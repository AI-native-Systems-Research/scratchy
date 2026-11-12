// SPDX-License-Identifier: Apache-2.0
//! Codegen: emit the per-(model × workload) forward fn + the
//! concrete `Weights` struct + `Weights::load` the caller uses.
//!
//! Scratchy owns weight loading. From the DSL + solver-picked Impls
//! the compiler knows every weight the forward needs — including
//! which fused Impls (e.g. `FusedQkvRopeCacheImpl`) want packed
//! weights built by concatenating multiple safetensors entries. So
//! the emitted module contains both:
//!
//! - `pub struct Weights { … }` — one field per unique `WeightAccessor`
//!   across every picked Impl in every workload tape_index. Fused
//!   accessors' fields are packed `LinearLayer`s produced by
//!   streaming concat at load time.
//! - `impl Weights { pub fn load(gw, stream) -> Result<Self> }` —
//!   reads safetensors via `GpuWeights` and produces the packed
//!   struct.
//! - `pub unsafe fn forward_m_<N>(wm: &Weights, ctx, device) -> OwnedTensor`
//!   per workload tape_index, plus a `forward(wm, ctx, device, num_tokens)`
//!   dispatcher.
//!
//! The caller's entire integration is two calls: `Weights::load(...)`
//! at startup and `forward(...)` per step.

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::classified::{OpKind, Program, UnrollIndex, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::interpreter_codegen::emit_bucket_static_slice;
use crate::weight_vocab::{SlotMap, WeightAccessor, WeightSlot};
use crate::workload::{WorkloadPoint, WorkloadPoints};

// ── Weights struct + loader emission ─────────────────────────────

/// Translate a DSL weight path (+ optional index) to the prefix the
/// safetensors file uses (without the trailing `.weight` / `.bias`).
///
/// HF decoder-only convention:
/// - `lm_head` → `lm_head`
/// - indexed weight like `self_attn.q_proj[L]` → `model.layers.<L>.self_attn.q_proj`
/// - other top-level (`embed_tokens`, `norm`, …) → `model.<dotted>`
///
/// Architectures that diverge (e.g. some models wrap lm_head in a
/// `model.` prefix) can override via a future per-arch conventions
/// mechanism; this covers Llama, Qwen2, Mistral, Gemma2.
pub(crate) fn safetensors_prefix(
    naming: crate::classified::WeightNaming<'_>,
    arch: &crate::arch_spec::DeclaredArchSpec,
    id: WeightId,
    index: Option<UnrollIndex>,
) -> String {
    let decoder_root = arch.decoder_root();
    let vision_layout = arch.safetensors.as_ref();
    // Weights at the checkpoint's top level beside the decoder root (as `lm_head`).
    let top_level = &arch.top_level_weights;
    // DSL idents can't start with a digit, so paths like
    // `merger.mlp.0` are written `merger.mlp_0` in the body /
    // manifest; translate `_<digit>` suffixes back to `.<digit>`
    // for the on-disk safetensors key (which uses Python-attribute
    // dotted form, including numeric submodule indices).
    let segs: Vec<String> = naming
        .weights
        .path(id)
        .iter()
        .map(|seg| translate_digit_suffix(seg))
        .collect();
    // DSL-leaf → disk-leaf rename (Gemma4: `self_attn.q_proj_global`
    // shares the on-disk leaf `self_attn.q_proj` with the sliding
    // class at a different shape; LocateAnything: the projector's
    // `linear_1`/`linear_2` disk leaves can't be named in the DSL —
    // a trailing `_<digit>` reads as a layer index — so
    // `mm.proj_in`/`mm.proj_out` rename here; GLM-4.5: the mid-path
    // `mlp.shared_expert → mlp.shared_experts`). Whole-segment,
    // longest-key-wins matching on the dotted DSL path
    // (`Program::rename_dotted`), applied BEFORE the vision subtree
    // resolution below so renamed segments flow into subtree paths.
    let joined = naming.rename_dotted(&segs.join("."));
    let segs: Vec<String> = joined.split('.').map(str::to_string).collect();
    let is_vision = matches!(naming.prelude, crate::classified::Prelude::Vision);
    if is_vision {
        // The per-arch vision layout is a REQUIRED declaration
        // (`const SAFETENSORS` on the carrier mod) — enforced at
        // config parse, so absence here is a compiler bug, not a
        // config gap.
        // Per-MODEL layout (`model.arch.safetensors` at the call site) —
        // NOT a crate-global: sibling variants of one arch can drift
        // (mlx_vlm repacks `visual.*` as `vision_tower.*`), and a
        // models[0]-keyed layout poisons every other variant's paths.
        let layout = vision_layout.expect("vision safetensors layout enforced at config parse");
        // Subtree override: when the DSL path's first segment maps
        // to a sibling subtree on disk, the override fully replaces
        // the `<default_root>(.<layered_subpath>.{l})?` prefix and
        // the matching subtree is treated as unindexed (today no
        // subtree consumer is per-block; add a per-subtree indexed
        // flag if a future arch needs it).
        if let Some(first) = segs.first()
            && let Some(disk) = layout.subtrees.get(first)
        {
            let rest = &segs[1..];
            return if rest.is_empty() {
                disk.clone()
            } else {
                format!("{disk}.{}", rest.join("."))
            };
        }
        return match index {
            Some(l) => format!(
                "{}.{}.{}.{}",
                layout.default_root, layout.layered_subpath, l, joined
            ),
            None => format!("{}.{}", layout.default_root, joined),
        };
    }
    // The decoder tree sits under `decoder_root`
    // ([`crate::arch_spec::DeclaredArchSpec::decoder_root`]): `model` for
    // HF decoders, `language_model.model` for Gemma3-MM-style wrappers,
    // `backbone` for Nemotron-H, the file's top level for a head stored bare
    // (an MLX multi-token-prediction drafter). `lm_head` sits beside that
    // tree at the file's top level in every checkpoint ordering.
    match (index, joined.as_str()) {
        (_, "lm_head") => "lm_head".to_string(),
        (None, w) if top_level.iter().any(|t| t == w) => w.to_string(),
        (Some(l), _) => under_root(&decoder_root, &format!("layers.{l}.{joined}")),
        (None, _) => under_root(&decoder_root, &joined),
    }
}

/// `rest` under `root` (`<root>.<rest>`), or `rest` itself for the empty (top-level) root.
pub(crate) fn under_root(root: &str, rest: &str) -> String {
    match root {
        "" => rest.to_string(),
        root => format!("{root}.{rest}"),
    }
}

/// The on-disk prefix `model` loads the lent weight `w` from: what a multi-token-prediction head
/// borrows from its target is found under the target's prefix and placed under the head's.
/// A tied lm_head loads from the embedding table. `None` when the program has no such weight.
pub(crate) fn lent_weight_prefix(
    program: &Program,
    fuf: &crate::fuf::Fuf,
    model: &crate::config::ModelParams,
    w: scratchy_forward_compiler::LentWeight,
) -> Option<String> {
    use scratchy_forward_compiler::LentWeight;
    // A head borrowing a subset (`borrows`, arch.json) reports none of the rest.
    let borrowed =
        |name: &str| (model.arch.borrows.as_ref()).is_none_or(|b| b.iter().any(|n| n == name));
    // The embedding table's prefix, wherever this arch keeps it: the table its Embed op reads,
    // named as the arch names it (Llama `embed_tokens`, Nemotron `embeddings`) — not a hardcoded
    // name, or a target naming it apart could not lend it. A tied lm_head shares that buffer.
    let embed_prefix = || {
        let role = crate::quantization::embed_table_role(program, fuf);
        let segs: Vec<&str> = role.split('.').collect();
        let id = program.weights.find(&segs)?;
        Some(safetensors_prefix(
            program.weight_naming(),
            &model.arch,
            id,
            None,
        ))
    };
    match w {
        LentWeight::LmHead if !borrowed("lm_head") => None,
        LentWeight::LmHead if !model.tie_word_embeddings => {
            let id = program.weights.find(&["lm_head"])?;
            Some(safetensors_prefix(
                program.weight_naming(),
                &model.arch,
                id,
                None,
            ))
        }
        // A head with an embedding of its own names its target's apart.
        LentWeight::EmbedTokens if program.weights.find(&["target_embed_tokens"]).is_some() => {
            let id = program.weights.find(&["target_embed_tokens"])?;
            Some(safetensors_prefix(
                program.weight_naming(),
                &model.arch,
                id,
                None,
            ))
        }
        LentWeight::EmbedTokens | LentWeight::LmHead => embed_prefix(),
    }
}

/// Translate trailing `_<digits>` in a DSL path segment back to
/// `.<digits>` so safetensors keys like `mlp.0` round-trip through
/// the DSL's ident-only path syntax.
fn translate_digit_suffix(seg: &str) -> String {
    if let Some(idx) = seg.rfind('_')
        && idx + 1 < seg.len()
        && seg[idx + 1..].chars().all(|c| c.is_ascii_digit())
    {
        format!("{}.{}", &seg[..idx], &seg[idx + 1..])
    } else {
        seg.to_string()
    }
}

/// How the `Weights::load` method constructs a field from safetensors.
#[derive(Clone)]
enum FieldLoad {
    /// `Embedding::load(gw, prefix)`.
    Embedding(String),
    /// MLX-affine int4 quantized embedding (Metal-only). Reads
    /// `<prefix>.{weight,scales,biases}` and CPU-dequantizes to BF16
    /// at load time so the rest of the forward path sees a normal
    /// dense embedding. Used by every `mlx-community/*-4bit` checkpoint
    /// in the int4 mandate's coverage matrix — they all ship the
    /// embedding as an affine triple,
    /// with the sole exception of Gemma-3-MM's language-model
    /// embedding (Gemma-3-MM is currently absent from the int4 P2
    /// gate). `group_size` / `bits` are taken from `quantization_config`
    /// at compile time. Emits `Embedding::load_affine_dequant`.
    EmbeddingAffine {
        prefix: String,
        group_size: u32,
        bits: u32,
    },
    /// Affine-quant embedding loaded into a DENSE `Embedding` by host/CPU
    /// dequant: `Embedding::load_affine_dequant(gw, prefix, group_size, bits)`.
    /// Chosen when the embed ACCESSOR is a dense `Embedding` (not the metal
    /// `AffineQuantEmbedding`) yet the on-disk storage is Affine — the
    /// storage-agnostic path `EmbedRefImpl` takes under spyre (the host worker
    /// dequantizes the embed weight), so the loaded field type matches the
    /// dense accessor. Distinct from [`Self::EmbeddingAffine`], which keeps the
    /// packed `AffineQuantEmbedding` for metal's fused gather+dequant kernel.
    EmbeddingAffineDequant {
        prefix: String,
        group_size: u32,
        bits: u32,
    },
    /// `RmsNorm::load(gw, prefix, eps)`. `eps` is baked in from the
    /// model config (`rms_norm_eps`).
    RmsNorm(String, f32),
    /// `LayerNorm::load(gw, prefix, eps)` — pulls `<prefix>.weight` AND
    /// optional `<prefix>.bias`. Used by `MeanSubRmsNormBiasAddImpl`
    /// (encoder models like ModernBERT, vision towers like Qwen2-VL).
    LayerNorm(String, f32),
    /// `GatedDeltaNetLayer::load(gw, prefix)` — pulls the 4 GDN
    /// per-layer weights `<prefix>.{conv1d.weight, A_log, dt_bias,
    /// norm.weight}`. `prefix` is the `linear_attn` prefix for the layer.
    /// Used by `GatedDeltaNetImpl` (Qwen3.5 / Qwen3-Next linear layers).
    GatedDeltaNet(String),
    /// `LinearLayer::load_dense(gw, prefix)`.
    LinearDense(String),
    /// `LinearLayer::load_dense_concat(gw, &[prefix0, prefix1, ...], stream)`.
    LinearConcat(Vec<String>),
    /// MLX-affine int4 quantized linear (Metal-only), single source.
    /// `group_size` and `bits` come from `quantization_config`. Under
    /// INT4 P2 emits `LinearLayer::load_affine_dequant_as_dense`
    /// (CPU dequant at load → Dense BF16). Under P3 will flip to
    /// `LinearLayer::load_affine_quant` for forward-time qmv.
    LinearAffine {
        prefix: String,
        group_size: u32,
        bits: u32,
        /// In-features (K) the arch manifest declares for this linear.
        /// The metal loader asserts the on-disk packed `.weight` width is
        /// `ceil(K*bits/32)`, rejecting a checkpoint quantized at different
        /// bits/group_size than this build's preset (else geometry is
        /// silently mis-derived from the on-disk shape).
        in_features: u32,
    },
    /// Fused-concat MLX-affine int4 quantized linear (Metal-only).
    /// `gate_proj` + `up_proj` (and q/k/v) ship as separate affine
    /// triples in mlx-community 4bit repos; the forward DSL fuses
    /// them into one `gate_up_proj` / `qkv_proj` linear so the metal
    /// `FusedGateUpSiluMul` impl pool matches the standard pattern.
    /// Emits `LinearLayer::load_affine_dequant_concat_as_dense`
    /// (CPU dequant per prefix, byte-concat along dim 0, alloc once).
    LinearAffineConcat {
        prefixes: Vec<String>,
        group_size: u32,
        bits: u32,
        /// Shared in-features (K) across the fused sources (gate_up / qkv all
        /// share K). The metal loader asserts each prefix's on-disk packed
        /// `.weight` width is `ceil(K*bits/32)`. See [`FieldLoad::LinearAffine`].
        in_features: u32,
    },
    /// One MLX-affine linear whose rows are several on-disk projections' rows in order, kept
    /// packed (weights.json `__packed_concats__`: Splash's fused QKV). Emits
    /// `LinearLayer::load_affine_quant_rows_concat`.
    LinearAffineRowConcat {
        prefixes: Vec<String>,
        group_size: u32,
        bits: u32,
        in_features: u32,
    },
    /// NVFP4 int4 quantized linear (Metal-only), single source. Emits
    /// `LinearLayer::load_nvfp4_quant` (forward-time `nvfp4_qmv` /
    /// `nvfp4_qmm_t`). `group_size` from `quantization_config` (16);
    /// bits is always 4.
    LinearNvfp4 { prefix: String, group_size: u32 },
    /// Fused-concat NVFP4 int4 quantized linear (Metal-only). `qkv_proj`
    /// / `gate_up_proj` ship as separate NVFP4 triples in ModelOpt
    /// checkpoints; the forward DSL fuses them. Emits
    /// `LinearLayer::load_nvfp4_quant_concat` (per-prefix fold, byte-
    /// concat packed weight + folded scales along dim 0). Concat is
    /// valid because each projection's global scale is folded into its
    /// per-group F16 scales before concatenation.
    LinearNvfp4Concat {
        prefixes: Vec<String>,
        group_size: u32,
    },
    /// `LinearLayer::load_raw(gw, key)` — reads `<key>` verbatim
    /// (no `.weight` / `.bias` suffix). Used for `nn.Parameter` weights
    /// (e.g. Gemma3 MM projector's `mm_input_projection_weight`) declared
    /// in the per-arch manifest with `kind: "raw_linear"`. Single-source
    /// only; fused-concat raw-linear isn't a real PyTorch shape and would
    /// be a manifest authoring error.
    RawLinear(String),
    /// A plain dense GpuTensor read verbatim via `gw.take` (the on-disk key is `<prefix>` itself —
    /// an `nn.Parameter`, like [`Self::RawLinear`]): gpt-oss's per-layer attention sinks, a
    /// DFlash2 drafter's per-layer convolution base kernels and its selector's codebooks. Never
    /// quantized, so [`affine_tensors_of`] lists no row for it.
    RawTensor(String),
    /// The model has `tie_word_embeddings: true`: `lm_head` shares
    /// its weight with `embed_tokens`. No safetensors read — build
    /// the `LinearLayer` from the already-loaded embedding field
    /// whose name is carried here.
    ///
    /// `affine` captures whether the source embedding is MLX-affine
    /// quantized (P6). When `Some((group_size, bits))`, the lm_head
    /// emits `LinearLayer::AffineQuant(...)` reading the embed's
    /// packed buffers; when `None`, it emits the legacy
    /// `LinearLayer::Dense(Linear::new(embed.weight, None))`.
    LinearTiedToEmbedding {
        embed_ident: syn::Ident,
        affine: Option<(u32, u32)>,
    },
    /// 4-bit packed INT4 linear that feeds a Marlin GEMM. Single-
    /// source (one prefix) or fused (multiple prefixes concat along
    /// dim N → one wider `MarlinLinear`). `format` selects the
    /// on-disk convention (AWQ vs GPTQ); only the loader fn name
    /// and a couple of storage-specific args (`desc_act` for GPTQ)
    /// vary, so a single arm emits both. Emits
    /// `MarlinLinear::load_{awq,gptq}[_concat]` against the
    /// ambient `__marlin_ws` / `__device_id` bindings that
    /// [`emit_weights_struct`] plants at the top of `Weights::load`
    /// when any Marlin accessor is present.
    MarlinLinear { prefixes: Vec<String> },
    /// BitsAndBytes 4-bit packed linear. Single prefix
    /// (`Bnb4bitLinear::load`) or fused across several
    /// (`Bnb4bitLinear::load_concat` — packed nibbles + absmax
    /// byte-concat along the N axis). Consumes the shared
    /// `__bnb_code` + `__bnb_scratch` bindings the
    /// [`emit_weights_struct`] prelude plants when any BNB4
    /// accessor is present.
    Bnb4Linear {
        prefixes: Vec<String>,
        /// Per-shard output dims — `sum()` is the fused
        /// `out_features` the loader hands to `Bnb4bitLinear`.
        /// Single-prefix loads carry one entry.
        out_features_per_shard: Vec<u32>,
        in_features: u32,
        blocksize: u32,
    },
    /// FP8 E4M3 linear. Single prefix (`Fp8Linear::load`) or fused
    /// across several (`Fp8Linear::load_concat` — max-scale merge,
    /// per Python vLLM's `requantize_with_max_scale`). The loader
    /// reads shapes + scale layout from the on-disk tensors so it
    /// handles per-tensor dynamic, per-channel, and online-quant
    /// (BF16 checkpoint) paths from the same arm. `output_dtype`
    /// is threaded in via the `__fp8_dtype` prelude binding.
    Fp8Linear { prefixes: Vec<String> },
    /// FP8 E4M3 blockwise-quantized linear (DeepSeek-V3-style 128×128
    /// block scales). Single prefix (`Fp8BlockLinear::load`) or fused
    /// across several (`Fp8BlockLinear::load_concat` — concat FP8
    /// weight and 2-D block-scale shards along N). Shares the
    /// `__fp8_dtype` prelude binding with `FieldLoad::Fp8Linear`.
    Fp8BlockLinear { prefixes: Vec<String> },
    /// DeepSeek V2/V3 MoE layer. One field per layer index; each
    /// calls `DeepSeekV2MoELayer::load` with the per-arch constants
    /// baked in as literals.
    DeepSeekV2Moe {
        prefix: String,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        /// True when `scoring_func="sigmoid"` + `topk_method="noaux_tc"` (DeepSeek V3 / Kimi K2).
        use_sigmoid: bool,
        /// Number of expert groups for grouped top-k (V3/Kimi K2). 0 = flat top-k.
        n_expert_group: usize,
        /// Number of groups to select in the first-stage grouped top-k. 0 = disabled.
        topk_group: usize,
    },
    /// FP8 blockwise-quantized analog of `DeepSeekV2Moe` — DeepSeek-V3
    /// official 671B and Kimi K2 official checkpoints.
    DeepSeekV2Fp8BlockMoe {
        prefix: String,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
    },
    /// GGML/GGUF analog of `DeepSeekV2Moe` — V2-Lite, Moonlight, K2 GGUFs.
    /// Calls `DeepSeekV2GgmlMoELayer::load_gguf` with the same per-arch
    /// constants. Expert weights stay quantized end-to-end.
    DeepSeekV2GgmlMoe {
        prefix: String,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
    },
    /// Mixtral-style fused MoE — no shared expert. Dispatches to
    /// `FusedMoELayer::load` (Dense BF16 expert weights, cuda) or
    /// `FusedMoELayer::load_affine` (MLX-affine int4 expert weights,
    /// metal) based on `affine`. Mixtral uses HF's
    /// `block_sparse_moe.experts.{e}.{w1,w2,w3}` layout.
    FusedMoe {
        prefix: String,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        /// `Some((group_size, bits))` when the macro detected
        /// `StorageFormat::Affine` on the MoE source weight — emits
        /// the Metal `load_affine` call site, `bits` each routed-expert
        /// projection's own width. `None` → Dense BF16 path → cuda
        /// `load` call site.
        affine: Option<(u32, crate::quantization::ExpertBits)>,
        /// Bit-width for the router `{prefix}.gate` dequant, resolved
        /// from the affine preset's `bits_overrides` (mlx MoE presets
        /// quantize the router gate at 8-bit while experts stay 4-bit).
        /// `None` when `affine` is, or when an MLX per-module map leaves
        /// the router dense (Nemotron-H); the section default when no
        /// override matches.
        gate_bits: Option<u32>,
    },
    /// Qwen-MoE-style fused MoE + shared expert. Cuda Dense via
    /// `SharedFusedMoELayer::load`, metal MLX-affine int4 via
    /// `SharedFusedMoELayer::load_affine`. Qwen2-MoE / Qwen3-MoE use
    /// HF's `experts.{e}.{gate,up,down}_proj` naming.
    SharedFusedMoe {
        prefix: String,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        /// `Some((group_size, bits))` when MLX-affine — see
        /// [`FieldLoad::FusedMoe::affine`].
        affine: Option<(u32, crate::quantization::ExpertBits)>,
        /// Router `{prefix}.gate` dequant bit-width — see
        /// [`FieldLoad::FusedMoe::gate_bits`].
        gate_bits: Option<u32>,
    },
    /// Gemma-4 router bundle (`GemmaMoe` op, base `router`). Dispatches to
    /// `GemmaRouterLayer::load`. The router's storage is resolved through
    /// `affine_role_widths` on the accessor's `{prefix}.proj` role — the same
    /// law the fingerprint's affine bit map uses — never defaulted: the
    /// backends used to bake an 8 no config stated, which left the variant's
    /// bit map silently disagreeing with the tensor the loader would read
    /// (#202), and the default kept fabricating an affine row for checkpoints
    /// whose router is DENSE — the RedHatAI FP8 26b lists `router.proj` in
    /// its own ignore list — so the fingerprint demanded `.scales` no such
    /// checkpoint ships and refused the only cached 26b FP8 weights at load.
    GemmaRouter {
        prefix: String,
        num_experts: usize,
        hidden_size: usize,
        /// `Some((group_size, bits))` when `router.proj` is MLX-affine — see
        /// [`FieldLoad::FusedMoe::affine`]. `None` when the merged config
        /// leaves it dense (fp8 checkpoints that skip the router, verbatim
        /// bf16): the loader takes the bf16 `router.proj.weight` verbatim.
        affine: Option<(u32, u32)>,
    },
    /// Gemma-4 SwitchGLU experts bundle (`GemmaMoe` op, base
    /// `experts.switch_glu`). Dispatches to `SwitchGluExpertsLayer::load`
    /// (metal-only). Always MLX-affine int4 (`group_size`, `bits`).
    GemmaSwitchGlu {
        prefix: String,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        group_size: u32,
        bits: u32,
    },
    /// gpt-oss fused MoE bundle (`GptOssMoe` op, base `mlp`). Dispatches
    /// to `GptOssMoELayer::load` (metal-only). One bundle carries the
    /// affine router gate at `{prefix}.router` (dequantized at
    /// `gate_bits` — gpt-oss-2Bit quantizes the router at the preset's
    /// own width), the F32 router bias, the pre-stacked affine experts,
    /// and the per-expert SwiGLU-OAI linear biases.
    GptOssMoe {
        prefix: String,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        /// `Some((group_size, bits))` when the experts are MLX-affine —
        /// the only storage with a metal realization. `None` tolerated
        /// for the dense/verbatim config (never runtime-selected).
        affine: Option<(u32, u32)>,
        /// Router `{prefix}.router` dequant bit-width, resolved through
        /// `affine_role_bits` like every other affine role — see
        /// [`FieldLoad::FusedMoe::gate_bits`]. The gpt-oss path names the
        /// gate `router`, not `gate`.
        gate_bits: Option<u32>,
    },
}

/// The MLX-affine `.weight` tensors this field load will read, each paired with
/// the `(bits, group_size)` THE LOAD CALL IS GIVEN.
///
/// ⛔ SINGLE SOURCE OF TRUTH for the fingerprint's affine bit table. The widths
/// are read straight off the `FieldLoad` the loader is about to be handed — not
/// re-derived — so the compiled fingerprint cannot claim a width the loader
/// won't use. A second resolver that merely *agreed* in the cases we tested is
/// what made #202's gate need a hand-maintained patch table for Gemma-4's
/// router.
///
/// Names are exact on-disk paths (the prefixes are layer-0 concrete, the same
/// ones `layer_templated_prefix_expr` templates for the load calls), so the
/// runtime does equality, never suffix matching: `mlp.gate_proj` can never
/// collide with `switch_mlp.gate_proj`.
///
/// Only tensors whose width this function can state are listed. Unlisted
/// tensors are simply not checked.
///
/// ⚠️ RESIDUAL RISK, now scoped: the GATED routed-expert stacks of the SHARED
/// MoE bundles are NOT listed (an ungated bank's are — see the arm).
/// `switch_mlp.{gate,up,down}_proj`, Mixtral's
/// `w1/w2/w3` and `GemmaSwitchGlu`'s sub-leaves are named inside the backend
/// loaders, not here, so this function cannot state their on-disk paths
/// without duplicating that naming — which is the duplication the whole
/// design exists to remove. The consequence is explicit: for those bundles
/// the expert-bit axis has NO fingerprint coverage, so a #202-class
/// divergence in EXPERT widths (rather than the router gates') would still
/// mis-select and surface down in the dequant path. No in-tree pair of
/// presets for those arches agrees on every listed width while differing in
/// expert bits, so the omission cannot bite today. `GptOssMoe` DOES list its
/// expert stacks (the macro states those leaf names — see the arm): its
/// b2-attn4/b3-attn4 preset pair agrees on attention, router and embed
/// widths and differs ONLY in expert bits, so the rows are load-bearing.
fn affine_tensors_of(fl: &FieldLoad) -> Vec<(String, u32, u32)> {
    let one = |p: &str, bits: u32, gs: u32| vec![(format!("{p}.weight"), bits, gs)];
    match fl {
        FieldLoad::EmbeddingAffine {
            prefix,
            group_size,
            bits,
            ..
        }
        | FieldLoad::EmbeddingAffineDequant {
            prefix,
            group_size,
            bits,
            ..
        }
        | FieldLoad::LinearAffine {
            prefix,
            group_size,
            bits,
            ..
        } => one(prefix, *bits, *group_size),
        FieldLoad::LinearAffineConcat {
            prefixes,
            group_size,
            bits,
            ..
        }
        | FieldLoad::LinearAffineRowConcat {
            prefixes,
            group_size,
            bits,
            ..
        } => prefixes
            .iter()
            .map(|p| (format!("{p}.weight"), *bits, *group_size))
            .collect(),
        // MoE router gate: `gate_bits` is what the `load_affine` call receives
        // for `{prefix}.gate`, and it is the role qwen3.5 and qwen3.6 disagree
        // about. A gated routed stack under the same prefix is omitted (see the
        // fn doc). An UNGATED stack is listed: its loader reads exactly the
        // pre-stacked `switch_mlp.fc1` / `fc2` (mlx `SwitchMLP` — no per-expert
        // layout exists for it), so this function can state them, and they are
        // load-bearing — Nemotron-3-Super's OptiQ map and a uniform 4-bit one
        // agree on every other width and differ in the experts' alone.
        FieldLoad::FusedMoe {
            prefix,
            affine: Some((group_size, bits)),
            gate_bits,
            ..
        }
        | FieldLoad::SharedFusedMoe {
            prefix,
            affine: Some((group_size, bits)),
            gate_bits,
            ..
        } => {
            let mut rows = match gate_bits {
                Some(b) => one(&format!("{prefix}.gate"), *b, *group_size),
                None => Vec::new(),
            };
            if let crate::quantization::ExpertBits::Ungated { .. } = bits {
                for (leaf, b) in bits.leaves() {
                    rows.push((format!("{prefix}.switch_mlp.{leaf}.weight"), b, *group_size));
                }
            }
            rows
        }
        FieldLoad::GemmaRouter {
            prefix,
            affine: Some((group_size, bits)),
            ..
        } => one(&format!("{prefix}.proj"), *bits, *group_size),
        // gpt-oss router gate: `gate_bits` is what the load call dequantizes
        // `{prefix}.router` at. The expert stacks under `{prefix}.experts`
        // ARE listed — unlike the other MoE bundles, the macro states their
        // on-disk leaf names here because it constructs the very load call
        // that reads them. They are not optional: the b2-attn4 and b3-attn4
        // presets agree on every OTHER fingerprinted width (attention 4,
        // router 8, embed/lm_head 4), so the expert bits are the only row
        // that separates the two variants — without it whichever variant is
        // consulted first claims the other's checkpoint and dies in the
        // loader's packed-width assert (#316).
        FieldLoad::GptOssMoe {
            prefix,
            affine: Some((group_size, expert_bits)),
            gate_bits: Some(gate_bits),
            ..
        } => {
            let mut rows = one(&format!("{prefix}.router"), *gate_bits, *group_size);
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                rows.push((
                    format!("{prefix}.experts.{proj}.weight"),
                    *expert_bits,
                    *group_size,
                ));
            }
            rows
        }
        _ => Vec::new(),
    }
}

/// Collapse exact per-layer affine rows into one row per role wherever every
/// layer agrees on `(bits, group_size)`, by replacing the layer index with `*`.
///
/// Keeps the table at one row per role for a uniform checkpoint (17 for
/// qwen3.5, not 40 x 17) while staying exact: a role whose width VARIES by layer
/// — MLX mixed/dynamic, i.e. OptiQ — keeps its concrete per-layer rows, because
/// starring it would claim layer 0's width for all 40 layers.
fn compress_affine_rows(rows: Vec<(String, u32, u32)>) -> Vec<(String, u32, u32)> {
    /// Replace a path's layer index with `*`. The FIRST all-digits segment is
    /// the layer slot (`model.layers.7.…`, `visual.blocks.7.…`); rows arrive one
    /// per concrete layer, so this must match any index, not just layer 0.
    fn starred(name: &str) -> Option<String> {
        let segments: Vec<&str> = name.split('.').collect();
        let at = segments
            .iter()
            .position(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))?;
        let mut out = segments;
        out[at] = "*";
        Some(out.join("."))
    }
    let mut widths: std::collections::BTreeMap<String, std::collections::BTreeSet<(u32, u32)>> =
        Default::default();
    for (name, bits, gs) in &rows {
        if let Some(pat) = starred(name) {
            widths.entry(pat).or_default().insert((*bits, *gs));
        }
    }
    let mut out: Vec<(String, u32, u32)> = Vec::new();
    for (name, bits, gs) in rows {
        match starred(&name) {
            // One width across every layer of this role — one starred row.
            Some(pat) if widths.get(&pat).is_some_and(|w| w.len() == 1) => {
                if !out.iter().any(|(n, _, _)| *n == pat) {
                    out.push((pat, bits, gs));
                }
            }
            // Per-layer variation, or not layered at all — keep it concrete.
            _ => out.push((name, bits, gs)),
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Per-storage-format parameters carried on a [`FieldLoad::MarlinLinear`].
/// `group_size` lives on the outer struct because both formats share
/// it; the enum captures the storage-specific bits.
#[derive(Clone, Copy, Debug)]
enum MarlinFormat {
    Awq,
    Gptq {
        /// Mirrors `quantization_config.desc_act`. `true` ⇒ loader
        /// reads `.g_idx`, argsort-permutes, and hands sort_indices
        /// to `gptq_repack_into` so same-group columns are
        /// contiguous post-repack.
        desc_act: bool,
        /// On-disk layout — `Qweight` for AutoGPTQ native
        /// (`.qweight [K/8, N]` + `.scales [num_groups, N]`) or
        /// `WeightPacked` for compressed-tensors repack
        /// (`.weight_packed [N, K/8]` + `.weight_scale [N,
        /// num_groups]`). The loader transposes CT tensors before
        /// `gptq_repack_into`, so the downstream kernel path is
        /// identical regardless of which layout this variant came
        /// from.
        layout: crate::quantization::GptqLayout,
    },
}

/// MoE config bits shared by `DeepSeekV2Moe` and `DeepSeekV2Fp8BlockMoe`
/// FieldLoad variants. Reads the per-variant config JSON once and falls
/// back to the model's `bounds` / `scalars` map when a key is absent.
struct DeepSeekMoeCfg {
    n_routed_experts: usize,
    n_shared_experts: usize,
    top_k: usize,
    moe_intermediate_size: usize,
    hidden_size: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    use_sigmoid: bool,
    n_expert_group: usize,
    topk_group: usize,
}

/// A fused-MoE accessor's affine plan: `(group_size, each routed-expert projection's width)`
/// when its bundle is MLX-affine, and the router gate's own width — `None` when the bundle is
/// dense, or when an MLX per-module map leaves the router dense. Resolved at the accessor's own
/// layer (per-layer expert bits under OptiQ), through the ONE per-projection resolver
/// [`crate::quantization::affine_moe_expert_bits`], in the expert body the config declares.
fn moe_affine_plan(
    program: &Program,
    fuf: &Fuf,
    model: &ModelParams,
    prefix: &str,
    accessor: &WeightAccessor,
) -> (Option<(u32, crate::quantization::ExpertBits)>, Option<u32>) {
    use crate::quantization::{ExpertBits, StorageFormat};
    let (src, idx) = accessor.source_weights[0];
    let StorageFormat::Affine { group_size, bits } =
        crate::quantization::storage_format_for_weight(program, fuf, src, idx, model)
    else {
        return (None, None);
    };
    let body = crate::config::expert_body(&model.expert_activation).unwrap_or_else(|act| {
        panic!(
            "{}: the config declares expert activation `{act}`, which no expert body realizes",
            model.name
        )
    });
    let method = model.quantization.as_ref().map(|qc| &qc.method);
    let resolved = method
        .and_then(|m| crate::quantization::affine_moe_expert_bits(program, m, prefix, idx, body));
    let expert = resolved.unwrap_or(match body {
        scratchy_subtile::subtile_ir::ExpertBody::Gated(_) => ExpertBits::Gated {
            gate: bits,
            up: bits,
            down: bits,
        },
        scratchy_subtile::subtile_ir::ExpertBody::Ungated(_) => ExpertBits::Ungated {
            fc1: bits,
            fc2: bits,
        },
    });
    // The mlx MoE presets quantize `{prefix}.gate` at 8-bit while experts stay 4-bit — via
    // preset `bits_overrides` OR (OptiQ) the per-module map, which `affine_role_bits` honors
    // both of (a router the map does not list is dense).
    let gate_path = format!("{prefix}.gate");
    let gate_bits =
        method.and_then(|m| crate::quantization::affine_role_bits(program, m, &gate_path, idx));
    (Some((group_size, expert)), gate_bits)
}

/// A bundle loader's dims, each its derived role's (`bundles::derive`, from the op that reads the
/// bundle and the model's config) evaluated under the model's bounds: the bake emits them as the
/// loader's constants, the same bundle rustc typed against the ops (`typed_shapes`) and the tests
/// hold to the checkpoints (`every_derived_bundle_is_its_checkpoints`).
#[derive(Clone, Copy, Debug)]
struct BundleDims {
    /// A router's `[width, experts]` (a scaled router's projection's).
    router: Option<(usize, usize)>,
    /// An expert bank's `[experts, width, inner]` (its up projection's).
    bank: Option<(usize, usize, usize)>,
    /// The shared experts' total inner width, if the bundle holds them.
    shared: Option<usize>,
}

impl BundleDims {
    /// The bundle `accessor` reads (its one source weight's), derived.
    fn of(accessor: &WeightAccessor, program: &Program, model: &ModelParams) -> Self {
        let (src, _) = accessor.source_weights[0];
        let path = program.weights.path(src).join(".");
        let config = crate::bundles::BundleConfig {
            bounds: &model.bounds,
            activation: &model.expert_activation,
        };
        let derived = crate::bundles::derive(program, &config)
            .unwrap_or_else(|e| panic!("`{}`: its bundles: {e}", model.source_stem));
        let bundle = derived.get(&path).unwrap_or_else(|| {
            panic!(
                "accessor `{}`: `{path}` is no bundle an op reads whole, to size its loader by",
                accessor.name
            )
        });
        use crate::bundles::BundleRole as R;
        let role = |role: R| -> Option<Vec<usize>> {
            let at = bundle.kind.roles().iter().position(|r| *r == role)?;
            let dims = (bundle.roles[at].iter())
                .map(|d| {
                    let v = crate::shape::eval_closed_dim(d, &model.bounds).unwrap_or_else(|| {
                        panic!(
                            "`{path}.{}`: {d:?} names a bound `{}` lacks",
                            role.name(),
                            model.source_stem
                        )
                    });
                    v as usize
                })
                .collect();
            Some(dims)
        };
        let pair = |s: Vec<usize>| (s[0], s[1]);
        Self {
            router: (role(R::Router))
                .or_else(|| role(R::GptOssRouter))
                .or_else(|| role(R::Proj))
                .map(pair),
            bank: role(R::Up).map(|s| (s[0], s[1], s[2])),
            shared: role(R::SharedGate).map(|s| s[1]),
        }
    }

    fn router(self, accessor: &WeightAccessor) -> (usize, usize) {
        (self.router)
            .unwrap_or_else(|| panic!("accessor `{}`: its bundle has no router", accessor.name))
    }

    fn bank(self, accessor: &WeightAccessor) -> (usize, usize, usize) {
        (self.bank).unwrap_or_else(|| {
            panic!(
                "accessor `{}`: its bundle has no expert bank",
                accessor.name
            )
        })
    }
}

/// The experts a token routes to (`num_experts_per_tok`): routing, a model bound.
fn routed_top_k(model: &ModelParams) -> usize {
    let k = model.bounds.get("num_experts_per_tok").unwrap_or_else(|| {
        panic!(
            "model `{}` states no `num_experts_per_tok` for its MoE",
            model.source_stem
        )
    });
    *k as usize
}

/// A DeepSeek MoE loader's constants: its dims from its derived bundle ([`BundleDims`]), its
/// routing knobs (`norm_topk_prob`, `routed_scaling_factor`, the scoring, the expert groups) from
/// the model's config, read at expansion.
fn read_deepseek_moe_cfg(
    model: &ModelParams,
    dims: BundleDims,
    accessor: &WeightAccessor,
) -> DeepSeekMoeCfg {
    let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
    let (hidden_size, n_routed_experts) = dims.router(accessor);
    let (_, _, moe_intermediate_size) = dims.bank(accessor);
    // The shared experts' total inner width is that many experts' inner widths.
    let shared = dims.shared.unwrap_or_else(|| {
        panic!(
            "accessor `{}`: a DeepSeek MoE bundle holds shared experts",
            accessor.name
        )
    });
    assert!(
        shared.is_multiple_of(moe_intermediate_size),
        "accessor `{}`: shared width {shared} is no whole number of {moe_intermediate_size}-wide experts",
        accessor.name
    );
    let n_shared_experts = shared / moe_intermediate_size;
    let top_k = routed_top_k(model);
    let norm_topk_prob = v
        .get("norm_topk_prob")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let routed_scaling_factor = v
        .get("routed_scaling_factor")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or_else(|| {
            model
                .scalars
                .get("routed_scaling_factor")
                .copied()
                .unwrap_or(1.0) as f32
        });
    let use_sigmoid = v
        .get("scoring_func")
        .and_then(|x| x.as_str())
        .map(|s| s == "sigmoid")
        .unwrap_or(false)
        && v.get("topk_method")
            .and_then(|x| x.as_str())
            .map(|s| s == "noaux_tc")
            .unwrap_or(false);
    let n_expert_group = v
        .get("n_group")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(0);
    let topk_group = v
        .get("topk_group")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or(0);
    DeepSeekMoeCfg {
        n_routed_experts,
        n_shared_experts,
        top_k,
        moe_intermediate_size,
        hidden_size,
        norm_topk_prob,
        routed_scaling_factor,
        use_sigmoid,
        n_expert_group,
        topk_group,
    }
}

/// Guard G5b: uniform actionable error for a storage-gated weight-load
/// arm whose accessor requires a specific on-disk storage the resolved
/// weight doesn't have. Names the config (`model.source_stem`) and the
/// offending accessor so the next person knows instantly WHICH config
/// variant (verbatim vs quantized) and WHICH weight slot tripped it —
/// the old `custom attribute panicked` text named neither. The five
/// storage-gated panics (Marlin/Bnb4/Fp8/AffineQuantEmbedding/
/// LinearAffine) route through here; they stay HARD ERRORS because their
/// matchers gate on storage, so a Dense source provably can't select
/// them — reaching the panic means a real matcher or manifest bug.
fn storage_requirement_error(
    model: &ModelParams,
    accessor: &WeightAccessor,
    required: &str,
    got: &str,
) -> String {
    format!(
        "[weights] config `{stem}`: accessor `{acc}` requires {required} weight storage, \
         but its source resolves to {got}. If this accessor is selected by an op (not by \
         storage) and the Dense/verbatim config legitimately can't run this specialization on \
         this backend, tolerate Dense and default the params (see the GemmaMoe arm); otherwise \
         the solver picked a quant impl for a dense weight (matcher bug) or the weights manifest \
         mis-declares this tensor's storage.",
        stem = model.source_stem,
        acc = accessor.name,
    )
}

/// Did the solver keep the `embed_tokens` of `accessors` as the packed `AffineQuantEmbedding`
/// (metal's fused gather+dequant) or dequantize it to a dense `Embedding` (spyre/cuda host path)?
/// A tied `lm_head` follows that representation.
fn embed_is_affine_repr(accessors: &[WeightAccessor]) -> bool {
    accessors.iter().any(|a| {
        let n = a.name.to_string();
        (n == "embed_tokens" || n.ends_with("_embed_tokens"))
            && a.rust_type
                .to_string()
                .replace(' ', "")
                .contains("AffineQuantEmbedding")
    })
}

/// The MLX-affine codes, `(bits, group)`, a field loaded by `plan` keeps packed on device: a
/// single-source or row-concatenated affine linear's, a tied head's over a packed embedding, and
/// that packed embedding's. `None` for any other load — the affine sources fused at load are
/// dequantized to one dense weight.
fn packed_affine_codes(plan: &FieldLoad) -> Option<(u32, u32)> {
    match *plan {
        FieldLoad::LinearAffine {
            group_size, bits, ..
        }
        | FieldLoad::LinearAffineRowConcat {
            group_size, bits, ..
        }
        | FieldLoad::LinearTiedToEmbedding {
            affine: Some((group_size, bits)),
            ..
        }
        | FieldLoad::EmbeddingAffine {
            group_size, bits, ..
        } => Some((bits, group_size)),
        _ => None,
    }
}

/// Distill a `WeightAccessor` into its field-load plan. Uses the
/// accessor's declared `rust_type` + `source_weights` and the
/// model's config (for `rms_norm_eps` / `tie_word_embeddings`).
fn plan_field_load(
    accessor: &WeightAccessor,
    program: &Program,
    fuf: &Fuf,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    // Whether the `embed_tokens` accessor the solver picked is the packed
    // `AffineQuantEmbedding` representation (metal's fused gather+dequant) vs a
    // dense `Embedding` (the host-dequant path spyre/cuda take). A tied `lm_head`
    // must follow the embedding's REPRESENTATION, not the raw on-disk storage:
    // when the embed is dense, the tied head reuses its dense weight; only when
    // the embed keeps the affine triple does the head read `scales`/`biases`.
    embed_is_affine_repr: bool,
) -> FieldLoad {
    let ty = accessor.rust_type.to_string().replace(' ', "");
    let is_deepseek_v2_moe = ty.ends_with("::DeepSeekV2MoELayer")
        || ty == "DeepSeekV2MoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2MoELayer");
    let is_deepseek_v2_fp8_block_moe = ty.ends_with("::DeepSeekV2Fp8BlockMoELayer")
        || ty == "DeepSeekV2Fp8BlockMoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2Fp8BlockMoELayer");
    let is_fused_moe = ty.ends_with("::FusedMoELayer")
        || ty == "FusedMoELayer"
        || ty.ends_with("layers_moe::FusedMoELayer");
    let is_shared_fused_moe = ty.ends_with("::SharedFusedMoELayer")
        || ty == "SharedFusedMoELayer"
        || ty.ends_with("layers_moe::SharedFusedMoELayer");
    let is_gemma_router = ty.ends_with("::GemmaRouterLayer")
        || ty == "GemmaRouterLayer"
        || ty.ends_with("layers_moe::GemmaRouterLayer");
    let is_gemma_switch_glu = ty.ends_with("::SwitchGluExpertsLayer")
        || ty == "SwitchGluExpertsLayer"
        || ty.ends_with("layers_moe::SwitchGluExpertsLayer");
    let is_gpt_oss_moe = ty.ends_with("::GptOssMoELayer")
        || ty == "GptOssMoELayer"
        || ty.ends_with("layers_moe::GptOssMoELayer");
    let is_deepseek_v2_ggml_moe = ty.ends_with("::DeepSeekV2GgmlMoELayer")
        || ty == "DeepSeekV2GgmlMoELayer"
        || ty.ends_with("layers_moe::DeepSeekV2GgmlMoELayer");
    let is_gated_delta_net = ty.ends_with("::GatedDeltaNetLayer")
        || ty == "GatedDeltaNetLayer"
        || ty.ends_with("layers::GatedDeltaNetLayer");
    let is_embedding =
        ty.ends_with("::Embedding") || ty == "Embedding" || ty.ends_with("layers::Embedding");
    // P6: MLX-affine int4 quantized embedding. Distinct type from
    // `Embedding` (carries packed U32 weight + F16 scales + F16
    // affine offsets); the field_load arm always routes to
    // `EmbeddingAffine` since the storage is known-Affine by
    // construction (`MetalAffineEmbedImpl::matches` gates on
    // `StorageFormat::Affine`).
    let is_affine_quant_embedding = ty.ends_with("::AffineQuantEmbedding")
        || ty == "AffineQuantEmbedding"
        || ty.ends_with("layers::AffineQuantEmbedding");
    let is_rmsnorm =
        ty.ends_with("::RmsNorm") || ty == "RmsNorm" || ty.ends_with("layers::RmsNorm");
    let is_layer_norm =
        ty.ends_with("::LayerNorm") || ty == "LayerNorm" || ty.ends_with("layers::LayerNorm");
    // A bare `GpuTensor` field (gpt-oss's sinks, a DFlash2 drafter's convolution kernels and
    // codebooks). The rotary (`WeightKind::CosSin`) is the only other GpuTensor on a Weights
    // struct and is planted directly by `emit_weights_struct`, never through this walk — so a
    // GpuTensor-typed accessor here is read verbatim.
    let is_raw_tensor =
        ty.ends_with("::GpuTensor") || ty == "GpuTensor" || ty.ends_with("tensor::GpuTensor");
    let is_linear =
        ty.ends_with("::LinearLayer") || ty == "LinearLayer" || ty.ends_with("layers::LinearLayer");
    let is_marlin = ty.ends_with("::MarlinLinear")
        || ty == "MarlinLinear"
        || ty.ends_with("layers::MarlinLinear");
    let is_bnb4 = ty.ends_with("::Bnb4bitLinear")
        || ty == "Bnb4bitLinear"
        || ty.ends_with("layers::Bnb4bitLinear");
    // Post-Fp8AnyLinear: the macro-emitted field type is always
    // `Fp8AnyLinear` (the wrapper enum). Whether to load as `Std`
    // or `Block` is decided per-weight from the on-disk storage
    // format below.
    let is_fp8_any = ty.ends_with("::Fp8AnyLinear")
        || ty == "Fp8AnyLinear"
        || ty.ends_with("layers::Fp8AnyLinear");

    let prefixes: Vec<String> = accessor
        .source_weights
        .iter()
        .map(|(id, idx)| safetensors_prefix(program.weight_naming(), &model.arch, *id, *idx))
        .collect();

    if is_marlin {
        // Marlin accessors are always emitted by a quant-aware impl
        // whose sources resolve to an AWQ or GPTQ storage format.
        // Group size + format must agree across every source of a
        // fused accessor (HF's fused-QKV/gate-up layers share one
        // quantization); mismatch is a data-integrity error in the
        // upstream HF repo, so we panic at macro-expansion time
        // rather than silently emit a wrong loader.
        let mut resolved: Option<(MarlinFormat, u32)> = None;
        for (wid, _idx) in &accessor.source_weights {
            let fmt =
                crate::quantization::storage_format_for_weight(program, fuf, *wid, None, model);
            let (mf, g) = match fmt {
                crate::quantization::StorageFormat::Awq { group_size: g, .. } => {
                    (MarlinFormat::Awq, g)
                }
                crate::quantization::StorageFormat::Gptq {
                    group_size: g,
                    desc_act,
                    layout,
                    ..
                } => (MarlinFormat::Gptq { desc_act, layout }, g),
                other => panic!(
                    "{}",
                    storage_requirement_error(
                        model,
                        accessor,
                        "Marlin (Awq|Gptq)",
                        &format!("{other:?}"),
                    )
                ),
            };
            match &resolved {
                None => resolved = Some((mf, g)),
                Some((existing_mf, existing_g)) => {
                    // Format mismatch (one source Awq, another Gptq)
                    // would require two different loaders for one
                    // fused accessor — HF never mixes formats within
                    // a single MergedColumnParallelLinear.
                    let format_match = matches!(
                        (existing_mf, &mf),
                        (MarlinFormat::Awq, MarlinFormat::Awq)
                            | (MarlinFormat::Gptq { .. }, MarlinFormat::Gptq { .. })
                    );
                    if !format_match {
                        panic!(
                            "accessor `{}` fuses sources with mismatched Marlin formats \
                                 ({existing_mf:?} vs {mf:?})",
                            accessor.name,
                        );
                    }
                    if *existing_g != g {
                        panic!(
                            "accessor `{}` fuses sources with mismatched group_size \
                                 (saw {existing_g} then {g})",
                            accessor.name,
                        );
                    }
                    // desc_act and on-disk layout have to agree
                    // across sources — `.g_idx` is shared at the K
                    // axis, so fused-QKV sub-weights either all
                    // carry it or none do; a mixed AutoGPTQ /
                    // compressed-tensors fusion would be an upstream
                    // packaging error that would produce garbage
                    // after the repack.
                    if let (
                        MarlinFormat::Gptq {
                            desc_act: existing_desc_act,
                            layout: existing_layout,
                        },
                        MarlinFormat::Gptq { desc_act, layout },
                    ) = (existing_mf, &mf)
                    {
                        if *existing_desc_act != *desc_act {
                            panic!(
                                "accessor `{}` fuses GPTQ sources with mismatched desc_act \
                                     (saw {existing_desc_act} then {desc_act})",
                                accessor.name,
                            );
                        }
                        if *existing_layout != *layout {
                            panic!(
                                "accessor `{}` fuses GPTQ sources with mismatched layouts \
                                     ({existing_layout:?} vs {layout:?})",
                                accessor.name,
                            );
                        }
                    }
                }
            }
        }
        // Resolved for its CHECK, not its value: the format/group-size it yields was stored on
        // the variant and never read back. The accessor must still declare a source weight.
        resolved.expect("MarlinLinear accessor declares at least one source weight");
        return FieldLoad::MarlinLinear { prefixes };
    }

    if is_bnb4 {
        // Each source weight's manifest shape evaluated against the
        // arch's bounds gives `[out_features, in_features]`. Fused
        // BNB4 concats along N, so every shard must agree on
        // `in_features`; the loader takes `out_features_per_shard`
        // + one shared `in_features`. Mismatched in_features is an
        // upstream repo bug — panic at macro-expansion time.
        let (mut out_per_shard, mut in_features, mut blocksize) =
            (Vec::<u32>::new(), None::<u32>, None::<u32>);
        for (wid, _idx) in &accessor.source_weights {
            let fmt =
                crate::quantization::storage_format_for_weight(program, fuf, *wid, None, model);
            let bs = match fmt {
                crate::quantization::StorageFormat::Bnb4 { blocksize: bs, .. } => bs,
                other => panic!(
                    "{}",
                    storage_requirement_error(model, accessor, "BNB4", &format!("{other:?}"))
                ),
            };
            if let Some(existing) = blocksize
                && existing != bs
            {
                panic!(
                    "accessor `{}` fuses BNB4 sources with mismatched blocksize \
                         ({existing} then {bs})",
                    accessor.name,
                );
            }
            blocksize = Some(bs);

            let segments = program.weights.path(*wid);
            let dotted = segments.join(".");
            let shape = manifest.lookup(segments).unwrap_or_else(|| {
                panic!(
                    "accessor `{}`: weight `{dotted}` missing from weights manifest; \
                     required for BNB4 out_features/in_features evaluation",
                    accessor.name,
                )
            });
            if shape.len() != 2 {
                panic!(
                    "accessor `{}`: BNB4 source weight `{dotted}` has shape len \
                     {} (expected 2 for a matmul)",
                    accessor.name,
                    shape.len(),
                );
            }
            let in_f =
                crate::shape::eval_closed_dim(&shape[0], &model.bounds).unwrap_or_else(|| {
                    panic!(
                        "accessor `{}`: can't resolve in_features for `{dotted}`",
                        accessor.name
                    )
                }) as u32;
            let out_f =
                crate::shape::eval_closed_dim(&shape[1], &model.bounds).unwrap_or_else(|| {
                    panic!(
                        "accessor `{}`: can't resolve out_features for `{dotted}`",
                        accessor.name
                    )
                }) as u32;
            if let Some(existing) = in_features
                && existing != in_f
            {
                panic!(
                    "accessor `{}` fuses BNB4 sources with mismatched in_features \
                         ({existing} then {in_f})",
                    accessor.name,
                );
            }
            in_features = Some(in_f);
            out_per_shard.push(out_f);
        }
        return FieldLoad::Bnb4Linear {
            prefixes,
            out_features_per_shard: out_per_shard,
            in_features: in_features.expect("at least one source weight"),
            blocksize: blocksize.expect("at least one source weight"),
        };
    }

    if is_fp8_any {
        // The accessor's emitted field type is `Fp8AnyLinear`. We
        // pick `FieldLoad::Fp8Linear` (per-tensor / per-channel,
        // wraps as `Fp8AnyLinear::Std`) or `FieldLoad::Fp8BlockLinear`
        // (blockwise, wraps as `Fp8AnyLinear::Block`) by reading the
        // source weight's on-disk storage format. All source weights
        // of one fused accessor share a format — HF never mixes
        // per-tensor and blockwise scales within a single
        // MergedColumnParallelLinear — so we read the FIRST source's
        // format and assert the rest agree.
        let mut block_size: Option<bool> = None;
        for (wid, _idx) in &accessor.source_weights {
            let fmt =
                crate::quantization::storage_format_for_weight(program, fuf, *wid, None, model);
            let is_block = match fmt {
                crate::quantization::StorageFormat::Fp8 { block_size, .. } => block_size.is_some(),
                other => panic!(
                    "{}",
                    storage_requirement_error(model, accessor, "FP8", &format!("{other:?}"))
                ),
            };
            match block_size {
                None => block_size = Some(is_block),
                Some(prev) => {
                    if prev != is_block {
                        panic!(
                            "accessor `{}` fuses sources with mismatched FP8 layout \
                             (one block-quant, the other per-tensor/per-channel)",
                            accessor.name,
                        );
                    }
                }
            }
        }
        return if block_size == Some(true) {
            FieldLoad::Fp8BlockLinear { prefixes }
        } else {
            FieldLoad::Fp8Linear { prefixes }
        };
    }

    if is_deepseek_v2_fp8_block_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2Fp8BlockMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let cfg = read_deepseek_moe_cfg(model, BundleDims::of(accessor, program, model), accessor);
        return FieldLoad::DeepSeekV2Fp8BlockMoe {
            prefix,
            n_routed_experts: cfg.n_routed_experts,
            n_shared_experts: cfg.n_shared_experts,
            top_k: cfg.top_k,
            moe_intermediate_size: cfg.moe_intermediate_size,
            hidden_size: cfg.hidden_size,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            use_sigmoid: cfg.use_sigmoid,
            n_expert_group: cfg.n_expert_group,
            topk_group: cfg.topk_group,
        };
    }

    if is_deepseek_v2_ggml_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2GgmlMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let cfg = read_deepseek_moe_cfg(model, BundleDims::of(accessor, program, model), accessor);
        return FieldLoad::DeepSeekV2GgmlMoe {
            prefix,
            n_routed_experts: cfg.n_routed_experts,
            n_shared_experts: cfg.n_shared_experts,
            top_k: cfg.top_k,
            moe_intermediate_size: cfg.moe_intermediate_size,
            hidden_size: cfg.hidden_size,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            use_sigmoid: cfg.use_sigmoid,
            n_expert_group: cfg.n_expert_group,
            topk_group: cfg.topk_group,
        };
    }

    if is_fused_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "FusedMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let dims = BundleDims::of(accessor, program, model);
        let (num_experts, hidden_size, intermediate_size) = dims.bank(accessor);
        let top_k = routed_top_k(model);
        // Probe storage format: if expert weights are MLX-affine int4
        // (mlx-community 4bit Mixtral / Qwen-MoE), emit the metal
        // `load_affine` call site; otherwise emit cuda `load`.
        // Same check the `MetalFusedMoeImpl::matches` uses via
        // `affine_gs_bits`. Source weights all share storage format
        // (the macro asserts this elsewhere for AffineQuantLinear).
        // Resolved at this accessor's layer index — MLX mixed/dynamic quant
        // (OptiQ) varies expert bits per numbered layer; emit_group_let
        // gathers the per-layer bits across the group.
        let (affine, gate_bits) = moe_affine_plan(program, fuf, model, &prefix, accessor);
        return FieldLoad::FusedMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        };
    }

    if is_shared_fused_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "SharedFusedMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let dims = BundleDims::of(accessor, program, model);
        // The experts' own width: the model's hidden size, or a latent MoE's narrower expert space.
        let (num_experts, hidden_size, moe_intermediate_size) = dims.bank(accessor);
        let top_k = routed_top_k(model);
        let shared_expert_intermediate_size = dims.shared.unwrap_or(0);
        let only_src = accessor.source_weights[0].0;
        // Single-owner rule (mirrors `MetalSharedFusedMoeImpl::fan_out`):
        // when the DSL claims `<moe>.shared_expert.*` leaves in the
        // manifest, the struct loader must not also load them — zero the
        // internal shared width so `load_affine`/dense load skip the
        // shared tensors and the DSL leaf accessors keep sole ownership.
        let moe_base = program.weights.path(only_src).first().cloned();
        let shared_expert_intermediate_size = match &moe_base {
            Some(b) if program.weights.has_subtree(&[b.as_str(), "shared_expert"]) => 0,
            _ => shared_expert_intermediate_size,
        };
        let (affine, gate_bits) = moe_affine_plan(program, fuf, model, &prefix, accessor);
        return FieldLoad::SharedFusedMoe {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        };
    }

    if is_gemma_router {
        assert_eq!(
            prefixes.len(),
            1,
            "GemmaRouterLayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let (hidden_size, num_experts) = BundleDims::of(accessor, program, model).router(accessor);
        // `router.proj`'s storage, resolved by the same law the fingerprint's
        // affine bit map uses — `affine_role_widths` on the accessor's
        // `{prefix}.proj` role, honoring the preset's `bits_overrides`
        // (gemma4-moe's MLX drift declares `router.proj: 8` over a 4-bit
        // model) and the MLX per-module map. The role form matters: the
        // router WeightId's registry path does not carry the `.proj` leaf,
        // so `storage_format_for_weight`'s suffix match cannot see the
        // override and would answer the section default. `None` — a
        // non-affine method (fp8), a per-module map that leaves the role
        // dense, or a verbatim model — means NO affine row and a dense
        // router load. Never defaulted: the old `unwrap_or(8)` fabricated
        // an affine row for dense-router checkpoints, and the fingerprint's
        // positive sniff then demanded `.scales` no such checkpoint ships.
        let (_, router_idx) = accessor.source_weights[0];
        let affine = model.quantization.as_ref().and_then(|qc| {
            crate::quantization::affine_role_widths(
                program,
                &qc.method,
                &format!("{prefix}.proj"),
                router_idx,
            )
        });
        return FieldLoad::GemmaRouter {
            prefix,
            num_experts,
            hidden_size,
            affine,
        };
    }

    if is_gemma_switch_glu {
        // G5b: this accessor is OP-keyed (selected by the `GemmaMoe` op),
        // not storage-gated — so the Dense/verbatim config reaches it too.
        // Op-keyed arms MUST default their storage params, never panic on
        // non-matching storage (unlike the storage-gated Marlin/Bnb4/Fp8/
        // Affine arms above, which route through storage_requirement_error).
        assert_eq!(
            prefixes.len(),
            1,
            "SwitchGluExpertsLayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let (num_experts, hidden_size, moe_intermediate_size) =
            BundleDims::of(accessor, program, model).bank(accessor);
        let top_k = routed_top_k(model);
        let only_src = accessor.source_weights[0].0;
        // Resolve at THIS accessor's unrolled layer index so MLX-native
        // mixed/dynamic quant (OptiQ) gives each layer's experts their own
        // bits — the middle layers ship 4-bit, the sensitive edge layers
        // 8-bit. emit_group_let gathers the per-layer bits across the group.
        let only_idx = accessor.source_weights[0].1;
        // The mlx-4bit checkpoint is the real runtime path (Affine gs/bits).
        // The verbatim BF16/Dense config also reaches here; it never becomes
        // the runtime-selected specialization on metal but must still codegen,
        // so default to the mlx checkpoint's g64/b4 rather than panicking
        // (same tolerance as the GemmaRouter group_size default above).
        let (group_size, bits) = match crate::quantization::storage_format_for_weight(
            program, fuf, only_src, only_idx, model,
        ) {
            crate::quantization::StorageFormat::Affine { group_size, bits } => (group_size, bits),
            // compressed-tensors INT4 experts (cyankiwi g32): pass the REAL
            // (group_size, bits) through. The marlin expert load uses
            // `cfg.group_size` to size `num_groups` and permute the scales,
            // so the mlx g64 default would misindex the g32 weight_scale
            // (half the groups) → garbage expert output.
            crate::quantization::StorageFormat::Gptq {
                group_size, bits, ..
            } => (group_size, bits),
            _ => (64, 4),
        };
        return FieldLoad::GemmaSwitchGlu {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            group_size,
            bits,
        };
    }

    if is_gpt_oss_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "GptOssMoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let (num_experts, hidden_size, intermediate_size) =
            BundleDims::of(accessor, program, model).bank(accessor);
        let top_k = routed_top_k(model);
        // Op-keyed arm (the `GptOssMoe` op selects it), so like GemmaSwitchGlu
        // it MUST tolerate the Dense/verbatim config and default its storage
        // params rather than panic — that config never becomes the
        // runtime-selected specialization on metal.
        let only_src = accessor.source_weights[0].0;
        let only_idx = accessor.source_weights[0].1;
        let affine = match crate::quantization::storage_format_for_weight(
            program, fuf, only_src, only_idx, model,
        ) {
            crate::quantization::StorageFormat::Affine { group_size, bits } => {
                Some((group_size, bits))
            }
            _ => None,
        };
        // Router gate bit-width at the gpt-oss path `{prefix}.router`
        // (NOT `.gate`): `affine_role_bits` honors preset `bits_overrides`
        // and MLX per-module maps alike; falls back to the expert bits
        // (the jesusoctavioas 2-bit checkpoint quantizes the router at
        // the preset's own width, so the fallback is the live path).
        let gate_bits = affine.map(|(_, expert_bits)| {
            let router_path = format!("{prefix}.router");
            model
                .quantization
                .as_ref()
                .and_then(|qc| {
                    crate::quantization::affine_role_bits(
                        program,
                        &qc.method,
                        &router_path,
                        only_idx,
                    )
                })
                .unwrap_or(expert_bits)
        });
        return FieldLoad::GptOssMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        };
    }

    if is_deepseek_v2_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2MoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        let cfg = read_deepseek_moe_cfg(model, BundleDims::of(accessor, program, model), accessor);
        return FieldLoad::DeepSeekV2Moe {
            prefix,
            n_routed_experts: cfg.n_routed_experts,
            n_shared_experts: cfg.n_shared_experts,
            top_k: cfg.top_k,
            moe_intermediate_size: cfg.moe_intermediate_size,
            hidden_size: cfg.hidden_size,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            use_sigmoid: cfg.use_sigmoid,
            n_expert_group: cfg.n_expert_group,
            topk_group: cfg.topk_group,
        };
    }

    if is_gated_delta_net {
        assert_eq!(
            prefixes.len(),
            1,
            "GatedDeltaNetLayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        return FieldLoad::GatedDeltaNet(prefixes.into_iter().next().unwrap());
    }

    if is_raw_tensor {
        assert_eq!(
            prefixes.len(),
            1,
            "raw tensor accessor `{}` with {} sources (expected 1)",
            accessor.name,
            prefixes.len(),
        );
        return FieldLoad::RawTensor(prefixes.into_iter().next().unwrap());
    }

    if is_embedding || is_affine_quant_embedding {
        assert_eq!(
            prefixes.len(),
            1,
            "Embedding accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        // MLX-affine int4 embedding: detect via the same
        // `storage_format_for_weight` lookup the `is_linear` arm uses
        // below. Single source by construction (assertion above).
        let only_src = accessor.source_weights[0].0;
        if let crate::quantization::StorageFormat::Affine { bits, group_size } =
            crate::quantization::storage_format_for_weight(program, fuf, only_src, None, model)
        {
            let prefix = prefixes.into_iter().next().unwrap();
            // The loader follows the ACCESSOR type the solver picked, so the
            // built field matches the struct field. A metal `AffineQuantEmbedding`
            // accessor keeps the packed bundle (`EmbeddingAffine`); a dense
            // `Embedding` accessor with Affine storage — the storage-agnostic
            // path `EmbedRefImpl` takes under spyre — dequantizes host-side.
            return if is_affine_quant_embedding {
                FieldLoad::EmbeddingAffine {
                    prefix,
                    group_size,
                    bits,
                }
            } else {
                FieldLoad::EmbeddingAffineDequant {
                    prefix,
                    group_size,
                    bits,
                }
            };
        }
        // If the accessor type is AffineQuantEmbedding but the source
        // weight is NOT Affine, the macro / impl pairing is broken —
        // MetalAffineEmbedImpl should only fire on Affine sources.
        if is_affine_quant_embedding {
            let got =
                crate::quantization::storage_format_for_weight(program, fuf, only_src, None, model);
            panic!(
                "{}",
                storage_requirement_error(
                    model,
                    accessor,
                    "Affine (MetalAffineEmbedImpl::matches should have rejected this)",
                    &format!("{got:?}"),
                )
            );
        }
        FieldLoad::Embedding(prefixes.into_iter().next().unwrap())
    } else if is_rmsnorm {
        assert_eq!(
            prefixes.len(),
            1,
            "RmsNorm accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        let eps = rms_norm_eps(model);
        FieldLoad::RmsNorm(prefixes.into_iter().next().unwrap(), eps)
    } else if is_layer_norm {
        assert_eq!(
            prefixes.len(),
            1,
            "LayerNorm accessor `{}` with {} sources",
            accessor.name,
            prefixes.len()
        );
        // Same eps source as RmsNorm — `rms_norm_eps` already accepts
        // the `layer_norm_eps` JSON key (CommandR convention) as a
        // fallback. ModernBERT writes `norm_eps` / `layer_norm_eps`,
        // both already in the fallback chain. Vision configs (Qwen2-VL
        // / Qwen2.5-VL / SigLIP) carry `vision_norm_eps` as a scalar;
        // `rms_norm_eps`'s JSON-only fallback chain misses that, so
        // vision-prefixed accessors override eps via the
        // `vision_norm_eps` scalar lookup once routed to a vision
        // loader at codegen time.
        let eps = rms_norm_eps(model);
        FieldLoad::LayerNorm(prefixes.into_iter().next().unwrap(), eps)
    } else if is_linear {
        // Tied-embedding special case: if this accessor is the
        // `lm_head` and the model's config.json has
        // `tie_word_embeddings: true`, there's no lm_head weight in
        // safetensors — its buffer is shared with `embed_tokens`.
        // HF convention: every decoder-only model that ties them
        // calls the sharing field `embed_tokens`; the macro looks
        // up that field by name.
        if accessor.name == "lm_head"
            && prefixes.len() == 1
            && (prefixes[0] == "lm_head" || prefixes[0].ends_with(".lm_head"))
            && tie_word_embeddings(model)
        {
            // P6: detect whether the source embedding is MLX-affine
            // quantized AND kept as the packed `AffineQuantEmbedding`
            // representation. The tied lm_head's `source_weights` resolves
            // to the embed_tokens weight ID — but it must follow the embed's
            // REPRESENTATION: only when the embed kept the affine triple
            // (`embed_is_affine_repr`, i.e. metal) does the tied head read
            // its `scales`/`biases`. When the embed was dequantized to a dense
            // `Embedding` (spyre/cuda host path) the tied head reuses the dense
            // weight, so `affine` stays `None`.
            let mut affine: Option<(u32, u32)> = None;
            if embed_is_affine_repr {
                for (wid, _idx) in &accessor.source_weights {
                    if let crate::quantization::StorageFormat::Affine { bits, group_size } =
                        crate::quantization::storage_format_for_weight(
                            program, fuf, *wid, None, model,
                        )
                    {
                        affine = Some((group_size, bits));
                        break;
                    }
                }
            }
            return FieldLoad::LinearTiedToEmbedding {
                embed_ident: syn::Ident::new("embed_tokens", proc_macro2::Span::call_site()),
                affine,
            };
        }
        // MLX-affine int4 storage: every source weight resolves to
        // `StorageFormat::Affine`. Single-source only — see the doc
        // comment on `FieldLoad::LinearAffine` for why concat is
        // excluded here. A mismatched fuse (e.g. one Affine, one
        // Dense) is an upstream weights-manifest authoring error and
        // panics so it surfaces at compile time, not at load time.
        let mut affine_params: Option<(u32, u32)> = None;
        let mut affine_in_features: Option<u32> = None;
        let mut any_non_affine = false;
        for (wid, idx) in &accessor.source_weights {
            // Resolve at the source weight's unrolled layer index — MLX
            // mixed/dynamic quant (OptiQ) varies bits per numbered layer.
            // For a fused accessor every source shares the same layer, so
            // any source's index is representative.
            let fmt =
                crate::quantization::storage_format_for_weight(program, fuf, *wid, *idx, model);
            match fmt {
                crate::quantization::StorageFormat::Affine { bits, group_size } => {
                    if let Some((eg, eb)) = affine_params
                        && (eg, eb) != (group_size, bits)
                    {
                        panic!(
                            "{}",
                            storage_requirement_error(
                                model,
                                accessor,
                                "uniform Affine (group_size, bits) across fused sources",
                                &format!(
                                    "mismatched Affine fuse ({eg}, {eb}) vs ({group_size}, {bits})"
                                ),
                            )
                        );
                    }
                    affine_params = Some((group_size, bits));

                    // Resolve the declared in_features (K) from the arch
                    // manifest so the loader can align the on-disk packed
                    // `.weight` width against `ceil(K*bits/32)` and reject a
                    // checkpoint quantized at different bits/group_size than
                    // this build's preset. Fused sources (gate_up / qkv) share
                    // K; a mismatch is an upstream manifest authoring error.
                    let segments = program.weights.path(*wid);
                    let dotted = segments.join(".");
                    let shape = manifest.lookup(segments).unwrap_or_else(|| {
                        panic!(
                            "accessor `{}`: weight `{dotted}` missing from weights manifest; \
                             required for Affine in_features alignment check",
                            accessor.name,
                        )
                    });
                    if shape.len() != 2 {
                        panic!(
                            "accessor `{}`: Affine source weight `{dotted}` has shape len {} \
                             (expected 2 for a matmul)",
                            accessor.name,
                            shape.len(),
                        );
                    }
                    let in_f = crate::shape::eval_closed_dim(&shape[0], &model.bounds)
                        .unwrap_or_else(|| {
                            panic!(
                                "accessor `{}`: can't resolve in_features for `{dotted}`",
                                accessor.name,
                            )
                        }) as u32;
                    if let Some(existing) = affine_in_features
                        && existing != in_f
                    {
                        panic!(
                            "accessor `{}` fuses Affine sources with mismatched in_features \
                             ({existing} then {in_f})",
                            accessor.name,
                        );
                    }
                    affine_in_features = Some(in_f);
                }
                _ => any_non_affine = true,
            }
        }
        if let Some((group_size, bits)) = affine_params {
            if any_non_affine {
                panic!(
                    "{}",
                    storage_requirement_error(
                        model,
                        accessor,
                        "all-Affine fused sources",
                        "a mix of Affine and non-Affine source weights (no unified Linear arm \
                         for mixed storage)",
                    )
                );
            }
            let in_features = affine_in_features.expect(
                "affine_params set implies at least one Affine source resolved in_features",
            );
            if prefixes.len() == 1 {
                // A weight the manifest packs from several on-disk projections' rows.
                let dotted = program.weights.path(accessor.source_weights[0].0).join(".");
                if let Some(parts) = manifest.packed_concats.get(&dotted) {
                    let prefix = &prefixes[0];
                    let stem = prefix.strip_suffix(dotted.as_str()).unwrap_or_else(|| {
                        panic!(
                            "accessor `{}`: prefix `{prefix}` does not end in its packed weight \
                             `{dotted}`",
                            accessor.name,
                        )
                    });
                    return FieldLoad::LinearAffineRowConcat {
                        prefixes: parts.iter().map(|p| format!("{stem}{p}")).collect(),
                        group_size,
                        bits,
                        in_features,
                    };
                }
                return FieldLoad::LinearAffine {
                    prefix: prefixes.into_iter().next().unwrap(),
                    group_size,
                    bits,
                    in_features,
                };
            }
            // Multi-source fuse (gate_up_proj / qkv_proj on mlx-community
            // 4bit repos): CPU-dequant per prefix, byte-concat along
            // dim 0 at load time. Result is one Dense BF16 weight that
            // FusedGateUpSiluMul / FusedQkv* claim normally.
            return FieldLoad::LinearAffineConcat {
                prefixes,
                group_size,
                bits,
                in_features,
            };
        }

        // NVFP4 int4 storage: every source weight resolves to
        // `StorageFormat::Nvfp4`. Single-source → LinearNvfp4; fused
        // (qkv_proj / gate_up_proj) → LinearNvfp4Concat. Mixed Nvfp4 /
        // non-Nvfp4 fuse is an upstream manifest authoring error and
        // panics at compile time.
        let mut nvfp4_gs: Option<u32> = None;
        let mut any_non_nvfp4 = false;
        for (wid, _idx) in &accessor.source_weights {
            match crate::quantization::storage_format_for_weight(program, fuf, *wid, None, model) {
                crate::quantization::StorageFormat::Nvfp4 { group_size } => {
                    if let Some(eg) = nvfp4_gs
                        && eg != group_size
                    {
                        panic!(
                            "accessor `{}` fuses Nvfp4 sources with mismatched group_size \
                             ({eg} vs {group_size})",
                            accessor.name,
                        );
                    }
                    nvfp4_gs = Some(group_size);
                }
                _ => any_non_nvfp4 = true,
            }
        }
        if let Some(group_size) = nvfp4_gs {
            if any_non_nvfp4 {
                panic!(
                    "accessor `{}` fuses Nvfp4 and non-Nvfp4 source weights — \
                     the macro can't emit a unified Linear arm for mixed storage",
                    accessor.name,
                );
            }
            if prefixes.len() == 1 {
                return FieldLoad::LinearNvfp4 {
                    prefix: prefixes.into_iter().next().unwrap(),
                    group_size,
                };
            }
            return FieldLoad::LinearNvfp4Concat {
                prefixes,
                group_size,
            };
        }

        if prefixes.len() == 1 {
            // `kind: "raw_linear"` opt-in (per the per-arch
            // weights manifest): the underlying tensor is an
            // `nn.Parameter`, not an `nn.Linear`. Looked up off
            // the DSL-side path of the source weight (same
            // convention as `manifest.lookup`). Fused-concat
            // RawLinear is not a real PyTorch shape — only the
            // single-source branch routes here.
            let segments = program.weights.path(accessor.source_weights[0].0);
            if matches!(
                manifest.kind(segments),
                crate::weights_manifest::ManifestEntryKind::RawLinear
            ) {
                return FieldLoad::RawLinear(prefixes.into_iter().next().unwrap());
            }
            FieldLoad::LinearDense(prefixes.into_iter().next().unwrap())
        } else {
            FieldLoad::LinearConcat(prefixes)
        }
    } else {
        // Unknown weight type — fall back to raw Embedding-shaped
        // load. New weight types (LayerNorm with bias, etc.) land
        // as new arms here alongside their Impl's `rust_type`.
        assert_eq!(
            prefixes.len(),
            1,
            "unhandled weight type `{ty}` for accessor `{}`",
            accessor.name,
        );
        FieldLoad::Embedding(prefixes.into_iter().next().unwrap())
    }
}

pub(crate) fn rms_norm_eps(model: &ModelParams) -> f32 {
    // HF configs carry `rms_norm_eps` (RMSNorm convention used by
    // Llama/Qwen2/Mistral/etc.) or `layer_norm_eps` (CohereLayerNorm
    // convention used by CommandR). Both name the same numerical
    // role — the eps inside the row-normalization kernel — so the
    // RmsNorm-typed weight loader accepts either. Reads the parsed
    // `scalars` table (every numeric config field, populated from the
    // NORMALIZED view — `normalize_hf_config` hoists nested
    // `text_config`) rather than re-parsing `source_path`: a raw
    // re-read sees only top-level keys, so a verbatim VL-wrapper
    // config (Qwen3.5 / Qwen3.5-MoE) would silently fall back to
    // 1e-5 while the checkpoint uses 1e-6.
    let read = |key: &str| model.scalars.get(key).map(|&x| x as f32);
    read("rms_norm_eps")
        .or_else(|| read("layer_norm_eps"))
        .or_else(|| read("vision_norm_eps"))
        .unwrap_or(1e-5)
}

/// Runtime RMSNorm gain offset for zero-centered (Gemma-style) norms.
/// When the config sets `"rms_norm_zero_centered": true` (Gemma2/3,
/// Qwen3.5/3.6/Next — input/post/final layernorms + per-head q/k norm
/// store gains zero-centered, effective gain `1 + weight`), every RMSNorm
/// op carries it in its gain convention (`to_wavefront`'s `RmsNorm` arm) and
/// `NORM_WEIGHT_OFFSET` bakes it for the kernels that read no convention. The
/// GDN gated RMSNorm uses its own kernel and is unaffected. Distinct
/// from the GGUF-only `norm_weight_offset` (a load-time SUBTRACTION that
/// un-bakes a converter's pre-applied constant). Default 0.0.
// `pub(crate)`: also consulted by `MetalRopeAppendNormedImpl::applies_to`
// (the synth norm prologue is only bit-correct for offset-0 models).
pub(crate) fn norm_weight_runtime_offset(model: &ModelParams) -> f32 {
    // Read the parsed `bounds` table (booleans land there as 0/1, and
    // `apply_arch_semantic_defaults` inserts the flag for the
    // Qwen3.5/3.6 family whose verbatim HF configs never carry it)
    // rather than re-parsing `source_path`, which sees only literal
    // top-level keys.
    if model
        .bounds
        .get("rms_norm_zero_centered")
        .copied()
        .unwrap_or(0)
        != 0
    {
        // The standard kernels fold `weight + 1.0`, assuming the
        // checkpoint stores the gain ZERO-CENTERED (`weight = gain - 1`,
        // the HF convention). But `mlx_lm.convert` "sanitizes" these
        // norms by pre-adding 1.0 (storing the full gain `1 + weight`),
        // so for MLX-affine checkpoints folding another +1.0 double-
        // counts the offset → gain ≈1.7× too large → garbage. Verified
        // on `mlx-community/Qwen3.5-0.8B-MLX-4bit` (input_layernorm.weight
        // is exactly +1.0 vs the official bf16). Use 0.0 there so the
        // kernel reads the pre-offset gain as-is. (GDN gated norm uses a
        // separate kernel with no offset and is unaffected either way.)
        let is_mlx_affine = matches!(
            model.quantization.as_ref().map(|qc| &qc.method),
            Some(crate::quantization::QuantMethod::Affine { .. })
        );
        if is_mlx_affine { 0.0 } else { 1.0 }
    } else {
        0.0
    }
}

pub fn tie_word_embeddings(model: &ModelParams) -> bool {
    // `ModelParams::tie_word_embeddings` is populated by the config
    // loader from the HF JSON; read directly rather than re-parsing
    // the file here.
    model.tie_word_embeddings
}

/// The routed experts' input width an ungated bank is packed at, probed at the last MoE
/// layer: the tensor spellings that may carry it, and the packed column count expected of
/// each. The one shape that tells a LatentMoE (experts on `moe_latent_size`) from a plain
/// one (experts on `hidden_size`) when both report one architecture string (Nemotron-H:
/// `NemotronHForCausalLM`) and share every other probed width — the latent projections only
/// one of them loads are not visible to the other.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ExpertInputProbe {
    /// The spellings of the width-carrying tensor, in probe order: the pre-stacked
    /// `switch_mlp.fc1` (mlx `SwitchMLP`'s own), then the per-expert names a conversion
    /// keeping nvidia's per-expert layout owns (`experts.0.fc1`, `experts.0.up_proj`) —
    /// the same chain `load_expert_projection` resolves. A checkpoint ships exactly one.
    tensors: Vec<String>,
    packed_cols: usize,
}

/// [`ExpertInputProbe`] for the model's last ungated `moe_block`: the experts' width is the op's
/// output (`sig_moe`), stored dense or MLX-affine at `fc1`'s own resolved width. `None` for a
/// model without an ungated bank.
fn expert_input_probe(
    program: &Program,
    fuf: &Fuf,
    model: &ModelParams,
) -> Option<ExpertInputProbe> {
    use scratchy_subtile::subtile_ir::ExpertBody;
    let body = crate::config::expert_body(&model.expert_activation).ok()?;
    if !matches!(body, ExpertBody::Ungated(_)) {
        return None;
    }
    let (node, id, index, storage) = fuf
        .nodes
        .iter()
        .rev()
        .filter(|n| n.op == OpKind::Moe)
        .find_map(|n| {
            n.inputs.iter().find_map(|i| match i {
                FufInput::Weight { id, index, storage } => Some((n, *id, *index, storage)),
                _ => None,
            })
        })?;
    let width = node.outputs.first()?.last()?.clone();
    let k = *crate::weight_vocab::eval_shape_with(&vec![width], &model.bounds)?.first()? as usize;
    let prefix = safetensors_prefix(program.weight_naming(), &model.arch, id, index);
    let packed_cols = match storage {
        crate::quantization::StorageFormat::Dense => k,
        crate::quantization::StorageFormat::Affine { bits, .. } => {
            let fc1 = model
                .quantization
                .as_ref()
                .and_then(|qc| {
                    crate::quantization::affine_moe_expert_bits(
                        program, &qc.method, &prefix, index, body,
                    )
                })
                .map_or(*bits, |b| b.up());
            (k * fc1 as usize).div_ceil(32)
        }
        _ => return None,
    };
    Some(ExpertInputProbe {
        tensors: vec![
            format!("{prefix}.switch_mlp.fc1.weight"),
            format!("{prefix}.experts.0.fc1.weight"),
            format!("{prefix}.experts.0.up_proj.weight"),
        ],
        packed_cols,
    })
}

/// Emit a `fingerprint_matches(gw)` method body — the per-variant
/// check the arch-level dispatcher uses to auto-detect which
/// compiled model a runtime `GpuWeights` corresponds to. All values
/// are baked at macro-expansion time from `model.bounds` +
/// `model.quantization`; the runtime cost is a handful of
/// `gw.contains` / `gw.tensor_info` lookups.
///
/// Checks:
/// 1. **Embedding shape** matches `(vocab_size, hidden_size)` from
///    config.json. Rules out arches with a different width or vocab.
/// 2. **Last-layer tensor present** (`model.layers.{N-1}.self_attn.q_proj.<suffix>`)
///    where `suffix` is `qweight` for AWQ/GPTQ, `weight` for dense.
/// 3. **Next-layer tensor absent** (same name with layer `N`). Rules
///    out larger compiled variants with the same suffix.
/// 4. **Opposite-suffix tensor absent**. Rules out the other
///    dense-vs-quant twin (GPTQ vs AWQ share the qweight suffix —
///    they're distinguished by check #5).
/// 5. **Quant-format shape check**. When the compiled variant is
///    AWQ or GPTQ, the `qweight` shape discriminates:
///    `[K, N/8]` is AWQ, `[K/8, N]` is GPTQ. For the q_proj
///    specifically both dims are `hidden_size`, so checking
///    `shape[0] == hidden_size` (AWQ) vs `shape[0] == hidden_size/8`
///    (GPTQ) is enough. Without this check a GPTQ model would
///    fingerprint-match an AWQ variant of the same arch+width and
///    the emitted `load_awq` loader would panic in
///    `awq_to_marlin_zero_points`.
/// 6. **RoPE base frequency** matches the `rope_theta` this variant
///    baked, when the config declared one and the caller supplied
///    one. The only discriminator for variant groups that are
///    identical on every axis above AND declare no `rope_scaling`
///    (granite 3.0 vs 3.1 vs 3.3; phi-4 vs phi-4-reasoning) — see
///    the `rope_theta_check` comment for why guessing wrong here is
///    silent rather than loud.
/// 7. **Per-role affine bit widths**. For MLX-affine variants, every
///    quantized weight under the decoder root must be packed at the
///    width this variant's baked bit map (`bits` + `bits_overrides`
///    / `per_module`) claims for its role. The only discriminator
///    for variants differing solely in that map — qwen3.5 vs
///    qwen3.6 35B-A3B, whose gates are 4- vs 8-bit with every other
///    axis identical. See the `affine_bit_map_gate` comment.
/// 8. **Experts' input width** of an ungated bank ([`ExpertInputProbe`]) —
///    LatentMoE vs plain MoE under one architecture string, probed at
///    whichever spelling of the width-carrying tensor the checkpoint ships
///    (pre-stacked `switch_mlp.fc1` or a per-expert name).
fn emit_fingerprint_check(
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    tp_world_size: u8,
    affine_rows: &[(String, u32, u32)],
    expert_input: Option<&ExpertInputProbe>,
) -> TokenStream {
    let num_hidden_layers = model.text.layers.get() as u64;
    let hidden_size = *model
        .bounds
        .get("hidden_size")
        .unwrap_or_else(|| panic!("model `{}` missing `hidden_size`", model.source_stem));
    let vocab_size = *model
        .bounds
        .get("vocab_size")
        .unwrap_or_else(|| panic!("model `{}` missing `vocab_size`", model.source_stem));
    // The embedding table's rows: `vocab_size`, unless the model declares a wider table than its
    // logits (`embed_vocab_size`: a draft head that drafts over a prefix of its target's
    // vocabulary embeds every token).
    let embed_rows = model
        .bounds
        .get("embed_vocab_size")
        .copied()
        .unwrap_or(vocab_size);

    // Pick the on-disk tensor suffix per compiled variant's
    // `quantization_config`. AutoGPTQ + AWQ both ship `.qweight`;
    // compressed-tensors INT4 ships `.weight_packed` with the axes
    // transposed. Dense ships `.weight`. bitsandbytes ships U8
    // `.weight` alongside a sibling `.weight.absmax` that's unique
    // to its storage layout — use that as the positive sniff so
    // it's disjoint from dense bf16 `.weight`. The `opposite_suffix`
    // is the negative check: if a compiled dense variant sees
    // `.qweight`, that's a quant model in disguise and the
    // fingerprint should miss.
    let (suffix, opposite_suffix) = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Gptq {
            layout: crate::quantization::GptqLayout::WeightPacked,
            ..
        }) => ("weight_packed", "weight"),
        Some(crate::quantization::QuantMethod::Bnb4 { .. }) => ("weight.absmax", "qweight"),
        Some(crate::quantization::QuantMethod::Fp8 { .. }) => ("weight_scale", "qweight"),
        // GGUF ships the same `.weight` suffix as dense (the loader
        // stores the quantized tensor under the HF-style name) — the
        // disambiguator vs Dense is `gw.is_gguf()` (added below in
        // qweight_shape_gate) plus the absence of fp8/bnb4 markers
        // at `.weight_scale` / `.weight.absmax`.
        Some(crate::quantization::QuantMethod::Ggml) => ("weight", "qweight"),
        // MLX-affine ships `.weight` (U32 packed) + `.scales` + `.biases`
        // sibling triple. Dense's same `.weight` suffix is disjoint
        // because the affine `tensor_info` shape gate above keys on
        // `hidden_size / pack_factor` rather than `hidden_size`.
        Some(crate::quantization::QuantMethod::Affine { .. }) => ("weight", "qweight"),
        // NVFP4 ships `.weight` (U8 packed E2M1) + `.weight_scale`
        // (fp8 block scale) + `.weight_scale_2` (f32 global). The
        // `.weight_scale_2` global is unique to NVFP4 — FP8 has only
        // `.weight_scale`, dense/AWQ/GPTQ have neither — so it's the
        // positive sniff that disambiguates NVFP4 from every other
        // format. `.qweight` (absent) is the negative.
        Some(crate::quantization::QuantMethod::Nvfp4 { .. }) => ("weight_scale_2", "qweight"),
        Some(_) => ("qweight", "weight"),
        None => ("weight", "qweight"),
    };

    let last_layer = num_hidden_layers.saturating_sub(1);
    // Per-arch decoder root for fingerprint-tensor names — the same
    // resolution [`safetensors_prefix`] keys the loader by, so the
    // fingerprint's `{dec_root}.embed_tokens.weight` /
    // `{dec_root}.layers.N.*` lookups name what the loader reads.
    let dec_root = model.arch.decoder_root();
    // Pick a layered tensor that ACTUALLY EXISTS ON DISK to use as the
    // fingerprint sniff. Packed parents come first (Phi-3 ships
    // `self_attn.qkv_proj.weight` on disk; ModernBERT ships
    // `attn.Wqkv.weight`; the per-slice virtual entries get carved at
    // load time, AFTER fingerprint matching). Then MLA archs (DeepSeek
    // V3) which use `q_a_proj` instead of `q_proj`. Then plain
    // `self_attn.q_proj` (llama-style decoder fleet) and finally
    // `attn.q_proj` (encoder-style without `self_` prefix). Without
    // this, the fingerprint misses and scratchy returns Ok(None) even
    // when it has a compiled variant for this arch.
    let fp_leaf_owned: String = if let Some(parent) = manifest
        .packed_splits
        .iter()
        .find(|(_, children)| {
            children
                .iter()
                .any(|c| c == "self_attn.q_proj" || c == "attn.q_proj")
        })
        .map(|(k, _)| k.clone())
    {
        parent
    } else if manifest.entries.contains_key("self_attn.q_a_proj") {
        "self_attn.q_a_proj".to_string()
    } else if manifest.entries.contains_key("self_attn.q_proj") {
        "self_attn.q_proj".to_string()
    } else if manifest.entries.contains_key("attn.q_proj") {
        "attn.q_proj".to_string()
    } else {
        "self_attn.q_proj".to_string()
    };
    // On disk under the arch's renames (nemotron-h: `attn.q_proj` →
    // `mixer.q_proj`), sniffed at the LAST layer that has it: a declared
    // block schedule (`attention_layers`) names it — nemotron-h's last
    // layer is a MoE block with no attention projection.
    let fp_leaf_owned =
        crate::classified::rename_dotted_under(&model.arch.weight_leaf_renames, &fp_leaf_owned);
    let fp_leaf: &str = fp_leaf_owned.as_str();
    let fp_layer = model
        .layer_sets
        .get("attention_layers")
        .and_then(|set| set.members().last().copied())
        .unwrap_or(last_layer);
    let last_tensor = under_root(&dec_root, &format!("layers.{fp_layer}.{fp_leaf}.{suffix}"));
    // MLX-affine `.scales` sibling of `last_tensor`. The affine group-size
    // gate must probe a layer that ACTUALLY HAS `fp_leaf` (= `self_attn.q_proj`).
    // It MUST NOT hardcode `layers.0`: hybrid arches (Qwen3.5 Gated-DeltaNet,
    // Mamba/Jamba) make layer 0 a linear-attention layer with NO `self_attn.q_proj`,
    // so `layers.0.self_attn.q_proj.scales` is absent there and the gate would
    // reject every variant → `ArchNotSupported`. `fp_layer` is the same layer
    // the present-check (`last_tensor`) already requires, so its `.scales`
    // sibling is guaranteed present for the affine checkpoint.
    let last_scales_tensor = under_root(&dec_root, &format!("layers.{fp_layer}.{fp_leaf}.scales"));
    // Whether this variant reads `fp_leaf` MLX-affine at all (a per-module map may leave it dense).
    let fp_leaf_affine = affine_rows.iter().any(|(name, _, _)| {
        let at = |layer: &str| under_root(&dec_root, &format!("layers.{layer}.{fp_leaf}.weight"));
        *name == at(&fp_layer.to_string()) || *name == at("*")
    });
    let one_past_tensor = under_root(
        &dec_root,
        &format!("layers.{num_hidden_layers}.{fp_leaf}.{suffix}"),
    );
    // The format-marker probes below read `fp_leaf`'s siblings at the FIRST layer that has it:
    // layer 0 on a uniform decoder, the first `attention_layers` member on a hybrid one (layer 0
    // is a Gated-DeltaNet / Mamba block there, and a probe of an absent leaf is inert).
    let marker_layer = model
        .layer_sets
        .get("attention_layers")
        .and_then(|set| set.members().first().copied())
        .unwrap_or(0);
    let opposite_tensor = under_root(
        &dec_root,
        &format!("layers.{marker_layer}.{fp_leaf}.{opposite_suffix}"),
    );
    // BNB4 checkpoints ship the U8-packed nibbles at `.weight`
    // (same suffix as dense bf16 weights) with a sibling
    // `.weight.absmax` that's unique to bitsandbytes. Dense + AWQ
    // + GPTQ + CT variants must reject when absmax is present; the
    // BNB4 variant itself uses `.weight.absmax` as the positive
    // sniff and doesn't need the rejection.
    let bnb4_exclusion = matches!(
        model.quantization.as_ref().map(|qc| &qc.method),
        Some(crate::quantization::QuantMethod::Bnb4 { .. })
    );
    let bnb4_marker_tensor = under_root(
        &dec_root,
        &format!("layers.{marker_layer}.{fp_leaf}.weight.absmax"),
    );
    let bnb4_marker_tensor = bnb4_marker_tensor.as_str();
    // FP8 checkpoints ship `.weight` (FP8E4M3 bytes — same suffix
    // as dense bf16) alongside a sibling `.weight_scale`. Dense /
    // AWQ / native-GPTQ / BNB4 variants must reject when
    // `.weight_scale` is present; the FP8 variant itself keys off
    // this tensor in its suffix-based checks and doesn't need the
    // rejection. Compressed-tensors INT4 (`GptqLayout::WeightPacked`)
    // ALSO ships `.weight_scale` (as the group-scale tensor) — its
    // own qweight-shape gate on `.weight_packed` already makes it
    // disjoint from FP8, so skip the fp8 exclusion for it to avoid
    // false-rejecting CT-INT4 checkpoints.
    // NVFP4 also carries `.weight_scale` (its fp8-e4m3 block scale), so
    // it must be excluded from the FP8-marker rejection or it would
    // reject its own checkpoint. Its positive sniff is `.weight_scale_2`,
    // which FP8 lacks, so the two stay disjoint.
    let fp8_exclusion = matches!(
        model.quantization.as_ref().map(|qc| &qc.method),
        Some(crate::quantization::QuantMethod::Fp8 { .. })
            | Some(crate::quantization::QuantMethod::Nvfp4 { .. })
            | Some(crate::quantization::QuantMethod::Gptq {
                layout: crate::quantization::GptqLayout::WeightPacked,
                ..
            })
    );
    let fp8_marker_tensor = under_root(
        &dec_root,
        &format!("layers.{marker_layer}.{fp_leaf}.weight_scale"),
    );
    let fp8_marker_tensor = fp8_marker_tensor.as_str();
    // MLX-affine marker exclusion. `mlx_lm.convert` ships every
    // quantized linear as a `.{weight,scales,biases}` triple. The
    // packed `.weight` (U32) shares its suffix with dense bf16 and
    // gptq's `.weight_packed`, so the only disjoint marker is the
    // `.scales` sibling. Dense / AWQ / native-GPTQ / CT-INT4 / BNB4 /
    // FP8 variants must reject when this sibling exists on layer.0
    // q_proj — otherwise an MLX 4bit checkpoint with a dense
    // `embed_tokens` (the untied case, e.g.
    // `mlx-community/Meta-Llama-3-8B-Instruct-4bit`) fingerprint-
    // matches the dense variant and silently corrupts decoding.
    // The affine variant itself doesn't need this rejection — it
    // keys off the embed shape (or, in the untied dense-embed case,
    // off the same `.scales` sibling on q_proj via the per-Impl
    // load-time gate).
    let mlx_affine_exclusion = matches!(
        model.quantization.as_ref().map(|qc| &qc.method),
        Some(crate::quantization::QuantMethod::Affine { .. })
    );
    let mlx_marker_tensor = under_root(
        &dec_root,
        &format!("layers.{marker_layer}.{fp_leaf}.scales"),
    );
    let mlx_marker_tensor = mlx_marker_tensor.as_str();

    // Each marker exclusion is a compile-time bool, so fold it in HERE rather
    // than emit `if !#exclusion && gw.contains(..)`: an excluded variant emits
    // nothing, a non-excluded one emits a plain `if gw.contains(..)`. Keeps the
    // generated fingerprint free of `!true`/`!false` (clippy nonminimal_bool).
    let bnb4_marker_check = if bnb4_exclusion {
        quote! {}
    } else {
        quote! { if gw.contains(#bnb4_marker_tensor) { return false; } }
    };
    let fp8_marker_check = if fp8_exclusion {
        quote! {}
    } else {
        quote! { if gw.contains(#fp8_marker_tensor) { return false; } }
    };
    let mlx_marker_check = if mlx_affine_exclusion {
        quote! {}
    } else {
        quote! { if gw.contains(#mlx_marker_tensor) { return false; } }
    };

    let hidden_lit = proc_macro2::Literal::usize_unsuffixed(hidden_size as usize);
    // GGUF's on-disk loader (`GgufGpuWeights::load`) pre-shards
    // `ShardDim0` tensors — including `embed_tokens` — at file-read
    // time.
    // At tp>1 the per-rank embed shape is `[vocab_size / tp, hidden]`,
    // so the fingerprint's vocab literal must match the sharded dim-0.
    // Safetensors variants keep the full tensor in memory (sharding
    // happens inside the codegen-emitted `_sharded` load helpers AFTER
    // `fingerprint_matches` runs), so their vocab literal stays whole.
    let vocab_for_fp = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Ggml) if tp_world_size > 1 => {
            embed_rows / (tp_world_size as u64)
        }
        _ => embed_rows,
    };
    let vocab_lit = proc_macro2::Literal::usize_unsuffixed(vocab_for_fp as usize);

    // HF-config disambiguator: variants of the same arch that share
    // on-disk tensor shapes (Phi-3-mini-4k vs Phi-3.5-mini-128k) only
    // differ in `rope_scaling.type` / its short/long factors. Both
    // are covered by `rope_scaling_type` + `rope_scaling_hash`
    // (below); `max_position_embeddings` is NOT checked because
    // mlx-community 4bit publishes a narrower context window than the
    // upstream base (`32768` vs `131072` on Qwen2.5-1.5B), and the
    // value doesn't drive forward-fn codegen — it's a runtime KV
    // sizing input. See `HfFingerprint`'s doc-comment.
    let max_pos_check: TokenStream = quote! {};
    let rope_scaling_expected: Option<&'static str> = match &model.rope_scaling {
        Some(crate::config::RopeScaling::Llama3 { .. }) => Some("llama3"),
        Some(crate::config::RopeScaling::LongRope { .. }) => Some("longrope"),
        Some(crate::config::RopeScaling::Yarn { .. }) => Some("yarn"),
        // Qwen2-VL / Qwen2.5-VL: `extract_rope_scaling` doesn't fold
        // `"mrope"` into a `RopeScaling` variant (it doesn't drive the
        // text-side `RotaryCache` — `mrope_section` lives on
        // `CanonicalParams::MROPE_SECTION` instead), but the fingerprint
        // still has to expect `Some("mrope")` from the live HF config
        // or the variant will reject its own checkpoint.
        None if model.mrope_section.is_some() => Some("mrope"),
        None => None,
    };
    let rope_scaling_check: TokenStream = match rope_scaling_expected {
        Some(kind) => quote! {
            // Manifest declares a non-trivial scaling; reject a
            // checkpoint whose HF config has a different (or absent)
            // rope_scaling.type.
            match hf.rope_scaling_type {
                Some(#kind) => {}
                None => {} // permissive when caller didn't supply it
                Some(_) => return false,
            }
        },
        None => quote! {
            // Manifest has no scaling; reject a checkpoint whose HF
            // config declares one (longrope / llama3). The
            // alphabetically-earlier variant's fingerprint would
            // otherwise win for a scaled checkpoint and bake the
            // wrong RoPE into its Weights.
            if hf.rope_scaling_type.is_some() {
                return false;
            }
        },
    };

    // RoPE base-frequency discriminator: reject a checkpoint whose declared
    // `rope_theta` isn't the one this variant BAKED.
    //
    // ⛔ THE ONLY THING SEPARATING SEVERAL IN-TREE VARIANT GROUPS. `rope_theta`
    // becomes a compile-time literal in `RotaryCache::new_from_stream` (see the
    // `rotary_load` emission), so matching the wrong variant silently applies the
    // wrong base frequency — every shape agrees, the load succeeds, and the model
    // emits fluent-looking token soup. `granite-3.0-2b-instruct` (10⁴) and
    // `granite-3.1-2b-instruct` (5×10⁶) are identical on every other axis here,
    // as are `granite-3.{0,1,3}-{2b,8b}-{base,instruct}` and `phi-4` (2.5×10⁵) vs
    // `phi-4-reasoning` (5×10⁵). None of them declare `rope_scaling`, so neither
    // the type nor the hash check above can tell them apart; before this gate the
    // winner was whichever `inventory` registered first.
    //
    // The gate compares against the θ this variant COMPUTES WITH — the config's typed
    // `RopeTheta`, resolved once at load (stated, hoisted from `rope_parameters`, aliased, or the
    // arch's own default for a config that omits it: llama-2-70b's 10⁴ is `LlamaConfig`'s). A
    // checkpoint that spells a different θ is not this variant; one whose config states none
    // passes `None` and stays permissive (`HfFingerprint::rope_theta`).
    //
    // The tolerance is relative and computed HERE, at expansion, so the emitted
    // code is a single compare against two literals — and so a config spelling
    // `1000000` where the checkpoint spells `1000000.0` cannot trip it. It is far
    // tighter than any real theta gap (the closest in-tree pair differs by 2×).
    let rope_theta_check: TokenStream = {
        let theta = model.text.rope_theta.get();
        let theta_lit = proc_macro2::Literal::f64_unsuffixed(theta);
        let tol_lit = proc_macro2::Literal::f64_unsuffixed(theta.abs() * 1e-9);
        quote! {
            // Permissive on `None` (see `HfFingerprint::rope_theta`): a caller
            // that cannot trust its source's value — the cuda worker for GGUF —
            // leaves the variant's baked value authoritative.
            if let Some(theta) = hf.rope_theta
                && (theta - #theta_lit).abs() > #tol_lit
            {
                return false;
            }
        }
    };

    // Content-hash discriminator: reject checkpoints whose
    // `rope_scaling` JSON (factor vectors included) doesn't
    // bit-identically match the manifest's. Discriminates
    // Phi-3.5-mini vs Phi-3-mini-128k, Phi-4-mini-instruct vs
    // Phi-4-mini-reasoning, etc. — same type + max_pos, different
    // short/long_factor values. Permissive when the executor
    // didn't compute the hash (`None`), strict otherwise.
    let rope_scaling_hash_check: TokenStream = match model.rope_scaling_hash {
        Some(hash) => {
            let lit = proc_macro2::Literal::u64_unsuffixed(hash);
            quote! {
                if let Some(h) = hf.rope_scaling_hash
                    && h != #lit
                {
                    return false;
                }
            }
        }
        None => quote! {},
    };

    // Per-format qweight-shape gate. AWQ/GPTQ/CT all pack 4-bit
    // weights but into different axis orders — this gate is the
    // only way to distinguish compiled variants of the same
    // arch+size that differ only in quantization method/layout.
    //
    //   AWQ            `.qweight`       [K, N/8]     → shape[0] == hidden
    //   GPTQ  native   `.qweight`       [K/8, N]     → shape[0] == hidden/8
    //   GPTQ  CT       `.weight_packed` [N, K/8]     → shape[0] == hidden (== N for q_proj, N==K)
    //
    // For q_proj specifically `N == K == hidden_size`, so the AWQ
    // and CT gates coincide on `shape[0] == hidden_size`. The
    // `suffix` picked above (`qweight` vs `weight_packed`) is what
    // makes them disjoint — a CT model doesn't ship `.qweight`.
    let qweight_shape_gate: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Awq { .. }) => {
            let k_lit = hidden_lit.clone();
            quote! {
                match gw.tensor_info(#last_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[0] == #k_lit => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Gptq {
            layout: crate::quantization::GptqLayout::Qweight,
            ..
        }) => {
            let k_packed = proc_macro2::Literal::usize_unsuffixed(hidden_size as usize / 8);
            quote! {
                match gw.tensor_info(#last_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[0] == #k_packed => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Gptq {
            layout: crate::quantization::GptqLayout::WeightPacked,
            ..
        }) => {
            // compressed-tensors `.weight_packed` is `[N, K/8]` (8 int4
            // per int32 along the input axis K). Gate on the PACKED INPUT
            // dim `shape[1] == hidden_size / 8`, not `shape[0] == hidden`:
            // the q_proj output `N` equals hidden_size only for square
            // attention (llama-style), but NOT for GQA archs whose
            // `n_heads * head_dim != hidden` (e.g. Gemma-4-MoE q_proj is
            // `[4096, 2816/8]` = `[16*256, 352]`, N=4096 != hidden=2816).
            // The input `K == hidden_size` holds for q_proj on every arch,
            // so `K/8` is the portable, correct discriminator.
            let k_packed = proc_macro2::Literal::usize_unsuffixed(hidden_size as usize / 8);
            quote! {
                match gw.tensor_info(#last_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[1] == #k_packed => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Bnb4 { .. }) => {
            // bitsandbytes fingerprint sniffs the `.weight.absmax`
            // sibling itself — presence is sufficient to distinguish
            // from dense bf16 `.weight`. No shape check: absmax
            // length varies with blocksize (64 default) and is
            // per-model, not worth baking into the compile-time
            // fingerprint.
            quote! {}
        }
        Some(crate::quantization::QuantMethod::Fp8 { .. }) => {
            // FP8 fingerprint sniffs `.weight_scale` on the first
            // q_proj — presence + shape distinguishes FP8 from every
            // other variant. The `bnb4_exclusion` block below also
            // rejects when `.weight.absmax` is present so FP8
            // doesn't fingerprint-match a BNB4 checkpoint of the
            // same arch+size.
            quote! {}
        }
        Some(crate::quantization::QuantMethod::Ggml) | None => quote! {},
        // The probe reads the fingerprint leaf's `.scales`, so it runs only where this variant
        // reads that leaf affine: an MLX per-module map may leave it dense (Nemotron-H's
        // experts-only parity checkpoint quantizes the routed experts alone; a draft head's only
        // affine weight may be the embedding its target lends it). Each module the map DOES list
        // must then ship its `.scales` sibling, and the per-role bit-map gate below carries the
        // positive affine sniff.
        Some(crate::quantization::QuantMethod::Affine { per_module, .. }) if !fp_leaf_affine => {
            let scales = per_module.iter().map(|(k, _)| format!("{k}.scales"));
            quote! { #(if !gw.contains(#scales) { return false; })* }
        }
        Some(crate::quantization::QuantMethod::Affine { group_size, .. }) => {
            // Group-size discriminator. The packed `.weight` (U32) has
            // the SAME shape for every group_size — 4-bit packs 8 values
            // per word along `in`, independent of the grouping — so g64
            // and g128 affine variants are otherwise fingerprint-
            // identical, and a mismatched checkpoint would dequantize
            // with the wrong group stride → silent garbage (not a
            // rejection). The `.scales` sibling is `[out, in/group_size]`;
            // for the q_proj fingerprint leaf `in == hidden_size`, so its
            // last dim must equal `hidden_size / group_size`. Reject
            // otherwise: a checkpoint whose group_size has no compiled
            // variant then gets the `ArchNotSupported` error (pointing at
            // quantizations.json) instead of corrupt output, and one that
            // DOES have a matching variant is routed to it deterministically
            // regardless of variant registration order. (bits are
            // discriminated by the `affine_bit_map_gate` below, which
            // reads the effective per-role bits rather than the section
            // default.)
            let scales_groups =
                proc_macro2::Literal::usize_unsuffixed(hidden_size as usize / *group_size as usize);
            quote! {
                match gw.tensor_info(#last_scales_tensor) {
                    Some((shape, _))
                        if shape.len() == 2 && shape[1] == #scales_groups => {}
                    _ => return false,
                }
            }
        }
        Some(crate::quantization::QuantMethod::Nvfp4 { .. }) => {
            // NVFP4 fingerprint has no compile-time shape gate — the
            // `.weight_scale` (float8_e4m3) + `.weight_scale_2` (f32)
            // sibling presence and the `U8` dtype on `<prefix>.weight`
            // are the load-time signals that distinguish it. Nothing
            // to fingerprint at the shape level.
            quote! {}
        }
    };

    // Per-role affine BIT-WIDTH gate. The group-size gate above probes one
    // tensor at one width; this one asserts the width of EVERY affine tensor the
    // emitted loader is about to read.
    //
    // ⛔ THE ONLY DISCRIMINATOR FOR VARIANTS THAT DIFFER ONLY IN THEIR PER-ROLE
    // BIT MAP. `Qwen3.5-35B-A3B-4bit` and `Qwen3.6-35B-A3B-4bit` agree on every
    // other axis here — hidden 2048, 40 layers, 256 experts, vocab 248320, g64,
    // no declared `rope_scaling` or top-level `rope_theta` — and differ only in
    // `mlp.gate` + `mlp.shared_expert_gate` being 4-bit on 3.5 and 8-bit on 3.6.
    // Without this gate the alphabetically-earlier variant claims both, and the
    // mismatch surfaces as a shape `ensure!` deep inside `affine_dequant_b4_bytes`
    // rather than as a rejection (#202).
    //
    // The rows come from `affine_tensors_of` — read off the same `FieldLoad`s the
    // load calls are emitted from — so the gate cannot assert a width the loader
    // won't use, and there is no second bit-resolution algorithm to drift from
    // `affine_role_bits`. Names are exact (patterns star only the layer index),
    // so nothing is matched by suffix and nothing is claimed about a tensor this
    // variant does not read.
    //
    // Symmetry comes from both variants listing the SAME tensor at their OWN
    // width: 3.5 lists `…layers.*.mlp.gate` at 4 and so rejects 3.6's 8-bit gate,
    // exactly as 3.6 lists it at 8 and rejects 3.5's.
    let affine_bit_map_gate: TokenStream = if affine_rows.is_empty() {
        quote! {}
    } else {
        let row_toks = affine_rows.iter().map(|(name, b, g)| {
            let name = proc_macro2::Literal::string(name);
            let b = proc_macro2::Literal::u32_unsuffixed(*b);
            let g = proc_macro2::Literal::u32_unsuffixed(*g);
            quote! { (#name, #b, #g) }
        });
        // The width check is silent on a listed tensor without `.scales`, so a variant the
        // group-size probe above does not cover (its `fp_leaf` is dense) must positively see the
        // checkpoint ship at least one of its affine tensors packed — or a DENSE checkpoint
        // satisfies an experts-only per-module variant vacuously and dies in its loader.
        let positive_sniff = if fp_leaf_affine {
            quote! {}
        } else {
            quote! {
                if !gw.ships_affine_row(__AFFINE_WIDTHS) {
                    return false;
                }
            }
        };
        quote! {
            const __AFFINE_WIDTHS: &[(&str, u32, u32)] = &[#(#row_toks),*];
            #positive_sniff
            if !gw.affine_widths_agree(__AFFINE_WIDTHS) {
                return false;
            }
        }
    };
    let expert_input_check: TokenStream = match expert_input {
        Some(ExpertInputProbe {
            tensors,
            packed_cols,
        }) => {
            let cols = proc_macro2::Literal::usize_unsuffixed(*packed_cols);
            // The width is the bank's, however the checkpoint spells its tensor: each
            // candidate present must carry it, and at least one must be present (a layout
            // the loaders resolve but the fingerprint never saw would otherwise refuse the
            // very checkpoint it was compiled for).
            let probes = tensors.iter().map(|tensor| {
                let tensor = proc_macro2::Literal::string(tensor);
                quote! {
                    if let Some((shape, _)) = gw.tensor_info(#tensor) {
                        if shape.last() != Some(&#cols) {
                            return false;
                        }
                        __EXPERT_INPUT_SEEN = true;
                    }
                }
            });
            quote! {
                let mut __EXPERT_INPUT_SEEN = false;
                #(#probes)*
                if !__EXPERT_INPUT_SEEN {
                    return false;
                }
            }
        }
        None => quote! {},
    };

    // Backing-store reject: every non-Ggml variant must reject a
    // GGUF-backed `GpuWeights`, and the Ggml variant must require
    // one. Without this the dense variant's shape-only fingerprint
    // matches a GGUF (since `gw.contains` checks the quantized map)
    // and the dense load body's fused-QKV path fires on Ggml
    // weights — panicking at runtime when a Cutlass instr calls
    // `dense_weight()` on a `LinearLayer::GgmlConcat`.
    let is_gguf_gate: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Ggml) => quote! {
            if !gw.is_gguf() {
                return false;
            }
        },
        _ => quote! {
            if gw.is_gguf() {
                return false;
            }
        },
    };

    // GPTQ-Qweight desc_act disambiguation. With overlay fan-out we
    // synthesize BOTH `gptq-sym` (desc_act=false) and
    // `gptq-sym-desc_act` (desc_act=true) variants per dense base —
    // both have the same `.qweight [K/8, N]` shape gate above. The
    // `.g_idx` tensor is what the runtime actually needs to
    // disambiguate: AutoGPTQ ships it iff desc_act=true (it encodes
    // the activation-order permutation the loader passes to
    // `gptq_repack_into`). desc_act=false repos either omit `.g_idx`
    // or carry it inert; we treat presence as the signal that
    // selects the desc_act=true variant. Without this disambiguation
    // the alphabetically-earlier `gptq-sym` would win for a
    // desc_act=true repo, threading `desc_act=false` through
    // `MarlinFormat::Gptq` and producing garbage weights.
    let g_idx_disambiguation: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Gptq {
            desc_act,
            layout: crate::quantization::GptqLayout::Qweight,
            ..
        }) => {
            let g_idx_tensor_owned = under_root(&dec_root, "layers.0.self_attn.q_proj.g_idx");
            let g_idx_tensor = g_idx_tensor_owned.as_str();
            if *desc_act {
                quote! {
                    if !gw.contains(#g_idx_tensor) {
                        return false;
                    }
                }
            } else {
                quote! {
                    if gw.contains(#g_idx_tensor) {
                        return false;
                    }
                }
            }
        }
        _ => quote! {},
    };

    // FP8 `activation_scheme` disambiguation. Overlay fan-out may
    // synthesize both `fp8-dynamic-per-tensor` and
    // `fp8-static-per-tensor` variants for a single dense base; they
    // share the same `.weight_scale` suffix gate above. The on-disk
    // signal that actually distinguishes them is the per-projection
    // `.input_scale` tensor — neuralmagic / RedHatAI static FP8
    // checkpoints carry it (it's the pre-calibrated per-tensor
    // activation scale), dynamic checkpoints omit it. Without this
    // disambiguation the alphabetically-earlier dynamic variant
    // would win for a static checkpoint and `Fp8Linear::forward`
    // would fall into the per-token dynamic-quant path instead of
    // the pre-calibrated static path.
    let input_scale_disambiguation: TokenStream =
        match model.quantization.as_ref().map(|qc| &qc.method) {
            Some(crate::quantization::QuantMethod::Fp8 {
                scheme: crate::quantization::Fp8ActivationScheme::Static,
                block_size: None,
            }) => {
                let input_scale_tensor_owned =
                    under_root(&dec_root, "layers.0.self_attn.q_proj.input_scale");
                let input_scale_tensor = input_scale_tensor_owned.as_str();
                quote! {
                    if !gw.contains(#input_scale_tensor) {
                        return false;
                    }
                }
            }
            Some(crate::quantization::QuantMethod::Fp8 {
                scheme: crate::quantization::Fp8ActivationScheme::Dynamic,
                block_size: None,
            }) => {
                let input_scale_tensor_owned =
                    under_root(&dec_root, "layers.0.self_attn.q_proj.input_scale");
                let input_scale_tensor = input_scale_tensor_owned.as_str();
                quote! {
                    if gw.contains(#input_scale_tensor) {
                        return false;
                    }
                }
            }
            _ => quote! {},
        };

    // FP8 blockwise disambiguation. Overlay fan-out may synthesize
    // both `fp8-*-per-tensor` and `fp8-block-*` variants for a single
    // dense base; they share the `.weight_scale` positive sniff. The
    // on-disk signal that distinguishes them is the scale tensor's
    // shape — per-tensor ships a scalar `[1]` or per-channel `[N, 1]`,
    // blockwise ships a 2-D `[ceil(N/bn), ceil(K/bk)]` with bk > 1.
    // Some blockwise repos name the tensor `.weight_scale_inv`
    // (DeepSeek-V3, Qwen3-MoE); its mere presence is a sufficient
    // block signal too. Without this, the alphabetically-earlier
    // per-tensor variant would win for a block checkpoint and
    // `Fp8Linear::load` would fail at runtime reading a 2-D scale
    // expecting a scalar.
    let block_disambiguation: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Fp8 {
            block_size: Some(_),
            ..
        }) => {
            // MLA archs ship `q_a_proj` instead of `q_proj` — match
            // the leaf the fingerprint already chose above so V3 / K2
            // FP8-block fixtures aren't silently rejected.
            let inv_tensor = under_root(
                &dec_root,
                &format!("layers.{marker_layer}.{fp_leaf}.weight_scale_inv"),
            );
            let scale_tensor = under_root(
                &dec_root,
                &format!("layers.{marker_layer}.{fp_leaf}.weight_scale"),
            );
            let inv_tensor = inv_tensor.as_str();
            let scale_tensor = scale_tensor.as_str();
            quote! {
                // Block variant: accept if `.weight_scale_inv` exists,
                // or if `.weight_scale` is 2-D with more than one
                // column (the per-tensor and per-channel layouts have
                // `shape.len() == 1` or `shape[1] == 1` respectively).
                let __fp8_is_block = gw.contains(#inv_tensor)
                    || matches!(
                        gw.tensor_info(#scale_tensor),
                        Some((shape, _)) if shape.len() == 2 && shape[1] > 1,
                    );
                if !__fp8_is_block {
                    return false;
                }
            }
        }
        Some(crate::quantization::QuantMethod::Fp8 {
            block_size: None, ..
        }) => {
            let inv_tensor = under_root(
                &dec_root,
                &format!("layers.{marker_layer}.{fp_leaf}.weight_scale_inv"),
            );
            let scale_tensor = under_root(
                &dec_root,
                &format!("layers.{marker_layer}.{fp_leaf}.weight_scale"),
            );
            let inv_tensor = inv_tensor.as_str();
            let scale_tensor = scale_tensor.as_str();
            quote! {
                // Per-tensor variant: reject block checkpoints.
                if gw.contains(#inv_tensor) {
                    return false;
                }
                if let Some((shape, _)) = gw.tensor_info(#scale_tensor)
                    && shape.len() == 2
                    && shape[1] > 1
                {
                    return false;
                }
            }
        }
        _ => quote! {},
    };

    // The embedding tensor's on-disk path varies per arch — llama uses
    // `model.embed_tokens.weight`, ModernBERT uses
    // `model.embeddings.tok_embeddings.weight`. The manifest entry whose
    // shape is `[vocab_size | embed_vocab_size, hidden_size]` is the embedding table; use
    // its key (with `model.` prefix + `.weight` suffix) as the
    // fingerprint sniff. Fall back to the llama-style path when no
    // entry matches, preserving the previous behavior for any arch
    // whose manifest predates this generalization.
    let embed_role: &str = manifest
        .entries
        .iter()
        .find(|(_, shape)| {
            shape.len() == 2
                && matches!(&shape[0], crate::shape::Dim::Bound(s)
                    if s == "vocab_size" || s == "embed_vocab_size")
                && matches!(&shape[1], crate::shape::Dim::Bound(s) if s == "hidden_size")
        })
        .map_or("embed_tokens", |(k, _)| k.as_str());
    let embed_path = under_root(&dec_root, &format!("{embed_role}.weight"));
    let embed_path_lit = proc_macro2::Literal::string(embed_path.as_str());

    // Hidden-size shape gate for the embedding fingerprint sniff.
    // Three on-disk shapes for `embed_tokens`:
    //   * dense `[vocab, hidden]` — every non-affine variant, AND
    //     the older mlx-affine convention for untied checkpoints
    //     (e.g. Llama-3-8B-Instruct-4bit).
    //   * packed `[vocab, ceil(hidden*bits/32)]` (8 for bits=4) — every
    //     mlx-affine TIED checkpoint
    //     checkpoint (embed IS the quantized lm_head, e.g.
    //     Llama-3.2-{1B,3B}-4bit) PLUS untied checkpoints whose
    //     preset opted in via `quant_embed: true` (e.g.
    //     Llama-3.1-8B-Instruct-4bit under the
    //     `mlx-affine-b4-g64-qembed` preset).
    //
    // Per-variant the shape is unambiguous — we pick the single
    // expected shape from (quant method, tie_word_embeddings,
    // quantize_embed). The `.scales`/`.biases` sibling on layer.0
    // q_proj (mlx_marker_tensor below) disambiguates affine-vs-dense;
    // here we just need a precise embed-shape gate so a checkpoint
    // doesn't false-positive against the wrong affine sub-variant.
    let embed_expects_packed: bool = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Affine {
            quantize_embed,
            bits_overrides,
            per_module,
            ..
        }) => {
            // MLX mixed/dynamic (OptiQ) lists `embed_tokens` in the per-module
            // map at its own bits WITHOUT setting the `quantize_embed` preset
            // flag — so the fingerprint must also consider per-module membership,
            // else the embedded (non-preset) OptiQ variant computes a DENSE embed
            // shape, never matches the real 8-bit-packed embed, and OptiQ is
            // forced onto the preset hybrid variant. `storage_format_for_weight`
            // already quantizes the embed in that case; keep the two in agreement.
            // A `bits_overrides` entry naming the embed implies packed the same
            // way (gpt-oss's proper low-bit recipes ship a 4-bit embed over a
            // 2/3-bit default) — same agreement, same reason.
            // A non-empty per-module map is authoritative, exactly as `storage_format_for_weight`
            // reads it: the embedding is packed iff the map lists IT (not a `target_embed_tokens`
            // a draft head borrows).
            match per_module.is_empty() {
                false => (per_module.iter())
                    .any(|(k, _)| crate::quantization::names_embed(k, embed_role)),
                true => {
                    model.tie_word_embeddings
                        || *quantize_embed
                        || bits_overrides
                            .iter()
                            .any(|(s, _)| crate::quantization::names_embed(s, embed_role))
                }
            }
        }
        _ => false,
    };
    let embed_hidden_lit: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(method @ crate::quantization::QuantMethod::Affine { .. }) if embed_expects_packed => {
            // Use the embed's OWN bits, not the section default: MLX
            // mixed/dynamic checkpoints (OptiQ) pack `embed_tokens` at
            // 8-bit (hidden/4) while the default is 4-bit (hidden/8).
            // Packed width follows the ceil(hidden*bits/32) law — bits=3
            // packs 8 elements per 3 bytes, so it is NOT hidden/(32/bits).
            let bits = crate::quantization::affine_embed_bits(method, embed_role).unwrap_or(4);
            let packed = (hidden_size * bits as u64).div_ceil(32) as usize;
            let lit = proc_macro2::Literal::usize_unsuffixed(packed);
            quote! { #lit }
        }
        _ => quote! { #hidden_lit },
    };
    let embed_shape_check: TokenStream = quote! {
        match gw.tensor_shape_any(#embed_path_lit) {
            Some(ref shape)
                if shape.len() >= 2
                    && shape[0] == #vocab_lit
                    && shape[1] == #embed_hidden_lit => {}
            _ => return false,
        }
    };

    quote! {
        /// Per-variant compile-time fingerprint check. See
        /// macro's `emit_fingerprint_check` for the rules.
        /// Backend-neutral — `gw` reads tensor shapes via the
        /// shared `GpuWeights` API, so the same body sniffs every target's checkpoints
        /// identically.
        pub fn fingerprint_matches(
            gw: &crate::__gpu::weights::GpuWeights,
            hf: ::scratchy_forward_compiler::HfFingerprint<'_>,
        ) -> bool {
            #embed_shape_check
            if !gw.contains(#last_tensor) {
                return false;
            }
            if gw.contains(#one_past_tensor) {
                return false;
            }
            if gw.contains(#opposite_tensor) {
                return false;
            }
            // BNB4 marker exclusion — reject dense/AWQ/GPTQ/CT
            // fingerprints when the checkpoint ships BNB4's
            // `.weight.absmax` sibling. The BNB4 variant itself
            // keys off this tensor in its suffix-based checks
            // above, so this exclusion runs only for non-BNB4
            // variants (empty for BNB4).
            #bnb4_marker_check
            // FP8 marker exclusion — reject dense/AWQ/GPTQ/CT/BNB4
            // fingerprints when the checkpoint ships FP8's
            // `.weight_scale` sibling. The FP8 variant itself keys
            // off `.weight_scale` as its positive suffix, so this
            // exclusion runs only for non-FP8 variants (empty for FP8).
            #fp8_marker_check
            // MLX-affine marker exclusion — every non-affine variant
            // rejects checkpoints that ship `.scales` on layer.0
            // q_proj. See `emit_fingerprint_check` for the rationale.
            #mlx_marker_check
            #qweight_shape_gate
            // Per-role affine bit widths — the only thing separating two
            // variants whose configs differ solely in `bits_overrides` /
            // `per_module` (qwen3.5 vs qwen3.6 35B-A3B). See
            // `emit_fingerprint_check`.
            #affine_bit_map_gate
            // An ungated bank's experts' input width — LatentMoE vs plain MoE.
            #expert_input_check
            #is_gguf_gate
            #g_idx_disambiguation
            #input_scale_disambiguation
            #block_disambiguation
            #max_pos_check
            #rope_scaling_check
            #rope_scaling_hash_check
            // RoPE base frequency — the only discriminator for several in-tree
            // variant groups (granite 3.0/3.1/3.3, phi-4 vs phi-4-reasoning),
            // which declare no `rope_scaling` and so are invisible to both
            // checks above. See `emit_fingerprint_check`.
            #rope_theta_check
            true
        }
    }
}

/// `emit_weights_struct` has two modes. `Canonical` defines its own
/// `pub struct Weights { ... }`; `Shim { canonical }` aliases the
/// struct to the canonical sibling module's Weights (for cross-
/// variant forward-fn dedup) while still emitting this variant's
/// own `load` + `fingerprint_matches` bodies.
pub(crate) enum WeightsEmitMode<'a> {
    Canonical,
    Shim { canonical: &'a Ident },
}

/// The affine rows a variant's fingerprint gate should carry. A shim's
/// `load` delegates to its canonical's body, where the affine
/// bit-widths are baked in — and a shim lowers no tape, so its own
/// accessor set is empty and a rowless shim would otherwise gate on
/// nothing and select anything. A shim with no rows of its own
/// inherits its canonical's; own rows always win when present.
fn effective_fingerprint_rows<'a>(
    mode: &WeightsEmitMode<'_>,
    own: &'a [(String, u32, u32)],
    canonical: Option<&'a [(String, u32, u32)]>,
) -> &'a [(String, u32, u32)] {
    match (mode, own.is_empty()) {
        (WeightsEmitMode::Shim { .. }, true) => canonical.unwrap_or(&[]),
        _ => own,
    }
}

/// Emit the `Weights` struct definition (or alias) + its `load` +
/// `fingerprint_matches` free fns. Returns the tokens alongside the
/// affine rows its fingerprint gate actually carries (the canonical's,
/// for a rowless shim) so the caller can hand them to that variant's
/// future shims.
#[allow(clippy::too_many_arguments)]
fn emit_weights_struct(
    program: &Program,
    fuf: &Fuf,
    // The accessor set, from the SHARED TAPE. `None` (no tape lowered
    // for this variant — a shim, or metal's first pass before the
    // front-end swap) emits a `Weights` with no accessor fields.
    tape_accessors: Option<&[WeightAccessor]>,
    // The fields' typed handles (none for a shim, whose `Weights` is its canonical's).
    dims: &ModelDims,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    mode: WeightsEmitMode<'_>,
    tp_world_size: u8,
    emit_fingerprint: bool,
    canonical_affine_rows: Option<&[(String, u32, u32)]>,
    emitter: &dyn crate::target::TargetEmitter,
) -> (TokenStream, Vec<(String, u32, u32)>) {
    let accessors: Vec<WeightAccessor> = tape_accessors.map(<[_]>::to_vec).unwrap_or_default();

    // Storage-format guard: a given accessor's `rust_type` must be
    // compatible with every one of its source weights' storage
    // formats. The allowed pairs today:
    //   `LinearLayer`   ↔ `Dense` | `Ggml`
    //   `Embedding`     ↔ `Dense`
    //   `RmsNorm`       ↔ `Dense`
    //   `MarlinLinear`  ↔ `Awq { .. }` | `Gptq { .. }`
    //   `Bnb4bitLinear` ↔ `Bnb4 { .. }`
    //   `Fp8Linear`     ↔ `Fp8 { .. }`
    //
    // GGUF rides on the `LinearLayer` accessor type because the
    // runtime enum already has a `Ggml(Box<GgmlLinear>)` arm that
    // dispatches at forward time — so codegen produces the same
    // accessor field type for Dense and Ggml; the FieldLoad arm
    // picks `take_quantized_linear` vs `take` based on storage.
    //
    // Any other pair means the solver picked an impl whose
    // declared accessor type doesn't match the bits on disk — a
    // matcher bug. Fail at macro-expansion time so new quant
    // formats can't slip through without a matching FieldLoad arm.
    for a in &accessors {
        let ty = a.rust_type.to_string().replace(' ', "");
        let accessor_is_marlin = ty.ends_with("::MarlinLinear")
            || ty == "MarlinLinear"
            || ty.ends_with("layers::MarlinLinear");
        let accessor_is_bnb4 = ty.ends_with("::Bnb4bitLinear")
            || ty == "Bnb4bitLinear"
            || ty.ends_with("layers::Bnb4bitLinear");
        // FP8 accessors come through `fp8_accessor_type_for`, which
        // post-Fp8AnyLinear-unblocker always returns `Fp8AnyLinear`.
        // The storage-format guard accepts that type for both
        // per-tensor / per-channel (`block_size: None`) and
        // blockwise (`block_size: Some(_)`) FP8 storage — the
        // Fp8AnyLinear enum dispatches at runtime on the loaded
        // variant.
        let accessor_is_fp8_any = ty.ends_with("::Fp8AnyLinear")
            || ty == "Fp8AnyLinear"
            || ty.ends_with("layers::Fp8AnyLinear")
            || ty.ends_with("::DeepSeekV2Fp8BlockMoELayer")
            || ty == "DeepSeekV2Fp8BlockMoELayer"
            || ty.ends_with("layers_moe::DeepSeekV2Fp8BlockMoELayer");
        // Gemma-4 SwitchGLU experts dequant their packed weights to dense BF16
        // at LOAD (no Marlin GEMM), so this accessor pairs with BOTH MLX-affine
        // (handled by the dense-style `(Affine, …)` arm below) AND
        // compressed-tensors `Gptq{WeightPacked}` (cyankiwi gemma-4-26B-A4B-it
        // AWQ-4bit) — the latter has no dense-style arm, so allow it explicitly.
        let accessor_is_switch_glu = ty.ends_with("::SwitchGluExpertsLayer")
            || ty == "SwitchGluExpertsLayer"
            || ty.ends_with("layers_moe::SwitchGluExpertsLayer");
        for (wid, _idx) in &a.source_weights {
            let fmt =
                crate::quantization::storage_format_for_weight(program, fuf, *wid, None, model);
            let ok = (accessor_is_switch_glu
                && matches!(
                    fmt,
                    crate::quantization::StorageFormat::Gptq {
                        layout: crate::quantization::GptqLayout::WeightPacked,
                        ..
                    }
                ))
                || matches!(
                    (
                        &fmt,
                        accessor_is_marlin,
                        accessor_is_bnb4,
                        accessor_is_fp8_any
                    ),
                    (
                        crate::quantization::StorageFormat::Dense,
                        false,
                        false,
                        false
                    ) | (
                        crate::quantization::StorageFormat::Awq { .. },
                        true,
                        false,
                        false
                    ) | (
                        crate::quantization::StorageFormat::Gptq { .. },
                        true,
                        false,
                        false
                    ) | (
                        crate::quantization::StorageFormat::Bnb4 { .. },
                        false,
                        true,
                        false
                    ) | (
                        crate::quantization::StorageFormat::Fp8 { .. },
                        false,
                        false,
                        true
                    ) | (
                        crate::quantization::StorageFormat::Ggml,
                        false,
                        false,
                        false
                    ) | (
                        // MLX-affine pairs with the dense-style accessor
                        // types (Embedding, LinearLayer, RmsNorm — though
                        // RmsNorm is never quantized on disk). Under INT4
                        // P2 the loader CPU-dequants at load time and
                        // returns the dense runtime type, so the accessor
                        // type stays `Embedding` / `LinearLayer` — no
                        // separate AffineLinear runtime type is generated
                        // by the macro.
                        crate::quantization::StorageFormat::Affine { .. },
                        false,
                        false,
                        false
                    ) | (
                        // NVFP4 pairs with the dense-style `LinearLayer`
                        // accessor too: the FieldLoad emits
                        // `LinearLayer::load_nvfp4_quant`, producing a
                        // `LinearLayer::Nvfp4(..)` at runtime — the accessor
                        // type stays `LinearLayer`, like Affine.
                        crate::quantization::StorageFormat::Nvfp4 { .. },
                        false,
                        false,
                        false
                    ),
                );
            // The storage/accessor match only matters for backends that emit a
            // concrete per-accessor host FieldLoad (metal). A target that
            // runs an embedded fused program loads + dequantizes weights itself
            // (no host FieldLoad), so the dense accessor is fine for any storage.
            if !ok && cfg!(feature = "metal") {
                let dotted = program.weights.path(*wid).join(".");
                let msg = format!(
                    "model `{stem}`: weight `{dotted}` has storage ({fmt:?}) that \
                     doesn't match accessor `{name}` (type `{ty}`). Add a quant-aware \
                     Impl + FieldLoad arm for this pair.",
                    stem = model.source_stem,
                    name = a.name,
                );
                return (quote! { compile_error!(#msg); }, Vec::new());
            }
        }
    }

    // Group accessors by base for the Vec<T> compression. Layered
    // groups collapse to one `pub <base>: Vec<T>` field + one
    // `(0..N).map(|layer| …).collect::<Result<Vec<_>>>()?` Vec-build
    // in `load_with`; unindexed groups keep the per-accessor field
    // and per-FieldLoad let. The same group list also drives
    // `emit_weights_accessor_methods`, so all three sites stay in
    // sync — adding a new accessor only changes the inputs here.
    let groups = group_accessors_by_base(&accessors);

    let fields: Vec<TokenStream> = groups
        .iter()
        .flat_map(|g| {
            let ty = dims.handle(&g.base).map_or(&g.rust_type, |h| &h.ty);
            match g.kind {
                AccessorGroupKind::Unindexed => {
                    let name = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
                    vec![quote! { pub #name: #ty, }]
                }
                AccessorGroupKind::LayeredContiguous => {
                    let name = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
                    vec![quote! { pub #name: ::std::vec::Vec<#ty>, }]
                }
                AccessorGroupKind::LayeredSparse => g
                    .entries
                    .iter()
                    .map(|(_, acc)| {
                        let n = &acc.name;
                        quote! { pub #n: #ty, }
                    })
                    .collect(),
            }
        })
        .collect();

    // Emit each group as its own let-binding in the load body.
    // Layered groups become `let <base>: Vec<T> = (0..N).map(…)
    // .collect()?;`. Unindexed groups keep the existing per-FieldLoad
    // let (computed by `plan_field_load` from the single accessor).
    // Order matters: a tied `lm_head` reads `embed_tokens.weight`,
    // and `groups` iterates BTreeMap-sorted by base, which puts
    // `embed_tokens` before `lm_head` alphabetically.
    //
    // Scan plans for the prelude-needed flags (any_marlin / any_bnb4
    // / any_fp8) over the FULL accessor list — keeps the prelude
    // logic identical to the pre-grouping version. The prelude
    // bindings (`__marlin_ws`, `__bnb_code`, `__fp8_dtype`, …) are
    // captured by the layered closures via lexical scope.
    let embed_is_affine_repr = embed_is_affine_repr(&accessors);
    let plans: Vec<FieldLoad> = accessors
        .iter()
        .map(|a| plan_field_load(a, program, fuf, model, manifest, embed_is_affine_repr))
        .collect();
    // The affine widths the emitted load calls will use, harvested from the very
    // `FieldLoad`s they are emitted from, then collapsed per role. This is what
    // the fingerprint's bit table is built from — see `affine_tensors_of`.
    let affine_rows = compress_affine_rows(plans.iter().flat_map(affine_tensors_of).collect());
    let fingerprint_rows =
        effective_fingerprint_rows(&mode, &affine_rows, canonical_affine_rows).to_vec();
    // Per-arch decoder root for embed_tokens probes etc.
    let dec_root_for_emit = model.arch.decoder_root();
    let embed_tokens_weight_path: String = under_root(&dec_root_for_emit, "embed_tokens.weight");
    let any_marlin = plans
        .iter()
        .any(|p| matches!(p, FieldLoad::MarlinLinear { .. }));
    let any_bnb4 = plans
        .iter()
        .any(|p| matches!(p, FieldLoad::Bnb4Linear { .. }));
    let any_fp8 = plans.iter().any(|p| {
        matches!(
            p,
            FieldLoad::Fp8Linear { .. } | FieldLoad::Fp8BlockLinear { .. }
        )
    });
    // `max(out_features * in_features)` across every BNB4 accessor
    // — sizes the per-model shared dequant scratch buffer. Zero
    // when the model has no BNB4 accessors (the prelude block is
    // then elided entirely).
    let bnb4_max_elements: usize = plans
        .iter()
        .filter_map(|p| match p {
            FieldLoad::Bnb4Linear {
                out_features_per_shard,
                in_features,
                ..
            } => {
                let total_out: usize = out_features_per_shard.iter().map(|o| *o as usize).sum();
                Some(total_out * (*in_features as usize))
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);
    // Map each accessor to its FieldLoad once so the unindexed-let
    // arm and the layered Vec-build arm can both look it up by
    // accessor identity. (`accessors` is sorted, plans was built in
    // the same order, so a name → plan lookup is fine.)
    let plan_by_name: std::collections::BTreeMap<String, &FieldLoad> = accessors
        .iter()
        .zip(plans.iter())
        .map(|(a, p)| (a.name.to_string(), p))
        .collect();

    let is_vision = matches!(program.prelude, crate::classified::Prelude::Vision);
    let lets: Vec<TokenStream> = groups
        .iter()
        .map(|g| emit_group_let(g, &plan_by_name, model, tp_world_size, is_vision))
        .collect();

    // Shared-per-model Marlin prelude: one workspace allocation
    // (GpuTensor is `Copy` — each MarlinLinear captures the same
    // buffer by value), one device-id query. Only planted when at
    // least one accessor resolves to a Marlin FieldLoad (AWQ or
    // GPTQ); dense models skip it so their `Weights::load` body is
    // byte-for-byte identical to before quantization landed.
    let marlin_prelude: TokenStream = if any_marlin {
        quote! {
            let __device = unsafe { crate::__gpu::driver::current_device()? };
            let __device_id: i32 = __device as i32;
            let __num_sm = unsafe { crate::__gpu::driver::device_get_num_sm(__device)? };
            let __marlin_ws =
                crate::__gpu::layers_quant::alloc_marlin_workspace(__num_sm, stream)?;
        }
    } else {
        quote! {}
    };

    // Shared-per-model BNB4 prelude: upload the NF4/FP4 LUT once
    // (16-entry f32 table) and allocate a single dequant scratch
    // buffer sized to `max(out*in)` across every BNB4 accessor.
    // Every `Bnb4bitLinear` on this device captures both tensors by
    // value (`GpuTensor: Copy`). Elided when the model has no BNB4.
    //
    // Compute dtype for the shared dequant scratch comes from
    // `embed_tokens.weight` — guaranteed present on every arch and
    // always stored in the model's compute dtype (never BNB-packed,
    // per bitsandbytes' default `llm_int8_skip_modules`). Reading
    // it off disk keeps bf16-compute and fp16-compute checkpoints
    // both correct without a config scrape.
    let bnb4_prelude: TokenStream = if any_bnb4 {
        let max_elements = proc_macro2::Literal::usize_unsuffixed(bnb4_max_elements);
        let code_expr = match model.quantization.as_ref().map(|qc| &qc.method) {
            Some(crate::quantization::QuantMethod::Bnb4 {
                quant_type: crate::quantization::BnbQuantType::NF4,
                ..
            }) => quote! { crate::__gpu::layers_quant::NF4_CODE },
            Some(crate::quantization::QuantMethod::Bnb4 {
                quant_type: crate::quantization::BnbQuantType::FP4,
                ..
            }) => quote! { crate::__gpu::layers_quant::FP4_CODE },
            _ => unreachable!("any_bnb4 implies QuantMethod::Bnb4"),
        };
        quote! {
            let __bnb_code =
                crate::__gpu::layers_quant::upload_bnb_code(&#code_expr, stream)?;
            let __bnb_dtype = gw
                .tensor_info(#embed_tokens_weight_path)
                .map(|(_, dt)| dt)
                .unwrap_or(crate::__gpu::dtype::DType::BF16);
            let __bnb_scratch = crate::__gpu::layers_quant::alloc_bnb_dequant_scratch(
                #max_elements,
                __bnb_dtype,
                stream,
            )?;
        }
    } else {
        quote! {}
    };

    // Shared-per-model FP8 prelude: read the compute dtype from
    // `embed_tokens.weight` (always present, always in compute dtype
    // — never FP8-quantized) and hand it to every `Fp8Linear::load`.
    // Matches the pattern used by the BNB4 prelude for its dequant
    // scratch allocation. Elided when no FP8 accessors are present.
    let fp8_prelude: TokenStream = if any_fp8 {
        quote! {
            let __fp8_dtype = gw
                .tensor_info(#embed_tokens_weight_path)
                .map(|(_, dt)| dt)
                .unwrap_or(crate::__gpu::dtype::DType::BF16);
        }
    } else {
        quote! {}
    };

    // Manifest-driven packed-tensor splits: archs whose checkpoints
    // ship fused qkv / gate_up under a single on-disk name declare
    // `__packed_splits__` in their `weights.json` (e.g. Phi-3 family).
    // For each (packed_prefix → [target …]) entry, we emit one call
    // per transformer layer that synthesizes the per-slice virtual
    // entries before any `FieldLoad`. Row counts come from looking up
    // each target in the manifest and evaluating the first dim against
    // `model.bounds`. The helper is a no-op when the packed parent
    // isn't present, so models without packed checkpoints are unaffected
    // even if they share the manifest (none do today).
    let is_vision = matches!(program.prelude, crate::classified::Prelude::Vision);
    let packed_splits_prelude: TokenStream = if manifest.packed_splits.is_empty() {
        quote! {}
    } else {
        // Vision configs carry `vision_depth` instead of
        // `num_hidden_layers`; the per-block prefix is
        // `visual.blocks.<L>.` instead of `model.layers.<L>.`.
        let layer_count_key = if is_vision {
            "vision_depth"
        } else {
            "num_hidden_layers"
        };
        // The packed-split SOURCE key (`<root>.<l>.attn.qkv`) must use the
        // SAME vision root the layered load reads back the carved q/k/v
        // under (codegen.rs ~4321 `vision_root_owned`). Derive it from the
        // per-arch `vision_safetensors_layout` (`vision_tower.blocks` for
        // Qwen3.5-VL) rather than hardcoding `visual.blocks` — otherwise
        // any non-`visual` Qwen-family tower carves qkv under a root the
        // loader never queries and load() fails on the missing split.
        let block_prefix_template: String = if is_vision {
            let layout = model
                .arch
                .safetensors
                .clone()
                .expect("vision safetensors layout enforced at config parse");
            format!("{}.{}", layout.default_root, layout.layered_subpath)
        } else {
            "model.layers".to_string()
        };
        let num_hidden_layers = match is_vision {
            false => model.text.layers.get(),
            true => *model.bounds.get(layer_count_key).unwrap_or_else(|| {
                panic!(
                    "model `{}` has `__packed_splits__` but no `{}` bound",
                    model.source_stem, layer_count_key,
                )
            }) as usize,
        };
        let mut calls: Vec<TokenStream> = Vec::new();
        for (packed_prefix, targets) in &manifest.packed_splits {
            let mut sized: Vec<(String, u64)> = Vec::with_capacity(targets.len());
            for target in targets {
                let shape = manifest.entries.get(target).unwrap_or_else(|| panic!(
                    "model `{}`: __packed_splits__ target `{target}` not declared in manifest entries",
                    model.source_stem,
                ));
                // Manifest convention is `[in_features, out_features]`
                // (Gemm expects `K = x.last == w.first`), so the
                // on-disk safetensors row count — what
                // `synthesize_packed_row_split_sizes` needs — is
                // `shape.last()`, the out dim. For a 1-D tensor (norm
                // weight), there's no "in" vs "out"; we still take the
                // sole dim, though packed-split targets are always 2-D
                // projection matrices in practice.
                let rows_dim = shape.last().unwrap_or_else(|| {
                    panic!(
                        "model `{}`: __packed_splits__ target `{target}` has empty shape",
                        model.source_stem,
                    )
                });
                let rows =
                    crate::shape::eval_closed_dim(rows_dim, &model.bounds).unwrap_or_else(|| {
                        panic!(
                            "model `{}`: __packed_splits__ target `{target}` out-dim `{:?}` \
                         did not resolve against bounds",
                            model.source_stem, rows_dim,
                        )
                    });
                // The helper takes the leaf suffix under the shared
                // grandparent (e.g. "q_proj" under "self_attn"); strip
                // the parent prefix if the target shares one with the
                // packed prefix (the common case), else pass the full
                // dotted suffix — `synthesize_packed_row_split_sizes`
                // joins grandparent + suffix either way.
                let packed_parent = packed_prefix.rsplit_once('.').map(|(p, _)| p).unwrap_or("");
                let suffix = target
                    .strip_prefix(&format!("{packed_parent}."))
                    .unwrap_or(target)
                    .to_string();
                sized.push((suffix, rows));
            }
            let pairs: Vec<TokenStream> = sized
                .iter()
                .map(|(suffix, rows)| {
                    let rows_lit = proc_macro2::Literal::usize_unsuffixed(*rows as usize);
                    quote! { (#suffix, #rows_lit) }
                })
                .collect();
            let tp_world_lit = proc_macro2::Literal::u8_unsuffixed(tp_world_size);
            calls.push(quote! {
                for __l in 0..#num_hidden_layers {
                    let __pp = ::std::format!(
                        concat!(#block_prefix_template, ".{}.{}"), __l, #packed_prefix,
                    );
                    // `_tp` variant handles the GGUF quantized parent at
                    // tp > 1 — fused `attn_qkv` / `ffn_up` are
                    // replicated on every rank by the GGUF loader's
                    // shard-kind rule table (the fused name isn't in
                    // there), so the packed-splits prelude carves
                    // per-rank views directly. At tp == 1 and for
                    // safetensors parents, delegates to the unsharded
                    // path below.
                    gw.synthesize_packed_row_split_sizes_tp(
                        &__pp,
                        &[ #(#pairs),* ],
                        tp_rank as usize,
                        #tp_world_lit as usize,
                    )?;
                }
            });
        }
        quote! {
            #(#calls)*
        }
    };

    // `Self { a, b, c }` shorthand — fields are the just-bound
    // locals. Unindexed and Vec-compressed groups contribute one
    // ident (the base name); sparse-layered groups contribute one
    // ident per per-layer accessor (matching the per-layer fields
    // in the struct + the per-layer let bindings in the body).
    // Each field from its load's `let`; a typed field rebound through the load mint right after
    // its load, which checks the loaded layer's extents against its type's numbers. The mint is a
    // statement beside its load, not an operand of the struct literal: its `?` then exits with
    // the locals the next load's does and shares that drop path, where a `?` among the literal's
    // operands (or every mint after every load) took a drop path of its own each — the qwen3.6
    // `load` 12-22x larger.
    let field_shorthand: Vec<syn::Ident> = groups
        .iter()
        .flat_map(|g| match g.kind {
            AccessorGroupKind::Unindexed | AccessorGroupKind::LayeredContiguous => {
                vec![syn::Ident::new(&g.base, proc_macro2::Span::call_site())]
            }
            AccessorGroupKind::LayeredSparse => {
                g.entries.iter().map(|(_, acc)| acc.name.clone()).collect()
            }
        })
        .collect();
    let mints = groups.iter().map(|g| {
        let Some(h) = dims.handle(&g.base) else {
            return TokenStream::new();
        };
        let (tensor, handle) = (&h.tensor, &h.ty);
        let mint = |name: &syn::Ident, ty: TokenStream| {
            quote! { let #name: #ty = ::scratchy_tensors::typed::Mint::mint(#name, #tensor)?; }
        };
        let base = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
        let minted: Vec<TokenStream> = match g.kind {
            AccessorGroupKind::Unindexed => vec![mint(&base, quote!(#handle))],
            AccessorGroupKind::LayeredContiguous => {
                vec![mint(&base, quote!(::std::vec::Vec<#handle>))]
            }
            AccessorGroupKind::LayeredSparse => (g.entries.iter())
                .map(|(_, acc)| mint(&acc.name, quote!(#handle)))
                .collect(),
        };
        quote!(#(#minted)*)
    });
    let lets: Vec<TokenStream> = (lets.iter().zip(mints))
        .map(|(load, mint)| quote!(#load #mint))
        .collect();
    // Vision encoders skip the fingerprint emission entirely — they
    // route through the hand-written `ScratchyMmRegistration` instead
    // of the inventory-based arch dispatcher, so `fingerprint_matches`
    // is unreachable. Skipping also avoids the `num_hidden_layers` /
    // `hidden_size` / `vocab_size` panics in `emit_fingerprint_check`
    // — vision configs (`vision_*` + `d_model` only) lack those keys.
    let fingerprint_method = if emit_fingerprint {
        emit_fingerprint_check(
            model,
            manifest,
            tp_world_size,
            &fingerprint_rows,
            expert_input_probe(program, fuf, model).as_ref(),
        )
    } else {
        TokenStream::new()
    };

    // Detect whether this arch uses `rotary_local` (dual-rotary,
    // e.g. Gemma3). If so, emit a `rotary_local: RotaryCache` field
    // on Weights + its construction in `load`. The global rotary
    // stays on ForwardCtx; only the local one lives here.
    let uses_rotary_local = fuf.nodes.iter().any(|n| {
        n.inputs.iter().any(|i| {
            matches!(
                i,
                crate::fuf::FufInput::Extern {
                    kind: crate::classified::ExternKind::RotaryLocal,
                    ..
                }
            )
        })
    });

    // The RoPE tables are `OnceLock`s: spyre builds them at load, metal when its pool is
    // built (`build_rope_tables`), for the positions the KV cache's block table reaches.
    let rotary_local_field: TokenStream = if uses_rotary_local {
        quote! { pub rotary_local: ::std::sync::OnceLock<crate::__gpu::rotary::RotaryCache>, }
    } else {
        quote! {}
    };

    let rotary_local_load_metal: TokenStream = if uses_rotary_local {
        let head_dim = model.text.head_dim.get() as usize;
        let local_theta = model.text.rope_local_for(&model.source_stem).get();
        quote! {
            let rotary_local = crate::__gpu::rotary::RotaryCache::new_from_gpuweights(
                gw,
                #head_dim,
                ::core::cmp::min(max_model_len, __rope_rows),
                #local_theta,
                None,
                crate::__gpu::dtype::DType::BF16,  // bf16 cos/sin (HEAD-original)
            )?;
        }
    } else {
        quote! {}
    };

    let rotary_local_init: TokenStream = if uses_rotary_local {
        quote! { rotary_local: ::std::sync::OnceLock::from(rotary_local), }
    } else {
        quote! {}
    };

    // Primary `rotary: RotaryCache` field. Scratchy owns rotary end-
    // to-end: `ForwardCtx` carries no rotary, the emitted forward
    // reads `wm.rotary`, and this block picks the right
    // `RotaryCache` constructor at macro-expansion time from the
    // manifest. Four cases cross-join `partial_rotary_factor` (None
    // ⇒ full head_dim, Some(f) ⇒ rotary_dim = f * head_dim) with
    // `rope_scaling` (None / Llama3 / LongRope).
    let uses_rotary = fuf.nodes.iter().any(|n| {
        n.inputs.iter().any(|i| {
            matches!(
                i,
                crate::fuf::FufInput::Extern {
                    kind: crate::classified::ExternKind::Rotary,
                    ..
                }
            )
        })
    });

    let rotary_field: TokenStream = if uses_rotary {
        quote! { pub rotary: ::std::sync::OnceLock<crate::__gpu::rotary::RotaryCache>, }
    } else {
        quote! {}
    };

    // Stream-free metal counterpart to `rotary_load`. The
    // `*_from_gpuweights` constructors cover every rope family the
    // cuda path emits — basic, Llama3, LongRoPE, YaRN, partial rotary
    // (combined with each scaling where the family allows it), and
    // the Gemma4 proportional-global class handled above — so the
    // metal match below is exhaustive with no refusal arm.
    //
    // Cache dtype is `BF16` to match the rest of the metal stack:
    // `CanonicalParams::METAL_DTYPE` defaults to bf16, the
    // `rope_append_bf16_specialized` shader binds `cos_sin` as
    // `device const bfloat*`, and Llama-3.x weights ship bf16 on
    // disk. (The previous codegen pinned this at BF16 too but the
    // kernel only had an `_f16_specialized` variant — bytes lined
    // up but the binding type didn't, so cosines/sines were
    // reinterpreted as fp16 and attention never aligned. Fixed by
    // landing the bf16 shader sibling.)
    //
    // Each table holds `min(max_model_len, __rope_rows)` rows: the runtime
    // `max_model_len` (a chat context of 4-8K on Llama-3.2's 131072 builds
    // 16-32× fewer), capped by `__rope_rows`, the most a request can reach:
    // `max_position_embeddings`. Whatever builds the tables binds `__rope_rows`
    // ahead of these loads: spyre's `load`, and metal's `build_rope_tables`,
    // whose `max_model_len` is also capped at the block table's reach.
    let rope_rows: TokenStream = if uses_rotary || uses_rotary_local {
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for Rotary")
            as usize;
        quote! { let __rope_rows: usize = #max_pos; }
    } else {
        quote! {}
    };
    let rotary_load_metal: TokenStream = if uses_rotary {
        let head_dim = model.text.head_dim.get() as usize;
        let rope_theta = model.text.rope_theta.get();
        let partial = model
            .scalars
            .get("partial_rotary_factor")
            .copied()
            .filter(|&f| (f - 1.0).abs() > 1e-9);
        let scaling = model.rope_scaling.clone();
        // YaRN (DeepSeek MLA): the rope portion uses `qk_rope_head_dim`
        // rather than the full `head_dim` — the same override the cuda
        // arm above applies.
        let yarn_rope_head_dim: Option<usize> =
            model.bounds.get("qk_rope_head_dim").map(|&v| v as usize);
        // rotary_dim = partial_rotary_factor * head_dim (e.g. Qwen3.5:
        // 0.25 * 256 = 64); `None` ⇒ full rotary (rotary_dim == head_dim).
        let rotary_dim_metal: Option<usize> =
            partial.map(|f| (f * head_dim as f64).round() as usize);
        // Gemma4 hybrid geometry: the `rotary` extern is the GLOBAL
        // class's cache — "proportional" rope on global_head_dim (512)
        // rotating global_partial_rotary_factor*512 = 128 dims with the
        // freq exponent denominator = the FULL head_dim (mlx
        // ProportionalRoPE). The sliding class keeps `rotary_local`
        // (full rotary at the base head_dim). Keyed on the
        // `global_partial_rotary_factor` scalar so no other arch routes
        // here.
        let global_proportional: Option<(usize, usize)> = model
            .scalars
            .get("global_partial_rotary_factor")
            .copied()
            .filter(|&f| (f - 1.0).abs() > 1e-9)
            .map(|f| {
                let g_hd = model
                    .bounds
                    .get("global_head_dim")
                    .map(|&v| v as usize)
                    .unwrap_or(head_dim);
                let g_rd = (f * g_hd as f64).round() as usize;
                (g_hd, g_rd)
            });
        if let Some((g_hd, g_rd)) = global_proportional {
            assert!(
                model.rope_scaling.is_none(),
                "proportional global rope cannot combine with rope_scaling"
            );
            let g_hd_lit = proc_macro2::Literal::usize_unsuffixed(g_hd);
            let g_rd_lit = proc_macro2::Literal::usize_unsuffixed(g_rd);
            quote! {
                let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                let rotary =
                    crate::__gpu::rotary::RotaryCache::new_proportional_from_gpuweights(
                        gw,
                        #g_hd_lit,
                        #g_rd_lit,
                        rope_max_pos,
                        #rope_theta,
                        crate::__gpu::dtype::DType::BF16,
                    )?;
            }
        } else {
            match (rotary_dim_metal, scaling) {
                (None, None) => quote! {
                    let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                    let rotary = crate::__gpu::rotary::RotaryCache::new_from_gpuweights(
                        gw,
                        #head_dim,
                        rope_max_pos,
                        #rope_theta,
                        None,
                        crate::__gpu::dtype::DType::BF16,  // bf16 cos/sin (HEAD-original)
                    )?;
                },
                (
                    None,
                    Some(crate::config::RopeScaling::Llama3 {
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                        original_max_position_embeddings,
                    }),
                ) => {
                    let orig = original_max_position_embeddings as usize;
                    quote! {
                        let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                        let rotary = crate::__gpu::rotary::RotaryCache::new_from_gpuweights(
                            gw,
                            #head_dim,
                            rope_max_pos,
                            #rope_theta,
                            Some(&crate::__gpu::rotary::Llama3RopeScaling {
                                factor: #factor,
                                low_freq_factor: #low_freq_factor,
                                high_freq_factor: #high_freq_factor,
                                original_max_position_embeddings: #orig,
                            }),
                            crate::__gpu::dtype::DType::BF16,  // bf16 cos/sin (HEAD-original)
                        )?;
                    }
                }
                // LongRoPE (Phi-3 family, `rope_scaling.type` =
                // "longrope" / "su"), full or partial rotary. The
                // per-channel factor table and mscale live in the
                // config; `RotaryCache::new_longrope_from_gpuweights`
                // does the same f32 math cuda's builder does, and
                // picks the short/long pair from `max_model_len`.
                (
                    rot_dim,
                    Some(crate::config::RopeScaling::LongRope {
                        short_factor,
                        long_factor,
                        original_max_position_embeddings,
                        short_mscale,
                        long_mscale,
                    }),
                ) => {
                    let rotary_dim_lit = rot_dim.unwrap_or(head_dim);
                    let orig = original_max_position_embeddings as usize;
                    let short: Vec<f64> = short_factor.clone();
                    let long: Vec<f64> = long_factor.clone();
                    let (sm, lm) = (short_mscale, long_mscale);
                    quote! {
                        let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                        let rotary = crate::__gpu::rotary::RotaryCache::new_longrope_from_gpuweights(
                            gw,
                            #head_dim,
                            #rotary_dim_lit,
                            crate::__gpu::rotary::LongRopeExtent {
                                max_pos: rope_max_pos,
                                max_model_len,
                            },
                            #rope_theta,
                            &crate::__gpu::rotary::LongRopeScaling {
                                short_factor: ::std::vec![#(#short),*],
                                long_factor: ::std::vec![#(#long),*],
                                original_max_position_embeddings: #orig,
                                short_mscale: #sm,
                                long_mscale: #lm,
                            },
                            crate::__gpu::dtype::DType::BF16,
                        )?;
                    }
                }
                // YaRN (`rope_scaling.type == "yarn"` — gpt-oss,
                // DeepSeek-V2), full or partial rotary. The host math
                // is shared with cuda's `new_yarn_from_stream` via
                // `yarn_cos_sin_table` (scratchy-layers), so the two
                // backends cannot drift; `qk_rope_head_dim` (MLA
                // models) overrides the rope dim exactly as the cuda
                // arm does.
                (
                    _,
                    Some(crate::config::RopeScaling::Yarn {
                        factor,
                        beta_fast,
                        beta_slow,
                        mscale,
                        mscale_all_dim,
                        original_max_position_embeddings,
                    }),
                ) => {
                    let orig = original_max_position_embeddings as usize;
                    let rope_hd = yarn_rope_head_dim.unwrap_or(head_dim);
                    quote! {
                        let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                        let rotary = crate::__gpu::rotary::RotaryCache::new_yarn_from_gpuweights(
                            gw,
                            #rope_hd,
                            rope_max_pos,
                            #rope_theta,
                            &crate::__gpu::rotary::YarnRopeScaling {
                                factor: #factor,
                                beta_fast: #beta_fast,
                                beta_slow: #beta_slow,
                                mscale: #mscale,
                                mscale_all_dim: #mscale_all_dim,
                                original_max_position_embeddings: #orig,
                            },
                            crate::__gpu::dtype::DType::BF16,  // bf16 cos/sin (HEAD-original)
                        )?;
                    }
                }
                // Partial rotary, no scaling (Qwen3.5 / Qwen3-Next:
                // partial_rotary_factor 0.25). Builds a `[max_pos, rotary_dim]`
                // cache; `W::ROT_DIM` (= rotary_dim) drives the kernel so the
                // trailing head_dim-rotary_dim channels pass through unrotated.
                (Some(rotary_dim), None) => quote! {
                    let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                    let rotary = crate::__gpu::rotary::RotaryCache::new_partial_from_gpuweights(
                        gw,
                        #head_dim,
                        #rotary_dim,
                        rope_max_pos,
                        #rope_theta,
                        None,
                        crate::__gpu::dtype::DType::BF16,
                    )?;
                },
                // Partial rotary + Llama3 scaling.
                (
                    Some(rotary_dim),
                    Some(crate::config::RopeScaling::Llama3 {
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                        original_max_position_embeddings,
                    }),
                ) => {
                    let orig = original_max_position_embeddings as usize;
                    quote! {
                        let rope_max_pos = ::core::cmp::min(max_model_len, __rope_rows);
                        let rotary = crate::__gpu::rotary::RotaryCache::new_partial_from_gpuweights(
                            gw,
                            #head_dim,
                            #rotary_dim,
                            rope_max_pos,
                            #rope_theta,
                            Some(&crate::__gpu::rotary::Llama3RopeScaling {
                                factor: #factor,
                                low_freq_factor: #low_freq_factor,
                                high_freq_factor: #high_freq_factor,
                                original_max_position_embeddings: #orig,
                            }),
                            crate::__gpu::dtype::DType::BF16,
                        )?;
                    }
                }
            }
        } // else: !global_proportional
    } else {
        quote! {}
    };

    let rotary_init: TokenStream = if uses_rotary {
        quote! { rotary: ::std::sync::OnceLock::from(rotary), }
    } else {
        quote! {}
    };
    // Metal's tables wait for its pool; the context `load` was asked for bounds them.
    let uses_rope = uses_rotary || uses_rotary_local;
    let rope_context_field: TokenStream = if uses_rope {
        quote! {
            /// The context `load` was asked for: with the block table's reach it bounds the
            /// rows of the RoPE tables the metal pool builds.
            pub max_model_len: usize,
        }
    } else {
        quote! {}
    };
    let rope_context_init: TokenStream = if uses_rope {
        quote! { max_model_len, }
    } else {
        quote! {}
    };
    let rotary_deferred_init: TokenStream = [
        (
            uses_rotary,
            quote! { rotary: ::std::sync::OnceLock::new(), },
        ),
        (
            uses_rotary_local,
            quote! { rotary_local: ::std::sync::OnceLock::new(), },
        ),
    ]
    .into_iter()
    .filter_map(|(used, init)| used.then_some(init))
    .collect();
    // `Weights::build_rope_tables`: the metal pool builds the RoPE tables before it binds them,
    // once the KV cache is sized, for the positions its block table reaches — the KV cap rung the
    // pool runs on, picked from `max_model_len` and the blocks the memory budget funds — rather
    // than at load for every position the checkpoint names. A launch whose KV cache holds less
    // than `max_position_embeddings` builds only the rows it can serve.
    let build_rope_tables: TokenStream = {
        let set_rotary = uses_rotary.then(|| {
            quote! {
                if self.rotary.set(rotary).is_err() {
                    ::anyhow::bail!("RoPE table built twice");
                }
            }
        });
        let set_rotary_local = uses_rotary_local.then(|| {
            quote! {
                if self.rotary_local.set(rotary_local).is_err() {
                    ::anyhow::bail!("local RoPE table built twice");
                }
            }
        });
        let body = if uses_rope {
            quote! {
                let reach = ctx.kv_cache.max_blocks_per_seq * ctx.kv_cache.block_size;
                let max_model_len = ::core::cmp::min(self.max_model_len, reach);
                let gw = &mut crate::__gpu::weights::GpuWeights::empty((*device.allocator).clone());
                #rope_rows
                #rotary_load_metal
                #rotary_local_load_metal
                #set_rotary
                #set_rotary_local
            }
        } else {
            quote! { let _ = (ctx, device); }
        };
        quote! {
            impl Weights {
                #[cfg(scratchy_target = "metal")]
                pub fn build_rope_tables(
                    &self,
                    ctx: &crate::__gpu::ForwardCtx,
                    device: &crate::__gpu::GpuDevice,
                ) -> ::anyhow::Result<()> {
                    #body
                    Ok(())
                }
            }
        }
    };

    // Per-arch accessor methods on `Weights`. Group accessor field
    // names by stripping any trailing `_<digits>` suffix; for each
    // base, emit `pub fn <base>(&self, layer: u32) -> &<Ty>` that
    // matches on the layer index. The host-interpreter's per-arch
    // enum variants carry `weight_fn: fn(&Weights, u32) -> &Ty`
    // pointing at one of these methods, so per-claim weight selection
    // is a const fn-pointer field rather than a baked-in `wm.<field>`
    // path. Non-layered accessors (no `_<digits>` suffix) get the
    // same signature for uniformity; their body is `&self.<field>`
    // and ignores the layer arg.
    // Weight-getter `impl Weights` (reads `self.<field>`). Emitted unconditionally
    // for every target — `crate::__gpu` resolves the field types per backend
    // (metal/spyre target crate, or `scratchy-layers` with no backend), so
    // the same accessors compile everywhere. No backend is special-cased.
    let accessor_methods = emit_weights_accessor_methods(&accessors, dims);

    // Rotary cos_sin accessors for the host-interpreter path. Both
    // `wm.rotary` and `wm.rotary_local` are conditionally-emitted
    // fields — the interpreter arm body is a single token stream
    // shared across every claim of an Impl on this arch, so it
    // can't directly write `wm.rotary_local.cos_sin_cache` (Llama
    // would fail to type-check). Per-claim selection rides on a
    // `cos_sin_fn: for<'a> fn(&'a Weights, u32) -> scratchy_tensors::GpuTensor`
    // OpInstance field; `fan_out` resolves it to one of these
    // accessor names. Llama's interpreter never sees `Weights::rotary_local_cos_sin`
    // because `RotaryLocal` extern doesn't appear in its FUF.
    let rotary_cos_sin_methods: TokenStream = match &mode {
        WeightsEmitMode::Canonical => {
            let main = if uses_rotary {
                quote! {
                    #[inline]
                    #[allow(dead_code)]
                    pub fn rotary_cos_sin(&self, _layer: u32)
                        -> crate::__gpu::tensor::GpuTensor
                    {
                        self.rotary.get().expect("RoPE tables are built at load, or with the metal pool (`build_rope_tables`)").cos_sin_cache
                    }
                }
            } else {
                quote! {}
            };
            let local = if uses_rotary_local {
                quote! {
                    #[inline]
                    #[allow(dead_code)]
                    pub fn rotary_local_cos_sin(&self, _layer: u32)
                        -> crate::__gpu::tensor::GpuTensor
                    {
                        self.rotary_local.get().expect("RoPE tables are built at load, or with the metal pool (`build_rope_tables`)").cos_sin_cache
                    }
                }
            } else {
                quote! {}
            };
            if uses_rotary || uses_rotary_local {
                quote! {
                    impl Weights {
                        #main
                        #local
                    }
                }
            } else {
                quote! {}
            }
        }
        // Shims share the canonical's Weights via type alias, so
        // they inherit these methods automatically.
        WeightsEmitMode::Shim { .. } => quote! {},
    };
    // Rotary cos_sin accessors read `self.rotary*` fields on `Weights`. Emitted
    // unconditionally for every target — `crate::__gpu` resolves the field type
    // per backend, so they compile everywhere. No backend is special-cased.

    // The `Weights` struct: one field per weight handle, emitted UNCONDITIONALLY
    // for every target. The field types are written through `crate::__gpu::…`,
    // which each backend crate (metal/spyre) provides — and `scratchy-layers`
    // provides in the no-backend default build. So the same struct compiles under
    // every target with no backend special-cased: spyre carries the same real
    // `Weights` metal does, differing only in which crate `__gpu` names.
    // The target's own fields on the canonical struct.
    let target_fields = emitter.weights_fields();
    let weights_def: TokenStream = match &mode {
        WeightsEmitMode::Canonical => quote! {
            /// Every weight the forward needs, packed for the solver-picked Impls.
            /// Construct via `load`; the fields are device weight handles.
            pub struct Weights {
                #(#fields)*
                #rotary_field
                #rotary_local_field
                #rope_context_field
                #target_fields
            }
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            /// Shim — shares canonical sibling's `Weights`.
            pub type Weights = super::#canonical::Weights;
        },
    };
    let weights_ctor: TokenStream = match &mode {
        WeightsEmitMode::Canonical => quote! { Weights },
        WeightsEmitMode::Shim { canonical } => quote! { super::#canonical::Weights },
    };
    let dims_module = match dims.module(model) {
        Ok(module) => module,
        Err(e) => {
            let msg = format!("model `{}`: its dims: {e}", model.source_stem);
            return (quote! { compile_error!(#msg); }, Vec::new());
        }
    };

    // The target's canonical `load`, over the shared per-accessor load plans.
    let target_load = emitter.load_fn(&crate::target::LoadCx {
        weights_ctor: &weights_ctor,
        lets: &lets,
        field_shorthand: &field_shorthand,
        packed_splits_prelude: &packed_splits_prelude,
        marlin_prelude: &marlin_prelude,
        bnb4_prelude: &bnb4_prelude,
        fp8_prelude: &fp8_prelude,
        rope_rows: &rope_rows,
        rotary_load: &rotary_load_metal,
        rotary_local_load: &rotary_local_load_metal,
        rotary_init: &rotary_init,
        rotary_local_init: &rotary_local_init,
        rotary_deferred_init: &rotary_deferred_init,
        rope_context_init: &rope_context_init,
    });

    // Canonical emits the full `load_with(marlin_storage)` body +
    // a thin `load()` wrapper that passes this variant's Marlin
    // format literal. Shim variants skip `load_with` entirely —
    // they just thread their own MarlinFormat into the canonical
    // sibling's `load_with`. rustc doesn't re-monomorphize the
    // shim's one-line delegation body, so the expensive load
    // compile work (N_layers × N_accessors lines) runs ONCE per
    // equivalence class.
    let tokens = match &mode {
        WeightsEmitMode::Canonical => quote! {
            #weights_def

            #dims_module

            #accessor_methods

            #rotary_cos_sin_methods

            #build_rope_tables

            #fingerprint_method

            #target_load
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            #weights_def

            #fingerprint_method

            /// Shim — delegates to its canonical's `load`.
            #[inline]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::LoadStream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                super::#canonical::load(gw, stream, max_model_len, tp_rank)
            }
        },
    };
    (tokens, fingerprint_rows)
}

/// A Weights field's `(base, layer)`: an accessor unrolled at layer `n` is named `{base}_{n}`
/// (`weight_field_stem`), its base that name less the suffix; an accessor of no unrolled layer is
/// its whole name. Layer-ness is the unroll index's, never a name's digits: a weight whose own path
/// ends in a numeral (Qwen2-VL's `merger.mlp.0` and `merger.mlp.2`) is two fields, not one
/// field's two layers.
pub(crate) fn base_layer(
    field_name: &str,
    layer: Option<UnrollIndex>,
) -> (String, Option<UnrollIndex>) {
    let suffix = layer.map(|n| format!("_{}", n.0));
    match suffix.as_deref().and_then(|s| field_name.strip_suffix(s)) {
        Some(base) => (base.to_string(), layer),
        None => (field_name.to_string(), None),
    }
}

/// An accessor's `(base, layer)` ([`base_layer`]), its layer its sources' unroll index.
fn accessor_base_layer(acc: &WeightAccessor) -> (String, Option<UnrollIndex>) {
    let layer = acc.source_weights.first().and_then(|(_, i)| *i);
    base_layer(&acc.name.to_string(), layer)
}

/// One row in the per-arch `Weights` shape. The field/struct emit,
/// the `impl Weights` accessor methods, and the `load_with` body
/// all walk the same group list so they stay in sync.
pub struct AccessorGroup<'a> {
    /// Field name on `Weights` — the accessor's stem with the
    /// `_<layer>` suffix stripped. Stays a plain `String` because
    /// callers also need it as a runtime format-string fragment.
    pub base: String,
    /// Shared element type. For Vec-compressed layered groups this
    /// is the `T` in `Vec<T>`; for unindexed and sparse-layered
    /// groups it's the field type as-is.
    pub rust_type: TokenStream,
    /// What shape this group lowers to.
    pub kind: AccessorGroupKind,
    /// `entries[i].0`: layer index. For `LayeredContiguous`,
    /// entries are sorted by layer and occupy `0..entries.len()`
    /// contiguously (the Vec-build relies on this). For
    /// `Unindexed`, a single entry with `None`. For `LayeredSparse`,
    /// entries are sorted by layer but may have gaps or start at
    /// `layer > 0` — the per-layer fallback handles either case.
    pub entries: Vec<(Option<UnrollIndex>, &'a WeightAccessor)>,
}

/// How an [`AccessorGroup`] is lowered. The Vec-compressed path is
/// only safe when the layered group fills `Vec[0..N]` contiguously;
/// real-world archs with conditional-per-layer accessors (e.g.
/// DeepSeek-V2's `moe` is layers 1..N — layer 0 is dense FFN) take
/// the legacy per-layer-fields fallback.
pub enum AccessorGroupKind {
    /// Single field, no layer arg. Field type is `T`. Accessor
    /// method ignores its `layer` arg and returns `&self.<base>`.
    Unindexed,
    /// Layered family that occupies `Vec[0..N]` contiguously.
    /// Field type is `Vec<T>`. Accessor method is
    /// `&self.<base>[layer as usize]`. Load body emits one
    /// `(0..N).map(|layer| …).collect()`.
    LayeredContiguous,
    /// Layered family with gaps or non-zero start (e.g. layers
    /// 1..N only). Per-layer fields `pub <base>_<L>: T` are
    /// emitted; accessor method is a `match layer { L => &self.<base>_<L>, … }`.
    /// Load body emits one `let <base>_<L> = …;` per entry.
    LayeredSparse,
}

/// The typed handles of a model's `Weights`: per field base, the handle its field holds (the
/// group's weight at its declared shape, spelled in the model's dim markers), and the bounds the
/// handles name, which [`ModelDims::module`] declares with the model's numbers.
#[derive(Default)]
pub(crate) struct ModelDims {
    handles: BTreeMap<String, TypedHandle>,
    bounds: std::collections::BTreeSet<String>,
    /// Each DSL weight a typed field holds, by `WeightId`: its marker in `__weights`.
    pub(crate) weight_markers: BTreeMap<u32, String>,
}

/// One field's typed handle: its path (`__handles::<Field>`, what the field, its accessor and
/// its source slot spell), its name there and the type it names, and the tensor the load mint
/// names on a mismatch.
pub(crate) struct TypedHandle {
    pub ty: TokenStream,
    /// The marker of the DSL weights the field stands in for (`__weights::<Marker>`): a field of
    /// one weight is that weight's marker, a packed pair's or a bundle's the field's own.
    pub weights: Ident,
    /// The markers of the DSL weights it holds.
    held: std::collections::BTreeSet<String>,
    name: Ident,
    def: TokenStream,
    pub tensor: String,
}

impl ModelDims {
    /// The handles of `groups`' fields: each group whose every layer is the same weight (or the
    /// same packed pair of weights) of a kind whose device layer states its declared extents, at
    /// the shape the arch's trace gives it. A packed pair is its two weights side by side along
    /// `N`, which only weights of one `K` are (`PackedN`).
    ///
    /// A linear every layer of which keeps the same MLX-affine codes packed on device (`codes`,
    /// `(bits, group)` from its load plan) is a quantized handle (`Q2`) of those codes; a group
    /// whose layers' codes differ (an OptiQ per-module map) gets none.
    ///
    /// A field holding a bundle an op reads whole (`bundles`, derived: `bundles::derive`) is a
    /// bundle handle (`Wb`) of its kind, each role at its derived shape.
    pub(crate) fn of(
        groups: &[AccessorGroup<'_>],
        program: &Program,
        inferred: &crate::shape::Inferred,
        codes: &dyn Fn(&WeightAccessor) -> Option<(u32, u32)>,
        bundles: &std::collections::BTreeMap<String, crate::bundles::Bundle>,
    ) -> Result<Self, String> {
        use crate::weight_vocab::WeightKind;
        let mut dims = Self::default();
        let w = quote!(::scratchy_tensors::typed);
        for g in groups {
            let sources = |acc: &WeightAccessor| -> Vec<WeightId> {
                acc.source_weights.iter().map(|(id, _)| *id).collect()
            };
            let Some((_, first)) = g.entries.first() else {
                continue;
            };
            let ids = sources(first);
            let kind = &first.kind;
            let untypable = |why: &str| {
                format!(
                    "field `{}` ({kind:?}) has no typed handle: {why}; the generator types every \
                     weight",
                    g.base
                )
            };
            if !g.entries.iter().all(|(_, acc)| sources(acc) == ids) {
                return Err(untypable("its layers read different weights"));
            }
            let layer = &g.rust_type;
            let def = 'def: {
                // A bundle an op reads whole: its kind and each role at its derived shape.
                let bundle = match ids[..] {
                    [id] => bundles.get(&program.weights.path(id).join(".")),
                    _ => None,
                };
                if let Some(bundle) = bundle {
                    let mut roles = Vec::new();
                    for shape in &bundle.roles {
                        let ds = (shape.iter())
                            .map(|d| dim_tokens(d, &mut dims.bounds))
                            .collect::<Result<Vec<_>, _>>()?;
                        roles.push(match &ds[..] {
                            [d] => quote!(#w::S1<#d>),
                            [k, n] => quote!(#w::S2<#k, #n>),
                            [e, k, n] => quote!(#w::S3<#e, #k, #n>),
                            _ => return Err(format!("a bundle role of rank {}", ds.len())),
                        });
                    }
                    let kind = format_ident!("{}", format!("K{:?}", bundle.kind));
                    break 'def quote!(#w::Wb<#w::#kind, (#(#roles,)*), #layer>);
                }
                // Every kind, by name: a new `WeightKind` is a compile error here, not a panic in
                // someone's build.
                match first.kind {
                    // Held by the shape handles below (`W1`, `W2`, `Q2`).
                    WeightKind::Linear
                    // An fp8 per-channel linear (`Fp8AnyLinear`): `[in, out]` with no affine codes,
                    // so the dense rank-2 handle (`W2`) holds it.
                    | WeightKind::Fp8
                    | WeightKind::Embedding
                    | WeightKind::AffineQuantEmbedding
                    | WeightKind::RmsNorm
                    | WeightKind::LayerNorm
                    | WeightKind::Tensor
                    | WeightKind::AttnSinks => {}
                    // Read whole as bundles: the bundle arm above holds them. Reaching here means
                    // the bundle table did not name this field.
                    WeightKind::FusedMoe
                    | WeightKind::SharedFusedMoe
                    | WeightKind::GemmaRouter
                    | WeightKind::GemmaSwitchGlu
                    | WeightKind::GptOssMoe
                    | WeightKind::GatedDeltaNet => {
                        return Err(untypable("a bundle kind the bundle table did not name"));
                    }
                    // No tape produces these as a struct field today: the tape's kinds are
                    // `weight_bindings::kind_of`'s, which names none of them (a gemm is Linear,
                    // Fp8 or an affine Linear; the rotary is no field). A producer that starts to
                    // emit one gives it a handle here first.
                    WeightKind::Marlin
                    | WeightKind::Bnb4
                    | WeightKind::DeepSeekMoe
                    | WeightKind::DeepSeekMoeFp8
                    | WeightKind::DeepSeekMoeGgml
                    | WeightKind::CosSin => {
                        return Err(untypable("no tape produces this kind as a field yet"));
                    }
                }
                let Some(shapes) = (ids.iter())
                    .map(|id| inferred.weights.get(id))
                    .collect::<Option<Vec<_>>>()
                else {
                    return Err(untypable("the trace gives a weight of it no shape"));
                };
                // Every layer's codes, which must be one: a group of two is no one handle.
                let layer_codes = codes(first);
                if g.entries.iter().any(|(_, acc)| codes(acc) != layer_codes) {
                    return Err(untypable("its layers hold different affine codes"));
                }
                let mut ty = |shape: &[crate::dim::Dim],
                              codes: Option<(u32, u32)>|
                 -> Result<Option<TokenStream>, String> {
                    let mut dim = |d| dim_tokens(d, &mut dims.bounds);
                    Ok(match (shape, codes) {
                        ([d], None) => {
                            let d = dim(d)?;
                            Some(quote!(#w::W1<#d, #layer>))
                        }
                        ([k, n], None) => {
                            let (k, n) = (dim(k)?, dim(n)?);
                            Some(quote!(#w::W2<#k, #n, #layer>))
                        }
                        ([k, n], Some((bits, group))) => {
                            let (k, n) = (dim(k)?, dim(n)?);
                            let (b, g) = (format_ident!("B{bits}"), format_ident!("G{group}"));
                            Some(quote!(#w::Q2<#k, #n, #w::#b, #w::#g, #layer>))
                        }
                        _ => None,
                    })
                };
                let handle = match &shapes[..] {
                    [one] => ty(one, layer_codes)?,
                    // A LayerNorm's weight and bias, each its width.
                    [weight, bias] if first.kind == WeightKind::LayerNorm && weight == bias => {
                        ty(weight, None)?
                    }
                    // A packed pair is dense: affine sources fused at load are dequantized.
                    [gate, up] if first.kind == WeightKind::Linear && layer_codes.is_none() => {
                        match (ty(gate, None)?, ty(up, None)?) {
                            (Some(gate), Some(up)) => {
                                Some(quote!(<(#gate, #up) as #w::PackedN>::Out))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                };
                let Some(def) = handle else {
                    let ranks: Vec<usize> = shapes.iter().map(|s| s.len()).collect();
                    return Err(untypable(&format!(
                        "no handle holds weights of ranks {ranks:?} with codes {layer_codes:?}"
                    )));
                };
                def
            };
            let tensor = (ids.iter())
                .map(|id| program.weights.path(*id).join("."))
                .collect::<Vec<_>>()
                .join(" + ");
            let name = format_ident!(
                "{}",
                scratchy_forward_compiler_frontend::typed_emit::marker(&g.base)
            );
            let ty = quote!(__handles::#name);
            // The DSL weights the field stands in for, each by its path's marker; a field of more
            // than one (a packed pair, a bundle) has its own marker holding each.
            let weight_marker = |id: &WeightId| {
                scratchy_forward_compiler_frontend::typed_emit::marker(
                    &program.weights.path(*id).join("_"),
                )
            };
            let held: std::collections::BTreeSet<String> = ids.iter().map(weight_marker).collect();
            for id in &ids {
                dims.weight_markers.insert(id.0, weight_marker(id));
            }
            let family = match held.iter().collect::<Vec<_>>().as_slice() {
                [one] => one.to_string(),
                _ => name.to_string(),
            };
            let handle = TypedHandle {
                ty,
                weights: format_ident!("{family}"),
                held,
                name,
                def,
                tensor,
            };
            dims.handles.insert(g.base.clone(), handle);
        }
        Ok(dims)
    }

    pub(crate) fn handle(&self, base: &str) -> Option<&TypedHandle> {
        self.handles.get(base)
    }

    /// `__weights`: a marker per DSL weight a typed field holds and per field of more than one,
    /// and what each field's holds (`HoldsWeight`), one concrete impl a weight.
    fn weights_module(&self) -> Option<TokenStream> {
        if self.handles.is_empty() {
            return None;
        }
        // Each DSL weight's marker, and each field's of more than one; each impl once (two
        // fields may hold one weight: a tied head and its embedding).
        let names: std::collections::BTreeSet<String> = (self.weight_markers.values().cloned())
            .chain(self.handles.values().map(|h| h.weights.to_string()))
            .collect();
        let markers = names.iter().map(|n| format_ident!("{n}"));
        let held: std::collections::BTreeSet<(String, &String)> = (self.handles.values())
            .flat_map(|h| h.held.iter().map(move |w| (h.weights.to_string(), w)))
            .collect();
        let holds = held.iter().map(|(field, w)| {
            let (field, w) = (format_ident!("{field}"), format_ident!("{w}"));
            quote!(impl ::scratchy_tensors::typed::HoldsWeight<#w> for #field {})
        });
        Some(quote! {
            /// The DSL weights the typed fields of `Weights` stand in for, as types, and what each
            /// field holds.
            pub mod __weights {
                #(pub enum #markers {})*
                #(#holds)*
            }
        })
    }

    /// Declare, beside the handles', every bound a tile output of `fuf` names that `model`
    /// states: the arena's extents (`arena_extent`) are spelled in `__dims` too, so a model has
    /// one marker per bound. A bucket's own bounds (`num_tokens`, `sk_bucket`: its rows, which
    /// [`arena_extent`] spells as types of their own) are no model's.
    pub(crate) fn declare_tile_bounds(&mut self, fuf: &crate::fuf::Fuf, model: &ModelParams) {
        fn named(dim: &crate::dim::Dim, out: &mut dyn FnMut(&str)) {
            use crate::dim::Dim;
            match dim {
                Dim::Bound(name) => out(name),
                Dim::Lit(_) => {}
                Dim::Mul(parts) | Dim::Add(parts) => parts.iter().for_each(|p| named(p, out)),
                Dim::Div(n, d) | Dim::FloorDiv(n, d) => {
                    named(n, out);
                    named(d, out);
                }
            }
        }
        let shapes = fuf.nodes.iter().flat_map(|n| n.outputs.iter().flatten());
        for dim in shapes {
            named(dim, &mut |name| {
                if model.bounds.contains_key(name) {
                    self.bounds.insert(name.to_string());
                }
            });
        }
    }

    /// `__dims`: one marker per bound the handles name, carrying `model`'s number for it. A bound
    /// the loader derived (`ModelParams::derived`) is spelled as its derivation over the bounds it
    /// reads, each a marker too, so the derivation's law is the type's: `head_dim` derived as
    /// `hidden_size / num_attention_heads` is that exact quotient, which rustc refuses where the
    /// heads do not divide the width. A derivation the bound's value is no longer (a later step
    /// restated it) is not the bound's, and its number is stated instead. Every bound the loader
    /// derived is declared, whether or not a handle names it (`vision_head_dim` as
    /// `vision_embed_dim / vision_num_heads`), so each derivation's law is a type of the model's.
    pub(crate) fn module(&self, model: &ModelParams) -> Result<TokenStream, String> {
        let mut named = self.bounds.clone();
        named.extend(
            model
                .derived
                .keys()
                .filter(|b| model.bounds.contains_key(*b))
                .cloned(),
        );
        let mut todo: Vec<String> = named.iter().cloned().collect();
        let mut markers = std::collections::BTreeMap::new();
        let mut spelled = false;
        while let Some(b) = todo.pop() {
            let n = *model.bounds.get(&b).ok_or_else(|| {
                format!("a weight's shape names `{b}`, which the model's config does not state")
            })?;
            u32::try_from(n).map_err(|_| format!("`{b}` = {n} exceeds u32"))?;
            let derived = (model.derived.get(&b))
                .filter(|d| crate::shape::eval_closed_dim(d, &model.bounds) == Some(n));
            let num = match derived {
                Some(dim) => {
                    let mut reads = std::collections::BTreeSet::new();
                    let t = dim_tokens(dim, &mut reads)?;
                    todo.extend(reads.into_iter().filter(|r| named.insert(r.clone())));
                    spelled = true;
                    quote!(<#t as ::scratchy_tensors::typed::Dim>::Num)
                }
                None => unsigned_tokens(n),
            };
            let marker = format_ident!(
                "{}",
                scratchy_forward_compiler_frontend::typed_emit::marker(&b)
            );
            markers.insert(
                b,
                quote! {
                    pub enum #marker {}
                    impl ::scratchy_tensors::typed::Dim for #marker {
                        type Num = #num;
                    }
                },
            );
        }
        let markers = markers.into_values();
        // A derivation names its reads as `__dims::<marker>`, as a handle does.
        let reads = spelled.then(|| {
            quote!(
                use super::__dims;
            )
        });
        let handles = (self.handles.values())
            .map(|TypedHandle { name, def, .. }| quote!(pub type #name = #def;));
        let handles = (!self.handles.is_empty()).then(|| {
            quote! {
                /// The handle each typed field of `Weights` holds, by the field's name.
                pub mod __handles {
                    use super::__dims;
                    #(#handles)*
                }
            }
        });
        let weights = self.weights_module();
        Ok(quote! {
            /// The model's dims, as types: each bound its weights' shapes name, with its number.
            pub mod __dims {
                #reads
                #(#markers)*
            }
            #handles
            #weights
        })
    }
}

/// `n` as its type-level number: a literal dim is the number itself
/// (`scratchy_tensors::typed::unsigned_path`).
fn unsigned_tokens(n: u64) -> TokenStream {
    (scratchy_tensors::typed::unsigned_path(n).parse()).expect("a typenum path is Rust tokens")
}

/// `dim` as a type in the model's `__dims`, each bound it names into `bounds`.
fn dim_tokens(
    dim: &crate::dim::Dim,
    bounds: &mut std::collections::BTreeSet<String>,
) -> Result<TokenStream, String> {
    dim_type(dim, &mut |name| {
        bounds.insert(name.to_string());
        let marker = bound_marker(name);
        Ok(quote!(__dims::#marker))
    })
}

/// ⭐ A MODEL'S ARENA TYPES, EACH DECLARED ONCE: every extent its buckets' tile outputs are, spelled
/// at a bucket's rows' parameters, as one marker (`E<n>`); every colour its buckets' colourings
/// make — a slot and the extents it hosts — as one colour type (`K<n>`); and every colouring as one
/// arena of them (`S<n>`). Buckets whose colourings agree share all three, at their own rows
/// (`S0<U8, U128>`, their numbers as `typenum` types).
#[derive(Default)]
struct ArenaTypes {
    /// Each extent's marker, by its spelling.
    extents: BTreeMap<String, String>,
    /// Each arena value's extent's marker: what the bake names a binding of it by.
    values: scratchy_subtile::handoff::ArenaExtents,
    extent_defs: Vec<TokenStream>,
    /// The extents that compute over a bucket's rows ([`computes_over_rows`]).
    computed_defs: Vec<TokenStream>,
    colours: BTreeMap<String, String>,
    colour_defs: Vec<TokenStream>,
    arenas: BTreeMap<Vec<String>, String>,
    arena_defs: Vec<TokenStream>,
}

impl ArenaTypes {
    /// The marker of extent `at` (spelled at the rows' parameters).
    /// `computed`: whether it computes over the rows ([`computes_over_rows`]).
    fn extent(&mut self, (at, computed): &(TokenStream, bool)) -> Ident {
        let n = self.extents.len();
        let defs = match computed {
            true => &mut self.computed_defs,
            false => &mut self.extent_defs,
        };
        let name = (self.extents.entry(at.to_string())).or_insert_with(|| {
            let name = format_ident!("E{n}");
            defs.push(quote!(#name = #at;));
            name.to_string()
        });
        format_ident!("{name}")
    }

    /// Note arena value `value`'s extent, `at`: its marker is the one a binding of it names.
    fn value(&mut self, value: scratchy_subtile::handoff::ArenaValue, at: &(TokenStream, bool)) {
        let marker = self.extent(at).to_string();
        self.values.insert(value, marker);
    }

    /// The colour type of slot `slot` hosting the extents `sized` (counted in its bytes) and
    /// `unsized_`.
    fn colour(&mut self, slot: u32, sized: &[Ident], unsized_: &[Ident]) -> Ident {
        let def = quote!(= #slot [#(#sized),*] [#(#unsized_),*]);
        let n = self.colours.len();
        let defs = &mut self.colour_defs;
        let name = (self.colours.entry(def.to_string())).or_insert_with(|| {
            let name = format_ident!("K{n}");
            defs.push(quote!(#name #def));
            name.to_string()
        });
        format_ident!("{name}")
    }

    /// The arena whose slot `k` is `colours[k]`.
    fn arena(&mut self, colours: &[Ident]) -> Ident {
        let n = self.arenas.len();
        let defs = &mut self.arena_defs;
        let key = colours.iter().map(Ident::to_string).collect();
        let name = (self.arenas.entry(key)).or_insert_with(|| {
            let name = format_ident!("S{n}");
            defs.push(quote!(#name [#(#colours),*]));
            name.to_string()
        });
        format_ident!("{name}")
    }

    /// The declarations, in the model's `__arena`.
    fn tokens(&self) -> TokenStream {
        let (extents, computed) = (&self.extent_defs, &self.computed_defs);
        let (colours, arenas) = (&self.colour_defs, &self.arena_defs);
        quote! {
            ::scratchy_target_metal::arena_extents! { <M, S> [#(#extents)*] [#(#computed)*] }
            ::scratchy_target_metal::arena_colours! { <M, S> #(#colours)* }
            ::scratchy_target_metal::arena_schemas! { <M, S> #(#arenas)* }
        }
    }
}

/// Whether an extent of `shape` computes over a bucket's rows: names `num_tokens` or `sk_bucket`
/// anywhere but as its rows themselves (`Act<M, D>`), so it is an extent only where its dims' laws
/// hold at the bucket's rows.
fn computes_over_rows(shape: &[crate::dim::Dim]) -> bool {
    use crate::dim::Dim;
    fn names_rows(dim: &Dim) -> bool {
        match dim {
            Dim::Bound(name) => name == "num_tokens" || name == "sk_bucket",
            Dim::Lit(_) => false,
            Dim::Mul(parts) | Dim::Add(parts) => parts.iter().any(names_rows),
            Dim::Div(n, d) | Dim::FloorDiv(n, d) => names_rows(n) || names_rows(d),
        }
    }
    match shape {
        [] => false,
        [d] => names_rows(d),
        [rows, rest @ ..] => {
            let plain_rows = matches!(rows, Dim::Bound(name) if name == "num_tokens");
            (!plain_rows && names_rows(rows)) || rest.iter().any(names_rows)
        }
    }
}

/// A tile output of `shape` as the arena extent a colour hosts, `Act<rows, columns>`: its first
/// dim its rows, the product of the rest its columns (a rank-1 value is one row). A bucket's own
/// bounds are the bucket's rows' types, `num_tokens` as `m` and `sk_bucket` as `sk` (a colour
/// type's parameters, or one bucket's numbers); any other is the model's `__dims`' marker
/// ([`ModelDims::declare_tile_bounds`]).
fn arena_extent(
    shape: &[crate::dim::Dim],
    m: &TokenStream,
    sk: &TokenStream,
) -> Result<TokenStream, String> {
    use crate::dim::Dim;
    let mut bound = |name: &str| -> Result<TokenStream, String> {
        match name {
            "num_tokens" => Ok(m.clone()),
            "sk_bucket" => Ok(sk.clone()),
            _ => {
                let marker = bound_marker(name);
                Ok(quote!(__dims::#marker))
            }
        }
    };
    let (rows, cols) = match shape {
        [] => (Dim::Lit(1), Dim::Lit(1)),
        [d] => (Dim::Lit(1), d.clone()),
        [r, c] => (r.clone(), c.clone()),
        [r, rest @ ..] => (r.clone(), Dim::Mul(rest.to_vec())),
    };
    let (rows, cols) = (dim_type(&rows, &mut bound)?, dim_type(&cols, &mut bound)?);
    // `Act` is in scope wherever an extent is spelled (`scratchy_tensors::typed::Act`).
    Ok(quote!(Act<#rows, #cols>))
}

/// The shape an arena value of `fuf` has: a tile output's, or — the one graph source a step
/// names in the arena, the embedded hidden — its `Embed` tile's.
fn arena_value_shape(
    fuf: &crate::fuf::Fuf,
    value: scratchy_subtile::handoff::ArenaValue,
) -> Result<&crate::dim::Shape, String> {
    use scratchy_subtile::handoff::ArenaValue;
    let output = |tile: u32, out: u8| {
        let node = fuf.nodes.get(tile as usize);
        node.and_then(|n| n.outputs.get(out as usize))
            .ok_or_else(|| format!("arena value of tile {tile} output {out}, which has none"))
    };
    match value {
        ArenaValue::Tile { tile, out } => output(tile, out),
        ArenaValue::Source(_) => {
            let embed = |n: &&crate::fuf::FufNode| n.op == crate::classified::OpKind::Embed;
            let mut embeds = fuf.nodes.iter().filter(embed);
            match (embeds.next(), embeds.next()) {
                (Some(embed), None) => output(embed.id.0, 0),
                _ => Err("the embedded hidden is not one `Embed` tile's output".to_string()),
            }
        }
    }
}

/// A config bound's marker type's name.
fn bound_marker(name: &str) -> Ident {
    format_ident!(
        "{}",
        scratchy_forward_compiler_frontend::typed_emit::marker(name)
    )
}

/// `dim` as a type, each bound it names spelled by `bound`: a product, sum or quotient over the
/// shared `scratchy_tensors::typed` operators, in the order the dim states it.
fn dim_type<F: FnMut(&str) -> Result<TokenStream, String>>(
    dim: &crate::dim::Dim,
    bound: &mut F,
) -> Result<TokenStream, String> {
    use crate::dim::Dim;
    let w = quote!(::scratchy_tensors::typed);
    match dim {
        Dim::Bound(name) => bound(name),
        Dim::Lit(n) => {
            u32::try_from(*n).map_err(|_| format!("literal dim {n} exceeds u32"))?;
            Ok(unsigned_tokens(*n))
        }
        Dim::Mul(parts) | Dim::Add(parts) => {
            let op = match dim {
                Dim::Mul(_) => quote!(Mul),
                _ => quote!(Add),
            };
            let types = (parts.iter())
                .map(|p| dim_type(p, bound))
                .collect::<Result<Vec<_>, _>>()?;
            let (last, rest) = types
                .split_last()
                .ok_or_else(|| format!("an empty {op} in a shape"))?;
            Ok((rest.iter().rev()).fold(last.clone(), |acc, t| quote!(#w::#op<#t, #acc>)))
        }
        Dim::Div(n, d) => {
            let (n, d) = (dim_type(n, bound)?, dim_type(d, bound)?);
            Ok(quote!(#w::Div<#n, #d>))
        }
        Dim::FloorDiv(n, d) => {
            let (n, d) = (dim_type(n, bound)?, dim_type(d, bound)?);
            Ok(quote!(#w::FloorDiv<#n, #d>))
        }
    }
}

/// Group `accessors` by their `(base, layer)` split.
///
/// Outcomes:
/// - Single unindexed accessor → `AccessorGroup { kind: Unindexed }`.
/// - Layered family covering `0..N` contiguously →
///   `AccessorGroup { kind: LayeredContiguous }` (Vec compression).
/// - Layered family with gaps or non-zero start →
///   `AccessorGroup { kind: LayeredSparse }` (per-layer fallback).
///
/// Panics on:
/// - a base mixing layered and unindexed entries;
/// - a base whose entries declare conflicting `rust_type`s.
pub fn group_accessors_by_base(accessors: &[WeightAccessor]) -> Vec<AccessorGroup<'_>> {
    use std::collections::BTreeMap;

    struct Bucket<'a> {
        rust_type: TokenStream,
        rust_type_str: String,
        layered_entries: BTreeMap<UnrollIndex, &'a WeightAccessor>,
        unindexed: Option<&'a WeightAccessor>,
    }

    let mut by_base: BTreeMap<String, Bucket<'_>> = BTreeMap::new();
    for acc in accessors {
        let (base, layer) = accessor_base_layer(acc);
        let ty_str = acc.rust_type.to_string();
        let entry = by_base.entry(base.clone()).or_insert_with(|| Bucket {
            rust_type: acc.rust_type.clone(),
            rust_type_str: ty_str.clone(),
            layered_entries: BTreeMap::new(),
            unindexed: None,
        });
        if entry.rust_type_str != ty_str {
            panic!(
                "Weights accessor base `{base}` has mismatched types across layers: \
                 `{}` vs `{ty_str}`",
                entry.rust_type_str,
            );
        }
        match layer {
            Some(n) => {
                if entry.layered_entries.insert(n, acc).is_some() {
                    panic!("Weights accessor base `{base}` has duplicate layer index {n}",);
                }
            }
            None => {
                if entry.unindexed.is_some() {
                    panic!("Weights accessor base `{base}` has multiple unindexed entries",);
                }
                entry.unindexed = Some(acc);
            }
        }
    }

    by_base
        .into_iter()
        .map(|(base, tape_index)| {
            match (tape_index.unindexed, tape_index.layered_entries.is_empty()) {
                (Some(acc), true) => AccessorGroup {
                    base,
                    rust_type: tape_index.rust_type,
                    kind: AccessorGroupKind::Unindexed,
                    entries: vec![(None, acc)],
                },
                (None, false) => {
                    let layers: Vec<UnrollIndex> =
                        tape_index.layered_entries.keys().copied().collect();
                    let starts_at_zero = layers.first().copied() == Some(UnrollIndex(0));
                    let contiguous = layers.iter().enumerate().all(|(i, l)| l.0 == i as u64);
                    let kind = if starts_at_zero && contiguous {
                        AccessorGroupKind::LayeredContiguous
                    } else {
                        // Real-world examples: DeepSeek MoE (layers
                        // 1..N), per-window-size attention overrides,
                        // any future per-layer-conditional accessor.
                        // Keep the legacy `match layer { … }` shape so
                        // these compile without forcing a Vec
                        // representation that doesn't fit.
                        AccessorGroupKind::LayeredSparse
                    };
                    let entries: Vec<(Option<UnrollIndex>, &WeightAccessor)> = layers
                        .iter()
                        .map(|l| (Some(*l), tape_index.layered_entries[l]))
                        .collect();
                    AccessorGroup {
                        base,
                        rust_type: tape_index.rust_type,
                        kind,
                        entries,
                    }
                }
                _ => panic!("Weights accessor base `{base}` mixes layered and non-layered fields",),
            }
        })
        .collect()
}

/// Convert an L=0-baked safetensors prefix (e.g.
/// `"model.layers.0.input_layernorm"` or
/// `"visual.blocks.0.attn.q"`) into a TokenStream that evaluates
/// to a runtime `String` for the layer in scope. The emitted tokens
/// reference a local `layer: u32` binding the caller plants in scope
/// (the closure arg of the Vec-build).
///
/// `vision_layered_root_with_zero` is the per-arch vision-side
/// `<default_root>.<layered_subpath>.0.` prefix (e.g.
/// `"visual.blocks.0."` for Qwen or
/// `"vision_tower.vision_model.encoder.layers.0."` for Gemma3-MM).
/// `None` for decoder bodies — the function only knows about
/// `model.layers.0.`.
fn layer_templated_prefix_expr(
    layer0_prefix: &str,
    vision_layered_root_with_zero: Option<&str>,
    decoder_layered_root_with_zero: Option<&str>,
) -> TokenStream {
    if let Some(tail) = layer0_prefix.strip_prefix("model.layers.0.") {
        // Route through `scratchy_forward_compiler::layer_weight_path(layer,
        // suffix)` instead of inlining `format!()`. The post-macro
        // expansion of `format!("model.layers.{}.X", layer)` is a
        // 5-line `::alloc::__export::must_use({
        //     ::alloc::fmt::format(format_args!(...))
        // })` block; the helper fn collapses every call site to
        // one line of expanded source. Fires per-layer per-accessor
        // per-canonical — thousands of times on llama.
        return quote! { ::scratchy_forward_compiler::layer_weight_path(layer, #tail) };
    }
    if layer0_prefix == "model.layers.0" {
        return quote! { ::std::format!("model.layers.{}", layer) };
    }
    if let Some(zero_prefix) = vision_layered_root_with_zero
        && let Some(tail) = layer0_prefix.strip_prefix(zero_prefix)
    {
        // Shave the trailing `.0.` from the zero_prefix to recover
        // the bare root (`visual.blocks` /
        // `vision_tower.vision_model.encoder.layers`) for the
        // runtime templater.
        let root = zero_prefix
            .strip_suffix(".0.")
            .or_else(|| zero_prefix.strip_suffix(".0"))
            .unwrap_or(zero_prefix);
        return quote! { ::scratchy_forward_compiler::vision_block_weight_path(#root, layer, #tail) };
    }
    if let Some(zero_prefix) = decoder_layered_root_with_zero
        && let Some(tail) = layer0_prefix.strip_prefix(zero_prefix)
    {
        // MM-decoder-prefixed root (e.g.
        // `language_model.model.layers`). Same shape as vision —
        // strip the trailing `.0.` to recover the bare root and emit
        // a `layer_weight_path_with_root` call.
        let root = zero_prefix
            .strip_suffix(".0.")
            .or_else(|| zero_prefix.strip_suffix(".0"))
            .unwrap_or(zero_prefix);
        return quote! { ::scratchy_forward_compiler::layer_weight_path_with_root(#root, layer, #tail) };
    }
    panic!(
        "layered accessor's L=0 prefix `{layer0_prefix}` doesn't \
         start with `model.layers.0` or the per-arch vision layered \
         root — codegen invariant violated"
    );
}

/// Strip the `model.layers.0.` prefix to recover the per-layer
/// suffix string the `load_layered_*` helpers in
/// `scratchy_layers::loaders` take. E.g. `"model.layers.0.input_layernorm"`
/// → `"input_layernorm"`. Panics on prefixes that don't start with
/// `model.layers.0.` — every layered accessor's L=0 prefix carries
/// that prefix by construction (see [`safetensors_prefix`]); a
/// mismatch surfaces a `default_required_weights` bug instead of
/// silently emitting a malformed helper call.
fn layered_suffix<'a>(
    layer0_prefix: &'a str,
    vision_layered_root_with_zero: Option<&str>,
    decoder_layered_root_with_zero: Option<&str>,
) -> &'a str {
    if let Some(s) = layer0_prefix.strip_prefix("model.layers.0.") {
        return s;
    }
    if let Some(zero_prefix) = vision_layered_root_with_zero
        && let Some(s) = layer0_prefix.strip_prefix(zero_prefix)
    {
        return s;
    }
    if let Some(zero_prefix) = decoder_layered_root_with_zero
        && let Some(s) = layer0_prefix.strip_prefix(zero_prefix)
    {
        return s;
    }
    panic!(
        "layered accessor's L=0 prefix `{layer0_prefix}` doesn't \
         start with `model.layers.0.` or the per-arch vision/decoder \
         layered root — codegen invariant violated"
    )
}

/// Emit the unindexed accessor's let-binding (kept as the
/// existing per-FieldLoad shape — the prefix is a baked `&str`
/// literal, so no runtime `format!` machinery is needed). The
/// returned tokens include the trailing `;` and the leading
/// `let #name = …`.
fn emit_unindexed_let(name: &syn::Ident, plan: &FieldLoad, tp_world_size: u8) -> TokenStream {
    use crate::tp_lowering::{ShardKind, shard_kind_for_dotted_prefix};
    let tp_world_lit = proc_macro2::Literal::u8_unsuffixed(tp_world_size);
    let sharded = tp_world_size > 1;
    match plan {
        FieldLoad::Embedding(prefix) => {
            // Embed paths route to vocab-parallel `_sharded` at tp>1
            // (matches Python vLLM `VocabParallelEmbedding`). Non-
            // embed names that happen to deserialize through the
            // `Embedding` FieldLoad arm fall back to `Replicate`.
            let kind = shard_kind_for_dotted_prefix(prefix);
            if sharded && kind == ShardKind::ShardDim0 {
                quote! {
                    let #name = crate::__gpu::layers::Embedding::load_sharded(
                        gw, #prefix, tp_rank as usize, #tp_world_lit as usize,
                    )?;
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers::Embedding::load(gw, #prefix)?;
                }
            }
        }
        FieldLoad::EmbeddingAffine {
            prefix,
            group_size,
            bits,
        } => {
            // MLX-affine int4 quantized embedding (Metal-only). P6 lift:
            // load the packed U32 weight + F16 scales + F16 affine
            // offsets verbatim into an `AffineQuantEmbedding` bundle;
            // the forward path emits `Instruction::AffineEmbed`, which
            // dispatches the fused `affine_embed_*_gs_*_b_4` gather +
            // dequant kernel (saves ~vocab * hidden * 1.5 bytes of
            // BF16 arena vs the old load-time CPU-dequant fallback).
            // No sharded variant — single-GPU is the only metal target.
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            quote! {
                let #name = crate::__gpu::layers::AffineQuantEmbedding::load(
                    gw, #prefix, #gs_lit, #bits_lit,
                )?;
            }
        }
        FieldLoad::EmbeddingAffineDequant {
            prefix,
            group_size,
            bits,
        } => {
            // Dense `Embedding` accessor + Affine storage (spyre's storage-agnostic
            // embed path): dequantize the affine triple host-side into a dense
            // embedding table so the loaded value matches the dense field type.
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            quote! {
                let #name = crate::__gpu::layers::Embedding::load_affine_dequant(
                    gw, #prefix, #gs_lit, #bits_lit,
                )?;
            }
        }
        FieldLoad::RmsNorm(prefix, eps) => quote! {
            let #name = crate::__gpu::layers::RmsNorm::load(gw, #prefix, #eps)?;
        },
        FieldLoad::GatedDeltaNet(prefix) => quote! {
            let #name = crate::__gpu::layers::GatedDeltaNetLayer::load(gw, #prefix)?;
        },
        FieldLoad::LayerNorm(prefix, eps) => quote! {
            let #name = crate::__gpu::layers::LayerNorm::load(gw, #prefix, #eps)?;
        },
        FieldLoad::LinearDense(prefix) => {
            // Per-Linear shard kind: q/k/v/gate/up/lm_head/embed →
            // ShardDim0 (column / vocab parallel); o/down → ShardDim1
            // (row parallel); else Replicate. lm_head's shard is what
            // makes Python's tied-weights case self-consistent — since
            // both embed_tokens and lm_head end up dim-0-sharded, the
            // tied `Linear::new(embed.weight, None)` in
            // `LinearTiedToEmbedding` below sees an already-sharded
            // weight without further work.
            let kind = shard_kind_for_dotted_prefix(prefix);
            match (sharded, kind) {
                (true, ShardKind::ShardDim0) => quote! {
                    let #name = crate::__gpu::layers::LinearLayer::load_dense_sharded(
                        gw, #prefix, 0usize, tp_rank as usize, #tp_world_lit as usize,
                    )?;
                },
                (true, ShardKind::ShardDim1) => quote! {
                    let #name = crate::__gpu::layers::LinearLayer::load_dense_sharded(
                        gw, #prefix, 1usize, tp_rank as usize, #tp_world_lit as usize,
                    )?;
                },
                _ => quote! {
                    // `load_dense_or_ggml`: try `take_quantized_linear`
                    // first (for `StorageFormat::Ggml` weights), fall
                    // back to dense safetensors path. Transparent on
                    // every existing safetensors model since the
                    // GGUF map is empty there.
                    let #name = crate::__gpu::layers::LinearLayer::load_dense_or_ggml(gw, #prefix)?;
                },
            }
        }
        FieldLoad::LinearConcat(prefixes) => {
            // Fused QKV / gate_up are always column-parallel — no
            // row-parallel concat exists in any current arch. Under
            // metal the macro emits a stream-free `_concat_packed`
            // path (CPU-concat then one allocator call); other builds
            // keep the GGUF/safetensors `_or_ggml` path. Metal is
            // single-GPU, so it emits no sharded variant.
            if cfg!(feature = "metal") {
                quote! {
                    let #name = crate::__gpu::layers::LinearLayer::load_dense_concat_packed(
                        gw,
                        &[ #(#prefixes),* ],
                    )?;
                }
            } else if sharded {
                quote! {
                    let #name = crate::__gpu::layers::LinearLayer::load_dense_concat_sharded(
                        gw,
                        &[ #(#prefixes),* ],
                        stream,
                        tp_rank as usize,
                        #tp_world_lit as usize,
                    )?;
                }
            } else {
                quote! {
                    // `load_dense_concat_or_ggml`: tries GGUF byte-pack
                    // first (when every prefix has a quantized linear),
                    // falls back to the existing safetensors concat
                    // path. Transparent on safetensors models.
                    let #name = crate::__gpu::layers::LinearLayer::load_dense_concat_or_ggml(
                        gw,
                        &[ #(#prefixes),* ],
                        stream,
                    )?;
                }
            }
        }
        FieldLoad::RawLinear(key) => quote! {
            // Raw nn.Parameter load — `<key>` is the verbatim
            // safetensors key (no `.weight` / `.bias` suffix).
            // No bias, no TP shard (raw projector weights are
            // global, not per-block / per-rank). The `LinearLayer`
            // wrapper carries the tensor through to the body's
            // gemm op via `dense_weight()`.
            let #name = crate::__gpu::layers::LinearLayer::load_raw(gw, #key)?;
        },
        FieldLoad::RawTensor(key) => quote! {
            let #name = gw.take(#key)?;
        },
        FieldLoad::LinearAffine {
            prefix,
            group_size,
            bits,
            in_features,
        } => {
            // MLX-affine int4 (Metal-only). No sharded variant —
            // single-GPU is the only metal target initially. No
            // GGUF fallback — affine and ggml are disjoint storage
            // formats.
            //
            // INT4 P3/P4 forward-time path (post-C4b): keeps the
            // packed / scales / biases on device so the solver's
            // `MetalAffineQmmImpl` can drive qmv (decode) / qmm_t
            // (prefill) kernels per `Instruction::AffineQmm`.
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            let inf_lit = proc_macro2::Literal::u32_unsuffixed(*in_features);
            quote! {
                let #name = crate::__gpu::layers::LinearLayer::load_affine_quant(
                    gw,
                    #prefix,
                    #gs_lit,
                    #bits_lit,
                    #inf_lit,
                )?;
            }
        }
        FieldLoad::LinearAffineConcat {
            prefixes,
            group_size,
            bits,
            in_features,
        } => {
            // Fused gate_up_proj / qkv_proj on mlx-community 4bit
            // repos — see `FieldLoad::LinearAffineConcat` doc.
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            let inf_lit = proc_macro2::Literal::u32_unsuffixed(*in_features);
            let prefix_lits: Vec<TokenStream> = prefixes
                .iter()
                .map(|p| {
                    let lit = syn::LitStr::new(p, proc_macro2::Span::call_site());
                    quote! { #lit }
                })
                .collect();
            quote! {
                let #name = crate::__gpu::layers::LinearLayer::load_affine_dequant_concat_as_dense(
                    gw,
                    &[#(#prefix_lits),*],
                    #gs_lit,
                    #bits_lit,
                    #inf_lit,
                )?;
            }
        }
        FieldLoad::LinearAffineRowConcat {
            prefixes,
            group_size,
            bits,
            in_features,
        } => {
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            let inf_lit = proc_macro2::Literal::u32_unsuffixed(*in_features);
            let prefix_lits = prefixes
                .iter()
                .map(|p| syn::LitStr::new(p, proc_macro2::Span::call_site()));
            quote! {
                let #name = crate::__gpu::layers::LinearLayer::load_affine_quant_rows_concat(
                    gw,
                    &[#(#prefix_lits),*],
                    #gs_lit,
                    #bits_lit,
                    #inf_lit,
                )?;
            }
        }
        FieldLoad::LinearNvfp4 { prefix, group_size } => {
            // NVFP4 int4 (Metal-only), single source. Keeps packed E2M1
            // weight + folded F16 scales on device for the forward-time
            // `nvfp4_qmv` / `nvfp4_qmm_t` kernels via
            // `Instruction::Nvfp4Qmm`.
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            quote! {
                let #name = crate::__gpu::layers::LinearLayer::load_nvfp4_quant(
                    gw,
                    #prefix,
                    #gs_lit,
                )?;
            }
        }
        FieldLoad::LinearNvfp4Concat {
            prefixes,
            group_size,
        } => {
            // Fused gate_up_proj / qkv_proj on ModelOpt NVFP4 repos —
            // see `FieldLoad::LinearNvfp4Concat` doc.
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let prefix_lits: Vec<TokenStream> = prefixes
                .iter()
                .map(|p| {
                    let lit = syn::LitStr::new(p, proc_macro2::Span::call_site());
                    quote! { #lit }
                })
                .collect();
            quote! {
                let #name = crate::__gpu::layers::LinearLayer::load_nvfp4_quant_concat(
                    gw,
                    &[#(#prefix_lits),*],
                    #gs_lit,
                )?;
            }
        }
        FieldLoad::LinearTiedToEmbedding {
            embed_ident,
            affine,
        } => {
            if let Some((group_size, bits)) = affine {
                // P6: tied lm_head + Affine source embedding. The
                // embed_tokens field is an `AffineQuantEmbedding`
                // carrying the packed U32 weight + F16 scales + F16
                // affine offsets; the lm_head reads those same
                // buffers via `LinearLayer::AffineQuant`. `GpuTensor`
                // is Copy on metal (raw pointer wrapper), so the
                // buffer triple is shared with the embedding's
                // forward-time gather kernel without ownership
                // gymnastics. `in_features = hidden_size`,
                // `out_features = vocab_size` (Linear matmul reads
                // hidden inputs and produces vocab logits).
                let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
                let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
                quote! {
                    let #name = crate::__gpu::layers::LinearLayer::AffineQuant(
                        Box::new(crate::__gpu::layers::AffineQuantLinear {
                            weight: #embed_ident.weight,
                            scales: #embed_ident.scales,
                            affine_biases: #embed_ident.affine_biases,
                            linear_bias: None,
                            in_features: #embed_ident.hidden_size,
                            out_features: #embed_ident.vocab_size,
                            group_size: #gs_lit,
                            bits: #bits_lit,
                        })
                    );
                }
            } else {
                quote! {
                    // Tied embedding (dense): lm_head reuses the
                    // `#embed_ident` field's weight tensor. Shape
                    // [vocab_size, hidden_size] works for both
                    // Embedding (gather rows) and LinearLayer
                    // (matmul against hidden_size). No bias.
                    let #name = crate::__gpu::layers::LinearLayer::Dense(
                        crate::__gpu::layers::Linear::new(
                            #embed_ident.weight,
                            None,
                        )
                    );
                }
            }
        }
        FieldLoad::MarlinLinear { prefixes, .. } => {
            // Every Marlin accessor emits the SAME call shape
            // regardless of AWQ/GPTQ/CT: the runtime
            // `MarlinFormat` discriminator is threaded in from
            // `load_with`'s `marlin_storage` param. That's what
            // lets cross-variant load-body dedup collapse
            // AWQ/GPTQ/CT variants of the same (arch, size) to
            // one canonical `load_with` body.
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                quote! {
                    let #name = crate::__gpu::layers::MarlinLinear::load(
                        gw,
                        #prefix,
                        marlin_storage,
                        __marlin_ws,
                        __device_id,
                    )?;
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers::MarlinLinear::load_concat(
                        gw,
                        &[ #(#prefixes),* ],
                        marlin_storage,
                        __marlin_ws,
                        __device_id,
                    )?;
                }
            }
        }
        FieldLoad::Fp8Linear { prefixes } => {
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                quote! {
                    let #name = crate::__gpu::layers::Fp8AnyLinear::Std(
                        crate::__gpu::layers::Fp8Linear::load(
                            gw,
                            #prefix,
                            __fp8_dtype,
                        )?
                    );
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers::Fp8AnyLinear::Std(
                        crate::__gpu::layers::Fp8Linear::load_concat(
                            gw,
                            &[ #(#prefixes),* ],
                            __fp8_dtype,
                        )?
                    );
                }
            }
        }
        FieldLoad::Fp8BlockLinear { prefixes } => {
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                quote! {
                    let #name = crate::__gpu::layers::Fp8AnyLinear::Block(
                        crate::__gpu::layers::Fp8BlockLinear::load(
                            gw,
                            #prefix,
                            __fp8_dtype,
                        )?
                    );
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers::Fp8AnyLinear::Block(
                        crate::__gpu::layers::Fp8BlockLinear::load_concat(
                            gw,
                            &[ #(#prefixes),* ],
                            __fp8_dtype,
                        )?
                    );
                }
            }
        }
        FieldLoad::Bnb4Linear {
            prefixes,
            out_features_per_shard,
            in_features,
            blocksize,
        } => {
            let in_features = *in_features as usize;
            let blocksize = *blocksize as usize;
            let outs: Vec<proc_macro2::Literal> = out_features_per_shard
                .iter()
                .map(|o| proc_macro2::Literal::usize_unsuffixed(*o as usize))
                .collect();
            if prefixes.len() == 1 {
                let prefix = &prefixes[0];
                let out = &outs[0];
                quote! {
                    let #name = crate::__gpu::layers::Bnb4bitLinear::load(
                        gw,
                        #prefix,
                        __bnb_code,
                        __bnb_scratch,
                        #out,
                        #in_features,
                        #blocksize,
                    )?;
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers::Bnb4bitLinear::load_concat(
                        gw,
                        &[ #(#prefixes),* ],
                        __bnb_code,
                        __bnb_scratch,
                        &[ #(#outs),* ],
                        #in_features,
                        #blocksize,
                    )?;
                }
            }
        }
        FieldLoad::DeepSeekV2Moe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                let #name = crate::__gpu::layers_moe::DeepSeekV2MoELayer::load(
                    gw,
                    #prefix,
                    #n_routed_experts,
                    #n_shared_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #norm_topk_prob,
                    #routed_scaling_factor,
                    #use_sigmoid,
                    #n_expert_group,
                    #topk_group,
                    stream,
                )?;
            }
        }
        FieldLoad::DeepSeekV2Fp8BlockMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                let #name = crate::__gpu::layers_moe::DeepSeekV2Fp8BlockMoELayer::load(
                    gw,
                    #prefix,
                    #n_routed_experts,
                    #n_shared_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #norm_topk_prob,
                    #routed_scaling_factor,
                    #use_sigmoid,
                    #n_expert_group,
                    #topk_group,
                    __fp8_dtype,
                    stream,
                )?;
            }
        }
        FieldLoad::DeepSeekV2GgmlMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                let #name = crate::__gpu::layers_moe::DeepSeekV2GgmlMoELayer::load_gguf(
                    gw,
                    #prefix,
                    #n_routed_experts,
                    #n_shared_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #norm_topk_prob,
                    #routed_scaling_factor,
                    #use_sigmoid,
                    #n_expert_group,
                    #topk_group,
                    stream,
                )?;
            }
        }
        FieldLoad::FusedMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        } => {
            let num_experts = *num_experts;
            let top_k = *top_k;
            let intermediate_size = *intermediate_size;
            let hidden_size = *hidden_size;
            if let Some((group_size, bits)) = *affine {
                let gate_bits = gate_bits.unwrap_or(bits.up());
                let bits = bits.tokens();
                quote! {
                    let #name = crate::__gpu::layers_moe::FusedMoELayer::load_affine(
                        gw,
                        #prefix,
                        #num_experts,
                        #top_k,
                        #intermediate_size,
                        #hidden_size,
                        #group_size,
                        #bits,
                        #gate_bits,
                    )?;
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers_moe::FusedMoELayer::load(
                        gw,
                        #prefix,
                        #num_experts,
                        #top_k,
                        #intermediate_size,
                        #hidden_size,
                        stream,
                    )?;
                }
            }
        }
        FieldLoad::SharedFusedMoe {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        } => {
            let num_experts = *num_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let shared_expert_intermediate_size = *shared_expert_intermediate_size;
            let hidden_size = *hidden_size;
            if let Some((group_size, bits)) = *affine {
                let gate_bits = gate_bits.unwrap_or(bits.up());
                let bits = bits.tokens();
                quote! {
                    let #name = crate::__gpu::layers_moe::SharedFusedMoELayer::load_affine(
                        gw,
                        #prefix,
                        #num_experts,
                        #top_k,
                        #moe_intermediate_size,
                        #shared_expert_intermediate_size,
                        #hidden_size,
                        #group_size,
                        #bits,
                        #gate_bits,
                    )?;
                }
            } else {
                quote! {
                    let #name = crate::__gpu::layers_moe::SharedFusedMoELayer::load(
                        gw,
                        #prefix,
                        #num_experts,
                        #top_k,
                        #moe_intermediate_size,
                        #shared_expert_intermediate_size,
                        #hidden_size,
                        stream,
                    )?;
                }
            }
        }
        FieldLoad::GemmaRouter {
            prefix,
            num_experts,
            hidden_size,
            affine,
        } => {
            let num_experts = *num_experts;
            let hidden_size = *hidden_size;
            let affine = gemma_router_affine_ts(*affine);
            quote! {
                let #name = crate::__gpu::layers_moe::GemmaRouterLayer::load(
                    gw,
                    #prefix,
                    #num_experts,
                    #hidden_size,
                    #affine,
                )?;
            }
        }
        FieldLoad::GemmaSwitchGlu {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            group_size,
            bits,
        } => {
            let num_experts = *num_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let group_size = *group_size;
            let bits = *bits;
            quote! {
                let #name = crate::__gpu::layers_moe::SwitchGluExpertsLayer::load(
                    gw,
                    #prefix,
                    #num_experts,
                    #top_k,
                    #moe_intermediate_size,
                    #hidden_size,
                    #group_size,
                    #bits,
                )?;
            }
        }
        FieldLoad::GptOssMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        } => {
            let num_experts = *num_experts;
            let top_k = *top_k;
            let intermediate_size = *intermediate_size;
            let hidden_size = *hidden_size;
            // Op-keyed arm: the Dense/verbatim config reaches here too and
            // must still codegen (it is never the runtime-selected
            // specialization on metal), so default its storage params —
            // the GemmaSwitchGlu tolerance.
            let (group_size, bits) = affine.unwrap_or((64, 4));
            let gate_bits = gate_bits.unwrap_or(bits);
            quote! {
                let #name = crate::__gpu::layers_moe::GptOssMoELayer::load(
                    gw,
                    #prefix,
                    #num_experts,
                    #top_k,
                    #intermediate_size,
                    #hidden_size,
                    #group_size,
                    #bits,
                    #gate_bits,
                )?;
            }
        }
    }
}

/// Quote a [`FieldLoad::GemmaRouter::affine`] `Option<(u32, u32)>` for the
/// emitted load call — the full `::core` path so it survives whatever `use`
/// the emitted module carries. Both load emissions (single and layered) take
/// the SAME token stream.
fn gemma_router_affine_ts(affine: Option<(u32, u32)>) -> TokenStream {
    match affine {
        Some((group_size, bits)) => {
            let group_size = proc_macro2::Literal::u32_unsuffixed(group_size);
            let bits = proc_macro2::Literal::u32_unsuffixed(bits);
            quote! { ::core::option::Option::Some((#group_size, #bits)) }
        }
        None => quote! { ::core::option::Option::None },
    }
}

/// The affine bits a layered plan's load call takes: one width, or a fused-MoE bank's
/// per-projection widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanBits {
    Width(u32),
    Expert(crate::quantization::ExpertBits),
}

impl quote::ToTokens for PlanBits {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Width(b) => proc_macro2::Literal::u32_unsuffixed(*b).to_tokens(tokens),
            Self::Expert(b) => b.tokens().to_tokens(tokens),
        }
    }
}

/// The on-disk affine bits a layered plan loads with, if any. Used
/// to detect MLX-native mixed/dynamic-quant groups (OptiQ) where numbered
/// layers differ in bits, so the layered emit can switch to a per-layer
/// loader. `None` for plans with no single affine bits.
fn field_load_affine_bits(plan: &FieldLoad) -> Option<PlanBits> {
    match plan {
        FieldLoad::LinearAffine { bits, .. }
        | FieldLoad::LinearAffineConcat { bits, .. }
        | FieldLoad::LinearAffineRowConcat { bits, .. }
        | FieldLoad::GemmaSwitchGlu { bits, .. } => Some(PlanBits::Width(*bits)),
        // MoE bundles carry the routed-expert bits in `affine`; the router
        // `gate_bits` is uniform across layers so it stays out of the
        // per-layer gather (the collapsed layer-0 plan supplies it).
        FieldLoad::FusedMoe { affine, .. } | FieldLoad::SharedFusedMoe { affine, .. } => {
            affine.map(|(_, b)| PlanBits::Expert(b))
        }
        FieldLoad::GptOssMoe { affine, .. } => affine.map(|(_, b)| PlanBits::Width(b)),
        _ => None,
    }
}

/// Emit the let-binding(s) for one [`AccessorGroup`].
///
/// - `Unindexed` → one `let <base> = <load_call>;` (delegates to
///   [`emit_unindexed_let`]).
/// - `LayeredContiguous` → one `let <base>: Vec<T> = (0..N).map(|layer|
///   { … }).collect()?;`. The per-iteration body comes from
///   [`emit_layered_load_body`], which rewrites the layer-0-baked
///   prefix(es) into runtime `format!()` calls.
/// - `LayeredSparse` → one `let <base>_<L> = …;` per entry — same
///   shape as the legacy pre-Vec-compression let chain. Used when
///   the layered family has gaps or doesn't start at layer 0
///   (e.g. DeepSeek MoE on layers 1..N), since `Vec[layer as usize]`
///   would be off-by-one without an offset.
fn emit_group_let(
    group: &AccessorGroup<'_>,
    plans: &std::collections::BTreeMap<String, &FieldLoad>,
    model: &ModelParams,
    tp_world_size: u8,
    is_vision: bool,
) -> TokenStream {
    match group.kind {
        AccessorGroupKind::Unindexed => {
            let base_ident = syn::Ident::new(&group.base, proc_macro2::Span::call_site());
            let acc = group.entries[0].1;
            let plan = plans
                .get(&acc.name.to_string())
                .copied()
                .expect("unindexed accessor missing from plan map");
            emit_unindexed_let(&base_ident, plan, tp_world_size)
        }
        AccessorGroupKind::LayeredSparse => {
            // Each entry keeps its per-layer field name and gets
            // its own per-FieldLoad let — same as the pre-grouping
            // codegen. Order follows `entries`'s sort by layer
            // index.
            let lets: Vec<TokenStream> = group
                .entries
                .iter()
                .map(|(_, acc)| {
                    let plan = plans
                        .get(&acc.name.to_string())
                        .copied()
                        .expect("sparse-layered accessor missing from plan map");
                    emit_unindexed_let(&acc.name, plan, tp_world_size)
                })
                .collect();
            quote! { #(#lets)* }
        }
        AccessorGroupKind::LayeredContiguous => {
            let base_ident = syn::Ident::new(&group.base, proc_macro2::Span::call_site());
            // Plan from the first entry (layer 0). All entries share
            // this plan modulo prefix; emit_layered_load_body
            // delegates to a `crate::__gpu::load_layered_*`
            // helper that owns the (0..N).map().collect() loop. The
            // call site collapses to one line of expanded source.
            let l0 = group.entries[0].1;
            let plan = plans
                .get(&l0.name.to_string())
                .copied()
                .expect("layered group's layer-0 accessor missing from plan map");
            // MLX-native mixed/dynamic quant (OptiQ): each per-layer
            // accessor was planned at its own layer index, so a group whose
            // layers ship different bit-widths yields differing plan bits.
            // Gather them; `Some(vec)` only when heterogeneous (uniform
            // groups keep the byte-identical single-bits fast path).
            let per_layer_bits: Option<Vec<PlanBits>> = {
                let bits: Vec<Option<PlanBits>> = group
                    .entries
                    .iter()
                    .map(|(_, acc)| {
                        plans
                            .get(&acc.name.to_string())
                            .and_then(|p| field_load_affine_bits(p))
                    })
                    .collect();
                match bits.iter().all(Option::is_some) {
                    true => {
                        let v: Vec<PlanBits> = bits.into_iter().map(Option::unwrap).collect();
                        (v.iter().any(|b| *b != v[0])).then_some(v)
                    }
                    false => None,
                }
            };
            // `n_layers` may be less than `num_hidden_layers` — a
            // contiguous-from-zero group can be partial (e.g.
            // DeepSeek's `mlp_down_proj` is layer 0 only; layers 1..N
            // use `moe` instead). The helper is parameterized by
            // `entries.len()`, and the static-slice rows only ever
            // index 0..n_layers, so partial coverage is fine.
            let n_layers = group.entries.len() as u32;
            // Vision-side layered root: `<default_root>.<layered_subpath>`
            // baked from `vision_safetensors_layout` in the per-arch config.
            // Examples: `visual.blocks` (Qwen), `vision_tower.vision_model
            // .encoder.layers` (Gemma3-MM). `None` for decoder bodies.
            let vision_root_owned: Option<String> = if is_vision {
                let layout = model
                    .arch
                    .safetensors
                    .clone()
                    .expect("vision safetensors layout enforced at config parse");
                Some(format!(
                    "{}.{}",
                    layout.default_root, layout.layered_subpath
                ))
            } else {
                None
            };
            // Decoder-side layered root, from the one decoder-root
            // resolution [`safetensors_prefix`] also keys by.
            let decoder_root_owned = under_root(&model.arch.decoder_root(), "layers");
            let call = emit_layered_load_body(
                plan,
                n_layers,
                tp_world_size,
                is_vision,
                vision_root_owned.as_deref(),
                &decoder_root_owned,
                per_layer_bits.as_deref(),
            );
            quote! {
                let #base_ident = #call;
            }
        }
    }
}

/// Emit the full `Result<Vec<T>>` expression for a layered
/// accessor's load. Delegates to a `crate::__gpu::load_layered_*`
/// helper that owns the `(0..N).map(|layer| Type::load(…)).collect()`
/// loop, replacing the previous emit-the-closure-body shape. Each
/// call site collapses from ~4 lines (typed Vec annotation, range,
/// closure, collect) to one.
///
/// `LinearTiedToEmbedding` is unreachable here — tied embedding
/// only attaches to the `lm_head` accessor, which is unindexed.
/// `DeepSeekV2Moe` falls through to a layered-MoE helper still
/// emitted inline (only DeepSeek arches use it; not worth a helper
/// crossing the scratchy-forward-compiler / target-crate seam).
fn emit_layered_load_body(
    plan: &FieldLoad,
    n_layers: u32,
    tp_world_size: u8,
    is_vision: bool,
    vision_layered_root: Option<&str>,
    decoder_layered_root: &str,
    // `Some(bits per layer)` when this layered group is MLX-native
    // mixed/dynamic quant (OptiQ) with differing per-layer bit-widths;
    // the affine arms then emit the per-layer loader. `None` = uniform.
    per_layer_bits: Option<&[PlanBits]>,
) -> TokenStream {
    use crate::tp_lowering::{ShardKind, shard_kind_for_dotted_prefix};
    let n_lit = proc_macro2::Literal::u32_unsuffixed(n_layers);
    // Literal `[b0, b1, ...]` per-layer bits slice for the mixed loaders.
    let per_layer_bits_lit: Option<TokenStream> =
        per_layer_bits.map(|bits| quote! { &[ #(#bits),* ] });
    let tp_world_lit = proc_macro2::Literal::u8_unsuffixed(tp_world_size);
    let sharded = tp_world_size > 1;
    // Build the `<root>.0.` form for `layered_suffix` /
    // `layer_templated_prefix_expr` consumers. `None` for decoder bodies.
    let vision_zero_prefix: Option<String> = vision_layered_root.map(|r| format!("{r}.0."));
    let vision_zero_prefix_ref: Option<&str> = vision_zero_prefix.as_deref();
    let vision_root_lit_opt: Option<TokenStream> = vision_layered_root.map(|root| {
        let lit = syn::LitStr::new(root, proc_macro2::Span::call_site());
        quote! { #lit }
    });
    // Decoder layered root literal for the `*_with_root`-aware
    // load_layered_* helpers. `"model.layers"` for text-only and
    // Qwen-style VL; `"language_model.model.layers"` (or wherever
    // `decoder_safetensors_prefix` points) for Gemma3-MM-style arches
    // where HF nests the text decoder.
    let dec_root_lit: TokenStream = {
        let lit = syn::LitStr::new(decoder_layered_root, proc_macro2::Span::call_site());
        quote! { #lit }
    };
    // `<decoder_root>.0.` form for `layered_suffix` /
    // `layer_templated_prefix_expr` consumers under decoder bodies
    // whose variant overrides the default `model.layers.<L>.<suffix>`
    // template (Gemma3-MM nests under `language_model.<...>`). Empty
    // when the root is the canonical `model.layers` (no override).
    let decoder_zero_prefix: Option<String> = if decoder_layered_root != "model.layers" {
        Some(format!("{decoder_layered_root}.0."))
    } else {
        None
    };
    let decoder_zero_prefix_ref: Option<&str> = decoder_zero_prefix.as_deref();
    // Root literal for the quantized layered loaders (affine / nvfp4),
    // whose helpers take the root as a plain string parameter: the
    // vision layout's `<root>.<subpath>` under vision bodies, else the
    // decoder root. Written when only text was ever quantized — a
    // quantized vision tower with `#dec_root_lit` queries
    // `model.layers.<L>.<leaf>` and fails the load on the first block.
    let quant_root_lit: TokenStream = if is_vision {
        vision_root_lit_opt
            .clone()
            .expect("is_vision=true requires vision_layered_root")
    } else {
        dec_root_lit.clone()
    };
    match plan {
        FieldLoad::Embedding(prefix) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            // Embedding is vocab-parallel at tp>1 (matches Python
            // VocabParallelEmbedding). Layered embed accessors don't
            // exist in any current arch but the helper is here for
            // codegen uniformity; non-`embed_tokens` last segments
            // fall back to the unsharded path.
            let kind = shard_kind_for_dotted_prefix(prefix);
            if sharded && kind == ShardKind::ShardDim0 {
                quote! {
                    crate::__gpu::load_layered_embedding_sharded(
                        gw, #n_lit, #dec_root_lit, #suffix, tp_rank as usize, #tp_world_lit as usize,
                    )?
                }
            } else {
                quote! {
                    crate::__gpu::load_layered_embedding(gw, #n_lit, #dec_root_lit, #suffix)?
                }
            }
        }
        FieldLoad::EmbeddingAffine { .. } | FieldLoad::EmbeddingAffineDequant { .. } => {
            // No per-arch model in the int4 P2 coverage matrix has a
            // *layered* affine embedding — `embed_tokens` is top-level
            // on every Llama / Qwen / Mistral / Gemma / DeepSeek-V3 4bit
            // checkpoint. The unindexed `emit_unindexed_let` arm covers
            // every concrete model; if a future arch surfaces a layered
            // affine embedding, add `load_layered_affine_dequant_embedding`
            // alongside the existing layered helpers.
            panic!(
                "codegen: layered FieldLoad::EmbeddingAffine{{,Dequant}} is not wired — no model \
                 in the int4 P2 coverage matrix exercises a per-layer affine embedding"
            );
        }
        FieldLoad::RmsNorm(prefix, eps) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            if is_vision {
                let root = vision_root_lit_opt
                    .clone()
                    .expect("is_vision=true requires vision_layered_root");
                quote! {
                    crate::__gpu::load_layered_rms_norm_vision(gw, #n_lit, #root, #suffix, #eps)?
                }
            } else {
                quote! {
                    crate::__gpu::load_layered_rms_norm(gw, #n_lit, #dec_root_lit, #suffix, #eps)?
                }
            }
        }
        FieldLoad::LayerNorm(prefix, eps) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            if is_vision {
                let root = vision_root_lit_opt
                    .clone()
                    .expect("is_vision=true requires vision_layered_root");
                quote! {
                    crate::__gpu::load_layered_layer_norm_vision(gw, #n_lit, #root, #suffix, #eps)?
                }
            } else {
                quote! {
                    crate::__gpu::load_layered_layer_norm(gw, #n_lit, #dec_root_lit, #suffix, #eps)?
                }
            }
        }
        FieldLoad::LinearDense(prefix) => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            let kind = shard_kind_for_dotted_prefix(prefix);
            match (sharded, kind, is_vision) {
                (true, ShardKind::ShardDim0, _) => quote! {
                    crate::__gpu::load_layered_linear_dense_sharded(
                        gw, #n_lit, #dec_root_lit, #suffix, 0usize, tp_rank as usize, #tp_world_lit as usize,
                    )?
                },
                (true, ShardKind::ShardDim1, _) => quote! {
                    crate::__gpu::load_layered_linear_dense_sharded(
                        gw, #n_lit, #dec_root_lit, #suffix, 1usize, tp_rank as usize, #tp_world_lit as usize,
                    )?
                },
                (_, _, true) => {
                    let root = vision_root_lit_opt
                        .clone()
                        .expect("is_vision=true requires vision_layered_root");
                    quote! {
                        crate::__gpu::load_layered_linear_dense_vision(gw, #n_lit, #root, #suffix)?
                    }
                }
                (_, _, false) => quote! {
                    crate::__gpu::load_layered_linear_dense(gw, #n_lit, #dec_root_lit, #suffix)?
                },
            }
        }
        FieldLoad::LinearConcat(prefixes) => {
            let suffixes: Vec<&str> = prefixes
                .iter()
                .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                .collect();
            // Always column-parallel — no row-parallel concat exists.
            // Under metal the layered helper routes to the stream-free
            // `_concat_packed` path; cuda keeps the existing stream-
            // based GGUF / vision / sharded variants.
            if cfg!(feature = "metal") {
                // Vision bodies use the per-MODEL vision layered root
                // (Qwen2.5-VL's fused gate_up under `vision_tower.
                // blocks`); the decoder root here would query
                // `model.layers.<L>.<leaf>` and fail the load.
                let root = if is_vision {
                    vision_root_lit_opt
                        .clone()
                        .expect("is_vision=true requires vision_layered_root")
                } else {
                    dec_root_lit.clone()
                };
                quote! {
                    crate::__gpu::load_layered_linear_dense_concat_packed(
                        gw,
                        #n_lit,
                        #root,
                        &[ #(#suffixes),* ],
                    )?
                }
            } else if sharded {
                quote! {
                    crate::__gpu::load_layered_linear_dense_concat_sharded(
                        gw,
                        #n_lit,
                        #dec_root_lit,
                        &[ #(#suffixes),* ],
                        stream,
                        tp_rank as usize,
                        #tp_world_lit as usize,
                    )?
                }
            } else if is_vision {
                // Vision-prelude per-block prefix is `<root>.<L>.`,
                // baked from `vision_safetensors_layout` in the per-
                // arch config (e.g. `visual.blocks` for Qwen,
                // `vision_tower.vision_model.encoder.layers` for
                // Gemma3-MM).
                let root = vision_root_lit_opt
                    .clone()
                    .expect("is_vision=true requires vision_layered_root");
                quote! {
                    crate::__gpu::load_layered_linear_dense_concat_vision(
                        gw,
                        #n_lit,
                        #root,
                        &[ #(#suffixes),* ],
                        stream,
                    )?
                }
            } else {
                quote! {
                    crate::__gpu::load_layered_linear_dense_concat(
                        gw,
                        #n_lit,
                        #dec_root_lit,
                        &[ #(#suffixes),* ],
                        stream,
                    )?
                }
            }
        }
        FieldLoad::LinearTiedToEmbedding { .. } => panic!(
            "LinearTiedToEmbedding is only ever used for the unindexed \
             `lm_head` accessor — should never appear in a layered group"
        ),
        FieldLoad::RawLinear(_) => panic!(
            "RawLinear (nn.Parameter) is global by construction — \
             should never appear in a layered group"
        ),
        FieldLoad::LinearAffine {
            prefix,
            group_size,
            bits,
            in_features,
        } => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            let inf_lit = proc_macro2::Literal::u32_unsuffixed(*in_features);
            // INT4 P3/P4 forward-time path (post-C4b): per-layer
            // AffineQuant LinearLayers kept on device for the qmv /
            // qmm_t dispatchers `MetalAffineQmmImpl` emits. When the group
            // is MLX-native mixed-bit (OptiQ), route to the per-layer
            // loader with the gathered bits slice; both paths thread the
            // manifest in_features so the loader's packed-width guard fires.
            match &per_layer_bits_lit {
                Some(bits_slice) => quote! {
                    crate::__gpu::load_layered_linear_affine_quant_mixed(
                        gw, #n_lit, #quant_root_lit, #suffix, #gs_lit, #bits_slice, #inf_lit,
                    )?
                },
                None => quote! {
                    crate::__gpu::load_layered_linear_affine_quant(
                        gw, #n_lit, #quant_root_lit, #suffix, #gs_lit, #bits_lit, #inf_lit,
                    )?
                },
            }
        }
        FieldLoad::LinearAffineConcat {
            prefixes,
            group_size,
            bits,
            in_features,
        } => {
            let suffixes: Vec<TokenStream> = prefixes
                .iter()
                .map(|p| {
                    let s = layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref);
                    quote! { #s }
                })
                .collect();
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            let inf_lit = proc_macro2::Literal::u32_unsuffixed(*in_features);
            match &per_layer_bits_lit {
                Some(bits_slice) => quote! {
                    crate::__gpu::load_layered_linear_affine_dequant_concat_as_dense_mixed(
                        gw, #n_lit, #quant_root_lit, &[ #(#suffixes),* ], #gs_lit, #bits_slice, #inf_lit,
                    )?
                },
                None => quote! {
                    crate::__gpu::load_layered_linear_affine_dequant_concat_as_dense(
                        gw, #n_lit, #quant_root_lit, &[ #(#suffixes),* ], #gs_lit, #bits_lit, #inf_lit,
                    )?
                },
            }
        }
        FieldLoad::LinearAffineRowConcat {
            prefixes,
            group_size,
            bits,
            in_features,
        } => {
            assert!(
                per_layer_bits_lit.is_none(),
                "a packed-rows linear loads at one width across its layers"
            );
            let suffixes = prefixes.iter().map(|p| {
                let s = layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref);
                quote! { #s }
            });
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let bits_lit = proc_macro2::Literal::u32_unsuffixed(*bits);
            let inf_lit = proc_macro2::Literal::u32_unsuffixed(*in_features);
            quote! {
                crate::__gpu::load_layered_linear_affine_quant_rows_concat(
                    gw, #n_lit, #quant_root_lit, &[ #(#suffixes),* ], #gs_lit, #bits_lit, #inf_lit,
                )?
            }
        }
        FieldLoad::LinearNvfp4 { prefix, group_size } => {
            let suffix = layered_suffix(prefix, vision_zero_prefix_ref, decoder_zero_prefix_ref);
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            quote! {
                crate::__gpu::load_layered_linear_nvfp4_quant(
                    gw, #n_lit, #quant_root_lit, #suffix, #gs_lit,
                )?
            }
        }
        FieldLoad::LinearNvfp4Concat {
            prefixes,
            group_size,
        } => {
            let suffixes: Vec<TokenStream> = prefixes
                .iter()
                .map(|p| {
                    let s = layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref);
                    quote! { #s }
                })
                .collect();
            let gs_lit = proc_macro2::Literal::u32_unsuffixed(*group_size);
            quote! {
                crate::__gpu::load_layered_linear_nvfp4_quant_concat(
                    gw, #n_lit, #quant_root_lit, &[ #(#suffixes),* ], #gs_lit,
                )?
            }
        }
        FieldLoad::MarlinLinear { prefixes, .. } => {
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                quote! {
                    crate::__gpu::load_layered_marlin_linear(
                        gw, #n_lit, #dec_root_lit, #suffix, marlin_storage, __marlin_ws, __device_id,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    crate::__gpu::load_layered_marlin_linear_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        marlin_storage, __marlin_ws, __device_id,
                    )?
                }
            }
        }
        FieldLoad::Fp8Linear { prefixes } => {
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                quote! {
                    crate::__gpu::load_layered_fp8_linear(
                        gw, #n_lit, #dec_root_lit, #suffix, __fp8_dtype,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    crate::__gpu::load_layered_fp8_linear_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        __fp8_dtype,
                    )?
                }
            }
        }
        FieldLoad::Fp8BlockLinear { prefixes } => {
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                quote! {
                    crate::__gpu::load_layered_fp8_block_linear(
                        gw, #n_lit, #dec_root_lit, #suffix, __fp8_dtype,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    crate::__gpu::load_layered_fp8_block_linear_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        __fp8_dtype,
                    )?
                }
            }
        }
        FieldLoad::Bnb4Linear {
            prefixes,
            out_features_per_shard,
            in_features,
            blocksize,
        } => {
            let in_features = *in_features as usize;
            let blocksize = *blocksize as usize;
            let outs: Vec<proc_macro2::Literal> = out_features_per_shard
                .iter()
                .map(|o| proc_macro2::Literal::usize_unsuffixed(*o as usize))
                .collect();
            if prefixes.len() == 1 {
                let suffix = layered_suffix(
                    &prefixes[0],
                    vision_zero_prefix_ref,
                    decoder_zero_prefix_ref,
                );
                let out = &outs[0];
                quote! {
                    crate::__gpu::load_layered_bnb4(
                        gw, #n_lit, #dec_root_lit, #suffix,
                        __bnb_code, __bnb_scratch,
                        #out, #in_features, #blocksize,
                    )?
                }
            } else {
                let suffixes: Vec<&str> = prefixes
                    .iter()
                    .map(|p| layered_suffix(p, vision_zero_prefix_ref, decoder_zero_prefix_ref))
                    .collect();
                quote! {
                    crate::__gpu::load_layered_bnb4_concat(
                        gw, #n_lit, #dec_root_lit,
                        &[ #(#suffixes),* ],
                        __bnb_code, __bnb_scratch,
                        &[ #(#outs),* ],
                        #in_features, #blocksize,
                    )?
                }
            }
        }
        FieldLoad::DeepSeekV2Moe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::DeepSeekV2MoELayer::load(
                            gw,
                            &#p,
                            #n_routed_experts,
                            #n_shared_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #norm_topk_prob,
                            #routed_scaling_factor,
                            #use_sigmoid,
                            #n_expert_group,
                            #topk_group,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::GatedDeltaNet(prefix) => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers::GatedDeltaNetLayer::load(gw, &#p)
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::RawTensor(prefix) => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        gw.take(&#p)
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::DeepSeekV2Fp8BlockMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::DeepSeekV2Fp8BlockMoELayer::load(
                            gw,
                            &#p,
                            #n_routed_experts,
                            #n_shared_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #norm_topk_prob,
                            #routed_scaling_factor,
                            #use_sigmoid,
                            #n_expert_group,
                            #topk_group,
                            __fp8_dtype,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::DeepSeekV2GgmlMoe {
            prefix,
            n_routed_experts,
            n_shared_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            norm_topk_prob,
            routed_scaling_factor,
            use_sigmoid,
            n_expert_group,
            topk_group,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let n_routed_experts = *n_routed_experts;
            let n_shared_experts = *n_shared_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let norm_topk_prob = *norm_topk_prob;
            let routed_scaling_factor = *routed_scaling_factor;
            let use_sigmoid = *use_sigmoid;
            let n_expert_group = *n_expert_group;
            let topk_group = *topk_group;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::DeepSeekV2GgmlMoELayer::load_gguf(
                            gw,
                            &#p,
                            #n_routed_experts,
                            #n_shared_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #norm_topk_prob,
                            #routed_scaling_factor,
                            #use_sigmoid,
                            #n_expert_group,
                            #topk_group,
                            stream,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::FusedMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let top_k = *top_k;
            let intermediate_size = *intermediate_size;
            let hidden_size = *hidden_size;
            if let Some((group_size, bits)) = *affine {
                let gate_bits = gate_bits.unwrap_or(bits.up());
                // Per-layer routed-expert bits for MLX mixed/dynamic quant
                // (OptiQ); single width otherwise.
                let bits_expr: TokenStream = match &per_layer_bits_lit {
                    Some(bits_slice) => quote! { (#bits_slice)[layer as usize] },
                    None => bits.tokens(),
                };
                quote! {
                    (0u32..#n_lit)
                        .map(|layer: u32| -> ::anyhow::Result<_> {
                            crate::__gpu::layers_moe::FusedMoELayer::load_affine(
                                gw,
                                &#p,
                                #num_experts,
                                #top_k,
                                #intermediate_size,
                                #hidden_size,
                                #group_size,
                                #bits_expr,
                                #gate_bits,
                            )
                        })
                        .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
                }
            } else {
                quote! {
                    (0u32..#n_lit)
                        .map(|layer: u32| -> ::anyhow::Result<_> {
                            crate::__gpu::layers_moe::FusedMoELayer::load(
                                gw,
                                &#p,
                                #num_experts,
                                #top_k,
                                #intermediate_size,
                                #hidden_size,
                                stream,
                            )
                        })
                        .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
                }
            }
        }
        FieldLoad::SharedFusedMoe {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let shared_expert_intermediate_size = *shared_expert_intermediate_size;
            let hidden_size = *hidden_size;
            if let Some((group_size, bits)) = *affine {
                let gate_bits = gate_bits.unwrap_or(bits.up());
                let bits_expr: TokenStream = match &per_layer_bits_lit {
                    Some(bits_slice) => quote! { (#bits_slice)[layer as usize] },
                    None => bits.tokens(),
                };
                quote! {
                    (0u32..#n_lit)
                        .map(|layer: u32| -> ::anyhow::Result<_> {
                            crate::__gpu::layers_moe::SharedFusedMoELayer::load_affine(
                                gw,
                                &#p,
                                #num_experts,
                                #top_k,
                                #moe_intermediate_size,
                                #shared_expert_intermediate_size,
                                #hidden_size,
                                #group_size,
                                #bits_expr,
                                #gate_bits,
                            )
                        })
                        .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
                }
            } else {
                quote! {
                    (0u32..#n_lit)
                        .map(|layer: u32| -> ::anyhow::Result<_> {
                            crate::__gpu::layers_moe::SharedFusedMoELayer::load(
                                gw,
                                &#p,
                                #num_experts,
                                #top_k,
                                #moe_intermediate_size,
                                #shared_expert_intermediate_size,
                                #hidden_size,
                                stream,
                            )
                        })
                        .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
                }
            }
        }
        FieldLoad::GemmaRouter {
            prefix,
            num_experts,
            hidden_size,
            affine,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let hidden_size = *hidden_size;
            let affine = gemma_router_affine_ts(*affine);
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::GemmaRouterLayer::load(
                            gw,
                            &#p,
                            #num_experts,
                            #hidden_size,
                            #affine,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::GemmaSwitchGlu {
            prefix,
            num_experts,
            top_k,
            moe_intermediate_size,
            hidden_size,
            group_size,
            bits,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let top_k = *top_k;
            let moe_intermediate_size = *moe_intermediate_size;
            let hidden_size = *hidden_size;
            let group_size = *group_size;
            let bits = *bits;
            // MLX-native mixed/dynamic quant (OptiQ): the experts ship at
            // 8-bit on the sensitive edge layers and 4-bit in the middle.
            // Index a per-layer bits slice by the loop's `layer` when the
            // group is heterogeneous; otherwise the single width.
            let bits_expr: TokenStream = match &per_layer_bits_lit {
                Some(bits_slice) => quote! { (#bits_slice)[layer as usize] },
                None => quote! { #bits },
            };
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::SwitchGluExpertsLayer::load(
                            gw,
                            &#p,
                            #num_experts,
                            #top_k,
                            #moe_intermediate_size,
                            #hidden_size,
                            #group_size,
                            #bits_expr,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
        FieldLoad::GptOssMoe {
            prefix,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            affine,
            gate_bits,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let top_k = *top_k;
            let intermediate_size = *intermediate_size;
            let hidden_size = *hidden_size;
            // Op-keyed arm — the Dense/verbatim config's storage params
            // default rather than panic (the unindexed arm's tolerance).
            let (group_size, bits) = affine.unwrap_or((64, 4));
            let gate_bits = gate_bits.unwrap_or(bits);
            // Per-layer expert bits (OptiQ) — same treatment as
            // GemmaSwitchGlu: index the slice when heterogeneous, else
            // the single width. `gate_bits` is uniform across layers
            // (the collapsed layer-0 plan supplies it).
            let bits_expr: TokenStream = match &per_layer_bits_lit {
                Some(bits_slice) => quote! { (#bits_slice)[layer as usize] },
                None => quote! { #bits },
            };
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::GptOssMoELayer::load(
                            gw,
                            &#p,
                            #num_experts,
                            #top_k,
                            #intermediate_size,
                            #hidden_size,
                            #group_size,
                            #bits_expr,
                            #gate_bits,
                        )
                    })
                    .collect::<::anyhow::Result<::std::vec::Vec<_>>>()?
            }
        }
    }
}

/// Emit one accessor method per base name on the per-arch `Weights`
/// struct. Layered bases (`input_layernorm_0`, `input_layernorm_1`,
/// …) — backed by a single `Vec<T>` field — produce
/// `pub fn input_layernorm(&self, layer: u32) -> &RmsNorm { &self.input_layernorm[layer as usize] }`.
/// Non-layered bases get the same signature for caller uniformity;
/// the body returns `&self.<base>` and ignores the layer arg.
///
/// Compresses what used to be a 40-arm `match layer` per accessor
/// into a single slice index — the 578-line `impl Weights { … }`
/// block on commandr collapses to ~50 lines, and llama's 28k-line
/// equivalent collapses by the same proportion.
///
/// Returns an `impl Weights { ... }` block. Empty (zero accessors)
/// is fine — the impl block is then elided entirely.
fn emit_weights_accessor_methods(accessors: &[WeightAccessor], dims: &ModelDims) -> TokenStream {
    let groups = group_accessors_by_base(accessors);
    if groups.is_empty() {
        return TokenStream::new();
    }
    // Per-method `#[cfg]` / `#[inline]` / `#[allow(dead_code)]` are
    // redundant — the impl block carries the cfg, the methods are
    // trivial enough that LLVM inlines them in release without the
    // hint, and the unused-warning is suppressed at the impl level.
    // Dropping per-method attrs collapses each accessor from ~7
    // lines to ~3.
    let methods: Vec<TokenStream> = groups
        .iter()
        .map(|g| {
            let base_ident = syn::Ident::new(&g.base, proc_macro2::Span::call_site());
            let ty = dims.handle(&g.base).map_or(&g.rust_type, |h| &h.ty);
            match g.kind {
                AccessorGroupKind::LayeredContiguous => quote! {
                    pub fn #base_ident(&self, layer: u32) -> &#ty {
                        unsafe { self.#base_ident.get_unchecked(layer as usize) }
                    }
                },
                AccessorGroupKind::Unindexed => quote! {
                    pub fn #base_ident(&self, _: u32) -> &#ty { &self.#base_ident }
                },
                AccessorGroupKind::LayeredSparse => {
                    // Per-layer fields → match arms. Codegen-issued
                    // static rows only ever pass layers that exist
                    // (guaranteed by fan_out's per-claim layer
                    // plumbing), so `unreachable_unchecked()` on non-
                    // emitted layers is safe.
                    let arms: Vec<TokenStream> = g
                        .entries
                        .iter()
                        .map(|(layer, acc)| {
                            let layer_lit = proc_macro2::Literal::u32_unsuffixed(
                                layer.expect("LayeredSparse entry has Some(layer)").0 as u32,
                            );
                            let fname = &acc.name;
                            quote! { #layer_lit => &self.#fname, }
                        })
                        .collect();
                    quote! {
                        pub fn #base_ident(&self, layer: u32) -> &#ty {
                            match layer {
                                #(#arms)*
                                _ => unsafe { ::core::hint::unreachable_unchecked() },
                            }
                        }
                    }
                }
            }
        })
        .collect();
    quote! {
        #[allow(dead_code)]
        impl Weights {
            #(#methods)*
        }
    }
}

/// The accessor RETURN TYPE for a weight kind — the second half of
/// the one accessor-ABI fact, beside
/// [`weight_kind_accessor_method`]. Both the `WeightAccessors` trait
/// emission and the tape-derived accessor set read this table, so a
/// new kind is a single-site change (adding a variant without a row
/// here is a non-exhaustive-match build error).
pub(crate) fn weight_kind_rust_type(kind: &crate::weight_vocab::WeightKind) -> TokenStream {
    // The mapping itself lives on `WeightKind` (`scratchy_subtile::handoff`) so metal's
    // compiler reads the SAME table when it compares accessor group signatures. Here it only
    // becomes tokens.
    kind.rust_type_name()
        .parse()
        .expect("weight kind rust type is a valid type path")
}

/// The `WeightAccessors` trait method a [`WeightKind`] resolves through — the key of
/// [`emit_weight_accessors_impl`]'s per-(op_idx, kind) slot-ordinal walk.
pub(crate) fn weight_kind_accessor_method(kind: &crate::weight_vocab::WeightKind) -> &'static str {
    use crate::weight_vocab::WeightKind;
    match kind {
        WeightKind::RmsNorm => "rms_norm_at",
        WeightKind::Embedding => "embedding_at",
        WeightKind::Linear => "linear_at",
        WeightKind::LayerNorm => "layer_norm_at",
        WeightKind::Marlin => "marlin_at",
        WeightKind::Bnb4 => "bnb4_at",
        WeightKind::Fp8 => "fp8_at",
        WeightKind::DeepSeekMoe => "deepseek_moe_at",
        WeightKind::DeepSeekMoeFp8 => "deepseek_moe_fp8_at",
        WeightKind::DeepSeekMoeGgml => "deepseek_moe_ggml_at",
        WeightKind::GatedDeltaNet => "gated_delta_net_at",
        WeightKind::FusedMoe => "fused_moe_at",
        WeightKind::SharedFusedMoe => "shared_fused_moe_at",
        WeightKind::GemmaRouter => "gemma_router_at",
        WeightKind::GemmaSwitchGlu => "gemma_switch_glu_at",
        WeightKind::GptOssMoe => "gpt_oss_moe_at",
        WeightKind::CosSin => "cos_sin_at",
        WeightKind::AffineQuantEmbedding => "affine_quant_embedding_at",
        WeightKind::AttnSinks => "sinks_at",
        WeightKind::Tensor => "tensor_at",
    }
}

/// Emit `impl ::scratchy_forward_compiler::WeightAccessors for Weights { ... }`
/// keyed on `(tape_index, op_idx)` per the typed-fanout design.
///
/// The variant of an `Instruction` at position `op_idx` in `tape_index`
/// fixes the *kind* of weight needed; the per-arch impl resolves
/// `(tape_index, op_idx)` → which named field on `Weights`. We walk
/// every tape_index × position, look up the recorded `weight_slots`,
/// and emit one match arm per (tape_index, op_idx, kind) triple.
///
/// Sprint 1 scope: `rms_norm_at` only. Sibling methods (`linear_at`,
/// `embedding_at`, `cos_sin_at`, MoE getters) land per sprint as
/// the matching `Instruction<W>` variants migrate off
/// `WtFn`/`CosSinFn`.
fn emit_weight_accessors_impl(
    canonical_lowered: &BTreeMap<WorkloadPoint, (CanonicalLowered, u32, u32, u32, SlotMap)>,
    variant_name: &str,
) -> TokenStream {
    use crate::weight_vocab::WeightKind;

    // Walk every BUCKET × position in declaration order. The tape_index
    // index in FORWARD_TABLE matches the (m, sk) row order we emit;
    // we use a synthetic "tape_index id" of `2 * row_idx + slice_idx`
    // where `slice_idx` is 0 for backbone, 1 for lm_head — matching
    // `BUCKET_BACKBONE` / `BUCKET_LM_HEAD` for the simple
    // single-tape_index case.
    //
    // For now (Sprint 1 + simple model lowering): every tape_index row
    // shares the same canonical's instruction stream, so the match
    // table only needs entries for `(BUCKET_BACKBONE, op_idx)` and
    // `(BUCKET_LM_HEAD, op_idx)` from any one tape_index row. We use
    // the first canonical entry's lowered slices as the source.
    //
    // When per-tape_index variation lands (different Impls per workload
    // point), this becomes per-tape_index-row and the tape_index encoding
    // expands; the runtime caller of `rms_norm_at` will pass the
    // matching encoded id from the FORWARD_TABLE row it dispatched
    // through.
    // For each kind, collect `(tape_index, op_idx, slot) => self.<base>(layer)`
    // match arms across every canonical's backbone and lm_head. `slot` is
    // the per-(tape_index, op_idx, kind) ordinal, derived by walking
    // `weight_slots` in declaration order and counting prior occurrences
    // of the same kind at the same op.
    use std::collections::HashMap;
    let mut by_kind: HashMap<&'static str, Vec<TokenStream>> = HashMap::new();
    // Parallel inventory used only to build the static diagnostic
    // string baked into each `*_at` body — never compiled into runtime
    // matching. `(tape_index, role, op_idx, slot, method, base)`.
    let mut slot_inventory: Vec<(u32, &'static str, u32, u32, &'static str, String)> = Vec::new();

    // Walk EVERY canonical lowered entry. Each distinct tape_index
    // (workload point) gets a pair of tape_index ids: 2*ci (backbone)
    // and 2*ci+1 (lm_head). FORWARD_TABLE rows pass these ids into
    // run/run_backbone, which forward them to run_slice → eval →
    // the per-arch WeightAccessors match arms.
    let emit_for =
        |tape_index: u32,
         role: &'static str,
         weight_slots: &[Vec<WeightSlot>],
         by_kind: &mut HashMap<&'static str, Vec<TokenStream>>,
         inventory: &mut Vec<(u32, &'static str, u32, u32, &'static str, String)>| {
            let tape_index_lit = proc_macro2::Literal::u32_unsuffixed(tape_index);
            for (op_idx, slots) in weight_slots.iter().enumerate() {
                let op_lit = proc_macro2::Literal::u32_unsuffixed(op_idx as u32);
                let kinds: Vec<WeightKind> = slots.iter().map(|s| s.kind.clone()).collect();
                for (i, slot) in slots.iter().enumerate() {
                    let key = weight_kind_accessor_method(&slot.kind);
                    let n = scratchy_subtile::handoff::accessor_slot(&kinds, i);
                    let slot_lit = proc_macro2::Literal::u32_unsuffixed(n);
                    inventory.push((
                        tape_index,
                        role,
                        op_idx as u32,
                        n,
                        key,
                        slot.base.to_string(),
                    ));
                    // `WeightSlot::base` is a `String` (it crosses into the metal compiler
                    // crate, which must not pull `syn`); the identifier is minted HERE, at the
                    // one place that emits tokens for it.
                    let base = quote::format_ident!("{}", slot.base);
                    // CosSin pulls from a `RotaryCache` field on the per-arch
                    // `Weights` struct (`wm.rotary` or `wm.rotary_local`),
                    // not from a `fn <base>(layer) -> &T` getter — rotary is
                    // shared across layers, and the cache itself owns a
                    // single `cos_sin_cache: GpuTensor`. `GpuTensor` is `Copy`,
                    // so reading the field copies the handle directly — no
                    // `.clone()` needed (it would just be a redundant copy).
                    let arm_body = match slot.kind {
                        WeightKind::CosSin => quote! {
                            self.#base.get().expect("RoPE tables are built at load, or with the metal pool (`build_rope_tables`)").cos_sin_cache
                        },
                        // The layered sinks accessor hands back a
                        // borrow; the trait hands the `Copy` handle
                        // by value.
                        WeightKind::AttnSinks => quote! { *self.#base(layer) },
                        _ => quote! { self.#base(layer) },
                    };
                    by_kind.entry(key).or_default().push(quote! {
                        (#tape_index_lit, #op_lit, #slot_lit) => #arm_body,
                    });
                }
            }
        };
    for (ci, (_wp, (cl, _, _, _, _))) in canonical_lowered.iter().enumerate() {
        let bb_id = (ci as u32) * 2;
        let lm_id = bb_id + 1;
        emit_for(
            bb_id,
            "backbone",
            &cl.backbone.weight_slots,
            &mut by_kind,
            &mut slot_inventory,
        );
        emit_for(
            lm_id,
            "lm_head",
            &cl.lm_head.weight_slots,
            &mut by_kind,
            &mut slot_inventory,
        );
    }
    // Build the per-method diagnostic summary baked into each `*_at`
    // panic body. Sort by (method, tape_index, op_idx, slot) so the
    // output reads top-to-bottom in the order the runtime resolves
    // bindings.
    let mut sorted_inv = slot_inventory.clone();
    sorted_inv.sort_by(|a, b| {
        a.4.cmp(b.4) // method
            .then(a.0.cmp(&b.0)) // tape_index
            .then(a.2.cmp(&b.2)) // op_idx
            .then(a.3.cmp(&b.3)) // slot
    });
    let method_summaries: HashMap<&'static str, String> = {
        #[allow(clippy::type_complexity)]
        let mut by_method: HashMap<
            &'static str,
            Vec<(u32, &'static str, u32, u32, String)>,
        > = HashMap::new();
        for (tape, role, op, slot, method, base) in &sorted_inv {
            by_method
                .entry(method)
                .or_default()
                .push((*tape, role, *op, *slot, base.clone()));
        }
        by_method
            .into_iter()
            .map(|(method, rows)| {
                let mut s = String::new();
                let mut cur_tape: Option<(u32, &'static str)> = None;
                for (tape, role, op, slot, base) in rows {
                    if cur_tape != Some((tape, role)) {
                        if cur_tape.is_some() {
                            s.push('\n');
                        }
                        s.push_str(&format!("    tape_index={tape} ({role}):"));
                        cur_tape = Some((tape, role));
                    }
                    s.push_str(&format!(" (op={op},slot={slot})→{base}"));
                }
                (method, s)
            })
            .collect()
    };
    // All-methods inventory for cross-kind context at the same op_idx.
    // Sorted by (tape_index, op_idx, method, slot).
    let mut all_sorted = slot_inventory.clone();
    all_sorted.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.2.cmp(&b.2))
            .then(a.4.cmp(b.4))
            .then(a.3.cmp(&b.3))
    });
    let all_summary: String = {
        let mut s = String::new();
        let mut cur_tape: Option<(u32, &'static str)> = None;
        let mut cur_op: Option<u32> = None;
        for (tape, role, op, slot, method, base) in &all_sorted {
            if cur_tape != Some((*tape, role)) {
                if cur_tape.is_some() {
                    s.push('\n');
                }
                s.push_str(&format!("  tape_index={tape} ({role}):\n"));
                cur_tape = Some((*tape, role));
                cur_op = None;
            }
            if cur_op != Some(*op) {
                if cur_op.is_some() {
                    s.push('\n');
                }
                s.push_str(&format!("    op_idx={op}:"));
                cur_op = Some(*op);
            }
            s.push_str(&format!(" {method}[slot={slot}]→{base}"));
        }
        s
    };

    let variant_lit = proc_macro2::Literal::string(variant_name);
    let all_summary_lit = proc_macro2::Literal::string(&all_summary);
    let method_emit = |method: &str, ret_ty: TokenStream| -> TokenStream {
        let method_id = syn::Ident::new(method, proc_macro2::Span::call_site());
        let arms = by_kind.get(method).cloned().unwrap_or_default();
        if arms.is_empty() {
            // Default trait body already returns `unreachable!()`. No
            // override needed when the arch never consumes this kind.
            return quote! {};
        }
        let method_summary = method_summaries
            .get(method)
            .cloned()
            .unwrap_or_else(|| "    (none recorded for this method)".to_string());
        let method_summary_lit = proc_macro2::Literal::string(&method_summary);
        quote! {
            fn #method_id(
                &self,
                tape_index: u32,
                op_idx: u32,
                slot: u32,
                layer: u32,
            ) -> #ret_ty {
                // `layer` is unused for `cos_sin_at` (rotary is layer-
                // independent); referenced explicitly here so the
                // generated body type-checks identically across every
                // method.
                let _ = layer;
                match (tape_index, op_idx, slot) {
                    #(#arms)*
                    _ => panic!(
                        "\nscratchy-forward-compiler weight-accessor coverage gap\n\
                         \n\
                         variant:  `{}::Weights`\n\
                         method:   `{}`\n\
                         asked:    (tape_index={}, op_idx={}, slot={}, layer={})\n\
                         \n\
                         This variant's recorded `{}` slots:\n\
                         {}\n\
                         \n\
                         Every weight-accessor slot recorded for this variant \
                         (any kind, at tape_index={}):\n\
                         {}\n\
                         \n\
                         A 'no match' here means worker code resolved a \
                         `Binding::Weight` whose `(tape_index, op_idx, slot)` \
                         locator does not appear in the macro-emitted slot \
                         table above. Two common causes:\n\
                         \n\
                         1. WRONG VARIANT WAS SELECTED at `scratchy_forward_compiler::try_load`.\n\
                            The compiled-variant fingerprint matched but the live\n\
                            checkpoint really wants a sibling (quantized) variant.\n\
                            Re-check `emit_fingerprint_check` in \
                            `crates/scratchy-forward-compiler-macro/src/codegen.rs`.\n\
                         2. IMPL `required_weights` UNDER-SUPPLIED its emits.\n\
                            The Impl that lowered op_idx={} declared fewer accessors\n\
                            than its emitted Instruction consumes at runtime.\n\
                            (Macro now panics at expansion when this is locally\n\
                            detectable — see the slot-distribution assertion in\n\
                            `interpreter_codegen::lower_bucket`.)\n",
                        #variant_lit, stringify!(#method_id),
                        tape_index, op_idx, slot, layer,
                        stringify!(#method_id), #method_summary_lit,
                        tape_index, #all_summary_lit,
                        op_idx,
                    ),
                }
            }
        }
    };

    // ONE accessor-ABI fact per kind: name and return type both come
    // from the tables, so the trait emission and the tape-derived
    // accessor set cannot drift.
    let method_emit_kind = |kind: &crate::weight_vocab::WeightKind| {
        method_emit(
            weight_kind_accessor_method(kind),
            weight_kind_rust_type(kind),
        )
    };
    let rms_norm = method_emit_kind(&crate::weight_vocab::WeightKind::RmsNorm);
    let embedding = method_emit_kind(&crate::weight_vocab::WeightKind::Embedding);
    let linear = method_emit_kind(&crate::weight_vocab::WeightKind::Linear);
    let layer_norm = method_emit_kind(&crate::weight_vocab::WeightKind::LayerNorm);
    let marlin = method_emit_kind(&crate::weight_vocab::WeightKind::Marlin);
    let bnb4 = method_emit_kind(&crate::weight_vocab::WeightKind::Bnb4);
    let fp8 = method_emit_kind(&crate::weight_vocab::WeightKind::Fp8);
    let dsmoe = method_emit_kind(&crate::weight_vocab::WeightKind::DeepSeekMoe);
    let dsmoe_fp8 = method_emit(
        "deepseek_moe_fp8_at",
        quote! { &crate::__gpu::layers_moe::DeepSeekV2Fp8BlockMoELayer },
    );
    let dsmoe_ggml = method_emit(
        "deepseek_moe_ggml_at",
        quote! { &crate::__gpu::layers_moe::DeepSeekV2GgmlMoELayer },
    );
    let gated_delta_net = method_emit_kind(&crate::weight_vocab::WeightKind::GatedDeltaNet);
    let fused_moe = method_emit_kind(&crate::weight_vocab::WeightKind::FusedMoe);
    let shared_moe = method_emit_kind(&crate::weight_vocab::WeightKind::SharedFusedMoe);
    let gemma_router = method_emit_kind(&crate::weight_vocab::WeightKind::GemmaRouter);
    let gemma_switch_glu = method_emit_kind(&crate::weight_vocab::WeightKind::GemmaSwitchGlu);
    let cos_sin = method_emit_kind(&crate::weight_vocab::WeightKind::CosSin);
    let attn_sinks = method_emit_kind(&crate::weight_vocab::WeightKind::AttnSinks);
    // `affine_quant_embedding_at` is gated `#[cfg(feature = "metal")]`
    // on the trait so we must emit the cfg attribute together with the
    // method body, or skip both when the arch never resolves an
    // `AffineQuantEmbedding` — otherwise a bare `#[cfg]` with no item
    // would surface as "expected item after attributes".
    let aqe_arms = by_kind
        .get("affine_quant_embedding_at")
        .cloned()
        .unwrap_or_default();
    let affine_quant_embedding = if aqe_arms.is_empty() {
        quote! {}
    } else {
        let aqe_summary = method_summaries
            .get("affine_quant_embedding_at")
            .cloned()
            .unwrap_or_else(|| "    (none recorded for this method)".to_string());
        let aqe_summary_lit = proc_macro2::Literal::string(&aqe_summary);
        quote! {
            #[cfg(scratchy_target = "metal")]
            fn affine_quant_embedding_at(
                &self,
                tape_index: u32,
                op_idx: u32,
                slot: u32,
                layer: u32,
            ) -> &crate::__gpu::layers::AffineQuantEmbedding {
                match (tape_index, op_idx, slot) {
                    #(#aqe_arms)*
                    _ => panic!(
                        "\nscratchy-forward-compiler weight-accessor coverage gap\n\
                         \n\
                         variant:  `{}::Weights`\n\
                         method:   `affine_quant_embedding_at`\n\
                         asked:    (tape_index={}, op_idx={}, slot={}, layer={})\n\
                         \n\
                         Recorded `affine_quant_embedding_at` slots:\n\
                         {}\n\
                         \n\
                         All slots at tape_index={}:\n\
                         {}\n",
                        #variant_lit,
                        tape_index, op_idx, slot, layer,
                        #aqe_summary_lit,
                        tape_index, #all_summary_lit,
                    ),
                }
            }
        }
    };

    // Emit the impl unconditionally — `CanonicalParams: WeightAccessors`
    // is an unconditional bound, so leaving this behind a feature gate
    // would surface as `Weights: WeightAccessors not satisfied` in any
    // arch whose own `metal` feature is empty
    // or absent (e.g. phi3 which declares `metal = []`).
    // The trait methods all return cross-backend types from
    // `scratchy_layers::layers{,_moe}` plus `GpuTensor` from
    // `scratchy-tensors` — both available regardless of backend feature.
    quote! {
        impl ::scratchy_forward_compiler::WeightAccessors for Weights {
            #rms_norm
            #embedding
            #linear
            #layer_norm
            #marlin
            #bnb4
            #fp8
            #dsmoe
            #dsmoe_fp8
            #dsmoe_ggml
            #gated_delta_net
            #fused_moe
            #shared_moe
            #gemma_router
            #gemma_switch_glu
            #cos_sin
            #attn_sinks
            #affine_quant_embedding
        }
    }
}

// ── Forward fn emission ──────────────────────────────────────────
//
// The macro lowers each canonical tape_index's solved FUF into a flat
// instruction list (a `&[Op]` static slice), emits one per-arch
// `Op` enum + one per-arch `__interpret` helper, then emits one
// thin per-tape_index fn per (forward, backbone) × workload-point that:
// 1. allocates the runtime tile table,
// 2. runs the alias prelude (zero-copy `View` aliases the lowering
//    surfaced via `output_alias`),
// 3. calls `__interpret(&FORWARD_M_<N>, &mut __tiles, …)`,
// 4. takes ownership of the slot the lowering tagged as final.
//
// Buckets that fold onto a canonical share the canonical's body via
// the bucket table.

/// Ident for a per-workload-tape_index `static <PREFIX>_<m>: &[Op]`.
/// Mirrors [`bucket_fn_ident`] in shape but uses upper-case so the
/// emitted module reads naturally — `FORWARD_M_64` /
/// `BACKBONE_M_64_SK_2048`.
fn bucket_static_ident(prefix: &str, wp: WorkloadPoint) -> proc_macro2::Ident {
    if wp.sk_bucket == 0 {
        format_ident!("{}_{}", prefix, wp.num_tokens)
    } else {
        format_ident!("{}_{}_SK_{}", prefix, wp.num_tokens, wp.sk_bucket)
    }
}

/// Per-canonical-tape_index lowering products. The `backbone` slice
/// carries everything the forward pass needs except the terminal
/// `lm_head` gemm; the `lm_head` slice carries that single row.
/// `forward` runs both, `forward_backbone` runs only the backbone
/// and DtoD-copies the backbone-output slot.
struct CanonicalLowered {
    backbone: crate::interpreter_codegen::LoweredBucket,
    lm_head: crate::interpreter_codegen::LoweredBucket,
}

/// Walk the FUF from the end, skipping nodes appended by lowering
/// passes that aren't the body's terminal output:
///
/// - `OpKind::MmEmbedSplice` — `tp_lowering::insert_mm_splices`
///   pushes one per image-bearing batch; semantically adjacent to
///   the Embed, lives at array tail for `push`-based insertion.
/// - `OpKind::LoadRows` — `rows_lowering::materialize_rows`
///   pushes one per row extern the body reads (`pixels`, `pos_embeds`,
///   `target_hidden`) to materialize it as a tile; semantically the FIRST
///   ops (everything reads from them), but they live at array tail for
///   the same `push`-based reason. Mistaking one for the terminal would
///   return its tile instead of the body's output.
///
/// The "last node" for backbone-output / terminal-subgraph
/// identification must be the body's actual terminal — the lm_head
/// Gemm (tp=1) or its AllGather wrapper (tp>1) for decoders, the
/// merger MLP's bias_add for vision encoders.
fn last_non_splice_node(fuf: &Fuf) -> Option<&crate::fuf::FufNode> {
    fuf.nodes.iter().rev().find(|n| match n.op {
        #[cfg(feature = "metal")]
        crate::classified::OpKind::MmEmbedSplice => false,
        crate::classified::OpKind::LoadRows(_) => false,
        _ => true,
    })
}

/// How the FUF's terminal node maps onto the backbone/lm_head split.
///
/// - [`BackboneLayout::Decoder`] — the FUF ends in `gemm(<tile>,
///   lm_head)` (or that gemm followed by an `AllGather` at tp>1).
///   The backbone is everything except the lm_head Gemm, whose
///   hidden-state input is what `forward_backbone` returns.
/// - [`BackboneLayout::Encoder`] — the FUF's terminal is NOT
///   `gemm(_, lm_head)`. Covers text-side encoders (ModernBERT)
///   AND every `#[vision_forward]` body (Qwen2-VL / Qwen2.5-VL /
///   SigLIP / …) since vision bodies don't have a trainable
///   lm_head. The whole pipeline is the backbone, the lm_head slice
///   is empty, and `forward` returns the FUF's last-node output
///   directly.
#[derive(Clone, Copy, Debug)]
enum BackboneLayout {
    Decoder,
    Encoder,
}

/// Classify the FUF's terminal as decoder vs encoder. A decoder
/// terminal is `gemm(<tile>, <lm_head_weight>)`, optionally followed
/// by an `AllGather` (inserted by tp>1 lowering on the vocab-parallel
/// lm_head Gemm). Walks past trailing `MmEmbedSplice` / `LoadRows`
/// nodes via [`last_non_splice_node`] — those are
/// appended by lowering passes but aren't the body's actual terminal.
/// The model's `METAL_OFF_TAPE` static: its argmax, grammar mask and sampler kernels baked for its
/// logits `width` and `dtype`, and — `head`, a multi-token-prediction head's drafts a step and the
/// width of the hidden rows its passes read — its chain between passes
/// (`scratchy_target_metal::off_tape`).
#[cfg(feature = "metal")]
fn metal_off_tape_tokens(
    width: u64,
    mc: &scratchy_ir::model_facts::ModelFacts,
    head: Option<(u32, u32)>,
    vocab_suffix: Option<scratchy_target_metal::off_tape::VocabSuffix>,
) -> TokenStream {
    use scratchy_target_metal::off_tape::{
        ByCodes, Dflash2Role, OffTape, SuffixKernels, TargetSampling, VocabSuffix,
    };
    use scratchy_target_metal::tape::ids::{BlockSize, HiddenSize, LogitsWidth, NumDrafts, RotDim};
    use scratchy_target_metal::tape::kernel_constants::{DraftAcceptConstants, MtpChainConstants};
    let width = u32::try_from(width).expect("logits width fits u32");
    use scratchy_ir::model_facts::DEFAULT_KV_PAGE;
    // The chain advances the full-context group's slots (KV group 0), at the global class's page;
    // a model without attention has no page of its own, and takes the default one.
    let page = mc
        .attention()
        .map_or(DEFAULT_KV_PAGE, |a| a.global().page());
    let block_size = BlockSize(page.get());
    // A draft block: its anchor row and its drafts.
    let block_rows = 1 + mc.spec_drafts().map_or(0, |d| d.get());
    let head = head.map(|(drafts, hidden)| MtpChainConstants {
        hidden: HiddenSize(hidden),
        drafts: NumDrafts(drafts),
        // An MRoPE head reads a cos/sin row a token in place of its position, its rot window
        // wide. The kernel's ABI spells the 1-D positions' path as 0.
        rope: RotDim(
            (mc.attention())
                .filter(|a| a.mrope().is_some())
                .map_or(0, |a| a.base().rot_dim().get()),
        ),
    });
    let keys = OffTape::keys(
        LogitsWidth(width),
        mc.metal_dtype(),
        block_size,
        head,
        mc.spec_drafts().is_some(),
        vocab_suffix,
        // A DFlash2 drafter (its block and the token its proposal rows embed), or a target one
        // drafts for (the rows a lane of its verify holds: the drafter's block).
        match (mc.drafter(), mc.capture_slots()) {
            (Some(drafter), _) => Dflash2Role::Drafter(DraftAcceptConstants {
                rows: block_rows,
                mask: drafter.mask_token().get(),
            }),
            (None, Some(_)) => Dflash2Role::Target { block_rows },
            (None, None) => Dflash2Role::None,
        },
    );
    let ids = keys.map(|key| scratchy_target_metal_compiler::static_tape::kernel_ref(&key));
    let OffTape {
        vocab,
        argmax,
        argmax_dual_write,
        logprobs,
        grammar_mask,
        sampler,
        chain_advance,
        mtp_chain,
        select_rows,
        vocab_suffix,
        dflash2_accept,
        dflash2_sampling,
    } = ids;
    let dflash2_sampling = dflash2_sampling.map(|TargetSampling { mass, search, draw }| {
        quote!(::scratchy_target_metal::off_tape::TargetSampling {
            mass: #mass,
            search: #search,
            draw: #draw,
        })
    });
    let vocab_suffix = vocab_suffix.map(
        |SuffixKernels {
             suffix,
             qmv,
             argmax,
         }| {
            let VocabSuffix {
                prefix,
                rows,
                hidden,
                bits,
                group,
                scale,
                verify_rows,
                window,
            } = suffix;
            use scratchy_target_metal::tape::quantized::ScaleDtype;
            // Minted in a const context: a width or group no kernel is instantiated for fails the
            // build.
            let (bits, group) = (bits.get(), group.get());
            let scale = match scale {
                ScaleDtype::F16 => {
                    quote!(::scratchy_target_metal::tape::quantized::ScaleDtype::F16)
                }
                ScaleDtype::Bf16 => {
                    quote!(::scratchy_target_metal::tape::quantized::ScaleDtype::Bf16)
                }
            };
            let ByCodes {
                as_written,
                offset8,
            } = qmv;
            quote!(::scratchy_target_metal::off_tape::SuffixKernels {
                suffix: ::scratchy_target_metal::off_tape::VocabSuffix {
                    prefix: #prefix,
                    rows: #rows,
                    hidden: #hidden,
                    bits: const { ::scratchy_target_metal::tape::ids::AffineBits::of(#bits) },
                    group: const { ::scratchy_target_metal::tape::ids::AffineGroupSize::of(#group) },
                    scale: #scale,
                    verify_rows: #verify_rows,
                    window: #window,
                },
                qmv: ::scratchy_target_metal::off_tape::ByCodes {
                    as_written: &[#(#as_written),*],
                    offset8: &[#(#offset8),*],
                },
                argmax: #argmax,
            })
        },
    );
    let vocab = vocab.get();
    let sampler = sampler.iter().map(|row| {
        let row = row.iter();
        quote!([#(#row),*])
    });
    let option = |k: Option<TokenStream>| match k {
        Some(k) => quote!(::core::option::Option::Some(#k)),
        None => quote!(::core::option::Option::None),
    };
    let (mtp_chain, select_rows, vocab_suffix, dflash2_accept, dflash2_sampling) = (
        option(mtp_chain),
        option(select_rows),
        option(vocab_suffix),
        option(dflash2_accept),
        option(dflash2_sampling),
    );
    quote! {
        /// This model's off-tape kernels, baked for its logits.
        #[cfg(scratchy_target = "metal")]
        pub static METAL_OFF_TAPE: ::scratchy_target_metal::off_tape::OffTapeKernels = {
            ::scratchy_target_metal::off_tape::OffTape {
                vocab: ::scratchy_target_metal::tape::ids::LogitsWidth(#vocab),
                argmax: #argmax,
                argmax_dual_write: #argmax_dual_write,
                logprobs: #logprobs,
                grammar_mask: #grammar_mask,
                sampler: [#(#sampler),*],
                chain_advance: #chain_advance,
                mtp_chain: #mtp_chain,
                select_rows: #select_rows,
                vocab_suffix: #vocab_suffix,
                dflash2_accept: #dflash2_accept,
                dflash2_sampling: #dflash2_sampling,
            }
        };
    }
}

/// A multi-token-prediction head drafting over a prefix of its target's vocabulary — its logits
/// `vocab_size` wide, its target's `target_vocab_size` — whose lm_head is MLX-affine: the rest of
/// its target's lm_head (`scratchy_target_metal::off_tape::VocabSuffix`), at the widths its
/// lm_head is quantized to, a pass reading at most its `spec_verify_rows` rows. `None` for any
/// other model; a head whose lm_head is stored otherwise (a dense bf16 build) has no kernel past
/// its prefix, and drafts over the prefix alone.
#[cfg(feature = "metal")]
fn metal_vocab_suffix(
    program: &Program,
    fuf: &crate::fuf::Fuf,
    model: &crate::config::ModelParams,
    mc: &scratchy_ir::model_facts::ModelFacts,
) -> Option<scratchy_target_metal::off_tape::VocabSuffix> {
    let (&prefix, &all) = (
        model.bounds.get("vocab_size")?,
        model.bounds.get("target_vocab_size")?,
    );
    if all <= prefix {
        return None;
    }
    let stem = &model.source_stem;
    let bound = |name: &str| {
        *(model.bounds.get(name)).unwrap_or_else(|| {
            panic!("`{stem}`: a head drafting over a vocabulary prefix declares its `{name}`")
        })
    };
    let (verify_rows, window) = (bound("spec_verify_rows"), bound("spec_wide_window"));
    let lm_head = lent_weight_prefix(
        program,
        fuf,
        model,
        scratchy_forward_compiler::LentWeight::LmHead,
    )?;
    let method = &model.quantization.as_ref()?.method;
    let (group, bits) = crate::quantization::affine_role_widths(program, method, &lm_head, None)?;
    let fits = |v: u64| u32::try_from(v).expect("a vocabulary fits u32");
    use scratchy_target_metal::tape::ids::{AffineBits, AffineGroupSize};
    Some(scratchy_target_metal::off_tape::VocabSuffix {
        prefix: fits(prefix),
        rows: fits(all - prefix),
        hidden: fits(mc.hidden().get() as u64),
        bits: AffineBits::new(bits).unwrap_or_else(|| {
            panic!("the lm_head's {bits}-bit width is one no affine kernel is instantiated for")
        }),
        group: AffineGroupSize::new(group).unwrap_or_else(|| {
            panic!("the lm_head's group of {group} is one no affine kernel is instantiated for")
        }),
        scale: mc.scale_dtype(),
        verify_rows: fits(verify_rows),
        window: fits(window),
    })
}

/// The model's `METAL_TQ_CODEBOOK`: its TurboQuant codebook at the global head dim, computed here
/// (`None` for a dense KV cache).
#[cfg(feature = "metal")]
fn metal_tq_codebook_tokens(mc: &scratchy_ir::model_facts::ModelFacts) -> TokenStream {
    use scratchy_target_metal::turboquant::TqCodebook;
    let codebook = match mc.kv_codec() {
        scratchy_forward_compiler::KvCodec::Dense => quote!(::core::option::Option::None),
        scratchy_forward_compiler::KvCodec::TurboQuant(bits) => {
            // The global class's head: the codebook rotates the full layers' keys.
            let attention =
                (mc.attention()).expect("ModelFacts::new: a TurboQuant KV codec is an attention's");
            let cb = TqCodebook::compute(attention.global().head_dim().get() as usize, bits);
            let table = |t: &[f32]| {
                let words = t.iter().map(|v| v.to_bits());
                quote!(&[#(f32::from_bits(#words)),*])
            };
            let (signs, centroids) = (table(cb.signs), table(cb.centroids));
            let (boundaries, bits, packed_dim) = (table(cb.boundaries), bits.get(), cb.packed_dim);
            quote!(::core::option::Option::Some(::scratchy_target_metal::turboquant::TqCodebook {
                bits: ::scratchy_forward_compiler::TqBits::new(#bits),
                packed_dim: #packed_dim,
                signs: #signs,
                centroids: #centroids,
                boundaries: #boundaries,
            }))
        }
    };
    quote! {
        /// This model's TurboQuant codebook (`None`: dense KV cache).
        #[cfg(scratchy_target = "metal")]
        pub static METAL_TQ_CODEBOOK: ::core::option::Option<
            ::scratchy_target_metal::turboquant::TqCodebook,
        > = #codebook;
    }
}

fn backbone_layout(fuf: &Fuf, program: &Program) -> BackboneLayout {
    const LM_HEAD_PREFIX: &str = "lm_head";

    let last_node = last_non_splice_node(fuf).expect("FUF must be non-empty to emit a forward fn");
    // Granite-class arches divide the logits by `logits_scaling`, so the FUF
    // terminates in `Mul(gemm_out, scalar)` rather than the lm_head Gemm
    // itself. That is still a DECODER: walk past the scalar multiply to the
    // Gemm, exactly as the AllGather arm below walks past tp>1's collective.
    // Misreading it as an encoder bakes `METAL_VOCAB_SIZE = hidden_size`, and
    // the argmax kernel then scans only the first `hidden_size` of
    // `vocab_size` logits — greedy decode picks low-id tokens (asterisks and
    // single characters) while sampling, which does not use that kernel,
    // stays clean.
    // Walk back over any chain of these; gemma2 has BOTH a softcap and (via
    // other stems) a scale, and a decoder must not be misread because its
    // logits get post-processed.
    let mut last_node = last_node;
    loop {
        let terminal_postop = match last_node.op {
            // granite `logits_scaling`, commandr `logit_scale`
            crate::classified::OpKind::Mul => last_node
                .inputs
                .iter()
                .any(|i| matches!(i, FufInput::Scalar(_))),
            // gemma2 `final_logit_softcapping`
            crate::classified::OpKind::TanhSoftCap => true,
            _ => false,
        };
        if !terminal_postop {
            break;
        }
        match last_node
            .inputs
            .iter()
            .find(|i| matches!(i, FufInput::Tile { .. }))
        {
            Some(FufInput::Tile { id, .. }) => last_node = fuf.get(*id),
            _ => break,
        }
    }
    let lm_head_node = if last_node.op == crate::classified::OpKind::AllGather {
        // tp>1: walk past the AllGather to the underlying Gemm.
        // AllGather has exactly one tile input by construction.
        match last_node.inputs.first() {
            Some(FufInput::Tile { id, .. }) => fuf.get(*id),
            _ => return BackboneLayout::Encoder,
        }
    } else {
        last_node
    };

    // Decoder shape: `gemm(<tile>, <lm_head weight>)`.
    if lm_head_node.op != crate::classified::OpKind::Gemm {
        return BackboneLayout::Encoder;
    }
    let weight_is_lm_head = matches!(
        lm_head_node.inputs.get(1),
        Some(FufInput::Weight { id, .. })
            if program
                .weights
                .path(*id)
                .first()
                .map(|s| s.as_str())
                == Some(LM_HEAD_PREFIX)
    );
    if !weight_is_lm_head {
        return BackboneLayout::Encoder;
    }
    match lm_head_node.inputs.first() {
        Some(FufInput::Tile { .. }) => BackboneLayout::Decoder,
        _ => BackboneLayout::Encoder,
    }
}

/// Build the workload-point bounds map the tape lowering consumes —
/// model.bounds + the workload-specific `num_tokens` and `sk_bucket`
/// overrides.
fn bounds_for_wp(
    model: &ModelParams,
    wp: WorkloadPoint,
    tp_world_size: u8,
) -> BTreeMap<String, u64> {
    let mut bounds = model.bounds.clone();
    bounds.insert("num_tokens".to_string(), wp.num_tokens);
    bounds.insert("sk_bucket".to_string(), wp.sk_bucket);
    // At tp>1, runtime weights are per-rank shards; the codegen
    // `gemm_nk_from_fuf` evaluates the FUF's symbolic `Shape`
    // against this bounds map and bakes the resulting (n, k) into
    // the emitted `Instruction` for `assert_weight_shape` to check
    // at runtime. Per-rank weights ⇒ per-rank bounds, or the
    // assertion (added in 812452cff) panics with a sharded/unsharded
    // K mismatch on the first row-parallel gemm.
    if tp_world_size > 1 {
        let tp = tp_world_size as u64;
        for k in [
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
        ] {
            if let Some(v) = bounds.get_mut(k) {
                *v = (*v / tp).max(1);
            }
        }
    }
    bounds
}

/// Whether this model uses rotary position embeddings — any `Rotary` extern
/// input in the Fuf. Drives `CanonicalParams::ROPE_ON_READ` (rope-on-read is
/// the universal default for rope models) and the matching
/// `ScratchyWeights::rope_on_read()` the worker reads. Same predicate as the
/// rotary-cache detection in `emit_weights_struct`.
pub fn fuf_uses_rotary(fuf: &Fuf) -> bool {
    fuf.nodes.iter().any(|n| {
        n.inputs.iter().any(|i| {
            matches!(
                i,
                crate::fuf::FufInput::Extern {
                    kind: crate::classified::ExternKind::Rotary,
                    ..
                }
            )
        })
    })
}

/// The layers whose ops read or write the paged KV cache, ascending: every layer of a uniform
/// decoder, a recurrent hybrid's attention layers only (its Mamba / Gated-DeltaNet layers hold
/// none), no layer of an encoder or a vision tower. The KV pool holds these layers' tensors. The
/// TurboQuant threshold does NOT count them: its `num_layers` input stays `num_hidden_layers`,
/// the input it was calibrated against — re-classifying existing models' codec is a policy
/// change and belongs in its own PR, not a side effect of a new arch.
#[derive(Default)]
pub struct KvCacheLayers(Vec<u32>);

impl KvCacheLayers {
    /// The layer index of every `kv_cache[layer]` the unrolled graph names.
    pub fn of(fuf: &Fuf) -> Self {
        let layers: std::collections::BTreeSet<u32> = fuf
            .nodes
            .iter()
            .flat_map(|n| &n.inputs)
            .filter_map(|i| match i {
                crate::fuf::FufInput::Extern {
                    kind: crate::classified::ExternKind::KvCache,
                    index,
                } => Some(index.map_or(0, |l| u32::try_from(l.0).expect("a u32 layer index"))),
                _ => None,
            })
            .collect();
        Self(layers.into_iter().collect())
    }

    /// The model reads or writes a paged KV cache at all.
    pub fn any(&self) -> bool {
        !self.0.is_empty()
    }

    /// How many layers hold KV.
    pub fn count(&self) -> usize {
        self.0.len()
    }

    /// The layers, when they are not all `num_hidden_layers` of them (nor none).
    pub fn proper_subset(&self, num_hidden_layers: usize) -> Option<&[u32]> {
        (self.any() && self.count() < num_hidden_layers).then_some(self.0.as_slice())
    }
}

/// The model's KV codec. TurboQuant under the `turboquant` feature, on metal
/// only, wherever `codec_for` takes its geometry; dense everywhere else. A
/// model the feature would compress but can't is named in the build output.
/// The geometry is the KV pool's: a head's KV is its target's capacity, a
/// pool mirroring the target's blocks, so it is decided at the target's layer
/// count (`target_num_hidden_layers`) — the target's codec.
fn kv_codec_for(
    model: &ModelParams,
    head_dim: u32,
    global_head_dim: u32,
    kv_layers: &KvCacheLayers,
    bits: scratchy_forward_compiler::TqBits,
) -> scratchy_forward_compiler::KvCodec {
    #[cfg(all(feature = "turboquant", feature = "metal"))]
    {
        use scratchy_ir::{DenseReason, KvGeometry, codec_for};
        let bound = |k: &str| *model.bounds.get(k).unwrap_or(&0) as usize;
        let pool_layers = (model.bounds.get("target_num_hidden_layers"))
            .map_or(model.text.layers.get(), |&n| n as usize);
        let geometry = KvGeometry {
            num_layers: pool_layers,
            num_kv_heads: bound("num_key_value_heads"),
            head_dim,
            global_head_dim,
            latent: bound("kv_lora_rank") > 0,
            has_kv_cache: kv_layers.any(),
        };
        match codec_for(geometry, bits) {
            Ok(bits) => scratchy_forward_compiler::KvCodec::TurboQuant(bits),
            Err(reason) => {
                if reason != DenseReason::NoKvCache {
                    println!(
                        "cargo:warning=turboquant: {} keeps a dense KV cache: {reason}",
                        model.source_stem
                    );
                }
                scratchy_forward_compiler::KvCodec::Dense
            }
        }
    }
    #[cfg(not(all(feature = "turboquant", feature = "metal")))]
    {
        let _ = (model, head_dim, global_head_dim, kv_layers, bits);
        scratchy_forward_compiler::KvCodec::Dense
    }
}

/// A canonical's facts as the expression its generated `CanonicalParams::FACTS` states: every
/// group through its own constructor (an attention through `Attention::of`, a GDN through
/// `Gdn::of`: their laws bounds on their numbers as types), so the emitted const re-checks each
/// law when rustc type-checks it. The one serializer of the value the fill built (and the metal
/// bake read).
fn model_facts_tokens(f: scratchy_ir::model_facts::ModelFacts) -> TokenStream {
    use scratchy_ir::model_facts::{
        Attention, AttnClass, KvRope, RopePairing, TargetRouting, TargetWidth, Vision,
        VisionRopePairing,
    };
    let m = quote! { ::scratchy_forward_compiler::model_facts };
    let u32_ = |v: u32| proc_macro2::Literal::u32_unsuffixed(v);
    let usize_ = |v: usize| proc_macro2::Literal::usize_unsuffixed(v);
    // Every float through its bits: the emitted value is the fill's value exactly.
    let f32_ = |v: f32| {
        let b = proc_macro2::Literal::u32_suffixed(v.to_bits());
        quote! { f32::from_bits(#b) }
    };
    let opt = |t: Option<TokenStream>| match t {
        Some(t) => quote! { ::core::option::Option::Some(#t) },
        None => quote! { ::core::option::Option::None },
    };
    let n = |v: u32| unsigned_tokens(u64::from(v));
    let class = |g: scratchy_ir::model_facts::HeadGeometry| {
        let (kv, hd, rot, page) = (
            n(g.kv_heads().get()),
            n(g.head_dim().get()),
            n(g.rot_dim().get()),
            n(g.page().get()),
        );
        quote! { #m::Class<#kv, #hd, #rot, #page> }
    };
    let attention = opt(f.attention().map(|a: Attention| {
        let (q, base, global) = (
            n(a.q_heads().get()),
            class(a.class(AttnClass::Sliding)),
            class(a.class(AttnClass::Full)),
        );
        let scale = f32_(a.scale().get());
        let softcap = opt(a.softcap().map(|c| {
            let c = f32_(c.get());
            quote! { #m::SoftCap::new(#c) }
        }));
        let window = opt(a.window().map(|w| {
            let w = u32_(w.get());
            quote! { #m::AttnWindow::new(#w) }
        }));
        let pairing = match a.pairing() {
            RopePairing::Standard => quote! { #m::RopePairing::Standard },
            RopePairing::Proportional => quote! { #m::RopePairing::Proportional },
        };
        let kv_rope = match a.kv_rope() {
            KvRope::AtWrite => quote! { #m::KvRope::AtWrite },
            KvRope::OnRead => quote! { #m::KvRope::OnRead },
        };
        let mrope = match a.mrope() {
            Some(s) => {
                let s = s.get().map(n);
                quote! { #m::Mrope<#(#s),*> }
            }
            None => quote! { #m::NoMrope },
        };
        quote! {
            #m::Attention::of::<#q, #base, #global, #mrope>(
                #m::AttnScale::new(#scale),
                #softcap,
                #window,
                #pairing,
                #kv_rope,
            )
        }
    }));
    let vision = opt(f.vision().map(|v: Vision| {
        let (heads, hd, inf) = (
            u32_(v.heads().get()),
            u32_(v.head_dim().get()),
            usize_(v.in_features().get()),
        );
        let pairing = match v.pairing() {
            VisionRopePairing::RotateHalf => quote! { #m::VisionRopePairing::RotateHalf },
            VisionRopePairing::Interleaved => quote! { #m::VisionRopePairing::Interleaved },
        };
        let scale = f32_(v.scale().get());
        let grid = opt(v.patch_grid_side().map(|s| {
            let s = u32_(s.get());
            quote! { #m::PatchGridSide::new(#s) }
        }));
        let pool = opt(v.pool_kernel().map(|k| {
            let k = u32_(k.get());
            quote! { #m::PoolKernel::new(#k) }
        }));
        quote! {
            #m::Vision::new(
                #m::QueryHeads::new(#heads),
                #m::HeadDim::new(#hd),
                #m::InFeatures::new(#inf),
                #pairing,
                #m::AttnScale::new(#scale),
                #grid,
                #pool,
            )
        }
    }));
    let gdn = opt(f.gdn().map(|g| {
        let (kh, vh, kd, vd, ck, cd) = (
            n(g.k_heads().get()),
            n(g.v_heads().get()),
            n(g.k_dim().get()),
            n(g.v_dim().get()),
            n(g.conv_kernel().get()),
            unsigned_tokens(g.conv_dim().get() as u64),
        );
        quote! { #m::Gdn::of::<#kh, #vh, #kd, #vd, #ck, #cd>() }
    }));
    let drafter = opt(f.drafter().map(|d| {
        let (g, r, t) = (
            u32_(d.conv_group().get()),
            u32_(d.selector_rank().get()),
            u32_(d.mask_token().get()),
        );
        quote! {
            #m::Drafter::new(#m::ConvGroup::new(#g), #m::SelectorRank::new(#r), #m::TokenId::new(#t))
        }
    }));
    let target = opt(f.target().map(|t| {
        let hidden = usize_(t.hidden().get());
        let routing = match t.routing() {
            TargetRouting::Dense => quote! { #m::TargetRouting::Dense },
            TargetRouting::Experts => quote! { #m::TargetRouting::Experts },
        };
        let ctor = match t.width() {
            TargetWidth::Own => quote! { new },
            TargetWidth::Lent => quote! { lent },
        };
        quote! { #m::Target::#ctor(#m::HiddenSize::new(#hidden), #routing) }
    }));
    let kv_layers = opt(f.kv_layers().map(|l| {
        let l = l.get().iter().map(|&v| u32_(v));
        quote! { #m::KvLayers::new(&[#(#l),*]) }
    }));
    let kv_codec = match f.kv_codec() {
        scratchy_ir::KvCodec::Dense => quote! { ::scratchy_forward_compiler::KvCodec::Dense },
        scratchy_ir::KvCodec::TurboQuant(bits) => {
            let b = u32_(bits.get());
            quote! {
                ::scratchy_forward_compiler::KvCodec::TurboQuant(::scratchy_forward_compiler::TqBits::new(#b))
            }
        }
    };
    let metal_dtype = match f.metal_dtype() {
        scratchy_ir::model_facts::MetalDtype::Bf16 => quote! { #m::MetalDtype::Bf16 },
        scratchy_ir::model_facts::MetalDtype::F16 => quote! { #m::MetalDtype::F16 },
        scratchy_ir::model_facts::MetalDtype::Int4 => quote! { #m::MetalDtype::Int4 },
    };
    let scale_dtype = match f.scale_dtype() {
        scratchy_ir::model_facts::ScaleDtype::Bf16 => quote! { #m::ScaleDtype::Bf16 },
        scratchy_ir::model_facts::ScaleDtype::F16 => quote! { #m::ScaleDtype::F16 },
    };
    let hidden = usize_(f.hidden().get());
    let intermediate = opt(f.intermediate().map(|i| {
        let i = usize_(i.get());
        quote! { #m::IntermediateSize::new(#i) }
    }));
    let vocab = opt(f.vocab().map(|v| {
        let v = usize_(v.get());
        quote! { #m::VocabSize::new(#v) }
    }));
    let (eps, offset) = (f32_(f.eps().get()), f32_(f.norm_offset().get()));
    let final_softcap = opt(f.final_softcap().map(|c| {
        let c = f32_(c.get());
        quote! { #m::SoftCap::new(#c) }
    }));
    let spec_drafts = opt(f.spec_drafts().map(|d| {
        let d = u32_(d.get());
        quote! { #m::NumDrafts::new(#d) }
    }));
    let capture_slots = opt(f.capture_slots().map(|s| {
        let s = u32_(s.get());
        quote! { #m::CaptureSlots::new(#s) }
    }));
    quote! {
        #m::ModelFacts::new(#m::FactsParts {
            metal_dtype: #metal_dtype,
            scale_dtype: #scale_dtype,
            hidden: #m::HiddenSize::new(#hidden),
            intermediate: #intermediate,
            vocab: #vocab,
            eps: #m::RmsNormEps::new(#eps),
            norm_offset: #m::NormWeightOffset::new(#offset),
            final_softcap: #final_softcap,
            kv_codec: #kv_codec,
            kv_layers: #kv_layers,
            attention: #attention,
            vision: #vision,
            gdn: #gdn,
            drafter: #drafter,
            spec_drafts: #spec_drafts,
            capture_slots: #capture_slots,
            target: #target,
        })
    }
}

/// How a head's width stands to its target's, from its program's `as_dim` casts: a head that
/// states its rows at its target's width (`as_dim(<head-wide>, backbone_hidden_size)`) is lent it,
/// and [`ModelFacts::new`](scratchy_ir::model_facts::ModelFacts::new) holds the two equal; a head
/// with no cast reads its target through its own projection. A cast of any other two widths has
/// no fact for that law to read, so it is refused.
pub(crate) fn target_width(
    program: &Program,
    inferred: &crate::shape::Inferred,
) -> scratchy_ir::model_facts::TargetWidth {
    use crate::classified::{Expr, LocalId, Stmt};
    use crate::dim::Dim;
    let bound = |d: &Dim, name: &str| matches!(d, Dim::Bound(b) if b == name);
    // Each cast's source: the local `x = as_dim(y, ..)` reads.
    fn sources(
        stmts: &[Stmt],
        casts: &std::collections::HashSet<LocalId>,
        out: &mut Vec<(LocalId, LocalId)>,
    ) {
        for s in stmts {
            match s {
                Stmt::Assign {
                    target,
                    value: Expr::Local(source),
                } if casts.contains(target) => out.push((*target, *source)),
                Stmt::For { body, .. } => sources(body, casts, out),
                Stmt::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    sources(then_body, casts, out);
                    sources(else_body, casts, out);
                }
                _ => {}
            }
        }
    }
    let casts = program.dim_casts.keys().copied().collect();
    let mut cast_sources = Vec::new();
    sources(&program.statements, &casts, &mut cast_sources);
    assert_eq!(
        cast_sources.len(),
        program.dim_casts.len(),
        "as_dim: every cast reads one local"
    );
    for (local, source) in cast_sources {
        let stated = &program.dim_casts[&local];
        let own = (inferred.locals.get(&source)).and_then(|shape| shape.last());
        assert!(
            own.is_some_and(|d| bound(d, "hidden_size")) && bound(stated, "backbone_hidden_size"),
            "as_dim: `{}` (line {}) states {own:?} as {stated:?}; the one cast a model's facts \
             hold is a head's hidden width as its target's (`hidden_size` as \
             `backbone_hidden_size`)",
            program.locals.name(local),
            (program.locals.line(local)).map_or("?".to_string(), |l| l.to_string()),
        );
    }
    match program.dim_casts.is_empty() {
        true => scratchy_ir::model_facts::TargetWidth::Own,
        false => scratchy_ir::model_facts::TargetWidth::Lent,
    }
}

/// Whether a body reads its target's final hidden rows (`RowsExtern::TargetHidden`): a head.
fn fuf_reads_target_hidden(fuf: &Fuf) -> bool {
    fuf.nodes.iter().any(|n| {
        n.op == crate::classified::OpKind::LoadRows(
            scratchy_forward_compiler::RowsExtern::TargetHidden,
        )
    })
}

/// A canonical's facts at `tp_world_size`, from its config, and the six MLA consts no fact covers
/// yet, as its `CanonicalParams` impl states them.
pub(crate) fn fill_canonical_params(
    model: &ModelParams,
    tp_world_size: u8,
    uses_rotary: bool,
    kv_layers: &KvCacheLayers,
    // For a body that reads its target's hidden rows (a head), how its width stands to its
    // target's ([`target_width`]): the facts state its target then.
    target: Option<scratchy_ir::model_facts::TargetWidth>,
) -> (scratchy_ir::model_facts::ModelFacts, TokenStream) {
    let tp = tp_world_size as u32;
    let tp_us = tp_world_size as usize;
    let head_dim = *model.bounds.get("head_dim").unwrap_or(&0) as u32;
    let num_q_heads = (*model.bounds.get("num_attention_heads").unwrap_or(&0) as u32) / tp;
    let num_kv_heads = (*model.bounds.get("num_key_value_heads").unwrap_or(&0) as u32) / tp;
    // For text bodies this comes straight off `intermediate_size`. For
    // vision-only bodies (like Qwen2.5-VL) the bound is absent, but
    // `vision_intermediate_size_padded` carries the SwiGLU intermediate
    // width that `silu_and_mul_fused`'s split-point reads out of
    // `W::INTERMEDIATE_SIZE` at runtime — so fall back to that. Vision
    // is replicated per-rank, so don't divide by tp.
    let intermediate_size = if let Some(v) = model.bounds.get("intermediate_size") {
        (*v as usize) / tp_us
    } else if let Some(v) = model.bounds.get("vision_intermediate_size_padded") {
        *v as usize
    } else {
        0
    };
    let q_size = (num_q_heads as usize) * (head_dim as usize);
    let kv_lora_rank = *model.bounds.get("kv_lora_rank").unwrap_or(&0) as usize;
    let qk_nope_head_dim = *model.bounds.get("qk_nope_head_dim").unwrap_or(&0) as usize;
    let qk_rope_head_dim = *model.bounds.get("qk_rope_head_dim").unwrap_or(&0) as usize;
    let v_head_dim = *model.bounds.get("v_head_dim").unwrap_or(&0) as usize;
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    let spec_drafts = *model.bounds.get("spec_drafts").unwrap_or(&0) as u32;
    // A DFlash2 drafter's geometry (its arch.json params), and the blocks of the rows a target
    // captures for one.
    let draft_conv_group = *model.bounds.get("conv_group_size").unwrap_or(&0) as u32;
    let draft_selector_rank = *model.bounds.get("selector_rank").unwrap_or(&0) as u32;
    let draft_mask_token = *model.bounds.get("mask_token_id").unwrap_or(&0) as u32;
    let draft_capture_slots = (model
        .layer_sets
        .get(crate::draft_capture::DRAFT_CAPTURE_LAYERS))
    .map_or(0, |s| s.members().len() as u32);

    // Vision-tower constants. Set in `#[vision_forward]` configs via
    // `vision_num_heads` / `vision_head_dim` bounds; absent in text
    // configs so the defaults (0 / 0.0) match the trait defaults
    // declared in `scratchy_forward_compiler::CanonicalParams`. Q_SIZE is the
    // rank-2 last-dim (`H*D`) the FUF carries; VISION_ATTN_SCALE is
    // `1/sqrt(head_dim)`.
    let vision_num_heads = *model.bounds.get("vision_num_heads").unwrap_or(&0) as u32;
    let vision_head_dim = *model.bounds.get("vision_head_dim").unwrap_or(&0) as u32;
    let vision_q_size = (vision_num_heads as usize) * (vision_head_dim as usize);
    let vision_in_features = *model.bounds.get("vision_in_features").unwrap_or(&0) as usize;
    let vision_attn_scale: f32 = if vision_head_dim > 0 {
        1.0_f32 / (vision_head_dim as f32).sqrt()
    } else {
        0.0
    };
    // Vision 2D-RoPE pairing convention (`vision_rope_style` config
    // key, validated at parse): absent/"neox_hw" → false (Qwen),
    // "interleaved_xy" → true (MoonViT / LocateAnything). Selects the
    // metal `vision_rope_2d[_interleaved]` entry point at lowering.
    let vision_rope_interleaved =
        matches!(model.arch.rope_style.as_deref(), Some("interleaved_xy"));

    // SigLIP-style patch grid side (square); zero for arches that
    // don't carry an image-tower patch grid. `Instruction::AvgPool2d`
    // reads this to fold flat-row index → (row, col) on the encoder's
    // pooling pass.
    let vision_patch_grid_side = *model.bounds.get("vision_patch_grid_side").unwrap_or(&0) as u32;
    let vision_pool_kernel = *model.bounds.get("vision_pool_kernel").unwrap_or(&0) as u32;

    // attention_multiplier (Granite override) → query_pre_attn_scalar
    // (Gemma2) → 1/sqrt(head_dim). Default 0.0 if no attention path.
    let attn_scale: f32 = if let Some(s) = model.scalars.get("attention_multiplier") {
        *s as f32
    } else if let Some(q) = model.scalars.get("query_pre_attn_scalar") {
        (*q as f32).powf(-0.5)
    } else if head_dim > 0 {
        1.0_f32 / (head_dim as f32).sqrt()
    } else {
        0.0
    };
    let attn_softcap: f32 = model
        .scalars
        .get("attn_logit_softcapping")
        .copied()
        .unwrap_or(0.0) as f32;
    let sliding_window: i32 = model
        .bounds
        .get("sliding_window")
        .map(|w| *w as i32)
        .unwrap_or(-1);
    let final_logit_softcapping: f32 = model
        .scalars
        .get("final_logit_softcapping")
        .copied()
        .unwrap_or(0.0) as f32;
    // MLA scale: 1/sqrt(qk_head_dim) with YaRN mscale_all_dim
    // correction. Mirrors MlaAttentionImpl's previous bake.
    let mla_attn_scale: f32 = if qk_head_dim > 0 {
        let base = 1.0_f32 / (qk_head_dim as f32).sqrt();
        match &model.rope_scaling {
            Some(crate::config::RopeScaling::Yarn {
                factor,
                mscale_all_dim,
                ..
            }) if *mscale_all_dim != 0.0 => {
                let mm = if *factor <= 1.0 {
                    1.0_f64
                } else {
                    0.1 * mscale_all_dim * factor.ln() + 1.0
                };
                base * (mm * mm) as f32
            }
            _ => base,
        }
    } else {
        0.0
    };

    // Vision-only bodies (the `#[vision_forward]` crates — qwen2-vl,
    // qwen2.5-vl, qwen3.5-vl, locateanything) carry NO text
    // `head_dim` / `num_attention_heads` / `hidden_size` bounds, so the
    // text-named consts above resolved to 0 (Q_SIZE = 0·0, HIDDEN_SIZE
    // fell back to q_size = 0). That's invisible for towers that read
    // only the `VISION_*` consts — but qwen2.5-vl's dense SwiGLU lowers
    // to `FusedGateUpSiluMul`, whose gate/up gemm K-dim is `W::HIDDEN_SIZE`
    // and inner width `W::INTERMEDIATE_SIZE`; a K=0 gemm zeros the whole
    // MLP. Re-derive the residual-stream consts from the vision bounds so
    // any `W::`-const reader on a vision Weights sees the tower's real
    // width. Vision attention itself reads `VISION_*` and is unchanged;
    // text crates (`num_q_heads > 0`) skip this branch entirely, so no
    // committed text/VL-text decoder shifts.
    let vision_embed_dim = *model.bounds.get("vision_embed_dim").unwrap_or(&0) as usize;
    let is_vision_only = num_q_heads == 0 && vision_num_heads > 0;
    let head_dim = if is_vision_only {
        vision_head_dim
    } else {
        head_dim
    };
    let num_q_heads = if is_vision_only {
        vision_num_heads
    } else {
        num_q_heads
    };
    let q_size = if is_vision_only {
        vision_q_size
    } else {
        q_size
    };

    // HIDDEN_SIZE = residual stream width = config.json `hidden_size`.
    // Distinct from Q_SIZE on GQA arches with head_dim != hidden/num_heads.
    // Vision-only bodies fall back to `vision_embed_dim` (not q_size) so
    // the SwiGLU K-dim above is the tower width even when H·D != embed.
    let hidden_size_for_const: usize = model
        .bounds
        .get("hidden_size")
        .copied()
        .map(|v| v as usize)
        .unwrap_or(if is_vision_only {
            vision_embed_dim
        } else {
            q_size
        });
    // A head's `target_hidden` rows: a DFlash2 drafter's captured layers side by side, else its
    // target's hidden width.
    let target_hidden_width =
        (model.bounds.get("backbone_hidden_size")).map_or(hidden_size_for_const, |&w| w as usize);
    let target_routes_experts = model
        .bounds
        .get("target_num_experts")
        .is_some_and(|&n| n > 0);
    // VOCAB_SIZE: lm_head output dim. Read from the verbatim HF config
    // (`vocab_size` bound). Used by the generic `Instruction::Gemm` arm
    // to identify lm_head and apply the last-token-per-seq narrow
    // before the GEMM. 0 → not set, the runtime falls back to the
    // legacy `n > INTERMEDIATE_SIZE` heuristic.
    let vocab_size_for_const: usize = model
        .bounds
        .get("vocab_size")
        .copied()
        .map(|v| v as usize)
        .unwrap_or(0);
    let kv_lora_rank_lit = proc_macro2::Literal::usize_unsuffixed(kv_lora_rank);
    let qk_nope_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_nope_head_dim);
    let qk_rope_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_rope_head_dim);
    let v_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(v_head_dim);
    let qk_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_head_dim);
    // Rotary dim for the metal rope kernel's `ROT_DIM` fn-const. Mirrors
    // the rope-cache builder (codegen.rs ~2758): `partial_rotary_factor
    // * head_dim` for partial-rope arches (Qwen3.5: 0.25*256 = 64), else
    // the full `head_dim`. Without this the impl falls back to the trait
    // default `ROT_DIM = HEAD_DIM`, so the kernel rotates head_dim dims
    // against a rotary_dim-wide cos/sin cache → partial-rope arches break.
    let rot_dim_val: u32 = model
        .scalars
        .get("partial_rotary_factor")
        .copied()
        .filter(|&f| (f - 1.0).abs() > 1e-9)
        .map(|f| (f * head_dim as f64).round() as u32)
        .unwrap_or(head_dim);
    // Per-layer-class geometry for hybrid sliding/global arches whose
    // classes differ in dims (Gemma4: sliding 256×8kv full-rope vs
    // global 512×1kv proportional-rope-128). Config keys
    // `global_head_dim` / `num_global_key_value_heads` /
    // `global_partial_rotary_factor` feed the GLOBAL_* consts read by
    // the full-attention metal lowering arms; absent keys fall back to
    // the base values (uniform models unchanged).
    let global_head_dim: u32 = model
        .bounds
        .get("global_head_dim")
        .map(|&v| v as u32)
        .unwrap_or(head_dim);
    let num_global_kv_heads: u32 = model
        .bounds
        .get("num_global_key_value_heads")
        .map(|&v| (v as u32) / tp)
        .unwrap_or(num_kv_heads);
    let global_rot_dim: u32 = model
        .scalars
        .get("global_partial_rotary_factor")
        .copied()
        .filter(|&f| (f - 1.0).abs() > 1e-9)
        .map(|f| (f * global_head_dim as f64).round() as u32)
        // Fallback must honor the BASE partial factor on uniform
        // models (global_head_dim == head_dim): Qwen3.5's full-
        // attention layers lower through the is_global arm, and the
        // old `global_head_dim` fallback rotated the FULL 256-dim
        // head against the 64-wide partial cos/sin table — rows past
        // the table read zeros and zeroed Q/K for every token beyond
        // ~4 (the "rambling, input-blind thinker" failure). Hybrid
        // arches with distinct global dims and genuinely full global
        // rope (no factor) keep the old fallback.
        .unwrap_or(if global_head_dim == head_dim {
            rot_dim_val
        } else {
            global_head_dim
        });
    // GUARD: the rope cos/sin table is
    // rotary_dim wide; a kernel-side rot const that disagrees reads
    // past the table into zeros and silently zeroes Q/K for every
    // token whose row falls outside it (the Qwen3.5 "input-blind
    // rambling thinker" regression). On uniform-geometry models the
    // global lowering arm serves the same heads as the base arm, so
    // the two rot consts MUST agree — enforced at expansion time.
    assert!(
        global_head_dim != head_dim || global_rot_dim == rot_dim_val,
        "model `{}`: GLOBAL_ROT_DIM {} != ROT_DIM {} on uniform geometry          (head_dim {}) — the is_global rope arm would rotate against a          mismatched cos/sin table width",
        model.source_stem,
        global_rot_dim,
        rot_dim_val,
        head_dim,
    );
    // Proportional rope marker — same predicate that routes the
    // rotary cache builder to `new_proportional_from_gpuweights`.
    // Drives the metal rope kernel's pairing offset (lane i pairs
    // with i + head_dim/2, not i + rot_dim/2).
    let rope_proportional = global_rot_dim != global_head_dim
        && model.scalars.contains_key("global_partial_rotary_factor");
    // GLOBAL_BLOCK_SIZE override — vLLM's hybrid-KV page-size unification
    // (`unify_kv_cache_spec_page_size`, kv_cache_utils.py:944:
    // `new_block_size = block_size * ratio`). When the full (global) and
    // sliding classes have different per-token KV footprints, vLLM scales
    // the SMALLER-page class's block_size UP by `max_page / its_page` so both
    // classes' blocks occupy identical bytes and share one physical tensor.
    // Gemma4: sliding `8·256 = 2048` vs global `2·512 = 1024` per token →
    // global is smaller → GLOBAL_BLOCK_SIZE = BLOCK_SIZE · (2048/1024) = 32.
    // Emitted only when the global class is the smaller page (the gemma4
    // direction); equal pages → no emit (default = BLOCK_SIZE). The reverse
    // direction (sliding smaller) would scale BLOCK_SIZE instead and is not
    // emitted by any current arch — guarded loudly below.
    // The base class's KV page is the paged pool's (`DEFAULT_KV_PAGE`, the runtime's
    // `BLOCK_SIZE`): a page is how the pool slices its cache, and no checkpoint states it. A
    // config's `block_size` is the model's own field: LocateAnything's (`"block_size": 6`) is its
    // SDLM block-diffusion generation window (nvidia/LocateAnything-3B `modeling_qwen2.py:1168`,
    // read only by `mask_sdpa_utils.py`'s attention masks), not a KV page.
    let base_block_size: u32 = scratchy_ir::model_facts::DEFAULT_KV_PAGE.get();
    let sliding_per_token = num_kv_heads * head_dim; // base class == sliding on gemma4
    let global_per_token = num_global_kv_heads * global_head_dim;
    let uniform_page =
        global_per_token == sliding_per_token || num_global_kv_heads == 0 || global_per_token == 0;
    let resolved_global_block_size: u32 = if uniform_page {
        base_block_size
    } else if global_per_token < sliding_per_token {
        // Global is the smaller page → scale its block_size up (gemma4). The two classes' pages
        // then hold the same elements: `Attention::of`'s bound, which the model's `FACTS` states,
        // so a ratio with a remainder (vLLM raises the same) fails the model's build (E0277).
        base_block_size * (sliding_per_token / global_per_token)
    } else {
        // Sliding is the smaller page — vLLM would scale BLOCK_SIZE (sliding)
        // up, leaving GLOBAL_BLOCK_SIZE == BLOCK_SIZE. No current arch hits
        // this; fail loud rather than silently mis-unify.
        panic!(
            "model `{}`: sliding page {} < global page {} — page unification \
             must scale the SLIDING block_size up (not yet wired; no arch needs it)",
            model.source_stem, sliding_per_token, global_per_token,
        );
    };
    let mla_attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(mla_attn_scale);

    // SCALE_DTYPE override — only matters under `--features metal`.
    // mlx-community 4bit convention (probed across cached HF snapshots):
    // Llama-3.x / Qwen2.5 / SmolLM ship F16 scales+biases+norm gains,
    // Qwen3 family ships BF16. Default in the trait is F16; emit a
    // `BF16` override for the Qwen3-family architectures so their
    // affine kernels and RMSNorm pipelines pick the matching
    // `_s_bf16_` symbol arms. Match on the HF `architectures` strings
    // baked into `model.architectures`.
    // Declared per arch (`const SCALE_DTYPE` on the carrier mod;
    // per-checkpoint repacks override via the `scale_dtype` JSON
    // drift key). Must stay in sync with the synth-kernel
    // `t_scale` gate above (same declared value).
    let scale_dtype_is_bf16 = {
        let is_bf16_scale = model.arch.scale_dtype.as_deref() == Some("bf16");
        // NVFP4 (NVIDIA ModelOpt) checkpoints ship BF16 RMSNorm gains
        // (and BF16 embed/lm_head), unlike the mlx-community 4bit Llama
        // convention of F16. `SCALE_DTYPE` selects the rmsnorm /
        // fused_add_rmsnorm `_s_<dtype>_` symbol arm that reads the gain
        // pointer; with the F16 default the BF16 gain bytes are
        // mis-read (0x3D41 → 1.31 instead of 0.047) → garbage output.
        // The NVFP4 GEMM scales are independently F16 (folded at load),
        // so its kernels keep their hard-coded `_s_f16_` names.
        let is_nvfp4 = matches!(
            model.quantization.as_ref().map(|qc| &qc.method),
            Some(crate::quantization::QuantMethod::Nvfp4 { .. })
        );
        is_bf16_scale || is_nvfp4
    };

    // TurboQuant KV bit-width policy (Metal). 3-bit (~4.7x) is validated
    // coherent for the Llama family; outlier-heavy KV (Qwen-class massive
    // activations) degrades at 3-bit and needs 4-bit. Promote arches to 3 here
    // as the validation sweep confirms them.
    let tq_bits = {
        let rides_3bit = model
            .architectures
            .iter()
            .any(|a| a.starts_with("Llama") || a.contains("TinyLlama"));
        scratchy_forward_compiler::TqBits::new(if rides_3bit { 3 } else { 4 })
    };
    let kv_codec = kv_codec_for(model, head_dim, global_head_dim, kv_layers, tq_bits);

    let num_hidden_layers = model.text.layers.get();

    let facts = {
        use scratchy_ir::model_facts as mf;
        let positive = |v: u32| (v > 0).then_some(v);
        // A decoder's attention. A vision-only body's text consts above carry its tower's numbers
        // (`is_vision_only`); it has no decoder attention, and its tower is `vision`.
        let attention = (!is_vision_only && head_dim > 0).then(|| {
            let class = |kv_heads: u32, head_dim: u32, rot_dim: u32, page: u32| {
                mf::HeadGeometry::new(
                    mf::QueryHeads::new(num_q_heads),
                    mf::KvHeads::new(kv_heads),
                    mf::HeadDim::new(head_dim),
                    mf::RotDim::new(rot_dim),
                    mf::BlockSize::new(page),
                )
            };
            mf::Attention::new(
                class(num_kv_heads, head_dim, rot_dim_val, base_block_size),
                class(
                    num_global_kv_heads,
                    global_head_dim,
                    global_rot_dim,
                    resolved_global_block_size,
                ),
                mf::AttnScale::new(attn_scale),
                (attn_softcap != 0.0).then(|| mf::SoftCap::new(attn_softcap)),
                (u32::try_from(sliding_window).ok())
                    .and_then(positive)
                    .map(mf::AttnWindow::new),
                mf::Rope::new(
                    match rope_proportional {
                        true => mf::RopePairing::Proportional,
                        false => mf::RopePairing::Standard,
                    },
                    match uses_rotary {
                        true => mf::KvRope::OnRead,
                        false => mf::KvRope::AtWrite,
                    },
                    model.mrope_section.map(mf::MropeSection::new),
                ),
            )
        });
        let vision = positive(vision_num_heads).map(|heads| {
            mf::Vision::new(
                mf::QueryHeads::new(heads),
                mf::HeadDim::new(vision_head_dim),
                mf::InFeatures::new(vision_in_features),
                match vision_rope_interleaved {
                    true => mf::VisionRopePairing::Interleaved,
                    false => mf::VisionRopePairing::RotateHalf,
                },
                mf::AttnScale::new(vision_attn_scale),
                positive(vision_patch_grid_side).map(mf::PatchGridSide::new),
                positive(vision_pool_kernel).map(mf::PoolKernel::new),
            )
        });
        let gdn = RecurrentDims::of(model).map(|d| {
            mf::Gdn::new(
                mf::GdnHeads::new(d.num_k_heads),
                mf::GdnHeads::new(d.num_v_heads),
                mf::GdnHeadDim::new(d.head_k_dim),
                mf::GdnHeadDim::new(d.head_v_dim),
                mf::ConvKernel::new(d.conv_kernel),
                mf::ConvDim::new(d.conv_dim as usize),
            )
        });
        let drafter = positive(draft_selector_rank).map(|rank| {
            mf::Drafter::new(
                mf::ConvGroup::new(draft_conv_group),
                mf::SelectorRank::new(rank),
                mf::TokenId::new(draft_mask_token),
            )
        });
        let target = target.map(|width| {
            let (hidden, routing) = (
                mf::HiddenSize::new(target_hidden_width),
                match target_routes_experts {
                    true => mf::TargetRouting::Experts,
                    false => mf::TargetRouting::Dense,
                },
            );
            match width {
                mf::TargetWidth::Own => mf::Target::new(hidden, routing),
                mf::TargetWidth::Lent => mf::Target::lent(hidden, routing),
            }
        });
        // The layers holding KV, when a proper subset: leaked once, in this build-script process,
        // so the bake's facts hold the same `'static` table the emitted const states.
        let kv_layers = (kv_layers.proper_subset(num_hidden_layers))
            .map(|layers| mf::KvLayers::new(Vec::leak(layers.to_vec())));
        mf::ModelFacts::new(mf::FactsParts {
            // No arch overrides the metal element dtype.
            metal_dtype: mf::MetalDtype::Bf16,
            scale_dtype: match scale_dtype_is_bf16 {
                true => mf::ScaleDtype::Bf16,
                false => mf::ScaleDtype::F16,
            },
            hidden: mf::HiddenSize::new(hidden_size_for_const),
            intermediate: (intermediate_size > 0)
                .then(|| mf::IntermediateSize::new(intermediate_size)),
            vocab: (vocab_size_for_const > 0).then(|| mf::VocabSize::new(vocab_size_for_const)),
            eps: mf::RmsNormEps::new(rms_norm_eps(model)),
            norm_offset: mf::NormWeightOffset::new(norm_weight_runtime_offset(model)),
            final_softcap: (final_logit_softcapping != 0.0)
                .then(|| mf::SoftCap::new(final_logit_softcapping)),
            kv_codec,
            kv_layers,
            attention,
            vision,
            gdn,
            drafter,
            spec_drafts: positive(spec_drafts).map(mf::NumDrafts::new),
            capture_slots: positive(draft_capture_slots).map(mf::CaptureSlots::new),
            target,
        })
    };
    (
        facts,
        quote! {
            const KV_LORA_RANK: usize = #kv_lora_rank_lit;
            const QK_NOPE_HEAD_DIM: usize = #qk_nope_head_dim_lit;
            const QK_ROPE_HEAD_DIM: usize = #qk_rope_head_dim_lit;
            const V_HEAD_DIM: usize = #v_head_dim_lit;
            const QK_HEAD_DIM: usize = #qk_head_dim_lit;
            const MLA_ATTN_SCALE: f32 = #mla_attn_scale_lit;
        },
    )
}

/// A canonical's `CanonicalParams` impl: its `facts` serialized as `FACTS`, and `mla_consts`.
fn emit_canonical_params_impl(
    facts: scratchy_ir::model_facts::ModelFacts,
    mla_consts: &TokenStream,
) -> TokenStream {
    let facts_tokens = model_facts_tokens(facts);
    quote! {
        // `CanonicalParams` is backend-agnostic — the trait, its
        // associated `const`s, and every callsite (`<W as
        // CanonicalParams>::HEAD_DIM`) live in `scratchy-forward-compiler` with
        // no backend gating. So this impl applies under every
        // backend feature.
        impl ::scratchy_forward_compiler::CanonicalParams for Weights {
            const FACTS: ::scratchy_forward_compiler::model_facts::ModelFacts = #facts_tokens;
            // The MLA dims: no fact covers them yet (no metal or spyre path reads them).
            #mla_consts
        }
    }
}

/// The one-slot-per-sequence recurrent-state geometry of a hybrid arch's
/// recurrent mixer, in the GDN state pool's terms — read from the config
/// once for both the `W::GDN_*` consts and the `GdnRuntimeConfig`.
///
/// A Mamba-2 mixer fills the same pool: its conv ring spans `xBC`
/// (`mamba_conv_dim`), its groups are the key heads (`n_groups`), its heads
/// the value heads (`mamba_num_heads`), each holding a `[head_dim, state]`
/// state matrix — `head_v_dim = mamba_head_dim`, `head_k_dim =
/// ssm_state_size`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RecurrentDims {
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
    pub conv_kernel: u32,
    pub conv_dim: u32,
}

impl RecurrentDims {
    /// `None` for an arch with no recurrent mixer.
    pub(crate) fn of(model: &ModelParams) -> Option<Self> {
        let get = |k: &str| model.bounds.get(k).map(|&v| v as u32);
        let gdn = || {
            Some(Self {
                num_k_heads: get("linear_num_key_heads")?,
                num_v_heads: get("linear_num_value_heads")?,
                head_k_dim: get("linear_key_head_dim")?,
                head_v_dim: get("linear_value_head_dim")?,
                conv_kernel: get("linear_conv_kernel_dim")?,
                conv_dim: get("gdn_conv_dim")?,
            })
        };
        let mamba = || {
            Some(Self {
                num_k_heads: get("n_groups")?,
                num_v_heads: get("mamba_num_heads")?,
                head_k_dim: get("ssm_state_size")?,
                head_v_dim: get("mamba_head_dim")?,
                conv_kernel: get("conv_kernel")?,
                conv_dim: get("mamba_conv_dim")?,
            })
        };
        gdn().or_else(mamba)
    }
}

/// Emit a per-variant arm body — the `GdnRuntimeConfig` literal for one
/// specialization. Returns `None` if the variant has no recurrent mixer
/// (`OpKind::GatedDeltaNet` / `OpKind::Mamba2`) — caller emits a `None` arm.
pub fn emit_gdn_runtime_config_arm_body(
    fuf: &Fuf,
    model: &ModelParams,
) -> Option<proc_macro2::TokenStream> {
    let num_hidden_layers = model.text.layers.get();
    let mut linear = vec![false; num_hidden_layers];
    let mut is_hybrid = false;
    for node in &fuf.nodes {
        if !matches!(node.op, OpKind::GatedDeltaNet | OpKind::Mamba2) {
            continue;
        }
        is_hybrid = true;
        for inp in &node.inputs {
            if let FufInput::Weight { index: Some(l), .. } = inp {
                let l = l.0 as usize;
                if l < num_hidden_layers {
                    linear[l] = true;
                }
            }
        }
    }
    if !is_hybrid {
        return None;
    }
    let RecurrentDims {
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        conv_kernel,
        conv_dim,
    } = RecurrentDims::of(model).unwrap_or_else(|| {
        panic!(
            "model `{}` runs a recurrent mixer but its config declares no GDN \
             (`linear_*`) or Mamba-2 (`mamba_*`, `n_groups`, `ssm_state_size`, \
             `conv_kernel`) state geometry",
            model.source_stem
        )
    });
    let spec_drafts = u8::try_from(*model.bounds.get("spec_drafts").unwrap_or(&0))
        .expect("spec_drafts: a slot's checkpoints count in a u8");
    let bits = linear.iter().copied();
    Some(quote! {
        ::core::option::Option::Some(
            ::scratchy_forward_compiler::gdn_state_layout::GdnRuntimeConfig {
                conv_dim: #conv_dim,
                conv_kernel: #conv_kernel,
                num_k_heads: #num_k_heads,
                num_v_heads: #num_v_heads,
                head_k_dim: #head_k_dim,
                head_v_dim: #head_v_dim,
                linear_layers: ::std::vec![ #(#bits),* ],
                spec_drafts: #spec_drafts,
            },
        )
    })
}

/// Emit the body for the per-arch `ScratchyWeights::per_layer_kv_token_elems`
/// override — `Some(vec![kv_heads_i * head_dim_i; num_layers])` for
/// hybrid-attention-geometry arches whose sliding and global classes
/// differ in dims (Gemma4: sliding 8×256, global 1×512).
///
/// The per-layer class mask is read straight off the unrolled FUF: every
/// `OpKind::SlidingAttention` tile carries its layer as the KvCache
/// extern index, so the mask is exactly the per-layer dispatch the
/// `#[forward]` `if` resolved at unroll time (IR-driven, not re-derived).
/// Returns `None` (trait default, uniform pool) when the body has no
/// sliding tiles OR when both classes share the same dims (Gemma2/3).
pub fn emit_per_layer_kv_token_elems_arm_body(
    fuf: &Fuf,
    model: &ModelParams,
) -> Option<proc_macro2::TokenStream> {
    let num_hidden_layers = model.text.layers.get();
    let mut sliding = vec![false; num_hidden_layers];
    let mut has_sliding = false;
    for node in &fuf.nodes {
        if node.op != OpKind::SlidingAttention {
            continue;
        }
        has_sliding = true;
        for inp in &node.inputs {
            if let FufInput::Extern {
                kind: crate::classified::ExternKind::KvCache,
                index: Some(l),
            } = inp
            {
                let l = l.0 as usize;
                if l < num_hidden_layers {
                    sliding[l] = true;
                }
            }
        }
    }
    if !has_sliding {
        return None;
    }
    let head_dim = model.text.head_dim.get() as usize;
    let num_kv = *model.bounds.get("num_key_value_heads").unwrap_or(&0) as usize;
    let g_hd = model
        .bounds
        .get("global_head_dim")
        .map(|&v| v as usize)
        .unwrap_or(head_dim);
    let g_kv = model
        .bounds
        .get("num_global_key_value_heads")
        .map(|&v| v as usize)
        .unwrap_or(num_kv);
    if g_hd == head_dim && g_kv == num_kv {
        // Sliding/global classes share dims (Gemma2/3) — uniform pool.
        return None;
    }
    let elems = sliding
        .iter()
        .map(|&is_sliding| {
            if is_sliding {
                num_kv * head_dim
            } else {
                g_kv * g_hd
            }
        })
        .map(proc_macro2::Literal::usize_unsuffixed);
    Some(quote! {
        ::core::option::Option::Some(::std::vec![ #(#elems),* ])
    })
}

/// The lanes a DFlash2 drafter's block draft runs in a step, each its own `spec_verify_rows` block:
/// Splash's SPLASH_MAXIMUM_BATCH_WIDTH. Its buckets through `lanes × spec_verify_rows` rows play the
/// block draft. The engine's lane cap (`scratchy_serving_engine::spec_decode::BLOCK_LANES`) is
/// asserted equal where both are in scope (`scratchy_forward_compiler::DFLASH2_BLOCK_LANES`).
const DFLASH2_BLOCK_LANES: u64 = scratchy_forward_compiler::DFLASH2_BLOCK_LANES as u64;

#[allow(clippy::too_many_arguments)]
pub fn emit_model(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    workloads: &WorkloadPoints,
    manifest: &crate::weights_manifest::WeightsManifest,
    // The arch's traced shapes — threaded only so the PD-wavefront
    // macro-emission (`dump_wavefront_mega`) can resolve weight shapes;
    // unused on the normal codegen path.
    inferred: &crate::shape::Inferred,
    canonical_override: Option<&Ident>,
    tp_world_size: u8,
    emit_fingerprint: bool,
    // The canonical's affine rows, for a shim whose own accessor set
    // yields none (a shim lowers no tape, so it has no accessors) — its
    // delegated load reads the canonical's widths, so its fingerprint
    // gate must too. `None` for canonicals themselves.
    canonical_affine_rows: Option<&[(String, u32, u32)]>,
    // The build's own output directory for a `superdsc-dump` build (`compile_carrier`'s caller passes
    // it); `None` writes no dump.
    dump_dir: Option<&std::path::Path>,
    // The target this build compiles for: what it bakes per model is its own.
    emitter: &dyn crate::target::TargetEmitter,
) -> (TokenStream, Vec<(String, u32, u32)>) {
    // A head reading its target's KV (`reads_target_kv`) binds its target's TurboQuant runtime.
    let reads_target_kv = model.arch.reads_target_kv;
    // Vision encoders have no terminal `gemm(<tile>, lm_head)`; the
    // entire FUF is the backbone. The `BackboneLayout::Encoder` arm
    // (which already covers text-side encoders like ModernBERT)
    // handles this naturally because `backbone_layout` reports
    // Encoder when the terminal isn't `gemm(_, lm_head)`. No
    // separate vision flag needed.

    if let Some(canonical) = canonical_override {
        return emit_shim_model(
            program,
            fuf,
            model,
            manifest,
            canonical,
            tp_world_size,
            emit_fingerprint,
            canonical_affine_rows,
            emitter,
        );
    }
    // The struct emitter. Metal calls it twice: once with no accessors
    // and — after the front-end swap has produced them — once with the
    // tape's. Returns the emitted tokens alongside the affine rows its
    // fingerprint gate carries (the tape call's rows are what survive
    // — `weights_affine_rows` is overwritten alongside `weights`).
    let emit_struct = |tape_accs: Option<&[WeightAccessor]>, dims: &ModelDims| {
        emit_weights_struct(
            program,
            fuf,
            tape_accs,
            dims,
            model,
            manifest,
            WeightsEmitMode::Canonical,
            tp_world_size,
            emit_fingerprint,
            None,
            emitter,
        )
    };
    #[cfg(feature = "metal")]
    let mut tape_accessors_for_struct: Option<Vec<WeightAccessor>> = None;
    // The fields' typed handles, from the tape's accessor set: the struct holds them, and the
    // metal resolver reaches each through its source slot.
    // Its `__dims` declares every bound its tile outputs name too: the arena's extents' markers.
    let model_dims = |accs: Option<&[WeightAccessor]>| {
        let mut dims = accs.map_or_else(ModelDims::default, |accs| {
            // Each field's codes as its load plan keeps them: the very plan its load is emitted
            // from.
            let embed_affine = embed_is_affine_repr(accs);
            let codes = |a: &WeightAccessor| {
                let plan = plan_field_load(a, program, fuf, model, manifest, embed_affine);
                packed_affine_codes(&plan)
            };
            // The bundles its ops read whole, as the bundle loaders' dims derive them.
            let config = crate::bundles::BundleConfig {
                bounds: &model.bounds,
                activation: &model.expert_activation,
            };
            let bundles = crate::bundles::derive(program, &config)
                .unwrap_or_else(|e| panic!("`{}`: its bundles: {e}", model.source_stem));
            let groups = group_accessors_by_base(accs);
            ModelDims::of(&groups, program, inferred, &codes, &bundles).unwrap_or_else(|e| {
                panic!("[weights] {}: its weights' types: {e}", model.source_stem)
            })
        });
        dims.declare_tile_bounds(fuf, model);
        dims
    };
    // ⭐ THE MODEL'S FACTS (#1083), computed ONCE here and threaded to every consumer below:
    // the bridge calls (the spyre weight struct, metal's tape fingerprint, metal's canonical
    // lowering) and the per-canonical `CanonicalParams` impl. Every fact is a field of this
    // one value; nothing below re-reads a config or a bound for a fact it states.
    let (model_facts, mla_consts) = fill_canonical_params(
        model,
        tp_world_size,
        fuf_uses_rotary(fuf),
        &KvCacheLayers::of(fuf),
        fuf_reads_target_hidden(fuf).then(|| target_width(program, inferred)),
    );
    // The config params and the typed facts are ONE model — the bridge takes them together.
    #[cfg(feature = "metal")]
    let bridge_model = crate::to_wavefront::BridgeModel {
        params: model,
        facts: &model_facts,
    };

    // THE MODEL'S WEIGHTS STRUCT. A target that derives it itself (spyre: from its decode tape's
    // weight bindings) hands it back; otherwise the accessor-less first pass, which the tape
    // re-emits below once the front-end swap has produced its accessors.
    let (weights, weights_affine_rows) = {
        let decode_point = workloads
            .iter()
            .find(|w| w.num_tokens == 1)
            .or_else(|| workloads.iter().next())
            .copied()
            .map(|wp| {
                (
                    bounds_for_wp(model, wp, tp_world_size),
                    wp.num_tokens as u32,
                )
            });
        let struct_of = |accs: Option<&[WeightAccessor]>| emit_struct(accs, &model_dims(accs));
        emitter
            .weights_struct(&crate::target::StructCx {
                program,
                model,
                facts: &model_facts,
                fuf,
                inferred,
                decode_point: decode_point.as_ref().map(|(b, m)| (b, *m)),
                struct_of: &struct_of,
            })
            .unwrap_or_else(|| struct_of(None))
    };

    // Group workload points by TAPE fingerprint. Buckets that lower to
    // the same tape produce byte-identical fn bodies, so the canonical
    // is lowered ONCE and duplicates delegate to it through the bucket
    // table. Public API (every `forward_m_<M>[_sk_<SK>]` /
    // `forward_backbone_m_<M>[_sk_<SK>]` name a user might take a
    // fn-pointer to) is preserved. Dedup runs over `(num_tokens,
    // sk_bucket)` 2-D points.
    //
    // Spyre folds every point onto the FIRST point: its per-bucket host
    // tapes are all empty (the model runs from the embedded KTIR bundle).
    let bucket_points: Vec<WorkloadPoint> = workloads.iter().copied().collect();
    #[cfg(not(feature = "metal"))]
    let bucket_canonical: Vec<WorkloadPoint> =
        bucket_points.iter().map(|_| bucket_points[0]).collect();
    #[cfg(feature = "metal")]
    let bucket_canonical: Vec<WorkloadPoint> = {
        let mut tape_to_canonical: HashMap<String, WorkloadPoint> = HashMap::new();
        let mut bucket_canonical = Vec::with_capacity(bucket_points.len());
        for wp in &bucket_points {
            let bounds = bounds_for_wp(model, *wp, tp_world_size);
            let fp = match crate::to_wavefront::lower_decode_to_wavefront(
                fuf,
                inferred,
                &bounds,
                &bridge_model,
                crate::to_wavefront::PrefixCapacity::new(std::num::NonZeroU32::new(8192).unwrap()),
                wp.num_tokens as u32,
            ) {
                Ok(l) => {
                    let m = wp.num_tokens as u32;
                    // A canonical realizes its codec guards at its own rows
                    // (`metal_guard_gates`), for every bucket it stands for: buckets whose
                    // realizations differ cannot share one (`lower_canonical` refuses it).
                    let guards = scratchy_target_metal::op_abi::metal_guard_gates(m);
                    let ops = l
                        .input
                        .ops
                        .iter()
                        .map(|od| {
                            // Row counts that track the bucket's `m`
                            // normalize to their EMISSION CLASS (decode
                            // m=1 vs prefill m>1 select different
                            // attention forms); fixed row counts stay
                            // verbatim. Normalizing to one sentinel
                            // collapsed m=1 with m=2 — the drift gate
                            // caught it against the solve's folding.
                            // The metal fold pass folds matvec ends on
                            // every bucket in the qmv matvec band (its
                            // floor: `QMV_MATVEC_BAND_ROWS`) and not
                            // above, so the classes split there too —
                            // a shared canonical would bake one band's
                            // steps at the other's bucket.
                            let band = scratchy_target_metal::tape::quantized::QMV_MATVEC_BAND_ROWS;
                            let mm = if od.m == m {
                                if m == 1 {
                                    u32::MAX
                                } else if m <= band {
                                    u32::MAX - 1
                                } else {
                                    u32::MAX - 2
                                }
                            } else {
                                od.m
                            };
                            format!("{:?}#{mm}#{:?};", od.op, od.inputs)
                        })
                        .collect::<String>();
                    format!("{guards:?}{ops}")
                }
                Err(e) => format!("REFUSED:{e:?}"),
            };
            bucket_canonical.push(*tape_to_canonical.entry(fp).or_insert(*wp));
        }
        bucket_canonical
    };

    // Lower every canonical tape_index once, backbone-shaped: skip the
    // terminal subgraph (the `gemm(<final_norm>, lm_head)` row) and
    // emit it separately as a tiny LM_HEAD slice. forward and
    // forward_backbone share the backbone slice; forward additionally
    // runs LM_HEAD; forward_backbone DtoD-copies the backbone-output
    // slot. No more pair of near-identical full slices per tape_index.
    let mut canonical_lowered: BTreeMap<
        WorkloadPoint,
        (
            CanonicalLowered,
            /* num_slots */ u32,
            /* backbone_slot */ u32,
            /* terminal_slot */ u32,
            /* slots: shared coloring across every tape_index in this
             * canonical's group. Per-tape_index arena_bytes (below in
             * the metal emission loop) computes its OWN sizes from
             * this map + that tape_index's bounds, so each tape_index gets
             * a tight arena sized for its own num_tokens. */
            crate::weight_vocab::SlotMap,
        ),
    > = BTreeMap::new();
    // `last_node_id` must be the lm_head Gemm, not one of the post-Embed `MmEmbedSplice` nodes
    // that sit at fuf-array-tail but semantically belong near the Embed; walking past them with
    // `last_non_splice_node` recovers the real terminal. The metal bucket facts walk the logits
    // chain from it (#784).
    let last_node_id = last_non_splice_node(fuf)
        .expect("non-empty FUF expected")
        .id;
    let layout = backbone_layout(fuf, program);
    // The canonical tuple starts as an empty placeholder and the
    // front-end swap below fills every field from the shared tape (a
    // swap that then refuses panics loudly there — an empty stream
    // cannot ship silently because the bucket table still reads these
    // fields).
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        canonical_lowered.insert(
            *wp,
            (
                CanonicalLowered {
                    backbone: crate::interpreter_codegen::LoweredBucket {
                        instances: Vec::new(),
                        weight_slots: Vec::new(),
                        barriers: Vec::new(),
                    },
                    lm_head: crate::interpreter_codegen::LoweredBucket {
                        instances: Vec::new(),
                        weight_slots: Vec::new(),
                        barriers: Vec::new(),
                    },
                },
                0,
                0,
                0,
                crate::weight_vocab::SlotMap::new(),
            ),
        );
    }

    // Per-canonical CanonicalParams impl + Instruction type alias.
    // The alias keeps every static-slice row short instead of
    // repeating `::scratchy_forward_compiler::Instruction::<Weights>::Variant(…)`.
    // One value, two consumers: the impl's `FACTS` (serialized) and the metal bake — the
    // `model_facts` hoisted above the bridge calls, computed once for both.
    let canonical_params_impl = emit_canonical_params_impl(model_facts, &mla_consts);
    // Per-canonical: alias the generic `Instruction<Weights>` for
    // the slice element type AND glob-import the variant
    // constructors so each static-slice row reads `Embed(...)` /
    // `RmsNorm(...)` instead of
    // `::scratchy_forward_compiler::Instruction::<Weights>::Embed(...)`.
    // The `__I` alias and the variant glob-import are backend-agnostic
    // — `Instruction<W>` is defined in `scratchy-forward-compiler` without a
    // backend gate. The per-tape_index static slices reference both, so
    // they compile under every backend feature.
    let instruction_alias = quote! {
        #[allow(non_camel_case_types, dead_code)]
        type __I = ::scratchy_forward_compiler::Instruction;
        use ::scratchy_forward_compiler::Instruction::*;
    };

    // Static slices: BACKBONE_M_<wp> + LM_HEAD_M_<wp> per CANONICAL
    // tape_index only. Non-canonical buckets share their canonical
    // sibling's slices via the FORWARD_TABLE entries below.
    let mut static_slices: Vec<TokenStream> = Vec::new();
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        let (lowered, _, _, _, _) = &canonical_lowered[wp];
        let backbone_static_ident = bucket_static_ident("BACKBONE_M", *wp);
        let lm_head_static_ident = bucket_static_ident("LM_HEAD_M", *wp);
        static_slices.push(emit_bucket_static_slice(
            &backbone_static_ident,
            &lowered.backbone.instances,
        ));
        static_slices.push(emit_bucket_static_slice(
            &lm_head_static_ident,
            &lowered.lm_head.instances,
        ));
        // Metal MTL4 barrier flags, computed at FUF/SlotMap level
        // (one bool per `OpInstance`). Length
        // tracks the corresponding instructions static. Runtime
        // consumes via `MetalBucketSpec.{backbone,lm_head}_barriers`.
        // The dataflow helper lives under `crate::metal`, which is
        // itself `#[cfg(feature = "metal")]`, so the call sites must
        // match — without the metal feature these statics aren't
        // referenced (METAL_BUCKETS is also metal-gated).
        #[cfg(feature = "metal")]
        {
            let bb_barriers_ident = bucket_static_ident("BACKBONE_BARRIERS_M", *wp);
            let lh_barriers_ident = bucket_static_ident("LM_HEAD_BARRIERS_M", *wp);
            static_slices.push(
                scratchy_target_metal_compiler::static_tape::emit_bucket_barriers_static(
                    &bb_barriers_ident,
                    &lowered.backbone.barriers,
                ),
            );
            static_slices.push(
                scratchy_target_metal_compiler::static_tape::emit_bucket_barriers_static(
                    &lh_barriers_ident,
                    &lowered.lm_head.barriers,
                ),
            );
        }
    }

    // `sk_axis_active`: true when the model declared `sk_buckets`;
    // otherwise every wp has sk_bucket==0 and the table emits
    // `[0, u64::MAX)` for sk on every entry.
    let sk_axis_active = workloads.iter().any(|wp| wp.sk_bucket != 0);
    let num_tokens_points: Vec<u64> = workloads.num_tokens_points();
    let mut sk_by_m: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for wp in workloads.iter() {
        sk_by_m.entry(wp.num_tokens).or_default().push(wp.sk_bucket);
    }
    for v in sk_by_m.values_mut() {
        v.sort();
        v.dedup();
    }

    // (m_min, m_max_excl, sk_min, sk_max_excl) for each workload.
    // M=1 is special-cased: the solver may pick M=1-only kernels
    // (cutlass_gemv) that fail at M>1, so the next tape_index starts at
    // M=2 even if the configured points list `[1, 8, ...]`.
    let m_max_excl_for = |m_idx: usize| -> u64 {
        if m_idx + 1 == num_tokens_points.len() {
            u64::MAX
        } else {
            num_tokens_points[m_idx + 1]
        }
    };
    let m_min_for = |m_idx: usize, m: u64| -> u64 {
        if m_idx > 0 && num_tokens_points[0] == 1 && num_tokens_points[m_idx - 1] == 1 {
            2
        } else {
            m
        }
    };
    let m_idx_of: HashMap<u64, usize> = num_tokens_points
        .iter()
        .enumerate()
        .map(|(i, &m)| (m, i))
        .collect();
    // Bucket-id assignment for the per-arch `WeightAccessors` impl:
    // canonical_lowered.iter() ordering pairs each canonical with
    // tape_index ids `(2*ci, 2*ci+1)` for backbone and lm_head. This
    // mapping must match the iteration order in
    // `emit_weight_accessors_impl` so the match-arm keys line up.
    let canonical_to_bucket_id: HashMap<_, u32> = canonical_lowered
        .iter()
        .enumerate()
        .map(|(ci, (wp, _))| (*wp, (ci as u32) * 2))
        .collect();

    // PD-wavefront macro-emission (env-gated, diagnostic + the const builder).
    // HERE — not in the pre-emit drive — because the decode bucket's
    // `weight_slots` (the real weight-locator source) only exists post-lowering
    // + post-loop-compression, and the `(bucket, op_idx, slot)` triples must
    // match the very match table `emit_weight_accessors_impl` builds from this
    // same `canonical_lowered`.
    // Wavefront-emitting builds only — spyre lowers the decode to a single
    // fused KTIR/SuperDSC bundle. Plain metal dispatches per-op and skips
    // this; the expensive lowering + its hard panic on un-lowerable models
    // must not run for them.

    // THE METAL FRONT END. Each decode canonical is lowered from the SHARED subtile tape by
    // `scratchy_target_metal_compiler::canonical::lower_canonical` — metal's orchestration lives in
    // the target's compiler crate. This computes the facts only the macro can (the weight table's
    // paths, the embed's storage, the MLP form, the per-module MoE widths), makes the one call, and
    // writes back what the bucket table, the arena statics, the accessor table and the bake read.
    #[cfg(feature = "metal")]
    let mut metal_steps: BTreeMap<
        WorkloadPoint,
        scratchy_target_metal_compiler::canonical::ClassedCanonical,
    > = BTreeMap::new();
    // THE MODEL'S SOURCE MANIFEST: every weight family its tapes bind, interned as the canonicals
    // lower, plus the class rotary tables rope-on-read binds (the same caches the Weights struct
    // declares: `rotary`, and `rotary_local` on a dual-rotary arch).
    #[cfg(feature = "metal")]
    let mut metal_sources = scratchy_target_metal_compiler::canonical::SourceManifest::default();
    #[cfg(feature = "metal")]
    let metal_rotary = fuf_uses_rotary(fuf).then(|| {
        let local = fuf.nodes.iter().any(|n| {
            n.inputs.iter().any(|i| {
                matches!(
                    i,
                    crate::fuf::FufInput::Extern {
                        kind: crate::classified::ExternKind::RotaryLocal,
                        ..
                    }
                )
            })
        });
        metal_sources.rotary_tables(local)
    });
    #[cfg(feature = "metal")]
    for (canonical, (_, ns_field, bb_slot_field, term_field, slots_dec)) in
        canonical_lowered.iter_mut()
    {
        use crate::weight_bindings::{EmbedStorage, MlpPacking, WeightAbi};
        use scratchy_target_metal::tape::step as st;
        use scratchy_target_metal_compiler::canonical as mc3;
        let (m, stem) = (canonical.num_tokens, &model.source_stem);
        let bounds = bounds_for_wp(model, *canonical, tp_world_size);
        let prefix =
            crate::to_wavefront::PrefixCapacity::new(std::num::NonZeroU32::new(8192).unwrap());
        let l = match crate::to_wavefront::lower_decode_to_wavefront(
            fuf,
            inferred,
            &bounds,
            &bridge_model,
            prefix,
            m as u32,
        ) {
            Ok(l) => l,
            // METAL HAS NO KERNEL (dense MoE) is not a front-end gap: the canonical gets no steps,
            // so the bake bakes NO variants and the pool refuses at load. Anything else has no
            // fallback.
            Err(e) => {
                assert!(
                    e.is_no_metal_realization(),
                    "[m2-flip] {stem} m={m}: the shared tape refused a DECLARED pilot ({e:?}) — \
                     with the solve off there is no fallback",
                );
                eprintln!("[m2-eq] {stem} m={m} wavefront REFUSED: {e}");
                continue;
            }
        };
        // A preset's widths and groups are the kernels' (`quantization`'s preset check refuses
        // any other), so each mints.
        let affine_group = |g: u32| {
            st::AffineGroupSize::new(g).unwrap_or_else(|| {
                panic!("an affine group of {g}, which no kernel is instantiated for")
            })
        };
        let affine_bits = |b: u32| {
            st::AffineBits::new(b).unwrap_or_else(|| {
                panic!("an affine width of {b} bits, which no kernel is instantiated for")
            })
        };
        let embed_quant = fuf.nodes.iter().find_map(|n| {
            match (n.op, crate::weight_vocab::weight_storage_of(n)) {
                (
                    OpKind::Embed,
                    Some(crate::quantization::StorageFormat::Affine { group_size, bits }),
                ) => Some((affine_group(*group_size), affine_bits(*bits))),
                _ => None,
            }
        });
        let fused_mlp = l.input.ops.iter().all(|od| {
            !matches!(
                od.op,
                scratchy_subtile::subtile_ir::SubOp::MatmulTile {
                    weight: scratchy_subtile::lower::GemmWeight::Affine { .. },
                    ..
                }
            )
        });
        let weight_paths: Vec<Vec<String>> = (program.weights.ids())
            .map(|id| program.weights.path(id).to_vec())
            .collect();
        let embed_base = weight_paths
            .iter()
            .map(|p| p.join("_"))
            .find(|p| p.contains("embed"));
        let embed_base = embed_base.unwrap_or_else(|| "embed_tokens".into());
        // Each DSL weight whose own subtree declares a `shared_expert`, by id: the bank a MoE op
        // reads is looked up by the weight it reads.
        let shared_expert_banks: std::collections::BTreeSet<u32> = (program.weights.ids())
            .filter(|&id| {
                let path = program.weights.path(id);
                let prefix: Vec<&str> = (path.iter().map(String::as_str))
                    .chain(["shared_expert"])
                    .collect();
                program.weights.has_subtree(&prefix)
            })
            .map(|id| id.0)
            .collect();
        let quant = model.quantization.as_ref();
        // A bank's widths, by the weight its op reads: its DSL path is the resolver's prefix.
        let widths = |bank: u32, layer: st::LayerId, body| {
            let qc = quant.expect("MoE widths are read for a quantized model only");
            let at = Some(UnrollIndex(layer.0 as u64));
            let prefix = program.weights.path(WeightId(bank)).join(".");
            let bits =
                crate::quantization::affine_moe_expert_bits(program, &qc.method, &prefix, at, body);
            bits.map_or([None; 3], |b| b.projections().map(|w| w.map(affine_bits)))
        };
        let facts = mc3::MetalStepFacts {
            weight_paths: &weight_paths,
            shared_expert_banks: &shared_expert_banks,
            embed_base: &embed_base,
            embed_quant,
            mlp: if fused_mlp {
                mc3::MlpForm::Packed
            } else {
                mc3::MlpForm::Split
            },
            moe_expert_bits: quant.map(|_| &widths as &mc3::MoeExpertBits<'_>),
        };
        let mc = &model_facts;
        let buckets: Vec<u64> = (bucket_points.iter().zip(&bucket_canonical))
            .filter(|&(_, c)| c == canonical)
            .map(|(wp, _)| wp.num_tokens)
            .collect();
        let at = mc3::CanonicalAt {
            stem,
            m,
            buckets: &buckets,
            rows: match layout {
                BackboneLayout::Decoder => scratchy_subtile::sample_rows::ResultRows::Sampled,
                BackboneLayout::Encoder => scratchy_subtile::sample_rows::ResultRows::Every,
            },
        };
        let lowered = mc3::lower_canonical_classed(&l, &facts, mc, at, &mut metal_sources)
            .unwrap_or_else(|e| panic!("[m2-flip] {stem} m={m}: {e}"));
        // The narrow band's shared bucket facts — the classed lowering proved the wide band
        // keeps every one of them (only the step tapes may differ).
        *ns_field = lowered.narrow.colours.get();
        *bb_slot_field = lowered.narrow.hidden.0;
        *term_field = lowered.narrow.result.index();
        // Arena sizing evaluates a FUF shape per colour, so colour 0 (the embedded hidden) is
        // keyed by the FUF Embed tile.
        let mut arena = lowered.narrow.arena.clone();
        if let Some(et) = fuf.nodes.iter().find(|nd| nd.op == OpKind::Embed) {
            arena.insert_at(et.id, 0, 0);
        }
        *slots_dec = arena;
        metal_steps.insert(*canonical, lowered);
        // THE WEIGHTS STRUCT, from the tape's bindings. Metal's one ABI fact in it: a gate/up
        // pair is ONE packed `__fused__` buffer.
        let abi = WeightAbi {
            mlp: if fused_mlp {
                MlpPacking::Packed
            } else {
                MlpPacking::Split
            },
            embed: match embed_quant {
                Some(_) => EmbedStorage::AffineQuant,
                None => EmbedStorage::Dense,
            },
        };
        let synth = crate::weight_bindings::from_tape(&l, program, abi)
            .unwrap_or_else(|e| panic!("[weight-bindings] {stem}: {e}"));
        tape_accessors_for_struct = Some(synth.accessors);
    }
    // The `(bucket, op_idx, slot)` accessor table. No stream records a slot, so it is the
    // trait's defaults; on metal the model's tensors resolve through the source manifest's
    // resolver instead.
    let weight_accessors_impl =
        emit_weight_accessors_impl(&canonical_lowered, model.source_stem.as_str());
    #[cfg(feature = "metal")]
    let dims = model_dims(tape_accessors_for_struct.as_deref());
    #[cfg(feature = "metal")]
    let resolver = metal_sources
        .resolver(&|base| {
            dims.handle(base).map(|h| {
                let weights = &h.weights;
                (h.ty.clone(), quote!(__weights::#weights))
            })
        })
        .unwrap_or_else(|e| panic!("[metal sources] {}: {e:?}", model.source_stem));
    #[cfg(feature = "metal")]
    let weight_accessors_impl = {
        let resolver = &resolver.tokens;
        quote! {
            #weight_accessors_impl
            #[cfg(scratchy_target = "metal")]
            #resolver
        }
    };
    // The Weights struct is emitted from the TAPE's accessor set.
    // Re-emitted HERE because the tape accessors only exist after the
    // front-end swap has lowered the canonical.
    #[cfg(feature = "metal")]
    let (weights, weights_affine_rows) = match tape_accessors_for_struct.as_ref() {
        Some(tape_accs) => emit_struct(Some(tape_accs), &dims),
        None => (weights, weights_affine_rows),
    };

    // The target's own baked program for this model, at the decode (num_tokens=1) point's
    // bounds — spliced into the per-model module; empty for a target that bakes none this way.
    let target_bake = {
        let decode_wp = bucket_points.iter().find(|wp| wp.num_tokens == 1).copied();
        let decode_bounds = decode_wp.map(|wp| bounds_for_wp(model, wp, tp_world_size));
        emitter.bake_model(&crate::target::BakeCx {
            program,
            model,
            facts: &model_facts,
            fuf,
            inferred,
            decode_bounds: decode_bounds.as_ref(),
            dump_dir,
        })
    };

    let mut bucket_table_entries: Vec<TokenStream> = Vec::new();
    for (i, wp) in bucket_points.iter().enumerate() {
        let canonical = bucket_canonical[i];
        let bb_static = bucket_static_ident("BACKBONE_M", canonical);
        let lm_static = bucket_static_ident("LM_HEAD_M", canonical);
        let bb_bucket_id = *canonical_to_bucket_id
            .get(&canonical)
            .expect("canonical_to_bucket_id missing entry for canonical workload point");
        let bb_bucket_lit = proc_macro2::Literal::u32_unsuffixed(bb_bucket_id);
        let lm_bucket_lit = proc_macro2::Literal::u32_unsuffixed(bb_bucket_id + 1);
        // Per-tape_index slot metadata. The colored slot map is built
        // per workload point (the solver may pick Impls that need
        // different intermediate-tile counts per tape_index — e.g.
        // CutlassGemmAdd fuses the residual into the GEMM at
        // prefill, freeing a slot vs the decode-tape_index's separate
        // Add). The non-canonical buckets share their canonical's
        // slot metadata since they share its static slices.
        let (_, num_slots_b, backbone_slot_b, terminal_slot_b, _) = &canonical_lowered[&canonical];
        let num_slots_lit = proc_macro2::Literal::u32_unsuffixed(*num_slots_b);
        let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(*backbone_slot_b);
        let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(*terminal_slot_b);
        let m_idx = m_idx_of[&wp.num_tokens];
        let m_min = if wp.num_tokens == 1 {
            1
        } else {
            m_min_for(m_idx, wp.num_tokens)
        };
        let m_max_excl = if wp.num_tokens == 1 {
            2
        } else {
            m_max_excl_for(m_idx)
        };
        let (sk_min, sk_max_excl) = if sk_axis_active {
            let sk_buckets = &sk_by_m[&wp.num_tokens];
            let j = sk_buckets
                .iter()
                .position(|&sk| sk == wp.sk_bucket)
                .unwrap();
            // The first sk tape_index per m group must accept sk < its
            // own configured value — old codegen routed this via a
            // `_ =>` fallback arm onto the smallest sk fn. So emit
            // sk_min=0 for j==0 instead of wp.sk_bucket; without
            // this, find_bucket misses on small max_seqlen_k (the
            // prefill case for short prompts) and the fallback to
            // table[0] silently picks an m=1 row.
            let sk_min = if j == 0 { 0 } else { wp.sk_bucket };
            let sk_max_excl = if j + 1 == sk_buckets.len() {
                u64::MAX
            } else {
                sk_buckets[j + 1]
            };
            (sk_min, sk_max_excl)
        } else {
            (0u64, u64::MAX)
        };
        let m_min_lit = proc_macro2::Literal::u64_unsuffixed(m_min);
        let m_max_lit = if m_max_excl == u64::MAX {
            quote! { u64::MAX }
        } else {
            let v = proc_macro2::Literal::u64_unsuffixed(m_max_excl);
            quote! { #v }
        };
        let sk_min_lit = proc_macro2::Literal::u64_unsuffixed(sk_min);
        let sk_max_lit = if sk_max_excl == u64::MAX {
            quote! { u64::MAX }
        } else {
            let v = proc_macro2::Literal::u64_unsuffixed(sk_max_excl);
            quote! { #v }
        };
        bucket_table_entries.push(quote! {
            __B(
                #m_min_lit, #m_max_lit, #sk_min_lit, #sk_max_lit,
                #bb_static, #lm_static,
                #num_slots_lit, #backbone_slot_lit, #terminal_slot_lit,
                #bb_bucket_lit, #lm_bucket_lit,
            ),
        });
    }

    // FORWARD_TABLE — one row per tape_index. `__B` aliases the
    // `BucketEntry` tuple-struct constructor so each row stays on a
    // single line in expanded source. `scr model info`'s per-variant
    // `dump()` walks it; metal's pool uses `METAL_BUCKETS` for the
    // forward and ignores the runtime fields here.
    let forward_table = quote! {
        #[cfg(scratchy_target = "metal")]
        static FORWARD_TABLE: &[::scratchy_forward_compiler::BucketEntry<__I>] = {
            use ::scratchy_forward_compiler::BucketEntry as __B;
            &[
                #(#bucket_table_entries)*
            ]
        };
    };

    // METAL_BUCKETS — one row per distinct `num_tokens` point per routed fold band: the bands
    // colour differently, so each carries its own slots, terminal and arena (twin rows, sharing
    // everything, when the bands fold the same step tape). The
    // metal pool dispatches on `num_tokens` only (no sk axis); for
    // models that declared `sk_buckets` we pick the `sk_bucket == 0`
    // canonical at each `m`, falling back to whichever wp exists for
    // that `m` if the model elided the sk=0 entry. TinyLlama-class
    // models have no sk axis, so this is a 1:1 enumeration of
    // `num_tokens_points`.
    //
    // Each row's `backbone` / `lm_head` ride on the per-canonical
    // static slices the cuda emission already produced — under both
    // backends the slices are unconditionally emitted (the
    // `instruction_alias` and `Weights` definitions are now
    // mutually-exclusive cfg-gated, but the slice statics
    // themselves are backend-agnostic).
    let mut metal_bucket_entries: Vec<TokenStream> = Vec::new();
    // Each row of `METAL_BUCKETS` as its bucket type, in row order: its rows, band and arena
    // (`baked_buckets!`).
    let mut metal_bucket_types: Vec<TokenStream> = Vec::new();
    let mut metal_arena_bytes_statics: Vec<TokenStream> = Vec::new();
    #[cfg(feature = "metal")]
    let mut metal_tape_cmds = scratchy_target_metal_compiler::static_tape::CommandPool::default();
    // The most arena slots a bucket has: how many the model's arena ranks rank (`RankedSlots`).
    let mut metal_arena_slots: u32 = 0;
    // Every colour and arena the buckets' colourings make, in the model's `__arena`.
    let mut metal_arena_types = ArenaTypes::default();
    // Whether the model's tapes read a paged KV pool. One that reads none (Splash's DFlash2
    // drafter, whose rings are its KV) holds any pool's block count at its ladder's floor.
    let pool_block_cap = match KvCacheLayers::of(fuf).any() {
        true => quote! { ctx.kv_cache.max_blocks_per_seq },
        false => quote! { 0 },
    };
    // The KV cap rungs every bucket bakes, topped by every block the model's positions fill.
    #[cfg(feature = "metal")]
    let metal_cap_ladder = {
        use scratchy_target_metal::tape::ids::{BlockSize, MaxPositions};
        let mc = &model_facts;
        let max_pos = KvCacheLayers::of(fuf).any().then(|| {
            let max_pos = model.bounds.get("max_position_embeddings").copied();
            max_pos
                .and_then(|p| u32::try_from(p).ok())
                .map(MaxPositions)
                .unwrap_or_else(|| {
                    panic!(
                        "model `{}`: no u32 max_position_embeddings to top the KV cap ladder",
                        model.source_stem
                    )
                })
        });
        // The pool's page is the base class's; a model without attention reads no pool (its
        // `max_pos` is `None`), and its ladder is the floor whatever the page.
        let page = mc
            .attention()
            .map_or(scratchy_ir::model_facts::DEFAULT_KV_PAGE, |a| {
                a.base().page()
            });
        let ladder = scratchy_target_metal_compiler::static_tape::kv_cap_ladder(
            max_pos,
            BlockSize(page.get()),
        );
        eprintln!(
            "[metal bake] {}: KV cap rungs {ladder:?}",
            model.source_stem
        );
        ladder
    };
    // Splash's two draft graphs (runtime/model/DFlashDraft.cpp `addContextCommit`, `addDecode`)
    // are one tape here, played apart. A DFlash2 drafter's block draft runs a step's lanes'
    // blocks: at most its target's verify rows (`spec_verify_rows`), so at the smallest bucket
    // that holds them (`plays_result_through`). A bucket past it plays the context commit alone,
    // and its arena holds what the commit's tiles (`commit_tiles`) write: the block's — its
    // lm_head's logits, a vocabulary a row, above all — are never written there.
    // The rows are a lane's (`spec_verify_rows`) times the lanes a step drafts
    // (`scratchy_serving_engine::spec_decode::BLOCK_LANES`, Splash's SPLASH_MAXIMUM_BATCH_WIDTH).
    let plays_result_through = match model.bounds.get("selector_rank") {
        Some(&rank) if rank > 0 => {
            (model.bounds.get("spec_verify_rows").copied()).and_then(|rows| {
                let rows = rows * DFLASH2_BLOCK_LANES;
                num_tokens_points
                    .iter()
                    .copied()
                    .filter(|&m| m >= rows)
                    .min()
            })
        }
        _ => None,
    };
    let commit_tiles: HashSet<TileId> = {
        let mut live: HashSet<TileId> = HashSet::new();
        let mut todo: Vec<TileId> = (fuf.nodes.iter())
            .filter(|n| n.op == crate::classified::OpKind::DraftContextKv)
            .map(|n| n.id)
            .collect();
        while let Some(id) = todo.pop() {
            if live.insert(id) {
                todo.extend(fuf.get(id).inputs.iter().filter_map(|i| match *i {
                    crate::fuf::FufInput::Tile { id, .. } => Some(id),
                    _ => None,
                }));
            }
        }
        live
    };
    for &m in &num_tokens_points {
        // Prefer the sk=0 canonical for this `m`; fall back to any wp
        // at `m` if the model never declared sk=0 explicitly.
        let wp = bucket_points
            .iter()
            .find(|wp| wp.num_tokens == m && wp.sk_bucket == 0)
            .copied()
            .or_else(|| bucket_points.iter().find(|wp| wp.num_tokens == m).copied())
            .expect("every num_tokens point has at least one wp");
        let i = bucket_points
            .iter()
            .position(|w| *w == wp)
            .expect("wp came from bucket_points");
        let canonical = bucket_canonical[i];
        let bb_static = bucket_static_ident("BACKBONE_M", canonical);
        let lm_static = bucket_static_ident("LM_HEAD_M", canonical);
        let bb_barriers_static = bucket_static_ident("BACKBONE_BARRIERS_M", canonical);
        let lh_barriers_static = bucket_static_ident("LM_HEAD_BARRIERS_M", canonical);
        let (_, num_slots_b, backbone_slot_b, terminal_slot_b, slots_b) =
            &canonical_lowered[&canonical];
        let bucket_m_lit = proc_macro2::Literal::u32_unsuffixed(m as u32);

        // Per-tape_index arena_bytes: register-coloring tells us which
        // (tile, output_slot) pairs share an arena slot; for THIS
        // tape_index M we evaluate each tile's output shape against this
        // tape_index's bounds and take the per-color max. Each tape_index
        // gets a tight arena sized exactly for its own num_tokens —
        // the worker pool elementwise-maxes across MetalBucketSpec
        // entries once at init time to size the per-worker arena
        // (init-time alloc, reused across every forward).
        let bp_bounds = bounds_for_wp(model, wp, tp_world_size);
        let plays_result = plays_result_through.is_none_or(|through| m <= through);
        // One fold band's bucket facts — its colouring's slot count, terminal, hidden and
        // per-slot bytes; each band's step tape was coloured under its own colouring, so its
        // bucket row carries its own. The logits: the terminal tile and the in-place logit
        // tail under it, back to the lm_head — the terminal slot's chain. A forward leaves only
        // the rows its step samples there, so they take that slot's bytes a row
        // (`MetalBucketSpec::logits_row_bytes`), its rows counted at load; an encoder's
        // terminal rows are every row, as are every row of a backend that samples none (metal
        // alone compacts them). A tile output the coloring did not register (a refused
        // canonical's empty map, an input the tape's arena keys by another tile) is in no
        // slot, so not the terminal's.
        let mut band_facts = |num_slots: u32,
                              terminal_slot: u32,
                              backbone_slot: u32,
                              slots: &crate::weight_vocab::SlotMap|
         -> (u32, u32, u32, u64, (Ident, Vec<String>)) {
            let in_terminal = |id: TileId, slot: u8| {
                (slots.iter()).any(|(at, colour)| at == (id, slot) && colour == terminal_slot)
            };
            let logits_tiles: Vec<(TileId, u8)> = match layout {
                BackboneLayout::Decoder if cfg!(feature = "metal") => {
                    let mut chain = vec![(last_node_id, 0u8)];
                    while let Some(&(tile, _)) = chain.last() {
                        let under = fuf.get(tile).inputs.iter().find_map(|i| match *i {
                            crate::fuf::FufInput::Tile { id, slot } if in_terminal(id, slot) => {
                                Some((id, slot))
                            }
                            _ => None,
                        });
                        match under {
                            Some(under) if !chain.contains(&under) => chain.push(under),
                            _ => break,
                        }
                    }
                    chain
                }
                BackboneLayout::Decoder | BackboneLayout::Encoder => Vec::new(),
            };
            let mut logits_row_bytes: u64 = 0;
            // The extents each colour hosts, each once (by its spelling, at the rows' parameters):
            // those its `Colour::BYTES` is the largest of, and those another size covers — the
            // logits rows (`logits_row_bytes`), the tiles a bucket that plays no result never
            // commits, the embedded hidden colour 0 is (`arena_colours!`).
            let no_slots =
                || vec![BTreeMap::<String, (TokenStream, bool)>::new(); num_slots as usize];
            let (mut sized, mut unsized_) = (no_slots(), no_slots());
            let extent = |shape: &crate::dim::Shape| {
                let at = arena_extent(shape, &quote!(M), &quote!(S))
                    .unwrap_or_else(|e| panic!("[metal arena] {}: m={m}: {e}", model.source_stem));
                (at, computes_over_rows(shape))
            };
            for ((tile_id, out_slot), color) in slots.iter() {
                let node = fuf.get(tile_id);
                let shape = &node.outputs[out_slot as usize];
                let e = extent(shape);
                let value = scratchy_subtile::handoff::ArenaValue::Tile {
                    tile: tile_id.0,
                    out: out_slot,
                };
                metal_arena_types.value(value, &e);
                if !plays_result && !commit_tiles.contains(&tile_id) {
                    unsized_[color as usize].insert(e.0.to_string(), e);
                } else if logits_tiles.contains(&(tile_id, out_slot)) {
                    // f16 = 2 bytes/element; metal kernels are f16-only.
                    let dims = crate::weight_vocab::eval_shape_with(shape, &bp_bounds)
                        .expect("a tile output names a bound the model lacks — codegen invariant");
                    let row: u64 = dims.iter().skip(1).product();
                    logits_row_bytes = logits_row_bytes.max(row.saturating_mul(2));
                    unsized_[color as usize].insert(e.0.to_string(), e);
                } else {
                    sized[color as usize].insert(e.0.to_string(), e);
                }
            }
            let embedded = scratchy_subtile::handoff::ArenaValue::Source(0);
            if let (Ok(shape), Some(zero)) =
                (arena_value_shape(fuf, embedded), unsized_.first_mut())
            {
                let e = extent(shape);
                metal_arena_types.value(embedded, &e);
                zero.insert(e.0.to_string(), e);
            }
            for (sized, unsized_) in sized.iter().zip(&mut unsized_) {
                unsized_.retain(|e, _| !sized.contains_key(e));
            }
            metal_arena_slots = metal_arena_slots.max(num_slots);
            // The bucket's arena: each slot's colour type, and the arena of them.
            let colours: Vec<Ident> = (sized.iter().zip(&unsized_).enumerate())
                .map(|(slot, (sized, unsized_))| {
                    let mut markers = |extents: &BTreeMap<String, (TokenStream, bool)>| {
                        let at = extents.values();
                        at.map(|at| metal_arena_types.extent(at))
                            .collect::<Vec<_>>()
                    };
                    let (sized, unsized_) = (markers(sized), markers(unsized_));
                    metal_arena_types.colour(slot as u32, &sized, &unsized_)
                })
                .collect();
            let arena = metal_arena_types.arena(&colours);
            let colours = colours.iter().map(Ident::to_string).collect::<Vec<_>>();
            (
                num_slots,
                terminal_slot,
                backbone_slot,
                logits_row_bytes,
                (arena, colours),
            )
        };
        // The narrow band's facts: the shared tuple the metal front-end swap wrote (which the
        // FORWARD_TABLE row reads too).
        let narrow = band_facts(*num_slots_b, *terminal_slot_b, *backbone_slot_b, slots_b);
        // The wide band's: its own canonical's when the bands fold differently at this bucket,
        // the narrow's twin when they fold the same (no routing ops, or the fold refused) —
        // either way the pool picks one row per bucket by its device's band.
        // `None`: the narrow's twin, whose arena it shares.
        #[cfg(feature = "metal")]
        let wide = (metal_steps.get(&canonical).and_then(|c| c.wide.as_ref())).map(|wide| {
            band_facts(
                wide.colours.get(),
                wide.result.index(),
                wide.hidden.0,
                &wide.arena,
            )
        });
        #[cfg(not(feature = "metal"))]
        let wide = None;
        // Bake this bucket's tape rungs at expansion — one per device
        // generation, addressing, KV cap and TurboQuant decode heads —
        // every fact baked; the pool picks one. See static_tape.rs.
        #[cfg(feature = "metal")]
        let tapes_static_toks = {
            let mc = &model_facts;
            // A canonical metal has no kernel for (`[m2-eq] … REFUSED`: it has no steps) bakes NO
            // variants, so `MetalWorkerPool::for_buckets` refuses at load (`NoRung`), naming
            // the bucket. Baking its empty step tape instead would build VALID empty-command tapes:
            // the forward would no-op and the tail would index the arena out of bounds.
            let tapes_expr = match metal_steps.get(&canonical) {
                None => {
                    eprintln!(
                        "[metal static tape] {}: bucket_m={m}: front-end REFUSED this canonical — \
                         baking an empty variant list so the pool refuses at load",
                        model.source_stem
                    );
                    quote! { &[] }
                }
                Some(classed) => {
                    use scratchy_target_metal_compiler::static_tape::{BandArena, BucketRows};
                    // Each band's colour types by slot, at the bucket's rows; a wide band that
                    // folds as the narrow one does is its twin.
                    let rows = BucketRows::At {
                        m: m as u32,
                        sk: wp.sk_bucket as u32,
                    };
                    let band_arena = |(_, colours): &(Ident, Vec<String>)| BandArena {
                        colours: colours.clone(),
                        rows,
                    };
                    let narrow_arena = band_arena(&narrow.4);
                    let wide_arena = band_arena(&wide.as_ref().unwrap_or(&narrow).4);
                    let input = scratchy_target_metal_compiler::static_tape::BucketLowerInput {
                        steps: &classed.narrow.steps,
                        wide_band: classed.wide.as_ref().map(|wide| {
                            scratchy_target_metal_compiler::static_tape::WideBand {
                                steps: &wide.steps,
                                num_arena_slots: wide.colours.get(),
                                operands: &wide.operands,
                                arena: &wide_arena,
                            }
                        }),
                        bucket_m: m as u32,
                        num_arena_slots: *num_slots_b,
                        rotary: metal_rotary,
                        cap_ladder: &metal_cap_ladder,
                        operands: &classed.narrow.operands,
                        arena: &narrow_arena,
                        // What each value its steps name is, as its colour hosts it.
                        extents: &metal_arena_types.values,
                    };
                    scratchy_target_metal_compiler::static_tape::bake_bucket_tapes(
                        mc,
                        &input,
                        &mut metal_tape_cmds,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "[metal static tape] {}: bucket_m={m}: {e}",
                            model.source_stem
                        )
                    })
                }
            };
            let ident = bucket_static_ident("METAL_TAPES_M", wp);
            metal_arena_bytes_statics.push(quote! {
                #[cfg(scratchy_target = "metal")]
                static #ident:
                    &[::scratchy_target_metal::tape::lowered::ClassedTape] = #tapes_expr;
            });
            quote! { #ident }
        };
        #[cfg(not(feature = "metal"))]
        let tapes_static_toks = quote! { &[] };
        // The weight bytes a forward of the bucket reads (`MetalStepTape::weight_bytes`).
        #[cfg(feature = "metal")]
        let (fixed, per_row) = (metal_steps.get(&canonical))
            .map(|classed| classed.narrow.steps.weight_bytes())
            .map_or((0, 0), |b| (b.fixed, b.per_row));
        #[cfg(not(feature = "metal"))]
        let (fixed, per_row) = (0u64, 0u64);
        let weight_bytes_toks = {
            let (fixed, per_row) = (
                proc_macro2::Literal::u64_unsuffixed(fixed),
                proc_macro2::Literal::u64_unsuffixed(per_row),
            );
            quote! {
                ::scratchy_target_metal::tape::step::WeightBytes { fixed: #fixed, per_row: #per_row }
            }
        };
        // The model's baked kernels (`CommandPool::into_tokens`).
        let kernels_toks = match cfg!(feature = "metal") {
            true => quote! { METAL_KERNELS },
            false => quote! { &[] },
        };
        // One row per fold band — the wide band's statics under its own names, the tapes and
        // kernels shared: the pick keeps the row whose band the device's class lowers.
        // A wide band that folds as the narrow one does is its twin.
        let band_rows = [
            ("Narrow", &narrow),
            ("Wide", wide.as_ref().unwrap_or(&narrow)),
        ];
        for (band, (num_slots, terminal_slot, backbone_slot, logits_row_bytes, (arena, _))) in
            band_rows
        {
            let num_slots_lit = proc_macro2::Literal::u32_unsuffixed(*num_slots);
            let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(*terminal_slot);
            let backbone_slot_lit = proc_macro2::Literal::u32_unsuffixed(*backbone_slot);
            let logits_row_bytes_lit = proc_macro2::Literal::u64_unsuffixed(*logits_row_bytes);
            // The band row's arena at the bucket's rows: each slot's bytes are its colour's.
            let rows = |n: u64| {
                let path = scratchy_tensors::typed::unsigned_alias_path(n);
                (path.parse::<TokenStream>()).unwrap_or_else(|e| {
                    panic!("[metal arena] {}: rows `{path}`: {e}", model.source_stem)
                })
            };
            let (m_rows, sk_rows) = (rows(m), rows(wp.sk_bucket));
            let arena_bytes = quote! {
                <__arena::#arena<#m_rows, #sk_rows> as ::scratchy_target_metal::tape::typed::Arena>
                    ::BYTES
            };
            let band_ident = proc_macro2::Ident::new(band, proc_macro2::Span::call_site());
            let bucket_type = format_ident!("B{}", metal_bucket_types.len());
            metal_bucket_types.push(quote! {
                #bucket_type = (#m_rows, #band_ident, super::__arena::#arena<#m_rows, #sk_rows>)
            });
            metal_bucket_entries.push(quote! {
                ::scratchy_target_metal::interpreter::metal::MetalBucketSpec {
                    bucket_m: #bucket_m_lit,
                    band: ::scratchy_target_metal::interpreter::metal::FoldBand::#band_ident,
                    num_arena_slots: #num_slots_lit,
                    terminal_slot: #terminal_slot_lit,
                    backbone_slot: #backbone_slot_lit,
                    arena_bytes: #arena_bytes,
                    logits_row_bytes: #logits_row_bytes_lit,
                    backbone: #bb_static,
                    lm_head: #lm_static,
                    backbone_barriers: #bb_barriers_static,
                    lm_head_barriers: #lh_barriers_static,
                    tapes: #tapes_static_toks,
                    kernels: #kernels_toks,
                    plays_result: #plays_result,
                    weight_bytes: #weight_bytes_toks,
                },
            });
        }
    }

    // THE ARENA RANKS: the slots by size at their largest bucket, largest first, ties by slot —
    // the order in which another model whose forwards never overlap this one's shares its arena
    // (`ArenaRanks`): every slot as a guest, all but the buckets' final hidden states as a host.
    // Read off `METAL_BUCKETS` as their `static` compiles (`RankedSlots`): each slot's bytes are
    // its colours'.
    let metal_ranked_slots = proc_macro2::Literal::usize_unsuffixed(metal_arena_slots as usize);
    let metal_arena_types = metal_arena_types.tokens();
    metal_arena_bytes_statics.push(quote! {
        /// The model's arena: the colours and arenas its buckets' colourings make, and its slots
        /// ranked by their bytes.
        #[cfg(scratchy_target = "metal")]
        mod __arena {
            use super::__dims;
            use ::scratchy_tensors::typed::Act;
            #metal_arena_types
            /// [`super::METAL_BUCKETS`]' arena slots, ranked by the bytes their colours host.
            pub(super) static RANKED: ::scratchy_target_metal::interpreter::metal::RankedSlots<
                #metal_ranked_slots,
            > = ::scratchy_target_metal::interpreter::metal::RankedSlots::of(super::METAL_BUCKETS);
        }

        /// The model's buckets: each row of [`METAL_BUCKETS`] as a type, and the enum a forward's
        /// bucket is, chosen at the door (`scratchy_target_metal::interpreter::metal::bucket`).
        #[cfg(scratchy_target = "metal")]
        mod __buckets {
            ::scratchy_target_metal::baked_buckets! {
                pub enum Bucket { #(#metal_bucket_types),* }
            }
        }

        #[cfg(scratchy_target = "metal")]
        impl ::scratchy_target_metal::interpreter::metal::BakedBuckets for Weights {
            type Bucket = __buckets::Bucket;
        }
    });

    // Largest `num_tokens` across all buckets — drives runtime
    // buffer sizing in the metal forward body's `RuntimeFactory`.
    let max_bucket_m_lit = {
        let max_m = num_tokens_points
            .iter()
            .copied()
            .max()
            .expect("at least one num_tokens point per canonical");
        proc_macro2::Literal::u32_unsuffixed(max_m as u32)
    };

    // Vision runtime-extern buffer sizes (bytes), baked from the largest
    // bucket + vision geometry. Zero (→ alloc's 16-byte floor) on
    // non-vision arches (the `vision_*` bounds are absent in text
    // configs). `freqs` is f32 `[total_L, vision_head_dim/2]`; `pixels`
    // is the model dtype (bf16) `[num_tokens, vision_in_features]`.
    // An MTP head's body reads its target's final hidden rows (`RowsExtern::TargetHidden`).
    let reads_target_hidden = fuf_reads_target_hidden(fuf);
    // The width of those rows (the `TargetHidden` extern's): its target's hidden size — which a
    // drafter of its own width (Gemma 4's assistant) reads and exports at, its passes chained at.
    let target_hidden_width: u64 = model
        .bounds
        .get("backbone_hidden_size")
        .or_else(|| model.bounds.get("hidden_size"))
        .copied()
        .unwrap_or(0);
    let (
        vision_freqs_bytes_lit,
        vision_pixels_bytes_lit,
        vision_posemb_bytes_lit,
        target_hidden_bytes_lit,
        mm_embeds_bytes_lit,
        mm_dst_rows_bytes_lit,
        mrope_cos_sin_bytes_lit,
    ) = {
        let max_m = num_tokens_points.iter().copied().max().unwrap_or(0);
        let vis_head_dim = model.bounds.get("vision_head_dim").copied().unwrap_or(0);
        let vis_in_features = model.bounds.get("vision_in_features").copied().unwrap_or(0);
        let vis_embed_dim = model.bounds.get("vision_embed_dim").copied().unwrap_or(0);
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(0);
        let freqs_bytes = max_m * (vis_head_dim / 2) * 4;
        let pixels_bytes = max_m * vis_in_features * 2;
        // pos_embeds: model-dtype (bf16) `[num_tokens, vision_embed_dim]`,
        // host-interpolated and added to the patch-embed output.
        let posemb_bytes = max_m * vis_embed_dim * 2;
        // target_hidden (MTP heads): model-dtype `[num_tokens, target_hidden_width]`, the
        // target's final hidden rows. Zero unless the body reads the extern.
        let target_hidden_bytes = match reads_target_hidden {
            true => max_m * target_hidden_width * 2,
            false => 0,
        };
        // mm splice (text decoder): mm_embeds = bf16 `[max_m, hidden]`;
        // mm_dst_rows = u32 `[max_m]`. Zero on non-MM arches.
        let mm_embeds_bytes = max_m * hidden_size * 2;
        let mm_dst_rows_bytes = max_m * 4;
        // MRoPE cos/sin override (MRoPE text decoders only): model-dtype
        // (f16/bf16 → 2 bytes) `[max_m, rot_dim]`. 0 on 1D-rope arches
        // (the `resolve_bindings` swap never fires, so the buffer is
        // never bound — `alloc`'s 16-byte floor covers the placeholder).
        // `rot_dim = partial_rotary_factor * head_dim` (Qwen3.5: 0.25*256
        // = 64), mirroring the rope-cache builder / `ROT_DIM` fn-const.
        let mrope_cos_sin_bytes = if model.mrope_section.is_some() {
            let head_dim = u64::from(model.text.head_dim.get());
            let rot_dim = model
                .scalars
                .get("partial_rotary_factor")
                .copied()
                .filter(|&f| (f - 1.0).abs() > 1e-9)
                .map(|f| (f * head_dim as f64).round() as u64)
                .unwrap_or(head_dim);
            max_m * rot_dim * 2
        } else {
            0
        };
        (
            proc_macro2::Literal::u64_unsuffixed(freqs_bytes),
            proc_macro2::Literal::u64_unsuffixed(pixels_bytes),
            proc_macro2::Literal::u64_unsuffixed(posemb_bytes),
            proc_macro2::Literal::u64_unsuffixed(target_hidden_bytes),
            proc_macro2::Literal::u64_unsuffixed(mm_embeds_bytes),
            proc_macro2::Literal::u64_unsuffixed(mm_dst_rows_bytes),
            proc_macro2::Literal::u64_unsuffixed(mrope_cos_sin_bytes),
        )
    };
    // The rows a target captures for its DFlash2 drafter (model dtype, `[max_m, slots · hidden]`;
    // 0 on a model no drafter reads), and a drafter's block (its drafts and their anchor).
    let (draft_capture_bytes_lit, draft_rows_lit, captured_rows) = {
        let max_m = num_tokens_points.iter().copied().max().unwrap_or(0);
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(0);
        let slots = (model
            .layer_sets
            .get(crate::draft_capture::DRAFT_CAPTURE_LAYERS))
        .map_or(0, |s| s.members().len() as u64);
        let rows = *model.bounds.get("spec_drafts").unwrap_or(&0) + 1;
        let bytes = max_m * slots * hidden_size * 2;
        // What a step's followup gets of them, from the worker's runtime bindings: they, or none
        // on a model no drafter reads.
        let captured_rows = match bytes > 0 {
            true => (quote! { runtime }, quote! { Some(&*runtime.draft_capture) }),
            false => (quote! { _ }, quote! { None }),
        };
        (
            proc_macro2::Literal::u64_unsuffixed(bytes),
            proc_macro2::Literal::u64_unsuffixed(rows),
            captured_rows,
        )
    };
    let (captured_runtime, captured_rows) = captured_rows;

    // Width of ONE row of the terminal arena slot, baked at
    // macro-expansion time: the metal forward shapes its returned
    // `OwnedTensor` as `[num_tokens, width]` over that slot, and the
    // argmax kernel strides by it.
    //
    // Decoder layouts terminate in the lm_head Gemm, so the width is
    // `vocab_size`. ENCODER layouts (ModernBERT and every
    // `#[vision_forward]` body) have no lm_head — the terminal slot
    // holds hidden states, so the width is `hidden_size`. Using vocab
    // there would stride rows by ~65x the real row and read past the
    // slot. Vision-only encoders carry no vocab at all and would get
    // 0. Cuda builds skip the constant entirely.
    let logits_width = match layout {
        BackboneLayout::Decoder => model.bounds.get("vocab_size").copied().unwrap_or(0),
        BackboneLayout::Encoder => model.bounds.get("hidden_size").copied().unwrap_or(0),
    };
    let vocab_size_lit = proc_macro2::Literal::u64_unsuffixed(logits_width);
    // The rows of the terminal slot a forward leaves for its caller: a decoder's logits of the
    // rows its step samples, at the front; an encoder's hidden state at every row.
    let terminal_rows = match layout {
        BackboneLayout::Decoder => quote! { inputs.sampled_rows.len() },
        BackboneLayout::Encoder => quote! { n },
    };

    // Per-forward MRoPE cos/sin override builder, spliced into the metal
    // `forward` / `forward_with_metal_followup` bodies just before
    // `ForwardInputs`. Only MRoPE text decoders (config carries
    // `rope_scaling.mrope_section`) build the per-token band-split table
    // and switch the rope kernel to identity positions (option (b)):
    // `positions[t] = t` indexes row `t` of the table, so the unmodified
    // 1D rope kernel reads the right (T/H/W band-split) cos/sin without an
    // in-kernel band-split. The worker's `resolve_bindings` redirects the
    // baked `WeightBundleKind::CosSin` pointer to `runtime.mrope_cos_sin`
    // under the same `W::MROPE_SECTION` gate. Every 1D-rope arch keeps
    // `positions` untouched and passes `mrope_cos_sin: None`. The `[3, n]`
    // vs broadcast `[n]` positions are disambiguated at runtime by
    // `ctx.positions` numel (the worker uploads `[3, n]` only when image
    // tokens are present).
    // MRoPE (Qwen3.5-VL): the per-token rotary cos/sin rows a forward reads in place of the
    // position-indexed cache, built on the host from its positions (`metal_rope_rows`); `None` on
    // 1D-rope arches, whose rope reads the positions.
    let metal_rope_rows_fn: proc_macro2::TokenStream = match model.mrope_section {
        Some([t, h, w]) => {
            let head_dim = u64::from(model.text.head_dim.get());
            let rot_dim_val = model
                .scalars
                .get("partial_rotary_factor")
                .copied()
                .filter(|&f| (f - 1.0).abs() > 1e-9)
                .map(|f| (f * head_dim as f64).round() as u32)
                .unwrap_or(head_dim as u32);
            let rope_theta_val: f64 = model.text.rope_theta.get();
            let rope_theta_lit = proc_macro2::Literal::f64_unsuffixed(rope_theta_val);
            let t_lit = proc_macro2::Literal::u32_unsuffixed(t);
            let h_lit = proc_macro2::Literal::u32_unsuffixed(h);
            let w_lit = proc_macro2::Literal::u32_unsuffixed(w);
            let rot_dim_lit2 = proc_macro2::Literal::u32_unsuffixed(rot_dim_val);
            quote! {
                /// The rotary cos/sin rows a forward of `n` tokens at `positions` (`[n]`, or MRoPE's
                /// `[3, n]`) reads in place of the position-indexed cache — its positions then each
                /// row's own index.
                #[cfg(scratchy_target = "metal")]
                pub fn metal_rope_rows(
                    positions: &[u32],
                    n: usize,
                ) -> ::core::option::Option<::std::vec::Vec<u8>> {
                    ::core::option::Option::Some(
                        ::scratchy_target_metal::interpreter::metal::build_mrope_cos_sin_override(
                            positions,
                            n,
                            #rot_dim_lit2 as usize,
                            #rope_theta_lit,
                            [#t_lit, #h_lit, #w_lit],
                            <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.metal_dtype(),
                        ),
                    )
                }
            }
        }
        None => quote! {
            /// None: this model's rope reads the positions.
            #[cfg(scratchy_target = "metal")]
            pub fn metal_rope_rows(
                _positions: &[u32],
                _n: usize,
            ) -> ::core::option::Option<::std::vec::Vec<u8>> {
                ::core::option::Option::None
            }
        },
    };
    let mrope_runtime_block = quote! {
        let mut __mrope_ident_vec: ::std::vec::Vec<u32> = ::std::vec::Vec::new();
        let __rope_rows = if !ctx.positions.as_raw().raw_ptr().is_null() {
            let __pos_all = ::std::slice::from_raw_parts(
                ctx.positions.as_raw().raw_ptr() as *const u32,
                ctx.positions.as_raw().numel(),
            );
            metal_rope_rows(__pos_all, n)
        } else {
            ::core::option::Option::None
        };
        let (positions, mrope_cos_sin): (&[u32], ::core::option::Option<&[u8]>) =
            match &__rope_rows {
                ::core::option::Option::Some(rows) => {
                    __mrope_ident_vec.extend(0..n as u32);
                    (__mrope_ident_vec.as_slice(), ::core::option::Option::Some(rows.as_slice()))
                }
                ::core::option::Option::None => (positions, ::core::option::Option::None),
            };
    };

    #[cfg(feature = "metal")]
    {
        let table = metal_tape_cmds
            .into_tokens(&resolver.typed, &resolver.named, &dims.weight_markers)
            .unwrap_or_else(|e| {
                panic!(
                    "[metal static tape] {}: command table: {e}",
                    model.source_stem
                )
            });
        metal_arena_bytes_statics.push(table);
    }
    #[cfg(feature = "metal")]
    {
        let mc = &model_facts;
        metal_arena_bytes_statics.push(metal_tq_codebook_tokens(mc));
        // A multi-token-prediction head (its body reads the target's hidden rows) chains its
        // passes on the device.
        let head = reads_target_hidden.then(|| {
            let drafts = model.bounds.get("spec_drafts").copied();
            let drafts =
                u32::try_from(drafts.expect("an MTP head's spec_drafts")).expect("drafts fit u32");
            let hidden = u32::try_from(target_hidden_width).expect("hidden width fits u32");
            (drafts, hidden)
        });
        let vocab_suffix = metal_vocab_suffix(program, fuf, model, mc);
        metal_arena_bytes_statics.push(metal_off_tape_tokens(logits_width, mc, head, vocab_suffix));
    }
    let metal_emission = quote! {
        #(#metal_arena_bytes_statics)*

        /// Per-canonical tape_index plan for the Metal pool. One row per
        /// `num_tokens` point per routed fold band (`FoldBand`), ordered ascending, a
        /// bucket's band rows adjacent. A pool keeps one band's rows (`pick_rung`), so its
        /// bucket index is no index into this table: a forward's slots come from its bucket's row
        /// (`Buckets::row`), the one the door chose.
        #[cfg(scratchy_target = "metal")]
        pub static METAL_BUCKETS:
            &[::scratchy_target_metal::interpreter::metal::MetalBucketSpec]
            = &[
                #(#metal_bucket_entries)*
            ];

        /// [`METAL_BUCKETS`]' baked rungs, which the worker picks the rung a pool runs on from
        /// before the pool exists: the KV pool's block-table width, the memory budget.
        #[cfg(scratchy_target = "metal")]
        pub static METAL_RUNGS: ::scratchy_target_metal::interpreter::metal::MetalRungs =
            ::scratchy_target_metal::interpreter::metal::MetalRungs {
                buckets: METAL_BUCKETS,
                tq: ::scratchy_target_metal::interpreter::metal::TqGroup::of(
                    <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS,
                ),
                // Checked against the buckets as this static compiles: every slot of every bucket
                // ranked once, and no bucket's final hidden states among the hosting ranks.
                ranks: __arena::RANKED.ranks(METAL_BUCKETS),
                door: ::scratchy_target_metal::interpreter::metal::bucket_row::<__buckets::Bucket>,
            };

        /// Largest `num_tokens` tape_index across [`METAL_BUCKETS`]. The
        /// metal forward body's `RuntimeFactory` allocates per-worker
        /// runtime buffers (input_ids/positions/slot_mapping/...) at
        /// `METAL_MAX_BUCKET_M * sizeof(u32)`.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MAX_BUCKET_M: u32 = #max_bucket_m_lit;

        /// Byte size of the vision 2D-RoPE `freqs` runtime buffer
        /// (f32 `[max_bucket_m, vision_head_dim/2]`). 0 on non-vision
        /// arches (`alloc` floors to 16 bytes).
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VISION_FREQS_BYTES: u64 = #vision_freqs_bytes_lit;

        /// Byte size of the vision `pixels` runtime buffer (model dtype
        /// `[max_bucket_m, vision_in_features]`). 0 on non-vision arches.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VISION_PIXELS_BYTES: u64 = #vision_pixels_bytes_lit;

        /// Byte size of the vision `pos_embeds` runtime buffer (model
        /// dtype `[max_bucket_m, vision_embed_dim]`). 0 on non-vision
        /// arches and on towers without a learned positional embedding.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VISION_POSEMB_BYTES: u64 = #vision_posemb_bytes_lit;

        /// Byte size of the `target_hidden` runtime buffer (model dtype
        /// `[max_bucket_m, hidden]`). 0 on every arch but an MTP head.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_TARGET_HIDDEN_BYTES: u64 = #target_hidden_bytes_lit;

        /// Byte size of the rows a target captures for its DFlash2 drafter (model dtype
        /// `[max_bucket_m, slots · hidden]`). 0 on a model no drafter reads.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_DRAFT_CAPTURE_BYTES: u64 = #draft_capture_bytes_lit;

        /// A DFlash2 drafter's block: its drafts and their anchor.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_DRAFT_ROWS: u64 = #draft_rows_lit;

        /// Byte size of the `mm_embeds` splice buffer (bf16
        /// `[max_bucket_m, hidden]`) and the `mm_dst_rows` buffer (u32
        /// `[max_bucket_m]`). 0 on arches without the multimodal splice.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MM_EMBEDS_BYTES: u64 = #mm_embeds_bytes_lit;
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MM_DST_ROWS_BYTES: u64 = #mm_dst_rows_bytes_lit;

        /// Byte size of the MRoPE cos/sin override buffer (model dtype
        /// `[max_bucket_m, rot_dim]`). 0 on 1D-rope arches (the rope
        /// kernel keeps the static cos/sin cache; `alloc` floors to 16
        /// bytes and the buffer is never bound).
        #[cfg(scratchy_target = "metal")]
        pub const METAL_MROPE_COS_SIN_BYTES: u64 = #mrope_cos_sin_bytes_lit;

        /// Width of one row of the terminal arena slot: `vocab_size`
        /// on decoder layouts (the lm_head Gemm's output),
        /// `hidden_size` on encoder layouts (no lm_head — the terminal
        /// IS the hidden state). The metal forward shapes its returned
        /// `OwnedTensor` as `[num_tokens, METAL_VOCAB_SIZE]` f16.
        #[cfg(scratchy_target = "metal")]
        pub const METAL_VOCAB_SIZE: u64 = #vocab_size_lit;

        /// Build a [`MetalWorkerPool`] for this canonical. Thin
        /// wrapper over [`MetalWorkerPool::for_buckets`] that threads
        /// the per-canonical [`METAL_BUCKETS`] static so callers don't
        /// have to construct the tape_index plan by hand. The arena layout
        /// is derived from each tape_index's `arena_bytes` field — taking
        /// the elementwise max so the single per-worker arena fits the
        /// largest activation across every tape_index.
        ///
        /// `weights` is borrowed once: the pool resolves the tapes'
        /// model sources at construction and keeps only their
        /// buffers, so it can live as a field on the `Weights` struct
        /// without an `Arc`-cycle.
        ///
        /// [`MetalWorkerPool`]: ::scratchy_target_metal::interpreter::metal::MetalWorkerPool
        /// [`MetalWorkerPool::for_buckets`]: ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets
        #[cfg(scratchy_target = "metal")]
        pub fn metal_pool(
            device: ::std::sync::Arc<
                ::scratchy_target_metal::interpreter::metal::__re::Device,
            >,
            weights: &Weights,
            allocator: ::std::sync::Arc<crate::__gpu::MetalAllocator>,
            runtime_factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory,
            max_workers: usize,
            // Runtime per-sequence block-table capacity
            // (`KvCachePool::max_blocks_per_seq`); picks the KV cap rung.
            block_cap: usize,
            addressing: ::scratchy_target_metal::tape::lowered::KvAddressing,
        ) -> ::core::result::Result<
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Weights>,
            ::scratchy_target_metal::interpreter::metal::PoolBuildError,
        > {
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                device,
                weights,
                allocator,
                &METAL_RUNGS,
                runtime_factory,
                max_workers,
                // Standalone factory: no device-budget context here, so keep
                // all buckets. The lazy-init path below passes the real cap.
                None,
                block_cap,
                addressing,
                // Nor the serving engine's batch: every bucket row's decode partials.
                None,
                // Its arena is its own.
                ::scratchy_target_metal::interpreter::metal::ArenaShare::Own,
                // Its callers read the logits on the host.
                ::scratchy_target_metal::interpreter::metal::TerminalReads::Host,
            )
        }

        /// The model's resident metal pool, built on its first forward.
        ///
        /// The factory closure captures (a) the metal device handle
        /// for runtime-buffer allocation, (b) the KV pool's block-table
        /// row stride and the serving batch, which size the block tables,
        /// (c) Arc-handle clones of the per-layer KV
        /// buffers from `ctx.kv_cache` so worker spawns inherit
        /// them without re-allocation. The closure is invoked
        /// once per worker spawn — the pool starts at size 1 so
        /// the first forward triggers the only factory invocation
        /// in single-worker configs. Every forward entry point
        /// reaches the pool through here, whichever runs first.
        #[cfg(scratchy_target = "metal")]
        fn resident_metal_pool<'w>(
            wm: &'w Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &crate::__gpu::GpuDevice,
        ) -> &'w ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Weights> {
            use ::scratchy_forward_compiler::CanonicalParams as _;
            use ::scratchy_target_metal::interpreter::metal::__re::{Buffer, MTLResourceOptions};

            wm.metal_pool.get_or_init(|| {
                // The RoPE tables the pool binds, for the positions this KV cache's block table
                // reaches.
                wm.build_rope_tables(ctx, device)
                    .expect("Weights::build_rope_tables: RoPE table build failed");
                let num_layers = ctx.kv_cache.num_layers;
                // Reactive (chunked) KV pool: bind the per-layer
                // chunk-address TABLE buffers (device uint64 arrays of
                // chunk gpuAddresses), not the cache data. The KV
                // kernels deref `table[block_id / BLOCKS_PER_CHUNK]`.
                // The table buffer identity is stable across the
                // pool's life (chunk-set growth edits its contents, not
                // its binding), so baking its gpuAddress once is sound.
                let kv_k: ::std::vec::Vec<Buffer> = (0..num_layers)
                    .map(|l| ctx.kv_cache.k_chunk_table_mem(l).buffer().clone())
                    .collect();
                let kv_v: ::std::vec::Vec<Buffer> = (0..num_layers)
                    .map(|l| ctx.kv_cache.v_chunk_table_mem(l).buffer().clone())
                    .collect();
                // KV-cache groups (vLLM hybrid layout). Uniform models have a
                // single group with an all-zero layer→group map (byte-identical
                // to the pre-hybrid single block-table path); gemma4 SWA reports
                // its real group count + per-layer mapping through the pool.
                let num_kv_groups: usize = ctx.kv_cache.num_kv_groups();
                let kv_layer_to_group: ::std::vec::Vec<u32> =
                    ctx.kv_cache.layer_to_group_u32();
                // GDN (Gated-DeltaNet) persistent state buffers, captured per
                // (global) layer from `ctx.gdn_state` for hybrid arches; empty
                // for non-hybrid. Non-linear layers reuse the first linear
                // layer's buffer as a never-bound placeholder (the
                // GatedDeltaNet lowering only emits GdnConvState/GdnSsmState on
                // linear layers, so the placeholder is never read).
                let (gdn_conv, gdn_ssm): (::std::vec::Vec<Buffer>, ::std::vec::Vec<Buffer>) =
                    match ctx.gdn_state {
                        ::core::option::Option::Some(gp) => {
                            match (0..gp.num_layers).find(|&l| gp.is_linear(l)) {
                                ::core::option::Option::Some(f0) => {
                                    let fc = gp.conv_layer_mem(f0).buffer().clone();
                                    let fs = gp.ssm_layer_mem(f0).buffer().clone();
                                    let conv = (0..gp.num_layers)
                                        .map(|l| if gp.is_linear(l) {
                                            gp.conv_layer_mem(l).buffer().clone()
                                        } else { fc.clone() })
                                        .collect();
                                    let ssm = (0..gp.num_layers)
                                        .map(|l| if gp.is_linear(l) {
                                            gp.ssm_layer_mem(l).buffer().clone()
                                        } else { fs.clone() })
                                        .collect();
                                    (conv, ssm)
                                }
                                ::core::option::Option::None => {
                                    (::std::vec::Vec::new(), ::std::vec::Vec::new())
                                }
                            }
                        }
                        ::core::option::Option::None => {
                            (::std::vec::Vec::new(), ::std::vec::Vec::new())
                        }
                    };
                // Its draft lanes, the same way (`GdnStatePool::lanes`). A pool without lanes — a
                // model that drafts nothing, or one whose run drafts with nothing (no head found,
                // `--num-speculative-tokens 0`) — runs only plain steps (its slot allocator builds
                // no other: `GdnStepError::Drafts`), which never touch a lane, so a drafting
                // model's tapes bind the layer's own state buffers there, never read.
                let (gdn_lanes_conv, gdn_lanes_ssm): (
                    ::std::vec::Vec<Buffer>,
                    ::std::vec::Vec<Buffer>,
                ) = match ctx.gdn_state {
                    ::core::option::Option::Some(gp) => {
                        let first = (0..gp.num_layers)
                            .find_map(|l| gp.conv_lanes_mem(l).zip(gp.ssm_lanes_mem(l)));
                        match first {
                            ::core::option::Option::Some((fc, fs)) => (0..gp.num_layers)
                                .map(|l| {
                                    let (c, s) = gp
                                        .conv_lanes_mem(l)
                                        .zip(gp.ssm_lanes_mem(l))
                                        .unwrap_or((fc, fs));
                                    (c.buffer().clone(), s.buffer().clone())
                                })
                                .unzip(),
                            ::core::option::Option::None => (gdn_conv.clone(), gdn_ssm.clone()),
                        }
                    }
                    ::core::option::Option::None => {
                        (::std::vec::Vec::new(), ::std::vec::Vec::new())
                    }
                };
                // A DFlash2 drafter's rings, the pool `ctx` lends (the sibling of `gdn_state`):
                // `Some` exactly for a drafter.
                let __draft_rings: ::core::option::Option<(
                    ::std::vec::Vec<Buffer>,
                    ::std::vec::Vec<Buffer>,
                    Buffer,
                )> = ctx.draft_ring.map(|pool| {
                    let (keys, values) = (0..pool.num_layers())
                        .map(|l| {
                            let (k, v) = pool.layer(l);
                            (k.buffer().clone(), v.buffer().clone())
                        })
                        .unzip();
                    (keys, values, pool.windows().buffer().clone())
                });
                // TurboQuant: captured (Copy) so the 'static factory closure can
                // provision the packed stores + scratch per worker for a model
                // whose `KV_CODEC` is TurboQuant. UNIFORM arches (num_kv_groups
                // == 1): every layer is "global", compressed at the base
                // geometry. HYBRID/SWA arches (gemma4: num_kv_groups > 1)
                // compress ONLY the GLOBAL (full-context, group 0) layers at the
                // GLOBAL geometry; the sliding layers stay fp16.
                // `is_global[L]` = (group_of(L) == 0): group 0 is always the
                // full-context group (see `compute_hybrid_kv_layout`); uniform →
                // all-zero map → all-true. A recurrent hybrid's layers that hold
                // no KV (`FACTS.kv_layers()`) get no store.
                let __tq_nb = ctx.kv_cache.num_blocks;
                let __tq_kv_layers = <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS
                    .kv_layers()
                    .map(::scratchy_forward_compiler::model_facts::KvLayers::get);
                // The attention's global class: the full layers' pages and kv heads. `None` for a
                // model without attention, which has neither.
                let __global = <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS
                    .attention()
                    .map(|a| a.global());
                let __tq_is_global: ::std::vec::Vec<bool> = ctx
                    .kv_cache
                    .layer_to_group_u32()
                    .iter()
                    .zip(0u32..)
                    .map(|(&g, l)| g == 0 && __tq_kv_layers.is_none_or(|kv| kv.contains(&l)))
                    .collect();
                // The stores register with the device's allocator, whose KV growth maps them.
                let __tq_allocator = device.allocator.clone();
                // A step's block tables: a row per sequence, at most the serving batch (every
                // row of the largest bucket when the worker has not stashed one), each the KV
                // pool's row stride — the KV cap rung it runs on, which the host pads every row
                // to.
                let __bt_stride = ctx.kv_cache.max_blocks_per_seq as u64;
                let __bt_rows = device.max_decode_rows.map_or(METAL_MAX_BUCKET_M, |r| {
                    r.get().min(METAL_MAX_BUCKET_M)
                }) as u64;
                let factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory =
                    ::scratchy_target_metal::interpreter::metal::RuntimeFactory::new(move |dev, host| {
                        let max_m = METAL_MAX_BUCKET_M as u64;
                        let alloc = |bytes: u64| {
                            use ::scratchy_target_metal::interpreter::metal::__re::MTLDevice as _;
                            dev.newBufferWithLength_options(
                                bytes.max(16) as usize,
                                MTLResourceOptions::StorageModeShared,
                            )
                            .expect("newBufferWithLength_options returned nil")
                        };
                        // Provision at the GLOBAL geometry: uniform arches have
                        // GLOBAL_* == base, so this is identical to the base-geometry
                        // call (every layer global). gemma4 provisions GLOBAL-sized
                        // packed/norms/scratch (head_dim 512, NUM_GLOBAL_KV_HEADS,
                        // GLOBAL_BLOCK_SIZE) for the group-0 layers only.
                        let __tq_prov = METAL_TQ_CODEBOOK.as_ref().map(|codebook| {
                            // A head reading its target's KV binds its target's packed stores.
                            if #reads_target_kv {
                                return host.tq.clone().expect(
                                    "a head reading its target's TurboQuant KV builds its pool in \
                                     its target's, which shares it",
                                );
                            }
                            let global = __global
                                .expect("ModelFacts::new: a TurboQuant KV codec is an attention's");
                            ::scratchy_target_metal::turboquant::build_tq_provision(
                                &__tq_allocator,
                                ::scratchy_target_metal::turboquant::TqPool {
                                    is_global: &__tq_is_global,
                                    num_blocks: __tq_nb,
                                    block_size: global.page().get() as usize,
                                    num_kv_heads: global.kv_heads().get() as usize,
                                    blocks_per_chunk: ::scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize,
                                },
                                codebook,
                                host.tq_scratch.as_ref(),
                            )
                        });
                        ::scratchy_target_metal::interpreter::metal::RuntimeBindings {
                            input_ids: alloc(max_m * 4),
                            positions: alloc(max_m * 4),
                            // One slot_mapping + block_table per KV-cache group
                            // (vLLM hybrid layout). Uniform models have one
                            // group; gemma4 SWA has full + N sliding. The worker
                            // sizes the group count + fills layer_to_group from
                            // the model's HybridKvLayout at runtime (stage 6);
                            // the default single group is byte-identical to the
                            // pre-hybrid path.
                            slot_mappings: (0..num_kv_groups).map(|_| alloc(max_m * 4)).collect(),
                            cu_seqlens_q: alloc((max_m + 1) * 4),
                            seq_used_k: alloc(max_m * 4),
                            // Per-token span label for block-diagonal span
                            // attention, for a step of one sequence: as many as
                            // its full-attention row reaches, the stride's blocks of
                            // `GLOBAL_BLOCK_SIZE` tokens. All-zero unless the
                            // request carries Relocatable spans.
                            span_ids: alloc(
                                __global.map_or(0, |g| __bt_stride * g.page().get() as u64 * 4),
                            ),
                            block_tables: (0..num_kv_groups)
                                .map(|_| alloc(__bt_rows * __bt_stride * 4))
                                .collect(),
                            layer_to_group: kv_layer_to_group.clone(),
                            // TurboQuant: when provisioned, the GLOBAL (group-0)
                            // layers' KV points at the one fp16 scratch (the
                            // injected per-layer dequant/quantize tape ops fill/drain
                            // it); SLIDING layers keep the normal fp16 pool. For
                            // uniform arches every layer is global → all scratch
                            // (byte-identical to the prior all-layers override).
                            // Without tq, every layer uses the fp16 pool.
                            kv_cache_k: __tq_prov
                                .as_ref()
                                .map(|p| {
                                    (0..num_layers)
                                        .map(|l| if __tq_is_global[l] {
                                            p.scratch.k_table.clone()
                                        } else {
                                            kv_k[l].clone()
                                        })
                                        .collect()
                                })
                                .unwrap_or_else(|| kv_k.clone()),
                            kv_cache_v: __tq_prov
                                .as_ref()
                                .map(|p| {
                                    (0..num_layers)
                                        .map(|l| if __tq_is_global[l] {
                                            p.scratch.v_table.clone()
                                        } else {
                                            kv_v[l].clone()
                                        })
                                        .collect()
                                })
                                .unwrap_or_else(|| kv_v.clone()),
                            // Spans rope-on-read per-layer flag mirror.
                            // 16-byte placeholder here; the worker sizes
                            // it to the pool's num_blocks and binds it
                            // only on W::ROPE_ON_READ arches (the metal
                            // lowering omits the binding otherwise, so
                            // this is never read on the non-spans path).
                            block_unrotated_flags: (0..num_layers)
                                .map(|_| alloc(16))
                                .collect(),
                            tq: __tq_prov,
                            // 4 bytes — worker writes the forward's
                            // `num_tokens` here before dispatch: the MoE
                            // sort counts the step's live pairs.
                            num_tokens_u32: alloc(4),
                            // 4 bytes — worker writes the sampled rows'
                            // count here before dispatch so the lm_head
                            // slice's gather / scatter know how many rows
                            // to move.
                            num_sample_rows_u32: alloc(4),
                            // [max_m] u32 — worker writes the sampled
                            // rows here before dispatch.
                            sample_indices: alloc(max_m * 4),
                            // GDN persistent state (per-layer) + per-forward
                            // indices/fresh flags (Shared, overwritten each
                            // forward). Index buffers sized to the max bucket
                            // (num_seqs <= num_tokens <= max_m): a 24-byte
                            // `GdnSeqSlots` record a sequence, a u32 step code.
                            gdn_state_conv: gdn_conv.clone(),
                            gdn_state_ssm: gdn_ssm.clone(),
                            gdn_lanes_conv: gdn_lanes_conv.clone(),
                            gdn_lanes_ssm: gdn_lanes_ssm.clone(),
                            gdn_state_indices: alloc(
                                max_m
                                    * ::std::mem::size_of::<
                                        ::scratchy_target_metal::gdn_state::GdnSeqSlots,
                                    >() as u64,
                            ),
                            gdn_is_fresh: alloc(max_m * 4),
                            // Vision externs: sized from the baked
                            // METAL_VISION_*_BYTES consts (16-byte floor
                            // on non-vision arches). Overwritten per
                            // forward by `write_runtime_inputs`.
                            vision_rope_freqs: alloc(METAL_VISION_FREQS_BYTES),
                            pixels: alloc(METAL_VISION_PIXELS_BYTES),
                            vision_pos_embeds: alloc(METAL_VISION_POSEMB_BYTES),
                            target_hidden: alloc(METAL_TARGET_HIDDEN_BYTES),
                            mm_embeds: alloc(METAL_MM_EMBEDS_BYTES),
                            mm_dst_rows: alloc(METAL_MM_DST_ROWS_BYTES),
                            mrope_cos_sin: alloc(METAL_MROPE_COS_SIN_BYTES),
                            // Qwen2.5-VL windowed-attention externs:
                            // i32/u32 rows bounded by the max bucket.
                            // Tiny — sized unconditionally.
                            vision_cu_seqlens_full: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_cu_seqlens_window: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_window_index: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_reverse_indices: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            vision_position_ids: alloc(
                                (METAL_MAX_BUCKET_M as u64 + 8) * 4,
                            ),
                            // The rows a target captures for its DFlash2 drafter: every row of
                            // the largest bucket, a block a captured layer (16-byte floor on a
                            // model no drafter reads).
                            draft_capture: alloc(METAL_DRAFT_CAPTURE_BYTES),
                            // A DFlash2 drafter's buffers: its rings, the pool `ctx` lends (kept
                            // across forwards, pinned by it), and a cycle's per-lane inputs and
                            // outputs, sized for the largest bucket's lanes.
                            draft: __draft_rings.as_ref().map(|(keys, values, windows)| {
                                let lanes = max_m / METAL_DRAFT_ROWS.max(1);
                                ::scratchy_target_metal::interpreter::metal::DraftRuntimeBuffers {
                                    ring_keys: keys.clone(),
                                    ring_values: values.clone(),
                                    ring_slots: alloc(lanes * 4),
                                    windows: windows.clone(),
                                    retained: alloc(lanes * 4),
                                    lane_params: alloc(
                                        lanes
                                            * ::std::mem::size_of::<
                                                ::scratchy_target_metal::dflash2::DraftLaneParams,
                                            >() as u64,
                                    ),
                                    outputs: ::scratchy_target_metal::dflash2::DraftOutput::ALL
                                        .map(|r| alloc(r.bytes(lanes))),
                                }
                            }),
                            written: ::std::default::Default::default(),
                        }
                    });
                ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                    device.device.clone(),
                    wm,
                    device.allocator.clone(),
                    &METAL_RUNGS,
                    factory,
                    1,
                    // Target-reactive cap stashed on the device by the worker
                    // after `determine_available_memory`. Prunes the compiled
                    // ladder so the colored arena fits the KV budget.
                    device.metal_bucket_max_m,
                    // Runtime per-sequence block-table capacity, read off the KV
                    // pool: it picks the KV cap rung — none for a model reading no KV.
                    #pool_block_cap,
                    device.kv_addressing,
                    // The serving engine's batch, stashed by the worker: the decode split's
                    // partials are reserved for that many rows.
                    device.max_decode_rows,
                    // How its arena is shared with that of a model whose forwards never overlap
                    // its own: an MTP target holds its head's (a fact of the target, set before
                    // its first forward), a head runs in its target's (its device's).
                    ::scratchy_target_metal::interpreter::metal::ArenaShare::of(
                        wm.metal_guest.get(),
                        device.host.as_deref(),
                    ),
                    // Whether its owner reads the logits on the host (a draft model's seed).
                    device.terminal_reads,
                )
                .expect("MetalWorkerPool::for_buckets: pool init failed")
            })
        }

        /// Per-canonical metal forward dispatch. Lazy-inits
        /// `weights.metal_pool` on the first call (factory closure
        /// captures the per-layer `metal::Buffer` Arc-handles from
        /// `ctx.kv_cache` and the `MAX_BLOCKS_PER_SEQ` block-table
        /// stride from `<Weights as CanonicalParams>`); on every call
        /// reads the runtime input slices off the host-visible
        /// `ctx.<input>` `TensorView`s — under metal those raw_ptrs
        /// are `metal::Buffer.contents()` so the slice borrow lives
        /// as long as the call — hands them to
        /// `MetalWorkerPool::forward`, and copies the tape_index's
        /// terminal arena slot out as a fresh `OwnedTensor` of
        /// `[num_tokens, vocab_size]` f16 logits.
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            unsafe { forward_with_metal_followup(wm, ctx, device, num_tokens, None) }
                .unwrap_or_else(|e| ::core::panic!("{e}"))
        }

        /// The runtime inputs `ctx` carries for a forward of `num_tokens` tokens, handed to `run`:
        /// they borrow the host slices `ctx`'s views point at and the rows built here. Refused
        /// when they name no rows to sample (`ForwardError::EmptySequence`).
        ///
        /// # Safety
        /// Every non-null view in `ctx` points at host-visible memory of its numel.
        #[cfg(scratchy_target = "metal")]
        unsafe fn with_forward_inputs<R>(
            ctx: &crate::__gpu::ForwardCtx,
            num_tokens: u64,
            run: impl FnOnce(&::scratchy_target_metal::interpreter::metal::ForwardInputs<'_>) -> R,
        ) -> ::core::result::Result<R, ::scratchy_target_metal::interpreter::metal::ForwardError> {
            // ── Read host-visible input slices off ctx ────────────
            //
            // Under metal, every `TensorView` in `ctx` resolves to a
            // pointer inside a `StorageModeShared` `MTLBuffer.contents()`,
            // so reading as `&[u32]` is a direct CPU borrow. The
            // shared buffer keeps backing the slice until the worker
            // memcopies through `write_runtime_inputs`.
            let n = num_tokens as usize;
            // Vision towers carry no token ids / positions — the tape
            // consumes pixels / freqs / cu_seqlens instead, so the macro's
            // ctx leaves these as `null_view`. `from_raw_parts(null, n)` is
            // UB even for an unused read (and `write_runtime_inputs` only
            // copies `len` bytes), so hand the worker an empty slice when
            // the pointer is null — same null-guard shape as the optional
            // decoder inputs (slot_mapping / cu_seqlens_q / …) just below.
            let input_ids: &[u32] = if ctx.input_ids.as_raw().raw_ptr().is_null() {
                &[]
            } else {
                ::std::slice::from_raw_parts(
                    ctx.input_ids.as_raw().raw_ptr() as *const u32,
                    n,
                )
            };
            let positions: &[u32] = if ctx.positions.as_raw().raw_ptr().is_null() {
                &[]
            } else {
                ::std::slice::from_raw_parts(
                    ctx.positions.as_raw().raw_ptr() as *const u32,
                    n,
                )
            };
            let slot_mapping = if !ctx.slot_mapping.as_raw().raw_ptr().is_null() {
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.slot_mapping.as_raw().raw_ptr() as *const u32,
                    n,
                ))
            } else {
                ::std::option::Option::None
            };
            let cu_seqlens_q = if !ctx.cu_seqlens_q.as_raw().raw_ptr().is_null() {
                let cu_n = ctx.cu_seqlens_q.as_raw().numel();
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.cu_seqlens_q.as_raw().raw_ptr() as *const u32,
                    cu_n,
                ))
            } else {
                ::std::option::Option::None
            };
            let seq_used_k = if !ctx.seqused_k.as_raw().raw_ptr().is_null() {
                let su_n = ctx.seqused_k.as_raw().numel();
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.seqused_k.as_raw().raw_ptr() as *const u32,
                    su_n,
                ))
            } else {
                ::std::option::Option::None
            };
            let block_table = if !ctx.block_table.as_raw().raw_ptr().is_null() {
                let bt_n = ctx.block_table.as_raw().numel();
                ::std::option::Option::Some(::std::slice::from_raw_parts(
                    ctx.block_table.as_raw().raw_ptr() as *const u32,
                    bt_n,
                ))
            } else {
                ::std::option::Option::None
            };
            // Sliding KV-cache GROUPS (gemma4 SWA): per sliding group, the
            // per-token slot_mapping + per-block block table. Empty on non-SWA
            // models. Group `s` here is KV-cache group `s + 1` (group 0 = full).
            let sliding_slot_mappings: ::std::vec::Vec<&[u32]> = ctx
                .sliding_slot_mappings
                .iter()
                .map(|v| ::std::slice::from_raw_parts(v.as_raw().raw_ptr() as *const u32, n))
                .collect();
            let sliding_block_tables: ::std::vec::Vec<&[u32]> = ctx
                .sliding_block_tables
                .iter()
                .map(|v| {
                    let sbt_n = v.as_raw().numel();
                    ::std::slice::from_raw_parts(v.as_raw().raw_ptr() as *const u32, sbt_n)
                })
                .collect();

            // The rows the forward samples (`None`: each sequence's last), resolved once here
            // for every reader of the inputs: the lm_head leaves their logits at the front of
            // its output, in order. A sequence of no rows has no last row: refused.
            let last_rows;
            let sampled_rows: &[u32] = match ctx.sampled_rows.as_ref() {
                ::std::option::Option::Some(view) => {
                    let raw = view.as_raw();
                    ::std::slice::from_raw_parts(raw.raw_ptr() as *const u32, raw.numel())
                }
                ::std::option::Option::None => {
                    last_rows = ::scratchy_target_metal::interpreter::metal::ForwardInputs::last_rows(
                        cu_seqlens_q,
                        num_tokens as u32,
                    )
                    .map_err(::scratchy_target_metal::interpreter::metal::ForwardError::EmptySequence)?;
                    last_rows.rows()
                }
            };

            // GDN per-forward indices (hybrid arches): read from ctx's
            // host-visible TensorViews (a `GdnSeqSlots` record — 6 i32s —
            // per seq, a step code per seq). `None` for non-hybrid arches
            // (ctx fields are None).
            let gdn_state_indices = ctx.gdn_state_indices.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const i32,
                    tv.as_raw().numel(),
                )
            });
            let gdn_is_fresh = ctx.gdn_is_fresh.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u32,
                    tv.as_raw().numel(),
                )
            });
            // Vision externs (vision towers): read as raw bytes from the
            // ctx TensorViews. `freqs` is f32, `pixels` is the model
            // dtype; the worker copies bytes verbatim. `None` for text.
            let vision_rope_freqs = ctx.vision_rope_freqs.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let pixels = ctx.pixels.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let pos_embeds = ctx.pos_embeds.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let target_hidden = ctx.target_hidden.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            // Qwen2.5-VL windowed-attention externs (i32/u32 byte
            // views; `None` on non-windowed towers and text bodies).
            let vision_cu_seqlens_full = ctx.vision_cu_seqlens_full.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_cu_seqlens_window = ctx.vision_cu_seqlens_window.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_window_index = ctx.vision_window_index.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_reverse_indices = ctx.vision_reverse_indices.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            let vision_position_ids = ctx.vision_position_ids.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            // Multimodal splice: vision embeddings (bytes) + a per-source-
            // row destination map built from `embed_patches`. Text-only
            // batches leave `embed_patches` empty → both `None` (no-op).
            let mm_embeds = ctx.mm_embeds.map(|tv| {
                ::std::slice::from_raw_parts(
                    tv.as_raw().raw_ptr() as *const u8,
                    tv.as_raw().size_bytes(),
                )
            });
            // ALWAYS materialized (all-`u32::MAX` for text-only batches):
            // the splice command sits in every decoder tape, so the kernel
            // must read `MAX` (= skip) for non-image rows — an unwritten
            // placeholder buffer would scatter garbage into the residual.
            let __mm_dst_rows_vec: ::std::vec::Vec<u32> = {
                let mut __v = ::std::vec![u32::MAX; n];
                let mut __src: usize = 0;
                for __p in ctx.embed_patches {
                    for __k in 0..(__p.length as usize) {
                        if __src < __v.len() {
                            __v[__src] = __p.token_offset + __k as u32;
                        }
                        __src += 1;
                    }
                }
                __v
            };
            let mm_dst_rows = ::core::option::Option::Some(__mm_dst_rows_vec.as_slice());
            // MRoPE (Qwen3.5-VL) only: build the per-token cos/sin override
            // table + identity positions (shadows `positions`); a no-op
            // `let mrope_cos_sin = None;` on 1D-rope arches.
            #mrope_runtime_block
            let inputs = ::scratchy_target_metal::interpreter::metal::ForwardInputs {
                num_tokens: num_tokens as u32,
                input_ids,
                positions,
                // Per-KV-cache-group slot_mappings / block tables (vLLM hybrid
                // layout): group 0 = full, then the sliding group(s). On non-SWA
                // models `sliding_*` are `None`, so this is a single-element vec
                // == the pre-hybrid single table (byte-identical). Stage 6
                // widens `ForwardCtx` to carry all N sliding groups.
                slot_mappings: slot_mapping
                    .into_iter()
                    .chain(sliding_slot_mappings)
                    .collect(),
                cu_seqlens_q,
                seq_used_k,
                // Block-diagonal span labels — wired from the request's block
                // annotations via ForwardCtx; `None` until that path is
                // populated (buffer stays zeroed ⇒ span mask inert).
                span_ids: ctx.span_ids.as_deref(),
                block_tables: block_table
                    .into_iter()
                    .chain(sliding_block_tables)
                    .collect(),
                sampled_rows,
                gdn_state_indices,
                gdn_is_fresh,
                vision_rope_freqs,
                pixels,
                pos_embeds,
                target_hidden,
                vision_cu_seqlens_full,
                vision_cu_seqlens_window,
                vision_window_index,
                vision_reverse_indices,
                vision_position_ids,
                mm_embeds,
                mm_dst_rows,
                mrope_cos_sin,
                draft: ctx.draft,
                device_inputs: ctx.device_inputs,
                deferred: ctx.deferred,
                reads: ctx.reads,
            };
            ::core::result::Result::Ok(run(&inputs))
        }

        #metal_rope_rows_fn

        /// Same as [`forward`] but takes an optional encoder-tail hook
        /// that's invoked on the same MTL4 compute encoder used to
        /// encode the forward, AFTER the bucket dispatches and BEFORE
        /// `endEncoding`. Lets the caller (the executor's argmax
        /// dispatch, today) append its own dispatches onto the same
        /// CB so forward + tail share one commit and one host wait.
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_with_metal_followup(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            followup: ::core::option::Option<::scratchy_target_metal::model::MetalForwardFollowup<'_>>,
        ) -> ::core::result::Result<crate::__gpu::OwnedTensor, ::scratchy_target_metal::model::GpuExecutionFailed> {
            let pool = resident_metal_pool(wm, ctx, device);
            // A guest's slots are its host's, which may be sparse: the host reads none of them.
            assert!(
                ctx.hidden_out.is_none()
                    || pool.role() != ::scratchy_target_metal::interpreter::metal::ArenaRole::Guest,
                "a forward asked the host for hidden rows of a pool that runs in another's arena \
                 (an MTP head's)"
            );
            let n = num_tokens as usize;

            // ── Run forward + copy logits out ─────────────────────
            //
            // The pool picks the tape_index from `num_tokens`, runs
            // `executeCommandsInBuffer` (interleaved with MPS GEMMs
            // on <M4 hardware), waits for completion, then invokes
            // the closure with `&MetalWorker` + `bucket_idx`. The
            // tape_index's terminal arena slot holds the lm_head output —
            // the logits of the rows the step samples, at its front;
            // we wrap that buffer in a fresh `OwnedTensor` (Arc-
            // handle clone — no copy) so the caller can read
            // `[sampled rows, vocab_size]` f16 logits without taking
            // ownership of the worker's arena.
            with_forward_inputs(ctx, num_tokens, |inputs| {
                let rows = #terminal_rows;
                // Adapter: pool's tail receives `(encoder, worker, runtime, bucket)`
                // and returns `Result<(), ForwardError>`. The public hook
                // is `(encoder, FollowupOutputs) -> Result<(), String>`.
                // Resolve the outputs from the worker's arena, its runtime bindings and the
                // canonical's compile-time vocab; map error variants.
                let tail_adapter = followup.map(|f| {
                    move |enc: &::scratchy_target_metal::model::ProtocolObject<
                              dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                          >,
                          worker: &::scratchy_target_metal::interpreter::metal::MetalWorker<Weights>,
                          #captured_runtime: &::scratchy_target_metal::interpreter::metal::RuntimeBindings,
                          bucket: &__buckets::Bucket|
                          -> ::core::result::Result<
                        (),
                        ::scratchy_target_metal::interpreter::metal::ForwardError,
                    > {
                        let spec = ::scratchy_target_metal::interpreter::metal::Buckets::row(bucket);
                        let outputs = ::scratchy_target_metal::model::FollowupOutputs {
                            logits: &worker.arena[spec.terminal_slot as usize],
                            hidden: &worker.arena[spec.backbone_slot as usize],
                            // A DFlash2 head's target's captured rows.
                            captured: #captured_rows,
                            rows: rows as u32,
                            vocab: METAL_VOCAB_SIZE as u32,
                        };
                        f(enc, outputs).map_err(
                            ::scratchy_target_metal::interpreter::metal::ForwardError::Followup,
                        )
                    }
                });
                pool.forward_with_tail(
                    inputs,
                    |worker, bucket| {
                        let spec = ::scratchy_target_metal::interpreter::metal::Buckets::row(bucket);
                        use ::scratchy_target_metal::interpreter::metal::__re::MTLBuffer as _;
                        // The caller's rows of the final hidden states (an MTP head's next input),
                        // before the worker goes back to the pool and its arena is reused.
                        if let Some(h) = ctx.hidden_out {
                            let row_bytes =
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.hidden().get() * 2;
                            let hidden = worker.arena[spec.backbone_slot as usize].contents();
                            let mut out = h.out.borrow_mut();
                            out.clear();
                            for &row in h.rows {
                                assert!((row as usize) < n, "hidden row {row} past the forward's {n} rows");
                                // SAFETY: the backbone slot holds `[n, HIDDEN_SIZE]` 2-byte rows (the
                                // bucket's arena is sized for at least `n`), and `row < n`.
                                out.extend_from_slice(unsafe {
                                    ::std::slice::from_raw_parts(
                                        (hidden.as_ptr() as *const u8).add(row as usize * row_bytes),
                                        row_bytes,
                                    )
                                });
                            }
                        }
                        let buf = worker.arena[spec.terminal_slot as usize].clone();
                        let vocab = METAL_VOCAB_SIZE as usize;
                        let shape = [rows, vocab];
                        let bytes = rows * vocab * 2; // bf16/f16 — both 2 bytes
                        let dtype = match <Weights as ::scratchy_forward_compiler::CanonicalParams>::FACTS.metal_dtype() {
                            ::scratchy_target_metal::interpreter::metal::MetalDtype::F16 =>
                                crate::__gpu::dtype::DType::F16,
                            ::scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 =>
                                crate::__gpu::dtype::DType::BF16,
                            ::scratchy_target_metal::interpreter::metal::MetalDtype::Int4 =>
                                ::core::unreachable!("Int4 has no logits dtype"),
                        };
                        // Null where the logits are a sparse (private) slot the host cannot read:
                        // the GPU's argmax and sampler bind the buffer, and a pool whose owner
                        // reads them on the host keeps the slot shared (`TerminalReads::Host`).
                        let inner = crate::__gpu::tensor::GpuTensor::new(
                            ::scratchy_target_metal::sparse_arena::host_ptr(&buf),
                            &shape,
                            dtype,
                        );
                        crate::__gpu::owned_from_metal_buffer(inner, buf, bytes)
                    },
                    tail_adapter,
                )
            })
            .and_then(::core::convert::identity)
            // A command buffer the GPU failed (an out-of-memory, say) fails this step, not the
            // worker thread: the executor fails its requests and schedules again. Any other error
            // is an invariant the forward broke (no bucket fits, a block table of the wrong width,
            // KV not mapped), which still stops the server.
            .map_err(|e| match e {
                ::scratchy_target_metal::interpreter::metal::ForwardError::ExecutionFailed(_) => {
                    ::scratchy_target_metal::model::GpuExecutionFailed(e.to_string())
                }
                // Its `Display` names `MetalWorkerPool::forward` already.
                other => ::core::panic!("{other}"),
            })
        }

        /// One forward encoded onto `encoder` — another model's command buffer, after that model's
        /// forward, whose outputs `ctx.device_inputs` read — with its input writes at its head
        /// (`MetalWorkerPool::forward_onto`). Its worker stays out of the pool until the returned
        /// forward drops, once that command buffer is done.
        ///
        /// # Safety
        /// Same as [`forward`].
        #[cfg(scratchy_target = "metal")]
        pub unsafe fn forward_onto<'w>(
            wm: &'w Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            encoder: &::scratchy_target_metal::model::ProtocolObject<
                dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
            >,
        ) -> ::core::result::Result<
            ::std::boxed::Box<dyn ::scratchy_target_metal::model::MetalForwardOnto + 'w>,
            ::std::string::String,
        > {
            struct Onto<'p>(::scratchy_target_metal::interpreter::metal::Onto<'p, Weights>);
            impl ::scratchy_target_metal::model::MetalForwardOnto for Onto<'_> {
                unsafe fn then(
                    &mut self,
                    ctx: crate::__gpu::ForwardCtxHandle<'_>,
                    num_tokens: u64,
                    encoder: &::scratchy_target_metal::model::ProtocolObject<
                        dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                    >,
                ) -> ::core::result::Result<(), ::std::string::String> {
                    let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                    with_forward_inputs(ctx, num_tokens, |inputs| self.0.then(inputs, encoder))
                        .and_then(::core::convert::identity)
                        .map_err(|e| format!("Onto::then: {e}"))
                }

                fn hidden(
                    &self,
                ) -> &::scratchy_target_metal::model::ProtocolObject<
                    dyn ::scratchy_target_metal::model::MTLBuffer,
                > {
                    let row = ::scratchy_target_metal::interpreter::metal::Buckets::row(self.0.bucket());
                    &self.0.worker().arena[row.backbone_slot as usize]
                }

                fn logits(
                    &self,
                ) -> &::scratchy_target_metal::model::ProtocolObject<
                    dyn ::scratchy_target_metal::model::MTLBuffer,
                > {
                    let row = ::scratchy_target_metal::interpreter::metal::Buckets::row(self.0.bucket());
                    &self.0.worker().arena[row.terminal_slot as usize]
                }

                fn draft_selection(
                    &self,
                ) -> ::core::option::Option<::scratchy_target_metal::model::DraftSelection<'_>> {
                    use ::scratchy_target_metal::dflash2::DraftOutput as O;
                    (self.0.runtime().draft.as_ref()).map(|d| {
                        ::scratchy_target_metal::model::DraftSelection {
                            tokens: d.output(O::Tokens),
                            candidates: d.output(O::Candidates),
                            q_probs: d.output(O::QProbs),
                            confidence: d.output(O::Confidence),
                            verify_rows: d.output(O::VerifyRows),
                        }
                    })
                }

                fn vocab(&self) -> u32 {
                    METAL_VOCAB_SIZE as u32
                }
            }
            let pool = resident_metal_pool(wm, ctx, device);
            with_forward_inputs(ctx, num_tokens, |inputs| pool.forward_onto(inputs, encoder))
                .and_then(::core::convert::identity)
                .map(|onto| ::std::boxed::Box::new(Onto(onto))
                    as ::std::boxed::Box<dyn ::scratchy_target_metal::model::MetalForwardOnto + 'w>)
                .map_err(|e| format!("MetalWorkerPool::forward_onto: {e}"))
        }

        /// Phase 6 chain entry point. Opens ONE MTL4 cmdbuf on the
        /// pool's MTL4 queue and invokes `body` with a
        /// [`ChainStepHandle`] adapter + the worker's runtime + the
        /// chain encoder. Caller drives K bucket forwards + per-iter
        /// argmax + chain_advance dispatches onto the same encoder;
        /// pool owns CB lifecycle (one commit, one host wait for
        /// the entire chain). Falls back to a clear error string when
        /// the bucket pick fails for the iter shape.
        ///
        /// `num_tokens` is the shape of EACH forward iter (constant
        /// across iters because the K-step decode shape is fixed).
        ///
        /// [`ChainStepHandle`]: ::scratchy_target_metal::model::ChainStepHandle
        #[cfg(scratchy_target = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_chain_with_encoder(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            body: ::scratchy_target_metal::model::MetalChainBody<'_>,
        ) -> ::core::result::Result<(), ::std::string::String> {
            let pool = resident_metal_pool(wm, ctx, device);
            with_forward_inputs(ctx, num_tokens, |inputs| {
                // The bucket of iter 0's tokens, at the door. The chain's shape is constant across
                // iters, so every iter plays it, at its tokens.
                let bucket = pool
                    .bucket_for(inputs)
                    .map_err(|e| format!("MetalWorkerPool::bucket_for: {e}"))?;
                let terminal_slot =
                    ::scratchy_target_metal::interpreter::metal::Buckets::row(&bucket).terminal_slot
                        as usize;
                let vocab = METAL_VOCAB_SIZE as u32;

                pool.with_chain_encoder(
                    inputs,
                    |worker, runtime, enc| {
                        // Concrete adapter that satisfies the non-generic
                        // `ChainStepHandle` trait. Holds the worker
                        // borrow + the bucket; calls
                        // `worker.run_bucket_mtl4` to encode one iter onto
                        // the encoder.
                        struct Adapter<'a, W: ::scratchy_forward_compiler::CanonicalParams
                            + ::scratchy_target_metal::interpreter::metal::BakedBuckets>
                        {
                            worker: &'a ::scratchy_target_metal::interpreter::metal::MetalWorker<W>,
                            bucket: W::Bucket,
                            terminal_slot: usize,
                            vocab: u32,
                        }
                        impl<W: ::scratchy_forward_compiler::CanonicalParams
                            + ::scratchy_target_metal::interpreter::metal::BakedBuckets>
                            ::scratchy_target_metal::model::ChainStepHandle for Adapter<'_, W>
                        {
                            fn run_forward_step(
                                &self,
                                encoder: &::scratchy_target_metal::model::ProtocolObject<
                                    dyn ::scratchy_target_metal::model::MTL4ComputeCommandEncoder,
                                >,
                                num_seqs: u32,
                                sampled_rows: ::scratchy_target_metal::model::SampledRowCount,
                            ) -> ::core::result::Result<(), ::std::string::String> {
                                self.worker
                                    .run_bucket_mtl4(
                                        &self.bucket,
                                        num_seqs,
                                        sampled_rows,
                                        ::scratchy_target_metal::interpreter::metal::ForwardReads::Outputs,
                                        encoder,
                                    )
                                    .map_err(|e| format!("run_bucket_mtl4: {e:?}"))
                            }

                            fn logits_buf(
                                &self,
                            ) -> &::scratchy_target_metal::model::ProtocolObject<
                                dyn ::scratchy_target_metal::model::MTLBuffer,
                            > {
                                &self.worker.arena[self.terminal_slot]
                            }

                            fn vocab(&self) -> u32 { self.vocab }
                        }
                        let adapter = Adapter {
                            worker,
                            bucket,
                            terminal_slot,
                            vocab,
                        };
                        // Erase the metal-typed `&RuntimeBindings` behind the
                        // neutral handle so `MetalChainBody` (defined in the
                        // target-free compiler) names no metal-target type; the
                        // worker's body closure recovers `&RuntimeBindings`.
                        body(
                            &adapter,
                            ::scratchy_target_metal::model::MetalRuntimeHandle::new(runtime),
                            enc,
                        )
                        .map_err(
                            ::scratchy_target_metal::interpreter::metal::ForwardError::Followup,
                        )
                    },
                )
                .map_err(|e| format!("MetalWorkerPool::with_chain_encoder: {e}"))
            })
            .map_err(|e| format!("forward_chain_with_encoder: {e}"))
            .and_then(::core::convert::identity)
        }
    };

    // Encoder layouts have no lm_head split — `forward_backbone` is
    // semantically identical to `forward`. Decoder layouts return the
    // pre-lm_head activation via a DtoD memcpy of the protected
    // backbone slot. Both shapes are dispatched via the same
    // `FORWARD_TABLE` row.
    let forward_backbone_fn = match layout {
        BackboneLayout::Decoder => quote! {},
        BackboneLayout::Encoder => quote! {},
    };

    // The per-arch device-runtime emit (weight-accessor impl, the static `Op`
    // tapes, `FORWARD_TABLE`) is the per-op host-dispatch surface — it exists
    // only for backends that dispatch each op through a host-emitted table
    // (metal). A target that runs an embedded fused program instead
    // (e.g. KTIR on the host) emits none of it. The macro is built per-backend,
    // so gate the EMISSION via `cfg!` rather than per-item `#[cfg]` attrs (the
    // static-slice repetition can't take one).
    let (canonical_params_impl, weight_accessors_impl, static_slices_emit, forward_table) =
        if cfg!(feature = "metal") {
            (
                canonical_params_impl,
                weight_accessors_impl,
                quote! { #(#static_slices)* },
                forward_table,
            )
        } else {
            (quote! {}, quote! {}, quote! {}, quote! {})
        };

    let tokens = quote! {
        #weights

        #target_bake

        #canonical_params_impl

        #weight_accessors_impl

        #instruction_alias

        #static_slices_emit

        #forward_table

        #metal_emission

        #forward_backbone_fn

        /// Walk `FORWARD_TABLE` and return one [`BucketDump`] per
        /// row, with backbone + lm_head normalized for non-generic
        /// inspection (no `&Weights`, no GPU). Used by
        /// `scr model info` via the inventory registry. Available
        /// under either backend so the CLI subcommand can dump
        /// metal-compiled arches too.
        #[cfg(scratchy_target = "metal")]
        pub fn dump() -> ::std::vec::Vec<::scratchy_forward_compiler::BucketDump> {
            FORWARD_TABLE
                .iter()
                .map(|e| ::scratchy_forward_compiler::BucketDump {
                    m_min: e.0,
                    m_max_excl: e.1,
                    sk_min: e.2,
                    sk_max_excl: e.3,
                    backbone: ::scratchy_forward_compiler::normalize_slice(e.4),
                    lm_head: ::scratchy_forward_compiler::normalize_slice(e.5),
                })
                .collect()
        }
    };
    (tokens, weights_affine_rows)
}

/// Unused — consumers used to reach this by name from tests.
/// Retained as a no-op type anchor while the trait-based shim is
/// being deleted from downstream call sites.
#[allow(dead_code)]
fn _unused(_: OpKind) {}

/// Emit a shim variant module: one whose forward-fn bodies are
/// byte-identical to a canonical sibling's. Instead of re-emitting
/// the bodies (which rustc would LLVM-optimize independently per
/// variant, compounding release-build time multiplicatively), we:
///
/// - `pub type Weights = super::<canonical>::Weights;` — share the
///   same struct layout; variants in the same equivalence class
///   end up wrapping the same concrete type at the arch-dispatcher
///   level, which is fine for `enum Outer { V1(T), V2(T) }`.
/// - `pub fn fingerprint_matches` — VARIANT-specific. The
///   tensor-suffix gate (e.g. dense `.weight` vs AWQ `.qweight` vs
///   CT `.weight_packed` vs BNB4 `.weight.absmax`) differs per
///   variant, so each ships its own sniff. The affine bit-map gate
///   inherits the CANONICAL's rows when the shim's own accessor set
///   yields none (a shim lowers no tape, so it is empty) — the
///   shim's `load` delegates to the canonical's body, so the
///   canonical's widths are what a selected shim actually reads.
/// - `pub fn load` — VARIANT-specific. The loader calls
///   `MarlinLinear::load_awq` vs `load_gptq` vs
///   `Bnb4bitLinear::load` etc. depending on the variant's
///   `quantization_config`, but constructs the same canonical
///   `Weights` struct at the end (same accessor shapes across the
///   equivalence class).
/// - `pub use super::<canonical>::{forward, forward_backbone,
///   forward_m_<N>, forward_backbone_m_<N>, ...};` — no fn-body
///   re-emit. rustc doesn't re-monomorphize `pub use` paths, so
///   the canonical's release-optimized forward is called directly
///   through this module without additional LLVM work.
#[allow(clippy::too_many_arguments)]
fn emit_shim_model(
    program: &Program,
    fuf: &Fuf,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    canonical: &Ident,
    tp_world_size: u8,
    emit_fingerprint: bool,
    // The canonical's affine rows — the fallback for this shim's
    // fingerprint gate when its own accessor set yields none.
    canonical_affine_rows: Option<&[(String, u32, u32)]>,
    emitter: &dyn crate::target::TargetEmitter,
) -> (TokenStream, Vec<(String, u32, u32)>) {
    let (weights, shim_affine_rows) = emit_weights_struct(
        program,
        fuf,
        None,
        &ModelDims::default(),
        model,
        manifest,
        WeightsEmitMode::Shim { canonical },
        tp_world_size,
        emit_fingerprint,
        canonical_affine_rows,
        emitter,
    );

    // Per-tape_index fn surfaces are gone — dispatch lives on the
    // canonical's `FORWARD_TABLE` + `find_bucket`. Re-export the
    // arch-level dispatchers only. Under `metal` the shim shares the
    // canonical's `METAL_BUCKETS` static + `metal_pool()` constructor
    // — variant-specific differences (quant format, fingerprint) are
    // load-time only; static tape_index plans are byte-identical.
    let target_shim_items = emitter.shim_items(canonical);
    let tokens = quote! {
        #weights

        // What the target re-exports from the canonical a shim shares its solve with.
        #target_shim_items

        // `dump` is the per-variant `scr model info` accessor —
        // available under either backend so shim variants register
        // under metal too.
        #[cfg(scratchy_target = "metal")]
        pub use super::#canonical::dump;

        #[cfg(scratchy_target = "metal")]
        pub use super::#canonical::{
            forward, forward_chain_with_encoder, forward_onto, forward_with_metal_followup,
            metal_rope_rows,
            METAL_BUCKETS, METAL_OFF_TAPE, METAL_RUNGS,
            metal_pool,
        };
    };
    (tokens, shim_affine_rows)
}

/// No bundle loader is sized off the config: `plan_field_load` (and the DeepSeek MoE reader it
/// calls) reads none of the config keys a bundle's dims could come from, so every dim it emits is
/// [`BundleDims`]', its derived bundle's roles evaluated under the model's bounds.
#[cfg(test)]
#[test]
fn no_bundle_loader_reads_a_dim_from_the_config() {
    let src = include_str!("codegen.rs");
    let body = |name: &str| -> &str {
        let start = src
            .find(&format!("\nfn {name}("))
            .unwrap_or_else(|| panic!("fn {name}"));
        let end = src[start + 1..]
            .find("\n}\n")
            .map_or(src.len(), |e| start + 1 + e);
        &src[start..end]
    };
    let keys = [
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "n_shared_experts",
        "moe_intermediate_size",
        "intermediate_size",
        "shared_expert_intermediate_size",
        "hidden_size",
    ];
    for name in ["plan_field_load", "read_deepseek_moe_cfg"] {
        let text = body(name);
        for key in keys {
            let read = format!("get(\"{key}\")");
            assert!(
                !text.contains(&read),
                "`{name}` reads `{key}` off the config"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_vocab::WeightAccessor;
    use quote::format_ident;
    use scratchy_ir::model_facts::raw;

    /// Helper: build a minimal `ModelParams` from raw bound values
    /// for canonical-params shard tests. Only the fields
    /// `emit_canonical_params_impl` actually reads are populated;
    /// everything else gets a sensible default.
    fn shard_test_model(
        num_attention_heads: u64,
        num_key_value_heads: u64,
        head_dim: u64,
        intermediate_size: u64,
    ) -> crate::config::ModelParams {
        use std::collections::BTreeMap;
        use std::path::PathBuf;

        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("num_attention_heads".into(), num_attention_heads);
        bounds.insert("num_key_value_heads".into(), num_key_value_heads);
        bounds.insert("head_dim".into(), head_dim);
        bounds.insert("intermediate_size".into(), intermediate_size);
        bounds.insert("num_hidden_layers".into(), 1);
        let scalars = BTreeMap::from([("rope_theta".to_string(), 10_000.0)]);
        let text = crate::config::TextGeometry::of(&bounds, &scalars, &PathBuf::new())
            .expect("the shard test model states its geometry");
        crate::config::ModelParams {
            name: "shard_test".into(),
            source_stem: "shard-test".into(),
            source_path: PathBuf::new(),
            bounds,
            scalars,
            layer_sets: Default::default(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            drafts_for: Vec::new(),
            drafter_repo_infix: None,
            checkpoint_prefix: None,
            published: None,
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            expert_activation: crate::config::ExpertActivation::Silu,
            derived: BTreeMap::new(),
            text,
        }
    }

    /// A uniform decoder's KV layers: all of them.
    fn every_layer(m: &ModelParams) -> KvCacheLayers {
        let layers = m.text.layers.get() as u32;
        KvCacheLayers((0..layers).collect())
    }

    /// ⭐ THE SCALE LAW, over EVERY checked-in config: the fill's `Attention::scale()`
    /// states main's `attention_scale_for` value BIT FOR BIT. The oracle below is a
    /// frozen copy of that deleted fn (verified identical to 64fd901475's), and the
    /// value under test is the REAL fill — `fill_canonical_params` →
    /// `attention().scale()` — over every `configs/<arch>/<stem>.json` the tree checks
    /// in, walked from the tree (never a hand list, never a named model). The loop
    /// also proves the fill's `head_dim > 0 … else 0.0` arm never fires for an
    /// attention model: the oracle has no 0.0 arm, so a config where it did would
    /// fail the bit comparison here.
    #[test]
    fn the_fill_states_the_frozen_scale_bit_for_bit_over_every_config() {
        // FROZEN: main's `weight_vocab::attention_scale_for`, transcribed expression
        // for expression. Priority, by the config key itself: `attention_multiplier`
        // (a direct override) → `query_pre_attn_scalar.powf(-0.5)` → `1/sqrt(head_dim)`.
        fn frozen_attention_scale_for(model: &ModelParams) -> f32 {
            if let Some(s) = model.scalars.get("attention_multiplier") {
                return *s as f32;
            }
            match model.scalars.get("query_pre_attn_scalar") {
                Some(q) => (*q as f32).powf(-0.5),
                None => {
                    let head_dim_f = *model
                        .bounds
                        .get("head_dim")
                        .expect("attention scale: head_dim missing from model config")
                        as f32;
                    1.0 / head_dim_f.sqrt()
                }
            }
        }
        // Every checked-in stem config: walk the arch configs dir, keeping model
        // stems and dropping the dir's own metadata (arch spec, quantization and
        // weights manifests, drift fragments). `load_file_declared` is the loader
        // the build's walk uses for one stem — arch spec and overrides applied, so
        // what loads here is what a build compiles.
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("read the arch configs dir") {
                let path = entry.expect("config entry").path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|e| e == "json")
                    && !matches!(
                        path.file_name().and_then(|n| n.to_str()),
                        Some("arch.json") | Some("quantizations.json") | Some("weights.json")
                    )
                    && !path.to_string_lossy().ends_with(".overrides.json")
                {
                    out.push(path);
                }
            }
        }
        let root =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/arch/configs");
        let mut paths = Vec::new();
        walk(&root, &mut paths);
        paths.sort();
        let mut stated = 0usize;
        for path in &paths {
            // Load-or-refuse: a config shipping a mixed-bit affine quantization
            // refuses under a backend that can't lower it (asserted by name
            // inside the helper) and is skipped; otherwise it loads.
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            let Some(model) = crate::config::assert_load_or_refuse_by_name(
                &stem,
                crate::config::load_file_declared(path),
            ) else {
                continue;
            };
            let facts = fill_canonical_params(&model, 1, false, &every_layer(&model), None).0;
            // A body with no decoder attention (a vision tower, a drafter head) states
            // no scale — the law is over the configs that do.
            let Some(attention) = facts.attention() else {
                continue;
            };
            stated += 1;
            assert_eq!(
                attention.scale().get().to_bits(),
                frozen_attention_scale_for(&model).to_bits(),
                "{}: the fill states a scale the frozen formula does not",
                model.source_stem
            );
            // And both scale SITES state their law over the REAL filled value.
            assert_eq!(
                scratchy_subtile::subtile_ir::AttnScaleSite::InKernel
                    .scale(attention)
                    .to_bits(),
                attention.scale().get().to_bits(),
                "{}: InKernel states the model's own scale",
                model.source_stem
            );
            assert_eq!(
                scratchy_subtile::subtile_ir::AttnScaleSite::FoldedIntoWeights
                    .scale(attention)
                    .to_bits(),
                1.0f32.to_bits(),
                "{}: FoldedIntoWeights states exactly 1.0",
                model.source_stem
            );
        }
        // ⛔ NOT VACUOUS: the walk must find the tree's attention stems, or the law
        // above ran over nothing. The stated count is the MEASURED result (the floor
        // is only the trap for a broken walk).
        eprintln!(
            "scale law: {stated} attention configs of {} checked-in stems",
            paths.len()
        );
        assert!(
            stated >= 100,
            "the scale law ran over only {stated} attention configs of {} stems — the walk \
             is wrong, not the tree",
            paths.len()
        );
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 1` is
    /// identity — bounds flow through unchanged. Pinning this so
    /// task #7's outer-loop fanout cannot accidentally regress the
    /// single-rank build (which is every per-arch-crate build under
    /// `--features cuda` until activation lands).
    #[test]
    fn the_generated_impl_states_facts_and_no_derived_const() {
        // Every const `CanonicalParams` derives from `FACTS` (`model_facts::raw`): a generated impl
        // that restated one would be a second source beside `FACTS`, overriding the default.
        const DERIVED: &[&str] = &[
            "HEAD_DIM",
            "NUM_Q_HEADS",
            "NUM_KV_HEADS",
            "Q_SIZE",
            "KV_SIZE",
            "INTERMEDIATE_SIZE",
            "ATTN_SCALE",
            "ATTN_SOFTCAP",
            "SLIDING_WINDOW",
            "FINAL_LOGIT_SOFTCAPPING",
            "MROPE_SECTION",
            "KV_CODEC",
            "KV_CACHE_LAYERS",
            "VISION_NUM_HEADS",
            "VISION_HEAD_DIM",
            "VISION_Q_SIZE",
            "VISION_IN_FEATURES",
            "VISION_ROPE_INTERLEAVED",
            "VISION_ATTN_SCALE",
            "VISION_PATCH_GRID_SIDE",
            "VISION_POOL_KERNEL",
            "RMS_NORM_EPS",
            "NORM_WEIGHT_OFFSET",
            "BLOCK_SIZE",
            "GLOBAL_BLOCK_SIZE",
            "GDN_NUM_K_HEADS",
            "GDN_NUM_V_HEADS",
            "GDN_HEAD_K_DIM",
            "GDN_HEAD_V_DIM",
            "GDN_CONV_KERNEL",
            "GDN_CONV_DIM",
            "SPEC_DRAFTS",
            "DRAFT_CONV_GROUP",
            "DRAFT_SELECTOR_RANK",
            "DRAFT_MASK_TOKEN",
            "TARGET_HIDDEN_WIDTH",
            "TARGET_ROUTES_EXPERTS",
            "DRAFT_CAPTURE_SLOTS",
            "ROT_DIM",
            "GLOBAL_HEAD_DIM",
            "NUM_GLOBAL_KV_HEADS",
            "GLOBAL_ROT_DIM",
            "GLOBAL_Q_SIZE",
            "ROPE_PROPORTIONAL",
            "ROPE_ON_READ",
            "METAL_DTYPE",
            "HIDDEN_SIZE",
            "VOCAB_SIZE",
            "SCALE_DTYPE",
        ];
        let m = shard_test_model(32, 8, 128, 11008);
        let (facts, mla) = fill_canonical_params(&m, 1, false, &every_layer(&m), None);
        let ts = emit_canonical_params_impl(facts, &mla).to_string();
        assert!(ts.contains("const FACTS"), "the impl states FACTS: {ts}");
        for name in DERIVED {
            assert!(
                !ts.contains(&format!("const {name} :")),
                "the generated impl restates {name}, which CanonicalParams derives from FACTS: {ts}"
            );
        }
    }

    #[test]
    fn canonical_params_at_tp_eq_1_is_identity() {
        // Llama-2-7B-ish numbers: 32 q heads, 32 kv heads,
        // head_dim=128, intermediate=11008.
        let m = shard_test_model(32, 32, 128, 11008);
        let facts = fill_canonical_params(&m, 1, false, &every_layer(&m), None).0;
        // Q size = 32 * 128 = 4096; KV size = 32 * 128 = 4096.
        assert!(
            raw::num_q_heads(facts) == 32,
            "tp=1 must keep NUM_Q_HEADS = 32; got {}",
            raw::num_q_heads(facts)
        );
        assert!(
            raw::num_kv_heads(facts) == 32,
            "tp=1 must keep NUM_KV_HEADS = 32; got {}",
            raw::num_kv_heads(facts)
        );
        assert!(
            raw::intermediate_size(facts) == 11008,
            "tp=1 must keep INTERMEDIATE_SIZE = 11008; got {}",
            raw::intermediate_size(facts)
        );
        assert!(
            raw::q_size(facts) == 4096,
            "tp=1 must keep Q_SIZE = 4096; got {}",
            raw::q_size(facts)
        );
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 2` shards
    /// every column-parallel dim by 2. The kernel-launch sizes that
    /// flow through `<W as CanonicalParams>::…` constants in the
    /// emitted `Instruction::eval` body MUST be the per-rank values
    /// — nothing else in the compiled body is tp-aware.
    #[test]
    fn canonical_params_at_tp_eq_2_shards_column_parallel_dims() {
        let m = shard_test_model(32, 32, 128, 11008);
        let facts = fill_canonical_params(&m, 2, false, &every_layer(&m), None).0;
        assert!(
            raw::num_q_heads(facts) == 16,
            "tp=2 must shard NUM_Q_HEADS to 16; got {}",
            raw::num_q_heads(facts)
        );
        assert!(
            raw::num_kv_heads(facts) == 16,
            "tp=2 must shard NUM_KV_HEADS to 16; got {}",
            raw::num_kv_heads(facts)
        );
        assert!(
            raw::intermediate_size(facts) == 5504,
            "tp=2 must shard INTERMEDIATE_SIZE to 5504; got {}",
            raw::intermediate_size(facts)
        );
        // Q_SIZE = (32/2) * 128 = 2048
        assert!(
            raw::q_size(facts) == 2048,
            "tp=2 must compute Q_SIZE from sharded heads to 2048; got {}",
            raw::q_size(facts)
        );
        assert!(
            raw::kv_size(facts) == 2048,
            "tp=2 must compute KV_SIZE from sharded heads to 2048; got {}",
            raw::kv_size(facts)
        );
        // HEAD_DIM is per-head and never sharded.
        assert!(
            raw::head_dim(facts) == 128,
            "tp=2 must keep HEAD_DIM = 128; got {}",
            raw::head_dim(facts)
        );
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 8` shards
    /// every column-parallel dim by 8. Pins the floor-divide
    /// behavior on a value that's the upper bound of the compile-
    /// time set — the compile() outer-loop fanout in task #7 stops
    /// at tp=8 by default, so this is the largest case that ever
    /// reaches emit.
    #[test]
    fn canonical_params_at_tp_eq_8_shards_column_parallel_dims() {
        // Llama-3-8B-ish numbers: 32 q heads, 8 kv heads,
        // head_dim=128, intermediate=14336.
        let m = shard_test_model(32, 8, 128, 14336);
        let facts = fill_canonical_params(&m, 8, false, &every_layer(&m), None).0;
        // 32 / 8 = 4
        assert!(
            raw::num_q_heads(facts) == 4,
            "tp=8 must shard NUM_Q_HEADS to 4; got {}",
            raw::num_q_heads(facts)
        );
        // 8 / 8 = 1
        assert!(
            raw::num_kv_heads(facts) == 1,
            "tp=8 must shard NUM_KV_HEADS to 1; got {}",
            raw::num_kv_heads(facts)
        );
        // 14336 / 8 = 1792
        assert!(
            raw::intermediate_size(facts) == 1792,
            "tp=8 must shard INTERMEDIATE_SIZE to 1792; got {}",
            raw::intermediate_size(facts)
        );
    }

    #[test]
    fn accessor_methods_compress_layered_fields_to_slice_index() {
        // Three layers' worth of `input_layernorm_<n>` fields collapse
        // into ONE `pub fn input_layernorm(&self, layer: u32) ->
        // &RmsNorm` whose body is a single slice index — no per-layer
        // arms, no 40-arm match. This is the load-bearing invariant
        // for the impl Weights compression: 40 arms × 5+ accessors
        // worth of Rust tokens disappear.
        let ty: TokenStream = quote! { crate::__gpu::layers::RmsNorm };
        let accessors = (0u64..3)
            .map(|n| WeightAccessor {
                name: format_ident!("input_layernorm_{}", n),
                rust_type: ty.clone(),
                kind: crate::weight_vocab::WeightKind::RmsNorm,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(n)))],
            })
            .collect::<Vec<_>>();
        let ts = emit_weights_accessor_methods(&accessors, &ModelDims::default()).to_string();
        assert!(ts.contains("impl Weights"));
        assert!(ts.contains("fn input_layernorm"));
        assert!(ts.contains("layer : u32"));
        // Single slice index — no match arm bloat. The body is
        // `unsafe { self.input_layernorm.get_unchecked(layer as usize) }`.
        assert!(ts.contains("self . input_layernorm . get_unchecked"));
        assert!(ts.contains("layer as usize"));
        // No 40-arm match anymore — the per-layer arms are gone.
        assert!(!ts.contains("match layer"));
        // No per-layer field references — the data lives in a single
        // `Vec<T>` field with the base name.
        assert!(!ts.contains("input_layernorm_0"));
        assert!(!ts.contains("input_layernorm_1"));
        assert!(!ts.contains("input_layernorm_2"));
        // No format-args bloat.
        assert!(!ts.contains("out of range"));
        assert!(!ts.contains("panic"));
    }

    #[test]
    fn accessor_methods_emit_unit_arm_for_unlayered_fields() {
        // `embed_tokens` has no trailing layer index; the method
        // ignores its layer arg and returns the field directly. Same
        // shape as before the Vec compression — unindexed accessors
        // never had a match.
        let accessors = vec![WeightAccessor {
            name: format_ident!("embed_tokens"),
            rust_type: quote! { crate::__gpu::layers::Embedding },
            kind: crate::weight_vocab::WeightKind::Embedding,
            source_weights: vec![],
        }];
        let ts = emit_weights_accessor_methods(&accessors, &ModelDims::default()).to_string();
        assert!(ts.contains("fn embed_tokens"));
        // Unindexed accessor's layer arg is `_: u32` — the underscore
        // prefix on `_layer` was unnecessary chars, dropped along
        // with per-method `#[cfg]` / `#[inline]` / `#[allow]` attrs.
        assert!(ts.contains("_ : u32"));
        assert!(ts.contains("& self . embed_tokens"));
        // No `match` block for non-layered accessors — the body is
        // a direct field reference, branchless. No slice index either.
        assert!(!ts.contains("match layer"));
        assert!(!ts.contains("get_unchecked"));
    }

    #[test]
    #[should_panic(expected = "mixes layered and non-layered")]
    fn accessor_methods_panic_on_mixed_layered_and_unlayered() {
        // Pathological: an accessor base with both an indexed and
        // an un-indexed field. Codegen must refuse — there's no
        // sensible single method body for the mix, and silently
        // picking one would mask a solver / Impl bug.
        let ty: TokenStream = quote! { crate::__gpu::layers::RmsNorm };
        let accessors = vec![
            WeightAccessor {
                name: format_ident!("norm"),
                rust_type: ty.clone(),
                kind: crate::weight_vocab::WeightKind::RmsNorm,
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("norm_0"),
                rust_type: ty.clone(),
                kind: crate::weight_vocab::WeightKind::RmsNorm,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(0)))],
            },
        ];
        let _ = emit_weights_accessor_methods(&accessors, &ModelDims::default());
    }

    #[test]
    #[should_panic(expected = "mismatched types")]
    fn accessor_methods_panic_on_type_disagreement_within_a_base() {
        // Two layers under the same base claim different `rust_type`s
        // — codegen invariant violation, not a recoverable case.
        let accessors = vec![
            WeightAccessor {
                name: format_ident!("input_layernorm_0"),
                rust_type: quote! { crate::__gpu::layers::RmsNorm },
                kind: crate::weight_vocab::WeightKind::RmsNorm,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(0)))],
            },
            WeightAccessor {
                name: format_ident!("input_layernorm_1"),
                rust_type: quote! { crate::__gpu::layers::LinearLayer },
                kind: crate::weight_vocab::WeightKind::Linear,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(1)))],
            },
        ];
        let _ = emit_weights_accessor_methods(&accessors, &ModelDims::default());
    }

    #[test]
    fn accessor_methods_empty_input_yields_empty_tokens() {
        // No accessors → no impl block. (Empty `impl Weights {}`
        // would be valid Rust but pointless; the codegen elides it.)
        let ts = emit_weights_accessor_methods(&[], &ModelDims::default()).to_string();
        assert!(ts.is_empty());
    }

    // ── group_accessors_by_base / Vec compression invariants ────────

    fn ln_acc(layer: Option<u64>) -> WeightAccessor {
        let name = match layer {
            Some(n) => format_ident!("input_layernorm_{}", n),
            None => format_ident!("input_layernorm"),
        };
        WeightAccessor {
            name,
            rust_type: quote! { crate::__gpu::layers::RmsNorm },
            kind: crate::weight_vocab::WeightKind::RmsNorm,
            // The layer rides the source weight, as `accessor_base_layer`
            // reads it in production.
            source_weights: vec![(crate::classified::WeightId(0), layer.map(UnrollIndex))],
        }
    }

    #[test]
    fn group_accessors_layered_collapses_to_one_group() {
        let accs: Vec<_> = (0u64..4).map(|n| ln_acc(Some(n))).collect();
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1, "all 4 layers share one base group");
        let g = &groups[0];
        assert_eq!(g.base, "input_layernorm");
        assert!(
            matches!(g.kind, AccessorGroupKind::LayeredContiguous),
            "all entries are indexed and start at layer 0 → LayeredContiguous"
        );
        assert_eq!(g.entries.len(), 4);
        for (i, (layer_opt, _acc)) in g.entries.iter().enumerate() {
            assert_eq!(
                *layer_opt,
                Some(UnrollIndex(i as u64)),
                "contiguous layers 0..N"
            );
        }
    }

    #[test]
    fn group_accessors_unindexed_is_one_entry_with_none() {
        let accs = vec![WeightAccessor {
            name: format_ident!("embed_tokens"),
            rust_type: quote! { crate::__gpu::layers::Embedding },
            kind: crate::weight_vocab::WeightKind::Embedding,
            source_weights: vec![],
        }];
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.base, "embed_tokens");
        assert!(matches!(g.kind, AccessorGroupKind::Unindexed));
        assert_eq!(g.entries.len(), 1);
        assert_eq!(g.entries[0].0, None);
    }

    #[test]
    fn group_accessors_orders_groups_alphabetically() {
        // Order matters: a tied `lm_head` references `embed_tokens`,
        // so `embed_tokens`'s let-binding must precede `lm_head`'s in
        // the load body. Alphabetical ordering preserves that without
        // a special sort.
        let accs = vec![
            WeightAccessor {
                name: format_ident!("lm_head"),
                rust_type: quote! { crate::__gpu::layers::LinearLayer },
                kind: crate::weight_vocab::WeightKind::Linear,
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("embed_tokens"),
                rust_type: quote! { crate::__gpu::layers::Embedding },
                kind: crate::weight_vocab::WeightKind::Embedding,
                source_weights: vec![],
            },
            ln_acc(Some(0)),
            ln_acc(Some(1)),
        ];
        let groups = group_accessors_by_base(&accs);
        let bases: Vec<_> = groups.iter().map(|g| g.base.as_str()).collect();
        assert_eq!(bases, vec!["embed_tokens", "input_layernorm", "lm_head"]);
    }

    #[test]
    #[should_panic(expected = "mixes layered and non-layered")]
    fn group_accessors_panics_on_mixed_layered_unindexed() {
        let accs = vec![ln_acc(None), ln_acc(Some(0))];
        let _ = group_accessors_by_base(&accs);
    }

    #[test]
    #[should_panic(expected = "mismatched types")]
    fn group_accessors_panics_on_type_disagreement() {
        let accs = vec![
            WeightAccessor {
                name: format_ident!("input_layernorm_0"),
                rust_type: quote! { crate::__gpu::layers::RmsNorm },
                kind: crate::weight_vocab::WeightKind::RmsNorm,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(0)))],
            },
            WeightAccessor {
                name: format_ident!("input_layernorm_1"),
                rust_type: quote! { crate::__gpu::layers::LinearLayer },
                kind: crate::weight_vocab::WeightKind::Linear,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(1)))],
            },
        ];
        let _ = group_accessors_by_base(&accs);
    }

    #[test]
    fn group_accessors_sparse_layered_falls_back_to_per_layer_fields() {
        // Layered group with a gap (layers 0 and 2, missing 1) →
        // LayeredSparse, not a Vec. Real-world archs hit sparse
        // layouts (DeepSeek MoE on layers 1..N, future per-window-
        // size attention overrides), so codegen must accept them
        // and emit per-layer fields + match-arm accessors.
        let accs = vec![ln_acc(Some(0)), ln_acc(Some(2))];
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!(matches!(g.kind, AccessorGroupKind::LayeredSparse));
        // Entries preserve the layer indices the group originally
        // had — no padding/None at the gap.
        assert_eq!(
            g.entries.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            vec![Some(UnrollIndex(0)), Some(UnrollIndex(2))]
        );
    }

    #[test]
    fn group_accessors_layered_starting_above_zero_is_sparse() {
        // DeepSeek-V2's `moe` accessor: layer 0 is dense FFN, layers
        // 1..N are MoE — `moe` only exists for 1..N. The Vec-compressed
        // path would be off-by-one (Vec[layer as usize] reads layer
        // L from index L instead of L-1), so non-zero-starting groups
        // also take the sparse fallback.
        let accs = vec![ln_acc(Some(1)), ln_acc(Some(2))];
        let groups = group_accessors_by_base(&accs);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!(matches!(g.kind, AccessorGroupKind::LayeredSparse));
    }

    #[test]
    fn accessor_methods_emit_match_for_sparse_layered_groups() {
        // The sparse path emits per-layer fields and a match-arm
        // accessor that's identical in shape to the legacy
        // pre-Vec-compression emit. Codegen-issued static rows only
        // pass layers that exist, so the catch-all is
        // `unreachable_unchecked`.
        let accs = vec![
            WeightAccessor {
                name: format_ident!("moe_1"),
                rust_type: quote! { crate::__gpu::layers_moe::DeepSeekV2MoELayer },
                kind: crate::weight_vocab::WeightKind::DeepSeekMoe,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(1)))],
            },
            WeightAccessor {
                name: format_ident!("moe_2"),
                rust_type: quote! { crate::__gpu::layers_moe::DeepSeekV2MoELayer },
                kind: crate::weight_vocab::WeightKind::DeepSeekMoe,
                source_weights: vec![(crate::classified::WeightId(0), Some(UnrollIndex(2)))],
            },
        ];
        // No typed dims: a DeepSeek MoE bundle is no handle kind, so its accessor returns the
        // bare layer.
        let ts = emit_weights_accessor_methods(&accs, &ModelDims::default()).to_string();
        assert!(ts.contains("fn moe"));
        assert!(ts.contains("match layer"));
        // `Literal::u32_unsuffixed` emits the bare integer; the
        // arm body resolves to it via the inferred match-arm type.
        assert!(ts.contains("1 => & self . moe_1"));
        assert!(ts.contains("2 => & self . moe_2"));
        assert!(ts.contains("unreachable_unchecked"));
        // Sparse groups don't take the Vec path.
        assert!(!ts.contains("get_unchecked"));
    }

    // ── Vec compression: layered prefix templating ──────────────────

    #[test]
    fn layer_template_replaces_layers_dot_zero_with_runtime_format() {
        let ts =
            layer_templated_prefix_expr("model.layers.0.input_layernorm", None, None).to_string();
        // Routes through the `layer_weight_path` helper so the
        // emitted load_with body has one fn-call per accessor per
        // layer instead of the 5-line `format!()` macro expansion.
        assert!(
            ts.contains("layer_weight_path"),
            "expected layer_weight_path call, got: {ts}"
        );
        assert!(
            ts.contains("\"input_layernorm\""),
            "expected suffix string literal, got: {ts}"
        );
    }

    #[test]
    #[should_panic(expected = "doesn't start with `model.layers.0`")]
    fn layer_template_rejects_unindexed_prefix() {
        // `lm_head` and `model.embed_tokens` are unindexed prefixes;
        // they must never reach the layered template helper. A panic
        // here surfaces a `default_required_weights` bug instead of
        // silently emitting a malformed Vec build.
        let _ = layer_templated_prefix_expr("model.embed_tokens", None, None);
    }

    #[test]
    fn emit_layered_load_body_uses_layer_template_for_rmsnorm() {
        let plan = FieldLoad::RmsNorm("model.layers.0.input_layernorm".to_string(), 1e-5);
        let ts =
            emit_layered_load_body(&plan, 32, 1, false, None, "model.layers", None).to_string();
        // Delegates to the load_layered_rms_norm helper in
        // scratchy-forward-compiler. The closure / collect / Vec annotation
        // the loop used to emit per accessor are now owned by the
        // helper — call sites collapse to one line.
        assert!(
            ts.contains("load_layered_rms_norm"),
            "expected helper call, got: {ts}"
        );
        // Threads (gw, n_layers, suffix, eps) — the suffix is the
        // post-`model.layers.0.` tail, n_layers is the count.
        assert!(ts.contains("\"input_layernorm\""));
        assert!(ts.contains("32"));
        // and bakes the static eps literal.
        assert!(ts.contains("0.00001"));
        // No closure / collect machinery on the call site.
        assert!(!ts.contains(". collect"));
        assert!(!ts.contains("| layer :"));
    }

    #[test]
    fn emit_layered_load_body_binds_locals_for_concat_prefixes() {
        // The `_concat` helpers take `&[&str]` of suffixes; the
        // emitter strips each `model.layers.0.` prefix down to the
        // tail and bakes them as a static `&[…]`. The per-iteration
        // String binds + as_str refs the previous shape needed are
        // now owned by `load_layered_linear_dense_concat`.
        let plan = FieldLoad::LinearConcat(vec![
            "model.layers.0.self_attn.q_proj".to_string(),
            "model.layers.0.self_attn.k_proj".to_string(),
            "model.layers.0.self_attn.v_proj".to_string(),
        ]);
        let ts =
            emit_layered_load_body(&plan, 32, 1, false, None, "model.layers", None).to_string();
        assert!(
            ts.contains("load_layered_linear_dense_concat"),
            "expected helper call, got: {ts}"
        );
        assert!(ts.contains("\"self_attn.q_proj\""));
        assert!(ts.contains("\"self_attn.k_proj\""));
        assert!(ts.contains("\"self_attn.v_proj\""));
        // No per-iteration String bind / .as_str() / closure tokens.
        assert!(!ts.contains("__p_0"));
        assert!(!ts.contains(". as_str ()"));
        assert!(!ts.contains("| layer :"));
    }

    /// At `tp_world_size = 1` every emitted load call must be
    /// byte-identical to the pre-task-#5 build — the existing tests
    /// above pin one direction (helper presence + suffix bake), this
    /// one pins the negative: the `_sharded` variant must NOT appear.
    /// Defends against a future regression that drops the `tp == 1`
    /// short-circuit and silently routes the dense build through
    /// `_sharded` with `world = 1`.
    #[test]
    fn emit_layered_load_body_at_tp_eq_1_emits_no_sharded_call() {
        let plans: &[FieldLoad] = &[
            FieldLoad::LinearDense("model.layers.0.self_attn.q_proj".to_string()),
            FieldLoad::LinearDense("model.layers.0.self_attn.o_proj".to_string()),
            FieldLoad::LinearDense("model.layers.0.input_layernorm".to_string()),
            FieldLoad::LinearConcat(vec![
                "model.layers.0.self_attn.q_proj".to_string(),
                "model.layers.0.self_attn.k_proj".to_string(),
                "model.layers.0.self_attn.v_proj".to_string(),
            ]),
            FieldLoad::Embedding("model.layers.0.embed_tokens".to_string()),
        ];
        for plan in plans {
            let ts =
                emit_layered_load_body(plan, 32, 1, false, None, "model.layers", None).to_string();
            assert!(
                !ts.contains("_sharded"),
                "tp=1 must never emit a `_sharded` helper call (got: {ts})",
            );
            assert!(
                !ts.contains("tp_rank"),
                "tp=1 must not reference `tp_rank` (got: {ts})",
            );
        }
    }

    /// At tp>1, layered `LinearDense` accessors route to the sharded
    /// helper with the right `dim` baked in: column-parallel
    /// (q/k/v/gate/up) → `dim = 0`; row-parallel (o/down) → `dim = 1`.
    /// The shard kind comes from the prefix's last segment via
    /// `tp_lowering::shard_kind_for_dotted_prefix`. A regression that
    /// flipped the dim or routed q_proj as row-parallel would mismatch
    /// the FUF's sharded `<W>::*` constants → kernel shape error at
    /// the first per-rank gemm; this test catches that at codegen time.
    #[test]
    fn emit_layered_load_body_at_tp_gt_1_dispatches_by_shard_kind() {
        // Column-parallel (ShardDim0): q_proj. Expect `dim = 0` lit.
        let q = FieldLoad::LinearDense("model.layers.0.self_attn.q_proj".to_string());
        let ts = emit_layered_load_body(&q, 32, 2, false, None, "model.layers", None).to_string();
        assert!(
            ts.contains("load_layered_linear_dense_sharded"),
            "tp=2 q_proj must route to sharded helper (got: {ts})",
        );
        assert!(
            ts.contains("0usize"),
            "q_proj must be dim=0 column-parallel (got: {ts})",
        );
        assert!(ts.contains("tp_rank"));

        // Row-parallel (ShardDim1): o_proj. Expect `dim = 1` lit.
        let o = FieldLoad::LinearDense("model.layers.0.self_attn.o_proj".to_string());
        let ts = emit_layered_load_body(&o, 32, 2, false, None, "model.layers", None).to_string();
        assert!(
            ts.contains("load_layered_linear_dense_sharded"),
            "tp=2 o_proj must route to sharded helper (got: {ts})",
        );
        assert!(
            ts.contains("1usize"),
            "o_proj must be dim=1 row-parallel (got: {ts})",
        );

        // Replicate (norm, etc.): NO sharded helper, even at tp>1.
        let n = FieldLoad::LinearDense("model.layers.0.input_layernorm".to_string());
        let ts = emit_layered_load_body(&n, 32, 2, false, None, "model.layers", None).to_string();
        assert!(
            !ts.contains("_sharded"),
            "Replicate path must not route to sharded helper at tp>1 (got: {ts})",
        );

        // Concat (always column-parallel). Under metal, which has no tensor-parallel load, the
        // layered helper routes every concat to the stream-free `_concat_packed`, whatever the
        // world; elsewhere a sharded world routes to `_concat_sharded`, which bakes the `world`
        // literal but no `dim` arg (concat-sharded is always dim=0 internally).
        let c = FieldLoad::LinearConcat(vec![
            "model.layers.0.mlp.gate_proj".to_string(),
            "model.layers.0.mlp.up_proj".to_string(),
        ]);
        let ts = emit_layered_load_body(&c, 32, 4, false, None, "model.layers", None).to_string();
        if cfg!(feature = "metal") {
            assert!(
                ts.contains("load_layered_linear_dense_concat_packed") && !ts.contains("_sharded"),
                "metal routes a tp=4 gate_up concat to concat_packed (got: {ts})",
            );
            return;
        }
        assert!(
            ts.contains("load_layered_linear_dense_concat_sharded"),
            "tp=4 gate_up concat must route to concat_sharded (got: {ts})",
        );
        // The macro bakes `tp_world_size` as an unsuffixed integer
        // literal cast to `usize` (e.g. `4 as usize`). Asserting on
        // the bare `4 as usize` token sequence — the unsuffixed form
        // is `proc_macro2::Literal::u8_unsuffixed`'s contract.
        assert!(
            ts.contains("4 as usize"),
            "expected `4 as usize` for baked tp_world_size literal (got: {ts})",
        );
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use crate::fixtures::{arch_configs, config_of, dsl_of, fuf_of, fuf_of_ast, variant_of};

    /// A shim's `load` delegates to its canonical's body — the affine
    /// bit-widths are baked into THAT body, and the shim's own accessor
    /// set is empty (a shim lowers no tape). So a rowless shim
    /// must inherit its canonical's rows or its fingerprint gate is
    /// vacuous and it selects any affine-compatible checkpoint (the
    /// gpt-oss b2/b3-attn4 collision: the b2 shim passed selection and
    /// its delegated bits-2 load died in the element-count assert).
    /// Own rows win when present; canonicals never inherit.
    #[test]
    fn a_rowless_shim_inherits_its_canonicals_affine_rows() {
        let canonical_rows = vec![
            (
                "model.layers.*.mlp.experts.gate_proj.weight".to_string(),
                2,
                64,
            ),
            ("model.layers.*.mlp.router".to_string(), 8, 64),
        ];
        let empty: Vec<(String, u32, u32)> = Vec::new();
        let canonical_mod = Ident::new("canonical_mod", proc_macro2::Span::call_site());
        let shim = WeightsEmitMode::Shim {
            canonical: &canonical_mod,
        };

        // Rowless shim → the canonical's rows.
        assert_eq!(
            effective_fingerprint_rows(&shim, &empty, Some(&canonical_rows)),
            canonical_rows.as_slice()
        );
        // Shim with its own rows keeps them.
        let own = vec![("model.layers.*.mlp.router".to_string(), 2, 64)];
        assert_eq!(
            effective_fingerprint_rows(&shim, &own, Some(&canonical_rows)),
            own.as_slice()
        );
        // A canonical never inherits.
        assert_eq!(
            effective_fingerprint_rows(&WeightsEmitMode::Canonical, &empty, Some(&canonical_rows)),
            empty.as_slice()
        );
    }

    /// Non-MLA arches (Llama / Qwen / etc.) keep the historical
    /// `q_proj` leaf — the fp_leaf selection is purely opt-in for
    /// archs whose manifest declares `q_a_proj`. (The MLA and flat-Q
    /// MLA variants of this pin left with their deepseek-v3 configs,
    /// #721.)
    // FP8-block presets are not claimable under metal (`quant_preset_active`), so no
    // FP8-block variant exists to fingerprint there; the spyre build's test step runs this.
    #[cfg(not(feature = "metal"))]
    #[test]
    fn fp8_block_disambiguation_uses_q_proj_for_non_mla_archs() {
        let dir = arch_configs("qwen3");
        let configs =
            crate::config::load_dir(&dir, &Default::default(), &crate::config::Scope::all())
                .expect("load qwen3 configs");
        let manifest =
            crate::weights_manifest::load_or_empty(&dir).expect("load qwen3 weights manifest");
        let model = configs
            .iter()
            .find(|c| {
                matches!(
                    c.quantization.as_ref().map(|qc| &qc.method),
                    Some(crate::quantization::QuantMethod::Fp8 {
                        block_size: Some(_),
                        ..
                    })
                )
            })
            .expect("at least one Qwen3 FP8-block variant");
        let ts = emit_fingerprint_check(model, &manifest, 1, &[], None).to_string();
        assert!(
            ts.contains("q_proj.weight_scale_inv"),
            "non-MLA FP8-block fingerprint should still use q_proj, got:\n{ts}",
        );
    }

    /// The `rope_theta` literal a variant's fingerprint gates on, if any.
    /// Parsed back out of the emitted token stream so the assertion is about
    /// what the generated code will actually compare, not about our intent.
    fn emitted_rope_theta(model: &crate::config::ModelParams, arch: &str) -> Option<f64> {
        let manifest = crate::weights_manifest::load_or_empty(&arch_configs(arch))
            .expect("load weights manifest");
        let ts = emit_fingerprint_check(model, &manifest, 1, &[], None).to_string();
        // `quote`'s stringification spaces tokens out: `theta - 100000.0`.
        let tail = ts.split("theta - ").nth(1)?;
        let lit: String = tail
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == 'e')
            .collect();
        Some(lit.parse().expect("emitted theta literal parses"))
    }

    /// ⛔ THE REGRESSION THIS GATE EXISTS FOR. `granite-3.0-2b-instruct` and
    /// `granite-3.1-2b-instruct` are byte-identical on every other fingerprint
    /// axis — same hidden/layers/vocab/heads/kv-heads, same (absent) quant
    /// config, and neither declares `rope_scaling`, so the type and hash checks
    /// cannot see them apart. They differ ONLY in `rope_theta` (10⁴ vs 5×10⁶),
    /// which is baked as a literal into each one's `RotaryCache`. Before the
    /// theta gate, a build carrying both — `model/granite`, or `model/all`,
    /// which is CI's scope — served whichever `inventory` registered first, and
    /// a mismatch is silent: shapes agree, the load succeeds, and the model
    /// emits fluent-looking token soup.
    #[test]
    fn granite_3_0_and_3_1_are_separated_by_the_rope_theta_gate() {
        let v30 = config_of("granite", "granite-3.0-2b-instruct");
        let v31 = config_of("granite", "granite-3.1-2b-instruct");

        // The premise: they really are indistinguishable without theta.
        for key in [
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
        ] {
            assert_eq!(
                v30.bounds.get(key),
                v31.bounds.get(key),
                "premise broken: granite 3.0 and 3.1 2b-instruct differ in {key}, so this test \
                 is no longer exercising the collision the theta gate was added for",
            );
        }
        assert!(
            v30.rope_scaling.is_none() && v31.rope_scaling.is_none(),
            "premise broken: one of these declares rope_scaling, so the pre-existing \
             rope_scaling checks would already separate them",
        );

        let t30 = emitted_rope_theta(&v30, "granite").expect("granite 3.0 must gate on rope_theta");
        let t31 = emitted_rope_theta(&v31, "granite").expect("granite 3.1 must gate on rope_theta");
        assert_ne!(
            t30, t31,
            "granite 3.0 and 3.1 2b-instruct must gate on DIFFERENT rope_theta literals — \
             equal literals mean both variants accept the other's checkpoint and the \
             collision is back",
        );
    }

    /// Every checked-in config of an arch whose body reads a LOCAL rotary table states the local
    /// class's base itself — the fact `TextGeometry::rope_local_for` takes, never defaulted to the
    /// global θ. A config that stopped stating it would be a build error; this names it first.
    #[test]
    fn every_local_rotary_config_states_its_local_base() {
        for arch in [
            "gemma3",
            "gemma4",
            "gemma4-moe",
            "gemma4-assistant",
            "modernbert",
        ] {
            let dir = arch_configs(arch);
            let mut n = 0;
            for e in std::fs::read_dir(&dir).expect("arch configs") {
                let path = e.expect("entry").path();
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if !name.ends_with(".json")
                    || ["arch.json", "quantizations.json", "weights.json"].contains(&name)
                    || name.ends_with(".overrides.json")
                {
                    continue;
                }
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                let Some(model) = crate::config::assert_load_or_refuse_by_name(
                    &stem,
                    crate::config::load_file_declared(&path),
                ) else {
                    continue;
                };
                assert!(
                    model.text.rope_local.is_some(),
                    "{arch}/{stem}: no local rotary base stated"
                );
                n += 1;
            }
            assert!(n > 0, "{arch}: no config checked");
        }
    }

    /// A config that omits `rope_theta` computes with its ARCH's default (llama: `LlamaConfig`'s
    /// 10⁴, `configs/llama/arch.json`), and the gate compares against that value — the one the
    /// model runs at — never against nothing.
    #[test]
    fn a_config_omitting_rope_theta_gates_on_its_arch_default() {
        let path = arch_configs("llama").join("llama-2-70b.json");
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        assert!(
            raw.get("rope_theta").is_none(),
            "premise broken: llama-2-70b now states a rope_theta, so it no longer exercises the \
             arch default — repoint this test at another config that omits it",
        );
        let model = config_of("llama", "llama-2-70b");
        assert_eq!(model.text.rope_theta.get(), 10_000.0);
        assert_eq!(emitted_rope_theta(&model, "llama"), Some(10_000.0));
    }

    /// `affine_role_bits` resolved against a variant's own config — the single
    /// resolver the load calls and the fingerprint table both go through.
    fn resolved_role_bits(model: &crate::config::ModelParams, role: &str) -> Option<u32> {
        let program = layout_test_program(&[]);
        crate::quantization::affine_role_bits(
            &program,
            &model.quantization.as_ref().expect("a quant variant").method,
            role,
            None,
        )
    }

    /// ⛔ THE REGRESSION THIS GATE EXISTS FOR (#202), at the layer that decides
    /// it. The two in-tree qwen3-5-moe stems are byte-identical on every other
    /// fingerprint axis — same hidden size, layer count, expert count, vocab, no
    /// `rope_scaling` and no top-level `rope_theta` — and their real checkpoints
    /// differ only in the router gates' width, which 3.6 declares through its
    /// `.overrides.json`. If both resolve the same width, both variants claim
    /// both checkpoints and the loser dequantizes with the wrong stride.
    #[test]
    fn the_two_qwen3_5_moe_stems_resolve_different_gate_widths() {
        let (Some(v3_5), Some(v3_6)) = (
            variant_of("qwen3-5-moe", "qwen3.5-35b-a3b", "mlx-affine-b4-g64-qembed"),
            variant_of("qwen3-5-moe", "qwen3.6-35b-a3b", "mlx-affine-b4-g64-qembed"),
        ) else {
            return;
        };

        // The premise: nothing else can tell them apart.
        for key in [
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "num_experts",
        ] {
            assert_eq!(
                v3_5.bounds.get(key),
                v3_6.bounds.get(key),
                "premise broken: the two stems now differ in {key}, so this test is no longer \
                 exercising the collision the width gate was added for",
            );
        }

        for role in ["mlp.gate", "mlp.shared_expert_gate"] {
            assert_eq!(
                resolved_role_bits(&v3_5, role),
                Some(4),
                "3.5 is uniformly 4-bit; {role} must resolve to 4",
            );
            assert_eq!(
                resolved_role_bits(&v3_6, role),
                Some(8),
                "3.6 ships {role} at 8-bit — that is what its `.overrides.json` declares, and \
                 what `affine_tensors_of` puts in the fingerprint table",
            );
        }
    }

    /// Gemma-4-MoE's router is 8-bit while the rest of the checkpoint is 4-bit.
    /// That width must come from the arch's DECLARED override, so the loader and
    /// the fingerprint read one value — it used to be hardcoded inside both
    /// backends' `Gemma4RouterOps::load`, leaving the compiled bit map silently
    /// disagreeing with the tensor the loader reads.
    #[test]
    fn the_gemma4_moe_router_width_comes_from_the_declared_override() {
        let Some(v) = variant_of("gemma4-moe", "gemma-4-26b-a4b-it", "mlx-affine-b4-g64") else {
            return;
        };
        assert_eq!(
            resolved_role_bits(&v, "router.proj"),
            Some(8),
            "router.proj must resolve to the 8 bits the checkpoint ships; without the declared \
             override it falls back to the 4-bit section default and the gate rejects \
             gemma-4-26b-a4b-it-4bit",
        );
        assert_eq!(
            resolved_role_bits(&v, "self_attn.q_proj"),
            Some(4),
            "the override must be scoped to the router, not widen the whole model",
        );
    }

    /// A drafter's capture is the compiler's (`draft_capture`), not an arch's, so every decoder
    /// arch hosts one: a target with no drafter beside it unrolls to the tiles its DSL alone does,
    /// and one whose drafter reads layer 1 captures that layer and no other.
    #[test]
    fn every_decoder_arch_hosts_a_drafter_capture() {
        use crate::config::LayerSet;
        use crate::draft_capture::DRAFT_CAPTURE_LAYERS;
        for (arch, stem) in [
            ("commandr", "c4ai-command-r-v01"),
            ("gemma2", "gemma2-27b"),
            ("gemma3", "gemma-3-12b-it"),
            ("gemma4", "gemma-4-12b-it"),
            ("gemma4-moe", "gemma-4-26b-a4b-it"),
            ("glm4-moe", "glm-4.5-air"),
            ("gpt-oss", "gpt-oss-20b"),
            ("granite", "granite-3.0-2b-base"),
            ("llama", "llama-3.2-1b"),
            ("mistral", "mistral-7b-instruct-v0.3"),
            ("mixtral", "mixtral-8x7b-instruct-v0.1"),
            ("nemotron-h", "nemotron-3-nano-30b-a3b"),
            ("nemotron-h-latent", "nemotron-3-super-120b-a12b"),
            ("phi3", "phi-3-5-mini-instruct"),
            ("qwen2", "qwen2.5-0.5b"),
            ("qwen2-moe", "slimed-qwen-3"),
            ("qwen3", "qwen3-0.6b"),
            ("qwen3-5", "qwen3.5-0.8b"),
            ("qwen3-5-moe", "qwen3.6-35b-a3b"),
            ("qwen3-moe", "qwen3-30b-a3b-instruct-2507"),
        ] {
            let path = arch_configs(arch).join(format!("{stem}.json"));
            let mut model = crate::config::load_file_declared(&path).expect("load");
            let (_, written) = fuf_of_ast(arch, &model, &dsl_of(arch));
            let (_, compiled) = fuf_of(arch, &model);
            assert_eq!(
                format!("{:?}", written.nodes),
                format!("{:?}", compiled.nodes),
                "{arch}: a target with no drafter must unroll as its DSL does",
            );
            let set = LayerSet::of([1]);
            model
                .layer_sets
                .insert(DRAFT_CAPTURE_LAYERS.to_string(), set);
            let (_, captured) = fuf_of(arch, &model);
            let captures = (captured.nodes.iter())
                .filter(|n| n.op == OpKind::CaptureHidden)
                .count();
            assert_eq!(
                captures, 1,
                "{arch}: a drafter reading layer 1 captures it alone"
            );
        }
    }

    /// The emitted body for one Nemotron-H variant, rows as given.
    fn nemotron_fingerprint(
        arch: &str,
        model: &crate::config::ModelParams,
        rows: &[(String, u32, u32)],
    ) -> String {
        let manifest =
            crate::weights_manifest::load_or_empty(&arch_configs(arch)).expect("manifest");
        let (program, fuf) = fuf_of(arch, model);
        let probe = expert_input_probe(&program, &fuf, model);
        emit_fingerprint_check(model, &manifest, 1, rows, probe.as_ref()).to_string()
    }

    /// A recurrent hybrid holds KV in its attention layers only: its KV layers are exactly its
    /// `attention_layers`, a proper subset the pool compacts to (`KV_CACHE_LAYERS`), and the
    /// TurboQuant threshold reads their bytes — Nano's 6 and Super's 8 attention layers of 2 KV
    /// heads x 128 are 6 and 8 KiB/token, below `MIN_KV_BYTES_PER_TOKEN`, where every one of
    /// their 52 / 88 layers would read as 52 / 88 KiB. Layers that all hold KV are no subset.
    #[test]
    #[cfg(feature = "metal")]
    fn a_recurrent_hybrids_kv_layers_are_its_attention_layers() {
        use scratchy_ir::{DenseReason, KvGeometry, TqBits, codec_for};
        let preset = "mlx-affine-b4-g64-qembed";
        for (arch, stem, kib) in [
            ("nemotron-h", "nemotron-3-nano-30b-a3b", Some(6)),
            ("nemotron-h-latent", "nemotron-3-super-120b-a12b", Some(8)),
            ("nemotron-h", "nemotron-3-nano-30b-a3b-parity-tiny", None),
            (
                "nemotron-h",
                "nemotron-3-nano-30b-a3b-mamba-parity-tiny",
                None,
            ),
        ] {
            let Some(model) = variant_of(arch, stem, preset) else {
                continue;
            };
            let (_, fuf) = fuf_of(arch, &model);
            let kv = KvCacheLayers::of(&fuf);
            let attention: Vec<u32> = model.layer_sets["attention_layers"]
                .members()
                .iter()
                .map(|&l| l as u32)
                .collect();
            let layers = model.bounds["num_hidden_layers"] as usize;
            assert!(!attention.is_empty() && attention.len() < layers, "{stem}");
            assert_eq!(
                kv.proper_subset(layers),
                Some(attention.as_slice()),
                "{stem}"
            );
            if let Some(kib) = kib {
                let geometry = KvGeometry {
                    num_layers: kv.count(),
                    num_kv_heads: model.bounds["num_key_value_heads"] as usize,
                    head_dim: model.bounds["head_dim"] as u32,
                    global_head_dim: model.bounds["head_dim"] as u32,
                    latent: false,
                    has_kv_cache: kv.any(),
                };
                assert_eq!(
                    codec_for(geometry, TqBits::new(4)),
                    Err(DenseReason::SmallKv {
                        bytes_per_token: kib * 1024
                    }),
                    "{stem}"
                );
            }
            assert_eq!(
                kv.proper_subset(kv.count()),
                None,
                "{stem}: every layer holds KV"
            );
        }
    }

    /// Nemotron-H (plain) and Nemotron-H with LatentMoE report ONE architecture string, so the
    /// fingerprint must tell them apart by tensors alone — and at EQUAL hidden size, where the
    /// embedding shape cannot (the two parity-tiny stems: hidden 128, vocab 512 both). The
    /// experts' input width does: the latent bank's `fc1` packs `moe_latent_size`, the plain
    /// one's `hidden_size`. Probed at the last MoE layer, at that layer's resolved `fc1` width.
    #[test]
    fn the_latent_and_plain_nemotron_moe_probe_different_expert_input_widths() {
        let preset = "mlx-affine-b4-g64-qembed";
        let probe = |arch: &str, stem: &str| -> Option<_> {
            let model = variant_of(arch, stem, preset)?;
            let (program, fuf) = fuf_of(arch, &model);
            let last_moe = *model.layer_sets["moe_layers"].members().last().unwrap();
            let probe = expert_input_probe(&program, &fuf, &model).expect("an ungated bank");
            assert_eq!(
                probe.tensors,
                vec![
                    format!("backbone.layers.{last_moe}.mixer.switch_mlp.fc1.weight"),
                    format!("backbone.layers.{last_moe}.mixer.experts.0.fc1.weight"),
                    format!("backbone.layers.{last_moe}.mixer.experts.0.up_proj.weight"),
                ],
                "{stem}: probed at the last MoE layer, in the loaders' spelling order"
            );
            Some((model.bounds["hidden_size"], probe.packed_cols))
        };
        let (Some((nano_h, nano)), Some((super_h, sup))) = (
            probe("nemotron-h", "nemotron-3-nano-30b-a3b-parity-tiny"),
            probe(
                "nemotron-h-latent",
                "nemotron-3-super-120b-a12b-parity-tiny",
            ),
        ) else {
            return;
        };
        assert_eq!(
            nano_h, super_h,
            "premise: the tiny pair shares its hidden size"
        );
        assert_eq!(nano, 128 * 4 / 32, "plain: hidden 128 at 4-bit");
        assert_eq!(
            sup,
            64 * 4 / 32,
            "latent: moe_latent_size 64 at layer 7's 4-bit fc1"
        );

        // The real checkpoints: Nano-4bit is uniform; Super's last MoE layer (87) is a 4/4 bundle
        // on the 1024-wide latent.
        assert_eq!(
            probe("nemotron-h", "nemotron-3-nano-30b-a3b")
                .expect("uniform variant loads")
                .1,
            2688 * 4 / 32
        );
        // Super's own checkpoint is an OptiQ mixed-bit variant: it asserts
        // under a backend that can lower it, and refuses (skipping) elsewhere.
        if let Some((_, sup_real)) = probe("nemotron-h-latent", "nemotron-3-super-120b-a12b") {
            assert_eq!(sup_real, 1024 * 4 / 32);
        }
        assert!(
            variant_of("qwen3-5-moe", "qwen3.5-35b-a3b", preset)
                .map(|model| {
                    expert_input_probe(
                        &layout_test_program(&[]),
                        &Fuf { nodes: Vec::new() },
                        &model,
                    )
                })
                .is_none_or(|probe| probe.is_none()),
            "a gated bank's leaf names live in the loaders; no probe is claimed for it",
        );
    }

    /// The Super OptiQ variant's fingerprint as emitted: the format-marker probes read a layer
    /// that HAS the fingerprint leaf (layer 0 is a Mamba block, so a `layers.0` probe was inert),
    /// the embedding gate expects the table packed at the per-module map's 4 bits (the map names
    /// `backbone.embeddings`, not `embed_tokens`), and the experts' input width is checked.
    #[test]
    fn the_super_optiq_fingerprint_probes_tensors_the_checkpoint_has() {
        let arch = "nemotron-h-latent";
        let Some(model) = variant_of(
            arch,
            "nemotron-3-super-120b-a12b",
            "mlx-affine-b4-g64-qembed",
        ) else {
            return;
        };
        let first_attn = *model.layer_sets["attention_layers"]
            .members()
            .first()
            .unwrap();
        let last_attn = *model.layer_sets["attention_layers"]
            .members()
            .last()
            .unwrap();
        assert_ne!(first_attn, 0, "premise: layer 0 is not an attention block");
        let q_rows = vec![("backbone.layers.*.mixer.q_proj.weight".to_string(), 4, 64)];
        let ts = nemotron_fingerprint(arch, &model, &q_rows);
        let squash = ts.replace(' ', "");
        assert!(
            squash.contains(&format!(
                "\"backbone.layers.{last_attn}.mixer.q_proj.scales\""
            )),
            "the group-size probe reads the last attention layer's q_proj: {ts}"
        );
        assert!(
            squash.contains("\"backbone.embeddings.weight\"")
                && squash.contains(&format!("shape[1]=={}", 4096 * 4 / 32)),
            "the embedding table is gated packed at 4 bits: {ts}"
        );
        assert!(
            squash.contains("\"backbone.layers.87.mixer.switch_mlp.fc1.weight\"")
                && squash.contains(&format!("Some(&{})", 1024 * 4 / 32)),
            "the experts' input width is gated: {ts}"
        );
        assert!(
            squash.contains("\"backbone.layers.87.mixer.experts.0.up_proj.weight\"")
                && squash.contains("__EXPERT_INPUT_SEEN"),
            "the width gate falls through to the per-expert spellings (Nemotron-3 Super's MTP \
             sidecar ships `experts.<e>.up_proj`, no pre-stacked `switch_mlp`): {ts}"
        );
        assert!(
            !squash.contains("ships_affine_row"),
            "q_proj is affine, so its group-size probe is the positive sniff already"
        );

        // The dense base rejects an MLX checkpoint by the `.scales` marker, at the FIRST layer
        // that has the leaf.
        let dense = config_of(arch, "nemotron-3-super-120b-a12b");
        let manifest =
            crate::weights_manifest::load_or_empty(&arch_configs(arch)).expect("manifest");
        let ts = emit_fingerprint_check(&dense, &manifest, 1, &[], None).to_string();
        let marker = format!("layers.{first_attn}.");
        assert!(
            ts.contains("q_proj.scales") && ts.contains(&marker) && !ts.contains("layers.0."),
            "no format marker may probe the Mamba layer 0: {ts}"
        );
    }

    /// The Nemotron-3 Super head's config is its target's verbatim, so its own block schedule
    /// restores through its dir's per-stem override, all three facts at once: the target's
    /// `hybrid_override_pattern` replaced by the checkpoint's `mtp_hybrid_override_pattern`
    /// (`*E`, an attention layer then a LatentMoE one), the head's layer count as
    /// `num_hidden_layers` — the schedule's length must equal it while the sets derive, and
    /// the schema's replacing param reads the same value back from `mtp_num_hidden_layers`,
    /// so the post-schema comparison KEEPS the derived sets instead of dropping them as
    /// another model's — and the target's count preserved as `target_num_hidden_layers`
    /// (the head's KV pool is the target's; the gemma4-assistant override carries its
    /// target's the same way). Its probe then names the sidecar's own layout: bare
    /// per-layer names (the worker lends `mtp.*` under them), per-expert spelling included.
    ///
    /// The 120B is the head's one target: its lm_head read spans the target's own vocabulary
    /// (no qwen-style prefix constant), so a second target of a different vocabulary cannot
    /// share the head arch — the head-facts registration refuses variants that disagree on
    /// `lm_head_rows` — which is why the parity-tiny (512 tokens) declares no
    /// `mtp_num_hidden_layers` and compiles no head.
    #[test]
    fn the_nemotron_head_restores_its_own_schedule_through_its_override() {
        let stem = "nemotron-3-super-120b-a12b";
        let model = crate::config::load_head_file(
            &arch_configs("nemotron-h-latent").join(format!("{stem}.json")),
            &arch_configs("nemotron-h-mtp"),
        )
        .unwrap_or_else(|e| panic!("load {stem} as a head: {e}"));
        assert_eq!(model.bounds["num_hidden_layers"], 2);
        assert_eq!(model.bounds["target_num_hidden_layers"], 88);
        assert_eq!(
            model.layer_sets["attention_layers"].members(),
            &[0u64],
            "the head's schedule is `*E`"
        );
        assert_eq!(model.layer_sets["moe_layers"].members(), &[1u64]);
        assert!(
            model.layer_sets["mamba_layers"].members().is_empty()
                && model.layer_sets["mlp_layers"].members().is_empty()
        );
        let (program, fuf) = fuf_of("nemotron-h-mtp", &model);
        let probe = expert_input_probe(&program, &fuf, &model).expect("an ungated bank");
        assert_eq!(
            probe.tensors,
            vec![
                "layers.1.mixer.switch_mlp.fc1.weight",
                "layers.1.mixer.experts.0.fc1.weight",
                "layers.1.mixer.experts.0.up_proj.weight",
            ],
            "the head's MoE layer is layer 1, its tensors bare post-lend"
        );
    }

    /// An MLX per-module variant whose fingerprint leaf is DENSE (the experts-only parity map)
    /// skips the group-size probe, and then must positively see one of its affine tensors packed
    /// — else a dense checkpoint satisfies its width gate vacuously (`NoVariantMatched` is the
    /// right answer, not a loader failure).
    #[test]
    fn an_experts_only_per_module_variant_sniffs_its_affine_tensors_positively() {
        let arch = "nemotron-h";
        let Some(model) = variant_of(
            arch,
            "nemotron-3-nano-30b-a3b-parity-tiny",
            "mlx-affine-b4-g64-qembed",
        ) else {
            return;
        };
        let expert_rows = vec![(
            "backbone.layers.1.mixer.switch_mlp.fc1.weight".to_string(),
            4,
            64,
        )];
        let squash = nemotron_fingerprint(arch, &model, &expert_rows).replace(' ', "");
        assert!(
            squash.contains("ships_affine_row(__AFFINE_WIDTHS)"),
            "{squash}"
        );
        assert!(
            !squash.contains("mixer.q_proj.scales\")"),
            "no group-size probe: {squash}"
        );
    }

    /// `affine_tensors_of` must report the width THE LOAD CALL GETS, so the
    /// fingerprint cannot assert something the loader won't do. The MoE arm's
    /// gate is the #202 role; the Gemma router is the one that used to be
    /// hardcoded.
    #[test]
    fn affine_tensors_of_reports_the_width_the_load_call_receives() {
        let moe = FieldLoad::SharedFusedMoe {
            prefix: "model.layers.0.mlp".to_string(),
            num_experts: 8,
            top_k: 2,
            moe_intermediate_size: 128,
            shared_expert_intermediate_size: 0,
            hidden_size: 128,
            affine: Some((
                64,
                crate::quantization::ExpertBits::Gated {
                    gate: 4,
                    up: 4,
                    down: 4,
                },
            )),
            gate_bits: Some(8),
        };
        assert_eq!(
            affine_tensors_of(&moe),
            vec![("model.layers.0.mlp.gate.weight".to_string(), 8, 64)],
            "the router gate must carry `gate_bits` (8), NOT the expert width (4) — they are \
             different tensors at different widths",
        );

        // Nemotron-3-Super's 4/2 split bundle: its dense router claims no width, and its
        // pre-stacked ungated experts are listed each at its own — the only rows that separate
        // the OptiQ variant from a uniform 4-bit one.
        let ungated = FieldLoad::SharedFusedMoe {
            prefix: "backbone.layers.8.mixer".to_string(),
            num_experts: 512,
            top_k: 22,
            moe_intermediate_size: 2688,
            shared_expert_intermediate_size: 0,
            hidden_size: 1024,
            affine: Some((
                64,
                crate::quantization::ExpertBits::Ungated { fc1: 4, fc2: 2 },
            )),
            gate_bits: None,
        };
        assert_eq!(
            affine_tensors_of(&ungated),
            vec![
                (
                    "backbone.layers.8.mixer.switch_mlp.fc1.weight".to_string(),
                    4,
                    64
                ),
                (
                    "backbone.layers.8.mixer.switch_mlp.fc2.weight".to_string(),
                    2,
                    64
                ),
            ],
        );

        let router = FieldLoad::GemmaRouter {
            prefix: "model.layers.0.router".to_string(),
            num_experts: 128,
            hidden_size: 2816,
            affine: Some((64, 8)),
        };
        assert_eq!(
            affine_tensors_of(&router),
            vec![("model.layers.0.router.proj.weight".to_string(), 8, 64)],
        );
        // A dense router (the fp8 26b's `ignore`d `router.proj`, verbatim
        // bf16) lists NOTHING — no fabricated width row, so the fingerprint's
        // positive sniff cannot demand `.scales` the checkpoint never ships.
        let dense_router = FieldLoad::GemmaRouter {
            prefix: "model.layers.0.router".to_string(),
            num_experts: 128,
            hidden_size: 2816,
            affine: None,
        };
        assert_eq!(affine_tensors_of(&dense_router), vec![]);

        let leaf = FieldLoad::LinearAffine {
            prefix: "model.layers.0.self_attn.q_proj".to_string(),
            group_size: 64,
            bits: 4,
            in_features: 128,
        };
        assert_eq!(
            affine_tensors_of(&leaf),
            vec![("model.layers.0.self_attn.q_proj.weight".to_string(), 4, 64)],
        );

        // gpt-oss lists its expert stacks at the EXPERT width. The
        // b2-attn4/b3-attn4 preset pair agrees on every other fingerprinted
        // width (attention 4, router 8, embed/lm_head 4), so these rows are
        // the only discriminator between the two variants — without them the
        // pair fingerprints identically and the first-consulted variant
        // claims the other's checkpoint (#316: the b2 variant's 180 packed
        // cols vs the b3 checkpoint's 270).
        let gptoss = FieldLoad::GptOssMoe {
            prefix: "model.layers.0.mlp".to_string(),
            num_experts: 128,
            top_k: 4,
            intermediate_size: 2880,
            hidden_size: 2880,
            affine: Some((64, 3)),
            gate_bits: Some(8),
        };
        assert_eq!(
            affine_tensors_of(&gptoss),
            vec![
                ("model.layers.0.mlp.router.weight".to_string(), 8, 64),
                (
                    "model.layers.0.mlp.experts.gate_proj.weight".to_string(),
                    3,
                    64
                ),
                (
                    "model.layers.0.mlp.experts.up_proj.weight".to_string(),
                    3,
                    64
                ),
                (
                    "model.layers.0.mlp.experts.down_proj.weight".to_string(),
                    3,
                    64
                ),
            ],
            "the expert stacks must be listed at the expert width (`affine`'s \
             bits), NOT the router's — they are the only width separating the \
             attn4-router8 presets from each other",
        );

        // Nothing is claimed for a dense load.
        assert!(affine_tensors_of(&FieldLoad::LinearDense("x".into())).is_empty());

        // ⛔ EVERY row names the `.weight` TENSOR, not the role. The runtime
        // matcher compares patterns against full tensor names, so a row that
        // stopped at the role would match nothing and the gate would go silently
        // dead — it fails open, so no test of the widths themselves would notice.
        for fl in [&moe, &router, &leaf, &gptoss] {
            for (name, _, _) in affine_tensors_of(fl) {
                assert!(
                    name.ends_with(".weight"),
                    "row `{name}` must name the .weight tensor",
                );
            }
        }
    }

    /// Compression keeps the table at one row per role when every layer agrees,
    /// and MUST fall back to concrete per-layer rows when they don't.
    ///
    /// ⛔ THE MLX MIXED/DYNAMIC (OptiQ) CASE. Those checkpoints give individual
    /// layers their own widths through the `per_module` map, so starring the
    /// layer index would claim layer 0's width for all of them — the exact kind
    /// of unbacked claim that makes a fingerprint false-reject. No OptiQ
    /// checkpoint is in tree today, which is why this is covered here rather
    /// than left to one.
    #[test]
    fn compression_stars_a_uniform_role_and_keeps_a_per_layer_one_concrete() {
        let uniform: Vec<(String, u32, u32)> = (0..3)
            .map(|l| (format!("model.layers.{l}.self_attn.q_proj.weight"), 4, 64))
            .collect();
        assert_eq!(
            compress_affine_rows(uniform),
            vec![("model.layers.*.self_attn.q_proj.weight".to_string(), 4, 64)],
            "a role every layer agrees on collapses to ONE starred row",
        );

        // Layer 1 is 8-bit (sensitivity-aware quant); 0 and 2 stay 4-bit.
        let mixed: Vec<(String, u32, u32)> = vec![
            ("model.layers.0.self_attn.q_proj.weight".to_string(), 4, 64),
            ("model.layers.1.self_attn.q_proj.weight".to_string(), 8, 64),
            ("model.layers.2.self_attn.q_proj.weight".to_string(), 4, 64),
        ];
        let got = compress_affine_rows(mixed.clone());
        assert_eq!(
            got.len(),
            3,
            "a role whose width varies by layer must keep every concrete row — starring it \
             would claim one layer's width for all of them. Got: {got:?}",
        );
        for row in &mixed {
            assert!(got.contains(row), "missing {row:?} from {got:?}");
        }
        assert!(
            !got.iter().any(|(n, _, _)| n.contains('*')),
            "no starred row may survive per-layer variation",
        );
    }

    /// The emitted gate calls the runtime helper with the collected rows. A
    /// variant with no affine tensors emits nothing at all.
    #[test]
    fn the_affine_width_gate_is_emitted_from_the_collected_rows() {
        let model = config_of("llama", "smollm2-135m");
        let manifest =
            crate::weights_manifest::load_or_empty(&arch_configs("llama")).expect("load manifest");

        let rows = vec![("model.layers.*.mlp.gate.weight".to_string(), 8, 64)];
        let ts = emit_fingerprint_check(&model, &manifest, 1, &rows, None).to_string();
        assert!(
            ts.contains("affine_widths_agree") && ts.contains("model.layers.*.mlp.gate.weight"),
            "the gate must pass the collected rows to the runtime helper, got:\n{ts}",
        );

        let ts_none = emit_fingerprint_check(&model, &manifest, 1, &[], None).to_string();
        assert!(
            !ts_none.contains("affine_widths_agree"),
            "a variant with no affine tensors must emit no width gate",
        );
    }

    /// The gate reads `hf.rope_theta` through `if let Some(..)`, so a caller
    /// that supplies nothing stays on the variant's baked value rather than
    /// being rejected. The cuda worker relies on this for GGUF, whose metadata
    /// routinely disagrees with the canonical `config.json`.
    #[test]
    fn the_theta_gate_is_permissive_when_the_caller_supplies_none() {
        let model = config_of("llama", "smollm2-135m");
        let manifest =
            crate::weights_manifest::load_or_empty(&arch_configs("llama")).expect("load manifest");
        let ts = emit_fingerprint_check(&model, &manifest, 1, &[], None).to_string();
        let gate = ts
            .split("rope_theta")
            .nth(1)
            .expect("smollm2-135m must gate on rope_theta");
        // `if let Some (theta) = hf . rope_theta && ..` — the `Some` binding is
        // what makes `None` fall through instead of rejecting.
        assert!(
            ts.contains("if let Some") && gate.contains("&&"),
            "the theta gate must be an `if let Some(..) = hf.rope_theta && ..` so a `None` \
             from the caller falls through, got:\n{ts}",
        );
    }

    /// The embed's packed-width literal the emitted fingerprint gates on,
    /// parsed back out of the token stream (same discipline as
    /// [`emitted_rope_theta`]: assert on what the generated code compares,
    /// not on our intent).
    fn emitted_embed_packed_width(model: &crate::config::ModelParams, arch: &str) -> Option<usize> {
        let manifest = crate::weights_manifest::load_or_empty(&arch_configs(arch))
            .expect("load weights manifest");
        let ts = emit_fingerprint_check(model, &manifest, 1, &[], None).to_string();
        // `match gw . tensor_shape_any ("model.embed_tokens.weight") { ..
        // shape [0] == 201088usize && shape [1] == 360usize .. }`
        // — anchor on `[1]` so the `>= 2usize` rank check can't be
        // mistaken for the width literal.
        let tail = ts.split("model.embed_tokens.weight").nth(1)?;
        let after_idx1 = tail.split("[1]").nth(1)?;
        let lit: String = after_idx1
            .chars()
            .skip_while(|c| *c != '=')
            .skip_while(|c| *c == '=' || *c == ' ')
            .take_while(|c| c.is_ascii_digit())
            .collect();
        lit.parse().ok()
    }

    /// ⛔ gpt-oss's `attn4-router8` presets: the embed's packed width must
    /// come from the OVERRIDE, not the section default. The b2-attn4 preset
    /// declares a 2-bit expert default with `embed_tokens: 4`; hidden 2880
    /// packs to 2880*4/32 = 360 at 4-bit but 2880*2/32 = 180 at the default.
    /// A regression that ignores the embed override computes the 180 literal
    /// while the loader dequantizes 4-bit — the fingerprint then
    /// false-rejects every proper checkpoint (or worse, accepts at the
    /// wrong width if the shapes happened to agree).
    #[test]
    fn gpt_oss_attn4_preset_gates_the_embed_at_the_override_width() {
        let (Some(proper), Some(plain)) = (
            variant_of(
                "gpt-oss",
                "gpt-oss-120b",
                "mlx-affine-b2-g64-attn4-router8-qembed",
            ),
            variant_of("gpt-oss", "gpt-oss-120b", "mlx-affine-b2-g64-qembed"),
        ) else {
            return;
        };

        // Premise: both variants are the same model at the same group size —
        // they differ ONLY in the per-role bit map.
        assert_eq!(
            proper.bounds.get("hidden_size"),
            plain.bounds.get("hidden_size"),
            "premise broken: the two variants disagree on hidden_size",
        );
        let proper_m = proper.quantization.as_ref().map(|qc| qc.method.clone());
        let plain_m = plain.quantization.as_ref().map(|qc| qc.method.clone());
        let (
            Some(crate::quantization::QuantMethod::Affine {
                bits: proper_bits,
                group_size: proper_gs,
                bits_overrides: proper_ov,
                ..
            }),
            Some(crate::quantization::QuantMethod::Affine {
                bits: plain_bits,
                group_size: plain_gs,
                bits_overrides: plain_ov,
                ..
            }),
        ) = (proper_m, plain_m)
        else {
            panic!("both variants must parse as MLX-affine");
        };
        assert_eq!((proper_bits, proper_gs), (plain_bits, plain_gs));
        assert!(
            plain_ov.is_empty() && !proper_ov.is_empty(),
            "premise broken: the presets differ in more than their bit map",
        );

        let proper_w = emitted_embed_packed_width(&proper, "gpt-oss")
            .expect("the b2-attn4 variant must gate on the embed shape");
        let plain_w = emitted_embed_packed_width(&plain, "gpt-oss")
            .expect("the plain b2 variant must gate on the embed shape");
        assert_eq!(
            proper_w, 360,
            "the 4-bit embed override must pack 2880*4/32 = 360, got {proper_w}",
        );
        assert_eq!(
            plain_w, 180,
            "the plain b2 default must pack 2880*2/32 = 180, got {plain_w}",
        );
    }

    /// Build a minimal `Program` whose `WeightTable` carries one
    /// entry — `lm_head` — at WeightId(0). Other tests can intern
    /// additional weights to push `lm_head` off slot 0; this helper
    /// keeps the encoder/decoder layout test focused on what matters
    /// (the dotted weight name carried at the FUF terminal).
    fn layout_test_program(weight_paths: &[&[&str]]) -> crate::classified::Program {
        let mut weights = crate::classified::WeightTable::default();
        for path in weight_paths {
            let segments: Vec<String> = path.iter().map(|s| (*s).to_string()).collect();
            let _ = weights.intern_str(segments);
        }
        crate::classified::Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            dim_casts: Default::default(),
            prelude: crate::classified::Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        }
    }

    fn shape_2d() -> Vec<crate::shape::Dim> {
        use crate::shape::Dim;
        vec![Dim::Lit(4), Dim::Lit(16)]
    }

    /// Decoder layout: FUF ends in `gemm(<tile>, lm_head)`.
    /// `backbone_layout` reports `Decoder` and the carried
    /// `(TileId, slot)` is the lm_head Gemm's hidden-state input.
    #[test]
    fn backbone_layout_recognizes_decoder_terminator() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        let program = layout_test_program(&[&["lm_head", "weight"]]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ],
        };
        match backbone_layout(&fuf, &program) {
            BackboneLayout::Decoder => {}
            BackboneLayout::Encoder => panic!("expected Decoder layout, got Encoder"),
        }
    }

    /// A decoder whose LOGITS GET POST-PROCESSED is still a decoder.
    ///
    /// granite divides logits by `logits_scaling` (terminal `Mul` with a
    /// scalar operand); gemma2/gemma-4 apply `final_logit_softcapping`
    /// (terminal `TanhSoftCap`); command-r has `logit_scale`. Reading any of
    /// them as an ENCODER bakes `METAL_VOCAB_SIZE = hidden_size`, and the
    /// argmax kernel then scans `hidden_size` of `vocab_size` logits — 2048
    /// of granite's 49155 — so greedy decode can only reach low-id tokens and
    /// emits `" * **P*h*o*t*o*s*y*n*t*h*e*"`. Sampling does not use that
    /// kernel, so this is invisible to any sampled test.
    #[test]
    fn backbone_layout_walks_past_logits_post_ops() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        // `embed -> gemm(_, lm_head) -> <post-ops...>` for each shape a real
        // arch produces. Every one must report Decoder with the Gemm's
        // hidden-state input as the backbone output.
        for (name, post) in [
            ("granite logits_scaling", vec![OpKind::Mul]),
            ("gemma final_logit_softcapping", vec![OpKind::TanhSoftCap]),
            ("softcap then scale", vec![OpKind::TanhSoftCap, OpKind::Mul]),
        ] {
            let program = layout_test_program(&[&["lm_head", "weight"]]);
            let mut nodes = vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ];
            for (k, op) in post.iter().enumerate() {
                let id = TileId(2 + k as u32);
                // A scalar `Mul` carries its constant as a bare Scalar input;
                // that is what distinguishes it from a tensor*tensor Mul.
                let mut inputs = vec![FufInput::Tile {
                    id: TileId(1 + k as u32),
                    slot: 0,
                }];
                if matches!(op, OpKind::Mul) {
                    inputs.push(FufInput::Scalar(0.125));
                }
                nodes.push(FufNode {
                    id,
                    op: *op,
                    inputs,
                    outputs: vec![shape_2d()],
                });
            }
            let fuf = Fuf { nodes };
            match backbone_layout(&fuf, &program) {
                BackboneLayout::Decoder => {}
                BackboneLayout::Encoder => {
                    panic!("{name}: a decoder with logits post-ops was read as an ENCODER")
                }
            }
        }
    }

    /// Encoder layout: FUF ends in something other than
    /// `gemm(_, lm_head)` (here a plain `Add` — same shape ModernBERT
    /// produces at the encoder output). `backbone_layout` returns
    /// `Encoder`; the FUF's terminal is itself the backbone output.
    #[test]
    fn backbone_layout_recognizes_encoder_terminator() {
        use crate::classified::{ExternKind, OpKind};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};

        let program = layout_test_program(&[]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Add,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ],
        };
        assert!(
            matches!(backbone_layout(&fuf, &program), BackboneLayout::Encoder),
            "FUF terminating in Add must classify as Encoder",
        );
    }

    /// Decoder terminator with a tp>1 AllGather appended after the
    /// lm_head Gemm. `backbone_layout` walks past the AllGather to
    /// find the underlying Gemm and reports `Decoder` with the right
    /// hidden-state input.
    #[test]
    fn backbone_layout_walks_past_allgather_to_decoder_gemm() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        let program = layout_test_program(&[&["lm_head", "weight"]]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(2),
                    op: OpKind::AllGather,
                    inputs: vec![FufInput::Tile {
                        id: TileId(1),
                        slot: 0,
                    }],
                    outputs: vec![shape_2d()],
                },
            ],
        };
        match backbone_layout(&fuf, &program) {
            BackboneLayout::Decoder => {}
            BackboneLayout::Encoder => panic!("expected Decoder past AllGather"),
        }
    }

    /// Terminal Gemm whose second input is some non-`lm_head` weight
    /// (e.g. a plain `down_proj`) is NOT a decoder lm_head row —
    /// classify it as Encoder so the lowering doesn't try to split a
    /// nonexistent lm_head off.
    #[test]
    fn backbone_layout_non_lm_head_gemm_is_encoder() {
        use crate::classified::{ExternKind, OpKind, WeightId};
        use crate::fuf::{Fuf, FufInput, FufNode, TileId};
        use crate::quantization::StorageFormat;

        let program = layout_test_program(&[&["mlp", "down_proj", "weight"]]);
        let fuf = Fuf {
            nodes: vec![
                FufNode {
                    id: TileId(0),
                    op: OpKind::Embed,
                    inputs: vec![FufInput::Extern {
                        kind: ExternKind::InputIds,
                        index: None,
                    }],
                    outputs: vec![shape_2d()],
                },
                FufNode {
                    id: TileId(1),
                    op: OpKind::Gemm,
                    inputs: vec![
                        FufInput::Tile {
                            id: TileId(0),
                            slot: 0,
                        },
                        FufInput::Weight {
                            id: WeightId(0),
                            index: None,
                            storage: StorageFormat::Dense,
                        },
                    ],
                    outputs: vec![shape_2d()],
                },
            ],
        };
        assert!(
            matches!(backbone_layout(&fuf, &program), BackboneLayout::Encoder),
            "Gemm whose 2nd input is not `lm_head` must classify as Encoder",
        );
    }
}
