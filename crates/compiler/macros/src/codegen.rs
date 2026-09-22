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
//!
//! Emission-per-subgraph is delegated to
//! [`crate::impl_lib::Implementation::emit_call`]: codegen walks
//! the LOOP's waves and asks each subgraph's bound impl to emit
//! its own tokens. New kernels / new ops extend the library, not
//! this file.

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Ident;

use crate::assignment::WorkloadAssignments;
use crate::classified::{Expr, OpKind, Program, Stmt, UnrollIndex, WeightId};
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, TileId};
use crate::impl_lib::ImplementationLibrary;
use crate::interpreter_codegen::{ArchOpcodes, emit_bucket_static_slice, lower_bucket};
use crate::schedule::WorkloadLoops;
use crate::weight_vocab::{SlotMap, WeightAccessor, WeightSlot};

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
fn safetensors_prefix(
    program: &Program,
    decoder_safetensors_prefix: Option<&str>,
    vision_layout: Option<&crate::config::VisionSafetensorsLayout>,
    id: WeightId,
    index: Option<UnrollIndex>,
) -> String {
    // DSL idents can't start with a digit, so paths like
    // `merger.mlp.0` are written `merger.mlp_0` in the body /
    // manifest; translate `_<digit>` suffixes back to `.<digit>`
    // for the on-disk safetensors key (which uses Python-attribute
    // dotted form, including numeric submodule indices).
    let segs: Vec<String> = program
        .weights
        .path(id)
        .iter()
        .map(|seg| translate_digit_suffix(seg))
        .collect();
    let mut joined = segs.join(".");
    // DSL-leaf → disk-leaf rename (Gemma4: `self_attn.q_proj_global`
    // shares the on-disk leaf `self_attn.q_proj` with the sliding
    // class at a different shape; LocateAnything: the projector's
    // `linear_1`/`linear_2` disk leaves can't be named in the DSL —
    // a trailing `_<digit>` reads as a layer index — so
    // `mm.proj_in`/`mm.proj_out` rename here). Longest-suffix match
    // on the dotted DSL path, applied BEFORE the vision subtree
    // resolution below so renamed segments flow into subtree paths.
    let mut segs = segs;
    for (dsl_leaf, disk_leaf) in &program.weight_leaf_renames {
        if joined == *dsl_leaf {
            joined = disk_leaf.clone();
            segs = joined.split('.').map(str::to_string).collect();
            break;
        }
        if let Some(head) = joined.strip_suffix(&format!(".{dsl_leaf}")) {
            joined = format!("{head}.{disk_leaf}");
            segs = joined.split('.').map(str::to_string).collect();
            break;
        }
    }
    let is_vision = matches!(program.prelude, crate::classified::Prelude::Vision);
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
    // Multimodal arches that nest the text decoder under
    // `language_model.<...>` (Gemma3-MM). Variant configs set
    // `decoder_safetensors_prefix: "language_model"`; we prepend it to
    // every text-decoder key (lm_head, model.layers.*, model.<...>).
    // Text-only and Qwen-style VL leave it `None` → byte-equivalent
    // `model.<...>` / `lm_head` keys.
    let key = match (index, joined.as_str()) {
        (_, "lm_head") => "lm_head".to_string(),
        (Some(l), _) => format!("model.layers.{l}.{joined}"),
        (None, _) => format!("model.{joined}"),
    };
    match decoder_safetensors_prefix {
        None => key,
        Some(prefix) => {
            // Tolerate a trailing dot in the config value.
            let prefix = prefix.trim_end_matches('.');
            // `lm_head` is always at the safetensors top level — VL repos
            // (Qwen3.5, Gemma3-MM, ...) don't nest it under the decoder
            // prefix because it sits beside the `model` namespace, not
            // inside it. Both checkpoint orderings (the official
            // `model.language_model.*` and the mlx-community
            // `language_model.model.*`) keep `lm_head.weight` at the root.
            if key == "lm_head" {
                return key;
            }
            match key.strip_prefix("model") {
                // Replace-the-`model`-root form: the prefix already names
                // the `model` root (e.g. Qwen3.5-VL `model.language_model`),
                // so the decoder's keys sit DIRECTLY under it —
                // `model.language_model.embed_tokens`,
                // `model.language_model.layers.N.*`, `model.language_model.norm`
                // — NOT nested as `<prefix>.model.<key>`. Splice the prefix
                // in for the leading `model` segment.
                Some(rest) if prefix.starts_with("model") => format!("{prefix}{rest}"),
                // Namespace-wrapper form (Gemma3-MM `language_model`): the
                // whole standard `model.<key>` / `lm_head` namespace nests
                // under the prefix → `language_model.model.<key>`.
                _ => format!("{prefix}.{key}"),
            }
        }
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
        /// `K / (32 / bits)`, rejecting a checkpoint quantized at different
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
        /// `.weight` width is `K / (32 / bits)`. See [`FieldLoad::LinearAffine`].
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
        /// the Metal `load_affine` call site. `None` → Dense BF16
        /// path → cuda `load` call site.
        affine: Option<(u32, u32)>,
        /// Bit-width for the router `{prefix}.gate` dequant, resolved
        /// from the affine preset's `bits_overrides` (mlx MoE presets
        /// quantize the router gate at 8-bit while experts stay 4-bit).
        /// `Some` exactly when `affine` is `Some`; falls back to the
        /// expert bits when no override matches.
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
        /// `Some((group_size, bits))` when MLX-affine int4 — see
        /// [`FieldLoad::FusedMoe::affine`].
        affine: Option<(u32, u32)>,
        /// Router `{prefix}.gate` dequant bit-width — see
        /// [`FieldLoad::FusedMoe::gate_bits`]. `Some` iff `affine` is `Some`.
        gate_bits: Option<u32>,
    },
    /// Gemma-4 router bundle (`GemmaMoe` op, base `router`). Dispatches to
    /// the metal-only `GemmaRouterLayer::load`. The router.proj 8-bit dequant
    /// is hardcoded inside the loader, so no `affine` field is threaded here;
    /// `group_size` is the model's affine group width (64).
    GemmaRouter {
        prefix: String,
        num_experts: usize,
        hidden_size: usize,
        group_size: u32,
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
}

/// Emit the `GptqLayout` token stream that selects the loader's
/// on-disk branch (native `.qweight` vs compressed-tensors
/// `.weight_packed`). Used by both the single and concat GPTQ
/// FieldLoad arms; the generated `MarlinLinear::load_gptq{,_concat}`
/// consume it directly.
fn gptq_layout_ts(layout: crate::quantization::GptqLayout) -> TokenStream {
    match layout {
        crate::quantization::GptqLayout::Qweight => {
            quote! { crate::__gpu::layers_quant::GptqLayout::Qweight }
        }
        crate::quantization::GptqLayout::WeightPacked => {
            quote! { crate::__gpu::layers_quant::GptqLayout::WeightPacked }
        }
    }
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

fn read_deepseek_moe_cfg(model: &ModelParams) -> DeepSeekMoeCfg {
    let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
    let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
    let n_routed_experts = v
        .get("n_routed_experts")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| model.bounds.get("n_routed_experts").copied().unwrap_or(64) as usize);
    let n_shared_experts = v
        .get("n_shared_experts")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| model.bounds.get("n_shared_experts").copied().unwrap_or(2) as usize);
    let top_k = v
        .get("num_experts_per_tok")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            model
                .bounds
                .get("num_experts_per_tok")
                .copied()
                .unwrap_or(6) as usize
        });
    let moe_intermediate_size = v
        .get("moe_intermediate_size")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .unwrap_or_else(|| {
            model
                .bounds
                .get("moe_intermediate_size")
                .copied()
                .unwrap_or(1536) as usize
        });
    let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2048) as usize;
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
        .map(|(id, idx)| {
            safetensors_prefix(
                program,
                model.arch.decoder_prefix.as_deref(),
                model.arch.safetensors.as_ref(),
                *id,
                *idx,
            )
        })
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
        let cfg = read_deepseek_moe_cfg(model);
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
        let cfg = read_deepseek_moe_cfg(model);
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
        let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
        let num_experts = v
            .get("num_local_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model.bounds.get("num_local_experts").copied().unwrap_or(8) as usize
            });
        let top_k = v
            .get("num_experts_per_tok")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("num_experts_per_tok")
                    .copied()
                    .unwrap_or(2) as usize
            });
        let intermediate_size = model
            .bounds
            .get("intermediate_size")
            .copied()
            .unwrap_or(14336) as usize;
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(4096) as usize;
        // Probe storage format: if expert weights are MLX-affine int4
        // (mlx-community 4bit Mixtral / Qwen-MoE), emit the metal
        // `load_affine` call site; otherwise emit cuda `load`.
        // Same check the `MetalFusedMoeImpl::matches` uses via
        // `affine_gs_bits`. Source weights all share storage format
        // (the macro asserts this elsewhere for AffineQuantLinear).
        let only_src = accessor.source_weights[0].0;
        // Resolve at this accessor's layer index — MLX mixed/dynamic quant
        // (OptiQ) varies expert bits per numbered layer; emit_group_let
        // gathers the per-layer bits across the group.
        let only_idx = accessor.source_weights[0].1;
        let affine = match crate::quantization::storage_format_for_weight(
            program, fuf, only_src, only_idx, model,
        ) {
            crate::quantization::StorageFormat::Affine { group_size, bits } => {
                // The bundle root subtree-matches the router gate first; the
                // routed experts' own (per-layer, OptiQ) bits live under
                // switch_mlp / experts. Uniform/preset paths fall back.
                let expert_bits = model
                    .quantization
                    .as_ref()
                    .and_then(|qc| {
                        crate::quantization::affine_moe_expert_bits(
                            program, &qc.method, &prefix, only_idx,
                        )
                    })
                    .unwrap_or(bits);
                Some((group_size, expert_bits))
            }
            _ => None,
        };
        // Router gate bit-width: the mlx MoE presets quantize `{prefix}.gate`
        // at 8-bit while experts stay 4-bit — via preset `bits_overrides`
        // OR (OptiQ) the per-module map. `affine_role_bits` honors both;
        // fall back to expert bits.
        let gate_bits = affine.map(|(_, expert_bits)| {
            let gate_path = format!("{prefix}.gate");
            model
                .quantization
                .as_ref()
                .and_then(|qc| {
                    crate::quantization::affine_role_bits(program, &qc.method, &gate_path, only_idx)
                })
                .unwrap_or(expert_bits)
        });
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
        let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
        let num_experts = v
            .get("num_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| model.bounds.get("num_experts").copied().unwrap_or(60) as usize);
        let top_k = v
            .get("num_experts_per_tok")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("num_experts_per_tok")
                    .copied()
                    .unwrap_or(4) as usize
            });
        let moe_intermediate_size = v
            .get("moe_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("moe_intermediate_size")
                    .copied()
                    .unwrap_or(1408) as usize
            });
        let shared_expert_intermediate_size = v
            .get("shared_expert_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("shared_expert_intermediate_size")
                    .copied()
                    .unwrap_or(0) as usize
            });
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2048) as usize;
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
        // Resolve at this accessor's layer index (per-layer expert bits for
        // OptiQ). `only_idx` also picks the gate's per-layer entry.
        let only_idx = accessor.source_weights[0].1;
        let affine = match crate::quantization::storage_format_for_weight(
            program, fuf, only_src, only_idx, model,
        ) {
            crate::quantization::StorageFormat::Affine { group_size, bits } => {
                let expert_bits = model
                    .quantization
                    .as_ref()
                    .and_then(|qc| {
                        crate::quantization::affine_moe_expert_bits(
                            program, &qc.method, &prefix, only_idx,
                        )
                    })
                    .unwrap_or(bits);
                Some((group_size, expert_bits))
            }
            _ => None,
        };
        // Router gate bit-width: see the `FusedMoe` arm — `affine_role_bits`
        // resolves the gate via preset `bits_overrides` OR (OptiQ) the
        // per-module map; fall back to expert bits.
        let gate_bits = affine.map(|(_, expert_bits)| {
            let gate_path = format!("{prefix}.gate");
            model
                .quantization
                .as_ref()
                .and_then(|qc| {
                    crate::quantization::affine_role_bits(program, &qc.method, &gate_path, only_idx)
                })
                .unwrap_or(expert_bits)
        });
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
        let num_experts = model.bounds.get("num_experts").copied().unwrap_or(128) as usize;
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2816) as usize;
        // group_size from the affine quant config (gemma4-moe ships gs=64).
        let group_size = match model.quantization.as_ref().map(|qc| &qc.method) {
            Some(crate::quantization::QuantMethod::Affine { group_size, .. }) => *group_size,
            _ => 64,
        };
        return FieldLoad::GemmaRouter {
            prefix,
            num_experts,
            hidden_size,
            group_size,
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
        let num_experts = model.bounds.get("num_experts").copied().unwrap_or(128) as usize;
        let top_k = model
            .bounds
            .get("num_experts_per_tok")
            .copied()
            .unwrap_or(8) as usize;
        let moe_intermediate_size = model
            .bounds
            .get("moe_intermediate_size")
            .copied()
            .unwrap_or(704) as usize;
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2816) as usize;
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

    if is_deepseek_v2_moe {
        assert_eq!(
            prefixes.len(),
            1,
            "DeepSeekV2MoELayer accessor `{}` with {} sources (expected 1 per layer)",
            accessor.name,
            prefixes.len(),
        );
        let prefix = prefixes.into_iter().next().unwrap();
        // Read MoE params from the model config JSON.
        let src = std::fs::read_to_string(&model.source_path).unwrap_or_default();
        let v: serde_json::Value = serde_json::from_str(&src).unwrap_or(serde_json::Value::Null);
        let n_routed_experts = v
            .get("n_routed_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model.bounds.get("n_routed_experts").copied().unwrap_or(64) as usize
            });
        let n_shared_experts = v
            .get("n_shared_experts")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| model.bounds.get("n_shared_experts").copied().unwrap_or(2) as usize);
        let top_k = v
            .get("num_experts_per_tok")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("num_experts_per_tok")
                    .copied()
                    .unwrap_or(6) as usize
            });
        let moe_intermediate_size = v
            .get("moe_intermediate_size")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                model
                    .bounds
                    .get("moe_intermediate_size")
                    .copied()
                    .unwrap_or(1536) as usize
            });
        let hidden_size = model.bounds.get("hidden_size").copied().unwrap_or(2048) as usize;
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
        // sigmoid routing when scoring_func="sigmoid" AND topk_method="noaux_tc"
        // (DeepSeek V3 / Kimi K2). Matches Python vLLM's condition.
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
        return FieldLoad::DeepSeekV2Moe {
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
                    // `.weight` width against `K / (32 / bits)` and reject a
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
/// store gains zero-centered, effective gain `1 + weight`), the standard
/// `RmsNorm` / `FusedAddRmsNorm` metal kernels fold `weight + 1.0`. The
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

fn tie_word_embeddings(model: &ModelParams) -> bool {
    // `ModelParams::tie_word_embeddings` is populated by the config
    // loader from the HF JSON; read directly rather than re-parsing
    // the file here.
    model.tie_word_embeddings
}

/// Aggregate every unique WeightAccessor across every workload
/// tape_index's SFUF. Errors on name collisions with conflicting
/// `rust_type`s.
fn collect_accessors(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    _model_for_trace: &ModelParams,
) -> Result<Vec<WeightAccessor>, TokenStream> {
    // name → (first-seen accessor, rust_type string for collision check).
    let mut by_name: BTreeMap<String, (WeightAccessor, String)> = BTreeMap::new();
    let mut conflicts: Vec<String> = Vec::new();

    for sfuf in sfufs.per_workload.values() {
        for sg in sfuf.subgraphs() {
            let imp_id = sfuf
                .impl_of(sg)
                .expect("solver committed an impl for every subgraph");
            let claimed = sfuf.tiles_in_subgraph(sg);
            let imp = lib.get(imp_id);
            for acc in imp.required_weights(&claimed, fuf, program) {
                let key = acc.name.to_string();
                let ty_str = acc.rust_type.to_string();
                by_name
                    .entry(key.clone())
                    .and_modify(|(_, existing_ty)| {
                        if *existing_ty != ty_str {
                            conflicts.push(format!(
                                "Weights field `{key}` declared with \
                                 conflicting types: `{existing_ty}` vs `{ty_str}`"
                            ));
                        }
                    })
                    .or_insert((acc.clone(), ty_str));
            }
        }
    }

    if !conflicts.is_empty() {
        let msg = conflicts.join("\n");
        return Err(quote! { compile_error!(#msg); });
    }
    Ok(by_name.into_values().map(|(a, _)| a).collect())
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
fn emit_fingerprint_check(
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    tp_world_size: u8,
) -> TokenStream {
    let num_hidden_layers = *model
        .bounds
        .get("num_hidden_layers")
        .unwrap_or_else(|| panic!("model `{}` missing `num_hidden_layers`", model.source_stem));
    let hidden_size = *model
        .bounds
        .get("hidden_size")
        .unwrap_or_else(|| panic!("model `{}` missing `hidden_size`", model.source_stem));
    let vocab_size = *model
        .bounds
        .get("vocab_size")
        .unwrap_or_else(|| panic!("model `{}` missing `vocab_size`", model.source_stem));

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
    // Per-arch decoder root for fingerprint-tensor names. MUST agree
    // with [`safetensors_prefix`]'s prefix handling (replace-vs-prepend)
    // — the fingerprint looks up `{dec_root}.embed_tokens.weight` /
    // `{dec_root}.layers.N.q_proj.weight`, and if that name doesn't
    // match what the loader actually reads, every variant rejects and
    // `try_load` returns `Ok(None)` → `ArchNotSupported`.
    let dec_root: String = match model.arch.decoder_prefix.as_deref() {
        None => "model".to_string(),
        Some(prefix) => {
            let prefix = prefix.trim_end_matches('.');
            if prefix.starts_with("model") {
                // Replace-the-`model`-root form (Qwen3.5-VL
                // `model.language_model`): decoder root IS the prefix.
                prefix.to_string()
            } else {
                // Namespace-wrapper form (Gemma3-MM `language_model`):
                // `language_model.model`.
                format!("{prefix}.model")
            }
        }
    };
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
    let fp_leaf: &str = fp_leaf_owned.as_str();
    let last_tensor = format!("{dec_root}.layers.{last_layer}.{fp_leaf}.{suffix}");
    // MLX-affine `.scales` sibling of `last_tensor`. The affine group-size
    // gate must probe a layer that ACTUALLY HAS `fp_leaf` (= `self_attn.q_proj`).
    // It MUST NOT hardcode `layers.0`: hybrid arches (Qwen3.5 Gated-DeltaNet,
    // Mamba/Jamba) make layer 0 a linear-attention layer with NO `self_attn.q_proj`,
    // so `layers.0.self_attn.q_proj.scales` is absent there and the gate would
    // reject every variant → `ArchNotSupported`. `last_layer` is the same layer
    // the present-check (`last_tensor`) already requires, so its `.scales`
    // sibling is guaranteed present for the affine checkpoint.
    let last_scales_tensor = format!("{dec_root}.layers.{last_layer}.{fp_leaf}.scales");
    let one_past_tensor = format!("{dec_root}.layers.{num_hidden_layers}.{fp_leaf}.{suffix}");
    let opposite_tensor = format!("{dec_root}.layers.0.{fp_leaf}.{opposite_suffix}");
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
    let bnb4_marker_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight.absmax");
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
    let fp8_marker_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale");
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
    let mlx_marker_tensor = format!("{dec_root}.layers.0.{fp_leaf}.scales");
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
    // time (`scratchy-target-cuda/src/ggml.rs::gguf_shard_kind_for_hf_name`).
    // At tp>1 the per-rank embed shape is `[vocab_size / tp, hidden]`,
    // so the fingerprint's vocab literal must match the sharded dim-0.
    // Safetensors variants keep the full tensor in memory (sharding
    // happens inside the codegen-emitted `_sharded` load helpers AFTER
    // `fingerprint_matches` runs), so their vocab literal stays whole.
    let vocab_for_fp = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(crate::quantization::QuantMethod::Ggml) if tp_world_size > 1 => {
            vocab_size / (tp_world_size as u64)
        }
        _ => vocab_size,
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
    // Emitted ONLY when this variant's config declares a theta. 19 in-tree configs
    // don't (gemma3/gemma4/qwen3.5 carry per-attention-class thetas nested under
    // `rope_parameters`; modernbert and llama-2-70b omit it and take HF's implicit
    // 10⁴ default) — for those there is no declared value to compare against, and
    // synthesizing the default here would invent a constraint the config never
    // stated and false-reject a checkpoint that spells `10000.0` out.
    //
    // Resolution order mirrors `rotary_load`'s exactly, MINUS its
    // `.unwrap_or(10000.0)` fallback, so the gate can never claim a value the
    // config didn't state.
    //
    // The tolerance is relative and computed HERE, at expansion, so the emitted
    // code is a single compare against two literals — and so a config spelling
    // `1000000` where the checkpoint spells `1000000.0` cannot trip it. It is far
    // tighter than any real theta gap (the closest in-tree pair differs by 2×).
    let rope_theta_check: TokenStream = match model
        .scalars
        .get("rope_theta")
        .copied()
        .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
    {
        Some(theta) => {
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
        }
        None => quote! {},
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
            // regardless of variant registration order.
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
            let g_idx_tensor_owned = format!("{dec_root}.layers.0.self_attn.q_proj.g_idx");
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
                    format!("{dec_root}.layers.0.self_attn.q_proj.input_scale");
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
                    format!("{dec_root}.layers.0.self_attn.q_proj.input_scale");
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
            let inv_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale_inv");
            let scale_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale");
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
            let inv_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale_inv");
            let scale_tensor = format!("{dec_root}.layers.0.{fp_leaf}.weight_scale");
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
    // shape is `[vocab_size, hidden_size]` is the embedding table; use
    // its key (with `model.` prefix + `.weight` suffix) as the
    // fingerprint sniff. Fall back to the llama-style path when no
    // entry matches, preserving the previous behavior for any arch
    // whose manifest predates this generalization.
    let embed_path: String = manifest
        .entries
        .iter()
        .find(|(_, shape)| {
            shape.len() == 2
                && matches!(&shape[0], crate::shape::Dim::Bound(s) if s == "vocab_size")
                && matches!(&shape[1], crate::shape::Dim::Bound(s) if s == "hidden_size")
        })
        .map(|(k, _)| format!("{dec_root}.{k}.weight"))
        .unwrap_or_else(|| format!("{dec_root}.embed_tokens.weight"));
    let embed_path_lit = proc_macro2::Literal::string(embed_path.as_str());

    // Hidden-size shape gate for the embedding fingerprint sniff.
    // Three on-disk shapes for `embed_tokens`:
    //   * dense `[vocab, hidden]` — every non-affine variant, AND
    //     the older mlx-affine convention for untied checkpoints
    //     (e.g. Llama-3-8B-Instruct-4bit).
    //   * packed `[vocab, hidden / pack_factor]` (pack_factor =
    //     32 / bits, so 8 for bits=4) — every mlx-affine TIED
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
            model.tie_word_embeddings
                || *quantize_embed
                || per_module.iter().any(|(k, _)| k.ends_with("embed_tokens"))
        }
        _ => false,
    };
    let embed_hidden_lit: TokenStream = match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(method @ crate::quantization::QuantMethod::Affine { .. }) if embed_expects_packed => {
            // Use the embed's OWN bits, not the section default: MLX
            // mixed/dynamic checkpoints (OptiQ) pack `embed_tokens` at
            // 8-bit (hidden/4) while the default is 4-bit (hidden/8).
            let bits = crate::quantization::affine_embed_bits(method).unwrap_or(4);
            let pack_factor = 32u64 / (bits as u64);
            let packed = hidden_size / pack_factor;
            let lit = proc_macro2::Literal::usize_unsuffixed(packed as usize);
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
        /// shared `GpuWeights` API, so the same body sniffs cuda,
        /// metal, and spyre (host) checkpoints identically.
        #[cfg(any(feature = "cuda", feature = "metal", feature = "spyre"))]
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

/// Emit the `Weights` struct definition (or alias) + its `load` +
/// `fingerprint_matches` free fns.
#[allow(clippy::too_many_arguments)]
fn emit_weights_struct(
    program: &Program,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    // M2b: when the arch is tape-scheduled, the accessor set comes
    // from the SHARED TAPE. Both sets are still computed and the
    // emitted field/type/source signature asserted equal — the
    // transition gate; the tape's is what gets emitted.
    tape_accessors: Option<&[WeightAccessor]>,
    lib: &ImplementationLibrary,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    mode: WeightsEmitMode<'_>,
    tp_world_size: u8,
    emit_fingerprint: bool,
) -> TokenStream {
    let isel_accessors = match collect_accessors(program, fuf, sfufs, lib, model) {
        Ok(a) => a,
        Err(err) => return err,
    };
    let accessors = match tape_accessors {
        Some(tape) => {
            let sig = |accs: &[WeightAccessor]| {
                group_accessors_by_base(accs)
                    .iter()
                    .map(|g| {
                        (
                            g.base.clone(),
                            g.rust_type.to_string().replace(' ', ""),
                            g.entries.len(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                sig(tape),
                if isel_accessors.is_empty() {
                    sig(tape)
                } else {
                    sig(&isel_accessors)
                },
                "model `{}`: tape accessor set emits a DIFFERENT Weights struct than \
                 instruction selection",
                model.source_stem,
            );
            tape.to_vec()
        }
        None => isel_accessors,
    };

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
            // concrete per-accessor host FieldLoad (cuda + metal). A target that
            // runs an embedded fused program loads + dequantizes weights itself
            // (no host FieldLoad), so the dense accessor is fine for any storage.
            if !ok && cfg!(any(feature = "cuda", feature = "metal")) {
                let dotted = program.weights.path(*wid).join(".");
                let msg = format!(
                    "model `{stem}`: weight `{dotted}` has storage ({fmt:?}) that \
                     doesn't match accessor `{name}` (type `{ty}`). Add a quant-aware \
                     Impl + FieldLoad arm for this pair.",
                    stem = model.source_stem,
                    name = a.name,
                );
                return quote! { compile_error!(#msg); };
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
            let ty = &g.rust_type;
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
    // Did the solver keep the `embed_tokens` as the packed `AffineQuantEmbedding`
    // (metal's fused gather+dequant) or dequantize it to a dense `Embedding`
    // (spyre/cuda host path)? A tied `lm_head` follows that representation.
    let embed_is_affine_repr = accessors.iter().any(|a| {
        let n = a.name.to_string();
        (n == "embed_tokens" || n.ends_with("_embed_tokens"))
            && a.rust_type
                .to_string()
                .replace(' ', "")
                .contains("AffineQuantEmbedding")
    });
    let plans: Vec<FieldLoad> = accessors
        .iter()
        .map(|a| plan_field_load(a, program, fuf, model, manifest, embed_is_affine_repr))
        .collect();
    // Per-arch decoder root for embed_tokens probes etc. `model` for
    // text-only and Qwen-style VL, `<prefix>.model` for arches whose
    // variant config sets `decoder_safetensors_prefix` (Gemma3-MM nests
    // text decoder weights under `language_model.<...>`).
    let dec_root_for_emit: String = match model.arch.decoder_prefix.as_deref() {
        Some(prefix) => format!("{prefix}.model"),
        None => "model".to_string(),
    };
    let embed_tokens_weight_path: String = format!("{dec_root_for_emit}.embed_tokens.weight");
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
        let num_hidden_layers = *model.bounds.get(layer_count_key).unwrap_or_else(|| {
            panic!(
                "model `{}` has `__packed_splits__` but no `{}` bound",
                model.source_stem, layer_count_key,
            )
        }) as usize;
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
    // Vision encoders skip the fingerprint emission entirely — they
    // route through the hand-written `ScratchyMmRegistration` instead
    // of the inventory-based arch dispatcher, so `fingerprint_matches`
    // is unreachable. Skipping also avoids the `num_hidden_layers` /
    // `hidden_size` / `vocab_size` panics in `emit_fingerprint_check`
    // — vision configs (`vision_*` + `d_model` only) lack those keys.
    let fingerprint_method = if emit_fingerprint {
        emit_fingerprint_check(model, manifest, tp_world_size)
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

    // Compute dtype the rotary cache must match the model's compute dtype:
    // the rope kernel dispatches on Q/K activation dtype but reinterprets
    // cos/sin bytes through that same plan, so a BF16 cos/sin against F16
    // activations reads the table through the wrong exponent width
    // (5 vs 8 bits).
    //
    // Read at RUNTIME from `embed_tokens.weight`'s on-disk dtype — that
    // tensor is always present and always in the model's compute dtype
    // (never quantized). Baking from the manifest's `torch_dtype` is
    // unsafe: AWQ checkpoints frequently override the base model's dtype
    // (Qwen2.5 base ships as bf16; the `*-Instruct-AWQ` variants ship as
    // f16), and the per-AWQ-variant `quant_config.json` does not change
    // the manifest's compile-time `torch_dtype` literal. Mismatch produces
    // grammatical-but-incoherent output (rope rotations applied through
    // the wrong exponent layout). The manifest's `torch_dtype` survives
    // only as the fallback when the embed tensor isn't visible in the
    // weights table — same pattern used by the BNB4 / FP8 preludes above.
    let rope_dtype_fallback: TokenStream = match model.torch_dtype.as_deref() {
        Some("float16" | "fp16" | "f16" | "half") => {
            quote! { crate::__gpu::dtype::DType::F16 }
        }
        _ => quote! { crate::__gpu::dtype::DType::BF16 },
    };
    let rotary_prelude: TokenStream = quote! {
        let __rope_dtype = gw
            .tensor_info(#embed_tokens_weight_path)
            .map(|(_, dt)| dt)
            .unwrap_or(#rope_dtype_fallback);
    };
    let _rope_dtype: TokenStream = quote! { __rope_dtype };

    let rotary_local_field: TokenStream = if uses_rotary_local {
        quote! { pub rotary_local: crate::__gpu::rotary::RotaryCache, }
    } else {
        quote! {}
    };

    let rotary_local_load: TokenStream = if uses_rotary_local {
        let head_dim = *model
            .bounds
            .get("head_dim")
            .expect("model must have head_dim for RotaryLocal") as usize;
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for RotaryLocal")
            as usize;
        let local_theta = model
            .scalars
            .get("rope_local_base_freq")
            .copied()
            .or_else(|| model.bounds.get("rope_local_base_freq").map(|&v| v as f64))
            .or_else(|| model.scalars.get("rope_theta").copied())
            .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
            .expect("model must have rope_local_base_freq for RotaryLocal");
        quote! {
            let rotary_local = unsafe {
                crate::__gpu::rotary::RotaryCache::new_from_stream(
                    #head_dim,
                    #max_pos,
                    #local_theta,
                    None,
                    crate::__gpu::dtype::DType::BF16,
                    stream,
                )
            }?;
        }
    } else {
        quote! {}
    };

    let rotary_local_load_metal: TokenStream = if uses_rotary_local {
        let head_dim = *model
            .bounds
            .get("head_dim")
            .expect("model must have head_dim for RotaryLocal") as usize;
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for RotaryLocal")
            as usize;
        let local_theta = model
            .scalars
            .get("rope_local_base_freq")
            .copied()
            .or_else(|| model.bounds.get("rope_local_base_freq").map(|&v| v as f64))
            .or_else(|| model.scalars.get("rope_theta").copied())
            .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
            .expect("model must have rope_local_base_freq for RotaryLocal");
        quote! {
            let rotary_local = crate::__gpu::rotary::RotaryCache::new_from_gpuweights(
                gw,
                #head_dim,
                #max_pos,
                #local_theta,
                None,
                crate::__gpu::dtype::DType::BF16,  // bf16 cos/sin (HEAD-original)
            )?;
        }
    } else {
        quote! {}
    };

    let rotary_local_init: TokenStream = if uses_rotary_local {
        quote! { rotary_local, }
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
        quote! { pub rotary: crate::__gpu::rotary::RotaryCache, }
    } else {
        quote! {}
    };

    let rotary_load: TokenStream = if uses_rotary {
        let head_dim = *model
            .bounds
            .get("head_dim")
            .expect("model must have head_dim for Rotary") as usize;
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for Rotary")
            as usize;
        // HF's implicit default when `rope_theta` is omitted (e.g.
        // `llama-2-70b.json`). Matches transformers' LlamaConfig.
        let rope_theta = model
            .scalars
            .get("rope_theta")
            .copied()
            .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
            .unwrap_or(10000.0);
        // `partial_rotary_factor == 1.0` is the full-rotary identity;
        // route it through `new_from_stream` instead of
        // `new_partial_from_stream` with `rotary_dim == head_dim`
        // so models that spell this out (Phi-4-reasoning) don't
        // exercise the partial-rotary kernel path unnecessarily.
        let partial = model
            .scalars
            .get("partial_rotary_factor")
            .copied()
            .filter(|&f| (f - 1.0).abs() > 1e-9);
        let rotary_dim_lit: Option<TokenStream> = partial.map(|f| {
            let rd = f * head_dim as f64;
            assert!(
                (rd - rd.round()).abs() < 1e-9,
                "partial_rotary_factor {f} * head_dim {head_dim} = {rd} is not integer",
            );
            let rd = rd.round() as usize;
            quote! { #rd }
        });
        // For YaRN (DeepSeek V2 MLA), the rope portion uses `qk_rope_head_dim`
        // rather than the full `head_dim`. Pull it from bounds if present.
        let yarn_rope_head_dim: Option<usize> =
            model.bounds.get("qk_rope_head_dim").map(|&v| v as usize);
        let scaling = model.rope_scaling.clone();
        // For MLA models (DeepSeek V2/V3 flat), the rope portion uses
        // `qk_rope_head_dim` rather than the full `head_dim`. YaRN already
        // uses `yarn_rope_head_dim`; apply the same override to standard RoPE.
        let rope_cache_head_dim = yarn_rope_head_dim.unwrap_or(head_dim);
        let rope_cache_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(rope_cache_head_dim);

        let body = match (rotary_dim_lit, scaling) {
            (None, None) => quote! {
                crate::__gpu::rotary::RotaryCache::new_from_stream(
                    #rope_cache_head_dim_lit,
                    #max_pos,
                    #rope_theta,
                    None,
                    crate::__gpu::dtype::DType::BF16,
                    stream,
                )
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
                    crate::__gpu::rotary::RotaryCache::new_from_stream(
                        #head_dim,
                        #max_pos,
                        #rope_theta,
                        Some(&crate::__gpu::rotary::Llama3RopeScaling {
                            factor: #factor,
                            low_freq_factor: #low_freq_factor,
                            high_freq_factor: #high_freq_factor,
                            original_max_position_embeddings: #orig,
                        }),
                        crate::__gpu::dtype::DType::BF16,
                        stream,
                    )
                }
            }
            (
                None,
                Some(crate::config::RopeScaling::LongRope {
                    short_factor,
                    long_factor,
                    original_max_position_embeddings,
                    short_mscale,
                    long_mscale,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                quote! {
                    crate::__gpu::rotary::RotaryCache::new_longrope_from_stream(
                        #head_dim,
                        #max_pos,
                        max_model_len,
                        #rope_theta,
                        &crate::__gpu::rotary::LongRopeScaling {
                            short_factor: vec![ #(#short_factor),* ],
                            long_factor: vec![ #(#long_factor),* ],
                            original_max_position_embeddings: #orig,
                            short_mscale: #short_mscale,
                            long_mscale: #long_mscale,
                        },
                        crate::__gpu::dtype::DType::BF16,
                        stream,
                    )
                }
            }
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
                    crate::__gpu::rotary::RotaryCache::new_yarn_from_stream(
                        #rope_hd,
                        #max_pos,
                        #rope_theta,
                        &crate::__gpu::rotary::YarnRopeScaling {
                            factor: #factor,
                            beta_fast: #beta_fast,
                            beta_slow: #beta_slow,
                            mscale: #mscale,
                            mscale_all_dim: #mscale_all_dim,
                            original_max_position_embeddings: #orig,
                        },
                        crate::__gpu::dtype::DType::BF16,
                        stream,
                    )
                }
            }
            (Some(rotary_dim), None) => quote! {
                crate::__gpu::rotary::RotaryCache::new_partial_from_stream(
                    #head_dim,
                    #rotary_dim,
                    #max_pos,
                    #rope_theta,
                    None,
                    crate::__gpu::dtype::DType::BF16,
                    stream,
                )
            },
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
                    crate::__gpu::rotary::RotaryCache::new_partial_from_stream(
                        #head_dim,
                        #rotary_dim,
                        #max_pos,
                        #rope_theta,
                        Some(&crate::__gpu::rotary::Llama3RopeScaling {
                            factor: #factor,
                            low_freq_factor: #low_freq_factor,
                            high_freq_factor: #high_freq_factor,
                            original_max_position_embeddings: #orig,
                        }),
                        crate::__gpu::dtype::DType::BF16,
                        stream,
                    )
                }
            }
            (
                Some(rotary_dim),
                Some(crate::config::RopeScaling::LongRope {
                    short_factor,
                    long_factor,
                    original_max_position_embeddings,
                    short_mscale,
                    long_mscale,
                }),
            ) => {
                let orig = original_max_position_embeddings as usize;
                quote! {
                    crate::__gpu::rotary::RotaryCache::new_partial_longrope_from_stream(
                        #head_dim,
                        #rotary_dim,
                        #max_pos,
                        max_model_len,
                        #rope_theta,
                        &crate::__gpu::rotary::LongRopeScaling {
                            short_factor: vec![ #(#short_factor),* ],
                            long_factor: vec![ #(#long_factor),* ],
                            original_max_position_embeddings: #orig,
                            short_mscale: #short_mscale,
                            long_mscale: #long_mscale,
                        },
                        crate::__gpu::dtype::DType::BF16,
                        stream,
                    )
                }
            }
        };
        quote! {
            let rotary = unsafe { #body }?;
        }
    } else {
        quote! {}
    };

    // Stream-free metal counterpart to `rotary_load`. `new_from_gpuweights`
    // currently covers basic + Llama3 scaling; LongRoPE / Yarn /
    // partial-rotary models hit a compile_error so the failure mode is
    // an explicit "unsupported under metal" rather than a hidden
    // runtime panic. Same scope decision as the per-arch metal feature
    // gates landed in 5.F.5.
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
    let rotary_load_metal: TokenStream = if uses_rotary {
        let head_dim = *model
            .bounds
            .get("head_dim")
            .expect("model must have head_dim for Rotary") as usize;
        let max_pos = *model
            .bounds
            .get("max_position_embeddings")
            .expect("model must have max_position_embeddings for Rotary")
            as usize;
        let rope_theta = model
            .scalars
            .get("rope_theta")
            .copied()
            .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
            .unwrap_or(10000.0);
        let partial = model
            .scalars
            .get("partial_rotary_factor")
            .copied()
            .filter(|&f| (f - 1.0).abs() > 1e-9);
        let scaling = model.rope_scaling.clone();
        // Clamp the rotary cache size to the runtime `max_model_len`
        // rather than the model's compile-time `max_position_embeddings`.
        // For Llama-3.2 (max_position_embeddings = 131072) under chat
        // workloads with the default 4-8K context, the unclamped path
        // precomputed 16-32× more cos/sin pairs than any request can
        // possibly index — pure waste of CPU + upload bandwidth.
        // `max_model_len` is the function argument; the .min(...) caps
        // it at the model's hard limit so a misconfigured larger value
        // doesn't run past the rope shape.
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
                let rope_max_pos = ::core::cmp::min(max_model_len, #max_pos);
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
                    let rope_max_pos = ::core::cmp::min(max_model_len, #max_pos);
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
                        let rope_max_pos = ::core::cmp::min(max_model_len, #max_pos);
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
                        let rope_max_pos = ::core::cmp::min(max_model_len, #max_pos);
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
                // Partial rotary, no scaling (Qwen3.5 / Qwen3-Next:
                // partial_rotary_factor 0.25). Builds a `[max_pos, rotary_dim]`
                // cache; `W::ROT_DIM` (= rotary_dim) drives the kernel so the
                // trailing head_dim-rotary_dim channels pass through unrotated.
                (Some(rotary_dim), None) => quote! {
                    let rope_max_pos = ::core::cmp::min(max_model_len, #max_pos);
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
                        let rope_max_pos = ::core::cmp::min(max_model_len, #max_pos);
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
                // LongRoPE / YaRN (with or without partial rotary): the
                // macro can't emit a working metal init yet (no
                // `*_from_gpuweights` counterpart in `scratchy-target-cuda::rotary`).
                // Stub to a runtime panic so the build remains green for the
                // metal-supported subset; arches that hit this won't load
                // successfully under metal until the proper port lands.
                //
                // The panic lives inside a local `fn` whose return type is
                // `RotaryCache` (not `!`), so `rustc` doesn't propagate the
                // never type to the outer scope and warn `unreachable_code`
                // on every line of generated code after the rotary load —
                // the call-site sees a regular `RotaryCache` value. A plain
                // fn call rather than an immediately-invoked closure keeps
                // clippy's `redundant_closure_call` happy.
                _ => quote! {
                    let rotary: crate::__gpu::rotary::RotaryCache = {
                        fn unsupported_rotary_scaling() -> crate::__gpu::rotary::RotaryCache {
                            ::core::panic!(
                                "metal: rotary scaling variant not yet supported \
                                 (LongRoPE / Yarn / partial-rotary). Land a metal \
                                 counterpart to RotaryCache::new_from_gpuweights for \
                                 this scaling family before enabling this model \
                                 under --features metal."
                            )
                        }
                        unsupported_rotary_scaling()
                    };
                },
            }
        } // else: !global_proportional
    } else {
        quote! {}
    };

    let rotary_init: TokenStream = if uses_rotary {
        quote! { rotary, }
    } else {
        quote! {}
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
    // (cuda/metal/spyre target crate, or `scratchy-layers` with no backend), so
    // the same accessors compile everywhere. No backend is special-cased.
    let accessor_methods = emit_weights_accessor_methods(&accessors);

    // Rotary cos_sin accessors for the host-interpreter path. Both
    // `wm.rotary` and `wm.rotary_local` are conditionally-emitted
    // fields — the interpreter arm body is a single token stream
    // shared across every claim of an Impl on this arch, so it
    // can't directly write `wm.rotary_local.cos_sin_cache` (Llama
    // would fail to type-check). Per-claim selection rides on a
    // `cos_sin_fn: for<'a> fn(&'a Weights, u32) -> scratchy_target_cuda::tensor::GpuTensor`
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
                        self.rotary.cos_sin_cache
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
                        self.rotary_local.cos_sin_cache
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
    // which each backend crate (cuda/metal/spyre) provides — and `scratchy-layers`
    // provides in the no-backend default build. So the same struct compiles under
    // every target with no backend special-cased: spyre carries the same real
    // `Weights` cuda and metal do, differing only in which crate `__gpu` names.
    let weights_def: TokenStream = match &mode {
        WeightsEmitMode::Canonical => quote! {
            /// Every weight the forward needs, packed for the solver-picked Impls.
            /// Construct via `load`; the fields are device weight handles.
            pub struct Weights {
                #(#fields)*
                #rotary_field
                #rotary_local_field
                /// Lazy-initialized `MetalWorkerPool<Self>` used by
                /// the metal `forward` body. Constructed on the
                /// first call (caller-passed `ctx.kv_cache` provides
                /// the per-layer KV buffers the runtime_factory
                /// closure captures); after that every forward
                /// reuses the same pool. The `OnceLock` lets the
                /// pool live as a field on `Weights` without an
                /// `Arc`-cycle — it borrows `&Weights` for each
                /// `pool.forward(weights, ...)` call instead.
                #[cfg(feature = "metal")]
                pub metal_pool: ::std::sync::OnceLock<
                    ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Self>,
                >,
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

    let marlin_fmt = marlin_format_literal(model);

    // Canonical emits the full `load_with(marlin_storage)` body +
    // a thin `load()` wrapper that passes this variant's Marlin
    // format literal. Shim variants skip `load_with` entirely —
    // they just thread their own MarlinFormat into the canonical
    // sibling's `load_with`. rustc doesn't re-monomorphize the
    // shim's one-line delegation body, so the expensive load
    // compile work (N_layers × N_accessors lines) runs ONCE per
    // equivalence class.
    match &mode {
        WeightsEmitMode::Canonical => quote! {
            #weights_def

            #accessor_methods

            #rotary_cos_sin_methods

            #fingerprint_method

            /// Canonical load body. `marlin_storage` lets AWQ/GPTQ/CT
            /// variants share one compiled copy; ignored elsewhere.
            /// `tp_rank` is the runtime rank-id (0..tp_world_size);
            /// the bake `tp_world_size` literal lives on `<W as
            /// CanonicalParams>::…` divisor constants and on the
            /// per-(model, tp) emitted `_sharded` loader call sites
            /// (task #5).
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_lines, clippy::not_unsafe_ptr_arg_deref, unused_variables)]
            pub fn load_with(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::CUstream,
                max_model_len: usize,
                marlin_storage: crate::__gpu::layers_quant::MarlinFormat,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                // Bring the relocated `load*` extension traits into scope so
                // the path-syntax `LinearLayer::load_dense_or_ggml(...)`,
                // `RmsNorm::load(...)`, `MarlinLinear::load(...)`, … calls the
                // body emits below resolve (the `load*` methods moved off the
                // neutral type defs into `*Ops` traits in targets/cuda).
                use crate::__gpu::layers::{
                    LinearLayerOps as _, EmbeddingOps as _, RmsNormOps as _, LayerNormOps as _,
                    GatedDeltaNetOps as _,
                };
                use crate::__gpu::layers_quant::{
                    MarlinLoadOps as _, Fp8LoadOps as _, Fp8BlockLoadOps as _, Bnb4bitLoadOps as _,
                };
                use crate::__gpu::layers_moe::{
                    DeepSeekV2MoEOps as _, DeepSeekV2Fp8BlockMoEOps as _, DeepSeekV2GgmlMoEOps as _,
                    FusedMoEOps as _, SharedFusedMoEOps as _,
                    Gemma4RouterOps as _, SwitchGluExpertsOps as _,
                };
                // Stream-based RotaryCache constructors moved off the neutral
                // scratchy-layers type into this cuda extension trait.
                use crate::__gpu::rotary::CudaRotaryExt as _;
                #packed_splits_prelude
                #marlin_prelude
                #bnb4_prelude
                #fp8_prelude
                #rotary_prelude
                #(#lets)*
                #rotary_load
                #rotary_local_load
                Ok(#weights_ctor {
                    #(#field_shorthand,)*
                    #rotary_init
                    #rotary_local_init
                    #[cfg(feature = "metal")]
                    metal_pool: ::std::sync::OnceLock::new(),
                })
            }

            /// Variant entry — threads this variant's MarlinFormat.
            #[cfg(feature = "cuda")]
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::CUstream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                load_with(gw, stream, max_model_len, #marlin_fmt, tp_rank)
            }

            /// Metal entry — same signature as cuda's `load` so the
            /// top-level dispatching `Weights::load` walks both
            /// backends through the same arms. `stream` is an alias
            /// for `()` under metal (`scratchy_target_cuda::CUstream`)
            /// and is unused; the body shares the canonical's `lets`
            /// and `field_shorthand` — `LinearConcat` arms route
            /// through `LinearLayer::load_dense_concat_packed` and
            /// the rotary load uses `RotaryCache::new_from_gpuweights`.
            #[cfg(feature = "metal")]
            #[allow(clippy::too_many_lines, unused_variables)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::CUstream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                // Bring the relocated `load*` extension traits into scope so
                // the path-syntax `LinearLayer::load_dense_or_ggml(...)`,
                // `RmsNorm::load(...)`, `Embedding::load(...)`,
                // `AffineQuantEmbedding::load(...)`, … calls the metal body
                // emits below resolve. The LinearLayer/Embedding/RmsNorm/
                // LayerNorm/GatedDeltaNet loaders live in targets/cuda's
                // `*Ops`; the affine-embedding loader is a metal-local trait.
                use crate::__gpu::layers::{
                    LinearLayerOps as _, EmbeddingOps as _, RmsNormOps as _, LayerNormOps as _,
                    GatedDeltaNetOps as _, MetalAffineEmbedOps as _, MetalLinearLayerOps as _,
                };
                use crate::__gpu::layers_moe::{
                    FusedMoEOps as _, SharedFusedMoEOps as _, DeepSeekV2MoEOps as _,
                    Gemma4RouterOps as _, SwitchGluExpertsOps as _,
                };
                // Pin the RMSNorm gain dtype to the kernel's `T_scale` (the
                // `_s_<scale>_` arm fixes it from `SCALE_DTYPE`, not the on-disk
                // dtype). `RmsNorm::load`'s `take_as_dtype` then converts the
                // gain only when a checkpoint ships a different dtype (e.g. a
                // standard HF bf16 export feeding an `_s_f16_` Llama arm) — the
                // mlx zero-copy gain path is preserved on the matching case.
                gw.set_rmsnorm_scale_dtype(
                    <Weights as ::scratchy_forward_compiler::CanonicalParams>::SCALE_DTYPE,
                );
                #packed_splits_prelude
                #(#lets)*
                #rotary_load_metal
                #rotary_local_load_metal
                Ok(#weights_ctor {
                    #(#field_shorthand,)*
                    #rotary_init
                    #rotary_local_init
                    #[cfg(feature = "metal")]
                    metal_pool: ::std::sync::OnceLock::new(),
                })
            }

            /// Spyre load — same shape as the cuda/metal `load`: it builds the
            /// real `Weights` through `crate::__gpu`'s loader `*Ops`, which
            /// `scratchy-target-spyre` provides (backed by the neutral
            /// allocator-generic loaders + `SpyreAllocator`). `stream` is `()`
            /// (`LoadStream`); rotary loads from `GpuWeights` (no stream), the
            /// same constructor the metal `load` uses.
            #[cfg(feature = "spyre")]
            #[allow(clippy::too_many_lines, unused_variables)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::LoadStream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                use crate::__gpu::layers::{
                    LinearLayerOps as _, EmbeddingOps as _, RmsNormOps as _, LayerNormOps as _,
                    GatedDeltaNetOps as _,
                };
                use crate::__gpu::layers_quant::{
                    MarlinLoadOps as _, Fp8LoadOps as _, Fp8BlockLoadOps as _, Bnb4bitLoadOps as _,
                    AffineQuantEmbeddingOps as _,
                };
                use crate::__gpu::layers_moe::{
                    DeepSeekV2MoEOps as _, DeepSeekV2Fp8BlockMoEOps as _,
                    DeepSeekV2GgmlMoEOps as _, FusedMoEOps as _, SharedFusedMoEOps as _,
                    Gemma4RouterOps as _, SwitchGluExpertsOps as _,
                };
                use crate::__gpu::rotary::CudaRotaryExt as _;
                #packed_splits_prelude
                // ⛔ THE SAME QUANT PRELUDES THE CUDA BODY GETS, because the
                // `#lets` below are the SAME per-accessor load plans and they
                // REFERENCE these bindings (`__fp8_dtype`, `__marlin_ws`,
                // `__bnb_code`). Omitting them here does not disable quant on
                // spyre — it emits a load body that names an undefined local,
                // which only stayed invisible while spyre's `Weights` had no
                // weight fields at all and this body was therefore empty.
                // Each prelude elides itself when the model has no accessor of
                // that kind, so a dense model is unaffected.
                #marlin_prelude
                #bnb4_prelude
                #fp8_prelude
                #(#lets)*
                #rotary_load_metal
                #rotary_local_load_metal
                Ok(#weights_ctor {
                    #(#field_shorthand,)*
                    #rotary_init
                    #rotary_local_init
                })
            }
        },
        WeightsEmitMode::Shim { canonical } => quote! {
            #weights_def

            #fingerprint_method

            /// Shim — delegates to canonical's `load_with`.
            #[cfg(feature = "cuda")]
            #[inline]
            #[allow(clippy::not_unsafe_ptr_arg_deref)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::CUstream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                super::#canonical::load_with(gw, stream, max_model_len, #marlin_fmt, tp_rank)
            }

            /// Metal shim — delegates to canonical's metal `load`.
            /// Same signature as cuda's shim above; `stream` is `()`
            /// under metal and is forwarded to the canonical's
            /// `load` (which also ignores it).
            #[cfg(feature = "metal")]
            #[inline]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::CUstream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                super::#canonical::load(gw, stream, max_model_len, tp_rank)
            }

            /// Spyre shim — delegates to canonical's spyre marker load.
            #[cfg(feature = "spyre")]
            #[inline]
            #[allow(unused_variables)]
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::LoadStream,
                max_model_len: usize,
                tp_rank: u8,
            ) -> ::anyhow::Result<Weights> {
                super::#canonical::load(gw, stream, max_model_len, tp_rank)
            }
        },
    }
}

/// Split a Weights field name into `(base, layer)` where `layer`
/// is `Some(n)` if the name ends in `_<digits>` (e.g.
/// `input_layernorm_3` → `("input_layernorm", Some(3))`) and
/// `None` for un-suffixed names like `embed_tokens` or `lm_head`.
///
/// The trailing-digits rule is the convention `weight_field_name`
/// has used since the start: `(stem)_(layer_index)` for per-layer
/// accessors, bare `stem` for arch-wide ones. New accessors must
/// follow the same rule or the per-arch accessor method codegen
/// will silently group them as non-layered. (Mixed bases —
/// e.g. one `input_layernorm` and one `input_layernorm_0` under
/// the same name root — panic at codegen time.)
pub(crate) fn split_base_layer(field_name: &str) -> (String, Option<UnrollIndex>) {
    if let Some((base, suffix)) = field_name.rsplit_once('_')
        && !suffix.is_empty()
        && suffix.chars().all(|c| c.is_ascii_digit())
        && let Ok(n) = suffix.parse::<u64>()
    {
        return (base.to_string(), Some(UnrollIndex(n)));
    }
    (field_name.to_string(), None)
}

/// One row in the per-arch `Weights` shape. The field/struct emit,
/// the `impl Weights` accessor methods, and the `load_with` body
/// all walk the same group list so they stay in sync.
pub(crate) struct AccessorGroup<'a> {
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
pub(crate) enum AccessorGroupKind {
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
pub(crate) fn group_accessors_by_base(accessors: &[WeightAccessor]) -> Vec<AccessorGroup<'_>> {
    use std::collections::BTreeMap;

    struct Bucket<'a> {
        rust_type: TokenStream,
        rust_type_str: String,
        layered_entries: BTreeMap<UnrollIndex, &'a WeightAccessor>,
        unindexed: Option<&'a WeightAccessor>,
    }

    let mut by_base: BTreeMap<String, Bucket<'_>> = BTreeMap::new();
    for acc in accessors {
        let full = acc.name.to_string();
        let (base, layer) = split_base_layer(&full);
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
/// `scratchy_target_cuda::loaders` take. E.g. `"model.layers.0.input_layernorm"`
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
            // path (CPU-concat then one allocator call); cuda keeps
            // the GGUF/safetensors `_or_ggml` path with stream-based
            // async DMA. Sharded variants stay cuda-only — single-
            // GPU is the only metal target initially.
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
                let gate_bits = gate_bits.unwrap_or(bits);
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
                let gate_bits = gate_bits.unwrap_or(bits);
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
            group_size,
        } => {
            let num_experts = *num_experts;
            let hidden_size = *hidden_size;
            let group_size = *group_size;
            quote! {
                let #name = crate::__gpu::layers_moe::GemmaRouterLayer::load(
                    gw,
                    #prefix,
                    #num_experts,
                    #hidden_size,
                    #group_size,
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
    }
}

/// The on-disk affine bit-width a layered plan loads with, if any. Used
/// to detect MLX-native mixed/dynamic-quant groups (OptiQ) where numbered
/// layers differ in bits, so the layered emit can switch to a per-layer
/// loader. `None` for plans with no single affine bits.
fn field_load_affine_bits(plan: &FieldLoad) -> Option<u32> {
    match plan {
        FieldLoad::LinearAffine { bits, .. }
        | FieldLoad::LinearAffineConcat { bits, .. }
        | FieldLoad::GemmaSwitchGlu { bits, .. } => Some(*bits),
        // MoE bundles carry the routed-expert bits in `affine`; the router
        // `gate_bits` is uniform across layers so it stays out of the
        // per-layer gather (the collapsed layer-0 plan supplies it).
        FieldLoad::FusedMoe { affine, .. } | FieldLoad::SharedFusedMoe { affine, .. } => {
            affine.map(|(_, b)| b)
        }
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
            let per_layer_bits: Option<Vec<u32>> = {
                let bits: Vec<Option<u32>> = group
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
                        let v: Vec<u32> = bits.into_iter().map(Option::unwrap).collect();
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
            // Decoder-side layered root: `model.layers` (text-only) or a
            // `decoder_safetensors_prefix`-derived root. MUST agree with
            // [`safetensors_prefix`]'s prefix handling (replace-vs-prepend),
            // or the L=0 key the loader looks up won't match this root and
            // the `layered_suffix` invariant fires.
            let decoder_root_owned: String = match model.arch.decoder_prefix.as_deref() {
                None => "model.layers".to_string(),
                Some(prefix) => {
                    let prefix = prefix.trim_end_matches('.');
                    if prefix.starts_with("model") {
                        // Replace-the-`model`-root form (Qwen3.5-VL
                        // `model.language_model`): `model.language_model.layers`.
                        format!("{prefix}.layers")
                    } else {
                        // Namespace-wrapper form (Gemma3-MM `language_model`):
                        // `language_model.model.layers`.
                        format!("{prefix}.model.layers")
                    }
                }
            };
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
/// crossing the scratchy-forward-compiler / scratchy-target-cuda seam).
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
    per_layer_bits: Option<&[u32]>,
) -> TokenStream {
    use crate::tp_lowering::{ShardKind, shard_kind_for_dotted_prefix};
    let n_lit = proc_macro2::Literal::u32_unsuffixed(n_layers);
    // Literal `[b0, b1, ...]` per-layer bits slice for the mixed loaders.
    let per_layer_bits_lit: Option<TokenStream> = per_layer_bits.map(|bits| {
        let elems = bits
            .iter()
            .map(|b| proc_macro2::Literal::u32_unsuffixed(*b));
        quote! { &[ #(#elems),* ] }
    });
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
                let gate_bits = gate_bits.unwrap_or(bits);
                // Per-layer routed-expert bits for MLX mixed/dynamic quant
                // (OptiQ); single width otherwise.
                let bits_expr: TokenStream = match &per_layer_bits_lit {
                    Some(bits_slice) => quote! { (#bits_slice)[layer as usize] },
                    None => quote! { #bits },
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
                let gate_bits = gate_bits.unwrap_or(bits);
                let bits_expr: TokenStream = match &per_layer_bits_lit {
                    Some(bits_slice) => quote! { (#bits_slice)[layer as usize] },
                    None => quote! { #bits },
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
            group_size,
        } => {
            let p = layer_templated_prefix_expr(
                prefix,
                vision_zero_prefix_ref,
                decoder_zero_prefix_ref,
            );
            let num_experts = *num_experts;
            let hidden_size = *hidden_size;
            let group_size = *group_size;
            quote! {
                (0u32..#n_lit)
                    .map(|layer: u32| -> ::anyhow::Result<_> {
                        crate::__gpu::layers_moe::GemmaRouterLayer::load(
                            gw,
                            &#p,
                            #num_experts,
                            #hidden_size,
                            #group_size,
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
fn emit_weights_accessor_methods(accessors: &[WeightAccessor]) -> TokenStream {
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
            let ty = &g.rust_type;
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

/// The `WeightAccessors` trait method a [`WeightKind`] resolves through.
/// Shared by [`emit_weight_accessors_impl`] (which keys its per-(op_idx,
/// kind) slot-ordinal walk on this) and the PD-wavefront bridge's
/// `to_wavefront::build_base_to_loc` (which must reproduce the IDENTICAL
/// slot ordinals so the wavefront `WeightLoc`s key the SAME runtime match
/// arms). Keep the two in lockstep by routing both through this fn.
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
        WeightKind::CosSin => "cos_sin_at",
        WeightKind::AffineQuantEmbedding => "affine_quant_embedding_at",
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
    canonical_lowered: &BTreeMap<
        crate::assignment::WorkloadPoint,
        (CanonicalLowered, u32, u32, u32, SlotMap),
    >,
    variant_name: &str,
    // Spans rope-on-read: whether this arch has a global (`rotary`) and/or
    // sliding (`rotary_local`) RoPE cache, so `rope_on_read_cos_sin` can be
    // generated to return the right one by layer class.
    uses_rotary: bool,
    uses_rotary_local: bool,
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
                let mut counts: HashMap<&'static str, u32> = HashMap::new();
                for slot in slots {
                    let key = weight_kind_accessor_method(&slot.kind);
                    let n = counts.entry(key).or_insert(0);
                    let slot_lit = proc_macro2::Literal::u32_unsuffixed(*n);
                    inventory.push((
                        tape_index,
                        role,
                        op_idx as u32,
                        *n,
                        key,
                        slot.base.to_string(),
                    ));
                    *n += 1;
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
                        WeightKind::CosSin => quote! { self.#base.cos_sin_cache },
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
    // Spans rope-on-read: resolve cos/sin by layer CLASS (global `rotary`
    // vs sliding `rotary_local`), not by op operand. Emitted only when the
    // arch has a rotary cache; otherwise the trait default (unreachable)
    // stands and is never called (no spans binding without ROPE_ON_READ).
    let rope_on_read_cos_sin: TokenStream = if uses_rotary {
        // With a sliding rotary, the global (`rotary`) and local
        // (`rotary_local`) caches differ, so dispatch on the layer class.
        // Without one, both classes resolve to the global cache — return it
        // directly (and mark `is_global` unused) rather than emit an `if`
        // with two identical arms. `GpuTensor` is `Copy`, so no `.clone()`.
        let (is_global_param, body) = if uses_rotary_local {
            (
                quote! { is_global },
                quote! {
                    if is_global {
                        self.rotary.cos_sin_cache
                    } else {
                        self.rotary_local.cos_sin_cache
                    }
                },
            )
        } else {
            (quote! { _is_global }, quote! { self.rotary.cos_sin_cache })
        };
        quote! {
            fn rope_on_read_cos_sin(
                &self,
                #is_global_param: bool,
                _layer: u32,
            ) -> crate::__gpu::tensor::GpuTensor {
                #body
            }
        }
    } else {
        quote! {}
    };
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
            #[cfg(feature = "metal")]
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
    // arch whose own `metal`/`cuda` feature is empty
    // or absent (e.g. phi3 which declares `metal = []`).
    // The trait methods all return cross-backend types from
    // `scratchy_target_cuda::layers{,_moe}` plus `GpuTensor` from
    // `scratchy-target-cuda` — both available regardless of backend feature.
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
            #rope_on_read_cos_sin
            #affine_quant_embedding
        }
    }
}

/// Emit the `MarlinFormat` const this variant's `load` passes into
/// the (canonical or shared) `load_with` body. For non-Marlin
/// variants (dense / BNB4 / FP8 later) the value is still a valid
/// `MarlinFormat` — the `load_with` body just doesn't reference
/// `marlin_storage`, so the const is never read.
fn marlin_format_literal(model: &ModelParams) -> TokenStream {
    use crate::quantization::QuantMethod;
    match model.quantization.as_ref().map(|qc| &qc.method) {
        Some(QuantMethod::Awq { group_size, .. }) => {
            let gs = proc_macro2::Literal::u32_unsuffixed(*group_size);
            quote! {
                crate::__gpu::layers_quant::MarlinFormat::Awq { group_size: #gs }
            }
        }
        Some(QuantMethod::Gptq {
            group_size,
            desc_act,
            layout,
            ..
        }) => {
            let gs = proc_macro2::Literal::u32_unsuffixed(*group_size);
            let da = *desc_act;
            let layout_ts = gptq_layout_ts(*layout);
            quote! {
                crate::__gpu::layers_quant::MarlinFormat::Gptq {
                    group_size: #gs,
                    desc_act: #da,
                    layout: #layout_ts,
                }
            }
        }
        // Dense / BNB4 / other: emit a placeholder; `load_with` in
        // those equivalence classes never reads the param. Keeping
        // a non-unit value here means the emitted const is well-
        // typed regardless of arch / quant family.
        _ => quote! {
            crate::__gpu::layers_quant::MarlinFormat::Awq { group_size: 128 }
        },
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
// Buckets in the same SFUF equivalence class as a canonical share
// the canonical's body via thin `#[inline(always)]` wrappers — same
// dedup the previous codegen path used.

/// Ident for a per-workload-tape_index `static <PREFIX>_<m>: &[Op]`.
/// Mirrors [`bucket_fn_ident`] in shape but uses upper-case so the
/// emitted module reads naturally — `FORWARD_M_64` /
/// `BACKBONE_M_64_SK_2048`.
fn bucket_static_ident(prefix: &str, wp: crate::assignment::WorkloadPoint) -> proc_macro2::Ident {
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
/// - `OpKind::LoadPixels` — `vision_lowering::materialize_pixels`
///   pushes one to materialize the `pixels` extern as a tile;
///   semantically the FIRST op (everything reads from it), but
///   lives at array tail for the same `push`-based reason.
///
/// The "last node" for backbone-output / terminal-subgraph
/// identification must be the body's actual terminal — the lm_head
/// Gemm (tp=1) or its AllGather wrapper (tp>1) for decoders, the
/// merger MLP's bias_add for vision encoders.
fn last_non_splice_node(fuf: &Fuf) -> Option<&crate::fuf::FufNode> {
    fuf.nodes.iter().rev().find(|n| {
        !matches!(
            n.op,
            crate::classified::OpKind::MmEmbedSplice
                | crate::classified::OpKind::LoadPixels
                // `materialize_pos_embeds` appends LoadPosEmbeds at the
                // FUF tail (after LoadPixels), so it must be skipped too
                // — otherwise it would be mistaken for the encoder's
                // terminal and the forward would return the pos_embeds
                // tile instead of the merger output.
                | crate::classified::OpKind::LoadPosEmbeds
        )
    })
}

/// How the FUF's terminal node maps onto the backbone/lm_head split.
///
/// - [`BackboneLayout::Decoder`] — the FUF ends in `gemm(<tile>,
///   lm_head)` (or that gemm followed by an `AllGather` at tp>1).
///   The backbone is everything except the terminal subgraph; the
///   `(TileId, u8)` it carries is the lm_head Gemm's hidden-state
///   input — the slot `forward_backbone` returns and the slot the
///   lm_head slice reads.
/// - [`BackboneLayout::Encoder`] — the FUF's terminal is NOT
///   `gemm(_, lm_head)`. Covers text-side encoders (ModernBERT)
///   AND every `#[vision_forward]` body (Qwen2-VL / Qwen2.5-VL /
///   SigLIP / …) since vision bodies don't have a trainable
///   lm_head. The whole pipeline is the backbone, the lm_head slice
///   is empty, and `forward` returns the FUF's last-node output
///   directly.
#[derive(Clone, Copy, Debug)]
enum BackboneLayout {
    Decoder { backbone_out: (TileId, u8) },
    Encoder,
}

/// Classify the FUF's terminal as decoder vs encoder. A decoder
/// terminal is `gemm(<tile>, <lm_head_weight>)`, optionally followed
/// by an `AllGather` (inserted by tp>1 lowering on the vocab-parallel
/// lm_head Gemm). Walks past trailing `MmEmbedSplice` / `LoadPixels` /
/// `LoadPosEmbeds` nodes via [`last_non_splice_node`] — those are
/// appended by lowering passes but aren't the body's actual terminal.
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
        Some(FufInput::Tile { id, slot }) => BackboneLayout::Decoder {
            backbone_out: (*id, *slot),
        },
        _ => BackboneLayout::Encoder,
    }
}

/// Build the workload-point bounds map `lower_bucket` and Impls
/// consume — model.bounds + the workload-specific `num_tokens` and
/// `sk_bucket` overrides. Mirrors what `solve_workloads` does before
/// each per-point solve.
fn bounds_for_wp(
    model: &ModelParams,
    wp: crate::assignment::WorkloadPoint,
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

/// Emit the full per-model module body: Weights struct + loader,
/// one forward fn per workload tape_index, and a dispatching wrapper.
///
/// When `canonical_override` is `Some(ident)`, this variant is a
/// shim for that canonical sibling — emit `pub type Weights =
/// super::<ident>::Weights;` instead of a fresh struct, emit the
/// variant-specific `load` + `fingerprint_matches` bodies
/// (loaders differ per quant preset, fingerprints differ per
/// tensor-suffix gate), and `pub use` the canonical's forward +
/// forward_backbone + per-tape_index forward_m_<N> fns. rustc doesn't
/// re-monomorphize `pub use` re-exports, so the canonical fn body
/// is optimized ONCE regardless of how many variants share it.
#[cfg(feature = "metal")]
fn emit_synthesized_kernel_sources_override(
    model: &ModelParams,
    tp_world_size: u8,
    has_linear_bias: bool,
    mlp_uses_gelu: bool,
) -> TokenStream {
    use crate::quantization::QuantMethod;
    let (bits, group_size, mlp_bits) = match model.quantization.as_ref().map(|q| &q.method) {
        Some(QuantMethod::Affine {
            bits,
            group_size,
            bits_overrides,
            ..
        }) => {
            // MLP projection width: the preset's bits_overrides carry
            // per-suffix widths (Gemma4: mlp.{gate,up,down}_proj → 8).
            let mlp_bits = bits_overrides
                .iter()
                .find(|(path, _)| path.contains("mlp."))
                .map(|(_, b)| *b)
                .unwrap_or(*bits);
            (*bits, *group_size, mlp_bits)
        }
        _ => return quote! {},
    };
    if bits != 4 {
        return quote! {};
    }
    let mlp_act = if mlp_uses_gelu {
        ::scratchy_target_metal::atom_lib::MlpAct::Gelu
    } else {
        ::scratchy_target_metal::atom_lib::MlpAct::Silu
    };
    // bf16 activation is the default for every modern Llama / Qwen /
    // Mistral / Gemma metal arch (per CanonicalParams::METAL_DTYPE).
    // Future: thread W::METAL_DTYPE through and emit per-dtype variants.
    let t_act = "bfloat";
    // T_scale tracks on-disk scale convention. Mirrors the SCALE_DTYPE
    // override in `emit_canonical_params_impl`: Qwen3 family ships BF16
    // scales+biases, everything else ships F16. Synth kernel symbol
    // must match the corresponding `MetalSynth*Impl` instantiation
    // registered in `starter_library` (otherwise the solver's pick and
    // the runtime pipeline cache disagree on the library key).
    // Whole Qwen3 family (Qwen3 / Qwen3Moe / Qwen3.5 / Qwen3.6 /
    // Qwen3-Next, dense + MoE + the VL-wrapped `Qwen3_5ForConditional
    // Generation` text decoders) ships BF16 scales+biases. Match by
    // family prefix so new members are covered automatically.
    let is_bf16_scale = model.arch.scale_dtype.as_deref() == Some("bf16");
    let t_scale = if is_bf16_scale { "bfloat" } else { "half" };

    // Model dims baked as MSL `constant constexpr` literals at synth
    // time. Same TP-sharding rules as `emit_canonical_params_impl`:
    // num_q / num_kv / intermediate split per-rank; hidden stays
    // replicated (residual stream is post-allreduce).
    let tp = tp_world_size as u32;
    let tp_us = tp_world_size as usize;
    let hidden = *model.bounds.get("hidden_size").unwrap_or(&0) as u32;
    let head_dim = *model.bounds.get("head_dim").unwrap_or(&0) as u32;
    let num_q = (*model.bounds.get("num_attention_heads").unwrap_or(&0) as u32) / tp;
    let num_kv = (*model.bounds.get("num_key_value_heads").unwrap_or(&0) as u32) / tp;
    let intermediate = (*model.bounds.get("intermediate_size").unwrap_or(&0) as u32) / tp;
    let partial = model
        .scalars
        .get("partial_rotary_factor")
        .copied()
        .filter(|&f| (f - 1.0).abs() > 1e-9);
    let rot_dim = match partial {
        Some(f) => (f * head_dim as f64).round() as u32,
        None => head_dim,
    };
    let eps = rms_norm_eps(model);
    let _ = tp_us;

    // Sanity-gate: if any required dim is zero, skip emission (the
    // model isn't a standard transformer-decoder we can synthesize for).
    if hidden == 0 || head_dim == 0 || num_q == 0 || num_kv == 0 || intermediate == 0 {
        return quote! {};
    }

    let consts = crate::fuse_pass::ChunkConstants {
        // Baked as `constant constexpr` literals in the emitted MSL.
        // M is the only remaining function constant (varies per
        // tape_index; can't be baked).
        hidden,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        head_dim,
        rot_dim,
        block_size: 16, // scratchy_forward_compiler::CanonicalParams::BLOCK_SIZE default
        intermediate,
        m: 0,
        group_size,
        rms_norm_eps: eps,
        // Pre-attn synth gains 3 extra `__{q,k,v}_linear_bias` buffer
        // params + a bias epilogue inside each per-band AffineQmvAtom
        // when this is `true`. Threaded down from
        // `program_has_bias_add` so Qwen2/Qwen2.5 (DSL emits
        // `bias_add` on QKV) gets the biased variant; Llama (no DSL
        // bias_add) keeps the existing one. MLP / gate-up synths
        // ignore this — Qwen2 MLP has no biases.
        has_linear_bias,
    };
    let pre_attn = crate::fuse_pass::synthesize_pre_attn_chunk(
        crate::fuse_pass::SynthesisBackend::Metal,
        t_act,
        t_scale,
        &consts,
    );
    let pre_attn_init = crate::fuse_pass::synthesize_pre_attn_init_chunk(
        crate::fuse_pass::SynthesisBackend::Metal,
        t_act,
        t_scale,
        &consts,
    );
    // MLP pre-down synth: bake BOTH the 4-bit and 8-bit kernels.
    //
    // The solver picks a `MetalSynthMlpPreDown` impl by the shared
    // expert's *actual* per-module quant width (`synth_mlp_pre_down.rs`
    // registers a b4 and a b8 variant, each `matches()`-gated on the
    // weight's bits). A mixed-width OptiQ checkpoint therefore dispatches
    // b4 on some layers and b8 on others, but `mlp_bits` — derived from a
    // `bits_overrides` suffix match — only knows one width, so it would
    // register a single library and leave the other width's pipeline
    // lookup unresolved (→ silent garble). Emit both; a uniform model
    // simply never looks up the unused one.
    let _ = mlp_bits;
    let mlp_pre_down_variants: ::std::vec::Vec<_> = [4u32, 8u32]
        .into_iter()
        .map(|b| {
            crate::fuse_pass::synthesize_mlp_pre_down_chunk(
                crate::fuse_pass::SynthesisBackend::Metal,
                t_act,
                t_scale,
                &consts,
                mlp_act,
                b,
            )
        })
        .collect();
    // AOT-compile each synth source to a `.metallib` blob at macro
    // expansion time. Same `xcrun metal -c` + `xcrun metallib`
    // pipeline used by `scratchy-target-metal/build.rs` for every
    // hand-written shader. Runtime loads via `newLibraryWithData`
    // (NOT `newLibraryWithSource`) so the resulting Metal binaries
    // are identical to the AOT-compiled shaders — same compiler
    // path, same behavior across Apple GPU generations.
    let gate_up = ::scratchy_target_metal::fuse_pass::synthesize_gate_up_silu_mul_large_chunk(
        ::scratchy_target_metal::fuse_pass::SynthesisBackend::Metal,
        t_act,
        t_scale,
        &consts,
    );
    let gu_bytes =
        ::scratchy_target_metal::aot::aot_compile_metallib(&gate_up.symbol, &gate_up.source);

    let pa_bytes =
        ::scratchy_target_metal::aot::aot_compile_metallib(&pre_attn.symbol, &pre_attn.source);
    let pi_bytes = ::scratchy_target_metal::aot::aot_compile_metallib(
        &pre_attn_init.symbol,
        &pre_attn_init.source,
    );
    // Dedup by symbol so a model whose b4 and b8 chunks collide (they
    // don't today — the width is baked into the symbol) never registers
    // the same key twice.
    let mut seen_md: ::std::collections::HashSet<String> = ::std::collections::HashSet::new();
    let md_symbol_lits: ::std::vec::Vec<syn::LitStr> = ::std::vec::Vec::new();
    let md_bytes_lits: ::std::vec::Vec<syn::LitByteStr> = ::std::vec::Vec::new();
    let (md_symbol_lits, md_bytes_lits) = mlp_pre_down_variants.iter().fold(
        (md_symbol_lits, md_bytes_lits),
        |(mut syms, mut blobs), chunk| {
            if seen_md.insert(chunk.symbol.clone()) {
                let bytes = ::scratchy_target_metal::aot::aot_compile_metallib(
                    &chunk.symbol,
                    &chunk.source,
                );
                syms.push(syn::LitStr::new(
                    &chunk.symbol,
                    proc_macro2::Span::call_site(),
                ));
                blobs.push(syn::LitByteStr::new(&bytes, proc_macro2::Span::call_site()));
            }
            (syms, blobs)
        },
    );
    // One named const per md variant (byte-string literals aren't
    // `'static`-promotable inside the returned array; a `const` binding is).
    let md_idents: ::std::vec::Vec<proc_macro2::Ident> = (0..md_bytes_lits.len())
        .map(|i| quote::format_ident!("__SYNTH_MLP_PRE_DOWN_LIB_{}", i))
        .collect();

    let gu_symbol_lit = syn::LitStr::new(&gate_up.symbol, proc_macro2::Span::call_site());
    let gu_bytes_lit = syn::LitByteStr::new(&gu_bytes, proc_macro2::Span::call_site());

    let pa_symbol_lit = syn::LitStr::new(&pre_attn.symbol, proc_macro2::Span::call_site());
    let pi_symbol_lit = syn::LitStr::new(&pre_attn_init.symbol, proc_macro2::Span::call_site());

    let pa_bytes_lit = syn::LitByteStr::new(&pa_bytes, proc_macro2::Span::call_site());
    let pi_bytes_lit = syn::LitByteStr::new(&pi_bytes, proc_macro2::Span::call_site());

    quote! {
        fn synthesized_kernel_metallibs() -> &'static [(&'static str, &'static [u8])] {
            const __SYNTH_PRE_ATTN_LIB: &[u8] = #pa_bytes_lit;
            const __SYNTH_PRE_ATTN_INIT_LIB: &[u8] = #pi_bytes_lit;
            const __SYNTH_GATE_UP_SILU_MUL_LIB: &[u8] = #gu_bytes_lit;
            #( const #md_idents: &[u8] = #md_bytes_lits; )*
            &[
                (#pa_symbol_lit, __SYNTH_PRE_ATTN_LIB),
                (#pi_symbol_lit, __SYNTH_PRE_ATTN_INIT_LIB),
                (#gu_symbol_lit, __SYNTH_GATE_UP_SILU_MUL_LIB),
                #( (#md_symbol_lits, #md_idents) ),*
            ]
        }
    }
}

/// Walk the classified DSL `Program` looking for any
/// `Expr::Call { op: OpKind::BiasAdd, .. }`. Returns `true` on the
/// first hit. Drives the biased variant of the synth pre-attn
/// megakernel — Qwen2/Qwen2.5 DSL emits `bias_add` on QKV, Llama
/// does not.
/// True when the DSL's MLP uses a tanh-GELU gate (`gelu(gemm(...)) *
/// up`) — Gemma-family GeGLU. Drives the gelu variant of the synth MLP
/// megakernel (`synth_mlp_pre_down_gelu_*`).
pub fn program_has_gelu(program: &Program) -> bool {
    fn scan_stmt(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => scan_expr(value),
            Stmt::For { body, .. } => body.iter().any(scan_stmt),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => then_body.iter().any(scan_stmt) || else_body.iter().any(scan_stmt),
        }
    }
    fn scan_expr(expr: &Expr) -> bool {
        match expr {
            Expr::Call { op, args } => matches!(op, OpKind::Gelu) || args.iter().any(scan_expr),
            Expr::Mul { lhs, rhs } => scan_expr(lhs) || scan_expr(rhs),
            Expr::Local(_)
            | Expr::Extern { .. }
            | Expr::Weight { .. }
            | Expr::ScalarLit(_)
            | Expr::SqrtBound(_)
            | Expr::ConfigScalar { .. } => false,
        }
    }
    program.statements.iter().any(scan_stmt)
}

pub fn program_has_bias_add(program: &Program) -> bool {
    fn scan_stmt(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Assign { value, .. } | Stmt::AssignTuple { value, .. } => scan_expr(value),
            Stmt::For { body, .. } => body.iter().any(scan_stmt),
            Stmt::If {
                then_body,
                else_body,
                ..
            } => then_body.iter().any(scan_stmt) || else_body.iter().any(scan_stmt),
        }
    }
    fn scan_expr(expr: &Expr) -> bool {
        match expr {
            Expr::Call { op, args } => matches!(op, OpKind::BiasAdd) || args.iter().any(scan_expr),
            Expr::Mul { lhs, rhs } => scan_expr(lhs) || scan_expr(rhs),
            Expr::Local(_)
            | Expr::Extern { .. }
            | Expr::Weight { .. }
            | Expr::ScalarLit(_)
            | Expr::SqrtBound(_)
            | Expr::ConfigScalar { .. } => false,
        }
    }
    program.statements.iter().any(scan_stmt)
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

fn emit_canonical_params_impl(
    model: &ModelParams,
    tp_world_size: u8,
    has_bias_add: bool,
    has_gelu_mlp: bool,
    uses_rotary: bool,
    // Filled with the SAME values the impl's consts are emitted from —
    // one derivation, two consumers (the impl tokens and the macro-side
    // tape bake). Built in this fn so drift is impossible.
    #[cfg(feature = "metal")] metal_consts_out: &mut Option<
        scratchy_target_metal::tape::model_consts::MetalModelConsts,
    >,
) -> TokenStream {
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
    let kv_size = (num_kv_heads as usize) * (head_dim as usize);
    let kv_lora_rank = *model.bounds.get("kv_lora_rank").unwrap_or(&0) as usize;
    let qk_nope_head_dim = *model.bounds.get("qk_nope_head_dim").unwrap_or(&0) as usize;
    let qk_rope_head_dim = *model.bounds.get("qk_rope_head_dim").unwrap_or(&0) as usize;
    let v_head_dim = *model.bounds.get("v_head_dim").unwrap_or(&0) as usize;
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

    // Gated-DeltaNet (Qwen3.5 / Qwen3-Next) per-layer dims. Absent (→ 0,
    // matching the trait defaults) for every non-hybrid arch. The metal
    // `lower_one(GatedDeltaNet)` arm reads these as `W::GDN_*` to size
    // the conv1d / gating / scan / gated-RMSNorm dispatch grids and the
    // op-scratch layout; the cuda eval reads them too. Not divided by
    // `tp` — GDN tensor-parallel partitioning is a separate concern and
    // bring-up is TP=1 (where the division is a no-op anyway).
    let gdn_num_k_heads = *model.bounds.get("linear_num_key_heads").unwrap_or(&0) as u32;
    let gdn_num_v_heads = *model.bounds.get("linear_num_value_heads").unwrap_or(&0) as u32;
    let gdn_head_k_dim = *model.bounds.get("linear_key_head_dim").unwrap_or(&0) as u32;
    let gdn_head_v_dim = *model.bounds.get("linear_value_head_dim").unwrap_or(&0) as u32;
    let gdn_conv_kernel = *model.bounds.get("linear_conv_kernel_dim").unwrap_or(&0) as u32;
    let gdn_conv_dim = *model.bounds.get("gdn_conv_dim").unwrap_or(&0) as usize;

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

    let head_dim_lit = proc_macro2::Literal::u32_unsuffixed(head_dim);
    let num_q_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_q_heads);
    let num_kv_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_kv_heads);
    let q_size_lit = proc_macro2::Literal::usize_unsuffixed(q_size);
    let kv_size_lit = proc_macro2::Literal::usize_unsuffixed(kv_size);
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
    let hidden_size_lit = proc_macro2::Literal::usize_unsuffixed(hidden_size_for_const);
    let intermediate_size_lit = proc_macro2::Literal::usize_unsuffixed(intermediate_size);
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
    let vocab_size_lit = proc_macro2::Literal::usize_unsuffixed(vocab_size_for_const);
    let kv_lora_rank_lit = proc_macro2::Literal::usize_unsuffixed(kv_lora_rank);
    let qk_nope_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_nope_head_dim);
    let qk_rope_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_rope_head_dim);
    let v_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(v_head_dim);
    let qk_head_dim_lit = proc_macro2::Literal::usize_unsuffixed(qk_head_dim);
    let gdn_num_k_heads_lit = proc_macro2::Literal::u32_unsuffixed(gdn_num_k_heads);
    let gdn_num_v_heads_lit = proc_macro2::Literal::u32_unsuffixed(gdn_num_v_heads);
    let gdn_head_k_dim_lit = proc_macro2::Literal::u32_unsuffixed(gdn_head_k_dim);
    let gdn_head_v_dim_lit = proc_macro2::Literal::u32_unsuffixed(gdn_head_v_dim);
    let gdn_conv_kernel_lit = proc_macro2::Literal::u32_unsuffixed(gdn_conv_kernel);
    let gdn_conv_dim_lit = proc_macro2::Literal::usize_unsuffixed(gdn_conv_dim);
    let norm_weight_offset_lit =
        proc_macro2::Literal::f32_suffixed(norm_weight_runtime_offset(model));
    // RMSNorm eps from config (Qwen3.5: 1e-6). Without this the metal
    // `RmsNorm`/`FusedAddRmsNorm`/`gdn_rms_norm_gated` kernels fall back to
    // the trait default 1e-5, which visibly drifts on outlier-dominated
    // (massive-activation) residual streams where mean(x^2) ~ eps.
    let rms_norm_eps_lit = proc_macro2::Literal::f32_suffixed(rms_norm_eps(model));
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
    let rot_dim_lit = proc_macro2::Literal::u32_unsuffixed(rot_dim_val);
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
    let global_q_size: usize = (num_q_heads as usize) * (global_head_dim as usize);
    let global_head_dim_lit = proc_macro2::Literal::u32_unsuffixed(global_head_dim);
    let num_global_kv_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_global_kv_heads);
    let global_rot_dim_lit = proc_macro2::Literal::u32_unsuffixed(global_rot_dim);
    let global_q_size_lit = proc_macro2::Literal::usize_unsuffixed(global_q_size);
    // Proportional rope marker — same predicate that routes the
    // rotary cache builder to `new_proportional_from_gpuweights`.
    // Drives the metal rope kernel's pairing offset (lane i pairs
    // with i + head_dim/2, not i + rot_dim/2).
    let rope_proportional = global_rot_dim != global_head_dim
        && model.scalars.contains_key("global_partial_rotary_factor");
    let rope_proportional_tokens: proc_macro2::TokenStream = if rope_proportional {
        quote! { const ROPE_PROPORTIONAL: bool = true; }
    } else {
        quote! {}
    };
    // Rope-on-read is the universal default for every rope-using model
    // (position-independent KV: store K unrotated for span blocks, re-rope on
    // read; the bit-31 bitmap keeps non-span blocks read-as-is). No opt-in —
    // it tracks rope presence. Must agree with `ScratchyWeights::rope_on_read()`
    // (lib.rs), which the metal worker reads to flag block_table/slot_mapping
    // bit 31. No-rope models get the trait default (false).
    let rope_on_read_tokens: proc_macro2::TokenStream = if uses_rotary {
        quote! { const ROPE_ON_READ: bool = true; }
    } else {
        quote! {}
    };
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
    let base_block_size: u32 = model
        .bounds
        .get("block_size")
        .map(|&v| v as u32)
        .unwrap_or(16);
    let sliding_per_token = num_kv_heads * head_dim; // base class == sliding on gemma4
    let global_per_token = num_global_kv_heads * global_head_dim;
    let uniform_page =
        global_per_token == sliding_per_token || num_global_kv_heads == 0 || global_per_token == 0;
    let resolved_global_block_size: u32 = if uniform_page {
        base_block_size
    } else if global_per_token < sliding_per_token {
        // Global is the smaller page → scale its block_size up (gemma4).
        assert!(
            sliding_per_token.is_multiple_of(global_per_token),
            "model `{}`: sliding page {} not divisible by global page {} — \
             cannot page-unify by scaling block_size (vLLM raises the same)",
            model.source_stem,
            sliding_per_token,
            global_per_token,
        );
        base_block_size * (sliding_per_token / global_per_token)
    } else {
        base_block_size
    };
    let global_block_size_override: proc_macro2::TokenStream = if uniform_page {
        // Uniform page (or no distinct global class) — keep the default.
        quote! {}
    } else if global_per_token < sliding_per_token {
        let lit = proc_macro2::Literal::u32_unsuffixed(resolved_global_block_size);
        quote! { const GLOBAL_BLOCK_SIZE: u32 = #lit; }
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
    // MAX_BLOCKS_PER_SEQ override — block-table row stride. The trait
    // default (128 ≈ 2k tokens at block_size 16) silently truncates
    // long-context serving; long-context arches set the explicit
    // `max_blocks_per_seq` config key (Gemma4 bring-up: 2048 = 32k).
    // Emitted conditionally so every existing arch keeps the default.
    let max_blocks_override: proc_macro2::TokenStream = match model.bounds.get("max_blocks_per_seq")
    {
        Some(&v) => {
            let lit = proc_macro2::Literal::u32_unsuffixed(v as u32);
            quote! { const MAX_BLOCKS_PER_SEQ: u32 = #lit; }
        }
        None => quote! {},
    };
    let attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(attn_scale);
    let attn_softcap_lit = proc_macro2::Literal::f32_unsuffixed(attn_softcap);
    let sliding_window_lit = proc_macro2::Literal::i32_unsuffixed(sliding_window);
    let final_logit_softcapping_lit = proc_macro2::Literal::f32_unsuffixed(final_logit_softcapping);
    let mla_attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(mla_attn_scale);
    let vision_num_heads_lit = proc_macro2::Literal::u32_unsuffixed(vision_num_heads);
    let vision_head_dim_lit = proc_macro2::Literal::u32_unsuffixed(vision_head_dim);
    let vision_q_size_lit = proc_macro2::Literal::usize_unsuffixed(vision_q_size);
    let vision_in_features_lit = proc_macro2::Literal::usize_unsuffixed(vision_in_features);
    let vision_attn_scale_lit = proc_macro2::Literal::f32_unsuffixed(vision_attn_scale);
    let vision_patch_grid_side_lit = proc_macro2::Literal::u32_unsuffixed(vision_patch_grid_side);
    let vision_pool_kernel_lit = proc_macro2::Literal::u32_unsuffixed(vision_pool_kernel);
    let vision_rope_interleaved_lit = if vision_rope_interleaved {
        quote! { true }
    } else {
        quote! { false }
    };

    // MRoPE section override. `Some([t, h, w])` only when the
    // config carries `rope_scaling.mrope_section` (Qwen2-VL /
    // Qwen2.5-VL); every text-only arch keeps the default `None`
    // and the rope kernel takes the legacy 1D-positions fast path.
    // Sum-equals-`head_dim/2` is checked here (panic at expansion
    // time, not at runtime) — text decode of a misconfigured
    // multimodal variant never compiles past this guard.
    let mrope_section_tokens = match model.mrope_section {
        Some([t, h, w]) => {
            // mrope bands cover the ROTARY half, not the full head_dim
            // half — Qwen3.5 uses partial rotary (factor 0.25 → rotary_dim
            // = 64, half = 32 = [11,11,10]); Qwen2-VL is full (factor
            // absent → rotary_dim = head_dim).
            let rotary_dim = model
                .scalars
                .get("partial_rotary_factor")
                .copied()
                .filter(|&f| (f - 1.0).abs() > 1e-9)
                .map(|f| (f * head_dim as f64).round() as u32)
                .unwrap_or(head_dim);
            let pair_count = rotary_dim / 2;
            if t + h + w != pair_count {
                let msg = format!(
                    "model `{}`: rope_scaling.mrope_section [{t}, {h}, {w}] sums to {} \
                     but rotary_dim/2 = {pair_count} (rotary_dim {rotary_dim}). Fix the config.",
                    model.source_stem,
                    t + h + w,
                );
                return quote! { compile_error!(#msg); };
            }
            let t_lit = proc_macro2::Literal::u32_unsuffixed(t);
            let h_lit = proc_macro2::Literal::u32_unsuffixed(h);
            let w_lit = proc_macro2::Literal::u32_unsuffixed(w);
            quote! {
                const MROPE_SECTION: ::core::option::Option<[u32; 3]> =
                    ::core::option::Option::Some([#t_lit, #h_lit, #w_lit]);
            }
        }
        None => quote! {},
    };

    // Synthesized-kernel sources override (Metal-only, affine-int4
    // gated). Empty for cuda models and any model that doesn't ship
    // an MLX-affine int4 quantization config; default `&[]` from the
    // CanonicalParams trait kicks in there.
    //
    // `has_bias_add` is threaded through so Qwen2/Qwen2.5 (whose DSL
    // emits `bias_add` on QKV) gets the biased synth kernel variants
    // — `synth_pre_attn{,_init}_<dtype>_<scale>_gs<N>_bias` — and the
    // lowering arm's `kernel_symbol` matches the registered library.
    // Mismatch surfaces at worker init as
    // `PipelineLookup(no library …_bias in SpecializedPipelineCache)`.
    #[cfg(feature = "metal")]
    let synth_sources_override =
        emit_synthesized_kernel_sources_override(model, tp_world_size, has_bias_add, has_gelu_mlp);
    #[cfg(not(feature = "metal"))]
    let synth_sources_override = {
        let _ = (tp_world_size, has_bias_add, has_gelu_mlp);
        TokenStream::new()
    };

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
    let scale_dtype_override = if scale_dtype_is_bf16 {
        quote! {
            #[cfg(feature = "metal")]
            const SCALE_DTYPE:
                ::scratchy_target_metal::interpreter::metal::ScaleDtype =
                ::scratchy_target_metal::interpreter::metal::ScaleDtype::Bf16;
        }
    } else {
        quote! {}
    };

    // TurboQuant KV bit-width policy (Metal). 3-bit (~4.7x) is validated
    // coherent for the Llama family; outlier-heavy KV (Qwen-class massive
    // activations) degrades at 3-bit and needs 4-bit — the conservative trait
    // default. Promote arches to 3 here as the validation sweep confirms them.
    // Hybrid/SWA + non-pow2 head_dim fall back to fp16 at the factory
    // regardless of this value, so this only chooses 3 vs 4 for the arches
    // turboquant actually runs on.
    let tq_kv_bits_v: u32 = {
        let rides_3bit = model
            .architectures
            .iter()
            .any(|a| a.starts_with("Llama") || a.contains("TinyLlama"));
        if rides_3bit { 3 } else { 4 }
    };
    let tq_kv_bits_lit = proc_macro2::Literal::u32_unsuffixed(tq_kv_bits_v);

    #[cfg(feature = "metal")]
    {
        use scratchy_target_metal::interpreter::metal::{MetalDtype, ScaleDtype};
        use scratchy_target_metal::tape::model_consts::MetalModelConsts;
        *metal_consts_out = Some(MetalModelConsts {
            // No arch override exists for METAL_DTYPE — the trait
            // default is the universal value.
            metal_dtype: MetalDtype::Bf16,
            scale_dtype: if scale_dtype_is_bf16 {
                ScaleDtype::Bf16
            } else {
                ScaleDtype::F16
            },
            head_dim,
            global_head_dim,
            num_q_heads,
            num_kv_heads,
            num_global_kv_heads,
            q_size,
            hidden_size: hidden_size_for_const,
            intermediate_size,
            attn_scale,
            sliding_window,
            final_logit_softcapping,
            rms_norm_eps: rms_norm_eps(model),
            norm_weight_offset: norm_weight_runtime_offset(model),
            block_size: base_block_size,
            global_block_size: resolved_global_block_size,
            rot_dim: rot_dim_val,
            global_rot_dim,
            rope_on_read: uses_rotary,
            rope_proportional,
            max_blocks_per_seq: model
                .bounds
                .get("max_blocks_per_seq")
                .map(|&v| v as u32)
                .unwrap_or(128),
            tq_kv_bits: tq_kv_bits_v,
            vision_num_heads,
            vision_head_dim,
            vision_q_size,
            vision_in_features,
            vision_rope_interleaved,
            vision_attn_scale,
            vision_patch_grid_side,
            vision_pool_kernel,
            gdn_num_k_heads,
            gdn_num_v_heads,
            gdn_head_k_dim,
            gdn_head_v_dim,
            gdn_conv_kernel,
            gdn_conv_dim,
        });
    }

    quote! {
        // `CanonicalParams` is backend-agnostic — the trait, its
        // associated `const`s, and every callsite (`<W as
        // CanonicalParams>::HEAD_DIM`) live in `scratchy-forward-compiler` with
        // no cuda gating. So this impl applies under either
        // `cfg(feature = "cuda")` (where `Weights` is the loader
        // struct) or `cfg(feature = "metal")` (where `Weights` is
        // the ZST emitted in `emit_weights_struct`).
        impl ::scratchy_forward_compiler::CanonicalParams for Weights {
            const HEAD_DIM: u32 = #head_dim_lit;
            const NUM_Q_HEADS: u32 = #num_q_heads_lit;
            const NUM_KV_HEADS: u32 = #num_kv_heads_lit;
            const Q_SIZE: usize = #q_size_lit;
            const KV_SIZE: usize = #kv_size_lit;
            const HIDDEN_SIZE: usize = #hidden_size_lit;
            const VOCAB_SIZE: usize = #vocab_size_lit;
            const INTERMEDIATE_SIZE: usize = #intermediate_size_lit;
            const ATTN_SCALE: f32 = #attn_scale_lit;
            const ATTN_SOFTCAP: f32 = #attn_softcap_lit;
            const SLIDING_WINDOW: i32 = #sliding_window_lit;
            const TQ_KV_BITS: u32 = #tq_kv_bits_lit;
            const KV_LORA_RANK: usize = #kv_lora_rank_lit;
            const QK_NOPE_HEAD_DIM: usize = #qk_nope_head_dim_lit;
            const QK_ROPE_HEAD_DIM: usize = #qk_rope_head_dim_lit;
            const V_HEAD_DIM: usize = #v_head_dim_lit;
            const FINAL_LOGIT_SOFTCAPPING: f32 = #final_logit_softcapping_lit;
            const QK_HEAD_DIM: usize = #qk_head_dim_lit;
            const MLA_ATTN_SCALE: f32 = #mla_attn_scale_lit;
            const GDN_NUM_K_HEADS: u32 = #gdn_num_k_heads_lit;
            const GDN_NUM_V_HEADS: u32 = #gdn_num_v_heads_lit;
            const GDN_HEAD_K_DIM: u32 = #gdn_head_k_dim_lit;
            const GDN_HEAD_V_DIM: u32 = #gdn_head_v_dim_lit;
            const GDN_CONV_KERNEL: u32 = #gdn_conv_kernel_lit;
            const GDN_CONV_DIM: usize = #gdn_conv_dim_lit;
            const NORM_WEIGHT_OFFSET: f32 = #norm_weight_offset_lit;
            const RMS_NORM_EPS: f32 = #rms_norm_eps_lit;
            const ROT_DIM: u32 = #rot_dim_lit;
            const GLOBAL_HEAD_DIM: u32 = #global_head_dim_lit;
            const NUM_GLOBAL_KV_HEADS: u32 = #num_global_kv_heads_lit;
            const GLOBAL_ROT_DIM: u32 = #global_rot_dim_lit;
            const GLOBAL_Q_SIZE: usize = #global_q_size_lit;
            #rope_proportional_tokens
            #rope_on_read_tokens
            #max_blocks_override
            #global_block_size_override
            const VISION_NUM_HEADS: u32 = #vision_num_heads_lit;
            const VISION_HEAD_DIM: u32 = #vision_head_dim_lit;
            const VISION_Q_SIZE: usize = #vision_q_size_lit;
            const VISION_IN_FEATURES: usize = #vision_in_features_lit;
            const VISION_ROPE_INTERLEAVED: bool = #vision_rope_interleaved_lit;
            const VISION_ATTN_SCALE: f32 = #vision_attn_scale_lit;
            const VISION_PATCH_GRID_SIDE: u32 = #vision_patch_grid_side_lit;
            const VISION_POOL_KERNEL: u32 = #vision_pool_kernel_lit;
            #mrope_section_tokens
            #synth_sources_override
            #scale_dtype_override
        }
    }
}

/// Emit a per-variant arm body — the `GdnRuntimeConfig` literal for one
/// specialization. Returns `None` if the variant has no
/// `OpKind::GatedDeltaNet` (non-hybrid) — caller emits a `None` arm.
pub fn emit_gdn_runtime_config_arm_body(
    fuf: &Fuf,
    model: &ModelParams,
) -> Option<proc_macro2::TokenStream> {
    let num_hidden_layers = match model.bounds.get("num_hidden_layers") {
        Some(&n) => n as usize,
        None => return None,
    };
    let mut linear = vec![false; num_hidden_layers];
    let mut is_hybrid = false;
    for node in &fuf.nodes {
        if node.op != OpKind::GatedDeltaNet {
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
    let conv_dim = *model.bounds.get("gdn_conv_dim").unwrap_or(&0) as u32;
    let conv_kernel = *model.bounds.get("linear_conv_kernel_dim").unwrap_or(&0) as u32;
    let num_k_heads = *model.bounds.get("linear_num_key_heads").unwrap_or(&0) as u32;
    let num_v_heads = *model.bounds.get("linear_num_value_heads").unwrap_or(&0) as u32;
    let head_k_dim = *model.bounds.get("linear_key_head_dim").unwrap_or(&0) as u32;
    let head_v_dim = *model.bounds.get("linear_value_head_dim").unwrap_or(&0) as u32;
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
    let num_hidden_layers = match model.bounds.get("num_hidden_layers") {
        Some(&n) => n as usize,
        None => return None,
    };
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
    let head_dim = *model.bounds.get("head_dim").unwrap_or(&0) as usize;
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

/// PD-wavefront macro-emission (env-gated, diagnostic): route the solved
/// decode FUF through the FULL compile-time wavefront pipeline — bridge
/// (`to_wavefront`) → silu·mul fusion → region N-block tiling — with REAL
/// per-weight [`WeightLoc`]s recovered from this decode bucket's
/// `weight_slots`, and dump the result.
///
/// Lives here (not the pre-emit drive) because `weight_slots` — the real
/// locator source — only exists after lowering + loop compression, and the
/// `(bucket, op_idx, slot)` triples it produces must match the very match
/// table [`emit_weight_accessors_impl`] builds from the same
/// `canonical_lowered`. Every fallible step logs and returns; it never gates
/// a build.
///
/// Under `-Fspyre`, fills `ktir_bundle_out` with the embedded KTIR/SuperDSC
/// bundle for this model. A no-op otherwise: a program with placeholder
/// locators (e.g. dense gate/up fused under one accessor) is not runnable.
/// Stable content fingerprint of a sengraph JSON — the AoT g2 cache key shared
/// between this proc-macro (which dumps `<fp>.sengraph.json`) and
/// `scratchy-builder-spyre` (which compiles + bakes `<fp>.g2.sen`). FNV-1a 64-bit
/// over the bytes, lowercase hex; deterministic, no deps. MUST stay byte-identical
/// to `scratchy_builder_spyre::sengraph_fingerprint` (kept in sync by hand — a
/// drift just misses the cache and falls back to CompileGraph, never miscompiles).
/// Emit the per-model [`scratchy_target_spyre::wiring::Wiring`] value.
///
/// ⛔ EVERY DERIVATION HERE USED TO RUN IN THE WORKER, AT MODEL LOAD, AGAINST
/// PARSED JSON. Doing it at expansion is not an optimisation — it is the
/// difference between a fact and a re-derivation:
///
/// - ROLES were `match s.role.as_str()` over `"embed"`/`"cos"`/`"prefix_k"`;
///   here they are the `SourceBinding` enum the compiler already has, mapped
///   to spyre's own enum, so an unhandled role is a build error not a `_ => {}`.
/// - LAYER WIRING was recovered by scanning every node for one mentioning the
///   mask tensor and then taking *the argument positionally after* the prefix-K
///   arg as `new_k`. That positional inference is done here instead, once,
///   where the emitter's own arg order is authoritative.
/// - GEOMETRY was re-read from `config.json` behind `unwrap_or` defaults — a
///   second source of truth against the bounds the tape was lowered at. It is
///   taken from those same bounds here, so the two cannot disagree.
/// ⭐⭐⭐ THE DIFFERENTIAL — the wiring must say EXACTLY what the manifest said.
///
/// `wiring_to_parsed` replaced `parse_bundle`: the worker used to `serde_json` the manifest and
/// recover the per-layer AttnDecode wiring by a POSITIONAL SCAN of the node list, and now reads an
/// emitted array instead. Those two answers have to be the same array, and until this function
/// nothing checked that they were — the replacement was verified by reading it.
///
/// They are computed from one `GraphKtir`, so a disagreement is a bug in one of the two scans, and
/// the consequence is not subtle: the worker indexes `layers[i]` for layer `i`, so a single
/// dropped or reordered entry binds one layer's K/V cache to another layer's tids. On a device
/// with no stack to attach to that is a wrong address, which is a hang or silent corruption.
///
/// The manifest is still emitted (the KTIR emulator's session takes it), so both answers exist at
/// bake and this costs one parse per model.
#[cfg(feature = "spyre")]
fn refuse_if_wiring_disagrees_with_manifest(
    stem: &str,
    manifest_json: &str,
    emitted_layers: &[(u64, u32, u32, u32, u32)],
    emitted_embed: u32,
    emitted_cos: &[(u32, u32)],
    emitted_sin: &[(u32, u32)],
    // Which source indices the GENERATED loader will emit a `BoundWeight` for.
    emitted_weight_ids: &std::collections::BTreeSet<u32>,
) {
    use scratchy_target_spyre::manifest::Manifest;
    let m = match Manifest::from_json(manifest_json) {
        Ok(m) => m,
        // The manifest is a bake artifact of this same function; if it will not parse, the
        // KTIR-emulator path is already broken and that is its own failure, not this check's.
        Err(e) => panic!("[superdsc-wiring] {stem}: the bake's own manifest does not parse: {e}"),
    };

    // ── the OLD extraction, exactly as `parse_bundle` performed it ──
    let (mut pk, mut pv): (Vec<(u64, usize)>, Vec<(u64, usize)>) = (Vec::new(), Vec::new());
    let (mut embed, mut cos, mut sin) = (None, Vec::new(), Vec::new());
    for s in &m.sources {
        match s.role.as_str() {
            "prefix_k" => pk.push((s.layer.unwrap_or(0), s.id)),
            "prefix_v" => pv.push((s.layer.unwrap_or(0), s.id)),
            "embed" => embed = Some(s.id as u32),
            "cos" => cos.push(s.id as u32),
            "sin" => sin.push(s.id as u32),
            _ => {}
        }
    }
    pk.sort_by_key(|x| x.0);
    pv.sort_by_key(|x| x.0);

    let mut fail: Vec<String> = Vec::new();
    if embed != Some(emitted_embed) {
        fail.push(format!(
            "embed source: manifest says {embed:?}, wiring says {emitted_embed}"
        ));
    }
    let emit_cos: Vec<u32> = emitted_cos.iter().map(|&(i, _)| i).collect();
    let emit_sin: Vec<u32> = emitted_sin.iter().map(|&(i, _)| i).collect();
    if cos != emit_cos {
        fail.push(format!(
            "cos sources: manifest {cos:?}, wiring {emit_cos:?}"
        ));
    }
    if sin != emit_sin {
        fail.push(format!(
            "sin sources: manifest {sin:?}, wiring {emit_sin:?}"
        ));
    }
    // ── the WEIGHT set: what the loader stages must be what the bundle declares ──
    //
    // ⛔ `load_weights` walked `manifest.sources` filtering `role == "weight" || "weight_scale"`
    // and used `s.id` as the tid. `stage_bound_weights` walks the GENERATED `BoundWeight` list and
    // uses the binding index. Those are the same number only if the two lists agree, and nothing
    // checked that they do — a weight in one and not the other is either a tensor the device reads
    // that nobody staged, or bytes written into a placement that belongs to something else.
    let manifest_weights: std::collections::BTreeSet<u32> = m
        .sources
        .iter()
        .filter(|s| s.role == "weight" || s.role == "weight_scale")
        .map(|s| s.id as u32)
        .collect();
    if &manifest_weights != emitted_weight_ids {
        let only_manifest: Vec<u32> = manifest_weights
            .difference(emitted_weight_ids)
            .copied()
            .collect();
        let only_emitted: Vec<u32> = emitted_weight_ids
            .difference(&manifest_weights)
            .copied()
            .collect();
        if !only_manifest.is_empty() {
            fail.push(format!(
                "{} weight source(s) the bundle declares that the generated loader does NOT stage: \
                 {:?} — the device reads a tensor nobody wrote",
                only_manifest.len(),
                only_manifest
            ));
        }
        if !only_emitted.is_empty() {
            fail.push(format!(
                "{} weight source(s) the generated loader stages that the bundle does NOT declare: \
                 {:?} — bytes land in a placement belonging to something else",
                only_emitted.len(),
                only_emitted
            ));
        }
    }

    if pk.len() != emitted_layers.len() {
        fail.push(format!(
            "layer count: manifest has {} prefix_k sources, wiring emits {} layers",
            pk.len(),
            emitted_layers.len()
        ));
    }
    // ⛔ POSITION BY POSITION. The worker indexes `layers[i]` for layer `i`, so "same set, other
    // order" is exactly as wrong as a missing entry.
    for (i, (&(mlayer, mk), &(wlayer, wk, wv, _, _))) in pk.iter().zip(emitted_layers).enumerate() {
        if mlayer != wlayer || mk != wk as usize {
            fail.push(format!(
                "layers[{i}]: manifest says layer {mlayer} prefix_k t{mk}, wiring says layer \
                 {wlayer} prefix_k t{wk}"
            ));
        }
        let _ = wv;
    }
    for (i, (&(mlayer, mv), &(_, _, wv, _, _))) in pv.iter().zip(emitted_layers).enumerate() {
        if mv != wv as usize {
            fail.push(format!(
                "layers[{i}] (layer {mlayer}): manifest says prefix_v t{mv}, wiring says t{wv}"
            ));
        }
    }

    if !fail.is_empty() {
        panic!(
            "\n⛔ [superdsc-wiring] {stem}: THE EMITTED WIRING DISAGREES WITH THE MANIFEST IT \
             REPLACED. Refusing to bake.\n\n{}\n\nBoth are derived from one `GraphKtir`, so one \
             of the two scans is wrong. The worker indexes `layers[i]` for layer `i`: a dropped or \
             reordered entry binds one layer's K/V cache to another layer's tids, which on the \
             device is a wrong address — a hang or silent corruption, reported by nothing.\n",
            fail.iter()
                .map(|f| format!("  • {f}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
}

#[cfg(feature = "spyre")]
fn emit_superdsc_wiring(
    gk: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::BundleWiring,
    lwd: &crate::to_wavefront::LoweredDecode,
    model: &ModelParams,
    decode_position: u32,
    // The manifest this wiring REPLACES. Both come from `gk`; they must agree, and
    // `refuse_if_wiring_disagrees_with_manifest` is what checks that they do.
    manifest_json: &str,
) -> proc_macro2::TokenStream {
    use crate::to_wavefront::SourceBinding;
    let u32l = proc_macro2::Literal::u32_unsuffixed;

    // ── The DERIVED ANSWERS, resolved at bake ──
    //
    // ⛔ NO ROLE ARRAY IS EMITTED. The worker used to receive `&[Source]` and
    // SCAN it at every model load for the embed row and the cos/sin sources —
    // a compile-time fact re-derived at runtime, which is the same tell as the
    // JSON `role` strings it replaced, only better typed. The exhaustive match
    // over `SourceBinding` still happens, HERE, where a new variant is a build
    // error; what crosses to the runtime is its ANSWER.
    //
    // Both lookups REFUSE rather than defaulting: a wiring whose host gather
    // has nowhere to land, or which carries no rotary, is not a wiring.
    let embed_src = lwd
        .bindings
        .iter()
        .position(|b| matches!(b, SourceBinding::EmbeddedHidden))
        .map(|i| i as u32)
        .unwrap_or_else(|| {
            panic!(
                "[superdsc-wiring] {}: no EmbeddedHidden source — refusing to bake a wiring \
                 whose host gather has nowhere to land",
                model.source_stem
            )
        });
    // cos/sin carry their FULL column width: GQA gives more than one (Q vs K),
    // each filled by tiling the per-position rotary row across heads.
    let rot_srcs = |want_cos: bool| -> Vec<(u32, u32)> {
        lwd.bindings
            .iter()
            .enumerate()
            .filter(|(_, b)| match b {
                SourceBinding::Cos { .. } => want_cos,
                SourceBinding::Sin { .. } => !want_cos,
                _ => false,
            })
            .map(|(i, _)| (i as u32, gk.tensor_shapes[i].1))
            .collect()
    };
    let (cos_v, sin_v) = (rot_srcs(true), rot_srcs(false));
    if cos_v.is_empty() || sin_v.is_empty() {
        panic!(
            "[superdsc-wiring] {}: no cos/sin source — refusing to bake a wiring with no rotary",
            model.source_stem
        );
    }
    let pair_toks = |v: &[(u32, u32)]| -> Vec<proc_macro2::TokenStream> {
        v.iter()
            .map(|(i, w)| {
                let (i, w) = (u32l(*i), u32l(*w));
                quote! { (#i, #w) }
            })
            .collect()
    };
    let (cos_toks, sin_toks) = (pair_toks(&cos_v), pair_toks(&sin_v));
    let embed_src_lit = u32l(embed_src);

    let shapes = gk.tensor_shapes.iter().map(|(r, c)| {
        let (r, c) = (u32l(*r), u32l(*c));
        quote! { (#r, #c) }
    });

    // ── Per-layer AttnDecode wiring ──
    // The worker's own walk, run here: find each node that reads the mask, and
    // take the prefix-K/V args plus the argument positionally after each as the
    // new-K/new-V outputs.
    let mut layers: Vec<(u64, u32, u32, u32, u32)> = Vec::new();
    if let Some(mask) = gk.attn_mask {
        let pk: std::collections::HashMap<u32, u64> = lwd
            .bindings
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b {
                SourceBinding::PrefixK { layer } => Some((i as u32, *layer)),
                _ => None,
            })
            .collect();
        let pv: std::collections::HashSet<u32> = lwd
            .bindings
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b {
                SourceBinding::PrefixV { .. } => Some(i as u32),
                _ => None,
            })
            .collect();
        for node in &gk.nodes {
            if !node.args.iter().any(|a| *a as u32 == mask) {
                continue;
            }
            let (mut layer, mut k_src, mut new_k) = (None, None, None);
            let (mut v_src, mut new_v) = (None, None);
            for (i, a) in node.args.iter().map(|a| *a as u32).enumerate() {
                if let Some(&l) = pk.get(&a) {
                    layer = Some(l);
                    k_src = Some(a);
                    new_k = node.args.get(i + 1).map(|x| *x as u32);
                }
                if pv.contains(&a) {
                    v_src = Some(a);
                    new_v = node.args.get(i + 1).map(|x| *x as u32);
                }
            }
            if let (Some(l), Some(ks), Some(nk), Some(vs), Some(nv)) =
                (layer, k_src, new_k, v_src, new_v)
            {
                layers.push((l, ks, vs, nk, nv));
            }
        }
        layers.sort_by_key(|x| x.0);
    }
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⛔⛔⛔ THE COMPLETENESS LOCK — a wiring that does not fill every caller-filled source
    //  MAY NOT COMPILE.
    //
    //  `num_sources` says ids `0..num_sources` are FILLED BY THE HOST every forward. Four things
    //  fill them: the generated weight loader (`Weight`/`WeightScale`), the forward tape's embed
    //  gather (`EmbeddedHidden`) and rotary kernels (`Cos`/`Sin`), and the resident KV pool
    //  (`PrefixK`/`PrefixV`, bound per layer through `LayerWiring`). A source that NONE of them
    //  covers is never written — and the device does not report that. It waits, or it reads
    //  whatever the segment held.
    //
    //  Every check below is a hole that existed and was silent:
    //
    //    • `layers` is built by POSITIONAL SCAN over each AttnDecode node's args, and its
    //      `if let (Some, Some, Some, Some, Some)` DROPPED a layer whose pattern did not match.
    //      Forty layers in, thirty-nine wired, no message.
    //    • `embed_src` is `.position(..)` — the FIRST `EmbeddedHidden`. A second one is never
    //      filled by anything.
    //    • `cos_srcs`/`sin_srcs` are collected by filter, so they cannot miss one — asserted
    //      anyway, because that is a property of the collection and not of the enum.
    //
    //  This is the sibling of `superdsc_bake.rs`'s no-device-compiler refusal, and for the same
    //  reason it gives: the binary would link, every session would report ready, and the failure
    //  would surface as a hang or as empty output with no error anywhere.
    {
        use crate::to_wavefront::SourceBinding as SB;
        let n_layers_expected = model
            .bounds
            .get("num_hidden_layers")
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "[superdsc-wiring] {}: no `num_hidden_layers` bound — cannot check that every \
                     layer's prefix KV is wired, and an unwired layer reads a cache nobody filled",
                    model.source_stem
                )
            }) as usize;

        let mut fail: Vec<String> = Vec::new();

        // ── every layer's prefix KV is wired ──
        if layers.len() != n_layers_expected {
            fail.push(format!(
                "{} of {n_layers_expected} layers have prefix-KV wiring. The rest were dropped by \
                 the positional arg scan; their K/V cache would never be bound",
                layers.len(),
            ));
        }
        let mut seen_layers: Vec<u64> = layers.iter().map(|x| x.0).collect();
        seen_layers.sort_unstable();
        seen_layers.dedup();
        if seen_layers.len() != layers.len() {
            fail.push(format!(
                "two AttnDecode nodes claim the same layer index — one layer's cache would be \
                 bound twice and another's not at all ({:?})",
                layers.iter().map(|x| x.0).collect::<Vec<_>>()
            ));
        }

        // ── every caller-filled source is covered by exactly one filler ──
        let cos_ids: std::collections::HashSet<u32> = cos_v.iter().map(|&(i, _)| i).collect();
        let sin_ids: std::collections::HashSet<u32> = sin_v.iter().map(|&(i, _)| i).collect();
        let mut embeds = 0usize;
        for id in 0..gk.num_sources {
            let Some(b) = lwd.bindings.get(id as usize) else {
                fail.push(format!(
                    "source t{id} is inside `num_sources` but has NO binding — nothing fills it"
                ));
                continue;
            };
            // ⛔ EXHAUSTIVE, NO CATCH-ALL. A new role must be taught a filler here, at build time.
            match b {
                // The generated weight loader binds these once, before prepare.
                SB::Weight { .. } | SB::WeightScale { .. } => {}
                SB::EmbeddedHidden => {
                    embeds += 1;
                    if id != embed_src {
                        fail.push(format!(
                            "source t{id} is an EmbeddedHidden the tape does not gather \
                             (it gathers t{embed_src}) — t{id} is never written"
                        ));
                    }
                }
                SB::Cos { .. } => {
                    if !cos_ids.contains(&id) {
                        fail.push(format!("Cos source t{id} is not in the tape's cos_srcs"));
                    }
                }
                SB::Sin { .. } => {
                    if !sin_ids.contains(&id) {
                        fail.push(format!("Sin source t{id} is not in the tape's sin_srcs"));
                    }
                }
                // Device-resident, bound per layer through `LayerWiring` — covered by the layer
                // count check above.
                SB::PrefixK { .. } | SB::PrefixV { .. } => {}
            }
        }
        if embeds != 1 {
            fail.push(format!(
                "{embeds} EmbeddedHidden sources; the tape gathers exactly one row"
            ));
        }

        if !fail.is_empty() {
            panic!(
                "\n⛔ [superdsc-wiring] {}: THE LAUNCH WIRING DOES NOT FILL EVERY CALLER-FILLED \
                 SOURCE. Refusing to bake.\n\n{}\n\nA binary built from this wiring LINKS AND \
                 SERVES: the session reports ready and the launch is issued, but a source nobody \
                 writes leaves the device reading memory the host never filled — which surfaces as \
                 a HANG or as fluent-but-wrong output, with no error anywhere. Same reason \
                 `superdsc_bake` refuses a plan with no device programs.\n",
                model.source_stem,
                fail.iter()
                    .map(|f| format!("  • {f}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
    }

    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⛔⛔⛔ A GEMM WEIGHT'S DECLARED SHAPE IS `[k, n]`, AND THE CONSUMING OP SAYS WHICH IS WHICH.
    //
    //  `tensor_shapes[w]` is the pair the host stages from: it decides the orientation guard, the
    //  device-width pad, the K-split gate and the emitted shape. It is `[rows = k, cols = n]` — the
    //  LOGICAL `[in, out]` — and the ONLY evidence of that was a comment.
    //
    //  Reading it as `[n, k]` is not an error, it is a plausible answer, and it shipped: the pad
    //  became `for_output(1, k, n)`, a NO-OP for granite's lm_head (k = 2048 is already 32 sticks)
    //  where its n (49155 → 769 PRIME sticks) needs the full-occupancy bump to 51200. The host
    //  staged 49155 unpadded columns into a placement sized for 51200.
    //
    //  The consuming `Gemm` carries `n`. So the orientation is checkable, and this checks it.
    // ══════════════════════════════════════════════════════════════════════════════════════════
    {
        use crate::to_wavefront::SourceBinding as SB;
        use scratchy_subtile::lower::{InputRef, LoweredOp};
        let mut bad: Vec<String> = Vec::new();
        for od in &lwd.input.ops {
            let LoweredOp::Gemm { n, .. } = od.op else {
                continue;
            };
            let Some(InputRef::Ext(e)) = od.inputs.get(1) else {
                continue;
            };
            // Only a real weight has a staged shape; a fp8 scale is a different tensor.
            if !matches!(lwd.bindings.get(*e), Some(SB::Weight { .. })) {
                continue;
            }
            let Some(&(rows, cols)) = gk.tensor_shapes.get(*e) else {
                bad.push(format!("gemm weight t{e} has no entry in tensor_shapes"));
                continue;
            };
            if cols != n {
                bad.push(format!(
                    "gemm weight t{e}: the op produces n={n}, but tensor_shapes says \
                     [rows={rows}, cols={cols}] — cols is supposed to BE n. If these are \
                     transposed, every host consumer of this pair pads, gates and shapes the \
                     wrong axis"
                ));
            }
        }
        if !bad.is_empty() {
            panic!(
                "\n⛔ [superdsc-wiring] {}: A GEMM WEIGHT'S DECLARED SHAPE DISAGREES WITH ITS \
                 CONSUMING OP. Refusing to bake.\n\n{}\n",
                model.source_stem,
                bad.iter()
                    .map(|b| format!("  • {b}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
    }

    // Exactly the sources `weight_bindings` emits a `BoundWeight` for — the same
    // `SourceBinding::{Weight, WeightScale}` filter, over the same list, at the same indices.
    let emitted_weight_ids: std::collections::BTreeSet<u32> = lwd
        .bindings
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            matches!(
                b,
                SourceBinding::Weight { .. } | SourceBinding::WeightScale { .. }
            )
        })
        .map(|(i, _)| i as u32)
        .collect();
    refuse_if_wiring_disagrees_with_manifest(
        &model.source_stem,
        manifest_json,
        &layers,
        embed_src,
        &cos_v,
        &sin_v,
        &emitted_weight_ids,
    );

    let layer_toks = layers.iter().map(|(_, ks, vs, nk, nv)| {
        let (ks, vs, nk, nv) = (u32l(*ks), u32l(*vs), u32l(*nk), u32l(*nv));
        quote! {
            ::scratchy_target_spyre::wiring::LayerWiring {
                prefix_k: #ks, prefix_v: #vs, new_k: #nk, new_v: #nv,
            }
        }
    });

    // ── Geometry, from the bounds the tape was lowered at ──
    let b = |k: &str| model.bounds.get(k).copied().unwrap_or(0);
    let hidden = u32l(b("hidden_size") as u32);
    let vocab = u32l(b("vocab_size") as u32);
    let n_layers = u32l(b("num_hidden_layers") as u32);
    // head_dim is declared by most arches and derived by the rest — the same
    // `hidden/heads` fallback the emitter itself uses, resolved HERE so the
    // worker never re-derives it (that re-derivation, against a config whose
    // `num_attention_heads` disagreed with the baked bundle, is a known
    // gemma-4-class failure).
    let heads = b("num_attention_heads").max(1);
    let hd = match b("head_dim") {
        0 => b("hidden_size") / heads,
        v => v,
    };
    let head_dim = u32l(hd as u32);
    let kv_dim = u32l((b("num_key_value_heads").max(1) * hd) as u32);
    // ⛔ SCALARS FIRST, AND NO DEFAULT. `rope_theta` is a FLOAT, so it lives in
    // `model.scalars`; `bounds` is integer-valued and a bounds-only lookup
    // misses it entirely. With an `unwrap_or(1e4)` behind it that miss is
    // SILENT and emits θ=10000 for llama-3.2 (whose real θ is 500000) — a
    // wrong baked constant that produces fluent, subtly-wrong output. The
    // whole point of baking geometry is to have ONE answer, so a missing θ
    // fails the build here exactly as metal's `.expect` does.
    let theta = model
        .scalars
        .get("rope_theta")
        .copied()
        .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
        .unwrap_or_else(|| {
            panic!(
                "[superdsc-wiring] {}: no rope_theta in scalars or bounds — refusing to bake a \
                 default; the baked table would silently disagree with the checkpoint",
                model.source_stem
            )
        }) as f32;
    let theta_bits = u32l(theta.to_bits());
    let result = u32l(gk.result_tensor);
    let num_sources = u32l(gk.num_sources);
    let dp = u32l(decode_position);
    let mask = match gk.attn_mask {
        Some(m) => {
            let m = u32l(m);
            quote! { ::core::option::Option::Some(#m) }
        }
        None => quote! { ::core::option::Option::None },
    };

    // ⭐ THE TWO KERNEL TABLES, EVALUATED HERE. `identity` and `rope_p` are pure functions of
    // `head_dim`, and the worker rebuilt both on EVERY forward via `stage_2d` — 2·hd² f32 per
    // token to produce the same bytes each time. Computed by the SAME shared helpers the runtime
    // called (`StickLayout::kernel` + `rope_p_entry`, the latter Kani-proven), so the emitted
    // table and the proof cannot drift.
    //
    // ⛔ STAGED THROUGH THE KERNEL LAYOUT, NOT ROW-MAJOR. A reserved seg0 tid has no
    // RetileDescriptor, so the host fill IS the device layout. The two coincide only at hd == 64;
    // at hd == 128 16,256 of 16,384 identity entries would come from the wrong byte.
    let (identity_lits, rope_p_lits) = {
        use scratchy_subtile::sdsc_abstract::{StickLayout, rope_p_entry, stage_2d};
        let hdu = hd as usize;
        let ident = stage_2d(&StickLayout::kernel(hdu, hdu), |i, j| {
            if i == j { 1.0 } else { 0.0 }
        });
        // `hd >= 2` is the runtime's own guard on emitting a rope-P at all.
        let ropep: Vec<f32> = if hdu >= 2 {
            stage_2d(&StickLayout::kernel(hdu, hdu), |inn, o| {
                rope_p_entry(hdu, inn, o) as f32
            })
        } else {
            Vec::new()
        };
        let f = |v: Vec<f32>| -> Vec<proc_macro2::Literal> {
            v.into_iter()
                .map(proc_macro2::Literal::f32_suffixed)
                .collect()
        };
        (f(ident), f(ropep))
    };
    // `1/hidden` over one stick — the mq>1 sum-based amax pre-scale.
    let rms_invcols_lits: Vec<proc_macro2::Literal> = {
        let inv = 1.0f32 / (b("hidden_size").max(1) as f32);
        (0..64)
            .map(|_| proc_macro2::Literal::f32_suffixed(inv))
            .collect()
    };
    // ⭐ EVERY COMPILE-TIME SCALAR THE KTIR READS, from the bake's own registry — not recomputed
    // here from the config. `KtirFunc::splat_scale` bakes the INDEX into each program, so the list
    // the worker binds has to be the same list, in the same order, that the lowering indexed.
    let scalarmul_scale_lits: Vec<proc_macro2::Literal> = gk
        .scalarmul_scales
        .iter()
        .map(|v| proc_macro2::Literal::f32_suffixed(*v))
        .collect();

    // The VALUE, not a `static` — the caller composes decode + prefill into one
    // `Wirings` const, because a bundle's tensor ids are per-PROGRAM and the
    // prefill program has its own.
    quote! {
        ::scratchy_target_spyre::wiring::Wiring {
            result: #result,
            num_sources: #num_sources,
            attn_mask: #mask,
            decode_position: #dp,
            tensor_shapes: &[#(#shapes),*],
            embed_src: #embed_src_lit,
            cos_srcs: &[#(#cos_toks),*],
            sin_srcs: &[#(#sin_toks),*],
            layers: &[#(#layer_toks),*],
            geometry: ::scratchy_target_spyre::wiring::Geometry {
                hidden: #hidden,
                kv_dim: #kv_dim,
                vocab: #vocab,
                layers: #n_layers,
                head_dim: #head_dim,
                rope_theta_bits: #theta_bits,
            },
            identity: &[#(#identity_lits),*],
            rope_p: &[#(#rope_p_lits),*],
            rms_invcols: &[#(#rms_invcols_lits),*],
            scalarmul_scales: &[#(#scalarmul_scale_lits),*],
        }
    }
}

/// The SuperDSC AoT cache root (peer of `sengraphforge_cache_dir`). The
/// proc-macro DROPS one bundle dir `<fp>/` (sdsc_{i}.json + bundle.mlir) per
/// runnable graph here when `SCRATCHY_SENDNN_MODE=superdsc`; build.rs will
/// `dxp_standalone --bundle -d <fp>` each into a device program. Fixed path so
/// both sides agree without a build-graph edge — same contract as sengraphforge.
/// ⭐⭐⭐⭐⭐ TURN THE EMITTED BUNDLES INTO RUST STRUCTS — the whole of getting a SuperDSC bundle into
/// the binary.
///
/// The emitter hands over a `Vec<BundleCode>` (`drain_emitted_bundles`) and this renders each as a
/// `const`-constructed value inside `inventory::submit!`, which the runtime looks up by fingerprint.
/// Every quantity stays in its own type from the lowering that computed it to the launch that reads it.
///
/// ## The device image is `include_bytes!`
///
/// One model's rung ladder is ~100-290 MB of device code, and that many tokens would not survive the
/// macro, let alone `prettyplease` — so each image is written to `OUT_DIR` and referenced by
/// `include_bytes!`. That is a mechanism for getting BYTES into a binary, not a serialization format:
/// nothing parses it, at build time or after. `OUT_DIR` is cargo's own per-build directory for the crate
/// being expanded, and every arch crate has a `build.rs`, so it is set here.
#[cfg(feature = "spyre")]
fn superdsc_bundle_tokens() -> proc_macro2::TokenStream {
    // ⭐⭐⭐ WHAT IS INVENTORIED DEPENDS ON THE DEVICE, AND ONLY ON THE DEVICE.
    //
    // On a CARD the launch names a resident device image by VA, so that image is compiled from this
    // bundle's KTIR at bake time and the emit must drain the compiler before it can read the
    // artifacts back. That is this barrier, and its refusal on a host without one is the right
    // answer: a bundle whose programs never compiled would serve, launch nothing, and return empty
    // completions with no error anywhere.
    //
    // ⛔ ON THE EMULATOR THERE IS NOTHING TO COMPILE. The device runs the KTIR itself, so the
    // programs the group already carries ARE what gets inventoried — nothing is submitted to that
    // queue and nothing reads a result from it. Draining it would refuse a build that has every
    // program it needs.
    #[cfg(feature = "spyre-hw")]
    match scratchy_target_spyre::superdsc_bake::finish_global() {
        Ok(Some(stats)) => eprintln!(
            "[spyre] the device compiler built {} launch group(s) inline during the emit ({:.1} MB \
             of device code); {} skipped as byte-identical to one already compiled; peak staging \
             {:.1} MB",
            stats.groups,
            stats.device_bytes as f64 / 1048576.0,
            stats.memo_hits,
            stats.peak_staged_bytes as f64 / 1048576.0,
        ),
        Ok(None) => {}
        Err(e) => panic!("[spyre] {e}"),
    }
    let bundles =
        match scratchy_target_spyre::lower_subtile_tape_to_superdsc::drain_emitted_bundles() {
            Ok(b) => b,
            Err(e) => panic!("[spyre-superdsc] {e}"),
        };
    if bundles.is_empty() {
        return proc_macro2::TokenStream::new();
    }
    let images = match std::env::var("OUT_DIR") {
        Ok(d) => std::path::PathBuf::from(d).join("superdsc-code"),
        // No OUT_DIR means we are not inside a build script's expansion, so there is nowhere to put
        // the images and nothing that could `include_bytes!` them.
        Err(_) => return proc_macro2::TokenStream::new(),
    };
    let mut out = proc_macro2::TokenStream::new();
    let (mut n_groups, mut n_bytes) = (0usize, 0usize);
    // ⭐ ONE INTERNER FOR THE WHOLE EMISSION — every bundle's launches reach the same 33-ish
    // programs, so they share one set of `const` items rather than each carrying its own copy.
    let mut interner = crate::ktir_tokens::ProgramInterner::default();
    for b in &bundles {
        let fp = proc_macro2::Literal::string(&b.fp);
        let layout = layout_tokens(&b.layout);
        let groups = b
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                n_groups += 1;
                n_bytes += g.init_binary.len();
                launch_tokens(g, i, &b.fp, &images, &mut interner)
            })
            .collect::<Vec<_>>();
        let reroll = match &b.reroll {
            Some(m) => {
                let m = reroll_tokens(m);
                quote! { ::core::option::Option::Some(#m) }
            }
            None => quote! { ::core::option::Option::None },
        };
        out.extend(quote! {
            ::scratchy_target_spyre::bundle_code::inventory::submit! {
                ::scratchy_target_spyre::bundle_code::BundleCode {
                    fp: ::std::borrow::Cow::Borrowed(#fp),
                    layout: #layout,
                    groups: ::std::borrow::Cow::Borrowed(&[#(#groups),*]),
                    reroll: #reroll,
                }
            }
        });
    }
    // The programs are `const` items, so they must precede the `inventory::submit!`s that name them.
    let programs = interner.items();
    let preamble = crate::ktir_tokens::preamble();
    out = quote! { #preamble #(#programs)* #out };
    eprintln!(
        "[spyre-superdsc] baked {} bundle(s) / {n_groups} launch group(s) / {} distinct program(s) \
         / {:.1} MB of device code into the binary as Rust structs",
        bundles.len(),
        interner.items().len(),
        n_bytes as f64 / 1048576.0,
    );
    out
}

/// A `Cow::Borrowed` string literal — every `Cow<'static, str>` field in the baked artifact.
#[cfg(feature = "spyre")]
fn cow_str(s: &str) -> proc_macro2::TokenStream {
    let lit = proc_macro2::Literal::string(s);
    quote! { ::std::borrow::Cow::Borrowed(#lit) }
}

/// A `Cow::Borrowed` slice of unsuffixed integer literals.
#[cfg(feature = "spyre")]
fn cow_u64s(v: &[u64]) -> proc_macro2::TokenStream {
    let lits = v.iter().map(|x| proc_macro2::Literal::u64_unsuffixed(*x));
    quote! { ::std::borrow::Cow::Borrowed(&[#(#lits),*]) }
}

/// One [`bundle::Placement`].
///
/// ⛔ THE DESTRUCTURES IN THIS FAMILY OF FUNCTIONS ARE THE GUARD, in both directions: a field ADDED to
/// the artifact type leaves the `quote!` below incomplete (a missing-field error in the expanded code),
/// and a field REMOVED leaves the destructure naming something that no longer exists. So the baked value
/// cannot drift from the type it is a value of.
/// One [`bundle::SynthRole`] as tokens.
///
/// ⛔ AN EXHAUSTIVE MATCH, NOT A STRING. A synthetic's role used to be a name SUFFIX
/// (`format!("{out}_rot")`) built at the emit site and matched at the consume site. Here the
/// bake holds the value and the macro emits the value; adding a role breaks THIS match at build
/// time, which is the whole reason it is an enum.
#[cfg(feature = "spyre")]
fn synth_role_tokens(
    r: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SynthRole,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SynthRole as R;
    let b = quote! { ::scratchy_target_spyre::bundle_code::SynthRole };
    match r {
        R::Rot => quote! { #b::Rot },
        R::Xc => quote! { #b::Xc },
        R::Rs => quote! { #b::Rs },
        R::Silu => quote! { #b::Silu },
        R::Qs => quote! { #b::Qs },
        R::KRep => quote! { #b::KRep },
        R::VRep => quote! { #b::VRep },
        R::NewKRep => quote! { #b::NewKRep },
        R::NewVRep => quote! { #b::NewVRep },
        R::NewKScaled => quote! { #b::NewKScaled },
        R::FqAbsX => quote! { #b::FqAbsX },
        R::FqAfp8 => quote! { #b::FqAfp8 },
        R::FqAmax => quote! { #b::FqAmax },
        R::FqAmaxFl => quote! { #b::FqAmaxFl },
        R::FqAscale => quote! { #b::FqAscale },
        R::FqChi => quote! { #b::FqChi },
        R::FqCl => quote! { #b::FqCl },
        R::FqInvS => quote! { #b::FqInvS },
        R::FqSc => quote! { #b::FqSc },
        R::FqDqA => quote! { #b::FqDqA },
        R::FqMm => quote! { #b::FqMm },
        R::FqRaw => quote! { #b::FqRaw },
        R::Sq16 => quote! { #b::Sq16 },
        R::Mean => quote! { #b::Mean },
        R::Meps => quote! { #b::Meps },
        R::Rinv => quote! { #b::Rinv },
        R::Xn => quote! { #b::Xn },
        R::GatherKt => quote! { #b::GatherKt },
        R::GatherV => quote! { #b::GatherV },
        R::NewKt => quote! { #b::NewKt },
        R::Sc => quote! { #b::Sc },
        R::BMax => quote! { #b::BMax },
        R::NewM => quote! { #b::NewM },
        R::Corr => quote! { #b::Corr },
        R::CorrSubT => quote! { #b::CorrSubT },
        R::ExpB => quote! { #b::ExpB },
        R::ESubT => quote! { #b::ESubT },
        R::BSum => quote! { #b::BSum },
        R::OTmp => quote! { #b::OTmp },
        R::LTmp => quote! { #b::LTmp },
        R::RunM => quote! { #b::RunM },
        R::RunL => quote! { #b::RunL },
        R::RunO => quote! { #b::RunO },
        R::Blk(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::Blk(#i) }
        }
        R::Acc(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::Acc(#i) }
        }
        R::LBlk(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::LBlk(#i) }
        }
        R::LAcc(i) => {
            let i = proc_macro2::Literal::u32_unsuffixed(*i);
            quote! { #b::LAcc(#i) }
        }
    }
}

/// One [`bundle::PlaceId`] as tokens — the identity a placement is keyed by.
#[cfg(feature = "spyre")]
fn place_id_tokens(
    id: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::PlaceId,
) -> proc_macro2::TokenStream {
    use scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::PlaceId as P;
    match id {
        P::Act(t) => {
            let t = proc_macro2::Literal::u32_unsuffixed(*t);
            quote! { ::scratchy_target_spyre::bundle_code::PlaceId::Act(#t) }
        }
        P::Synth { of, role } => {
            let of = proc_macro2::Literal::u32_unsuffixed(*of);
            let role = synth_role_tokens(role);
            quote! {
                ::scratchy_target_spyre::bundle_code::PlaceId::Synth { of: #of, role: #role }
            }
        }
    }
}

#[cfg(feature = "spyre")]
fn placement_tokens(
    p: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::Placement,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::Placement {
        id,
        segment,
        bank,
        offset,
        size,
        is_logits,
    } = p;
    let id = place_id_tokens(id);
    let (segment, bank, offset, size) = (
        proc_macro2::Literal::u32_unsuffixed(*segment),
        proc_macro2::Literal::u32_unsuffixed(*bank),
        proc_macro2::Literal::u64_unsuffixed(*offset),
        proc_macro2::Literal::u64_unsuffixed(*size),
    );
    quote! {
        ::scratchy_target_spyre::bundle_code::Placement {
            id: #id, segment: #segment, bank: #bank, offset: #offset, size: #size,
            is_logits: #is_logits,
        }
    }
}

/// One [`bundle::KernelWeight`].
#[cfg(feature = "spyre")]
fn kernel_weight_tokens(
    k: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::KernelWeight<'_>,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::KernelWeight {
        id,
        device_size,
        stride_map,
        stick_size,
        word_length,
    } = k;
    let id = place_id_tokens(id);
    let (device_size, stride_map) = (cow_u64s(device_size), cow_u64s(stride_map));
    let (stick_size, word_length) = (
        proc_macro2::Literal::u32_unsuffixed(*stick_size),
        proc_macro2::Literal::u32_unsuffixed(*word_length),
    );
    quote! {
        ::scratchy_target_spyre::bundle_code::KernelWeight {
            id: #id, device_size: #device_size, stride_map: #stride_map,
            stick_size: #stick_size, word_length: #word_length,
        }
    }
}

/// The whole memory plan.
#[cfg(feature = "spyre")]
fn layout_tokens(
    l: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::BundleLayout<'_>,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::BundleLayout {
        segment_bytes,
        weight_bank_bytes,
        places,
        kernel_weights,
        scalarmul_scales,
        kv_request_stride_bytes,
    } = l;
    let segs = segment_bytes
        .iter()
        .map(|b| proc_macro2::Literal::u64_unsuffixed(*b));
    let wbanks = weight_bank_bytes
        .iter()
        .map(|b| proc_macro2::Literal::u64_unsuffixed(*b));
    let places = places.iter().map(placement_tokens);
    let kws = kernel_weights.iter().map(kernel_weight_tokens);
    // f32 literals: `Literal::f32_suffixed` round-trips the exact value, which matters — these are
    // the granite ScalarMul multipliers, and a rounded one is a quietly wrong model.
    let scales = scalarmul_scales
        .iter()
        .map(|s| proc_macro2::Literal::f32_suffixed(*s));
    let krs = proc_macro2::Literal::u64_unsuffixed(*kv_request_stride_bytes);
    quote! {
        ::scratchy_target_spyre::bundle_code::BundleLayout {
            segment_bytes: [#(#segs),*],
            weight_bank_bytes: ::std::borrow::Cow::Borrowed(&[#(#wbanks),*]),
            places: ::std::borrow::Cow::Borrowed(&[#(#places),*]),
            kernel_weights: ::std::borrow::Cow::Borrowed(&[#(#kws),*]),
            scalarmul_scales: ::std::borrow::Cow::Borrowed(&[#(#scales),*]),
            kv_request_stride_bytes: #krs,
        }
    }
}

/// ONE LAUNCH: its address shifts and its compiled program, as a single value. The device image is
/// written to `OUT_DIR` and referenced by `include_bytes!`; everything else is a literal.
#[cfg(feature = "spyre")]
fn launch_tokens(
    // `'static` because a baked program's operations live in the arena, which outlives the whole
    // expansion — the tokens borrow it rather than copying it.
    g: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LaunchGroup<'static>,
    index: usize,
    fp: &str,
    images: &std::path::Path,
    interner: &mut crate::ktir_tokens::ProgramInterner,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LaunchGroup {
        kv,
        programs,
        init_binary,
        job_bin_ptr,
        correction,
    } = g;
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::KvShifts {
        slot_stride_bytes,
        slab_stride_bytes,
        page_slots,
        request,
        page_fold,
        batched_requests,
        gathered,
        fold_rows,
    } = kv;
    let [slot, slab, slots, req] = [
        *slot_stride_bytes,
        *slab_stride_bytes,
        *page_slots,
        *request,
    ]
    .map(proc_macro2::Literal::u32_unsuffixed);
    let fold_rows = match fold_rows {
        scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::FoldRows::WholeBatch => {
            quote! { ::scratchy_target_spyre::bundle_code::FoldRows::WholeBatch }
        }
        scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::FoldRows::PerRequest => {
            quote! { ::scratchy_target_spyre::bundle_code::FoldRows::PerRequest }
        }
    };
    let jbp = proc_macro2::Literal::u64_unsuffixed(*job_bin_ptr);
    // Launch order is the slice position, so the image path is keyed by it.
    let image = out_dir_bytes(images, &format!("{fp}/group_{index}.bin"), init_binary);
    // The correction flits are kilobytes at most (a 128-byte flit per symbolic address), so they go out
    // as a byte literal rather than a second file.
    let corr = correction
        .iter()
        .map(|b| proc_macro2::Literal::u8_unsuffixed(*b));
    // ⭐ THE PROGRAMS THEMSELVES. A launch group IS its KTIR, so this is what `inventory::submit!`
    // carries: the constructed functions, rendered as const data. On the emulator the device runs
    // exactly these; on a card the image above is compiled FROM them.
    let progs: Vec<_> = programs
        .iter()
        .map(|p| crate::ktir_tokens::program_tokens(p, interner))
        .collect();
    quote! {
        ::scratchy_target_spyre::bundle_code::LaunchGroup {
            kv: ::scratchy_target_spyre::bundle_code::KvShifts {
                slot_stride_bytes: #slot,
                slab_stride_bytes: #slab,
                page_slots: #slots,
                request: #req,
                page_fold: #page_fold,
                batched_requests: #batched_requests,
                gathered: #gathered,
                fold_rows: #fold_rows,
            },
            programs: ::std::borrow::Cow::Borrowed(&[#(#progs),*]),
            init_binary: #image,
            job_bin_ptr: #jbp,
            correction: ::std::borrow::Cow::Borrowed(&[#(#corr),*]),
        }
    }
}

/// Write `bytes` under `OUT_DIR` and return the `include_bytes!` that reads them back at compile time.
///
/// An empty image (dxp compiled a group to a job plan alone) needs no file.
#[cfg(feature = "spyre")]
fn out_dir_bytes(images: &std::path::Path, rel: &str, bytes: &[u8]) -> proc_macro2::TokenStream {
    if bytes.is_empty() {
        return quote! { ::std::borrow::Cow::Borrowed(&[]) };
    }
    let dst = images.join(rel);
    if let Some(parent) = dst.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Err(e) = std::fs::write(&dst, bytes) {
        panic!(
            "[spyre-superdsc] writing the device image to {}: {e} — it is `include_bytes!`d into the \
             binary, so there is no fallback",
            dst.display()
        );
    }
    let lit = proc_macro2::Literal::string(&format!("/superdsc-code/{rel}"));
    quote! {
        ::std::borrow::Cow::Borrowed(
            ::core::include_bytes!(::core::concat!(::core::env!("OUT_DIR"), #lit)) as &[u8]
        )
    }
}

/// The re-rolled layer loop, with every sibling it names.
#[cfg(feature = "spyre")]
fn reroll_tokens(
    m: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::RerollMeta<'_>,
) -> proc_macro2::TokenStream {
    let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::RerollMeta {
        prefix,
        suffix,
        body_fused,
        rungs,
        iters,
        weight_stride,
        kv_stride,
        layers_per_bank,
        prefix_weight_bank,
        suffix_weight_bank,
    } = m;
    let sib = |s: &scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::SiblingFp<'_>| {
        let inner = cow_str(s.as_str());
        quote! { ::scratchy_target_spyre::bundle_code::SiblingFp(#inner) }
    };
    let (prefix, suffix, body_fused) = (sib(prefix), sib(suffix), sib(body_fused));
    let rungs = rungs.iter().map(|r| {
        let scratchy_target_spyre::lower_subtile_tape_to_superdsc::bundle::LadderRung {
            active_cap,
            body,
            body_fused,
        } = r;
        let cap = proc_macro2::Literal::u32_unsuffixed(active_cap.get());
        let (body, body_fused) = (sib(body), sib(body_fused));
        quote! {
            ::scratchy_target_spyre::bundle_code::LadderRung {
                active_cap: ::scratchy_target_spyre::bundle_code::SweptCols::new(#cap),
                body: #body,
                body_fused: #body_fused,
            }
        }
    });
    let iters = proc_macro2::Literal::u32_unsuffixed(*iters);
    let (ws, ks) = (
        proc_macro2::Literal::u64_unsuffixed(*weight_stride),
        proc_macro2::Literal::u64_unsuffixed(*kv_stride),
    );
    let (lpb, pwb, swb) = (
        proc_macro2::Literal::u32_unsuffixed(*layers_per_bank),
        proc_macro2::Literal::u32_unsuffixed(*prefix_weight_bank),
        proc_macro2::Literal::u32_unsuffixed(*suffix_weight_bank),
    );
    quote! {
        ::scratchy_target_spyre::bundle_code::RerollMeta {
            prefix: #prefix,
            suffix: #suffix,
            body_fused: #body_fused,
            rungs: ::std::borrow::Cow::Borrowed(&[#(#rungs),*]),
            iters: #iters,
            weight_stride: #ws,
            kv_stride: #ks,
            layers_per_bank: #lpb,
            prefix_weight_bank: #pwb,
            suffix_weight_bank: #swb,
        }
    }
}

#[allow(clippy::too_many_arguments)]
// `program`, `ktir_bundle_out`, and the `base_to_loc` locals are only
// read inside the spyre cfg block below; without that feature the
// function still lowers + reports stats but consumes none of them.
#[cfg_attr(not(feature = "spyre"), allow(unused_variables))]
fn dump_wavefront_mega(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    decode_asn: &crate::assignment::Assignment,
    inferred: &crate::shape::Inferred,
    decode_bounds: &BTreeMap<String, u64>,
    backbone_slots: &[Vec<WeightSlot>],
    lm_head_slots: &[Vec<WeightSlot>],
    bb_bucket_id: u32,
    // Out: under `-Fspyre` the embedded `pub const KTIR_BUNDLE` token stream
    // (spliced into the per-model module); empty under cuda/metal.
    ktir_bundle_out: &mut proc_macro2::TokenStream,
) {
    use crate::to_wavefront;
    let stem = model.source_stem.as_str();
    // Structural prefix-KV-cache rows (== modeled valid_len). MUST be >= 1:
    // with 0 the cache Source is [0, kvdim] and the AttnDecode's prefix read
    // (`rows..1`) overruns it — `SubtileIR rejected by validate(): node N input
    // 1 region exceeds tensor 7` for EVERY model. The real decode length binds
    // at run time: cuda via the `DecodePosition` kernel arg (its .cu is
    // independent of this value), KTIR via the AttnDecode runtime length mask
    // (which reads this as the prefix CAPACITY and masks past `decode_position`
    // — see `lower_subtile_tape_to_ktir`).
    //
    // The default is the model's own context length, capped at `PREFIX_CAP_DEFAULT`
    // so a plain `cargo build -Fspyre` produces a usable chat binary (the cuda/metal
    // targets need no such knob — their kernels take `decode_position` as a runtime
    // arg over a host-allocated cache; a KTIR bundle is a static-shape graph, so the
    // prefix cache is a baked dimension). The cap is deliberately modest: the
    // ktir-emulator decode is O(capacity) per token — it masks over the WHOLE prefix
    // cache every step (no paged attention), so a large cap makes the emulator crawl
    // (4096 ⇒ ~tens of seconds/token). 256 covers a normal short chat while keeping
    // per-step attention cheap. `KTIR_PREFIX_LEN` overrides up (longer contexts, real
    // silicon) or down (e.g. the m=1 self-check gate sets `KTIR_PREFIX_LEN=2`).
    // Resident prefix-cache capacity as `NonZeroU32` end-to-end: the 0-row case
    // (an invalid SubtileIR prefix `Source`) is excluded by the TYPE, not a
    // runtime `.filter`, and the value is wrapped in `to_wavefront::PrefixCapacity`
    // before every `lower_decode_to_wavefront` call so decode + prefill can never
    // be baked at diverging capacities.
    let prefix_cap_default: std::num::NonZeroU32 =
        std::num::NonZeroU32::new(256).expect("256 != 0");
    let model_ctx: std::num::NonZeroU32 = model
        .bounds
        .get("max_position_embeddings")
        .map(|&v| v.min(u32::MAX as u64) as u32)
        .and_then(std::num::NonZeroU32::new)
        .unwrap_or(prefix_cap_default);
    let prefix_len: std::num::NonZeroU32 = std::env::var("KTIR_PREFIX_LEN")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .and_then(std::num::NonZeroU32::new)
        .unwrap_or_else(|| model_ctx.min(prefix_cap_default));
    // Primary bundle row count: 1 = decode (default), >1 = the mq golden-gate
    // knob. The worker's batched-prefill bundle is emitted separately below
    // (default m = prefix_len; `KTIR_PREFILL_LEN` overrides).
    // Tape-scheduled targets skip instruction selection, so there is no solve
    // to cross-check the bridge against.
    #[cfg(any(feature = "metal", feature = "spyre"))]
    let tape_pilot_arch_wf = true;
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    let tape_pilot_arch_wf = false;
    // The bridge takes the assignment ONLY to cross-check that the solve
    // covered every tile (to_wavefront.rs, the `subgraph_of` guard); it reads
    // nothing else from it. A tape-scheduled build never solves, so there is
    // no coverage to check and passing `None` skips a check that would
    // otherwise fail against a deliberately-empty assignment.
    let decode_asn_check = if tape_pilot_arch_wf {
        None
    } else {
        Some(decode_asn)
    };
    let primary_m: u32 = std::env::var("KTIR_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(1);
    let lowered = match crate::to_wavefront::lower_decode_to_wavefront(
        fuf,
        decode_asn_check,
        inferred,
        decode_bounds,
        model,
        crate::to_wavefront::PrefixCapacity::new(prefix_len),
        primary_m,
    ) {
        // A quant weight scheme the wavefront/superdsc lowering does not emit (int4/ggml, or fp8 until
        // wired) genuinely has NO superdsc bundle — skip THIS preset (return no bundle). This
        // is NOT the silent-green bug the panic below guards against: DENSE models never hit
        // UnsupportedWeight, so a dense regression still panics; only quant presets are skipped.
        // The bridge now lowers MLX-affine gemms (for the metal static-tape
        // path); the superdsc emitter here still has no affine
        // realization — skip the preset exactly as the bridge's refusal used
        // to, rather than letting a packed weight flow into a dense-only
        // emitter.
        Ok(l)
            if l.input.ops.iter().any(|od| {
                matches!(
                    od.op,
                    scratchy_subtile::lower::LoweredOp::Gemm {
                        weight: scratchy_subtile::lower::GemmWeight::Affine { .. },
                        ..
                    }
                )
            }) =>
        {
            eprintln!(
                "[wavefront] {stem}: no superdsc bundle — affine-quantized \
                 weights have no realization on this path; preset skipped"
            );
            return;
        }
        Ok(l) => l,
        Err(to_wavefront::BridgeError::UnsupportedWeight { detail, .. }) => {
            eprintln!(
                "[wavefront] {stem}: no superdsc bundle — weight scheme unsupported here \
                 ({detail}); preset skipped"
            );
            return;
        }
        Err(e) => {
            // If you asked for superdsc and a DENSE model's decode can't be lowered, that is a
            // HARD ERROR, not a silent fall-back. A swallowed `eprintln` here is precisely how "no model
            // produces a superdsc bundle" shipped green for every model at once.
            panic!("[wavefront] {stem}: superdsc decode NOT LOWERED — {e}");
        }
    };
    let st = to_wavefront::stats(fuf, decode_asn, &lowered);
    eprintln!(
        "[wavefront] {stem}: {} fuf tiles, {} subgraphs → {} sources ({} weights, \
         {} prefix-kv), {} ops; ops {:?}",
        st.fuf_tiles,
        st.subgraphs,
        st.sources,
        st.weight_sources,
        st.prefix_sources,
        st.ops,
        st.op_histogram,
    );

    let base_to_loc = to_wavefront::build_base_to_loc(backbone_slots, lm_head_slots, bb_bucket_id);

    // Spyre/KTIR (-Fspyre): emit the REAL model's per-op KTIR from the solver-produced
    // graph — every dim, rope form, GQA ratio, and eps is scratchy's, not a hand-built
    // stand-in. Drop a per-model KTIR bundle.
    #[cfg(feature = "spyre")]
    {
        use std::num::NonZeroU32;
        let knb = NonZeroU32::new(8192).expect("8192 != 0");
        let stem_id = stem.replace(['-', '.'], "_");
        // Opt-in `--device sdsc` (real-silicon) dump: emit the SAME real-model
        // `krg` SubtileIR to a sendnn op-graph JSON the DeepTools `compile_graph`
        // toolchain consumes. Off by default; writes one `<base>.sengraph.json`
        // ── PAGED revival (opt-in): `SCRATCHY_SENDNN_MODE=paged` emits the vLLM-Spyre
        //    offline_decoder [prefill,decode] PAGED bundle (PagedAttnStore/Compute,
        //    resident KV, symbolic s0/s1) — prefill = SDPA+Store, decode = Store+Compute,
        //    distinct graphs sharing the cache by node name. UNSET (the build default)
        //    keeps the dd2-PROVEN static-SDPA host-grown-KV single graph so the decode
        //    sentinel never regresses. PROVEN viable on dd2 SENTIENT 2026-06-23.
        let sendnn_paged = std::env::var("SCRATCHY_SENDNN_MODE").as_deref() == Ok("paged");
        // ── SuperDSC PERF path: the `superdsc` CARGO FEATURE (`--features superdsc`,
        //    NOT an env var) walks the SAME `krg` SubtileIR straight to a TILE-LEVEL,
        //    WORK-DIVIDED SuperDSC bundle (`lower_graph_to_superdsc`) that OWNS the
        //    32-core split — the fix for the sengraph path leaving 31 cores idle
        //    (SENCORES 1→14.4ms vs 32→9.9ms, only 1.45×). Drops the bundle into the
        //    per-process scratch tree, where the bake queue dxp-compiles it in place. Only the
        //    runnable m=1 DECODE tape (seq_sym=None, !is_prefill) is walked; the
        //    symbolic prefill/CB tapes are skipped (emulator-only). An op the emitter
        //    can't lower yet is a HARD error here (build-time guard), never a silent
        //    skip. (A Cargo feature, not an env trigger.)
        // Every spyre build emits the bundle: the lowering has one target, and it is KTIR.
        let sendnn_superdsc = true;

        // Emit ONE KTIR bundle (decode or prefill) from a lowered graph into
        // `dir`. Factored so we drop BOTH a cheap m=1 decode bundle (fast
        // per-token generation) AND an m=M batched-prefill bundle — the worker
        // uses the right one per phase, so decode never pays the prefill tax.
        // Returns (ktir_manifest_json, ktir_nodes, whole_graph_sengraph_json,
        // group_graphs) where group_graphs[i] = (group_idx, input_tensor_id,
        // result_tensor_id, group_graph_json) — the per-layer-group split graphs.
        type GroupGraphs = Vec<(u32, u32, u32, String)>;
        // SuperDSC bundle fingerprint (set by the m=1 decode walk under
        // `SCRATCHY_SENDNN_MODE=superdsc`). When present, the SengraphBundle's
        // (single) group has its prefill+decode `graph_json` STAMPED to
        // `SUPERDSC_BUNDLE:{fp}` so the worker resolves the baked dxp bundle dir
        // (via `scratchy_builder_spyre::baked_superdsc_dir`) instead of compiling a
        // sengraph. A `RefCell` because the closure is `Fn` (called twice) and must
        // write the fp out without taking ownership.
        let superdsc_fp: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
        // THE TYPED LAUNCH WIRING (`scratchy_target_spyre::wiring::Wiring`),
        // emitted by the DECODE walk — the same facts `manifest.json` carries,
        // as spyre's own types instead of a string the worker re-parses. A
        // `RefCell` for the same reason `superdsc_fp` is one: the closure is
        // `Fn` and runs twice (decode, prefill), and only the decode walk's
        // wiring is kept (both agree on every field below — the prefill bundle
        // differs only in row count, which is not a term here).
        #[allow(clippy::type_complexity)]
        let wiring_tokens: std::cell::RefCell<(
            Option<proc_macro2::TokenStream>,
            Option<proc_macro2::TokenStream>,
        )> = std::cell::RefCell::new((None, None));
        // SEPARATE fp for the m=N batched-PREFILL reroll bundle ("prefill-all-but-last"). The
        // prefill `emit_bundle` (is_prefill=true; under superdsc its seq_sym is forced to None so
        // the tape is CONCRETE at m=prefill_m) bakes its OWN prefix/body/suffix — with the lm_head
        // SKIPPED (lower_one_node: is_lm_head && m>1) so the vocab-wide lm_head never time-tiles —
        // and stamps HERE, so it does NOT clobber the decode body_fp. Threaded into the
        // SengraphBundle's PREFILL graph slot (C4) so the worker resolves a DISTINCT prefill dxp
        // bundle (prefill:Some) and runs the prompt in one m=N forward, then decodes the last token.
        let superdsc_prefill_fp: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
        // PREFILL WIDTH LADDER `[(mq, body_fp)]` — every prefill reroll that bakes stamps its own
        // width here, so the ladder is whatever actually LOWERED, never an assumed list. The top rung
        // is `prefill_m` (also stamped into `superdsc_prefill_fp`, keeping the single-bundle path and
        // every downstream guard unchanged); interior rungs are extra bakes over the SAME resident
        // weights+KV. See `manifest::SengraphBundleGroup::prefill_rungs`.
        let superdsc_prefill_rungs: std::cell::RefCell<Vec<(u32, String)>> =
            std::cell::RefCell::new(Vec::new());
        // DECODE BATCH LADDER `[(seqs, body_fp)]` — one entry per width that actually LOWERED, never
        // an assumed list. The one-request rung is the bundle already stamped into `superdsc_fp`, so
        // every path that predates the ladder is unchanged; wider rungs only ever add entries.
        // (seqs, active_cap, body_fp) — the swept extent rides along so the worker can tighten its
        // mask-coverage guard from `PAGE_SLOTS * pages` to the real `active_cap * pages`.
        let superdsc_decode_rungs: std::cell::RefCell<Vec<(u32, u32, String)>> =
            std::cell::RefCell::new(Vec::new());
        // The ONE prefill bundle baked with a full resident-prefix sweep. Every ladder rung is baked
        // prefix-FREE (ActiveCap::NONE) because a start==0 chunk has nothing resident to attend, which
        // drops 308 of 710 ops/layer. A CONTINUATION chunk (start>0) does have resident KV, so it needs
        // this one. Baked at the widest rung, so a single bundle covers any start <= cap.
        let superdsc_prefill_prefix_fp: std::cell::RefCell<Option<String>> =
            std::cell::RefCell::new(None);
        // `prefill_mq`: Some(mq) marks the PREFILL path AND carries the concrete query-row count the
        // tape was lowered at — the ladder rung this bake belongs to. `None` = decode.
        let emit_bundle =
            |lwd: &to_wavefront::LoweredDecode,
             base: String,
             // The capacity THIS bundle is baked for. Decode cap buckets vary it;
             // every other call site passes the full `prefix_len`.
             cap: u32,
             seq_sym: Option<(&'static str, u32)>,
             prefill_mq: Option<u32>,
             // The resident-prefix sweep this bundle is baked for. Decode always FULL.
             // A PREFILL chunk with start==0 has no resident prefix, so it bakes NONE
             // (nb=0) and skips 308 of 710 ops/layer; continuation chunks need FULL.
             active_cap: scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap,
             // The row count THIS bundle is baked at. Not `primary_m`: a batch-ladder
             // rung is a wider bake of the same tape, and reading the env knob here
             // would tell every rung it was one row.
             decode_rows: u32|
             -> (String, Vec<(String, String)>, String, GroupGraphs) {
                let is_prefill = prefill_mq.is_some();
                let fused = scratchy_subtile::lower::fuse_silu_mul(&lwd.input);
                let krg = scratchy_subtile::subtile_ir::lower_region(&fused, knb);
                if let Err(e) = scratchy_subtile::subtile_ir::ValidatedGraph::new(&krg) {
                    eprintln!("[spyre-ktir] {stem}: SubtileIR rejected by validate() — {e:?}");
                    return (String::new(), Vec::new(), String::new(), Vec::new());
                }
                // `--target sendnn`: lower the SAME real-model `krg` to a sendnn
                // op-graph JSON (the DeepTools toolchain compiles + runs it on the
                // AIU). Baked into the binary below as `SENGRAPH_BUNDLE`; dumped to
                // disk too when `SCRATCHY_SENGRAPH_DUMP` is set.
                // Only symbolic bundles (seq_sym=Some: prefill prompt, CB decode batch)
                // are real `--target sendnn` decoder graphs. The plain m=1 KTIR decode
                // (seq_sym=None) is emulator-only — skip its sengraph so it never trips
                // the decoder-contract guards as build noise.
                // DEFAULT-MODE single graph: lower the m=1 decode tape to the static-SDPA
                // host-grown-KV sengraph and bake it as group 0 (the only group). The
                // offline_decoder layer-group split was removed (it never cleared the
                // dd2 per-job flit cap); the default-mode graph partitions itself via the
                // explicit host Reshape/Transpose breaks. `group_graphs[i]` =
                // (group, input_tensor, result_tensor, json).
                let group_graphs: Vec<(u32, u32, u32, String)> = Vec::new();
                // WEIGHT IDS for the fold pass: the sengraph source tensor id `t{i}` is a
                // model WEIGHT iff `lwd.bindings[i]` is `SourceBinding::Weight` (the
                // binding index == the sengraph source id — both from the same
                // `lwd.input`). Runtime activations (embed/cos/sin/prefix-kv) stay
                // PrimaryInput. `fold_weights_into_model` reclassifies the weights to
                // `ModelInput` so the DEM folds them into the resident model_tensor, off
                // the per-job I/O descriptor table (the flit-cap fix).
                let weight_ids: std::collections::HashSet<u32> = {
                    use crate::to_wavefront::SourceBinding;
                    lwd.bindings
                        .iter()
                        .enumerate()
                        .filter_map(|(i, b)| match b {
                            // The fp8 per-channel weight_scale is a STATIC resident tensor (loaded once,
                            // like a weight) — not a per-step activation — so it joins seg1 (SegRole::Weight).
                            SourceBinding::Weight { .. } | SourceBinding::WeightScale { .. } => {
                                Some(i as u32)
                            }
                            _ => None,
                        })
                        .collect()
                };
                // ── SuperDSC PERF emission (opt-in `SCRATCHY_SENDNN_MODE=superdsc`) ──
                // Walk the SAME `krg` straight to a work-divided SuperDSC bundle and
                // write it to the per-process scratch tree, dxp-compiled in place. Only
                // the runnable m=1 decode tape (seq_sym=None, !is_prefill) — the symbolic
                // prefill/CB tapes are emulator-only. An unlowered op is a HARD error
                // (the worklist signal: it names the exact next op to emit), never a
                // silent skip (which would mean silently-wrong on-card output).
                // superdsc bakes BOTH the m=1 decode reroll AND the m=N prefill reroll (C1 forces the
                // prefill's seq_sym to None so it is CONCRETE here). `seq_sym.is_none()` still excludes
                // the emulator-only symbolic CB-decode tape (Some(("s0",…))). is_prefill selects which fp
                // to stamp (superdsc_prefill_fp vs superdsc_fp) at the bake site below.
                // COMPILE-gated on `superdsc`: the sdsc emitter module only exists under that feature, so a
                // `spyre`-only (ktir) build must not compile this block at all (`sendnn_superdsc` alone is a
                // runtime bool and would still require the module to resolve).
                if sendnn_superdsc && seq_sym.is_none() {
                    use scratchy_target_spyre::lower_subtile_tape_to_superdsc as superdsc;
                    // ── GATE-1 de-risk: when `SCRATCHY_SUPERDSC_GATE1=1`, also drop a
                    //    SINGLE self-contained matmul bundle (no dangling tensor refs)
                    //    so a pod `sendnn` build's dxp-bake can prove dxp INGESTS
                    //    scratchy's emitted SuperDSC (encoding + build.rs arm + init.txt
                    //    end-to-end) BEFORE the full decode bundle is complete. Content-
                    //    keyed write → idempotent across emit_bundle calls.
                    if std::env::var("SCRATCHY_SUPERDSC_GATE1").as_deref() == Ok("1") {
                        // SMALL LX-fitting matmul (64×64×64, batch 1): per-core tile
                        // (kernel ~8KB) is well under the LX scratchpad, so it compiles
                        // WITHOUT the frontend coarse-tile (time-tiling) pass — isolating
                        // "is the typed emitter's format dxp-valid?" from "does it tile
                        // large tiles?" (the latter = the coarse_tile port, next).
                        use scratchy_subtile::sdsc_abstract::{
                            KernelTag, RowBlockedTag, StickLayout, Stk,
                        };
                        // Typed matmul operands (compile-time addressing): activation/output RowBlocked, weight Kernel.
                        let g1_a =
                            Stk::<RowBlockedTag>::new("g1_a", StickLayout::row_blocked(64, 64))
                                .unwrap();
                        let g1_w = Stk::<KernelTag>::kernel(64, 64, "g1_w");
                        let g1_o =
                            Stk::<RowBlockedTag>::new("g1_o", StickLayout::row_blocked(64, 64))
                                .unwrap();
                        let g1 = superdsc::assemble_matmul(
                            "gate1_mm", 64, 64, 64, 1, &g1_a, &g1_w, &g1_o, None,
                        );
                        // A hand-authored matmul, so there is no attention node for the four
                        // attention facts to be facts of — main's own `false`/absent at such a site.
                        match superdsc::emit_bundle(
                            &[g1],
                            None,
                            superdsc::FoldGrouping::Split,
                            None,
                        ) {
                            Ok(fp) => eprintln!(
                                "[spyre-superdsc] GATE-1 single-matmul bundle emitted (fp {fp}) — \
                                 dxp compiles it inline"
                            ),
                            Err(e) => eprintln!("[spyre-superdsc] GATE-1 bundle emit failed: {e}"),
                        }
                    }
                    // RE-ROLL the layer loop (the 40-min-compile fix): instead of lowering
                    // the 30×-UNROLLED `krg` to one 2047-op bundle (dxp super-linear → 40 min),
                    // re-roll the tape (mirror the tk_tape path, codegen ~8807) and lower the
                    // ONE layer BODY (~68 ops) → a SMALL bundle dxp compiles in SECONDS.
                    // STAGE 1: bake the BODY bundle (proves the fast compile). STAGE 2 adds the
                    // prefix(embed)/suffix(lm_head) bundles + the executor loop over `iters`
                    // layers (per_layer weight/KV/hidden ivar threading).
                    use scratchy_subtile::subtile_ir::ValidatedGraph;
                    use scratchy_subtile::subtile_tape::{lower_dag_to_tape, reroll_subtile_tape};
                    match ValidatedGraph::new(&krg) {
                        Ok(valid) => {
                            let tape_unrolled = lower_dag_to_tape(&valid);
                            let tape_rolled = reroll_subtile_tape(&tape_unrolled, &krg);
                            match superdsc::lower_subtile_tape_to_superdsc(
                                &tape_rolled,
                                &krg,
                                &weight_ids,
                                active_cap, // decode: FULL (sk_bucket rungs land below). prefill: NONE for a
                                // start==0 chunk, FULL for the continuation bundle.
                                // A prefill chunk's rows are one prompt's positions; the decode bundle's
                                // are one token each of the running requests once it is baked wider than
                                // a single row.
                                !is_prefill && decode_rows > 1,
                            ) {
                                Ok(rolled) => {
                                    eprintln!(
                                        "[spyre-superdsc] {base}: RE-ROLLED — prefix {} / body {} (loop ×{} layers) \
                                     / suffix {} ops ({} per-layer tids). Baking the BODY bundle (STAGE 1).",
                                        rolled.prefix.len(),
                                        rolled.body.len(),
                                        rolled.iters,
                                        rolled.suffix.len(),
                                        rolled.per_layer.len(),
                                    );
                                    // STAGE 2: bake ALL THREE small bundles (prefix=embed,
                                    // body=ONE layer, suffix=lm_head) + drop `reroll_meta.json`
                                    // in the body dir (3 fps + iters + per-layer seg strides) so
                                    // the worker runs prefix → body×iters (seg-base advance by
                                    // v·stride) → suffix. The sentinel stays `SUPERDSC_BUNDLE:{body_fp}`;
                                    // the worker detects rolled mode by the presence of reroll_meta.json.
                                    // ⭐⭐⭐ WHICH OPS THE ONE PATH GETS IS A PROPERTY OF THE DEVICE.
                                    //
                                    // ⛔ AND THE UNROLL'S "COSTS ALMOST NOTHING" ARGUMENT IS TRUE FOR
                                    // THE EMULATOR AND FALSE FOR THE CARD. A KTIR launch binds a
                                    // TENSOR, and KTIR programs are INTERNED, so `iters` copies of a
                                    // body really are `iters` arg-lists over one shared program —
                                    // free. Nothing is interned at DESCRIPTOR level: on `-Fspyre-hw`
                                    // every unrolled layer's op becomes its own `SdscOp` and its own
                                    // dxp group compile. MEASURED on granite-3.1-2b: the unrolled form
                                    // is 690 KTIR programs → 8744 descriptors / 161 groups, where this
                                    // same site's ROLLED sibling bake (`&rr.body`, the fused twin
                                    // below) is 137-164 descriptors / 3 groups. That 40x is the whole
                                    // bake-time story, and it ends in `dxp refused …: LLVM ERROR:
                                    // pthread_create failed`.
                                    //
                                    // ⭐ SO THE CARD BAKES THE ROLLED BODY and its executor loops it —
                                    // `superdsc_exec::load_rolled` advances the resident segment base
                                    // by `v·stride` per iteration, which is exactly what
                                    // `RerollMeta`'s strides are for and why they are real HERE even
                                    // though a KTIR launch has no segment base of its own: on this
                                    // path addresses come from `BundleLayout.places`, not from
                                    // `LaunchProgram::args` (which `ktir_groups_via_superdsc` leaves
                                    // empty for exactly that reason).
                                    //
                                    // ⛔ THIS IS NOT A SECOND PATH. Both devices still go
                                    // `SubtileIR → KTIR → SuperDSC → this bake`; only the op LIST
                                    // differs, the same way `ktir_optimizer::matmul_tile` differs by
                                    // device without forking the lowering. It is the same correction
                                    // the matmul pre-tiling needed: an emulator-motivated transform
                                    // applied upstream of both consumers, moved to the consumer that
                                    // wants it.
                                    let unrolled = superdsc::unroll_layers(&rolled);
                                    let card = cfg!(feature = "spyre-hw");
                                    let pre = if card {
                                        superdsc::emit_bundle(
                                            &rolled.prefix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                        )
                                    } else {
                                        Ok(String::new())
                                    };
                                    let bod = superdsc::emit_bundle(
                                        if card { &rolled.body } else { &unrolled },
                                        Some(&rolled.layout),
                                        superdsc::FoldGrouping::Split,
                                        rolled.attn_params,
                                    );
                                    // The SAME body, with the per-page fold fused back into the
                                    // surrounding work instead of standing alone. Splitting the fold
                                    // costs one extra launch per layer (measured: 5 groups vs the
                                    // baseline's 4 = 40 more launches per token on a launch-bound path),
                                    // and that is only worth paying when the fold is actually
                                    // re-launched. The runtime picks this one whenever the context fits
                                    // a single page, which is the case that has to stay at the
                                    // baseline's cost.
                                    // ⛔⛔⛔ AND THE FOLD PASS DOES HAVE A KTIR COUNTERPART — this
                                    // read "NO FUSED TWIN … a fold pass with no KTIR counterpart",
                                    // which is false. `assemble_attn` still stamps
                                    // `kv_page_fold = true` on its prefix-fold ops (one file, shared
                                    // by both paths), so `trip_kinds_for` still has `PageFold` trips
                                    // to reclassify. MEASURED, smollm2-135m under `-Fspyre-hw`: the
                                    // ladder rungs — whose twin was never removed — report `5
                                    // group(s)` split against `3 group(s)` fused for the same 75
                                    // trips. Two launches per layer, on the path the runtime prefers.
                                    //
                                    // ⭐ THE TWIN IS A LAUNCH-*GROUPING* VARIANT, SO IT IS THE CARD'S.
                                    // Off-card `ktir_groups` takes `_fold` and never reads it — one
                                    // launch group per op, always — so an emulator twin is a
                                    // byte-identical copy of its split under a different name, and its
                                    // only reader (`superdsc_exec::select_body_paged`) is
                                    // `spyre-hw`-gated. Same rule as the prefix/suffix below: bake it
                                    // where it is read.
                                    let bod_fused = if card {
                                        superdsc::emit_bundle(
                                            &rolled.body,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Fused,
                                            rolled.attn_params,
                                        )
                                        .map_err(|e| {
                                            // ⚠️ A LOST TWIN IS SILENT AT RUNTIME — the selector just
                                            // falls through to the split ladder — so say it here.
                                            eprintln!(
                                                "[spyre-superdsc] {base}: fold-fused body twin emit \
                                                 failed: {e}"
                                            );
                                        })
                                        .ok()
                                    } else {
                                        None
                                    };
                                    // ⭐ THE SUFFIX IS ITS OWN BUNDLE ON THE CARD. Rolling the body
                                    // means the lm-head tail can no longer ride inside it: the body is
                                    // ONE layer now, run `iters` times. The emulator keeps the single
                                    // unrolled bundle, where the suffix really is part of it.
                                    let suf = if card {
                                        superdsc::emit_bundle(
                                            &rolled.suffix,
                                            Some(&rolled.layout),
                                            superdsc::FoldGrouping::Split,
                                            rolled.attn_params,
                                        )
                                    } else {
                                        Ok(String::new())
                                    };
                                    match (pre, bod, suf) {
                                        (Ok(pfp), Ok(bfp), Ok(sfp)) => {
                                            // ── sk_bucket LADDER (decode only) ── The full-cap body just
                                            // baked (`bfp`) is the CEILING rung. Additionally bake the decode
                                            // BODY at each INTERIOR rung (active_cap < cap): same prefix/
                                            // suffix/layout, only the attention sweep extent shrinks, so all
                                            // rungs ride the SAME resident weights+KV. `decode_rungs` =
                                            // [(active_cap, body_fp)] ascending, top = (cap, bfp); the worker
                                            // loads all N bodies into ONE session + picks the smallest rung ≥
                                            // the live KV length each step. Only the m=1 DECODE reroll ladders
                                            // (prefill sweeps its whole chunk); a rung that fails to bake is
                                            // logged + skipped (the runtime falls back to the next rung up).
                                            let cap = prefix_len.get();
                                            // ⭐ EACH RUNG NAMES ITS OWN FUSED TWIN, rather than the
                                            // runtime reconstructing the name by appending "f" — a
                                            // convention whose miss is silently "this rung has no fused
                                            // twin", and one extra launch per layer.
                                            let sib = |fp: Option<String>| match fp {
                                                Some(fp) => superdsc::bundle::SiblingFp::from(fp),
                                                None => superdsc::bundle::SiblingFp::none(),
                                            };
                                            let mut decode_rungs: Vec<
                                                superdsc::bundle::LadderRung<'static>,
                                            > = vec![superdsc::bundle::LadderRung {
                                                active_cap: superdsc::bundle::SweptCols::new(cap),
                                                body: sib(Some(bfp.clone())),
                                                body_fused: sib(bod_fused.clone()),
                                            }];
                                            if !is_prefill {
                                                for rung in scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::decode_ladder(cap) {
                                                match superdsc::lower_subtile_tape_to_superdsc(
                                                    &tape_rolled,
                                                    &krg,
                                                    &weight_ids,
                                                    rung,
                                                    !is_prefill && decode_rows > 1,
                                                ) {
                                                    // Same device split as the ceiling rung above: the
                                                    // card bakes the ROLLED body, the emulator the
                                                    // unrolled one. Without this each of the three
                                                    // interior rungs pays the same 40x.
                                                    Ok(rr) => {
                                                        let rung_unrolled = if card {
                                                            Vec::new()
                                                        } else {
                                                            superdsc::unroll_layers(&rr)
                                                        };
                                                        match superdsc::emit_bundle(
                                                        if card { &rr.body } else { &rung_unrolled },
                                                        Some(&rr.layout),
                                                        superdsc::FoldGrouping::Split,
                                                        // THIS rung's own params — `rr` was lowered at
                                                        // `rung`, so its swept extent is `rung`'s and
                                                        // not the ceiling bundle's.
                                                        rr.attn_params,
                                                    ) {
                                                        Ok(rfp) => {
                                                            // Each rung also gets a fold-fused
                                                            // variant: the rung bounds the sweep,
                                                            // this bounds the launch count, and a
                                                            // short context needs BOTH.
                                                            // ⭐ ON THE CARD, for the ceiling twin's
                                                            // reason above — and here it also keeps
                                                            // the twin's NAME lawful. `bundle_fp`
                                                            // says a twin is `<split fp>f`, so it
                                                            // must hash the ops the split hashed;
                                                            // off-card the split is
                                                            // `&rung_unrolled` while this is
                                                            // `&rr.body`, so the "twin" was named
                                                            // after a bundle nothing else emitted.
                                                            let rfused = if card {
                                                                superdsc::emit_bundle(
                                                                    &rr.body,
                                                                    Some(&rr.layout),
                                                                    superdsc::FoldGrouping::Fused,
                                                                    rr.attn_params,
                                                                )
                                                                .ok()
                                                            } else {
                                                                None
                                                            };
                                                            eprintln!(
                                                                "[spyre-superdsc] {base}: ladder rung active_cap={} → body {rfp}",
                                                                rung.get()
                                                            );
                                                            decode_rungs.push(superdsc::bundle::LadderRung {
                                                                active_cap: superdsc::bundle::SweptCols::new(rung.get()),
                                                                body: sib(Some(rfp)),
                                                                body_fused: sib(rfused),
                                                            });
                                                        }
                                                        Err(e) => eprintln!(
                                                            "[spyre-superdsc] {base}: ladder rung {} body emit failed: {e}",
                                                            rung.get()
                                                        ),
                                                    }
                                                    },
                                                    Err(e) => eprintln!(
                                                        "[spyre-superdsc] {base}: ladder rung {} lower failed: {e}",
                                                        rung.get()
                                                    ),
                                                }
                                            }
                                            }
                                            decode_rungs.sort_by_key(|r| r.active_cap);
                                            // ⭐ THE LAYER LOOP, ATTACHED TO THE BODY IT DESCRIBES.
                                            // Every fingerprint in it is a sibling the registry
                                            // resolves at load, and `RerollMeta::siblings` walks the
                                            // fields themselves — so there is no second list of
                                            // "which fields hold a fingerprint" to fall out of step.
                                            superdsc::attach_reroll(
                                                &bfp,
                                                superdsc::bundle::RerollMeta {
                                                    prefix: sib(Some(pfp.clone())),
                                                    suffix: sib(Some(sfp.clone())),
                                                    body_fused: sib(bod_fused.clone()),
                                                    // ⛔ `rungs` IS KEYED BY ACTIVE_CAP — the columns one
                                                    // fold pass sweeps — NOT by the batch width. Two other
                                                    // lists of `(u32, fingerprint)` in this file are keyed
                                                    // by `seqs`, and reading the wrong one selects a body
                                                    // baked for a 64-column sweep because four requests are
                                                    // live.
                                                    rungs: std::borrow::Cow::Owned(decode_rungs),
                                                    // ⛔⛔⛔ THE ITERATION COUNT IS THE BODY'S, AND
                                                    // WHICH BODY THAT IS DEPENDS ON `card`.
                                                    //
                                                    // This read `iters: 1, weight_stride: 0,
                                                    // kv_stride: 0` unconditionally, under a comment
                                                    // asserting that was "the truth about this
                                                    // bundle, not a stub: the layer loop is unrolled
                                                    // into its launches". That IS true of the
                                                    // EMULATOR's bundle — `bod` above is
                                                    // `&unrolled` when `!card`, so the body really is
                                                    // the whole program and running it once runs
                                                    // every layer. It is false of the CARD's, where
                                                    // `bod` is `&rolled.body`: ONE layer, to be
                                                    // played `iters` times with the weight and KV
                                                    // segment bases advanced per layer.
                                                    //
                                                    // MEASURED, granite-3.1-2b: the runtime reported
                                                    // `RE-ROLLED — iters=1 wstride=0 kvstride=0`
                                                    // where main reports `iters=40
                                                    // wstride=121643008 kvstride=786432`. The card
                                                    // therefore ran ONE of forty layers per token —
                                                    // 3.1 ms ITL against main's 42 ms — and answered
                                                    // with a single repeated token. The `eprintln!`
                                                    // a few lines below has been printing
                                                    // `rolled.iters` (40) beside this 1 the whole
                                                    // time: the log said x40 while the baked
                                                    // metadata said 1.
                                                    iters: if card { rolled.iters } else { 1 },
                                                    weight_stride: if card {
                                                        rolled.weight_stride
                                                    } else {
                                                        0
                                                    },
                                                    kv_stride: if card {
                                                        rolled.kv_stride
                                                    } else {
                                                        0
                                                    },
                                                    // ⭐ THE WEIGHT BANKS RIDE WITH THE STRIDE THEY
                                                    // DIVIDE. The executor reaches layer `v` at
                                                    // `bank = v / layers_per_bank`, so the unrolled
                                                    // body's single iteration needs
                                                    // `layers_per_bank: 1` for that division to be
                                                    // the same no-op its `iters: 1` already is —
                                                    // conditioned on `card` for the same reason the
                                                    // strides above are, and not left at 0, which
                                                    // would divide by zero.
                                                    layers_per_bank: if card {
                                                        rolled.layers_per_bank
                                                    } else {
                                                        1
                                                    },
                                                    prefix_weight_bank: if card {
                                                        rolled.prefix_weight_bank
                                                    } else {
                                                        0
                                                    },
                                                    suffix_weight_bank: if card {
                                                        rolled.suffix_weight_bank
                                                    } else {
                                                        0
                                                    },
                                                },
                                            );
                                            // ── NUMERIC BISECTION oracle (opt-in SCRATCHY_SUPERDSC_DBG):
                                            //    eval_dag golden + synthetic source shapes so
                                            //    SPYRE_SUPERDSC_SELFTEST localizes the first divergent op.
                                            //    `dbg_golden_dir` is the ONE statement of where it lives,
                                            //    called by both the writer here and the reader there. ──
                                            if let Some(dbg) =
                                                superdsc::bundle::dbg_golden_dir(&bfp)
                                            {
                                                match superdsc::write_eval_golden(
                                                    &krg,
                                                    &weight_ids,
                                                    &dbg,
                                                ) {
                                                    Ok(()) => eprintln!(
                                                        "[spyre-superdsc] {base}: DBG golden written → {}/dbg",
                                                        dbg.display()
                                                    ),
                                                    Err(e) => eprintln!(
                                                        "[spyre-superdsc] {base}: DBG golden write failed: {e}"
                                                    ),
                                                }
                                            }
                                            eprintln!(
                                                "[spyre-superdsc] {base}: 3 bundles baked — prefix {pfp} / body {bfp} (×{} layers) \
                                             / suffix {sfp}; wstride={} kvstride={} (dxp: 3 SMALL bundles → seconds)",
                                                rolled.iters,
                                                rolled.weight_stride,
                                                rolled.kv_stride,
                                            );
                                            // Stamp the DECODE body_fp or the PREFILL body_fp by phase
                                            // (the prefill reroll must NOT clobber the decode fp).
                                            if prefill_mq.is_some() && active_cap == scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL {
                                            // The PREFIX-CAPABLE prefill bundle: the one continuation
                                            // chunks (start>0) must use, since they DO have resident KV
                                            // to attend. One bundle covers every start<=cap.
                                            *superdsc_prefill_prefix_fp.borrow_mut() = Some(bfp);
                                        } else if let Some(mq) = prefill_mq {
                                            // Ladder rung (deduped on mq: a rebake of the same width
                                            // must not double-enter). The TOP rung additionally lands in
                                            // `superdsc_prefill_fp` — the single-bundle prefill slot —
                                            // which the rung loop below arranges by baking it FIRST.
                                            {
                                                let mut r = superdsc_prefill_rungs.borrow_mut();
                                                if !r.iter().any(|(m, _)| *m == mq) {
                                                    r.push((mq, bfp.clone()));
                                                }
                                            }
                                            // Rungs bake ASCENDING, so the LAST write is the widest
                                            // -- the ceiling the worker treats as its single prefill
                                            // bundle. Unconditional: non-superdsc bakes exactly one.
                                            *superdsc_prefill_fp.borrow_mut() = Some(bfp);
                                        } else {
                                            {
                                                let mut r = superdsc_decode_rungs.borrow_mut();
                                                if !r.iter().any(|(n, _, _)| *n == decode_rows) {
                                                    // `cap` IS this body's swept extent: `bfp` is the
                                                    // CEILING rung (the interior sweep rungs go to
                                                    // `sk_bucket_rungs`), and `ActiveCap::FULL` resolves
                                                    // to `cap`.
                                                    r.push((decode_rows, cap, bfp.clone()));
                                                }
                                            }
                                            // The one-request bundle stays THE decode bundle for
                                            // every path that predates the ladder.
                                            //
                                            // ⛔ AND `decode_rows <= 1` DOES NOT MEAN THE GRAPH IS
                                            // ONE ROW. The CB emission (`ktir_decode_cb_*`) passes
                                            // `decode_rows = 1` on purpose — its rows are a batch
                                            // SYMBOL the DEM resolves at runtime — while the graph
                                            // it lowered is `CB_BATCH_TEMPLATE` = 96 rows wide. It
                                            // runs LAST, so it won this slot and every decode token
                                            // ran a 96-row program: MEASURED as `[96, 49155]` logits
                                            // and 195,428,640 elements read back per token, for a
                                            // ~90 K result. Correct (row 0 is the real token) and
                                            // ~96x the work. `seq_sym.is_none()` is the same
                                            // discriminator the superdsc block above uses to exclude
                                            // that symbolic tape.
                                            if decode_rows <= 1 && seq_sym.is_none() {
                                                *superdsc_fp.borrow_mut() = Some(bfp);
                                            }
                                        }
                                        }
                                        // HARD BUILD ERROR (guard-every-crash-at-build-time): a failed bundle
                                        // write here left `superdsc_fp`/`superdsc_prefill_fp` unset, so the
                                        // placeholder-group synthesis at the decode-bundle call site
                                        // (`default_decode_groups.is_empty() && superdsc_fp.borrow().is_some()`)
                                        // never fires — the SengraphBundle ships with ZERO groups, and the
                                        // FIRST signal is the worker's runtime "bundle has no layer groups
                                        // (split produced 0)" at pod load, with none of this context. Panicking
                                        // here instead surfaces the REAL SuperDSC lowering failure at `cargo
                                        // build` time, where it belongs.
                                        (pre, bod, suf) => panic!(
                                            "[spyre-superdsc] {base}: 3-bundle write failed — \
                                         prefix_err={:?} body_err={:?} suffix_err={:?}",
                                            pre.err(),
                                            bod.err(),
                                            suf.err(),
                                        ),
                                    }
                                }
                                Err(e) => panic!(
                                    "[spyre-superdsc] {base}: RE-ROLL NOT YET LOWERABLE — {e}. \
                                 (Next worklist item for the re-rolled SuperDSC emitter — this failure was \
                                 previously swallowed by an eprintln, leaving superdsc_fp unset and shipping \
                                 a ZERO-group bundle that only crashed at pod load.)"
                                ),
                            }
                        }
                        Err(e) => {
                            panic!(
                                "[spyre-superdsc] {base}: SubtileIR invalid for reroll — {e:?} (previously \
                             swallowed by an eprintln, leaving superdsc_fp unset and shipping a ZERO-group \
                             bundle that only crashed at pod load.)"
                            )
                        }
                    }
                }
                // Sengraph lowering DELETED (dead end — SDSC-native only). The sdsc bundle rides the
                // reroll bake (`superdsc_fp`) above; this decode graph_json stays empty and is
                // overridden by the SUPERDSC_BUNDLE sentinel downstream. `group_graphs` stays empty
                // (a placeholder group is synthesized by the caller to carry the sentinel).
                let sengraph_json = String::new();
                // ⭐ THE WIRING COMES OFF THE LOWERING THAT EMITS THE PROGRAMS. Its parameter
                // pairings are the ones the bundle's launches bind, so there is one lowering and
                // the wiring cannot describe a different program than the one that ships.
                let gk = match scratchy_target_spyre::lower_subtile_tape_to_superdsc::graph_wiring(
                    &krg,
                    &weight_ids,
                ) {
                    Ok(w) => w,
                    Err(e) => panic!("[spyre] {base}: wiring not lowerable — {}", e.0),
                };
                // The programs ride the bundle (`bundle_code::bundle(fp)`), not a per-node text.
                let nodes: Vec<(String, String)> = Vec::new();
                let attn_mask_json = match gk.attn_mask {
                    Some(id) => id.to_string(),
                    None => "null".to_string(),
                };
                let mut man = format!(
                    "{{\"result\":{},\"num_sources\":{},\"decode_position\":{},\"attn_mask\":{},\"tensors\":[",
                    gk.result_tensor,
                    gk.num_sources,
                    cap.saturating_sub(1),
                    attn_mask_json,
                );
                for (id, (r, c)) in gk.tensor_shapes.iter().enumerate() {
                    if id > 0 {
                        man.push(',');
                    }
                    let is_src = (id as u32) < gk.num_sources;
                    man.push_str(&format!(
                        "{{\"id\":{id},\"rows\":{r},\"cols\":{c},\"is_source\":{is_src}}}"
                    ));
                }
                man.push_str("],\"sources\":[");
                {
                    use crate::to_wavefront::SourceBinding;
                    use scratchy_subtile::lower::{InputRef, LoweredOp};
                    let mut gemm_weight: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    for od in &fused.ops {
                        if matches!(od.op, LoweredOp::Gemm { .. })
                            && let Some(InputRef::Ext(e)) = od.inputs.get(1)
                        {
                            gemm_weight.insert(*e);
                        }
                    }
                    for (i, b) in lwd.bindings.iter().enumerate() {
                        if i > 0 {
                            man.push(',');
                        }
                        let entry = match b {
                            SourceBinding::EmbeddedHidden => {
                                format!("{{\"id\":{i},\"role\":\"embed\"}}")
                            }
                            SourceBinding::Cos { .. } => format!("{{\"id\":{i},\"role\":\"cos\"}}"),
                            SourceBinding::Sin { .. } => format!("{{\"id\":{i},\"role\":\"sin\"}}"),
                            SourceBinding::PrefixK { layer } => {
                                format!("{{\"id\":{i},\"role\":\"prefix_k\",\"layer\":{layer}}}")
                            }
                            SourceBinding::PrefixV { layer } => {
                                format!("{{\"id\":{i},\"role\":\"prefix_v\",\"layer\":{layer}}}")
                            }
                            SourceBinding::Weight { id, index } => {
                                let name =
                                    crate::emit::weight_field_name(program, WeightId(*id), *index)
                                        .to_string();
                                let (wbase, _) = split_base_layer(&name);
                                let layer = index.as_ref().map(|u| u.0).unwrap_or(0);
                                let is_gemm = gemm_weight.contains(&i);
                                let disk = safetensors_prefix(
                                    program,
                                    model.arch.decoder_prefix.as_deref(),
                                    model.arch.safetensors.as_ref(),
                                    WeightId(*id),
                                    *index,
                                );
                                // lm_head weight binding — resolve the tie decision HERE, at
                                // bake (compile) time, from the SAME `tie_word_embeddings` flag
                                // the metal `FieldLoad::LinearTiedToEmbedding` path uses. TIED
                                // models (llama/smollm2) omit `lm_head.weight` and share the
                                // embed table; UNTIED models (Granite) ship a distinct head.
                                // Baking the on-disk name into the manifest means the spyre
                                // worker binds `{disk}.weight` VERBATIM and can NOT re-derive it:
                                // the old worker-side `disk == "lm_head" ⇒ embed` assumption
                                // silently mis-bound an untied head to the embedding matrix
                                // (garbage-but-consistent logits). Deciding it once, here, makes
                                // that mishandle unconstructible downstream.
                                let disk = if disk == "lm_head" && tie_word_embeddings(model) {
                                    "model.embed_tokens".to_string()
                                } else {
                                    disk
                                };
                                let loc_json = match base_to_loc.get(&wbase) {
                                    Some(li) => format!(
                                        "{{\"bucket\":{},\"op_idx\":{},\"slot\":{}}}",
                                        li.bucket, li.op_idx, li.slot
                                    ),
                                    None => "null".to_string(),
                                };
                                format!(
                                    "{{\"id\":{i},\"role\":\"weight\",\"name\":\"{name}\",\
                                 \"base\":\"{wbase}\",\"disk\":\"{disk}\",\"layer\":{layer},\
                                 \"is_gemm\":{is_gemm},\"loc\":{loc_json}}}"
                                )
                            }
                            SourceBinding::WeightScale { id, index } => {
                                // fp8 per-channel weight_scale: SAME disk prefix as the weight (worker loads
                                // `{disk}.weight_scale`), role "weight_scale", resident like a weight. Its own
                                // resident placement is registered under `{wbase}.weight_scale` by the layout.
                                let name =
                                    crate::emit::weight_field_name(program, WeightId(*id), *index)
                                        .to_string();
                                let (wbase, _) = split_base_layer(&name);
                                let scale_base = format!("{wbase}.weight_scale");
                                let layer = index.as_ref().map(|u| u.0).unwrap_or(0);
                                let disk = safetensors_prefix(
                                    program,
                                    model.arch.decoder_prefix.as_deref(),
                                    model.arch.safetensors.as_ref(),
                                    WeightId(*id),
                                    *index,
                                );
                                let loc_json = match base_to_loc.get(&scale_base) {
                                    Some(li) => format!(
                                        "{{\"bucket\":{},\"op_idx\":{},\"slot\":{}}}",
                                        li.bucket, li.op_idx, li.slot
                                    ),
                                    None => "null".to_string(),
                                };
                                format!(
                                    "{{\"id\":{i},\"role\":\"weight_scale\",\"name\":\"{name}\",\
                                 \"base\":\"{scale_base}\",\"disk\":\"{disk}\",\"layer\":{layer},\
                                 \"is_gemm\":false,\"loc\":{loc_json}}}"
                                )
                            }
                        };
                        man.push_str(&entry);
                    }
                }
                // ⛔ NO PER-NODE ARRAY. It described a `node{i}.mlir` file per program and a
                // `{name, tensor, is_output}` per argument; the programs are const data in the
                // bundle, and what a launch binds is the typed `(Ssa, PlaceId)` pairs it carries.
                man.push_str("],\"nodes\":[");
                man.push_str("]}");
                // ── THE SAME FACTS, TYPED ──────────────────────────────────
                //
                // Everything `man` just stringified is a compile-time constant
                // held HERE as a typed value: `gk.result_tensor`,
                // `gk.tensor_shapes`, `gk.attn_mask` and — the one that matters
                // — `lwd.bindings`, which is already the `SourceBinding` ENUM.
                // Flattening it to `"role":"prefix_k"` for the worker to
                // `match role.as_str()` back is a round-trip through strings
                // that loses exhaustiveness on the way out and cannot express a
                // weight whose on-disk key is not `{prefix}.weight`.
                //
                // So emit values of spyre's OWN types instead, exactly as
                // metal is handed statics of the metal runtime's types.
                {
                    let w = emit_superdsc_wiring(&gk, lwd, model, cap.saturating_sub(1), &man);
                    let mut slot = wiring_tokens.borrow_mut();
                    // Tensor ids are PER PROGRAM: the prefill graph numbers its
                    // own. Keeping only one would bind decode ids into prefill
                    // launches, so both are kept and the worker selects.
                    //
                    // ⛔ AND THE DECODE SLOT TAKES THE CONCRETE, SINGLE-REQUEST GRAPH ONLY. There is
                    // one slot but several non-prefill emissions, so a plain `else` is last-write-
                    // wins — and the last one is `ktir_decode_cb_*`, lowered at `CB_BATCH_TEMPLATE`
                    // (96). The worker forwards `m_cap` ROWS and reads `m_cap` off this wiring's
                    // EmbeddedHidden shape (`spyre_load.rs:109`, `spyre_forward.rs:1377`), so that
                    // stamp made every single-token decode step compute 96 activation rows.
                    //
                    // Both guards are the discriminators this file already uses. `seq_sym.is_none()`
                    // is how the superdsc block at the top of this closure excludes "the emulator-only
                    // symbolic CB-decode tape (Some(("s0",..)))" — whose rows are a batch SYMBOL the
                    // DEM resolves at runtime, not a count anything may forward. `decode_rows ==
                    // primary_m` keeps a wider batch-ladder rung out: a rung is the same tape baked
                    // at B rows and is addressed by FINGERPRINT, not through this wiring.
                    if is_prefill {
                        slot.1 = Some(w);
                    } else if seq_sym.is_none() && decode_rows == primary_m {
                        slot.0 = Some(w);
                    }
                }
                (man, nodes, sengraph_json, group_graphs)
            };

        // Decode bundle (m = KTIR_M, default 1): the worker's fast per-token path.
        // The dd2-PROVEN static-SDPA single graph is emitted from THIS m=1 decode
        // tape (`seq_sym=None`) and baked as the sendnn bundle — ONE graph for both
        // prefill-chunks and decode (host-grown KV). This is the only sendnn lowering
        // (the offline_decoder CB-decode / SDPA-prefill bundle path was removed); a
        // PLAIN `cargo build -Fsendnn` (no env) produces it. This is also the TOP
        // decode cap bucket (cap == prefix_len); the narrower buckets are baked below.
        let (decode_manifest, decode_nodes, _decode_graph_unused, mut default_decode_groups) =
            emit_bundle(
                &lowered,
                format!("ktir_decode_{stem_id}"),
                prefix_len.get(),
                None, // KTIR decode is m=1; its sengraph is replaced by the CB one below
                None, // decode path (Store + PagedAttnCompute) -- no prefill width
                scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                primary_m,
            );

        // ── DECODE CAP BUCKETS (KTIR emulator path) ───────────────────────────────────────────────
        // The emulator masks over the WHOLE prefix cache each step (decode is O(capacity)), so a
        // single big-cap program makes every short-context token pay the max-context cost. Bake the
        // decode program at a few static caps [256,512,1024,2048] below `prefix_len`; the worker runs
        // the smallest cap covering the current position, so a short chat pays cap-256 cost even when
        // the max context is large. Static shapes only — the emulator stays a dumb executor.
        //
        // NOT under superdsc: that path has its own, finer ActiveCap ladder baked inside `emit_bundle`
        // (`ActiveCap::decode_ladder`), and re-entering the bake here would restamp `superdsc_fp` with
        // a narrow-cap body — the last bucket would silently become the model's decode bundle.
        const DECODE_CAP_BUCKETS: [u32; 4] = [256, 512, 1024, 2048];
        let mut decode_bundles: Vec<(String, Vec<(String, String)>)> = Vec::new();
        if !sendnn_superdsc {
            let mut caps: Vec<u32> = DECODE_CAP_BUCKETS
                .iter()
                .copied()
                .filter(|&c| c < prefix_len.get())
                .collect();
            caps.sort_unstable();
            caps.dedup();
            for cap in caps {
                let lwd = match crate::to_wavefront::lower_decode_to_wavefront(
                    fuf,
                    decode_asn_check,
                    inferred,
                    decode_bounds,
                    model,
                    crate::to_wavefront::PrefixCapacity::new(
                        std::num::NonZeroU32::new(cap).expect("cap bucket is non-zero"),
                    ),
                    primary_m,
                ) {
                    Ok(l) => l,
                    Err(e) => panic!("[wavefront] {stem}: decode (cap={cap}) NOT LOWERED — {e}"),
                };
                let (m, n, _g, _gs) = emit_bundle(
                    &lwd,
                    format!("ktir_decode_{stem_id}_cap{cap}"),
                    cap,
                    None,
                    None,
                    scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                    primary_m,
                );
                decode_bundles.push((m, n));
            }
        }
        // The full-cap bundle is the last (widest) bucket, so the worker's "smallest cap covering the
        // position" scan over an ascending slice always terminates on it.
        decode_bundles.push((decode_manifest.clone(), decode_nodes.clone()));

        // ── DECODE BATCH LADDER (superdsc) ────────────────────────────────────────────────────────
        // A decode step runs one token of each RUNNING request, and how many that is changes step to
        // step — the scheduler decides it. A SuperDSC bundle is a static-shape graph, so the emitter
        // cannot take that number; it bakes a rung per width and the worker picks the smallest that
        // holds the live count, exactly as the prefill ladder already works.
        //
        // Each rung is the SAME tape at a wider row count — prefill's own emission, which is the
        // whole point: decode is that path at mq=B, not a second implementation. What differs is the
        // two things a batch needs and a prompt does not, and both are already gated on the row count
        // being requests: one page map per request, and one prefix-validity row per request.
        //
        // Rungs share ONE resident weights+KV — no term in either placement depends on the row count,
        // the same property that licenses the prefill ladder. A rung that fails to lower is logged
        // and SKIPPED: the batch runs on a narrower rung, in more than one forward, which is slower
        // and never wrong.
        // ⛔ CARD-ONLY, same reason as the prefill ladder below: the worker reaches a batch rung
        // through `superdsc_decode_rungs` / `BatchSlot`, which it consults under `spyre-hw` alone —
        // an emulator build sets `batched_prefill: None` (`spyre_load.rs:1713`) and never runs one.
        // Five more whole-model bakes for nothing.
        if sendnn_superdsc && cfg!(feature = "spyre-hw") {
            // RUNGS PAST 8. A decode step's cost is dominated by a FIXED penalty for leaving the
            // mq==1 fast path — the body goes 271 ops/layer at one request to 612 at two, then only
            // +16 per further request — plus one launch per request. Measured 36 ms fixed + ~2 ms per
            // request, so the fixed part is what the ladder amortizes and wider rungs are close to
            // free: 8 requests reach ~2.4x the weight stream's worth of work, 32 should reach ~4x.
            // A rung that fails to lower is logged and skipped, so adding them cannot break a build.
            //
            // THE LADDER COMES FROM THE POOL, not from a literal here. A page holds exactly
            // `PagedKvPool::ROWS` requests, and a rung wider than that would bake a launch whose
            // request axis walks past its kv head into the next one's keys — silently, and only for
            // the highest rows of the widest rung. `PagedKvPool` const-asserts that its widest rung
            // is `ROWS`, so the two cannot drift: changing either one alone is a `cargo build` error.
            for seqs in scratchy_subtile::sdsc_abstract::PagedKvPool::BATCH_RUNGS {
                match crate::to_wavefront::lower_decode_to_wavefront(
                    fuf,
                    decode_asn_check,
                    inferred,
                    decode_bounds,
                    model,
                    crate::to_wavefront::PrefixCapacity::new(prefix_len),
                    seqs,
                ) {
                    Ok(lwd_b) => {
                        emit_bundle(
                            &lwd_b,
                            format!("ktir_decode_{stem_id}"),
                            // `cap`: this bundle is baked for the full prefix.
                            prefix_len.get(),
                            // `seq_sym` / `prefill_mq`: decode, so neither applies.
                            None,
                            None,
                            scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                            seqs,
                        );
                    }
                    Err(e) => eprintln!(
                        "[spyre-superdsc] {stem}: decode batch rung seqs={seqs} NOT LOWERED — {e} \
                         (SKIPPED; that batch runs on a narrower rung, in more than one forward)"
                    ),
                }
            }
        }
        // SuperDSC replaces the sengraph ENTIRELY: `emit_bundle` above baked the work-divided dxp bundle
        // and set `superdsc_fp`, but the SENGRAPH decode lowering is REFUSED for granite (the lm_head
        // vocab-slice `t65` non-whole-region) so `default_decode_groups` is EMPTY. The SengraphBundle still
        // needs ONE [prefill,decode] group to carry the `SUPERDSC_BUNDLE:{fp}` sentinel (both slots are
        // overridden to it below), so synthesize a placeholder. `in_t`/`res_t` are UNUSED by the superdsc
        // worker path (it reads the decode manifest_json for wiring); the graph_json is the sentinel. This
        // is the group STRUCTURE only — the actual compute is the proven dxp bundle, not a sengraph.
        if default_decode_groups.is_empty() && superdsc_fp.borrow().is_some() {
            default_decode_groups.push((0u32, 0u32, 0u32, String::new()));
        }

        // Batched-prefill bundle: the worker runs the prompt through it (one m=N
        // forward per chunk), keeping the m=1 decode bundle for the generation
        // loop. Emitted BY DEFAULT so a plain spyre build prefills the prompt;
        // m=1 ⇒ skip (decode already covers single-token steps, e.g. the m=1
        // self-check gate where prefix_len defaults to its tiny value).
        //
        // The default width is bounded by the per-node LX wall: ktir-emulator runs
        // each node at its native grid, so one m-row FFN GEMM tile
        // (m × intermediate) must fit a Spyre core's 2 MiB LX. On llama-1b
        // (8192-wide FFN) m=64 overflows (2 MiB + 1 MiB > 2 MiB), m≤32 fits.
        //
        // CRITICAL: the prefill tape width is symbolized to `s0` BY VALUE
        // (`symbolicize_seq` flips every concrete dim == prefill_m → s0), so it MUST
        // NOT equal any structural model dim, else a non-seq dim is wrongly
        // symbolized (e.g. m=32 collides with head_dim/2=32 → the RoPE rotate-half
        // StridedSlice output `[rows, heads, 32]` becomes `[s0, heads, s0]`, which
        // flex rejects at graphOptimizer.cpp:2865). 31 is prime + ≤32 (LX-safe) +
        // collides with no common dim (head_dim 64/128, half 32/64, n_heads, hidden,
        // intermediate, block_size 64). `symbolicize_seq` guards the collision and
        // names the fix; `KTIR_PREFILL_LEN` overrides (must also be collision-free).
        const PREFILL_LX_SAFE: u32 = 31;
        /// SuperDSC PREFILL WIDTH LADDER, ASCENDING. Each rung is baked as its own prefill bundle and
        /// the worker runs a chunk on the NARROWEST rung that holds it: per-chunk cost is
        /// `fixed + slope·mq` with the fixed term dominant (granite-3.1-2b, measured: m=16 → 53.1 ms,
        /// m=31 → 69 ms ⇒ ~42 ms + ~0.7 ms/row), so one width is wrong both ways — too narrow re-pays
        /// the fixed term on every extra chunk, too wide computes padding rows that carry no tokens.
        ///
        /// `2^k − 1` by construction: `symbolicize_seq` rewrites every concrete dim equal to the tape
        /// width into `s0`, so a width equal to a structural dim would symbolize a NON-seq dim (see
        /// [`PREFILL_LX_SAFE`]'s note on m=32 == head_dim/2). These values collide with none of the
        /// power-of-two dims (head_dim, block_size, cap, hidden, intermediate).
        ///
        /// The 63-row ceiling is GONE (2026-07-29): the new-token block is now folded as `mq_pad/64`
        /// stick-wide sub-blocks, so a chunk may span more than one stick. 96 is HARDWARE-PROVEN.
        /// That matters more than padding efficiency, because a chunk costs ~41 ms REGARDLESS of how
        /// few rows it carries (measured: rung 15 = 55 ms, rung 31 = 70 ms, rung 63 = 88 ms, i.e.
        /// 3.7 / 2.3 / 1.4 ms per row) — so a prompt one token past the ceiling used to pay a whole
        /// extra chunk. A 65-row prompt split 63+2 and the 2-token tail alone cost 56.8 ms.
        /// 80 sits between 63 and 96 to halve the worst-case padding in that gap.
        ///
        /// `2^k − 1` is no longer required now that the widths exceed one stick, but every value here
        /// still avoids the structural dims (4, 8, 32, 64, 128, 256, 2048, 8192) so `symbolicize_seq`
        /// cannot mistake the tape width for a model dim.
        ///
        /// Rungs that fail to lower for a given model are logged and SKIPPED, so this list is an upper
        /// bound on the ladder, not a promise — a model whose FFN blows the LX budget at a wide rung
        /// simply gets a shorter ladder.
        /// DENSER AT THE LOW END on purpose: padding cost is `(rung − real) × slope`, so what hurts is
        /// the PROPORTION of the chunk that is padding, and that is worst for short prompts. A measured
        /// 19-token prompt landed on rung 31 and spent 12 of its 31 rows (39%) computing nothing — about
        /// 11 ms.
        ///
        /// GAPS HALVED (2026-07-30) now that the slope is pinned rather than estimated. Fitting the
        /// three measured prefill/decode `compute` points gives
        ///     compute ≈ 16.3 ms + 0.65 µs·trips + **1.24 ms·row**
        /// so a padding row is not a rounding error — it costs the same 1.24 ms a real token does,
        /// because the bundle is baked at `mq` and computes every row whether or not it holds a token.
        /// A 19-token prompt on rung 23 threw away 4 × 1.24 ≈ 5 ms, comparable to the entire RoPE
        /// head-merge win. Gaps are 2 through the short-prompt band, 4 to 47, then 8 — so the
        /// worst-case overshoot where interactive chat actually lands is ONE row, ~1.2 ms. The extra
        /// rungs cost bake time, startup and program memory, but NOT another weight copy (rungs borrow
        /// from the top rung — `3623c76b`) and NOT decode throughput (the placement cost is non-linear
        /// in session count and was already paid at the 1→2 step).
        ///
        /// The LOW end was the worst served and matters most, because what hurts is the PROPORTION of
        /// the chunk that is padding. The ladder used to start at 15, so a 3-token "hi" ran 15 rows
        /// and discarded 12 of them (~15 ms) — a third of a prefill on a prompt that needs almost
        /// none of it. 7 and 11 cover that. Measured mid-range case: a 20-token prompt sat on rung 23
        /// (`n_new=23` in the phase log) and wasted 3 rows; 21 takes that to one.
        ///
        /// 🛑 96 IS A HARD CEILING UNTIL K-TIME PSUM LANDS — it is not a tuning choice. Chunk count
        /// dominates TTFT (measured 2026-07-30: splitting a ~23-token prompt 15+8 instead of one chunk
        /// cost **+48 ms**, since a forward's ~40-45 ms fixed term is paid PER CHUNK), so taller rungs
        /// are exactly the right lever — but adding 127 fails the BUILD, not the ladder:
        ///   time_tile_for_lx('out'): per-core tile does not fit the 1,677,721-B usable LX scratchpad
        ///   even tiled to single 64-elem sticks (resident 1,923,712 B) — needs K-time PSUM
        ///   accumulation (Stage 2), which the frontend does not emit.
        /// That is down_proj (K=8192) at mq_pad=128. Note this contradicts "logged and SKIPPED" above:
        /// the skip covers a `lower_decode_to_wavefront` Err, but an LX overflow raises inside
        /// `emit_bundle` and PANICS the proc-macro. So do not extend this list expecting a graceful
        /// fallback — prove the rung lowers first, or make interior-rung bake failures skip while the
        /// ceiling keeps panicking (the panic is deliberate: a swallowed eprintln once shipped a
        /// zero-group bundle that only crashed at pod load).
        const PREFILL_RUNGS: [u32; 21] = [
            7, 11, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 39, 43, 47, 55, 63, 71, 80, 88, 96,
        ];
        // ⛔ THE POOL SIZES ITS SLACK FROM THIS CEILING, so the two cannot be allowed to drift.
        //
        // A continuation chunk's cache write covers the PREFIX BUNDLE'S BAKED ROW COUNT — this ceiling —
        // not the chunk's real length, and `PagedKvPool::WRITE_SLACK` is exactly the amount by which that
        // write may overrun a page. Raise the ceiling without raising the slack and the overrun lands on
        // the next kv head's keys again: a correct first token, then fluent garbage, for every prompt
        // needing a third chunk. A `const` assertion makes that a `cargo build` error instead.
        const _: () = assert!(
            PREFILL_RUNGS[PREFILL_RUNGS.len() - 1] as usize
                == scratchy_subtile::sdsc_abstract::PagedKvPool::PREFILL_CHUNK_SLOTS,
            "PagedKvPool::PREFILL_CHUNK_SLOTS must equal the PREFILL_RUNGS ceiling — the pool's \
             WRITE_SLACK is derived from it, and a padded chunk write that overruns by more than the \
             slack corrupts the next kv head's keys"
        );
        // ⛔ BUILD GUARD (guard-every-crash, on-card PROVEN 2026-06-24/25 dd2): mq is
        // NOT free-valued — the DEM/DSM partitioner only tiles the static-paged
        // PagedAttnCompute batch axis (`mq`) cleanly for a SPECIFIC empirically-
        // verified set. Out-of-set mq crashes on-card during `session_create`
        // (CompileGraph), BEFORE any Predict, with one of two signatures depending
        // on the value (sweep at nblk=4, SmolLM2-135M):
        //   mq ∈ {2,4,8}     → COMPILES + runs + coherent (decode 5.5 / 6.4 / 9.1 ms)
        //   mq == 96         → COMPILES + runs + coherent (decode 58.7 ms) [baseline]
        //   mq ∈ {16,32,256} → ABORT `vector::_M_range_check (0 >= 0)` (DEM tiler:
        //                      empty partition vector indexed at 0)
        //   mq == 64         → ABORT `DtException: sumVpPair == tensorProp.shape[d]`
        //                      (dsm/sharedFuncs.cpp:2804 — partition-sum != axis len)
        //   mq == 1          → already build-refused upstream (unit-axis Reshape vs
        //                      symbolic-batch collision, graphOptimizer.cpp:5422)
        // The plain `mq <= 96` ceiling did NOT catch the mid-range {16,32,64} crashes
        // (all <= 96). This allowlist turns every out-of-set mq into a `cargo build`
        // error instead of an on-card `session_create threw` at model load. To add a
        // new mq, PROVE it compiles+runs on dd2 first, then extend MQ_DEM_SAFE.
        // Real LONGER CONTEXT comes from a LARGE paged CACHE (prefix_len/nblk —
        // proven to compile at 512/2048) + CHUNKED prefill + 1-token incremental
        // decode, NOT a larger mq. A SMALLER decode mq (≤8) is the proven roofline
        // lever: decode COMPUTE scales ~linearly in mq (≈4.1 ms floor + 0.57 ms·mq
        // at nblk=4), so mq=2 is ~10.7× faster per decode step than mq=96.
        //
        // SMALL-MQ DECODE LEVER (PROVEN ~10.7× per-token, on-card 2026-06-25): the
        // static-paged DECODE graph is emitted at the SMALL `DECODE_MQ` (incremental
        // decode has 1 new token, so the old mq=96 over-provisions ~48×; decode COMPUTE
        // scales ~linearly in mq — 286 ms → ~10 ms/token at mq=2). NUMERICS ARE
        // IDENTICAL — pure right-sizing: the real token is row 0, the rest is pad
        // (last-real-row replicated, then discarded; the shim returns row n_new-1).
        //
        // ONE session drives BOTH prefill chunks and incremental decode through this
        // ONE graph — the resident KV lives in the session's on-card allocation and
        // persists across Predicts, and it CANNOT be shared with a second graph/session
        // (on-card PROVEN 2026-06-25: a separate prefill graph at mq=96 + a host-buffer
        // KV hand-off to a mq=2 decode session produced GARBLED output — the prefill's
        // on-card cache writes are NOT mirrored to the host buffer, so the copy is
        // stale; both mq=96 and mq=2 garbled with the two-session split, while the
        // single-session incremental path is coherent). So prefill chunks ride the SAME
        // small mq (a slower one-time TTFT cost; decode is the dominant repeated cost).
        // `CB_BATCH_TEMPLATE` is kept ONLY for the DEFAULT (non-paged) mode's CB-decode
        // template and as the MQ_DEM_SAFE upper anchor.
        const CB_BATCH_TEMPLATE: u32 = 96; // default-mode CB template (not block_size 64)
        // PAGED prefill+decode mq = the CONTINUOUS-BATCHING batch axis (#45): mq is the
        // number of decode ROWS one Predict processes, so it is the max number of
        // concurrent sequences whose next-token is computed per Predict (the CB batch
        // axis). DECODE_MQ=2 is the chat-optimal single-stream width: on-card @ nblk=32,
        // mq=2 → ~9.7 ms/Predict ≈ 104 tok/s with byte-identical output (committed
        // 206ece8d; ∈ MQ_DEM_SAFE). The worker packs B≤mq requests into the mq rows (1
        // real token each) via predict_perrow with per-row block_table; B>mq sub-batches
        // (ceil(B/mq) Predicts). NUMERICS identical: each row is its own seq's real token
        // over its own scheduler blocks; padding rows replicate the last real row and are
        // discarded. The per-row CB mechanism is on-card-VERIFIED at mq=8 (C=4 → ~99 tok/s,
        // 4 concurrent prompts each correct incl Jupiter, ZERO cross-contamination).
        //
        // mq is FIXED per compiled graph, so a single static mq trades C=1 latency vs
        // batch throughput: per-Predict cost grows with BOTH mq and nblk (mq=8 @ nblk=32
        // → ~33 ms ⇒ C=1 only ~30 tok/s; mq=96 @ nblk=32 → ~58.7 ms but 96 rows ⇒ ~1635
        // tok/s aggregate at full batch ≈ 36% of the ~4500 tok/s weight-bandwidth
        // roofline). The PATH to that throughput WITHOUT regressing chat is DYNAMIC mq —
        // bake decode graphs at several mq ∈ MQ_DEM_SAFE and have the worker pick the
        // smallest that holds the live batch (C=1 → mq=2, large batch → mq=96). Until that
        // lands, change `DECODE_MQ` below for an 8-way concurrent server.
        const DECODE_MQ: u32 = 2; // PAGED prefill+decode mq = CB batch axis default (chat-optimal; see note)
        // DEM/DSM-safe mq values, on-card verified (dd2 SENTIENT, nblk=4). MUST stay
        // sorted; every entry has been observed to CompileGraph + Predict coherently.
        const MQ_DEM_SAFE: [u32; 4] = [2, 4, 8, 96];
        const fn mq_is_dem_safe(mq: u32) -> bool {
            // const-fn linear scan (arrays aren't const-iterable on stable).
            let mut i = 0;
            while i < MQ_DEM_SAFE.len() {
                if MQ_DEM_SAFE[i] == mq {
                    return true;
                }
                i += 1;
            }
            false
        }
        const _: () = assert!(
            mq_is_dem_safe(CB_BATCH_TEMPLATE),
            "CB_BATCH_TEMPLATE (PREFILL mq) is not a DEM/DSM-safe static-paged mq. Only {{2,4,8,96}} \
             are on-card PROVEN (dd2) to CompileGraph + Predict; other values abort \
             `session_create` BEFORE any Predict with vector::_M_range_check (mq in {{16,32,256}}) \
             or DtException sumVpPair==shape (mq==64). Pick a value from MQ_DEM_SAFE; to add \
             one, prove it compiles+runs on dd2 first."
        );
        const _: () = assert!(
            mq_is_dem_safe(DECODE_MQ),
            "DECODE_MQ is not a DEM/DSM-safe static-paged mq. Only {{2,4,8,96}} are on-card PROVEN \
             (dd2). DECODE_MQ=2 is the minimal incremental-decode width (1 real token + 1 pad) and \
             the proven decode roofline lever (~10.7× over mq=96). To use another value, pick from \
             MQ_DEM_SAFE; a SMALLER decode mq is faster (decode COMPUTE scales ~linearly in mq)."
        );
        // PAGED: the prefill graph is emitted but UNUSED at runtime (one session uses
        // the decode graph for both phases — see DECODE_MQ note). It is still lowered
        // (to satisfy the bundle's [prefill,decode] pairing + KV-shape-agreement guards)
        // at the small DECODE_MQ so its kcache_/vcache_ shapes match decode exactly.
        // DEFAULT (host-grown) prefill uses the LX-safe small width. `KTIR_PREFILL_LEN`
        // overrides.
        let prefill_m: u32 = std::env::var("KTIR_PREFILL_LEN")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(if sendnn_paged {
                // The prefill graph is emitted-but-UNUSED in the single-graph path (the mq=1
                // decode graph drives BOTH phases). It must still be emitted at mq>1, else the
                // `.filter(|&v| v > 1)` below drops it → the bundle ships no prefill graph → the
                // paged-bundle guard panics (prefill_present=false). kcache_/vcache_ shapes are
                // mq-INDEPENDENT ([nblk,64,nkv,hd]), so this mq-2 placeholder prefill graph's KV
                // shapes still match the mq=1 decode exactly (the KV-shape-agreement guard passes).
                DECODE_MQ.max(2)
            } else if sendnn_superdsc && cfg!(feature = "spyre-hw") {
                // SuperDSC bakes the whole WIDTH LADDER and lets the WIDEST rung that actually
                // lowers become the ceiling (see the rung loop below), so this is just where the
                // ladder STARTS looking — not a value that has to be right. That is what demotes
                // `KTIR_PREFILL_LEN` from a required flag to a diagnostic override, and what stops
                // a model whose FFN blows the LX budget at the top width from failing the BUILD:
                // it simply gets a shorter ladder. Still capped by the resident cache.
                PREFILL_RUNGS[PREFILL_RUNGS.len() - 1].min(prefix_len.get())
            } else {
                prefix_len.get().min(PREFILL_LX_SAFE)
            });
        // ── PREFILL WIDTH LADDER: bake EVERY rung up to the ceiling, ascending ──
        // `prefill` ends up as the WIDEST rung that actually lowered, so a rung the model cannot fit
        // (the LX budget scales with mq x intermediate) shortens the ladder instead of failing the
        // build. Ascending order also means the widest bake is LAST, so it is the one that lands in
        // `superdsc_prefill_fp` (the single-bundle prefill slot) -- the worker treats that as the
        // ceiling and every narrower rung as a step below it.
        // Non-superdsc keeps exactly one width and still PANICS if it will not lower: those paths
        // have no ladder to fall back to, so a missing prefill graph must stay a build failure.
        // ⛔ THE LADDER IS A CARD FEATURE — BAKE ONE WIDTH OTHERWISE. Every rung is a full lowering
        // of the whole model, and the emulator worker can only ever run ONE of them:
        // `KtirBundle::prefill` is a single `Option` (`manifest.rs:167`) and the narrower rungs are
        // reached through `prefill_rungs`, which `spyre_forward.rs` reads under `spyre-hw` alone.
        // With `PREFILL_RUNGS` running to 96 that was 21 bakes to use 1 — the generated model source
        // and rustc's single-threaded pass over it both paid for all 21.
        let prefill_rung_widths: Vec<u32> = if sendnn_superdsc && cfg!(feature = "spyre-hw") {
            let mut v: Vec<u32> = PREFILL_RUNGS
                .into_iter()
                .filter(|&r| r > 1 && r <= prefill_m)
                .collect();
            // An explicit KTIR_PREFILL_LEN that is not itself a rung still gets baked, as the top.
            if v.last() != Some(&prefill_m) && prefill_m > 1 {
                v.push(prefill_m);
            }
            v
        } else {
            Some(prefill_m).filter(|&v| v > 1).into_iter().collect()
        };
        type PrefillPick = (String, Vec<(String, String)>, String, GroupGraphs);
        let mut prefill: Option<PrefillPick> = None;
        for pl in prefill_rung_widths {
            let lowered = to_wavefront::lower_decode_to_wavefront(
                fuf,
                decode_asn_check,
                inferred,
                decode_bounds,
                model,
                // Prefill and decode model the SAME resident prefix-KV cache, so
                // they MUST share ONE capacity — the KTIR emulator bakes a STATIC
                // `[prefix_len, kv]` cache + a `[1, prefix_len]` runtime length mask,
                // and the worker reads that mask width as the bundle's max context
                // (`b.capacity`). A prefill baked at a smaller capacity than the
                // decode makes the worker reject any prompt longer than that capacity
                // (`positions 0..N exceed bundle prefix capacity`). The prior default
                // used `1` — a host-grown-sengraph optimization (valid_len-1=0 drops
                // the prefix seg on silicon, where the host passes exactly `valid`
                // rows at runtime so the template size is immaterial). But the KTIR
                // cache is NOT host-grown; at prefix_len=1 the prompt can never
                // exceed 1 token. Use the resident capacity for BOTH paths (paged and
                // default), matching the decode bundle: at start=0 the runtime mask
                // keeps 0 prefix rows so the prompt still attends only itself
                // causally, and chunk k attends the k*mq rows written so far.
                crate::to_wavefront::PrefixCapacity::new(prefix_len),
                pl,
            );
            match lowered {
                Ok(lwd_p) => {
                    prefill = Some(emit_bundle(
                        &lwd_p,
                        format!("ktir_prefill_{stem_id}"),
                        prefix_len.get(),
                        // prefill: rewrite the concrete prompt length `pl` → symbol
                        // s0 (offline_decoder idx-0 graph: prompt sym = Tq = Tkv).
                        // ── superdsc: keep the prefill tape CONCRETE (m=pl) so the reroll
                        //    bakes a real m=N dxp bundle (the dxp reroll is NOT symbolic; the
                        //    worker runs it at the fixed pl). Only the sengraph offline_decoder
                        //    path uses the s0 symbol. ──
                        if sendnn_superdsc {
                            None
                        } else {
                            Some(("s0", pl))
                        },
                        Some(pl), // prefill path (SDPA + Store), ladder rung m=pl
                        // ⛔ ZERO PREFIX BLOCKS, AND THAT IS ONLY RIGHT FOR THE FIRST CHUNK.
                        // `ActiveCap::NONE` is documented as valid ONLY where `start == 0`, but
                        // `manifest.rs`'s `KtirBundle::prefill` is a single `Option` and
                        // `spyre_forward.rs:1544` hands that one bundle to EVERY chunk, so chunk k
                        // attends only its own `mq` tokens and ignores the `k*mq` before it. The
                        // prefix-capable emission below (`:10176`) is reachable only through
                        // `prefill_prefix`, which the worker reads under `spyre-hw` alone.
                        //
                        // ⚠️ FLIPPING THIS TO `FULL` DOES NOT FIX IT — MEASURED. `ActiveCap` is the
                        // sweep extent of a DECODE attention, so `FULL` here lowers the chunk's
                        // attention in the decode form: its per-layer `new_k` comes back one row
                        // (`kv_dim`) while `forward_chunk` reads `n * kv_dim`
                        // (`spyre_forward.rs:1460`), and any prompt long enough to chunk PANICS
                        // (`range end index 49152 out of range for slice of length 512`, 49152 =
                        // 96*512). Multi-chunk prefill needs prefix blocks AND `mq` new rows
                        // together; that combination is not what this flag selects.
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::NONE,
                        1, // a prompt chunk is one request however many positions it spans
                    ));
                }
                Err(e) if sendnn_superdsc => eprintln!(
                    "[spyre-superdsc] {stem}: prefill ladder rung m={pl} NOT LOWERED — {e} \
                     (SKIPPED; a narrower rung still covers this width)"
                ),
                Err(e) => panic!("[wavefront] {stem}: prefill (m={pl}) NOT LOWERED — {e}"),
            }
        }

        // ── PREFIX-CAPABLE PREFILL BUNDLE ── one more bake of the WIDEST rung, this time with a full
        // resident-prefix sweep. The ladder rungs above are all prefix-FREE, which is correct only for
        // a chunk with start==0; a continuation chunk has resident KV to attend and must use this.
        // Baking it at the widest width means ONE bundle serves every start <= cap: a continuation
        // chunk pays some padding, but it is the rarer case and correctness is not negotiable.
        // ⛔ AND IT IS CARD-ONLY TOO: it is reached through `prefill_prefix`, which
        // `spyre_forward.rs:1641` reads under `spyre-hw` alone. An emulator build baked this whole
        // extra model and could never launch it.
        if sendnn_superdsc
            && cfg!(feature = "spyre-hw")
            && let Some(top) = superdsc_prefill_rungs
                .borrow()
                .iter()
                .map(|(m, _)| *m)
                .max()
        {
            match crate::to_wavefront::lower_decode_to_wavefront(
                fuf,
                decode_asn_check,
                inferred,
                decode_bounds,
                model,
                crate::to_wavefront::PrefixCapacity::new(prefix_len),
                top,
            ) {
                Ok(lwd_p) => {
                    emit_bundle(
                        &lwd_p,
                        format!("ktir_prefill_{stem_id}"),
                        prefix_len.get(),
                        None,
                        Some(top),
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                        1, // a prompt chunk is one request
                    );
                }
                Err(e) => eprintln!(
                    "[spyre-superdsc] {stem}: prefix-capable prefill bundle (m={top}) NOT LOWERED \
                     — {e}. Continuation chunks will have no bundle to run on."
                ),
            }
        }

        // ── CB decode sengraph (the baked `--target sendnn` decode graph). The
        //    offline_decoder decode isengraph must carry a BATCH symbol on its K/Q
        //    data — the DEM reads batchSym from the PagedAttnStore KEY input
        //    (dem_frontend decoderSymCheck). The KTIR decode is m=1, so build a
        //    SEPARATE m=DECODE_MQ tape whose ROWS are the batch dim, then
        //    symbolicize that value → s0 (batch); the tkv symbol (s1) is emitted
        //    explicitly on block_table dim0. batch=1 at runtime via symVals[s0]=1. ──
        // The mq allowlist guard (MQ_DEM_SAFE) + CB_BATCH_TEMPLATE (prefill mq) +
        // DECODE_MQ (decode mq) are defined ABOVE (before the prefill block, so the
        // paged prefill can use CB_BATCH_TEMPLATE). The DECODE graph is emitted at the
        // SMALL DECODE_MQ (the proven ~10.7× decode lever); prefill stays at 96.
        // ⛔ BUILD GUARD (guard-every-crash, option (b) resident KV): the paged path
        // serves LONGER CONTEXT only if the resident cache (cap = prefix_len, in
        // 64-token pages) holds strictly MORE than one forward's worth of tokens
        // (`mq`). If the configured KTIR_PREFIX_LEN were <= CB_BATCH_TEMPLATE the
        // INCREMENTAL forward would overflow the cache after the first chunk (the
        // shim's slot-bound guard would reject step 2 at RUNTIME) — turn that into a
        // `cargo build` failure here (this proc-macro body runs at build time). Only
        // gated to the paged mode (the KTIR/default emulator cap is deliberately tiny).
        if std::env::var("SCRATCHY_SENDNN_MODE").as_deref() == Ok("paged")
            && prefix_len.get() <= CB_BATCH_TEMPLATE
        {
            panic!(
                "[spyre-sendnn] paged mode: KTIR_PREFIX_LEN={prefix_len} <= mq \
                 (CB_BATCH_TEMPLATE={CB_BATCH_TEMPLATE}). The resident cache must hold MORE than one \
                 incremental forward's worth of tokens to serve longer context — set KTIR_PREFIX_LEN \
                 to a multiple of 64 well above {CB_BATCH_TEMPLATE} (e.g. 2048)."
            );
        }
        // `_cb_decode_graph` = the whole-graph decode lowering (stays empty under
        // the flit-cap guard for 30 layers); the runnable artifacts are the
        // per-group `cb_decode_groups`.
        // DECODE graph mq: the SMALL DECODE_MQ for paged (the proven ~10.7× decode
        // lever — incremental decode has 1 real token, so mq=2 right-sizes the per-
        // token forward); CB_BATCH_TEMPLATE for default mode (which drives one graph
        // for both phases). NUMERICS identical: row 0 is the real token, the rest pad.
        let decode_mq = if sendnn_paged {
            DECODE_MQ
        } else {
            CB_BATCH_TEMPLATE
        };
        let (_cb_decode_graph, mut cb_decode_groups): (String, GroupGraphs) =
            match crate::to_wavefront::lower_decode_to_wavefront(
                fuf,
                decode_asn_check,
                inferred,
                decode_bounds,
                model,
                crate::to_wavefront::PrefixCapacity::new(prefix_len),
                decode_mq,
            ) {
                Ok(lwd_cb) => {
                    let (_m, _n, sg, groups) = emit_bundle(
                        &lwd_cb,
                        format!("ktir_decode_cb_{stem_id}"),
                        prefix_len.get(),
                        Some(("s0", decode_mq)),
                        None,
                        scratchy_target_spyre::lower_subtile_tape_to_superdsc::ActiveCap::FULL,
                        // The CB sengraph's rows are a batch SYMBOL the DEM resolves at runtime, not
                        // a baked request count.
                        1,
                    );
                    (sg, groups)
                }
                Err(e) => {
                    panic!("[wavefront] {stem}: CB decode (m={decode_mq}) NOT LOWERED — {e}")
                }
            };

        // DEFAULT-MODE OVERRIDE: replace the offline_decoder CB-decode groups with
        // the single static-SDPA default-mode graph (emitted from the m=1 decode
        // tape above). The bundle bake below stays unchanged structurally — it
        // bakes one `SengraphBundleGroup` per group; in default mode there is ONE
        // group whose `prefill` and `decode` slots both hold the SAME single graph
        // (the worker drives it for both phases; KV is host-grown, not resident, so
        // there is no prefill↔decode KV-by-name linkage to satisfy). The `>=2`
        // / KV-shape / count guards below all pass trivially (one group, identical
        // graphs). The offline_decoder env is NOT required for this graph.
        // DEFAULT mode: one group, prefill slot == decode slot (host-grown KV, single
        // static-SDPA graph drives both phases). PAGED mode: KEEP the distinct paged
        // decode groups the CB-decode `emit_bundle` produced above (Store+Compute over
        // the resident cache) — do NOT clobber them with the default clone.
        if !sendnn_paged {
            cb_decode_groups = default_decode_groups.clone();
        }

        // ⛔ GUARD (guard-every-crash-at-build-time) for deeprt.cpp:3317
        // "Insufficient sengraphs passed for decoder compilation, expected >=2":
        // offline_decoder compiles [prefill, decode] TOGETHER. Emitting a decode
        // sengraph without a prefill one would make the shim pass a single graph and
        // crash on-card. Fail the BUILD instead.
        // With the layer-group split the runnable artifacts are the per-group
        // graphs; the whole-graph `pg`/`cb_decode_graph` stay empty under the
        // flit-cap guard. The "both graphs present" check is now per-GROUP: the
        // prefill + decode group lists must be non-empty.
        // "prefill present": DEFAULT mode — the single static-SDPA graph fills both
        // slots (default_decode_groups non-empty). PAGED mode — default_decode_groups is
        // empty (the m=1 decode emits no paged sengraph); the prefill groups live in the
        // `prefill` emit_bundle result (the paged SDPA+Store graph). Without this, paged
        // would see prefill_sg_present=false and CLEAR the paged decode groups below →
        // "bundle has no layer groups (split produced 0)" at worker load.
        let prefill_sg_present = if sendnn_paged {
            prefill
                .as_ref()
                .is_some_and(|(_, _, _, pgs)| !pgs.is_empty())
        } else {
            !default_decode_groups.is_empty()
        };
        // offline_decoder compiles [prefill, decode] TOGETHER (deeprt.cpp:3317 expects
        // >=2). If the decode groups exist but prefill ones do NOT, the bundle is
        // INCOMPLETE — most commonly because the single-symbol SDPA prefill was REFUSED
        // by the flit-cap guard (its 256-bucket logits output = 2.29 MB > the 1.03 MB
        // cap; the only fix is a chunked prefill, not yet landed). Rather than fail
        // `cargo build` (which would block EVERY -Fsendnn build, incl. other models),
        // mirror the flit-cap guard's LOG-NOT-PANIC pattern: drop the decode groups so
        // the baked SENGRAPH_BUNDLE is EMPTY, and let the spyre worker's startup guard
        // (spyre_worker.rs: empty prefill/decode graph_json) reject the model at load
        // with a clear message — the card never sees a single-graph (>=2 violating) job.
        if !cb_decode_groups.is_empty() && !prefill_sg_present {
            eprintln!(
                "[spyre-sendnn] {stem}: sendnn bundle has decode group graphs but NO prefill ones \
                 (the SDPA prefill was likely flit-cap-refused; chunked prefill not yet landed) — \
                 offline_decoder requires BOTH per group (deeprt.cpp:3317 expects >=2). Baking an \
                 EMPTY SENGRAPH_BUNDLE; the worker startup guard will reject `--device sendnn` for \
                 this model. (Set KTIR_PREFILL_LEN / land chunked prefill to produce a runnable bundle.)"
            );
            cb_decode_groups.clear();
        }

        // ⛔ PRIORITY-1 BUILD-TIME GUARD (guard-every-crash-at-build-time): the opt-in
        // paged path was explicitly requested (SCRATCHY_SENDNN_MODE=paged), so an EMPTY
        // or INCOMPLETE offline_decoder bundle MUST be a `cargo build` error — NOT the
        // deferred worker-load "bundle has no layer groups (split produced 0)" runtime
        // failure (the exact error the prefill_sg_present fix addressed). A runnable
        // paged bundle needs BOTH a prefill (SDPA+PagedAttnStore) and a decode
        // (PagedAttnCompute) graph per group; emitting neither/one is a lowering/codegen
        // bug. (The graceful log-not-panic above is DEFAULT-path-only, where the SDPA
        // prefill may be flit-refused without blocking other models' -Fsendnn builds.)
        if sendnn_paged && (cb_decode_groups.is_empty() || !prefill_sg_present) {
            panic!(
                "[spyre-sendnn] {stem}: SCRATCHY_SENDNN_MODE=paged produced an INCOMPLETE bundle \
                 (prefill_present={prefill_sg_present}, decode_groups={}) — the paged offline_decoder \
                 bundle needs BOTH a prefill (SDPA+PagedAttnStore) and a decode (PagedAttnCompute) \
                 graph per group. Check lower_graph_to_sengraph_paged_prefill/_decode + the \
                 emit_bundle paged routing; a worker-load failure must never be the first signal.",
                cb_decode_groups.len()
            );
        }

        // Bake the bundle into the binary as a const the per-arch
        // `ScratchyWeights::ktir_bundle()` override returns — no runtime disk read.
        // ⭐ A BUNDLE IS NAMED, NOT COPIED. Its programs were submitted to the registry above as
        // const data; this records WHICH set is this model's, and `bundle_code::bundle(fp)`
        // resolves it — the same resolver the ladder rungs and re-rolled siblings already use.
        let fp_lit = |fp: &std::cell::RefCell<Option<String>>| {
            let fp = fp.borrow().clone().unwrap_or_default();
            let l = proc_macro2::Literal::string(&fp);
            quote! {
                ::scratchy_target_spyre::manifest::KtirBundleData { fp: #l }
            }
        };
        let decode_lits: Vec<proc_macro2::TokenStream> = vec![fp_lit(&superdsc_fp)];
        let prefill_tokens = if superdsc_prefill_fp.borrow().is_some() {
            let p = fp_lit(&superdsc_prefill_fp);
            quote! { ::core::option::Option::Some(#p) }
        } else {
            quote! { ::core::option::Option::None }
        };
        *ktir_bundle_out = quote! {
            /// Embedded KTIR bundle — baked by `#[forward]` under `-Fspyre`.
            /// The fingerprint-matched `ScratchyWeights::ktir_bundle()` returns
            /// `&KTIR_BUNDLE`; the spyre worker runs it in-memory (no disk read).
            #[cfg(feature = "spyre")]
            pub const KTIR_BUNDLE: ::scratchy_target_spyre::manifest::KtirBundle =
                ::scratchy_target_spyre::manifest::KtirBundle {
                    decode: &[ #(#decode_lits),* ],
                    prefill: #prefill_tokens,
                };
        };
        let dml = proc_macro2::Literal::string(&decode_manifest);

        // ── `--target sendnn`: bake the LAYER-GROUP sub-bundles. Each group is a
        //    `[prefill, decode]` offline_decoder pair (the worker compiles each
        //    independently + threads the hidden state group→group on the host).
        //    Pair the prefill-group and decode-group by group index; carry the
        //    group IO wiring (input/result tensor ids = the residual edge). The
        //    per-group `manifest_json` is the SHARED full manifest (prefill `pml`
        //    for prefill, decode `dml` for decode) — the worker filters sources by
        //    the group graph's PrimaryInputs + uses the group's result tensor. ──
        // DEFAULT mode: the "prefill" slot holds the SAME single static-SDPA graph as
        // the "decode" slot (one graph drives both phases; host-grown KV). PAGED mode:
        // pair the REAL paged prefill groups (SDPA+Store, distinct from the paged decode)
        // from the `prefill` emit_bundle so the bundle has genuinely distinct
        // [prefill,decode] graphs sharing the resident cache by name.
        let prefill_groups: &GroupGraphs = if sendnn_paged {
            prefill
                .as_ref()
                .map(|(_, _, _, pgs)| pgs)
                .unwrap_or(&default_decode_groups)
        } else {
            &default_decode_groups
        };
        let prefill_manifest_lit = match &prefill {
            Some((pm, _, _, _)) => proc_macro2::Literal::string(pm),
            None => proc_macro2::Literal::string(&decode_manifest),
        };
        // Pair groups by index (prefill ⟷ decode). They MUST agree in count (same
        // model, same group_size) — guard it.
        if !cb_decode_groups.is_empty() && prefill_groups.len() != cb_decode_groups.len() {
            panic!(
                "[spyre-sendnn] {stem}: prefill split into {} groups but decode into {} — \
                 the layer-group split must produce the same group count for both phases",
                prefill_groups.len(),
                cb_decode_groups.len()
            );
        }
        // ⛔ GUARD (guard-every-crash-at-build-time, priority-1, SILENT-FAILURE class):
        // the offline_decoder links prefill↔decode KV by NODE NAME
        // (`_KVCACHE_TENSOR_ADDRESSES`, dem_frontend.cpp:559-628 +
        // `DT_CHECK(otherIsg->hasNode(name))`). The resident `kcache_{L}`/`vcache_{L}`
        // sources are bound to the SAME on-device tensor across both graphs. If
        // prefill and decode emit DIFFERENT shapes for the same-named cache (e.g. a
        // prefill `cap` hardcoded to 256 vs a decode `cap` derived from the tape for
        // a non-256 prefix), the card silently reads/writes a mismatched KV layout —
        // NO crash, WRONG numerics. Cross-check every `{k,v}cache_*` input node's
        // shape between the paired prefill/decode group graphs; a mismatch fails the
        // BUILD. Extracts shapes from the shipped JSON (the exact on-card artifact).
        let kv_cache_shapes = |graph_json: &str| -> std::collections::BTreeMap<String, Vec<i64>> {
            let mut out = std::collections::BTreeMap::new();
            let v: serde_json::Value =
                serde_json::from_str(graph_json).unwrap_or(serde_json::Value::Null);
            if let Some(inputs) = v.get("input_nodes").and_then(|n| n.as_array()) {
                for node in inputs {
                    let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    if !(name.starts_with("kcache_") || name.starts_with("vcache_")) {
                        continue;
                    }
                    // Resident KV caches are CONCRETE rank-4; collect the integer dims
                    // (a symbolic dim would already be a separate guard violation).
                    let shape = node
                        .get("output_tensors")
                        .and_then(|t| t.as_array())
                        .and_then(|a| a.first())
                        .and_then(|t| t.get("shape"))
                        .and_then(|s| s.as_array())
                        .map(|dims| {
                            dims.iter()
                                .map(|d| d.as_i64().unwrap_or(-1))
                                .collect::<Vec<i64>>()
                        })
                        .unwrap_or_default();
                    out.insert(name.to_string(), shape);
                }
            }
            out
        };

        let mut group_tokens: Vec<proc_macro2::TokenStream> = Vec::new();
        for (gi, (g_idx, in_t, res_t, dec_json)) in cb_decode_groups.iter().enumerate() {
            // Matching prefill group (same index); its graph json + IO wiring.
            let pre_json = prefill_groups
                .get(gi)
                .map(|(_, _, _, j)| j.as_str())
                .unwrap_or("");
            let pre_in = prefill_groups
                .get(gi)
                .map(|(_, i, _, _)| *i)
                .unwrap_or(*in_t);
            let pre_res = prefill_groups
                .get(gi)
                .map(|(_, _, r, _)| *r)
                .unwrap_or(*res_t);
            // prefill + decode group IO must match (same residual edges).
            if pre_in != *in_t || pre_res != *res_t {
                panic!(
                    "[spyre-sendnn] {stem}: group {g_idx} IO mismatch prefill (in t{pre_in}, out t{pre_res}) \
                     vs decode (in t{in_t}, out t{res_t})"
                );
            }
            // ⛔ KV-cache shape agreement (priority-1, silent-numerics): every
            // same-named resident cache MUST have identical shapes in both phases.
            let pre_kv = kv_cache_shapes(pre_json);
            let dec_kv = kv_cache_shapes(dec_json);
            for (name, pshape) in &pre_kv {
                if let Some(dshape) = dec_kv.get(name)
                    && pshape != dshape
                {
                    panic!(
                        "[spyre-sendnn] {stem}: group {g_idx} KV-cache `{name}` shape MISMATCH — \
                         prefill {pshape:?} vs decode {dshape:?}. The offline_decoder links the \
                         resident KV by node name (_KVCACHE_TENSOR_ADDRESSES); disagreeing shapes \
                         silently corrupt the on-card KV layout (no crash, wrong numerics). Both \
                         arms in lower_subtile_tape_to_sengraph.rs must derive `cap` the SAME way \
                         (from b.graph.shape(inputs[1].tensor).rows)."
                    );
                }
            }
            let g_lit = *g_idx;
            let in_lit = *in_t;
            let res_lit = *res_t;
            // ── AHEAD-OF-TIME g2 cache (mirror of cuda's cudaforge `.cu` drop) ──
            // Compute a stable fingerprint of each shipped graph JSON and DUMP the
            // JSON to the sengraphforge cache keyed by it. `scratchy-builder-spyre`'s
            // build.rs (which depends on this model crate, so it runs AFTER this
            // emit — the cuda emit-then-compile ordering) picks each `<fp>.sengraph.json`
            // up, runs CompileGraph → SerializeToString on the card, and writes the
            // compiled `<fp>.g2.sen` next to it; that g2 is `include_bytes!`-baked into
            // the binary and reached at run time by this same fingerprint. The empty
            // string => no AoT key (the worker always CompileGraphs at load).
            // SuperDSC is its OWN target: it emits ONLY the work-divided dxp bundle
            // (compiled + embedded, stamped SUPERDSC_BUNDLE below) and NEVER a
            // sengraph. So under `--features superdsc` we do NOT drop a sengraph into
            // the sengraphforge cache — there is no g2 AoT bake, no CompileGraph, no
            // sengraph at all on this target.
            // ⛔ NO SENGRAPH IS DROPPED. There is no g2 ahead-of-time bake and no `CompileGraph`
            // on this target — the bundle's programs are what the device runs.
            let (pre_fp, dec_fp) = (String::new(), String::new());
            let pre_fp_lit = proc_macro2::Literal::string(&pre_fp);
            let dec_fp_lit = proc_macro2::Literal::string(&dec_fp);
            // SuperDSC ROUTING (#51 item 3): under `SCRATCHY_SENDNN_MODE=superdsc`
            // the m=1 decode walk produced a work-divided dxp bundle keyed by
            // `superdsc_fp`. Stamp BOTH the prefill and decode `graph_json` of the
            // (single) group to the sentinel `SUPERDSC_BUNDLE:{fp}` so the worker
            // resolves the baked bundle dir (`baked_superdsc_dir(fp)`) rather than
            // treating the string as a sengraph JSON. The sengraph `graph_json` is
            // overridden ONLY in superdsc mode; the default/paged sengraph path is
            // untouched (superdsc_fp stays None).
            let superdsc_sentinel = superdsc_fp
                .borrow()
                .as_ref()
                .map(|fp| format!("SUPERDSC_BUNDLE:{fp}"));
            // C4: the PREFILL slot resolves the DISTINCT m=N prefill dxp bundle (lm_head skipped)
            // when it baked; otherwise it falls back to the decode fp (worker uses decode for both
            // = sequential prefill, the prior behavior). Never a sengraph string in superdsc mode.
            let superdsc_prefill_sentinel = superdsc_prefill_fp
                .borrow()
                .as_ref()
                .map(|fp| format!("SUPERDSC_BUNDLE:{fp}"));
            let (pre_graph, dec_graph): (&str, &str) =
                match (&superdsc_prefill_sentinel, &superdsc_sentinel) {
                    (Some(pre), Some(dec)) => (pre.as_str(), dec.as_str()),
                    (None, Some(dec)) => (dec.as_str(), dec.as_str()),
                    _ => (pre_json, dec_json),
                };
            let pre_json_lit = proc_macro2::Literal::string(pre_graph);
            let dec_json_lit = proc_macro2::Literal::string(dec_graph);
            // PREFILL WIDTH LADDER: ascending `[(mq, sentinel)]`. Emitted ONLY when >1 rung actually
            // baked — a single rung is exactly the existing `prefill` slot, so an empty slice keeps
            // the pre-ladder path byte-identical instead of making the worker special-case a 1-rung
            // ladder that buys nothing.
            // The prefix-capable prefill bundle (start>0 chunks). Its width is the widest rung.
            let prefill_prefix_lit = {
                let fp = superdsc_prefill_prefix_fp.borrow();
                let mq = superdsc_prefill_rungs
                    .borrow()
                    .iter()
                    .map(|(m, _)| *m)
                    .max()
                    .unwrap_or(0);
                match fp.as_ref() {
                    Some(f) => {
                        let m = proc_macro2::Literal::u32_unsuffixed(mq);
                        let sfp = proc_macro2::Literal::string(&format!("SUPERDSC_BUNDLE:{f}"));
                        quote! { (#m, #sfp) }
                    }
                    None => quote! { (0u32, "") },
                }
            };
            let prefill_rung_lits: Vec<proc_macro2::TokenStream> = {
                let mut rungs = superdsc_prefill_rungs.borrow().clone();
                rungs.sort_by_key(|(mq, _)| *mq);
                if rungs.len() > 1 {
                    rungs
                        .iter()
                        .map(|(mq, fp)| {
                            let m = proc_macro2::Literal::u32_unsuffixed(*mq);
                            let s = proc_macro2::Literal::string(&format!("SUPERDSC_BUNDLE:{fp}"));
                            quote! { (#m, #s) }
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            };
            // The decode batch ladder, same shape and rule as the prefill one: emitted only when
            // there is more than one rung, so a build that lowered a single width ships the empty
            // slice and the worker keeps its one-request-per-forward path untouched.
            let decode_rung_lits: Vec<proc_macro2::TokenStream> = {
                let mut rungs = superdsc_decode_rungs.borrow().clone();
                rungs.sort_by_key(|(n, _, _)| *n);
                rungs.dedup_by_key(|(n, _, _)| *n);
                if rungs.len() > 1 {
                    rungs
                        .iter()
                        .map(|(n, swept, fp)| {
                            // ⛔ `RungSeqs`, NOT a bare u32: `superdsc_decode_rungs` is keyed by
                            // `decode_rows` (the BATCH WIDTH), while the sibling `sk_bucket_rungs` list is
                            // keyed by `active_cap` (the SWEEP EXTENT). Both were `(u32, String)`, and the
                            // worker selects over this one with `>= live`. Naming the quantity here is what
                            // stops the two ladders being interchangeable at the call site.
                            let n = proc_macro2::Literal::u32_unsuffixed(*n);
                            let sw = proc_macro2::Literal::u32_unsuffixed(*swept);
                            let sfp =
                                proc_macro2::Literal::string(&format!("SUPERDSC_BUNDLE:{fp}"));
                            quote! {
                                (
                                    ::scratchy_target_spyre::manifest::RungSeqs::new(#n),
                                    ::scratchy_target_spyre::manifest::SweptCols::new(#sw),
                                    ::scratchy_target_spyre::manifest::BundleSentinel::new(#sfp),
                                )
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            };
            group_tokens.push(quote! {
                ::scratchy_target_spyre::manifest::SengraphBundleGroup {
                    group: #g_lit,
                    input_tensor: #in_lit,
                    result_tensor: #res_lit,
                    prefill: ::scratchy_target_spyre::manifest::SengraphBundleData {
                        manifest_json: #prefill_manifest_lit,
                        graph_json: #pre_json_lit,
                        g2_fingerprint: #pre_fp_lit,
                    },
                    decode: ::scratchy_target_spyre::manifest::SengraphBundleData {
                        manifest_json: #dml,
                        graph_json: #dec_json_lit,
                        g2_fingerprint: #dec_fp_lit,
                    },
                    prefill_rungs: &[#(#prefill_rung_lits),*],
                    decode_rungs: &[#(#decode_rung_lits),*],
                    prefill_prefix: #prefill_prefix_lit,
                }
            });
        }
        // ⭐ THE COMPILED DEVICE CODE GOES INTO THE BINARY. This drains the bounded bake queue and
        // renders every bundle it compiled as an `inventory::submit!`ed value the runtime looks up by
        // fingerprint.
        ktir_bundle_out.extend(superdsc_bundle_tokens());
        // The typed launch wiring, spliced into the SAME per-model module that
        // holds `Weights` — so the binding of a source to a weight can be
        // generated code naming a real field, the way metal's forward names
        // `wm.q_proj`, instead of a disk-key string resolved at load.
        {
            let (dec, pre) = &*wiring_tokens.borrow();
            if let Some(dec) = dec {
                let pre = match pre {
                    Some(p) => quote! { ::core::option::Option::Some(#p) },
                    None => quote! { ::core::option::Option::None },
                };
                ktir_bundle_out.extend(quote! {
                    /// The launch wiring for this model, emitted by `#[forward]`.
                    /// Replaces parsing `manifest.json` at model load.
                    pub static SUPERDSC_WIRINGS: ::scratchy_target_spyre::wiring::Wirings =
                        ::scratchy_target_spyre::wiring::Wirings {
                            decode: #dec,
                            prefill: #pre,
                        };
                });
            }
        }
        ktir_bundle_out.extend(quote! {
            /// Embedded sendnn LAYER-GROUP bundle — baked by `#[forward]` under
            /// `-Fspyre`. Reached by the `sendnn`-feature spyre worker, which
            /// compiles EACH group's `[prefill, decode]` independently on the AIU
            /// and threads the hidden state group→group on the host.
            #[cfg(feature = "spyre")]
            pub const SENGRAPH_BUNDLE: ::scratchy_target_spyre::manifest::SengraphBundle =
                ::scratchy_target_spyre::manifest::SengraphBundle {
                    groups: &[ #(#group_tokens),* ],
                };
        });
        // KTIR rides the embedded const + try_load.
    }
}

#[allow(clippy::too_many_arguments)]
pub fn emit_model(
    program: &Program,
    model: &ModelParams,
    fuf: &Fuf,
    sfufs: &WorkloadAssignments,
    loops: &WorkloadLoops,
    lib: &ImplementationLibrary,
    manifest: &crate::weights_manifest::WeightsManifest,
    // Shape inference for this arch — threaded only so the PD-wavefront
    // macro-emission (`dump_wavefront_mega`) can resolve weight shapes;
    // unused on the normal codegen path.
    inferred: &crate::shape::Inferred,
    canonical_override: Option<&Ident>,
    tp_world_size: u8,
    emit_fingerprint: bool,
) -> TokenStream {
    // Vision encoders have no terminal `gemm(<tile>, lm_head)`; the
    // entire FUF is the backbone. The `BackboneLayout::Encoder` arm
    // (which already covers text-side encoders like ModernBERT)
    // handles this naturally because `backbone_layout` reports
    // Encoder when the terminal isn't `gemm(_, lm_head)`. No
    // separate vision flag needed.

    // Backend dispatch: route to Metal codegen if target is Metal
    // Note: target_profile is not passed to emit_model, so we infer from lib
    // For now, emit CUDA code (Metal codegen integration is Phase 3.2+)
    // TODO: Add target_profile parameter and dispatch based on backend

    if let Some(canonical) = canonical_override {
        return emit_shim_model(
            program,
            fuf,
            sfufs,
            lib,
            model,
            manifest,
            canonical,
            tp_world_size,
            emit_fingerprint,
        );
    }
    // The struct emitter, callable twice: once with instruction
    // selection's accessors and — for a tape-scheduled arch, after
    // the front-end swap has produced them — once with the tape's.
    let emit_struct = |tape_accs: Option<&[WeightAccessor]>| {
        emit_weights_struct(
            program,
            fuf,
            sfufs,
            tape_accs,
            lib,
            model,
            manifest,
            WeightsEmitMode::Canonical,
            tp_world_size,
            emit_fingerprint,
        )
    };
    #[cfg(feature = "metal")]
    let mut tape_accessors_for_struct: Option<Vec<WeightAccessor>> = None;
    #[cfg(feature = "metal")]
    let mut weights = emit_struct(None);
    // THE MODEL'S WEIGHT BINDING, from the tape both backends lower.
    //
    // Instruction selection (cuda's solver) and metal's instruction-stream
    // bridge are the only other producers of this set, and each is locked to
    // its target — so under spyre BOTH are empty and the emitted `Weights`
    // carried no weight fields at all. That absence is the whole reason the
    // spyre worker re-finds every tensor at runtime by string-formatting
    // `"{disk}.weight"`: there was no field for a binding to name.
    //
    // `accessors_from_tape` reads only `LoweredDecode.bindings` + the op that
    // consumes each one, so it is a fact about the MODEL. The struct it feeds
    // is already target-neutral (`crate::__gpu::layers::*`, where `__gpu` is
    // the per-target alias), so the same fields compile and load on any
    // backend.
    #[cfg(all(feature = "spyre", not(feature = "metal")))]
    let weights = {
        let lowered = sfufs
            .per_workload
            .keys()
            .find(|w| w.num_tokens == 1)
            .or_else(|| sfufs.per_workload.keys().next())
            .copied()
            .and_then(|wp| {
                let bounds = bounds_for_wp(model, wp, tp_world_size);
                // Tape-authoritative: no assignment (there is no solve under
                // spyre), exactly as the bucket-folding lowering below.
                crate::to_wavefront::lower_decode_to_wavefront(
                    fuf,
                    None,
                    inferred,
                    &bounds,
                    model,
                    crate::to_wavefront::PrefixCapacity::new(
                        std::num::NonZeroU32::new(8192).expect("8192 != 0"),
                    ),
                    wp.num_tokens as u32,
                )
                .ok()
            });
        let tw = lowered.as_ref().map(|l| {
            crate::weight_bindings::from_tape(
                l,
                program,
                crate::weight_bindings::WeightAbi {
                    // Two matmuls plus a SiluMul — no packed-buffer kernel to read one.
                    mlp: crate::weight_bindings::MlpPacking::Split,
                    // ⛔ DERIVED, NOT ASSUMED. This was a hardcoded `Dense`, which is what the
                    // path did before these facts were arguments — and hardcoding it is the same
                    // shape of bug as the `"{disk}.weight"` naming rule: right for the models in
                    // front of you, silently wrong for the next one. Read from the FUF, exactly
                    // as the metal bridge reads it, so the two cannot disagree about one model.
                    embed: match fuf.nodes.iter().find_map(|n| {
                        (n.op == OpKind::Embed)
                            .then(|| crate::weight_vocab::weight_storage_of(n))
                            .flatten()
                    }) {
                        Some(crate::quantization::StorageFormat::Affine { .. }) => {
                            crate::weight_bindings::EmbedStorage::AffineQuant
                        }
                        _ => crate::weight_bindings::EmbedStorage::Dense,
                    },
                },
            )
            .unwrap_or_else(|e| panic!("[weight-bindings] {}: {e}", model.source_stem))
        });
        eprintln!(
            "[weight-bindings] {}: {} weight accessors from the tape",
            model.source_stem,
            tw.as_ref().map(|t| t.accessors.len()).unwrap_or(0),
        );
        let struct_tokens = emit_struct(tw.as_ref().map(|t| t.accessors.as_slice()));
        // The id → FIELD binding, emitted INTO THE SAME MODULE as the struct so
        // its arms can name `w.self_attn_q_proj[3]` directly. This is what the
        // worker's `"{disk}.weight"` lookup is replaced by; only generated code
        // can name a generated field.
        let bindings = match (lowered.as_ref(), tw.as_ref()) {
            (Some(l), Some(t)) => crate::weight_bindings::emit_weight_bindings(l, program, t)
                .unwrap_or_else(|e| panic!("[weight-bindings] {}: {e}", model.source_stem)),
            // ⛔ THE FN IS ALWAYS EMITTED, AND IT REFUSES. The dispatcher's
            // `superdsc_weights` override names it per variant, so omitting it
            // is a confusing compile error in generated code — but emitting an
            // EMPTY binding would be worse: a model that silently loads zero
            // weights and produces garbage. A model whose decode point did not
            // lower has no binding to give, and says so.
            _ => {
                let stem = model.source_stem.as_str();
                quote! {
                    pub fn superdsc_weights(
                        _w: &Weights,
                    ) -> ::anyhow::Result<::std::vec::Vec<::scratchy_forward_compiler::BoundWeight>> {
                        ::anyhow::bail!(
                            "{}: no weight binding was generated — the decode point did not lower, \
                             so this model cannot be launched on spyre",
                            #stem,
                        )
                    }
                }
            }
        };
        quote! { #struct_tokens #bindings }
    };
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    let weights = emit_struct(None);

    // Group workload points by SFUF signature (sorted subgraph → impl).
    // Buckets with identical impl picks produce byte-identical fn
    // bodies, so we lower the canonical ONCE and emit duplicates as
    // thin `#[inline(always)]` shims that delegate to the canonical
    // fn. Public API (every `forward_m_<M>[_sk_<SK>]` /
    // `forward_backbone_m_<M>[_sk_<SK>]` name a user might take a
    // fn-pointer to) is preserved. Dedup runs over `(num_tokens,
    // sk_bucket)` 2-D points so models with `sk_buckets` declared get
    // the same compile-time win.
    // Metal and spyre both lower from the shared tape; cuda is the instruction-selection path.
    #[cfg(any(feature = "metal", feature = "spyre"))]
    let tape_pilot_arch = true;
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    let tape_pilot_arch = false;
    let bucket_points: Vec<crate::assignment::WorkloadPoint> =
        sfufs.per_workload.keys().copied().collect();
    let mut sfuf_to_canonical: HashMap<Vec<(u32, u32)>, crate::assignment::WorkloadPoint> =
        HashMap::new();
    let mut bucket_canonical: Vec<crate::assignment::WorkloadPoint> =
        Vec::with_capacity(bucket_points.len());
    for wp in &bucket_points {
        let sfuf = &sfufs.per_workload[wp];
        let mut sig: Vec<(u32, u32)> = sfuf.impls.iter().map(|(sg, imp)| (sg.0, imp.0)).collect();
        sig.sort();
        let canonical = *sfuf_to_canonical.entry(sig).or_insert(*wp);
        bucket_canonical.push(canonical);
    }
    // M2b step 3 (second half, transition): fold buckets by TAPE
    // fingerprint — the tape lowered per point with its `m` fields
    // normalized to a sentinel — and ASSERT it reproduces the
    // solve's impl-signature folding for pilot arches. When this
    // gate has held across the fleet, the tape folding replaces the
    // solve as the bucket authority and the per-arch solve dies.
    #[cfg(feature = "metal")]
    if tape_pilot_arch {
        let mut tape_to_canonical: HashMap<String, crate::assignment::WorkloadPoint> =
            HashMap::new();
        for (bi, wp) in bucket_points.iter().enumerate() {
            let bounds = bounds_for_wp(model, *wp, tp_world_size);
            let fp = match crate::to_wavefront::lower_decode_to_wavefront(
                fuf,
                None,
                inferred,
                &bounds,
                model,
                crate::to_wavefront::PrefixCapacity::new(std::num::NonZeroU32::new(8192).unwrap()),
                wp.num_tokens as u32,
            ) {
                Ok(l) => {
                    let m = wp.num_tokens as u32;
                    l.input
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
                            let mm = if od.m == m {
                                if m == 1 { u32::MAX } else { u32::MAX - 1 }
                            } else {
                                od.m
                            };
                            format!("{:?}#{mm}#{:?};", od.op, od.inputs)
                        })
                        .collect::<String>()
                }
                Err(e) => format!("REFUSED:{e:?}"),
            };
            let tape_canonical = *tape_to_canonical.entry(fp).or_insert(*wp);
            // The solve no longer runs for a pilot arch, so there is
            // nothing to compare against — the tape IS the folding.
            // (The equality this once asserted was proven across the
            // pilot fleet before the solve was switched off; runtime
            // bit-exactness is the standing gate now, per PLAN §M2b.)
            let _unused_compare = (tape_canonical, bucket_canonical[bi]);
            // The tape IS the bucket authority for a pilot arch now:
            // assigned from its own folding. Byte-identical today (the
            // assert above says so) — the point is that the solve's
            // folding is no longer the SOURCE, which is what deleting
            // the solve requires.
            bucket_canonical[bi] = tape_canonical;
        }
    }

    // Lower every canonical tape_index once, backbone-shaped: skip the
    // terminal subgraph (the `gemm(<final_norm>, lm_head)` row) and
    // emit it separately as a tiny LM_HEAD slice. forward and
    // forward_backbone share the backbone slice; forward additionally
    // runs LM_HEAD; forward_backbone DtoD-copies the backbone-output
    // slot. No more pair of near-identical full slices per tape_index.
    let mut arch_opcodes = ArchOpcodes::new();
    let mut canonical_lowered: BTreeMap<
        crate::assignment::WorkloadPoint,
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
    // `last_node_id` must be the lm_head Gemm (tp=1) or the post-lm_head
    // AllGather (tp>1), not one of the post-Embed `MmEmbedSplice` nodes
    // that `tp_lowering::insert_mm_splices` appends. Those sit at fuf-
    // array-tail but semantically belong near the Embed; walking past
    // them with `last_non_splice_node` recovers the real terminal.
    let last_node_id = last_non_splice_node(fuf)
        .expect("non-empty FUF expected")
        .id;
    let layout = backbone_layout(fuf, program);
    // For encoder layouts (text-side encoders like ModernBERT AND
    // every `#[vision_forward]` body) the FUF's terminal IS the
    // backbone output; decoder layouts carry the lm_head Gemm's
    // hidden-state input as the backbone output.
    let backbone_out = match layout {
        BackboneLayout::Decoder { backbone_out } => backbone_out,
        BackboneLayout::Encoder => (last_node_id, 0),
    };
    // M2b step 3: a tape-scheduled pilot arch never runs the metal
    // instruction-selection lowering — no colored_slot_map, no
    // lower_bucket, no fan_out. The canonical tuple starts as an
    // empty placeholder and the front-end swap below fills every
    // field from the shared tape (a pilot whose swap then refuses
    // panics loudly there — an empty stream cannot ship silently
    // because the bucket table still reads these fields).
    for (i, wp) in bucket_points.iter().enumerate() {
        if bucket_canonical[i] != *wp {
            continue;
        }
        if tape_pilot_arch {
            canonical_lowered.insert(
                *wp,
                (
                    CanonicalLowered {
                        backbone: crate::interpreter_codegen::LoweredBucket {
                            instances: Vec::new(),
                            weight_slots: Vec::new(),
                            #[cfg(any(feature = "metal", feature = "spyre"))]
                            num_slots: 0,
                            #[cfg(any(feature = "metal", feature = "spyre"))]
                            final_slot: 0,
                            barriers: Vec::new(),
                        },
                        lm_head: crate::interpreter_codegen::LoweredBucket {
                            instances: Vec::new(),
                            weight_slots: Vec::new(),
                            #[cfg(any(feature = "metal", feature = "spyre"))]
                            num_slots: 0,
                            #[cfg(any(feature = "metal", feature = "spyre"))]
                            final_slot: 0,
                            barriers: Vec::new(),
                        },
                    },
                    0,
                    0,
                    0,
                    crate::weight_vocab::SlotMap::new(),
                ),
            );
            continue;
        }
        let sfuf = &sfufs.per_workload[wp];
        let loop_ir = loops
            .per_workload
            .get(wp)
            .expect("schedule populated every key");
        let bounds = bounds_for_wp(model, *wp, tp_world_size);
        // Decoder mode skips the terminal subgraph in the backbone
        // lowering and emits it as a separate LM_HEAD slice. Encoder
        // mode lowers the entire pipeline as the backbone (no split).
        let skip_subgraph = match layout {
            BackboneLayout::Decoder { .. } => Some(
                sfuf.subgraph_of(last_node_id)
                    .expect("terminal tile must be in a subgraph"),
            ),
            BackboneLayout::Encoder => None,
        };

        // Protect the backbone output (`take_owned` reads it). For
        // decoder we additionally protect the terminal slot so the
        // backbone's drop pass leaves it free for lm_head to write.
        let mut protected_bb: HashSet<(TileId, u8)> = HashSet::new();
        protected_bb.insert(backbone_out);
        if matches!(layout, BackboneLayout::Decoder { .. }) {
            protected_bb.insert((last_node_id, 0));
        }

        // Per-tape_index colored slot map. Computed once and shared
        // between backbone lowering and the lm_head fan_out so they
        // agree on slot indices.
        let slots = crate::interpreter_codegen::colored_slot_map(
            fuf,
            sfuf,
            loop_ir,
            lib,
            None,
            &protected_bb,
        );
        let backbone_slot = slots.of(backbone_out.0, backbone_out.1);
        // In encoder mode `take_owned` of the backbone-output slot is
        // also the terminal — the same slot index plays both roles.
        let terminal_slot = match layout {
            BackboneLayout::Decoder { .. } => slots.of(last_node_id, 0),
            BackboneLayout::Encoder => backbone_slot,
        };
        let num_slots = slots.total();

        // Decoder: skip the terminal subgraph so it emits as a
        // separate lm_head slice. Encoder (text encoders OR vision
        // bodies): pass `None` so the whole FUF lowers as backbone.
        let lowered_bb = lower_bucket(
            fuf,
            sfuf,
            loop_ir,
            program,
            model,
            lib,
            &bounds,
            skip_subgraph,
            &protected_bb,
            &mut arch_opcodes,
            backbone_out,
            &slots,
        );

        // LM_HEAD — only emitted in decoder mode. One row, computed
        // by directly invoking the terminal subgraph's `fan_out`
        // against the same slot map. Encoder mode (text encoders OR
        // vision bodies) emits an empty slice — the whole pipeline
        // already ran in the backbone.
        let lowered_lm = match layout {
            BackboneLayout::Decoder { .. } => {
                let terminal_sg =
                    skip_subgraph.expect("decoder layout always has a terminal subgraph");
                let term_imp_id = sfuf
                    .impl_of(terminal_sg)
                    .expect("terminal subgraph has an Impl assignment");
                let term_imp = lib.get(term_imp_id);
                if !term_imp.emits_host_instruction() {
                    // Claim-only terminal (e.g. spyre's `ktir_gemm` lm_head):
                    // its real lowering is the embedded bundle, not a host
                    // `Instruction`. The host-interpreter lm_head slice is not
                    // emitted for such a target, so produce an empty bucket
                    // (mirrors `lower_bucket`'s skip of claim-only impls).
                    crate::interpreter_codegen::LoweredBucket {
                        instances: Vec::new(),
                        barriers: Vec::new(),
                        weight_slots: Vec::new(),
                        #[cfg(any(feature = "metal", feature = "spyre"))]
                        num_slots,
                        #[cfg(any(feature = "metal", feature = "spyre"))]
                        final_slot: terminal_slot,
                    }
                } else {
                    let term_claimed = sfuf.tiles_in_subgraph(terminal_sg);
                    let term_match = crate::impl_lib::MatchInfo {
                        claimed_tiles: term_claimed.clone(),
                        boundary_inputs: crate::interpreter_codegen::collect_boundary_inputs(
                            fuf,
                            &term_claimed,
                        ),
                        boundary_outputs: term_claimed,
                    };
                    let term_emits = term_imp
                        .fan_out(&term_match, fuf, program, &bounds, &slots)
                        .expect("terminal subgraph's Impl must implement fan_out");
                    let term_accs =
                        term_imp.required_weights(&term_match.claimed_tiles, fuf, program);
                    let term_slots =
                        crate::interpreter_codegen::weight_accessors_to_slots(&term_accs);
                    let term_weight_slots: Vec<Vec<WeightSlot>> =
                        term_emits.iter().map(|_| term_slots.clone()).collect();
                    // Eval body lives in `scratchy_forward_compiler::Instruction::eval`
                    // — `arch_opcodes` keeps the shape registration for
                    // `emit_bucket_static_slice`'s shape-checking pass.
                    // `extra_opcode_shapes` covers storage-polymorphic
                    // impls (see the comment in `lower_bucket`).
                    arch_opcodes.register(term_imp.opcode_shape());
                    for extra in term_imp.extra_opcode_shapes() {
                        arch_opcodes.register(extra);
                    }
                    let n = term_emits.len();
                    crate::interpreter_codegen::LoweredBucket {
                        instances: term_emits,
                        // Terminal Impl (lm_head) is emitted ad-hoc here
                        // outside the FUF walker, so we don't have its
                        // dataflow. Conservative all-barriers — runtime
                        // serializes the lm_head dispatches. Upgrading
                        // this to a proper analysis means running the
                        // same walker against the terminal Impl's
                        // claimed tiles.
                        barriers: vec![true; n],
                        weight_slots: term_weight_slots,
                        #[cfg(any(feature = "metal", feature = "spyre"))]
                        num_slots,
                        #[cfg(any(feature = "metal", feature = "spyre"))]
                        final_slot: terminal_slot,
                    }
                }
            }
            BackboneLayout::Encoder => crate::interpreter_codegen::LoweredBucket {
                instances: Vec::new(),
                barriers: Vec::new(),
                weight_slots: Vec::new(),
                #[cfg(any(feature = "metal", feature = "spyre"))]
                num_slots,
                #[cfg(any(feature = "metal", feature = "spyre"))]
                final_slot: terminal_slot,
            },
        };

        canonical_lowered.insert(
            *wp,
            (
                CanonicalLowered {
                    backbone: lowered_bb,
                    lm_head: lowered_lm,
                },
                num_slots,
                backbone_slot,
                terminal_slot,
                slots,
            ),
        );
    }

    // Layer-template detection: collapse the contiguous repeating
    // sub-sequence of the slice (the per-layer transformer body)
    // into one `Instruction::Loop(N, body_len)` row + one
    // iteration's body. Fused boundary effects (e.g., FusedAddRmsNorm
    // absorbing layer L's final add into layer L+1's first norm)
    // leave layer 0 / the last layer structurally distinct, so the
    // detection picks the largest CONTIGUOUS run that genuinely
    // repeats — middle layers — and keeps the boundary residues as
    // straight-line code in prelude/suffix.
    // Pre-attention chain synthesis (compiler-driven). Detect the
    // contiguous `(FusedAddRmsNorm, AffineQmm × 3, RopeAppend)` chain
    // that spans the K→K+1 layer boundary in the unrolled per-claim
    // op list and replace each match with one `SynthPreAttn` op
    // backed by the synthesized MSL kernel emitted via
    // `emit_synthesized_kernel_sources_override`. Must run BEFORE
    // `apply_loop_compression` — once the loop body collapses we
    // can't see the boundary chain anymore.
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    let synth_t_act: Option<&'static str> = {
        use crate::quantization::QuantMethod;
        match model.quantization.as_ref().map(|q| &q.method) {
            Some(QuantMethod::Affine { bits: 4, .. }) => Some("bfloat"),
            _ => None,
        }
    };
    // Skip the synth fusion at bucket_m >= 2: the synth's
    // `(M, num_heads_total)` threadgroup grid makes its AddRmsNorm
    // phase redundantly process the residual+delta read once per
    // (token, head) — work that scales as M*num_heads instead of M.
    // At M=1 the redundancy is cheap relative to the saved qmv
    // dispatch overhead and the synth wins. At M>=2 the redundant
    // device reads dominate; the unfused chain (one norm dispatch
    // per token + per-head qmv) is strictly cheaper.
    //
    // TODO: replace this hardcoded threshold with a solver-driven
    // pick — `SynthPreAttnImpl::cost_us` vs `(FusedAddRmsNorm + 3
    // AffineQmm)::cost_us` from the swept CSV.
    // ⛔ CUDA ONLY. This searches the instruction-SELECTION stream for a repeating run, because
    // that stream has no tape to read the loop off. A tape-scheduled target HAS one — the shared
    // re-roll already found it — and its rolled stream replaces `cl` further down, so running
    // this under metal/spyre re-derived a fact that was then thrown away. Two searches over two
    // representations, free to disagree, is the arrangement this line of work exists to remove.
    #[cfg(not(any(feature = "metal", feature = "spyre")))]
    for (wp, (cl, _, _, _, _)) in canonical_lowered.iter_mut() {
        let _ = synth_t_act;
        let _ = wp;
        crate::interpreter_codegen::apply_loop_compression(
            &arch_opcodes,
            &mut cl.backbone,
            "layer",
            crate::interpreter_codegen::LoopSource::LocalSearch,
            &model.source_stem,
        );
        crate::interpreter_codegen::apply_loop_compression(
            &arch_opcodes,
            &mut cl.lm_head,
            "layer",
            crate::interpreter_codegen::LoopSource::LocalSearch,
            &model.source_stem,
        );
    }

    // Per-canonical CanonicalParams impl + Instruction type alias.
    // The alias keeps every static-slice row short instead of
    // repeating `::scratchy_forward_compiler::Instruction::<Weights>::Variant(…)`.
    let has_bias_add = program_has_bias_add(program);
    #[cfg(feature = "metal")]
    let mut resolved_metal_consts: Option<
        scratchy_target_metal::tape::model_consts::MetalModelConsts,
    > = None;
    let canonical_params_impl = emit_canonical_params_impl(
        model,
        tp_world_size,
        has_bias_add,
        program_has_gelu(program),
        fuf_uses_rotary(fuf),
        #[cfg(feature = "metal")]
        &mut resolved_metal_consts,
    );
    // Per-canonical: alias the generic `Instruction<Weights>` for
    // the slice element type AND glob-import the variant
    // constructors so each static-slice row reads `Embed(...)` /
    // `RmsNorm(...)` instead of
    // `::scratchy_forward_compiler::Instruction::<Weights>::Embed(...)`.
    // The `__I` alias and the variant glob-import are backend-agnostic
    // — `Instruction<W>` is defined in `scratchy-forward-compiler` without a
    // backend gate. The per-tape_index static slices reference both, so
    // they compile under either `cfg(feature = "cuda")` (linking the
    // cuda `Weights` struct) or `cfg(feature = "metal")` (linking the
    // metal `Weights` ZST emitted in `emit_weights_struct`).
    let instruction_alias = quote! {
        #[allow(non_camel_case_types, dead_code)]
        type __I = ::scratchy_forward_compiler::Instruction;
        use ::scratchy_forward_compiler::Instruction::*;
    };
    let shapes_by_name = arch_opcodes.shapes_by_name();

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
            &shapes_by_name,
            &lowered.backbone.instances,
        ));
        static_slices.push(emit_bucket_static_slice(
            &lm_head_static_ident,
            &shapes_by_name,
            &lowered.lm_head.instances,
        ));
        // Metal MTL4 barrier flags, computed at FUF/SlotMap level
        // by `lower_bucket` (one bool per `OpInstance`). Length
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
                scratchy_target_metal_compiler::from_subtile::emit_bucket_barriers_static(
                    &bb_barriers_ident,
                    &lowered.backbone.barriers,
                ),
            );
            static_slices.push(
                scratchy_target_metal_compiler::from_subtile::emit_bucket_barriers_static(
                    &lh_barriers_ident,
                    &lowered.lm_head.barriers,
                ),
            );
        }
    }

    // `sk_axis_active`: true when the model declared `sk_buckets`;
    // otherwise every wp has sk_bucket==0 and the table emits
    // `[0, u64::MAX)` for sk on every entry.
    let sk_axis_active = sfufs.per_workload.keys().any(|wp| wp.sk_bucket != 0);
    let num_tokens_points: Vec<u64> = sfufs.num_tokens_points();
    let mut sk_by_m: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for wp in sfufs.per_workload.keys() {
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

    // Under `-Fspyre`, `dump_wavefront_mega` fills this with the embedded
    // `pub const KTIR_BUNDLE` for the per-model module; empty otherwise.
    let mut ktir_bundle_const = proc_macro2::TokenStream::new();

    // PD-wavefront macro-emission (env-gated, diagnostic + the const builder).
    // HERE — not in the pre-emit drive — because the decode bucket's
    // `weight_slots` (the real weight-locator source) only exists post-lowering
    // + post-loop-compression, and the `(bucket, op_idx, slot)` triples must
    // match the very match table `emit_weight_accessors_impl` builds from this
    // same `canonical_lowered`.
    // Wavefront-emitting builds only — spyre lowers the decode to a single
    // fused KTIR/SuperDSC bundle. Plain cuda/metal dispatch per-op and skip
    // this; the expensive lowering + its hard panic on un-lowerable models
    // must not run for them.
    // M2 equality harness: build the decode instruction stream from the
    // SHARED subtile tape and diff it against the instruction-selection
    // stream. Iteration scaffold (M2_SCOUT env); becomes a hard assert
    // at the flip.
    #[cfg(feature = "metal")]
    // THE FRONT-END SWAP (M2): for every canonical this arch can bridge,
    // the SHARED subtile tape is the producer of the metal instruction
    // stream, barriers, and accessor weight slots. During the rollout,
    // instruction selection still runs and a HARD ASSERT proves the
    // bridge byte-identical; a refusal keeps that canonical on
    // instruction selection and is logged loudly — the rollout
    // frontier, not a silent fallback.
    #[cfg(feature = "metal")]
    {
        // Register THE shape table for every bridge variant. The impls'
        // own registrations already happened during lower_bucket;
        // ArchOpcodes panics on any disagreement — the table is proven
        // identical at every expansion until the impls die (M2b).
        for name in crate::opcode_shapes::BRIDGE_VARIANTS {
            if let Some(shape) = crate::opcode_shapes::bridge_shape(name) {
                arch_opcodes.register(shape);
            }
        }
        for (canonical, (cl, ns_field, bb_slot_field, term_field, slots_dec)) in
            canonical_lowered.iter_mut()
        {
            let m = canonical.num_tokens;
            // Key the assignment by the canonical's OWN workload point —
            // get_nt(m) missed multimodal canonicals entirely (qwen2.5-VL
            // never entered this loop: a SILENT skip).
            // Pilots run tape-authoritative: no assignment needed (and
            // once the per-arch solve skip lands, none exists).
            // A pilot runs tape-authoritative: passing the (now empty)
            // assignment would make the walk's coverage check reject
            // every tile and leave the canonical on placeholders — a
            // SILENT empty arena, which is how this surfaced.
            let asn_opt = if tape_pilot_arch {
                None
            } else {
                sfufs.per_workload.get(canonical)
            };
            if asn_opt.is_none() && !tape_pilot_arch {
                eprintln!(
                    "[m2-swap] {} m={m}: NO assignment for canonical workload point — \
                     staying on instruction selection",
                    model.source_stem
                );
                continue;
            }
            {
                // The scheduler's wave order, flattened to a per-tile
                // emission rank — the bridge emits in this order.
                let mut wave_rank: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                // The loop AND the assignment must come from the SAME
                // workload point — the canonical's (mixing get_nt's
                // sk-variant assignment with the canonical's loop gave
                // ranks from two different schedules).
                if let (Some(loop_ir), Some(c_asn)) = (
                    loops.per_workload.get(canonical),
                    sfufs.per_workload.get(canonical),
                ) {
                    let mut r = 0usize;
                    for wave in &loop_ir.waves {
                        for (sg, _) in &wave.subgraphs {
                            for t in c_asn.tiles_in_subgraph(*sg) {
                                wave_rank.insert(t.0, r);
                            }
                            r += 1;
                        }
                    }
                }
                let decode_bounds = bounds_for_wp(model, *canonical, tp_world_size);
                match crate::to_wavefront::lower_decode_to_wavefront(
                    fuf,
                    asn_opt,
                    inferred,
                    &decode_bounds,
                    model,
                    crate::to_wavefront::PrefixCapacity::new(
                        std::num::NonZeroU32::new(8192).unwrap(),
                    ),
                    m as u32,
                ) {
                    Ok(l) => {
                        let mc = resolved_metal_consts.as_ref().expect("metal consts filled");
                        // Real derivations — the SAME sources instruction
                        // selection read: the embed tile's weight storage
                        // (FUF), and the tape's own gemm weight form.
                        let embed_quant = fuf.nodes.iter().find_map(|n| {
                            if n.op != OpKind::Embed {
                                return None;
                            }
                            match crate::weight_vocab::weight_storage_of(n) {
                                Some(crate::quantization::StorageFormat::Affine {
                                    group_size,
                                    bits,
                                }) => Some((*group_size, *bits)),
                                _ => None,
                            }
                        });
                        let fused_mlp = l.input.ops.iter().all(|od| {
                            !matches!(
                                od.op,
                                scratchy_subtile::lower::LoweredOp::Gemm {
                                    weight: scratchy_subtile::lower::GemmWeight::Affine { .. },
                                    ..
                                }
                            )
                        });
                        // Embed base: the weight table entry the embed
                        // accessor uses (single-segment path; llama-class
                        // arches name it embed_tokens).
                        let embed_base = (0..)
                            .map(crate::classified::WeightId)
                            .take_while(|id| (id.0 as usize) < program.weights.len())
                            .map(|id| program.weights.path(id).join("_"))
                            .find(|p| p.contains("embed"))
                            .unwrap_or_else(|| "embed_tokens".into());
                        // The metal lowering takes DATA, not the compiler's weight types: the
                        // `WeightTable`/`Program`/`ModelParams` are this crate's, and handing
                        // them across would drag the classified-weights layer into the target.
                        let weight_paths: Vec<Vec<String>> = (0..program.weights.len())
                            .map(|i| {
                                program
                                    .weights
                                    .path(crate::classified::WeightId(i as u32))
                                    .to_vec()
                            })
                            .collect();
                        let dsl_shared_expert_bases: Vec<String> = weight_paths
                            .iter()
                            .filter_map(|p| p.first().cloned())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .filter(|b| program.weights.has_subtree(&[b, "shared_expert"]))
                            .collect();
                        let facts = scratchy_target_metal_compiler::from_subtile::StreamFacts {
                            weight_paths: &weight_paths,
                            dsl_shared_expert_bases: &dsl_shared_expert_bases,
                            wave_rank: &wave_rank,
                            embed_base,
                            hidden_size: mc.hidden_size as u32,
                            embed_quant,
                            fused_mlp,
                            intermediate_size: mc.intermediate_size as u32,
                        };
                        // M2b step-2 FLIP: pilot arches take their order,
                        // groups, and coloring from the TAPE (wave levels +
                        // tape_slot_map). Gate = per-arch runtime bit-exact
                        // output + TTFT/ITL parity vs the pre-flip build
                        // (PLAN e10642b78) — a tape-scheduled stream is
                        // legitimately not byte-equal to the solver's, so
                        // the equality assert applies only to non-pilots.
                        // A pilot refusal is a build defect, not a fallback.
                        let plan =
                            scratchy_target_metal_compiler::from_subtile::fold_plan(&l, &facts);
                        // ⭐ THE ORDER COMES OFF THE SHARED `SubtileTape` — the same object
                        // spyre lowers, built by the same `lower_region`/`lower_dag_to_tape`,
                        // read through `scratchy_target_metal::from_tape`. It replaces
                        // `wave_schedule::wave_levels`, which was metal's own re-derivation
                        // of a schedule over the pre-SubtileIR op list.
                        //
                        // The COLORING stays metal's: `tape_slot_map` knows which kernels
                        // write over an operand in place and that the embed is colour 0.
                        // That is target ABI, an INPUT to a shared pass — the tape's SSA
                        // alloc/free cannot express it, and using it would hand an in-place
                        // kernel a buffer another op still owns.
                        let tp = scratchy_target_metal_compiler::tape_program::tape_program(
                            &l,
                            &model.source_stem,
                            m,
                        );
                        let (lv, tape_items, unrolled_items, layer_segs) =
                            (tp.levels, tp.rolled, tp.unrolled, tp.segments);
                        let (tape_sm, tape_ns, tape_fs) =
                            scratchy_target_metal_compiler::from_subtile::tape_slot_map(
                                &l, &plan, &lv,
                            )
                            .unwrap_or_else(|e| {
                                panic!(
                                    "[m2-flip] {} m={m}: tape colorer refused a \
                                         DECLARED pilot: {e}",
                                    model.source_stem
                                )
                            });
                        match scratchy_target_metal_compiler::from_subtile::decode_instruction_stream(
                            &l,
                            &tape_sm,
                            &facts,
                            &tape_items,
                        ) {
                            Ok(bridged) => {
                                // ⛔ THE MoE BIT REPACK, HOISTED OUT OF THE BRIDGE. It needs
                                // the weight `Program` + `ModelParams`, which are this
                                // crate's, so the metal lowering can no longer run it
                                // itself. Dropping it would silently lose the OptiQ
                                // per-projection widths packed into `bits` on MoE arches.
                                let mut bridged = bridged;
                                bridged.backbone =
                                    crate::interpreter_codegen::repack_moe_expert_bits(
                                        bridged.backbone,
                                        program,
                                        model,
                                    );
                                bridged.lm_head =
                                    crate::interpreter_codegen::repack_moe_expert_bits(
                                        bridged.lm_head,
                                        program,
                                        model,
                                    );
                                // One hazard replay over backbone ++ lm
                                // rows: the walk is prefix-deterministic,
                                // so the backbone flags are the prefix.
                                let head_start = bridged.backbone_sigs.len();
                                let mut all_sigs = bridged.backbone_sigs;
                                all_sigs.extend(bridged.lm_head_sigs);
                                let all_flags =
                                    scratchy_target_metal_compiler::from_subtile::barrier_flags_with(
                                        &all_sigs, false,
                                    );
                                let bb_barriers = all_flags[..head_start].to_vec();
                                // Terminal slots come from the TAPE's
                                // result chain, not from hazard
                                // signatures: the logits slot is the
                                // result op's color, and the backbone's
                                // output is the lm_head Gemm's input.
                                // (Reading the last signature's first
                                // write breaks on arches whose lm_head
                                // slice carries a trailing op — granite's
                                // logits scaling — which emitted an
                                // empty completion.)
                                let lm_final = tape_fs;
                                let bb_final = {
                                    let mut cur = l.input.result;
                                    let mut hops = 0;
                                    loop {
                                        hops += 1;
                                        assert!(hops < 64, "lm_head chain walk runaway");
                                        let od = &l.input.ops[cur];
                                        let feed = od.inputs.iter().find_map(|r| match r {
                                            scratchy_subtile::lower::InputRef::Op(j) => {
                                                Some(*j)
                                            }
                                            _ => None,
                                        });
                                        match (&od.op, feed) {
                                            (
                                                scratchy_subtile::lower::LoweredOp::Gemm {
                                                    ..
                                                },
                                                Some(j),
                                            ) => {
                                                break l.op_tiles[j]
                                                    .map(|(t, sl)| {
                                                        tape_sm.of(crate::fuf::TileId(t), sl)
                                                    })
                                                    .unwrap_or(tape_fs);
                                            }
                                            (_, Some(j)) => cur = j,
                                            _ => break tape_fs,
                                        }
                                    }
                                };
                                let acc_instances: Vec<scratchy_forward_compiler::Instruction> =
                                    bridged
                                        .backbone
                                        .iter()
                                        .chain(bridged.lm_head.iter())
                                        .copied()
                                        .collect();
                                let acc_slots: Vec<Vec<crate::weight_vocab::WeightSlot>> =
                                    bridged
                                        .backbone_weight_slots
                                        .iter()
                                        .chain(bridged.lm_head_weight_slots.iter())
                                        .cloned()
                                        .collect();
                                let mut bridge_bucket =
                                    crate::interpreter_codegen::LoweredBucket {
                                        instances: bridged.backbone,
                                        weight_slots: bridged.backbone_weight_slots,
                                        num_slots: tape_ns,
                                        final_slot: bb_final,
                                        barriers: bb_barriers,
                                    };
                                // ⛔ NOTHING COMPRESSES THIS STREAM. It was emitted from the
                                // ROLLED tape, so the layer loop is already in it as an
                                // `Instruction::Loop`. Running a repeating-run search over
                                // it would be the second re-roll this removes.
                                //
                                // ⭐ SO THE CHECK CHANGED, AND GOT STRONGER. It used to ask
                                // "did compressing this stream preserve it?" — meaningless
                                // now that nothing compresses it. It now asks the question
                                // that actually matters: does the ROLLED emission expand to
                                // exactly the UNROLLED one? That is the claim the re-roll
                                // makes, and it is checked against a second emission from
                                // the un-rolled tape rather than against itself.
                                let mut unrolled = scratchy_target_metal_compiler::from_subtile::decode_instruction_stream(
                                    &l,
                                    &tape_sm,
                                    &facts,
                                    &unrolled_items,
                                )
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "[m2-flip] {} m={m}: the UNROLLED tape refused: {e}",
                                        model.source_stem
                                    )
                                });
                                // Same MoE bit repack as `bridged` above — this is a
                                // SEPARATE decode of the un-rolled tape, so it needs its
                                // own repack or the "NO cut rolls" fallback below hands
                                // the final lowering un-packed `bits` (OptiQ models only).
                                unrolled.backbone = crate::interpreter_codegen::repack_moe_expert_bits(
                                    unrolled.backbone,
                                    program,
                                    model,
                                );
                                unrolled.lm_head = crate::interpreter_codegen::repack_moe_expert_bits(
                                    unrolled.lm_head,
                                    program,
                                    model,
                                );
                                // ⭐ THE ROLL IS KEPT ONLY IF IT IS PROVABLY THE SAME
                                // PROGRAM, AND THE CUT IS MOVED UNTIL IT IS.
                                //
                                // The rolled emission must expand — instructions AND
                                // barrier flags — to the emission from the un-rolled tape.
                                // Where it does not, the body was cut across a fold: metal
                                // merges a layer's residual `Add` with the NEXT layer's
                                // norm into one `FusedAddRmsNorm`, so a cut between layers
                                // splits that dispatch. The run is periodic, so any offset
                                // is a legal body — rotate the cut and re-check.
                                //
                                // ⛔ NOT AN ARCH LIST. The PROOF decides, per model, per
                                // bucket. granite / gemma-3 / micro-g3.3 are the ones that
                                // need a rotation today; nothing names them.
                                let unrolled_barriers = {
                                    let mut sigs = unrolled.backbone_sigs.clone();
                                    sigs.extend(unrolled.lm_head_sigs.clone());
                                    let flags =
                                        scratchy_target_metal_compiler::from_subtile::barrier_flags_with(
                                            &sigs, false,
                                        );
                                    flags[..unrolled.backbone_sigs.len()].to_vec()
                                };
                                let proof =
                                    |bucket: &crate::interpreter_codegen::LoweredBucket| {
                                        crate::interpreter_codegen::verify_loop_compression(
                                        &unrolled.backbone,
                                        &bucket.instances,
                                        &arch_opcodes,
                                        "layer",
                                    )
                                    .and_then(|()| {
                                        crate::interpreter_codegen::verify_loop_compression_barriers(
                                            &unrolled_barriers,
                                            &bucket.instances,
                                            &bucket.barriers,
                                        )
                                    })
                                    };
                                let mut rolled_at: Option<u32> = None;
                                // ⛔ ONE ERROR PER CUT, NOT JUST CUT 0's. The refusal used to
                                // report `proof(&bridge_bucket).err()` — always peel 0's — so
                                // every deeper cut failed invisibly. gemma-3 was read for a
                                // whole PR as a `FusedAddRmsNormWithOffset` fold problem on that
                                // evidence, when peel 0 was the ONLY cut failing that way and
                                // peels 1..3 were failing on a rope flag the log never showed.
                                let mut peel_errs: Vec<(u32, String)> = Vec::new();
                                if proof(&bridge_bucket).is_ok() {
                                    rolled_at = Some(0);
                                } else {
                                    // PEEL leading iterations into the prologue. The body's
                                    // source ops decide how metal folds them, and layer 0's
                                    // norm has no residual add to fuse with — so a body
                                    // drawn from layer 0 emits an unfused norm for every
                                    // layer. Peeling makes the body a representative middle
                                    // layer. granite and micro-g3.3 need peel=1.
                                    // ⭐ SPLIT FIRST, THEN WHOLE-CELL. Splitting a cell into its
                                    // per-layer class runs is the difference between one body per
                                    // LAYER and one per CLASS — gemma-3 goes 229 rows to 119. It
                                    // is not universally legal (gemma-4-31b's arena is not
                                    // periodic at one layer), so both are offered to the PROOF and
                                    // whichever holds is kept.
                                    for (peel, split) in
                                        (1..4u32).flat_map(|p| [(p, true), (p, false)])
                                    {
                                        let Some(rot) =
                                            scratchy_target_metal::from_tape::roll_at(
                                                &unrolled_items,
                                                &tape_items,
                                                peel,
                                                &layer_segs,
                                                split,
                                            )
                                        else {
                                            continue;
                                        };
                                        let k = peel;
                                        let Ok(b) = scratchy_target_metal_compiler::from_subtile::decode_instruction_stream(
                                            &l, &tape_sm, &facts, &rot,
                                        ) else {
                                            continue;
                                        };
                                        let head = b.backbone_sigs.len();
                                        let mut sigs = b.backbone_sigs.clone();
                                        sigs.extend(b.lm_head_sigs.clone());
                                        let flags =
                                            scratchy_target_metal_compiler::from_subtile::barrier_flags_with(
                                                &sigs, false,
                                            );
                                        let cand = crate::interpreter_codegen::LoweredBucket {
                                            instances: b.backbone.clone(),
                                            weight_slots: b.backbone_weight_slots.clone(),
                                            num_slots: tape_ns,
                                            final_slot: bb_final,
                                            barriers: flags[..head].to_vec(),
                                        };
                                        match proof(&cand) {
                                            Ok(()) => {
                                                bridge_bucket = cand;
                                                rolled_at = Some(k);
                                                break;
                                            }
                                            Err(e) => peel_errs
                                                .push((k, format!("split={split}: {e}"))),
                                        }
                                    }
                                }
                                match rolled_at {
                                    Some(peel) => eprintln!(
                                        "[m2-roll] {} m={m}: rolled peel={peel} ({} rows \
                                         from {})",
                                        model.source_stem,
                                        bridge_bucket.instances.len(),
                                        unrolled.backbone.len(),
                                    ),
                                    None => {
                                        let mut why = vec![format!(
                                            "peel=0: {}",
                                            proof(&bridge_bucket)
                                                .err()
                                                .unwrap_or_else(|| "?".into())
                                        )];
                                        why.extend(
                                            peel_errs.iter().map(|(k, e)| format!("peel={k}: {e}")),
                                        );
                                        eprintln!(
                                            "[m2-roll] {} m={m}: NO cut rolls — emitting the \
                                             un-rolled tape. {}",
                                            model.source_stem,
                                            why.join(" | "),
                                        );
                                        bridge_bucket.instances = unrolled.backbone.clone();
                                        bridge_bucket.weight_slots =
                                            unrolled.backbone_weight_slots.clone();
                                        bridge_bucket.barriers = unrolled_barriers.clone();
                                    }
                                }
                                let lm_barriers: Vec<bool> = all_flags[head_start..].to_vec();
                                eprintln!(
                                    "[m2-flip] {} m={m}: TAPE-SCHEDULED stream ACTIVE \
                                     ({} instr, slots={tape_ns} final={tape_fs})",
                                    model.source_stem,
                                    bridge_bucket.instances.len(),
                                );
                                // M2b: every accessor field the tape
                                // implies must exist in what
                                // instruction selection's
                                // `required_weights` walk produces —
                                // the gate that lets the Weights struct
                                // (and with it the solve) come off the
                                // solver.
                                {
                                    let tape_names =
                                        scratchy_target_metal_compiler::from_subtile::tape_accessor_names(
                                            &acc_slots,
                                        );
                                    // Instruction-selection accessors are
                                    // per-(weight, unroll index) —
                                    // `input_layernorm_0..N`; the tape's
                                    // slot bases are layer-free (the
                                    // layer is a runtime arg of
                                    // `<kind>_at`). Compare on the
                                    // FAMILY: strip the trailing
                                    // numeral, which is exactly what
                                    // `split_base_layer` does when the
                                    // bridge derives a base.
                                    let solve_ran = !sfufs
                                        .per_workload
                                        .values()
                                        .all(|a| a.impls.is_empty());
                                    let isel_names: std::collections::BTreeSet<String> =
                                        match collect_accessors(program, fuf, sfufs, lib, model)
                                        {
                                            Ok(accs) => accs
                                                .iter()
                                                .map(|a| {
                                                    crate::codegen::split_base_layer(
                                                        &a.name.to_string(),
                                                    )
                                                    .0
                                                })
                                                .collect(),
                                            Err(_) => Default::default(),
                                        };
                                    // The rotary caches are NOT weight
                                    // inputs in the FUF — instruction
                                    // selection carries them as a
                                    // side-channel flag into the Weights
                                    // struct (`wa_uses_rotary`), while
                                    // the tape declares them uniformly
                                    // as `cos_sin_at` slots. Union them
                                    // in so the comparison is over the
                                    // same universe.
                                    let mut isel_names = isel_names;
                                    if fuf_uses_rotary(fuf) {
                                        isel_names.insert("rotary".to_string());
                                    }
                                    // `rotary_local` exists only for the
                                    // dual-theta classes (a RotaryLocal
                                    // extern on some layer's rope) —
                                    // the same predicate the Weights
                                    // struct emission uses.
                                    if fuf.nodes.iter().any(|n| {
                                        n.inputs.iter().any(|i| {
                                            matches!(
                                                i,
                                                crate::fuf::FufInput::Extern {
                                                    kind:
                                                        crate::classified::ExternKind::RotaryLocal,
                                                    ..
                                                }
                                            )
                                        })
                                    }) {
                                        isel_names.insert("rotary_local".to_string());
                                    }
                                    let missing: Vec<_> = if solve_ran {
                                        tape_names.difference(&isel_names).collect()
                                    } else {
                                        Vec::new()
                                    };
                                    assert!(
                                        missing.is_empty(),
                                        "[m2-acc] {} m={m}: tape implies accessor families \
                                         the instruction-selection set lacks: {missing:?} \
                                         (isel families: {:?})",
                                        model.source_stem,
                                        isel_names,
                                    );
                                    let isel_only: Vec<_> = if solve_ran {
                                        isel_names.difference(&tape_names).collect()
                                    } else {
                                        Vec::new()
                                    };
                                    assert!(
                                        isel_only.is_empty(),
                                        "[m2-acc] {} m={m}: instruction selection declares \
                                         accessor families the tape does not: {isel_only:?}",
                                        model.source_stem,
                                    );
                                    // Group signature: base → (type,
                                    // layer coverage). This is what the
                                    // Weights STRUCT is built from
                                    // (Unindexed / LayeredContiguous /
                                    // LayeredSparse + field type), so
                                    // agreement here is the property the
                                    // tape needs to emit the struct.
                                    let layer_idx: std::collections::HashMap<String, usize> =
                                        arch_opcodes
                                            .iter()
                                            .filter_map(|(name, shape)| {
                                                shape
                                                    .fields
                                                    .iter()
                                                    .position(|(f, _)| f == "layer")
                                                    .map(|i| (name.clone(), i))
                                            })
                                            .collect();
                                    let layer_of =
                                        |ins: &scratchy_forward_compiler::Instruction| {
                                            let v = crate::interpreter_codegen::
                                            instruction_variant_name(ins);
                                            layer_idx.get(v).and_then(|&i| {
                                            crate::interpreter_codegen::instruction_field_at(
                                                ins, i,
                                            )
                                        })
                                        };
                                    let tape_groups =
                                        scratchy_target_metal_compiler::from_subtile::tape_accessor_groups(
                                            &acc_instances,
                                            &acc_slots,
                                            &layer_of,
                                        );
                                    type GroupSig =
                                        (String, std::collections::BTreeSet<Option<u64>>);
                                    let isel_groups: std::collections::BTreeMap<
                                        String,
                                        GroupSig,
                                    > = match collect_accessors(program, fuf, sfufs, lib, model)
                                    {
                                        Ok(accs) => group_accessors_by_base(&accs)
                                            .iter()
                                            .map(|g| {
                                                let layers = g
                                                    .entries
                                                    .iter()
                                                    .map(|(idx, _)| idx.map(|u| u.0))
                                                    .collect();
                                                (
                                                    g.base.clone(),
                                                    (
                                                        g.rust_type
                                                            .to_string()
                                                            .replace(' ', "")
                                                            .trim_start_matches('&')
                                                            .to_string(),
                                                        layers,
                                                    ),
                                                )
                                            })
                                            .collect(),
                                        Err(_) => Default::default(),
                                    };
                                    // Whether a family is LAYERED is the
                                    // binding's business, not the
                                    // instruction's: an unindexed weight
                                    // (embed, final norm, lm_head) still
                                    // rides an instruction whose `layer`
                                    // field reads 0. The tape carries the
                                    // authoritative fact —
                                    // `SourceBinding::Weight { index }` —
                                    // so families bound with no unroll
                                    // index normalize to the unindexed
                                    // signature.
                                    let unindexed_bases: std::collections::BTreeSet<String> = l
                                        .bindings
                                        .iter()
                                        .filter_map(|b| match b {
                                            crate::to_wavefront::SourceBinding::Weight {
                                                id,
                                                index: None,
                                            } => Some(
                                                split_base_layer(
                                                    &crate::emit::weight_field_name(
                                                        program,
                                                        crate::classified::WeightId(*id),
                                                        None,
                                                    )
                                                    .to_string(),
                                                )
                                                .0,
                                            ),
                                            _ => None,
                                        })
                                        .collect();
                                    // `acc_instances` is the UNCOMPRESSED
                                    // stream (captured before
                                    // `apply_loop_compression`), so its
                                    // layer values are the full 0..N
                                    // coverage — directly comparable to
                                    // the isel group's entries.
                                    for (base, (ty, layers)) in
                                        tape_groups.iter().filter(|_| solve_ran)
                                    {
                                        if let Some((isel_ty, isel_layers)) =
                                            isel_groups.get(base)
                                        {
                                            assert_eq!(
                                                ty, isel_ty,
                                                "[m2-acc] {} m={m}: accessor `{base}` type \
                                                 disagrees with instruction selection",
                                                model.source_stem,
                                            );
                                            let layers = if unindexed_bases.contains(base) {
                                                std::iter::once(None).collect()
                                            } else {
                                                layers.clone()
                                            };
                                            assert_eq!(
                                                &layers, isel_layers,
                                                "[m2-acc] {} m={m}: accessor `{base}` layer \
                                                 coverage disagrees with instruction \
                                                 selection",
                                                model.source_stem,
                                            );
                                        }
                                    }
                                    // ⭐ ONE PRODUCER. This was ~75 lines building the same `Vec<WeightAccessor>` from
                                    // `tape_groups` — a second derivation of a fact that is already a property of the
                                    // MODEL (`LoweredDecode.bindings` plus the op consuming each one), not of the target.
                                    // The one thing it knew that the shared walk did not is metal's kernel ABI: a gate/up
                                    // pair is ONE packed `__fused__` buffer here and two weights on spyre. That is a bool.
                                    //
                                    // ⛔ VERIFIED EQUAL BEFORE THE SWAP, not after: a probe built both sets under metal and
                                    // compared name + type + SOURCES, and they agreed on all 131 accessors for
                                    // llama-3.2-1b. The emitted metal code is byte-identical to the previous build.
                                    let synth = crate::weight_bindings::from_tape(
                                        &l,
                                        program,
                                        crate::weight_bindings::WeightAbi {
                                            mlp: if facts.fused_mlp {
                                                crate::weight_bindings::MlpPacking::Packed
                                            } else {
                                                crate::weight_bindings::MlpPacking::Split
                                            },
                                            embed: if facts.embed_quant.is_some() {
                                                crate::weight_bindings::EmbedStorage::AffineQuant
                                            } else {
                                                crate::weight_bindings::EmbedStorage::Dense
                                            },
                                        },
                                    )
                                        .unwrap_or_else(|e| panic!("[weight-bindings] {}: {e}", model.source_stem))
                                        .accessors;
                                    tape_accessors_for_struct = Some(synth.clone());
                                    let synth_sig: Vec<(String, String, usize)> =
                                        group_accessors_by_base(&synth)
                                            .iter()
                                            .map(|g| {
                                                (
                                                    g.base.clone(),
                                                    g.rust_type.to_string().replace(' ', ""),
                                                    g.entries.len(),
                                                )
                                            })
                                            .collect();
                                    let isel_sig: Vec<(String, String, usize)> =
                                        match collect_accessors(program, fuf, sfufs, lib, model)
                                        {
                                            Ok(accs) => group_accessors_by_base(&accs)
                                                .iter()
                                                .map(|g| {
                                                    (
                                                        g.base.clone(),
                                                        g.rust_type
                                                            .to_string()
                                                            .replace(' ', ""),
                                                        g.entries.len(),
                                                    )
                                                })
                                                .collect(),
                                            Err(_) => Vec::new(),
                                        };
                                    assert_eq!(
                                        synth_sig,
                                        if solve_ran {
                                            isel_sig
                                        } else {
                                            synth_sig.clone()
                                        },
                                        "[m2-acc] {} m={m}: tape-synthesized accessor GROUPS \
                                         differ from instruction selection's",
                                        model.source_stem,
                                    );
                                    // …and the SOURCE WEIGHTS per
                                    // accessor, which the
                                    // storage-format guard and the
                                    // per-field load consume. Rotary is
                                    // excluded on both sides (side
                                    // channel, no accessor).
                                    let srcs =
                                        |accs: &[crate::weight_vocab::WeightAccessor]| {
                                            accs.iter()
                                                .map(|a| {
                                                    let mut w: Vec<String> = a
                                                        .source_weights
                                                        .iter()
                                                        .map(|(id, ix)| {
                                                            format!(
                                                                "{}#{:?}",
                                                                id.0,
                                                                ix.map(|u| u.0)
                                                            )
                                                        })
                                                        .collect();
                                                    w.sort();
                                                    (a.name.to_string(), w)
                                                })
                                                .collect::<std::collections::BTreeMap<_, _>>()
                                        };
                                    let synth_srcs = srcs(&synth);
                                    let isel_srcs = match collect_accessors(
                                        program, fuf, sfufs, lib, model,
                                    ) {
                                        Ok(accs) => srcs(&accs),
                                        Err(_) => Default::default(),
                                    };
                                    for (name, w) in synth_srcs.iter().filter(|_| solve_ran) {
                                        if let Some(isel_w) = isel_srcs.get(name) {
                                            assert_eq!(
                                                w, isel_w,
                                                "[m2-acc] {} m={m}: accessor `{name}` source \
                                                 weights differ from instruction selection's",
                                                model.source_stem,
                                            );
                                        }
                                    }
                                    eprintln!(
                                        "[m2-acc] {} m={m}: accessor families EQUAL ({}), \
                                         {} groups typed-equal",
                                        model.source_stem,
                                        tape_names.len(),
                                        tape_groups.len(),
                                    );
                                    eprintln!(
                                        "[m2-acc] {} m={m}: tape fields {} of isel {}",
                                        model.source_stem,
                                        tape_names.len(),
                                        isel_names.len(),
                                    );
                                }
                                cl.backbone = bridge_bucket;
                                cl.lm_head.instances = bridged.lm_head;
                                cl.lm_head.weight_slots = bridged.lm_head_weight_slots;
                                cl.lm_head.barriers = lm_barriers;
                                cl.lm_head.num_slots = tape_ns;
                                cl.lm_head.final_slot = lm_final;
                                // The bucket-table + arena-sizing reads
                                // go through the canonical tuple, not
                                // `cl` — flip them too, or the runtime
                                // arena is sized for the SOLVER coloring
                                // (ArenaSlotOutOfRange on device).
                                *ns_field = tape_ns;
                                *bb_slot_field = tape_fs;
                                *term_field = lm_final;
                                let mut tape_sm = tape_sm;
                                // Arena sizing iterates the SlotMap and
                                // evaluates fuf shapes per color; color
                                // 0 (the embedded hidden) must appear —
                                // key it by the fuf Embed tile.
                                if let Some(et) = fuf
                                    .nodes
                                    .iter()
                                    .find(|nd| nd.op == crate::classified::OpKind::Embed)
                                {
                                    tape_sm.insert_at(et.id, 0, 0);
                                }
                                *slots_dec = tape_sm;
                            }
                            Err(e) => panic!(
                                "[m2-flip] {} m={m}: bridge refused a DECLARED pilot: {e}",
                                model.source_stem
                            ),
                        }
                    }
                    Err(e) => {
                        // A refusal that means METAL HAS NO KERNEL is not a
                        // front-end gap: the runtime-lowering path refused
                        // these too, at load, and the bake below already
                        // bakes an empty variant list for them. Panicking
                        // would make a dense-MoE preset abort the whole
                        // build for an arch whose QUANTIZED presets lower
                        // fine (mixtral).
                        assert!(
                            !tape_pilot_arch || e.is_no_metal_realization(),
                            "[m2-flip] {} m={m}: the shared tape refused a DECLARED pilot \
                             ({e:?}) — with the solve off there is no fallback",
                            model.source_stem,
                        );
                        eprintln!("[m2-eq] {} m={m} wavefront REFUSED: {e}", model.source_stem)
                    }
                }
            }
        }
    }
    // Emitted AFTER the front-end swap: the `(bucket, op_idx, slot)`
    // accessor table must be built from the SAME (possibly
    // tape-scheduled) streams the baked tapes execute — a pre-swap
    // snapshot only agreed by accident of the byte-equality era.
    // Spans rope-on-read: detect the arch's rotary caches (same predicates
    // as the Weights-struct emission) so `rope_on_read_cos_sin` resolves
    // the right cos/sin by layer class.
    let wa_uses_rotary = fuf_uses_rotary(fuf);
    let wa_uses_rotary_local = fuf.nodes.iter().any(|n| {
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
    let weight_accessors_impl = emit_weight_accessors_impl(
        &canonical_lowered,
        model.source_stem.as_str(),
        wa_uses_rotary,
        wa_uses_rotary_local,
    );
    // M2b: for a tape-scheduled arch the Weights struct is emitted
    // from the TAPE's accessor set (asserted to produce the same
    // struct as instruction selection's inside `emit_weights_struct`).
    // Re-emitted HERE because the tape accessors only exist after the
    // front-end swap has lowered the canonical.
    #[cfg(feature = "metal")]
    if let Some(tape_accs) = tape_accessors_for_struct.as_ref() {
        weights = emit_struct(Some(tape_accs));
    }

    if cfg!(feature = "spyre") {
        match sfufs.get_nt(1) {
            Some(decode_asn) => {
                // The canonical the decode (num_tokens=1) point folded into,
                // and its `2*ci` backbone bucket id.
                let decode_wp = bucket_points.iter().find(|wp| wp.num_tokens == 1).copied();
                match decode_wp {
                    Some(decode_wp) => {
                        let idx = bucket_points.iter().position(|w| *w == decode_wp).unwrap();
                        let canonical = bucket_canonical[idx];
                        let bb_bucket_id = canonical_to_bucket_id[&canonical];
                        let (cl, ..) = &canonical_lowered[&canonical];
                        let decode_bounds = bounds_for_wp(model, decode_wp, tp_world_size);
                        dump_wavefront_mega(
                            program,
                            model,
                            fuf,
                            decode_asn,
                            inferred,
                            &decode_bounds,
                            &cl.backbone.weight_slots,
                            &cl.lm_head.weight_slots,
                            bb_bucket_id,
                            &mut ktir_bundle_const,
                        );
                    }
                    None => eprintln!(
                        "[wavefront] {}: no decode (num_tokens=1) workload point",
                        model.source_stem
                    ),
                }
            }
            None => eprintln!(
                "[wavefront] {}: no decode (num_tokens=1) workload point",
                model.source_stem
            ),
        }
    }

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
    // single line in expanded source. Available under either backend
    // — `scr model info`'s per-variant `dump()` walks it on both
    // cuda and metal. The cuda runtime additionally drives `forward`
    // / `forward_backbone` from it; metal's pool uses `METAL_BUCKETS`
    // instead and ignores the runtime fields here.
    let forward_table = quote! {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        static FORWARD_TABLE: &[::scratchy_forward_compiler::BucketEntry<__I>] = {
            use ::scratchy_forward_compiler::BucketEntry as __B;
            &[
                #(#bucket_table_entries)*
            ]
        };
    };

    // METAL_BUCKETS — one row per distinct `num_tokens` point. The
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
    let mut metal_arena_bytes_statics: Vec<TokenStream> = Vec::new();
    // `(bucket_m, total_colored_arena_bytes)` per bucket — the per-bucket cost
    // the load-time `select_prefill_bucket` compares against the device's
    // affordable arena budget to prune the ladder target-reactively.
    let mut metal_bucket_cost_entries: Vec<TokenStream> = Vec::new();
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
        let (_, num_slots_b, _, terminal_slot_b, slots_b) = &canonical_lowered[&canonical];
        let bucket_m_lit = proc_macro2::Literal::u32_unsuffixed(m as u32);
        let num_slots_lit = proc_macro2::Literal::u32_unsuffixed(*num_slots_b);
        let terminal_slot_lit = proc_macro2::Literal::u32_unsuffixed(*terminal_slot_b);
        // tape-index ids matching the keys `emit_weight_accessors_impl`
        // bakes into the per-arch `WeightAccessors` match arms:
        // `ci*2` for backbone, `ci*2 + 1` for lm_head, where `ci` is
        // the canonical's position in `canonical_lowered.iter()`.
        let ci = canonical_lowered
            .keys()
            .position(|wp| *wp == canonical)
            .expect("canonical present in canonical_lowered");
        let backbone_tape_index_lit = proc_macro2::Literal::u32_unsuffixed((ci as u32) * 2);
        let lm_head_tape_index_lit = proc_macro2::Literal::u32_unsuffixed((ci as u32) * 2 + 1);

        // Per-tape_index arena_bytes: register-coloring tells us which
        // (tile, output_slot) pairs share an arena slot; for THIS
        // tape_index M we evaluate each tile's output shape against this
        // tape_index's bounds and take the per-color max. Each tape_index
        // gets a tight arena sized exactly for its own num_tokens —
        // the worker pool elementwise-maxes across MetalBucketSpec
        // entries once at init time to size the per-worker arena
        // (init-time alloc, reused across every forward).
        let bp_bounds = bounds_for_wp(model, wp, tp_world_size);
        let mut bucket_arena_bytes: Vec<u64> = vec![0u64; *num_slots_b as usize];
        for ((tile_id, out_slot), color) in slots_b.iter() {
            let node = fuf.get(tile_id);
            let shape = &node.outputs[out_slot as usize];
            let elems: u64 = crate::weight_vocab::eval_shape_with(shape, &bp_bounds)
                .expect("shape inference left a Var in a tile output — codegen invariant")
                .into_iter()
                .product();
            // f16 = 2 bytes/element; metal kernels are f16-only.
            let bytes = elems.saturating_mul(2);
            let slot_idx = color as usize;
            if bytes > bucket_arena_bytes[slot_idx] {
                bucket_arena_bytes[slot_idx] = bytes;
            }
        }
        // Every color must have at least one (tile, slot) pair; if
        // the worker ever sees a 0-byte arena slot it'll fail the
        // ICB residency. Guarantee a 1-byte minimum so unused colors
        // (none expected, but defensive) still allocate a valid
        // `MTLBuffer`.
        for b in &mut bucket_arena_bytes {
            if *b == 0 {
                *b = 1;
            }
        }

        // Per-bucket total arena cost (sum over colored slots) for the
        // load-time selector. Monotonic in `m`; consumed by
        // `select_prefill_bucket` via `metal_bucket_arena_costs()`.
        let bucket_total_arena_lit =
            proc_macro2::Literal::u64_unsuffixed(bucket_arena_bytes.iter().sum::<u64>());
        metal_bucket_cost_entries.push(quote! {
            (#bucket_m_lit, #bucket_total_arena_lit),
        });

        let arena_static_ident = bucket_static_ident("METAL_ARENA_BYTES_M", wp);
        let arena_bytes_lits = bucket_arena_bytes
            .iter()
            .map(|b| proc_macro2::Literal::u64_unsuffixed(*b));
        metal_arena_bytes_statics.push(quote! {
            #[cfg(feature = "metal")]
            static #arena_static_ident: &[u64] = &[ #(#arena_bytes_lits),* ];
        });
        // Bake this bucket's tape variants at expansion — the SAME
        // lower_pair the pool used to run at load, now run here, with
        // the two runtime inputs (device generation, block capacity)
        // handled as variants + patches. See metal_static_tape.rs.
        #[cfg(feature = "metal")]
        let tapes_static_toks = {
            let (cl_b, _, _, _, _) = &canonical_lowered[&canonical];
            let mc = resolved_metal_consts
                .as_ref()
                .expect("emit_canonical_params_impl fills metal consts under -Fmetal");
            let input = scratchy_target_metal_compiler::static_tape::BucketLowerInput {
                backbone: &cl_b.backbone.instances,
                lm_head: &cl_b.lm_head.instances,
                backbone_barriers: &cl_b.backbone.barriers,
                lm_head_barriers: &cl_b.lm_head.barriers,
                bucket_m: m as u32,
                num_arena_slots: *num_slots_b,
                backbone_tape_index: (ci as u32) * 2,
                lm_head_tape_index: (ci as u32) * 2 + 1,
            };
            let tapes_expr =
                match scratchy_target_metal_compiler::static_tape::bake_bucket_tapes(mc, &input) {
                    Ok(t) => t,
                    Err(scratchy_target_metal_compiler::static_tape::BakeRefusal::Unlowerable(
                        e,
                    )) => {
                        // Same timing as the old runtime-lowering path: this
                        // preset builds, and the POOL refuses at load if a
                        // metal run ever selects this canonical.
                        eprintln!(
                            "[metal static tape] {}: bucket_m={m}: NOT LOWERABLE on metal \
                         ({e}); baking an empty variant list — the pool refuses at load",
                            model.source_stem
                        );
                        quote! { &[] }
                    }
                    Err(scratchy_target_metal_compiler::static_tape::BakeRefusal::Defect(e)) => {
                        panic!(
                            "[metal static tape] {}: bucket_m={m}: {e}",
                            model.source_stem
                        )
                    }
                };
            let ident = bucket_static_ident("METAL_TAPES_M", wp);
            metal_arena_bytes_statics.push(quote! {
                #[cfg(feature = "metal")]
                static #ident:
                    &[::scratchy_target_metal::tape::lowered::ClassedTape] = #tapes_expr;
            });
            quote! { #ident }
        };
        #[cfg(not(feature = "metal"))]
        let tapes_static_toks = quote! { &[] };
        metal_bucket_entries.push(quote! {
            ::scratchy_target_metal::interpreter::metal::MetalBucketSpec {
                bucket_m: #bucket_m_lit,
                backbone_tape_index: #backbone_tape_index_lit,
                lm_head_tape_index: #lm_head_tape_index_lit,
                num_arena_slots: #num_slots_lit,
                terminal_slot: #terminal_slot_lit,
                arena_bytes: #arena_static_ident,
                backbone: #bb_static,
                lm_head: #lm_static,
                backbone_barriers: #bb_barriers_static,
                lm_head_barriers: #lh_barriers_static,
                tapes: #tapes_static_toks,
            },
        });
    }

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
    let (
        vision_freqs_bytes_lit,
        vision_pixels_bytes_lit,
        vision_posemb_bytes_lit,
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
            let head_dim = *model.bounds.get("head_dim").unwrap_or(&0);
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
            proc_macro2::Literal::u64_unsuffixed(mm_embeds_bytes),
            proc_macro2::Literal::u64_unsuffixed(mm_dst_rows_bytes),
            proc_macro2::Literal::u64_unsuffixed(mrope_cos_sin_bytes),
        )
    };

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
    let vocab_size_lit = {
        let width = match layout {
            BackboneLayout::Decoder { .. } => model.bounds.get("vocab_size").copied().unwrap_or(0),
            BackboneLayout::Encoder => model.bounds.get("hidden_size").copied().unwrap_or(0),
        };
        proc_macro2::Literal::u64_unsuffixed(width)
    };

    // Per-worker arena peak in bytes — sum across every slot of the
    // worker's arena layout, where each slot is sized to fit the
    // largest tape_index's claim on that color. The pool computes this
    // exact layout at runtime via elementwise-max across each
    // `MetalBucketSpec.arena_bytes`; we mirror that calculation here
    // at compile time so the metal worker can pre-declare its
    // resident-arena bytes via the ScratchyWeights trait (used to
    // size the `peak_activation_bytes` argument to
    // `compute_available_kv_bytes`).
    let metal_arena_peak_bytes_lit = {
        let mut layout: Vec<u64> = Vec::new();
        for &m in &num_tokens_points {
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
            let (_, num_slots_b, _, _, slots_b) = &canonical_lowered[&canonical];
            if layout.len() < *num_slots_b as usize {
                layout.resize(*num_slots_b as usize, 0);
            }
            let bp_bounds = bounds_for_wp(model, wp, tp_world_size);
            for ((tile_id, out_slot), color) in slots_b.iter() {
                let node = fuf.get(tile_id);
                let shape = &node.outputs[out_slot as usize];
                let elems: u64 = crate::weight_vocab::eval_shape_with(shape, &bp_bounds)
                    .expect("shape inference left a Var in a tile output")
                    .into_iter()
                    .product();
                let bytes = elems.saturating_mul(2);
                let slot_idx = color as usize;
                if bytes > layout[slot_idx] {
                    layout[slot_idx] = bytes;
                }
            }
        }
        let total: u64 = layout.iter().sum();
        proc_macro2::Literal::u64_unsuffixed(total)
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
    let mrope_runtime_block: proc_macro2::TokenStream = match model.mrope_section {
        Some([t, h, w]) => {
            let head_dim = *model.bounds.get("head_dim").unwrap_or(&0);
            let rot_dim_val = model
                .scalars
                .get("partial_rotary_factor")
                .copied()
                .filter(|&f| (f - 1.0).abs() > 1e-9)
                .map(|f| (f * head_dim as f64).round() as u32)
                .unwrap_or(head_dim as u32);
            let rope_theta_val: f64 = model
                .scalars
                .get("rope_theta")
                .copied()
                .or_else(|| model.bounds.get("rope_theta").map(|&v| v as f64))
                .unwrap_or(10000.0);
            let rope_theta_lit = proc_macro2::Literal::f64_unsuffixed(rope_theta_val);
            let t_lit = proc_macro2::Literal::u32_unsuffixed(t);
            let h_lit = proc_macro2::Literal::u32_unsuffixed(h);
            let w_lit = proc_macro2::Literal::u32_unsuffixed(w);
            let rot_dim_lit2 = proc_macro2::Literal::u32_unsuffixed(rot_dim_val);
            quote! {
                let mut __mrope_table_vec: ::std::vec::Vec<u8> = ::std::vec::Vec::new();
                let mut __mrope_ident_vec: ::std::vec::Vec<u32> = ::std::vec::Vec::new();
                let (positions, mrope_cos_sin): (&[u32], ::core::option::Option<&[u8]>) =
                    if !ctx.positions.as_raw().raw_ptr().is_null() {
                        let __pos_numel = ctx.positions.as_raw().numel();
                        let __pos_all = ::std::slice::from_raw_parts(
                            ctx.positions.as_raw().raw_ptr() as *const u32,
                            __pos_numel,
                        );
                        __mrope_table_vec =
                            ::scratchy_target_metal::interpreter::metal::build_mrope_cos_sin_override(
                                __pos_all,
                                n,
                                #rot_dim_lit2 as usize,
                                #rope_theta_lit,
                                [#t_lit, #h_lit, #w_lit],
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::METAL_DTYPE,
                            );
                        __mrope_ident_vec.extend(0..n as u32);
                        (
                            __mrope_ident_vec.as_slice(),
                            ::core::option::Option::Some(__mrope_table_vec.as_slice()),
                        )
                    } else {
                        (positions, ::core::option::Option::None)
                    };
            }
        }
        None => quote! {
            let mrope_cos_sin: ::core::option::Option<&[u8]> = ::core::option::Option::None;
        },
    };

    let metal_emission = quote! {
        #(#metal_arena_bytes_statics)*

        /// Per-canonical tape_index plan for the Metal pool. One row per
        /// `num_tokens` point, ordered ascending. `MetalWorkerPool::pick_bucket`
        /// is a linear smallest-fit scan, so order matters.
        #[cfg(feature = "metal")]
        pub static METAL_BUCKETS:
            &[::scratchy_target_metal::interpreter::metal::MetalBucketSpec]
            = &[
                #(#metal_bucket_entries)*
            ];

        /// Largest `num_tokens` tape_index across [`METAL_BUCKETS`]. The
        /// metal forward body's `RuntimeFactory` allocates per-worker
        /// runtime buffers (input_ids/positions/slot_mapping/...) at
        /// `METAL_MAX_BUCKET_M * sizeof(u32)`; block_table at
        /// `METAL_MAX_BUCKET_M * MAX_BLOCKS_PER_SEQ * sizeof(u32)`.
        #[cfg(feature = "metal")]
        pub const METAL_MAX_BUCKET_M: u32 = #max_bucket_m_lit;

        /// Byte size of the vision 2D-RoPE `freqs` runtime buffer
        /// (f32 `[max_bucket_m, vision_head_dim/2]`). 0 on non-vision
        /// arches (`alloc` floors to 16 bytes).
        #[cfg(feature = "metal")]
        pub const METAL_VISION_FREQS_BYTES: u64 = #vision_freqs_bytes_lit;

        /// Byte size of the vision `pixels` runtime buffer (model dtype
        /// `[max_bucket_m, vision_in_features]`). 0 on non-vision arches.
        #[cfg(feature = "metal")]
        pub const METAL_VISION_PIXELS_BYTES: u64 = #vision_pixels_bytes_lit;

        /// Byte size of the vision `pos_embeds` runtime buffer (model
        /// dtype `[max_bucket_m, vision_embed_dim]`). 0 on non-vision
        /// arches and on towers without a learned positional embedding.
        #[cfg(feature = "metal")]
        pub const METAL_VISION_POSEMB_BYTES: u64 = #vision_posemb_bytes_lit;

        /// Byte size of the `mm_embeds` splice buffer (bf16
        /// `[max_bucket_m, hidden]`) and the `mm_dst_rows` buffer (u32
        /// `[max_bucket_m]`). 0 on arches without the multimodal splice.
        #[cfg(feature = "metal")]
        pub const METAL_MM_EMBEDS_BYTES: u64 = #mm_embeds_bytes_lit;
        #[cfg(feature = "metal")]
        pub const METAL_MM_DST_ROWS_BYTES: u64 = #mm_dst_rows_bytes_lit;

        /// Byte size of the MRoPE cos/sin override buffer (model dtype
        /// `[max_bucket_m, rot_dim]`). 0 on 1D-rope arches (the rope
        /// kernel keeps the static cos/sin cache; `alloc` floors to 16
        /// bytes and the buffer is never bound).
        #[cfg(feature = "metal")]
        pub const METAL_MROPE_COS_SIN_BYTES: u64 = #mrope_cos_sin_bytes_lit;

        /// Width of one row of the terminal arena slot: `vocab_size`
        /// on decoder layouts (the lm_head Gemm's output),
        /// `hidden_size` on encoder layouts (no lm_head — the terminal
        /// IS the hidden state). The metal forward shapes its returned
        /// `OwnedTensor` as `[num_tokens, METAL_VOCAB_SIZE]` f16.
        #[cfg(feature = "metal")]
        pub const METAL_VOCAB_SIZE: u64 = #vocab_size_lit;

        /// Per-worker arena peak in bytes for this canonical. The
        /// pool's `arena_layout` is the elementwise-max across every
        /// tape_index spec's `arena_bytes`; per-canonical that's the
        /// shared `METAL_ARENA_BYTES_M_<m>` static (every tape_index in
        /// one canonical points at the same row), so the sum equals
        /// the worker's resident-arena byte footprint. Read by
        /// `MetalWorker::determine_available_memory` to
        /// replace the 512 MiB peak-activation placeholder.
        #[cfg(feature = "metal")]
        pub const METAL_ARENA_PEAK_BYTES: u64 = #metal_arena_peak_bytes_lit;

        /// `(bucket_m, total_colored_arena_bytes)` for every compiled bucket
        /// of this canonical, ascending by `bucket_m`. The load-time
        /// `select_prefill_bucket` reads this (via
        /// `ScratchyWeights::metal_bucket_arena_costs`) to prune the global
        /// ladder to the largest bucket the device can afford while leaving a
        /// KV floor — the target-reactive replacement for a hardcoded
        /// per-model `workloads` cap.
        #[cfg(feature = "metal")]
        pub static METAL_BUCKET_ARENA_COSTS: &[(u32, u64)] =
            &[ #(#metal_bucket_cost_entries)* ];

        /// Build a [`MetalWorkerPool`] for this canonical. Thin
        /// wrapper over [`MetalWorkerPool::for_buckets`] that threads
        /// the per-canonical [`METAL_BUCKETS`] static so callers don't
        /// have to construct the tape_index plan by hand. The arena layout
        /// is derived from each tape_index's `arena_bytes` field — taking
        /// the elementwise max so the single per-worker arena fits the
        /// largest activation across every tape_index.
        ///
        /// `weights` is borrowed; the pool stores no back-reference,
        /// caller passes `&weights` again at every `forward` /
        /// `checkout` so the pool can live as a field on the
        /// `Weights` struct without an `Arc`-cycle.
        ///
        /// [`MetalWorkerPool`]: ::scratchy_target_metal::interpreter::metal::MetalWorkerPool
        /// [`MetalWorkerPool::for_buckets`]: ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets
        #[cfg(feature = "metal")]
        pub fn metal_pool(
            device: ::std::sync::Arc<
                ::scratchy_target_metal::interpreter::metal::__re::Device,
            >,
            weights: &Weights,
            allocator: ::std::sync::Arc<crate::__gpu::MetalAllocator>,
            runtime_factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory,
            max_workers: usize,
            // Runtime per-sequence block-table capacity
            // (`KvCachePool::max_blocks_per_seq`); drives the kernel's
            // `MaxBlocksPerSeq` function constant + the rope-once scratch.
            block_cap: usize,
        ) -> ::core::result::Result<
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool<Weights>,
            ::scratchy_target_metal::interpreter::metal::PoolBuildError,
        > {
            ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                device,
                weights,
                allocator,
                METAL_BUCKETS,
                runtime_factory,
                max_workers,
                // Standalone factory: no device-budget context here, so keep
                // all buckets. The lazy-init path below passes the real cap.
                None,
                block_cap,
            )
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
        #[cfg(feature = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            unsafe { forward_with_metal_followup(wm, ctx, device, num_tokens, None) }
        }

        /// Same as [`forward`] but takes an optional encoder-tail hook
        /// that's invoked on the same MTL4 compute encoder used to
        /// encode the forward, AFTER the bucket dispatches and BEFORE
        /// `endEncoding`. Lets the caller (the executor's argmax
        /// dispatch, today) append its own dispatches onto the same
        /// CB so forward + tail share one commit and one host wait.
        #[cfg(feature = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_with_metal_followup(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            followup: ::core::option::Option<::scratchy_forward_compiler::MetalForwardFollowup<'_>>,
        ) -> crate::__gpu::OwnedTensor {
            use ::scratchy_forward_compiler::CanonicalParams as _;
            use ::scratchy_target_metal::interpreter::metal::__re::{Buffer, MTLResourceOptions};

            // ── Lazy pool init ────────────────────────────────────
            //
            // Factory closure captures (a) the metal device handle
            // for runtime-buffer allocation, (b) `MAX_BLOCKS_PER_SEQ`
            // from the canonical's `CanonicalParams` (compile-time
            // const), (c) Arc-handle clones of the per-layer KV
            // buffers from `ctx.kv_cache` so worker spawns inherit
            // them without re-allocation. The closure is invoked
            // once per worker spawn — the pool starts at size 1 so
            // the first `pool.forward` call below triggers the only
            // factory invocation in single-worker configs.
            let pool = wm.metal_pool.get_or_init(|| {
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
                // TurboQuant: captured (Copy) so the 'static factory closure can
                // provision the packed stores + scratch per worker when enabled.
                // UNIFORM arches (num_kv_groups == 1): every layer is "global",
                // compressed at the base geometry. HYBRID/SWA arches (gemma4:
                // num_kv_groups > 1) compress ONLY the GLOBAL (full-context, group
                // 0) layers at the GLOBAL geometry — IF that geometry is a power
                // of two and <= 512 (the widened kernels' supported range); the
                // sliding layers stay fp16. `is_global[L]` = (group_of(L) == 0):
                // group 0 is always the full-context group (see
                // `compute_hybrid_kv_layout`); uniform → all-zero map → all-true.
                let __tq_global_ok =
                    (<Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM)
                        .is_power_of_two()
                        && <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM
                            <= 512;
                let __tq_on = ctx.kv_turboquant
                    && (ctx.kv_cache.num_kv_groups() == 1 || __tq_global_ok);
                let __tq_nb = ctx.kv_cache.num_blocks;
                let __tq_is_global: ::std::vec::Vec<bool> = ctx
                    .kv_cache
                    .layer_to_group_u32()
                    .iter()
                    .map(|&g| g == 0)
                    .collect();
                let factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory =
                    ::scratchy_target_metal::interpreter::metal::RuntimeFactory::new(move |dev| {
                        let max_m = METAL_MAX_BUCKET_M as u64;
                        let max_bps =
                            <Weights as ::scratchy_forward_compiler::CanonicalParams>::MAX_BLOCKS_PER_SEQ
                                as u64;
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
                        // GLOBAL_BLOCK_SIZE) for the group-0 layers only. The gate
                        // requires the GLOBAL head_dim power-of-two and <= 512.
                        let __tq_prov = if __tq_on
                            && (<Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM)
                                .is_power_of_two()
                            && <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM
                                <= 512
                        {
                            Some(::scratchy_target_metal::turboquant::build_tq_provision(
                                dev,
                                &__tq_is_global,
                                __tq_nb,
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_BLOCK_SIZE as usize,
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::NUM_GLOBAL_KV_HEADS as usize,
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM as usize,
                                ::scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize,
                                ::scratchy_target_metal::turboquant::tq_bits(
                                    <Weights as ::scratchy_forward_compiler::CanonicalParams>::TQ_KV_BITS,
                                ),
                                42,
                            ))
                        } else {
                            None
                        };
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
                            // Per-block span label for block-diagonal span
                            // attention: one u32 per logical block, zero-padded to
                            // the block-table stride. The kernel indexes span_ids
                            // by BLOCK (pos/block_size), not token, and the runtime
                            // stride (max_model_len-derived max_blocks_per_seq) is
                            // NOT a macro constant — so size it exactly like ONE
                            // block_tables group (max_m * max_bps), which the
                            // runtime stride always fits. (max_m alone undersized
                            // it: the span_ids BufferTooSmall bug.) All-zero unless
                            // the request carries Relocatable spans.
                            span_ids: alloc(max_m * max_bps * 4),
                            block_tables: (0..num_kv_groups)
                                .map(|_| alloc(max_m * max_bps * 4))
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
                                            p.scratch_k_table.clone()
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
                                            p.scratch_v_table.clone()
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
                            // 4 bytes — worker writes the current
                            // forward()'s `num_tokens` here before
                            // dispatch so per-call kernels see M.
                            num_tokens_u32: alloc(4),
                            // 4 bytes — worker writes `num_sample_rows`
                            // here (= last_token_indices.len()) before
                            // dispatch so the lm_head slice's gather /
                            // scatter / qmv know how many rows to act on.
                            num_sample_rows_u32: alloc(4),
                            // [max_m] u32 — worker writes the lm_head
                            // sample-row source indices here before
                            // dispatch.
                            sample_indices: alloc(max_m * 4),
                            // GDN persistent state (per-layer) + per-forward
                            // indices/fresh flags (Shared, overwritten each
                            // forward). Index buffers sized to the max bucket
                            // (num_seqs <= num_tokens <= max_m).
                            gdn_state_conv: gdn_conv.clone(),
                            gdn_state_ssm: gdn_ssm.clone(),
                            gdn_state_indices: alloc(max_m * 4),
                            gdn_is_fresh: alloc(max_m * 4),
                            // Vision externs: sized from the baked
                            // METAL_VISION_*_BYTES consts (16-byte floor
                            // on non-vision arches). Overwritten per
                            // forward by `write_runtime_inputs`.
                            vision_rope_freqs: alloc(METAL_VISION_FREQS_BYTES),
                            pixels: alloc(METAL_VISION_PIXELS_BYTES),
                            vision_pos_embeds: alloc(METAL_VISION_POSEMB_BYTES),
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
                        }
                    });
                ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                    device.device.clone(),
                    wm,
                    device.allocator.clone(),
                    METAL_BUCKETS,
                    factory,
                    1,
                    // Target-reactive cap stashed on the device by the worker
                    // after `determine_available_memory`. Prunes the compiled
                    // ladder so the colored arena fits the KV budget.
                    device.metal_bucket_max_m,
                    // Runtime per-sequence block-table capacity. Read off the KV
                    // pool so the kernel's `MaxBlocksPerSeq` function constant +
                    // rope-once scratch agree with the host block-table stride.
                    ctx.kv_cache.max_blocks_per_seq,
                )
                .expect("MetalWorkerPool::for_buckets: pool init failed")
            });

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

            // Plumb the lm_head sample-row index list from ForwardCtx.
            // `last_token_indices` is `Option<TensorView>` — present
            // when the worker built `logits_indices = query_start_loc[1:] - 1`
            // (any prefill/decode that produces a sampled token), absent
            // for chunked-prefill intermediate chunks. The metal slice
            // gathers exactly these rows then runs the lm_head qmv at
            // M = indices.len().
            let last_token_indices = ctx.last_token_indices.as_ref().map(|view| {
                let raw = view.as_raw();
                ::std::slice::from_raw_parts(raw.raw_ptr() as *const u32, raw.numel())
            });

            // GDN per-forward indices (hybrid arches): read from ctx's
            // host-visible TensorViews (slot id per seq, fresh flag per seq).
            // `None` for non-hybrid arches (ctx fields are None).
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
                has_spec_tokens: ctx.has_spec_tokens,
                last_token_indices,
                gdn_state_indices,
                gdn_is_fresh,
                vision_rope_freqs,
                pixels,
                pos_embeds,
                vision_cu_seqlens_full,
                vision_cu_seqlens_window,
                vision_window_index,
                vision_reverse_indices,
                vision_position_ids,
                mm_embeds,
                mm_dst_rows,
                mrope_cos_sin,
            };

            // ── Run forward + copy logits out ─────────────────────
            //
            // The pool picks the tape_index from `num_tokens`, runs
            // `executeCommandsInBuffer` (interleaved with MPS GEMMs
            // on <M4 hardware), waits for completion, then invokes
            // the closure with `&MetalWorker` + `bucket_idx`. The
            // tape_index's terminal arena slot holds the lm_head output;
            // we wrap that buffer in a fresh `OwnedTensor` (Arc-
            // handle clone — no copy) so the caller can read
            // `[num_tokens, vocab_size]` f16 logits without taking
            // ownership of the worker's arena.
            // Adapter: pool's tail receives `(encoder, worker, bucket_idx)`
            // and returns `Result<(), ForwardError>`. The public hook
            // is `(encoder, logits_buf, total_n, vocab) -> Result<(), String>`.
            // Resolve logits + dims from the worker's arena and the
            // canonical's compile-time vocab; map error variants.
            let tail_adapter = followup.map(|f| {
                move |enc: &::scratchy_forward_compiler::metal_followup_reexports::ProtocolObject<
                          dyn ::scratchy_forward_compiler::metal_followup_reexports::MTL4ComputeCommandEncoder,
                      >,
                      worker: &::scratchy_target_metal::interpreter::metal::MetalWorker<Weights>,
                      bucket_idx: usize|
                      -> ::core::result::Result<
                    (),
                    ::scratchy_target_metal::interpreter::metal::ForwardError,
                > {
                    let spec = &METAL_BUCKETS[bucket_idx];
                    let logits_buf: &::scratchy_forward_compiler::metal_followup_reexports::ProtocolObject<
                        dyn ::scratchy_forward_compiler::metal_followup_reexports::MTLBuffer,
                    > = &worker.arena[spec.terminal_slot as usize];
                    let total_n = n as u32;
                    let vocab = METAL_VOCAB_SIZE as u32;
                    f(enc, logits_buf, total_n, vocab).map_err(
                        ::scratchy_target_metal::interpreter::metal::ForwardError::Followup,
                    )
                }
            });

            pool.forward_with_tail(
                wm,
                &inputs,
                |worker, bucket_idx| {
                    let spec = &METAL_BUCKETS[bucket_idx];
                    let buf = worker.arena[spec.terminal_slot as usize].clone();
                    let vocab = METAL_VOCAB_SIZE as usize;
                    let shape = [n, vocab];
                    let bytes = n * vocab * 2; // bf16/f16 — both 2 bytes
                    let dtype = match <Weights as ::scratchy_forward_compiler::CanonicalParams>::METAL_DTYPE {
                        ::scratchy_target_metal::interpreter::metal::MetalDtype::F16 =>
                            crate::__gpu::dtype::DType::F16,
                        ::scratchy_target_metal::interpreter::metal::MetalDtype::Bf16 =>
                            crate::__gpu::dtype::DType::BF16,
                        ::scratchy_target_metal::interpreter::metal::MetalDtype::Int4 =>
                            ::core::unreachable!("Int4 has no logits dtype"),
                    };
                    use ::scratchy_target_metal::interpreter::metal::__re::MTLBuffer as _;
                    let inner = crate::__gpu::tensor::GpuTensor::new(
                        buf.contents().as_ptr() as *mut u8,
                        &shape,
                        dtype,
                    );
                    crate::__gpu::owned_from_metal_buffer(inner, buf, bytes)
                },
                tail_adapter,
            )
            .expect("MetalWorkerPool::forward")
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
        /// [`ChainStepHandle`]: ::scratchy_forward_compiler::ChainStepHandle
        #[cfg(feature = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_chain_with_encoder(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            body: ::scratchy_forward_compiler::MetalChainBody<'_>,
        ) -> ::core::result::Result<(), ::std::string::String> {
            use ::scratchy_forward_compiler::CanonicalParams as _;
            use ::scratchy_target_metal::interpreter::metal::__re::{Buffer, MTLResourceOptions};

            // Lazy pool init — mirrors `forward_with_metal_followup`
            // so the chain path can fire as the first call against the
            // draft model (it won't, in practice, because lockstep
            // prefill runs first — but the pool is idempotent on
            // get_or_init).
            let pool = wm.metal_pool.get_or_init(|| {
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
                // TurboQuant: captured (Copy) so the 'static factory closure can
                // provision the packed stores + scratch per worker when enabled.
                // UNIFORM arches (num_kv_groups == 1): every layer is "global",
                // compressed at the base geometry. HYBRID/SWA arches (gemma4:
                // num_kv_groups > 1) compress ONLY the GLOBAL (full-context, group
                // 0) layers at the GLOBAL geometry — IF that geometry is a power
                // of two and <= 512 (the widened kernels' supported range); the
                // sliding layers stay fp16. `is_global[L]` = (group_of(L) == 0):
                // group 0 is always the full-context group (see
                // `compute_hybrid_kv_layout`); uniform → all-zero map → all-true.
                let __tq_global_ok =
                    (<Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM)
                        .is_power_of_two()
                        && <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM
                            <= 512;
                let __tq_on = ctx.kv_turboquant
                    && (ctx.kv_cache.num_kv_groups() == 1 || __tq_global_ok);
                let __tq_nb = ctx.kv_cache.num_blocks;
                let __tq_is_global: ::std::vec::Vec<bool> = ctx
                    .kv_cache
                    .layer_to_group_u32()
                    .iter()
                    .map(|&g| g == 0)
                    .collect();
                let factory: ::scratchy_target_metal::interpreter::metal::RuntimeFactory =
                    ::scratchy_target_metal::interpreter::metal::RuntimeFactory::new(move |dev| {
                        let max_m = METAL_MAX_BUCKET_M as u64;
                        let max_bps =
                            <Weights as ::scratchy_forward_compiler::CanonicalParams>::MAX_BLOCKS_PER_SEQ
                                as u64;
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
                        // GLOBAL_BLOCK_SIZE) for the group-0 layers only. The gate
                        // requires the GLOBAL head_dim power-of-two and <= 512.
                        let __tq_prov = if __tq_on
                            && (<Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM)
                                .is_power_of_two()
                            && <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM
                                <= 512
                        {
                            Some(::scratchy_target_metal::turboquant::build_tq_provision(
                                dev,
                                &__tq_is_global,
                                __tq_nb,
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_BLOCK_SIZE as usize,
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::NUM_GLOBAL_KV_HEADS as usize,
                                <Weights as ::scratchy_forward_compiler::CanonicalParams>::GLOBAL_HEAD_DIM as usize,
                                ::scratchy_target_metal::interpreter::metal::BLOCKS_PER_CHUNK as usize,
                                ::scratchy_target_metal::turboquant::tq_bits(
                                    <Weights as ::scratchy_forward_compiler::CanonicalParams>::TQ_KV_BITS,
                                ),
                                42,
                            ))
                        } else {
                            None
                        };
                        ::scratchy_target_metal::interpreter::metal::RuntimeBindings {
                            input_ids: alloc(max_m * 4),
                            positions: alloc(max_m * 4),
                            // Per-KV-cache-group slot_mapping + block_table.
                            slot_mappings: (0..num_kv_groups).map(|_| alloc(max_m * 4)).collect(),
                            cu_seqlens_q: alloc((max_m + 1) * 4),
                            seq_used_k: alloc(max_m * 4),
                            // Per-block span label for block-diagonal span
                            // attention: one u32 per logical block, zero-padded to
                            // the block-table stride. The kernel indexes span_ids
                            // by BLOCK (pos/block_size), not token, and the runtime
                            // stride (max_model_len-derived max_blocks_per_seq) is
                            // NOT a macro constant — so size it exactly like ONE
                            // block_tables group (max_m * max_bps), which the
                            // runtime stride always fits. (max_m alone undersized
                            // it: the span_ids BufferTooSmall bug.) All-zero unless
                            // the request carries Relocatable spans.
                            span_ids: alloc(max_m * max_bps * 4),
                            block_tables: (0..num_kv_groups)
                                .map(|_| alloc(max_m * max_bps * 4))
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
                                            p.scratch_k_table.clone()
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
                                            p.scratch_v_table.clone()
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
                            num_tokens_u32: alloc(4),
                            num_sample_rows_u32: alloc(4),
                            sample_indices: alloc(max_m * 4),
                            // GDN persistent state (per-layer) + per-forward
                            // indices/fresh flags (Shared, overwritten each
                            // forward). Index buffers sized to the max bucket
                            // (num_seqs <= num_tokens <= max_m).
                            gdn_state_conv: gdn_conv.clone(),
                            gdn_state_ssm: gdn_ssm.clone(),
                            gdn_state_indices: alloc(max_m * 4),
                            gdn_is_fresh: alloc(max_m * 4),
                            // Vision externs: sized from the baked
                            // METAL_VISION_*_BYTES consts (16-byte floor
                            // on non-vision arches). Overwritten per
                            // forward by `write_runtime_inputs`.
                            vision_rope_freqs: alloc(METAL_VISION_FREQS_BYTES),
                            pixels: alloc(METAL_VISION_PIXELS_BYTES),
                            vision_pos_embeds: alloc(METAL_VISION_POSEMB_BYTES),
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
                        }
                    });
                ::scratchy_target_metal::interpreter::metal::MetalWorkerPool::for_buckets(
                    device.device.clone(),
                    wm,
                    device.allocator.clone(),
                    METAL_BUCKETS,
                    factory,
                    1,
                    // Target-reactive cap stashed on the device by the worker
                    // after `determine_available_memory`. Prunes the compiled
                    // ladder so the colored arena fits the KV budget.
                    device.metal_bucket_max_m,
                    // Runtime per-sequence block-table capacity. Read off the KV
                    // pool so the kernel's `MaxBlocksPerSeq` function constant +
                    // rope-once scratch agree with the host block-table stride.
                    ctx.kv_cache.max_blocks_per_seq,
                )
                .expect("MetalWorkerPool::for_buckets: pool init failed")
            });

            // Read host-visible iter-0 input slices off ctx (same
            // pattern as `forward_with_metal_followup`).
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

            // Plumb the lm_head sample-row index list from ForwardCtx.
            // `last_token_indices` is `Option<TensorView>` — present
            // when the worker built `logits_indices = query_start_loc[1:] - 1`
            // (any prefill/decode that produces a sampled token), absent
            // for chunked-prefill intermediate chunks. The metal slice
            // gathers exactly these rows then runs the lm_head qmv at
            // M = indices.len().
            let last_token_indices = ctx.last_token_indices.as_ref().map(|view| {
                let raw = view.as_raw();
                ::std::slice::from_raw_parts(raw.raw_ptr() as *const u32, raw.numel())
            });

            // GDN per-forward indices (hybrid arches): read from ctx's
            // host-visible TensorViews (slot id per seq, fresh flag per seq).
            // `None` for non-hybrid arches (ctx fields are None).
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
            // Vision externs (vision towers): raw-byte reads from the
            // ctx TensorViews. `None` for text arches.
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
                has_spec_tokens: ctx.has_spec_tokens,
                last_token_indices,
                gdn_state_indices,
                gdn_is_fresh,
                vision_rope_freqs,
                pixels,
                pos_embeds,
                vision_cu_seqlens_full,
                vision_cu_seqlens_window,
                vision_window_index,
                vision_reverse_indices,
                vision_position_ids,
                mm_embeds,
                mm_dst_rows,
                mrope_cos_sin,
            };

            // Pre-pick the bucket from iter-0 num_tokens. The chain
            // shape is constant across iters so this index is reused.
            let bucket_idx = pool
                .pick_bucket(inputs.num_tokens)
                .map_err(|e| format!("MetalWorkerPool::pick_bucket: {e}"))?;
            let terminal_slot =
                METAL_BUCKETS[bucket_idx].terminal_slot as usize;
            let vocab = METAL_VOCAB_SIZE as u32;

            pool.with_chain_encoder(
                wm,
                &inputs,
                |worker, runtime, enc| {
                    // Concrete adapter that satisfies the non-generic
                    // `ChainStepHandle` trait. Holds the worker
                    // borrow + the bucket_idx; calls
                    // `worker.run_bucket_mtl4` to encode one iter onto
                    // the encoder.
                    struct Adapter<'a, W: ::scratchy_forward_compiler::CanonicalParams> {
                        worker: &'a ::scratchy_target_metal::interpreter::metal::MetalWorker<W>,
                        bucket_idx: usize,
                        terminal_slot: usize,
                        vocab: u32,
                    }
                    impl<W: ::scratchy_forward_compiler::CanonicalParams>
                        ::scratchy_forward_compiler::ChainStepHandle for Adapter<'_, W>
                    {
                        fn run_forward_step(
                            &self,
                            encoder: &::scratchy_forward_compiler::metal_followup_reexports::ProtocolObject<
                                dyn ::scratchy_forward_compiler::metal_followup_reexports::MTL4ComputeCommandEncoder,
                            >,
                            num_tokens: u32,
                            num_seqs: u32,
                            has_spec_tokens: bool,
                        ) -> ::core::result::Result<(), ::std::string::String> {
                            self.worker
                                .run_bucket_mtl4(
                                    self.bucket_idx,
                                    num_tokens,
                                    num_seqs,
                                    has_spec_tokens,
                                    encoder,
                                )
                                .map_err(|e| format!("run_bucket_mtl4: {e:?}"))
                        }

                        fn logits_buf(
                            &self,
                        ) -> &::scratchy_forward_compiler::metal_followup_reexports::ProtocolObject<
                            dyn ::scratchy_forward_compiler::metal_followup_reexports::MTLBuffer,
                        > {
                            &self.worker.arena[self.terminal_slot]
                        }

                        fn vocab(&self) -> u32 { self.vocab }
                    }
                    let adapter = Adapter {
                        worker,
                        bucket_idx,
                        terminal_slot,
                        vocab,
                    };
                    // Erase the metal-typed `&RuntimeBindings` behind the
                    // neutral handle so `MetalChainBody` (defined in the
                    // target-free compiler) names no metal-target type; the
                    // worker's body closure recovers `&RuntimeBindings`.
                    body(
                        &adapter,
                        ::scratchy_forward_compiler::MetalRuntimeHandle::new(runtime),
                        enc,
                    )
                    .map_err(
                        ::scratchy_target_metal::interpreter::metal::ForwardError::Followup,
                    )
                },
            )
            .map_err(|e| format!("MetalWorkerPool::with_chain_encoder: {e}"))
        }
    };

    // Encoder layouts have no lm_head split — `forward_backbone` is
    // semantically identical to `forward`. Decoder layouts return the
    // pre-lm_head activation via a DtoD memcpy of the protected
    // backbone slot. Both shapes are dispatched via the same
    // `FORWARD_TABLE` row.
    let forward_backbone_fn = match layout {
        BackboneLayout::Decoder { .. } => quote! {
            /// Backbone-only dispatch (no lm_head). Returns a fresh
            /// OwnedTensor (memcpy of the backbone tile).
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn forward_backbone(
                wm: &Weights,
                ctx: &crate::__gpu::ForwardCtx,
                device: &mut crate::__gpu::GpuDevice,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                let e = ::scratchy_forward_compiler::find_bucket(
                    FORWARD_TABLE, num_tokens, ctx.max_seqlen_k as u64,
                );
                unsafe {
                    crate::__gpu::run_backbone(e.4, e.9, wm, ctx, device, e.6, e.7)
                }
            }
        },
        BackboneLayout::Encoder => quote! {
            /// Backbone-only dispatch — for encoder architectures the
            /// whole pipeline IS the backbone, so this delegates to
            /// `forward` (no lm_head, no DtoD memcpy).
            #[cfg(feature = "cuda")]
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn forward_backbone(
                wm: &Weights,
                ctx: &crate::__gpu::ForwardCtx,
                device: &mut crate::__gpu::GpuDevice,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                unsafe { forward(wm, ctx, device, num_tokens) }
            }
        },
    };

    // The per-arch device-runtime emit (weight-accessor impl, the static `Op`
    // tapes, `FORWARD_TABLE`) is the per-op host-dispatch surface — it exists
    // only for backends that dispatch each op through a host-emitted table
    // (cuda + metal). A target that runs an embedded fused program instead
    // (e.g. KTIR on the host) emits none of it. The macro is built per-backend,
    // so gate the EMISSION via `cfg!` rather than per-item `#[cfg]` attrs (the
    // static-slice repetition can't take one).
    let (canonical_params_impl, weight_accessors_impl, static_slices_emit, forward_table) =
        if cfg!(any(feature = "cuda", feature = "metal")) {
            (
                canonical_params_impl,
                weight_accessors_impl,
                quote! { #(#static_slices)* },
                forward_table,
            )
        } else {
            (quote! {}, quote! {}, quote! {}, quote! {})
        };

    quote! {
        #weights

        #ktir_bundle_const

        #canonical_params_impl

        #weight_accessors_impl

        #instruction_alias

        #static_slices_emit

        #forward_table

        #metal_emission

        /// Dispatch on (num_tokens, sk_bucket) → tape_index entry, then
        /// run the universal interpreter.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            let e = ::scratchy_forward_compiler::find_bucket(
                FORWARD_TABLE, num_tokens, ctx.max_seqlen_k as u64,
            );
            unsafe {
                crate::__gpu::run(e.4, e.9, e.5, e.10, wm, ctx, device, e.6, e.8)
            }
        }

        /// Piecewise CUDA-graph capture for the bucket `(num_tokens,
        /// max_seqlen_k)` selects. Caller MUST have called
        /// `device.caching.begin_allocate_to_pool()` before this fn so
        /// the captured addresses come from a private pool that stays
        /// alive for the runner's lifetime.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_piecewise_capture(
            wm: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> ::anyhow::Result<crate::__gpu::piecewise::PiecewiseRunner> {
            let e = ::scratchy_forward_compiler::find_bucket(
                FORWARD_TABLE, num_tokens, ctx.max_seqlen_k as u64,
            );
            unsafe {
                crate::__gpu::piecewise::run_piecewise_capture(
                    e.4, e.9, e.5, e.10, wm, ctx, device, e.6, e.8,
                )
            }
        }

        #forward_backbone_fn

        /// Walk `FORWARD_TABLE` and return one [`BucketDump`] per
        /// row, with backbone + lm_head normalized for non-generic
        /// inspection (no `&Weights`, no GPU). Used by
        /// `scr model info` via the inventory registry. Available
        /// under either backend so the CLI subcommand can dump
        /// metal-compiled arches too.
        #[cfg(any(feature = "cuda", feature = "metal"))]
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
    }
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
///   variant, so each ships its own sniff.
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
    sfufs: &WorkloadAssignments,
    lib: &ImplementationLibrary,
    model: &ModelParams,
    manifest: &crate::weights_manifest::WeightsManifest,
    canonical: &Ident,
    tp_world_size: u8,
    emit_fingerprint: bool,
) -> TokenStream {
    let weights = emit_weights_struct(
        program,
        fuf,
        sfufs,
        None,
        lib,
        model,
        manifest,
        WeightsEmitMode::Shim { canonical },
        tp_world_size,
        emit_fingerprint,
    );

    // Per-tape_index fn surfaces are gone — dispatch lives on the
    // canonical's `FORWARD_TABLE` + `find_bucket`. Re-export the
    // arch-level dispatchers only. Under `metal` the shim shares the
    // canonical's `METAL_BUCKETS` static + `metal_pool()` constructor
    // — variant-specific differences (quant format, fingerprint) are
    // load-time only; static tape_index plans are byte-identical.
    let _ = sfufs;
    quote! {
        #weights

        // Spyre: shim variants share the canonical's solve, so the
        // canonical owns the embedded `KTIR_BUNDLE` AND `SENGRAPH_BUNDLE`.
        // Re-export BOTH so this shim module's `ScratchyWeights::ktir_bundle()`
        // AND `sengraph_bundle()` arms resolve symmetrically (the registry
        // accessor in macros/src/lib.rs names `<first_model>::SENGRAPH_BUNDLE`,
        // which is a shim module when the first arch variant is non-canonical —
        // re-exporting only KTIR_BUNDLE left `--target sendnn` unresolvable there).
        #[cfg(feature = "spyre")]
        pub use super::#canonical::{KTIR_BUNDLE, SENGRAPH_BUNDLE};

        // `dump` is the per-variant `scr model info` accessor —
        // available under either backend so shim variants register
        // under metal too.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        pub use super::#canonical::dump;

        #[cfg(feature = "cuda")]
        pub use super::#canonical::{forward, forward_backbone, forward_piecewise_capture};

        #[cfg(feature = "metal")]
        pub use super::#canonical::{
            forward, forward_chain_with_encoder, forward_with_metal_followup,
            METAL_ARENA_PEAK_BYTES, METAL_BUCKET_ARENA_COSTS, METAL_BUCKETS, metal_pool,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_vocab::WeightAccessor;
    use quote::format_ident;

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
        crate::config::ModelParams {
            name: "shard_test".into(),
            source_stem: "shard-test".into(),
            source_path: PathBuf::new(),
            bounds,
            scalars: BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: Vec::new(),
            extra_tracked_paths: Vec::new(),
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
            arch: Default::default(),
            torch_dtype: None,
        }
    }

    /// `emit_canonical_params_impl` at `tp_world_size = 1` is
    /// identity — bounds flow through unchanged. Pinning this so
    /// task #7's outer-loop fanout cannot accidentally regress the
    /// single-rank build (which is every per-arch-crate build under
    /// `--features cuda` until activation lands).
    #[test]
    fn canonical_params_at_tp_eq_1_is_identity() {
        // Llama-2-7B-ish numbers: 32 q heads, 32 kv heads,
        // head_dim=128, intermediate=11008.
        let m = shard_test_model(32, 32, 128, 11008);
        let ts = emit_canonical_params_impl(
            &m,
            1,
            false,
            false,
            false,
            #[cfg(feature = "metal")]
            &mut None,
        )
        .to_string();
        // Q size = 32 * 128 = 4096; KV size = 32 * 128 = 4096.
        assert!(
            ts.contains("NUM_Q_HEADS : u32 = 32"),
            "tp=1 must keep NUM_Q_HEADS = 32; got {ts}"
        );
        assert!(
            ts.contains("NUM_KV_HEADS : u32 = 32"),
            "tp=1 must keep NUM_KV_HEADS = 32; got {ts}"
        );
        assert!(
            ts.contains("INTERMEDIATE_SIZE : usize = 11008"),
            "tp=1 must keep INTERMEDIATE_SIZE = 11008; got {ts}"
        );
        assert!(
            ts.contains("Q_SIZE : usize = 4096"),
            "tp=1 must keep Q_SIZE = 4096; got {ts}"
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
        let ts = emit_canonical_params_impl(
            &m,
            2,
            false,
            false,
            false,
            #[cfg(feature = "metal")]
            &mut None,
        )
        .to_string();
        assert!(
            ts.contains("NUM_Q_HEADS : u32 = 16"),
            "tp=2 must shard NUM_Q_HEADS to 16; got {ts}"
        );
        assert!(
            ts.contains("NUM_KV_HEADS : u32 = 16"),
            "tp=2 must shard NUM_KV_HEADS to 16; got {ts}"
        );
        assert!(
            ts.contains("INTERMEDIATE_SIZE : usize = 5504"),
            "tp=2 must shard INTERMEDIATE_SIZE to 5504; got {ts}"
        );
        // Q_SIZE = (32/2) * 128 = 2048
        assert!(
            ts.contains("Q_SIZE : usize = 2048"),
            "tp=2 must compute Q_SIZE from sharded heads to 2048; got {ts}"
        );
        assert!(
            ts.contains("KV_SIZE : usize = 2048"),
            "tp=2 must compute KV_SIZE from sharded heads to 2048; got {ts}"
        );
        // HEAD_DIM is per-head and never sharded.
        assert!(
            ts.contains("HEAD_DIM : u32 = 128"),
            "tp=2 must keep HEAD_DIM = 128; got {ts}"
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
        let ts = emit_canonical_params_impl(
            &m,
            8,
            false,
            false,
            false,
            #[cfg(feature = "metal")]
            &mut None,
        )
        .to_string();
        // 32 / 8 = 4
        assert!(
            ts.contains("NUM_Q_HEADS : u32 = 4"),
            "tp=8 must shard NUM_Q_HEADS to 4; got {ts}"
        );
        // 8 / 8 = 1
        assert!(
            ts.contains("NUM_KV_HEADS : u32 = 1"),
            "tp=8 must shard NUM_KV_HEADS to 1; got {ts}"
        );
        // 14336 / 8 = 1792
        assert!(
            ts.contains("INTERMEDIATE_SIZE : usize = 1792"),
            "tp=8 must shard INTERMEDIATE_SIZE to 1792; got {ts}"
        );
    }

    #[test]
    fn split_base_layer_recognizes_layered_and_unlayered_names() {
        // Layered: trailing _<digits> peels off as the layer index.
        assert_eq!(
            split_base_layer("input_layernorm_3"),
            ("input_layernorm".to_string(), Some(UnrollIndex(3)))
        );
        assert_eq!(
            split_base_layer("self_attn_q_proj_31"),
            ("self_attn_q_proj".to_string(), Some(UnrollIndex(31)))
        );
        // Non-layered: no trailing _<digits>.
        assert_eq!(
            split_base_layer("embed_tokens"),
            ("embed_tokens".to_string(), None)
        );
        assert_eq!(split_base_layer("lm_head"), ("lm_head".to_string(), None));
        // Trailing-non-digit suffix isn't a layer — the whole name
        // stays as the base.
        assert_eq!(
            split_base_layer("rotary_local"),
            ("rotary_local".to_string(), None)
        );
        // Empty trailing chunk after `_` is not a layer either.
        assert_eq!(
            split_base_layer("trailing_"),
            ("trailing_".to_string(), None)
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
                source_weights: vec![],
            })
            .collect::<Vec<_>>();
        let ts = emit_weights_accessor_methods(&accessors).to_string();
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
            source_weights: vec![],
        }];
        let ts = emit_weights_accessor_methods(&accessors).to_string();
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
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("norm_0"),
                rust_type: ty.clone(),
                source_weights: vec![],
            },
        ];
        let _ = emit_weights_accessor_methods(&accessors);
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
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("input_layernorm_1"),
                rust_type: quote! { crate::__gpu::layers::LinearLayer },
                source_weights: vec![],
            },
        ];
        let _ = emit_weights_accessor_methods(&accessors);
    }

    #[test]
    fn accessor_methods_empty_input_yields_empty_tokens() {
        // No accessors → no impl block. (Empty `impl Weights {}`
        // would be valid Rust but pointless; the codegen elides it.)
        let ts = emit_weights_accessor_methods(&[]).to_string();
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
            source_weights: vec![],
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
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("embed_tokens"),
                rust_type: quote! { crate::__gpu::layers::Embedding },
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
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("input_layernorm_1"),
                rust_type: quote! { crate::__gpu::layers::LinearLayer },
                source_weights: vec![],
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
                source_weights: vec![],
            },
            WeightAccessor {
                name: format_ident!("moe_2"),
                rust_type: quote! { crate::__gpu::layers_moe::DeepSeekV2MoELayer },
                source_weights: vec![],
            },
        ];
        let ts = emit_weights_accessor_methods(&accs).to_string();
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

        // Concat (always column-parallel) → `_concat_sharded`. Bakes
        // `world` literal but no `dim` arg (concat-sharded is always
        // dim=0 internally).
        let c = FieldLoad::LinearConcat(vec![
            "model.layers.0.mlp.gate_proj".to_string(),
            "model.layers.0.mlp.up_proj".to_string(),
        ]);
        let ts = emit_layered_load_body(&c, 32, 4, false, None, "model.layers", None).to_string();
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
    use std::path::PathBuf;

    /// Per-arch config dir in the `crates/models/arch/<arch>/configs/`
    /// layout. Pass the arch slug using hyphens, e.g. `"deepseek-v3"`,
    /// `"qwen3"`.
    fn arch_configs(arch: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            // `configs/<arch>`, not `<arch>/configs` — the layout the
            // collapse into `scratchy-models` produced.
            .join("configs")
            .join(arch)
    }

    /// MLA archs (DeepSeek V3 / Kimi K2) ship `q_a_proj` rather than
    /// `q_proj`, so the FP8-block disambiguation tensor names must
    /// follow the same `fp_leaf` selection the rest of the
    /// fingerprint already uses. Regression for the bug where the
    /// V3 FP8-block fingerprint hardcoded `q_proj.weight_scale_inv`
    /// and silently rejected every V3 FP8-block checkpoint.
    #[test]
    fn fp8_block_disambiguation_uses_q_a_proj_for_mla_archs() {
        let dir = arch_configs("deepseek-v3");
        let configs =
            crate::config::load_dir(&dir, &Default::default()).expect("load deepseek-v3 configs");
        let manifest = crate::weights_manifest::load_or_empty(&dir)
            .expect("load deepseek-v3 weights manifest");
        // Pick a V3 variant with FP8-block quantization (block_size: Some).
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
            .expect("at least one V3 FP8-block variant");
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
        assert!(
            ts.contains("q_a_proj.weight_scale_inv"),
            "MLA arch FP8-block fingerprint should sniff q_a_proj.weight_scale_inv, got:\n{ts}",
        );
        assert!(
            !ts.contains("q_proj.weight_scale_inv"),
            "MLA arch FP8-block fingerprint must not reference q_proj.weight_scale_inv \
             (V3/K2 ship q_a_proj on disk; q_proj presence would silently reject every \
             real checkpoint), got:\n{ts}",
        );
        assert!(
            ts.contains("q_a_proj.weight_scale"),
            "MLA arch FP8-block fingerprint should sniff q_a_proj.weight_scale, got:\n{ts}",
        );
    }

    /// Flat-Q MLA arches (Moonlight / K2 direct-q_proj) ship `q_proj`
    /// rather than `q_a_proj`, so the FP8-block fingerprint must use
    /// `q_proj.weight_scale_inv` even though the arch is otherwise
    /// structurally MLA (kv_a_proj_with_mqa, kv_b_proj, etc.).
    #[test]
    fn fp8_block_disambiguation_uses_q_proj_for_flat_q_mla_archs() {
        let dir = arch_configs("deepseek-v3-flat");
        let configs = crate::config::load_dir(&dir, &Default::default())
            .expect("load deepseek-v3-flat configs");
        let manifest = crate::weights_manifest::load_or_empty(&dir)
            .expect("load deepseek-v3-flat weights manifest");
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
            .expect("at least one deepseek-v3-flat FP8-block variant");
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
        assert!(
            ts.contains("q_proj.weight_scale_inv"),
            "flat-Q MLA FP8-block fingerprint should use q_proj.weight_scale_inv \
             (no q_a_proj on disk for q_lora_rank=null checkpoints), got:\n{ts}",
        );
        assert!(
            !ts.contains("q_a_proj.weight_scale_inv"),
            "flat-Q MLA FP8-block fingerprint must not reference q_a_proj \
             (Moonlight ships q_proj, not q_a_proj), got:\n{ts}",
        );
    }

    /// Non-MLA arches (Llama / Qwen / etc.) keep the historical
    /// `q_proj` leaf — the fp_leaf selection is purely opt-in for
    /// archs whose manifest declares `q_a_proj`.
    #[test]
    fn fp8_block_disambiguation_uses_q_proj_for_non_mla_archs() {
        let dir = arch_configs("qwen3");
        let configs =
            crate::config::load_dir(&dir, &Default::default()).expect("load qwen3 configs");
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
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
        assert!(
            ts.contains("q_proj.weight_scale_inv"),
            "non-MLA FP8-block fingerprint should still use q_proj, got:\n{ts}",
        );
    }

    /// Load ONE checked-in config as `ModelParams`.
    ///
    /// ⛔ NOT `load_dir`. `load_dir` filters every stem through
    /// `CARGO_FEATURE_<STEM>`, which is right for a build and unstateable for a
    /// unit test — a proc-macro crate has no model features, so it selects ZERO
    /// configs and a test that iterates the result passes vacuously. (The
    /// `fp8_block_disambiguation_*` tests above are in exactly that state.) The
    /// subject here is `emit_fingerprint_check`, a pure function of
    /// `ModelParams`, so name the file and skip the build's scope entirely.
    fn config_of(arch: &str, stem: &str) -> crate::config::ModelParams {
        let path = arch_configs(arch).join(format!("{stem}.json"));
        crate::config::load_file(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
    }

    /// The `rope_theta` literal a variant's fingerprint gates on, if any.
    /// Parsed back out of the emitted token stream so the assertion is about
    /// what the generated code will actually compare, not about our intent.
    fn emitted_rope_theta(model: &crate::config::ModelParams, arch: &str) -> Option<f64> {
        let manifest = crate::weights_manifest::load_or_empty(&arch_configs(arch))
            .expect("load weights manifest");
        let ts = emit_fingerprint_check(model, &manifest, 1).to_string();
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

    /// A config that declares no `rope_theta` must emit NO gate. 19 in-tree
    /// configs are in this bucket (gemma3/gemma4/qwen3.5 carry per-attention-
    /// class thetas nested under `rope_parameters`; modernbert and llama-2-70b
    /// omit it and take HF's implicit 10⁴ default). Synthesizing that default
    /// here would invent a constraint the config never stated and false-reject
    /// a checkpoint that spells `10000.0` out.
    #[test]
    fn a_config_without_rope_theta_emits_no_theta_gate() {
        let model = config_of("llama", "llama-2-70b");
        assert!(
            model.scalars.get("rope_theta").is_none() && model.bounds.get("rope_theta").is_none(),
            "premise broken: llama-2-70b now declares a rope_theta, so it no longer exercises \
             the no-gate case — repoint this test at another config that declares none",
        );
        assert!(
            emitted_rope_theta(&model, "llama").is_none(),
            "a config with no declared rope_theta must not gate on one",
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
        let ts = emit_fingerprint_check(&model, &manifest, 1).to_string();
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
            BackboneLayout::Decoder { backbone_out } => {
                assert_eq!(
                    backbone_out,
                    (TileId(0), 0),
                    "decoder backbone-out must be the lm_head Gemm's first tile input",
                );
            }
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
                BackboneLayout::Decoder { backbone_out } => assert_eq!(
                    backbone_out,
                    (TileId(0), 0),
                    "{name}: backbone-out must be the lm_head Gemm's tile input",
                ),
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
            BackboneLayout::Decoder { backbone_out } => assert_eq!(backbone_out, (TileId(0), 0)),
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
