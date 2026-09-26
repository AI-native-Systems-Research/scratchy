// SPDX-License-Identifier: Apache-2.0
//! `#[forward]` attribute macro — compile a DSL forward pass into
//! specialized Rust + CUDA per model architecture.
//!
//! This crate is the proc-macro shell. Every compiler pass (parse,
//! classify, shape-infer, CFG, unroll, solve, schedule) lives in
//! internal modules here so they can be unit-tested in isolation.
//! The attribute macro drives them end-to-end at macro expansion
//! time: it reads the configs + target profile, runs the whole
//! pipeline for every (model × workload-point), and emits the
//! generated code.
//!
//! Until codegen (PLAN task #5) lands, the emitted code is a
//! placeholder `pub mod <model>` per model containing constants
//! derived from the real pipeline (tile count, wave count,
//! predicted cost per workload). Integration tests assert those
//! constants, proving the pipeline actually executes at compile
//! time.

use std::path::Path;

// Backends are mutually exclusive — the emit is single-backend (one `__gpu`
// alias, one `Weights` shape, one impl-library target). Enabling two at once
// produces conflicting dual-emit (duplicate `__gpu`, missing `metal_pool`, a
// solver with no impl for the other backend's ops). `impl_lib.rs` already
// assumes this guard exists ("no model crate ever builds with both backends
// on; the macro's own `compile_error!` upstream rejects the combo"). Fail here,
// with a clear message, rather than mid-build.
#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("features 'cuda' and 'metal' are mutually exclusive — enable exactly one backend");
#[cfg(all(feature = "cuda", feature = "spyre"))]
compile_error!("features 'cuda' and 'spyre' are mutually exclusive — enable exactly one backend");
#[cfg(all(feature = "metal", feature = "spyre"))]
compile_error!("features 'metal' and 'spyre' are mutually exclusive — enable exactly one backend");

use proc_macro2::Span;
use quote::quote;
use syn::Ident;

mod alias_rules;
mod arch_spec;
mod ast;
#[cfg(feature = "metal")]
use scratchy_target_metal::atom;
mod cfg;
mod classified;
mod classify;
mod codegen;
#[cfg(feature = "cuda")]
mod concurrency;
mod config;
/// The baked program, as const tokens — see its own header.
#[cfg(feature = "spyre")]
mod ktir_tokens;
pub use config::{total_models_emitted, unmatched_build_filter_tags};
#[cfg(feature = "metal")]
use scratchy_target_metal::fuse_pass;
mod assignment;
#[cfg(feature = "cuda")]
mod cost;
mod emit;
mod fuf;
mod impl_lib;
mod interpreter_codegen;
#[cfg(feature = "metal")]
mod opcode_shapes;
// The DSL front end: a `dsl/<arch>.py` carrier → `Ast`.
mod parse_python;
mod quantization;
/// The front end, exposed for the `scratchy-forwards` build driver in
/// scratchy-models.
pub use parse_python::{PythonCarrier, parse_python_file};
// Exposed so build scripts (`hf_registry_build.rs`) can parse a Hub
// candidate's OWN `quantization_config` with the exact same logic that
// decides what a checkpoint means at compile time — not a second,
// drifting copy of "what counts as e.g. 4-bit affine".
pub use quantization::{ParseError, QuantMethod, QuantizationConfig};
mod render;
/// The build script's other half: `compile_carrier` makes the tokens,
/// `render_tokens` turns them into the text rustc reads.
pub use render::render_tokens;
mod schedule;
mod shape;
#[cfg(feature = "cuda")]
mod solver;
mod target;
mod to_wavefront;
mod tp_lowering;
mod vision_lowering;
// ⭐ THE ONE PRODUCER OF THE MODEL'S WEIGHT ACCESSOR SET, for every target.
//
// It used to be gated `all(feature = "spyre", not(feature = "metal"))` because metal derived
// the same set inline from `tape_groups` — a cfg that selects between two bodies, which keeps
// both. The target being a compile-time singleton is what makes ONE body viable: there is no
// second target in the build to compromise for. Packing gate/up into a single `__fused__`
// buffer is metal's kernel ABI, so it arrives as a bool argument rather than a second walk.
//
// Gated to the targets that consume it: metal (via `emit_weight_bindings` /
// `from_tape` in the `#[cfg(feature = "metal")]` arm of `emit_model`) and
// spyre (the `all(feature = "spyre", not(feature = "metal"))` arm). cuda
// derives its accessor set through ISel and calls nothing here, so without
// the gate every item in the module is dead code under `-Fcuda`.
#[cfg(any(feature = "metal", feature = "spyre"))]
mod weight_bindings;
mod weight_vocab;
mod weights_manifest;

mod vision_glue;

// ── Carrier decorator arguments ───────────────────────────────────

struct ForwardArgs {
    /// Discrete `num_tokens` points to solve at. Non-empty.
    workloads: Vec<u64>,
    /// Discrete `sk_bucket` (KV-cache span in tokens) points to solve
    /// at. Empty ⇒ the solver sweeps only the `num_tokens` axis with
    /// `sk_bucket = 0` (the sentinel "sk axis unused"). Declare this
    /// for models where attention dispatch wants to pick different
    /// kernels at different KV spans — e.g. FlashInfer decode wins on
    /// long sk, FA2 wins at small prefill.
    sk_buckets: Vec<u64>,
    /// Path to the per-arch CPU pixel-pack fn. Optional for
    /// `@vision_forward`, ignored by `@forward`. The fn signature
    /// must match `fn(&VisionConfig, &[f32], u32, u32) -> (Vec<u16>,
    /// (u32, u32, u32))`. The macro-emitted `VisionArchWeights` impl
    /// forwards its `pixel_pack` associated fn to this path.
    pixel_pack: Option<syn::Path>,
    /// Path to a `pub const PROCESSOR: scratchy_vision::MmMetadata` in
    /// the per-arch crate declaring CPU-side host preprocessing
    /// metadata (placeholder token id key, size policy, tokens-per-
    /// image policy, preprocess fn). Required for `@vision_forward`,
    /// ignored by `@forward`. Baked into every emitted
    /// `ScratchyMmRegistration` row. scratchy stays arch-agnostic — every
    /// arch-specific knob is data on the const, not a switch in scratchy.
    processor: Option<syn::Path>,
    /// Span used for error reporting when a required arg is
    /// missing.
    span: Span,
}

impl ForwardArgs {
    /// The carrier decorator's arguments, with the defaults applied.
    fn from_carrier(carrier: &PythonCarrier) -> Self {
        Self {
            // Omitting `workloads` gives the global default ladder.
            // Per-bucket affordability is decided at load time by
            // `select_prefill_bucket`, so declaring the full ladder here
            // is free on small devices — they just prune it. Vision
            // carriers always declare theirs (image-patch counts, not
            // the text-decode ladder).
            workloads: carrier
                .workloads
                .clone()
                .unwrap_or_else(|| DEFAULT_DECODER_WORKLOADS.to_vec()),
            // Omitting `sk_buckets` gives the standard KV-span ladder
            // (every decoder arch wants the same one). An explicit empty
            // list `sk_buckets=[]` opts into the legacy 1-D sweep, where
            // `sk_bucket = 0` is the sentinel that non-sk-constrained
            // impls (Any / NumTokensRange) accept unconditionally and FI
            // impls with a real sk range can never match.
            sk_buckets: carrier
                .sk_buckets
                .clone()
                .unwrap_or_else(|| DEFAULT_SK_BUCKETS.to_vec()),
            pixel_pack: carrier.pixel_pack.clone(),
            processor: carrier.processor.clone(),
            span: Span::call_site(),
        }
    }
}

/// Format a microsecond value adaptively for human scanning:
/// `<1000µs` as `Nµs`, `<100ms` as `N.Xms`, else `Nms`.
/// Cross-variant forward-fn dedup. Returns a map `variant_idx →
/// canonical_module_ident`, where `canonical_module_ident` is the
/// module name of the variant chosen to carry the full emitted
/// forward-fn bodies for its equivalence class. Variants whose
/// canonical is themselves emit full bodies; others emit shims.
///
/// The key — what makes two variants' forward fn bodies
/// byte-identical — is:
/// 1. The arch's DSL (always shared within a `#[forward]` call).
/// 2. The variant's integer `bounds` (baked as literals in
///    `ctx.bound(...)` and the unroll trip counts).
/// 3. The variant's float `scalars` (baked as literals in
///    `attention_scale_for` / similar).
/// 4. The SFUF per workload point (which `Impl` runs at each
///    subgraph → whose `emit_call` output lands in the body).
///
/// Among AWQ / GPTQ / CT variants of the same dense base, items
/// 1-3 are identical and item 4 collapses because they all resolve
/// to `Marlin*Impl`. Dense + BNB4 stay separate because their
/// `Impl` picks differ from Marlin's and from each other's.
///
/// Canonical selection: the variant with the earliest
/// `source_stem` in the equivalence class wins. Deterministic
/// across macro re-expansions so cargo's incremental cache stays
/// stable.
fn compute_canonical_variants(
    solved: &[impl HasSolvedSig],
) -> std::collections::HashMap<usize, Ident> {
    let mut by_sig: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, sm) in solved.iter().enumerate() {
        by_sig.entry(sm.dedup_signature()).or_default().push(i);
    }

    let mut out: std::collections::HashMap<usize, Ident> = std::collections::HashMap::new();
    for members in by_sig.values() {
        // Pick the member with the alphabetically-earliest stem as
        // canonical. All other members point to it.
        let mut ordered = members.clone();
        ordered.sort_by_key(|&i| solved[i].source_stem().to_string());
        let canonical_idx = ordered[0];
        let canonical_ident = Ident::new(
            solved[canonical_idx].model_name(),
            proc_macro2::Span::call_site(),
        );
        for &idx in &ordered {
            out.insert(idx, canonical_ident.clone());
        }
    }
    out
}

/// Trait over the per-variant fields `compute_canonical_variants`
/// needs; implemented inline on the `SolvedModel` wrapper inside
/// `forward_impl`. Keeps the helper callable without plumbing the
/// concrete `SolvedModel` type through.
trait HasSolvedSig {
    fn dedup_signature(&self) -> String;
    fn source_stem(&self) -> &str;
    fn model_name(&self) -> &str;
}

/// One stable string per QuantMethod *FieldLoad arm* the codegen
/// will emit. AWQ/GPTQ/CT collapse to `q:awq` / `q:gptq` because
/// they all share the `MarlinLinear` arm (the runtime
/// `marlin_storage` discriminator covers their on-disk split).
/// FP8 splits on `block_size`: per-tensor / per-channel goes to
/// `Fp8Linear` (1D scale), blockwise goes to `Fp8BlockLinear`
/// (2D scale) — different `load_with` bodies, so they cannot
/// share a canonical.
///
/// Threaded into `dedup_signature` so equivalence-class hashing
/// keeps FP8 block / std variants in separate canonicals. Without
/// this discriminator, qwen3's `fp8-block-128x128` and
/// `fp8-dynamic-per-tensor` hash identical (Impl picks match,
/// bounds match), `block` wins canonical alphabetically, the
/// `dynamic` shim calls `Fp8BlockLinear::load` on a 1D scale, and
/// the kernel panics during graph capture with `FP8 block scale
/// must be 2D, got 1D`.
fn dedup_quant_sig(method: Option<&crate::quantization::QuantMethod>) -> String {
    match method {
        None => "q:dense".to_string(),
        Some(crate::quantization::QuantMethod::Awq { .. }) => "q:awq".to_string(),
        Some(crate::quantization::QuantMethod::Gptq { .. }) => "q:gptq".to_string(),
        Some(crate::quantization::QuantMethod::Bnb4 { .. }) => "q:bnb4".to_string(),
        Some(crate::quantization::QuantMethod::Fp8 { block_size, .. }) => {
            if block_size.is_some() {
                "q:fp8-block".to_string()
            } else {
                "q:fp8-std".to_string()
            }
        }
        Some(crate::quantization::QuantMethod::Ggml) => "q:ggml".to_string(),
        Some(crate::quantization::QuantMethod::Affine {
            bits,
            group_size,
            quantize_embed,
            bits_overrides,
            per_module,
        }) => {
            let qe = if *quantize_embed { "-qe" } else { "" };
            // The per-role (`bits_overrides`) and per-module (`per_module`,
            // MLX dynamic/mixed-bit like OptiQ) bit assignments MUST be in
            // the signature: two affine variants that share a top-level
            // `bits`/`group_size` but differ in their per-layer bit map
            // emit DIFFERENT forwards (per-layer kernel selection), so they
            // cannot collapse to one canonical. Omitting them let OptiQ
            // (per_module mixed 4/8-bit) dedup onto the uniform
            // `mlp8-router8` variant → OptiQ ran the wrong per-role forward
            // → garbage whenever both were compiled into one binary. Both
            // vecs are already sorted, so the string is deterministic.
            let ov = bits_overrides
                .iter()
                .map(|(k, b)| format!("{k}={b}"))
                .collect::<Vec<_>>()
                .join(",");
            let pm = per_module
                .iter()
                .map(|(k, (b, g))| format!("{k}={b}/{g}"))
                .collect::<Vec<_>>()
                .join(",");
            format!("q:affine-b{bits}-g{group_size}{qe}|ov[{ov}]|pm[{pm}]")
        }
        Some(crate::quantization::QuantMethod::Nvfp4 { group_size }) => {
            format!("q:nvfp4-g{group_size}")
        }
    }
}

/// Truthy when `SCRATCHY_DEBUG` is set to anything non-empty other
/// than "0". Gates the noisier solver-internals output (per-phase
/// timings, per-model solve time) at proc-macro time.
///
/// Caveat: cargo doesn't track plain env vars across proc-macro
/// invocations (the rebuild-tracking variant `tracked_env::var` is
/// nightly-only), so flipping `SCRATCHY_DEBUG` between cached builds
/// won't re-run the macro on its own — touch a model source file or
/// run `cargo clean -p scratchy-models` to force re-expansion.
pub(crate) fn scratchy_debug() -> bool {
    std::env::var("SCRATCHY_DEBUG")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Stable per-tp-world-size discriminator threaded into
/// `dedup_signature` so the canonical-equivalence-class hash splits
/// the (variant × tp) fanout — a model compiled at tp=1 cannot share
/// a canonical with the same model compiled at tp=2 because the
/// emitted body sees different sharded `INTERMEDIATE_SIZE` /
/// `NUM_ATTENTION_HEADS` / `NUM_KEY_VALUE_HEADS` constants on
/// `<W as CanonicalParams>` and (for ShardDim1 weights) inserts
/// `Instruction::AllReduce` rows the tp=1 body lacks.
///
/// Compile-time set: `{1, 2, 4, 8}` (+16 behind the future
/// `tp-frontier` feature for NVL72-class models). Past 16 is rare
/// enough to keep behind a cargo feature gate.
fn dedup_tp_sig(tp_world_size: u8) -> String {
    format!("tp:{tp_world_size}")
}

/// Adaptive µs-to-string formatter for build-log scoring lines.
/// Used by the per-M score emission landed in the activation phase
/// (`889c44b2f`); kept as `#[allow(dead_code)]` for the foundation
/// commits that introduce it before the consumer lands during
/// the rebase replay.
#[allow(dead_code)]
fn fmt_us(us: f64) -> String {
    if us < 1000.0 {
        format!("{us:.0}µs")
    } else if us < 100_000.0 {
        format!("{:.1}ms", us / 1000.0)
    } else {
        format!("{:.0}ms", us / 1000.0)
    }
}

// ── Pipeline entry point ──────────────────────────────────────────

/// Global, target-reactive prefill/decode bucket ladder. Every `#[forward]`
/// arch that does not explicitly override `workloads` compiles THIS ladder
/// (for both the CUDA and Metal codegen). The full ladder is always emitted;
/// the affordable subset is chosen per device at load time by
/// `select_prefill_bucket` (Metal prunes its colored arena, CUDA its captured
/// graph shapes). Mirrors the llama arch's historical ladder.
const DEFAULT_DECODER_WORKLOADS: &[u64] = &[1, 2, 4, 8, 64, 512, 1024, 2048, 4096];

/// Default KV-cache-span (`sk_bucket`) ladder. Every `#[forward]` arch that
/// does not explicitly override `sk_buckets` solves at THESE spans; all
/// current decoder archs share this list. Override with an explicit list, or
/// `sk_buckets = []` for the legacy 1-D (sk-axis-unused) sweep.
const DEFAULT_SK_BUCKETS: &[u64] = &[128, 512, 2048, 8192];

/// Per-carrier overrides on the shared compile pipeline. Selected by
/// the carrier's decorator; threaded through the prelude / fanout /
/// lowering passes so the body of [`compile_carrier`] stays
/// almost-uniform across the decoder and vision variants.
#[derive(Clone, Copy)]
struct CompileMode {
    /// Selects the DSL extern set used by [`classify::classify_with`].
    prelude: classified::Prelude,
    /// True for `#[forward]` (decoder), false for `#[vision_forward]`.
    /// Gates the row-parallel AllReduce + lm_head AllGather lowering
    /// passes: vision is replicated in v1 so they're skipped.
    apply_tp_lowering: bool,
    /// True for `#[forward]`, false for `#[vision_forward]`. Gates
    /// the post-Embed multimodal splice pass — that splice belongs
    /// on the decoder's text-side hidden states, not the encoder's
    /// patch hidden states. Only consulted under `cuda` (the splice
    /// pass is cuda-specific); declared cuda-only so non-cuda builds
    /// don't carry a dead field.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    apply_mm_splice: bool,
    /// True for `#[forward]` (which fans out over `{1, 2, 4, 8}` at
    /// nccl-enabled), false for `#[vision_forward]` (always tp=1).
    enable_tp_fanout: bool,
    /// True for `#[forward]`, false for `#[vision_forward]`. Gates
    /// emission of the arch-level dispatcher (`enum Weights`,
    /// `ScratchyArchRegistration` inventory submission, per-variant
    /// HF-bounds accessors). Vision encoders reach their compiled
    /// `Weights` via the hand-written `ScratchyMmRegistration` in
    /// each VL crate's `vision.rs`; HF `architectures` strings like
    /// `Qwen2VLForConditionalGeneration` are claimed by the text-side
    /// `qwen2` arch, not the vision encoder. Skipping here also
    /// avoids the `collect_dispatch_bounds` panic — vision configs
    /// don't carry `num_hidden_layers` / `hidden_size` /
    /// `num_attention_heads` / `vocab_size`.
    emit_arch_dispatch: bool,
}

impl CompileMode {
    const DECODER: Self = Self {
        prelude: classified::Prelude::Decoder,
        apply_tp_lowering: true,
        #[cfg(any(feature = "cuda", feature = "metal"))]
        apply_mm_splice: true,
        enable_tp_fanout: true,
        emit_arch_dispatch: true,
    };
    const VISION: Self = Self {
        prelude: classified::Prelude::Vision,
        apply_tp_lowering: false,
        #[cfg(any(feature = "cuda", feature = "metal"))]
        apply_mm_splice: false,
        enable_tp_fanout: false,
        emit_arch_dispatch: false,
    };
}

/// The whole pipeline: a parsed `dsl/<arch>.py` carrier, compiled against
/// its arch's `configs/<arch>/` directory — in vision mode for a
/// `@vision_forward` carrier, decoder mode for `@forward`.
pub fn compile_carrier(
    carrier: PythonCarrier,
    models_dir: &std::path::Path,
) -> syn::Result<proc_macro2::TokenStream> {
    let args = &ForwardArgs::from_carrier(&carrier);
    let mode = if carrier.vision {
        CompileMode::VISION
    } else {
        CompileMode::DECODER
    };
    let name_span = args.span;
    let PythonCarrier { ast, arch_name, .. } = carrier;

    // `pixel_pack = path::to::fn` is OPTIONAL under VISION mode.
    // When unset, the trait's default `pixel_pack` (which delegates
    // to `VisionConfig::patches_from_normalized_chw`, the spatial-
    // merge order Qwen2-VL / Qwen2.5-VL / any arch with the same
    // `patch_size · spatial_merge_size` convention share) is used.
    // Override only for arches with different patch ordering
    // (SigLIP raster, etc.). Decoder mode ignores the arg if set.

    // ── Front end: classify ────────────────────────────────────────
    let classified = classify::classify_with(&ast, mode.prelude)
        .map_err(|e| syn::Error::new(args.span, format!("classify: {e}")))?;

    // ── Load configs + manifest + target ──────────────────────────
    // Shape inference needs both the arch's `weights.json` manifest
    // (for declared weight shapes) and one model's bounds (for
    // numerical-equivalence anchoring). The prober's cross-size
    // validation guarantees every model's bounds resolve the
    // manifest's formulas consistently, so any one model's bounds
    // suffice — we use the first (alphabetical) model.
    // THE arch's declarations: `configs/<arch>/arch.json`. Facts about
    // an arch that aren't derivable from a verbatim HF config.json live
    // in that file, alongside the configs they describe — not in the
    // DSL, which declares only the arch's MATH. Missing file → an arch
    // that declares nothing (every fact derivable), which is the
    // common case.
    let spec = config::load_arch_json(models_dir).map_err(|e| {
        syn::Error::new(
            name_span,
            format!("models_dir `{}`: {e}", models_dir.display()),
        )
    })?;
    let models = match mode.prelude {
        // `#[vision_forward]` configs are verbatim VL-wrapper HF
        // checkpoints; the vision loader derives the `vision_*`
        // bound set from the nested `vision_config` block via the
        // declared `Params` schema instead of the decoder's flat
        // top-level harvest.
        classified::Prelude::Vision => config::load_dir_vision(models_dir, &spec),
        _ => config::load_dir(models_dir, &spec),
    }
    .map_err(|e| {
        syn::Error::new(
            name_span,
            format!("models_dir `{}`: {e}", models_dir.display()),
        )
    })?;
    if models.is_empty() {
        // A `<stem>`/`quant-*`/arch selection filtering out every model in
        // this dir is OK — emit an empty crate so a build scoped to (say)
        // `llama-3.2-1b` succeeds for other arches whose configs
        // weren't selected at all. Only a genuinely empty configs
        // dir (no *.json at all — a misconfigured arch, not a filtering
        // outcome) is a real error.
        let has_any_config = std::fs::read_dir(models_dir).is_ok_and(|rd| {
            rd.flatten().any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.ends_with(".json")
                    && name != "quantizations.json"
                    && name != "weights.json"
                    && !name.ends_with(".overrides.json")
            })
        });
        if has_any_config {
            return Ok(quote! {});
        }
        return Err(syn::Error::new(
            name_span,
            format!("no *.json configs in {}", models_dir.display()),
        ));
    }
    let manifest = weights_manifest::load_or_empty(models_dir).map_err(|e| {
        syn::Error::new(
            name_span,
            format!("weights.json in {}: {e}", models_dir.display()),
        )
    })?;

    // Per-arch declarations (resolved with per-checkpoint drift in
    // `config::resolve_arch_spec`) flow to codegen through the
    // classified program — EXCEPT the vision safetensors layout,
    // which is per-MODEL (sibling variants drift: mlx_vlm repacks
    // rename `visual.*` to `vision_tower.*`) and is read from
    // `model.arch.safetensors` at each codegen call site instead.
    // ⛔ NOT FROM `models[0]` — THESE DRIFT BETWEEN SIBLINGS, exactly as the vision layout above
    // does. gemma-3-12b nests its decoder under `language_model`; gemma-3-1b does not. Taking the
    // prefix from whichever config sorts first gave 1b the 12b prefix in any build containing
    // both, its weight paths then missed the manifest, shape inference fell into reshape
    // RECOVERY, and `apply_reshape_hints` inserted two reshapes per layer — 708 FUF nodes alone
    // against 760 alongside 12b, for the same model. Output stayed correct (the inserted
    // reshapes are no-ops) so nothing caught it; the tape just silently grew.
    //
    // They are set per MODEL below, on that model's own clone of the program.

    // ── Guard G2: DSL weight leaf ending in `_<digit>` → dotted on disk
    // `translate_digit_suffix` (codegen.rs) resolves a leaf like
    // `merger.mlp_0` to the on-disk key `...merger.mlp.0` (dotted
    // submodule index). That is correct for real `nn.Sequential`
    // indices (qwen2-vl/qwen2-5-vl) but a SILENT bug when the on-disk
    // tensor actually uses an underscore. The macro reads no checkpoint
    // and the manifest is DSL-keyed, so it cannot tell the two apart —
    // hence this is a WARNING (eprintln), never a hard error. The
    // rename-intercept test mirrors `safetensors_prefix` (codegen.rs
    // ~88-98) exactly: full-string OR `.{dsl_leaf}` suffix match.
    {
        fn ends_in_underscore_digit(seg: &str) -> bool {
            seg.rfind('_')
                .map(|i| i + 1 < seg.len() && seg[i + 1..].chars().all(|c| c.is_ascii_digit()))
                .unwrap_or(false)
        }
        for id in 0..classified.weights.len() {
            let path = classified.weights.path(classified::WeightId(id as u32));
            let Some(leaf) = path.last() else { continue };
            if !ends_in_underscore_digit(leaf) {
                continue;
            }
            let dotted = path.join(".");
            // Does a rename intercept this exact path? (mirror codegen.rs)
            let handled = classified.weight_leaf_renames.iter().any(|(dsl_leaf, _)| {
                dotted == *dsl_leaf || dotted.ends_with(&format!(".{dsl_leaf}"))
            });
            if handled {
                continue;
            }
            let cut = leaf.rfind('_').unwrap();
            let dotted_disk = {
                let mut segs: Vec<String> = path.to_vec();
                let last = segs.last_mut().unwrap();
                *last = format!("{}.{}", &last[..cut], &last[cut + 1..]);
                segs.join(".")
            };
            eprintln!(
                "warning: [{arch}] DSL weight leaf `{leaf}` ends in `_<digit>`; \
                 translate_digit_suffix (codegen.rs) will resolve it to the on-disk \
                 safetensors key `{dotted_disk}` (dotted submodule index). If the on-disk \
                 tensor literally uses an underscore (`{dotted}`), add a WEIGHT_LEAF_RENAMES \
                 entry in lib.rs mapping a non-digit DSL leaf to it. If the on-disk key really \
                 is a dotted nn.Sequential/submodule index, this is intended — ignore.",
                arch = arch_name,
            );
        }
    }

    // Shape inference may flag reshape-recoverable mismatches (e.g.
    // per-head QK-norm in Qwen3/Gemma3). Catch those, synthesize the
    // Reshape stmts into `classified`, and retry — downstream passes
    // (CFG, FUF, solver, codegen) see a program with explicit reshape
    // tiles. Bounded to one recovery pass: a clean hint set resolves
    // on the second pass. Anything that doesn't is a compiler bug
    // we'd rather surface than loop on.
    // ⛔ RECOVER ON A COPY. This pass exists to VALIDATE (the unresolved-weight diagnostic below);
    // it runs once, against `models[0].bounds`, and it used to synthesize its reshape hints into
    // the SHARED `classified` that every model then clones. So whichever config happened to sort
    // first decided the program structure for all of them: gemma-3-12b needs the per-head QK-norm
    // recovery and gemma-3-1b does not, and `gemma-3-12b-it` sorts before `gemma-3-1b-it`, so a
    // build containing both gave 1b TWO EXTRA RESHAPES PER LAYER — 707 FUF nodes alone against
    // 759 alongside 12b, for the same model with identical bounds. The output stayed correct (the
    // inserted reshapes are no-ops), so only the tape size showed it.
    //
    // Every model re-infers on its own clone below and recovers against its OWN bounds, so the
    // shared program must stay pristine.
    let infer_bounds = &models[0].bounds;
    let mut rep_prog = classified.clone();
    let inferred = match shape::infer(&rep_prog, &manifest, infer_bounds) {
        Ok(inf) => inf,
        Err(shape::ShapeError::ReshapeRecovery { hints }) => {
            shape::apply_reshape_hints(&mut rep_prog, &hints);
            shape::infer(&rep_prog, &manifest, infer_bounds).map_err(|e| {
                // ── Guard G6: enrich the post-reshape-recovery mismatch.
                // Pure diagnostic — same hard error, richer message. When
                // the second-pass failure is a Mismatch, name the producer
                // local(s) recovery rewrote (the prime suspect, e.g. `kvg`),
                // the two conflicting dims, and the sibling-flatten-back
                // hint that this failure mode usually means was missed.
                let detail = match &e {
                    shape::ShapeError::Mismatch { lhs, rhs } => {
                        let producers: Vec<String> = hints
                            .iter()
                            .map(|h| classified.locals.name(h.producer_local).to_string())
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect();
                        format!(
                            "\n  the two dims that won't reconcile: `{lhs}` vs `{rhs}`\
                             \n  reshape recovery rewrote producer local(s): {prod}\
                             \n  HINT: a producer feeding BOTH a weighted per-head norm AND a \
                             weightless `rmsnorm_unit` (k_eq_v V-norm) needs a flatten-back for \
                             the UNIT-norm sibling too — see \
                             shape.rs::rmsnorm_unit_sibling_flatten_backs. If this is a new \
                             reshape pattern, the unit/sibling consumer's output was left in the \
                             per-head `[rows, {lhs}]` view while a downstream consumer expects the \
                             flat `[rows, {lhs} * <heads>]` view.",
                            lhs = shape::show_dim_pub(lhs),
                            rhs = shape::show_dim_pub(rhs),
                            prod = if producers.is_empty() {
                                "(none)".to_string()
                            } else {
                                producers.join(", ")
                            },
                        )
                    }
                    _ => String::new(),
                };
                syn::Error::new(
                    args.span,
                    format!("shape infer (after reshape recovery): {e}{detail}"),
                )
            })?
        }
        Err(e) => {
            return Err(syn::Error::new(args.span, format!("shape infer: {e}")));
        }
    };

    // ── Guard G1: every referenced weight accessor must resolve ───────
    // A referenced `Expr::Weight` whose CLOSED shape is empty (rank 0)
    // was pinned by NEITHER the manifest NOR dataflow — it interned to
    // zero real weight edges (shape.rs installs `Vec::new()` and no op
    // signature grew it). Bundle-struct args (router/experts/moe/mlp/
    // linear_attn) get no dataflow rank, so for them a missing manifest
    // entry is silently fatal downstream (the op gets 0 weight inputs →
    // no solver candidates → a red-herring "tile N op Mul unclaimed").
    // Turn it into a precise source error. The `!shape.is_empty()`
    // exemption is load-bearing: `.bias` args (gemma3-mm) are pinned to
    // rank-1 by dataflow and the manifest declares only the parent stem,
    // so they must stay exempt.
    {
        let mut unresolved: Vec<String> = Vec::new();
        for (wid, shape) in &inferred.weights {
            if !shape.is_empty() {
                continue; // dataflow- or manifest-pinned → fine
            }
            let dotted = classified.weights.path(*wid).join(".");
            if manifest.entries.contains_key(&dotted) {
                continue; // declared (marker family or explicit leaf) → fine
            }
            unresolved.push(dotted);
        }
        if !unresolved.is_empty() {
            unresolved.sort();
            unresolved.dedup();
            let first = unresolved.first().cloned().unwrap_or_default();
            return Err(syn::Error::new(
                args.span,
                format!(
                    "[{arch}] weight accessor(s) not declared in configs/weights.json: {names}. \
                     Each resolved to zero weight edges (no manifest entry and no dataflow-pinned \
                     shape), so the op consuming it has no weight inputs and the solver will fail \
                     downstream with an unrelated unclaimed-tile error. Declare each in \
                     {wjpath} — a marker/bundle family as \
                     `\"{first}\": [\"hidden_size\"]`, or the explicit leaf shape.",
                    arch = arch_name,
                    names = unresolved.join(", "),
                    wjpath = models_dir.join("weights.json").display(),
                    first = first,
                ),
            ));
        }
    }

    // Backend selection via feature flags (compile-time, not runtime)
    #[cfg(feature = "cuda")]
    let target_profile = {
        let target_def =
            scratchy_target_cuda::targets::detect().map_err(|e| syn::Error::new(name_span, e))?;
        target::from_profile_def(target_def)
    };

    // ⛔ DETECTION ONLY, AND THAT IS THE WHOLE POINT. A metal build runs no solver, so the
    // `TargetProfile` this used to convert the probe into was built and dropped — a cost model
    // for a search that is not compiled in. What still has to happen is the PROBE: it fails the
    // build when no Metal device is present.
    //
    // Compile-time profile only — does NOT require Metal 4 (unlike the runtime
    // `detect_device()`), so models build on non-Metal-4 hosts.
    #[cfg(feature = "metal")]
    scratchy_target_metal::device::detect_metal_profile().ok_or_else(|| {
        syn::Error::new(
            name_span,
            "No Metal device detected. Metal backend requires macOS with Apple Silicon.",
        )
    })?;

    // Spyre/KTIR is a host backend with no GPU cost model. The solver only
    // needs a profile to cover the FUF with spyre's own claim impls, which the
    // KTIR emitter then lowers — so the profile is honestly `Backend::Spyre`
    // (no Cuda masquerade), carries no GPU hardware spec, and uses the universal
    // cost table (analytic fallback). The op set is driven by spyre's registered
    // impls, not by the backend discriminant.
    // Spyre is tape-scheduled too, so nothing consumes a cost profile there;
    // the binding is kept (and unused) so the shape stays visible next to the
    // cuda and metal arms.
    #[cfg(feature = "spyre")]
    let _target_profile = target::TargetProfile {
        name: "spyre".to_string(),
        source_path: std::path::PathBuf::new(),
        backend: target::Backend::Spyre,
        peak_tflops_fp16: 100.0,
        memory_bandwidth_gbps: 1000.0,
        backend_spec: target::BackendSpec::Spyre,
        cost_table: target::CostTable::universal(),
    };

    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "spyre")))]
    compile_error!(
        "scratchy-forward-compiler-macro requires a backend feature: 'cuda', 'metal', or 'spyre'"
    );

    let library = impl_lib::starter_library();

    // Stable rebuild-on-JSON-change: emit `const _: &str =
    // include_str!("<abs path>");` for every file the macro read.
    // Rustc treats `include_str!` paths as source-dependency inputs
    // and cargo rebuilds the caller when any of them change. Works
    // on stable; no build.rs or nightly feature required.
    let mut tracked: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut tracked_paths: std::collections::BTreeSet<std::path::PathBuf> =
        std::collections::BTreeSet::new();
    for m in &models {
        tracked_paths.insert(m.source_path.clone());
        for extra in &m.extra_tracked_paths {
            tracked_paths.insert(extra.clone());
        }
    }
    // Arch-level metadata files: the weights manifest and the
    // quantization preset declaration. They influence compiled
    // output (manifest shapes, synthesized variant set) without
    // being per-variant source paths.
    let quantizations_path = models_dir.join("quantizations.json");
    if quantizations_path.exists() {
        tracked_paths.insert(quantizations_path);
    }
    // De-dupe before emitting; multiple variants share preset /
    // override paths. Target profile is no longer a separate file —
    // it's compiled into scratchy-target-cuda, so cargo's normal
    // dep edge invalidates this crate when the profile changes.
    for p in &tracked_paths {
        let s = p.to_string_lossy().into_owned();
        let lit = syn::LitStr::new(&s, proc_macro2::Span::call_site());
        tracked.push(quote! { const _: &str = include_str!(#lit); });
    }

    // ── Per-model × per-workload pipeline ─────────────────────────
    //
    // Two passes: solve every variant first (gather fuf + sfufs + loops
    // per variant), then group variants whose emitted forward-fn
    // bodies would be byte-identical and emit each group's canonical
    // variant in full. Non-canonical variants emit `pub use`
    // re-exports of the canonical's forward fns + their own variant-
    // specific `Weights` type alias / `load` / `fingerprint_matches`.
    //
    // The dedup key captures every input that flows into the emitted
    // forward fn body: model bounds (baked as int literals in emit),
    // scalars (baked as float literals), and the SFUF-per-workload-
    // point (which Impl runs at each subgraph → which `emit_call`
    // output ends up in the body). Two variants that share this
    // tuple compile to byte-identical forward fns — across AWQ /
    // GPTQ / CT of the same (arch, size), for instance, because
    // they all resolve to the same `Marlin*Impl` family and the
    // quant knobs (`desc_act`, `sym`, etc.) only change the
    // per-variant `load`, never the forward.
    struct SolvedModel<'a> {
        model: &'a config::ModelParams,
        /// Shapes inferred from THIS model's bounds — never the
        /// arch-level `models[0]` inference (see the unroll site).
        inferred: shape::Inferred,
        /// The classified body THIS model was lowered against, with
        /// its own reshape recovery applied. Emission must read the
        /// same program the FUF was built from.
        prog: classified::Program,
        fuf: fuf::Fuf,
        sfufs: crate::assignment::WorkloadAssignments,
        loops: schedule::WorkloadLoops,
        stub_items: proc_macro2::TokenStream,
        /// Tensor-parallel world size this model was solved at. The
        /// (variant × tp) fanout constructs one SolvedModel per
        /// (model, tp_world_size) pair. At `tp_world_size = 1` (every
        /// emission when `CARGO_FEATURE_NCCL` is unset) sharding is
        /// identity. Threaded into `dedup_signature` so the
        /// canonical-equivalence-class hash keeps each tp on its own
        /// canonical, and into the `tp_lowering::insert_all_reduces`
        /// call so the FUF receives a row-parallel AllReduce only
        /// when it should.
        tp_world_size: u8,
        /// Per-(model, tp) Rust-ident form of the emitted module. At
        /// tp=1 this is `model.name` verbatim (preserves the
        /// `scratchy_models::<arch>::<model>::Weights` path callers
        /// already use); at tp>1 it gets a `_tp{N}` suffix.
        mod_name: String,
        /// Per-(model, tp) HF-form stem used as the alphabetical
        /// canonical-selection key inside `compute_canonical_variants`.
        /// At tp=1 = `model.source_stem`; at tp>1 it's
        /// `format!("{}_tp{}", model.source_stem, tp_world_size)` so
        /// every member of a tp-N equivalence class shares the suffix
        /// and within-class ordering is preserved.
        canon_stem: String,
    }

    impl HasSolvedSig for SolvedModel<'_> {
        fn dedup_signature(&self) -> String {
            let mut parts: Vec<String> = Vec::new();
            // Bounds get baked as integer literals in the emitted
            // forward body (`ctx.bound("hidden_size")` expands to the
            // concrete number at macro expansion). Variants with
            // differing bounds produce different literal output.
            for (k, v) in &self.model.bounds {
                parts.push(format!("b:{k}={v}"));
            }
            // Scalars get baked as float literals (attention_scale_for,
            // softcap). Same reasoning.
            for (k, v) in &self.model.scalars {
                parts.push(format!("s:{k}={v}"));
            }
            // `tie_word_embeddings` routes lm_head through
            // `FieldLoad::LinearTiedToEmbedding` (no safetensors
            // read) vs `FieldLoad::LinearDense` — two different
            // bodies, so variants with different tie settings
            // can't share a compiled `load_with`.
            parts.push(format!("t:{}", self.model.tie_word_embeddings));
            // `rope_scaling` (short/long_factor, type, orig_max) is
            // baked into `RotaryCache::new_*` as literal arguments in
            // `load_with`. Two variants with identical bounds +
            // scalars but different `rope_scaling` (Phi-4-mini-instruct
            // vs Phi-4-mini-reasoning: all-1.0 short_factor vs the
            // non-trivial vector) MUST NOT share a canonical — the
            // shim would bake the canonical's rotary for both.
            parts.push(format!("r:{}", self.model.rope_scaling_hash.unwrap_or(0)));
            // Resolved per-arch declaration (+ per-checkpoint drift):
            // safetensors layout, decoder prefix, scale dtype, rope
            // style, … all bake into the emitted loader / glue
            // (`vision_tower.*` vs `visual.*` paths, `_s_bf16_`
            // symbol arms). Two variants with identical bounds but
            // drifted specs (qwen2-vl-2b-instruct vs the
            // mlx_vlm-repacked qwen2-vl-2b-mlx) MUST NOT share a
            // canonical — folding them emits conflicting
            // `VisionArchWeights` impls (E0119) or, worse, one
            // variant silently loading the other's paths.
            parts.push(format!("a:{:?}", self.model.arch));
            parts.push(dedup_quant_sig(
                self.model.quantization.as_ref().map(|qc| &qc.method),
            ));
            // Tensor-parallel canonicalization axis. The
            // SolvedModel.tp_world_size field is the per-(model, tp)
            // discriminator; until task #7's outer-loop fanout lands
            // every SolvedModel has `tp_world_size = 1`, so every
            // dedup string still ends in `tp:1`. When the fanout
            // turns on, two SolvedModels of the same model at tp=1
            // vs tp=2 hash to different signatures and pick separate
            // canonicals — pinned by `tp_world_sizes_pick_separate_
            // canonicals`.
            parts.push(dedup_tp_sig(self.tp_world_size));
            // SFUF per (num_tokens, sk_bucket) point: which Impl runs
            // at each subgraph. Identical SFUFs → each impl's
            // `emit_call` produces identical output at identical
            // positions in the body.
            let mut wps: Vec<_> = self.sfufs.per_workload.iter().collect();
            wps.sort_by_key(|(wp, _)| (wp.num_tokens, wp.sk_bucket));
            for (wp, sf) in wps {
                let mut impls: Vec<(u32, u32)> =
                    sf.impls.iter().map(|(sg, i)| (sg.0, i.0)).collect();
                impls.sort();
                parts.push(format!("w:{}-{}-{:?}", wp.num_tokens, wp.sk_bucket, impls));
            }
            parts.join("|")
        }
        fn source_stem(&self) -> &str {
            &self.canon_stem
        }
        fn model_name(&self) -> &str {
            &self.mod_name
        }
    }

    // Compile-time tp set. The per-arch
    // crate's `nccl` cargo feature transitively enables
    // `scratchy-forward-compiler-macro/nccl`, recompiling THIS proc-macro with
    // its own `nccl` feature on. `cfg!(feature = "nccl")` then reads
    // `true` at expand time and the macro fans out every variant
    // over `{1, 2, 4, 8}`. Without it, only tp=1 emits —
    // byte-identical to the pre-fanout build. (Cargo caches the
    // proc-macro per-feature-set, so consumers without nccl still
    // get the fast tp=1-only macro.)
    let nccl_enabled = cfg!(feature = "nccl");
    // Vision macros opt out of the tp fanout entirely — the encoder
    // is replicated in v1 (no AllReduce / AllGather lowering, no
    // sharded weight surface). Decoder macros keep the existing
    // {1, 2, 4, 8} set under nccl-enabled, [1] otherwise.
    let tp_set: &[u8] = if mode.enable_tp_fanout && nccl_enabled {
        &[1, 2, 4, 8]
    } else {
        &[1]
    };

    let mut solved: Vec<SolvedModel<'_>> = Vec::with_capacity(models.len() * tp_set.len());

    for &tp_world_size in tp_set {
        for model in &models {
            // Skip (variant, tp) tuples whose column-parallel dims don't
            // divide evenly. KV replication when `num_kv_heads < tp_size`
            // is task #6's loader-sharding work — until then, an
            // indivisible KV head count drops the (variant, tp) tuple
            // from emission rather than baking a `NUM_KV_HEADS = 0`
            // canonical that would silently fail at runtime. Hits e.g.
            // SmolLM-135M (3 KV heads) at tp ∈ {2, 4, 8}, Llama 3.2-1B
            // (8 KV heads) at tp=16 (not in the default set).
            if tp_world_size > 1 {
                let na = *model.bounds.get("num_attention_heads").unwrap_or(&1);
                let nkv = *model.bounds.get("num_key_value_heads").unwrap_or(&1);
                let inter = *model.bounds.get("intermediate_size").unwrap_or(&1);
                let tp = tp_world_size as u64;
                if na % tp != 0 || nkv % tp != 0 || inter % tp != 0 {
                    eprintln!(
                        "  scratchy · {variant:<30} · skip tp={tp_world_size} (heads={na}/{nkv}, inter={inter} not divisible)",
                        variant = model.source_stem,
                    );
                    continue;
                }
            }

            // Per-(model, tp) ident form. tp=1 keeps the existing names
            // verbatim so callers' `scratchy_models::<arch>::<model>::Weights`
            // paths stay valid; tp>1 gets a `_tp{N}` suffix.
            let mod_name = if tp_world_size == 1 {
                model.name.clone()
            } else {
                format!("{}_tp{}", model.name, tp_world_size)
            };
            let canon_stem = if tp_world_size == 1 {
                model.source_stem.clone()
            } else {
                format!("{}_tp{}", model.source_stem, tp_world_size)
            };

            // 🛑 PER-MODEL shape inference, over a PER-MODEL copy of the
            // classified body.
            //
            // The arch-level `inferred` is computed from
            // `models[0].bounds` — whichever config sorts FIRST — and
            // feeding it to a sibling's unroll bakes the first config's
            // geometry into that sibling's FUF. Observed: compiling
            // `gemma-3-1b` (text-only, hidden 1152 / 26 layers) beside
            // `gemma-3-4b` drove the 4b's FUF from 992 tiles to 924
            // with a reordered q/k norm pair, and the 4b then emitted
            // fluent garbage.
            //
            // Reshape RECOVERY has the same shape of bug one level
            // down: it MUTATES the program, and models of one arch do
            // not need the same hints (the 1b needs none; the 4b needs
            // two). Recovering into the shared body would let whichever
            // model recovered first rewrite its siblings. So each model
            // recovers into its OWN clone, and that clone is what its
            // cfg / FUF / solve / emit all read.
            let mut model_prog = classified.clone();
            if matches!(mode.prelude, classified::Prelude::Decoder) {
                model_prog.decoder_safetensors_prefix = model.arch.decoder_prefix.clone();
            }
            model_prog.weight_leaf_renames = model.arch.weight_leaf_renames.clone();
            let model_inferred = match shape::infer(&model_prog, &manifest, &model.bounds) {
                Ok(inf) => inf,
                Err(shape::ShapeError::ReshapeRecovery { hints }) => {
                    shape::apply_reshape_hints(&mut model_prog, &hints);
                    shape::infer(&model_prog, &manifest, &model.bounds).map_err(|e| {
                        syn::Error::new(
                            args.span,
                            format!(
                                "shape infer after reshape recovery [{}]: {e}",
                                model.source_stem
                            ),
                        )
                    })?
                }
                Err(e) => {
                    return Err(syn::Error::new(
                        args.span,
                        format!("shape infer [{}]: {e}", model.source_stem),
                    ));
                }
            };
            let model_cfg = cfg::build_cfg(&model_prog, model).map_err(|e| {
                syn::Error::new(args.span, format!("cfg [{}]: {e}", model.source_stem))
            })?;
            let mut model_fuf = fuf::unroll(&model_cfg, &model_inferred).map_err(|e| {
                syn::Error::new(args.span, format!("unroll [{}]: {e}", model.source_stem))
            })?;
            model_fuf.annotate_storage_formats(&model_prog, model);
            // Tensor-parallel lowering pass. At tp=1 (every existing
            // SolvedModel until task #7's canonical fanout lands) this is
            // a strict no-op — the FUF flowing into the solver is
            // byte-identical to single-rank builds. Vision encoders
            // skip the pass entirely (no AllReduce/AllGather; replicated
            // in v1 per the G.3 handoff resolution).
            if mode.apply_tp_lowering {
                tp_lowering::insert_all_reduces(&mut model_fuf, &model_prog, tp_world_size);
                tp_lowering::insert_lm_head_allgather(&mut model_fuf, &model_prog, tp_world_size);
            }
            // Multimodal post-Embed splice. Unconditional at every tp
            // (including tp=1) — runtime no-op for text-only batches.
            // Must run AFTER `insert_all_reduces` so at tp>1 the
            // splice sits on the reduced embedding (not each rank's
            // partial masked-gather, which the pre-refactor inline
            // splice inside `Instruction::Embed::eval` mistakenly
            // overwrote). Vision encoders skip — splice belongs on
            // the decoder side, not the encoder side.
            // MmEmbedSplice's only matcher is the CUDA-only
            // `MmEmbedSpliceImpl` (D2D-copy via cuMemcpyDtoDAsync). Under
            // `--features metal` the impl pool can't claim the synthesized
            // splice node, so the solver explodes with "no Impl matched
            // Inserts a post-Embed `OpKind::MmEmbedSplice` that D2D-copies
            // the vision-encoder embeddings into the image-placeholder
            // rows. Text-only batches make it a runtime no-op
            // (`embed_patches` empty) on both backends. Now wired on metal
            // too (metal `MmEmbedSpliceImpl` + `SpliceMmEmbeds` lowering).
            #[cfg(any(feature = "cuda", feature = "metal"))]
            if mode.apply_mm_splice {
                tp_lowering::insert_mm_splices(&mut model_fuf, &model_prog);
            }
            // Vision-prelude `pixels` extern → tile materialization
            // (G.5.e.1). Synthesizes a single `OpKind::LoadPixels`
            // node and rewrites every downstream `FufInput::Extern`
            // referencing pixels to read its slot 0. No-op when the
            // body has no pixels reference. Vision-only — decoder
            // bodies have no `Pixels` extern (the prelude split
            // makes the two extern sets disjoint).
            if mode.prelude == classified::Prelude::Vision {
                vision_lowering::materialize_pixels(&mut model_fuf);
                // Qwen3.5-VL `pos_embeds` extern → tile (sibling of the
                // pixels materialization above). No-op for towers without
                // a learned positional embedding.
                vision_lowering::materialize_pos_embeds(&mut model_fuf);
            }

            // At tp>1, the runtime weight tensors are per-rank shards
            // (column-parallel q/k/v/gate/up halve dim 0; row-parallel
            // o/down halve dim 1). The codegen-baked weight shapes
            // (`assert_weight_shape` checks them at every Gemm-class
            // eval) must match those per-rank shapes, so the solver
            // and `gemm_nk_from_fuf` (which evaluates symbolic
            // `Shape` against bounds) need a sharded view of
            // `num_attention_heads / num_key_value_heads /
            // intermediate_size`. tp=1 keeps the unsharded bounds
            // verbatim — byte-identical to the pre-fanout build.
            #[cfg(feature = "cuda")]
            let solve_bounds = if tp_world_size > 1 {
                let mut b = model.bounds.clone();
                let tp = tp_world_size as u64;
                for k in [
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                ] {
                    if let Some(v) = b.get_mut(k) {
                        *v = (*v / tp).max(1);
                    }
                }
                b
            } else {
                model.bounds.clone()
            };
            let t_solve = std::time::Instant::now();
            // M2b step 3: a tape-scheduled arch does not run the
            // solver at all. Its emitted artifacts — stream,
            // schedule, coloring, accessors, Weights struct, bucket
            // folding — all come from the shared tape (each gated
            // equal to instruction selection's before this switch).
            // The workload POINTS are still needed as map keys, so
            // an empty `Assignment` per point stands in; nothing on
            // the pilot path reads their contents.
            #[cfg(any(feature = "metal", feature = "spyre"))]
            let tape_pilot = true;
            #[cfg(not(any(feature = "metal", feature = "spyre")))]
            let tape_pilot = false;
            let sfufs = if tape_pilot {
                let sk_eff: Vec<u64> = if args.sk_buckets.is_empty() {
                    vec![0]
                } else {
                    args.sk_buckets.clone()
                };
                let mut per_workload = std::collections::BTreeMap::new();
                for m in &args.workloads {
                    for sk in &sk_eff {
                        per_workload.insert(
                            crate::assignment::WorkloadPoint {
                                num_tokens: *m,
                                sk_bucket: *sk,
                            },
                            crate::assignment::Assignment {
                                cover: Default::default(),
                                impls: Default::default(),
                                predicted_us: 0.0,
                            },
                        );
                    }
                }
                crate::assignment::WorkloadAssignments { per_workload }
            } else {
                #[cfg(feature = "cuda")]
                {
                    solver::solve_with_arch_filter(
                        &model_fuf,
                        &library,
                        &target_profile,
                        Some((&model_prog, model)),
                        &model_inferred,
                        &solve_bounds,
                        &args.workloads,
                        &args.sk_buckets,
                        tp_world_size,
                    )
                    .map_err(|e| {
                        syn::Error::new(args.span, format!("solve [{}]: {e}", model.source_stem))
                    })?
                }
                #[cfg(not(feature = "cuda"))]
                {
                    // Unreachable on metal and spyre: `tape_pilot` above is
                    // unconditionally true there, so this arm never runs — which
                    // is what lets the solver be absent from the build entirely.
                    unreachable!("metal builds are always tape-scheduled")
                }
            };
            // Only `cost::refresh_predicted_us` mutates this, and it is not
            // compiled on a metal-only build.
            #[cfg(feature = "cuda")]
            let mut sfufs = sfufs;
            let d_solve = t_solve.elapsed();

            let loops = schedule::schedule_workloads(&model_fuf, &sfufs);
            #[cfg(feature = "cuda")]
            cost::refresh_predicted_us(
                &model_fuf,
                &mut sfufs,
                &loops,
                &library,
                &target_profile,
                &solve_bounds,
            );

            // PD-wavefront macro-emission moved into `codegen::emit_model`
            // (`dump_wavefront_mega`): the decode bucket's `weight_slots` —
            // the real weight-locator source — only exists after lowering +
            // loop compression, which happens inside `emit_model`, not in
            // this pre-emit solve drive.

            let max_waves = loops
                .per_workload
                .values()
                .map(|l| l.num_waves())
                .max()
                .unwrap_or(0);
            // Kernel-class summary lifted from HEAD (`9189c3147`).
            // Classification must be TOTAL: any impl name that doesn't
            // map to a known class fails the build, prompting us to
            // add the kernel to the explicit table.
            //
            // Three semantic axes:
            //   - attention backend: fa2 / fi / mla
            //   - GEMM backend (pure or fused-with-GEMM): cublas /
            //     cutlass / marlin. fp8 folds into cutlass (uses
            //     `cutlass_scaled_mm_with_bias`); bnb4 folds into
            //     cublas (dequant + cuBLAS matmul).
            //   - non-gemm: kernels that are neither attention nor
            //     GEMM-bearing — element-wise, reshapes, residual
            //     adds, standalone norms. Surfaced because their
            //     existence is usually a "why didn't we fuse this?"
            //     signal.
            const CLASS_LABELS: [&str; 8] = [
                "fa2", "fi", "mla", "cublas", "cutlass", "marlin", "non-gemm", "comm",
            ];
            // Names that are non-gemm despite a `fused_` prefix
            // (norm-side fusions with no matmul).
            const NON_GEMM_NAMES: &[&str] = &[
                "embed_ref",
                "rmsnorm_ref",
                "add_ref",
                "reshape_ref",
                "rope_append_ref",
                "silu_ref",
                "elem_mul_ref",
                "scalar_mul_inplace",
                "scalar_offset_rms_norm",
                "tanh_softcap_inplace",
                // Gemma4 singletons: unit-gain rmsnorm (v_norm) and
                // the per-layer [1]-weight multiply (layer_scalar).
                "rmsnorm_unit",
                "scalar_weight_mul",
                // Gemma4 post-FFN tail fusion (rmsnorm+add+scalar_mul
                // in one kernel — norm-side, no matmul).
                "metal_norm_add_scalar_mul_f16",
                "metal_norm_add_scalar_mul_bf16",
                // Gemma4 pre-attn tail fusion (q/k norms + v unit-norm
                // inside the rope dispatch — norm-side, no matmul).
                "metal_rope_append_normed_f16",
                "metal_rope_append_normed_bf16",
                "softcap",
                "nosoftcap",
                "deepseek_moe_ref",
                "deepseek_moe_fp8_block",
                "deepseek_moe_ggml",
                "fused_moe_ref",
                "shared_fused_moe_ref",
                "cuda_gemma_moe",
                // Qwen3.5 hybrid ops — host-callback dispatch wrappers whose
                // internals (conv1d / recurrent scan / deinterleave / sigmoid
                // gate) carry no matmul; the projections are separate DSL
                // gemms. Same accounting class as the `*_ref` MoE siblings.
                "gated_delta_net_ref",
                "gate_split_ref",
                "gate_apply_ref",
                "gate_scale_ref",
                // Metal MoE Impls. Same "host-callback dispatch
                // wrapper, internal compute steps already classified
                // (Gemm via metal_gemm_, gather_qmv via
                // metal_affine_qmm_)" shape as the cuda *_ref
                // siblings — bucket them under non-gemm for
                // accounting.
                #[cfg(feature = "metal")]
                "metal_fused_moe",
                #[cfg(feature = "metal")]
                "metal_shared_fused_moe",
                #[cfg(feature = "metal")]
                "metal_gemma_moe",
                "fused_add_rms_norm",
                "fused_add_rms_norm_with_offset",
                "mean_sub_rms_norm",
                "mean_sub_rms_norm_bias_add",
                // Vision-side unary elementwise ops (G.4). Shape-
                // preserving, no matmul — same class as the text-side
                // `scalar_mul_inplace` / `tanh_softcap_inplace` lines.
                "quick_gelu_inplace",
                "gelu_erf_inplace",
                "gelu_tanh_inplace",
                // Metal kernels (Phase 5.F: the proc-macro now runs
                // under `--features metal`, so the classifier sees
                // these names alongside the CUDA ones). Hand-rolled
                // norm / elementwise / fused-MLP / RoPE — same shape
                // class as the CUDA `*_ref` siblings, just emitting
                // MSL instead of CUDA. `metal_attention_*` and
                // `metal_gemm_*` get their own prefix arms below
                // (fa2 / cutlass-equivalent). Gated on `metal` so the
                // CUDA build doesn't carry dead names in its classifier.
                #[cfg(feature = "metal")]
                "metal_add_f16",
                #[cfg(feature = "metal")]
                "metal_add_bf16",
                #[cfg(feature = "metal")]
                "metal_embed_f16",
                #[cfg(feature = "metal")]
                "metal_embed_bf16",
                #[cfg(feature = "metal")]
                "metal_affine_embed_f16",
                #[cfg(feature = "metal")]
                "metal_affine_embed_bf16",
                #[cfg(feature = "metal")]
                "metal_reshape",
                #[cfg(feature = "metal")]
                "metal_bias_add_f16",
                #[cfg(feature = "metal")]
                "metal_bias_add_bf16",
                #[cfg(feature = "metal")]
                "metal_rmsnorm_f16",
                #[cfg(feature = "metal")]
                "metal_rmsnorm_bf16",
                #[cfg(feature = "metal")]
                "metal_fused_add_rmsnorm_f16",
                #[cfg(feature = "metal")]
                "metal_fused_add_rmsnorm_bf16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_silu_mul_f16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_silu_mul_bf16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_gelu_mul_f16",
                #[cfg(feature = "metal")]
                "metal_fused_gate_up_gelu_mul_bf16",
                #[cfg(feature = "metal")]
                "metal_rope_append_f16",
                #[cfg(feature = "metal")]
                "metal_rope_append_bf16",
                // CommandR and other models use the interleaved rope
                // variant; same shape class as the regular rope_append.
                #[cfg(feature = "metal")]
                "metal_rope_append_interleaved_f16",
                #[cfg(feature = "metal")]
                "metal_rope_append_interleaved_bf16",
                #[cfg(feature = "metal")]
                "metal_fatrelu_f16",
                // Metal counterparts of the CUDA `scalar_mul_inplace`
                // and `tanh_softcap_inplace` non-gemm in-place
                // mutators. Same kernel class — bandwidth-bound
                // elementwise unary.
                #[cfg(feature = "metal")]
                "metal_scalar_mul_f16",
                #[cfg(feature = "metal")]
                "metal_scalar_mul_bf16",
                #[cfg(feature = "metal")]
                "metal_tanh_softcap_f16",
                #[cfg(feature = "metal")]
                "metal_tanh_softcap_bf16",
                // Vision-prelude pixels materialization (G.5.e.1).
                // Synthesized by `vision_lowering::materialize_pixels`;
                // emits a single D2D copy that wraps `ctx.fwd.pixels`
                // into a tile-table OwnedTensor. Not a compute kernel.
                "load_pixels",
                // Qwen3.5-VL pos_embeds materialization (sibling of
                // load_pixels) — wraps `ctx.fwd.pos_embeds` into a tile.
                "load_pos_embeds",
                // Vision-side varlen attention + vision rope.
                // Shape-preserving non-gemm primitives.
                "varlen_attention",
                "vision_rope",
                // Row-permutation gather (G.6.4). Used by Qwen2.5-VL's
                // window-attention dispatch — same class as the other
                // memory-bound vision primitives.
                "embedding_gather",
                // 2-D average pool over the patch grid (G.7(b)). Used
                // by Gemma3-MM's SigLIP→text projector to reduce the
                // 64×64 patch grid down to 16×16 = 256 tokens. Memory-
                // bound with one thread per output cell; non-gemm.
                "avg_pool_2d",
                // Vision learned positional embedding lookup (G.7(c.1)).
                // Reuses the decoder's `embedding_gather_masked` kernel;
                // same memory-bound class as `embed_ref`.
                "pos_embed_ref",
            ];
            let mut classes_used = [false; 8];
            let mut unknown_names: std::collections::BTreeSet<&'static str> =
                std::collections::BTreeSet::new();
            for assignment in sfufs.per_workload.values() {
                for impl_id in assignment.impls.values() {
                    let name = library.get(*impl_id).name();
                    // Most kernel-name prefixes here are CUDA-specific
                    // (flashinfer/mla/cutlass/marlin/fp8/bnb4/ggml/cublas-via-fused_/
                    // NCCL collectives + the multimodal D2D splice).
                    // Gating each behind `cfg!(feature = "cuda")` keeps
                    // the metal-only build's classifier from carrying
                    // dead arms and prevents a hypothetical
                    // metal-emitted impl that happens to start with
                    // `cutlass` etc. from being silently mis-classed.
                    let bucket = if cfg!(feature = "cuda") && name.starts_with("flashinfer") {
                        Some(1) // fi
                    } else if cfg!(feature = "cuda") && name.starts_with("flash_attention_3") {
                        // FA3 paged decode (Hopper-native, sm_90+).
                        // Same class as FlashInfer for the cost-model
                        // mix line — both are persistent-scheduler
                        // attention kernels and exclude each other in
                        // the per-cell solver pick.
                        Some(1) // fi
                    } else if name.starts_with("mla_") {
                        // MLA singletons (`mla_split_ref`, `mla_attention_ref`)
                        // and `DeepSeekMoeRefImpl`-family are registered under
                        // both backends — runtime support diverges, but the
                        // classifier just buckets by name for the build-time
                        // mix line.
                        Some(2) // mla
                    } else if name.starts_with("attention_")
                        || name.starts_with("sliding_attention_")
                        || name.starts_with("fa2_")
                        || name == "encoder_attention"
                        || (cfg!(feature = "metal") && name.starts_with("metal_attention_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_sliding_attention_"))
                    {
                        Some(0) // fa2
                    } else if cfg!(feature = "cuda") && name.starts_with("marlin") {
                        Some(5) // marlin
                    } else if cfg!(feature = "cuda") && name.starts_with("fp8") {
                        Some(4) // cutlass (fp8 uses cutlass_scaled_mm)
                    } else if cfg!(feature = "cuda")
                        && (name.starts_with("bnb4") || name.starts_with("ggml"))
                    {
                        // cublas: bnb4 dequant + cuBLAS matmul; ggml
                        // dequant_mul_mat_vec at decode + cuBLAS at prefill.
                        Some(3)
                    } else if (cfg!(feature = "cuda") && name.starts_with("cutlass"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_gemm_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_affine_qmm_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_nvfp4_qmm_"))
                        || (cfg!(feature = "metal") && name.starts_with("metal_synth_"))
                    {
                        // Metal GEMM is currently routed through MPS
                        // matmul2d (see scratchy-target-metal::gemm);
                        // metal int4 GEMM routes through the
                        // qmv/qmm_t kernels (see scratchy-target-metal::
                        // quantized). Both treated as cutlass-equivalent
                        // for class accounting — same "specialized
                        // matmul tile" shape from the cost-model's
                        // perspective.
                        Some(4) // cutlass
                    } else if NON_GEMM_NAMES.contains(&name) {
                        Some(6) // non-gemm
                    } else if name.starts_with("ktir_") {
                        // `ktir_*` claim impls are host-callback placeholders that
                        // let the solver cover each op; the real kernel is the
                        // embedded KTIR bundle, so there is no host GEMM/attention
                        // kernel to account for in this build-time mix line. (No
                        // cuda/metal impl uses this prefix, so the arm is inert
                        // under those backends — no need to name a target.)
                        Some(6) // non-gemm
                    } else if cfg!(feature = "cuda")
                        && (name == "all_reduce" || name == "all_gather")
                    {
                        // Tensor-parallel collectives inserted by
                        // `tp_lowering` at tp>1 (AllReduce after
                        // row-parallel gemms + vocab-parallel embed;
                        // AllGather after lm_head). Maps to NCCL —
                        // semantically distinct from compute kernels.
                        Some(7) // comm
                    } else if (cfg!(feature = "cuda") || cfg!(feature = "metal"))
                        && name == "mm_embed_splice"
                    {
                        // Multimodal post-Embed D2D splice inserted by
                        // `tp_lowering::insert_mm_splices`. Not a
                        // compute kernel — runs a sequence of
                        // memcpy_dtod_async calls per image placeholder.
                        // Bucketed alongside the comm kernels since
                        // they share the "not a GEMM / not a normal
                        // per-token kernel" shape.
                        Some(7) // comm
                    } else if name.starts_with("fused_") || name == "gemm_ref" {
                        // `fused_gemm_bias` (qwen2 K/V) and the gemma
                        // fusion families (`fused_add_rms_norm`,
                        // `fused_add_rms_norm_with_offset`,
                        // `scalar_offset_rms_norm`) live in both backends
                        // now. cuda routes through cuBLAS gemm_bias; metal
                        // routes through its own GEMM path. Same
                        // build-time class for accounting.
                        Some(3) // cublas / cublas-equivalent
                    } else {
                        None
                    };
                    match bucket {
                        Some(b) => classes_used[b] = true,
                        None => {
                            unknown_names.insert(name);
                        }
                    }
                }
            }
            if !unknown_names.is_empty() {
                return Err(syn::Error::new(
                    args.span,
                    format!(
                        "[{}] kernel-class summary: no class assigned for impl name(s): {}. \
                         Add a class (or extend an existing prefix) in lib.rs.",
                        model.source_stem,
                        unknown_names.iter().copied().collect::<Vec<_>>().join(", "),
                    ),
                ));
            }
            let kernel_mix: String = classes_used
                .iter()
                .zip(CLASS_LABELS.iter())
                .filter(|(seen, _)| **seen)
                .map(|(_, label)| format!(" {label}"))
                .collect();
            // Per-M scoring: gated on SCRATCHY_DEBUG so the default
            // build log stays terse (one line per (variant, tp)).
            let per_m_part: String = if scratchy_debug() {
                let sk_axis_active = sfufs.per_workload.keys().any(|wp| wp.sk_bucket != 0);
                sfufs
                    .per_workload
                    .iter()
                    .map(|(wp, a)| {
                        if sk_axis_active {
                            format!(
                                " M={}sk={}→{}",
                                wp.num_tokens,
                                wp.sk_bucket,
                                fmt_us(a.predicted_us)
                            )
                        } else {
                            format!(" M={}→{}", wp.num_tokens, fmt_us(a.predicted_us))
                        }
                    })
                    .collect()
            } else {
                String::new()
            };
            let solve_ms_part = if scratchy_debug() {
                format!(" · {:>3} ms", d_solve.as_millis())
            } else {
                String::new()
            };
            eprintln!(
                "  scratchy · {variant:<30} · {tiles:>4} tiles · {waves:>3} waves{solve_ms_part} · tp={tp:<1} ·{kernel_mix}{per_m_part}",
                variant = canon_stem,
                tp = tp_world_size,
                tiles = model_fuf.len(),
                waves = max_waves,
            );

            let stub_items = emit_model_stub_items(&model_fuf, &sfufs, &loops);

            solved.push(SolvedModel {
                model,
                inferred: model_inferred,
                prog: model_prog,
                fuf: model_fuf,
                sfufs,
                loops,
                stub_items,
                tp_world_size,
                mod_name,
                canon_stem,
            });
        }
    }

    // Cross-variant forward-fn dedup. Key each variant by its
    // (bounds, scalars, SFUF-per-workload-point) tuple and pick the
    // earliest-by-source_stem variant in each equivalence class as
    // the canonical. Non-canonical variants emit thin shims that
    // `pub use` the canonical's forward fns (one-line re-exports —
    // rustc doesn't re-monomorphize them, so LLVM optimization work
    // scales with `#distinct equivalence classes`, not
    // `#variants`).
    let canonical_for: std::collections::HashMap<usize, Ident> =
        compute_canonical_variants(&solved);

    let mut per_model_ts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut arch_dispatch_arms: Vec<DispatchArm> = Vec::new();

    for (idx, sm) in solved.iter().enumerate() {
        let model_mod = Ident::new(&sm.mod_name, Span::call_site());
        let canonical_ident = &canonical_for[&idx];
        let is_canonical = *canonical_ident == model_mod;
        let canonical_override = if is_canonical {
            None
        } else {
            Some(canonical_ident.clone())
        };

        let codegen_items = codegen::emit_model(
            &sm.prog,
            sm.model,
            &sm.fuf,
            &sm.sfufs,
            &sm.loops,
            &library,
            &manifest,
            &sm.inferred,
            canonical_override.as_ref(),
            sm.tp_world_size,
            mode.emit_arch_dispatch,
        );
        let stub_items = &sm.stub_items;
        // Vision arch glue: per-variant `VisionArchWeights` impl,
        // `try_load_mm` with d_model fingerprint, inventory submits
        // for tp ∈ {1,2,4,8}. Decoder mode emits empty TokenStream.
        // `pixel_pack` is None for arches that use the trait's
        // default (delegating to `VisionConfig::patches_from_normalized_chw`).
        let vision_glue = if matches!(mode.prelude, classified::Prelude::Vision) {
            let processor = args.processor.as_ref().ok_or_else(|| {
                syn::Error::new(
                    args.span,
                    "#[vision_forward] missing required `processor = path::PROCESSOR` arg \
                     (path to a `pub const PROCESSOR: scratchy_vision::MmMetadata`)",
                )
            })?;
            vision_glue::emit_per_variant(
                sm.model,
                &arch_name,
                args.pixel_pack.as_ref(),
                &manifest.pad_to_mult8,
                processor,
            )
        } else {
            proc_macro2::TokenStream::new()
        };
        per_model_ts.push(quote! {
            pub mod #model_mod {
                #stub_items
                #codegen_items
                #vision_glue
            }
        });

        // Decoder fan-in to the arch-level dispatcher. Vision
        // encoders skip — see `CompileMode::emit_arch_dispatch`.
        // `collect_dispatch_bounds` panics on configs lacking the
        // decoder-only `DISPATCH_FIELDS` (vision configs carry
        // `vision_*` keys instead), so the call itself is gated.
        if mode.emit_arch_dispatch {
            arch_dispatch_arms.push(DispatchArm {
                model_ident: model_mod,
                source_stem: sm.model.source_stem.clone(),
                bounds: collect_dispatch_bounds(sm.model),
                tp_world_size: sm.tp_world_size,
            });
        }
    }

    // Union of HF `architectures: [..]` strings across every compiled
    // model — the set of safetensors `arch_hint` values
    // `scratchy_forward_compiler::try_load` will route to this arch. Deduped +
    // sorted for determinism.
    //
    // GGUF dispatch is on a SEPARATE field (`gguf_archs` below).
    // GGUFs report family-level tags (`"deepseek2"` covers V2 + V3-LoRA +
    // V3-flat; `"llama"` covers Llama + Mistral); the dispatcher tries
    // every claimant of that tag and `fingerprint_matches` discriminates.
    let mut hf_arches: Vec<String> = models
        .iter()
        .flat_map(|m| m.architectures.iter().cloned())
        .collect();
    hf_arches.sort();
    hf_arches.dedup();

    // Vision encoders intentionally skip the arch-dispatcher emission
    // (no `enum Weights`, no `ScratchyArchRegistration` inventory). The
    // hand-written `ScratchyMmRegistration` in each VL crate's
    // `vision.rs` claims its HF arch string and constructs the
    // multimodal forward over the macro-emitted per-variant
    // `Weights` types directly. `arch_dispatch_arms` is empty under
    // VISION mode, so this is also covered by `emit_arch_dispatcher`'s
    // empty-arms early-return — but the explicit skip here makes the
    // intent visible at the call site.
    let arch_dispatch_ts = if mode.emit_arch_dispatch {
        let arch_ident = Ident::new(&arch_name, name_span);
        // Hybrid (Gated-DeltaNet) arches: the per-arch
        // `ScratchyWeights::gdn_runtime_config` override the worker reads
        // to size + allocate the GDN state pool. Empty for non-hybrid
        // arches (they keep the trait default `None`). Per-variant arms
        // dispatch on the matched `Weights::Variant(_)` so each variant
        // returns its own (conv_dim, num_v_heads, linear_layers) — load-
        // bearing for arches with multiple sizes (e.g. Qwen3.5 0.8b vs
        // 9b: 24 layers vs 32, different conv_dim). Across workload
        // buckets within one variant the FUF op graph is identical so
        // we read it off the first bucket per variant.
        // Per-variant `gdn_runtime_config` arms — each `Weights::Variant(_)`
        // returns its own (conv_dim, num_v_heads, linear_layers) literal or
        // None for non-hybrid variants. Load-bearing for arches with
        // multiple sizes (Qwen3.5 0.8b vs 9b: 24 vs 32 layers, different
        // conv_dim). Workload-bucket dups collapse via `mod_name` dedup;
        // FUF op graph is bucket-invariant for the GDN predicate.
        let gdn_runtime_config_tokens = {
            let mut per_variant: std::collections::BTreeMap<
                String,
                Option<proc_macro2::TokenStream>,
            > = std::collections::BTreeMap::new();
            for sm in &solved {
                let key = sm.mod_name.clone();
                per_variant.entry(key).or_insert_with(|| {
                    codegen::emit_gdn_runtime_config_arm_body(&sm.fuf, sm.model)
                });
            }
            if per_variant.values().all(|v| v.is_none()) {
                proc_macro2::TokenStream::new()
            } else {
                let arms: Vec<proc_macro2::TokenStream> = arch_dispatch_arms
                    .iter()
                    .map(|a| {
                        let variant_ident = pascal_case(&a.model_ident);
                        let body = per_variant
                            .get(&a.model_ident.to_string())
                            .cloned()
                            .flatten()
                            .unwrap_or_else(|| quote! { ::core::option::Option::None });
                        quote! { Weights::#variant_ident(_) => #body, }
                    })
                    .collect();
                quote! {
                    fn gdn_runtime_config(
                        &self,
                    ) -> ::core::option::Option<
                        ::scratchy_forward_compiler::gdn_state_layout::GdnRuntimeConfig,
                    > {
                        match self {
                            #(#arms)*
                        }
                    }
                }
            }
        };
        // Per-variant `per_layer_kv_token_elems` arms — Some(vec![..])
        // for hybrid-attention-geometry arches (Gemma4), None elsewhere.
        // Same per-variant dedup mechanics as gdn_runtime_config above.
        let per_layer_kv_elems_tokens = {
            let mut per_variant: std::collections::BTreeMap<
                String,
                Option<proc_macro2::TokenStream>,
            > = std::collections::BTreeMap::new();
            for sm in &solved {
                let key = sm.mod_name.clone();
                per_variant.entry(key).or_insert_with(|| {
                    codegen::emit_per_layer_kv_token_elems_arm_body(&sm.fuf, sm.model)
                });
            }
            if per_variant.values().all(|v| v.is_none()) {
                proc_macro2::TokenStream::new()
            } else {
                let arms: Vec<proc_macro2::TokenStream> = arch_dispatch_arms
                    .iter()
                    .map(|a| {
                        let variant_ident = pascal_case(&a.model_ident);
                        let body = per_variant
                            .get(&a.model_ident.to_string())
                            .cloned()
                            .flatten()
                            .unwrap_or_else(|| quote! { ::core::option::Option::None });
                        quote! { Weights::#variant_ident(_) => #body, }
                    })
                    .collect();
                quote! {
                    fn per_layer_kv_token_elems(
                        &self,
                    ) -> ::core::option::Option<::std::vec::Vec<usize>> {
                        match self {
                            #(#arms)*
                        }
                    }
                }
            }
        };
        // Per-variant `max_blocks_per_seq` arms — the metal block-table
        // row stride (`CanonicalParams::MAX_BLOCKS_PER_SEQ`). Emitted
        // only when some variant overrides the trait default (128) via
        // the `max_blocks_per_seq` config key (Gemma4: 2048); the
        // executor packs host-side block_table rows at this stride so
        // multi-seq rows land where the kernels read them.
        let max_blocks_per_seq_tokens = {
            let mut per_variant: std::collections::BTreeMap<String, Option<u64>> =
                std::collections::BTreeMap::new();
            for sm in &solved {
                let key = sm.mod_name.clone();
                per_variant
                    .entry(key)
                    .or_insert_with(|| sm.model.bounds.get("max_blocks_per_seq").copied());
            }
            if per_variant.values().all(|v| v.is_none() || *v == Some(128)) {
                proc_macro2::TokenStream::new()
            } else {
                let arms: Vec<proc_macro2::TokenStream> = arch_dispatch_arms
                    .iter()
                    .map(|a| {
                        let variant_ident = pascal_case(&a.model_ident);
                        let v = per_variant
                            .get(&a.model_ident.to_string())
                            .copied()
                            .flatten()
                            .unwrap_or(128);
                        let lit = proc_macro2::Literal::usize_unsuffixed(v as usize);
                        quote! { Weights::#variant_ident(_) => #lit, }
                    })
                    .collect();
                quote! {
                    fn max_blocks_per_seq(&self) -> usize {
                        match self {
                            #(#arms)*
                        }
                    }
                }
            }
        };
        // Per-variant `rope_on_read` arms — spans / position-independent KV.
        // Rope-on-read is the universal default for rope-using models; this
        // mirrors the `ROPE_ON_READ` CanonicalParams const (both derive from
        // `fuf_uses_rotary`) and is what the metal worker reads to decide it is
        // safe to OR bit 31 into block_table/slot_mapping. No-rope variants get
        // the trait default (false); emitted only when some variant ropes.
        let rope_on_read_tokens = {
            let mut per_variant: std::collections::BTreeMap<String, bool> =
                std::collections::BTreeMap::new();
            for sm in &solved {
                let key = sm.mod_name.clone();
                per_variant
                    .entry(key)
                    .or_insert_with(|| codegen::fuf_uses_rotary(&sm.fuf));
            }
            if per_variant.values().all(|&v| !v) {
                proc_macro2::TokenStream::new()
            } else {
                let arms: Vec<proc_macro2::TokenStream> = arch_dispatch_arms
                    .iter()
                    .map(|a| {
                        let variant_ident = pascal_case(&a.model_ident);
                        let v = per_variant
                            .get(&a.model_ident.to_string())
                            .copied()
                            .unwrap_or(false);
                        quote! { Weights::#variant_ident(_) => #v, }
                    })
                    .collect();
                quote! {
                    fn rope_on_read(&self) -> bool {
                        match self {
                            #(#arms)*
                        }
                    }
                }
            }
        };
        let gdn_runtime_config_tokens = quote! {
            #gdn_runtime_config_tokens
            #per_layer_kv_elems_tokens
            #max_blocks_per_seq_tokens
            #rope_on_read_tokens
        };

        emit_arch_dispatcher(
            &arch_ident,
            &hf_arches,
            &arch_dispatch_arms,
            models_dir,
            name_span,
            gdn_runtime_config_tokens,
        )?
    } else {
        proc_macro2::TokenStream::new()
    };

    // Emit items INLINE at the carrier's scope (no wrapping mod).
    // The carrier fn itself is consumed — it was only a host for
    // the DSL body + the arch ident. The file-module that contains
    // the #[forward] invocation becomes the public entry point:
    // if `scratchy-models/src/llama.rs` contains
    // `#[forward] fn llama() { ... }`, the caller accesses
    // `scratchy_models::llama::Weights` directly (no
    // `::arch::` / `::llama::` / etc.).
    Ok(quote! {
        // Backend crate alias. Every emitted reference to the active
        // GPU runtime goes through `crate::__gpu::…` so the macro's
        // output is hardware-agnostic — the cfg selects which target
        // crate `__gpu` resolves to: the cuda runtime under `cuda`, the
        // metal runtime under `metal`. The third (backend-less) arm
        // matters because some arch crates are pulled into a backend's
        // workspace build WITHOUT a backend feature (e.g. phi3 is not
        // metal-capable, so it compiles feature-less under `--features
        // metal`): it still emits ungated `crate::__gpu::…` field types
        // (the `Weights` struct), so the alias must resolve there too —
        // every arch crate deps `scratchy-target-cuda` unconditionally,
        // so the cuda crate is the safe backend-less target.
        #[cfg(feature = "cuda")]
        #[allow(unused_imports)]
        use ::scratchy_target_cuda as __gpu;
        #[cfg(feature = "metal")]
        #[allow(unused_imports)]
        use ::scratchy_target_metal as __gpu;
        // Spyre: the host target re-exports the neutral surface (`weights`
        // defaulted to `SpyreAllocator`, `LoadStream`, `BackendAllocator`) the
        // emitted load/fingerprint code names through `crate::__gpu::…`.
        #[cfg(feature = "spyre")]
        #[allow(unused_imports)]
        use ::scratchy_target_spyre as __gpu;
        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "spyre")))]
        #[allow(unused_imports)]
        use ::scratchy_layers as __gpu;

        // Rebuild-on-change for every JSON the macro read.
        #(#tracked)*

        #(#per_model_ts)*

        #arch_dispatch_ts
    })
}

/// The identifying HF-config fields the arch-level `Weights::load`
/// matches on, in a fixed order. Any two compiled models that agree
/// on all seven values would collide; bump this if you add an arch
/// where that happens.
const DISPATCH_FIELDS: &[&str] = &[
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
];

/// One row of the arch-dispatch table — a single (variant, tp)
/// tuple. The compile-loop builds one of these per (model, tp) pair
/// the outer-loop fanout produced; the dispatcher uses them to build
/// the unified `Weights` enum, the per-tp `inventory::submit!` blocks,
/// and the per-variant `BackboneDumpRegistration` rows for
/// `scr model info`.
struct DispatchArm {
    model_ident: Ident,
    /// Source-config stem (e.g. `"qwen2.5-3b"`). Same for every tp
    /// of the same model — used as the variant_stem label in
    /// `BackboneDumpRegistration::dump_all` so `scr model info`
    /// can print human-readable model names regardless of the
    /// `_tp{N}`-suffixed module path.
    source_stem: String,
    /// Values aligned with [`DISPATCH_FIELDS`] — read by the
    /// per-arch accessor methods on the dispatcher's `Weights`
    /// enum. Same for every tp of the same model (the ScratchyWeights
    /// trait surface is the un-sharded model config — sharded values
    /// flow through `<W as CanonicalParams>` instead).
    bounds: Vec<u64>,
    tp_world_size: u8,
}

fn collect_dispatch_bounds(model: &config::ModelParams) -> Vec<u64> {
    DISPATCH_FIELDS
        .iter()
        .map(|k| {
            *model.bounds.get(*k).unwrap_or_else(|| {
                panic!(
                    "model `{}` is missing required bound `{k}`; add it to \
                     config.json or to config::derive_implicit_bounds",
                    model.source_stem,
                )
            })
        })
        .collect()
}

/// Arch-level dispatcher: an enum over every compiled variant plus
/// a `load` that auto-detects the right variant by walking each
/// variant's compile-emitted `fingerprint_matches(gw)` until one
/// claims the runtime `GpuWeights`. Accessor methods
/// (`num_hidden_layers`, …) delegate to per-variant constants baked
/// from each model's config.json.
///
/// Also emits the auto-registration with
/// `scratchy_target_cuda::try_load`: an `impl ScratchyWeights for Weights`
/// that routes the trait methods to the just-emitted accessors +
/// `forward` / `forward_backbone`, and an `inventory::submit!` block
/// carrying the union of HF `architectures` strings this arch
/// claims. The top-level loader walks the inventory at runtime; no
/// hand-written per-arch entry anywhere in the caller's codebase.
fn emit_arch_dispatcher(
    arch_ident: &Ident,
    hf_arches: &[String],
    arms: &[DispatchArm],
    models_dir: &Path,
    error_span: Span,
    gdn_runtime_config_tokens: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    if arms.is_empty() {
        return Ok(quote! {});
    }

    // Variant ident = PascalCase of the model ident (e.g.
    // `llama_3_2_1b` → `Llama_3_2_1b`, `llama_3_2_1b_tp2` →
    // `Llama_3_2_1b_Tp2`). Keep the underscores — they carry meaning
    // (dotted-version components, tp suffix) and collapsing them
    // would create ambiguity between e.g. `llama32` and `llama_3_2`.
    let variants: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            // Box the per-canonical `Weights` so the enum is pointer-sized
            // regardless of variant-size disparity (clippy large_enum_variant);
            // it's built once at load and every `forward*` takes `&Weights`, so
            // `&Box<W>` deref-coerces at each call — no usage change, and only
            // one deref per forward call (not per weight).
            quote! { #variant_ident(::std::boxed::Box<#model_ident::Weights>) }
        })
        .collect();

    // Per-variant spyre arms for `ScratchyWeights::ktir_bundle()`: each variant
    // returns its module's embedded `KTIR_BUNDLE` const as a neutral `&dyn Any`
    // (the worker downcasts it). Emitted inside the `#[cfg(feature = "spyre")]`
    // `ktir_bundle` override, so under cuda/metal these tokens are cfg'd out.
    let ktir_bundle_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => &#model_ident::KTIR_BUNDLE
                    as &'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync),
            }
        })
        .collect();

    // Per-variant arms for `ScratchyWeights::superdsc_weights()` — the
    // GENERATED source-id → `Weights` FIELD binding. Unlike `ktir_bundle`
    // these arms BIND the inner weights (`(w)`, not `(_)`): the whole point is
    // to reach real fields, which is why this cannot be a lookup table in the
    // worker.
    let superdsc_weight_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => #model_ident::superdsc_weights(w),
            }
        })
        .collect();

    // Per-variant arms for `ScratchyWeights::sengraph_bundle()` — the same
    // per-variant const the registration fn-pointer serves, reachable from the
    // LOADED model so the bundle and the weights cannot come from two
    // different variant matches.
    let superdsc_embed_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => #model_ident::superdsc_embed(w),
            }
        })
        .collect();

    let superdsc_wiring_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => &#model_ident::SUPERDSC_WIRINGS
                    as &'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync),
            }
        })
        .collect();

    let sengraph_bundle_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => &#model_ident::SENGRAPH_BUNDLE
                    as &'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync),
            }
        })
        .collect();

    // Distinct tp values across all arms, in ascending order. One
    // `inventory::submit!` per value; one match arm in `Weights::load`
    // per value. At nccl-disabled this is just `[1]` and the dispatcher
    // is byte-identical to the pre-fanout build.
    let mut tp_values: Vec<u8> = arms.iter().map(|a| a.tp_world_size).collect();
    tp_values.sort_unstable();
    tp_values.dedup();

    // `Weights::load` dispatch: one flat `if tp_world_size == N && <fingerprint>`
    // per variant. Gating tp inside each check (rather than `match tp` or an
    // outer `if tp == N { .. }` per group) avoids clippy single_match /
    // collapsible_if on single-variant groups. `arms` is in declaration order,
    // so fingerprint-sharing variants still resolve to the earliest declared.
    let load_match_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(a.tp_world_size);
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                if tp_world_size == #tp_lit && #model_ident::fingerprint_matches(gw, hf) {
                    return Ok(Some(Self::#variant_ident(::std::boxed::Box::new(
                        #model_ident::load(gw, stream, max_model_len, tp_rank)?,
                    ))));
                }
            }
        })
        .collect();

    let forward_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();
    let forward_backbone_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_backbone(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();
    let forward_piecewise_capture_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_piecewise_capture(w, ctx, device, num_tokens)
                },
            }
        })
        .collect();

    // Per-variant dispatch arms for `forward_with_metal_followup`.
    let forward_with_followup_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_with_metal_followup(w, ctx, device, num_tokens, followup)
                },
            }
        })
        .collect();

    // Per-variant dispatch arms for `forward_chain_with_encoder`.
    // Phase 6 spec-decode K-step chain entry — each canonical's
    // metal_emission emits its own `forward_chain_with_encoder` that
    // wraps `pool.with_chain_encoder` with a per-canonical
    // `ChainStepHandle` adapter.
    let forward_chain_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(w) => unsafe {
                    #model_ident::forward_chain_with_encoder(w, ctx, device, num_tokens, body)
                },
            }
        })
        .collect();

    // Per-variant `METAL_ARENA_PEAK_BYTES` reads. Each canonical mod
    // emits this const from the macro's per-canonical metal_emission;
    // shim variants re-export the canonical's. The trait impl below
    // dispatches on `Weights` variant and returns the matched module's
    // const so `determine_available_memory` reads the right per-arch
    // peak rather than a 512 MiB placeholder.
    let metal_arena_peak_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => #model_ident::METAL_ARENA_PEAK_BYTES,
            }
        })
        .collect();

    // Per-variant `METAL_BUCKET_ARENA_COSTS` reads — the `(bucket_m, bytes)`
    // table the worker feeds to `select_prefill_bucket` for target-reactive
    // bucket pruning. Mirrors `metal_arena_peak_arms`.
    let metal_bucket_costs_arms: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let variant_ident = pascal_case(&a.model_ident);
            let model_ident = &a.model_ident;
            quote! {
                Weights::#variant_ident(_) => #model_ident::METAL_BUCKET_ARENA_COSTS,
            }
        })
        .collect();

    // Accessor methods on Weights — each returns a per-variant
    // constant from the matched model's bounds. Consumers (e.g.
    // scratchy-serving-worker's CudaModel enum) delegate their own accessor
    // arms to these, replacing N duplicated `m.model.layers[0].foo`
    // walks with a single method call. Trait surface is un-sharded
    // — kernel launches read sharded values via `<W as
    // CanonicalParams>::…` instead, so every (variant, tp) pair of
    // the same model returns the same num_attention_heads / etc.
    let accessor_methods: Vec<proc_macro2::TokenStream> = DISPATCH_FIELDS
        .iter()
        .map(|field| {
            let method_name = Ident::new(field, Span::call_site());
            let arms_ts: Vec<proc_macro2::TokenStream> = arms
                .iter()
                .map(|a| {
                    let variant_ident = pascal_case(&a.model_ident);
                    let idx = DISPATCH_FIELDS
                        .iter()
                        .position(|f| f == field)
                        .expect("field in DISPATCH_FIELDS");
                    let val = proc_macro2::Literal::u64_unsuffixed(a.bounds[idx]);
                    quote! { Weights::#variant_ident(_) => #val, }
                })
                .collect();
            quote! {
                #[doc = concat!(
                    "The matched variant's `", stringify!(#method_name),
                    "` — from the model's config.json at macro-expansion time."
                )]
                pub fn #method_name(&self) -> u64 {
                    match self {
                        #(#arms_ts)*
                    }
                }
            }
        })
        .collect();

    // Literals the auto-emitted `ScratchyWeights` impl + inventory
    // registration reference.
    let arch_name_lit = proc_macro2::Literal::string(&arch_ident.to_string());
    let hf_arch_lits: Vec<proc_macro2::Literal> = hf_arches
        .iter()
        .map(|s| proc_macro2::Literal::string(s))
        .collect();

    // GGUF spec (per-arch). Loaded once: drives both
    // (a) the `gguf_archs` field of `ScratchyArchRegistration` (every arch
    //     with a `"ggml"` entry claims its gguf tag for dispatch), and
    // (b) the optional `scratchy_quantizations::gguf::register!` emission below (only
    //     the canonical owner per gguf_arch tag — `register_spec=false`
    //     opts out for non-canonical claimants like Mistral or
    //     deepseek-v3-flat).
    let gguf_spec = config::load_gguf_spec(models_dir).map_err(|e| {
        syn::Error::new(
            error_span,
            format!(
                "quantizations.json gguf spec in {}: {e}",
                models_dir.display()
            ),
        )
    })?;
    let gguf_arch_lits: Vec<proc_macro2::Literal> = match &gguf_spec {
        Some(spec) => vec![proc_macro2::Literal::string(
            spec.gguf_arch
                .clone()
                .unwrap_or_else(|| arch_ident.to_string())
                .as_str(),
        )],
        None => vec![],
    };

    // Per-variant `(stem, dump_fn)` rows for the backbone-dump
    // registry. Each variant's `mod <model_ident>` emits a
    // `pub fn dump() -> Vec<BucketDump>`; here we name them so a
    // single `BackboneDumpRegistration` per arch can iterate them.
    // Iterates over EVERY (model, tp) tuple — `scr model info`
    // shows each tp variant separately (the per-tp module's `dump()`
    // reflects the sharded canonical's bucket layout).
    let dump_rows: Vec<proc_macro2::TokenStream> = arms
        .iter()
        .map(|a| {
            let model_ident = &a.model_ident;
            let stem_lit = proc_macro2::Literal::string(&a.source_stem);
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(a.tp_world_size);
            quote! {
                ::scratchy_forward_compiler::VariantDump {
                    variant_stem: #stem_lit,
                    tp_world_size: #tp_lit,
                    buckets: #model_ident::dump(),
                }
            }
        })
        .collect();

    // Per-tp inventory submission. Each closure passes its own tp
    // constant into `Weights::load(...)`; the matching arm there
    // walks only the tp-N variants. The submit! block is a static
    // initializer — N submissions at compile time, walked once at
    // runtime by `scratchy_forward_compiler::try_load(arch_hint, tp_world_size)`.
    // The registration's `ktir_bundle` accessor — spyre's forward representation
    // (the peer of the cuda/metal `FORWARD_TABLE`). Under spyre it returns the
    // embedded `KTIR_BUNDLE` const as a neutral `&dyn Any` (the cfg-free registry
    // can't name the spyre `KtirBundle`); under cuda/metal the forward runs
    // through device kernels, so there's no bundle and the field is `None`.
    //
    // An arch registers MANY distinct base models (llama-2-13b … llama-3.2-3b),
    // each with its own `KTIR_BUNDLE` (different layer count). Pick the ONE whose
    // `fingerprint_matches` accepts the live checkpoint — the same per-variant
    // sniff `Weights::load` uses (read-only, so the worker keeps its tensors).
    // Quant variants of one base model share that base's bundle (shims re-export
    // it), so any matching arm returns the right const. Own-target gating
    // (parallel to how cuda/metal gate their `FORWARD_TABLE` / `METAL_BUCKETS`):
    // `KTIR_BUNDLE` / `fingerprint_matches` only exist under this arch's spyre
    // feature, so the `Some` arm can only compile there; other backends emit `None`.
    #[cfg(feature = "spyre")]
    let ktir_bundle_field = {
        let bundle_match_arms: Vec<proc_macro2::TokenStream> = arms
            .iter()
            .map(|a| {
                let model_ident = &a.model_ident;
                quote! {
                    if #model_ident::fingerprint_matches(__gw, __hf) {
                        return ::core::option::Option::Some(
                            &#model_ident::KTIR_BUNDLE
                                as &'static (dyn ::core::any::Any
                                    + ::core::marker::Send
                                    + ::core::marker::Sync),
                        );
                    }
                }
            })
            .collect();
        quote! {
            ktir_bundle: ::core::option::Option::Some(
                |__handle: ::scratchy_forward_compiler::GpuWeightsHandle<'_>,
                 __hf: ::scratchy_forward_compiler::HfFingerprint<'_>| {
                    // Recover the concrete `&mut GpuWeights` the same way
                    // `try_load` does; `fingerprint_matches` only sniffs
                    // shapes (its `&mut` reborrows to `&`), never `take`s a
                    // tensor — so the worker keeps them all.
                    let __gw: &mut crate::__gpu::weights::GpuWeights =
                        unsafe { __handle.as_mut() };
                    #(#bundle_match_arms)*
                    ::core::option::Option::None
                },
            ),
        }
    };
    #[cfg(not(feature = "spyre"))]
    let ktir_bundle_field = quote! { ktir_bundle: ::core::option::Option::None, };
    // Peer accessor for the `--target sendnn` bundle (the baked `SENGRAPH_BUNDLE`).
    #[cfg(feature = "spyre")]
    let sengraph_bundle_field = {
        // Same per-variant fingerprint walk as `ktir_bundle` above, and for
        // the same reason: an arch registers MANY base models, each with its
        // own bundle. This used to hand back `arms.first()` unconditionally,
        // so a build carrying granite-3.1-2b AND -8b served the 2b's bundle
        // for both — caught only by the loader's shape guard.
        let sengraph_match_arms: Vec<proc_macro2::TokenStream> = arms
            .iter()
            .map(|a| {
                let model_ident = &a.model_ident;
                quote! {
                    if #model_ident::fingerprint_matches(__gw, __hf) {
                        return ::core::option::Option::Some(
                            &#model_ident::SENGRAPH_BUNDLE
                                as &'static (dyn ::core::any::Any
                                    + ::core::marker::Send
                                    + ::core::marker::Sync),
                        );
                    }
                }
            })
            .collect();
        quote! {
            sengraph_bundle: ::core::option::Option::Some(
                |__handle: ::scratchy_forward_compiler::GpuWeightsHandle<'_>,
                 __hf: ::scratchy_forward_compiler::HfFingerprint<'_>| {
                    let __gw: &mut crate::__gpu::weights::GpuWeights =
                        unsafe { __handle.as_mut() };
                    #(#sengraph_match_arms)*
                    ::core::option::Option::None
                },
            ),
        }
    };
    #[cfg(not(feature = "spyre"))]
    let sengraph_bundle_field = quote! { sengraph_bundle: ::core::option::Option::None, };
    let inventory_submits: Vec<proc_macro2::TokenStream> = tp_values
        .iter()
        .map(|tp| {
            let tp_lit = proc_macro2::Literal::u8_unsuffixed(*tp);
            quote! {
                #[cfg(any(feature = "cuda", feature = "metal", feature = "spyre"))]
                ::scratchy_forward_compiler::inventory::submit! {
                    ::scratchy_forward_compiler::ScratchyArchRegistration {
                        arch_name: #arch_name_lit,
                        hf_arches: &[#(#hf_arch_lits),*],
                        gguf_archs: &[#(#gguf_arch_lits),*],
                        tp_world_size: #tp_lit,
                        // The registry fn-pointer names the allocator-erased
                        // `GpuWeightsHandle`; recover the concrete
                        // `&mut GpuWeights<BackendAllocator>` the per-variant
                        // `Weights::load` needs before calling it. Non-capturing
                        // so it coerces to the `ArchTryLoadFn` fn-pointer.
                        try_load: |__gw_handle: ::scratchy_forward_compiler::GpuWeightsHandle<'_>, stream, max_model_len, tp_rank, hf| {
                            let gw: &mut crate::__gpu::weights::GpuWeights =
                                unsafe { __gw_handle.as_mut() };
                            Weights::load(
                                gw, stream, max_model_len, hf, #tp_lit, tp_rank,
                            ).map(|opt| {
                                opt.map(|w| ::std::boxed::Box::new(w)
                                    as ::std::boxed::Box<dyn ::scratchy_forward_compiler::ScratchyWeights>)
                            })
                        },
                        #ktir_bundle_field
                        #sengraph_bundle_field
                    }
                }
            }
        })
        .collect();

    // GGUF spec registration. Conditional on (a) the arch having a
    // `"ggml"` entry in `quantizations.json` AND (b) `register_spec`
    // being true (default). Non-canonical owners of a gguf tag set
    // `register_spec: false` so only one crate per gguf_arch supplies
    // the inventory-side spec data — find_spec stays deterministic.
    // The arch is still routed for that tag via the `gguf_archs`
    // field of its `ScratchyArchRegistration` above.
    let gguf_register_emit: proc_macro2::TokenStream = match gguf_spec {
        None => quote! {},
        Some(ref spec) if !spec.register_spec => quote! {},
        Some(spec) => {
            let gguf_arch_lit = proc_macro2::Literal::string(
                &spec.gguf_arch.unwrap_or_else(|| arch_ident.to_string()),
            );
            let qk = spec.qk_permute;
            let rope = spec.llama3_rope_scaling_inference;
            let nwo = proc_macro2::Literal::f32_suffixed(spec.norm_weight_offset);
            let renames: Vec<proc_macro2::TokenStream> = spec
                .tensor_renames
                .iter()
                .map(|(g, h)| {
                    let g = proc_macro2::Literal::string(g);
                    let h = proc_macro2::Literal::string(h);
                    quote! { (#g, #h) }
                })
                .collect();
            let m_u32: Vec<proc_macro2::TokenStream> = spec
                .metadata_u32
                .iter()
                .map(|(g, e)| {
                    let g = proc_macro2::Literal::string(g);
                    let e = proc_macro2::Literal::string(e);
                    quote! { (#g, #e) }
                })
                .collect();
            let m_f32: Vec<proc_macro2::TokenStream> = spec
                .metadata_f32
                .iter()
                .map(|(g, e)| {
                    let g = proc_macro2::Literal::string(g);
                    let e = proc_macro2::Literal::string(e);
                    quote! { (#g, #e) }
                })
                .collect();
            let d_u32: Vec<proc_macro2::TokenStream> = spec
                .metadata_defaults_u32
                .iter()
                .map(|(k, v)| {
                    let k = proc_macro2::Literal::string(k);
                    let v = proc_macro2::Literal::u32_suffixed(*v);
                    quote! { (#k, #v) }
                })
                .collect();
            let d_f32: Vec<proc_macro2::TokenStream> = spec
                .metadata_defaults_f32
                .iter()
                .map(|(k, v)| {
                    let k = proc_macro2::Literal::string(k);
                    let v = proc_macro2::Literal::f32_suffixed(*v);
                    quote! { (#k, #v) }
                })
                .collect();
            quote! {
                #[cfg(feature = "cuda")]
                ::scratchy_quantizations::register! {
                    gguf_arch = #gguf_arch_lit,
                    qk_permute = #qk,
                    tensor_renames = [ #(#renames),* ],
                    metadata_u32 = [ #(#m_u32),* ],
                    metadata_f32 = [ #(#m_f32),* ],
                    metadata_defaults_u32 = [ #(#d_u32),* ],
                    metadata_defaults_f32 = [ #(#d_f32),* ],
                    llama3_rope_scaling_inference = #rope,
                    norm_weight_offset = #nwo,
                }
            }
        }
    };

    Ok(quote! {
        /// One variant per compiled model config. Holds that
        /// model's specialized `Weights`. Same shape under both
        /// backends — the variant's per-canonical `Weights` struct
        /// itself is cfg-mutex'd internally (cuda fields gated
        /// `cfg(feature = "cuda")`, metal fields gated
        /// `cfg(feature = "metal")`).
        #[cfg(any(feature = "cuda", feature = "metal", feature = "spyre"))]
        pub enum Weights {
            #(#variants),*
        }

        #[cfg(any(feature = "cuda", feature = "metal", feature = "spyre"))]
        impl Weights {
            #(#accessor_methods)*

            /// Auto-detect the compiled variant by sniffing the
            /// runtime `GpuWeights` against each variant's compile-
            /// baked fingerprint (embedding shape + last-layer
            /// tensor presence + quant suffix), then load.
            ///
            /// Returns `Ok(Some(..))` on a variant hit, `Ok(None)`
            /// when no compiled variant's fingerprint accepted the
            /// live `GpuWeights` (caller falls back to a hand-written
            /// path), or `Err(..)` only when a matched variant's
            /// `Weights::load` itself failed (I/O, shape mismatch
            /// inside a loader). A fingerprint miss is not an error —
            /// scratchy's job is to cover the storage formats it
            /// compiled for, not every format on disk.
            pub fn load(
                gw: &mut crate::__gpu::weights::GpuWeights,
                stream: crate::__gpu::LoadStream,
                max_model_len: usize,
                hf: ::scratchy_forward_compiler::HfFingerprint<'_>,
                tp_world_size: u8,
                tp_rank: u8,
            ) -> ::anyhow::Result<Option<Self>> {
                #(#load_match_arms)*
                Ok(None)
            }
        }

        /// Dispatching forward. Matches the `Weights` variant and
        /// calls the per-model specialized `forward`. Same body and
        /// signature under both backends — the cfg-mutex'd
        /// `GpuDevice` and `OwnedTensor` re-exports resolve to the
        /// matching backend's struct, and per-canonical `forward`
        /// fns now exist in both `cfg(cuda)` and `cfg(metal)` arms.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live backend device.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            match w {
                #(#forward_arms)*
            }
        }

        /// Same as [`forward`] but with an MTL4 encoder-tail hook.
        /// See `MetalForwardFollowup` for semantics.
        ///
        /// # Safety
        /// Same as [`forward`].
        #[cfg(feature = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_with_metal_followup(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            followup: ::core::option::Option<::scratchy_forward_compiler::MetalForwardFollowup<'_>>,
        ) -> crate::__gpu::OwnedTensor {
            match w {
                #(#forward_with_followup_arms)*
            }
        }

        /// Phase 6 spec-decode K-step chain dispatcher. One MTL4 CB,
        /// caller drives K bucket forwards + per-iter argmax +
        /// chain_advance from inside `body`. See
        /// `::scratchy_forward_compiler::MetalChainBody` for the body signature.
        ///
        /// # Safety
        /// Same as [`forward`]; additionally, `body` must not retain
        /// any references to the encoder past its return.
        #[cfg(feature = "metal")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_chain_with_encoder(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
            body: ::scratchy_forward_compiler::MetalChainBody<'_>,
        ) -> ::core::result::Result<(), ::std::string::String> {
            match w {
                #(#forward_chain_arms)*
            }
        }

        /// Dispatching backbone-only forward (no lm_head). Returns
        /// `[num_tokens, hidden_size]` as an independently-owned
        /// `OwnedTensor`. For pipeline-parallel intermediate ranks
        /// that hand hidden states to the next rank — cuda-only
        /// today; metal has no PP fanout, so the per-canonical
        /// `forward_backbone` is cfg(cuda)-gated and this dispatcher
        /// matches.
        ///
        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_backbone(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> crate::__gpu::OwnedTensor {
            match w {
                #(#forward_backbone_arms)*
            }
        }

        /// Dispatching piecewise CUDA-graph capture. Matches the
        /// `Weights` variant and calls the per-canonical
        /// `forward_piecewise_capture`. cuda-only; metal has no
        /// piecewise path because nccl isn't a metal concept.
        #[cfg(feature = "cuda")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn forward_piecewise_capture(
            w: &Weights,
            ctx: &crate::__gpu::ForwardCtx,
            device: &mut crate::__gpu::GpuDevice,
            num_tokens: u64,
        ) -> ::anyhow::Result<crate::__gpu::piecewise::PiecewiseRunner> {
            match w {
                #(#forward_piecewise_capture_arms)*
            }
        }

        // ── Auto-registration with scratchy_forward_compiler::try_load ──────
        //
        // `ScratchyWeights` impl routes trait methods to the just-
        // emitted accessors + `forward` / `forward_backbone`.
        // `inventory::submit!` adds this arch to the global registry.
        // No hand-written central list anywhere.

        #[cfg(any(feature = "cuda", feature = "metal", feature = "spyre"))]
        impl ::scratchy_forward_compiler::ScratchyWeights for Weights {
            fn arch_name(&self) -> &'static str { #arch_name_lit }
            fn num_hidden_layers(&self) -> u64 { self.num_hidden_layers() }
            fn hidden_size(&self) -> u64 { self.hidden_size() }
            fn intermediate_size(&self) -> u64 { self.intermediate_size() }
            fn num_attention_heads(&self) -> u64 { self.num_attention_heads() }
            fn num_key_value_heads(&self) -> u64 { self.num_key_value_heads() }
            fn head_dim(&self) -> u64 { self.head_dim() }
            fn vocab_size(&self) -> u64 { self.vocab_size() }

            #gdn_runtime_config_tokens

            // Spyre: hand the host worker the embedded KTIR bundle for the
            // fingerprint-matched variant — each variant names its module's
            // `KTIR_BUNDLE` const (baked into the binary by the macro). Returned
            // as a neutral `&dyn Any` (the cfg-free trait can't name the spyre
            // `KtirBundle`); the worker downcasts it back. No disk read.
            #[cfg(feature = "spyre")]
            fn ktir_bundle(
                &self,
            ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
            {
                Some(match self {
                    #(#ktir_bundle_arms)*
                })
            }

            // Spyre: the generated source-id → weight-field binding. Emitted
            // per model because only generated code can name a generated
            // field — the same reason metal's forward reads `wm.q_proj`
            // instead of resolving a string.
            #[cfg(feature = "spyre")]
            fn superdsc_weights(
                &self,
            ) -> Option<::anyhow::Result<Vec<::scratchy_forward_compiler::BoundWeight>>> {
                Some(match self {
                    #(#superdsc_weight_arms)*
                })
            }

            // Spyre: the launch wiring, emitted as a static instead of a JSON
            // string the worker parses back into the same shape.
            #[cfg(feature = "spyre")]
            fn superdsc_wiring(
                &self,
            ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
            {
                Some(match self {
                    #(#superdsc_wiring_arms)*
                })
            }

            // Spyre: the embedding table, taken out of the weight store by the
            // generated loader and therefore only reachable from the field.
            #[cfg(feature = "spyre")]
            fn superdsc_embed(
                &self,
            ) -> Option<::anyhow::Result<::scratchy_target_spyre::GpuTensor>> {
                Some(match self {
                    #(#superdsc_embed_arms)*
                })
            }

            // Spyre: the embedded sendnn bundle for THIS loaded variant — so
            // the bundle and the weights come from one match, not two walks.
            #[cfg(feature = "spyre")]
            fn sengraph_bundle(
                &self,
            ) -> Option<&'static (dyn ::core::any::Any + ::core::marker::Send + ::core::marker::Sync)>
            {
                Some(match self {
                    #(#sengraph_bundle_arms)*
                })
            }

            // Device-runtime forward methods — cuda/metal only (the trait gates
            // them off under spyre, whose forward runs through the KTIR bundle).
            // The cfg-free trait takes the neutral `ForwardCtxHandle` /
            // `ForwardDeviceHandle` (so it names no cuda/metal type); recover
            // the concrete `&ForwardCtx` / `&mut GpuDevice` here before
            // delegating to the per-arch free fns, which keep concrete types.
            #[cfg(any(feature = "cuda", feature = "metal"))]
            unsafe fn forward(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward(self, ctx, device, num_tokens) }
            }

            #[cfg(any(feature = "cuda", feature = "metal"))]
            unsafe fn forward_backbone(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
            ) -> crate::__gpu::OwnedTensor {
                #[cfg(feature = "cuda")]
                {
                    let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                    let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                    unsafe { forward_backbone(self, ctx, device, num_tokens) }
                }
                #[cfg(feature = "metal")]
                {
                    let _ = (ctx, device, num_tokens);
                    unimplemented!(
                        "metal forward_backbone — pipeline-parallel intermediate \
                         ranks aren't supported on metal yet (no PP fanout)"
                    )
                }
            }

            #[cfg(feature = "cuda")]
            unsafe fn forward_piecewise_capture(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
            ) -> ::anyhow::Result<::std::boxed::Box<dyn ::core::any::Any + ::core::marker::Send>> {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                // The concrete `PiecewiseRunner` is boxed as a neutral
                // `dyn Any + Send` (the trait can't name a cuda type); the cuda
                // worker downcasts it back.
                let runner = unsafe { forward_piecewise_capture(self, ctx, device, num_tokens) }?;
                ::core::result::Result::Ok(::std::boxed::Box::new(runner))
            }

            #[cfg(feature = "metal")]
            fn metal_arena_peak_bytes(&self) -> u64 {
                match self {
                    #(#metal_arena_peak_arms)*
                }
            }

            #[cfg(feature = "metal")]
            fn metal_bucket_arena_costs(&self) -> &'static [(u32, u64)] {
                match self {
                    #(#metal_bucket_costs_arms)*
                }
            }

            #[cfg(feature = "metal")]
            unsafe fn forward_with_metal_followup(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
                followup: ::core::option::Option<::scratchy_forward_compiler::MetalForwardFollowup<'_>>,
            ) -> crate::__gpu::OwnedTensor {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward_with_metal_followup(self, ctx, device, num_tokens, followup) }
            }

            #[cfg(feature = "metal")]
            unsafe fn metal_chain_with_encoder(
                &self,
                ctx: crate::__gpu::ForwardCtxHandle<'_>,
                device: crate::__gpu::ForwardDeviceHandle<'_>,
                num_tokens: u64,
                body: ::scratchy_forward_compiler::MetalChainBody<'_>,
            ) -> ::core::result::Result<(), ::std::string::String> {
                let ctx: &crate::__gpu::ForwardCtx = unsafe { ctx.as_ref() };
                let device: &mut crate::__gpu::GpuDevice = unsafe { device.as_mut() };
                unsafe { forward_chain_with_encoder(self, ctx, device, num_tokens, body) }
            }
        }

        // One `ScratchyArchRegistration` per distinct tp value across
        // the compiled (variant, tp) tuples. Each closure passes its
        // own tp constant into `Weights::load(...)` so the matching
        // arm there walks only the tp-N variants. At nccl-disabled
        // this expands to a single submission with `tp_world_size:
        // 1u8` — byte-identical to the pre-fanout build.
        #(#inventory_submits)*

        // One `BackboneDumpRegistration` per arch, fanning out over
        // every (model, tp) variant via `dump_rows`. Independent of
        // the per-tp `ScratchyArchRegistration` above — `scr model
        // info` walks this registry separately, with no runtime GPU
        // or weight loading. Gated on either backend feature so the
        // metal CLI sees compiled-in metal arches too.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        ::scratchy_forward_compiler::inventory::submit! {
            ::scratchy_forward_compiler::BackboneDumpRegistration {
                arch_name: #arch_name_lit,
                dump_all: || vec![ #(#dump_rows),* ],
            }
        }

        // GGUF arch registration: empty when the arch's
        // `quantizations.json` has no `"ggml"` entry; otherwise one
        // `scratchy_quantizations::gguf::register! { ... }` block. All per-arch GGUF
        // data flows through `quantizations.json`.
        #gguf_register_emit
    })
}

/// PascalCase a snake_case ident while preserving underscores
/// between segments (they're version separators in our model names).
fn pascal_case(ident: &Ident) -> Ident {
    let s = ident.to_string();
    let mut out = String::with_capacity(s.len());
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' {
            out.push('_');
            capitalize_next = true;
        } else if capitalize_next {
            out.extend(c.to_uppercase());
            capitalize_next = false;
        } else {
            out.push(c);
        }
    }
    Ident::new(&out, Span::call_site())
}

/// Emit `pub const NUM_TILES: usize = …` per canonical. The only
/// observability surface that catches a real regression — if the
/// FUF lowering or fusion logic changes the tile count, the test
/// `assert_eq!(llama_3_2_1b::NUM_TILES, 243)` fires.
///
/// Per-bucket `m_X[_sk_Y]::{NUM_SUBGRAPHS, NUM_WAVES, PREDICTED_US}`
/// modules used to live here too — they were brittle (PREDICTED_US
/// drifts every time `target_profiles/*.csv` is regenerated; the
/// other two were probed without an assertion). The test invariant
/// they really cared about (prefill cost ≫ decode cost) lives in
/// `solver::tests` now, calling `solve()` directly. ~9k lines off
/// cargo expand workspace-wide.
fn emit_model_stub_items(
    fuf: &fuf::Fuf,
    _sfufs: &crate::assignment::WorkloadAssignments,
    _loops: &schedule::WorkloadLoops,
) -> proc_macro2::TokenStream {
    let num_tiles = fuf.len();
    quote! {
        pub const NUM_TILES: usize = #num_tiles;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::{BnbQuantType, Fp8ActivationScheme, GptqLayout, QuantMethod};

    /// `dedup_quant_sig` must keep FP8 block and per-tensor / per-
    /// channel variants on separate canonicals — the load_with body
    /// emits `Fp8BlockLinear::load` for one and `Fp8Linear::load` for
    /// the other, and the runtime kernels can't share a path
    /// (`Fp8BlockLinear::load` panics on a 1D scale tensor).
    #[test]
    fn dedup_quant_sig_separates_fp8_block_from_fp8_std() {
        let block = QuantMethod::Fp8 {
            scheme: Fp8ActivationScheme::Dynamic,
            block_size: Some([128, 128]),
        };
        let std_dyn = QuantMethod::Fp8 {
            scheme: Fp8ActivationScheme::Dynamic,
            block_size: None,
        };
        let std_static = QuantMethod::Fp8 {
            scheme: Fp8ActivationScheme::Static,
            block_size: None,
        };
        // Block vs std must differ — that's the load-arm split.
        assert_ne!(
            dedup_quant_sig(Some(&block)),
            dedup_quant_sig(Some(&std_dyn))
        );
        // Within `std`, dynamic vs static can share a canonical:
        // both go through `Fp8Linear::load`, the activation_scheme
        // is a runtime Impl detail (Fp8GemmImpl handles both).
        assert_eq!(
            dedup_quant_sig(Some(&std_dyn)),
            dedup_quant_sig(Some(&std_static))
        );
    }

    /// AWQ and GPTQ all collapse to one Marlin discriminator —
    /// AWQ/GPTQ/CT variants of the same dense base SHOULD share
    /// a canonical (the runtime `marlin_storage` param threads in
    /// the on-disk format). Compressed-tensors INT4 routes through
    /// `QuantMethod::Gptq`, so it lands on `q:gptq` alongside
    /// AutoGPTQ.
    #[test]
    fn dedup_quant_sig_collapses_awq_and_gptq_internally() {
        let awq_a = QuantMethod::Awq {
            bits: 4,
            group_size: 128,
            zero_point: true,
            version: crate::quantization::AwqVersion::Gemm,
        };
        let awq_b = QuantMethod::Awq {
            bits: 4,
            group_size: 64,
            zero_point: true,
            version: crate::quantization::AwqVersion::Gemm,
        };
        // Same arm, regardless of group_size.
        assert_eq!(dedup_quant_sig(Some(&awq_a)), dedup_quant_sig(Some(&awq_b)));
        let gptq = QuantMethod::Gptq {
            bits: 4,
            group_size: 128,
            desc_act: true,
            sym: true,
            layout: GptqLayout::Qweight,
        };
        // AWQ and GPTQ must NOT collapse — different MarlinFormat
        // discriminator threaded at load time, but more importantly
        // their `marlin_storage` literal differs in the prelude.
        assert_ne!(dedup_quant_sig(Some(&awq_a)), dedup_quant_sig(Some(&gptq)));
    }

    /// `dedup_tp_sig` must give different strings for different
    /// tp_world_size values so the (variant × tp) fanout's canonical
    /// hash separates them. Mirrors
    /// `dedup_quant_sig_separates_fp8_block_from_fp8_std`'s shape.
    #[test]
    fn dedup_tp_sig_separates_each_compile_time_tp() {
        // The compile-time set is {1,2,4,8}.
        let sigs: Vec<String> = [1u8, 2, 4, 8].iter().map(|t| dedup_tp_sig(*t)).collect();
        for i in 0..sigs.len() {
            for j in (i + 1)..sigs.len() {
                assert_ne!(
                    sigs[i],
                    sigs[j],
                    "tp={} and tp={} must hash to different signatures",
                    [1, 2, 4, 8][i],
                    [1, 2, 4, 8][j],
                );
            }
        }
    }

    /// `dedup_tp_sig(n)` is deterministic — same input → same output
    /// across calls. Cargo's incremental cache hashes the dedup
    /// signature, so non-determinism would force spurious rebuilds.
    #[test]
    fn dedup_tp_sig_is_deterministic() {
        for tp in [1u8, 2, 4, 8, 16] {
            assert_eq!(dedup_tp_sig(tp), dedup_tp_sig(tp));
        }
    }

    /// `dedup_tp_sig` format is `tp:<n>` — pinning this so the
    /// signature stays human-readable in cargo expand and golden
    /// diffs, matching the `q:`/`b:`/`r:`/`w:` prefixes already
    /// used by the other dedup parts.
    #[test]
    fn dedup_tp_sig_format() {
        assert_eq!(dedup_tp_sig(1), "tp:1");
        assert_eq!(dedup_tp_sig(8), "tp:8");
    }

    /// BNB4 has its own FieldLoad arm (`Bnb4bitLinear::load`),
    /// distinct from FP8 / Marlin / Dense.
    #[test]
    fn dedup_quant_sig_keeps_bnb4_separate() {
        let bnb = QuantMethod::Bnb4 {
            quant_type: BnbQuantType::NF4,
            blocksize: 64,
        };
        let dense = None;
        assert_ne!(dedup_quant_sig(Some(&bnb)), dedup_quant_sig(dense));
    }

    /// `compute_canonical_variants` end-to-end on the (variant × tp)
    /// fanout: two SolvedModels of the same variant compiled at
    /// different `tp_world_size` values must NOT collapse to one
    /// canonical. The dedup string differs only in the `tp:N` part,
    /// which is enough to keep them in separate equivalence classes.
    /// This is the regression for task #7's outer-loop fanout — a
    /// future change that drops `dedup_tp_sig(self.tp_world_size)`
    /// from the dedup string would fail this test.
    #[test]
    fn tp_world_sizes_pick_separate_canonicals() {
        struct Fake {
            sig: String,
            stem: String,
            name: String,
        }
        impl HasSolvedSig for Fake {
            fn dedup_signature(&self) -> String {
                self.sig.clone()
            }
            fn source_stem(&self) -> &str {
                &self.stem
            }
            fn model_name(&self) -> &str {
                &self.name
            }
        }
        // Same variant compiled at tp=1, 2, 4, 8 — every other
        // dedup part identical, only `tp:N` differs.
        let solved = vec![
            Fake {
                sig: format!("x|{}", dedup_tp_sig(1)),
                stem: "command-r-1l".into(),
                name: "command_r_1l".into(),
            },
            Fake {
                sig: format!("x|{}", dedup_tp_sig(2)),
                stem: "command-r-1l_tp2".into(),
                name: "command_r_1l_tp2".into(),
            },
            Fake {
                sig: format!("x|{}", dedup_tp_sig(4)),
                stem: "command-r-1l_tp4".into(),
                name: "command_r_1l_tp4".into(),
            },
            Fake {
                sig: format!("x|{}", dedup_tp_sig(8)),
                stem: "command-r-1l_tp8".into(),
                name: "command_r_1l_tp8".into(),
            },
        ];
        let map = compute_canonical_variants(&solved);
        // Each tp picks its own variant as canonical — no collapse.
        assert_eq!(map[&0].to_string(), "command_r_1l");
        assert_eq!(map[&1].to_string(), "command_r_1l_tp2");
        assert_eq!(map[&2].to_string(), "command_r_1l_tp4");
        assert_eq!(map[&3].to_string(), "command_r_1l_tp8");
    }

    /// `compute_canonical_variants` end-to-end: with the q-sig
    /// discriminator in `dedup_signature`, FP8 block stays its own
    /// canonical while dynamic+static fold together (alphabetically
    /// `dynamic` wins). This is the regression test for the
    /// qwen3-fp8-dynamic graph-capture panic.
    #[test]
    fn fp8_block_and_std_pick_separate_canonicals() {
        struct Fake {
            sig: String,
            stem: String,
            name: String,
        }
        impl HasSolvedSig for Fake {
            fn dedup_signature(&self) -> String {
                self.sig.clone()
            }
            fn source_stem(&self) -> &str {
                &self.stem
            }
            fn model_name(&self) -> &str {
                &self.name
            }
        }
        // All four share every other dedup key (bounds, scalars,
        // rope, SFUF) and differ only in the q-sig — exactly the
        // qwen3 case the bug surfaced on.
        let solved = vec![
            Fake {
                sig: "x|q:fp8-block".into(),
                stem: "qwen3-0.6b-fp8-block-128x128".into(),
                name: "qwen3_0_6b_fp8_block_128x128".into(),
            },
            Fake {
                sig: "x|q:fp8-std".into(),
                stem: "qwen3-0.6b-fp8-dynamic-per-tensor".into(),
                name: "qwen3_0_6b_fp8_dynamic_per_tensor".into(),
            },
            Fake {
                sig: "x|q:fp8-std".into(),
                stem: "qwen3-0.6b-fp8-static-per-tensor".into(),
                name: "qwen3_0_6b_fp8_static_per_tensor".into(),
            },
        ];
        let map = compute_canonical_variants(&solved);
        // Block stays its own canonical.
        assert_eq!(map[&0].to_string(), "qwen3_0_6b_fp8_block_128x128");
        // Dynamic + static collapse to dynamic (alphabetically
        // earliest stem in the {dynamic, static} pair).
        assert_eq!(map[&1].to_string(), "qwen3_0_6b_fp8_dynamic_per_tensor");
        assert_eq!(map[&2].to_string(), "qwen3_0_6b_fp8_dynamic_per_tensor");
    }
}
