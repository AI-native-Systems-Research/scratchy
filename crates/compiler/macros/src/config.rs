// SPDX-License-Identifier: Apache-2.0
//! Phase 3: read `config.json` files from a `crates/models/arch/configs/<arch>/`
//! directory into a `Vec<ModelParams>`.
//!
//! Each `ModelParams` has:
//!   - a `name` (file stem, normalized to a Rust identifier) used
//!     by downstream codegen to name the emitted specialization;
//!   - a `bounds` map of every top-level integer field in the JSON.
//!
//! The bounds map is populated purely from the JSON — no
//! per-architecture knowledge lives here. Downstream passes
//! (shape inference, CFG, codegen) decide which of those bounds
//! they care about by name (`num_hidden_layers`, `hidden_size`,
//! ...). That keeps this pass generic across architectures.

// This module is Phase 3's deliverable. Consumers land in Phase 4
// (shape inference resolves symbolic shapes using these bounds);
// removed then.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::quantization::QuantizationConfig;

/// One model's parameters, loaded from one `config.json`.
#[derive(Clone, Debug)]
pub struct ModelParams {
    /// File stem, normalized to a valid Rust identifier. Stored as
    /// `String` (not `Ident`) so `ModelParams` is `Send + Sync` —
    /// the macro drive parallelizes per-model work and `Ident`
    /// wraps rustc's thread-local bridge.
    pub name: String,
    /// Original file stem (pre-normalization) and the path it was
    /// loaded from. For diagnostics.
    pub source_stem: String,
    pub source_path: PathBuf,
    /// Every top-level integer field from the JSON. Keys are the
    /// config.json field names verbatim (`num_hidden_layers`,
    /// `hidden_size`, etc.).
    pub bounds: BTreeMap<String, u64>,
    /// Every top-level float field from the JSON. Mirror of
    /// [`bounds`](Self::bounds) for non-integer scalars like
    /// `query_pre_attn_scalar` (Gemma2), `attn_logit_softcapping`,
    /// `rms_norm_eps`. Downstream callers read these by name; this
    /// module doesn't know which ones are used where.
    pub scalars: BTreeMap<String, f64>,
    /// Parsed `quantization_config` subobject, if present in the
    /// JSON. `None` for plain dense models. Consumers resolve
    /// per-weight storage format via
    /// [`crate::quantization::storage_format_for_weight`].
    pub quantization: Option<QuantizationConfig>,
    /// HF's `tie_word_embeddings` flag. When `true`, `lm_head`
    /// shares its weight buffer with `embed_tokens` and has no
    /// on-disk `lm_head.*` tensors — the codegen FieldLoad plan
    /// falls back to [`crate::codegen::FieldLoad::LinearTiedToEmbedding`]
    /// and the quant resolver keeps `lm_head` dense even under an
    /// AWQ config whose `modules_to_not_convert` doesn't list it.
    pub tie_word_embeddings: bool,
    /// HF `architectures: [..]` strings from this model's
    /// `config.json`. The macro unions these across every compiled
    /// model in an arch directory to produce the `hf_arches` list
    /// baked into the arch's
    /// `scratchy_target_cuda::ScratchyArchRegistration` registration,
    /// driving the runtime `try_load(..., arch_hint)` dispatch.
    pub architectures: Vec<String>,
    /// Extra JSON files that contributed to this variant's final
    /// config — the quantization preset and per-size override
    /// files that the loader deep-merged onto the dense base.
    /// Empty for dense variants; populated for synthesized
    /// `<size>-<preset>` variants. The `#[forward]` macro adds
    /// each to its `include_str!` tracking list so cargo rebuilds
    /// on any overlay/override edit, not just on dense-base edits.
    pub extra_tracked_paths: Vec<PathBuf>,
    /// Parsed `rope_scaling` subobject, if present. Drives the
    /// `RotaryCache` constructor picked in the emitted `Weights::load`.
    /// `None` = no scaling (standard `new_from_stream` with plain
    /// base freqs).
    pub rope_scaling: Option<RopeScaling>,
    /// Deterministic content hash of the raw `rope_scaling` JSON
    /// subobject, baked into this variant's `fingerprint_matches`.
    /// Discriminates checkpoints that share `type` +
    /// `max_position_embeddings` but differ in `short_factor` /
    /// `long_factor` — e.g. Phi-3.5-mini vs Phi-3-mini-128k,
    /// Phi-4-mini-instruct vs Phi-4-mini-reasoning. `None` when the
    /// config has no `rope_scaling`.
    pub rope_scaling_hash: Option<u64>,
    /// Qwen2-VL / Qwen2.5-VL `rope_scaling.mrope_section`: rotary-pair
    /// counts for the (T, H, W) axes of the multimodal RoPE
    /// generalization. Sums to `head_dim/2`. Drives the per-variant
    /// `const MROPE_SECTION: Option<[u32; 3]> = Some([..]);` override
    /// emitted on `impl CanonicalParams`. `None` for every text-only
    /// arch (the rope kernel takes the legacy 1D-positions path).
    /// See `~/.claude/plans/distributed-mapping-map.md` Phase A/B.
    pub mrope_section: Option<[u32; 3]>,
    /// Everything the arch crate DECLARES about itself (safetensors
    /// layout, fingerprint, rope style, decoder prefix, bound
    /// defaults, weight renames, …) — see [`crate::arch_spec`]. The
    /// declared values arrive from the `#[forward]` /
    /// `#[vision_forward]` attribute args; per-checkpoint JSON drift
    /// (`.overrides.json` / explicit config fields) is merged on top
    /// during parse, so consumers read ONE resolved spec. Replaces
    /// the former per-quirk `Option` fields — model-arch knowledge
    /// lives in the arch crate, never as switches in this crate.
    pub arch: crate::arch_spec::DeclaredArchSpec,
    /// HF `torch_dtype` string, lowercased. Read by codegen as the
    /// FALLBACK for the rotary cache compute dtype when the embed
    /// tensor isn't visible in the weights table at load time. Today's
    /// primary path reads `embed_tokens.weight`'s on-disk dtype at
    /// runtime — see `emit_weights_struct`'s `rotary_prelude`. AWQ
    /// variants frequently override the base dtype (Qwen2.5 base
    /// ships bf16; `*-Instruct-AWQ` ships f16) while keeping the
    /// manifest `torch_dtype` literal unchanged, so the runtime read
    /// is load-bearing for AWQ correctness. `None` when the JSON
    /// omits the field.
    pub torch_dtype: Option<String>,
}

/// On-disk safetensors layout for a vision tower. Drives
/// [`crate::codegen::safetensors_prefix`] for `Prelude::Vision`
/// programs. Field semantics in [`ModelParams::vision_layout`].
#[derive(Clone, Debug)]
pub struct VisionSafetensorsLayout {
    /// Disk root for unindexed weights (e.g. `visual` for Qwen2-VL,
    /// `vision_tower.vision_model` for Gemma3-MM SigLIP).
    pub default_root: String,
    /// Suffix appended after `default_root` for indexed (per-block)
    /// weights — e.g. `blocks` (Qwen2-VL: `visual.blocks.{l}.*`),
    /// `encoder.layers` (Gemma3-MM: `vision_tower.vision_model.encoder.layers.{l}.*`).
    pub layered_subpath: String,
    /// First-DSL-segment → disk-prefix overrides for sibling
    /// subtrees. Lookups treat the matching subtree as unindexed —
    /// the override fully replaces the `<default_root>(.<layered_subpath>.{l})?`
    /// prefix. Example: `{"mm" → "multi_modal_projector"}` routes
    /// Gemma3-MM's `mm.mm_soft_emb_norm` to disk
    /// `multi_modal_projector.mm_soft_emb_norm`.
    pub subtrees: BTreeMap<String, String>,
}

impl VisionSafetensorsLayout {}

/// d_model fingerprint key + dim — see [`ModelParams::vision_d_model_fingerprint`].
#[derive(Clone, Debug)]
pub struct VisionDModelFingerprint {
    pub key: String,
    pub dim: usize,
}

impl VisionDModelFingerprint {}

/// Patch-embed flatten target — see [`ModelParams::vision_patch_embed_flatten`].
#[derive(Clone, Debug)]
pub struct VisionPatchEmbedFlatten {
    pub key: String,
    pub leading_dim: usize,
    /// `true` when the on-disk conv weight is channels-LAST
    /// (`[out, kt, kh, kw, in]`, MLX convention) and must be permuted to
    /// `[out, in, kt, kh, kw]` before flattening so it pairs with the
    /// channels-FIRST `[in, kt, kh, kw]` patch packing. `false` (the
    /// default) does a plain reshape — correct for channels-first
    /// (torch/SigLIP `[out, in, p, p]`) checkpoints.
    pub channels_last: bool,
}

impl VisionPatchEmbedFlatten {}

/// Arch-agnostic rope-scaling flavor parsed from `config.json`.
/// Integer-valued `original_max_position_embeddings` is kept as
/// `u64` so both `bounds` readers and the rotary kernels (which
/// want `usize`) can consume it without ambiguity.
#[derive(Clone, Debug)]
pub enum RopeScaling {
    /// `rope_scaling.type == "llama3"` — frequency-bucketed scaling
    /// used by Llama-3.x. All fields come from `rope_scaling.*`.
    Llama3 {
        factor: f64,
        low_freq_factor: f64,
        high_freq_factor: f64,
        original_max_position_embeddings: u64,
    },
    /// `rope_scaling.type == "longrope"` or `"su"` — Phi-3 LongRoPE
    /// (su-scaling). Factor vectors have length `rotary_dim/2`.
    /// `short_mscale` / `long_mscale` default to the Phi-3 paper
    /// formula when omitted from the config — Python vLLM's
    /// `Phi3LongRoPEScaledRotaryEmbedding.__init__` materializes
    /// the default from `scaling_factor = sqrt(1 + ln(max_pos/orig_max) / ln(orig_max))`
    /// when the caller passes `None`.
    LongRope {
        short_factor: Vec<f64>,
        long_factor: Vec<f64>,
        original_max_position_embeddings: u64,
        short_mscale: f64,
        long_mscale: f64,
    },
    /// `rope_scaling.type == "yarn"` — DeepSeek-V2 YaRN NTK-by-parts
    /// interpolation with mscale correction.
    Yarn {
        factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        mscale: f64,
        mscale_all_dim: f64,
        original_max_position_embeddings: u64,
    },
}

/// Errors produced while loading configs.
#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    NotADirectory(PathBuf),
    BadStem {
        path: PathBuf,
        reason: &'static str,
    },
    Quantization {
        path: PathBuf,
        source: crate::quantization::ParseError,
    },
    BadQuantizations {
        path: PathBuf,
        reason: &'static str,
    },
    /// A `#[vision_forward]` config (verbatim HF VL-wrapper
    /// checkpoint) is missing a field the per-family `vision_*`
    /// bound derivation needs, or its arch family is unknown to
    /// [`VisionFamily`].
    VisionDerivation {
        path: PathBuf,
        reason: String,
    },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Json { path, source } => write!(f, "parsing {}: {source}", path.display()),
            Self::NotADirectory(p) => write!(f, "not a directory: {}", p.display()),
            Self::BadStem { path, reason } => {
                write!(f, "bad file stem for {}: {reason}", path.display())
            }
            Self::Quantization { path, source } => {
                write!(f, "quantization_config in {}: {source}", path.display())
            }
            Self::BadQuantizations { path, reason } => {
                write!(f, "{}: {reason}", path.display())
            }
            Self::VisionDerivation { path, reason } => {
                write!(f, "vision config {}: {reason}", path.display())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// Load every `*.json` file in `dir` as a `ModelParams`, then
/// synthesize additional `<size>-<preset>` variants for every
/// `quantizations.json`-declared preset × dense base. Dense bases,
/// preset overlays, and optional `<size>-<preset>.overrides.json`
/// files are deep-merged at load time.
///
/// Files skipped from the dense-base scan:
/// - `weights.json` — per-arch shape manifest, loaded separately.
/// - `quantizations.json` — preset declaration list.
/// - `*.overrides.json` — per-(size, preset) drift overrides
///   applied during synthesis.
///
/// Overlay presets are looked up under `<dir>/../quantizations/`
/// (sibling of the arch directory). Each preset's JSON fragment
/// deep-merges onto the base — typically just
/// `{quantization_config: {...}}`, but any top-level field is
/// allowed.
///
/// Results are sorted alphabetically by file stem for build
/// determinism; synthesized variants appear after their base.
pub fn load_dir(
    dir: &Path,
    spec: &crate::arch_spec::DeclaredArchSpec,
) -> Result<Vec<ModelParams>, ConfigError> {
    load_dir_mode(dir, ConfigMode::Decoder, spec)
}

/// [`load_dir`] for a `#[vision_forward]` crate's configs/ dir.
/// Same file discovery / overlay synthesis / filtering; per-file
/// `ModelParams` construction goes through the vision path
/// (per-family `vision_*` bound derivation from the nested HF
/// `vision_config` block) instead of the flat decoder harvest.
pub fn load_dir_vision(
    dir: &Path,
    spec: &crate::arch_spec::DeclaredArchSpec,
) -> Result<Vec<ModelParams>, ConfigError> {
    load_dir_mode(dir, ConfigMode::Vision, spec)
}

/// Process-wide set of `SCRATCHY_BUILD_FILTER` tags that matched at least
/// one stem, across every arch directory `load_dir`/`load_dir_vision` has
/// processed so far. One arch not containing a tag (e.g. "3.2-instruct"
/// matching granite but not llama) is expected and not an error — only a
/// tag matching NOTHING anywhere is, which `unmatched_build_filter_tags`
/// (called once, after every arch has been emitted) catches.
fn build_filter_seen() -> &'static std::sync::Mutex<std::collections::HashSet<String>> {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// `SCRATCHY_BUILD_FILTER` tags that never matched a single stem in any
/// arch, across the whole build. Call once, after every `load_dir`/
/// `load_dir_vision` invocation has run (i.e. from `scratchy-forwards.rs`
/// after its `par_iter` over arches completes) — a typo'd tag should fail
/// the build loudly instead of silently compiling nothing.
pub fn unmatched_build_filter_tags() -> Vec<String> {
    let requested: Vec<String> = std::env::var("SCRATCHY_BUILD_FILTER")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|x| x.trim().to_lowercase())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let seen = build_filter_seen().lock().unwrap();
    requested
        .into_iter()
        .filter(|t| !seen.contains(t))
        .collect()
}

/// Process-wide count of models actually forward-expanded (dense bases +
/// synthesized quant variants), across every arch directory processed so
/// far. `scratchy-models` has no default model/quant features (deliberately
/// — a plain `-Fmetal` with nothing else selected should fail loudly, not
/// silently link a binary with zero models). Call once, after every
/// `load_dir`/`load_dir_vision` invocation has run (i.e. from
/// `scratchy-forwards.rs` after its `par_iter` over arches completes).
fn total_models_emitted_counter() -> &'static std::sync::atomic::AtomicUsize {
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    &COUNT
}

pub fn total_models_emitted() -> usize {
    total_models_emitted_counter().load(std::sync::atomic::Ordering::Relaxed)
}

/// Which macro is consuming the configs — `#[forward]` (Decoder) or
/// `#[vision_forward]` (Vision). The configs/ files themselves are
/// VERBATIM HF checkpoint configs either way; the mode selects which
/// view of the checkpoint the crate compiles:
///
/// - **Decoder** harvests the flat top-level fields (after
///   `normalize_hf_config` hoists `text_config` / `rope_parameters`)
///   — the text decoder's identity.
/// - **Vision** derives the `vision_*` bound set + `d_model` +
///   `vision_norm_eps` from the nested `vision_config` block,
///   arch-family-keyed, and deliberately harvests NOTHING else from
///   the top level (a VL-wrapper's text fields like
///   `intermediate_size` / `head_dim` would otherwise leak into the
///   vision expansion's `W::` consts — e.g. shadowing Qwen2.5-VL's
///   `vision_intermediate_size_padded` SwiGLU split-point).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ConfigMode {
    Decoder,
    Vision,
}

static STEM_LOG_INIT: std::sync::Once = std::sync::Once::new();

fn init_stem_logging() {
    STEM_LOG_INIT.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_writer(std::io::stderr)
            .try_init();
    });
}

fn load_dir_mode(
    dir: &Path,
    mode: ConfigMode,
    spec: &crate::arch_spec::DeclaredArchSpec,
) -> Result<Vec<ModelParams>, ConfigError> {
    init_stem_logging();
    if !dir.is_dir() {
        return Err(ConfigError::NotADirectory(dir.to_path_buf()));
    }

    // Per-model scoping: every checked-in `configs/<arch>/<stem>.json` has
    // its own Cargo feature (crates/models/arch/Cargo.toml) — bare `<stem>`
    // normally. No `model-` text prefix: `crates/cli/scr/Cargo.toml`
    // depends on this crate under the local rename `model = { package =
    // "scratchy-models", ... }`, so Cargo's `pkg/feature` CLI syntax already
    // reads as `--features model/granite-3.1-2b-instruct` — the alias
    // carries the "this is a model" meaning.
    //
    // A handful of stems are byte-identical files checked into two arch
    // dirs at once (the same checkpoint, loadable via a vision arch or its
    // non-vision base arch — e.g. `gemma-3-12b-it` under both `gemma3/` and
    // `gemma3-mm/`); for those the vision arch keeps the bare stem and the
    // base arch's copy is `<stem>-text-only` (VISION_ARCH_FOR_STEM below,
    // verified per pair against crates/models/arch/Cargo.toml — mirrors the
    // Cargo.toml feature-name generator exactly, must stay in sync with it).
    // This table matters for correctness, not just naming: the bare `<stem>`
    // feature genuinely exists (scoped to the vision arch), so a naive
    // "check bare OR arch-prefixed OR text-only-suffixed, any hit counts"
    // probe would cross-contaminate — enabling the vision arch's bare
    // feature would ALSO look enabled to the base arch's identical-stem
    // file, since Cargo features are one flat namespace with no directory
    // scoping baked into the env var. So each (arch, stem) pair here checks
    // EXACTLY the one candidate name that arch could legitimately own, never
    // a blind OR across candidates that belong to a different arch.
    //
    // Any other collision not covered by the table below falls back to
    // `<arch>-<stem>` — safe to also probe bare `<stem>` there without
    // cross-contamination risk, because in that fallback shape Cargo never
    // declares a bare `<stem>` feature for EITHER colliding arch (only the
    // two `<arch>-<stem>` forms), so the bare env var can never be set.
    //
    // A base whose feature isn't enabled is skipped BEFORE `read_json_file`
    // ever opens it below — the point being an unselected model's JSON is
    // never even read, let alone forward-expanded. Read straight from
    // Cargo's own `CARGO_FEATURE_*` env vars — this runs in the same
    // build-script process as the feature-gated crate, so there's no
    // proc-macro/IPC boundary to relay across and no need for a self-invented
    // env var. Mirrors Cargo's own feature-name -> env-var transform
    // (uppercase, `-` -> `_`, every other byte verbatim — including `.`).
    const VISION_ARCH_FOR_STEM: &[(&str, &str)] = &[
        ("qwen2-vl-2b-mlx", "qwen2-vl"),
        ("qwen2.5-vl-3b-mlx", "qwen2-5-vl"),
        ("locateanything-3b", "locateanything"),
        ("qwen3.5-9b", "qwen3-5-vl"),
        ("gemma-3-12b-it", "gemma3-mm"),
        ("gemma-3-27b-it", "gemma3-mm"),
        ("gemma-3-4b-it", "gemma3-mm"),
    ];
    let arch_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| ConfigError::NotADirectory(dir.to_path_buf()))?;
    let feature_env_var = |feature: &str| -> String {
        format!("CARGO_FEATURE_{}", feature.to_uppercase().replace('-', "_"))
    };
    let model_feature_enabled = |stem: &str| -> bool {
        if let Some((_, vision_arch)) = VISION_ARCH_FOR_STEM.iter().find(|(s, _)| *s == stem) {
            let candidate = if arch_name == *vision_arch {
                stem.to_string()
            } else {
                format!("{stem}-text-only")
            };
            return std::env::var(feature_env_var(&candidate)).is_ok();
        }
        std::env::var(feature_env_var(stem)).is_ok()
            || std::env::var(feature_env_var(&format!("{arch_name}-{stem}"))).is_ok()
    };

    // Quant scoping. `SCRATCHY_QUANTS=preset1,preset2,...` restricts the
    // synthesized `<base>-<preset>` variants to those presets (dense bases
    // always emit). The per-arch build.rs forward-maps the enabled `quant-*`
    // Cargo features (CARGO_FEATURE_QUANT_*) to preset names and forwards them
    // here via `cargo:rustc-env` — so this is driven by Cargo FEATURES, not a
    // hand-set env var. UNSET = legacy behaviour (every backend-applicable
    // preset emits), so un-migrated arch crates are unaffected; SET (even to an
    // empty list) = emit only the listed presets, i.e. dense-only by default.
    let enabled_quants: Option<std::collections::HashSet<String>> =
        std::env::var("SCRATCHY_QUANTS").ok().map(|s| {
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        });
    // Whether `preset_name` both survived Cargo-feature quant scoping AND is
    // one this backend can actually claim. Shared by the dense-suppression
    // pre-check (below, before the base loop) and the overlay-synthesis
    // loop (further down) so the two can never drift apart — a preset that
    // wouldn't actually synthesize must never suppress the dense fallback,
    // or a model ends up with ZERO compiled variants.
    //
    // Quant feature scoping: when SCRATCHY_QUANTS is set (the arch build.rs
    // forward-maps the enabled `quant-*`-equivalent Cargo features from
    // scratchy-quantizations), only listed presets are active. Unset ⇒
    // legacy all-presets behaviour.
    //
    // Backend claimability — metal: `mlx-affine-*` variants reach the
    // solver as Dense (codegen dequantizes at load time), and `nvfp4`
    // reaches it as a genuine metal quant claimed by `MetalNvfp4QmmImpl`
    // (forward-time E2M1 dequant-on-read qmv/qmm_t). Every other preset
    // routes through CUDA-only Impls (Marlin / Bnb4 / Fp8 / Ggml) that the
    // metal impl pool has no claimants for, so the solver would explode
    // with `UnclaimedTile`.
    //
    // Backend claimability — cuda: the mirror. `mlx-affine-*` weights are
    // an Apple checkpoint format with no CUDA Impl in the pool (the Affine
    // quant flow lives entirely in the metal kernels). `nvfp4` is currently
    // metal-only too (E2M1 dequant qmv/qmm_t shaders + `MetalNvfp4QmmImpl`);
    // a CUDA NVFP4 path is future work. Skip both so the cuda solver
    // doesn't fail with `UnclaimedTile` on `Embed`/`Gemm` for those storage
    // tags.
    let quant_preset_active = |preset_name: &str| -> bool {
        if let Some(q) = &enabled_quants
            && !q.contains(preset_name)
        {
            return false;
        }
        if cfg!(feature = "metal")
            && !preset_name.starts_with("mlx-affine-")
            && preset_name != "nvfp4"
        {
            return false;
        }
        if cfg!(feature = "cuda")
            && (preset_name.starts_with("mlx-affine-") || preset_name == "nvfp4")
        {
            return false;
        }
        true
    };

    // Stopgap for the one precision `arch-*`/`size-*`/`quant-*` can't express:
    // picking a specific version + base-vs-instruct pair (e.g. a size tier
    // still spans every generation and variant in that param range).
    // `SCRATCHY_BUILD_FILTER=<tag>,<tag>,...` (case-insensitive substring,
    // OR'd across tags) restricts both base configs and synthesized quant
    // variants to stems containing at least one tag. UNSET = no filter, same
    // as every other scoping knob in this file. Hand-set, so it needs
    // `cargo:rerun-if-env-changed=SCRATCHY_BUILD_FILTER` in
    // scratchy-forwards.rs (already wired) to avoid a stale-cache footgun.
    // Tags are recorded as matched in `build_filter_seen`
    // so `unmatched_build_filter_tags` can catch a typo'd tag at the end of
    // the whole build instead of silently compiling nothing.
    // Stems are `<arch>-<version>-<size>-<variant>` (e.g.
    // `granite-3.2-2b-instruct`), so a tag like `3.2-instruct` never
    // appears as one contiguous substring — `2b` sits between the version
    // and the variant. Each tag is matched as a SET of `-`-separated parts,
    // ALL of which must appear as substrings of the stem (any order, not
    // necessarily adjacent); tags themselves are OR'd.
    let build_filter: Vec<Vec<String>> = std::env::var("SCRATCHY_BUILD_FILTER")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|tag| tag.trim().to_lowercase())
                .filter(|tag| !tag.is_empty())
                .map(|tag| tag.split('-').map(|p| p.to_string()).collect())
                .collect()
        })
        .unwrap_or_default();

    let stem_matches_build_filter = |stem: &str| -> bool {
        if build_filter.is_empty() {
            return true;
        }
        let lower = stem.to_lowercase();
        let mut any = false;
        for parts in &build_filter {
            if parts.iter().all(|p| lower.contains(p.as_str())) {
                build_filter_seen().lock().unwrap().insert(parts.join("-"));
                any = true;
            }
        }
        any
    };

    let all_json: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|source| ConfigError::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();

    let is_overrides = |p: &Path| {
        p.file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.ends_with(".overrides.json"))
            .unwrap_or(false)
    };

    let mut base_paths: Vec<PathBuf> = all_json
        .iter()
        .filter(|p| {
            let name = p.file_name().and_then(|s| s.to_str());
            !matches!(name, Some("weights.json") | Some("quantizations.json")) && !is_overrides(p)
        })
        .cloned()
        .collect();
    base_paths.sort_by(|a, b| {
        a.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .cmp(b.file_stem().and_then(|s| s.to_str()).unwrap_or(""))
    });

    // A quant preset means "compile this precision instead of dense", not
    // "compile this precision too" — if this arch has at least one preset
    // that's both selected (via `quant/<preset>`, forwarded through
    // SCRATCHY_QUANTS) and actually claimable on this backend, every
    // selected base in this arch gets its quant variant(s) INSTEAD of the
    // dense/bf16 one below, not alongside it. Deliberately arch-wide, not
    // per-base: `quantizations.json` is arch-wide too, so there is no
    // per-model "just this one dense" escape hatch today — if that's ever
    // needed, `SCRATCHY_BUILD_FILTER` can still narrow further.
    let arch_has_active_quant = declared_preset_names(&dir.join("quantizations.json"))?
        .iter()
        .any(|name| quant_preset_active(name));

    let mut out: Vec<ModelParams> = Vec::new();
    // Remember each base's raw JSON so we can deep-merge overlays
    // onto it without re-reading and without copying the ModelParams
    // structure (synthesized variants have different bounds/quant
    // and need to re-parse from the merged raw JSON).
    let mut base_raw: Vec<(PathBuf, String, serde_json::Value)> = Vec::new();
    for p in &base_paths {
        let stem = stem_of(p)?;
        // Per-model gate: skip entirely — never even `read_json_file` —
        // when this base's `<stem>` (or `<arch>-<stem>`) feature isn't enabled.
        // Every quant variant synthesized from this base below shares its
        // raw JSON via `base_raw`, so skipping here (instead of after
        // parsing) also means an unselected base contributes no quant
        // variants, not just no dense emission.
        if !model_feature_enabled(&stem) {
            continue;
        }
        let (raw, mut json) = read_json_file(p)?;
        // Dense-base drift surface: `<stem>.overrides.json` deep-merges
        // onto the verbatim checkpoint config (same mechanism the
        // synthesized quant variants already get). Tracked so cargo
        // rebuilds on override edits.
        let mut tracked: Vec<PathBuf> = Vec::new();
        let override_path = dir.join(format!("{stem}.overrides.json"));
        if override_path.exists() {
            let (_, override_json) = read_json_file(&override_path)?;
            deep_merge(&mut json, &override_json);
            tracked.push(override_path);
        }
        // Build filter gates the DENSE emission only — a filter naming a
        // quant suffix never matches a base's own stem, but the base still
        // needs to stay in `base_raw` so a matching quant variant can be
        // synthesized from it below. `arch_has_active_quant` gates it too:
        // a selected quant preset replaces dense, it doesn't add to it.
        if stem_matches_build_filter(&stem)
            && !arch_has_active_quant
            && (cfg!(feature = "metal") || !ships_mixed_bit_affine(&json))
        {
            tracing::info!("[COMPILING] base {}", stem);
            let model = model_params_from_json_mode(&json, &stem, p, tracked, mode, spec)?;
            out.push(model);
            total_models_emitted_counter().fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        base_raw.push((p.clone(), stem, json));
        let _ = raw;
    }

    // Overlay synthesis. Each arch's `quantizations.json` is a flat
    // list of preset names; every listed preset fans out across
    // every dense base in the arch to produce one compiled variant
    // per `(size, preset)` pair. Per-HF-repo drift (e.g.
    // TinyLlama-GPTQ's vocab_size=32003 vs the dense base's 32000)
    // lands as `<size>-<preset>.overrides.json` files that deep-
    // merge last.
    //
    // Universal fan-out is only compile-affordable because codegen
    // does cross-variant forward-fn deduplication — variants with
    // identical Impl-set signatures share one emitted body. Without
    // that dedup, N sizes × M presets quickly blow up release-build
    // LLVM work.
    // Under `--features metal` we skip CUDA-only quant variants
    // (awq-gemm / gptq-* / bnb-nf4-dq / fp8-* / ct-int4-sym / ggml) —
    // their on-disk layouts route through `MarlinFusedGateUpSiluMul` /
    // `Bnb4Linear` / `Fp8Linear` / `GgmlLinear` impls that the metal
    // impl pool has no claimants for, so the solver explodes with
    // `UnclaimedTile` on the first quant tile. A CUDA-only preset being
    // skipped this way on metal means it was never `quant_preset_active`
    // in the first place, so it also never suppressed the dense base
    // above — dense only drops out for a preset that actually synthesizes.
    //
    // MLX-affine presets (`mlx-affine-b<bits>-g<gs>`) are the
    // exception: their codegen materializes a Dense `LinearLayer` at
    // load time via `LinearLayer::load_affine_dequant_as_dense`, so
    // the forward path remains the existing bf16 Gemm path that the
    // metal impl pool already claims. The preset-filter inside the
    // overlay loop below keeps CUDA-only presets out while still
    // synthesizing the affine variant.
    let quantizations_path = dir.join("quantizations.json");
    let presets = declared_preset_names(&quantizations_path)?;
    if !presets.is_empty() {
        // Presets live in the shared `quantization` crate.
        // Walk up from `dir` (which is `<crate>/configs/`) to find
        // the workspace root (first ancestor with a `Cargo.toml`
        // containing `[workspace]`), then descend to
        // `crates/models/quantization/presets/`.
        let preset_root = {
            let mut cur: Option<&Path> = Some(dir);
            let mut found: Option<PathBuf> = None;
            while let Some(d) = cur {
                let cargo_toml = d.join("Cargo.toml");
                if cargo_toml.exists()
                    && let Ok(s) = fs::read_to_string(&cargo_toml)
                    && s.contains("[workspace]")
                {
                    found = Some(
                        d.join("crates")
                            .join("models")
                            .join("quantization")
                            .join("presets"),
                    );
                    break;
                }
                cur = d.parent();
            }
            found.ok_or_else(|| ConfigError::NotADirectory(dir.to_path_buf()))?
        };

        for preset_name in &presets {
            // Quant feature scoping (SCRATCHY_QUANTS) + backend claimability
            // (metal only claims mlx-affine-*/nvfp4 as those route through
            // Dense/MetalNvfp4QmmImpl; cuda is the mirror — see
            // `quant_preset_active`'s definition above for the full backend
            // reasoning, shared with the dense-suppression pre-check so the
            // two can't drift apart).
            if !quant_preset_active(preset_name) {
                continue;
            }
            let preset_path = preset_root.join(format!("{preset_name}.json"));
            let (_, preset_json) = read_json_file(&preset_path)?;

            for (base_path, base_stem, base_json) in &base_raw {
                let variant_stem = format!("{base_stem}-{preset_name}");
                if !stem_matches_build_filter(&variant_stem) {
                    continue;
                }
                let override_path = dir.join(format!("{variant_stem}.overrides.json"));

                let mut merged = base_json.clone();
                deep_merge(&mut merged, &preset_json);
                let mut extra_tracked = vec![preset_path.clone()];
                if override_path.exists() {
                    let (_, override_json) = read_json_file(&override_path)?;
                    deep_merge(&mut merged, &override_json);
                    extra_tracked.push(override_path);
                }

                tracing::info!("[COMPILING] variant {}", variant_stem);
                let variant = model_params_from_json_mode(
                    &merged,
                    &variant_stem,
                    base_path,
                    extra_tracked,
                    mode,
                    spec,
                )?;
                out.push(variant);
                total_models_emitted_counter().fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    // Drop any explicitly-quantized base configs under
    // `--features metal` *except* MLX-affine, which routes through the
    // existing Dense Impl pool at solve time (codegen materializes a
    // Dense `LinearLayer` at load via
    // `LinearLayer::load_affine_dequant_as_dense`). Other
    // checked-in `<size>-<preset>.json` base configs
    // (e.g. `qwen3-0.6b-bnb-4bit.json`) still reach the solver as
    // quantized and the metal impl pool has no claimants — the
    // solver would explode on the first quantized tile.
    if cfg!(feature = "metal") {
        out.retain(|m| {
            m.quantization.is_none()
                || matches!(
                    m.quantization.as_ref().map(|qc| &qc.method),
                    Some(crate::quantization::QuantMethod::Affine { .. })
                        | Some(crate::quantization::QuantMethod::Nvfp4 { .. })
                )
        });
    }

    // Keep the final list sorted by stem so emitted
    // arch-dispatcher arm order stays deterministic across
    // `quantizations.json` edits.
    out.sort_by(|a, b| a.source_stem.cmp(&b.source_stem));

    Ok(out)
}

/// Load a single config.json as a dense-base ModelParams.
///
/// Test-only: the macro reads configs through the directory walk, not one file at a time.
#[cfg(test)]
pub fn load_file(path: &Path) -> Result<ModelParams, ConfigError> {
    let (_, json) = read_json_file(path)?;
    let stem = stem_of(path)?;
    model_params_from_json(
        &json,
        &stem,
        path,
        Vec::new(),
        &crate::arch_spec::DeclaredArchSpec::default(),
    )
}

/// Read + parse a JSON file, returning the raw string (for
/// diagnostics) and the parsed `Value`. Centralized so
/// `ConfigError::{Io, Json}` always carry the right path.
/// Whether a base config ships its own MIXED-BIT affine quantization — an
/// OptiQ checkpoint: a `quantization_config` carrying PER-TENSOR overrides
/// (`"...embed_tokens": { "bits": 8, .. }`) on top of the scalar
/// `group_size`/`bits`/`mode`, rather than one width for the whole model.
///
/// Only metal lowers those today: the per-projection widths are read back by
/// `unpack_moe_expert_bits` / `affine_gather_qmv_symbol`, and cuda's ISel has
/// no `Embed` Impl that claims a mixed-bit quantized embedding — it refuses
/// the solve outright ("no Impl in the library matched tile 0 op Embed").
///
/// So `all` is BACKEND-RELATIVE for these, exactly as it already is for
/// synthesized quant variants via `quant_preset_active` above: the same
/// feature names every checked-in config, and each backend emits the subset
/// it can actually lower. Derived from the config's own contents, not a stem
/// list, so a new OptiQ checkpoint is covered the day it lands.
fn ships_mixed_bit_affine(json: &serde_json::Value) -> bool {
    json.get("quantization_config")
        .and_then(|q| q.as_object())
        .is_some_and(|q| q.values().any(|v| v.is_object()))
}

fn read_json_file(path: &Path) -> Result<(String, serde_json::Value), ConfigError> {
    let contents = fs::read_to_string(path).map_err(|source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let json: serde_json::Value =
        serde_json::from_str(&contents).map_err(|source| ConfigError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    Ok((contents, json))
}

/// Preset names declared in an arch's `quantizations.json` (empty if the
/// arch has none). Each entry is either a bare preset name (`"fp8-..."`,
/// `"ggml"`) or a single-key object carrying registration data for that
/// preset's runtime side (the object form's structured fields are picked up
/// separately by `load_gguf_spec` — this only needs the names). Shared by
/// the dense-suppression pre-check and the overlay-synthesis loop in
/// `load_dir_mode` so they can never see a different preset list.
fn declared_preset_names(quantizations_path: &Path) -> Result<Vec<String>, ConfigError> {
    if !quantizations_path.is_file() {
        return Ok(Vec::new());
    }
    let (_, qjson) = read_json_file(quantizations_path)?;
    let entries = qjson
        .get("quantizations")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ConfigError::BadQuantizations {
            path: quantizations_path.to_path_buf(),
            reason: "expected `{\"quantizations\": [<entry>, ...]}` array",
        })?;
    Ok(entries
        .iter()
        .filter_map(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else if let Some(obj) = v.as_object()
                && obj.len() == 1
            {
                obj.keys().next().cloned()
            } else {
                None
            }
        })
        .collect())
}

/// Per-arch GGUF registration data harvested from `quantizations.json`.
/// Populated only when the arch's quantizations list contains a `"ggml"`
/// entry (string or object form). Forwarded into the
/// `scratchy_quantizations::gguf::register!` call the macro emits.
#[derive(Debug, Clone)]
pub struct GgufSpec {
    /// GGUF `general.architecture` override. Defaults to the `arch`
    /// ident on the `#[forward]` block; set explicitly when the arch
    /// reports a non-matching tag (Mistral GGUFs report `"llama"`,
    /// DeepSeek-V2 `"deepseek2"`, etc.).
    pub gguf_arch: Option<String>,
    pub qk_permute: bool,
    pub llama3_rope_scaling_inference: bool,
    /// Per-suffix tensor renames: `(gguf_suffix, hf_suffix)` pairs.
    pub tensor_renames: Vec<(String, String)>,
    /// `(gguf_key_template, extra_key)` u32 reads. The template may
    /// contain `{arch}` which `apply_metadata` substitutes.
    pub metadata_u32: Vec<(String, String)>,
    pub metadata_f32: Vec<(String, String)>,
    /// Constants always inserted into `HfModelConfig.extra` (split by
    /// numeric type so the macro can emit the right `GgufDefault`
    /// variant).
    pub metadata_defaults_u32: Vec<(String, u32)>,
    pub metadata_defaults_f32: Vec<(String, f32)>,
    /// Subtracted from every rmsnorm weight at GGUF load time. Lets
    /// archs (Gemma2/3) whose llama.cpp converter pre-bakes a
    /// constant into the stored weight recover the canonical "raw w"
    /// shape so the runtime kernel's `(w + offset)` fold isn't
    /// double-applied. Default 0.0.
    pub norm_weight_offset: f32,
    /// Whether this arch is the canonical owner of the gguf tag —
    /// the only crate that emits the inventory `scratchy_quantizations::gguf::register!`
    /// call. Default `true`. Set `false` on non-canonical claimants
    /// (e.g. Mistral for `"llama"`, deepseek-v3-flat for `"deepseek2"`)
    /// so a single deterministic spec covers each gguf_arch. The arch
    /// still gets a `-ggml` overlay variant + `gguf_archs` entry in
    /// its `ScratchyArchRegistration` so the dispatcher tries it.
    pub register_spec: bool,
}

impl Default for GgufSpec {
    fn default() -> Self {
        Self {
            gguf_arch: None,
            qk_permute: false,
            llama3_rope_scaling_inference: false,
            tensor_renames: Vec::new(),
            metadata_u32: Vec::new(),
            metadata_f32: Vec::new(),
            metadata_defaults_u32: Vec::new(),
            metadata_defaults_f32: Vec::new(),
            norm_weight_offset: 0.0,
            register_spec: true,
        }
    }
}

/// Read the arch's `quantizations.json` and return its GGUF spec —
/// `Some` when the list contains a `"ggml"` entry (string or object
/// form), `None` otherwise.
///
/// Schema for the structured form (all fields optional):
///
/// ```json
/// {"ggml": {
///   "qk_permute": true,
///   "gguf_arch": "llama",
///   "tensor_renames": {
///     "attn_qkv.weight": "self_attn.qkv_proj.weight"
///   },
///   "metadata_u32": {"{arch}.attention.sliding_window": "sliding_window"},
///   "metadata_f32": {"{arch}.attention.scale": "attention_multiplier"},
///   "metadata_defaults": {"sliding_window_pattern": 6},
///   "llama3_rope_scaling_inference": true
/// }}
/// ```
pub fn load_gguf_spec(dir: &Path) -> Result<Option<GgufSpec>, ConfigError> {
    let path = dir.join("quantizations.json");
    if !path.exists() {
        return Ok(None);
    }
    let (_, qjson) = read_json_file(&path)?;
    let arr = match qjson.get("quantizations").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(None),
    };
    let mut data: Option<&serde_json::Value> = None;
    let mut found_bare = false;
    for entry in arr {
        if entry.as_str() == Some("ggml") {
            found_bare = true;
        }
        if let Some(obj) = entry.as_object()
            && let Some(d) = obj.get("ggml")
        {
            data = Some(d);
            break;
        }
    }
    if !found_bare && data.is_none() {
        return Ok(None);
    }
    let mut spec = GgufSpec::default();
    let Some(data) = data else {
        return Ok(Some(spec));
    };
    let obj = data
        .as_object()
        .ok_or_else(|| ConfigError::BadQuantizations {
            path: path.clone(),
            reason: "`ggml` entry must be a string or `{\"ggml\": {<fields>}}` object",
        })?;
    let bad = || ConfigError::BadQuantizations {
        path: path.clone(),
        reason: "ggml field has wrong shape",
    };
    if let Some(v) = obj.get("gguf_arch").and_then(|v| v.as_str()) {
        spec.gguf_arch = Some(v.to_string());
    }
    if let Some(v) = obj.get("qk_permute").and_then(|v| v.as_bool()) {
        spec.qk_permute = v;
    }
    if let Some(v) = obj
        .get("llama3_rope_scaling_inference")
        .and_then(|v| v.as_bool())
    {
        spec.llama3_rope_scaling_inference = v;
    }
    if let Some(v) = obj.get("norm_weight_offset").and_then(|v| v.as_f64()) {
        spec.norm_weight_offset = v as f32;
    }
    if let Some(v) = obj.get("register_spec").and_then(|v| v.as_bool()) {
        spec.register_spec = v;
    }
    if let Some(v) = obj.get("tensor_renames") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            spec.tensor_renames
                .push((k.clone(), val.as_str().ok_or_else(bad)?.to_string()));
        }
    }
    if let Some(v) = obj.get("metadata_u32") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            spec.metadata_u32
                .push((k.clone(), val.as_str().ok_or_else(bad)?.to_string()));
        }
    }
    if let Some(v) = obj.get("metadata_f32") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            spec.metadata_f32
                .push((k.clone(), val.as_str().ok_or_else(bad)?.to_string()));
        }
    }
    if let Some(v) = obj.get("metadata_defaults") {
        let map = v.as_object().ok_or_else(bad)?;
        for (k, val) in map {
            // Pick the variant by JSON type — integers go to u32,
            // numbers to f32. Bools/strings are not currently used.
            if let Some(n) = val.as_u64() {
                spec.metadata_defaults_u32
                    .push((k.clone(), n.try_into().map_err(|_| bad())?));
            } else if let Some(f) = val.as_f64() {
                spec.metadata_defaults_f32.push((k.clone(), f as f32));
            } else {
                return Err(bad());
            }
        }
    }
    Ok(Some(spec))
}

/// Extract a `Path::file_stem` as a String, erroring if it's
/// missing or non-UTF-8 (neither should happen on real disks;
/// guards against the `*.json.bak` / `~` editor-swap case).
fn stem_of(path: &Path) -> Result<String, ConfigError> {
    path.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| ConfigError::BadStem {
            path: path.to_path_buf(),
            reason: "file has no stem or non-UTF-8 stem",
        })
}

/// Build a `ModelParams` from an already-parsed JSON value (possibly
/// the result of deep-merging a dense base with a quantization
/// preset + per-size overrides). `source_stem` is the variant name
/// that ends up on the generated Weights enum variant + inventory
/// registration. `source_path` points at the dense base for
/// synthesized variants; `extra_tracked_paths` carries the preset +
/// override files so the `#[forward]` macro can `include_str!` them
/// for cargo change detection.
/// Normalize an HF `config.json` for macro consumption. VL-wrapper
/// checkpoints (Qwen3.5 / Qwen3.5-MoE / …) nest the text decoder's
/// fields under `text_config`, and newer transformers nest rotary
/// params under `rope_parameters` — while the per-field readers
/// (`extract_bounds`, `extract_scalars`, `extract_mrope_section`, …)
/// all consume the flat single-decoder view. Hoist both levels.
///
/// An explicit top-level field always wins (hoisting never
/// overwrites), and `vision_config` deliberately stays nested — the
/// vision glue owns that subtree. The `configs/` files themselves are
/// VERBATIM copies of the HF checkpoint configs (fetched by
/// `probe-weights`); this normalization is the macro's job, never an
/// edit to the json.
fn normalize_hf_config(
    json: &serde_json::Value,
    aliases: &[(String, String)],
) -> serde_json::Value {
    let mut out = json.clone();
    let Some(obj) = out.as_object_mut() else {
        return out;
    };
    if let Some(text) = obj.get("text_config").cloned()
        && let Some(text_obj) = text.as_object()
    {
        for (k, v) in text_obj {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    if let Some(rope) = obj.get("rope_parameters").cloned()
        && let Some(rope_obj) = rope.as_object()
    {
        for (k, v) in rope_obj {
            obj.entry(k.clone()).or_insert(v.clone());
        }
    }
    // Arch-declared key aliases (`CONFIG_ALIASES` on the carrier mod):
    // when the standard key `.0` is absent, read it from the arch's
    // alternate spelling `.1`. The alt spelling may be a DOTTED PATH
    // (`rope_parameters.full_attention.rope_theta`) — newer
    // transformers nests rope scalars per attention class — resolved
    // via `json_path`. Alias, don't rename: the config stays verbatim
    // and an explicit standard-name field still wins. Resolve every
    // alt against a snapshot taken AFTER the hoists above so paths may
    // reference hoisted subtrees.
    if !aliases.is_empty() {
        let snapshot = serde_json::Value::Object(obj.clone());
        for (std_key, alt_key) in aliases {
            if obj.contains_key(std_key) {
                continue;
            }
            if let Some(v) = crate::arch_spec::json_path(&snapshot, alt_key) {
                obj.insert(std_key.clone(), v.clone());
            }
        }
    }
    out
}

fn model_params_from_json_mode(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
    mode: ConfigMode,
    spec: &crate::arch_spec::DeclaredArchSpec,
) -> Result<ModelParams, ConfigError> {
    match mode {
        ConfigMode::Decoder => {
            model_params_from_json(json, source_stem, source_path, extra_tracked_paths, spec)
        }
        ConfigMode::Vision => {
            vision_params_from_json(json, source_stem, source_path, extra_tracked_paths, spec)
        }
    }
}

/// Shared field parsers for the decoder / vision `ModelParams`
/// heads. Both construction paths call these on their respective
/// json view (decoder: normalized/hoisted; vision: raw) so a field's
/// extraction logic lives once.
fn parse_name(source_stem: &str, source_path: &Path) -> Result<String, ConfigError> {
    stem_to_ident(source_stem).map_err(|reason| ConfigError::BadStem {
        path: source_path.to_path_buf(),
        reason,
    })
}

fn parse_architectures(json: &serde_json::Value) -> Vec<String> {
    json.get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_quantization(
    json: &serde_json::Value,
    source_path: &Path,
) -> Result<Option<crate::quantization::QuantizationConfig>, ConfigError> {
    crate::quantization::QuantizationConfig::parse(json).map_err(|e| ConfigError::Quantization {
        path: source_path.to_path_buf(),
        source: e,
    })
}

/// `None` (field absent) lets `apply_arch_semantic_defaults` supply
/// the family's modeling-code default (HF's own default is TRUE;
/// gemma2/gemma3 checkpoints rely on it); explicit json wins.
fn parse_tie_word_embeddings(json: &serde_json::Value) -> Option<bool> {
    json.get("tie_word_embeddings").and_then(|v| v.as_bool())
}

/// Top-level `torch_dtype` → `text_config.torch_dtype` →
/// `text_config.dtype` (newer transformers serialization: Qwen3.5
/// wrappers carry the compute dtype only as `text_config.dtype`,
/// which the hoist surfaces as `dtype`, never `torch_dtype`).
fn parse_torch_dtype(json: &serde_json::Value) -> Option<String> {
    json.get("torch_dtype")
        .or_else(|| {
            json.get("text_config")
                .and_then(|t| t.get("torch_dtype").or_else(|| t.get("dtype")))
        })
        .and_then(|v| v.as_str())
        .map(|s| s.to_ascii_lowercase())
}

/// Resolve the final per-variant arch spec: the arch crate's DECLARED
/// values as the base, with per-checkpoint JSON drift (explicit config
/// fields / `.overrides.json`) winning field-by-field. Consumers read
/// one resolved spec; neither this crate nor codegen ever switches on
/// an arch name.
fn resolve_arch_spec(
    declared: &crate::arch_spec::DeclaredArchSpec,
    json: &serde_json::Value,
    source_stem: &str,
) -> crate::arch_spec::DeclaredArchSpec {
    let mut a = declared.clone();
    if let Some(l) = extract_vision_layout(json) {
        a.safetensors = Some(l);
    }
    if let Some(f) = extract_vision_d_model_fingerprint(json) {
        a.fingerprint = Some(f);
    }
    if let Some(p) = extract_vision_patch_embed_flatten(json) {
        a.patch_embed_flatten = Some(p);
    }
    if let Some(k) = json.get("vision_pos_embed_key").and_then(|v| v.as_str()) {
        a.pos_embed_key = Some(k.to_string());
    }
    if let Some(s) = json.get("vision_rope_style").and_then(|v| v.as_str()) {
        assert!(
            matches!(s, "neox_hw" | "interleaved_xy"),
            "config `{source_stem}`: unknown vision_rope_style `{s}`",
        );
        a.rope_style = Some(s.to_string());
    }
    if let Some(s) = json.get("vision_pos_emb_interp").and_then(|v| v.as_str()) {
        assert!(
            matches!(s, "bilinear" | "bicubic"),
            "config `{source_stem}`: unknown vision_pos_emb_interp `{s}`",
        );
        a.pos_emb_interp = Some(s.to_string());
    }
    if let Some(p) = json
        .get("decoder_safetensors_prefix")
        .and_then(|v| v.as_str())
    {
        a.decoder_prefix = Some(p.to_string());
    }
    if let Some(s) = json.get("scale_dtype").and_then(|v| v.as_str()) {
        assert!(
            matches!(s, "f16" | "bf16"),
            "config `{source_stem}`: unknown scale_dtype `{s}` (f16|bf16)",
        );
        a.scale_dtype = Some(s.to_string());
    }
    if let Some(m) = json.get("weight_leaf_renames").and_then(|v| v.as_object()) {
        let mut v: Vec<(String, String)> = m
            .iter()
            .map(|(k, val)| {
                (
                    k.clone(),
                    val.as_str()
                        .expect("weight_leaf_renames values must be strings")
                        .to_string(),
                )
            })
            .collect();
        v.sort();
        a.weight_leaf_renames = v;
    }
    a
}

fn model_params_from_json(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
    spec: &crate::arch_spec::DeclaredArchSpec,
) -> Result<ModelParams, ConfigError> {
    let json = &normalize_hf_config(json, &spec.config_aliases);
    let name = parse_name(source_stem, source_path)?;
    let mut bounds = extract_bounds(json);
    derive_sliding_window_pattern(json, &mut bounds);
    derive_implicit_bounds(&mut bounds);
    // Arch-declared bound defaults (zero-centered norms, sliding
    // cadence, router renorm, …): explicit config values always win.
    for (k, v) in &spec.bound_defaults {
        bounds.entry(k.clone()).or_insert(*v);
    }
    // Arch-declared `Params` schema (additive over the standard flat
    // harvest for decoder crates that need non-standard fields).
    spec.eval_params(json, &mut bounds)
        .map_err(|reason| ConfigError::VisionDerivation {
            path: source_path.to_path_buf(),
            reason,
        })?;
    let mut scalars = extract_scalars(json);
    // Arch-declared scalar defaults (`SCALAR_DEFAULTS`): arch-constant
    // floats the checkpoint omits (Gemma-4's `attention_multiplier`).
    // Explicit config values always win.
    for (k, v) in &spec.scalar_defaults {
        scalars.entry(k.clone()).or_insert(*v);
    }
    let quantization = parse_quantization(json, source_path)?;
    let tie_word_embeddings = parse_tie_word_embeddings(json)
        .or(spec.tie_default)
        .unwrap_or(false);
    let architectures = parse_architectures(json);
    let rope_scaling = extract_rope_scaling(json);
    let rope_scaling_hash = json.get("rope_scaling").map(hash_json_value);
    let mrope_section = extract_mrope_section(json);
    let arch = resolve_arch_spec(spec, json, source_stem);
    let torch_dtype = parse_torch_dtype(json);

    apply_generic_bound_defaults(&mut bounds);

    Ok(ModelParams {
        name,
        source_stem: source_stem.to_string(),
        source_path: source_path.to_path_buf(),
        bounds,
        scalars,
        quantization,
        tie_word_embeddings,
        architectures,
        extra_tracked_paths,
        rope_scaling,
        rope_scaling_hash,
        mrope_section,
        arch,
        torch_dtype,
    })
}

/// Build a `#[vision_forward]` variant's `ModelParams` from a
/// VERBATIM HF VL-wrapper config.json.
///
/// Everything the vision pipeline needs is DERIVED here from the
/// nested `vision_config` block via the ARCH-DECLARED key spellings
/// (`config_keys = (…)` on the `#[vision_forward]` attribute — see
/// [`crate::arch_spec`]) plus generic geometry formulas, so the
/// configs/ files stay byte-verbatim checkpoint copies and this
/// crate never names an arch. Flat top-level `vision_*` / `d_model`
/// keys always win when present — that is the `.overrides.json`
/// drift surface.
///
/// Deliberately NOT harvested (unlike the decoder path):
/// - top-level / `text_config` text-decoder fields — they would leak
///   into the vision expansion's `W::` consts (see [`ConfigMode`]);
/// - `rope_scaling` / `mrope_section` — text-side rotary identity;
///   the tower's 2D rope is the DSL's `vision_rope` op and the mm
///   registration carries no rope fingerprint.
fn vision_params_from_json(
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
    extra_tracked_paths: Vec<PathBuf>,
    spec: &crate::arch_spec::DeclaredArchSpec,
) -> Result<ModelParams, ConfigError> {
    let name = parse_name(source_stem, source_path)?;
    let architectures = parse_architectures(json);

    // Flat harvest restricted to the vision namespace: `d_model` +
    // `vision_*` integers, minus the wrapper's `vision_*_token_id`
    // text-tokenizer ids (those are decoder-side splice identity,
    // not tower geometry). These are the `.overrides.json` drift
    // surface — present keys win over the schema below.
    let mut bounds: BTreeMap<String, u64> = json
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter(|(k, _)| {
                    (*k == "d_model" || k.starts_with("vision_")) && !k.ends_with("_token_id")
                })
                .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default();
    // THE arch's `Params` schema: every vision bound — geometry,
    // derived widths, the lot — is declared (typed, named) in the
    // crate's carrier mod and evaluated here against the verbatim
    // config. No family table, no generic derivation chain.
    spec.eval_params(json, &mut bounds)
        .map_err(|reason| ConfigError::VisionDerivation {
            path: source_path.to_path_buf(),
            reason,
        })?;

    // Mirror of the decoder path's int/float double-counting: every
    // derived integer bound is also visible as a scalar, plus the
    // one real float the vision glue reads (`vision_norm_eps`).
    let mut scalars: BTreeMap<String, f64> =
        bounds.iter().map(|(k, v)| (k.clone(), *v as f64)).collect();
    // Block-norm eps: explicit flat key (drift surface) → arch-declared
    // value (modeling-code hardcodes differ per arch) → 1e-6.
    let norm_eps = json
        .get("vision_norm_eps")
        .and_then(|v| v.as_f64())
        .or(spec.vision_norm_eps)
        .unwrap_or(1e-6);
    scalars.insert("vision_norm_eps".to_string(), norm_eps);

    let tie_word_embeddings = parse_tie_word_embeddings(json)
        .or(spec.tie_default)
        .unwrap_or(false);
    let quantization = parse_quantization(json, source_path)?;
    let arch = resolve_arch_spec(spec, json, source_stem);

    // A vision arch MUST declare its on-disk structure — there is no
    // arch-keyed fallback table in this crate. Missing ⇒ the crate's
    // `#[vision_forward]` attribute needs the declaration.
    for (present, what, example) in [
        (
            arch.safetensors.is_some(),
            "safetensors",
            "safetensors = (root = \"…\", blocks = \"…\")",
        ),
        (
            arch.fingerprint.is_some(),
            "fingerprint",
            "fingerprint = (key = \"…\", dim = 0)",
        ),
        (
            arch.patch_embed_flatten.is_some(),
            "patch_embed_flatten",
            "patch_embed_flatten = (key = \"…\")",
        ),
    ] {
        if !present {
            return Err(ConfigError::VisionDerivation {
                path: source_path.to_path_buf(),
                reason: format!(
                    "vision arch is missing its `{what}` declaration — add \
                     `{example}` to the #[vision_forward] attribute args",
                ),
            });
        }
    }

    let torch_dtype = parse_torch_dtype(json);
    apply_generic_bound_defaults(&mut bounds);

    Ok(ModelParams {
        name,
        source_stem: source_stem.to_string(),
        source_path: source_path.to_path_buf(),
        bounds,
        scalars,
        quantization,
        tie_word_embeddings,
        architectures,
        extra_tracked_paths,
        rope_scaling: None,
        rope_scaling_hash: None,
        mrope_section: None,
        arch,
        torch_dtype,
    })
}

/// Parse `vision_safetensors_layout`, if present. Missing or
/// malformed → `None`; codegen falls back to
/// [`VisionSafetensorsLayout::qwen_default`] in that case.
fn extract_vision_layout(json: &serde_json::Value) -> Option<VisionSafetensorsLayout> {
    let obj = json.get("vision_safetensors_layout")?.as_object()?;
    let default_root = obj.get("default_root")?.as_str()?.to_string();
    let layered_subpath = obj.get("layered_subpath")?.as_str()?.to_string();
    let mut subtrees = BTreeMap::new();
    if let Some(s) = obj.get("subtrees").and_then(|v| v.as_object()) {
        for (k, v) in s {
            if let Some(disk) = v.as_str() {
                subtrees.insert(k.clone(), disk.to_string());
            }
        }
    }
    Some(VisionSafetensorsLayout {
        default_root,
        layered_subpath,
        subtrees,
    })
}

fn extract_vision_d_model_fingerprint(json: &serde_json::Value) -> Option<VisionDModelFingerprint> {
    let obj = json.get("vision_d_model_fingerprint")?.as_object()?;
    let key = obj.get("key")?.as_str()?.to_string();
    let dim = obj.get("dim")?.as_u64()? as usize;
    Some(VisionDModelFingerprint { key, dim })
}

fn extract_vision_patch_embed_flatten(json: &serde_json::Value) -> Option<VisionPatchEmbedFlatten> {
    let obj = json.get("vision_patch_embed_flatten")?.as_object()?;
    let key = obj.get("key")?.as_str()?.to_string();
    let leading_dim = obj.get("leading_dim")?.as_u64()? as usize;
    let channels_last = obj
        .get("channels_last")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    Some(VisionPatchEmbedFlatten {
        key,
        leading_dim,
        channels_last,
    })
}

/// Extract `mrope_section` as a fixed `[u32; 3]`, from a TOP-LEVEL
/// `mrope_section` first, else `rope_scaling.mrope_section`. Returns
/// `None` when absent (every text-only arch) or malformed (<3 entries).
/// Qwen2-VL/2.5-VL ship it under `rope_scaling` in the HF config (so the
/// checkpoint carries it and the rope_scaling fingerprint still matches).
/// Qwen3.5-VL DOESN'T carry it in the checkpoint (it's a model-code
/// default `[11,11,10]`) — so its scratchy config sets it TOP-LEVEL,
/// keeping `rope_scaling` absent (== the checkpoint) so the
/// `HfFingerprint` rope_scaling_hash disambiguation still matches.
fn extract_mrope_section(json: &serde_json::Value) -> Option<[u32; 3]> {
    let arr = json
        .get("mrope_section")
        .or_else(|| {
            json.get("rope_scaling")
                .and_then(|rs| rs.get("mrope_section"))
        })?
        .as_array()?;
    if arr.len() < 3 {
        return None;
    }
    let a = arr[0].as_u64()? as u32;
    let b = arr[1].as_u64()? as u32;
    let c = arr[2].as_u64()? as u32;
    Some([a, b, c])
}

/// Macro-side mirror of `scratchy_forward_compiler::hash_json_value` — the
/// macro can't depend on `scratchy-forward-compiler` (cycle), so the same
/// function body is duplicated here. Changes MUST stay in lockstep:
/// the manifest hash baked at compile time only matches the
/// runtime hash if both hashers agree bit-for-bit.
fn hash_json_value(v: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    fn recurse<H: Hasher>(v: &serde_json::Value, h: &mut H) {
        match v {
            serde_json::Value::Null => 0u8.hash(h),
            serde_json::Value::Bool(b) => {
                1u8.hash(h);
                b.hash(h);
            }
            serde_json::Value::Number(n) => {
                2u8.hash(h);
                let f = n.as_f64().unwrap_or(0.0);
                f.to_bits().hash(h);
            }
            serde_json::Value::String(s) => {
                3u8.hash(h);
                s.hash(h);
            }
            serde_json::Value::Array(arr) => {
                4u8.hash(h);
                arr.len().hash(h);
                for v in arr {
                    recurse(v, h);
                }
            }
            serde_json::Value::Object(obj) => {
                5u8.hash(h);
                let mut keys: Vec<&String> = obj.keys().collect();
                keys.sort();
                keys.len().hash(h);
                for k in keys {
                    k.hash(h);
                    recurse(&obj[k], h);
                }
            }
        }
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    recurse(v, &mut h);
    h.finish()
}

/// Parse the `rope_scaling` subobject into a [`RopeScaling`]. Returns
/// `None` when the key is absent or the `type`/`rope_type` is neither
/// `llama3`, `longrope`, nor `su`. Integer-ish fields are extracted
/// via `as_u64` / `as_f64` so HF configs that write `4096` or `4096.0`
/// both parse. `attention_factor` falls back to the Phi-3 paper
/// formula when the config omits it.
fn extract_rope_scaling(json: &serde_json::Value) -> Option<RopeScaling> {
    let rs = json.get("rope_scaling")?;
    let rope_type = rs
        .get("rope_type")
        .or_else(|| rs.get("type"))
        .and_then(|v| v.as_str())?;
    match rope_type {
        "llama3" => {
            let factor = rs.get("factor").and_then(|v| v.as_f64())?;
            let low_freq_factor = rs
                .get("low_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            let high_freq_factor = rs
                .get("high_freq_factor")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0);
            let original_max_position_embeddings = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192);
            Some(RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            })
        }
        "longrope" | "su" => {
            let parse_factors = |key: &str| -> Option<Vec<f64>> {
                rs.get(key)?
                    .as_array()?
                    .iter()
                    .map(|v| v.as_f64())
                    .collect()
            };
            let short_factor = parse_factors("short_factor")?;
            let long_factor = parse_factors("long_factor")?;
            let original_max_position_embeddings = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .or_else(|| {
                    json.get("original_max_position_embeddings")
                        .and_then(|v| v.as_u64())
                })
                .unwrap_or(4096);
            let max_pos = json
                .get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096);
            // Python vLLM's `Phi3LongRoPEScaledRotaryEmbedding.__init__`:
            //   scale = max_pos / orig_max
            //   scaling_factor = 1.0 if scale <= 1.0 else sqrt(1 + log(scale)/log(orig_max))
            //   short_mscale = short_mscale or scaling_factor
            //   long_mscale  = long_mscale  or scaling_factor
            let scaling_factor = {
                let scale = max_pos as f64 / original_max_position_embeddings as f64;
                if scale <= 1.0 {
                    1.0
                } else {
                    (1.0 + scale.ln() / (original_max_position_embeddings as f64).ln()).sqrt()
                }
            };
            let short_mscale = rs
                .get("short_mscale")
                .and_then(|v| v.as_f64())
                .unwrap_or(scaling_factor);
            let long_mscale = rs
                .get("long_mscale")
                .and_then(|v| v.as_f64())
                .unwrap_or(scaling_factor);
            Some(RopeScaling::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
                short_mscale,
                long_mscale,
            })
        }
        "yarn" => {
            let factor = rs.get("factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let beta_fast = rs.get("beta_fast").and_then(|v| v.as_f64()).unwrap_or(32.0);
            let beta_slow = rs.get("beta_slow").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mscale = rs.get("mscale").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mscale_all_dim = rs
                .get("mscale_all_dim")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let original_max_position_embeddings = rs
                .get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096);
            Some(RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
                original_max_position_embeddings,
            })
        }
        _ => None,
    }
}

/// Recursive deep-merge of JSON objects. When both sides agree on
/// a key whose value is an object, merge keys recursively; else
/// the overlay wins at that leaf. Used to stack a dense base with
/// a quantization preset (+ optional per-size overrides) into one
/// final variant JSON.
fn deep_merge(base: &mut serde_json::Value, overlay: &serde_json::Value) {
    use serde_json::Value;
    match (base, overlay) {
        (Value::Object(base_map), Value::Object(overlay_map)) => {
            for (k, v) in overlay_map {
                let entry = base_map.entry(k.clone()).or_insert(Value::Null);
                deep_merge(entry, v);
            }
        }
        (slot, overlay_val) => {
            *slot = overlay_val.clone();
        }
    }
}

/// Every top-level integer field becomes a bound. Anything else
/// (strings, bools, floats, nested objects) is ignored — the DSL
/// only quantifies over integers.
fn extract_bounds(json: &serde_json::Value) -> BTreeMap<String, u64> {
    json.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| {
                    // Integers → as-is. Booleans → 0/1 (HF stores
                    // model config booleans like `norm_topk_prob`,
                    // `tie_word_embeddings`, `use_qk_norm` here; the
                    // bounds map is the only u64 table downstream
                    // consumers read).
                    v.as_u64()
                        .map(|n| (k.clone(), n))
                        .or_else(|| v.as_bool().map(|b| (k.clone(), b as u64)))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Every top-level number field (int OR float) becomes a scalar.
/// Integer fields also go to `bounds` via [`extract_bounds`] — they
/// double-count into both tables because some HF configs write
/// scale-type fields as integer literals (e.g. Gemma3's
/// `query_pre_attn_scalar: 256`, `rope_theta: 1000000`), while
/// other configs write them as floats (Gemma2 uses `256.0`,
/// `10000.0`). Readers of physics-scale values (softmax scales,
/// rope thetas, norm epsilons) look in `scalars`; readers of
/// shape-determining values look in `bounds`. Both views must
/// agree on integer-valued scale fields.
fn extract_scalars(json: &serde_json::Value) -> BTreeMap<String, f64> {
    json.as_object()
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                .collect()
        })
        .unwrap_or_default()
}

/// Apply HF's implicit config.json defaults to the bounds map.
///
/// HF configs are allowed to omit certain fields that have well-
/// defined defaults. Without these, shape inference on an older
/// config (like `llama-2-13b/config.json`, which omits `head_dim`)
/// would leave `num_heads * head_dim` unclosed. The defaults are
/// universal across every HF transformer — they live here, in the
/// config loader, rather than getting baked into per-arch shape
/// anchoring.
///
/// Defaults applied:
/// - **`head_dim`** ← `hidden_size / num_attention_heads`. Llama-2
///   and Llama-3 (pre-3.2) omit the field; Llama-3.2+, Qwen2.5+,
///   Gemma2 list it explicitly. Both forms are valid HF JSON.
/// - **`num_key_value_heads`** ← `num_attention_heads`. Configs
///   predating grouped-query attention assume MHA and don't list
///   the field.
///
/// Explicit values in the JSON always win — we only fill absent
/// keys.
/// Derive `sliding_window_pattern` from the verbatim `layer_types`
/// array when the scalar field is absent. Newer transformers configs
/// (Gemma-3/4) ship the per-layer attention schedule as an explicit
/// `layer_types: ["sliding_attention", …, "full_attention", …]` list
/// rather than a period scalar — mlx-lm reads this list directly and
/// only falls back to its `sliding_window_pattern` dataclass default
/// when it's absent (so the default can disagree with the checkpoint,
/// as it does for Gemma-4 12B/31B: default 5 vs actual period 6).
///
/// The schedule is regular (every Nth layer is global), so the period
/// is the index of the first layer whose type differs from layer 0,
/// plus one. Explicit `sliding_window_pattern` in the config wins.
fn derive_sliding_window_pattern(json: &serde_json::Value, bounds: &mut BTreeMap<String, u64>) {
    if bounds.contains_key("sliding_window_pattern") {
        return;
    }
    let Some(types) = json.get("layer_types").and_then(|v| v.as_array()) else {
        return;
    };
    let Some(first) = types.first().and_then(|v| v.as_str()) else {
        return;
    };
    if let Some(idx) = types.iter().position(|t| t.as_str() != Some(first)) {
        bounds.insert("sliding_window_pattern".to_string(), (idx + 1) as u64);
    }
}

fn derive_implicit_bounds(bounds: &mut BTreeMap<String, u64>) {
    // MLA (DeepSeek family) head_dim — MUST run before the generic
    // hidden/heads rule: MLA's per-head Q/K width is
    // `qk_nope_head_dim + qk_rope_head_dim` (V2-Lite: 128+64=192),
    // NOT hidden/heads (2048/16=128). The verbatim HF configs carry
    // the two summands but no `head_dim`.
    if !bounds.contains_key("head_dim")
        && let (Some(&nope), Some(&rope)) = (
            bounds.get("qk_nope_head_dim"),
            bounds.get("qk_rope_head_dim"),
        )
    {
        bounds.insert("head_dim".to_string(), nope + rope);
    }
    if !bounds.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (bounds.get("hidden_size"), bounds.get("num_attention_heads"))
        && heads != 0
        && hidden.is_multiple_of(heads)
    {
        bounds.insert("head_dim".to_string(), hidden / heads);
    }
    // MLA projection out-dims referenced by the deepseek crates'
    // weights.json shape exprs — pure arithmetic over the verbatim
    // checkpoint fields. `q_proj_out` is q_proj's (or q_b_proj's,
    // under Q-LoRA) output width; `kv_a_proj_out` is the compressed
    // KV + rope-K width; `kv_lora_out` is kv_b_proj's decompressed
    // output; `attn_out` is o_proj's input (heads · v_head_dim,
    // consumed by the non-flat deepseek-v3 manifest). Explicit
    // values (synthetic test configs) always win.
    if let (Some(&heads), Some(&nope), Some(&rope), Some(&vhd)) = (
        bounds.get("num_attention_heads"),
        bounds.get("qk_nope_head_dim"),
        bounds.get("qk_rope_head_dim"),
        bounds.get("v_head_dim"),
    ) {
        bounds
            .entry("q_proj_out".to_string())
            .or_insert(heads * (nope + rope));
        bounds
            .entry("kv_lora_out".to_string())
            .or_insert(heads * (nope + vhd));
        bounds.entry("attn_out".to_string()).or_insert(heads * vhd);
        if let Some(&kv_lora_rank) = bounds.get("kv_lora_rank") {
            bounds
                .entry("kv_a_proj_out".to_string())
                .or_insert(kv_lora_rank + rope);
        }
    }
    if !bounds.contains_key("num_key_value_heads")
        && let Some(&heads) = bounds.get("num_attention_heads")
    {
        bounds.insert("num_key_value_heads".to_string(), heads);
    }
    if !bounds.contains_key("sliding_window_global_remainder")
        && let Some(&p) = bounds.get("sliding_window_pattern")
        && p > 0
    {
        bounds.insert("sliding_window_global_remainder".to_string(), p - 1);
    }
    // All-MoE decoders (Qwen3.5-MoE) have no dense MLP, so their HF
    // configs omit `intermediate_size` — but the bound is a universal
    // DISPATCH_FIELDS constant and the MLP-fusion seams
    // (SynthGateUpSiluMul, FusedGateUpSiluMul) size their SwiGLU from
    // it. For such configs the model's only non-routed SwiGLU is the
    // shared expert, so derive its width (0 when there is no shared
    // expert either). Dense / hybrid models ship `intermediate_size`
    // explicitly — explicit values always win.
    if !bounds.contains_key("intermediate_size") && bounds.contains_key("moe_intermediate_size") {
        let shared = bounds
            .get("shared_expert_intermediate_size")
            .copied()
            .unwrap_or(0);
        bounds.insert("intermediate_size".to_string(), shared);
    }
    // Geometry dims referenced by weights.json shape exprs — pure
    // arithmetic over checkpoint fields, derived here so the configs/
    // files stay verbatim HF copies. Explicit values always win.
    if !bounds.contains_key("attn_q_dim")
        && let (Some(&heads), Some(&hd)) =
            (bounds.get("num_attention_heads"), bounds.get("head_dim"))
    {
        bounds.insert("attn_q_dim".to_string(), heads * hd);
    }
    if !bounds.contains_key("q_gate_dim")
        && let Some(&q) = bounds.get("attn_q_dim")
    {
        // `attn_output_gate` (Qwen3.5 family) doubles q_proj's output:
        // per head `[query | gate]`.
        let gate = bounds.get("attn_output_gate").copied().unwrap_or(0) != 0;
        bounds.insert("q_gate_dim".to_string(), if gate { 2 * q } else { q });
    }
    if !bounds.contains_key("kv_dim")
        && let (Some(&kvh), Some(&hd)) = (bounds.get("num_key_value_heads"), bounds.get("head_dim"))
    {
        bounds.insert("kv_dim".to_string(), kvh * hd);
    }
    if !bounds.contains_key("gdn_value_dim")
        && let (Some(&vh), Some(&vd)) = (
            bounds.get("linear_num_value_heads"),
            bounds.get("linear_value_head_dim"),
        )
    {
        bounds.insert("gdn_value_dim".to_string(), vh * vd);
    }
    // Gated-DeltaNet conv channel count: q and k at key width plus v
    // at value width (`in_proj_qkv`'s output / `conv1d`'s channels).
    if !bounds.contains_key("gdn_conv_dim")
        && let (Some(&kh), Some(&kd), Some(&vdim)) = (
            bounds.get("linear_num_key_heads"),
            bounds.get("linear_key_head_dim"),
            bounds.get("gdn_value_dim"),
        )
    {
        bounds.insert("gdn_conv_dim".to_string(), 2 * kh * kd + vdim);
    }
}

/// Arch-AGNOSTIC bound defaults applied to every parsed config
/// (decoder and vision paths alike). Per-arch modeling defaults
/// (zero-centered norms, sliding cadence, tie defaults, …) are NOT
/// here — they are `bound_defaults` / `tie_default` declarations on
/// the owning crate's `#[forward]` attribute (see [`crate::arch_spec`]).
fn apply_generic_bound_defaults(bounds: &mut BTreeMap<String, u64>) {
    // Per-class attention geometry defaults: uniform-geometry arches
    // never declare global_* keys, but the shape sigs anchor the
    // `attention()` (global-class) tile to `global_head_dim` /
    // `num_global_key_value_heads` unconditionally — default them to
    // the base values so every existing arch resolves identically.
    if !bounds.contains_key("global_head_dim")
        && let Some(&hd) = bounds.get("head_dim")
    {
        bounds.insert("global_head_dim".to_string(), hd);
    }
    if !bounds.contains_key("num_global_key_value_heads")
        && let Some(&kv) = bounds.get("num_key_value_heads")
    {
        bounds.insert("num_global_key_value_heads".to_string(), kv);
    }
    // Global-attention-class projection widths (Gemma-4's dual-class
    // attention: the global layers run a wider head with their own KV
    // head count). `q_global_dim` is q_proj_global's output width;
    // `k_global_dim` the single shared global K/V projection's width.
    // Derived AFTER the `global_head_dim` / `num_global_key_value_heads`
    // defaults above so uniform-geometry arches (where those fall back
    // to `head_dim` / `num_key_value_heads`) resolve these identically
    // to `attn_q_dim` / `kv_dim`. Explicit config values always win.
    if !bounds.contains_key("q_global_dim")
        && let (Some(&heads), Some(&ghd)) = (
            bounds.get("num_attention_heads"),
            bounds.get("global_head_dim"),
        )
    {
        bounds.insert("q_global_dim".to_string(), heads * ghd);
    }
    if !bounds.contains_key("k_global_dim")
        && let (Some(&gkvh), Some(&ghd)) = (
            bounds.get("num_global_key_value_heads"),
            bounds.get("global_head_dim"),
        )
    {
        bounds.insert("k_global_dim".to_string(), gkvh * ghd);
    }
}

/// Normalize a file stem into a valid Rust identifier:
///   - replace runs of non-alphanumeric chars with `_`
///   - prepend `m_` if the result starts with a digit.
fn stem_to_ident(stem: &str) -> Result<String, &'static str> {
    if stem.is_empty() {
        return Err("empty stem");
    }
    let mut out = String::with_capacity(stem.len() + 2);
    let mut prev_underscore = false;
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    // Trim leading/trailing underscores and collapse.
    let trimmed = out.trim_matches('_').to_string();
    let final_ = if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("m_{trimmed}")
    } else {
        trimmed
    };
    if final_.is_empty() {
        return Err("stem normalizes to empty identifier");
    }
    Ok(final_)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute path to `crates/models/arch/<arch>/configs/` from
    /// this crate's manifest dir, for tests that load real configs.
    fn repo_model_archs(arch: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            // `configs/<arch>`, not `<arch>/configs` — the layout the
            // collapse into `scratchy-models` produced.
            .join("configs")
            .join(arch)
    }

    #[test]
    fn stem_normalization() {
        assert_eq!(
            stem_to_ident("llama-3.2-1b").unwrap().to_string(),
            "llama_3_2_1b"
        );
        assert_eq!(stem_to_ident("qwen2-7b").unwrap().to_string(), "qwen2_7b");
        assert_eq!(
            stem_to_ident("qwen2.5-0.5b").unwrap().to_string(),
            "qwen2_5_0_5b"
        );
        // Leading digit → "m_" prefix.
        assert_eq!(stem_to_ident("3b-model").unwrap().to_string(), "m_3b_model");
        // Double-dashes collapse.
        assert_eq!(stem_to_ident("a--b").unwrap().to_string(), "a_b");
        assert!(stem_to_ident("").is_err());
    }

    #[test]
    fn load_real_llama_configs() {
        let dir = repo_model_archs("llama");
        let configs = load_dir(&dir, &Default::default()).expect("load llama configs");

        // 12 dense bases (9 Llama + 2 smollm2 + 1 tinyllama) ×
        // (1 dense + 8 quant presets from `quantizations.json`:
        //  awq-gemm, bnb-nf4-dq, gptq-sym, gptq-sym-desc_act,
        //  ct-int4-sym, fp8-dynamic-per-tensor, fp8-static-per-tensor,
        //  ggml) = 108. Individual sizes don't always have real HF
        //  repos in every preset, but the compiler emits variants for
        //  all of them so fingerprint dispatch stays open-set at
        //  runtime.
        assert_eq!(
            configs.len(),
            108,
            "expected 108 Llama variants (12 bases × 9 variants)"
        );

        // Ground-truth check on llama-3.2-1b. Published values:
        //   num_hidden_layers = 16
        //   hidden_size       = 2048
        //   intermediate_size = 8192
        //   num_attention_heads = 32
        //   num_key_value_heads = 8
        //   head_dim            = 64
        //   vocab_size          = 128256
        let cfg = configs
            .iter()
            .find(|c| c.source_stem == "llama-3.2-1b")
            .expect("llama-3.2-1b present");
        assert_eq!(cfg.name.to_string(), "llama_3_2_1b");
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&16));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&2048));
        assert_eq!(cfg.bounds.get("intermediate_size"), Some(&8192));
        assert_eq!(cfg.bounds.get("num_attention_heads"), Some(&32));
        assert_eq!(cfg.bounds.get("num_key_value_heads"), Some(&8));
        assert_eq!(cfg.bounds.get("head_dim"), Some(&64));
        assert_eq!(cfg.bounds.get("vocab_size"), Some(&128256));
        // source_path preserved for later rebuild tracking and
        // diagnostics.
        assert!(cfg.source_path.ends_with("llama-3.2-1b.json"));
    }

    #[test]
    fn load_real_qwen2_configs() {
        let dir = repo_model_archs("qwen2");
        let configs = load_dir(&dir, &Default::default()).expect("load qwen2 configs");
        // 13 dense (11 text + qwen2-vl-2b + qwen2.5-vl-3b text
        // decoders) × (1 dense + 7 presets: awq-gemm, bnb-nf4-dq,
        // ct-int4-sym, gptq-sym, fp8-dynamic-per-tensor,
        // fp8-static-per-tensor, ggml) = 104.
        assert_eq!(
            configs.len(),
            104,
            "expected 104 Qwen2 variants (13 bases × 8 variants)"
        );

        // Ground-truth check on Qwen2-0.5B:
        //   num_hidden_layers    = 24
        //   hidden_size          = 896
        //   intermediate_size    = 4864
        //   num_attention_heads  = 14
        //   num_key_value_heads  = 2
        //   vocab_size           = 151936
        let cfg = configs
            .iter()
            .find(|c| c.source_stem == "qwen2-0.5b")
            .expect("qwen2-0.5b present");
        assert_eq!(cfg.name.to_string(), "qwen2_0_5b");
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&24));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&896));
        assert_eq!(cfg.bounds.get("intermediate_size"), Some(&4864));
        assert_eq!(cfg.bounds.get("num_attention_heads"), Some(&14));
        assert_eq!(cfg.bounds.get("num_key_value_heads"), Some(&2));
        assert_eq!(cfg.bounds.get("vocab_size"), Some(&151936));
    }

    #[test]
    fn load_real_qwen2_vl_vision_configs() {
        // Vision configs (G.5.b) carry only `d_model` + `vision_*`
        // bounds — no decoder-only keys (`hidden_size`,
        // `num_attention_heads`, etc.). This test pins three things:
        //
        //   1. `load_dir` parses each variant cleanly without
        //      requiring the decoder fields,
        //   2. `derive_implicit_bounds` does NOT spuriously synthesize
        //      `head_dim` / `num_key_value_heads` (its source keys are
        //      absent → every branch short-circuits),
        //   3. every `vision_*` bound the codegen reads
        //      (`vision_num_heads`, `vision_head_dim`, plus the shape-
        //      anchoring `vision_in_features` / `vision_rope_half_dim`)
        //      lands in `bounds`.
        let dir = repo_model_archs("qwen2-vl");
        let configs =
            load_dir_vision(&dir, &Default::default()).expect("load qwen2-vl vision configs");
        assert_eq!(
            configs.len(),
            3,
            "expected 3 Qwen2-VL vision variants (2b/7b/72b)"
        );

        // Per-variant `d_model` (= text-decoder hidden), vision tower
        // is shape-identical otherwise.
        let d_model_for = |stem: &str| -> u64 {
            *configs
                .iter()
                .find(|c| c.source_stem == stem)
                .unwrap_or_else(|| panic!("{stem} present"))
                .bounds
                .get("d_model")
                .expect("d_model")
        };
        assert_eq!(d_model_for("qwen2-vl-2b-instruct"), 1536);
        assert_eq!(d_model_for("qwen2-vl-7b-instruct"), 3584);
        assert_eq!(d_model_for("qwen2-vl-72b-instruct"), 8192);

        let cfg = configs
            .iter()
            .find(|c| c.source_stem == "qwen2-vl-2b-instruct")
            .expect("qwen2-vl-2b-instruct present");

        // Bounds the codegen reads via `emit_canonical_params_impl`
        // and `extern_shape`.
        assert_eq!(cfg.bounds.get("vision_num_heads"), Some(&16));
        assert_eq!(cfg.bounds.get("vision_head_dim"), Some(&80));
        assert_eq!(cfg.bounds.get("vision_in_features"), Some(&1176));
        assert_eq!(cfg.bounds.get("vision_rope_half_dim"), Some(&40));
        // Bounds the manifest formulas anchor on.
        assert_eq!(cfg.bounds.get("vision_embed_dim"), Some(&1280));
        assert_eq!(cfg.bounds.get("vision_mlp_hidden"), Some(&5120));
        assert_eq!(cfg.bounds.get("vision_merge_hidden"), Some(&5120));
        assert_eq!(cfg.bounds.get("vision_depth"), Some(&32));
        assert_eq!(cfg.bounds.get("vision_spatial_merge_size"), Some(&2));
        assert_eq!(cfg.bounds.get("vision_merge_factor"), Some(&4));
        assert_eq!(cfg.bounds.get("vision_patch_size"), Some(&14));
        assert_eq!(cfg.bounds.get("vision_temporal_patch_size"), Some(&2));
        assert_eq!(cfg.bounds.get("vision_in_chans"), Some(&3));

        // Decoder-only keys are absent — `derive_implicit_bounds`
        // must not have synthesized them.
        assert!(!cfg.bounds.contains_key("hidden_size"));
        assert!(!cfg.bounds.contains_key("num_attention_heads"));
        assert!(!cfg.bounds.contains_key("num_key_value_heads"));
        assert!(!cfg.bounds.contains_key("head_dim"));
        assert!(!cfg.bounds.contains_key("num_hidden_layers"));
        assert!(!cfg.bounds.contains_key("vocab_size"));

        // Float scalars land in `scalars`. Vision norm eps for the
        // pre/post-attn LayerNorms; auto-extracted by
        // `extract_scalars` from any f64 field.
        assert_eq!(cfg.scalars.get("vision_norm_eps"), Some(&1e-6));

        // HF arch claim string + tie flag. The 2B checkpoint TIES
        // embed/lm_head (`tie_word_embeddings: true` in the verbatim
        // HF config — the old invented config wrongly said false);
        // inert vision-side (no lm_head in the tower), but the value
        // must mirror the checkpoint.
        assert_eq!(
            cfg.architectures,
            vec!["Qwen2VLForConditionalGeneration".to_string()]
        );
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn qwen2_vl_vision_weights_manifest_anchors_on_vision_bounds() {
        // The vision weights.json declares per-tensor shapes anchored
        // on `vision_*` bounds (and `d_model` on `merger.mlp.2`).
        // Pin a few representative shapes so a future edit that
        // accidentally swaps in decoder-side bounds (e.g.
        // `hidden_size`) trips this guard at unit-test time, before
        // it reaches a #[vision_forward] expansion.
        use crate::weights_manifest::load_or_empty;
        let dir = repo_model_archs("qwen2-vl");
        let manifest = load_or_empty(&dir).expect("load qwen2-vl weights.json");

        // G.5.f flipped `attn.qkv` from a top-level shape entry to a
        // `__packed_splits__` mapping → `[attn.q, attn.k, attn.v]`. The
        // body writes three separate gemms (text-side qwen2 pattern),
        // so the post-split keys are what land in `entries`.
        let splits = manifest
            .packed_splits
            .get("attn.qkv")
            .expect("attn.qkv in __packed_splits__");
        assert_eq!(
            splits,
            &vec!["attn.q".to_string(), "attn.k".into(), "attn.v".into()]
        );
        let q = manifest.lookup(&["attn", "q"]).expect("attn.q in manifest");
        // `[vision_embed_dim, vision_embed_dim]` — K, N order.
        assert_eq!(q.len(), 2);

        let merger_proj = manifest
            .lookup(&["merger", "mlp_2"])
            .expect("merger.mlp_2 in manifest");
        // `[vision_merge_hidden, d_model]` in K, N order.
        assert_eq!(merger_proj.len(), 2);

        let patch_embed = manifest
            .lookup(&["patch_embed", "proj"])
            .expect("patch_embed.proj in manifest");
        // `[vision_in_features, vision_embed_dim]` in K, N order.
        assert_eq!(patch_embed.len(), 2);

        let norm1 = manifest.lookup(&["norm1"]).expect("norm1 in manifest");
        assert_eq!(norm1.len(), 1);
    }

    #[test]
    fn bounds_are_sorted_by_stem_for_determinism() {
        let dir = repo_model_archs("llama");
        let configs = load_dir(&dir, &Default::default()).unwrap();
        let stems: Vec<&str> = configs.iter().map(|c| c.source_stem.as_str()).collect();
        let mut sorted = stems.clone();
        sorted.sort();
        assert_eq!(stems, sorted, "configs should be alphabetically sorted");
    }

    /// The global-attention-class projection widths are DERIVED from
    /// the standard checkpoint fields (`num_attention_heads`,
    /// `num_global_key_value_heads`, `global_head_dim`) so Gemma-4's
    /// `configs/` stay verbatim HF copies — no hand-stamped
    /// `q_global_dim` / `k_global_dim`. Mirrors the 12B geometry:
    /// 16 q heads × 512 global head dim = 8192; 1 global KV head × 512
    /// = 512.
    #[test]
    fn global_attn_dims_derive_from_standard_fields() {
        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("hidden_size".into(), 3840);
        bounds.insert("num_attention_heads".into(), 16);
        bounds.insert("num_key_value_heads".into(), 8);
        bounds.insert("head_dim".into(), 256);
        bounds.insert("global_head_dim".into(), 512);
        bounds.insert("num_global_key_value_heads".into(), 1);

        derive_implicit_bounds(&mut bounds);
        apply_generic_bound_defaults(&mut bounds);

        // Sliding-class dims (already derived pre-change) — sanity.
        assert_eq!(bounds.get("attn_q_dim"), Some(&4096)); // 16 * 256
        assert_eq!(bounds.get("kv_dim"), Some(&2048)); // 8 * 256
        // Global-class dims — the new derivation.
        assert_eq!(bounds.get("q_global_dim"), Some(&8192)); // 16 * 512
        assert_eq!(bounds.get("k_global_dim"), Some(&512)); // 1 * 512
    }

    /// 31B geometry: 32 q heads, 4 global KV heads, 512 global head dim
    /// → q_global_dim 16384, k_global_dim 2048. Confirms the derivation
    /// tracks a different size with >1 global KV head, with no config
    /// edit needed.
    #[test]
    fn global_attn_dims_track_31b_geometry() {
        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("hidden_size".into(), 5376);
        bounds.insert("num_attention_heads".into(), 32);
        bounds.insert("num_key_value_heads".into(), 16);
        bounds.insert("head_dim".into(), 256);
        bounds.insert("global_head_dim".into(), 512);
        bounds.insert("num_global_key_value_heads".into(), 4);

        derive_implicit_bounds(&mut bounds);
        apply_generic_bound_defaults(&mut bounds);

        assert_eq!(bounds.get("q_global_dim"), Some(&16384)); // 32 * 512
        assert_eq!(bounds.get("k_global_dim"), Some(&2048)); // 4 * 512
    }

    /// Uniform-geometry arches never declare the global_* fields; they
    /// fall back to `head_dim` / `num_key_value_heads` so the global
    /// dims resolve identically to the sliding ones (no spurious wider
    /// head). An explicit config value would still win.
    #[test]
    fn global_attn_dims_collapse_to_base_when_uniform() {
        let mut bounds: BTreeMap<String, u64> = BTreeMap::new();
        bounds.insert("hidden_size".into(), 2048);
        bounds.insert("num_attention_heads".into(), 32);
        bounds.insert("num_key_value_heads".into(), 8);
        bounds.insert("head_dim".into(), 64);

        derive_implicit_bounds(&mut bounds);
        apply_generic_bound_defaults(&mut bounds);

        assert_eq!(bounds.get("q_global_dim"), bounds.get("attn_q_dim"));
        assert_eq!(bounds.get("k_global_dim"), bounds.get("kv_dim"));
        assert_eq!(bounds.get("q_global_dim"), Some(&2048)); // 32 * 64
        assert_eq!(bounds.get("k_global_dim"), Some(&512)); // 8 * 64
    }

    /// `sliding_window_pattern` is read off the verbatim `layer_types`
    /// schedule (Gemma-4 12B/31B: full_attention every 6th layer →
    /// period 6), NOT a hardcoded default. An explicit scalar wins.
    #[test]
    fn sliding_window_pattern_derives_from_layer_types() {
        let mk = |types: serde_json::Value| {
            let json = serde_json::json!({ "layer_types": types });
            let mut bounds = BTreeMap::new();
            derive_sliding_window_pattern(&json, &mut bounds);
            bounds.get("sliding_window_pattern").copied()
        };
        let period6 = serde_json::json!([
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention"
        ]);
        assert_eq!(mk(period6), Some(6));

        // Explicit scalar already present → not overwritten.
        let json = serde_json::json!({
            "sliding_window_pattern": 4,
            "layer_types": ["sliding_attention", "sliding_attention", "full_attention"]
        });
        let mut bounds = BTreeMap::new();
        bounds.insert("sliding_window_pattern".to_string(), 4);
        derive_sliding_window_pattern(&json, &mut bounds);
        assert_eq!(bounds.get("sliding_window_pattern"), Some(&4));

        // No layer_types → no-op.
        assert_eq!(mk(serde_json::json!(null)), None);
    }

    /// `CONFIG_ALIASES` resolves DOTTED paths into nested subtrees so
    /// Gemma-4's per-attention-class `rope_parameters` surface as the
    /// flat scalar names the rope codegen reads — config stays verbatim.
    #[test]
    fn config_aliases_resolve_nested_rope_parameters() {
        let json = serde_json::json!({
            "hidden_size": 3840,
            "rope_parameters": {
                "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        });
        let aliases = vec![
            (
                "rope_theta".to_string(),
                "rope_parameters.full_attention.rope_theta".to_string(),
            ),
            (
                "rope_local_base_freq".to_string(),
                "rope_parameters.sliding_attention.rope_theta".to_string(),
            ),
            (
                "global_partial_rotary_factor".to_string(),
                "rope_parameters.full_attention.partial_rotary_factor".to_string(),
            ),
        ];
        let normed = normalize_hf_config(&json, &aliases);
        // Surfaced at top level → visible to extract_scalars (floats).
        let scalars = extract_scalars(&normed);
        assert_eq!(scalars.get("rope_theta"), Some(&1_000_000.0));
        assert_eq!(scalars.get("rope_local_base_freq"), Some(&10_000.0));
        assert_eq!(scalars.get("global_partial_rotary_factor"), Some(&0.25));

        // Explicit top-level value wins over the alias.
        let json2 = serde_json::json!({
            "rope_theta": 5.0,
            "rope_parameters": { "full_attention": { "rope_theta": 1000000.0 } }
        });
        let normed2 = normalize_hf_config(
            &json2,
            &[(
                "rope_theta".to_string(),
                "rope_parameters.full_attention.rope_theta".to_string(),
            )],
        );
        assert_eq!(extract_scalars(&normed2).get("rope_theta"), Some(&5.0));
    }

    /// Spec mirroring the gemma4 `#[forward]` carrier-mod declarations
    /// (`DECODER_PREFIX`, `BOUND_DEFAULTS`, `SCALAR_DEFAULTS`,
    /// `CONFIG_ALIASES`) — used to load the verbatim configs in tests.
    fn gemma4_carrier_spec() -> crate::arch_spec::DeclaredArchSpec {
        crate::arch_spec::DeclaredArchSpec {
            decoder_prefix: Some("language_model".into()),
            bound_defaults: vec![("max_blocks_per_seq".into(), 2048)],
            scalar_defaults: vec![("attention_multiplier".into(), 1.0)],
            config_aliases: vec![
                (
                    "global_partial_rotary_factor".into(),
                    "rope_parameters.full_attention.partial_rotary_factor".into(),
                ),
                (
                    "rope_local_base_freq".into(),
                    "rope_parameters.sliding_attention.rope_theta".into(),
                ),
                (
                    "rope_theta".into(),
                    "rope_parameters.full_attention.rope_theta".into(),
                ),
            ],
            ..Default::default()
        }
    }

    /// Load a gemma4 size's verbatim config through the full
    /// `model_params_from_json` with the carrier spec. Returns `None`
    /// if the config file isn't present in this checkout.
    fn load_verbatim_gemma4(stem: &str) -> Option<ModelParams> {
        let path = repo_model_archs("gemma4").join(format!("{stem}.json"));
        let bytes = std::fs::read(&path).ok()?;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let spec = gemma4_carrier_spec();
        Some(
            model_params_from_json(
                &normalize_hf_config(&json, &spec.config_aliases),
                stem,
                &path,
                Vec::new(),
                &spec,
            )
            .unwrap_or_else(|e| panic!("{stem} config parses: {e:?}")),
        )
    }

    /// The scalars + per-class rope geometry are family-constant, so
    /// assert them once for any gemma4 size loaded from a VERBATIM
    /// config — no hand-stamped keys.
    fn assert_gemma4_family_constants(mp: &ModelParams) {
        // Derived from the layer_types schedule (period 6: 5 sliding,
        // then 1 full).
        assert_eq!(mp.bounds.get("sliding_window_pattern"), Some(&6));
        assert_eq!(mp.bounds.get("sliding_window_global_remainder"), Some(&5));
        // Family deployment default.
        assert_eq!(mp.bounds.get("max_blocks_per_seq"), Some(&2048));
        // Rope scalars hoisted from nested rope_parameters.
        assert_eq!(mp.scalars.get("rope_theta"), Some(&1_000_000.0));
        assert_eq!(mp.scalars.get("rope_local_base_freq"), Some(&10_000.0));
        assert_eq!(mp.scalars.get("global_partial_rotary_factor"), Some(&0.25));
        // Arch-constant softmax scale (mlx hardcodes 1.0).
        assert_eq!(mp.scalars.get("attention_multiplier"), Some(&1.0));
        // Logit softcap is verbatim in-config.
        assert_eq!(mp.scalars.get("final_logit_softcapping"), Some(&30.0));
        assert_eq!(mp.tie_word_embeddings, true);
    }

    /// End-to-end: the VERBATIM upstream `gemma-4-12b-it` config (the
    /// canonical `google/gemma-4-12B-it` config.json, no hand-stamped
    /// keys) resolves to the values the prior hand-curated config
    /// carried explicitly — a regression lock for replacing the
    /// curated file with the verbatim one.
    #[test]
    fn verbatim_gemma4_12b_config_resolves_all_dsl_fields() {
        let Some(mp) = load_verbatim_gemma4("gemma-4-12b-it") else {
            return;
        };
        assert_eq!(mp.bounds.get("hidden_size"), Some(&3840));
        assert_eq!(mp.bounds.get("num_hidden_layers"), Some(&48));
        assert_eq!(mp.bounds.get("num_attention_heads"), Some(&16));
        assert_eq!(mp.bounds.get("num_key_value_heads"), Some(&8));
        assert_eq!(mp.bounds.get("num_global_key_value_heads"), Some(&1));
        // Derived dims — match the prior hand-stamped values exactly.
        assert_eq!(mp.bounds.get("attn_q_dim"), Some(&4096)); // 16 * 256
        assert_eq!(mp.bounds.get("kv_dim"), Some(&2048)); // 8 * 256
        assert_eq!(mp.bounds.get("q_global_dim"), Some(&8192)); // 16 * 512
        assert_eq!(mp.bounds.get("k_global_dim"), Some(&512)); // 1 * 512
        assert_gemma4_family_constants(&mp);
    }

    /// Same, for the verbatim 31B (60 layers, 4 global KV heads).
    #[test]
    fn verbatim_gemma4_31b_config_resolves_all_dsl_fields() {
        let Some(mp) = load_verbatim_gemma4("gemma-4-31b-it") else {
            return;
        };
        assert_eq!(mp.bounds.get("hidden_size"), Some(&5376));
        assert_eq!(mp.bounds.get("num_hidden_layers"), Some(&60));
        assert_eq!(mp.bounds.get("num_attention_heads"), Some(&32));
        assert_eq!(mp.bounds.get("num_global_key_value_heads"), Some(&4));
        assert_eq!(mp.bounds.get("q_global_dim"), Some(&16384)); // 32 * 512
        assert_eq!(mp.bounds.get("k_global_dim"), Some(&2048)); // 4 * 512
        assert_gemma4_family_constants(&mp);
    }

    #[test]
    fn missing_dir_errors_cleanly() {
        let result = load_dir(
            Path::new("/nonexistent/path/to/configs"),
            &Default::default(),
        );
        assert!(matches!(result, Err(ConfigError::NotADirectory(_))));
    }

    #[test]
    fn float_fields_land_in_scalars_integer_fields_do_not() {
        let tmp = std::env::temp_dir().join("scratchy_forward_compiler_scalars_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(
            tmp.join("m.json"),
            r#"{
                "num_hidden_layers": 16,
                "hidden_size": 2048,
                "rms_norm_eps": 0.000001,
                "query_pre_attn_scalar": 256.0,
                "attn_logit_softcapping": 50.0
            }"#,
        )
        .unwrap();
        let cfg = load_file(&tmp.join("m.json")).unwrap();

        // Integers go to bounds AND to scalars (as f64). Floats go to
        // scalars only. This double-entry for integers is required so
        // readers of scale-type fields find values regardless of how
        // the upstream HF config formatted them.
        assert_eq!(cfg.bounds.get("num_hidden_layers"), Some(&16));
        assert_eq!(cfg.bounds.get("hidden_size"), Some(&2048));
        assert_eq!(cfg.scalars.get("num_hidden_layers"), Some(&16.0));
        assert_eq!(cfg.scalars.get("hidden_size"), Some(&2048.0));

        assert_eq!(cfg.scalars.get("rms_norm_eps"), Some(&0.000001));
        assert_eq!(cfg.scalars.get("query_pre_attn_scalar"), Some(&256.0));
        assert_eq!(cfg.scalars.get("attn_logit_softcapping"), Some(&50.0));

        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn llama_configs_have_no_gemma_scalars() {
        // Regression guard: if someone adds `query_pre_attn_scalar`
        // to a Llama config, that would silently change the
        // hardcoded softmax scale in `AttentionViaCacheImpl`. The
        // Llama path has no such field today; lock that in.
        let dir = repo_model_archs("llama");
        let configs = load_dir(&dir, &Default::default()).expect("load llama configs");
        for cfg in &configs {
            assert!(
                !cfg.scalars.contains_key("query_pre_attn_scalar"),
                "{} unexpectedly has query_pre_attn_scalar",
                cfg.source_stem
            );
            assert!(
                !cfg.scalars.contains_key("attn_logit_softcapping"),
                "{} unexpectedly has attn_logit_softcapping",
                cfg.source_stem
            );
        }
    }

    #[test]
    fn non_json_files_are_ignored() {
        // Create a temp dir with one json and one txt file.
        let tmp = std::env::temp_dir().join("scratchy_forward_compiler_phase3_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("model.json"), r#"{"num_hidden_layers": 4}"#).unwrap();
        fs::write(tmp.join("README.txt"), "ignore me").unwrap();

        let configs = load_dir(&tmp, &Default::default()).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].source_stem, "model");

        fs::remove_dir_all(&tmp).ok();
    }
}
