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

use crate::draft_capture::DRAFT_CAPTURE_LAYERS;
use crate::hf_config::{extract_bounds, normalize_hf_config};
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
    /// Config-derived layer-index sets, keyed by the name a DSL
    /// `if layer in <name>:` predicate reads (`mamba_layers`, …) — see
    /// [`derive_layer_sets`]. Empty for configs without a per-layer
    /// block schedule.
    pub layer_sets: BTreeMap<String, LayerSet>,
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
    /// `scratchy_forward_compiler::ArchRegistration` row,
    /// driving the runtime `try_load(..., arch_hint)` dispatch.
    pub architectures: Vec<String>,
    /// The checkpoint architectures of the model this arch is a part of — its config's own
    /// `architectures` when the arch declares its identity (`hf_architectures`: a
    /// multi-token-prediction head, whose config is its target's); empty otherwise.
    pub drafts_for: Vec<String>,
    /// How the repo of a head published apart from its target is named from the target's
    /// (`drafter_repo_infix`, arch.json).
    pub drafter_repo_infix: Option<String>,
    /// Where a head's tensors sit inside its target's checkpoint (`checkpoint_prefix`, arch.json).
    pub checkpoint_prefix: Option<String>,
    /// The drafter its target drafts with by default: where it is published, and its target's
    /// shape (`published_as`, arch.json). `None` for every other model.
    pub published: Option<PublishedDrafter>,
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
    /// The activation a routed expert applies to its in-projection, as the config declares it
    /// — see [`ExpertActivation`].
    pub expert_activation: ExpertActivation,
    /// Each bound the loader derived where the config states none, as the dim it derived it from
    /// (`head_dim` as `hidden_size / num_attention_heads`, `attn_q_dim` as
    /// `num_attention_heads * head_dim`, an arch.json `expr` param such as `vision_head_dim` as
    /// `vision_embed_dim / vision_num_heads`, …): the model's dims as types spell each so where it still
    /// evaluates to the bound's value, which makes its derivation's law (an exact quotient) a bound
    /// rustc checks.
    pub derived: BTreeMap<String, crate::dim::Dim>,
    /// ⭐ THE TEXT BACKBONE'S GEOMETRY, typed at load ([`TextGeometry`]): the layer count, the
    /// attention head dim and the RoPE base, each resolved ONCE where the config is read. A config
    /// that cannot state one is a load error every scope reaches; no reader re-reads the keys, and
    /// none defaults them.
    pub text: TextGeometry,
}

/// The decoder layer count a config states (`num_hidden_layers`, or the length of its block
/// schedule). Never zero: a decoder of no layers is not a config.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerCount(std::num::NonZeroU32);

impl LayerCount {
    pub fn get(self) -> usize {
        self.0.get() as usize
    }
}

/// The attention head dim a config states, or the one the loader derives where it states none
/// (`hidden_size / num_attention_heads`, MLA's `qk_nope + qk_rope`: [`derive_implicit_bounds`],
/// the one derivation). Never zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadDim(std::num::NonZeroU32);

impl HeadDim {
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// The RoPE base frequency θ the model computes with: the config's stated `rope_theta` (after
/// [`normalize_hf_config`]'s hoists and the arch's `config_aliases`), else the ARCH's own default
/// (`scalar_defaults.rope_theta` in its `arch.json` — the architecture's config class default, the
/// value its checkpoints were trained against when they omit it, never a value this compiler
/// picks). A config that states none of an arch that declares none is a load error. Finite and
/// positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeTheta(f64);

impl RopeTheta {
    pub fn get(self) -> f64 {
        self.0
    }
}

/// The text backbone's geometry, typed at load — see [`ModelParams::text`]. A `#[vision_forward]`
/// config is a VL wrapper whose `text_config` states the decoder it feeds; its tower's own
/// geometry is the `vision_*` bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextGeometry {
    pub layers: LayerCount,
    pub head_dim: HeadDim,
    pub rope_theta: RopeTheta,
    /// The sliding (local) attention class's RoPE base, when the config states one
    /// (`rope_local_base_freq`, after the hoists and the arch's aliases: gemma-3/4's
    /// `rope_parameters.sliding_attention.rope_theta`, modernbert's `local_rope_theta`). NOT
    /// defaulted to [`Self::rope_theta`]: no architecture in tree defines its local base as the
    /// global one, so a body that reads a local rotary table takes it through
    /// [`Self::rope_local_for`], which refuses a config that states none.
    pub rope_local: Option<RopeTheta>,
}

impl TextGeometry {
    /// Resolve the geometry from a config's normalized json and its harvested bounds/scalars (after
    /// every derivation and arch default has run), or name the fact the config does not state.
    pub(crate) fn of(
        bounds: &BTreeMap<String, u64>,
        scalars: &BTreeMap<String, f64>,
        source_path: &Path,
    ) -> Result<Self, ConfigError> {
        let missing = |reason: String| ConfigError::Geometry {
            path: source_path.to_path_buf(),
            reason,
        };
        let layers = bounds
            .get("num_hidden_layers")
            .and_then(|&n| u32::try_from(n).ok())
            .and_then(std::num::NonZeroU32::new)
            .map(LayerCount)
            .ok_or_else(|| {
                missing(
                    "no `num_hidden_layers` (nor a block schedule to count): a decoder states \
                     how many layers it has"
                        .to_string(),
                )
            })?;
        let head_dim = bounds
            .get("head_dim")
            .and_then(|&hd| u32::try_from(hd).ok())
            .and_then(std::num::NonZeroU32::new)
            .map(HeadDim)
            .ok_or_else(|| {
                missing(format!(
                    "no nonzero `head_dim`, stated or derived (`hidden_size / \
                     num_attention_heads` where it divides): `head_dim` {:?}, `hidden_size` {:?}, \
                     `num_attention_heads` {:?}",
                    bounds.get("head_dim"),
                    bounds.get("hidden_size"),
                    bounds.get("num_attention_heads"),
                ))
            })?;
        let theta = scalars.get("rope_theta").copied().ok_or_else(|| {
            missing(
                "no `rope_theta`, and its arch declares no default \
                 (`scalar_defaults.rope_theta` in the arch's arch.json, cited to the \
                 architecture's own config class)"
                    .to_string(),
            )
        })?;
        if !(theta.is_finite() && theta > 0.0) {
            return Err(missing(format!("`rope_theta` is {theta}")));
        }
        let rope_local = match scalars.get("rope_local_base_freq").copied() {
            None => None,
            Some(t) if t.is_finite() && t > 0.0 => Some(RopeTheta(t)),
            Some(t) => return Err(missing(format!("`rope_local_base_freq` is {t}"))),
        };
        Ok(Self {
            layers,
            head_dim,
            rope_theta: RopeTheta(theta),
            rope_local,
        })
    }

    /// The local rotary class's base for a body that READS a local rotary table (`RotaryLocal`):
    /// the config's stated value, or a build error naming the config — a local table baked at the
    /// global base would rotate every sliding layer at the wrong frequency, silently.
    pub fn rope_local_for(&self, stem: &str) -> RopeTheta {
        self.rope_local.unwrap_or_else(|| {
            panic!(
                "model `{stem}`: its body reads a local (sliding) rotary table, but its config \
                 states no `rope_local_base_freq` (nor an arch alias for one) — the local base is \
                 the checkpoint's fact, never the global base"
            )
        })
    }
}

pub use crate::bundles::ExpertActivation;

/// The expert body `act` is — or, `Err`, the declared name no body realizes.
pub fn expert_body(
    act: &ExpertActivation,
) -> Result<scratchy_subtile::subtile_ir::ExpertBody, &str> {
    use scratchy_subtile::subtile_ir::{ExpertBody, GatedAct, UngatedAct};
    match act {
        ExpertActivation::Silu => Ok(ExpertBody::Gated(GatedAct::Silu)),
        ExpertActivation::Relu2 => Ok(ExpertBody::Ungated(UngatedAct::Relu2)),
        ExpertActivation::Unrealized(act) => Err(act),
    }
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
    /// A malformed `configs/<arch>/arch.json` — bad shape, bad enum
    /// value, or an unknown key. Unknown keys are an ERROR rather than
    /// a silent no-op: a typo'd declaration that quietly did nothing
    /// would mis-emit the arch, which is exactly what the token
    /// surface this file replaces was careful to prevent.
    ArchJson {
        path: PathBuf,
        reason: String,
    },
    /// A per-layer block schedule (`hybrid_override_pattern` /
    /// `layers_block_type` / `layer_types`) that names an unknown block
    /// or disagrees with `num_hidden_layers`.
    LayerPattern {
        path: PathBuf,
        reason: String,
    },
    /// A Mamba-2 `time_step_limit` that clips: the gating kernels run `dt = softplus(dt +
    /// dt_bias)` unclipped — mlx-lm's default `(0, inf)` — so a config whose limit can bind is
    /// refused rather than run as if it could not ([`binding_time_step_limit`]).
    TimeStepLimit {
        path: PathBuf,
        limit: serde_json::Value,
    },
    /// A config's quant spec carries per-module (mixed-bit, OptiQ/DWQ) affine
    /// overrides that this backend cannot lower (only metal reads the
    /// per-tensor widths back: `unpack_moe_expert_bits` /
    /// `affine_gather_qmv_symbol`). Raised at the ONE place the quant spec is
    /// decided — `parse_quantization` — naming the config and WHICH tensors
    /// are mixed-bit. The build refuses; it never silently emits nothing and
    /// lets a later "no models selected" panic stand in for the fact.
    MixedBitAffine {
        stem: String,
        tensors: Vec<String>,
    },
    /// A config whose text geometry ([`TextGeometry`]) cannot be stated: no layer count, a zero
    /// head dim, no RoPE base where neither it nor its arch states one.
    Geometry {
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
            Self::ArchJson { path, reason } => {
                write!(f, "arch declaration {}: {reason}", path.display())
            }
            Self::LayerPattern { path, reason } => {
                write!(f, "layer schedule in {}: {reason}", path.display())
            }
            Self::TimeStepLimit { path, limit } => write!(
                f,
                "{}: time_step_limit {limit} clips Mamba-2's dt, which no kernel realizes \
                 (only the unclipped default, absent or `[0.0, null]`, is supported)",
                path.display()
            ),
            Self::MixedBitAffine { stem, tensors } => write!(
                f,
                "{stem}: ships a mixed-bit (OptiQ) affine quantization — per-tensor width \
                 overrides only metal lowers; this backend cannot compile it. Mixed-bit \
                 tensors: {}",
                tensors.join(", ")
            ),
            Self::Geometry { path, reason } => {
                write!(f, "text geometry of {}: {reason}", path.display())
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
/// - `arch.json` — the arch's own declaration file (see
///   [`load_arch_json`]), not a checkpoint.
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
    scope: &Scope,
) -> Result<Vec<ModelParams>, ConfigError> {
    load_dir_mode(dir, ConfigMode::Decoder, spec, scope)
}

/// [`load_dir`] for a `#[vision_forward]` crate's configs/ dir.
/// Same file discovery / overlay synthesis / filtering; per-file
/// `ModelParams` construction goes through the vision path
/// (per-family `vision_*` bound derivation from the nested HF
/// `vision_config` block) instead of the flat decoder harvest.
pub fn load_dir_vision(
    dir: &Path,
    spec: &crate::arch_spec::DeclaredArchSpec,
    scope: &Scope,
) -> Result<Vec<ModelParams>, ConfigError> {
    load_dir_mode(dir, ConfigMode::Vision, spec, scope)
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

/// Process-wide set of the base configs (`configs/<arch>/<stem>.json`) whose model feature this
/// build enables, across every arch directory processed so far — the scope anything else baked
/// per model (its tokenizer) follows. Read after every `load_dir`/`load_dir_vision` has run.
fn selected_configs() -> &'static std::sync::Mutex<std::collections::BTreeSet<PathBuf>> {
    static SELECTED: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<PathBuf>>> =
        std::sync::OnceLock::new();
    SELECTED.get_or_init(Default::default)
}

pub fn selected_config_paths() -> Vec<PathBuf> {
    selected_configs().lock().unwrap().iter().cloned().collect()
}

/// What a build selects: the one fact every config-loading pass reads — which models, which
/// heads, which quant presets. A PURE VALUE: no fact enters a scope except as an argument,
/// and this crate reads no environment. The build script of the crate that owns the feature
/// flags states it — [`Scope::from_feature_flags`] with cargo's own delivery of those flags
/// (the `CARGO_FEATURE_*` names for the stem list, `cfg!` for `mtp`/`mtp-heads`) plus the
/// `quant/<preset>` union via [`Scope::with_quants`]; a test states [`Scope::all`] or
/// [`Scope::stems`] — the scope it means, never an env var hand-set into a test invocation.
#[derive(Clone, Debug)]
pub struct Scope {
    /// Which models are selected, as the Cargo feature names their gates probe.
    stems: StemSelection,
    /// `mtp` (`spec/mtp`): drafters published as checkpoints of their own compile too.
    mtp: bool,
    /// `mtp-heads`: heads shipping inside their targets' checkpoints compile beside them.
    mtp_heads: bool,
    /// The quant presets the enabled `quant/<preset>` features named, as the
    /// build computed them. `None` = unset = legacy behaviour (every
    /// backend-applicable preset; see `quant_preset_active`).
    quants: Option<std::collections::HashSet<String>>,
}

/// The selected-models predicate's data. Cargo feature names are one flat namespace with no
/// directory scoping, so the scope stores NAMES and the predicate probes the exact candidates
/// an `(arch, stem)` pair legitimately owns — never a blind OR across candidates belonging to
/// a different arch (the `VISION_ARCH_FOR_STEM` cross-contamination trap, documented at its
/// table).
#[derive(Clone, Debug)]
enum StemSelection {
    /// Every stem selected — what a test means by "the configs on disk, all of them".
    All,
    /// The enabled feature names, exactly as [`Scope::from_feature_flags`] received them.
    Features(std::collections::HashSet<String>),
}

/// A handful of stems are byte-identical files checked into two arch dirs at once (the same
/// checkpoint, loadable via a vision arch or its non-vision base arch — e.g. `qwen2-vl-2b-mlx`
/// under both `qwen2/` and `qwen2-vl/`); for those the vision arch keeps the bare stem and the
/// base arch's copy is `<stem>-text-only` (this table, verified per pair against
/// crates/models/arch/Cargo.toml — mirrors the Cargo.toml feature-name generator exactly,
/// must stay in sync with it). The table matters for correctness, not just naming: the bare
/// `<stem>` feature genuinely exists (scoped to the vision arch), so a naive "check bare OR
/// arch-prefixed OR text-only-suffixed, any hit counts" probe would cross-contaminate —
/// enabling the vision arch's bare feature would ALSO look enabled to the base arch's
/// identical-stem file, since Cargo features are one flat namespace with no directory
/// scoping baked into the env var. So each (arch, stem) pair here checks EXACTLY the one
/// candidate name that arch could legitimately own, never a blind OR across candidates that
/// belong to a different arch.
///
/// Any other collision not covered by the table falls back to `<arch>-<stem>` — safe to also
/// probe bare `<stem>` there without cross-contamination risk, because in that fallback
/// shape Cargo never declares a bare `<stem>` feature for EITHER colliding arch (only the
/// two `<arch>-<stem>` forms), so the bare env var can never be set.
const VISION_ARCH_FOR_STEM: &[(&str, &str)] = &[
    ("qwen2-vl-2b-mlx", "qwen2-vl"),
    ("qwen2.5-vl-3b-mlx", "qwen2-5-vl"),
    ("locateanything-3b", "locateanything"),
    ("qwen3.5-9b", "qwen3-5-vl"),
    ("qwen3.6-27b", "qwen3-5-vl"),
    ("qwen3.8-27b", "qwen3-5-vl"),
];

/// The Cargo feature names an `(arch, stem)` pair's gate probes, in the priority the ambient
/// reads established: a `VISION_ARCH_FOR_STEM` stem owns EXACTLY ONE candidate (the bare
/// stem under its vision arch, `<stem>-text-only` under the base arch sharing the identical
/// file); any other stem falls back to the bare name OR'd with `<arch>-<stem>`, where Cargo
/// never declares the bare name for either colliding arch, so probing it is safe.
fn feature_candidates(arch_name: &str, stem: &str) -> [String; 2] {
    match VISION_ARCH_FOR_STEM.iter().find(|(s, _)| *s == stem) {
        Some((_, vision_arch)) if arch_name == *vision_arch => [stem.to_string(), String::new()],
        Some(_) => [format!("{stem}-text-only"), String::new()],
        None => [stem.to_string(), format!("{arch_name}-{stem}")],
    }
}

/// Cargo's own feature-name → env-var transform: uppercase, `-` → `_`, every other byte
/// verbatim — including `.`.
fn feature_env_name(feature: &str) -> String {
    feature.to_uppercase().replace('-', "_")
}

impl Scope {
    /// The build's own statement, as pure values. `stems` is cargo's delivery of the
    /// target crate's `model/*` feature flags — the `CARGO_FEATURE_*` names, read by the
    /// build script of the crate that owns them (the ONE dynamic read: `cfg!` cannot
    /// loop over the stem list, and a hand-restated probe table would be a second copy
    /// of Cargo.toml). `mtp`/`mtp_heads` arrive as the booleans that crate's own `cfg!`
    /// states — valid there, invalid here, where `cfg!` would see only the macro
    /// crate's features. Quant presets are not part of this call: they live on
    /// scratchy-quantizations and arrive via [`Scope::with_quants`].
    pub fn from_feature_flags<I, S>(stems: I, mtp: bool, mtp_heads: bool) -> Scope
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Scope {
            stems: StemSelection::Features(stems.into_iter().map(Into::into).collect()),
            mtp,
            mtp_heads,
            quants: None,
        }
    }

    /// Every model selected, no heads, every backend-applicable preset —
    /// the scope a test means when it pins what an arch's configs/ on disk contain. An
    /// explicit scope selecting nothing is caught by those pins; vacuity is impossible by
    /// construction.
    pub fn all() -> Scope {
        Scope {
            stems: StemSelection::All,
            mtp: false,
            mtp_heads: false,
            quants: None,
        }
    }

    /// The same scope with its presets stated: `Some` of exactly these names — an empty
    /// iteration is the build's "no quant features enabled" (dense-only), a non-empty one
    /// is the forwarded `quant/<preset>` union — replacing the every-preset `None` default.
    pub fn with_quants<I, S>(mut self, presets: I) -> Scope
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.quants = Some(presets.into_iter().map(Into::into).collect());
        self
    }

    /// A test's statement of the models it means, as explicit `(arch, stem)` pairs. Each
    /// pair enables exactly the feature name its own gate probes: the one candidate a
    /// `VISION_ARCH_FOR_STEM` pair owns, else the arch-qualified `<arch>-<stem>` — never
    /// the bare `<stem>`, which belongs to no arch and would leak into every other arch
    /// probing the same name.
    pub fn stems(pairs: &[(&str, &str)]) -> Scope {
        let mut on = std::collections::HashSet::new();
        for (arch, stem) in pairs {
            let name = match VISION_ARCH_FOR_STEM.iter().find(|(s, _)| *s == *stem) {
                Some((_, vision_arch)) if *arch == *vision_arch => stem.to_string(),
                Some(_) => format!("{stem}-text-only"),
                None => format!("{arch}-{stem}"),
            };
            on.insert(feature_env_name(&name));
        }
        Scope {
            stems: StemSelection::Features(on),
            mtp: false,
            mtp_heads: false,
            quants: None,
        }
    }

    /// Whether this scope enables the model feature `feature` — membership in the names
    /// [`Scope::from_feature_flags`] received (or [`Scope::stems`] stated), never an env
    /// probe.
    fn feature_on(&self, feature: &str) -> bool {
        match &self.stems {
            StemSelection::All => true,
            StemSelection::Features(on) => on.contains(&feature_env_name(feature)),
        }
    }

    /// Whether `(arch, stem)` is selected: the pair's own candidates, probed against the
    /// scope's stated names.
    fn model_selected(&self, arch_name: &str, stem: &str) -> bool {
        match &self.stems {
            StemSelection::All => true,
            StemSelection::Features(_) => feature_candidates(arch_name, stem)
                .iter()
                .filter(|c| !c.is_empty())
                .any(|c| self.feature_on(c)),
        }
    }

    /// Whether `arch` compiles in this build: its `arch-<name>` feature is on, and so is a
    /// backend — probed against the scope's stated feature names, never the environment.
    /// `scratchy-models`' build script runs an arch's pipeline only then (its `src/lib.rs`
    /// gates each arch's `pub mod` the same way), and a target bakes in the facts of the
    /// heads that compile beside it alone. Whether an arch ACTUALLY lowers is each
    /// lowering's verdict — an op with no spyre lowering is a build error naming it, not a
    /// silent skip here.
    pub fn arch_compiled(&self, arch: &str) -> bool {
        match &self.stems {
            StemSelection::All => true,
            StemSelection::Features(on) => {
                arch_compiled_with(arch, |feature| on.contains(&feature_env_name(feature)))
            }
        }
    }
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
    scope: &Scope,
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
    // carries the "this is a model" meaning. Which features are enabled is
    // the SCOPE's statement (see `Scope`): the build states
    // `Scope::from_feature_flags`, a test states `Scope::all`/`Scope::stems`
    // — the candidate-name law
    // itself (which name an (arch, stem) pair legitimately owns) lives at
    // `VISION_ARCH_FOR_STEM` and `feature_candidates` above.
    //
    // A base whose feature isn't enabled is skipped BEFORE `read_json_file`
    // ever opens it below — the point being an unselected model's JSON is
    // never even read, let alone forward-expanded.
    let arch_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| ConfigError::NotADirectory(dir.to_path_buf()))?;
    let model_feature_enabled =
        |arch_name: &str, stem: &str| -> bool { scope.model_selected(arch_name, stem) };
    // A head (an arch registering under an identity of its own, `hf_architectures`) compiles beside
    // the targets it drafts for: one that ships inside its targets' checkpoints (`checkpoint_of`)
    // under scratchy-models' `mtp-heads` feature (which `metal` turns on, and `mtp` implies), on
    // every selected config of those arches that carries it — its config is the target's own
    // file; one published as a checkpoint of its own (`head_targets`), on each of its configs
    // whose target is selected ([`compiled_head_targets`]): under `mtp-heads` the one its target
    // drafts with by default (`published_as`), under `mtp` (`spec/mtp`) every one, which drafts
    // when `--speculative-model` names it. Either way the target's verify steps carry the head's
    // drafts (`spec_drafts`, the head's arch.json).
    let mtp = scope.mtp;
    let mtp_heads = scope.mtp_heads;
    let configs_root = dir
        .parent()
        .ok_or_else(|| ConfigError::NotADirectory(dir.to_path_buf()))?;
    let head_target_selected = |head_path: &Path| -> bool {
        let stem = head_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let target = compiled_head_targets(spec, mtp).find(|(s, _)| s == stem);
        mtp_heads
            && target.is_some_and(|(_, target)| {
                let (arch, stem) = target.split_once('/').unwrap_or_default();
                model_feature_enabled(arch, stem)
            })
    };
    // What the heads compiled beside a target config bake into it: the most drafts one of them
    // carries a sequence (its verify steps hold that many), and the layers whose outputs one reads
    // (`target_capture_layers`, its arch.json: a path into its own config), which the target
    // captures. A head that does not compile in this build ([`Scope::arch_compiled`]: a metal-only
    // drafter under cuda) bakes nothing into it.
    let head_facts = |stem: &str, json: &serde_json::Value| -> Result<HeadFacts, ConfigError> {
        let mut facts = HeadFacts::default();
        if !mtp_heads || !spec.hf_architectures.is_empty() {
            return Ok(facts);
        }
        let this = format!("{arch_name}/{stem}");
        for arch_dir in arch_dirs(configs_root)? {
            let head_arch = (arch_dir.file_name()).and_then(|n| n.to_str());
            if !head_arch.is_some_and(|a| scope.arch_compiled(a)) {
                continue;
            }
            let head = load_arch_json(&arch_dir)?;
            let configs: Vec<String> = (compiled_head_targets(&head, mtp))
                .filter(|(_, t)| *t == this)
                .map(|(s, _)| s.clone())
                .collect();
            let in_checkpoint =
                head.checkpoint_of.iter().any(|a| a == arch_name) && carries_head(&head, json);
            if configs.is_empty() && !in_checkpoint {
                continue;
            }
            if let Some(&(_, d)) = (head.bound_defaults.iter()).find(|(k, _)| k == "spec_drafts") {
                facts.drafts = Some(facts.drafts.map_or(d, |m: u64| m.max(d)));
            }
            let Some(path) = &head.target_capture_layers else {
                continue;
            };
            for config in &configs {
                let head_path = arch_dir.join(format!("{config}.json"));
                let (_, head_json) = read_json_file(&head_path)?;
                let layers: Vec<u64> = crate::arch_spec::json_path(&head_json, path)
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.iter().map(|v| v.as_u64()).collect())
                    .ok_or_else(|| ConfigError::ArchJson {
                        path: head_path.clone(),
                        reason: format!("`{path}` must be an array of layer indices"),
                    })?;
                match &facts.capture_layers {
                    Some(other) if *other != layers => {
                        return Err(ConfigError::ArchJson {
                            path: head_path,
                            reason: format!(
                                "two heads of `{this}` read different target layers \
                                 ({other:?} and {layers:?}): a target captures one set"
                            ),
                        });
                    }
                    _ => facts.capture_layers = Some(layers),
                }
            }
        }
        Ok(facts)
    };

    // Quant scoping. The scope's `quants` restricts the synthesized
    // `<base>-<preset>` variants to those presets (dense bases always emit).
    // The build states them as a value — the union of the enabled
    // `quant/<preset>` features (scratchy-quantizations, via its own `cfg!`),
    // handed to the scope by the build script; a test states `None`.
    // `None` = legacy behaviour (every backend-applicable preset emits);
    // `Some` (even an empty set) = emit only the listed presets, i.e.
    // dense-only by default.
    let enabled_quants = &scope.quants;
    // Whether `preset_name` both survived Cargo-feature quant scoping AND is
    // one this backend can actually claim. Shared by the dense-suppression
    // pre-check (below, before the base loop) and the overlay-synthesis
    // loop (further down) so the two can never drift apart — a preset that
    // wouldn't actually synthesize must never suppress the dense fallback,
    // or a model ends up with ZERO compiled variants.
    //
    // Quant feature scoping: when the scope's `quants` is `Some` (the build
    // computed the union of the enabled `quant/<preset>` features from
    // scratchy-quantizations), only listed presets are active. `None` ⇒
    // legacy all-presets behaviour.
    //
    // Backend claimability — metal: `mlx-affine-*` variants reach the
    // solver as Dense (codegen dequantizes at load time), and `nvfp4`
    // reaches it as a genuine metal quant claimed by `MetalNvfp4QmmImpl`
    // (forward-time E2M1 dequant-on-read qmv/qmm_t). Every other preset
    // routes through Impls (Marlin / Bnb4 / Fp8 / Ggml) that the metal
    // impl pool has no claimants for, so the solver would explode
    // with `UnclaimedTile`.
    let quant_preset_active = |preset_name: &str| -> bool {
        if let Some(q) = &enabled_quants
            && !q.contains(preset_name)
        {
            return false;
        }
        !(cfg!(feature = "metal")
            && !preset_name.starts_with("mlx-affine-")
            && preset_name != "nvfp4")
    };

    // An in-checkpoint head's configs are the selected configs of its target arches that carry it.
    let mut in_checkpoint: Vec<PathBuf> = Vec::new();
    for target_arch in spec.checkpoint_of.iter().filter(|_| mtp_heads) {
        for p in model_configs(&configs_root.join(target_arch))? {
            if model_feature_enabled(target_arch, &stem_of(&p)?)
                && carries_head(spec, &read_json_file(&p)?.1)
            {
                in_checkpoint.push(p);
            }
        }
    }
    let mut base_paths = model_configs(dir)?;
    base_paths.extend(in_checkpoint.iter().cloned());
    base_paths.sort_by(|a, b| {
        a.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .cmp(b.file_stem().and_then(|s| s.to_str()).unwrap_or(""))
    });

    // A quant preset means "compile this precision instead of dense", not
    // "compile this precision too" — if this arch has at least one preset
    // that's both selected (via `quant/<preset>`, forwarded through
    // the scope's `quants`) and actually claimable on this backend, every
    // selected base in this arch gets its quant variant(s) INSTEAD of the
    // dense/bf16 one below, not alongside it. Deliberately arch-wide, not
    // per-base: `quantizations.json` is arch-wide too, so there is no
    // per-model "just this one dense" escape hatch today.
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
        // when this base's `<stem>` (or `<arch>-<stem>`) feature isn't enabled
        // and it isn't the head of a selected target (`in_checkpoint`,
        // `head_target_selected`). Every quant variant synthesized from this
        // base below shares its raw JSON via `base_raw`, so skipping here
        // (instead of after parsing) also means an unselected base contributes
        // no quant variants, not just no dense emission.
        if !in_checkpoint.contains(p)
            && !model_feature_enabled(arch_name, &stem)
            && !head_target_selected(p)
        {
            continue;
        }
        selected_configs().lock().unwrap().insert(p.clone());
        let (raw, mut json) = read_json_file(p)?;
        let facts = head_facts(&stem, &json)?;
        if let Some(obj) = json.as_object_mut() {
            if let Some(drafts) = facts.drafts {
                obj.insert("spec_drafts".to_string(), drafts.into());
            }
            if let Some(layers) = facts.capture_layers {
                obj.insert(DRAFT_CAPTURE_LAYERS.to_string(), layers.into());
            }
        }
        // Dense-base drift surface: `<stem>.overrides.json` deep-merges
        // onto the verbatim checkpoint config (same mechanism the
        // synthesized quant variants already get). Tracked so cargo
        // rebuilds on override edits.
        let mut tracked: Vec<PathBuf> = Vec::new();
        apply_overrides_file(dir, &stem, &mut json, &mut tracked)?;
        // `arch_has_active_quant` gates the dense emission: a selected quant
        // preset replaces dense, it doesn't add to it. The base still needs
        // to stay in `base_raw` below so a quant variant can be synthesized
        // from it.
        if !arch_has_active_quant {
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
    // Under `--features metal` we skip quant variants metal can't claim
    // (awq-gemm / gptq-* / bnb-nf4-dq / fp8-* / ct-int4-sym / ggml) —
    // their on-disk layouts route through `MarlinFusedGateUpSiluMul` /
    // `Bnb4Linear` / `Fp8Linear` / `GgmlLinear` impls that the metal
    // impl pool has no claimants for, so the solver explodes with
    // `UnclaimedTile` on the first quant tile. A preset being
    // skipped this way on metal means it was never `quant_preset_active`
    // in the first place, so it also never suppressed the dense base
    // above — dense only drops out for a preset that actually synthesizes.
    //
    // MLX-affine presets (`mlx-affine-b<bits>-g<gs>`) are the
    // exception: their codegen materializes a Dense `LinearLayer` at
    // load time via `LinearLayer::load_affine_dequant_as_dense`, so
    // the forward path remains the existing bf16 Gemm path that the
    // metal impl pool already claims. The preset-filter inside the
    // overlay loop below keeps unclaimable presets out while still
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
            // Quant feature scoping (the scope's `quants`) + backend claimability
            // (metal only claims mlx-affine-*/nvfp4 as those route through
            // Dense/MetalNvfp4QmmImpl — see
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
                let mut extra_tracked = vec![preset_path.clone()];
                let merged = merge_variant_config(
                    dir,
                    base_json,
                    &variant_stem,
                    &preset_json,
                    &mut extra_tracked,
                )?;

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

/// An arch dir's model configs: its `*.json` but the arch's own declarations (`arch.json`,
/// `weights.json`, `quantizations.json`) and the drift files (`*.overrides.json`).
fn model_configs(dir: &Path) -> Result<Vec<PathBuf>, ConfigError> {
    let io = |source| ConfigError::Io {
        path: dir.to_path_buf(),
        source,
    };
    let mut configs = Vec::new();
    for entry in fs::read_dir(dir).map_err(io)? {
        let path = entry.map_err(io)?.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let declaration = matches!(name, "weights.json" | "quantizations.json" | ARCH_JSON);
        if name.ends_with(".json") && !declaration && !name.ends_with(".overrides.json") {
            configs.push(path);
        }
    }
    Ok(configs)
}

/// [`Scope::arch_compiled`] under the features `on` reports.
fn arch_compiled_with(arch: &str, on: impl Fn(&str) -> bool) -> bool {
    on(&format!("arch-{arch}")) && (on("metal") || on("spyre"))
}

/// Every arch dir under `configs/`.
fn arch_dirs(configs_root: &Path) -> Result<Vec<PathBuf>, ConfigError> {
    let io = |source| ConfigError::Io {
        path: configs_root.to_path_buf(),
        source,
    };
    let mut dirs = Vec::new();
    for entry in fs::read_dir(configs_root).map_err(io)? {
        let path = entry.map_err(io)?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    Ok(dirs)
}

/// Whether a checkpoint config carries the in-checkpoint head `head` (`checkpoint_of`): it declares
/// the head's layers — the config field the head's `num_hidden_layers` replaces its target's from
/// (`mtp_num_hidden_layers`) — above zero.
pub(crate) fn carries_head(
    head: &crate::arch_spec::DeclaredArchSpec,
    json: &serde_json::Value,
) -> bool {
    crate::hf_config::carries_head(&head.params, &head.config_aliases, json)
}

/// Load a single config.json as a dense-base ModelParams.
///
/// Test-only: the macro reads configs through the directory walk, not one file at a time.
#[cfg(test)]
pub fn load_file(path: &Path) -> Result<ModelParams, ConfigError> {
    let (_, json) = read_json_file(path)?;
    let stem = stem_of(path)?;
    // The arch's declarations are part of what a config means (its RoPE base where the config
    // omits it, its key aliases): read them as the walk does, never a default spec.
    let dir = path
        .parent()
        .ok_or_else(|| ConfigError::NotADirectory(path.to_path_buf()))?;
    model_params_from_json(&json, &stem, path, Vec::new(), &load_arch_json(dir)?)
}

/// Deep-merge `<stem>.overrides.json` onto `json` when the file exists, pushing
/// its path onto `tracked` so cargo rebuilds on override edits.
///
/// Both drift tiers go through here: the dense base (`<stem>`) and a synthesized
/// quant variant (`<stem>-<preset>`).
fn apply_overrides_file(
    dir: &Path,
    stem: &str,
    json: &mut serde_json::Value,
    tracked: &mut Vec<PathBuf>,
) -> Result<(), ConfigError> {
    let path = dir.join(format!("{stem}.overrides.json"));
    if path.exists() {
        let (_, override_json) = read_json_file(&path)?;
        deep_merge(json, &override_json);
        tracked.push(path);
    }
    Ok(())
}

/// Build the config a synthesized `(stem, preset)` variant compiles from, in
/// precedence order: `base_json` (the verbatim checkpoint config with its own
/// `<stem>.overrides.json` already applied) → the preset fragment → the
/// per-`(stem, preset)` `<variant_stem>.overrides.json`.
///
/// ⛔ ONE definition, called by both [`load_dir_mode`]'s overlay loop and
/// [`load_file_with_preset`]. A test that rebuilt this sequence by hand could
/// assert against a merge the build never performs — and did: it applied the
/// variant drift file while skipping the base's own `<stem>.overrides.json`.
fn merge_variant_config(
    dir: &Path,
    base_json: &serde_json::Value,
    variant_stem: &str,
    preset_json: &serde_json::Value,
    tracked: &mut Vec<PathBuf>,
) -> Result<serde_json::Value, ConfigError> {
    let mut merged = base_json.clone();
    deep_merge(&mut merged, preset_json);
    apply_overrides_file(dir, variant_stem, &mut merged, tracked)?;
    Ok(merged)
}

/// Load a single config.json with one quant preset deep-merged onto it —
/// the `(size, preset)` variant the overlay-synthesis loop in [`load_dir`]
/// builds, reachable one pair at a time.
///
/// Test-only, and for the same reason [`load_file`] is: a proc-macro crate has
/// no model/quant features, so `load_dir` selects ZERO configs there and any
/// test iterating its result passes vacuously. Naming the base and the preset
/// explicitly skips the build's scope.
///
/// The merge itself is NOT rebuilt here — it goes through the same
/// [`apply_overrides_file`] + [`merge_variant_config`] the build uses, and the
/// arch's own [`load_arch_json`] spec is loaded too, so what comes back is what
/// a real build compiles. Both matter: a hand-rolled sequence previously skipped
/// the base's `<stem>.overrides.json`, and a `DeclaredArchSpec::default()` drops
/// arch-level facts like `decoder_safetensors_prefix`, either of which leaves a
/// test asserting against something the build never produces.
#[cfg(test)]
pub fn load_file_with_preset(
    config_path: &Path,
    preset_path: &Path,
) -> Result<ModelParams, ConfigError> {
    let base_stem = stem_of(config_path)?;
    let preset_stem = stem_of(preset_path)?;
    let variant_stem = format!("{base_stem}-{preset_stem}");
    let dir = config_path
        .parent()
        .ok_or_else(|| ConfigError::NotADirectory(config_path.to_path_buf()))?;

    let (_, mut base_json) = read_json_file(config_path)?;
    let mut tracked: Vec<PathBuf> = Vec::new();
    apply_overrides_file(dir, &base_stem, &mut base_json, &mut tracked)?;
    let (_, preset_json) = read_json_file(preset_path)?;
    let merged = merge_variant_config(dir, &base_json, &variant_stem, &preset_json, &mut tracked)?;

    let spec = load_arch_json(dir)?;
    model_params_from_json(&merged, &variant_stem, config_path, tracked, &spec)
}

/// One dense base as the macro's directory walk loads it: its `<stem>.overrides.json` and its
/// arch's declarations (the `arch.json` beside it) applied, which [`load_file`] leaves out.
#[cfg(test)]
pub fn load_file_declared(config_path: &Path) -> Result<ModelParams, ConfigError> {
    load_file_declared_mode(config_path, ConfigMode::Decoder)
}

/// The load-OR-refuse law, stated once for tests: a config either loads, or —
/// under a backend that cannot lower its mixed-bit (OptiQ) affine
/// quantization — the loader refuses NAMING the config and its mixed-bit
/// tensors. A well-formed refusal is asserted here and returned as `None`
/// (the caller skips the config); any other error still fails the test
/// loudly. Under a backend that lowers mixed-bit affine this never refuses.
#[cfg(test)]
pub(crate) fn assert_load_or_refuse_by_name(
    stem: &str,
    result: Result<ModelParams, ConfigError>,
) -> Option<ModelParams> {
    match result {
        Ok(model) => Some(model),
        Err(ConfigError::MixedBitAffine {
            stem: refused,
            tensors,
        }) => {
            assert_eq!(refused, stem, "the refusal names its config");
            assert!(
                !tensors.is_empty(),
                "{stem}: the refusal names its mixed-bit tensors"
            );
            None
        }
        Err(e) => panic!("{stem}: {e}"),
    }
}

/// [`load_file_declared`] for a `@vision_forward` carrier's config: its `vision_*` bounds derived
/// as [`load_dir_vision`] derives them.
#[cfg(test)]
pub(crate) fn load_file_declared_vision(config_path: &Path) -> Result<ModelParams, ConfigError> {
    load_file_declared_mode(config_path, ConfigMode::Vision)
}

#[cfg(test)]
fn load_file_declared_mode(
    config_path: &Path,
    mode: ConfigMode,
) -> Result<ModelParams, ConfigError> {
    let stem = stem_of(config_path)?;
    let dir = config_path
        .parent()
        .ok_or_else(|| ConfigError::NotADirectory(config_path.to_path_buf()))?;
    let (_, mut json) = read_json_file(config_path)?;
    let mut tracked: Vec<PathBuf> = Vec::new();
    apply_overrides_file(dir, &stem, &mut json, &mut tracked)?;
    let spec = load_arch_json(dir)?;
    model_params_from_json_mode(&json, &stem, config_path, tracked, mode, &spec)
}

/// Load one in-checkpoint head base — its TARGET's config through the head arch's own spec
/// and drift tier — the sequence `load_dir_mode`'s base loop performs for an `in_checkpoint`
/// head, reachable one pair at a time.
///
/// Test-only, for the same reasons [`load_file`] is, plus one more: the head's spec and
/// `.overrides.json` live in the HEAD's configs dir while its base config lives in the
/// target's, so no single-dir loader can express it. The merge sequence is NOT rebuilt
/// here — it is the walk's own (read the target's json, deep-merge the head dir's
/// `<stem>.overrides.json`, parse under the head's `load_arch_json` spec), so a test
/// asserts against what a real build compiles.
#[cfg(test)]
pub fn load_head_file(
    target_config_path: &Path,
    head_arch_dir: &Path,
) -> Result<ModelParams, ConfigError> {
    let stem = stem_of(target_config_path)?;
    let (_, mut json) = read_json_file(target_config_path)?;
    let mut tracked: Vec<PathBuf> = Vec::new();
    apply_overrides_file(head_arch_dir, &stem, &mut json, &mut tracked)?;
    let spec = load_arch_json(head_arch_dir)?;
    model_params_from_json(&json, &stem, target_config_path, tracked, &spec)
}

/// Read + parse a JSON file, returning the raw string (for
/// diagnostics) and the parsed `Value`. Centralized so
/// `ConfigError::{Io, Json}` always carry the right path.
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
    /// its `ArchRegistration` row so the dispatcher tries it.
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

/// The `head_targets` configs of `head` that compile when their target is selected, under
/// `mtp-heads`: every one under `mtp` (`spec/mtp`); else the one its target drafts with by default
/// (`published_as`) — every variant of it this build compiles, though only those `published_as`
/// names a repo for draft with no `--speculative-model`.
fn compiled_head_targets(
    head: &crate::arch_spec::DeclaredArchSpec,
    mtp: bool,
) -> impl Iterator<Item = &(String, String)> {
    (head.head_targets.iter())
        .filter(move |(stem, _)| mtp || head.published_as.iter().any(|p| p.config == *stem))
}

/// A drafter published as a checkpoint of its own that its target drafts with by default
/// (`published_as`, arch.json): the Hub repo it loads from, and the shape of the target config it
/// compiled against — what tells that target's checkpoints apart from the others of its
/// architecture before a load (Gemma 4's 26B-A4B and 31B are both
/// `Gemma4ForConditionalGeneration`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PublishedDrafter {
    pub repo: String,
    pub target_hidden_size: u64,
    pub target_num_hidden_layers: u64,
}

/// What a head knows of its target: the architectures it drafts for — its target config's, named
/// by its arch's `head_targets` when it has a config of its own (`source_path`'s stem: a quant
/// variant's base, whose own stem carries its preset); else its config's own (it is its target's,
/// verbatim) — for the variant `source_stem` of the drafter its target drafts with by default
/// (`published_as`), where it is published and its target's shape, and the experts its target
/// routes each row through ([`routed_experts`]).
fn head_target(
    spec: &crate::arch_spec::DeclaredArchSpec,
    json: &serde_json::Value,
    source_stem: &str,
    source_path: &Path,
) -> Result<(Vec<String>, Option<PublishedDrafter>, u64), ConfigError> {
    let stem = stem_of(source_path)?;
    let Some((_, target)) = spec.head_targets.iter().find(|(s, _)| *s == stem) else {
        return Ok((parse_architectures(json), None, routed_experts(json)));
    };
    let configs = (source_path.parent().and_then(Path::parent))
        .ok_or_else(|| ConfigError::NotADirectory(source_path.to_path_buf()))?;
    let path = configs.join(format!("{target}.json"));
    let (_, target_json) = read_json_file(&path)?;
    // A composite checkpoint config (gemma4's) nests its text backbone's fields under
    // `text_config`.
    let backbone = |key: &str| {
        (target_json.get(key))
            .or_else(|| target_json.get("text_config")?.get(key))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ConfigError::ArchJson {
                path: path.clone(),
                reason: format!("the target of `{stem}` declares no `{key}`"),
            })
    };
    let published = (spec.published_as.iter().find(|p| p.variant == source_stem))
        .map(|p| -> Result<_, ConfigError> {
            Ok(PublishedDrafter {
                repo: p.repo.clone(),
                target_hidden_size: backbone("hidden_size")?,
                target_num_hidden_layers: backbone("num_hidden_layers")?,
            })
        })
        .transpose()?;
    let experts = routed_experts(&target_json);
    Ok((parse_architectures(&target_json), published, experts))
}

/// The experts a target config's MoE layers route among (`num_experts`, under a composite
/// checkpoint's `text_config`); 0 for a dense target.
fn routed_experts(json: &serde_json::Value) -> u64 {
    (json.get("num_experts"))
        .or_else(|| json.get("text_config")?.get("num_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
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
    stem: &str,
    source_path: &Path,
) -> Result<Option<crate::quantization::QuantizationConfig>, ConfigError> {
    let parsed = crate::quantization::QuantizationConfig::parse(json).map_err(|e| {
        ConfigError::Quantization {
            path: source_path.to_path_buf(),
            source: e,
        }
    })?;
    // THE one place mixed-bit-ness is decided: the quant spec itself. A
    // per-module (OptiQ/DWQ) override map is a fact of the checkpoint; a
    // backend that cannot lower it refuses here, by name, listing the
    // tensors — no second, downstream re-derivation from the raw JSON.
    if let Some(cfg) = &parsed
        && let crate::quantization::QuantMethod::Affine { per_module, .. } = &cfg.method
        && !per_module.is_empty()
        && !cfg!(feature = "metal")
    {
        return Err(ConfigError::MixedBitAffine {
            stem: stem.to_string(),
            tensors: per_module.iter().map(|(k, _)| k.clone()).collect(),
        });
    }
    Ok(parsed)
}

/// `None` (field absent) lets `apply_arch_semantic_defaults` supply
/// the family's modeling-code default (HF's own default is TRUE;
/// gemma2/gemma3 checkpoints rely on it); explicit json wins.
fn parse_tie_word_embeddings(json: &serde_json::Value) -> Option<bool> {
    json.get("tie_word_embeddings").and_then(|v| v.as_bool())
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
    if let Some(r) = json
        .get("decoder_safetensors_root")
        .and_then(|v| v.as_str())
    {
        a.decoder_root = Some(r.to_string());
    }
    assert!(
        a.decoder_prefix.is_none() || a.decoder_root.is_none(),
        "config `{source_stem}`: decoder_safetensors_prefix (a wrapper around `model`) and \
         decoder_safetensors_root (a rename of `model`) are exclusive",
    );
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

/// The config's Mamba-2 `time_step_limit` when it can bind — mlx-lm `compute_dt`'s
/// `clip(softplus(dt + dt_bias), min, max)`. `softplus` is positive, so the clip is a no-op only
/// for `min <= 0` with no `max` (`null`: the `inf` JSON cannot spell); absent is mlx-lm's default
/// `(0, inf)`. Anything else — a finite `max`, a positive `min`, a malformed value — is returned.
fn binding_time_step_limit(json: &serde_json::Value) -> Option<&serde_json::Value> {
    let limit = json.get("time_step_limit")?;
    let unclipped = match limit.as_array().map(Vec::as_slice) {
        Some([min, max]) => min.as_f64().is_some_and(|m| m <= 0.0) && max.is_null(),
        _ => false,
    };
    (!unclipped).then_some(limit)
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
    let layer_sets =
        derive_layer_sets(json, &mut bounds).map_err(|reason| ConfigError::LayerPattern {
            path: source_path.to_path_buf(),
            reason,
        })?;
    if let Some(limit) = binding_time_step_limit(json) {
        return Err(ConfigError::TimeStepLimit {
            path: source_path.to_path_buf(),
            limit: limit.clone(),
        });
    }
    // A gated derived dim's flag, by the one rule the macros crate's build script reads it with
    // (`hf_config::flag_value`: the config's, else the arch's default, else 0), so the width this
    // model derives and the width its carrier's typed program states agree.
    for flag in crate::dim::DERIVED_DIMS
        .iter()
        .filter_map(|d| d.rule.gate_flag())
    {
        let v =
            crate::hf_config::flag_value(json, &spec.config_aliases, &spec.bound_defaults, flag);
        if v != 0 {
            bounds.insert(flag.to_string(), v);
        }
    }
    let mut derived = derive_implicit_bounds(&mut bounds);
    // Arch-declared bound defaults (zero-centered norms, sliding
    // cadence, router renorm, …): explicit config values always win.
    for (k, v) in &spec.bound_defaults {
        bounds.entry(k.clone()).or_insert(*v);
    }
    // Arch-declared `Params` schema (additive over the standard flat
    // harvest for decoder crates that need non-standard fields).
    let scheduled_layers = bounds.get("num_hidden_layers").copied();
    derived.extend(spec.eval_params(json, &mut bounds).map_err(|reason| {
        ConfigError::VisionDerivation {
            path: source_path.to_path_buf(),
            reason,
        }
    })?);
    // A schema that replaces the layer count leaves the block schedule another model's: an MTP
    // head's config is its target's verbatim, so its sets name the target's layers, not the head's.
    let own_layers = bounds.get("num_hidden_layers").copied() == scheduled_layers;
    let mut layer_sets = match own_layers {
        true => layer_sets,
        false => BTreeMap::new(),
    };
    // Every decoder's layer loop tests the layers a drafter beside it captures (`draft_capture`),
    // so every model has the set: empty without a drafter, and empty on another model's layers.
    let captured = (json.get(DRAFT_CAPTURE_LAYERS).and_then(|v| v.as_array()))
        .filter(|_| own_layers)
        .map(|a| a.iter().filter_map(|v| v.as_u64()).collect::<Vec<_>>())
        .unwrap_or_default();
    layer_sets.insert(DRAFT_CAPTURE_LAYERS.to_string(), LayerSet::of(captured));
    let mut scalars = extract_scalars(json);
    // Arch-declared scalar defaults (`SCALAR_DEFAULTS`): arch-constant
    // floats the checkpoint omits (Gemma-4's `attention_multiplier`).
    // Explicit config values always win.
    for (k, v) in &spec.scalar_defaults {
        scalars.entry(k.clone()).or_insert(*v);
    }
    let quantization = parse_quantization(json, source_stem, source_path)?;
    // An in-checkpoint head reads its target's config, whose `tie_word_embeddings` says where the
    // TARGET keeps its lm_head (Qwen3.5-0.8B: in its embedding table). The head's lm_head is a
    // slot of its own the target lends either way (`target.lent_weight_prefix`), never its embedding.
    let tie_word_embeddings = spec.checkpoint_of.is_empty()
        && parse_tie_word_embeddings(json)
            .or(spec.tie_default)
            .unwrap_or(false);
    // An arch that is a part of another model's checkpoint (an MTP head) registers under the
    // identity it declares, not the architecture its (the whole checkpoint's) config names.
    let (architectures, (drafts_for, published, target_experts)) =
        match spec.hf_architectures.is_empty() {
            true => (parse_architectures(json), (Vec::new(), None, 0)),
            false => (
                spec.hf_architectures.clone(),
                head_target(spec, json, source_stem, source_path)?,
            ),
        };
    // A head's verify budget prices the expert routes of the rows its target verifies.
    if target_experts > 0 {
        bounds.insert("target_num_experts".to_string(), target_experts);
    }
    let rope_scaling = extract_rope_scaling(json);
    let rope_scaling_hash = json.get("rope_scaling").map(hash_json_value);
    let mrope_section = extract_mrope_section(json);
    let arch = resolve_arch_spec(spec, json, source_stem);
    let expert_activation = ExpertActivation::of(json);

    derived.extend(apply_generic_bound_defaults(&mut bounds));
    let text = TextGeometry::of(&bounds, &scalars, source_path)?;

    Ok(ModelParams {
        name,
        source_stem: source_stem.to_string(),
        source_path: source_path.to_path_buf(),
        bounds,
        scalars,
        layer_sets,
        quantization,
        tie_word_embeddings,
        architectures,
        drafts_for,
        drafter_repo_infix: spec.drafter_repo_infix.clone(),
        checkpoint_prefix: spec.checkpoint_prefix.clone(),
        published,
        extra_tracked_paths,
        rope_scaling,
        rope_scaling_hash,
        mrope_section,
        arch,
        expert_activation,
        derived,
        text,
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
    let mut derived =
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
    let quantization = parse_quantization(json, source_stem, source_path)?;
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

    let expert_activation = ExpertActivation::of(json);
    derived.extend(apply_generic_bound_defaults(&mut bounds));
    // The decoder this tower feeds, as the wrapper's `text_config` states it — read through the
    // decoder path's own harvest (normalize, bounds, derivations, the arch's defaults), so the two
    // loaders state one geometry for one config.
    let text = {
        let text_json = normalize_hf_config(json, &spec.config_aliases);
        let mut text_bounds = extract_bounds(&text_json);
        derive_layer_sets(&text_json, &mut text_bounds).map_err(|reason| {
            ConfigError::LayerPattern {
                path: source_path.to_path_buf(),
                reason,
            }
        })?;
        derive_implicit_bounds(&mut text_bounds);
        let mut text_scalars = extract_scalars(&text_json);
        for (k, v) in &spec.scalar_defaults {
            text_scalars.entry(k.clone()).or_insert(*v);
        }
        TextGeometry::of(&text_bounds, &text_scalars, source_path)?
    };

    Ok(ModelParams {
        name,
        source_stem: source_stem.to_string(),
        source_path: source_path.to_path_buf(),
        bounds,
        scalars,
        layer_sets: BTreeMap::new(),
        quantization,
        tie_word_embeddings,
        architectures,
        drafts_for: Vec::new(),
        drafter_repo_infix: None,
        checkpoint_prefix: None,
        published: None,
        extra_tracked_paths,
        rope_scaling: None,
        rope_scaling_hash: None,
        mrope_section: None,
        arch,
        expert_activation,
        derived,
        text,
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

/// Load `configs/<arch>/arch.json` — THE arch's own declaration file.
///
/// This is the per-arch tier of the same JSON surface the
/// per-checkpoint `<stem>.overrides.json` files already use, and it
/// deliberately reuses their key names (`scale_dtype`,
/// `decoder_safetensors_prefix`, `vision_safetensors_layout`,
/// `vision_d_model_fingerprint`, `vision_patch_embed_flatten`, …) and
/// their extractors, so there is ONE vocabulary for "facts about an
/// arch that aren't derivable from a verbatim HF config.json" rather
/// than two. Precedence is unchanged in spirit: HF `config.json`
/// (per size) → `arch.json` (per arch) → `<stem>.overrides.json`
/// (per checkpoint), each winning over the one before it.
///
/// A missing file is not an error: an arch whose every fact IS
/// derivable from its config.json declares nothing, and its DSL file
/// stays a bare `#[forward] fn`.
pub fn load_arch_json(dir: &Path) -> Result<crate::arch_spec::DeclaredArchSpec, ConfigError> {
    let path = dir.join(ARCH_JSON);
    let mut spec = crate::arch_spec::DeclaredArchSpec::default();
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(spec);
    };
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|source| ConfigError::Json {
            path: path.clone(),
            source,
        })?;

    // Vision layout / fingerprint / patch-embed flatten: the SAME
    // three extractors `resolve_arch_spec` runs over the per-checkpoint
    // drift JSON, so the two tiers can never disagree about shape.
    spec.safetensors = extract_vision_layout(&json);
    spec.fingerprint = extract_vision_d_model_fingerprint(&json);
    spec.patch_embed_flatten = extract_vision_patch_embed_flatten(&json);

    // Scalars and strings. Enum-valued keys are validated here, at the
    // single point they enter the compiler — the same check
    // `resolve_arch_spec` applies to the drift tier.
    let as_str = |k: &str| json.get(k).and_then(|v| v.as_str()).map(str::to_string);
    spec.pos_embed_key = as_str("vision_pos_embed_key");
    spec.decoder_prefix = as_str("decoder_safetensors_prefix");
    spec.decoder_root = as_str("decoder_safetensors_root");
    if spec.decoder_prefix.is_some() && spec.decoder_root.is_some() {
        return Err(ConfigError::ArchJson {
            path: path.clone(),
            reason: "decoder_safetensors_prefix and decoder_safetensors_root are exclusive"
                .to_string(),
        });
    }
    spec.drafter_repo_infix = as_str("drafter_repo_infix");
    if let Some(v) = json.get("hf_architectures") {
        spec.hf_architectures = v
            .as_array()
            .and_then(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect())
            .ok_or_else(|| ConfigError::ArchJson {
                path: path.clone(),
                reason: "`hf_architectures` must be an array of strings".to_string(),
            })?;
    }
    spec.vision_norm_eps = json.get("vision_norm_eps").and_then(|v| v.as_f64());
    spec.tie_default = json.get("tie_default").and_then(|v| v.as_bool());
    spec.head_targets = str_str_pairs(&path, &json, "head_targets")?;
    // A variant is its config's stem (dense) or `<config>-<preset>`, as the variant loop of
    // `load_dir_mode` names it.
    let presets = declared_preset_names(&dir.join("quantizations.json"))?;
    let config_of = |variant: &str| {
        (spec.head_targets.iter()).find(|(config, _)| {
            config == variant || presets.iter().any(|p| format!("{config}-{p}") == variant)
        })
    };
    let mut published_as = Vec::new();
    for (variant, repo) in str_str_pairs(&path, &json, "published_as")? {
        let Some((config, target)) = config_of(&variant) else {
            return Err(ConfigError::ArchJson {
                path: path.clone(),
                reason: format!(
                    "`published_as` names `{variant}`, neither a `head_targets` config nor one at \
                     a preset its quantizations.json declares"
                ),
            });
        };
        let other = (published_as.iter())
            .filter_map(|p: &crate::arch_spec::PublishedAs| config_of(&p.variant))
            .find(|(c, t)| t == target && c != config);
        if let Some((other, _)) = other {
            return Err(ConfigError::ArchJson {
                path: path.clone(),
                reason: format!(
                    "`published_as` names `{config}` and `{other}`, both drafters of `{target}`: \
                     a target drafts with one by default"
                ),
            });
        }
        published_as.push(crate::arch_spec::PublishedAs {
            config: config.clone(),
            variant,
            repo,
        });
    }
    spec.published_as = published_as;
    spec.checkpoint_prefix = as_str("checkpoint_prefix");
    spec.reads_target_kv = json.get("reads_target_kv").and_then(|v| v.as_bool()) == Some(true);
    let strings = |key: &str| -> Result<Option<Vec<String>>, ConfigError> {
        crate::hf_config::string_list(&json, key).map_err(|reason| ConfigError::ArchJson {
            path: path.clone(),
            reason,
        })
    };
    spec.borrows = strings("borrows")?;
    spec.target_capture_layers = as_str("target_capture_layers");
    spec.top_level_weights = strings("top_level_weights")?.unwrap_or_default();
    // The facts the bounds read, by the one reader the macros crate's build script shares.
    let facts = crate::hf_config::arch_facts(&json).map_err(|reason| ConfigError::ArchJson {
        path: path.clone(),
        reason,
    })?;
    spec.checkpoint_of = facts.checkpoint_of;
    if spec.checkpoint_of.is_empty() != spec.checkpoint_prefix.is_none() {
        return Err(ConfigError::ArchJson {
            path: path.clone(),
            reason: "`checkpoint_of` and `checkpoint_prefix` declare one fact (a head inside its \
                     target's checkpoint): both or neither"
                .to_string(),
        });
    }

    if let Some(s) = as_str("vision_rope_style") {
        check_enum(
            &path,
            "vision_rope_style",
            &s,
            &["neox_hw", "interleaved_xy"],
        )?;
        spec.rope_style = Some(s);
    }
    if let Some(s) = as_str("vision_pos_emb_interp") {
        check_enum(&path, "vision_pos_emb_interp", &s, &["bilinear", "bicubic"])?;
        spec.pos_emb_interp = Some(s);
    }
    if let Some(s) = as_str("scale_dtype") {
        check_enum(&path, "scale_dtype", &s, &["f16", "bf16"])?;
        spec.scale_dtype = Some(s);
    }

    spec.bound_defaults = facts.bound_defaults;
    spec.scalar_defaults = str_f64_pairs(&path, &json, "scalar_defaults")?;
    spec.config_aliases = facts.config_aliases;
    spec.weight_leaf_renames = str_str_pairs(&path, &json, "weight_leaf_renames")?;
    spec.params = facts.params;

    // Typos must not silently no-op — the failure mode the token
    // surface was careful about, kept here.
    reject_unknown_keys(&path, &json)?;
    Ok(spec)
}

/// Reserved (non-model) filename for the per-arch declaration file.
pub(crate) const ARCH_JSON: &str = "arch.json";

/// What the heads compiled beside a target config bake into it (`load_dir_mode`'s `head_facts`).
#[derive(Default)]
struct HeadFacts {
    /// The most drafts one of them carries a sequence.
    drafts: Option<u64>,
    /// The target layers whose outputs one reads.
    capture_layers: Option<Vec<u64>>,
}

/// Every key `load_arch_json` understands. Anything else in an
/// `arch.json` is a typo or a stale key, and is rejected.
const ARCH_JSON_KEYS: &[&str] = &[
    "vision_safetensors_layout",
    "vision_d_model_fingerprint",
    "vision_patch_embed_flatten",
    "vision_pos_embed_key",
    "vision_rope_style",
    "vision_pos_emb_interp",
    "vision_norm_eps",
    "decoder_safetensors_prefix",
    "decoder_safetensors_root",
    "hf_architectures",
    "drafter_repo_infix",
    "head_targets",
    "published_as",
    "checkpoint_of",
    "checkpoint_prefix",
    "reads_target_kv",
    "borrows",
    "target_capture_layers",
    "top_level_weights",
    "scale_dtype",
    "tie_default",
    "bound_defaults",
    "scalar_defaults",
    "config_aliases",
    "weight_leaf_renames",
    "params",
];

fn reject_unknown_keys(path: &Path, json: &serde_json::Value) -> Result<(), ConfigError> {
    let Some(obj) = json.as_object() else {
        return Err(ConfigError::ArchJson {
            path: path.to_path_buf(),
            reason: "top level must be a JSON object".to_string(),
        });
    };
    for k in obj.keys() {
        // `_comment` is the established escape hatch in the existing
        // `.overrides.json` files; keep it valid here too.
        if k == "_comment" || ARCH_JSON_KEYS.contains(&k.as_str()) {
            continue;
        }
        return Err(ConfigError::ArchJson {
            path: path.to_path_buf(),
            reason: format!("unknown key `{k}` (known: {})", ARCH_JSON_KEYS.join(", ")),
        });
    }
    Ok(())
}

fn check_enum(path: &Path, key: &str, got: &str, allowed: &[&str]) -> Result<(), ConfigError> {
    if allowed.contains(&got) {
        return Ok(());
    }
    Err(ConfigError::ArchJson {
        path: path.to_path_buf(),
        reason: format!("unknown {key} `{got}` ({})", allowed.join("|")),
    })
}

/// `{"a": 1, "b": 2}` → sorted `[(a,1), (b,2)]`. Sorted so the emit is
/// order-stable regardless of how the file was written.
fn str_f64_pairs(
    path: &Path,
    json: &serde_json::Value,
    key: &str,
) -> Result<Vec<(String, f64)>, ConfigError> {
    let Some(obj) = json.get(key) else {
        return Ok(Vec::new());
    };
    let obj = obj.as_object().ok_or_else(|| ConfigError::ArchJson {
        path: path.to_path_buf(),
        reason: format!("`{key}` must be an object"),
    })?;
    let mut out: Vec<(String, f64)> = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let n = v.as_f64().ok_or_else(|| ConfigError::ArchJson {
            path: path.to_path_buf(),
            reason: format!("`{key}.{k}` must be a number"),
        })?;
        out.push((k.clone(), n));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn str_str_pairs(
    path: &Path,
    json: &serde_json::Value,
    key: &str,
) -> Result<Vec<(String, String)>, ConfigError> {
    crate::hf_config::object_str_pairs(json, key).map_err(|reason| ConfigError::ArchJson {
        path: path.to_path_buf(),
        reason,
    })
}

/// The `params` schema: an ORDERED array, because `expr` fields read
/// bounds that earlier fields defined (`eval_params` walks in
/// declaration order). A JSON object would not preserve that order,
/// so this is a list of `{"name": …, <source>}` entries with exactly
/// one source key each.
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
            // HF signature parity: `YarnRotaryEmbedding.__init__(mscale=1.0,
            // mscale_all_dim=1.0)`. gpt-oss omits the pair — with the 1.0/1.0
            // defaults the mscale ratio is 1.0 (plain interpolation, no
            // attention temperature), which is what mlx-lm/vLLM compute for
            // it. Every in-tree yarn config carries an explicit pair, so this
            // default is dead code for them.
            let mscale = rs.get("mscale").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mscale_all_dim = rs
                .get("mscale_all_dim")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
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

/// One block of a hybrid per-layer schedule — what mlx-lm's
/// `nemotron_h.ModelArgs._block_type_to_char` normalizes every spelling
/// to (`M` / `*` / `E` / `-`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum HybridBlock {
    Mamba,
    Attention,
    Moe,
    Mlp,
}

impl HybridBlock {
    const ALL: [Self; 4] = [Self::Mamba, Self::Attention, Self::Moe, Self::Mlp];

    /// A `hybrid_override_pattern` character.
    fn from_pattern_char(c: char) -> Option<Self> {
        match c {
            'M' => Some(Self::Mamba),
            '*' => Some(Self::Attention),
            'E' => Some(Self::Moe),
            '-' => Some(Self::Mlp),
            _ => None,
        }
    }

    /// A transformers >= 5.13 `layers_block_type` / `layer_types` entry,
    /// legacy spellings included.
    fn from_block_type(s: &str) -> Option<Self> {
        match s {
            "linear_attention" | "mamba" | "conv" => Some(Self::Mamba),
            "full_attention" | "attention" => Some(Self::Attention),
            "moe" => Some(Self::Moe),
            "mlp" => Some(Self::Mlp),
            _ => None,
        }
    }

    /// The layer-set name a DSL predicate reads (`if layer in mamba_layers:`).
    pub fn set_name(self) -> &'static str {
        match self {
            Self::Mamba => "mamba_layers",
            Self::Attention => "attention_layers",
            Self::Moe => "moe_layers",
            Self::Mlp => "mlp_layers",
        }
    }
}

/// A set of layer indices derived from the config's block schedule,
/// read by `if layer in <name>:`. Sorted and deduplicated by
/// construction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerSet(Vec<u64>);

impl LayerSet {
    /// The set of `members`, sorted and deduplicated.
    pub fn of(members: impl IntoIterator<Item = u64>) -> Self {
        let mut v: Vec<u64> = members.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        Self(v)
    }

    pub fn members(&self) -> &[u64] {
        &self.0
    }
}

/// Derive the per-block layer sets (`mamba_layers`, `attention_layers`,
/// `moe_layers`, `mlp_layers`) from the config's per-layer block
/// schedule, the way mlx-lm's `nemotron_h.ModelArgs.__post_init__`
/// normalizes it: `hybrid_override_pattern` (a string, or a list of
/// one-character strings) wins; otherwise `layers_block_type`, then
/// `layer_types`, mapped through [`HybridBlock::from_block_type`].
///
/// `hybrid_override_pattern` is unambiguous, so an unknown character is
/// an error. `layer_types` is shared with non-hybrid schedules (Gemma's
/// `sliding_attention`), so a list naming any non-block entry derives
/// nothing. A schedule's length IS the layer count: it fills an absent
/// `num_hidden_layers` and must equal a present one. All four sets are
/// inserted once a schedule exists, so an absent block kind reads as the
/// empty set rather than an unknown name.
pub(crate) fn derive_layer_sets(
    json: &serde_json::Value,
    bounds: &mut BTreeMap<String, u64>,
) -> Result<BTreeMap<String, LayerSet>, String> {
    let pattern_block = |c: char| {
        HybridBlock::from_pattern_char(c)
            .ok_or_else(|| format!("hybrid_override_pattern: unknown block `{c}`"))
    };
    let schedule: Vec<HybridBlock> = match json.get("hybrid_override_pattern") {
        Some(serde_json::Value::String(p)) => {
            p.chars().map(pattern_block).collect::<Result<_, _>>()?
        }
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|v| {
                let s = v.as_str().unwrap_or_default();
                let mut chars = s.chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) => pattern_block(c),
                    _ => Err(format!(
                        "hybrid_override_pattern: `{v}` is not one block character"
                    )),
                }
            })
            .collect::<Result<_, _>>()?,
        Some(other) => {
            return Err(format!(
                "hybrid_override_pattern: expected a string or a list, got {other}"
            ));
        }
        None => {
            let Some(types) = ["layers_block_type", "layer_types"]
                .iter()
                .find_map(|k| json.get(k).and_then(|v| v.as_array()))
            else {
                return Ok(BTreeMap::new());
            };
            let Some(schedule) = types
                .iter()
                .map(|t| t.as_str().and_then(HybridBlock::from_block_type))
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(BTreeMap::new());
            };
            schedule
        }
    };
    if schedule.is_empty() {
        return Err("the block schedule is empty".to_string());
    }
    let n = schedule.len() as u64;
    match bounds.get("num_hidden_layers") {
        Some(&declared) if declared != n => {
            return Err(format!(
                "the block schedule names {n} layers but num_hidden_layers is {declared}"
            ));
        }
        Some(_) => {}
        None => {
            bounds.insert("num_hidden_layers".to_string(), n);
        }
    }
    Ok(HybridBlock::ALL
        .into_iter()
        .map(|block| {
            let members = (0..n).filter(|&l| schedule[l as usize] == block).collect();
            (block.set_name().to_string(), LayerSet(members))
        })
        .collect())
}

/// Returns each bound it derived, as the dim it derived it from.
fn derive_implicit_bounds(bounds: &mut BTreeMap<String, u64>) -> BTreeMap<String, crate::dim::Dim> {
    use crate::dim::Dim;
    let mut derived = BTreeMap::new();
    let bound = |name: &str| Dim::Bound(name.to_string());
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
        derived.insert(
            "head_dim".to_string(),
            crate::dim::canonical_add(vec![bound("qk_nope_head_dim"), bound("qk_rope_head_dim")]),
        );
    }
    if !bounds.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (bounds.get("hidden_size"), bounds.get("num_attention_heads"))
        && heads != 0
        && hidden.is_multiple_of(heads)
    {
        bounds.insert("head_dim".to_string(), hidden / heads);
        derived.insert(
            "head_dim".to_string(),
            Dim::Div(
                Box::new(bound("hidden_size")),
                Box::new(bound("num_attention_heads")),
            ),
        );
    }
    if !bounds.contains_key("num_key_value_heads")
        && let Some(&heads) = bounds.get("num_attention_heads")
    {
        bounds.insert("num_key_value_heads".to_string(), heads);
        derived.insert(
            "num_key_value_heads".to_string(),
            bound("num_attention_heads"),
        );
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
    crate::dim::derive_dims(bounds, crate::dim::DeriveStage::Implicit, &mut derived);
    derived
}

/// Arch-AGNOSTIC bound defaults applied to every parsed config
/// (decoder and vision paths alike). Per-arch modeling defaults
/// (zero-centered norms, sliding cadence, tie defaults, …) are NOT
/// here — they are `bound_defaults` / `tie_default` declarations on
/// the owning crate's `#[forward]` attribute (see [`crate::arch_spec`]).
///
/// Returns each bound it derived, as the dim it derived it from.
fn apply_generic_bound_defaults(
    bounds: &mut BTreeMap<String, u64>,
) -> BTreeMap<String, crate::dim::Dim> {
    use crate::dim::Dim;
    let mut derived = BTreeMap::new();
    let bound = |name: &str| Dim::Bound(name.to_string());
    // A drafter reading its target's hidden states and embedding reads them at its target's
    // residual width (Gemma 4's assistant declares it); a head of its target's width omits it.
    if !bounds.contains_key("backbone_hidden_size")
        && let Some(&h) = bounds.get("hidden_size")
    {
        bounds.insert("backbone_hidden_size".to_string(), h);
        derived.insert("backbone_hidden_size".to_string(), bound("hidden_size"));
    }
    // Per-class attention geometry defaults: uniform-geometry arches
    // never declare global_* keys, but the shape sigs anchor the
    // `attention()` (global-class) tile to `global_head_dim` /
    // `num_global_key_value_heads` unconditionally — default them to
    // the base values so every existing arch resolves identically.
    if !bounds.contains_key("global_head_dim")
        && let Some(&hd) = bounds.get("head_dim")
    {
        bounds.insert("global_head_dim".to_string(), hd);
        derived.insert("global_head_dim".to_string(), bound("head_dim"));
    }
    if !bounds.contains_key("num_global_key_value_heads")
        && let Some(&kv) = bounds.get("num_key_value_heads")
    {
        bounds.insert("num_global_key_value_heads".to_string(), kv);
        derived.insert(
            "num_global_key_value_heads".to_string(),
            bound("num_key_value_heads"),
        );
    }
    // Global-attention-class projection widths (Gemma-4's dual-class
    // attention: the global layers run a wider head with their own KV
    // head count). `q_global_dim` is q_proj_global's output width;
    // `k_global_dim` the single shared global K/V projection's width.
    // Derived AFTER the `global_head_dim` / `num_global_key_value_heads`
    // defaults above so uniform-geometry arches (where those fall back
    // to `head_dim` / `num_key_value_heads`) resolve these identically
    // to `attn_q_dim` / `kv_dim`. Explicit config values always win.
    crate::dim::derive_dims(bounds, crate::dim::DeriveStage::AfterDefaults, &mut derived);
    derived
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

    /// The presets of an arch's `quantizations.json` the backend this test binary is built
    /// for can lower (`quant_preset_active`): metal claims the MLX affine presets and
    /// NVFP4; any other backend claims every preset.
    #[cfg(feature = "metal")]
    const LLAMA_PRESETS: &[&str] = &["mlx-affine-b4-g64", "mlx-affine-b4-g64-qembed", "nvfp4"];
    #[cfg(not(feature = "metal"))]
    const LLAMA_PRESETS: &[&str] = &[
        "awq-gemm",
        "bnb-nf4-dq",
        "ct-int4-sym",
        "fp8-dynamic-per-tensor",
        "fp8-static-per-tensor",
        "ggml",
        "gptq-sym",
        "gptq-sym-desc_act",
        "mlx-affine-b4-g64",
        "mlx-affine-b4-g64-qembed",
        "nvfp4",
    ];
    #[cfg(feature = "metal")]
    const QWEN2_PRESETS: &[&str] = &["mlx-affine-b4-g64", "mlx-affine-b4-g64-qembed"];
    #[cfg(not(feature = "metal"))]
    const QWEN2_PRESETS: &[&str] = &[
        "awq-gemm",
        "bnb-nf4-dq",
        "ct-int4-sym",
        "fp8-dynamic-per-tensor",
        "fp8-static-per-tensor",
        "ggml",
        "gptq-sym",
        "mlx-affine-b4-g64",
        "mlx-affine-b4-g64-qembed",
    ];

    /// What `dir`'s configs load as, under a build of every one of them: dense-only (no
    /// preset forwarded), one variant per base; and with every preset the backend claims,
    /// each base's variants exactly `<base>-<preset>` for `presets` — a selected preset
    /// replaces the dense variant, it doesn't add to it. Returns the dense variants.
    fn every_variant_of(dir: &Path, presets: &[&str]) -> Vec<ModelParams> {
        let spec =
            load_arch_json(dir).unwrap_or_else(|e| panic!("{} arch.json: {e}", dir.display()));
        let dense = load_dir(dir, &spec, &Scope::all().with_quants(Vec::<String>::new()))
            .unwrap_or_else(|e| panic!("load {} dense: {e}", dir.display()));
        let mut bases: Vec<&str> = dense.iter().map(|c| c.source_stem.as_str()).collect();
        bases.sort();
        let mut files: Vec<String> = model_configs(dir)
            .expect("list configs")
            .iter()
            .map(|p| stem_of(p).expect("stem"))
            .collect();
        files.sort();
        assert_eq!(bases, files, "every config loads as one dense variant");

        let quantized = load_dir(dir, &spec, &Scope::all())
            .unwrap_or_else(|e| panic!("load {} with every preset: {e}", dir.display()));
        let mut got: Vec<&str> = quantized.iter().map(|c| c.source_stem.as_str()).collect();
        got.sort();
        let mut want: Vec<String> = bases
            .iter()
            .flat_map(|b| presets.iter().map(move |p| format!("{b}-{p}")))
            .collect();
        want.sort();
        assert_eq!(got, want, "every base × every preset this backend claims");
        dense
    }

    #[test]
    fn load_real_llama_configs() {
        let configs = every_variant_of(&repo_model_archs("llama"), LLAMA_PRESETS);

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
        let configs = every_variant_of(&repo_model_archs("qwen2"), QWEN2_PRESETS);

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
        let configs = load_dir_vision(
            &dir,
            &load_arch_json(&dir).expect("qwen2-vl arch.json"),
            &Scope::all(),
        )
        .expect("load qwen2-vl vision configs");
        let mut stems: Vec<&str> = configs.iter().map(|c| c.source_stem.as_str()).collect();
        stems.sort();
        assert_eq!(
            stems,
            [
                "qwen2-vl-2b-instruct",
                "qwen2-vl-2b-instruct-parity-tiny",
                "qwen2-vl-2b-mlx",
                "qwen2-vl-72b-instruct",
                "qwen2-vl-7b-instruct",
            ],
            "every Qwen2-VL vision config loads"
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
        let configs = load_dir(&dir, &load_arch_json(&dir).unwrap(), &Scope::all()).unwrap();
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

    fn layer_sets(json: serde_json::Value) -> Result<BTreeMap<String, Vec<u64>>, String> {
        derive_layer_sets(&json, &mut BTreeMap::new()).map(|sets| {
            sets.into_iter()
                .map(|(k, v)| (k, v.members().to_vec()))
                .collect()
        })
    }

    /// Every spelling of a hybrid block schedule derives the same four
    /// sets: the pattern string, the list-of-characters form, and the
    /// transformers >= 5.13 `layers_block_type` / `layer_types` names
    /// (legacy `mamba` / `attention` included).
    #[test]
    fn layer_sets_derive_from_every_schedule_spelling() {
        let want: BTreeMap<String, Vec<u64>> = [
            ("mamba_layers", vec![0, 2, 5]),
            ("attention_layers", vec![3]),
            ("moe_layers", vec![1, 4]),
            ("mlp_layers", vec![6]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let types = [
            "mamba",
            "moe",
            "linear_attention",
            "full_attention",
            "moe",
            "conv",
            "mlp",
        ];
        for json in [
            serde_json::json!({ "hybrid_override_pattern": "MEM*EM-" }),
            serde_json::json!({ "hybrid_override_pattern": ["M", "E", "M", "*", "E", "M", "-"] }),
            serde_json::json!({ "layers_block_type": types }),
            serde_json::json!({ "layer_types": types }),
        ] {
            assert_eq!(layer_sets(json).as_ref(), Ok(&want));
        }
        // The pattern wins over a block-type list.
        let json = serde_json::json!({
            "hybrid_override_pattern": "MEM*EM-",
            "layers_block_type": ["moe"],
        });
        assert_eq!(layer_sets(json), Ok(want));
    }

    /// A schedule's length is the layer count: it fills an absent
    /// `num_hidden_layers` and must agree with a present one. An unknown
    /// pattern character is an error; a `layer_types` list naming a
    /// non-block entry (Gemma's `sliding_attention`) derives nothing.
    #[test]
    fn layer_sets_check_the_layer_count_and_the_vocabulary() {
        let mut bounds = BTreeMap::new();
        let json = serde_json::json!({ "hybrid_override_pattern": "M*E" });
        derive_layer_sets(&json, &mut bounds).expect("derive");
        assert_eq!(bounds.get("num_hidden_layers"), Some(&3));
        bounds.insert("num_hidden_layers".to_string(), 4);
        assert!(derive_layer_sets(&json, &mut bounds).is_err());
        assert!(layer_sets(serde_json::json!({ "hybrid_override_pattern": "M*X" })).is_err());
        assert!(layer_sets(serde_json::json!({ "hybrid_override_pattern": ["ME"] })).is_err());
        assert!(layer_sets(serde_json::json!({ "hybrid_override_pattern": "" })).is_err());
        let gemma = serde_json::json!({ "layer_types": ["sliding_attention", "full_attention"] });
        assert_eq!(layer_sets(gemma), Ok(BTreeMap::new()));
        assert_eq!(layer_sets(serde_json::json!({})), Ok(BTreeMap::new()));
    }

    /// A Mamba-2 `time_step_limit` that can clip `dt` is a compile-time
    /// error: the checked-in Nano config with a binding limit refuses, the
    /// unclipped spellings (absent, `[0.0, null]`, a non-positive minimum)
    /// parse. mlx-community's Nano ships `[0.0, Infinity]`, which is the
    /// unclipped default but not JSON — the checked-in base omits it.
    #[test]
    fn a_binding_time_step_limit_is_a_compile_time_error() {
        let dir = repo_model_archs("nemotron-h");
        let spec = load_arch_json(&dir).expect("arch.json");
        let stem = "nemotron-3-nano-30b-a3b";
        let path = dir.join(format!("{stem}.json"));
        let base: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
        let parse = |limit: Option<serde_json::Value>| {
            let mut json = base.clone();
            if let Some(limit) = limit {
                json["time_step_limit"] = limit;
            }
            model_params_from_json(&json, stem, &path, Vec::new(), &spec)
        };
        assert!(parse(None).is_ok());
        assert!(parse(Some(serde_json::json!([0.0, null]))).is_ok());
        assert!(parse(Some(serde_json::json!([-1.0, null]))).is_ok());
        for binding in [
            serde_json::json!([0.0, 100.0]),
            serde_json::json!([0.001, null]),
            serde_json::json!([0.001, 100.0]),
            serde_json::json!(0.1),
        ] {
            assert!(
                matches!(
                    parse(Some(binding.clone())),
                    Err(ConfigError::TimeStepLimit { .. })
                ),
                "{binding} must refuse"
            );
        }
    }

    /// The checked-in Nemotron-3 configs, through the full decoder parse
    /// with their arch.json: the irregular pattern's sets, the
    /// `backbone` decoder root, the `layer_norm_epsilon` alias, and the
    /// Mamba-2 widths the arch's `params` derive.
    #[test]
    fn nemotron_h_configs_derive_their_schedule_and_widths() {
        for (arch, stem, layers, attn, mamba_heads) in [
            ("nemotron-h", "nemotron-3-nano-30b-a3b", 52, 6, 64),
            (
                "nemotron-h-latent",
                "nemotron-3-super-120b-a12b",
                88,
                8,
                128,
            ),
        ] {
            let dir = repo_model_archs(arch);
            let spec = load_arch_json(&dir).expect("arch.json");
            let path = dir.join(format!("{stem}.json"));
            let json: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
            let mp = model_params_from_json(&json, stem, &path, Vec::new(), &spec).expect("parse");
            let set = |name: &str| mp.layer_sets.get(name).expect(name).members().to_vec();
            let (mamba, attention, moe, mlp) = (
                set("mamba_layers"),
                set("attention_layers"),
                set("moe_layers"),
                set("mlp_layers"),
            );
            assert_eq!(mp.bounds["num_hidden_layers"], layers);
            assert_eq!(attention.len(), attn, "{stem}");
            assert_eq!(mamba.len() + attention.len() + moe.len(), layers as usize);
            assert!(mlp.is_empty());
            assert_eq!(
                (mamba[0], moe[0]),
                (0, 1),
                "{stem}: the schedule opens `ME`"
            );
            assert_eq!(
                *moe.last().unwrap(),
                layers - 1,
                "{stem}: and closes on `E`"
            );
            assert_eq!(mp.arch.decoder_root(), "backbone");
            assert_eq!(mp.scalars.get("rms_norm_eps"), Some(&1e-5));
            assert_eq!(
                mp.expert_activation,
                ExpertActivation::Relu2,
                "{stem}: `mlp_hidden_act` declares the ungated relu2 expert body"
            );
            let (h, p, g, s) = (mamba_heads, 64, 8, 128);
            assert_eq!(mp.bounds["mamba_d_inner"], h * p);
            assert_eq!(mp.bounds["mamba_conv_dim"], h * p + 2 * g * s);
            assert_eq!(mp.bounds["mamba_in_proj_dim"], 2 * h * p + 2 * g * s + h);
        }
        // Super: in_proj 2·8192 + 2·8·128 + 128, conv_dim 10240 (brief).
        let dir = repo_model_archs("nemotron-h-latent");
        let spec = load_arch_json(&dir).expect("arch.json");
        let path = dir.join("nemotron-3-super-120b-a12b.json");
        let json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
        let mp = model_params_from_json(&json, "s", &path, Vec::new(), &spec).expect("parse");
        assert_eq!(mp.bounds["mamba_conv_dim"], 10240);
        assert_eq!(mp.bounds["mamba_in_proj_dim"], 18560);
    }

    /// A vision tower's `expr` params are derivations of the model's, as a decoder's `head_dim`
    /// default is: qwen2.5-vl's `vision_head_dim` is the exact quotient of its embedding over its
    /// heads, which `__dims` spells so rustc checks the division.
    #[test]
    fn a_vision_expr_param_is_recorded_as_its_derivation() {
        use crate::dim::Dim;
        let bound = |b: &str| Box::new(Dim::Bound(b.to_string()));
        let path = repo_model_archs("qwen2-5-vl").join("qwen2.5-vl-3b-instruct.json");
        let mp = load_file_declared_vision(&path).expect("parse");
        assert_eq!(
            mp.derived.get("vision_head_dim"),
            Some(&Dim::Div(
                bound("vision_embed_dim"),
                bound("vision_num_heads")
            ))
        );
        assert_eq!(
            crate::shape::eval_closed_dim(&mp.derived["vision_head_dim"], &mp.bounds),
            Some(mp.bounds["vision_head_dim"])
        );
    }

    /// An arch compiles under its `arch-<name>` feature and a backend — the ONE law, no
    /// capability list. The arch name comes from `configs/` (the tree is the list), never
    /// a literal, so the test cannot shadow it.
    #[test]
    fn an_arch_compiles_with_its_feature_and_a_backend_of_its_own() {
        let configs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/arch/configs");
        let arch = arch_dirs(&configs)
            .expect("walk configs/")
            .iter()
            .map(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .expect("dir name")
                    .to_string()
            })
            .next()
            .expect("at least one arch dir");
        let feature = format!("arch-{arch}");
        let with = |backend: &'static str| {
            let f = feature.clone();
            move |feat: &str| feat == backend || feat == f
        };
        assert!(arch_compiled_with(&arch, with("metal")));
        assert!(arch_compiled_with(&arch, with("spyre")));
        // Both backends off: not compiled, whatever the arch feature.
        let neither = {
            let f = feature.clone();
            move |feat: &str| feat == f
        };
        assert!(!arch_compiled_with(&arch, neither));
        // Its arch feature off: not compiled, whatever the backend.
        let no_arch_feature = |feat: &str| feat == "metal";
        assert!(
            !arch_compiled_with(&arch, no_arch_feature),
            "its arch feature is off"
        );
    }

    /// The tree law: EVERY arch dir under `configs/` compiles under `spyre` when its
    /// `arch-<name>` feature is on — the build script's target filter admits it, and what
    /// happens next (lowering or a named refusal) is the lowering's verdict. No name is
    /// written here; the tree is the list.
    #[test]
    fn every_arch_in_the_tree_compiles_under_spyre_with_its_feature_on() {
        let configs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/arch/configs");
        for dir in arch_dirs(&configs).expect("walk configs/") {
            let arch = dir.file_name().and_then(|s| s.to_str()).expect("dir name");
            let feature = format!("arch-{arch}");
            let spyre = move |feat: &str| feat == "spyre" || feat == feature;
            assert!(
                arch_compiled_with(arch, spyre),
                "{arch} compiles under spyre with its feature on"
            );
        }
    }

    /// Qwen3.6's MTP head compiles on its target's config verbatim, whose block schedule names the
    /// target's 40 layers; the head's schema makes it 1. The target keeps its sets; the head gets
    /// none (a set naming layer 39 would have its fingerprint probe a layer it does not have) but
    /// the capture set every decoder's layer loop tests, empty: its target's drafter captures the
    /// target's layers, not the head's.
    #[test]
    fn a_schema_replacing_the_layer_count_leaves_no_layer_sets() {
        let target_path = repo_model_archs("qwen3-5-moe").join("qwen3.6-35b-a3b.json");
        let mut json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&target_path).expect("read")).expect("json");
        // What the directory walk writes into a target's config with a drafter beside it.
        json[DRAFT_CAPTURE_LAYERS] = serde_json::json!([1, 10, 19, 28, 37]);
        let parse = |arch: &str| {
            let spec = load_arch_json(&repo_model_archs(arch)).expect("arch.json");
            model_params_from_json(&json, "qwen3.6-35b-a3b", &target_path, Vec::new(), &spec)
                .expect("parse")
        };
        let target = parse("qwen3-5-moe");
        assert_eq!(target.bounds["num_hidden_layers"], 40);
        let attention = target.layer_sets["attention_layers"].members();
        assert_eq!(attention.last(), Some(&39));
        let captured = target.layer_sets[DRAFT_CAPTURE_LAYERS].members();
        assert_eq!(captured, [1, 10, 19, 28, 37]);
        let head = parse("qwen3-5-mtp");
        assert_eq!(head.bounds["num_hidden_layers"], 1);
        let none = BTreeMap::from([(DRAFT_CAPTURE_LAYERS.to_string(), LayerSet::default())]);
        assert_eq!(head.layer_sets, none);
        assert_eq!(
            head.checkpoint_prefix.as_deref(),
            Some("language_model.mtp")
        );
    }

    /// Qwen3.5-0.8B ties its lm_head to its embedding; its dense MTP head, read off the same
    /// config, does not: the target lends the head both apart, so every variant of the head loads
    /// them from the same places.
    #[test]
    fn a_tied_target_lends_its_head_an_untied_lm_head() {
        let target_path = repo_model_archs("qwen3-5").join("qwen3.5-0.8b.json");
        let (_, json) = read_json_file(&target_path).expect("json");
        let parse = |arch: &str| {
            let spec = load_arch_json(&repo_model_archs(arch)).expect("arch.json");
            model_params_from_json(&json, "qwen3.5-0.8b", &target_path, Vec::new(), &spec)
                .expect("parse")
        };
        assert!(parse("qwen3-5").tie_word_embeddings);
        assert!(!parse("qwen3-5-dense-mtp").tie_word_embeddings);
    }

    /// The MTP head compiles on exactly the configs of `checkpoint_of` that declare its layers:
    /// every Qwen3.5/3.6 MoE checkpoint, the 122B's among them; a MoE checkpoint of another family
    /// declares none.
    #[test]
    fn the_mtp_head_compiles_on_the_moe_configs_that_carry_it() {
        let head = load_arch_json(&repo_model_archs("qwen3-5-mtp")).expect("arch.json");
        assert_eq!(head.checkpoint_of, ["qwen3-5-moe"]);
        let carries = |arch: &str| -> Vec<(String, bool)> {
            let configs = model_configs(&repo_model_archs(arch)).expect("configs");
            let mut carried: Vec<_> = configs
                .iter()
                .map(|p| {
                    let (_, json) = read_json_file(p).expect("json");
                    (stem_of(p).expect("stem"), carries_head(&head, &json))
                })
                .collect();
            carried.sort();
            carried
        };
        for (stem, carried) in carries("qwen3-5-moe") {
            assert!(carried, "{stem} declares its MTP layers");
        }
        assert!(
            carries("qwen3-5-moe")
                .iter()
                .any(|(s, _)| s == "qwen3.5-122b-a10b")
        );
        for (stem, carried) in carries("qwen3-moe") {
            assert!(!carried, "{stem} declares no MTP layers");
        }
        assert!(
            model_configs(&repo_model_archs("qwen3-5-mtp"))
                .expect("configs")
                .is_empty()
        );
    }

    /// Gemma 4 26B-A4B's assistants draft for their target's architecture in every variant — a
    /// quant variant's stem carries its preset, so its target is its base config's. Google's is
    /// the target's default drafter: its MLX 4-bit g32 variant names the repo it loads from, with
    /// its target's shape, which Gemma 4 31B (the same architecture) does not share; its other
    /// variants, which that repo does not load into, name none. The QAT one names none: only
    /// `spec/mtp` compiles it.
    #[test]
    fn the_gemma_assistant_is_its_targets_default_drafter_at_mlx_4_bit_g32() {
        let dir = repo_model_archs("gemma4-assistant");
        let spec = load_arch_json(&dir).expect("arch.json");
        let google = "gemma-4-26b-a4b-it-assistant";
        let by_default: Vec<_> = compiled_head_targets(&spec, false)
            .map(|(stem, _)| stem.as_str())
            .collect();
        assert_eq!(by_default, [google]);
        assert_eq!(
            compiled_head_targets(&spec, true).count(),
            spec.head_targets.len()
        );
        let parse = |base: &str, stem: &str| {
            let path = dir.join(format!("{base}.json"));
            let (_, mut json) = read_json_file(&path).expect("json");
            apply_overrides_file(&dir, base, &mut json, &mut Vec::new()).expect("overrides");
            model_params_from_json(&json, stem, &path, Vec::new(), &spec).expect("parse")
        };
        let g32 = format!("{google}-mlx-affine-b4-g32");
        let g64 = format!("{google}-mlx-affine-b4-g64");
        let qat = "gemma-4-26b-a4b-it-qat-q4_0-unquantized-assistant";
        let qat_g64 = format!("{qat}-mlx-affine-b4-g64");
        let variants = [
            (google, google),
            (google, &g32),
            (google, &g64),
            (qat, &qat_g64),
        ];
        for (base, stem) in variants {
            let head = parse(base, stem);
            assert_eq!(
                head.drafts_for,
                ["Gemma4ForConditionalGeneration"],
                "{stem}"
            );
            let published = (stem == g32).then(|| PublishedDrafter {
                repo: "guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit".to_string(),
                target_hidden_size: 2816,
                target_num_hidden_layers: 30,
            });
            assert_eq!(head.published, published, "{stem}");
        }
        let (_, big) =
            read_json_file(&repo_model_archs("gemma4").join("gemma-4-31b-it.json")).expect("json");
        assert_eq!(big["architectures"][0], "Gemma4ForConditionalGeneration");
        assert_ne!(big["text_config"]["hidden_size"], 2816);
    }

    /// The parity-tiny configs keep every property the carrier gate
    /// needs: all three block kinds, an aperiodic schedule, grouped
    /// B/C (`n_groups > 1`) shared by several heads each, GQA, top-k > 1,
    /// and (Super) a latent width below the hidden width.
    #[test]
    fn nemotron_h_parity_tiny_configs_exercise_every_path() {
        for (arch, stem, latent) in [
            ("nemotron-h", "nemotron-3-nano-30b-a3b-parity-tiny", false),
            (
                "nemotron-h-latent",
                "nemotron-3-super-120b-a12b-parity-tiny",
                true,
            ),
        ] {
            let dir = repo_model_archs(arch);
            let spec = load_arch_json(&dir).expect("arch.json");
            let path = dir.join(format!("{stem}.json"));
            let json: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
            // Load-or-refuse: the parity-tiny ships a mixed-bit affine quantization,
            // so under a backend that can't lower it the refusal is the outcome.
            let Some(mp) = assert_load_or_refuse_by_name(
                stem,
                model_params_from_json(&json, stem, &path, Vec::new(), &spec),
            ) else {
                continue;
            };
            let b = |k: &str| mp.bounds[k];
            let set = |name: &str| mp.layer_sets[name].members().to_vec();
            for kind in ["mamba_layers", "attention_layers", "moe_layers"] {
                assert!(!set(kind).is_empty(), "{stem}: no {kind}");
            }
            let n = b("num_hidden_layers") as usize;
            let schedule: Vec<&str> = (0..n as u64)
                .map(|l| {
                    [
                        "mamba_layers",
                        "attention_layers",
                        "moe_layers",
                        "mlp_layers",
                    ]
                    .into_iter()
                    .find(|k| mp.layer_sets[*k].members().contains(&l))
                    .expect("every layer has a block")
                })
                .collect();
            assert!(
                (1..n).all(|p| (0..n).any(|l| schedule[l] != schedule[l % p])),
                "{stem}: schedule is periodic"
            );
            assert!(b("n_groups") > 1 && b("mamba_num_heads") / b("n_groups") > 1);
            assert!(b("num_attention_heads") / b("num_key_value_heads") > 1);
            assert!(b("num_experts_per_tok") > 1);
            assert_eq!(latent, mp.bounds.contains_key("moe_latent_size"));
            if latent {
                assert!(b("moe_latent_size") < b("hidden_size"));
            }
        }
    }

    /// A Mamba-2 mixer fills the GDN state pool's geometry — groups as key heads, heads as value
    /// heads, `[mamba_head_dim, ssm_state_size]` a head, the conv over `xBC` — and a GDN arch
    /// keeps its own (`linear_*`).
    #[test]
    fn recurrent_state_dims_come_from_either_mixer() {
        use crate::codegen::RecurrentDims;
        // Load-or-refuse: the parity-tiny ships a mixed-bit affine quantization,
        // so under a backend that can't lower it the refusal is the outcome
        // (`None`); a dense config loads on every backend.
        let dims = |arch: &str, stem: &str| -> Option<RecurrentDims> {
            let dir = repo_model_archs(arch);
            let spec = load_arch_json(&dir).expect("arch.json");
            let path = dir.join(format!("{stem}.json"));
            let json: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("json");
            let mp = assert_load_or_refuse_by_name(
                stem,
                model_params_from_json(&json, stem, &path, Vec::new(), &spec),
            )?;
            Some(RecurrentDims::of(&mp).expect("a recurrent mixer"))
        };
        let mamba = |groups, heads, head_dim| RecurrentDims {
            num_k_heads: groups,
            num_v_heads: heads,
            head_k_dim: 128,
            head_v_dim: head_dim,
            conv_kernel: 4,
            conv_dim: heads * head_dim + 2 * groups * 128,
        };
        assert_eq!(
            dims("nemotron-h", "nemotron-3-nano-30b-a3b"),
            Some(mamba(8, 64, 64))
        );
        let super_dims = dims("nemotron-h-latent", "nemotron-3-super-120b-a12b");
        assert_eq!(super_dims, Some(mamba(8, 128, 64)));
        assert_eq!(super_dims.expect("dense config loads").conv_dim, 10240);
        if let Some(tiny) = dims("nemotron-h", "nemotron-3-nano-30b-a3b-parity-tiny") {
            assert_eq!(
                (
                    tiny.num_k_heads,
                    tiny.num_v_heads,
                    tiny.head_k_dim,
                    tiny.head_v_dim
                ),
                (2, 8, 16, 16)
            );
        }
        let gdn = dims("qwen3-5", "qwen3.5-0.8b").expect("dense config loads");
        assert_eq!(
            (
                gdn.num_k_heads,
                gdn.num_v_heads,
                gdn.head_k_dim,
                gdn.conv_dim
            ),
            (16, 16, 128, 6144)
        );
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

    /// A yarn `rope_scaling` block that omits the mscale pair (gpt-oss)
    /// parses to HF's signature defaults 1.0/1.0 — with that pair the
    /// cos/sin mscale ratio is exactly 1.0 (plain interpolation, no
    /// attention temperature), which is what mlx-lm/vLLM compute for
    /// gpt-oss. The pre-fix default (`mscale_all_dim` 0.0) made the
    /// ratio `0.1·ln(factor)+1 ≈ 1.3466` at factor 32 — a wrong
    /// attention temperature folded into the cache.
    #[test]
    fn yarn_config_without_mscale_pair_parses_to_hf_defaults() {
        let json = serde_json::json!({
            "rope_scaling": {
                "type": "yarn",
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096
            }
        });
        match extract_rope_scaling(&json) {
            Some(RopeScaling::Yarn {
                factor,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
                original_max_position_embeddings,
            }) => {
                assert_eq!(factor, 32.0);
                assert_eq!(beta_fast, 32.0);
                assert_eq!(beta_slow, 1.0);
                assert_eq!(mscale, 1.0);
                assert_eq!(mscale_all_dim, 1.0);
                assert_eq!(original_max_position_embeddings, 4096);
            }
            other => panic!("expected Yarn scaling, got {other:?}"),
        }
    }

    /// Spec mirroring the gemma4 `#[forward]` carrier-mod declarations
    /// (`DECODER_PREFIX`, `BOUND_DEFAULTS`, `SCALAR_DEFAULTS`,
    /// `CONFIG_ALIASES`) — used to load the verbatim configs in tests.
    fn gemma4_carrier_spec() -> crate::arch_spec::DeclaredArchSpec {
        crate::arch_spec::DeclaredArchSpec {
            decoder_prefix: Some("language_model".into()),
            bound_defaults: Vec::new(),
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
        // No config caps the context: the block table's width is the KV
        // pool's, picked at load from `--max-model-len` and the memory budget.
        assert_eq!(mp.bounds.get("max_blocks_per_seq"), None);
        // Rope scalars hoisted from nested rope_parameters.
        assert_eq!(mp.scalars.get("rope_theta"), Some(&1_000_000.0));
        assert_eq!(mp.scalars.get("rope_local_base_freq"), Some(&10_000.0));
        assert_eq!(mp.scalars.get("global_partial_rotary_factor"), Some(&0.25));
        // Arch-constant softmax scale (mlx hardcodes 1.0).
        assert_eq!(mp.scalars.get("attention_multiplier"), Some(&1.0));
        // Logit softcap is verbatim in-config.
        assert_eq!(mp.scalars.get("final_logit_softcapping"), Some(&30.0));
        assert!(mp.tie_word_embeddings);
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
            &Scope::all(),
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
                "num_attention_heads": 16,
                "rope_theta": 10000.0,
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
        let configs = load_dir(
            &dir,
            &load_arch_json(&dir).expect("llama arch.json"),
            &Scope::all(),
        )
        .expect("load llama configs");
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
        fs::write(
            tmp.join("model.json"),
            r#"{"num_hidden_layers": 4, "rope_theta": 10000.0, "head_dim": 64}"#,
        )
        .unwrap();
        fs::write(tmp.join("README.txt"), "ignore me").unwrap();

        let configs = load_dir(&tmp, &Default::default(), &Scope::all()).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].source_stem, "model");

        fs::remove_dir_all(&tmp).ok();
    }

    /// The expert body's declared fact: `mlp_hidden_act` (Nemotron-H) wins over `hidden_act`,
    /// HF's `silu` default stands in for neither, and a name no body realizes is carried for the
    /// refusal rather than read as SiLU.
    #[test]
    fn expert_activation_reads_the_declared_activation() {
        let act = |v: serde_json::Value| ExpertActivation::of(&v);
        assert_eq!(act(serde_json::json!({})), ExpertActivation::Silu);
        assert_eq!(
            act(serde_json::json!({"hidden_act": "silu"})),
            ExpertActivation::Silu
        );
        assert_eq!(
            act(serde_json::json!({"hidden_act": "silu", "mlp_hidden_act": "relu2"})),
            ExpertActivation::Relu2
        );
        let gelu = act(serde_json::json!({"hidden_act": "gelu_pytorch_tanh"}));
        assert_eq!(
            gelu,
            ExpertActivation::Unrealized("gelu_pytorch_tanh".to_string())
        );
        assert_eq!(expert_body(&gelu), Err("gelu_pytorch_tanh"));
    }
}
