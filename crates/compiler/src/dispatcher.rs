// SPDX-License-Identifier: Apache-2.0
//! Cross-arch dispatcher: the cfg-free `ScratchyWeights` trait + the
//! `inventory`-driven `ArchRegistration` / `try_load` registry
//! seam every `#[forward] fn <arch>()` registers through.
//!
//! Every `#[forward]` macro invocation auto-emits an
//! `impl ScratchyWeights for Weights` + an `inventory::submit!`
//! registration, so `try_load` discovers every compiled arch
//! without a hand-written central list. Adding a new arch touches
//! only its source file plus a single `pub mod <arch>;` in the
//! calling crate's lib.rs (Rust module system requirement).
//!
//! `ScratchyWeights` carries only what every target's loaded model answers: its geometry and
//! the model facts the engine sizes from. How a model RUNS is its target's: each target crate
//! extends this trait with its own (`<target>Weights: ScratchyWeights`) and registers its loaders
//! under its own registration type, which wraps the shared [`crate::ArchRegistration`].

/// A weight a multi-token-prediction head does not carry and borrows from its target model: the
/// token embedding it embeds its input tokens with, and the lm_head it drafts through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LentWeight {
    EmbedTokens,
    LmHead,
}

impl LentWeight {
    /// Every lent weight.
    pub const ALL: [Self; 2] = [Self::EmbedTokens, Self::LmHead];
}

/// Arch-agnostic handle for a loaded model. Every
/// `#[forward] fn <arch>()` emits an `impl ScratchyWeights` for
/// its per-arch `Weights` type; callers hold
/// `Box<dyn ScratchyWeights>` and never need to know which arch
/// they got.
pub trait ScratchyWeights: Send + Sync {
    fn arch_name(&self) -> &'static str;
    fn num_hidden_layers(&self) -> u64;
    fn hidden_size(&self) -> u64;
    fn intermediate_size(&self) -> u64;
    fn num_attention_heads(&self) -> u64;
    fn num_key_value_heads(&self) -> u64;
    fn head_dim(&self) -> u64;
    fn vocab_size(&self) -> u64;
    /// Gated-DeltaNet (linear-attention) runtime config for hybrid arches
    /// (Qwen3.5 / Qwen3-Next). Drives the worker's `GdnStatePool` sizing +
    /// allocation. The proc-macro emits a per-arch override returning
    /// `Some(_)` for arches whose forward body contains a `gated_delta_net`
    /// op; the default (non-hybrid arches) returns `None`, so no GDN state
    /// pool is allocated and the `gdn_state` ForwardCtx field stays `None`.
    fn gdn_runtime_config(&self) -> Option<crate::gdn_state_layout::GdnRuntimeConfig> {
        None
    }

    /// Whether this model's forward reads `target_hidden`: a multi-token-prediction head, which
    /// drafts from its target model's final hidden states. The macro emits `true` for such
    /// forwards; the default is `false`.
    fn reads_target_hidden(&self) -> bool {
        false
    }

    /// The on-disk prefix (no `.weight` / `.scales` / `.biases`) this model loads `w` from — a
    /// tied lm_head's is its embedding's — so a head can be lent its target's tensors under its
    /// own names. `None` when the model has no such weight. Emitted by the macro.
    fn lent_weight_prefix(&self, _w: LentWeight) -> Option<&'static str> {
        None
    }

    /// Per-layer `kv_heads * head_dim` for hybrid-attention-geometry
    /// arches whose sliding and global classes differ in dims
    /// (Gemma4: sliding 8×256 = 2048 elems/token, global 1×512 =
    /// 512). Drives per-layer KV pool sizing. The proc-macro emits
    /// a per-arch override derived from the unrolled IR (which
    /// layers carry `sliding_attention` vs `attention` tiles) and
    /// the GLOBAL_* CanonicalParams; the default `None` keeps the
    /// uniform pool layout (every same-dims arch, incl. Gemma2/3).
    fn per_layer_kv_token_elems(&self) -> Option<Vec<usize>> {
        None
    }

    /// The loaded variant's `CanonicalParams::KV_CACHE_LAYERS`: the layers that
    /// hold KV when not every layer does (a recurrent hybrid's attention
    /// layers). The worker's KV pool holds tensors for these only and sizes its
    /// budget by them. Default `None`: every layer holds KV.
    fn kv_cache_layers(&self) -> Option<&'static [u32]> {
        None
    }

    /// Spans / position-independent KV caching (`W::ROPE_ON_READ`). When
    /// true, this arch's paged attention + rope_append kernels mask
    /// block_table / slot_mapping **bit 31** and treat it as the
    /// "stored unrotated → rotate on read" flag. A worker reads
    /// this to decide whether it is SAFE to OR bit 31 into the block_table
    /// / slot_mapping it uploads (a non-rope-on-read arch's kernels read
    /// those raw, so bit 31 would corrupt the physical block id). Default
    /// `false`; the macro emits `true` for arches with `ROPE_ON_READ`.
    fn rope_on_read(&self) -> bool {
        false
    }

    /// The drafts each sequence of a speculative verify step carries (`CanonicalParams::
    /// SPEC_DRAFTS`): an MTP head's, and its target's when this build compiles the head. 0 for a
    /// model without one.
    fn spec_drafts(&self) -> u32 {
        0
    }
}

/// Minimal HF-config view threaded into `try_load` so per-variant
/// `fingerprint_matches` can disambiguate checkpoints that share
/// on-disk tensor shapes but differ in config-only fields.
/// Phi-3-mini-4k (`max_position_embeddings=4096`, `rope_scaling=null`)
/// and Phi-3.5-mini-128k (`131072`, `{type:"longrope", …}`) have
/// identical weight shapes; without the config view the
/// alphabetically-earlier variant's fingerprint wins and its
/// (manifest-baked) RoPE cache gets used for the wrong model.
///
/// Kept as a struct of plain `Option<primitive>` so scratchy
/// stays decoupled from the caller's full HF-config parser —
/// add fields here only when a future arch truly needs them to
/// disambiguate.
#[derive(Clone, Copy, Debug, Default)]
pub struct HfFingerprint<'a> {
    pub rope_scaling_type: Option<&'a str>,
    /// Deterministic hash of the full `rope_scaling` JSON
    /// subobject (or `None` when the checkpoint has no
    /// rope_scaling). Discriminates checkpoints that share
    /// `rope_scaling_type` but differ in `short_factor` /
    /// `long_factor` / `original_max_position_embeddings`
    /// values — e.g. Phi-3.5-mini vs Phi-3-mini-128k,
    /// Phi-4-mini-instruct vs Phi-4-mini-reasoning. Computed
    /// with [`crate::hash_json_value`].
    ///
    /// `max_position_embeddings` is intentionally not on this
    /// fingerprint. It doesn't affect forward-fn codegen (MPE
    /// is a runtime KV-cache sizing input), and including it
    /// rejected checkpoints whose published `config.json` has
    /// a context window narrower than the upstream base
    /// (e.g. mlx-community 4bit Qwen2.5-1.5B publishes
    /// `max_position_embeddings: 32768` against the upstream
    /// `131072`). The rope-scaling-type and rope-scaling-hash
    /// disambiguation already catches every Phi-3-style
    /// long/short fork — `MPE` was redundant.
    pub rope_scaling_hash: Option<u64>,
    /// The checkpoint's declared `rope_theta`.
    ///
    /// ⛔ THIS IS NOT REDUNDANT WITH THE TWO FIELDS ABOVE, AND ITS ABSENCE WAS A
    /// SILENT-WRONG-OUTPUT BUG. `rope_theta` is baked as a compile-time literal
    /// into the generated `RotaryCache::new_from_stream` call, so a checkpoint
    /// that fingerprint-matches the wrong variant gets that variant's RoPE base
    /// frequency. Nothing downstream can detect it: the shapes all agree, every
    /// load succeeds, and the model emits fluent-looking token soup.
    ///
    /// Unlike the Phi-3 long/short forks, the colliding variants here declare NO
    /// `rope_scaling` at all, so neither `rope_scaling_type` nor
    /// `rope_scaling_hash` can separate them. These groups ship in-tree today and
    /// are byte-identical on every other fingerprint axis (same
    /// hidden/layers/vocab/heads/kv-heads, same quant format):
    ///   - `granite-3.0-2b-instruct` (10⁴) vs `granite-3.1-2b-instruct` /
    ///     `granite-3.2-2b-instruct` (5×10⁶) — and the same three-way split on
    ///     `granite-3.{0,1,3}-2b-base`, `-8b-base` and `-8b-instruct` (10⁴ vs 10⁷);
    ///   - `phi-4` (2.5×10⁵) vs `phi-4-reasoning` / `phi-4-reasoning-plus` (5×10⁵).
    ///
    /// Under `model/granite` — or `model/all`, which is CI's scope — whichever
    /// variant `inventory` happened to register first won.
    ///
    /// `None` is permissive (the variant's baked value stays authoritative),
    /// matching [`Self::rope_scaling_hash`]. A caller that cannot trust its
    /// source's value passes `None` rather than a guess (GGUF metadata, for
    /// one, routinely disagrees with the canonical `config.json`). Note this is the checkpoint's TOP-LEVEL
    /// `rope_theta` only — arches that carry per-attention-class thetas nested
    /// under `rope_parameters` (gemma3/gemma4/qwen3.5) declare no top-level value,
    /// so they pass `None` here and are unaffected either way.
    pub rope_theta: Option<f64>,
}
