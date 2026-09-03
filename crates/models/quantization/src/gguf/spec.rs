// SPDX-License-Identifier: Apache-2.0
//! Per-arch GGUF support: alignment between a GGUF
//! `general.architecture` value and a `scratchy-models` arch.
//!
//! Each arch registers one [`GgufArchSpec`] declaring the
//! GGUF tag it claims, the HF arch class to stamp, the qk-permute
//! flag, and any per-arch tensor renames + metadata reads.
//!
//! **All fields are pure data** — no function pointers. Per-arch
//! knowledge enters this crate exclusively through the arch's
//! `configs/quantizations.json` `ggml` entry; the `scratchy-forward-compiler`
//! macro forwards each field straight into the struct literal it
//! emits via [`crate::register!`]. There are no per-arch code paths
//! in this crate.
//!
//! Construction goes through [`crate::register!`] only — the struct
//! is `#[doc(hidden)]` and not part of the public API.

use scratchy_core_model::weight::HfModelConfig;

/// A constant value to insert into `HfModelConfig.extra` when an arch
/// declares a `metadata_defaults` entry. Variant chosen by the JSON
/// type at macro-expansion time.
#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub enum GgufDefault {
    U32(u32),
    F32(f32),
}

/// Per-arch GGUF binding. See module docs.
#[doc(hidden)]
pub struct GgufArchSpec {
    /// GGUF `general.architecture` value this spec handles.
    pub gguf_arch: &'static str,
    /// Whether to un-permute q/k rows on load (Llama-derived archs).
    pub qk_permute: bool,
    /// Per-suffix tensor-name overrides. Each `(gguf_suffix, hf_suffix)`
    /// applies to `blk.{N}.<suffix>` tensors. Looked up before the
    /// default Llama-shape rename.
    pub tensor_renames: &'static [(&'static str, &'static str)],
    /// Per-arch u32 reads. Each `(gguf_key, extra_key)` says: read
    /// the GGUF metadata at `gguf_key` (`{arch}` substituted for
    /// `gguf_arch`) as a u32, store under `HfModelConfig.extra[extra_key]`.
    pub metadata_u32: &'static [(&'static str, &'static str)],
    /// Same as [`Self::metadata_u32`] but read as f32.
    pub metadata_f32: &'static [(&'static str, &'static str)],
    /// Constants always inserted into `HfModelConfig.extra` when the
    /// key isn't already set by another reader.
    pub metadata_defaults: &'static [(&'static str, GgufDefault)],
    /// Heuristic flag for Llama-3.x checkpoints whose GGUF omits the
    /// rope.scaling.* keys (unsloth GGUFs do this). When true and
    /// `rope_theta == 500000` and `max_position_embeddings > 8192`,
    /// `gguf_model_config` stamps the canonical llama3 scaling block.
    pub llama3_rope_scaling_inference: bool,
    /// Constant value baked into every rmsnorm weight by llama.cpp's
    /// GGUF converter. Subtracted at load time to recover the raw
    /// safetensors `w` so the runtime kernel's `(w + offset)` fold
    /// produces the right `(1+w_orig)`. Gemma2 / Gemma3 ship norms as
    /// `1+w` (the converter pre-adds 1 so vanilla `rmsnorm(x, w)`
    /// matches HF's `rmsnorm(x, 1+w)`); scratchy's DSL still uses
    /// `weight + 1.0`, so without this subtraction we'd double-apply
    /// the offset and double-scale every norm output. Default `0.0`
    /// for archs whose GGUF stores raw `w` (everything except Gemma).
    pub norm_weight_offset: f32,
}

inventory::collect!(GgufArchSpec);

/// Look up the spec for a given GGUF `general.architecture` value.
pub fn find_spec(gguf_arch: &str) -> Option<&'static GgufArchSpec> {
    inventory::iter::<GgufArchSpec>()
        .into_iter()
        .find(|s| s.gguf_arch == gguf_arch)
}

/// Look up a per-suffix tensor rename. Returns `None` when the spec
/// has no override for this suffix; caller falls through to the
/// default Llama-shape rename.
pub fn lookup_rename(spec: &GgufArchSpec, gguf_suffix: &str) -> Option<&'static str> {
    spec.tensor_renames
        .iter()
        .find_map(|&(g, h)| (g == gguf_suffix).then_some(h))
}

/// Apply the spec's metadata reads, defaults, and llama3-rope flag to
/// `config.extra`. Called by `gguf_model_config` after the generic
/// header reads have populated the standard fields.
pub fn apply_metadata(spec: &GgufArchSpec, gguf: &super::GgufFile, config: &mut HfModelConfig) {
    let arch = spec.gguf_arch;
    let resolve = |key: &str| key.replace("{arch}", arch);

    for &(gguf_key, extra_key) in spec.metadata_u32 {
        if let Some(v) = gguf.get_metadata_u32(&resolve(gguf_key)) {
            config
                .extra
                .insert(extra_key.to_string(), serde_json::json!(v));
        }
    }
    for &(gguf_key, extra_key) in spec.metadata_f32 {
        if let Some(v) = gguf.get_metadata_f32(&resolve(gguf_key)) {
            config
                .extra
                .insert(extra_key.to_string(), serde_json::json!(v as f64));
        }
    }
    for &(extra_key, val) in spec.metadata_defaults {
        let v = match val {
            GgufDefault::U32(n) => serde_json::json!(n),
            GgufDefault::F32(f) => serde_json::json!(f as f64),
        };
        // Dotted keys splice into a nested object inside `extra`. Today
        // only one level of nesting is supported (`parent.child`); used
        // for filling in missing `rope_scaling.{mscale, mscale_all_dim,
        // beta_fast, beta_slow}` on YaRN GGUFs whose converter omitted
        // those tuning knobs (e.g. mradermacher's DeepSeek-V2-Lite).
        if let Some((parent_key, child_key)) = extra_key.split_once('.') {
            let parent = config
                .extra
                .entry(parent_key.to_string())
                .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            if let Some(obj) = parent.as_object_mut()
                && !obj.contains_key(child_key)
            {
                obj.insert(child_key.to_string(), v);
            }
            continue;
        }
        if config.extra.contains_key(extra_key) {
            continue;
        }
        config.extra.insert(extra_key.to_string(), v);
    }

    // Llama-3.x rope-scaling inference: unsloth Llama-3 GGUFs omit
    // the rope.scaling.* keys. Recognize Llama-3 by
    // (rope_theta == 500000, ctx > 8192) and stamp the canonical
    // scaling shared by Llama-3.1 / 3.2 / 3.3.
    if spec.llama3_rope_scaling_inference && !config.extra.contains_key("rope_scaling") {
        let theta = config.rope_theta.unwrap_or(10000.0);
        let ctx = config.max_position_embeddings.unwrap_or(0);
        if (theta - 500000.0).abs() < 1.0 && ctx > 8192 {
            let mut s = serde_json::Map::new();
            s.insert("rope_type".into(), "llama3".into());
            s.insert("factor".into(), 32.0.into());
            s.insert("low_freq_factor".into(), 1.0.into());
            s.insert("high_freq_factor".into(), 4.0.into());
            s.insert(
                "original_max_position_embeddings".into(),
                serde_json::Value::from(8192u64),
            );
            config
                .extra
                .insert("rope_scaling".into(), serde_json::Value::Object(s));
        }
    }
}

/// The only sanctioned construction site for `GgufArchSpec`. Driven
/// by data — every field is a literal forwarded directly from the
/// arch's `configs/quantizations.json` `ggml` entry.
#[macro_export]
macro_rules! register {
    (
        gguf_arch = $gguf:literal
        $(, qk_permute = $perm:literal)?
        $(, tensor_renames = [ $(($tg:literal, $th:literal)),* $(,)? ])?
        $(, metadata_u32 = [ $(($mu_g:literal, $mu_e:literal)),* $(,)? ])?
        $(, metadata_f32 = [ $(($mf_g:literal, $mf_e:literal)),* $(,)? ])?
        $(, metadata_defaults_u32 = [ $(($mdu_k:literal, $mdu_v:literal)),* $(,)? ])?
        $(, metadata_defaults_f32 = [ $(($mdf_k:literal, $mdf_v:literal)),* $(,)? ])?
        $(, llama3_rope_scaling_inference = $rope:literal)?
        $(, norm_weight_offset = $nwo:literal)?
        $(,)?
    ) => {
        $crate::inventory::submit! {
            $crate::GgufArchSpec {
                gguf_arch: $gguf,
                qk_permute: $crate::register!(@bool false $($perm)?),
                tensor_renames: &[ $($( ($tg, $th) ),*)? ],
                metadata_u32: &[ $($( ($mu_g, $mu_e) ),*)? ],
                metadata_f32: &[ $($( ($mf_g, $mf_e) ),*)? ],
                metadata_defaults: &[
                    // Trailing commas (rather than between-item commas)
                    // so concatenating the two repetition groups keeps
                    // valid syntax when either side is empty.
                    $($( ($mdu_k, $crate::GgufDefault::U32($mdu_v)), )*)?
                    $($( ($mdf_k, $crate::GgufDefault::F32($mdf_v)), )*)?
                ],
                llama3_rope_scaling_inference:
                    $crate::register!(@bool false $($rope)?),
                norm_weight_offset:
                    $crate::register!(@f32 0.0 $($nwo)?),
            }
        }
    };
    (@bool $default:literal) => { $default };
    (@bool $_default:literal $v:literal) => { $v };
    (@f32 $default:literal) => { $default };
    (@f32 $_default:literal $v:literal) => { $v };
}
