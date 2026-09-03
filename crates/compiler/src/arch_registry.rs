// SPDX-License-Identifier: Apache-2.0
//! Cross-arch load registry: the `inventory`-driven
//! `ScratchyArchRegistration` / `try_load` seam every `#[forward] fn <arch>()`
//! registers through, plus its multimodal sibling
//! (`ScratchyMmRegistration` / `try_load_mm`).
//!
//! The per-arch loaders fill the concrete `GpuWeights<A>` weight-store, a type
//! this cfg-free crate cannot name (it has no backend allocator). The registry
//! fn-pointers therefore take the allocator-erased
//! [`scratchy_tensors::GpuWeightsHandle`]; the generic [`try_load`] /
//! [`try_load_mm`] entry points are monomorphized over the concrete
//! `GpuWeights<A>` at the worker call site (where `A` is the backend
//! allocator), build the handle, and pass it through the fn-pointer. The
//! macro-emitted closure recovers the concrete `&mut GpuWeights<A>` from the
//! handle before calling the per-variant `Weights::load`. This keeps the
//! registry — and the `#[forward]` macro's `inventory::submit!` target — free
//! of any backend coupling, so it can live below the neutral `ScratchyWeights`
//! contract the compiler owns.

// MM sibling registry names `scratchy-vision` (`MmMetadata`) + device-runtime
// types, neither available under spyre; the text-side registry below is
// allocator-generic + vision-free, so it compiles under spyre too.
#[cfg(feature = "vision")]
use crate::mm::MultimodalForward;
use crate::{HfFingerprint, ScratchyWeights};
use scratchy_tensors::{DeviceAllocator, GpuWeightsHandle, LoadStream};

/// One registration per `#[forward] fn <arch>()`. The macro
/// emits an `inventory::submit!` block that constructs this.
/// `try_load` function-pointer signature — extracted as a type
/// alias so the registration struct doesn't trip clippy's
/// `type_complexity` lint.
///
/// The weight parameter is the allocator-erased [`GpuWeightsHandle`] — the
/// macro-emitted closure recovers the concrete `&mut GpuWeights<A>` the
/// per-variant `Weights::load` loaders need. The stream parameter is the
/// backend-neutral [`LoadStream`] (cuda `CUstream`; `()` elsewhere), not the
/// cuda-only `CUstream` type, so the signature is identical across backends and
/// a third target slots into the same registry.
pub type ArchTryLoadFn = fn(
    GpuWeightsHandle<'_>,
    LoadStream,
    usize, // max_model_len — runtime value (CLI --max-model-len or HF config fallback)
    u8,    // tp_rank — runtime value, must be < tp_world_size
    HfFingerprint<'_>,
) -> ::anyhow::Result<Option<Box<dyn ScratchyWeights>>>;

/// `ktir_bundle` accessor signature — extracted as a type alias (like
/// [`ArchTryLoadFn`]) so the registration struct doesn't trip
/// `clippy::type_complexity`. The struct is not cfg-gated, so this type is
/// compiled under every backend (the field is just `None` off spyre). Same
/// erased-handle + fingerprint inputs as the per-variant `fingerprint_matches`;
/// returns the matched variant's KTIR bundle as a neutral `&dyn Any`.
pub type ArchKtirBundleFn = fn(
    GpuWeightsHandle<'_>,
    HfFingerprint<'_>,
) -> Option<&'static (dyn ::core::any::Any + Send + Sync)>;

pub struct ScratchyArchRegistration {
    /// The arch's identifier — `"llama"`, `"qwen2"`, … — from
    /// the carrier fn name. Used in logs.
    pub arch_name: &'static str,
    /// HF `architectures` strings this arch claims to handle.
    /// Harvested by the macro from the union of every compiled
    /// model's `config.json` `architectures: [..]` field.
    pub hf_arches: &'static [&'static str],
    /// GGUF `general.architecture` tags this arch claims to handle.
    /// Family-level — multiple forward arches can claim the same
    /// gguf tag (`"deepseek2"` covers V2, V3-LoRA, V3-flat;
    /// `"llama"` covers Llama and Mistral). The dispatcher tries
    /// each claimant in turn and the per-variant
    /// `fingerprint_matches` discriminates by bounds.
    ///
    /// Populated by the macro from the arch's `quantizations.json`
    /// `ggml.gguf_arch` field (defaults to `arch_name` when the
    /// entry is bare). Empty for arches without a `"ggml"`
    /// quantization preset.
    pub gguf_archs: &'static [&'static str],
    /// Compile-time tensor-parallel world size this registration
    /// covers. The macro emits one `ScratchyArchRegistration` per
    /// `(arch, tp_world_size)` tuple — at task #7's outer-loop
    /// fanout that's `{1, 2, 4, 8}` per arch; until then every
    /// emitted registration is `tp_world_size = 1`. The top-level
    /// `try_load` matches on this so the runtime tp size picks
    /// the right pre-compiled variant set.
    pub tp_world_size: u8,
    /// Try to load the arch's compiled variants. Internally
    /// iterates per-variant fingerprint sniffs. Returns
    /// `Ok(Some(..))` on a variant hit, `Ok(None)` when the arch
    /// matched by name but no compiled variant's fingerprint
    /// sniff accepted the live `GpuWeights` (caller should fall
    /// back to a hand-written path), or `Err(..)` only on a
    /// genuine load failure (I/O, shape mismatch inside a matched
    /// variant, …).
    pub try_load: ArchTryLoadFn,
    /// Read-only accessor for this arch's embedded KTIR bundle — spyre's forward
    /// representation (the peer of cuda's `FORWARD_TABLE` / metal's
    /// `METAL_BUCKETS`). `Some` under `-Fspyre`, `None` under cuda/metal (whose
    /// forward runs through device kernels, not a bundle). Returned as a neutral
    /// `&dyn Any` (this cfg-free crate cannot name the spyre `KtirBundle`); the
    /// spyre worker downcasts it.
    ///
    /// Takes the same `(GpuWeightsHandle, HfFingerprint)` a variant's
    /// `fingerprint_matches` needs and returns the bundle of the ONE variant
    /// whose fingerprint accepts the live checkpoint — an arch like `llama`
    /// registers many distinct base models (llama-2-13b … llama-3.2-3b), each
    /// with its own bundle + layer count, so a by-arch "canonical" pick would
    /// hand back the wrong model's bundle. Only READS `gw` (shape sniffing),
    /// leaving every tensor for the worker to stream into the KTIR runner.
    pub ktir_bundle: Option<ArchKtirBundleFn>,
    /// Read-only accessor for this arch's embedded **sendnn** bundle (the
    /// `--target sendnn` silicon path; peer of [`Self::ktir_bundle`]). `Some`
    /// under `-Fspyre`. Returned as a neutral `&dyn Any` the spyre worker
    /// downcasts to `SengraphBundle`.
    pub sengraph_bundle: Option<ArchKtirBundleFn>,
}

inventory::collect!(ScratchyArchRegistration);

/// Resolve the embedded KTIR bundle for the checkpoint in `gw`, read-only.
/// Walks registrations claiming `arch_hint` for `tp_world_size == 1` (spyre is
/// single-rank) and, within the matching arch, returns the bundle of the ONE
/// compiled variant whose `fingerprint_matches` accepts the live weights — the
/// same per-variant sniff `try_load` uses, so bundle selection and weight-load
/// variant selection can never diverge. Only reads `gw` (no `take`), so the
/// worker keeps every tensor to stream into the runner. Returns `None` when no
/// compiled spyre variant matches (e.g. the model wasn't built into this binary).
#[cfg(feature = "spyre")]
pub fn resolve_ktir_bundle<A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    arch_hint: &str,
    hf: HfFingerprint<'_>,
) -> Option<&'static (dyn ::core::any::Any + Send + Sync)> {
    let handle = GpuWeightsHandle::new(gw);
    inventory::iter::<ScratchyArchRegistration>()
        .filter(|reg| {
            (reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint))
                && reg.tp_world_size == 1
        })
        .find_map(|reg| reg.ktir_bundle.and_then(|f| f(handle, hf)))
}

/// Resolve the embedded sendnn bundle (the `--target sendnn` path) for an arch
/// by name. Peer of [`resolve_ktir_bundle`]; the spyre worker downcasts the
/// returned `&dyn Any` to `SengraphBundle`.
#[cfg(feature = "spyre")]
pub fn resolve_sengraph_bundle<A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    arch_hint: &str,
    hf: HfFingerprint<'_>,
) -> Option<&'static (dyn ::core::any::Any + Send + Sync)> {
    // 🛑 Takes the checkpoint fingerprint for the SAME reason
    // `resolve_ktir_bundle` does, and it used to not.
    //
    // This picked `arms.first()` — the first variant REGISTERED for the
    // arch, unconditionally. With one model compiled that is right by
    // accident. With two sizes of one arch it is a coin flip: a build with
    // both granite-3.1-2b and granite-3.1-8b compiled in (same arch, two
    // size tiers) then ran the 8b against the 2B's bundle, and
    // load_weights refused with
    // "q_proj.weight is 16777216 bytes but the manifest declares
    // 2048*2048*1 = 4194304" — the 8b's real weights (4096²) against the
    // 2b's manifest (2048²). The guard caught it; without it the bundle
    // mismatch would have been garbage tokens.
    let handle = GpuWeightsHandle::new(gw);
    inventory::iter::<ScratchyArchRegistration>()
        .filter(|reg| {
            (reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint))
                && reg.tp_world_size == 1
        })
        .find_map(|reg| reg.sengraph_bundle.and_then(|f| f(handle, hf)))
}

// ── Multimodal sibling surface ────────────────────────────────
//
// `MultimodalForward` is a sibling trait, NOT a default-method
// extension on `ScratchyWeights`. Text-only arches don't
// implement it. Discovery uses a sibling inventory row so the
// text-side `ScratchyArchRegistration` stays untouched and
// text-only arches never need to know MM exists. The gpu worker
// calls `try_load_mm` after `try_load` succeeds; an arch with
// no MM submission yields `Ok(None)` and the worker proceeds
// text-only. Phase B of the multimodal plan
// (`~/.claude/plans/distributed-mapping-map.md`) lands the
// surface; per-arch impls land in Phase D.

/// Sibling MM-load fn. Same dispatch shape as [`ArchTryLoadFn`]
/// — caller's `GpuWeights` already carries every tensor on disk
/// (including `visual.*` for an MM checkpoint), so this fn
/// extracts the vision sub-tree and returns a handle. Returns
/// `Ok(None)` when the arch claims the HF arch string but the
/// live checkpoint has no vision tensors (text-only checkpoint
/// loaded through an MM-capable arch entry — falls through).
#[cfg(feature = "vision")]
pub type MmTryLoadFn = fn(
    GpuWeightsHandle<'_>,
    LoadStream,
    usize, // max_model_len
    u8,    // tp_rank
    HfFingerprint<'_>,
) -> ::anyhow::Result<Option<Box<dyn MultimodalForward>>>;

/// Sibling registration row. One per arch that ships a vision
/// encoder. `hf_arches` and `gguf_archs` follow the same rules
/// as [`ScratchyArchRegistration`] but the keys typically only
/// list the multimodal-conditional-generation variants
/// (e.g. `Qwen2VLForConditionalGeneration`, NOT plain
/// `Qwen2ForCausalLM`).
///
/// `mm_metadata` is the per-arch CPU-side preprocessing
/// declaration baked from a `pub const PROCESSOR: MmMetadata` in
/// the arch crate, threaded through `#[vision_forward(processor =
/// path::PROCESSOR, ...)]`. The loader stays arch-agnostic: every
/// arch-specific knob (placeholder token id key, size policy,
/// tokens-per-image policy, preprocess fn) is data on this row,
/// not a switch in the loader or the macro.
#[cfg(feature = "vision")]
pub struct ScratchyMmRegistration {
    pub arch_name: &'static str,
    pub hf_arches: &'static [&'static str],
    pub gguf_archs: &'static [&'static str],
    pub tp_world_size: u8,
    pub try_load_mm: MmTryLoadFn,
    pub mm_metadata: scratchy_vision::MmMetadata,
}

#[cfg(feature = "vision")]
inventory::collect!(ScratchyMmRegistration);

/// Walk the [`ScratchyMmRegistration`] inventory and return the
/// first registration whose `hf_arches` claims any of the
/// supplied HF architecture strings. Returns `None` for text-
/// only models. Used by serve at startup to pick MM metadata
/// without knowing any arch names itself.
///
/// `tp_world_size` filter is intentionally not applied here —
/// MM metadata is identical across the tp variants of an arch
/// (preprocessing is host-side and replicated), so the first
/// hit is sufficient.
#[cfg(feature = "vision")]
pub fn resolve_mm_metadata(hf_arches: &[String]) -> Option<&'static ScratchyMmRegistration> {
    inventory::iter::<ScratchyMmRegistration>().find(|reg| {
        hf_arches
            .iter()
            .any(|a| reg.hf_arches.contains(&a.as_str()))
    })
}

/// Top-level architecture loader. Walks every `#[forward]`-registered
/// arch; the first whose `hf_arches` list contains `arch_hint`
/// AND whose `tp_world_size` matches the runtime `tp_world_size`
/// wins and attempts to load. Returns `Ok(None)` when either
/// (a) no registered (arch, tp) pair claims the request, or
/// (b) a pair matched but no compiled variant's fingerprint sniff
/// accepted the live `GpuWeights`. Both cases let the caller
/// fall back to the hand-written path without hard-failing.
///
/// Generic over the backend allocator `A`: the worker passes its concrete
/// `&mut GpuWeights<A>` and `A` is inferred. The borrow is wrapped in a
/// [`GpuWeightsHandle`] (allocator-erased) before being threaded through the
/// fn-pointer; the macro-emitted closure recovers the same concrete type.
///
/// Until task #7's outer-loop fanout lands, every emitted
/// registration is at `tp_world_size = 1`, so callers passing
/// `tp_world_size > 1` always see `Ok(None)` (and fall back) —
/// matching the current behavior, since the gpu worker already gates
/// eligibility on `!use_tp`.
pub fn try_load<A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    stream: LoadStream,
    arch_hint: &str,
    tp_world_size: u8,
    tp_rank: u8,
    max_model_len: usize,
    hf: HfFingerprint<'_>,
) -> ::anyhow::Result<Option<Box<dyn ScratchyWeights>>> {
    let handle = GpuWeightsHandle::new(gw);
    // Walk every registration that claims this HF arch identifier
    // for this TP world size. `Ok(Some(_))` and `Err(_)` terminate;
    // `Ok(None)` (this registration's variants all rejected the
    // live `GpuWeights`) falls through to the next claimant —
    // required when more than one registration claims the same
    // HF arch (e.g. the `deepseek-v3` LoRA-Q variants alongside the
    // `deepseek-v3-flat` direct-Q variants for `DeepseekV3ForCausalLM`
    // checkpoints with `q_lora_rank=null`, and the `mistral` arch
    // claiming `LlamaForCausalLM` as an alias for GGUFs whose
    // `general.architecture = "llama"` flattens Llama-2 and Mistral
    // together).
    // The inner `transpose` flips `Result<Option<W>>` →
    // `Option<Result<W>>` so `find_map` treats `Ok(None)` as
    // "keep looking" and any other shape as a hit.
    inventory::iter::<ScratchyArchRegistration>()
        .filter(|reg| {
            (reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint))
                && reg.tp_world_size == tp_world_size
        })
        .find_map(|reg| (reg.try_load)(handle, stream, max_model_len, tp_rank, hf).transpose())
        .transpose()
}

/// MM-handle counterpart to [`try_load`]. Walks
/// [`ScratchyMmRegistration`] rows for the same
/// `(arch_hint, tp_world_size)` filter. Returns `Ok(None)`
/// when no MM-capable arch claims the HF identifier — the
/// expected case for text-only checkpoints, where the gpu worker
/// proceeds with `embed_patches: &[]`. Phase D of the
/// multimodal plan lands the first row.
#[cfg(feature = "vision")]
pub fn try_load_mm<A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    stream: LoadStream,
    arch_hint: &str,
    tp_world_size: u8,
    tp_rank: u8,
    max_model_len: usize,
    hf: HfFingerprint<'_>,
) -> ::anyhow::Result<Option<Box<dyn MultimodalForward>>> {
    let handle = GpuWeightsHandle::new(gw);
    inventory::iter::<ScratchyMmRegistration>()
        .filter(|reg| {
            (reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint))
                && reg.tp_world_size == tp_world_size
        })
        .find_map(|reg| (reg.try_load_mm)(handle, stream, max_model_len, tp_rank, hf).transpose())
        .transpose()
}
