// SPDX-License-Identifier: Apache-2.0
//! Cross-arch load registry: the `inventory`-driven [`ArchRegistration`] / [`try_load`] seam
//! every compiled arch registers through, the target-neutral [`ArchFacts`] every compiled arch
//! states, and the multimodal sibling (`ScratchyMmRegistration` / `try_load_mm`).
//!
//! ⭐ THE ROW IS SHARED, THE REGISTRY IS THE TARGET'S. A loaded model is a target's own trait
//! object (`dyn <Target>Weights`, extending [`ScratchyWeights`]), so the loader fn-pointer's
//! return type is the target's, and so is the inventory it is collected in: each target crate
//! wraps [`ArchRegistration`] in a type of its own, collects that, and implements
//! [`TargetRegistration`] for it. [`try_load`] is the one walk every target runs, generic over
//! that wrapper. (A shared generic row cannot be collected per target directly: `inventory`'s
//! `Collect` is a foreign trait, and an impl of it for `ArchRegistration<dyn LocalTrait>` is an
//! orphan.)
//!
//! The per-arch loaders fill the concrete `GpuWeights<A>` weight-store, a type this crate cannot
//! name (it has no backend allocator). The registry fn-pointers therefore take the
//! allocator-erased [`scratchy_tensors::GpuWeightsHandle`]; [`try_load`] is monomorphized over
//! the concrete `GpuWeights<A>` at the worker call site (where `A` is the backend allocator),
//! builds the handle, and passes it through the fn-pointer. The emitted closure recovers the
//! concrete `&mut GpuWeights<A>` from the handle before calling the per-variant `Weights::load`.

#[cfg(feature = "vision")]
use crate::mm::MultimodalForward;
use crate::{HeadRegistration, HfFingerprint, ScratchyWeights};
use scratchy_tensors::{DeviceAllocator, GpuWeightsHandle, LoadStream};

/// The loader fn-pointer of one [`ArchRegistration`], returning the target's loaded model `W`
/// (a `dyn <Target>Weights`). Extracted as an alias so the row doesn't trip clippy's
/// `type_complexity`.
///
/// The weight parameter is the allocator-erased [`GpuWeightsHandle`] — the emitted closure
/// recovers the concrete `&mut GpuWeights<A>` the per-variant `Weights::load` loaders need. The
/// stream parameter is the backend-neutral [`LoadStream`], so the signature is the same for
/// every target.
pub type ArchTryLoadFn<W> = fn(
    GpuWeightsHandle<'_>,
    LoadStream,
    usize, // max_model_len — runtime value (CLI --max-model-len or HF config fallback)
    u8,    // tp_rank — runtime value, must be < tp_world_size
    HfFingerprint<'_>,
) -> ::anyhow::Result<Option<Box<W>>>;

/// One registration per compiled `(arch, tp_world_size)`, inside the target's own row type.
pub struct ArchRegistration<W: ?Sized + 'static> {
    /// The arch's identifier — `"llama"`, `"qwen2"`, … — from the carrier fn name. Used in logs.
    pub arch_name: &'static str,
    /// HF `architectures` strings this arch claims to handle. Harvested from the union of every
    /// compiled model's `config.json` `architectures: [..]` field.
    pub hf_arches: &'static [&'static str],
    /// GGUF `general.architecture` tags this arch claims to handle. Family-level — multiple
    /// forward arches can claim the same gguf tag (`"deepseek2"` covers V2, V3-LoRA, V3-flat;
    /// `"llama"` covers Llama and Mistral). The dispatcher tries each claimant in turn and the
    /// per-variant `fingerprint_matches` discriminates by bounds.
    ///
    /// Populated from the arch's `quantizations.json` `ggml.gguf_arch` field (defaults to
    /// `arch_name` when the entry is bare). Empty for arches without a `"ggml"` preset.
    pub gguf_archs: &'static [&'static str],
    /// Compile-time tensor-parallel world size this registration covers — one registration per
    /// `(arch, tp_world_size)` the build baked. [`try_load`] matches on it, so the runtime tp
    /// size picks the right pre-compiled variant set.
    pub tp_world_size: u8,
    /// Try to load the arch's compiled variants. Internally iterates per-variant fingerprint
    /// sniffs. Returns `Ok(Some(..))` on a variant hit, `Ok(None)` when the arch matched by
    /// name but no compiled variant's fingerprint sniff accepted the live `GpuWeights`, or
    /// `Err(..)` only on a genuine load failure (I/O, shape mismatch inside a matched variant).
    pub try_load: ArchTryLoadFn<W>,
}

impl<W: ?Sized> ArchRegistration<W> {
    /// Whether this row claims the checkpoint architecture `arch_hint` at `tp_world_size`.
    pub fn claims(&self, arch_hint: &str, tp_world_size: u8) -> bool {
        (self.hf_arches.contains(&arch_hint) || self.gguf_archs.contains(&arch_hint))
            && self.tp_world_size == tp_world_size
    }
}

/// A target's registration row: the type its crate collects (`inventory::collect!`), wrapping
/// the shared [`ArchRegistration`] over the target's loaded-model trait object.
pub trait TargetRegistration: inventory::Collect {
    /// The target's loaded model: `dyn <Target>Weights`.
    type Weights: ?Sized + ScratchyWeights + 'static;
    /// The shared row.
    fn arch(&self) -> &ArchRegistration<Self::Weights>;
}

/// What every compiled arch states about itself, whatever target compiled it: the facts a
/// server reads before (or without) loading a model. One per arch.
pub struct ArchFacts {
    /// The arch's identifier, as in its [`ArchRegistration`]s.
    pub arch_name: &'static str,
    /// HF `architectures` strings this arch claims, as in its [`ArchRegistration`]s.
    pub hf_arches: &'static [&'static str],
    /// `Some` for a multi-token-prediction head (its forward reads `target_hidden`): the facts a
    /// loader needs before the head loads. `None` for every other arch.
    pub head: Option<HeadRegistration>,
}

inventory::collect!(ArchFacts);

/// Outcome of [`try_load`]. The two miss cases are DISTINCT, and callers want different
/// diagnostics for them:
///
/// * [`ArchLoad::ArchNotCompiled`] — nothing claims this HF arch at this tp size. The build
///   genuinely has no implementation, so "unsupported arch" is the honest answer.
/// * [`ArchLoad::NoVariantMatched`] — the arch IS compiled, but no compiled variant's
///   fingerprint accepted the live `GpuWeights`: none of its `(stem, preset)` pairs describes
///   THIS checkpoint. Reporting that as "unsupported arch" sends the reader hunting for a
///   missing backend when the actual fix is the build's model/quant scope or a
///   `quantizations.json` entry.
///
/// Returning them as data rather than a collapsed `Option` is what lets every caller map each to
/// its own error without re-deriving the registry filter.
pub enum ArchLoad<W: ?Sized> {
    Loaded(Box<W>),
    ArchNotCompiled,
    NoVariantMatched,
}

/// Top-level architecture loader over the target registry `R`. Walks every row of `R`; the first
/// whose `hf_arches` list contains `arch_hint` AND whose `tp_world_size` matches the runtime
/// `tp_world_size` wins and attempts to load. See [`ArchLoad`] for the two miss cases.
///
/// Generic over the backend allocator `A`: the worker passes its concrete `&mut GpuWeights<A>`
/// and `A` is inferred. The borrow is wrapped in a [`GpuWeightsHandle`] (allocator-erased)
/// before being threaded through the fn-pointer; the emitted closure recovers the same concrete
/// type.
pub fn try_load<R: TargetRegistration, A: DeviceAllocator>(
    gw: &mut scratchy_layers::weights::GpuWeights<A>,
    stream: LoadStream,
    arch_hint: &str,
    tp_world_size: u8,
    tp_rank: u8,
    max_model_len: usize,
    hf: HfFingerprint<'_>,
) -> ::anyhow::Result<ArchLoad<R::Weights>> {
    let handle = GpuWeightsHandle::new(gw);
    // Walk every registration that claims this HF arch identifier for this TP world size. A
    // variant hit and an `Err` terminate; a registration whose variants all rejected the live
    // `GpuWeights` falls through to the next claimant — required when more than one
    // registration claims the same HF arch (e.g. the `deepseek-v3` LoRA-Q variants alongside the
    // `deepseek-v3-flat` direct-Q variants for `DeepseekV3ForCausalLM` checkpoints with
    // `q_lora_rank=null`, and the `mistral` arch claiming `LlamaForCausalLM` as an alias for
    // GGUFs whose `general.architecture = "llama"` flattens Llama-2 and Mistral together).
    //
    // `claimed` is what separates the two miss cases: it records that SOME registration owned
    // the arch name even though every one of its variants declined the checkpoint.
    let mut claimed = false;
    for reg in inventory::iter::<R>()
        .map(TargetRegistration::arch)
        .filter(|reg| reg.claims(arch_hint, tp_world_size))
    {
        claimed = true;
        if let Some(weights) = (reg.try_load)(handle, stream, max_model_len, tp_rank, hf)? {
            return Ok(ArchLoad::Loaded(weights));
        }
    }
    Ok(if claimed {
        ArchLoad::NoVariantMatched
    } else {
        ArchLoad::ArchNotCompiled
    })
}

// ── Multimodal sibling surface ────────────────────────────────
//
// `MultimodalForward` is a sibling trait, NOT a default-method
// extension on `ScratchyWeights`. Text-only arches don't
// implement it. Discovery uses a sibling inventory row so the
// text-side `ArchRegistration` stays untouched and
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
/// as [`ArchRegistration`] but the keys typically only
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
