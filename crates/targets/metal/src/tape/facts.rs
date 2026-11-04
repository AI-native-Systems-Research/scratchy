// SPDX-License-Identifier: Apache-2.0
//! The optional groups of a model's facts, as the lowering reads them.
//!
//! The lowering takes a canonical's [`ModelFacts`]. A step that needs a group the model lacks is
//! [`LoweringError::ModelLacks`] naming it, never a zero read in its place.

use super::lowered::{Lacked, LoweringError};
use scratchy_ir::model_facts::{
    Attention, AttnClass, AttnWindow, Drafter, Gdn, HeadGeometry, IntermediateSize, KvRope,
    ModelFacts, SoftCap, Target, Vision, VocabSize,
};

/// The decoder's attention.
pub(crate) fn attention(p: &ModelFacts) -> Result<Attention, LoweringError> {
    p.attention()
        .ok_or(LoweringError::ModelLacks(Lacked::Attention))
}

/// The sliding layers' window.
pub(crate) fn window(p: &ModelFacts) -> Result<AttnWindow, LoweringError> {
    (attention(p)?.window()).ok_or(LoweringError::ModelLacks(Lacked::SlidingWindow))
}

/// The vision tower.
pub(crate) fn vision(p: &ModelFacts) -> Result<Vision, LoweringError> {
    p.vision().ok_or(LoweringError::ModelLacks(Lacked::Vision))
}

/// The Gated-DeltaNet layers.
pub(crate) fn gdn(p: &ModelFacts) -> Result<Gdn, LoweringError> {
    p.gdn().ok_or(LoweringError::ModelLacks(Lacked::Gdn))
}

/// The DFlash2 drafter's geometry.
pub(crate) fn drafter(p: &ModelFacts) -> Result<Drafter, LoweringError> {
    p.drafter()
        .ok_or(LoweringError::ModelLacks(Lacked::Drafter))
}

/// The feed-forward's inner width.
pub(crate) fn intermediate(p: &ModelFacts) -> Result<IntermediateSize, LoweringError> {
    p.intermediate()
        .ok_or(LoweringError::ModelLacks(Lacked::Intermediate))
}

/// The lm_head's vocabulary.
pub(crate) fn vocab(p: &ModelFacts) -> Result<VocabSize, LoweringError> {
    p.vocab().ok_or(LoweringError::ModelLacks(Lacked::Vocab))
}

/// The final logits' soft cap.
pub(crate) fn final_softcap(p: &ModelFacts) -> Result<SoftCap, LoweringError> {
    p.final_softcap()
        .ok_or(LoweringError::ModelLacks(Lacked::FinalSoftcap))
}

/// The target a head reads.
pub(crate) fn target(p: &ModelFacts) -> Result<Target, LoweringError> {
    p.target().ok_or(LoweringError::ModelLacks(Lacked::Target))
}

/// The drafts a verify step carries, as a kernel's draft-count constant spells them: the kernels'
/// ABI writes "no drafts" as 0. Only at that seam is the absent fact a number.
pub(crate) fn kernel_drafts(p: &ModelFacts) -> u32 {
    p.spec_drafts().map_or(0, |d| d.get())
}

/// The attention's base class: every layer's on a uniform model, the sliding layers' on a hybrid.
pub(crate) fn base(p: &ModelFacts) -> Result<HeadGeometry, LoweringError> {
    attention(p).map(|a| a.base())
}

/// The attention's global class (equal to the base on a uniform model).
pub(crate) fn global(p: &ModelFacts) -> Result<HeadGeometry, LoweringError> {
    attention(p).map(|a| a.global())
}

/// The query heads every attention layer projects.
pub(crate) fn q_heads(p: &ModelFacts) -> Result<u32, LoweringError> {
    attention(p).map(|a| a.q_heads().get())
}

/// Whether the attention re-ropes its keys on each read of the cache.
pub(crate) fn rope_on_read(p: &ModelFacts) -> Result<bool, LoweringError> {
    attention(p).map(|a| a.kv_rope() == KvRope::OnRead)
}

/// The attention class a layer of `class` reads.
pub(crate) fn class(p: &ModelFacts, class: AttnClass) -> Result<HeadGeometry, LoweringError> {
    attention(p).map(|a| a.class(class))
}

/// Whether the global class's rope pairs across the head's halves (Gemma 4's proportional rope).
pub(crate) fn proportional(p: &ModelFacts) -> Result<bool, LoweringError> {
    attention(p).map(|a| a.pairing() == scratchy_ir::model_facts::RopePairing::Proportional)
}

/// The sliding window as the attention kernels' `ATTN_WINDOW` constant spells it: a signed `int`,
/// whose 0 (no window) only a full layer passes. The window is a positive count read from the
/// config's signed bound, so it fits.
pub(crate) fn kernel_window(p: &ModelFacts) -> Result<i32, LoweringError> {
    window(p).map(|w| w.get() as i32)
}
