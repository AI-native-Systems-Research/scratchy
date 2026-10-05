//! Multi-token-prediction heads: what a loader needs before it loads one, and the lookups from a
//! checkpoint's architecture to the head this build compiled. Compiled without a target too: a
//! server resolves its speculative decoding before it knows it has nothing to serve.

/// What a loader needs to know about a multi-token-prediction head before it loads: where it
/// loads the weights its target lends it ([`crate::LentWeight`]), how many of the target's
/// lm_head rows it reads — its logits' width, a prefix of the target's vocabulary when it drafts
/// over one — and whose head it is. Every compiled variant of the arch agrees (asserted at
/// expansion).
#[derive(Clone, Copy, Debug)]
pub struct HeadRegistration {
    pub embed_tokens: &'static str,
    pub lm_head: &'static str,
    pub lm_head_rows: u32,
    /// The checkpoint architectures of the targets it drafts for: its config is its target's.
    pub drafts_for: &'static [&'static str],
    /// How its repo is named when published apart from its target's (an MLX conversion strips the
    /// head's tensors from the target and publishes them alone): inserted before the target
    /// repo's last `-`-delimited token. `None`: it ships only inside its target's checkpoint.
    pub repo_infix: Option<&'static str>,
}

impl HeadRegistration {
    /// The on-disk prefix the head loads `w` from.
    pub fn prefix(&self, w: crate::LentWeight) -> &'static str {
        match w {
            crate::LentWeight::EmbedTokens => self.embed_tokens,
            crate::LentWeight::LmHead => self.lm_head,
        }
    }

    /// The leading rows of the target's `w` the head reads: every embedding row (an input token
    /// can be any), the lm_head's first [`Self::lm_head_rows`].
    pub fn rows(&self, w: crate::LentWeight) -> Option<usize> {
        match w {
            crate::LentWeight::EmbedTokens => None,
            crate::LentWeight::LmHead => Some(self.lm_head_rows as usize),
        }
    }
}

/// The registration of the multi-token-prediction head `arch_hint` names, if it names one.
pub fn draft_head(arch_hint: &str) -> Option<HeadRegistration> {
    registered()
        .find(|(hf_arches, _)| hf_arches.contains(&arch_hint))
        .map(|(_, head)| head)
}

/// The compiled multi-token-prediction head of a target whose checkpoint architecture is
/// `target_arch`, if this build compiled one.
pub fn head_of(target_arch: &str) -> Option<HeadRegistration> {
    registered()
        .map(|(_, head)| head)
        .find(|head| head.drafts_for.contains(&target_arch))
}

/// Every compiled head, with its arch's checkpoint architectures. A build without a target
/// compiles no model (the registry is a target's, `arch_registry`), so no head.
fn registered() -> impl Iterator<Item = (&'static [&'static str], HeadRegistration)> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "spyre",
        feature = "vision"
    ))]
    return inventory::iter::<crate::arch_registry::ScratchyArchRegistration>()
        .filter_map(|reg| Some((reg.hf_arches, reg.head?)));
    #[cfg(not(any(
        feature = "cuda",
        feature = "metal",
        feature = "spyre",
        feature = "vision"
    )))]
    std::iter::empty()
}

impl HeadRegistration {
    /// The repo of this head published apart from its target's, `target_repo` (`org/X-4bit` →
    /// `org/X-MTP-4bit`); `None` when it is not published apart or the id has no `-` token.
    pub fn repo_of(&self, target_repo: &str) -> Option<String> {
        let (name, last) = target_repo.rsplit_once('-')?;
        Some(format!("{name}{}-{last}", self.repo_infix?))
    }
}
