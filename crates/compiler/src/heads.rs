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
    /// The architecture it registers under (its arch.json `hf_architectures`): what a loader
    /// dispatches it by, wherever its tensors are found.
    pub arch: &'static str,
    pub embed_tokens: &'static str,
    /// `None`: the head's logits are its own (Gemma 4's assistant: its tied embedding).
    pub lm_head: Option<&'static str>,
    pub lm_head_rows: u32,
    /// Its attention reads its target's KV and writes none (Gemma 4's assistant): it has no KV pool
    /// of its own, its query sits at the position after the row it reads, and its chained drafts
    /// keep that position.
    pub reads_target_kv: bool,
    /// The checkpoint architectures of the targets it drafts for: its config is its target's.
    pub drafts_for: &'static [&'static str],
    /// How its repo is named when published apart from its target's (an MLX conversion strips the
    /// head's tensors from the target and publishes them alone): inserted before the target
    /// repo's last `-`-delimited token. `None`: it ships only inside its target's checkpoint.
    pub repo_infix: Option<&'static str>,
    /// Where its tensors sit inside its target's checkpoint (`language_model.mtp`), for a head
    /// that ships there: a target checkpoint holding tensors under it carries the head, which
    /// loads from the target's own store. `None`: it never ships inside its target's checkpoint.
    pub checkpoint_prefix: Option<&'static str>,
    /// The drafters of the arch published as checkpoints of their own that their targets draft
    /// with by default (Gemma 4's assistant: `published_as`, its arch.json), one a target shape.
    /// Empty for a head that ships inside its target's checkpoint.
    pub published: &'static [PublishedDrafter],
    /// The most tokens it drafts a sequence each step (`spec_drafts`, its arch.json): its target's
    /// verify steps carry up to them, as both models' kernels bake them.
    pub drafts: u8,
    /// The most rows a verify step of its target holds (`spec_verify_rows`, its arch.json): each
    /// sequence's token and its drafts, which a step of more sequences shrinks to fit, drafting
    /// nothing once one draft a sequence does not (`scratchy_serving_engine::spec_decode::
    /// DraftBudget`).
    ///
    /// TODO: one budget per declared device profile and target, computed at expansion from the
    /// verify and plain tapes' costs, in place of one count for every GPU and every target of the
    /// arch (`scratchy_serving_engine::spec_decode::mtp`'s module docs: what it is measured at,
    /// and what blocks it).
    pub verify_rows: u8,
}

/// A drafter published as a checkpoint of its own that its target drafts with by default: no
/// checkpoint of the target carries it (Google ships Gemma 4's assistant only as a repo of its
/// own), so the compiled drafter names the repo it loads from.
#[derive(Clone, Copy, Debug)]
pub struct PublishedDrafter {
    /// The Hub repo it loads from (`guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit`).
    pub repo: &'static str,
    /// The shape of the target config it compiled against.
    pub target: TargetShape,
}

/// What tells a target's checkpoints apart from the others of its architecture before a load
/// (Gemma 4's 26B-A4B and 31B are both `Gemma4ForConditionalGeneration`): its text backbone's
/// width and depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetShape {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
}

impl HeadRegistration {
    /// The on-disk prefix the head loads `w` from, when it borrows `w`.
    pub fn prefix(&self, w: crate::LentWeight) -> Option<&'static str> {
        match w {
            crate::LentWeight::EmbedTokens => Some(self.embed_tokens),
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

    /// The repo of the drafter a target of `shape` drafts with by default, when this build
    /// compiled one ([`Self::published`]).
    pub fn published_for(&self, shape: TargetShape) -> Option<&'static str> {
        (self.published.iter())
            .find(|drafter| drafter.target == shape)
            .map(|drafter| drafter.repo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma 4's assistant drafts by default only for the target shape it compiled against: 31B
    /// shares 26B-A4B's architecture, not its shape.
    #[test]
    fn a_published_drafter_drafts_only_for_its_targets_shape() {
        const GEMMA_26B: TargetShape = TargetShape {
            hidden_size: 2816,
            num_hidden_layers: 30,
        };
        let head = HeadRegistration {
            arch: "Gemma4AssistantForCausalLM",
            embed_tokens: "model.target_embed_tokens",
            lm_head: None,
            lm_head_rows: 262_144,
            reads_target_kv: true,
            drafts_for: &["Gemma4ForConditionalGeneration"],
            repo_infix: None,
            checkpoint_prefix: None,
            published: &[PublishedDrafter {
                repo: "guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit",
                target: GEMMA_26B,
            }],
            drafts: 2,
            verify_rows: 3,
        };
        assert_eq!(
            head.published_for(GEMMA_26B),
            Some("guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit")
        );
        let gemma_31b = TargetShape {
            hidden_size: 5376,
            num_hidden_layers: 60,
        };
        assert_eq!(head.published_for(gemma_31b), None);
    }
}
