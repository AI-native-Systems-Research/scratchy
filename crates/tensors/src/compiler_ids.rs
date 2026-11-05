// SPDX-License-Identifier: Apache-2.0
//! The forward compiler's small shared identities: a weight's id, its per-layer unroll index, and
//! a host-staged row source. Here, beneath every crate that names them (`scratchy-ir`,
//! `scratchy-subtile`, the macros crate), so the macros crate's build script, which compiles the
//! crate's front end by path, reaches them without the rest of the compiler. Each crate that used to
//! define one re-exports it at its old path: one type each.

/// A weight tensor's identity in the model's weight table. The same id spans every layer;
/// [`UnrollIndex`] picks which one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(pub u32);

/// The unrolled (former loop-variable) position of a weight reference. `Some(UnrollIndex(n))`
/// names the `n`th per-layer instance of a layered weight; `None` a global tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnrollIndex(pub u64);

impl std::fmt::Display for UnrollIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A runtime tensor of `[num_tokens, width]` rows a forward reads as an input, delivered by the
/// host before the forward runs — what `Instruction::LoadRows` copies into a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowsExtern {
    /// Vision patches, `[num_tokens, vision_in_features]`.
    Pixels,
    /// Qwen3.5-VL's learned positional embedding, interpolated on the host,
    /// `[num_tokens, vision_embed_dim]`.
    PosEmbeds,
    /// Another forward's final hidden states, `[num_tokens, hidden_size]` — what a
    /// multi-token-prediction head reads from its target model.
    TargetHidden,
}

impl RowsExtern {
    /// Every source, in the order a forward materializes the ones it reads.
    pub const ALL: [Self; 3] = [Self::Pixels, Self::PosEmbeds, Self::TargetHidden];
}
