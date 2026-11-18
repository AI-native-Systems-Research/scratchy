// SPDX-License-Identifier: Apache-2.0
//! The forward compiler's small shared identities: a weight's id, its per-layer unroll index, and
//! a host-staged row source. Here, beneath every crate that names them (`scratchy-ir`,
//! `scratchy-subtile`, the macros crate), so the macros crate's build script, which compiles the
//! crate's front end by path, reaches them without the rest of the compiler. Each crate that used to
//! define one re-exports it at its old path: one type each. A state op's identity and the counts
//! its declaration states ([`Arity`], [`OutputCount`]) live here too, so the front end that
//! classifies a call and the subtile op that carries the op's facts read the same types.

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

/// A `&'static str` newtype minted only by its `const fn new`.
#[macro_export]
macro_rules! static_name {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(&'static str);

        impl $name {
            pub const fn new(name: &'static str) -> Self {
                Self(name)
            }

            pub const fn get(self) -> &'static str {
                self.0
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.0)
            }
        }

        /// A table keyed by the name is looked up by the text a body writes.
        impl ::std::borrow::Borrow<str> for $name {
            fn borrow(&self) -> &str {
                self.0
            }
        }
    };
}

static_name!(
    /// The name a DSL body calls an op by.
    OpName
);
static_name!(
    /// The name a DSL body reads a decoder state's input by.
    InputName
);
static_name!(
    /// A decoder state's name, as its diagnostics print it.
    StateName
);

impl StateName {
    /// Whether the two names are the same text, compared byte by byte so a `const` can ask.
    pub const fn same_as(self, other: Self) -> bool {
        let (a, b) = (self.0.as_bytes(), other.0.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }
}

/// The number of arguments an op takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Arity(u8);

impl Arity {
    pub const fn new(args: u8) -> Self {
        Self(args)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// The number of values an op produces: at least one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputCount(std::num::NonZeroU8);

impl OutputCount {
    /// An op that produces one value.
    pub const ONE: Self = Self(std::num::NonZeroU8::MIN);

    pub const fn new(outputs: std::num::NonZeroU8) -> Self {
        Self(outputs)
    }

    pub const fn get(self) -> std::num::NonZeroU8 {
        self.0
    }
}

/// A state op's identity: its decoder state and its name, the pair that tells it from every other op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateOpId {
    state: StateName,
    op: OpName,
}

impl StateOpId {
    pub const fn new(state: StateName, op: OpName) -> Self {
        Self { state, op }
    }

    pub const fn state(self) -> StateName {
        self.state
    }

    pub const fn name(self) -> OpName {
        self.op
    }
}

/// A state input's identity: its decoder state and its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StateInputId {
    state: StateName,
    input: InputName,
}

impl StateInputId {
    pub const fn new(state: StateName, input: InputName) -> Self {
        Self { state, input }
    }

    pub const fn state(self) -> StateName {
        self.state
    }

    pub const fn name(self) -> InputName {
        self.input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_state_op_id_is_its_state_and_name() {
        let id = StateOpId::new(StateName::new("r"), OpName::new("op"));
        assert_eq!(id, StateOpId::new(StateName::new("r"), OpName::new("op")));
        assert_ne!(id, StateOpId::new(StateName::new("s"), OpName::new("op")));
        assert_ne!(
            id,
            StateOpId::new(StateName::new("r"), OpName::new("other"))
        );
        assert_eq!(id.state(), StateName::new("r"));
        assert_eq!(id.name(), OpName::new("op"));

        let input = StateInputId::new(StateName::new("r"), InputName::new("state"));
        assert_ne!(
            input,
            StateInputId::new(StateName::new("s"), InputName::new("state"))
        );
        assert_eq!(input.name(), InputName::new("state"));
    }

    #[test]
    fn a_state_name_is_const_constructible() {
        const STATE: StateName = StateName::new("r");
        const OP: OpName = OpName::new("op");
        const INPUT: InputName = InputName::new("state");
        const ID: StateOpId = StateOpId::new(STATE, OP);
        const INPUT_ID: StateInputId = StateInputId::new(STATE, INPUT);
        assert_eq!((STATE.get(), OP.get(), INPUT.get()), ("r", "op", "state"));
        assert_eq!((ID.state(), INPUT_ID.name()), (STATE, INPUT));
    }

    #[test]
    fn a_const_compares_state_names_by_text() {
        const {
            assert!(StateName::new("mixer").same_as(StateName::new("mixer")));
            assert!(!StateName::new("mix").same_as(StateName::new("mixer")));
            assert!(!StateName::new("mixer").same_as(StateName::new("mixed")));
        }
    }
}
