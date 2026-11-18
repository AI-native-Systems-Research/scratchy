// SPDX-License-Identifier: Apache-2.0
//! Multi-head latent attention (MLA): the keys and values are decompressed per head from one
//! compressed latent, beside one rope key every head shares. This crate is the decoder state
//! ([`MLA`]), the typed signatures of its DSL ops and state input ([`ops`]), each op stated once
//! there, their lowerings ([`lower`]) and the subtile ops those push ([`MLA_LATENT`] and the rows
//! beside it); nothing shared names it, so deleting the crate and its one list entry deletes the
//! decoder state.

// The signatures path is absolute, as the typed carriers call it from another crate; the decoder state
// reaches its own ops through the same path.
extern crate self as scratchy_state_decoder_mla;

pub mod lower;
pub mod ops;

use scratchy_decoder_state::{DeriveStage, DerivedDim, DerivedRule, StateName};

/// The decoder state's name, which each of its subtile ops is stated under.
pub const STATE: StateName = StateName::new("mla");

/// The per-head query/key width: the no-rope part and the rope part. Derived before the fallbacks,
/// so it wins over hidden over heads; a config stating `head_dim` wins over it.
pub const HEAD_DIM: DerivedDim = DerivedDim {
    name: "head_dim",
    stage: DeriveStage::BeforeFallbacks,
    rule: DerivedRule::Add(&[
        DerivedRule::Bound("qk_nope_head_dim"),
        DerivedRule::Bound("qk_rope_head_dim"),
    ]),
};

/// q_proj's (or q_b_proj's, under Q-LoRA) output: per head, the no-rope and rope parts of a query.
pub const Q_PROJ_OUT: DerivedDim = DerivedDim {
    name: "q_proj_out",
    stage: DeriveStage::Implicit,
    rule: DerivedRule::Mul(&[
        DerivedRule::Bound("num_attention_heads"),
        DerivedRule::Add(&[
            DerivedRule::Bound("qk_nope_head_dim"),
            DerivedRule::Bound("qk_rope_head_dim"),
        ]),
    ]),
};

/// kv_b_proj's decompressed output: per head, a key's no-rope part and a value.
pub const KV_LORA_OUT: DerivedDim = DerivedDim {
    name: "kv_lora_out",
    stage: DeriveStage::Implicit,
    rule: DerivedRule::Mul(&[
        DerivedRule::Bound("num_attention_heads"),
        DerivedRule::Add(&[
            DerivedRule::Bound("qk_nope_head_dim"),
            DerivedRule::Bound("v_head_dim"),
        ]),
    ]),
};

/// o_proj's input: every head's value.
pub const ATTN_OUT: DerivedDim = DerivedDim {
    name: "attn_out",
    stage: DeriveStage::Implicit,
    rule: DerivedRule::Mul(&[
        DerivedRule::Bound("num_attention_heads"),
        DerivedRule::Bound("v_head_dim"),
    ]),
};

/// kv_a_proj_with_mqa's output: the compressed KV, then the one shared rope key.
pub const KV_A_PROJ_OUT: DerivedDim = DerivedDim {
    name: "kv_a_proj_out",
    stage: DeriveStage::Implicit,
    rule: DerivedRule::Add(&[
        DerivedRule::Bound("kv_lora_rank"),
        DerivedRule::Bound("qk_rope_head_dim"),
    ]),
};

scratchy_decoder_state::subtile_op! {
    /// The latent: the first out-width columns of its operand, the compressed KV of
    /// kv_a_proj_with_mqa's output.
    pub static MLA_LATENT: name = mla_latent, state = STATE, arity = 1,
        rows = In0, tp = Whole, weight = none;
    /// The shared rope key: the last out-width columns of its operand, the rope part of
    /// kv_a_proj_with_mqa's output.
    pub static MLA_ROPE_KEY: name = mla_rope_key, state = STATE, arity = 1,
        rows = In0, tp = Whole, weight = none;
    /// `[q, cos, sin]`: q with each head's rope part roped by the cos/sin pair, which is every
    /// head's rope part wide (heads times the rope width). The output is q's own width.
    pub static MLA_ROPE_Q: name = mla_rope_q, state = STATE, arity = 3,
        rows = In0, tp = Whole, weight = none;
    /// `[latent, k_pe, cos, sin]`: k_pe roped by the cos/sin pair, one rope part wide, and written
    /// with the latent as the row `[latent | roped k_pe]` into the cache of the op's layer. Its
    /// output is the roped k_pe.
    pub static MLA_CACHE_WRITE: name = mla_cache_write, state = STATE, arity = 4,
        rows = In0, tp = Whole, weight = none;
    /// `[q, latent, k_pe, kv_b]`: attention over the cache of the op's layer, with `kv_b`, its
    /// operand 3, decompressing the latent into each head's no-rope key and value. One row per
    /// token, every head's value.
    pub static MLA_ATTEND: name = mla_attend, state = STATE, arity = 4,
        rows = M, tp = Whole, weight = 3;
}

scratchy_decoder_state::decoder_state! {
    /// The multi-head latent attention decoder state: its three DSL ops and their lowerings, one
    /// layer's latent cache as its state input, the five subtile ops the lowerings push, and the
    /// projection widths the latent attention's weights are read at.
    pub static MLA: name = STATE, signatures = ::scratchy_state_decoder_mla::ops,
        ops = [
            mla_split => lower::split,
            mla_append => lower::append,
            mla_attention => lower::attention,
        ],
        inputs = [mla_cache],
        subtile = [MLA_LATENT, MLA_ROPE_KEY, MLA_ROPE_Q, MLA_CACHE_WRITE, MLA_ATTEND],
        dims = [HEAD_DIM, Q_PROJ_OUT, KV_LORA_OUT, ATTN_OUT, KV_A_PROJ_OUT];
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU8;

    use scratchy_forward_compiler_frontend::state::{
        Arity, InputName, OpName, OutputCount, SignaturePath, StateName,
    };
    use scratchy_subtile::state_op::StateRows;
    use scratchy_subtile::tensor_parallel::TpRule;

    use super::MLA;

    /// Each DSL op's arity and output count, counted from its signature, are the ones a carrier's
    /// call binds, and each has a lowering. Each subtile op states its own facts, under the
    /// decoder state's name.
    #[test]
    fn mla_states_each_op_once() {
        let count = |n: u8| OutputCount::new(NonZeroU8::MIN.saturating_add(n - 1));
        let vocabulary = MLA.vocabulary();
        assert_eq!(vocabulary.name(), StateName::new("mla"));
        assert_eq!(
            vocabulary.signatures(),
            SignaturePath::new("::scratchy_state_decoder_mla::ops")
        );
        let dsl = [
            ("mla_split", 1, 2),
            ("mla_append", 6, 3),
            ("mla_attention", 6, 1),
        ];
        assert_eq!(vocabulary.ops().len(), dsl.len());
        for (decl, (name, arity, outputs)) in vocabulary.ops().iter().zip(dsl) {
            assert_eq!(decl.name, OpName::new(name));
            assert_eq!(decl.arity, Arity::new(arity));
            assert_eq!(decl.outputs, count(outputs));
            assert!(MLA.lowering(decl.name).is_some(), "{name} has a lowering");
            assert!(MLA.subtile(decl.name).is_none(), "{name} is a DSL op");
        }
        let subtile = [
            ("mla_latent", 1, StateRows::In0, None),
            ("mla_rope_key", 1, StateRows::In0, None),
            ("mla_rope_q", 3, StateRows::In0, None),
            ("mla_cache_write", 4, StateRows::In0, None),
            ("mla_attend", 4, StateRows::M, Some(3)),
        ];
        for (name, arity, rows, weight) in subtile {
            let facts = MLA
                .subtile(OpName::new(name))
                .unwrap_or_else(|| panic!("{name} states its facts"));
            assert_eq!(facts.id().state(), StateName::new("mla"));
            assert_eq!(facts.id().name(), OpName::new(name));
            assert_eq!(facts.arity(), Arity::new(arity));
            assert_eq!(facts.outputs(), OutputCount::ONE);
            assert_eq!(facts.rows(), rows);
            assert_eq!(facts.tp(), TpRule::Whole);
            assert_eq!(facts.weight(), weight);
            assert!(
                MLA.lowering(OpName::new(name)).is_none(),
                "{name} is a subtile op"
            );
        }
        let [input] = vocabulary.inputs() else {
            panic!("MLA declares one state input");
        };
        assert_eq!(input.name, InputName::new("mla_cache"));
        assert!(MLA.subtile(OpName::new("mla_cache")).is_none());
        let dims: Vec<_> = vocabulary.derived_dims().iter().map(|d| d.name).collect();
        assert_eq!(
            dims,
            [
                "head_dim",
                "q_proj_out",
                "kv_lora_out",
                "attn_out",
                "kv_a_proj_out"
            ]
        );
    }
}
