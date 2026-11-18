// SPDX-License-Identifier: Apache-2.0
//! Multi-head latent attention (MLA): the keys and values are decompressed per head from one
//! compressed latent, beside one rope key every head shares. This crate is the decoder state's DSL
//! vocabulary ([`Mla`]) and the typed signatures of its ops and state input ([`ops`]); nothing
//! shared names it, so deleting the crate and its one list entry deletes the decoder state.

use std::num::NonZeroU8;

use scratchy_forward_compiler_frontend::state::{
    Arity, InputName, OpName, OutputCount, SignaturePath, StateInputDecl, StateName, StateOpDecl,
    StateVocabulary,
};

pub mod ops;

/// `mla_split` binds two values: the compressed KV and the shared rope key.
const TWO: NonZeroU8 = NonZeroU8::MIN.saturating_add(1);
/// `mla_append` passes its three rows on: the query, the latent and the roped rope key.
const THREE: NonZeroU8 = TWO.saturating_add(1);

static OPS: [StateOpDecl; 3] = [
    // `(latent, k_pe) = mla_split(kv_a)`.
    StateOpDecl {
        name: OpName::new("mla_split"),
        arity: Arity::new(1),
        outputs: OutputCount::new(TWO),
    },
    // `(q, latent, k_pe) = mla_append(q, latent, k_pe, positions, rotary, mla_cache[layer])`.
    StateOpDecl {
        name: OpName::new("mla_append"),
        arity: Arity::new(6),
        outputs: OutputCount::new(THREE),
    },
    // `mla_attention(q, latent, k_pe, kv_b, mla_cache[layer], block_table)`.
    StateOpDecl {
        name: OpName::new("mla_attention"),
        arity: Arity::new(6),
        outputs: OutputCount::ONE,
    },
];

static INPUTS: [StateInputDecl; 1] = [
    // One layer's latent cache: the compressed KV and the rope key, per token.
    StateInputDecl {
        name: InputName::new("mla_cache"),
    },
];

/// The multi-head latent attention decoder state.
pub struct Mla;

impl StateVocabulary for Mla {
    fn name(&self) -> StateName {
        StateName::new("mla")
    }

    fn ops(&self) -> &'static [StateOpDecl] {
        &OPS
    }

    fn inputs(&self) -> &'static [StateInputDecl] {
        &INPUTS
    }

    fn signatures(&self) -> SignaturePath {
        SignaturePath::new("::scratchy_state_decoder_mla::ops")
    }
}
