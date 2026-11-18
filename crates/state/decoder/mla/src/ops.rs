// SPDX-License-Identifier: Apache-2.0
//! The MLA ops' typed signatures, each named as a carrier calls it and stated once with its row
//! law and sharding rule, and the constructor of its state input. A state op has no rank-local
//! form the sharding pass could split it into, so each states `tp = Whole`. A product or sum is
//! written in the order `canonical_mul` / `canonical_add` give a `weights.json` dim (a head count
//! before a per-head sum, the latent before the rope key), which this module's
//! `mla_products_order` pins.

use scratchy_typed_shapes::ops::as_dim;
use scratchy_typed_shapes::{Act, Add, BlockTable, Mul, Rotary, Tokens, W2};

/// One layer's latent cache: opaque to shapes.
#[derive(Clone, Copy)]
pub struct MlaCache;

scratchy_decoder_state::state_op! {
    /// `mla_split(kv_a: [T, R + P]) -> ([T, R], [T, P])`: the compressed KV, then the one rope key
    /// every head shares. Each row splits into its two parts, so both outputs keep operand 0's
    /// rows.
    pub fn mla_split<T, R, P>(x: Act<T, Add<R, P>>) -> (Act<T, R>, Act<T, P>) {
        (as_dim(x), as_dim(x))
    }
    rows = In0, tp = Whole;
}

scratchy_decoder_state::state_op! {
    /// `mla_append(q, latent, k_pe, positions, rotary, mla_cache)`: rope (interleaved) on each
    /// head's rope part of q and on the shared rope key, and the latent row `[latent | roped k_pe]`
    /// written to the layer's cache. Each row keeps its shape: q and the latent pass on, k_pe comes
    /// back roped, so the outputs keep operand 0's rows.
    pub fn mla_append<T, Q, L, P>(
        q: Act<T, Q>,
        latent: Act<T, L>,
        k_pe: Act<T, P>,
        _positions: Tokens<T>,
        _rotary: Rotary,
        _cache: MlaCache,
    ) -> (Act<T, Q>, Act<T, L>, Act<T, P>) {
        (q, latent, k_pe)
    }
    rows = In0, tp = Whole;
}

scratchy_decoder_state::state_op! {
    /// `mla_attention(q, latent, k_pe, kv_b, mla_cache, block_table)`: q is `H` heads of a no-rope
    /// part `Nope` and a rope part `Rope`; `kv_b` decompresses the `R`-wide latent into each head's
    /// no-rope key and `V`-wide value; the shared `k_pe` is one rope part wide. The output is every
    /// head's value, one row per token.
    pub fn mla_attention<T, H, Nope, Rope, R, V>(
        q: Act<T, Mul<H, Add<Nope, Rope>>>,
        _latent: Act<T, R>,
        _k_pe: Act<T, Rope>,
        _kv_b: W2<R, Mul<H, Add<Nope, V>>>,
        _cache: MlaCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<H, V>> {
        as_dim(q)
    }
    rows = M, tp = Whole;
}

/// `mla_cache[layer]`: one layer's latent cache.
pub fn mla_cache() -> MlaCache {
    MlaCache
}

#[cfg(test)]
mod tests {
    use scratchy_forward_compiler_frontend::dim::{Dim, canonical_add, canonical_mul};
    use scratchy_typed_shapes::{DimType, NumAttentionHeads};

    /// The signatures above write each product and sum in the order `canonical_mul` and
    /// `canonical_add` give its `weights.json` dims: a head count before a per-head sum, a head
    /// count before the value width, and the compressed KV before the shared rope key.
    #[test]
    fn mla_products_order() {
        let bound = |name: &str| Dim::Bound(name.to_string());
        let heads = NumAttentionHeads::dim();
        let (nope, rope, v) = (
            bound("qk_nope_head_dim"),
            bound("qk_rope_head_dim"),
            bound("v_head_dim"),
        );
        let latent = bound("kv_lora_rank");
        for (first, second) in [(nope.clone(), rope.clone()), (nope.clone(), v.clone())] {
            let per_head = canonical_add(vec![second.clone(), first.clone()]);
            assert_eq!(per_head, Dim::Add(vec![first, second]));
            assert_eq!(
                canonical_mul(vec![per_head.clone(), heads.clone()]),
                Dim::Mul(vec![heads.clone(), per_head])
            );
        }
        assert_eq!(
            canonical_mul(vec![v.clone(), heads.clone()]),
            Dim::Mul(vec![heads, v])
        );
        assert_eq!(
            canonical_add(vec![rope.clone(), latent.clone()]),
            Dim::Add(vec![latent, rope])
        );
    }
}
