// SPDX-License-Identifier: Apache-2.0
//! How the MLA ops lower into the subtile IR: each DSL op becomes single-output subtile ops
//! ([`crate::MLA_LATENT`] and the other rows in the crate root), and each DSL output that passes a
//! value on points at that value's producer. A lowering reads widths only through `out_cols` and
//! config bounds only through `bound`, so it restates no shape the carrier already fixed.

use scratchy_decoder_state::lower::{
    ArgIndex, LowerError, OutIndex, StateLowering, StateSubOp, StateWeight,
};

use crate::{MLA_ATTEND, MLA_CACHE_WRITE, MLA_LATENT, MLA_ROPE_KEY, MLA_ROPE_Q};

/// `mla_split(kv_a)`: the latent is the first out-0-wide columns of `kv_a`, the rope key the last
/// out-1-wide, one subtile op each.
pub fn split(l: &mut dyn StateLowering) -> Result<(), LowerError> {
    let x = l.input(ArgIndex::new(0))?;
    for (facts, out) in [
        (&MLA_LATENT, OutIndex::new(0)),
        (&MLA_ROPE_KEY, OutIndex::new(1)),
    ] {
        let cols = l.out_cols(out)?;
        let part = l.push(StateSubOp::new(facts, |_| cols), vec![x])?;
        l.produce(out, part);
    }
    Ok(())
}

/// `mla_append(q, latent, k_pe, positions, rotary, mla_cache)`: q's rope parts are roped over a
/// cos/sin pair as wide as every head's rope part; the latent passes on as its own producer; k_pe
/// is roped over a pair one rope part wide and written beside the latent into the cache of the
/// layer `mla_cache` names. Positions and rotary reach the ops only through the cos/sin pairs.
pub fn append(l: &mut dyn StateLowering) -> Result<(), LowerError> {
    let q = l.input(ArgIndex::new(0))?;
    let latent = l.input(ArgIndex::new(1))?;
    let k_pe = l.input(ArgIndex::new(2))?;
    let rope = l.bound("qk_rope_head_dim")?;
    let heads = l.bound("num_attention_heads")?;

    let q_rope = heads.checked_mul(rope).ok_or(LowerError::WidthOverflow {
        what: "every head's rope part",
    })?;
    let (cos_q, sin_q) = l.cos_sin(q_rope);
    let q_cols = l.out_cols(OutIndex::new(0))?;
    let roped_q = l.push(
        StateSubOp::new(&MLA_ROPE_Q, |_| q_cols),
        vec![q, cos_q, sin_q],
    )?;
    l.produce(OutIndex::new(0), roped_q);

    l.produce(OutIndex::new(1), latent);

    let (cos_k, sin_k) = l.cos_sin(rope);
    let k_cols = l.out_cols(OutIndex::new(2))?;
    let layer = l.layer(ArgIndex::new(5))?;
    let written = l.push(
        StateSubOp::new(&MLA_CACHE_WRITE, |_| k_cols).at_layer(layer),
        vec![latent, k_pe, cos_k, sin_k],
    )?;
    l.produce(OutIndex::new(2), written);
    Ok(())
}

/// `mla_attention(q, latent, k_pe, kv_b, mla_cache, block_table)`: one subtile op over q, the
/// latent, k_pe and the `kv_b` weight, reading the cache of the layer `mla_cache` names. The block
/// table reaches it through that layer's cache, not as an operand.
pub fn attention(l: &mut dyn StateLowering) -> Result<(), LowerError> {
    let q = l.input(ArgIndex::new(0))?;
    let latent = l.input(ArgIndex::new(1))?;
    let k_pe = l.input(ArgIndex::new(2))?;
    let StateWeight { source, scheme } = l.weight(ArgIndex::new(3))?;
    let layer = l.layer(ArgIndex::new(4))?;
    let cols = l.out_cols(OutIndex::new(0))?;
    let attended = l.push(
        StateSubOp::new(&MLA_ATTEND, |_| cols)
            .at_layer(layer)
            .weighted(scheme),
        vec![q, latent, k_pe, source],
    )?;
    l.produce(OutIndex::new(0), attended);
    Ok(())
}

#[cfg(test)]
mod tests {
    use scratchy_decoder_state::lower::{
        ArgIndex, GemmWeight, InputRef, LowerError, OutIndex, Recorder, StateLowering, StateWeight,
        UnrollIndex,
    };

    use super::{append, attention, split};
    use crate::{MLA, MLA_ATTEND, MLA_CACHE_WRITE, MLA_LATENT, MLA_ROPE_KEY, MLA_ROPE_Q};

    const HEADS: u32 = 4;
    const NOPE: u32 = 16;
    const ROPE: u32 = 8;
    const LATENT: u32 = 64;
    const VALUE: u32 = 32;

    fn mla_bounds(r: Recorder) -> Recorder {
        r.with_bound("num_attention_heads", HEADS)
            .with_bound("qk_rope_head_dim", ROPE)
    }

    /// The latent and the rope key are two slices of the one operand, each as wide as its output.
    #[test]
    fn split_is_two_slices() {
        let x = InputRef::Op(3);
        let mut r = Recorder::new(MLA, vec![x], vec![LATENT, ROPE]);
        split(&mut r).expect("split lowers");
        let [(latent, latent_in), (key, key_in)] = r.pushes() else {
            panic!("split pushes two ops");
        };
        assert_eq!(latent.id(), MLA_LATENT.id());
        assert_eq!(latent.out_cols().first(), LATENT);
        assert_eq!(latent_in, &[x]);
        assert_eq!(key.id(), MLA_ROPE_KEY.id());
        assert_eq!(key.out_cols().first(), ROPE);
        assert_eq!(key_in, &[x]);
        assert_eq!(
            r.produces(),
            [
                (OutIndex::new(0), InputRef::Op(0)),
                (OutIndex::new(1), InputRef::Op(1))
            ]
        );
    }

    /// q is roped over every head's rope part, the cache write ropes the one shared key at the
    /// cache's layer, and the latent output is the latent input's own producer, so it adds no op.
    #[test]
    fn append_ropes_q_and_writes_the_cache() {
        let (q, latent, k_pe) = (InputRef::Op(1), InputRef::Op(2), InputRef::Ext(0));
        let q_cols = HEADS * (NOPE + ROPE);
        let mut r = mla_bounds(Recorder::new(
            MLA,
            vec![q, latent, k_pe],
            vec![q_cols, LATENT, ROPE],
        ))
        .with_layer(ArgIndex::new(5), UnrollIndex(5));
        append(&mut r).expect("append lowers");
        let [(q_pair, cos_q, sin_q), (k_pair, cos_k, sin_k)] = *r.cos_sins() else {
            panic!("append asks for two cos/sin pairs");
        };
        assert_eq!(q_pair, HEADS * ROPE);
        assert_eq!(k_pair, ROPE);
        let [(roped, roped_in), (write, write_in)] = r.pushes() else {
            panic!("append pushes two ops");
        };
        assert_eq!(roped.id(), MLA_ROPE_Q.id());
        assert_eq!(roped.out_cols().first(), q_cols);
        assert_eq!(roped.layer(), None);
        assert_eq!(roped_in, &[q, cos_q, sin_q]);
        assert_eq!(write.id(), MLA_CACHE_WRITE.id());
        assert_eq!(write.out_cols().first(), ROPE);
        assert_eq!(write.layer(), Some(UnrollIndex(5)));
        assert_eq!(write_in, &[latent, k_pe, cos_k, sin_k]);
        assert_eq!(
            r.produces(),
            [
                (OutIndex::new(0), InputRef::Op(0)),
                (OutIndex::new(1), latent),
                (OutIndex::new(2), InputRef::Op(1))
            ]
        );
    }

    /// A rope width past a `u32` is refused at expansion, before any op is pushed.
    #[test]
    fn append_refuses_a_rope_width_past_u32() {
        let mut r = Recorder::new(MLA, vec![InputRef::Op(0); 3], vec![1, 1, 1])
            .with_bound("num_attention_heads", u32::MAX)
            .with_bound("qk_rope_head_dim", 2);
        assert!(matches!(
            append(&mut r),
            Err(LowerError::WidthOverflow { .. })
        ));
        assert!(r.pushes().is_empty());
    }

    scratchy_decoder_state::subtile_op! {
        /// A row named as MLA's latent, under MLA's name, that MLA does not list.
        static UNLISTED_LATENT: name = mla_latent, state = crate::STATE, arity = 1,
            rows = In0, tp = Whole, weight = none;
    }

    /// A lowering may push only its decoder state's own listed rows: a row MLA does not list,
    /// even one sharing a listed row's name, is refused, and nothing is pushed.
    #[test]
    fn a_row_the_state_does_not_list_is_refused() {
        let mut r = Recorder::new(MLA, vec![InputRef::Op(0)], vec![LATENT]);
        let pushed = r.push(
            scratchy_decoder_state::lower::StateSubOp::new(&UNLISTED_LATENT, |_| LATENT),
            vec![InputRef::Op(0)],
        );
        assert_eq!(
            pushed,
            Err(LowerError::NotOwnRow {
                op: UNLISTED_LATENT.id()
            })
        );
        assert!(r.pushes().is_empty());
        assert!(
            r.push(
                scratchy_decoder_state::lower::StateSubOp::new(&MLA_LATENT, |_| LATENT),
                vec![InputRef::Op(0)]
            )
            .is_ok()
        );
    }

    /// The attention op carries the weight's scheme and the cache's layer, and reads the weight's
    /// source as its operand 3.
    #[test]
    fn attention_carries_its_weight_and_layer() {
        let (q, latent, k_pe) = (InputRef::Op(0), InputRef::Op(1), InputRef::Op(2));
        let kv_b = InputRef::Ext(4);
        let mut r = Recorder::new(MLA, vec![q, latent, k_pe], vec![HEADS * VALUE])
            .with_weight(
                ArgIndex::new(3),
                StateWeight {
                    source: kv_b,
                    scheme: GemmWeight::Dense,
                },
            )
            .with_layer(ArgIndex::new(4), UnrollIndex(2));
        attention(&mut r).expect("attention lowers");
        let [(attend, attend_in)] = r.pushes() else {
            panic!("attention pushes one op");
        };
        assert_eq!(attend.id(), MLA_ATTEND.id());
        assert_eq!(attend.scheme(), Some(GemmWeight::Dense));
        assert_eq!(attend.layer(), Some(UnrollIndex(2)));
        assert_eq!(attend.out_cols().first(), HEADS * VALUE);
        assert_eq!(attend_in, &[q, latent, k_pe, kv_b]);
        assert_eq!(MLA_ATTEND.weight(), Some(3));
        assert_eq!(r.produces(), [(OutIndex::new(0), InputRef::Op(0))]);
    }
}
