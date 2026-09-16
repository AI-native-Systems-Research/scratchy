// SPDX-License-Identifier: Apache-2.0
//! The per-head-axis COUNTS of a model, as distinct types.
//!
//! ⭐ MOVED HERE BECAUSE `addr` AND `sdsc_abstract` CANNOT LEAVE WITHOUT [`Gqa`], and those two are
//! mutually recursive so they move as one unit. `addr::shape::Shape::gqa` calls `Gqa::new` and
//! `sdsc_abstract`'s `KvHead::of_query` / `KvHead::group_first_query` take a `Gqa`, so the group
//! size a device address is built from is a device fact these modules name directly. It was the
//! ONLY unresolved path when they moved (one `E0432`, verified by compiler run, not by reading).
//!
//! ⛔ THE DOORS DID NOT COME WITH IT, AND THAT IS THE LINE THIS CRATE MUST NOT CROSS. What stayed in
//! `scratchy_subtile::model_geometry` is everything that knows about *models*: the
//! `include!(OUT_DIR/config_geometry.rs)` the build script writes from every checked-in
//! `crates/models/arch/*/configs/*.json`, the two value→const doors
//! `with_config_attn_geometry`/`with_config_head_dim` and their generated arms, and
//! `geometry_sources()`. So no model inventory travelled into this leaf crate.
//! `model_geometry` re-exports the four counts, so every existing
//! `model_geometry::HeadDim::new(..)` / `Gqa` path in the tree still resolves.
//!
//! ⭐ [`ModelAttnGeometry`] — the VALUE side, with its GQA divisibility proof spent at the mint —
//! followed the counts here, because the lowering's request type named it. It is now the caller's own
//! argument at the geometry door (`scratchy-target-spyre`'s `BundleAttnParams`), which is where model
//! facts belong; this module keeps the type because `model_geometry`'s doors name it through here. Its
//! mint is pure arithmetic over the four counts above, so it brought no model inventory with it.
//!
//! ⚠️ NOTE, NOT FIXED: `addr::shape`'s doc on the `Gqa` re-export still says it is "Re-exported here
//! rather than declared here: `addr` is the address-side face of the model's parameters, but the
//! parameters themselves are the tape's, and the divisibility proof travels with them." That
//! sentence describes the arrangement this move inverts for `Gqa`'s *declaration*, and the proof
//! itself has since followed it here — it is spent at [`ModelAttnGeometry::mint`], one module below.
//! Left as-is per "a move is a move". ⛔ WHAT IS STILL UNSETTLED: whether `Gqa` belongs to the DEVICE
//! (it is the group size every KV address is built from) or to the TAPE (it is a parsed config
//! ratio). This move asserts the former by necessity — the `E0432` above — not by argument.

/// A COUNT along one head axis of a model. Distinct types, so a query-head count cannot be handed
/// to something that wanted a kv-head count — under GQA they are different numbers, and passing one
/// where the other belongs addresses another head's keys, which is fluent wrong output rather than
/// a fault. The same discipline as `addr::shape`'s `Rows`/`PadRows`/`Slabs`, and this is where
/// [`Gqa`] — the group size those two counts determine — lives too.
macro_rules! head_counts {
    ($($t:ident => $doc:literal),* $(,)?) => { $(
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $t(u32);
        impl $t {
            /// The count as the config declares it.
            pub const fn new(v: u32) -> Self { Self(v) }
            /// The count as a value, for the derivations that need one.
            pub const fn get(self) -> u32 { self.0 }
        }
    )* };
}

head_counts! {
    QueryHeads => "How many QUERY heads the model has — `num_attention_heads`.",
    KvHeads    => "How many KEY/VALUE heads the model has — `num_key_value_heads`. Equal to \
                   [`QueryHeads`] only for a non-GQA model, which is exactly why it is a different \
                   type.",
    HeadDim    => "How wide ONE head is — `head_dim`. Not a hidden size, not a stick width, not a \
                   lane count: the three quantities a head dim is most often confused with, all of \
                   which are also small powers of two.",
    Gqa        => "Query heads per kv head. Never written by a call site: it is \
                   `ModelAttnGeometry`'s own division, performed once at the mint where the \
                   divisibility is proven.",
}

/// ⭐⭐⭐⭐⭐ ONE MODEL'S ATTENTION GEOMETRY AS A VALUE, WITH ITS GQA PROOF SPENT AT THE MINT.
///
/// The tape carries this — not three adjacent `u32` fields — so the head counts cannot be
/// transposed in a struct literal, a head dim from another op cannot be paired with them, and the
/// group size cannot be recomputed anywhere downstream.
///
/// [`mint`](Self::mint) is the ONE place divisibility is decided. `None` there is a config whose
/// kv-head count does not divide its query-head count: such a model has no GQA grouping at all, and
/// the arithmetic that used to paper over it — `(nqh / nkvh.max(1)).max(1)` — yields a plausible
/// group size whose attention reads another head's keys. Holding a `ModelAttnGeometry` IS holding
/// the fact that its [`gqa`](Self::gqa) is exact, so no consumer checks again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ModelAttnGeometry {
    nqh: QueryHeads,
    nkvh: KvHeads,
    hd: HeadDim,
    gqa: Gqa,
}

impl ModelAttnGeometry {
    /// THE MINT — and the one evaluation of the GQA division in the whole pipeline.
    ///
    /// `None` for a zero count, a zero head dim, or a kv-head count that does not divide the
    /// query-head count. The caller is the parse boundary from the model config (the macro's
    /// wavefront bridge), so a `None` there is a build error naming the model, not a runtime arm.
    pub const fn mint(nqh: QueryHeads, nkvh: KvHeads, hd: HeadDim) -> Option<ModelAttnGeometry> {
        if nqh.get() == 0 || nkvh.get() == 0 || hd.get() == 0 {
            return None;
        }
        if !nqh.get().is_multiple_of(nkvh.get()) {
            return None;
        }
        Some(ModelAttnGeometry {
            nqh,
            nkvh,
            hd,
            gqa: Gqa::new(nqh.get() / nkvh.get()),
        })
    }

    /// The query-head count.
    pub const fn nqh(self) -> QueryHeads {
        self.nqh
    }

    /// The kv-head count.
    pub const fn nkvh(self) -> KvHeads {
        self.nkvh
    }

    /// The head dim.
    pub const fn hd(self) -> HeadDim {
        self.hd
    }

    /// The GQA group size, divided out at the mint. Exact by construction — there is no rounding
    /// and no `max(1)` here, because a non-dividing pair never became a value of this type.
    pub const fn gqa(self) -> Gqa {
        self.gqa
    }

    /// The `[rows, nqh*hd]` token-stream width — the query projection's and the attention output's
    /// column count. The ONE place this product is written.
    pub const fn q_width(self) -> u32 {
        self.nqh.get() * self.hd.get()
    }

    /// The `[rows, nkvh*hd]` kv-stream width — one K or V projection's column count, narrower than
    /// [`q_width`](Self::q_width) by exactly the GQA group size.
    pub const fn kv_width(self) -> u32 {
        self.nkvh.get() * self.hd.get()
    }
}

impl std::fmt::Display for ModelAttnGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nqh={} nkvh={} head_dim={}",
            self.nqh.get(),
            self.nkvh.get(),
            self.hd.get()
        )
    }
}
