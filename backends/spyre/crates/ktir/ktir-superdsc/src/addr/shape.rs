// SPDX-License-Identifier: Apache-2.0
//! MODEL PARAMS AS TYPES. Every extent any address is allowed to use comes from here.
//!
//! The bug population this replaces was all one shape: a stride written by hand from the only model
//! that had ever run. `DOWNPROJ_B = 16` is `8192/512`, granite-3.1-2b's K. `b*stick*hd` is the V cache
//! block stride only when a head is one stick. `mq_pad*stick` is the slab stride only when the padded
//! and real row counts coincide. Each was correct for 2b, silently wrong for 8b, and unreachable by any
//! test, because a device address and a plain integer are both `u32`.
//!
//! The fix is not to correct those numbers. It is to make them unwritable. A [`Shape`] carries the
//! model's parameters as CONST GENERICS, and it is the only thing that can hand out a [`Nest`]. So an
//! address cannot be built from an extent someone typed in — it is built from an axis NAME on a shape
//! the model chose. `hd`, `nqh`, `cap` stop being integers in scope that arithmetic can reach for.
//!
//! WHY CONST GENERICS AND NOT FIELDS: this crate runs inside the `#[forward]` proc-macro, so every one
//! of these is known when the bundle is baked — a bundle is baked per (model, prefill rung), and `mq`
//! is fixed within one. Making them const parameters means a shape mismatch is a TYPE error at the
//! call site rather than a wrong number that bakes cleanly and garbles on-card. It also makes the
//! reward-hack mechanically impossible: there is no `if HD == 64` to write, because the two head_dims
//! instantiate the same code and any branch on them would have to be written as a branch on a const,
//! which shows up as a special case in review instead of hiding inside an offset expression.
//!
//! The derived quantities (`slabs`, `head_stride`, ...) are the ONLY place a `/ 64` or a `* hd` is
//! allowed to appear. Every one of this week's bugs was one of those expressions written somewhere
//! else.

use super::Nest;
use crate::superdsc_opspec::Df;

/// A COUNT along one model axis. Distinct types, so a row count cannot be handed to something that
/// wanted a slab count, and neither can be multiplied into an offset by accident -- the arithmetic
/// that used to produce a stride has no operands of the right type to work with.
///
/// `get` is the ONE exit back to `u32`, and the only callers allowed to take it are the derived
/// quantities below. Anywhere else, a bare `u32` in an addressing expression is the bug this prevents.
macro_rules! counts {
    ($($t:ident => $doc:literal),* $(,)?) => { $(
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
        pub struct $t(u32);
        impl $t {
            pub const fn new(v: u32) -> Self { Self(v) }
            pub const fn get(self) -> u32 { self.0 }
        }
    )* };
}
counts! {
    Rows  => "Query rows in this chunk — `mq` at prefill, 1 at decode. NEVER the stick-padded count.",
    PadRows => "The STICK-PADDED row count. Deliberately a different type from [`Rows`]: confusing the \
                two is the replicate-copy bug, where a slab was strided by the padded count while its \
                producer packed by the real one.",
    Slabs => "Head-dim slabs at some data format — `head_dim / lanes`, never a literal.",
    Gqa => "Query heads per kv head — the group size every KV address is built from: \
            [`Shape::gqa`] divides it out of the shape's head counts, and `sdsc_abstract`'s \
            `KvHead::of_query`/`group_first_query` take one. The address-side twin of the \
            model facts' GQA group (`scratchy_ir::model_facts::HeadGeometry::gqa`), divided \
            out once at the facts' constructor; this crate takes no scratchy dependency, so \
            it cannot name that one.",
}

/// One model's attention geometry, as types. `HD` is head_dim, `NQH`/`NKVH` the query and key/value
/// head counts, `CAP` the resident KV capacity.
///
/// There is deliberately no constructor taking runtime values: a `Shape` is named, not built, so a
/// caller cannot invent one that disagrees with the model.
#[derive(Clone, Copy, Debug, Default)]
pub struct Shape<const HD: u32, const NQH: u32, const NKVH: u32, const CAP: u32>;

impl<const HD: u32, const NQH: u32, const NKVH: u32, const CAP: u32> Shape<HD, NQH, NKVH, CAP> {
    /// [`slabs`](Self::slabs) for a head_dim not known as a const. The emitter is one binary serving
    /// every prefill rung, so some extents arrive as values; this keeps the DERIVATION single-homed
    /// even where the type cannot be.
    pub const fn slabs_of(hd: u32, df: Df) -> Slabs {
        let lanes = df.elems_per_stick();
        Slabs::new(if hd < lanes { 1 } else { hd / lanes })
    }

    /// Head-dim slabs at this data format. THE one place `hd / lanes` may be written.
    ///
    /// At `HD == lanes` this is 1 and every slab term below vanishes, which is why hd=64 models never
    /// exercised any of them — not because those models take a different path, but because the same
    /// path degenerates. That is the property that must hold: one path, and the arithmetic collapses.
    pub const fn slabs(df: Df) -> Slabs {
        let lanes = df.elems_per_stick();
        Slabs::new(if HD < lanes { 1 } else { HD / lanes })
    }

    /// GQA group size — how many query heads share one kv head.
    pub const fn gqa() -> Gqa {
        Gqa::new(match NQH.checked_div(NKVH) {
            Some(g) => g,
            None => 1,
        })
    }

    /// The token stream `[rows, NQH*HD]` — roped Q, and the attention output. Head `h`'s slab `s` is a
    /// coordinate on this nest, never a product a caller writes.
    pub fn token_stream(rows: Rows, df: Df) -> Nest {
        Nest::new(&["row", "head", "feat"], &[rows.get(), NQH, HD], df)
    }

    /// The kv-width token stream `[rows, NKVH*HD]` — roped new K/V before replication.
    pub fn kv_stream(rows: Rows, df: Df) -> Nest {
        Nest::new(&["row", "head", "feat"], &[rows.get(), NKVH, HD], df)
    }

    /// The resident V cache, `[CAP, HD]` per query head, stick-major on the feature axis.
    pub fn v_cache(df: Df) -> Nest {
        Nest::new(&["slot", "head", "feat"], &[CAP, NQH, HD], df)
    }

    /// The resident Kᵀ kernel, `[HD, CAP]` per kv head, stick-major on the slot axis.
    pub fn kt_cache(df: Df) -> Nest {
        Nest::new(&["feat", "head", "slot"], &[HD, NKVH, CAP], df)
    }

    /// The head-major online-softmax buffers, `[NQH*rows, HD]`.
    pub fn head_major(rows: Rows, df: Df) -> Nest {
        Nest::new(&["head", "row", "feat"], &[NQH, rows.get(), HD], df)
    }
}
