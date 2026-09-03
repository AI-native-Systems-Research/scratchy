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
}

/// Query heads per kv head — the SAME type the tape's geometry carries, so the group size a
/// device address is built from is the one divided out at
/// [`ModelAttnGeometry::mint`](crate::model_geometry::ModelAttnGeometry::mint) and nowhere else.
/// Re-exported here rather than declared here: `addr` is the address-side face of the model's
/// parameters, but the parameters themselves are the tape's, and the divisibility proof travels
/// with them.
pub use crate::model_geometry::Gqa;

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

    /// Is a tensor's `[rows, heads*head_dim]` form BYTE-IDENTICAL to its `[heads*rows, head_dim]`
    /// head-major view? Only when a head is exactly one stick: element `(r, h*hd+d)` sits at
    /// `((h*hd+d)/lanes)*(rows*lanes) + r*lanes + (d%lanes)`, and the head-major view puts the same
    /// element at `(d/lanes)*(heads*rows*lanes) + (h*rows+r)*lanes + (d%lanes)`. Those agree for every
    /// `(h, r, d)` iff `hd == lanes`, and diverge for most of them otherwise.
    ///
    /// Several emitter fast paths collapse a per-head loop into one whole-tensor op using exactly this
    /// identity — the batched attention, and both RoPE row-batches. They are NOT arbitrary
    /// head_dim-64 special cases: this is their precondition, and it genuinely fails above one stick.
    /// Stating it here means the condition reads as the property it is, and a future format whose
    /// lanes differ gets the right answer instead of one keyed to the number 64.
    /// ⭐ FROM `HD`, THE TYPE'S OWN CONST — no argument, because there is nothing to pass.
    ///
    /// This crate is driven by a PROC MACRO that reads `head_dim` out of the model config at expansion
    /// time and interpolates it as a literal (`codegen.rs`), so the head dim is a COMPILE-TIME CONSTANT.
    /// Asking this question with a runtime `hd` demotes a constant to a value, and a value cannot be
    /// guarded at build time — which is why the hd=64 / hd=128 divergence has been a runtime branch
    /// rather than two instantiations, and why every attempt to lock it down had nothing to hold onto.
    pub const fn head_major_collapse_valid_here(df: Df) -> bool {
        HD == df.elems_per_stick()
    }

    // ⛔⛔⛔ THE RUNTIME TWIN IS DELETED. It took the head dim as a VALUE — `Shape::<0,0,0,0>` built and
    // then bypassed — so every site that used it was a place where a `cargo build` error was
    // IMPOSSIBLE: a constant the proc macro already knows, demoted to something only a runtime branch
    // could test. Its own doc said "do not add new callers" and carried a ⏭ to remove it once the
    // lowering gained `HD` as a const generic. The lowering has, so it is gone, and
    // [`Self::head_major_collapse_valid_here`] — which reads `HD` off the type — is the only way left
    // to ask. The predicate can no longer be reached with a value, so it can no longer be answered too
    // late to matter.

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

// ⭐⭐ THE COLLAPSE FACT, PINNED AT BUILD TIME — module-level `const _: ()` so it SELF-EVALUATES.
//
// ⛔ A NAMED associated const would be INERT: it is only checked if something forces it, and "never used"
// is a warning, not an error. That mistake has already been made in this tree (an inverted
// `CHUNK_FITS_ONE_PAGE` compiled clean), so these live at module level where the compiler must evaluate
// them.
//
// What they pin is the precondition of every emitter fast path that collapses a per-head loop into one
// whole-tensor op: `[rows, heads*hd]` is byte-identical to `[heads*rows, hd]` ONLY when a head is exactly
// one stick. head_dim 64 satisfies it; head_dim 128 does not. Those two facts decide which RoPE form and
// which attention form a bundle gets, and until now they were computed from a runtime `hd`.
const _: () = assert!(
    Shape::<64, 32, 8, 128>::head_major_collapse_valid_here(Df::Fp16),
    "head_dim 64 IS one fp16 stick, so the head-major collapse must be valid — the batched attention and      both RoPE row-batches are gated on exactly this"
);
const _: () = assert!(
    !Shape::<128, 32, 8, 128>::head_major_collapse_valid_here(Df::Fp16),
    "head_dim 128 spans TWO fp16 sticks, so the head-major collapse must be INVALID. This is the fact      that sends a head_dim-128 bundle down the slab RoPE path — the one commented `PREFILL (mq>1)` — and      therefore the reason a decode batch at head_dim 128 is emitted by prefill machinery"
);
// The same law, from the OTHER direction: fp8 packs 128 lanes to a stick, so at fp8 a 128-wide head IS
// one stick. The condition is about LANES, never about the number 64 — pinning both formats keeps a
// future format honest.
const _: () = assert!(Shape::<128, 32, 8, 128>::head_major_collapse_valid_here(
    Df::Fp8
));
const _: () = assert!(!Shape::<64, 32, 8, 128>::head_major_collapse_valid_here(
    Df::Fp8
));
