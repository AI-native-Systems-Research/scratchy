// SPDX-License-Identifier: Apache-2.0
//! The per-model launch wiring, as SPYRE'S OWN TYPES — emitted by `#[forward]`
//! as a `static`, never parsed.
//!
//! ⛔ WHAT THIS REPLACES, AND WHY IT IS NOT A FILE FORMAT.
//!
//! Every fact below is a compile-time constant the macro already holds. It used
//! to hold them as the TYPED `SourceBinding` enum, flatten them to JSON strings
//! (`{"id":7,"role":"prefix_k","layer":3}`), bake the string into the binary,
//! and have the worker `serde_json`-parse it back and `match role.as_str()`
//! into the same distinction it started as. A compile-time enum round-tripped
//! through runtime strings, on every model load.
//!
//! That round-trip is not merely wasteful, it is where model facts go to
//! disagree: the string form has no exhaustiveness, so a new role reaches the
//! worker as an unmatched `_ => {}` instead of a compile error, and the
//! `"{disk}.weight"` naming rule it carried could not express a bare
//! `nn.Parameter` (gemma-4's `layer_scalar`) or a non-default decoder root.
//!
//! So the roles are an ENUM here and the macro emits values of it. Adding a
//! role breaks every match that must learn it, at build time, in one place.

use crate::bundle_code as bundle;
use scratchy_tensors::GpuTensor;

// ⭐ `act_name` MOVED TO `ktir_superdsc::place`, BESIDE THE `PlaceId` WHOSE `Display` IT CALLS. Its
// whole body was `bundle::PlaceId::Act(tid).to_string()`, and that module's header already claimed
// to hold "the ONE site that spells an operand" — so this was simply left behind when `PlaceId` and
// the `SynthRole` vocabulary went. Its 22 call sites are all in the lowering, which is now in that
// crate too.

/// The staged tensor of a linear layer, whatever its quant variant.
///
/// ⛔ NOT `LinearLayer::dense_weight()`, WHICH PANICS ON EVERY QUANT VARIANT.
/// That accessor exists for CUTLASS, which only takes dense bf16. Spyre stages
/// BYTES and retiles them for the PT array, so a quantized weight is not a
/// special case here — it is the point (fp8 W8A8 is the bandwidth win). A
/// `Result` rather than a panic because a variant with no single backing
/// tensor (a GGML concat of several) is a real "this model cannot be staged"
/// answer, not a bug to abort on.
pub fn linear_weight(l: &scratchy_layers::layers::LinearLayer) -> anyhow::Result<GpuTensor> {
    use scratchy_layers::layers::LinearLayer as L;
    Ok(match l {
        L::Dense(x) => x.weight,
        L::Fp8(x) => x.weight,
        other => anyhow::bail!(
            "spyre: linear weight variant {} has no single staged tensor",
            std::any::type_name_of_val(other)
        ),
    })
}

/// The fp8 code tensor — the 1-byte payload staged verbatim.
pub fn fp8_weight(l: &scratchy_layers::layers::Fp8AnyLinear) -> GpuTensor {
    use scratchy_layers::layers::Fp8AnyLinear as F;
    match l {
        F::Std(x) => x.weight,
        F::Block(x) => x.weight,
    }
}

/// The fp8 per-channel scale — the sibling of [`fp8_weight`], bound at the
/// `WeightScale` source of the same gemm.
pub fn fp8_scale(l: &scratchy_layers::layers::Fp8AnyLinear) -> GpuTensor {
    use scratchy_layers::layers::Fp8AnyLinear as F;
    match l {
        F::Std(x) => x.weight_scale,
        // The block-scaled variant names it `weight_scale_inv` — it stores the
        // RECIPROCAL, per the block-fp8 checkpoint convention.
        F::Block(x) => x.weight_scale_inv,
    }
}

// ── WHAT THE BAKE PLACED ────────────────────────────────────────────────────

/// The bundle's own answers about what it placed.
///
/// ⛔ THE WORKER USED TO ASK THE ARTIFACT, ONE QUESTION AT A TIME. `load_inner` carried six
/// mutable locals — `prefill_uses_ones`, `prefill_ones_len`, `prefill_uses_identity`,
/// `decode_uses_identity`, `scalarmul_scales_v`, `baked_lm_head_ksplit` — each declared next to
/// the others, assigned inside a `#[cfg(feature = "spyre-hw")]` branch, and read somewhere else
/// entirely. Every one of them needed `#[allow(unused_mut, unused_assignments)]` to compile,
/// which is the shape of the problem: an `#[allow]` was load-bearing, and it is what let
/// `baked_lm_head_ksplit` go on being derived for a bind that no longer existed.
///
/// Asked once, here, where the artifact lives.
#[derive(Clone, Copy, Debug)]
pub struct BakeFacts {
    /// The bundle placed `ONES_REDUCE_TID` — it was emitted with the matmul-by-ones row sum, so
    /// the worker must bind the all-ones seg0 weight. NOT a runtime env: a bundle-vs-env mismatch
    /// leaves the matmul reading an unbound (zero) seg0, and `recip(0) = inf`.
    pub uses_ones_reduce: bool,
    /// Length in f32 elements of that placement (`size / 2`). The fp8-prefill L2 amax enlarges
    /// ONES to `[max(hidden, max fp8-K), stick]`, so binding only `hidden * 64` would leave
    /// down_proj's tail rows reading adjacent seg0 consts. 0 when unplaced.
    pub ones_reduce_len: usize,
    /// The bundle placed `IDENTITY_TID` — the KV cachewr / GQA-replicate matmul-by-identity.
    pub uses_identity: bool,
    /// granite ScalarMul multipliers; index `i` is const tid `scalarmul_scale_tid(i)`.
    pub scalarmul_scales: &'static [f32],
}

impl BakeFacts {
    /// A bundle that placed nothing — the KTIR-emulator path, which has no SuperDSC layout to
    /// ask. ⛔ NOT A DEFAULT FOR THE CARD PATH: `require_identity` refuses this on decode.
    pub const NONE: BakeFacts = BakeFacts {
        uses_ones_reduce: false,
        ones_reduce_len: 0,
        uses_identity: false,
        scalarmul_scales: &[],
    };

    /// Read them off the generated layout.
    pub fn of(layout: &'static bundle::BundleLayout<'static>) -> Self {
        use crate::lower_subtile_tape_to_superdsc as sd;
        let ones = layout.place_of_tid(sd::ONES_REDUCE_TID);
        BakeFacts {
            uses_ones_reduce: ones.is_some(),
            ones_reduce_len: ones.map_or(0, |p| (p.size / 2) as usize),
            uses_identity: layout.place_of_tid(sd::IDENTITY_TID).is_some(),
            scalarmul_scales: match &layout.scalarmul_scales {
                std::borrow::Cow::Borrowed(v) => v,
                // A generated layout is always `Cow::Borrowed`; the owned arm exists for the
                // emit-side type and cannot outlive this call, so it is not a case to serve.
                std::borrow::Cow::Owned(_) => &[],
            },
        }
    }

    /// ⛔ REFUSE a decode bundle with no identity. Every model with attention references `ident`
    /// unconditionally (`assemble_attn`'s GQA new_k/new_v replication), so a bundle that did not
    /// place it runs those matmuls against an UNINITIALISED buffer — silently wrong attention at
    /// every layer of every step, the "stable but incoherent" class. Not zero-safe, so not a
    /// default.
    pub fn require_identity(&self, what: &str) -> Result<(), String> {
        if self.uses_identity {
            return Ok(());
        }
        Err(format!(
            "superdsc decode bundle {what} did NOT place IDENTITY_TID, but assemble_attn's GQA \
             new_k/new_v replication references it unconditionally for every AttnDecode node — \
             decode would run with an unfilled identity buffer. Refusing to serve: \
             compute_bundle_layout's has_attn placement gate and this lookup have desynced."
        ))
    }

    /// The constant environment for a forward of `rows` rows at this geometry.
    ///
    /// One call instead of the eight-field literal both binders spelled out — and the two of them
    /// drifted twice before, each time leaving a placed constant unfilled on one path only
    /// (`IDENTITY_TID` and `ATTN_ZERO_TID`, both found on card).
    pub fn constant_env(
        &self,
        // ⭐ THE WHOLE WIRING, not just its geometry: the identity and rope-P tables are EMITTED
        // constants that live beside it, and taking them from the same static the geometry came
        // from is what stops a caller pairing one model's dims with another's tables.
        w: &'static Wiring,
        mq_pad: usize,
        // ⛔ THE LAUNCH'S ROWS. `reductions = rows > 1` decides whether the reduce constants are
        // bound at all, so a capacity here binds them on a decode step that never reads them.
        rows: StagedRows,
    ) -> ConstantEnv {
        ConstantEnv {
            hidden: w.geometry.hidden as usize,
            head_dim: w.geometry.head_dim as usize,
            mq_pad,
            uses_ones_reduce: self.uses_ones_reduce,
            ones_reduce_len: self.ones_reduce_len,
            uses_identity: self.uses_identity,
            rows: rows.get(),
            scalarmul_scales: self.scalarmul_scales,
            identity: w.identity,
            rope_p: w.rope_p,
            rms_invcols: w.rms_invcols,
        }
    }
}

impl Wiring {
    /// ⛔ THE EMITTED KERNEL TABLES, CHECKED AGAINST A FRESH COMPUTATION — ONCE, AT LOAD.
    ///
    /// [`Self::identity`] and [`Self::rope_p`] are `stage_2d` output the macro evaluated at bake.
    /// They are correct BY CONSTRUCTION — the emitter calls the same two shared helpers the
    /// runtime used to call per forward — but "by construction" is exactly the argument that was
    /// also true of `t{tid}` before the ids and the spellings drifted, and of the K-split zero the
    /// worker bound for a tensor the emitter named nowhere.
    ///
    /// So it is checked. The cost is one `stage_2d` per model load against `hd²` elements, versus
    /// the two it used to pay on every token, and a mismatch means the emitted table and the
    /// device layout law have desynced — which is silent otherwise: an identity that is wrong off
    /// the diagonal zeroes every copy-via-matmul it drives, and a wrong rope-P rotates into the
    /// wrong lane. Both read as fluent-but-incoherent output, not as an error.
    pub fn verify_kernel_tables(&self) -> Result<(), String> {
        use scratchy_subtile::sdsc_abstract::{StickLayout, rope_p_entry, stage_2d};
        let hd = self.geometry.head_dim as usize;
        let want_ident = stage_2d(
            &StickLayout::kernel(hd, hd),
            |i, j| {
                if i == j { 1.0 } else { 0.0 }
            },
        );
        if self.identity != want_ident.as_slice() {
            return Err(format!(
                "emitted identity table ({} elems) does not match `stage_2d(kernel({hd},{hd}))` \
                 ({} elems) — the bake and the device layout law disagree",
                self.identity.len(),
                want_ident.len(),
            ));
        }
        let want_rope: Vec<f32> = if hd >= 2 {
            stage_2d(&StickLayout::kernel(hd, hd), |inn, o| {
                rope_p_entry(hd, inn, o) as f32
            })
        } else {
            Vec::new()
        };
        if self.rope_p != want_rope.as_slice() {
            return Err(format!(
                "emitted rope-P table ({} elems) does not match `rope_p_entry` over \
                 `kernel({hd},{hd})` ({} elems)",
                self.rope_p.len(),
                want_rope.len(),
            ));
        }
        Ok(())
    }

    /// The query-row CAPACITY this bundle was baked at — the embedding activation's placement rows.
    ///
    /// ⛔⛔⛔ THIS IS A CAPACITY, NOT THE ROW COUNT OF A LAUNCH, AND IT IS NOT WHAT THE TAPE
    /// STAGES. A decode step fills ONE row; a prompt chunk fills `mq`, which is at most this. I
    /// briefly made `forward_shape` take its `rows` from here on the reasoning that "the bundle
    /// already knows" — it knows how many rows it can HOLD. On granite that is 96, so every decode
    /// forward staged 96 rows of embedding, 96 rotary rows and a 96-row causal mask where one row
    /// was read, and the card stopped responding.
    ///
    /// Kept because the launch row count must be checked AGAINST it: a launch wider than the bake
    /// stages past the placement. That check is [`Wiring::forward_shape`]'s.
    pub fn baked_row_capacity(&self) -> RowCapacity {
        RowCapacity((self.tensor_shapes[self.embed_src as usize].0 as usize).max(1))
    }

    /// THE FORWARD TAPE FOR THIS BUNDLE.
    ///
    /// ⭐ EVERY FIELD IS GENERATED DATA. The shape used to be assembled in the worker out of a
    /// `BundleMeta` the worker had itself built by scanning the artifact; here it is a projection
    /// of the emitted wiring, so there is nothing for a second opinion to disagree with.
    ///
    /// `owns_prefix` remains the caller's because it is the EXECUTOR's placement answer
    /// (`pmask_slots`), which additionally requires the mask to be a whole number of pages — a
    /// stricter question than "did the wiring name an attn_mask".
    pub fn forward_shape(
        &self,
        // ⛔ THE ROWS THIS LAUNCH FILLS — 1 for a decode step, `mq` for a prompt chunk. NOT the
        // bundle's capacity: see [`Self::baked_row_capacity`], which this is checked against.
        rows: StagedRows,
        owns_prefix: bool,
        n_consts: usize,
    ) -> crate::forward_tape::ForwardShape {
        self.forward_shape_within(self.baked_row_capacity(), rows, owns_prefix, n_consts)
    }

    /// [`Self::forward_shape`] with the capacity supplied by the CALLER rather than read off this
    /// wiring.
    ///
    /// ⭐ WHY A CALLER MAY KNOW BETTER. A decode batch rung is the same tape baked at `B` rows and is
    /// addressed by fingerprint, not through a wiring — so `Wirings.decode` carries every rung's
    /// tensor ids correctly while carrying only the single-request graph's ROW COUNT. A launch on the
    /// `B`-row rung is legitimate and [`Self::baked_row_capacity`] answers 1 for it, so checking
    /// against the wiring rejects a correct launch. The rung's width comes from its manifest's own
    /// `RungSeqs`, which is a stronger authority than an embedding placement belonging to a different
    /// bundle.
    ///
    /// ⛔ THE GUARD IS NOT WEAKENED, IT IS POINTED AT THE RIGHT NUMBER. `RowCapacity` still has no
    /// integer door: the only way to obtain one is this wiring's own placement
    /// ([`Self::baked_row_capacity`]) or a baked rung width
    /// ([`RowCapacity::of_baked_rung`]) — both artifact-derived. A caller cannot widen the check by
    /// arithmetic.
    pub fn forward_shape_within(
        &self,
        // The rows the bundle ACTUALLY RUNNING was baked to hold. Equal to
        // [`Self::baked_row_capacity`] whenever the wiring describes that bundle.
        cap: RowCapacity,
        // ⛔ THE ROWS THIS LAUNCH FILLS — 1 for a decode step, `mq` for a prompt chunk.
        rows: StagedRows,
        owns_prefix: bool,
        n_consts: usize,
    ) -> crate::forward_tape::ForwardShape {
        assert!(
            cap.admits(rows),
            "forward tape asked for {} row(s) from a bundle baked to hold {}: every host buffer \
             would be staged at a shape the placement cannot take",
            rows.get(),
            cap.stated(),
        );
        crate::forward_tape::ForwardShape {
            rows: rows.get(),
            embed_src: self.embed_src,
            cos_srcs: self.cos_srcs,
            sin_srcs: self.sin_srcs,
            prefix_mask: owns_prefix,
            n_consts,
        }
    }
}

// ── THE TWO ROW COUNTS, WHICH ARE NOT THE SAME NUMBER ───────────────────────

/// ⭐⭐⭐ THE TOKEN ROWS **ONE LAUNCH STAGES**.
///
/// ⛔ NAMED `StagedRows`, NOT `StagedRows`, BECAUSE THAT NAME IS TAKEN — `sdsc_abstract::
/// LaunchRows` is a borrowed view of a decode batch's rows (which request sits in which slot),
/// an entirely different thing that this same file already imports. Two types with one name, in
/// one module, differing only by crate path is the collision this whole newtype exists to
/// prevent; introducing one while fixing another would be its own joke. One for a decode step, `mq` for a prompt chunk.
///
/// ⛔ THIS IS NOT [`RowCapacity`], AND THE DIFFERENCE HAS NOW COST TWO BUGS IN ONE FILE.
/// A bundle is baked to HOLD some number of rows; a launch FILLS some number, at most that. Both
/// are `usize`, both are called "rows", and granite's decode bundle holds 96 while a decode step
/// fills 1 — so passing the wrong one is not a crash, it is a plausible number:
///
///   * `ForwardShape::rows` took the capacity, so every decode step staged 96 rows of embedding,
///     96 rotary rows and a 96-row causal mask into buffers one row is read from.
///   * `ConstantEnv::rows` took the capacity, so `reductions = rows > 1` was TRUE on the decode
///     path and bound two reduce constants the pre-tape decode binder never bound.
///
/// Both were written by the same hand, in the same file, from the same assumption — which is why
/// the fix is a TYPE and not two corrected call sites. There is no `From<usize>` and no
/// conversion from a capacity: the only way in is [`Self::of_launch`], at the site that knows how
/// many rows it is about to stage.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct StagedRows(usize);

impl StagedRows {
    /// The rows this launch fills. `None` at zero — a forward that stages nothing is not a
    /// forward, and zero is the value this hardware reads as a maximum.
    pub fn of_launch(rows: usize) -> Option<StagedRows> {
        (rows >= 1).then_some(StagedRows(rows))
    }

    /// Exactly one row — a decode step. Named, so the common case cannot be spelled with a bare
    /// integer that a capacity could also be spelled with.
    pub const ONE: StagedRows = StagedRows(1);

    pub const fn get(self) -> usize {
        self.0
    }
}

/// ⭐ THE ROWS A BUNDLE WAS BAKED TO HOLD — its embedding activation's placement rows.
///
/// ⛔ A CEILING, NOT A COUNT. It answers "how wide a launch may this bundle take", and the only
/// thing it may be used for is checking a [`StagedRows`] against it. It deliberately has no
/// accessor that yields a bare `usize` usable as a row count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowCapacity(usize);

impl RowCapacity {
    /// Does this bundle admit a launch that wide?
    pub const fn admits(self, rows: StagedRows) -> bool {
        rows.0 <= self.0
    }
    /// For diagnostics only — a message saying how wide the bundle is.
    pub const fn stated(self) -> usize {
        self.0
    }

    /// ⭐ THE CAPACITY OF A **BAKED LADDER RUNG**, which is NOT readable from the wiring it runs
    /// through.
    ///
    /// A decode batch rung is the SAME TAPE baked at `B` rows, and the emitter says so outright:
    /// *"a rung is the same tape baked at B rows and is addressed by FINGERPRINT, not through this
    /// wiring"* (`codegen.rs`, the `slot.0` guard). So one `Wirings.decode` describes every rung's
    /// tensor ids correctly while describing only the single-request graph's ROW COUNT — and
    /// [`Wiring::baked_row_capacity`], which reads the embedding placement, therefore answers 1 for
    /// a launch that is legitimately `B` rows wide.
    ///
    /// ⛔ THIS IS THE REGRESSION THAT BROKE BATCHED DECODE. Before the KTIR unification the wiring
    /// slot was last-write-wins, and because rungs bake ASCENDING it happened to hold the WIDEST
    /// rung's wiring — so a batched launch passed the check by accident. Tightening that slot to the
    /// single-request graph (a correct fix: `ktir_decode_cb_*` was stamping `m_cap = 96`, making
    /// every single-token decode compute 96 activation rows) left the batched path checking its
    /// width against a wiring that no longer describes it, and a 2-row launch became
    /// *"forward tape asked for 2 row(s) from a bundle baked to hold 1"*.
    ///
    /// Minted ONLY from a [`RungWidth`](scratchy_subtile::sdsc_abstract::RungWidth) — a width that
    /// came from a manifest's own `RungSeqs`, i.e. from the artifact rather than from a host guess —
    /// so this cannot become a door for waving an arbitrary integer past the guard.
    pub fn of_baked_rung(width: scratchy_subtile::sdsc_abstract::RungWidth) -> RowCapacity {
        RowCapacity(width.count())
    }
}

/// One layer's AttnDecode wiring.
///
/// ⛔ THE WORKER USED TO REDISCOVER THIS BY SEARCHING. It walked every manifest
/// node for one whose args mentioned the mask tensor, then inferred `new_k` /
/// `new_v` as *the argument positionally after* the prefix-K / prefix-V args.
/// Positional inference over a parsed node list, to recover a fact the emitter
/// knew exactly. Emitted directly, it is four integers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayerWiring {
    pub prefix_k: u32,
    pub prefix_v: u32,
    pub new_k: u32,
    pub new_v: u32,
}

/// Model geometry, baked.
///
/// ⛔ THE WORKER USED TO RE-READ THIS FROM `config.json` AT RUNTIME
/// (`num_attention_heads`, `rope_theta`, `head_dim`, `num_hidden_layers`), each
/// behind an `unwrap_or` default — a SECOND source of truth against the bundle
/// that was lowered from the compiler's own bounds. When the two disagreed the
/// bundle won silently and the output was garbage. Baked from the same bounds
/// the tape was lowered at, they cannot disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Geometry {
    pub hidden: u32,
    pub kv_dim: u32,
    pub vocab: u32,
    pub layers: u32,
    pub head_dim: u32,
    /// Rotary base, as bits — `f32` is not structurally-eq, so a `const`-able
    /// struct cannot hold one and still derive `Eq`. Read via [`Self::rope_theta`].
    pub rope_theta_bits: u32,
}

impl Geometry {
    #[inline]
    pub const fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }
}

/// Everything the worker needs to launch one baked bundle, emitted whole.
pub struct Wiring {
    /// Tensor id of the logits.
    pub result: u32,
    /// Tensor ids `0..num_sources` are caller-filled.
    pub num_sources: u32,
    /// The AttnDecode runtime length mask, if the model has a maskable decode.
    pub attn_mask: Option<u32>,
    /// Baked `valid_len - 1`; a worker overrides per generated token.
    pub decode_position: u32,
    /// `[rows, cols]` per tensor id.
    pub tensor_shapes: &'static [(u32, u32)],
    /// The host-gathered hidden row's source id.
    ///
    /// ⛔ THE ANSWER, NOT THE DATA TO COMPUTE IT FROM. This used to be a
    /// `&[Source]` role array that the worker SCANNED at every model load
    /// (`.position(|s| matches!(s, EmbeddedHidden))`) — a compile-time fact
    /// re-derived at runtime, which is the same tell as the JSON `role`
    /// strings this replaced, just with a nicer type. The roles are still an
    /// exhaustive enum where they belong: in the macro, over `SourceBinding`.
    pub embed_src: u32,
    /// Rotary `cos` / `sin` sources as `(id, full column width)`. GQA gives
    /// more than one width (Q vs K), so this is a list, not a pair.
    pub cos_srcs: &'static [(u32, u32)],
    pub sin_srcs: &'static [(u32, u32)],
    /// Per-layer AttnDecode wiring, in layer order.
    pub layers: &'static [LayerWiring],
    pub geometry: Geometry,
    /// The `[hd, hd]` identity, STAGED THROUGH THE KERNEL LAYOUT — the matmul-by-identity the KV
    /// cachewr and the GQA new_k/new_v replication read.
    ///
    /// ⛔ EMITTED, NOT REBUILT. This is `stage_2d(&StickLayout::kernel(hd, hd), ..)`, a pure
    /// function of `head_dim`, and it was recomputed on EVERY forward — 16,384 f32 for hd=128,
    /// per token, to produce the same bytes every time. Row-major coincides with the kernel layout
    /// only at hd == 64; at hd == 128, 16,256 of those entries come from a different byte, which is
    /// why this cannot be a `vec![]` of an obvious pattern.
    pub identity: &'static [f32],
    /// The RoPE rotate-half permutation P — the KERNEL of `rot = matmul(x, P)`, from the shared
    /// Kani-proven `rope_p_entry` so the on-card fill and the proof cannot drift. Same story as
    /// [`Self::identity`]: a pure function of `head_dim`, formerly rebuilt per forward.
    pub rope_p: &'static [f32],
    /// `1/hidden` broadcast over one stick — the mq>1 sum-based amax pre-scale. A pure function of
    /// `hidden`, formerly `vec![1.0 / h as f32; 64]` per forward.
    pub rms_invcols: &'static [f32],
    /// EVERY compile-time scalar this program's KTIR reads, in registry order: entry `i` is bound at
    /// `lower_subtile_tape_to_superdsc::scalarmul_scale_tid(i)` as a `[1,1]` fp16 — the model's own
    /// multipliers and RMSNorm epsilons, exactly the set and order `subtile→superdsc` registers.
    ///
    /// ⛔ NOTHING IS SEEDED AND NOTHING EXTRA IS PUSHED, because the INDEX IS THE DEVICE TID. Two
    /// algebraic identities were once seeded at the front for this construction's `linalg.*` `outs`
    /// seeds; those are immediates now, and so are the rmsnorm epsilon and the attention multiplier
    /// that the PROGRAM states (`f32_splat`, `self.scalar`) — the descriptor still reads the epsilon
    /// from this registry, via a carried slot.
    ///
    /// ⛔ IT IS HERE BECAUSE A ScalarMul SCALE REACHES THE DEVICE AS A BOUND `[1,1]` CONST — that is
    /// `subtile→superdsc`'s own mechanism, so those constants appear in `func.arguments` like any
    /// other buffer and BOTH consumers must fill them: the card through [`constant_steps`], the
    /// emulator through its own source binding. `BakeFacts` carries the same list for the card off the
    /// `BundleLayout`; this is the copy the emulator path has, on a program it resolved by fingerprint
    /// rather than through a layout it never asks for.
    pub scalarmul_scales: &'static [f32],
}

/// The per-model wiring set: one [`Wiring`] per baked PROGRAM.
///
/// ⛔ NOT ONE WIRING. Tensor ids are numbered per program, so the prefill graph
/// has its own — binding decode ids into a prefill launch is silent corruption,
/// not an error. The worker selects by phase.
pub struct Wirings {
    pub decode: Wiring,
    /// `None` when the model bakes no batched-prefill program.
    pub prefill: Option<Wiring>,
}

impl Wiring {
    /// Element count of tensor `id`.
    #[inline]
    pub fn tensor_len(&self, id: u32) -> usize {
        let (r, c) = self.tensor_shapes[id as usize];
        r as usize * c as usize
    }

    /// The prefix-capacity the bundle was baked at (the mask's column count).
    #[inline]
    pub fn capacity(&self) -> Option<usize> {
        self.attn_mask
            .map(|m| self.tensor_shapes[m as usize].1 as usize)
    }
}

// ── SYNTHETIC CONSTANTS ─────────────────────────────────────────────────────

/// What the synthetic-constant set depends on. Everything here is either model
/// geometry or a fact read off the BAKED bundle's own placements — never an
/// environment variable, and never a phase label.
#[derive(Clone, Copy, Debug)]
pub struct ConstantEnv {
    pub hidden: usize,
    pub head_dim: usize,
    /// Padded query rows this bundle was baked at (1 for decode).
    pub mq_pad: usize,
    /// From the bundle's own placements, not a runtime flag.
    pub uses_ones_reduce: bool,
    /// Baked `ONES_REDUCE` placement length in f32 elements; 0 ⇒ fall back to
    /// `hidden * 64`.
    pub ones_reduce_len: usize,
    /// From the bundle's own placements.
    pub uses_identity: bool,
    /// The EMITTED `[hd,hd]` identity and rope-P kernels — see [`Wiring::identity`].
    pub identity: &'static [f32],
    pub rope_p: &'static [f32],
    /// The EMITTED `1/hidden` stick — see [`Wiring::rms_invcols`].
    pub rms_invcols: &'static [f32],
    /// The bundle's BAKED query-row count — 1 for decode, M for batched prefill.
    ///
    /// ⛔ THIS REPLACED A HAND-PASSED `reductions: bool`, which was the one
    /// input here that was a PHASE LABEL rather than a bundle fact. The reduce
    /// constants belong to the multi-row rmsnorm path, and whether a bundle
    /// takes it is `rows > 1` — a number the bake already decided.
    ///
    /// A placement query does NOT work for this one: `RMS_INVCOLS` is placed
    /// under the same guard as `RMS_HALF` ("the tape has any RmsNorm"), so a
    /// decode bundle PLACES it and simply never reads it. Placed is not read.
    pub rows: usize,
    /// Per-index ScalarMul multipliers from `bundle_layout.json`.
    /// ⛔ `'static`, NOT `'a`: a `[1,1]` scalarmul const is ONE ELEMENT of this array, handed out
    /// as a subslice rather than copied into a fresh `vec![sc]` per scale per forward. The array
    /// is already a `static` the macro emitted onto the layout, so the borrow is real.
    pub scalarmul_scales: &'static [f32],
}

// ── THE CONSTANTS THAT ARE LITERALLY CONSTANT ───────────────────────────────
//
// ⛔ THESE WERE `vec![0.5f32; 64]` AND FRIENDS, REBUILT ON EVERY FORWARD. Their
// values depend on nothing — not the model, not the launch, not the bake — so a
// heap allocation per token to hold four known numbers is the "everything is a
// constant" rule broken in the smallest possible way. They are `static` now, and
// [`ConstValues`] is what lets the list hold them beside the computed ones.

/// RMSNorm Newton-rsqrt bound: 0.5, exact in fp16.
static RMS_HALF: [f32; 64] = [0.5; 64];
/// fp8 E4M3 clamp bounds and the `qfp8ch` amax reciprocal.
static FP8_POS448: [f32; 64] = [448.0; 64];
static FP8_NEG448: [f32; 64] = [-448.0; 64];
static FP8_INV448: [f32; 64] = [1.0 / 448.0; 64];

/// A constant's values: BORROWED from the binary when the value is known ahead
/// of the run, OWNED while it is still computed at load.
///
/// ⭐ THE POINT IS THE TYPE, NOT THE ALLOCATION. `Borrowed` is the shape every
/// entry is headed for — the `#[forward]` macro already emits `scalarmul_scales`
/// and the whole `BundleLayout` as `static`s, and these belong beside them. The
/// `Owned` arm is the WORK REMAINING, and it is now visible in the type rather
/// than buried in a `vec!` inside a per-forward function: `IDENTITY` and
/// `ROPE_P` are pure functions of `head_dim`, `RMS_INVCOLS` of `hidden`,
/// `ONES_REDUCE` of the baked placement length. Every one is a bake fact.
///
/// ⛔ `ATTN_ZERO` IS THE ONE THAT IS NOT, and it is worth naming: its LENGTH is
/// `mq_pad * head_dim`, and `mq_pad` is the launch's padded row count. Its
/// values are zeros, so a `static` at the bundle's capacity would serve every
/// launch by slicing — but that is a change to what the bind reads, not a
/// retyping, so it is not folded in here.
/// ONE BIND: which tensor, and the row it gets.
///
/// ⭐ `Cow` IS THE WHOLE POINT. A kernel output is computed per forward and owns its rows; a
/// constant is a table the `#[forward]` macro put in the binary and is borrowed straight into
/// the bind loop, which only ever reads it (`f32_to_f16_le`). Named because the pair appears in
/// eleven signatures and clippy is right that the raw form is unreadable at that count.
pub type Bind = (bundle::PlaceId, std::borrow::Cow<'static, [f32]>);

pub enum ConstValues {
    /// A table the `#[forward]` macro put in the binary.
    Borrowed(&'static [f32]),
    /// `len` copies of one compile-time value.
    ///
    /// ⭐ THE VALUE IS THE CONSTANT; THE LENGTH IS NOT, AND THAT IS THE WHOLE DISTINCTION. Both
    /// members are uniform fills whose extent is decided elsewhere: `ONES_REDUCE`'s comes from the
    /// baked placement (the fp8-prefill L2 amax enlarges it past `hidden`), and `ATTN_ZERO`'s is
    /// `mq_pad · head_dim` — the LAUNCH's padded row count. Emitting them as slices would mean
    /// inventing a bound for the length, so the shape says what is actually known instead.
    Fill { value: f32, len: usize },
}

impl ConstValues {
    /// The values for the bind ABI, BORROWED WHERE THEY CAN BE.
    ///
    /// ⭐ THIS IS WHY THE ENUM EARNS ITS KEEP. `run_step` only ever reads the row (it feeds
    /// `f32_to_f16_le`), so an emitted table has no reason to be copied on the way there — and
    /// the two biggest constants are the `[hd,hd]` identity and rope-P kernels, `2·hd²` floats
    /// that were memcpy'd per token to be read once. A `Fill` still materialises, because `len`
    /// copies of a value is not a slice that exists anywhere.
    pub fn to_cow(&self) -> std::borrow::Cow<'static, [f32]> {
        match self {
            ConstValues::Borrowed(v) => std::borrow::Cow::Borrowed(v),
            ConstValues::Fill { value, len } => std::borrow::Cow::Owned(vec![*value; *len]),
        }
    }

    /// The values, materialised.
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            ConstValues::Borrowed(v) => v.to_vec(),
            ConstValues::Fill { value, len } => vec![*value; *len],
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ConstValues::Borrowed(v) => v.len(),
            ConstValues::Fill { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// THE synthetic constants a bundle needs, as `(tid, values)` in bind order.
///
/// ⛔ ONE FILL SITE, AND THE SECOND ONE WAS A REAL BUG TWICE. These were built
/// by hand in `run_prefill_batch` AND in `superdsc_forward_chunk`, and the two
/// lists drifted — twice, in the same way, both found only on-card:
///
///   * `IDENTITY_TID`: its placement and its `uses_identity` flag were both
///     fixed, but the only FILL was inside `run_prefill_batch`, which decode
///     never calls. Decode's identity stayed zero-initialised, silently zeroing
///     every copy-via-matmul-by-identity op it runs (`new_k`/`new_v` GQA
///     replication, kv cachewr).
///   * `ATTN_ZERO_TID`: same shape of gap one level removed — placement and
///     consumer fixed together, the worker BIND still missing for decode, so
///     the pad-row zero-copy read unbound seg0 garbage.
///
/// Both are "declared, gated correctly, and never filled on one of the two
/// paths". A single list makes that unrepresentable: the two callers now differ
/// only in the [`ConstantEnv`] they pass, which is DATA.
pub fn synthetic_constants(env: &ConstantEnv) -> Vec<(u32, ConstValues)> {
    use crate::lower_subtile_tape_to_superdsc as sd;
    let mut out: Vec<(u32, ConstValues)> = Vec::new();
    let (hd, h) = (env.head_dim, env.hidden);

    // RMSNorm Newton-rsqrt bound const: 0.5 (EXACT in fp16). The on-card
    // rmsnorm derives 1.0/1.5/-1.0 from it — exact math, not tunable.
    out.push((sd::RMS_HALF_TID, ConstValues::Borrowed(&RMS_HALF)));

    // fp8 W8A8 activation-quant clamp consts — E4M3 bounds ±448 plus 1/448 for
    // the `qfp8ch` per-token amax→a_scale. Bound unconditionally: the per-step
    // refill skips them on a dense bundle where they are not placed. (Unbound
    // was the "clamp consts never bound ⇒ garbage quant" gap.)
    out.push((sd::FP8_POS448_TID, ConstValues::Borrowed(&FP8_POS448)));
    out.push((sd::FP8_NEG448_TID, ConstValues::Borrowed(&FP8_NEG448)));
    out.push((sd::FP8_INV448_TID, ConstValues::Borrowed(&FP8_INV448)));

    // Multi-row rmsnorm only: at rows == 1 the reduce is a plain reduce-MAX and
    // neither constant is read (a decode bundle still PLACES `RMS_INVCOLS`,
    // which is why this asks the row count and not the placement).
    let reductions = env.rows > 1;
    if reductions {
        // 1/hidden — the mq>1 sum-based amax pre-scale/un-scale (reduce-MAX is
        // unusable on a multi-row tensor, so amax = sum|x| via a multi-row SUM).
        out.push((sd::RMS_INVCOLS_TID, ConstValues::Borrowed(env.rms_invcols)));
    }
    // ⛔ BOTH REDUCE CONSTS ARE GATED THE SAME WAY, not just RMS_INVCOLS.
    // `ONES_REDUCE` and `RMS_INVCOLS` belong to the SAME fact — this bundle
    // takes the multi-row reduce path — so gating them differently would let a
    // decode bundle bind one and not the other. Decode is rows==1 and takes
    // neither; that was the previous behaviour and it is now one condition
    // rather than two omissions.
    if reductions && env.uses_ones_reduce {
        // Bind the FULL baked placement length: the fp8-prefill L2 amax enlarges
        // ONES to cover down_proj (K = intermediate > hidden), and binding only
        // `hidden*64` leaves the tail rows reading adjacent seg0 consts.
        let len = if env.ones_reduce_len > 0 {
            env.ones_reduce_len
        } else {
            h * 64
        };
        out.push((sd::ONES_REDUCE_TID, ConstValues::Fill { value: 1.0, len }));
    }
    if env.uses_identity {
        // Staged THROUGH the kernel layout: a reserved seg0 tid has no
        // RetileDescriptor, so the host fill IS the device layout, and it is
        // consumed as a `[hd, hd]` matmul KERNEL. Row-major coincides with that
        // only at hd == 64; at hd == 128, 16256 of 16384 entries would come from
        // the wrong byte.
        out.push((sd::IDENTITY_TID, ConstValues::Borrowed(env.identity)));
    }

    // Zeros for new_k/new_v's PADDING rows [mq..mq_pad), which the emitter
    // clears via a real copy from this tensor.
    out.push((
        sd::ATTN_ZERO_TID,
        ConstValues::Fill {
            value: 0.0,
            len: env.mq_pad * hd,
        },
    ));

    if hd >= 2 {
        // RoPE rotate-half permutation P — the KERNEL of `rot = matmul(x, P)`,
        // filled from the SHARED Kani-proven entry fn so the on-card fill and
        // the proof cannot drift. Staged through the kernel layout for the same
        // reason as IDENTITY.
        out.push((sd::ROPE_P_TID, ConstValues::Borrowed(env.rope_p)));
    }

    // granite ScalarMul multipliers: `t{scalarmul_scale_tid(i)} = [scale_i]`, a
    // [1,1] fp16 the on-device pointwise `mul` reads. Unbound = 0 = wrong output.
    for i in 0..env.scalarmul_scales.len() {
        // A SUBSLICE of the emitted scales, not a copy of one: the layout already carries this
        // array as a `static`, and a `[1,1]` const is one element of it.
        out.push((
            sd::scalarmul_scale_tid(i),
            ConstValues::Borrowed(&env.scalarmul_scales[i..=i]),
        ));
    }
    out
}

// ── THE CONSTANTS, AS TAPE STEPS ────────────────────────────────────────────

/// One `ToDevice` operand per constant, in bind order.
///
/// Split from [`constant_steps`] so the steps can BORROW this: a step's operand
/// list is a slice, and the caller owns the storage. That is the same shape the
/// generated form takes, where both are `static`.
pub fn constant_operands(
    consts: &[(u32, ConstValues)],
) -> Vec<scratchy_subtile::host_tape::Operand> {
    use scratchy_subtile::host_tape::{Operand, TensorId, Transfer};
    consts
        .iter()
        .map(|(tid, _)| Operand {
            tensor: TensorId(*tid),
            // A constant is produced on the HOST and read by the device, so it
            // goes up. This is the `ToDevice`-output case — the one the first
            // cut of the player could not express.
            transfer: Transfer::ToDevice,
        })
        .collect()
}

/// The constants as a tape: one HOST step per constant (it is computed, not
/// read from a device), each uploading its own output.
///
/// ⛔ `KernelId(i)` INDEXES THE CONSTANT LIST. A constant IS a kernel — a tiny
/// CPU one that fills a buffer — so it needs no new step kind, exactly as a
/// weight load needs none. The launcher's `host_call(KernelId(i))` produces
/// `consts[i]`, and the `ToDevice` output then uploads it.
pub fn constant_steps(
    ops: &[scratchy_subtile::host_tape::Operand],
) -> Vec<scratchy_subtile::host_tape::Step<'_>> {
    use scratchy_subtile::host_tape::{KernelId, Site, Step};
    (0..ops.len())
        .map(|i| Step {
            kernel: KernelId(i as u32),
            site: Site::Host,
            inputs: &[],
            outputs: &ops[i..i + 1],
        })
        .collect()
}

/// The launcher that turns tape steps into the `acts` list a SuperDSC step
/// binds — spyre's `h2d`, for constants.
///
/// ⛔ IT STAGES, THEN UPLOADS, rather than looking the value up on `h2d`. The
/// shortcut (have `h2d` find the constant by tid and skip `host_call`) would
/// work and would be shorter, but it would model a host kernel as doing
/// nothing, and the whole claim being tested is that a constant IS a kernel
/// whose output happens to go up. Staging keeps the two halves distinct, and
/// the tid assertion below then actually checks that the step's kernel and its
/// operand agree — a check the shortcut cannot make because it has no second
/// opinion to compare against.
pub struct ActsLauncher<'a> {
    consts: &'a [(u32, ConstValues)],
    staged: Option<(u32, &'a ConstValues)>,
    /// The binds, in play order — handed to `run_step` verbatim.
    pub acts: Vec<Bind>,
}

/// A tape step named a kernel and an operand that disagree.
#[derive(Debug)]
pub struct LaunchError(pub String);

impl<'a> ActsLauncher<'a> {
    pub fn new(consts: &'a [(u32, ConstValues)]) -> Self {
        Self {
            consts,
            staged: None,
            acts: Vec::with_capacity(consts.len()),
        }
    }
}

impl scratchy_subtile::host_tape::Launcher for ActsLauncher<'_> {
    type Error = LaunchError;

    fn host_call(&mut self, k: scratchy_subtile::host_tape::KernelId) -> Result<(), Self::Error> {
        let c = self
            .consts
            .get(k.0 as usize)
            .ok_or_else(|| LaunchError(format!("no constant kernel {}", k.0)))?;
        self.staged = Some((c.0, &c.1));
        Ok(())
    }

    fn h2d(&mut self, t: scratchy_subtile::host_tape::TensorId) -> Result<(), Self::Error> {
        let (tid, vals) = self
            .staged
            .take()
            .ok_or_else(|| LaunchError(format!("h2d t{} with nothing staged", t.0)))?;
        // The step's OPERAND and its KERNEL must name the same tensor. They are
        // built from one list so they agree by construction — which is exactly
        // why checking is cheap and why a future generated tape, where they
        // come from different emission sites, gets the check for free.
        if tid != t.0 {
            return Err(LaunchError(format!(
                "step uploads t{} but its kernel produced t{tid}",
                t.0
            )));
        }
        self.acts.push((bundle::PlaceId::Act(tid), vals.to_cow()));
        Ok(())
    }

    fn d2h(&mut self, t: scratchy_subtile::host_tape::TensorId) -> Result<(), Self::Error> {
        Err(LaunchError(format!(
            "d2h t{} — the constant tape reads nothing back",
            t.0
        )))
    }

    fn launch(&mut self, k: scratchy_subtile::host_tape::KernelId) -> Result<(), Self::Error> {
        Err(LaunchError(format!(
            "launch {} — the constant tape has no device step",
            k.0
        )))
    }
}

/// Play the constant tape and return the binds, in order.
///
/// This is what the per-forward binder calls instead of pushing
/// [`synthetic_constants`] directly. Same tids, same values, same order —
/// pinned by `playing_the_constant_tape_matches_synthetic_constants`.
pub fn constant_acts(env: &ConstantEnv) -> Result<Vec<Bind>, LaunchError> {
    let consts = synthetic_constants(env);
    let ops = constant_operands(&consts);
    let steps = constant_steps(&ops);
    let mut l = ActsLauncher::new(&consts);
    scratchy_subtile::host_tape::play(
        &scratchy_subtile::host_tape::HostTape { steps: &steps },
        &mut l,
    )?;
    Ok(l.acts)
}

#[cfg(test)]
mod constant_tape_tests {
    use super::*;

    /// ⭐⭐⭐ A CAPACITY CANNOT BE PASSED WHERE A LAUNCH'S ROWS ARE WANTED. That is the whole
    /// point of the two types, and this pins the properties the compiler rests on: there is no
    /// `From<usize>` for `StagedRows` (the only doors are `of_launch` and `ONE`), and
    /// `RowCapacity` yields no bare count at all — only `admits` and a diagnostic `stated`.
    ///
    /// Both bugs this prevents were real and were made twice in one file: a decode step staging 96
    /// rows because it took the bundle's capacity, and the reduce constants binding on decode
    /// because `rows > 1` was asked of that same capacity.
    #[test]
    fn a_staged_width_is_not_a_capacity() {
        let cap = RowCapacity(96);
        let one = StagedRows::ONE;
        assert_eq!(one.get(), 1);
        assert!(cap.admits(one), "a 96-row bundle takes a 1-row launch");
        assert!(
            cap.admits(StagedRows::of_launch(96).unwrap()),
            "and a full one"
        );
        assert!(
            !cap.admits(StagedRows::of_launch(97).unwrap()),
            "but not a wider one"
        );
        // ⛔ Zero is not a launch: it is also the value this hardware reads as a maximum.
        assert!(StagedRows::of_launch(0).is_none());
    }

    use scratchy_subtile::host_tape::{HostTape, KernelId, Launcher, TensorId, play};

    #[derive(Default)]
    struct Trace(Vec<String>);
    impl Launcher for Trace {
        type Error = ();
        fn h2d(&mut self, t: TensorId) -> Result<(), ()> {
            self.0.push(format!("h2d {}", t.0));
            Ok(())
        }
        fn d2h(&mut self, t: TensorId) -> Result<(), ()> {
            self.0.push(format!("d2h {}", t.0));
            Ok(())
        }
        fn launch(&mut self, k: KernelId) -> Result<(), ()> {
            self.0.push(format!("launch {}", k.0));
            Ok(())
        }
        fn host_call(&mut self, k: KernelId) -> Result<(), ()> {
            self.0.push(format!("host {}", k.0));
            Ok(())
        }
    }

    /// The two emitted kernel tables at `head_dim == 64`, built here the way the macro builds
    /// them — at hd == 64 the kernel layout coincides with row-major, which is why the identity
    /// reads as the obvious pattern and why hd == 128 would NOT.
    fn kernel_tables() -> (Vec<f32>, Vec<f32>) {
        use scratchy_subtile::sdsc_abstract::{StickLayout, rope_p_entry, stage_2d};
        (
            stage_2d(
                &StickLayout::kernel(64, 64),
                |i, j| if i == j { 1.0 } else { 0.0 },
            ),
            stage_2d(&StickLayout::kernel(64, 64), |inn, o| {
                rope_p_entry(64, inn, o) as f32
            }),
        )
    }

    fn env(rows: usize) -> ConstantEnv {
        // Leaked so the fixture can hand out the `&'static` the emitted tables have in a real
        // build; a test binary's lifetime is the process.
        let (ident, ropep) = kernel_tables();
        ConstantEnv {
            hidden: 2048,
            head_dim: 64,
            mq_pad: 64,
            uses_ones_reduce: true,
            ones_reduce_len: 0,
            uses_identity: true,
            rows,
            scalarmul_scales: &[1.5, 2.5],
            identity: Vec::leak(ident),
            rope_p: Vec::leak(ropep),
            rms_invcols: Vec::leak(vec![1.0f32 / 2048.0; 64]),
        }
    }

    /// ⭐ THE EQUIVALENCE THAT MAKES THE PORT SAFE: playing the constant tape
    /// visits exactly the tids `synthetic_constants` produces, in exactly that
    /// order. The live binder pushes that list directly today; this proves the
    /// tape formulation is the same sequence before anything is rewired.
    ///
    /// Run for BOTH row regimes, because the set differs between them (the two
    /// reduce constants are multi-row only) — an equivalence that only held for
    /// decode would hide precisely the drift this work is removing.
    #[test]
    fn playing_the_constant_tape_matches_synthetic_constants() {
        for rows in [1usize, 64] {
            let consts = synthetic_constants(&env(rows));
            let ops = constant_operands(&consts);
            let steps = constant_steps(&ops);

            let mut t = Trace::default();
            play(&HostTape { steps: &steps }, &mut t).expect("plays");

            // Every constant is: run its host kernel, then upload it.
            let want: Vec<String> = consts
                .iter()
                .enumerate()
                .flat_map(|(i, (tid, _))| [format!("host {i}"), format!("h2d {tid}")])
                .collect();
            assert_eq!(
                t.0, want,
                "rows={rows}: the tape must visit the same constants, in the same order, \
                 as the list the live binder pushes"
            );
        }
    }

    /// ⭐ THE FUNCTION THE WORKER ACTUALLY CALLS, locked against the list it
    /// replaced. `playing_the_constant_tape_matches_synthetic_constants` pins
    /// the TRACE; this pins the RESULT — the `(name, values)` pairs handed to
    /// `run_step` — so the wired path cannot drift from the direct one.
    #[test]
    fn constant_acts_equals_pushing_the_list_directly() {
        for rows in [1usize, 64] {
            let want: Vec<Bind> = synthetic_constants(&env(rows))
                .into_iter()
                .map(|(tid, v)| (bundle::PlaceId::Act(tid), v.to_cow()))
                .collect();
            let got = constant_acts(&env(rows)).expect("plays");
            assert_eq!(got.len(), want.len(), "rows={rows}: same number of binds");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0, "rows={rows}: same bind name, same order");
                assert_eq!(g.1, w.1, "rows={rows}: same values for {}", g.0);
            }
        }
    }

    /// The row regimes really do differ, so the test above is not vacuously
    /// comparing two identical lists. Multi-row adds `RMS_INVCOLS` and
    /// `ONES_REDUCE`; decode has neither.
    #[test]
    fn the_multi_row_regime_adds_exactly_the_two_reduce_constants() {
        use crate::lower_subtile_tape_to_superdsc as sd;
        let ids = |rows| -> Vec<u32> {
            synthetic_constants(&env(rows))
                .into_iter()
                .map(|(t, _)| t)
                .collect()
        };
        let (one, many) = (ids(1), ids(64));
        let extra: Vec<u32> = many.iter().filter(|t| !one.contains(t)).copied().collect();
        assert_eq!(
            extra,
            vec![sd::RMS_INVCOLS_TID, sd::ONES_REDUCE_TID],
            "multi-row adds the reduce pair and nothing else"
        );
    }
}
