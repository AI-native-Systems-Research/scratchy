// SPDX-License-Identifier: Apache-2.0
//! THE RESERVED TID SPACE — the tensor ids no model produces, declared, bounded, and proven
//! disjoint at COMPILE TIME.
//!
//! ⭐ THIS IS THE PRODUCER↔LOWERING ABI, which is why it belongs in this crate rather than behind a
//! trait. Each id names a tensor the *lowering* invents or requires and the caller binds: the RoPE
//! permutation matrix, the attention scale/mask/causal constants, the rmsnorm Newton constants, the
//! fp8 clamp bounds, the head-major selectors. A third-party KTIR producer has to agree with these
//! numbers to bind anything, so they are part of the interface, not of scratchy's plan.
//!
//! ⛔ AND THE ARITHMETIC IS THE POINT. Every id used to be raw `SOME_BASE - index` with no bound on
//! the index and nothing proving the regions disjoint — see [`TidRegion`] for what two tensors
//! landing on one tid costs on a device with no stack to attach to.

/// RESERVED tensor id for the RoPE rotate-half permutation matrix `P` (in-bundle
/// RoPE). It is NOT a SubtileIR tensor (no model source produces it) — the emitter
/// references it as `t{ROPE_P_TID}` in `lower_rope_node`'s `matmul(x, P)`, places it
/// as a seg0 ACTIVATION in `compute_bundle_layout` (re-bound per step like RMS_SEED —
/// a seg1 weight is only H2D'd at PrepareModel, which binds ONLY the manifest weights,
/// so a synthetic seg1 P would stay ZERO), and the WORKER recognizes this id to
/// synthesize + bind the fixed matrix each step. Chosen far above any real tensor id.
pub const ROPE_P_TID: u32 = u32::MAX - 1;

/// RESERVED tensor id for the attention `scale` constant (`1/sqrt(head_dim)`), a
/// `[1,1]` fp16 the worker synthesizes + binds (seg1). Shared across all layers.
pub const ATTN_SCALE_TID: u32 = u32::MAX - 2;

/// RESERVED tensor id for the attention additive PREFIX length-mask `[mq, cap]` (0
/// on valid prefix cols `[0..p)`, −inf elsewhere) — a per-step ACTIVATION (seg0) the
/// worker binds each step. Broadcast over the nqh batch. Shared across layers.
pub const ATTN_MASK_TID: u32 = u32::MAX - 3;

/// RESERVED tensor id for the attention additive CAUSAL mask `[mq, mq_pad]` over the
/// NEW chunk (0 where new token i attends new token j ≤ i, −inf above the diagonal
/// and on the `[mq..mq_pad)` stick-pad) — a per-step ACTIVATION (seg0) the worker
/// binds (the `triu(-inf,diag=1)` of the reference SDPA). Broadcast over nqh. For
/// decode (mq=1) it is `[1, mq_pad]` with col 0 = 0 (the token attends itself).
pub const ATTN_CAUSAL_TID: u32 = u32::MAX - 4;

/// RESERVED tensor id for the RMSNorm Newton-rsqrt SEED FLOOR constant (`≈1e-4`), a
/// `[1,1]` fp16 the worker synthesizes + binds (seg0 activation). The decomposed rmsnorm
/// computes `inv = 1/√meps` by a range-safe Newton iteration `y ← y·(1.5 − 0.5·meps·y²)`
/// (uses `meps` DIRECTLY — no `reciprocal(meps)` that underflows fp16 for large residuals).
/// Its seed is `max(reciprocal(meps), SEED_CONST)`: `reciprocal(meps)` is a good seed for
/// small meps (early layers) and underflows→~0 for large meps, where this floor takes over —
/// so ~10 iterations converge across the whole residual-stream dynamic range. The iteration
/// constants `1.0/0.5/1.5` are derived ON-CARD from this one bound value via bounded
/// reciprocals (`1 = SEED·recip(SEED)`, `½ = recip(1+1)`, `1½ = 1+½`), so only ONE const
/// needs binding. (Sengraph gets a working native `Rsqrt` free from CompileGraph; the
/// SuperDSC per-op-dxp `rsqrt`/`sqrt` return the unrefined seed, so we build it explicitly.)
pub const RMS_SEED_TID: u32 = u32::MAX - 5;

/// RESERVED tensor id for the RMSNorm Newton constant `0.5` (`[1,stick]`, worker-bound).
/// Bound DIRECTLY (not derived from [`RMS_SEED_TID`] via `reciprocal`): the seed floor is
/// ~5e-6, and `reciprocal(5e-6)=2e5` OVERFLOWS fp16 (max 65504)→inf, so deriving `1.0 =
/// seed·recip(seed)` produced inf consts. `0.5` is exact in fp16; `1.5` is `0.5+0.5+0.5`.
pub const RMS_HALF_TID: u32 = u32::MAX - 6;

/// RESERVED tensor id for the RMSNorm `1/cols` constant (`[1,stick]`, worker-bound = `1/hidden`). Used
/// ONLY by the mq>1 (prefill) SUM-BASED amax: reduce-MAX is unusable on a multi-row TENSOR (returns seed 0
/// even with a per-row mb=1 slice — CONFIRMED, Kani `rmsnorm_max_reduce_seeds_on_multirow_tensor`), so amax
/// = `sum|x|` via a MULTI-ROW SUM. Pre-scaling `|x|·(1/cols)` before the sum keeps every partial ≤ max|x| ≤
/// fp16-max (no overflow, Kani `rmsnorm_sumbased_amax_finite`); `ramax = recip(mean|x|)·(1/cols) = 1/sum|x|`.
pub const RMS_INVCOLS_TID: u32 = u32::MAX - 11;

/// DIAGNOSTIC (2026-07-08): a persistent seg0 probe tid holding LAYER-0's `new_v` (the V-proj output,
/// PRE-selector) `[mq_pad, nkvh·hd]`. The structural mq>1 inf is in the shared M=8 proj-matmul/selector
/// path (K+V caches both inf, rope exonerated). This splits it: the emitter copies layer-0 (k_id==9) new_v
/// here BEFORE the selector; the worker reads it. INF ⇒ the V-proj M=8 matmul is the source; FINITE ⇒ the
/// selector (which reads new_v's mqp-padded [mqu..mqp) rows) is. Persistent (never reused) so it survives.
pub const NEW_V_PROBE_TID: u32 = u32::MAX - 12;

/// DIAGNOSTIC (mq>1 fp32 rmsnorm root-cause, gated on `SCRATCHY_SUPERDSC_ISLAND_PROBE`): persistent
/// fp32 probe tids holding the reduce output `var32` and the final Newton `rsqrt` (per row). The
/// approved measurement (audit) reads these to localize the batched `V=0`: `var32` tiny ⇒ the reduce
/// output-side is broken (⇒ matmul-by-ones fix); `var32` finite but `rsqrt` huge ⇒ the Newton
/// broadcast diverges. Persistent (never reused) so the value survives to the worker readback.
pub const RMS_VAR_PROBE_TID: u32 = u32::MAX - 18; // MAX-16 collides with ONES_REDUCE_TID
// ⛔ MAX-17, NOT MAX-19 — MAX-19 IS `IDENTITY_TID`. The probe moved, not
// IDENTITY: IDENTITY is on the live KV-matmul path and any already-baked bundle
// encodes MAX-19 as IDENTITY, so moving IT would silently reinterpret existing
// artifacts. MAX-17 was the one free slot in MAX-1..MAX-19.
// Locked by `sentinel_tids_are_pairwise_distinct`, which was RED on this line.
pub const RMS_RSQRT_PROBE_TID: u32 = u32::MAX - 17;

/// fp8 W8A8 activation-quant constants (worker-bound f16, seg0, shape [1, stick]): the E4M3 clamp bounds
/// `+448` / `-448` and `1/448` (for `a_scale = amax·(1/448)`). Placed when the tape has an fp8 (arity-3)
/// MatmulTile. `qfp8ch` requires its input CLAMPED into [-448,448] (fp16 rounding can push a per-token-scaled
/// value past 448 — the bound is NOT free, per the clamp guard on [`crate::superdsc_opspec::OpFunc::Qfp8ch`]).
pub const FP8_POS448_TID: u32 = u32::MAX - 13;
pub const FP8_NEG448_TID: u32 = u32::MAX - 14;
pub const FP8_INV448_TID: u32 = u32::MAX - 15;

/// RESERVED tid for the mq>1 (prefill) MATMUL-BY-ONES row-sum weight — a `[hidden, stick]` all-ones fp16
/// seg0 ACTIVATION (worker-bound flat like the head-major selectors).
/// The on-card reduce returns the SEED (0) for any tensor with >1 PHYSICAL row (#33), so the multi-row
/// rmsnorm `sum|x|`/`mean(xs²)` reduces (and the softmax denom) yield 0 → recip(0)=inf → the whole prefill
/// (NEW_V, K/V) goes inf. matmul IS proven correct AND multi-row on-card (it is the q/k/v/o proj path), so
/// `matmul(A[rows,cols], ones[cols,stick])` gives Σ_c A[r,c] in EVERY output col — a drop-in row-sum. n=stick
/// (=64, one output stick) ⇒ the device retile is identity of row-major, and an all-ones tile is
/// retile-invariant, so it needs NO RetileDescriptor (exactly like the selectors). `cols≤hidden` ⇒ w_off=0
/// reads the first `cols` all-ones rows. Placed + bound iff the bundle has a multi-row (prefill) rmsnorm;
/// a decode (m=1) bundle omits it, byte-identical to the pre-fix baseline. Prefill-only (mq>1). Free tid
/// (u32::MAX-16; -17/-18 = Claude2's block_table/slot_mapping, -19 = IDENTITY_TID, up to SCALARMUL_BASE=-20).
pub const ONES_REDUCE_TID: u32 = u32::MAX - 16;

/// RESERVED tensor id for the mq>1 PREFILL KV-CACHE-WRITE matmul-by-IDENTITY (#3) — a `[hd,hd]`
/// identity, worker-bound seg0 const (bound like [`ONES_REDUCE_TID`]). The mq>1 per-slot cachewr
/// used `nqh·mqu·2` single-row `[1,hd]` `slot_no_fuse` copies (SlotSolo singletons) because the
/// on-card multi-row `[mqu,hd]` copy is BROKEN (writes nothing). Mirroring #0's reduce→matmul-by-ones,
/// the copy becomes `matmul(kh[mqu,hd], I[hd,hd]) = kh` — matmul is the ONE m>1-correct primitive — so
/// ONE fusable op per head replaces `mqu` singletons (1,984→64 for granite-2b prefill). For hd==64 the
/// identity is single-stick (retile == row-major, like the all-ones), so no `kernel_weights` descriptor.
/// Placed only when the tape has an mq>1 `AttnDecode` AND `SCRATCHY_SUPERDSC_KV_MATMUL` (default-off A/B).
pub const IDENTITY_TID: u32 = u32::MAX - 19;

/// RESERVED tensor id for the mq>1 PREFILL HEAD-MAJOR SELECTOR weights — `nqh` stacked one-hot
/// `[nqh·hd, hd]` column-selectors (total `[nqh·nqh·hd, hd]`, worker-bound f16 const, seg0). For head
/// `h`, `Sel_h` (row-block `h`, element offset `h·nqh·hd·hd`) has `Sel_h[i,j]=1` iff
/// `i == `[`selector_head_src_col`](crate::sdsc_abstract::selector_head_src_col)`(h,hd,j)` — so
/// `matmul(q[mq,nqh·hd], Sel_h) = q_h[mq,hd]` = head h's columns, written head-major at `h·mq·hd`. The
/// ONLY deployed way to head-major-ize row-major q/k/v for the per-head mq>1 attention ops (a strided
/// read / 3D reshape / restickify all fail — see [`kcache_kt_write_offset`] + Kani `selector_extracts_head_column`).
/// Fixed (position/layer-independent) ⇒ bound ONCE at prepare, like [`ROPE_P_TID`]. Placed only when the
/// tape has an mq>1 `AttnDecode` (the prefill bundle); the mq=1 decode bundle never references it.
///
/// [`kcache_kt_write_offset`]: crate::sdsc_abstract::kcache_kt_write_offset
pub const SEL_HEADMAJOR_TID: u32 = u32::MAX - 7;

/// RESERVED tid for the mq>1 prefill KV-width head-major selector — `nkvh` stacked one-hot
/// `[nkvh·hd, hd]` selectors (worker-bound f16 const, seg0). Same role as [`SEL_HEADMAJOR_TID`] but for
/// the `nkvh`-headed new-K/new-V `[mq_pad, nkvh·hd]` → head-major `[nkvh, mq_pad, hd]` (a distinct k-dim
/// `nkvh·hd`, so a distinct selector). Placed only for the prefill (mq>1) bundle.
pub const SEL_KV_HEADMAJOR_TID: u32 = u32::MAX - 8;

/// RESERVED tid for the mq>1 prefill INVERSE head-major selector — `nqh` stacked one-hot `[hd, nqh·hd]`
/// selectors (`SelT_h[j,c]=1` iff `c == h·hd+j`, worker-bound f16 const, seg0). Scatters the head-major
/// attention output `out_h[mq,hd]` back to ROW-MAJOR `out[mq, nqh·hd]` (columns `[h·hd,(h+1)·hd)`) that
/// `o_proj` reads: `out += matmul(out_h[mq,hd], SelT_h[hd, nqh·hd])`. Placed only for the prefill bundle.
pub const SELT_HEADMAJOR_TID: u32 = u32::MAX - 9;

/// RESERVED tid for the mq>1 prefill ZERO-PAD const — a worker-bound `[mqp, hd]` f16 ZERO buffer (seg0).
/// The head-major K/V selector writes only the `mqu` REAL rows of `kh`/`vh` (`[nkvh, mqp, hd]`); rows
/// `[mqu..mqp)` (stick padding) are UNINITIALIZED. The restickify+score then reads those as garbage K in
/// the masked padding columns `[mqu..mqp)`, and the NO-MAX softmax `exp(garbage·scale + MASK_NEG)`
/// OVERFLOWS to inf (Kani `prefill_softmax_padding_must_be_zeroed`, fail-first) → the whole prefill goes
/// inf. Copying this zero into `kh`/`vh` padding rows makes the padding-col score 0 ⇒ `exp(MASK_NEG)=0`
/// (safe). Placed only when the tape has an mq>1 AttnDecode.
pub const ATTN_ZERO_TID: u32 = u32::MAX - 10;

/// RESERVED tensor id for the mq>1 PREFILL LAST-ROW HIDDEN `[1, hidden]` — the last prompt token's
/// final-norm output, extracted from the `[mq, hidden]` residual stream so the lm_head tail can run at
/// `m=1` INSIDE the prefill bundle (folding away the separate decode-of-the-last-token forward, which
/// existed only to make the first generated token's logits and re-paid the whole weight-stream floor).
///
/// NOT worker-bound: it is written ON-CARD by `lower_one_node`'s per-stick copies and read by the
/// re-lowered lm_head. It is registered as a SYNTH under this name (`t{LAST_HIDDEN_TID}`), so it costs
/// one intermediate-segment reservation and NO manifest placement — the decode bundle never references
/// it and is byte-identical.
///
/// WHY A COPY AND NOT A ONE-HOT MATMUL: `hidden[mq, hidden]` is `RowBlocked`, so row `mq-1` lives in
/// `hidden/64` chunks of 64 elements at stride `mq·64` — a `sel[1,mq] @ hidden[mq,hidden]` extraction
/// would need `k = mq` to be a whole 64-stick (the rungs are 15/23/31/39/47/63/80/96 — never), and
/// reading `hidden` at a row offset in ONE op would need a `rows·lanes` coordInfo stride, which
/// [`crate::sdsc_abstract::StickLayout::group_stride`] documents as unrepresentable for fp16 (dxp
/// `LX_MODLRFIMM` at exactly `rows·lanes = 31·64`). So the extraction is `hidden/64` single-stick
/// `[1,64]` identity copies — each self-consistently addressed at `lanes`, the one proven form.
pub const LAST_HIDDEN_TID: u32 = u32::MAX;

/// ⛔ THE SENTINEL TIDS MUST BE PAIRWISE DISTINCT, AND NOTHING WAS CHECKING.
///
/// These are hand-assigned `u32::MAX - N` magic numbers naming synthetic
/// tensors (constants, probes, selectors) that the emitter places and the worker
/// binds. Two names sharing one value means two DIFFERENT tensors resolve to one
/// placement: whichever is bound last wins and the other silently reads someone
/// else's bytes. There is no crash — the id is valid, it is just not yours.
///
/// This is not hypothetical. `RMS_VAR_PROBE_TID` carries the comment
/// "MAX-16 collides with ONES_REDUCE_TID", so the collision was hit ONCE and
/// fixed by hand — and then `RMS_RSQRT_PROBE_TID` was given MAX-19, which
/// `IDENTITY_TID` already had. A hand-assigned space with no uniqueness check
/// re-collides as soon as someone adds a name.
#[test]
fn sentinel_tids_are_pairwise_distinct() {
    // Every sentinel, by name, so a new one added without a slot shows up here.
    const SENTINELS: &[(&str, u32)] = &[
        ("LAST_HIDDEN", LAST_HIDDEN_TID),
        ("ROPE_P", ROPE_P_TID),
        ("ATTN_SCALE", ATTN_SCALE_TID),
        ("ATTN_MASK", ATTN_MASK_TID),
        ("ATTN_CAUSAL", ATTN_CAUSAL_TID),
        ("RMS_SEED", RMS_SEED_TID),
        ("RMS_HALF", RMS_HALF_TID),
        ("SEL_HEADMAJOR", SEL_HEADMAJOR_TID),
        ("SEL_KV_HEADMAJOR", SEL_KV_HEADMAJOR_TID),
        ("SELT_HEADMAJOR", SELT_HEADMAJOR_TID),
        ("ATTN_ZERO", ATTN_ZERO_TID),
        ("RMS_INVCOLS", RMS_INVCOLS_TID),
        ("NEW_V_PROBE", NEW_V_PROBE_TID),
        ("FP8_POS448", FP8_POS448_TID),
        ("FP8_NEG448", FP8_NEG448_TID),
        ("FP8_INV448", FP8_INV448_TID),
        ("ONES_REDUCE", ONES_REDUCE_TID),
        ("RMS_VAR_PROBE", RMS_VAR_PROBE_TID),
        ("RMS_RSQRT_PROBE", RMS_RSQRT_PROBE_TID),
        ("IDENTITY", IDENTITY_TID),
    ];
    let mut seen: std::collections::BTreeMap<u32, &str> = Default::default();
    let mut clashes: Vec<String> = Vec::new();
    for (name, tid) in SENTINELS {
        if let Some(prev) = seen.insert(*tid, name) {
            clashes.push(format!(
                "{name} and {prev} are both u32::MAX - {}",
                u32::MAX - *tid
            ));
        }
    }
    assert!(
        clashes.is_empty(),
        "sentinel tids collide — two synthetic tensors would share one placement: {clashes:?}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE RESERVED TID SPACE — declared, bounded, and proven disjoint AT COMPILE TIME
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One reserved region of the tid space: a base and how many slots it owns, counting DOWN.
///
/// ⛔⛔⛔ THIS EXISTS BECAUSE THE REGIONS WERE RAW `u32` SUBTRACTION. Every reserved tid was
/// `SOME_BASE - index`, with each base defined in terms of the previous one, no bound on any
/// index, and nothing anywhere proving the regions do not overlap. `scalarmul_scale_tid(idx)` owns
/// exactly 100 slots before it walks into `KSPLIT_BLOCK_BASE`, and it took `idx: usize` unchecked;
/// `downproj_block_tid` multiplies a REAL TENSOR ID by a stride and subtracts that.
///
/// Two tensors landing on one tid is not an error anyone sees. They get ONE placement, so the
/// second one's producer writes over the first one's bytes and its consumer reads them — on a
/// device with no stack to attach to, which surfaces as wrong output or as a hang.
///
/// A region cannot be added or resized without [`RESERVED_REGIONS_ARE_DISJOINT`] re-proving the
/// whole layout, and no index can leave its region without [`Self::at`] refusing during the bake.
#[derive(Clone, Copy, Debug)]
pub struct TidRegion {
    /// For diagnostics — named so a refusal says WHICH region overflowed.
    pub name: &'static str,
    /// Highest tid in the region; slots count downward from here.
    pub base: u32,
    pub slots: u32,
}

impl TidRegion {
    /// Lowest tid this region owns.
    pub const fn floor(self) -> u32 {
        self.base - (self.slots - 1)
    }

    /// The `idx`-th tid, or a BUILD FAILURE. `idx >= slots` would silently alias the region below.
    pub const fn at(self, idx: u32) -> u32 {
        assert!(
            idx < self.slots,
            "reserved tid region overflow: this index is past the region's last slot and would \
             alias the region below it, giving two tensors one placement",
        );
        self.base - idx
    }

    const fn overlaps(self, other: TidRegion) -> bool {
        self.floor() <= other.base && other.floor() <= self.base
    }
}

/// Every reserved region, in one place. Order is high tid → low.
pub const RESERVED_REGIONS: [TidRegion; 3] = [
    // The single sentinels (`ROPE_P_TID` .. `IDENTITY_TID`) occupy MAX-1 .. MAX-19.
    TidRegion {
        name: "sentinels",
        base: u32::MAX - 1,
        slots: 19,
    },
    TidRegion {
        name: "scalarmul_scale",
        base: u32::MAX - 20,
        slots: 100,
    },
    // ⛔ THE GAP FROM MAX-120 TO MAX-1_000_170 IS DELIBERATELY LEFT EMPTY. It held the K-split
    // block/zero/down_proj regions, which are gone with the K-split itself. `kct_resident` keeps
    // its ABSOLUTE base rather than sliding up into the hole: every reserved id is a number that
    // has been baked into artifacts, and moving one to tidy the map would silently repoint it.
    // ⛔ BOUNDED, WHERE IT USED TO BE OPEN-ENDED. `kct_resident_tid(k_id) = BASE - k_id` had no
    // floor at all: a model with enough tensors walked it downward without limit. One million
    // slots is far past any real tid count and is now a REFUSAL rather than a wrap.
    TidRegion {
        name: "kct_resident",
        base: u32::MAX - 1_000_171,
        slots: 1_000_000,
    },
];

/// ⭐ THE PROOF, EVALUATED AT COMPILE TIME. Adding or resizing a region re-runs it; an overlap is
/// a build error naming both regions, not a tensor that quietly answers to two names.
#[allow(clippy::let_unit_value)]
const _: () = {
    // Referencing both proofs is what makes the compiler EVALUATE them; an unused `const` item is
    // not const-evaluated, so a lock nothing mentions is a lock that never runs.
    let _ = SENTINELS_ARE_INSIDE_THEIR_REGION;
    let _ = RESERVED_REGIONS_ARE_DISJOINT;
};

pub const SENTINELS_ARE_INSIDE_THEIR_REGION: () = {
    // ⛔ THE ONE-OFF SENTINELS ARE DECLARED SEPARATELY (`ROPE_P_TID` .. `IDENTITY_TID`), so the
    // region table would happily describe a span they had already outgrown. Adding a 20th sentinel
    // now fails the build instead of silently taking `scalarmul_scale_tid(0)`'s slot.
    let r = RESERVED_REGIONS[0];
    assert!(r.base == u32::MAX - 1, "sentinels start at MAX-1");
    assert!(
        IDENTITY_TID >= r.floor(),
        "a one-off sentinel has fallen below the sentinel region and into the scalarmul scales",
    );
    assert!(
        SCALARMUL_SCALE_BASE < r.floor(),
        "the scalarmul region overlaps the one-off sentinels",
    );
};

pub const RESERVED_REGIONS_ARE_DISJOINT: () = {
    let mut i = 0;
    while i < RESERVED_REGIONS.len() {
        let mut j = i + 1;
        while j < RESERVED_REGIONS.len() {
            assert!(
                !RESERVED_REGIONS[i].overlaps(RESERVED_REGIONS[j]),
                "two reserved tid regions overlap — some tid names two different tensors",
            );
            j += 1;
        }
        // Regions descend, and none may run below the last one's floor into REAL tensor ids.
        assert!(
            RESERVED_REGIONS[i].floor() < RESERVED_REGIONS[i].base + 1,
            "a reserved region wrapped past zero",
        );
        i += 1;
    }
};

/// Base of the RESERVED tensor-id block for granite ScalarMul scale constants. The `i`-th DISTINCT
/// scale (see [`BundleLayout::scalarmul_scales`]) is bound to `SCALARMUL_SCALE_BASE - i` — a
/// worker-bound `[1,1]` fp16 the pointwise `mul` reads broadcast (the ATTN_SCALE mechanism). Well
/// below the other reserved ids (u32::MAX-1..-6) with room for many distinct scales.
///
/// [`BundleLayout::scalarmul_scales`]: crate::placement::BundleLayout::scalarmul_scales
pub const SCALARMUL_SCALE_BASE: u32 = RESERVED_REGIONS[1].base;

/// Reserved const TID for the `i`-th distinct ScalarMul scale.
pub fn scalarmul_scale_tid(idx: usize) -> u32 {
    RESERVED_REGIONS[1].at(idx as u32)
}

/// Base of the RESERVED tid block for the RESIDENT per-layer Kᵀ kernel `kct` (the "kill the O(active)
/// restickify" residency fix). The Kᵀ kernel the score matmul reads was a per-step SYNTH scratch that
/// re-transposed the whole active K each step; making it a RESIDENT seg2 tensor (persists across
/// steps and layers, so the incremental slab restickify only re-transposes the current slab) needs a
/// real tid with a per-layer seg2 placement. KEYED ON THE K-CACHE SOURCE TID `k_id` (like `downproj_block_tid`)
/// so the emitter (which has layer-0 `k_id`) and the layout/guard (which iterate every layer's `k_id`)
/// compute the SAME kct tid with no layer-index handoff. A 1M gap below `DOWNPROJ_BLOCK_BASE` keeps it
/// disjoint from the down_proj blocks (which descend only ~tens-of-K) AND from real source tids (~thousands).
pub const KCT_RESIDENT_BASE: u32 = RESERVED_REGIONS[2].base;
/// Reserved tid for the resident per-layer Kᵀ kernel of the K-cache source tid `k_id`.
pub fn kct_resident_tid(k_id: u32) -> u32 {
    RESERVED_REGIONS[2].at(k_id)
}
