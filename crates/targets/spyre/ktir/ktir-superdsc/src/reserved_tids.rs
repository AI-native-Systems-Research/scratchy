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

/// ⭐⭐⭐⭐⭐ RESERVED tensor id for the KV BLOCK INDEX — the index tensor a gathered KV read is
/// addressed through, and the last piece the hardware gather needs.
///
/// Today the block table is HOST-SIDE state (`fold_plan`'s `block_tables: Vec<Vec<i64>>`), consulted
/// per launch to compute ONE page base. `SessionKv::fold_requests` names what that costs: *"ONE LAUNCHED
/// OP HAS ONE PAGE BASE, so B requests cannot share a pass: the fold runs `pages x requests` times"*.
/// A gather replaces that scalar with a tensor, so every row reads its own keys in ONE launch — which is
/// what batch-size-invariant attention means.
///
/// ## What the worker stages here, and why the arithmetic is already settled
/// int32 GLOBAL 4096-ELEMENT BLOCK INDICES. `addr = idx * skip_addr + base` with `skip_addr = 4096`,
/// which is what the score leg's Kᵗ operand already emits as its entry (MEASURED: per-core `in=64`,
/// `out=64`). That works because the whole pool is a uniform array of 4096-element blocks — stick-groups
/// at 0/4096/8192/12288, `plane_block_elems = 4*4096`, and `plane_stride`/`layer_stride`/`page_stride`
/// all multiples of it — so one index reaches any cell with no relayout and no `device_extent` change.
///
/// It is the same affine map the host already computes: `fold_plan::page_base_bytes` is
/// `phys * page_stride_bytes`, and dxp's `ConvertData_gather_idx` is `idx * skip_addr + base_addr`.
///
/// ## Why a NEW region rather than a 20th sentinel
/// The sentinel region is FULL — `MAX-1 .. MAX-19` are all taken and `MAX-20` is
/// `scalarmul_scale`'s base. Growing the sentinels would slide that region down, and this file warns
/// exactly against it: *"every reserved id is a number that has been baked into artifacts, and moving
/// one to tidy the map would silently repoint it"*. So this takes the top of the deliberately empty gap
/// and declares its own region, which the compile-time disjointness proof then covers.
pub const KV_BLOCK_INDEX_TID: u32 = u32::MAX - 120;

/// ⭐⭐⭐⭐⭐ THE ROPE-P **CLASS** REGION — one P placement per DISTINCT rope head dim.
///
/// A hybrid-attention model (gemma-4) carries TWO rope classes in one tape — sliding `hd` and a wider
/// global `hd` — and a composite single P is IMPOSSIBLE, not merely awkward: the big class's
/// `[hd,hd]` contraction sweeps every input lane, so it would read the small class's nonzero block as
/// spurious ±1 terms. Each class needs its OWN `[hd_c, hd_c]` table and its OWN seg0 placement.
///
/// ## Class ordering, and why class 0 keeps the old sentinel
/// Classes are the tape's DISTINCT rope head dims sorted DESCENDING, so class 0 is the widest. Class 0
/// answers [`ROPE_P_TID`] itself: a UNIFORM model (every rope node one head dim) has exactly one class,
/// so its ops still name `ROPE_P_TID` and its bundle is byte-identical to the pre-class layout. Classes
/// `1..N-1` take this region's slots. The alternative — renumbering every class — would change the tid a
/// uniform model's ops name, for no benefit and one silent-repoint risk more.
///
/// [`rope_p_class_hds`]: crate::reserved_tids::rope_p_class_hds
pub const ROPE_P_CLASS_BASE: u32 = u32::MAX - 121;

/// Reserved tid for the rope-P table of rope class `idx` (distinct rope head dims, sorted descending;
/// class 0 = the widest = [`ROPE_P_TID`] itself, so a uniform model's bundle is unchanged).
pub fn rope_p_class_tid(idx: usize) -> u32 {
    if idx == 0 {
        // ⭐ DOCUMENTED WHY: class 0 IS the old single-class tid. Keeping it there is what makes a
        // uniform model's ops name `ROPE_P_TID` exactly as before — byte-identical bundles — instead
        // of renumbering every class and silently repointing ids already baked into artifacts.
        return ROPE_P_TID;
    }
    // Class 1 takes the region's BASE slot (class 0 lives on the sentinel, so no slot is wasted).
    reserved_region("rope_p_class").at(idx as u32 - 1)
}

/// ⭐⭐⭐⭐⭐ THE IDENTITY **CLASS** REGION — one `[hd,hd]` identity per DISTINCT attention head dim.
///
/// Same defect, same fix, one axis over: `IDENTITY_TID`'s placement was sized at the WIDEST
/// `AttnDecode` head dim while the bound value was staged at the BASE head dim, so on a hybrid
/// (gemma-4 tiny-allglobal: classes 128 and 64) the wide class's GQA krep/vrep matmuls read slab 1 of
/// a `[128,128]` placement holding only a `[64,64]` table — lanes 64..127 all zero (dump-proven,
/// `op11_prog0_seg0_tt108_newkrep.bin`). Attention classes are the tape's DISTINCT `AttnDecode` head
/// dims sorted DESCENDING; class 0 = the widest = [`IDENTITY_TID`] itself, for the same
/// byte-identity reason as rope class 0.
pub const IDENTITY_CLASS_BASE: u32 = u32::MAX - 129;

/// Reserved tid for the identity table of attention class `idx` (distinct AttnDecode head dims,
/// sorted descending; class 0 = the widest = [`IDENTITY_TID`] itself).
pub fn identity_class_tid(idx: usize) -> u32 {
    if idx == 0 {
        // ⭐ Same law as `rope_p_class_tid(0)`: the widest class keeps the old single-class tid so a
        // uniform model's bundle is byte-identical.
        return IDENTITY_TID;
    }
    // Class 1 takes the region's BASE slot (class 0 lives on the sentinel, so no slot is wasted).
    reserved_region("identity_class").at(idx as u32 - 1)
}

/// ⭐⭐⭐⭐⭐ THE ROUTER **CONST** REGION — the token-independent rows and tables the MoE router
/// doors read, worker-bound like the rope-P tables.
///
/// The router quartet's doors (`RouteArgsort`, `RouteTopK`, `RouteGatherScores`,
/// `RouteExpertScale`) decompose the emu's compare/reduce semantics into pointwise
/// `lesserthan`/`equal` over broadcast operands plus native row reduces — the
/// `broadcast_ops.ddl` compare/select family, the same wire shape as the card-proven
/// add/maximum. Every one of those forms needs a FACTOR that is a pure function of the
/// router's geometry (the expert count), not of the tokens:
///
///   * **`rank_tie`** — the stable-argsort tie-break matrix `tie[j,h'] = 𝟙[h' < j]`, a
///     `[W,W]` 0/1 table (W = the padded expert width, 64-lane sticks), read
///     mb-broadcast in the rank compare's `equal`-leg multiply.
///   * **`topk_iota`** — the `[1,W]` iota row `h ↦ h`, the value RouteTopK's selector
///     sum multiplies its match mask by.
///   * **`topk_onehot(j)`** — the `[1,W]` one-hot row with lane `j` hot, the row the
///     combine-chain shape writes logical lane `j` of an output stick through.
///   * **`pad_hi`/`pad_lo`** — the `[1,W]` sanitize rows (+inf sorts last under
///     `lesserthan`, −inf zeroes under softmax's `exp`), the narrow-tensor law's remedy:
///     `maximum(x, pad_row)` forces lanes `E..W` of a padded router stick inert.
///
/// Same mechanism as the rope-P classes: the DOOR resolves the row/table's tid here, the
/// placement pass mints its seg0 footprint, and the worker's load-time bind
/// (`wiring::synthetic_constants`) builds the value from the same bake-carried geometry.
/// SIZED IN SLOTS: one `rank_tie`, one `topk_iota`, one `pad_hi`, one `pad_lo`, one
/// `pad_mask`, and one `topk_onehot` per top-k slot — k ≤ 16 covers every checked-in MoE
/// config (the widest is k=8), and a config beyond it overflows the region and fails the
/// bake rather than aliasing a neighbouring tensor's tid.
pub const ROUTER_CONST_BASE: u32 = u32::MAX - 137;

/// The `[W,W]` stable-argsort tie-break table `tie[j,h'] = 𝟙[h' < j]` (j on rows, h' on
/// lanes — the orientation the rank compare's mb-broadcast multiply reads).
pub fn router_rank_tie_tid() -> u32 {
    reserved_region("router_const").at(0)
}

/// The `[1,W]` iota row `h ↦ h` the top-k selector sums its match mask against.
pub fn router_topk_iota_tid() -> u32 {
    reserved_region("router_const").at(1)
}

/// The `[1,W]` one-hot row with lane `j` hot (one slot per top-k index).
pub fn router_topk_onehot_tid(j: u32) -> u32 {
    reserved_region("router_const").at(2 + j)
}

/// The `[1,W]` +inf sanitize row (argsort pad: sorts last, so pad lanes never win ranks).
pub fn router_pad_hi_tid() -> u32 {
    reserved_region("router_const").at(19)
}

/// The `[1,W]` −inf sanitize row (softmax/gather pad: `exp(−inf)=0`, so pad lanes never
/// contribute to scores).
pub fn router_pad_lo_tid() -> u32 {
    reserved_region("router_const").at(20)
}

/// The `[1,W]` ARGSORT PAD-MASK row: `0` in lanes `0..E` (the real experts — `maximum(x, mask)`
/// leaves them alone) and `+inf` in lanes `E..W` (the producer's zero-pad — `maximum` forces them
/// to sort last, so a zero pad lane can never outrank a negative real score). Distinct from
/// [`router_pad_hi_tid`] (the uniform +inf row the softmax-side sanitize and the top-k one-hot
/// forms use): this row is E-dependent, so its VALUE comes from the geometry handoff the same way
/// the one-hot count does.
pub fn router_pad_mask_tid() -> u32 {
    reserved_region("router_const").at(21)
}

/// Does `tid` name a KERNEL-TABLE class table of family `what` (`"rope-P"` or `"identity"`)?
///
/// The load-time placement cross-check walks every PLACED tid and asks this, because the reverse
/// direction of the check ("every placed class tid is in the wiring's set") cannot enumerate the
/// class ids from a placement — it can only test a tid it finds. Class 0 is the family's sentinel;
/// classes 1.. live in the family's region.
pub fn is_kernel_table_class_tid(tid: u32, what: &str) -> bool {
    match what {
        "rope-P" => {
            tid == ROPE_P_TID || {
                let r = reserved_region("rope_p_class");
                (r.floor()..=r.base).contains(&tid)
            }
        }
        "identity" => {
            tid == IDENTITY_TID || {
                let r = reserved_region("identity_class");
                (r.floor()..=r.base).contains(&tid)
            }
        }
        _ => false,
    }
}

/// ⭐ THE ONE SOURCE OF TRUTH FOR A CLASS SET, IN THE ORDER THE TIDS ARE ASSIGNED.
///
/// Both consumers of the class list — the PLACEMENT site (`compute_bundle_layout`, which sizes each
/// class's seg0 placement) and the macro's `gk` (which bakes the list into the wiring the worker's
/// load-time bind reads) — must produce the SAME set in the SAME order or a class tid names a
/// placement sized for a different class. This is the shared helper both call: distinct values,
/// sorted DESCENDING so class 0 is the widest (and a uniform model's single class is class 0, which
/// [`rope_p_class_tid`] answers with the pre-class sentinel).
///
/// ⛔ TAKES THE HEAD DIMS AS PLAIN DATA, not a `SubtileIR`: this crate cannot depend on
/// `scratchy-subtile` (the dependency runs the other way), so the callers iterate their own rope/attn
/// nodes and hand the dims over. The ORDERING LAW lives here, once, where both callers meet it.
pub fn rope_p_class_hds(hds: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut out: Vec<u32> = hds.collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out.dedup();
    out
}

/// Every reserved region, in one place. Order is high tid → low.
pub const RESERVED_REGIONS: [TidRegion; 7] = [
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
    // ⭐ THE KV BLOCK INDEX takes the TOP of the old K-split gap — see [`KV_BLOCK_INDEX_TID`] for why it
    // is a new region instead of a 20th sentinel (the sentinel region is full, and sliding
    // `scalarmul_scale` down would repoint ids already baked into artifacts).
    TidRegion {
        name: "kv_block_index",
        base: KV_BLOCK_INDEX_TID,
        slots: 1,
    },
    // ⭐ THE ROPE-P CLASS REGION — one slot per rope head-dim class past the first (class 0 keeps
    // `ROPE_P_TID`; see [`rope_p_class_tid`] for why). 8 slots: two classes is the gemma-4 shape, and
    // a model with more DISTINCT rope head dims than that is a config error the region's own overflow
    // assert turns into a bake refusal.
    TidRegion {
        name: "rope_p_class",
        base: ROPE_P_CLASS_BASE,
        slots: 8,
    },
    // ⭐ THE IDENTITY CLASS REGION — the same law one rung down, for the attention classes (see
    // [`identity_class_tid`]).
    TidRegion {
        name: "identity_class",
        base: IDENTITY_CLASS_BASE,
        slots: 8,
    },
    // ⭐ THE ROUTER CONST REGION — the MoE router doors' token-independent factors (see
    // [`ROUTER_CONST_BASE`]): the tie table, the iota row, the one-hot rows, the sanitize rows.
    // 22 slots: 1 tie + 1 iota + 16 one-hots + 2 sanitize + 1 argsort pad-mask. Sits in the top
    // of the gap below `identity_class` (whose floor is MAX-136) and above `kct_resident`
    // (MAX-1_000_171).
    TidRegion {
        name: "router_const",
        base: ROUTER_CONST_BASE,
        slots: 22,
    },
    // ⛔ THE GAP FROM MAX-137 TO MAX-1_000_170 IS DELIBERATELY LEFT EMPTY. It held the K-split
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

/// ⛔⛔⛔ A REGION IS ADDRESSED BY ITS **NAME**, NEVER BY ITS ARRAY INDEX — and this function exists
/// because indexing it by position has already cost exactly the bug the table was built to prevent.
///
/// `KCT_RESIDENT_BASE` and `kct_resident_tid` were spelled `RESERVED_REGIONS[2]`. Inserting the
/// `kv_block_index` region AT position 2 (it belongs there — the table is ordered high tid → low)
/// slid `kct_resident` to 3 and silently repointed both: `kct_resident_tid(0)` began answering
/// `KV_BLOCK_INDEX_TID`, i.e. the resident Kᵗ kernel of layer 0 and the gather's index tensor became
/// ONE tid with ONE placement. Only the region's `slots: 1` made it loud — `kct_resident_tid(k_id)`
/// for any `k_id >= 1` overflowed a one-slot region and failed the bake. A wider new region would
/// have aliased in silence, which is precisely "two tensors landing on one tid is not an error anyone
/// sees" from this file's own header.
///
/// A name cannot be shifted by an insertion, and a name that is not in the table is a BUILD failure
/// rather than a neighbouring region's base.
pub const fn reserved_region(name: &str) -> TidRegion {
    let want = name.as_bytes();
    let mut i = 0;
    while i < RESERVED_REGIONS.len() {
        let have = RESERVED_REGIONS[i].name.as_bytes();
        if have.len() == want.len() {
            let mut k = 0;
            let mut same = true;
            while k < have.len() {
                if have[k] != want[k] {
                    same = false;
                }
                k += 1;
            }
            if same {
                return RESERVED_REGIONS[i];
            }
        }
        i += 1;
    }
    panic!("no reserved tid region by that name — a region was renamed or removed");
}

/// ⭐ THE PROOF, EVALUATED AT COMPILE TIME. Adding or resizing a region re-runs it; an overlap is
/// a build error naming both regions, not a tensor that quietly answers to two names.
#[allow(clippy::let_unit_value)]
const _: () = {
    // Referencing both proofs is what makes the compiler EVALUATE them; an unused `const` item is
    // not const-evaluated, so a lock nothing mentions is a lock that never runs.
    let _ = SENTINELS_ARE_INSIDE_THEIR_REGION;
    let _ = RESERVED_REGIONS_ARE_DISJOINT;
    let _ = EVERY_NAMED_REGION_RESOLVES;
};

/// ⭐⭐⭐ EVERY NAME THIS MODULE ADDRESSES A REGION BY, RESOLVED AT COMPILE TIME.
///
/// [`reserved_region`] panics on a name the table does not hold, and a `const fn` only panics when it
/// is EVALUATED — so a rename would break the bake at whatever site first called it, or not at all if
/// that site is behind a `cfg`. Naming them here makes the lookup run at every build.
pub const EVERY_NAMED_REGION_RESOLVES: () = {
    assert!(reserved_region("sentinels").base == u32::MAX - 1);
    assert!(reserved_region("scalarmul_scale").slots > 0);
    assert!(reserved_region("kv_block_index").base == KV_BLOCK_INDEX_TID);
    assert!(reserved_region("rope_p_class").base == ROPE_P_CLASS_BASE);
    assert!(reserved_region("identity_class").base == IDENTITY_CLASS_BASE);
    assert!(reserved_region("router_const").base == ROUTER_CONST_BASE);
    assert!(reserved_region("router_const").floor() < ROUTER_CONST_BASE);
    assert!(reserved_region("kct_resident").slots > 0);
};

pub const SENTINELS_ARE_INSIDE_THEIR_REGION: () = {
    // ⛔ THE ONE-OFF SENTINELS ARE DECLARED SEPARATELY (`ROPE_P_TID` .. `IDENTITY_TID`), so the
    // region table would happily describe a span they had already outgrown. Adding a 20th sentinel
    // now fails the build instead of silently taking `scalarmul_scale_tid(0)`'s slot.
    let r = reserved_region("sentinels");
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
pub const SCALARMUL_SCALE_BASE: u32 = reserved_region("scalarmul_scale").base;

/// Reserved const TID for the `i`-th distinct ScalarMul scale.
pub fn scalarmul_scale_tid(idx: usize) -> u32 {
    reserved_region("scalarmul_scale").at(idx as u32)
}

/// Base of the RESERVED tid block for the RESIDENT per-layer Kᵀ kernel `kct` (the "kill the O(active)
/// restickify" residency fix). The Kᵀ kernel the score matmul reads was a per-step SYNTH scratch that
/// re-transposed the whole active K each step; making it a RESIDENT seg2 tensor (persists across
/// steps and layers, so the incremental slab restickify only re-transposes the current slab) needs a
/// real tid with a per-layer seg2 placement. KEYED ON THE K-CACHE SOURCE TID `k_id` (like `downproj_block_tid`)
/// so the emitter (which has layer-0 `k_id`) and the layout/guard (which iterate every layer's `k_id`)
/// compute the SAME kct tid with no layer-index handoff. A 1M gap below `DOWNPROJ_BLOCK_BASE` keeps it
/// disjoint from the down_proj blocks (which descend only ~tens-of-K) AND from real source tids (~thousands).
///
/// ⛔ ADDRESSED BY NAME. This was `RESERVED_REGIONS[2]`, and inserting the one-slot `kv_block_index`
/// region above it made `kct_resident_tid(0)` return `KV_BLOCK_INDEX_TID` — see [`reserved_region`].
pub const KCT_RESIDENT_BASE: u32 = reserved_region("kct_resident").base;
/// Reserved tid for the resident per-layer Kᵀ kernel of the K-cache source tid `k_id`.
pub fn kct_resident_tid(k_id: u32) -> u32 {
    reserved_region("kct_resident").at(k_id)
}

/// ⛔⛔⛔ THE REGRESSION THAT ADDING A REGION CAUSED, PINNED — a resident Kᵗ kernel and the gather's
/// index tensor must not be one tid.
///
/// `kct_resident_tid` / `KCT_RESIDENT_BASE` were `RESERVED_REGIONS[2]`, and the `kv_block_index` region
/// was inserted at 2 (correctly — the table is ordered high tid → low). `kct_resident_tid(0)` then
/// answered `KV_BLOCK_INDEX_TID`, and layer 0's resident Kᵗ kernel and the index table shared one
/// placement. That is exactly this file's own stated failure — *"whichever is bound last wins and the
/// other silently reads someone else's bytes"* — with the additional twist that the Kᵗ kernel is what
/// the gather READS THROUGH the index.
///
/// It surfaced only because the new region has ONE slot, so every `k_id >= 1` overflowed and failed the
/// bake. A region with room would have aliased in silence. This test does not depend on that luck.
#[test]
fn a_named_region_survives_an_insertion_above_it() {
    assert_ne!(
        kct_resident_tid(0),
        KV_BLOCK_INDEX_TID,
        "layer 0's resident Kᵗ kernel resolves to the gather's INDEX tid — the region lookup is \
         positional again, so the tensor a gather reads through the index IS the index"
    );
    assert_eq!(
        KCT_RESIDENT_BASE,
        reserved_region("kct_resident").base,
        "KCT_RESIDENT_BASE names a different region than kct_resident_tid does"
    );
    assert_eq!(
        SCALARMUL_SCALE_BASE,
        reserved_region("scalarmul_scale").base,
        "SCALARMUL_SCALE_BASE names a different region than scalarmul_scale_tid does"
    );
    // Every region this module has a door for, resolved by name and asserted to hold the tid that
    // door hands out — so a rename shows up here rather than as a neighbouring region's base.
    assert!(
        reserved_region("kct_resident").floor() <= kct_resident_tid(999),
        "kct_resident_tid walked below its own region"
    );
    assert_eq!(
        reserved_region("kv_block_index").at(0),
        KV_BLOCK_INDEX_TID,
        "the kv_block_index region's only slot is not KV_BLOCK_INDEX_TID"
    );
}

/// ⭐ THE CLASS LAWS, PINNED — a class tid answers the right table, and a uniform model sees none of it.
///
/// Class 0 of both class families KEEPS the pre-class sentinel, which is what makes a uniform model's
/// bundle byte-identical (its ops name `ROPE_P_TID`/`IDENTITY_TID` exactly as before). Classes 1.. live
/// in their own named regions, disjoint from every neighbour — including each other, which is the
/// pairing this file exists to make un-silentable: a rope class tid and an identity class tid are both
/// "MAX-minus-small" numbers a hand-assigned space would happily have collide.
#[test]
fn class_tids_keep_class_zero_on_the_sentinel_and_descend_their_own_regions() {
    // Class 0 IS the old single-class tid, for both families.
    assert_eq!(rope_p_class_tid(0), ROPE_P_TID);
    assert_eq!(identity_class_tid(0), IDENTITY_TID);
    // Class 1 descends into its own region, one rung below the kv_block_index region.
    assert_eq!(rope_p_class_tid(1), ROPE_P_CLASS_BASE);
    assert_eq!(identity_class_tid(1), IDENTITY_CLASS_BASE);
    // The two class families never share a tid.
    for i in 0..8u32 {
        for j in 0..8u32 {
            assert_ne!(
                reserved_region("rope_p_class").at(i),
                reserved_region("identity_class").at(j),
                "rope class {i} and identity class {j} are one tid — two kernel tables share one \
                 placement"
            );
        }
    }
    // A class tid never leaves its region (the floor check `at` already refuses the overflow, but the
    // law is worth stating where the class scheme is defined). With 8 slots and class 0 on the
    // sentinel, classes 1..8 are the addressable ones.
    assert!(
        rope_p_class_tid(8) >= reserved_region("rope_p_class").floor(),
        "rope class 8 walked below the rope_p_class region"
    );
}

/// ⛔ THE OVERFLOW IS A REFUSAL, NOT A WRAP — a 9th rope class must not alias the identity classes.
/// `#[should_panic]` is the honest shape here: `TidRegion::at` is a `const fn` assert the bake hits.
#[test]
#[should_panic(expected = "reserved tid region overflow")]
fn a_ninth_rope_class_is_a_refusal_not_an_alias() {
    let _ = rope_p_class_tid(9);
}

/// The class set is DISTINCT and DESCENDING — the order the tids are assigned in, so class 0 is the
/// widest head dim and a uniform model's single class is class 0.
#[test]
fn the_class_set_is_distinct_and_descending() {
    assert_eq!(
        rope_p_class_hds([128u32, 64, 128, 256, 64].into_iter()),
        vec![256, 128, 64]
    );
    assert_eq!(rope_p_class_hds([64u32].into_iter()), vec![64]);
    let empty: std::iter::Empty<u32> = std::iter::empty();
    assert_eq!(rope_p_class_hds(empty), Vec::<u32>::new());
}
