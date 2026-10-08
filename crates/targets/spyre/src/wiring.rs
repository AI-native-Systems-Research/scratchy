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

// ── The MoE bundle tensors, one accessor per (bundle, tensor) ─────────────────
//
// A MoE bundle field holds SEVERAL tensors where every other weight kind holds one, so
// `superdsc_weights`'s per-source arms reach INSIDE the field through these. They return the
// tensor VERBATIM — staging (orientation, device-width pad) is the worker's, decided by the
// source's baked shape, exactly as for a single-tensor weight.

/// The gemma-4 router's dense `[num_experts, hidden]` projection.
pub fn gemma_router_gate(l: &scratchy_layers::layers_moe::GemmaRouterLayer) -> GpuTensor {
    l.gate
}

/// The gemma-4 router's `[num_experts]` per-expert score scale.
pub fn gemma_router_per_expert_scale(
    l: &scratchy_layers::layers_moe::GemmaRouterLayer,
) -> GpuTensor {
    l.per_expert_scale
}

/// The gemma-4 router's `[hidden]` RMSNorm gain on its input.
pub fn gemma_router_scale(l: &scratchy_layers::layers_moe::GemmaRouterLayer) -> GpuTensor {
    l.scale
}

/// The SwitchGLU experts' gate-projection codes, stacked `[E, out, in]`.
pub fn switch_glu_gate_w(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_gate_w
}

/// The SwitchGLU experts' gate-projection per-channel scale, stacked `[E, out, 1]`.
pub fn switch_glu_gate_s(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_gate_scales
}

pub fn switch_glu_up_w(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_up_w
}

pub fn switch_glu_up_s(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_up_scales
}

pub fn switch_glu_down_w(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_down_w
}

pub fn switch_glu_down_s(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_down_scales
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
    /// ⭐ THE ROUTER CONST GEOMETRY `(padded expert width W, expert count E)`, off the
    /// router_const region's own placements: W from the `[W,W]` tie table's size
    /// (`sqrt(size/2)`), and E from the layout's own `router_experts` registry entry — the one
    /// fact no placement size carries, because the pad-mask row's VALUE needs the UNPADDED
    /// count (0 below E, +inf above). k rides the top-k targets table's OWN PLACEMENT (its
    /// row count), read back by the bind when the table exists — a bundle with no RouteTopK
    /// node places no targets table and binds no k. `(0, 0)` when the bundle has no MoE
    /// router. Same "asked of the artifact" law as `ones_reduce_len`.
    pub router: (usize, usize),
    /// The top-k row count, off the `[k,W]` TARGETS table's own placement — the bind
    /// builds the target ranks `E−k+j` from it. 0 when the bundle placed no targets
    /// table (no RouteTopK node).
    pub router_k: usize,
    /// ⭐⭐⭐⭐⭐ The bundle placed `KV_BLOCK_INDEX_TID` — it emitted a GATHERED KV read, so the forward
    /// tape carries a [`ForwardKernel::KvBlockIndex`](crate::forward_tape::ForwardKernel::KvBlockIndex)
    /// step and the launch must stage a block table.
    ///
    /// Same mechanism and the same reason as [`Self::uses_identity`]: the descriptor half and the
    /// staging half of a feature must be decided by ONE artifact-derived question, because they fail
    /// asymmetrically. A step this bundle has no gather for binds bytes nothing reads; a gather with no
    /// step reads an unstaged index, and an unstaged index is ZERO — a VALID block — so every row
    /// attends row 0's keys, fluently and wrongly.
    pub gathers_kv: bool,
}

impl BakeFacts {
    /// A bundle that placed nothing — the KTIR-emulator path, which has no SuperDSC layout to
    /// ask. ⛔ NOT A DEFAULT FOR THE CARD PATH: `require_identity` refuses this on decode.
    pub const NONE: BakeFacts = BakeFacts {
        uses_ones_reduce: false,
        ones_reduce_len: 0,
        uses_identity: false,
        scalarmul_scales: &[],
        gathers_kv: false,
        router: (0, 0),
        router_k: 0,
    };

    /// Read them off the generated layout.
    pub fn of(layout: &'static bundle::BundleLayout<'static>) -> Self {
        use crate::lower_subtile_tape_to_superdsc as sd;
        let ones = layout.place_of_tid(sd::ONES_REDUCE_TID);
        // The router geometry, off the tie table's placement and the layout's own
        // `router_experts` registry entry (the E fact no placement size carries — the
        // pad-mask row's VALUE needs the unpadded count). The placement pass mints the tie
        // table at exactly `[W, W]` fp16, so `sqrt(size/2)` IS W. k rides the targets
        // table's own placement (`[k, W]` ⇒ `size/2/W` rows).
        let router = match layout.place_of_tid(sd::router_rank_tie_tid()) {
            Some(p) => {
                let w = ((p.size / 2) as f64).sqrt() as usize;
                let e = layout.router_experts as usize;
                (w, e)
            }
            None => (0, 0),
        };
        let router_k = layout
            .place_of_tid(sd::router_topk_targets_tid())
            .map(|p| (p.size / 2) as usize / router.0.max(1))
            .filter(|&k| k > 0)
            .unwrap_or(0);
        BakeFacts {
            uses_ones_reduce: ones.is_some(),
            ones_reduce_len: ones.map_or(0, |p| (p.size / 2) as usize),
            uses_identity: layout.place_of_tid(sd::IDENTITY_TID).is_some(),
            gathers_kv: layout.place_of_tid(sd::KV_BLOCK_INDEX_TID).is_some(),
            router,
            router_k,
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
            rope_class_hds: w.rope_class_hds,
            attn_class_hds: w.attn_class_hds,
            rms_invcols: w.rms_invcols,
            router: self.router,
            router_k: self.router_k,
        }
    }
}

impl Wiring {
    /// ⛔ THE CLASS SETS, CHECKED AGAINST THE PLACEMENTS — ONCE, AT LOAD.
    ///
    /// The bake used to emit the identity and rope-P tables as f32 LITERALS staged at the BASE head
    /// dim, while the placement pass sized their tids at the MAX head dim over the tape's classes.
    /// Both halves were individually "correct by construction" — the same shared helpers, the same
    /// `ir` — and their disagreement was the gemma-4 card-garbage defect: a `[64,64]` table in a
    /// `[128,128]` placement, the wide class's rope rotating pairs at half=32 and its GQA krep
    /// reading slab 1 as zeros. No layer of the old check could see it, because the old check
    /// (`verify_kernel_tables`'s literal-vs-freshcomputation form) re-staged at the SAME base head
    /// dim the emission used.
    ///
    /// The class fix moves the tables to LOAD-TIME staging (a 512² literal table is 262k f32 of
    /// expansion), so what the bake emits is the class SET — and the check that replaces the old
    /// one is the fact that actually failed: every rope-P class tid must be placed at exactly
    /// `hd_c² · 2` bytes, every identity class tid likewise, and every PLACED class tid must be in
    /// the wiring's set. A mismatch means the wiring and the layout come from different bakes —
    /// which is silent otherwise (fluent, incoherent output), and is exactly what an
    /// "emu green, card garbage" divergence looks like from the inside: the emulator executes the
    /// KTIR dialect ops and never reads these tids at all.
    pub fn verify_class_placements(
        &self,
        layout: &bundle::BundleLayout<'_>,
    ) -> Result<(), String> {
        use crate::lower_subtile_tape_to_superdsc as sd;
        // One class family's check: every class in the wiring's set is placed at exactly hd²·2
        // bytes, and every placed tid of that family is in the set. Both directions, because a
        // one-sided check passes on exactly the desync that produced this bug (a stale set naming
        // a tid the newer layout never placed, or a newer layout placing a class the wiring
        // predates).
        let check = |hds: &'static [u32],
                     tid_of: fn(usize) -> u32,
                     what: &str|
         -> Result<(), String> {
            for (class, &hd) in hds.iter().enumerate() {
                let tid = tid_of(class);
                let want = hd as u64 * hd as u64 * 2;
                match layout.place_of_tid(tid) {
                    Some(p) if p.size == want => {}
                    Some(p) => {
                        return Err(format!(
                            "class {class} of {what} (t{tid}, head_dim {hd}) is placed at {} B, \
                             but its table is {want} B — the wiring and the layout come from \
                             different bakes; a table that does not fill its placement is the \
                             wide-class-reads-zeros defect",
                            p.size,
                        ))
                    }
                    None => {
                        return Err(format!(
                            "class {class} of {what} (t{tid}, head_dim {hd}) is in the wiring's \
                             class set but the layout never placed it — the wiring and the layout \
                             come from different bakes"
                        ))
                    }
                }
            }
            // The other direction: a placed tid of this family that the wiring's set does not
            // resolve to. The family is identified by tid (`is_kernel_table_class_tid`), not by
            // size — a size match is what a desynced pair would get WRONG, not right.
            for p in layout.places.iter() {
                let bundle::PlaceId::Act(tid) = p.id else {
                    continue;
                };
                if sd::is_kernel_table_class_tid(tid, what)
                    && !hds.iter().enumerate().any(|(class, _)| tid_of(class) == tid)
                {
                    return Err(format!(
                        "the layout placed t{tid} ({} B) as a {what} class table, but the \
                         wiring's class set {hds:?} does not name it — the wiring and the layout \
                         come from different bakes",
                        p.size,
                    ));
                }
            }
            Ok(())
        };
        check(
            self.rope_class_hds,
            sd::rope_p_class_tid,
            "rope-P",
        )?;
        check(self.attn_class_hds, sd::identity_class_tid, "identity")?;
        Ok(())
    }

    /// ⭐⭐⭐⭐⭐ THE CAUSAL MASK'S ROW EXTENT vs ITS PLACEMENT — ONCE, AT LOAD.
    ///
    /// The placement reserves `nqh · mq · mq_pad` fp16 elements for `ATTN_CAUSAL_TID`
    /// (`lower_subtile_tape_to_superdsc.rs`, the `cmbytes` law), and the host stages exactly
    /// `causal_tiled(cmask, mq, mq_pad, nqh)` into it. The row count both halves share is the
    /// head count — which every host staging site used to DERIVE as `hidden / head_dim`. That
    /// quotient is the head count only for a SQUARE Q projection (`hidden == nqh·hd`): true of
    /// granite/llama, false of gemma-4 (production hidden 3840 against nqh·hd 4096/8192; the
    /// tiny-allglobal fixture's hidden 128 against 4·64). A host staging at the quotient binds
    /// only that many of the baked `nqh·mq` rows, and `refill_activations`' bind guard refuses
    /// OVER-binds only — the shortfall is silent, and the ops read the unstaged tail of the
    /// ADDITIVE mask as ZERO, i.e. as "no mask". MEASURED on the tiny fixture: heads 2-3 of
    /// every layer ran with no causal mask, heads 0-1 exact.
    ///
    /// The fix bakes the count ([`Geometry::nqh`], from the same bounds the tape was lowered
    /// at). This check is its guard: the placement's byte size must be exactly
    /// `nqh · mq · score_width · 2`, so a staging shape built on any other head count — the
    /// quotient, a stale wiring, a different model's geometry — is a load-time refusal instead
    /// of a silent partial mask. Same shape of check as [`Self::verify_class_placements`], for
    /// the same reason: the emulator executes the KTIR dialect ops and never reads this tid, so
    /// nothing on its path can see the disagreement.
    pub fn verify_causal_placement(
        &self,
        layout: &bundle::BundleLayout<'_>,
        mq: u32,
        score_width: u32,
    ) -> Result<(), String> {
        use crate::lower_subtile_tape_to_superdsc as sd;
        let nqh = self.geometry.nqh as u64;
        let want = nqh * mq as u64 * score_width as u64 * 2;
        match layout.place_of_tid(sd::ATTN_CAUSAL_TID) {
            Some(p) if p.size == want => Ok(()),
            Some(p) => Err(format!(
                "the causal mask (t{}) is placed at {} B, but this wiring's geometry bakes \
                 {nqh} query head(s) × {mq} row(s) × {score_width} score width = {want} B. The \
                 host stages `nqh·mq` mask rows; a placement that does not match is a \
                 wiring/layout desync — and a STAGING at any other head count (the old \
                 `hidden / head_dim` derivation, right only for a square Q projection) binds \
                 only part of the placement, which the bind guard cannot see because it \
                 refuses over-binds only. The unstaged tail of an ADDITIVE mask reads as ZERO, \
                 i.e. as no mask.",
                sd::ATTN_CAUSAL_TID,
                p.size,
            )),
            None => Err(format!(
                "the causal mask (t{}) is nowhere in this layout, but every attention bundle \
                 places it — the wiring and the layout come from different bakes",
                sd::ATTN_CAUSAL_TID,
            )),
        }
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
        gathers: GathersKv,
        n_consts: usize,
    ) -> crate::forward_tape::ForwardShape {
        self.forward_shape_within(
            self.baked_row_capacity(),
            rows,
            owns_prefix,
            gathers,
            n_consts,
        )
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
        // ⛔ A WITNESS, NOT A `bool` — mintable only from a [`BakeFacts`] read off the bundle's own
        // layout. See [`GathersKv`].
        gathers: GathersKv,
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
            kv_block_index: gathers.get(),
            n_consts,
        }
    }
}

/// ⭐⭐⭐⭐⭐ THE PROOF THAT "THIS BUNDLE GATHERS" CAME FROM THE BUNDLE — a witness with exactly one
/// door, [`BakeFacts`], which reads the `KV_BLOCK_INDEX_TID` placement off the generated layout.
///
/// ⛔ WHY NOT A `bool`. The two halves of a gather are the emitted descriptor and the staged index
/// table, and their failure modes are not symmetric: a table nobody gathers through is inert, while a
/// gather with no table reads an UNSTAGED index — which is zero, and block 0 is a real address, so
/// every row of the batch attends row 0's keys. That is fluent, wrong output from a clean bake with no
/// counter moved. A `bool` parameter is exactly the arrangement in which a caller can answer that
/// question from something other than the artifact — a phase, a rung width, an `is_some()` on the
/// wrong option — and one wrong answer in one direction is silent corruption.
///
/// So the only way to obtain one is to hold the bundle's own facts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GathersKv(bool);

impl GathersKv {
    /// The bundle's own answer.
    pub fn of(facts: &BakeFacts) -> Self {
        GathersKv(facts.gathers_kv)
    }

    /// The same answer straight off a LAYOUT, for callers that hold one but no [`BakeFacts`] — the
    /// load-time placement check. It is the same lookup [`BakeFacts::of`] performs, spelled once here
    /// rather than duplicated at the call site, and it takes a plain reference so a non-`'static`
    /// layout can be asked.
    pub fn of_layout(layout: &bundle::BundleLayout<'_>) -> Self {
        GathersKv(
            layout
                .place_of_tid(crate::lower_subtile_tape_to_superdsc::KV_BLOCK_INDEX_TID)
                .is_some(),
        )
    }

    /// ⭐⭐⭐ THE **PER-BODY** ANSWER, off the body's own LAUNCH GROUPS — each group's
    /// `bundle::KvShifts::gathered`, which the emitter stamps on the fold group it built the gather into.
    ///
    /// ⛔ WHY A THIRD DOOR RATHER THAN REUSING THE LAYOUT ONE. [`Self::of_layout`] answers about a
    /// BUNDLE: `KV_BLOCK_INDEX_TID` is placed if ANY body of it gathers. A decode bundle holds a whole
    /// sk_bucket ladder of bodies plus their fold-fused twins, and which one a step runs is the
    /// selector's answer from the live context length — so the bundle's placement cannot answer "does
    /// the body I am about to launch gather", and answering it from the bundle is exactly the mistake
    /// this file has now recorded three times (see `superdsc_exec::Executor::step_body`).
    ///
    /// The ops are the right authority for the same reason `latch_paged_geometry` reads `page_slots` off
    /// them: a bake fact rides on the launch group it describes, so it cannot be true of a body that
    /// does not carry it.
    pub fn of_launch_groups(gathered: impl IntoIterator<Item = bool>) -> Self {
        GathersKv(gathered.into_iter().any(|g| g))
    }

    /// ⭐ THE ONE CALLER-SIDE `false` THAT IS SOUND: a launch path with no SuperDSC layout to ask (the
    /// KTIR emulator). It is spelled as its own named door rather than as `GathersKv(false)` so it
    /// appears in a grep for who claims a bundle does not gather.
    pub const fn no_layout_to_ask() -> Self {
        GathersKv(false)
    }

    pub const fn get(self) -> bool {
        self.0
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
    /// THIS layer's per-token KV width (`kv_heads * head_dim`), off the prefix-K
    /// source's own baked shape. Hybrid-attention arches (gemma-4: sliding
    /// 8×256 vs global 1×512) give different layers different widths, so a
    /// single `Geometry::kv_dim` cannot size a per-layer buffer or readback —
    /// this is the per-layer answer, from the same `tensor_shapes` the tape
    /// was lowered at.
    pub kv_width: u32,
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
    /// ⭐⭐⭐⭐⭐ THE QUERY-HEAD COUNT, baked — `num_attention_heads` from the same
    /// bounds the tape was lowered at.
    ///
    /// ⛔⛔⛔ NOT `hidden / head_dim`, WHICH IS WHAT EVERY HOST STAGING SITE USED TO
    /// DERIVE. That quotient is the head count only when the Q projection is SQUARE
    /// (`hidden == nqh·hd`) — true of granite/llama and false of gemma-4, whose
    /// `hidden` (3840) is not a multiple of any class's `nqh·hd` (16·256 = 4096,
    /// 16·512 = 8192). A host that stages the causal mask or the prefix mask tiled
    /// to `hidden/head_dim` head rows binds only that many of the baked `nqh·mq`
    /// rows — the bind guard refuses OVER-binds only, so the shortfall is silent,
    /// and the ops read the unstaged tail of an ADDITIVE mask as ZERO, i.e. as
    /// "no mask". MEASURED, gemma-4 tiny-allglobal (hidden 128, nqh 4, hd 64):
    /// the causal mask covered heads 0-1 and heads 2-3 ran with NO causal mask,
    /// every layer, prefill — card dump `sc` rows 22..43 unmasked, heads 0-1 exact.
    /// The fix is the count itself, baked here from the bounds, so the staged row
    /// count and the baked placement cannot disagree.
    pub nqh: u32,
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

/// ⭐ WHICH ROTARY TABLE A COS/SIN SOURCE CARRIES — the class of the layer(s) it
/// rotates. A uniform model has ONE class; a hybrid-attention arch (gemma-4:
/// sliding layers vs every-6th global layers) has TWO, and a table built from
/// the wrong one is silent garbage — the rotation multiplies Q/K by wrong
/// frequencies, which is fluent-but-wrong output, not a fault.
///
/// ⛔ NOT A `bool`. `local`/`global` are negations of each other and both are in
/// scope at every construction site — the exact shape that lets one land in the
/// other. The two classes here have DIFFERENT ARITY of parameters (the
/// proportional class carries its own rotated-dim count), which a bool cannot.
///
/// Every class's frequency law, from mlx `rope_utils.py` (the reference):
/// - `Default` — `freq_i = θ^(-2i/hd)` for the full head dim, NeoX pairing
///   (d, d+hd/2).
/// - `Proportional` — mlx `ProportionalRoPE`: `freq_i = θ^(-2i/hd)` for
///   `2i < rotated`, INFINITE past it (the pass-through lanes — identity, the
///   tail stays unrotated). The exponent's denominator is the FULL hd, not
///   `rotated`: that is the class's defining difference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RotaryKind {
    /// The base class: full rotary at the class's own head dim and θ.
    Default {
        /// The class's head dim — the width of one head in THIS table's
        /// consumers (sliding: `head_dim`; a uniform model: `head_dim`).
        head_dim: u32,
        /// The class's rope base θ.
        theta_bits: u32,
    },
    /// mlx `ProportionalRoPE` — partial rotary whose exponent denominator is
    /// the FULL head dim (gemma-4 global layers: hd 512, rotated 128, θ 1e6).
    Proportional {
        head_dim: u32,
        theta_bits: u32,
        /// How many leading dims of the head actually rotate; the rest are
        /// pass-through (freq = ∞ ⇒ cos 1, sin 0).
        rotated: u32,
    },
}

impl RotaryKind {
    /// The class's head dim — the width one row of its table spans.
    pub const fn head_dim(self) -> u32 {
        match self {
            Self::Default { head_dim, .. } | Self::Proportional { head_dim, .. } => head_dim,
        }
    }

    /// The class's rope base θ, as bits (see [`Geometry::rope_theta_bits`]).
    pub const fn theta(self) -> f32 {
        match self {
            Self::Default { theta_bits, .. } | Self::Proportional { theta_bits, .. } => {
                f32::from_bits(theta_bits)
            }
        }
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
    /// Rotary `cos` / `sin` sources as `(id, full column width, rotary class)`.
    /// GQA gives more than one width (Q vs K), so this is a list, not a pair;
    /// a hybrid-attention arch (gemma-4) additionally gives CLASSES different
    /// rotary tables, so every source carries its own [`RotaryKind`].
    pub cos_srcs: &'static [(u32, u32, RotaryKind)],
    pub sin_srcs: &'static [(u32, u32, RotaryKind)],
    /// Per-layer AttnDecode wiring, in layer order.
    pub layers: &'static [LayerWiring],
    pub geometry: Geometry,
    /// ⭐ THE KERNEL-TABLE CLASS SETS — this program's DISTINCT rope head dims and DISTINCT
    /// attention head dims, each sorted descending. Index `i` of the rope set ↔ the tid
    /// `rope_p_class_tid(i)` holding that class's `[hd,hd]` P table; index `i` of the attention
    /// set ↔ `identity_class_tid(i)` holding that class's `[hd,hd]` identity. Class 0 of each is
    /// the widest and keeps the pre-class sentinel (`ROPE_P_TID`/`IDENTITY_TID`), so a uniform
    /// model names the same tids it always did.
    ///
    /// ⛔ THE TABLES ARE BUILT AT LOAD, NOT BAKED AS LITERALS — a hybrid model's widest class is
    /// `hd = 512` (gemma-4 production), and a 512² table is 262,144 f32 literals for one field,
    /// which is the expansion-cost rule broken in its most expensive form. The bake carries the
    /// class HEAD DIMS (a handful of u32s); `constant_acts` stages each class's table at load
    /// through the SAME shared helpers the emission used (`stage_kernel_table` + `rope_p_entry`),
    /// so the value and the proof still cannot drift — and [`Wiring::verify_kernel_tables`]
    /// cross-checks the class set against the PLACEMENTS the bake made.
    ///
    /// (This replaces the two baked single-table fields `identity: &[f32]` / `rope_p: &[f32]`:
    /// they were staged at the BASE head dim while the placement was sized at the MAX — the
    /// gemma-4 defect, a `[64,64]` table in a `[128,128]` placement with the wide class reading
    /// zeros.)
    pub rope_class_hds: &'static [u32],
    pub attn_class_hds: &'static [u32],
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
    /// ⭐ THE DENSE fp8 K-PADS — for every arity-3 MatmulTile weight whose contraction k is not a
    /// whole 128-elem SEN143_FP8 stick: `(source id, k_pad)`. The FIRST fp8 model with a
    /// non-legal K is gemma-4-26b (dense GeGLU DOWN, k=2112 → 2176): the quantize chain and the
    /// packed weight both run at `k_pad`, so the worker's staging must WIDEN the on-disk
    /// `[n, k]` buffer's INNER axis to match the retile — a strided rebuild, not the outer-axis
    /// `resize` the N-pad uses. Baked here (from the same `ir` the retile reads) because the
    /// bank/weight orientation is a BAKE fact the worker cannot re-derive from `tensor_shapes`
    /// alone (a bank's N axis is its ROWS). Empty for every 128-aligned model — granite, g8b
    /// and g12b stage byte-identically.
    pub fp8_k_pads: &'static [(u32, u32)],
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
    /// ⭐ THE KERNEL-TABLE CLASS SETS (see [`Wiring::rope_class_hds`]). The tables themselves are
    /// BUILT HERE, at load, by [`synthetic_constants`] through the same shared helpers the
    /// emission used — the bake carries only the head dims, because a widest-class `hd=512` table
    /// is 262,144 f32 literals the expansion would otherwise emit.
    pub rope_class_hds: &'static [u32],
    pub attn_class_hds: &'static [u32],
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
    /// ⭐ THE ROUTER CONST GEOMETRY, off the bundle's own placements: `(padded expert width W,
    /// expert count E)`. W derives from the router_const region's placement exactly as the
    /// placement pass minted it (W = the `[W,W]` tie table's placement width), E from the
    /// layout's `router_experts` registry entry (the pad-mask row's value needs it and no
    /// placement size carries it). `(0, 0)` when the bundle has no MoE router — a non-MoE
    /// model binds nothing.
    pub router: (usize, usize),
    /// The top-k row count, off the `[k,W]` TARGETS table's own placement (`size/2/W`) —
    /// the bind builds the target ranks `E−k+j` from it. 0 when the bundle placed no
    /// targets table (no RouteTopK node), and then no targets are bound.
    pub router_k: usize,
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
    /// A table BUILT AT LOAD — the per-class kernel tables. Their values are pure functions of a
    /// head dim the bake carries, but a widest-class table at `hd=512` is 262,144 f32 literals the
    /// expansion would have to emit, so the bake carries the head dims and the load stages the
    /// bytes through the same shared helpers the emission used. OWNED because no `static` holds
    /// them; the bind loop reads each exactly once per step, the same read pattern `Borrowed` has.
    Owned(Vec<f32>),
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
            ConstValues::Owned(v) => std::borrow::Cow::Owned(v.clone()),
            ConstValues::Fill { value, len } => std::borrow::Cow::Owned(vec![*value; *len]),
        }
    }

    /// The values, materialised.
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            ConstValues::Borrowed(v) => v.to_vec(),
            ConstValues::Owned(v) => v.clone(),
            ConstValues::Fill { value, len } => vec![*value; *len],
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ConstValues::Borrowed(v) => v.len(),
            ConstValues::Owned(v) => v.len(),
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
    use scratchy_subtile::sdsc_abstract::{rope_p_entry, stage_kernel_table};
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
        // ⭐ ONE IDENTITY PER ATTENTION CLASS, BUILT AT LOAD. A reserved seg0 tid has no
        // RetileDescriptor, so the host fill IS the device layout, and each class's table is
        // consumed as that class's `[hd_c, hd_c]` matmul KERNEL — staged through the SAME
        // `stage_kernel_table` the emission's addressing law comes from. One table per DISTINCT
        // attention head dim because a hybrid model's single max-hd placement holding a base-hd
        // table zeroes the wide class's upper lanes (the gemma-4 defect, dump-proven).
        for (class, &hd) in env.attn_class_hds.iter().enumerate() {
            let table = stage_kernel_table(hd as usize, |i, j| {
                if i == j { 1.0 } else { 0.0 }
            });
            out.push((sd::identity_class_tid(class), ConstValues::Owned(table)));
        }
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
        // ⭐ ONE ROPE-P PER ROPE CLASS, BUILT AT LOAD — the KERNEL of `rot = matmul(x, P)`, filled
        // from the SHARED Kani-proven `rope_p_entry` so the on-card fill and the proof cannot
        // drift, and staged through the kernel layout for the same reason as the identity. A
        // composite single P is IMPOSSIBLE on a hybrid tape (the big class's contraction reads the
        // small class's nonzero block as spurious ±1 terms), so one table per DISTINCT rope head
        // dim. `hd >= 2` is the runtime's own guard on emitting a rope-P at all — a model with no
        // rope has an empty class set and binds nothing, byte-identical to before.
        for (class, &hd_c) in env.rope_class_hds.iter().enumerate() {
            if hd_c < 2 {
                continue;
            }
            let table = stage_kernel_table(hd_c as usize, |inn, o| {
                rope_p_entry(hd_c as usize, inn, o) as f32
            });
            out.push((sd::rope_p_class_tid(class), ConstValues::Owned(table)));
        }
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

    // ⭐ THE ROUTER CONST ROWS — the MoE router doors' token-independent factors, BUILT HERE at
    // load exactly like the identity/rope-P tables: the values are pure functions of the router
    // geometry the placements carry, so the bake ships only the geometry and this bind stages the
    // bytes. `(0, 0)` (no MoE router) binds nothing — a non-MoE model is byte-identical.
    let (w, e) = env.router;
    if w > 0 && e > 0 {
        // The stable-argsort tie-break table `tie[j,h'] = 𝟙[h' < j]` — j on rows, h' on lanes,
        // the orientation the rank compare's mb-broadcast multiply reads. fp16 0/1.
        let mut tie = vec![0.0f32; w * w];
        for j in 0..w {
            for h in 0..j {
                tie[j * w + h] = 1.0;
            }
        }
        out.push((sd::router_rank_tie_tid(), ConstValues::Owned(tie)));
        // The identity table, row j = the one-hot row with lane j hot. ONE [W,W] table serves
        // every per-slot one-hot read — the argsort's per-EXPERT lane writes and the top-k
        // selector's per-SLOT writes are rows of the same E-sized identity.
        let mut ident = vec![0.0f32; w * w];
        for j in 0..w {
            ident[j * w + j] = 1.0;
        }
        out.push((sd::router_identity_tid(), ConstValues::Owned(ident)));
        // The iota row `h ↦ h` (the value the top-k selector's match mask multiplies).
        let iota: Vec<f32> = (0..w).map(|h| h as f32).collect();
        out.push((sd::router_topk_iota_tid(), ConstValues::Owned(iota)));
        // The top-k TARGET-RANK table `[k,W]`, row j a uniform splat of `E−k+j` — the
        // compare factor the top-k door's full-width `equal` reads mb-broadcast.
        // Bound only when the bundle PLACED it (a tape with a RouteTopK node); the
        // row count is the placement's own.
        if env.router_k > 0 {
            let k = env.router_k;
            let mut targets = vec![0.0f32; k * w];
            for j in 0..k {
                let t = (e - k + j) as f32;
                targets[j * w..(j + 1) * w].fill(t);
            }
            out.push((sd::router_topk_targets_tid(), ConstValues::Owned(targets)));
            // The SOFTMAX PAD-MASK row: 0 in lanes 0..k (the real top-k scores —
            // `x + mask` leaves them bit-identical), −inf in lanes k..W (the pad —
            // `exp(−inf)=0`, so a pad lane never contributes to the denominator).
            let sm_mask: Vec<f32> = (0..w)
                .map(|h| if h < k { 0.0 } else { f32::NEG_INFINITY })
                .collect();
            out.push((sd::router_sm_mask_tid(), ConstValues::Owned(sm_mask)));
        }
        // The two uniform sanitize rows: +inf (sorts last) and −inf (zeroes under exp).
        out.push((
            sd::router_pad_hi_tid(),
            ConstValues::Fill {
                value: f32::INFINITY,
                len: w,
            },
        ));
        out.push((
            sd::router_pad_lo_tid(),
            ConstValues::Fill {
                value: f32::NEG_INFINITY,
                len: w,
            },
        ));
        // The ARGSORT PAD-MASK row: 0 in lanes 0..E (the real experts — `maximum(x, mask)` leaves
        // them alone), +inf in lanes E..W (the producer's zero-padded lanes — `maximum` forces
        // them to sort last, so a zero pad lane can never outrank a negative real score).
        let mask: Vec<f32> = (0..w)
            .map(|h| if h < e { 0.0 } else { f32::INFINITY })
            .collect();
        out.push((sd::router_pad_mask_tid(), ConstValues::Owned(mask)));
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

    fn env(rows: usize) -> ConstantEnv {
        ConstantEnv {
            hidden: 2048,
            head_dim: 64,
            mq_pad: 64,
            uses_ones_reduce: true,
            ones_reduce_len: 0,
            uses_identity: true,
            rows,
            scalarmul_scales: &[1.5, 2.5],
            // A uniform-model class set: one rope class, one attention class, both the base 64 —
            // which resolves to the class-0 sentinels `ROPE_P_TID`/`IDENTITY_TID`, exactly what a
            // single-class model's bind must still name.
            rope_class_hds: &[64],
            attn_class_hds: &[64],
            rms_invcols: Vec::leak(vec![1.0f32 / 2048.0; 64]),
            // No MoE router in this fixture's model — the router consts bind nothing.
            router: (0, 0),
            router_k: 0,
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

    /// ⭐ THE PER-CLASS BIND — a hybrid class set binds one P and one identity PER CLASS, at that
    /// class's own tid, each table staged through `stage_kernel_table` at that class's head dim.
    /// This is the load-time half of the gemma-4 fix: the bake carries the head dims, and the
    /// tables are built HERE (a 512² literal table would be 262k f32 of expansion).
    #[test]
    fn a_hybrid_class_set_binds_one_table_per_class_at_its_own_tid() {
        use crate::lower_subtile_tape_to_superdsc as sd;
        use scratchy_subtile::sdsc_abstract::{rope_p_entry, stage_kernel_table};
        let mut env = env(1);
        env.rope_class_hds = &[128, 64];
        env.attn_class_hds = &[128, 64];
        env.uses_identity = true;
        let consts = synthetic_constants(&env);
        let get = |tid: u32| {
            consts
                .iter()
                .find(|(t, _)| *t == tid)
                .map(|(_, v)| v.to_vec())
        };
        // One P per class, at the class tids (class 0 = the sentinel).
        for (class, hd) in [(0usize, 128u32), (1, 64)] {
            let tid = sd::rope_p_class_tid(class);
            let table = get(tid).unwrap_or_else(|| panic!("rope class {class} (t{tid}) not bound"));
            let want = stage_kernel_table(hd as usize, |inn, o| {
                rope_p_entry(hd as usize, inn, o) as f32
            });
            assert_eq!(table.len(), hd as usize * hd as usize);
            assert_eq!(table, want, "rope class {class}'s table is hd={hd}'s P");
            // The identity, likewise.
            let itid = sd::identity_class_tid(class);
            let itable = get(itid)
                .unwrap_or_else(|| panic!("identity class {class} (t{itid}) not bound"));
            let iwant = stage_kernel_table(hd as usize, |i, j| {
                if i == j { 1.0 } else { 0.0 }
            });
            assert_eq!(itable, iwant, "identity class {class}'s table is hd={hd}'s");
        }
        // And the tables are DIFFERENT — the composite-single-P mistake would make class 1's table
        // the max-hd one (the nonzero block in the wrong places for a 64-wide rotation).
        assert_ne!(
            get(sd::rope_p_class_tid(0)).unwrap(),
            get(sd::rope_p_class_tid(1)).unwrap(),
            "the two rope classes bound the SAME P — the composite-single-P defect"
        );
    }

    /// ⭐ THE LOAD-TIME CROSS-CHECK — `verify_class_placements` fires on exactly the desync that
    /// produced the card garbage: a wiring/layout pair whose class set and placements disagree.
    /// Both directions, because a one-sided check passes on the mismatch that actually happened
    /// (the value/placement size disagreement) while missing the stale-set direction.
    #[test]
    fn the_class_placement_cross_check_fires_on_a_desynced_pair() {
        use crate::lower_subtile_tape_to_superdsc as sd;
        // A wiring with the gemma-4 tiny-allglobal class sets.
        let wiring = Wiring {
            rope_class_hds: &[128, 64],
            attn_class_hds: &[128, 64],
            ..uniform_wiring()
        };
        // A layout that places BOTH rope classes at the MAX size — the defect's exact shape: the
        // placement pass's max-fold, held against the class set.
        let lay_max = class_layout(&[(sd::rope_p_class_tid(0), 128), (sd::rope_p_class_tid(1), 128)]);
        let why = wiring
            .verify_class_placements(&lay_max)
            .expect_err("a class-1 placement sized at the max must refuse");
        assert!(
            why.contains("t") && why.to_lowercase().contains("class 1"),
            "the refusal names the class and its tid, got {why:?}"
        );
        // A layout that places only class 0 — the stale-set direction.
        let lay_missing = class_layout(&[(sd::rope_p_class_tid(0), 128)]);
        assert!(
            wiring.verify_class_placements(&lay_missing).is_err(),
            "a class the layout never placed must refuse"
        );
        // And the AGREED pair passes — the check is not a blanket refusal.
        let lay_ok = class_layout(&[
            (sd::rope_p_class_tid(0), 128),
            (sd::rope_p_class_tid(1), 64),
            (sd::identity_class_tid(0), 128),
            (sd::identity_class_tid(1), 64),
        ]);
        wiring
            .verify_class_placements(&lay_ok)
            .expect("an agreed wiring/layout pair passes");

        // ⭐ A UNIFORM MODEL'S PAIR, STILL ONE CLASS — the byte-identity case: class 0 is the
        // sentinel and the check is satisfied by the single placement it always had.
        let uniform = Wiring {
            rope_class_hds: &[64],
            attn_class_hds: &[64],
            ..uniform_wiring()
        };
        let lay_uniform = class_layout(&[
            (sd::rope_p_class_tid(0), 64),
            (sd::identity_class_tid(0), 64),
        ]);
        uniform
            .verify_class_placements(&lay_uniform)
            .expect("a uniform model's single-class pair passes");
    }

    /// ⭐⭐⭐⭐⭐ THE CAUSAL-MASK HEAD-COUNT DEFECT, PINNED — gemma-4 tiny-allglobal's exact
    /// geometry (hidden 128, nqh 4, head_dim 64), where `hidden / head_dim` = 2 ≠ 4.
    ///
    /// The host used to stage `causal_tiled(cmask, mq, mq_pad, hidden / head_dim)` rows into a
    /// placement sized `nqh · mq · mq_pad`. The quotient is the head count only for a SQUARE Q
    /// projection (`hidden == nqh·hd`) — true of granite/llama, false of gemma-4 at production
    /// scale too (hidden 3840 against nqh·hd 4096/8192). The shortfall was silent: the bind
    /// guard refuses OVER-binds only, and the unstaged tail of the ADDITIVE mask reads as ZERO,
    /// i.e. as "no mask". MEASURED on the tiny fixture: heads 2-3 of every layer ran unmasked.
    ///
    /// This test pins both halves of the fix: the BAKED count stages the full placement (the
    /// cross-check passes at `Geometry::nqh`), and the QUOTIENT's staging does not cover the
    /// placement — which the load-time cross-check refuses, because its `want` is computed from
    /// the bake's own `nqh` and not from any host arithmetic.
    #[test]
    fn the_causal_mask_head_count_is_the_bake_not_the_hidden_quotient() {
        use crate::lower_subtile_tape_to_superdsc as sd;
        // The gemma-4 tiny-allglobal geometry: hidden 128, FOUR query heads, head_dim 64.
        // `hidden / head_dim` = 2 — the wrong count every host staging site used to derive.
        let wiring = Wiring {
            geometry: Geometry {
                hidden: 128,
                head_dim: 64,
                nqh: 4,
                ..uniform_wiring().geometry
            },
            ..uniform_wiring()
        };
        // The tiny fixture's prefill: mq = 11, mq_pad = 64. The placement law:
        // `nqh · mq · score_width · 2` bytes.
        let (mq, width) = (11u32, 64u32);
        let placed_bytes = 4u64 * mq as u64 * width as u64 * 2;
        let mut lay = class_layout(&[]);
        lay.places.to_mut().push(bundle::Placement {
            id: bundle::PlaceId::Act(sd::ATTN_CAUSAL_TID),
            segment: 0,
            bank: 0,
            offset: 0,
            size: placed_bytes,
            is_logits: false,
        });
        // The BAKED count's staging covers the placement — the pair agrees.
        wiring
            .verify_causal_placement(&lay, mq, width)
            .expect("the bake's own nqh stages exactly its placement");
        // THE DEFECT'S SHAPE: a staging at the QUOTIENT covers only `2·mq` of the `4·mq` rows.
        // The quotient staging itself is silent at bind time (under-binding is allowed), so the
        // CHECK is what must fire — its `want` comes from the bake's `nqh`, never from host
        // arithmetic, and the two counts disagree on exactly this geometry.
        let quotient = wiring.geometry.hidden / wiring.geometry.head_dim;
        assert_ne!(
            quotient, wiring.geometry.nqh,
            "the fixture must be one where the quotient is NOT the head count, or this test \
             pins nothing"
        );
        let quotient_bytes = quotient as u64 * mq as u64 * width as u64 * 2;
        assert!(
            quotient_bytes < placed_bytes,
            "the quotient's staging is a PARTIAL bind, which is the silent direction of the \
             bind guard"
        );
        // And the cross-check fires when the placement does not match the bake's head count —
        // here simulated from the layout side (a placement sized at the quotient), which is the
        // same disagreement seen from the other half.
        let mut lay_quotient = class_layout(&[]);
        lay_quotient.places.to_mut().push(bundle::Placement {
            id: bundle::PlaceId::Act(sd::ATTN_CAUSAL_TID),
            segment: 0,
            bank: 0,
            offset: 0,
            size: quotient_bytes,
            is_logits: false,
        });
        let why = wiring
            .verify_causal_placement(&lay_quotient, mq, width)
            .expect_err("a placement sized at the hidden/head_dim quotient must refuse");
        assert!(
            why.contains("4 query head"),
            "the refusal names the bake's head count, got {why:?}"
        );
        // And a layout with no causal placement at all refuses — the stale-bake direction.
        let lay_missing = class_layout(&[]);
        assert!(
            wiring.verify_causal_placement(&lay_missing, mq, width).is_err(),
            "a wiring whose layout never placed the causal mask must refuse"
        );
    }

    /// A minimal `Wiring` for the cross-check tests — only the class fields are load-bearing.
    fn uniform_wiring() -> Wiring {
        Wiring {
            result: 0,
            num_sources: 0,
            attn_mask: None,
            decode_position: 0,
            tensor_shapes: &[],
            embed_src: 0,
            cos_srcs: &[],
            sin_srcs: &[],
            layers: &[],
            geometry: Geometry {
                hidden: 128,
                kv_dim: 128,
                vocab: 512,
                layers: 1,
                head_dim: 64,
                nqh: 2,
                rope_theta_bits: 0,
            },
            rope_class_hds: &[],
            attn_class_hds: &[],
            rms_invcols: &[],
            scalarmul_scales: &[],
            fp8_k_pads: &[],
        }
    }

    /// A layout placing one `[hd,hd]` fp16 table per `(tid, hd)` pair.
    fn class_layout(tables: &[(u32, u32)]) -> bundle::BundleLayout<'static> {
        bundle::BundleLayout {
            places: std::borrow::Cow::Owned(
                tables
                    .iter()
                    .map(|&(tid, hd)| bundle::Placement {
                        id: bundle::PlaceId::Act(tid),
                        segment: 0,
                        bank: 0,
                        offset: 0,
                        size: hd as u64 * hd as u64 * 2,
                        is_logits: false,
                    })
                    .collect(),
            ),
            ..Default::default()
        }
    }
}
