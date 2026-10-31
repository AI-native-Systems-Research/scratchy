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
use crate::lower_subtile_tape_to_superdsc::{DeviceWidth, row_footprint_bytes};
pub use ktir_superdsc::element_arrangement::{
    DL16TOFP32_PERMUTATION, ElementArrangement, dl16tofp32_stagger,
};
use ktir_superdsc::superdsc_opspec::Df;
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
            rope_class_hds: w.rope_class_hds,
            attn_class_hds: w.attn_class_hds,
            rms_invcols: w.rms_invcols,
        }
    }
}

impl Wiring {
    /// ⭐ IS THE RUNG'S RESULT THE PICKED TOKEN ITSELF? — the one question the
    /// worker asks of [`Self::result_dtype`]. An integer result (the device
    /// argmax's `i32` idx stick today) states the device already picked: the
    /// step leaves the result on the device, reads the stick's raw bytes at
    /// this dtype, and the host's row-argmax is replaced by the stick's first
    /// word. Keyed on the DTYPE the result's own producer stated, never on the
    /// op's name — a second kernel with an index result rides the same branch.
    pub const fn result_is_index(&self) -> bool {
        matches!(
            self.result_dtype,
            scratchy_tensors::DType::I32 | scratchy_tensors::DType::U32
        )
    }

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
    pub fn verify_class_placements(&self, layout: &bundle::BundleLayout<'_>) -> Result<(), String> {
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
                        ));
                    }
                    None => {
                        return Err(format!(
                            "class {class} of {what} (t{tid}, head_dim {hd}) is in the wiring's \
                             class set but the layout never placed it — the wiring and the layout \
                             come from different bakes"
                        ));
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
                    && !hds
                        .iter()
                        .enumerate()
                        .any(|(class, _)| tid_of(class) == tid)
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
        check(self.rope_class_hds, sd::rope_p_class_tid, "rope-P")?;
        check(self.attn_class_hds, sd::identity_class_tid, "identity")?;
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
    pub fn forward_shape(
        &self,
        // ⛔ THE ROWS THIS LAUNCH FILLS — 1 for a decode step, `mq` for a prompt chunk. NOT the
        // bundle's capacity: see [`Self::baked_row_capacity`], which this is checked against.
        rows: StagedRows,
        n_consts: usize,
    ) -> crate::forward_tape::ForwardShape {
        self.forward_shape_within(self.baked_row_capacity(), rows, self.paged, n_consts)
    }

    /// [`Self::forward_shape`] with the capacity and the paged step supplied by the CALLER rather
    /// than read off this wiring.
    ///
    /// ⭐ WHY A CALLER MAY KNOW BETTER. A decode batch rung is the same tape baked at `B` rows and is
    /// addressed by fingerprint, not through a wiring — so `Wirings.decode` carries every rung's
    /// tensor ids correctly while carrying only the single-request graph's ROW COUNT and step. A
    /// launch on the `B`-row rung is legitimate and [`Self::baked_row_capacity`] answers 1 for it, so
    /// checking against the wiring rejects a correct launch; and its step is the batch's, at the walk
    /// of the body that runs ([`PagedStep::on_body`]).
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
        // The step of the body that runs — this wiring's own for its own bundle.
        paged: Option<PagedStep>,
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
            paged,
            n_consts,
            // The phase projects straight off the wiring's own field — the
            // tip's law: `phase: self.phase`, never a second derivation.
            phase: self.phase,
        }
    }
}

/// ⭐⭐⭐ WHETHER A BODY GATHERS — read off its own launch groups ([`Self::of_launch_groups`]), the
/// only door: a fact that differs per BODY, never asked of a bundle or a caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GathersKv(bool);

impl GathersKv {
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

/// One layer's KV pools — the K and V cache tensors its paged attention gathers from and its KV
/// write scatters into (see [`PagedStep`]).
///
/// ⛔ THE WORKER USED TO REDISCOVER THIS BY SEARCHING. It walked every manifest node for one whose
/// args mentioned the mask tensor and inferred the rest positionally. Emitted directly, it is two
/// integers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayerWiring {
    pub prefix_k: u32,
    pub prefix_v: u32,
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

/// ⭐ THE PAGED STEP — what a paged forward's host fills each step, and the pool it fills it against.
///
/// The attention reads its keys and values out of each layer's K and V POOLS (the cache tensors,
/// head-major `[pages, nkvh, page_slots, hd]`), gathering the pages a request's block table names;
/// the KV writers scatter the step's rows into them first. spyre-inference's own operands carry the
/// step (`crate::paged_step` builds each as their metadata builder does): the PAGE INDEX TABLE
/// (`[walk, 32]` int32, block `i`'s page at column 0), the MASK STACK (`[walk, rows, page_slots]`
/// additive, one tile per walked block), the KV WRITE INDEX (`nkvh` `[rows]` int32 indices, one per
/// kv head), the QUERY ROW INDEX (`[rows]` int32) and `kv_row_pool` (`[pages, nkvh, 1]` int32); a
/// DECODE BATCH's attention reads `rep_row_ids`, `chunk_page_ids` and `mask_by_chunk` instead
/// ([`Self::paged_rows`], [`Self::batch`]). The pools themselves are never re-staged: the device writes them and they
/// persist across steps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PagedStep {
    pub mask: u32,
    pub page_table: u32,
    pub write_rows: u32,
    pub query_rows: u32,
    pub kv_row_pool: u32,
    pub rep_rows: u32,
    pub chunk_pages: u32,
    pub chunk_masks: u32,
    /// The int32 zero stick staged beside `kv_row_pool` ([`Self::card_int32_zero_bytes`]).
    pub int32_zero: u32,
    /// Pages in each layer's pools — the scheduler's block count.
    pub pages: u32,
    /// Token slots per page — the scheduler's block size.
    pub page_slots: u32,
    /// Query rows one forward of this bundle carries.
    pub rows: u32,
    /// KV heads per layer.
    pub nkvh: u32,
    /// Blocks of the page table this bundle's attention walks — the bake's
    /// `scratchy_triton_splice::walk_pages` for the rung, so at most [`Self::pages`].
    pub walk: u32,
    /// What the rows are: one sequence's, or a decode batch's — one token of each of `rows`
    /// sequences, read by `batched_decode_head_major` ([`Self::batch`]).
    pub paged_rows: crate::paged_step::PagedRows,
}

impl PagedStep {
    /// The blocks the scheduler may hand out — every page but the null page
    /// ([`crate::paged_step::page_of_block`]).
    pub const fn host_blocks(self) -> u32 {
        self.pages - 1
    }

    /// The pool's token capacity: every page's slots.
    pub const fn capacity(self) -> usize {
        self.pages as usize * self.page_slots as usize
    }

    /// ⭐ THE BUNDLE'S REACH — the positions its attention's walk covers. A step whose positions end at
    /// or below this can run on the bundle from ANY start: the mask stack decides which walked keys each
    /// row attends, so a prompt chunk over a cached prefix is this bundle's to run whenever the prefix
    /// and the chunk both sit inside the walk. A step that ends past it would lose keys.
    pub const fn reach(self) -> usize {
        self.walk as usize * self.page_slots as usize
    }

    /// A decode batch's chunking of this bundle's walk (`batched_decode_chunking`), derived from the
    /// rows and the walk so a rung's walk cannot leave a stale chunking behind; `None` for one
    /// sequence's rows.
    pub const fn batch(self) -> Option<crate::paged_step::BatchChunking> {
        match self.paged_rows {
            crate::paged_step::PagedRows::Batch => {
                Some(crate::paged_step::BatchChunking::of(self.rows, self.walk))
            }
            crate::paged_step::PagedRows::Sequence => None,
        }
    }

    /// Whether this bundle's attention reads `kv_row_pool` — spyre-inference's dispatch: the decode
    /// kernel at one query row of one sequence does; the prefill kernel past it and the batched
    /// decode kernel do not.
    pub const fn reads_kv_row_pool(self) -> bool {
        self.rows == 1 && self.batch().is_none()
    }

    /// ⭐ THE WALK'S STEP ON THE CARD — the bytes one trip moves the mask tile, and so the pitch the
    /// card holds the trip's page ids at: the relaunch shifts every step source in the activation
    /// segment by that one step (`loop_fold`'s pitched index table). The stack is stick-major
    /// (`paged_step::card_mask`), so one trip's tile is its rows' sticks in every plane of it, 128 B
    /// a row: `rows` of them for one sequence (`[rows, page_slots]`),
    /// so page table row `i` sits at `i · rows · 128` B (one int32 stick at decode), and
    /// `entries · nkvh` of them for a decode batch (`[J, B, nkvh, 1, page_slots]`), so chunk `c`'s
    /// ids sit at `c · entries · nkvh · 128` B.
    pub const fn card_walk_step(self) -> u64 {
        let rows = match self.batch() {
            Some(c) => c.entries() * self.nkvh,
            None => self.rows,
        };
        rows as u64 * Df::Fp16.elems_per_stick() as u64 * 2
    }

    /// `kv_row_pool` on the card: each page's run of `nkvh` rows held at
    /// [`INDEX_RUN_WORDS`](ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS) int32
    /// words, the run the decode walk's chained index copies per page.
    pub const fn card_kv_row_pool_bytes(self) -> u64 {
        self.pages as u64 * ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as u64 * 4
    }

    /// The int32 zero stick on the card: one run of
    /// [`INDEX_RUN_WORDS`](ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS) i32 zeros,
    /// the add identity the decode walk's chained index copies each run against.
    pub const fn card_int32_zero_bytes(self) -> u64 {
        ktir_superdsc::emit::lower_ktir_to_superdsc::INDEX_RUN_WORDS as u64 * 4
    }

    /// ⭐ THE STEP'S CARD PLACEMENTS — `(tid, seat, bytes)` for every source this bundle's attention
    /// reads, sized for the pool's whole walk (a body walking fewer blocks stages and reads a
    /// prefix). The one statement the bake places by (`compute_bundle_layout`) and the card stagers
    /// ([`crate::paged_step::card_step_sources`], [`crate::paged_step::card_batch_sources`],
    /// [`crate::paged_step::card_statics`]) stage within; `self` is the bundle's own step, at its
    /// whole walk.
    pub fn card_placements(self) -> Vec<(u32, CardSeat, u64)> {
        let int32_stick = u64::from(scratchy_triton_splice::INT32_STICK);
        let sticks = |words: u64| words.div_ceil(int32_stick) * int32_stick * 4;
        let (slots, nkvh) = (u64::from(self.page_slots), u64::from(self.nkvh));
        let mut v = vec![(
            self.write_rows,
            CardSeat::Step,
            sticks(u64::from(self.rows * self.nkvh)),
        )];
        match self.batch() {
            Some(c) => {
                let (chunks, entries) = (u64::from(c.num_chunks), u64::from(c.entries()));
                v.extend([
                    (self.rep_rows, CardSeat::Step, sticks(entries)),
                    (
                        self.chunk_pages,
                        CardSeat::Step,
                        chunks * self.card_walk_step(),
                    ),
                    (
                        self.chunk_masks,
                        CardSeat::Step,
                        chunks * entries * nkvh * slots * 2,
                    ),
                ]);
            }
            None => v.extend([
                (
                    self.page_table,
                    CardSeat::Step,
                    u64::from(self.pages) * self.card_walk_step(),
                ),
                (
                    self.mask,
                    CardSeat::Step,
                    u64::from(self.pages) * u64::from(self.rows) * slots * 2,
                ),
                (
                    self.query_rows,
                    CardSeat::Step,
                    sticks(u64::from(self.rows)),
                ),
            ]),
        }
        if self.reads_kv_row_pool() {
            v.extend([
                (
                    self.kv_row_pool,
                    CardSeat::Static,
                    self.card_kv_row_pool_bytes(),
                ),
                (
                    self.int32_zero,
                    CardSeat::Static,
                    self.card_int32_zero_bytes(),
                ),
            ]);
        }
        v
    }

    /// ⭐ THE STEP OF THE BODY THAT RUNS — this bundle's pool and tids at a rung's own rows (a batch or
    /// prompt-width rung is the same tape baked at `rows` rows), walking the blocks the body sweeps:
    /// `swept` slots, whole pages for every body (a `walk_buckets` rung, or the ceiling's walk).
    /// `None` when `swept` is no walk of this pool.
    pub fn on_body(
        self,
        rows: u32,
        paged_rows: crate::paged_step::PagedRows,
        swept: u32,
    ) -> Option<PagedStep> {
        let walk = swept / self.page_slots;
        (rows > 0 && walk > 0 && swept.is_multiple_of(self.page_slots) && walk <= self.pages)
            .then_some(PagedStep {
                rows,
                paged_rows,
                walk,
                ..self
            })
    }
}

/// Where a paged step's card source sits ([`PagedStep::card_placements`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardSeat {
    /// Beside the activations: staged every step, and moved by a walk's relaunch.
    Step,
    /// In the segment NEITHER relaunch shifts — not a layer's (which moves the weight and KV
    /// segments to that layer's copies) and not a walk's (which moves the step's tables): staged once,
    /// at load, and read at its placement by every layer and every trip.
    Static,
}

/// Everything the worker needs to launch one baked bundle, emitted whole.
pub struct Wiring {
    /// Tensor id of the logits.
    pub result: u32,
    /// ⭐ THE RESULT'S OWN DTYPE — the fact that decides how the worker reads the
    /// rung's answer back ([`Self::result_is_index`] is the one question asked of
    /// it). Stated by the result producer's own kernel signature when a spliced
    /// kernel produced the result, and the f16 row every native lm-head tail
    /// emits otherwise.
    ///
    /// ⛔ THE ROW REMAINS A PLACED TENSOR, READABLE BY TID, when the result is an
    /// index stick: the answer re-point is the greedy contract's redirect of the
    /// ANSWER (the host reads the picked token, not the row it was picked from),
    /// never a deletion of the row. Any consumer that needs the row can still
    /// name its tid and read it; the mint widens it, places it, and leaves every
    /// writer exactly where it was.
    pub result_dtype: scratchy_tensors::DType,
    /// Tensor ids `0..num_sources` are caller-filled.
    pub num_sources: u32,
    /// The AttnDecode runtime length mask, if the model has a maskable decode.
    pub attn_mask: Option<u32>,
    /// The paged step's sources and pool — see [`PagedStep`]. `None` for a graph with no attention.
    pub paged: Option<PagedStep>,
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
    /// The ScalarMul weight folds the baked programs carry: `(tensor id, multiplier)` for each
    /// matmul weight the worker must scale at load — `(x·W)·s == x·(W·s)`, so the folded
    /// multiplier's device op is gone and the scale rides the staged weight bytes instead
    /// ([`crate::spyre_load`]'s `stage_bound_weights` applies it, the one runtime half of the fold).
    ///
    /// ⛔ THE MULTIPLY IS POST-STAGING, PRE-CARD — the same staged fp16 bytes the un-folded bundle
    /// would have shipped, times the multiplier, once, at load. Not a per-forward cost and not a
    /// requantize: dense weights scale in place, fp8 dequant rows scale their `ws` row.
    pub weight_scale_folds: &'static [(u32, f32)],
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
    /// alone. Empty for every 128-aligned model — granite, g8b
    /// and g12b stage byte-identically.
    pub fp8_k_pads: &'static [(u32, u32)],
    /// ⭐ THE LOAD-TIME CONSTANTS — `(tid, kind)` for every resident buffer the
    /// graph's own producer minted below the `SourceBinding` layer: staged ONCE
    /// at load through the kind's builder ([`LoadConst::build`]), never a
    /// per-step source, never baked as literals. The rope tables' own law,
    /// stated as a LIST: a second spliced kernel with staged tables adds one
    /// [`LoadConst`] kind and rides this same field — no new wiring surface, no
    /// per-op field, no consumer that names the op. The ids are the graph's own
    /// tensor ids (the same ids the placement pass places).
    pub load_consts: &'static [(u32, LoadConst)],
    /// ⭐ THE SAMPLER'S PER-STEP SEAT — `Some` exactly when the rung carries the
    /// token draw (the minted `Sample` node), naming the TWO sources the draw
    /// reads FRESH EVERY DECODE STEP. A NAMED CLASS, never a bare tid pair
    /// riding an arithmetic: the mint inserts the pair at the graph's source
    /// ceiling, and a consumer that recomputed those tids would be a second
    /// derivation free to disagree with the mint (the `weight_copies` one-mint
    /// law). The class is the source gate's own spelling too: the seat's tids
    /// count as filled ONLY when bound THIS STEP — a decode launch without a
    /// fresh seed/inv_t bind REFUSES LOUDLY (the PrefixMask precedent), never
    /// a held default, because a held seed row is a held TOKEN — every
    /// subsequent draw would repeat the same pick wherever temperature was
    /// live. `None` on every rung that mints no draw (prefill rungs, host-sampled
    /// emulators), and a `None` on a rung the worker samples through is a
    /// wiring/consumer mismatch the launch gate names.
    pub sample_seat: Option<SampleSeat>,
    /// ⭐ THE FORWARD'S OWN PHASE — the same
    /// [`crate::forward_tape::ForwardPhase`] the tape's shape carries, spelled
    /// once here and projected into the shape (`phase: self.phase`, never a
    /// second derivation). `Decode` carries the chained head's stick sources
    /// ([`ChainWiring`], the mint's payload); `Prefill` is every unchained
    /// program, byte-identical behavior to before the mint existed.
    ///
    /// ⛔ THIS ONE FIELD IS THE WORKER'S WHOLE CONTRACT with the chain: on
    /// `Decode`, the per-step embed-row push is REPLACED by the sticks' seed
    /// (the decompose of the token the step feeds, bound through the raw u32
    /// channel) — the gather reads the sticks, never a host-pushed row. The
    /// per-step path STAGES the seed every step: an idempotent restatement of
    /// what the device's own ChainStage wrote (step t feeds the token step
    /// t-1 picked, which is exactly what step t-1's stage wrote into the
    /// sticks), so the first play — the prompt's final token, which no stage
    /// has written yet — is the one that matters. The block path stages the
    /// seed once per block; the device's own stage writes carry every step
    /// after.
    ///
    /// A decode wiring that reaches this field carries `Decode` BY
    /// CONSTRUCTION: the expansion's mint refuses any decode-slot rung that
    /// would not mint (untied embedding, no EmbeddedHidden source), so no
    /// emitted decode wiring exists without the chain.
    pub phase: crate::forward_tape::ForwardPhase,
}

/// The chained head's stick sources — the persistent state the mint threaded
/// between steps ([`Wiring::phase`]'s `Decode` payload). `s_src`/`l_src` are the
/// `[1, CHAIN_IDX_BLOCK]` SENUINT32 sources the ChainStage WRITES each step
/// and the EmbedRowGather READS the next: the slab-gather entry words and the
/// transposed row's own stick ramp. `k` is the entry count
/// (`chain_slab_entries`' answer for the graph's hidden, stated once at the
/// mint): both host seeds compute the s stick's lane pattern
/// `k·(t>>6) + (i mod k)` from it. `hidden` is the graph's own hidden width
/// (the EmbedRowGather node's own field, stated once at the mint): the l
/// seed's LANE RAMP `64·(i mod (hidden/64)) + (t&63)` reads its entry count
/// from it — the one spelling every l-law site shares, never re-derived
/// downstream. `replaces` is the row the chain REPLACED —
/// the caller-filled `[1, hidden]` source whose host filler the chain removed
/// (the tape's ChainSeed steps replaced the EmbedRow push), stated once at
/// the mint and read by the layout stamp's resident excuse and the bake's
/// binding-check arm; the worker never names it.
#[derive(Clone, Copy, Debug)]
pub struct ChainWiring {
    pub s_src: u32,
    pub l_src: u32,
    pub k: u32,
    pub hidden: u32,
    pub replaces: u32,
}

/// ⭐ THE GRAPH'S OWN CHAIN FACT — read once off the mint by `chain_of_ir`
/// (lower_subtile_tape_to_ktir), the one derivation. `Chained` carries the
/// payload BY TYPE; `Unchained` is every graph the mint did not touch —
/// prefill walks, batch ladder rungs, the symbolic CB batch, the emulator
/// tapes — not a missing anything, the graph's own state: a graph landing
/// there is one of those or a defect to investigate, never a silent
/// fallback. The EMITTER derives the forward phase from the SLOT
/// ([`Wiring::phase`]); this fact is the payload, and the two disagreeing is
/// the E0004 belt at emission.
#[derive(Clone, Copy)]
pub enum ChainFact {
    Chained { chain: ChainWiring },
    Unchained,
}

/// ⭐ A LOAD-TIME CONSTANT'S KIND — the shared builder that makes its bytes.
/// A kind names a BUILDER ([`Self::build`]), never a consumer: this enum is
/// the producer's statement of its host contract, translated once into the
/// generic [`Wiring::load_consts`] list, and everything downstream (the bake's
/// numeric gate, the worker's load staging) iterates the list op-blind,
/// choosing fp32-verbatim vs fp16-convert off the ENTRY'S OWN dtype.
#[derive(Clone, Copy, Debug)]
pub enum LoadConst {
    /// One of the sampler's twenty operand tables, by the kernel's operand
    /// order (the table's own position among the node's table inputs — the
    /// row, the seed and the temperature precede them and are never load
    /// constants). Bytes from [`sample_staged_tables`] — the one derivation
    /// every stager reads, so a table defect cannot hide on one leg only.
    /// `arrangement` is
    /// the BAKE'S OWN propagation fact (the propagation pass `ktir_superdsc::emit::whole_function::staggered_sources`): the table the
    /// sampler's staggered pointwise island reads lane-for-lane is staged
    /// ALREADY IN ITS ARRANGEMENT ([`dl16tofp32_stagger`]) — `build()` is pure
    /// in this field, so the worker stages what the bake states and decides
    /// nothing.
    SampleTable {
        vocab: u32,
        /// The RUNG'S OWN ROW COUNT — the mint's own `rows` fact (the solo rung
        /// at 1, every batched rung at its baked width). The ten UNIFORM tables
        /// (the hash multipliers, the wrap-fold's 2^8, the fold's five f32
        /// constants, the FEST clamp) stage the SAME word at EVERY row of a
        /// `[rows, 1]` table — the seed words' own class, so the door's `Col`
        /// operand mode reads row r's stick head like it does a seed word, at
        /// every band height (the auditor's 2026-10-27 ruling). The per-lane
        /// tables ignore it (their shapes are whole-row, functions of the vocab
        /// alone).
        rows: u32,
        operand: u8,
        arrangement: ElementArrangement,
    },
    /// One of the chained head's two decompose tables, by the ChainStage's
    /// operand order (1-based over the stage's table inputs, past the pick
    /// and the two sticks — the same convention as the sampler's nine).
    /// Bytes from [`chain_staged_tables`] — the one derivation every stager
    /// reads, so a table defect cannot hide on one leg only. `k` is the
    /// slab-gather entry count the mint stated into the node
    /// (scratchy-subtile's `chain_slab_entries` — the lane pattern's
    /// modulus; cited not linked, the leaf-crate law), and `hidden` is the
    /// graph's own hidden width (the EmbedRowGather node's own field, stated
    /// at the mint beside it — the l-tab's LANE RAMP
    /// `64·(i mod (hidden/64)) + (t&63)` reads its entry count from it), so
    /// the staging is self-contained: the fill never re-derives either.
    ChainStageTable {
        pad_vocab: u32,
        k: u32,
        hidden: u32,
        operand: u8,
    },
}

impl LoadConst {
    /// The const's bytes, true dtype, and `[rows, cols]` shape — ONE
    /// derivation per kind, shared by every stager (the bake's numeric gate
    /// and the worker's load staging both call here, so a defect cannot hide
    /// on one leg only). PURE IN THE ARRANGEMENT: a `Dl16ToFp32` entry's bytes
    /// come out already permuted into the stagger the consuming island reads,
    /// so every leg that calls here stages identical bytes by construction.
    pub fn build(&self) -> (Vec<u8>, scratchy_tensors::DType, [usize; 2]) {
        match *self {
            LoadConst::SampleTable {
                vocab,
                rows,
                operand,
                arrangement,
            } => {
                let (bytes, dt, shape) = sample_staged_tables(vocab, rows)
                    .into_iter()
                    .nth((operand - 1) as usize)
                    .expect("the mint emits operand 1..=20 — the registry's own row");
                let bytes = match (dt, arrangement) {
                    // The stagger is an f32-word-stream fact; an f16 table never carries it.
                    (scratchy_tensors::DType::F32, ElementArrangement::Dl16ToFp32) => {
                        let words: Vec<f32> = bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|p| f32::from_le_bytes(*p))
                            .collect();
                        dl16tofp32_stagger(&words)
                            .into_iter()
                            .flat_map(|w| w.to_le_bytes())
                            .collect()
                    }
                    _ => bytes,
                };
                (bytes, dt, shape)
            }
            // The fill reads the kind's own `hidden` — the ramp's entry
            // count, the mint's own statement — never re-derived here.
            LoadConst::ChainStageTable {
                pad_vocab,
                k,
                hidden,
                operand,
            } => {
                let (bytes, dt, shape) = chain_staged_tables(pad_vocab, k, hidden)
                    .into_iter()
                    .nth((operand - 1) as usize)
                    .expect("the mint emits operand 1..=2 — the registry's own row");
                (bytes, dt, shape)
            }
        }
    }

    /// The ten UNIFORM `[rows, 1]` tables — the seeds' own class, by the
    /// kernel's operand order: `hc1`/`hc2`/`hc3`/`c256` (10..=13) and
    /// `fm8`/`ftop`/`fzero`/`fhalf`/`fm22` (15..=19) are 4-byte words, `cmin`
    /// (20) is the fp16 clamp. Every other operand is whole-row (`lane` is
    /// 14) or `[1, BLOCK]` — their staging is verbatim, never spread.
    pub fn is_uniform_col(&self) -> bool {
        match *self {
            LoadConst::SampleTable { operand, .. } => {
                (10..=13).contains(&operand) || (15..=20).contains(&operand)
            }
            // The chain's decompose tables are whole-row multi-stick ramps —
            // never the seeds' one-word-per-row class.
            LoadConst::ChainStageTable { .. } => false,
        }
    }

    /// ⭐ THE CARD FORM of this const's bytes — [`Self::build`]'s VALUES
    /// arranged the way the card's placements read them. The ten UNIFORM
    /// `[rows, 1]` tables spread to each row's stick head through
    /// [`stage_col_stick_heads`] — the seat's own law, ONE function for every
    /// card stager of the class — and every other table is verbatim (a
    /// whole-row table places exactly its bytes). The worker's
    /// `bind_load_consts` stages THIS form; the emulator's legs stage
    /// [`Self::build`]'s dense form, the same split the seat already states
    /// (card: stick heads; emulator: packed at `[rows, 1]`).
    ///
    /// ⛔ A UNIFORM COL TABLE NEVER CARRIES THE STAGGER. The `Dl16ToFp32`
    /// arrangement is a lane-for-lane fact of a whole-row table the staggered
    /// pointwise island reads; a one-word-per-row table has no lane stream to
    /// permute, and a mark on one is a mint/propagation disagreement — refuse
    /// it, never permute the rows' words.
    pub fn card_bytes(&self) -> (Vec<u8>, scratchy_tensors::DType, [usize; 2]) {
        let (bytes, dt, shape) = self.build();
        // Only the sampler's own tables carry the uniform-col class; every
        // other kind — the chain's decompose ramps — falls through VERBATIM,
        // and a second such kind arrives here by this match failing to
        // compile, never by silently taking the per-row spread.
        let (rows, operand, arrangement) = match *self {
            LoadConst::SampleTable {
                rows,
                operand,
                arrangement,
                ..
            } => (rows, operand, arrangement),
            LoadConst::ChainStageTable { .. } => return (bytes, dt, shape),
        };
        if self.is_uniform_col() {
            if matches!(arrangement, ElementArrangement::Dl16ToFp32) {
                panic!(
                    "the sampler's uniform col table (operand {operand}) carries the \
                     Dl16ToFp32 stagger — a `[rows, 1]` table has no lane stream to permute; \
                     the propagation marked a table the mint knows is per-row"
                );
            }
            let stride = col_row_stride_bytes(dt);
            // The dense form carries row r's word at `r · elem` — the spread
            // reads exactly that element and lands it at the row's stick head.
            let bytes = match dt {
                scratchy_tensors::DType::I32 | scratchy_tensors::DType::F32 => {
                    stage_col_stick_heads::<4>(rows as usize, stride, |r| {
                        bytes[r * 4..r * 4 + 4]
                            .try_into()
                            .expect("a whole i32/f32 element")
                    })
                }
                scratchy_tensors::DType::F16 => {
                    stage_col_stick_heads::<2>(rows as usize, stride, |r| {
                        bytes[r * 2..r * 2 + 2]
                            .try_into()
                            .expect("a whole f16 element")
                    })
                }
                other => panic!(
                    "the sampler's uniform col table (operand {operand}) builds as {other:?} \
                     — no `[rows, 1]` card spelling is stated for that dtype"
                ),
            };
            return (bytes, dt, shape);
        }
        (bytes, dt, shape)
    }
}

/// ⭐ THE CARD'S ROW STRIDE for a `[rows, 1]` table of dtype `dt` — the
/// placement's own reservation, [`row_footprint_bytes`] at one column, so the
/// stager and the reservation cannot drift (the same derivation the seat's
/// `seed_row_words` states in words: 128 B, a whole 32-element u32 stick, for
/// the i32/f32 tables; the fp16 col table's own footprint for `cmin`).
pub fn col_row_stride_bytes(dt: scratchy_tensors::DType) -> usize {
    let df = match dt {
        scratchy_tensors::DType::F32 => Df::Fp32,
        scratchy_tensors::DType::I32 => Df::SenUint32,
        scratchy_tensors::DType::F16 => Df::Fp16,
        other => panic!(
            "no `[rows, 1]` card table of {other:?} — the sampler's col class is i32, f32 and \
             the one fp16 clamp alone"
        ),
    };
    row_footprint_bytes(1, df) as usize
}

/// ⭐⭐⭐ ONE STAGING FUNCTION FOR EVERY `[rows, 1]` CARD TABLE — the seed
/// words' own class (the auditor's 2026-10-27 ruling, Finding 1). A
/// `[rows, 1]` placement reserves ONE WHOLE STICK FOOTPRINT PER ROW
/// ([`col_row_stride_bytes`]), and the Col operand mode reads row r's word at
/// its STICK HEAD — byte `r · stride`. The worker's per-step seat
/// (`sample_binds`) and the bake's ten UNIFORM tables
/// ([`LoadConst::card_bytes`]) stage through THIS fn, so the seat's words and
/// the uniform tables can never disagree about where a row's word lives.
///
/// ⛔ THE DEFECT THIS REPLACES was measured in the code, not on the card: the
/// uniform tables were staged DENSE — `rows` words, row 0's stick alone — so
/// at any rung with `rows > 1` every row past 0 read ZERO at its stick head
/// (a zero hash multiplier folds every row's noise to the same value). The
/// seat already staged the correct `r · seed_row_words` form; this fn is that
/// law, stated once for both stagers.
pub fn stage_col_stick_heads<const N: usize>(
    rows: usize,
    stride_bytes: usize,
    elem: impl Fn(usize) -> [u8; N],
) -> Vec<u8> {
    assert!(
        stride_bytes >= N,
        "a `[rows, 1]` row stride of {stride_bytes} B cannot hold a {N}-byte element — the \
         placement's own reservation law and the element's dtype disagree"
    );
    let mut out = vec![0u8; rows * stride_bytes];
    for r in 0..rows {
        let head = r * stride_bytes;
        out[head..head + N].copy_from_slice(&elem(r));
    }
    out
}

/// ⭐ THE CHAINED HEAD'S TWO DECOMPOSE TABLES — the ChainStage's own host
/// contract, stated ONCE for every stager (the same law as
/// [`sample_staged_tables`]): the bake-time numeric gate and the worker's
/// load-time staging read this same fn, so a table defect cannot hide on one
/// leg only. In the stage's OPERAND ORDER — `stab, ltab` — each entry is
/// `(bytes, dtype, [rows, cols])`:
///
/// * `stab` — row `t`'s LANE PATTERN: lane `i` stages the slab-gather entry
///   word `k·(t>>6) + (i mod k)`, the picked token's quarter of its
///   superword index under the door's K-way page split (core `q` of the
///   gather reads word `q` as entry `q`'s index — the IBR's single-stick
///   modulo law). `k` arrives as the mint's stated entry count, never
///   re-derived here. `k = 1` degenerates to the splat byte-for-byte
///   (`i mod 1 = 0`).
/// * `ltab` — row `t`'s LANE RAMP: the pick's WHOLE l index,
///   `32·chain_l_sticks(hidden)` words, word `i` staging SOURCE STICK
///   `64·(i mod (hidden/64)) + (t&63)` of the transposed row
///   (`chain_l_stick_ramp`, the ONE spelling — self-sized at the stick
///   count, the row gather's `hidden/64` one-stick entries spread over the
///   per-stick cut's index sticks), never a splat. `hidden` arrives as the
///   mint's stated width, never re-derived here.
///
/// The decompose is ONE derivation shared by host and device: the sticks
/// staged here, the mint's reference eval, and the worker's seed (staged per
/// step in the per-step path — idempotent with the device's own writes — and
/// once per block in the block path) all compute `s = t>>6` / `l = t&63`
/// from the same pick and encode `s` in the same `k·s + (i mod k)` lane
/// pattern. Tables are functions of `pad_vocab`, `k`, and `hidden`, constant
/// across decode steps, staged ONCE.
///
/// The SENTINEL (all-NaN row, #666): the device answers `PAD_V + 4160`, so
/// the tables run to `pad_vocab + SAMPLE_SENTINEL_OFFSET + 1` rows to keep
/// that pick's gather IN EXTENT. Past `pad_vocab` every row stages the
/// sentinel's own values — `stab` = the s=0 LANE PATTERN (every lane names a
/// quarter of slab 0, the clamp's bus safety: no special-cased row class),
/// `ltab` = the SENTINEL RAMP (`chain_l_sentinel_ramp` — lanes naming the
/// transposed placement's unwritten zero tail, word 0 = `hidden` the mark: a
/// real pick's word 0 is a lane `≤ 63 < 64 ≤ hidden`, so the mark never
/// collides): the readback refusal (`checked_picked_token`) errors the step
/// BEFORE any consumer of the sentinel rows runs, so those rows are bus
/// safety, never a value the forward reads.
///
/// ⛔ THE WORDS ARE THE DEVICE ENCODING — SENUINT32 lane values as u32 LE
/// words, `DType::I32`. The staging door for I32 consts is the VERBATIM
/// one (`raw_ids`' law: LE bytes are the device encoding for raw integer
/// words), never the IEEE→SEN convert `bind_weight` would run.
pub fn chain_staged_tables(
    pad_vocab: u32,
    k: u32,
    hidden: u32,
) -> Vec<(Vec<u8>, scratchy_tensors::DType, [usize; 2])> {
    // The mint stated k; the fill TRUSTS it — and a k outside the law's
    // window would not mis-fill, it would UNNAME (k = 0 dies a divide-by-
    // zero at `i % k`; a k past CHAIN_IDX_BLOCK wraps the entry words past
    // the stick's own lane count). The law's one refusal site is the mint;
    // this guard re-states the window so a caller defect is a named
    // refusal, never an arithmetic casualty.
    assert!(
        (1..=scratchy_subtile::subtile_ir::CHAIN_IDX_BLOCK).contains(&k),
        "chain_staged_tables: k is {k} — the entry count must sit in \
         [1, CHAIN_IDX_BLOCK = {}], the window chain_slab_entries answers in \
         (the mint's stated k; a fill outside the window is a caller defect)",
        scratchy_subtile::subtile_ir::CHAIN_IDX_BLOCK
    );
    let rows =
        pad_vocab as usize + scratchy_subtile::subtile_ir::SAMPLE_SENTINEL_OFFSET as usize + 1;
    let lanes = scratchy_subtile::subtile_ir::CHAIN_IDX_BLOCK as usize;
    // The stab row is the LANE PATTERN, not a splat: lane i names entry
    // k·(t>>6) + (i mod k) of the weight's slab-major staging. Past
    // pad_vocab the base clamps to s = 0, so every lane stays a quarter of
    // slab 0 — the clamp's bus safety under the entry-word encoding.
    let mut stab = Vec::with_capacity(rows * lanes * 4);
    for t in 0..rows as u32 {
        let base = if t < pad_vocab { k * (t >> 6) } else { 0 };
        for i in 0..lanes as u32 {
            stab.extend_from_slice(&(base + i % k).to_le_bytes());
        }
    }
    // The ltab row is the pick's WHOLE LANE RAMP, not a splat: word `i` names
    // SOURCE STICK `64·(i mod (hidden/64)) + (t&63)` of the transposed row
    // (`chain_l_stick_ramp`, the ONE spelling the tape seed, the worker's
    // twin push, and this fill all share — SELF-SIZED at the stick count, so
    // one row is the whole multi-stick l index the row gather's per-stick cut
    // walks). Past pad_vocab the row is the SENTINEL RAMP
    // (`chain_l_sentinel_ramp`) — words naming the transposed placement's
    // unwritten zero tail, word 0 = `hidden` the mark.
    let l_sticks = scratchy_subtile::subtile_ir::chain_l_sticks(hidden) as usize;
    let l_words = l_sticks * lanes;
    let mut ltab = Vec::with_capacity(rows * l_words * 4);
    for t in 0..rows as u32 {
        let row = if t < pad_vocab {
            scratchy_subtile::subtile_ir::chain_l_stick_ramp(t & 63, hidden)
        } else {
            scratchy_subtile::subtile_ir::chain_l_sentinel_ramp(hidden)
        };
        // The fill and the shape are ONE law: the ramp is self-sized at the
        // stick count, so a row's length IS the placement's word count —
        // asserted here, never trusted, so the two cannot drift by a silent
        // edit in either spelling.
        assert_eq!(
            row.len(),
            l_words,
            "chain_staged_tables: the l ramp answered {} words for hidden {hidden} — \
             chain_l_sticks says {l_sticks} sticks = {l_words} words; the ONE spelling has \
             drifted",
            row.len()
        );
        for w in row {
            ltab.extend_from_slice(&w.to_le_bytes());
        }
    }
    vec![
        (stab, scratchy_tensors::DType::I32, [rows, lanes]),
        (ltab, scratchy_tensors::DType::I32, [rows, l_words]),
    ]
}

// ⭐ THE TWO CROSS-PINS — the move of `chain_slab_entries` down to
// scratchy-subtile (the leaf-crate law: the mint and both reference arms are
// unconditional subtile code) spells the budget and the cap in a second
// crate. These close the two-spelling hole: a drift in EITHER crate is a
// cargo-build failure of this one.
const _: () = assert!(
    scratchy_subtile::subtile_ir::CHAIN_SLAB_LX_BUDGET
        == ktir_superdsc::superdsc_opspec::VENDOR_LX_SCHED_BYTES,
    "the slab-gather LX budget: scratchy-subtile's spelled const must equal \
     ktir-superdsc's VENDOR_LX_SCHED_BYTES"
);
const _: () = assert!(
    scratchy_subtile::subtile_ir::CHAIN_IDX_BLOCK as u64
        == ktir_superdsc::sdsc_abstract::CopyDims::ENTRIES_PER_OP as u64,
    "the entry cap: CHAIN_IDX_BLOCK (the u32 index stick's width) must equal \
     CopyDims::ENTRIES_PER_OP (the IBR ceiling) — one stick, two spellings"
);

/// ⭐⭐⭐ THE GATE TABLE'S β IMAGE — the ONE `[rows, 1]` table whose chain crosses the
/// i32→f32 convert, staged at the convert's own device sweep (front-b-noise, auditor
/// ruling 2026-10-27). The table's reservation is [`gate_row_stride_bytes`] (two u32
/// sticks per row), and the convert's `[rows, 64]` addressing over it is STICK-BLOCKED —
/// row r's first stick is flat stick `r`, its second flat stick `rows + r` — so the image
/// is the one-stick head staging (row r's gate at byte `r · 128`, the SAME head
/// [`stage_col_stick_heads`] ever wrote, the Col operand mode's own address) followed by
/// the sweep's second sticks, ZERO: the widened words are converted and never read, and a
/// zero-filled reservation is deterministic — no uninitialized tail for the sweep to
/// consume. `head_stride` is the seed words' own one-stick stride
/// ([`col_row_stride_bytes`]); `sweep_stride` is the gate's own reservation
/// ([`gate_row_stride_bytes`]) — both passed in so this fn stages LAWS, never
/// recomputations.
pub fn stage_gate_sweep<const N: usize>(
    rows: usize,
    head_stride: usize,
    sweep_stride: usize,
    elem: impl Fn(usize) -> [u8; N],
) -> Vec<u8> {
    assert!(
        sweep_stride.is_multiple_of(head_stride) && sweep_stride >= head_stride,
        "the gate's sweep stride {sweep_stride} B is not a whole multiple of the head stride \
         {head_stride} B — the stick-blocked image requires the sweep to be whole head-sticks"
    );
    let mut out = stage_col_stick_heads::<N>(rows, head_stride, elem);
    out.resize(rows * sweep_stride, 0);
    out
}

/// ⭐ THE GATE TABLE'S OWN ROW RESERVATION — `DeviceWidth::for_convert(1)` in i32 words
/// (64 = two u32 sticks, 256 B), the SAME one decision the convert door sweeps, the
/// whole-function mint reserves, and the layout's `nbytes` places — stated here so the
/// worker's per-row write and the placement's reservation cannot drift, exactly
/// [`col_row_stride_bytes`]'s own contract for the one-stick tables.
pub fn gate_row_stride_bytes() -> usize {
    DeviceWidth::for_convert(1).get() as usize * 4
}

/// ⭐ THE SAMPLER'S PER-STEP SEAT — the named class of the five sources the
/// token draw reads FRESH EVERY DECODE STEP (the seed quadruple's four word
/// tables and inv_t). Derived ONCE off the minted
/// `Sample` node ([`lower_subtile_tape_to_ktir::sample_seat_of_ir`], the same
/// one-derivation law [`LoadConst`] rides) and carried on
/// [`Wiring::sample_seat`]; the mint's own facts, never recomputed.
///
/// ⛔ COUNTED AS FILLED ONLY WHEN BOUND THIS STEP. The source gate treats the
/// seat's tids as a class apart from every other source: a weight is staged
/// once at load, a load-const once at load, and an activation is bound before
/// the launch that reads it — but the seed words and inv_t go STALE the
/// moment a step completes, because the seed's step word is the step's own
/// freshness. A decode launch without a fresh bind of EVERY SEAT OPERAND
/// REFUSES LOUDLY, the PrefixMask precedent, never a held default: a held
/// seed row is a held
/// TOKEN (every subsequent draw repeats the same pick wherever temperature is
/// live), which is the silent-wrong-answer class the whole wiring exists to
/// refuse.
#[derive(Clone, Copy, Debug)]
pub struct SampleSeat {
    /// The SEED WORD sources' tids — the four `[rows, 1]` i32 per-row word
    /// tables of the quadruple `(seed_lo, seed_hi, step, greedy_gate)`,
    /// in that order, each `invt`'s own class (one stick per row, the word
    /// at the stick head), staged EVERY STEP by the worker (the noise itself
    /// is computed on the card from them). ⭐ NO ROW WORD (auditor ruling
    /// 2026-10-27, the P3 localization): the hash key is request-owned
    /// alone — the seed and the request's own step count — so a request
    /// draws the same noise in any baked row of any batched launch. Four
    /// tables, not one `[rows, 4]`: the Col operand mode that broadcasts a
    /// word reads the stick head ALONE (the int-bcast card leg's measured
    /// law), so every word rides its own corner-free table — the
    /// auditor's wall-4 ruling (a).
    pub seed_words: [u32; 4],
    /// The temperature reciprocal's tid — the rung's `[rows, 1]` f32 inv_t
    /// column, staged every step with the seed words.
    pub invt: u32,
    /// The rung's OWN row count — the mint's own `rows` fact (the solo rung at
    /// 1, every batched rung at its baked width), the row count of both
    /// per-row operands and of the draw's grid.
    pub rows: u32,
    /// ⭐ THE LOGITS ROW's tid — the Sample node's `inputs[0]`, the tensor the
    /// mint keeps placed and readable by tid even as it re-points the rung's
    /// RESULT at the idx stick (the pick is `[rows, BLOCK]`, not `rows` rows
    /// of vocab, so reading the result as logits refused every rung at load).
    /// The worker reads the launch's logits through this tid — the same fact
    /// the solo launch reads the pick through; a bake without a draw keeps the
    /// result AS the logits, so that consumer falls back to `result_id`.
    pub row: u32,
    /// The EMITTED vocab — the row's own width, the sentinel threshold the
    /// consumer refuses past.
    pub vocab: u32,
    /// `sample_pad_vocab(vocab)` — the width the f32 tables share.
    pub pad_vocab: u32,
    /// Every seed word table's ROW STRIDE in i32 words on the CARD — the
    /// placement's own reservation law for a `[rows, 1]` SenUint32 table
    /// (`row_footprint_bytes(1, SenUint32) / 4`, a whole 32-element u32
    /// stick = 32 words, `invt`'s own stride), so the worker's per-row write
    /// and the placement's per-row reservation cannot drift. The word of
    /// row `r` in table `k` lands at word `r · seed_row_words` of that
    /// table's region — the STICK HEAD the Col operand mode reads.
    pub seed_row_words: u32,
    /// ⭐⭐⭐ THE GATE TABLE'S OWN ROW STRIDE in i32 words — the ONE table of
    /// the quadruple whose chain crosses the i32→f32 convert
    /// (`sgate.to(tl.float32)`), so the β ruling (front-b-noise, auditor
    /// 2026-10-27) reserves it at the convert's own device sweep:
    /// `DeviceWidth::for_convert(1)` = 64 words = TWO u32 sticks per row,
    /// against the other three tables' one. The GATE ITSELF still lands at
    /// row r's stick HEAD — byte `r · 128`, the same address the one-stick
    /// law ever stated, because the convert's `[rows, 64]` addressing is
    /// stick-blocked (row r's first stick is flat stick `r`) — and the
    /// widened words past the head are the sweep's own padding: converted,
    /// never read. Staged by [`stage_gate_sweep`]; the worker checks it
    /// against [`gate_row_stride_bytes`] the same way `seed_row_words` is
    /// checked against [`col_row_stride_bytes`].
    pub gate_row_words: u32,
}

/// ⭐ THE SAMPLER'S TWENTY STAGED TABLES — the kernel's own host contract
/// (`kernels/sample_fwd.py`'s docstring), stated ONCE for every stager: the
/// bake-time numeric gate and the worker's load-time staging read this same
/// fn, so a table defect cannot hide on one leg only (the synth weights' own
/// law). The tables are functions of the vocab alone, constant across decode
/// steps, staged ONCE.
///
/// ⭐ THE SELECTION TABLES ARE F32 — the selection's own dtype. The score is a
/// sum `x·inv_t + g`; comparing it in f16 would round near-ties into ties the
/// host reference never sees, so the whole selection runs f32 and the tables
/// state the SAME numbers the f16 argmax tables carried (#813's own values),
/// re-stated in the wider dtype — exact integers either way.
///
/// ⭐ THE NOISE SECTION'S ELEVEN ARE DTYPE-FAITHFUL TO THE KERNEL'S LOADS:
/// `hc1`/`hc2`/`hc3`/`c256` are i32 (the hash's wrapping-integer class —
/// SenUint32 shares IEEE_FP32's 4-byte geometry, so the staging is verbatim
/// LE words), `lane` and the fold's five constants are f32, and `cmin` is
/// fp16 (2^-14, the smallest normal — the clamp never leaves the fp16 stick;
/// staged through the SEN convert like every fp16 weight, a REAL conversion
/// since SEN169 ≠ IEEE binary16).
///
/// In the kernel's OPERAND ORDER — `bramp, lramp, binc, linc, ones, sixtyfour,
/// padoff, gramp, ginc, hc1, hc2, hc3, c256, lane, fm8, ftop, fzero, fhalf,
/// fm22, cmin` — each entry is `(bytes, dtype, [rows, cols])`:
///
/// * `bramp` — strictly DECREASING in the block index WITHIN EACH GROUP and
///   PERIODIC across groups (`GROUP_BLOCKS - (b % GROUP_BLOCKS)`), so the
///   second `max` lands on the FIRST block holding the row's maximum inside
///   the first maximal group; its lane values top out at 64, exact at
///   ANY vocab. 0 on the pad lanes, which can never win a max the real lanes
///   contest.
/// * `lramp` — strictly decreasing within each block, PERIODIC (`BLOCK - l`),
///   so the third `max` lands on the first lane of that block.
/// * `binc` / `linc` — the INCREASING 0-based block-within-group / lane index
///   tables the one-hot reads the selected numbers back out through
///   (`binc` periodic, `GROUP_BLOCKS` wide); on the pad lanes they stage
///   `GROUP_BLOCKS` / `BLOCK` — the SENTINEL's own values.
/// * `ones` / `sixtyfour` — the `[1, BLOCK]` widen tables: the ones broadcast
///   the selected numbers out to the output stick, the scale row carries the
///   `·64` of the wide integer combine (used twice: once for the block term,
///   once squared for the group term's `·4096`; exact in f32 below 2^24).
/// * `padoff` — THE PAD LAW: 0.0 on `[0, vocab)`, f32::MIN (−FLT_MAX,
///   0xFF7FFFFF) on the pads — the fp32 reduce template's own "negInf"
///   (`summeanmaxexx2_fp32.ddl:71`), NEVER IEEE −inf (an exponent-all-ones
///   lane reads as NaN on the card and poisons the max). The kernel adds it
///   to the score LAST, so a pad lane can never win whatever the row's pad
///   bytes hold (the lm-head's zero-pad columns write 0.0 there every step;
///   NaN rides through, keeping the all-NaN sentinel intact). Carried as a
///   staged table — the same host contract as every other constant — never
///   a V-constexpr mask.
/// * `gramp` — the TWO-LEVEL RAMP's outer level: strictly DECREASING in the
///   group index (`G - g`), constant across each group's 4096 lanes, 0 on the
///   pad lanes. Its ceiling is the group count, which the admission gate
///   keeps ≤ 2048, so the FIRST maximal GROUP wins the first `max` at any
///   vocab the gate admits (a single flat block ramp would need NB = 8192 at
///   gemma-4's PAD_V 524288, past fp16's exact range: that is why the ramp is
///   two-level).
/// * `ginc` — the INCREASING 0-based group index table; `G` on the pad lanes
///   (the sentinel's group half).
/// * `hc1` / `hc2` / `hc3` — the counter hash's multipliers, the staged
///   spellings of `0x9E3779B1` / `0x85EBCA77` / `0xC2B2AE3D` (multiplication
///   by an odd constant is a bijection mod 2^32; the squares between them
///   break the linear structure an xor-shift mix would break on a richer
///   vocabulary). ⭐ `[rows, 1]` — the same word at every row's stick head,
///   THE SEEDS' OWN CLASS (auditor ruling 2026-10-27): the bake stages them
///   at the rung's own rows, so the worker writes nothing for them, and the
///   consuming multiplies read them through the int-bcast operand mode like
///   seed words at every band height — never whole-row splats (the wall-5
///   measured law: a `[1, PAD_V]` splat burns PAD_V·4 B of local store to
///   carry one repeated word).
/// * `c256` — the wrap-fold's 2^8: `h·2^8 mod 2^32` extracts the low 24 bits
///   with no compare (the device has no lane-wise integer compare). `[rows, 1]`,
///   one word per row — the seeds' own class.
/// * `lane` — the per-lane index `0..PAD_V-1` as f32 (exact below 2^24),
///   ingested by the index-cast law — the no-arange law's own reading: a
///   per-lane fact is host-staged, never a splatted interior value. The one
///   GENUINELY per-lane fact of the noise section's eleven: whole-row.
/// * `fm8` / `ftop` / `fzero` / `fhalf` / `fm22` — the fold's constants
///   `2^-8`, `2^23 - 1`, `0`, `0.5`, `2^-22`: the abs fold's scale, the
///   strict-interior reflect point, the floor catch, and the uniform grid's
///   half-ulp and scale — `u = (np + 0.5)·2^-22` lands strictly inside
///   `(0, 1)` on every attainable `np`. `[rows, 1]` each — the seeds' own
///   class, broadcast by their consuming ops.
/// * `cmin` — the fp16 island's clamp `2^-14`, the FEST log's domain guard on
///   both entries (subnormal/FTZ behavior never reached). `[rows, 1]` — the
///   `maximum`'s broadcast operand, the seeds' own class.
///
/// The SENTINEL (all-NaN row, #666): the one-hot falls on the pad lanes,
/// where the tables stage `ginc = G`, `binc = GROUP_BLOCKS`, `linc = BLOCK`,
/// so idx = `G·4096 + GROUP_BLOCKS·64 + BLOCK = PAD_V + 4160`, past the last
/// real lane.
///
/// Every selection value is a small exact integer (`G ≤ 2048` by the admission
/// gate; `padoff` stages only 0.0 and −inf); every noise value is either an
/// exact i32 constant or an exact f32/fp16 power of two, so the staging round
/// trips bit-exactly.
///
/// ⭐ THE COL ENTRIES ARE THE VALUE FORM — `rows` words, memref-dense, the
/// spelling the emulator's legs stage (a `[rows, 1]` view over these bytes
/// reads row r's word at `r`). The CARD's placements reserve a whole stick
/// footprint per row and the Col operand mode reads the STICK HEAD, so the
/// worker stages [`LoadConst::card_bytes`] — these same values spread through
/// [`stage_col_stick_heads`], the seat's own law (Finding 1, 2026-10-27: the
/// dense form staged onto the card left every row past 0 reading ZERO).
pub fn sample_staged_tables(
    vocab: u32,
    rows: u32,
) -> Vec<(Vec<u8>, scratchy_tensors::DType, [usize; 2])> {
    let pad_v = scratchy_subtile::subtile_ir::sample_pad_vocab(vocab) as usize;
    let block = scratchy_subtile::subtile_ir::SAMPLE_BLOCK as usize;
    let group_blocks = scratchy_subtile::subtile_ir::SAMPLE_GROUP_BLOCKS as usize;
    let groups = scratchy_subtile::subtile_ir::sample_groups(pad_v as u32) as usize;
    let v = vocab as usize;
    // A whole-row f32 table under a per-lane rule: lane `i < vocab` takes
    // `real(i)`, a pad lane takes `pad`.
    let f32_row = |real: &dyn Fn(usize) -> usize, pad: usize| -> Vec<u8> {
        (0..pad_v)
            .flat_map(|i| {
                let x = if i < v { real(i) } else { pad };
                (x as f32).to_le_bytes()
            })
            .collect()
    };
    // A whole-row i32 splat's per-row form — the hash constants' own value
    // (SenUint32, the same 4-byte geometry as f32; the LE words ARE the device
    // encoding), ONE WORD PER ROW of a `[rows, 1]` table. ⭐ THE WALL-5 MEASURED
    // LAW holds — a uniform constant is never a whole-row splat (which burns
    // PAD_V·4 B of local store to carry one repeated word; ten of them tip the
    // 2 MiB LX) — and ⭐ THE 2026-10-27 RULING restages it as the SEED WORDS'
    // OWN CLASS: `[rows, 1]` at the rung's own row count, the same word at
    // every row's stick head, so the `Col` operand mode reads row r's word
    // exactly as it reads a seed word, at every band height (a `[1, 1]` table
    // at a multi-row rung would demand a scalar broadcast no door states).
    let i32_col =
        |x: i32| -> Vec<u8> { (0..rows as usize).flat_map(|_| x.to_le_bytes()).collect() };
    // A whole-row f32 splat's per-row form — the fold's constants, the same law.
    let f32_col =
        |x: f32| -> Vec<u8> { (0..rows as usize).flat_map(|_| x.to_le_bytes()).collect() };
    vec![
        // bramp: GROUP_BLOCKS - (b % GROUP_BLOCKS), periodic, pads 0.
        (
            f32_row(&|i| group_blocks - (i / block) % group_blocks, 0),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // lramp: BLOCK - l, periodic, pads 0.
        (
            f32_row(&|i| block - i % block, 0),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // binc: the block-within-group index, periodic, pads GROUP_BLOCKS.
        (
            f32_row(&|i| (i / block) % group_blocks, group_blocks),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // linc: the lane index, pads BLOCK.
        (
            f32_row(&|i| i % block, block),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // ones: [1, BLOCK] of 1.0.
        (
            (0..block).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
            scratchy_tensors::DType::F32,
            [1, block],
        ),
        // sixtyfour: [1, BLOCK] of 64.0.
        (
            (0..block).flat_map(|_| 64.0f32.to_le_bytes()).collect(),
            scratchy_tensors::DType::F32,
            [1, block],
        ),
        // padoff: THE PAD LAW — 0.0 on the real lanes, f32::MIN (−FLT_MAX,
        // 0xFF7FFFFF) on the pads. Not an `f32_row` operand: the pad value is
        // −FLT_MAX, not a small integer. ⭐ NEVER IEEE −inf: the hardware's
        // fp32 domain does not use IEEE inf — the fp32 reduce template's own
        // "negInf" seed is 0xFF7FFFFF, a FINITE number
        // (`summeanmaxexx2_fp32.ddl:71`), and an exponent-all-ones lane reads
        // as NaN on the card, so an IEEE −inf staged verbatim poisons the max
        // into NaN. −FLT_MAX loses every max a real lane contests exactly the
        // way −inf does (both are below every finite score), with no NaN to
        // ride through. (#813's f16 pads never hit this: `ieee_to_sen`
        // saturates −inf to SEN's finite minimum on that path.)
        (
            (0..pad_v)
                .flat_map(|i| {
                    let x = if i < v { 0.0 } else { f32::MIN };
                    x.to_le_bytes()
                })
                .collect(),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // gramp: G - g, constant per group, pads 0 — the two-level ramp's
        // outer level.
        (
            f32_row(&|i| groups - i / (block * group_blocks), 0),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // ginc: the group index, pads G — the sentinel's group half.
        (
            f32_row(&|i| i / (block * group_blocks), groups),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // hc1 / hc2 / hc3: the counter hash's multipliers, i32 — the
        // staged spellings of 0x9E3779B1 / 0x85EBCA77 / 0xC2B2AE3D. `[rows, 1]`,
        // the same word at every row's stick head (the seeds' own class): uniform
        // constants broadcast by their consuming multiplies through the int-bcast
        // operand mode, never whole-row splats.
        (
            i32_col(-1640531535),
            scratchy_tensors::DType::I32,
            [rows as usize, 1],
        ),
        (
            i32_col(-2048144777),
            scratchy_tensors::DType::I32,
            [rows as usize, 1],
        ),
        (
            i32_col(-1028477379),
            scratchy_tensors::DType::I32,
            [rows as usize, 1],
        ),
        // c256: the wrap-fold's 2^8, i32 — `[rows, 1]`, one word per row.
        (
            i32_col(256),
            scratchy_tensors::DType::I32,
            [rows as usize, 1],
        ),
        // lane: the per-lane index 0..PAD_V-1 as f32 (exact below 2^24) —
        // every lane its own index, pads included (the hash runs per lane;
        // the pad lanes lose on the pad law, never on the noise).
        (
            (0..pad_v).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            scratchy_tensors::DType::F32,
            [1, pad_v],
        ),
        // fm8 / ftop / fzero / fhalf / fm22: the fold's f32 constants —
        // 2^-8, 2^23 - 1, 0, 0.5, 2^-22. `[rows, 1]`, the same word at every
        // row's stick head (the seeds' own class): uniform constants broadcast
        // by their consuming ops, never whole-row splats.
        (
            f32_col(0.00390625),
            scratchy_tensors::DType::F32,
            [rows as usize, 1],
        ),
        (
            f32_col(8388607.0),
            scratchy_tensors::DType::F32,
            [rows as usize, 1],
        ),
        (
            f32_col(0.0),
            scratchy_tensors::DType::F32,
            [rows as usize, 1],
        ),
        (
            f32_col(0.5),
            scratchy_tensors::DType::F32,
            [rows as usize, 1],
        ),
        (
            f32_col(2f32.powi(-22)),
            scratchy_tensors::DType::F32,
            [rows as usize, 1],
        ),
        // cmin: the fp16 island's clamp 2^-14 (fp16's smallest normal) —
        // fp16, the FEST log's domain guard; staged through the SEN convert
        // like every fp16 weight (a REAL conversion: SEN169 ≠ IEEE binary16).
        // `[rows, 1]` — the same word at every row's stick head, the
        // `maximum`'s broadcast operand in the seeds' own class.
        (
            (0..rows as usize)
                .flat_map(|_| half::f16::from_f32(2f32.powi(-14)).to_le_bytes())
                .collect(),
            scratchy_tensors::DType::F16,
            [rows as usize, 1],
        ),
    ]
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

    /// The KV capacity the bundle was baked at — its pools' token slots.
    #[inline]
    pub fn capacity(&self) -> Option<usize> {
        self.paged.map(PagedStep::capacity)
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
            let table = stage_kernel_table(hd as usize, |i, j| if i == j { 1.0 } else { 0.0 });
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
            let itable =
                get(itid).unwrap_or_else(|| panic!("identity class {class} (t{itid}) not bound"));
            let iwant = stage_kernel_table(hd as usize, |i, j| if i == j { 1.0 } else { 0.0 });
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
        let lay_max = class_layout(&[
            (sd::rope_p_class_tid(0), 128),
            (sd::rope_p_class_tid(1), 128),
        ]);
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

    /// A minimal `Wiring` for the cross-check tests — only the class fields are load-bearing.
    fn uniform_wiring() -> Wiring {
        Wiring {
            result: 0,
            result_dtype: scratchy_tensors::DType::F16,
            num_sources: 0,
            attn_mask: None,
            paged: None,
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
            weight_scale_folds: &[],
            load_consts: &[],
            sample_seat: None,
            phase: crate::forward_tape::ForwardPhase::Prefill,
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
