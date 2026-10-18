// SPDX-License-Identifier: Apache-2.0
//! THE COMPOSITE DEVICE ARGMAX — #603 PR3a's loop-closure group (opus-ruled 2026-10-18).
//!
//! # THE CHAIN, AND WHY EVERY LINK IS WHAT IT IS
//!
//! ```text
//!   1. m      = max(logits)              summeanmaxexx2's max bind   (fp16, stick family)
//!   2. mask   = (logits == m)            broadcast_ops' equal        (fp16, out-broadcast m)
//!   3. mask32 = dl16tofp32(mask)         quantization_double_pad     (the {0,1} mask ONLY)
//!   4. w      = mask32 · ramp            broadcast_ops' mul          (fp32, ramp = V−iota)
//!   5. j      = max(w)                   summeanmaxexx2_fp32's max   (fp32, stick family)
//!   6. idx32  = V − j                    broadcast_ops' subtract     (fp32, one stick)
//!   7. idx    = fp32toint32(idx32)       unary_parallel              (SENUINT32, 4-byte)
//!   8. row    = gather(table, idx)       the identity gather-copy    (the tied lm_head bank)
//! ```
//!
//! • **First-max BY ARITHMETIC — no topk, no tie probe.** The resident fp32 ramp carries
//!   `V − i` at column `i`, so the FIRST maximal index gets the LARGEST weight; the ramp
//!   values are pairwise distinct, so op 5's maximum is UNIQUE — a tie cannot exist in this
//!   chain, which is why it needs neither `topk.ddl`'s tie convention nor its probe. This is
//!   the #631 law (host, device, metal all first-max) made structural instead of tested.
//!
//! # THE PAD-LANE FORM (opus, 2026-10-18) — sub-stick vocab SERVED, not fenced
//!
//! granite-3.1's vocab is 49155 = 768·64 + 3 — three lanes past the last fp16 stick — but
//! its logits do not live in a 49155-lane buffer: the lm_head's out axis is STICK-PADDED
//! (49155 real lanes in a 49216-lane staging), so the chain runs at the padded width over
//! ONE staged constant bias:
//!
//!   • **link 0 (new, before the max): `biased = logits + bias`** — fp16, the bias
//!     mb-broadcast exactly the way the ramp is: `[1, stick_pad(V)]` with 0 on the real
//!     lanes and −65504 (the fp16 floor — `manifest::argmax`'s own running-max init, the
//!     device's `0xFFFE`) on the pad lanes. The real lanes add 0 EXACTLY, so first-max
//!     semantics and NaN handling are unchanged (a NaN pad stays NaN through `+bias` and
//!     never wins the maxnumf); a pad lane can win the max only what an all-floor real row
//!     could.
//!   • **links 1–5 run at `stick_pad(V)` reading `biased`**, and the ramp is staged over
//!     the SAME padded width as `max(V − i, 0)`. The clamp is VACUOUS on real lanes
//!     (`V − i ≥ 1` — the landed `V − iota` verbatim) and ZERO on pad lanes, which is the
//!     form's index-boundedness law: `w ≥ 0` on every lane, so `j ≥ 0` and
//!     `idx = V − j ≤ V` — a pad lane that beat every real one AND the all-NaN row both
//!     land EXACTLY on the sentinel row V, never past it (an unclamped pad winner at
//!     `i > V` would gather past the sentinel, out of the table's view).
//!   • **the subtract's constant stays the REAL V**, and the table view states
//!     `[V + 1, H]` — the sentinel row is the first zero row of the stick-padded staging,
//!     so the extension the whole-stick form already reserves covers it: no extra bytes.
//!   • **a whole-stick vocab (granite-3.3's 49152) pads to itself** and assembles the
//!     eight landed links VERBATIM — byte-identical descriptors, the content-key law; the
//!     bias tid is NAMED by the door but never placed, declared, or staged, so those
//!     bundles' wires stay byte-identical too.
//! • **Exact end to end.** The fp16 max value and the eq are exact (both operands are input
//!   elements); every fp32 value is ≤ V < 2²⁴, so `V − j` recovers the index exactly. The
//!   LOGITS ARE NEVER CONVERTED — only the {0,1} mask widens (98→196 KB at granite), which is
//!   the whole point of comparing in fp16: the counter law (fp16 exact integers stop at 2048
//!   ≪ 49152) forbids the index arithmetic in fp16, not the comparison.
//! • **The staging form, unpadded.** The lm_head emits `[mb = m, out = vocab]` with VOCAB on
//!   the stick and `m` NOT a stick axis (`matmul/dims.rs`), so at m=1 the logits row is ONE
//!   row × `vocab/64` sticks — 98 KB at granite. The `[vocab, m_pad=64]` K-form volume is
//!   born at the Kᵀ restickify and exists only for the topk pair's token-stick law; THIS
//!   chain never enters it. The max bind is the STICK family (slice+stick order-fixed on
//!   `%reduce_dim#0`), which the vocab-sticked staging matches directly — reduce over the
//!   stick dim, no restickify anywhere.
//! • **Multi-core, the anti-topk.** `summeanmaxexx2.ddl` scopes `core_to_core_communication`
//!   to the REDUCE dims with the FMAX partial combine — the vocab scan splits across all 32
//!   cores. (`topk.ddl`'s ring rode `%k` and stranded the scan on one core; that shape is
//!   why the one-op device argmax was price-dead.)
//! • **The gather reads the tied bank IN PLACE.** granite ties (`tie_word_embeddings:
//!   true`, 2b and 8b), and the lm_head weight's device staging is out-major `[vocab,
//!   hidden-sticks]` — token v's embedding IS out-slot v, one contiguous stick range. The
//!   gather-copy's expert-bank run shape (entries = ids relative to the bank's base,
//!   `idx·skip_addr` supplying the slab stride) reads it with ZERO new memory. The
//!   [`TiedLmHeadEmbedding`] token is the build-time refusal that keeps an UNTIED model
//!   (llama) off this path until its staging fallback exists.
//! • **k=1, w=1 — the first cut.** At one row every packing seam is trivial (rank-3 flat,
//!   stick-major and rank-2 coincide byte-for-byte at rows == 1, and the index table's word
//!   0 IS lane 0). The w>1 widths are second-wave work gated on the index-table packing:
//!   the per-request run at k=1 is ONE u32 word, sub-stick, so the router's FLAT-run
//!   convert (`[W, m_pad]`, whole-stick runs) does not apply; that wave lands with PR3b's
//!   run-ahead, which needs the widths anyway. Consequence for the bake: this group adds
//!   ONE dxp compile, not one per width.
//! • **THE CONTENT-KEY LAW (opus, 2026-10-18): rung-independent by construction, asserted
//!   not conventions.** Every operand slot this chain touches (the logits slot, the ramp
//!   and V constants, the scratch, the index slot, the row destination, the table) sits at
//!   a rung-independent fixed address — the PR2 canvas slots — so no per-rung offset ever
//!   enters a descriptor and the bake's content key (`superdsc_bake`: "a hash of exactly
//!   what dxp will read") collapses the group across every cap rung. The emission-side
//!   half of the guard is `zz_the_argmax_composite_probes.rs`'s digest test; the ladder-side
//!   assert lands with the wiring.
//!
//! # THE PROBE MATRIX (cardless, `dxp_standalone` rc=0, quiet pod) — RUN GREEN 2026-10-18
//!
//! The eight descriptors staged as one probe GROUP (`bundle.mlir` + `sdsc_N.json`, the
//! dump's own dir contract — see `zz_the_argmax_composite_probes.rs`'s staging test) and
//! compiled by the real bake driver, pod nickm4, load ~4.8 (quiet): **`dxp_standalone`
//! rc=0 at the SENTIENT target (the dbo pipeline — the bake's own path) AND at the
//! senulator target (the imperative `runDxp`)**, all eight links, producing a real
//! `spyreCodeDir` (one `ComputeOnDevice` job, 790 KB image). A refusal surfaces as rc=1
//! with the failing program named — the run that preceded the env fix did exactly that —
//! so rc=0 is each link's selection AND schedule accepting, not a skip. (`ddc_standalone
//! -s` is NOT this gate: it demands the hand-written fixture form — chip descriptors the
//! real pipeline fills in code — so even the real dump files abort in it with `map::at`;
//! reproduced and pinned. The env the driver needs: `DEEPTOOLS_PATH` at the installed
//! `share/` tree, `SENARCH=rcudd1a`.)
//!
//! 1. max16 — the stick-family max on the staging form `[1, vocab]` (probe:
//!    `argmax_max16.json`).
//! 2. equal with a COMPUTED out-broadcast `[1,1]` operand — the card-proven splat is a
//!    constant; this chain's `m` is the reduce's own output (probe: `argmax_eq.json`;
//!    fallback if refused: a `[1, vocab]` splat-copy, +2 µs).
//! 3. dl16tofp32 with the stick SHARED on vocab both sides (probe: `argmax_cvt.json`) —
//!    the probed form in the ExpertMatmul mint was flat/dim-0; same-dim, different
//!    position.
//! 4. mul in fp32, the ramp mb-broadcast (probe: `argmax_mul.json`).
//! 5. max32 — the fp32 sibling's stick family (probe: `argmax_max32.json`).
//! 6. subtract at the one-fp16-stick tail width (64 elems = two fp32 sticks), the V
//!    scalar broadcast (probe: `argmax_sub.json`).
//! 7. fp32toint32 at `[1, 64]` — the same one-fp16-stick width (probe:
//!    `argmax_cvt_idx.json`).
//! 8. the gather-copy against the tied bank's pitch (probe: `argmax_gather.json`).

use crate::emit::{
    EmittedOp, In, SdscFoldSet, assemble_convert, assemble_pointwise_broadcast_df, emit_sdsc_tiled,
    rb,
};
use crate::ir::bridge::tiled_op_sdsc_op::pointwise::gather_copy_opspec;
use crate::ir::bridge::tiled_op_sdsc_op::reduce::{reduce_opspec, reduce_opspec_df};
use crate::superdsc_opspec::{
    DataFormat, Df, EntryBase, Fp16, GatherIndex, GatherRun, OpFunc, PageExtent,
};

/// The chain's geometry. `vocab` and `hidden` come from the model; `rows` is the batch
/// width, which the first cut pins to 1 (see the module doc's k=1/w=1 law).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgmaxGeom {
    /// The batch width — the decode bundle's `mb` axis. ONE row per in-flight request.
    /// The first cut emits 1 only; the w>1 packing is the second wave.
    pub rows: u32,
    /// The vocab axis V — the REAL vocab count (49155 at granite-3.1, 49152 at
    /// granite-3.3). The chain's links run at its STICK PADDING (the lm_head's out axis
    /// is stick-padded; a whole-stick vocab pads to itself), derived inside — and a
    /// sub-stick vocab selects the pad-lane form (the module doc's pad-lane section).
    pub vocab: u32,
    /// The hidden axis — one embedding row's width in elements (2048 at granite-2b). The
    /// gather's slab; must be a whole fp16-stick multiple (it is, everywhere we ship).
    pub hidden: u32,
}

impl ArgmaxGeom {
    /// The k=1/w=1 form — PR3a's landing shape, and the only one the first cut admits.
    pub fn k1(vocab: u32, hidden: u32) -> Self {
        Self {
            rows: 1,
            vocab,
            hidden,
        }
    }
}

/// The chain's operand slots. INPUT/OUTPUT mark the group's boundary; everything else is
/// scratch the caller places (at rung-independent fixed addresses — the content-key law).
#[derive(Clone, Copy, Debug)]
pub struct ArgmaxSlots<'a> {
    /// INPUT — the lm_head output slot, `[rows, stick_pad(vocab)]` vocab-sticked (the
    /// staging form; at a whole-stick vocab the padding IS the vocab).
    pub logits: &'a str,
    /// PAD-LANE FORM ONLY — link 0's output, the bias-applied logits row (fp16, the
    /// stick-padded width). The whole-stick arm never reads this slot.
    pub biased: &'a str,
    /// RESIDENT CONSTANT (pad-lane form only) — the fp16 `[1, stick_pad(vocab)]` lane
    /// bias: 0 on the real lanes, −65504 (the fp16 floor) on the pad ones. Named by the
    /// door at [`crate::reserved_tids::argmax_bias_tid`]; placed and staged only by a
    /// pad-lane bundle.
    pub bias: &'a str,
    /// scratch — the per-row max (the reduce's RedStick accum).
    pub m: &'a str,
    /// scratch — the {0,1} fp16 mask.
    pub mask: &'a str,
    /// scratch — the widened fp32 mask.
    pub mask32: &'a str,
    /// RESIDENT CONSTANT — the fp32 `[1, vocab]` descending ramp `V − iota` (bake-time,
    /// written once; the FIRST maximal index gets the LARGEST value).
    pub ramp: &'a str,
    /// RESIDENT CONSTANT — the fp32 `[1,1]` scalar V (the vocab count as f32).
    pub v_const: &'a str,
    /// scratch — the fp32 weighted mask.
    pub w: &'a str,
    /// scratch — the per-row max of `w` (the fp32 reduce's RedStick accum).
    pub j: &'a str,
    /// scratch — the fp32 index `[rows, one fp32 stick]`.
    pub idx32: &'a str,
    /// OUTPUT — the SENUINT32 index slot the next step's gather reads.
    pub idx_u32: &'a str,
    /// RESIDENT — the tied lm_head embedding staging, the gather's bank (see
    /// [`TiedLmHeadEmbedding`]).
    pub table: &'a str,
    /// OUTPUT — the next step's input-activation slot (the gathered embedding row).
    pub row: &'a str,
}

/// ⛔ THE TIED FACT, AS A TYPE — the gather reads the lm_head's own staging in place, so
/// the chain may only be assembled for a model whose `tie_word_embeddings` is true. The
/// token is constructed ONLY through [`Self::from_tied_config`], which refuses an untied
/// model with the named reason — so an untied model (llama) cannot take this path until
/// its staging fallback exists; the refusal is a build failure, never a silent gather of
/// the wrong bank. (granite-3.3 2b and 8b both tie; the PR3a gates are granite.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TiedLmHeadEmbedding {
    _private: (),
}

impl TiedLmHeadEmbedding {
    /// The only constructor. `tie_word_embeddings` is the CONFIG's own field, read at the
    /// expansion where the model is in hand — the caller cannot hand over a bare `true`.
    pub fn from_tied_config(tie_word_embeddings: bool) -> Result<Self, String> {
        if tie_word_embeddings {
            Ok(Self { _private: () })
        } else {
            Err(
                "the device argmax's embedding gather reads the lm_head's own staging in \
                 place, which is only the embedding table when tie_word_embeddings is true; \
                 this model is untied and has no staging fallback yet — refusing rather than \
                 gathering the wrong bank"
                    .to_string(),
            )
        }
    }
}

/// Assemble the composite argmax chain — one [`EmittedOp`] per link, in dependency order.
/// The `Result` form for door arms that can name their own blame; the `#[forward]` form
/// is [`assemble_argmax`].
///
/// Every link rides an existing, mostly card-proven builder (the module doc's table):
/// the reduces are the stick-family seeded reduce, the eq/mul/sub are the broadcast
/// pointwise family (the eq's `m` operand is the out-broadcast `[rows,1]` column form —
/// RmsNorm's inv_rms class, card-proven 2026-06-25), the converts are the staged dtype
/// converts, and the gather is the identity gather-copy in its expert-bank run shape.
pub fn try_assemble_argmax(
    geom: ArgmaxGeom,
    s: &ArgmaxSlots<'_>,
    _tied: TiedLmHeadEmbedding,
    sym_id_base: &mut i64,
    layout: Option<&crate::placement::BundleLayout>,
) -> Result<Vec<EmittedOp>, String> {
    if geom.rows != 1 {
        return Err(format!(
            "try_assemble_argmax: rows = {} — the first cut emits the k=1/w=1 form only; \
             the w>1 index-table packing (the router's FLAT-run convert at whole-stick \
             runs) is second-wave work that lands with PR3b's run-ahead",
            geom.rows
        ));
    }
    // The gather's slab must be whole sticks of the value format (the one-stick sub-row
    // law); `hidden` is a stick multiple everywhere we ship, so this is a typed restatement,
    // not a new constraint.
    if !geom.hidden.is_multiple_of(Fp16::ELEMS_PER_STICK) {
        return Err(format!(
            "try_assemble_argmax: hidden = {} is not a whole {}-element stick multiple — \
             the embedding row cannot be gathered as one-stick sub-rows",
            geom.hidden,
            Fp16::ELEMS_PER_STICK
        ));
    }
    let rows = geom.rows;
    let vocab = geom.vocab;
    let hidden = geom.hidden;
    // ⭐ THE STAGING WIDTH and the form flag — the lm_head's out axis is stick-padded, so
    //    the chain runs at `stick_pad(V)`; a whole-stick vocab pads to itself and the
    //    eight landed links below are assembled VERBATIM (byte-identical descriptors —
    //    the content-key law). A sub-stick vocab (granite-3.1's 49155 → 49216) takes the
    //    NINE-link pad-lane form: link 0 adds the staged bias BEFORE the max, and links
    //    1–5 read the biased row at the padded width (the module doc's pad-lane section).
    let pad_lanes = !vocab.is_multiple_of(Fp16::ELEMS_PER_STICK);
    let v_pad = if pad_lanes {
        vocab.div_ceil(Fp16::ELEMS_PER_STICK) * Fp16::ELEMS_PER_STICK
    } else {
        vocab
    };
    // The typed handles. `rb` mints RowBlocked handles (annotation-only — the builders read
    // the name and the kind); the eq's `m` and the sub's `V` ride the broadcast `In` doors.
    let logits_h = rb(s.logits, rows, v_pad);
    let biased_h = rb(s.biased, rows, v_pad);
    let bias_h = rb(s.bias, 1, v_pad);
    let m_h = rb(s.m, rows, Fp16::ELEMS_PER_STICK);
    let mask_h = rb(s.mask, rows, v_pad);
    let mask32_h = rb(s.mask32, rows, v_pad);
    let ramp_h = rb(s.ramp, 1, v_pad);
    let v_h = rb(s.v_const, 1, Df::Fp32.elems_per_stick());
    let j_h = rb(s.j, rows, Df::Fp32.elems_per_stick());
    let idx32_h = rb(s.idx32, rows, Fp16::ELEMS_PER_STICK);
    let idx_u32_h = rb(s.idx_u32, rows, Fp16::ELEMS_PER_STICK);
    let mut ops = Vec::with_capacity(9);
    // Links 1–2 read the BIASED row in the pad-lane form, the logits themselves in the
    // whole-stick one — one operand name, chosen once here.
    let (read, read_h) = if pad_lanes {
        (s.biased, &biased_h)
    } else {
        (s.logits, &logits_h)
    };

    // 0 (pad-lane form only). biased = logits + bias — fp16, the resident bias
    //    mb-broadcast (the ramp's own form). The real lanes add 0 EXACTLY; the pad lanes
    //    add the fp16 floor `manifest::argmax` itself initializes its running max at, so
    //    a pad lane wins only what an all-floor real row could.
    if pad_lanes {
        ops.push(assemble_pointwise_broadcast_df(
            "argmax_bias",
            "add",
            rows,
            v_pad,
            &[In::full(&logits_h).ew(), In::mb(&bias_h).ew()],
            s.biased,
            Df::Fp16,
            sym_id_base,
            layout,
        ));
    }
    // 1. m = max(biased) — the fp16 stick-family max over the vocab-sticked staging.
    let max16 = reduce_opspec(OpFunc::Max, rows, v_pad, read, s.m, false)?;
    let folds = SdscFoldSet::new(max16.iter.cores_used());
    ops.push(
        emit_sdsc_tiled("argmax_max16", &max16, &folds, sym_id_base, layout).map_err(|e| e.0)?,
    );
    // 2. mask = (biased == m) — fp16, `m` out-broadcast (the inv_rms column form).
    ops.push(assemble_pointwise_broadcast_df(
        "argmax_eq",
        "equal",
        rows,
        v_pad,
        &[In::full(read_h).ew(), In::col(&m_h).ew()],
        s.mask,
        Df::Fp16,
        sym_id_base,
        layout,
    ));
    // 3. mask32 = dl16tofp32(mask) — the staged convert, the stick SHARED on vocab both
    //    sides (quantization_double_pad's own law).
    ops.push(assemble_convert(
        "argmax_cvt",
        "dl16tofp32",
        rows,
        v_pad,
        &mask_h,
        &mask32_h,
        sym_id_base,
        layout,
    ));
    // 4. w = mask32 · ramp — fp32, the resident ramp mb-broadcast (the gamma form).
    ops.push(assemble_pointwise_broadcast_df(
        "argmax_mul",
        "multiply",
        rows,
        v_pad,
        &[In::full(&mask32_h).ew(), In::mb(&ramp_h).ew()],
        s.w,
        Df::Fp32,
        sym_id_base,
        layout,
    ));
    // 5. j = max(w) — the fp32 sibling's stick-family max.
    let max32 = reduce_opspec_df(OpFunc::Max, rows, v_pad, s.w, s.j, false, Df::Fp32)?;
    let folds = SdscFoldSet::new(max32.iter.cores_used());
    ops.push(
        emit_sdsc_tiled("argmax_max32", &max32, &folds, sym_id_base, layout).map_err(|e| e.0)?,
    );
    // 6. idx32 = V − j — the tail runs ONE FP16-STICK wide (64 elems = the shared width the
    //    mixed-format legs can all tile: two fp32 sticks, two u32 sticks, one fp16 stick).
    //    `V` is the scalar broadcast and `j` rides the SAME lane-0 column broadcast the eq's
    //    `m` rode (the inv_rms form), so every lane holds the true index; the gather reads
    //    word 0.
    ops.push(assemble_pointwise_broadcast_df(
        "argmax_sub",
        "subtract",
        rows,
        Fp16::ELEMS_PER_STICK,
        &[In::scalar(&v_h).ew(), In::col(&j_h).ew()],
        s.idx32,
        Df::Fp32,
        sym_id_base,
        layout,
    ));
    // 7. idx = fp32toint32(idx32) — the SENUINT32 index slot, both operands 4-byte (the
    //    DDL's slice equality), at the tail's one-fp16-stick width.
    ops.push(assemble_convert(
        "argmax_cvt_idx",
        "fp32toint32",
        rows,
        Fp16::ELEMS_PER_STICK,
        &idx32_h,
        &idx_u32_h,
        sym_id_base,
        layout,
    ));
    // 8. row = gather(table, idx) — the identity gather-copy in the expert-bank run shape:
    //    entries are token ids relative to the bank's base, one entry covers the whole row
    //    (page = hidden/64 one-stick sub-rows), and `skip_addr = page · 64 = hidden` is the
    //    bank's own row pitch. The index is the chain's own u32 output (word 0 of its
    //    stick — the only entry at rows == 1).
    let sub_rows = hidden / Fp16::ELEMS_PER_STICK;
    let gather = gather_copy_opspec(
        s.table,
        s.row,
        sub_rows,
        Fp16::ELEMS_PER_STICK,
        Df::Fp16,
        GatherIndex::of_scratch_rows(
            s.idx_u32.to_string(),
            PageExtent::of_positions(sub_rows),
            EntryBase::ZERO,
        ),
        GatherRun::CONTIGUOUS,
    )?;
    let folds = SdscFoldSet::new(gather.iter.cores_used());
    ops.push(
        emit_sdsc_tiled("argmax_gather", &gather, &folds, sym_id_base, layout).map_err(|e| e.0)?,
    );
    Ok(ops)
}

/// [`try_assemble_argmax`] — the `#[forward]` form, for callers inside an expansion where
/// a refusal IS the build error.
pub fn assemble_argmax(
    geom: ArgmaxGeom,
    s: &ArgmaxSlots<'_>,
    tied: TiedLmHeadEmbedding,
    sym_id_base: &mut i64,
    layout: Option<&crate::placement::BundleLayout>,
) -> Vec<EmittedOp> {
    try_assemble_argmax(geom, s, tied, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_argmax: {e}"))
}

// ════════════════════════════════════════════════════════════════════════════════════════════════
//  THE WALK MINT — the same computation as a KTIR program
// ════════════════════════════════════════════════════════════════════════════════════════════════

/// THE BINDING CONVENTION, stated once for both expressions of the computation.
///
/// The composite is ONE computation with TWO bodies: this KTIR program (which the EMULATOR
/// executes — the interpreter runs the ops directly, so the parity gate can check the chain
/// against the host argmax on the same logits) and the descriptor group
/// [`try_assemble_argmax`] mints (which the CARD runs). The two agree by construction, not
/// by convention-matching: they read the same three boundary buffers, in the same order —
///
/// * **parameter 0 — `logits`**: the lm_head's output, the graph result, viewed `[1,
///   stick_pad(V)]` — the STAGING width (V itself when the vocab is stick-aligned). The
///   program loads it WHOLE (one access tile at the origin) and reduces over both axes;
///   in the pad-lane form the reduce reads the bias-applied row, link 0 below.
/// * **parameter 1 — `table`**: the tied lm_head weight, the gather's bank, viewed
///   `[V + 1, H]` row-major — the `+1` is the SENTINEL row, the zero row the `idx = V`
///   gather reads when nothing real won (the all-NaN row, or a pad lane that beat every
///   real one; the staging's zero-filled shadow IS that row, so no extra bytes). The
///   program's only access tile over it carries the RUNTIME index (`%idx`, the chain's
///   own output) as its row corner, so `regions()` falls back to the whole-view window
///   for it — exactly the geometry the descriptors state.
/// * **parameter 2 — `row`**: the next step's input-activation slot, viewed `[1, H]`. The
///   program's ONE store writes through a `[1, H]` tile at the origin, which is the
///   `written_args` chain the emulator's output readback and the door's `is_out` both key
///   on — the row IS the program's output.
///
/// The scratch and the three constants never cross this boundary: the descriptors place them
/// as synthetics keyed off the row tid and `argmax_const` reserved tids (the door's own
/// mint), and the program needs neither — the ramp and the pad-lane bias are COMPUTED
/// in-func (`linalg.index` arithmetic), because an emulator source binds fp16 and the fp32
/// `[1, V]` ramp is past fp16's exact-integer range at every real vocab.
///
/// ⭐ THE NAME KEYS ON THE ROW, not a node index: every other program is `<kind>_s<node
/// index>`, but the composite is minted by the WALK over the graph's result — there is no
/// node. `argmax_o{row_tid}` follows the door's own op-name law (`<kind>_o{output
/// tensor}`), which also makes it rung-stable: the ladder's cap rungs share the row tid, so
/// the program interns once and every rung's launch binds the same one.
pub fn mint_argmax_node(
    vocab: u32,
    hidden: u32,
    logits_tid: u32,
    table_tid: u32,
    row_tid: u32,
    tied: TiedLmHeadEmbedding,
) -> crate::ktir_node::KtirNode {
    use ktir_core::arena::Arena;
    use ktir_core::attrkey::AttrKey;
    use ktir_core::dtypes::DType;
    use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
    use ktir_core::irtype::IrType;
    use ktir_core::opkind::OpKind;

    let a = Arena::global();
    // SSA numbering: the three pointer parameters first (the `arguments` list references
    // them), then one fresh id per program value, in mint order.
    let (lg_ptr, tb_ptr, row_ptr) = (Ssa(0), Ssa(1), Ssa(2));
    let mut next = 3u32;
    let mut fresh = || {
        let s = Ssa(next);
        next += 1;
        s
    };
    let il = |v: Vec<i64>| Attr::IntList(a.ints(v));

    // ⭐ THE FORM — a sub-stick vocab (granite-3.1's 49155) takes the pad-lane form: the
    //    views state the STICK-PADDED staging width, link 0 applies an in-func bias
    //    before the max, and the ramp clamps at zero (the module doc's pad-lane section).
    //    A whole-stick vocab pads to itself and the program is the landed eight-link one.
    let pad_lanes = !vocab.is_multiple_of(Fp16::ELEMS_PER_STICK);
    let v_pad = if pad_lanes {
        vocab.div_ceil(Fp16::ELEMS_PER_STICK) * Fp16::ELEMS_PER_STICK
    } else {
        vocab
    };

    // ── the three boundary views, one per parameter (the every-parameter-states-its-width
    //    contract the door's `regions()` enforces) ──
    let view = |ptr: Ssa, rows: i64, cols: i64, res: Ssa| -> Operation<'static> {
        Operation::new(a, Some(res), OpKind::KtdpConstructMemoryView, &[ptr])
            .with_attr(a, AttrKey::Shape, il(vec![rows, cols]))
            .with_attr(a, AttrKey::Strides, il(vec![cols, 1]))
            .with_attr(a, AttrKey::MemorySpace, Attr::Str("HBM"))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F16))
    };
    let lg_v = fresh();
    let tb_v = fresh();
    let row_v = fresh();

    // ── the corner zero, shared by every const-cornered tile ──
    let c0 = fresh();
    let c0_op = Operation::new(a, Some(c0), OpKind::ArithConstant, &[]).with_attr(
        a,
        AttrKey::Value,
        Attr::Int(0),
    );

    // ── the logits, loaded whole at the origin, at the STICK-PADDED staging width ──
    let lg_t = fresh();
    let lg_t_op = Operation::new(
        a,
        Some(lg_t),
        OpKind::KtdpConstructAccessTile,
        &[lg_v, c0, c0],
    )
    .with_attr(a, AttrKey::Shape, il(vec![1, i64::from(v_pad)]));
    let lg = fresh();
    let lg_load = Operation::new(a, Some(lg), OpKind::KtdpLoad, &[lg_t]);

    // The scalar V — constructed here (before the bias link, whose compare reads it) and
    // listed with the ramp seed below; an SSA id is a name, not a position.
    let vc = fresh();
    let vc_op = Operation::new(a, Some(vc), OpKind::ArithConstant, &[]).with_attr(
        a,
        AttrKey::Value,
        Attr::Float(f64::from(vocab)),
    );

    // 0 (pad-lane form only). the bias, COMPUTED in-func — the ramp's own law (the
    //    program is self-contained; the descriptors' copy is the staged fp16 const):
    //    `(i ≥ V) · (−65504)` over `[1, v_pad]` fp32, the Bool widened through the f32
    //    multiply — 0 on the real lanes, the fp16 floor (manifest::argmax's own
    //    running-max init, the device's 0xFFFE) on the pad ones.
    let bias_link =
        if pad_lanes {
            let seed = fresh();
            let seed_op = Operation::new(a, Some(seed), OpKind::ArithConstant, &[])
                .with_attr(a, AttrKey::IsTensor, Attr::Bool(true))
                .with_attr(a, AttrKey::Shape, il(vec![1, i64::from(v_pad)]))
                .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F32))
                .with_attr(a, AttrKey::Value, Attr::Float(0.0));
            let acc = fresh();
            let bias = fresh();
            let bias_gen =
                {
                    let i = fresh();
                    let if_ = fresh();
                    let ge = fresh();
                    let neg = fresh();
                    let b = fresh();
                    let index_op = Operation::new(a, Some(i), OpKind::LinalgIndex, &[]).with_attr(
                        a,
                        AttrKey::Dim,
                        Attr::Int(1),
                    );
                    let sitofp_op = Operation::new(a, Some(if_), OpKind::ArithSitofp, &[i]);
                    // `i ≥ V` — true exactly on the pad lanes (the column index runs 0..v_pad).
                    let cmp_op = Operation::new(a, Some(ge), OpKind::ArithCmpf, &[if_, vc])
                        .with_attr(a, AttrKey::Predicate, Attr::Str("oge"));
                    // The −65504 is a TENSOR constant (the seeds' own form), not a scalar:
                    // a Bool tile × f32 SCALAR lands on the Bool grid (binary_float's mixed
                    // arm keeps the tile's dtype — 0/1, the magnitude destroyed), while
                    // tile × tile lands `result_float_dtype(Bool, F32)` = F32 — the Bool
                    // widens through the f32 multiply only when the other operand is a
                    // tile. This is the same law the landed `w = mask · ramp` multiply
                    // already rides (Bool tile × F32 tile → F32).
                    let neg_op = Operation::new(a, Some(neg), OpKind::ArithConstant, &[])
                        .with_attr(a, AttrKey::IsTensor, Attr::Bool(true))
                        .with_attr(a, AttrKey::Shape, il(vec![1, i64::from(v_pad)]))
                        .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F32))
                        .with_attr(a, AttrKey::Value, Attr::Float(-65504.0));
                    let mul_op = Operation::new(a, Some(b), OpKind::ArithMulf, &[ge, neg]);
                    let yield_op = Operation::new(a, None, OpKind::LinalgYield, &[b]);
                    let body = a.ops(vec![index_op, sitofp_op, cmp_op, neg_op, mul_op, yield_op]);
                    Operation {
                        regions: a.regions(vec![body]),
                        ..Operation::new(a, Some(bias), OpKind::LinalgGeneric, &[seed])
                            .with_attr(a, AttrKey::NIns, Attr::Int(0))
                            .with_attr(a, AttrKey::Bb0Names, Attr::Ssas(a.ssa(vec![acc])))
                    }
                };
            // biased = logits + bias — fp16 (one fp16 operand: `result_float_dtype`), the
            // SAME IEEE fp16 sum the descriptor chain's pointwise link computes, so the two
            // bodies of one computation round identically.
            let biased = fresh();
            let add_op = Operation::new(a, Some(biased), OpKind::ArithAddf, &[lg, bias]);
            Some((seed_op, bias_gen, add_op, biased))
        } else {
            None
        };
    // Links 1–2 read the BIASED row in the pad-lane form, the logits themselves in the
    // whole-stick one.
    let read = bias_link.as_ref().map_or(lg, |l| l.3);

    // 1. m = max(read) — both axes reduced, so the result is a SCALAR.
    let m = fresh();
    let m_red = Operation::new(a, Some(m), OpKind::LinalgReduce, &[read])
        .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithMaxnumf))
        .with_attr(a, AttrKey::Dimensions, il(vec![0, 1]));
    // 2. mask = (read == m) — `oeq`, the #631 law's compare; the scalar broadcasts.
    let eq = fresh();
    let eq_op = Operation::new(a, Some(eq), OpKind::ArithCmpf, &[read, m]).with_attr(
        a,
        AttrKey::Predicate,
        Attr::Str("oeq"),
    );
    // The ramp's zero seed, at the same stick-padded width.
    let seed = fresh();
    let seed_op = Operation::new(a, Some(seed), OpKind::ArithConstant, &[])
        .with_attr(a, AttrKey::IsTensor, Attr::Bool(true))
        .with_attr(a, AttrKey::Shape, il(vec![1, i64::from(v_pad)]))
        .with_attr(a, AttrKey::Dtype, Attr::Dtype(DType::F32))
        .with_attr(a, AttrKey::Value, Attr::Float(0.0));
    // 3–4. the ramp, COMPUTED: `max(V − iota, 0)` over `[1, v_pad]` fp32. On the real
    //    lanes the clamp is VACUOUS (`V − i ≥ 1` — the landed `V − iota` verbatim); on
    //    the pad lanes it is ZERO — the pad-lane form's index-boundedness law: `w ≥ 0` on
    //    every lane, so `j ≥ 0` and `idx = V − j ≤ V`, and nothing-real-won lands EXACTLY
    //    on the sentinel row V, never past it. The generic runs once, vectorized over the
    //    seed's shape; `linalg.index {dim: 1}` is the column index, and the bb0 name list
    //    names the (unused) outs accumulator so the region's argument names are stated —
    //    the first body op (`linalg.index`) has no operands to infer them from.
    let acc = fresh();
    let ramp = fresh();
    let ramp_gen = {
        let i = fresh();
        let if_ = fresh();
        let d = fresh();
        let index_op = Operation::new(a, Some(i), OpKind::LinalgIndex, &[]).with_attr(
            a,
            AttrKey::Dim,
            Attr::Int(1),
        );
        let sitofp_op = Operation::new(a, Some(if_), OpKind::ArithSitofp, &[i]);
        let subf_op = Operation::new(a, Some(d), OpKind::ArithSubf, &[vc, if_]);
        let (r, clamp) = if pad_lanes {
            let z = fresh();
            let r = fresh();
            let zero_op = Operation::new(a, Some(z), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(0.0),
            );
            let maxf_op = Operation::new(a, Some(r), OpKind::ArithMaxf, &[d, z]);
            (r, vec![zero_op, maxf_op])
        } else {
            (d, Vec::new())
        };
        let yield_op = Operation::new(a, None, OpKind::LinalgYield, &[r]);
        let mut body = vec![index_op, sitofp_op, subf_op];
        body.extend(clamp);
        body.push(yield_op);
        let body = a.ops(body);
        Operation {
            regions: a.regions(vec![body]),
            ..Operation::new(a, Some(ramp), OpKind::LinalgGeneric, &[seed])
                .with_attr(a, AttrKey::NIns, Attr::Int(0))
                .with_attr(a, AttrKey::Bb0Names, Attr::Ssas(a.ssa(vec![acc])))
        }
    };
    // 4. w = mask · ramp — the Bool tile widens through the f32 multiply
    //    (`result_float_dtype`: neither operand is fp16, so the landing is fp32).
    let w = fresh();
    let w_op = Operation::new(a, Some(w), OpKind::ArithMulf, &[eq, ramp]);
    // 5. j = max(w) — the first maximum's weight, UNIQUE by the ramp's construction.
    let j = fresh();
    let j_red = Operation::new(a, Some(j), OpKind::LinalgReduce, &[w])
        .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithMaxnumf))
        .with_attr(a, AttrKey::Dimensions, il(vec![0, 1]));
    // 6. idx32 = V − j — the FIRST maximal index, exactly (every value ≤ V < 2²⁴).
    let idx32 = fresh();
    let idx32_op = Operation::new(a, Some(idx32), OpKind::ArithSubf, &[vc, j]);
    // 7. idx = fp32toint32(idx32) — the scalar truncation, the access tile's row index.
    let idx = fresh();
    let idx_op = Operation::new(a, Some(idx), OpKind::ArithFptosi, &[idx32]);
    // 8. row = gather(table, idx) — the tied bank read AT THE RUNTIME ROW: one access tile
    //    over the table view whose row corner is `%idx` (bounds-checked against `[V+1, H]`
    //    by the interpreter — the sentinel row is IN the view, so the nothing-real-won
    //    gather is in-bounds by construction), loaded and stored through the row's own
    //    origin tile.
    let src_t = fresh();
    let src_t_op = Operation::new(
        a,
        Some(src_t),
        OpKind::KtdpConstructAccessTile,
        &[tb_v, idx, c0],
    )
    .with_attr(a, AttrKey::Shape, il(vec![1, i64::from(hidden)]));
    let row_val = fresh();
    let row_load = Operation::new(a, Some(row_val), OpKind::KtdpLoad, &[src_t]);
    let row_t = fresh();
    let row_t_op = Operation::new(
        a,
        Some(row_t),
        OpKind::KtdpConstructAccessTile,
        &[row_v, c0, c0],
    )
    .with_attr(a, AttrKey::Shape, il(vec![1, i64::from(hidden)]));
    let store = Operation::new(a, None, OpKind::KtdpStore, &[row_val, row_t]);

    let mut ops = vec![
        c0_op,
        view(lg_ptr, 1, i64::from(v_pad), lg_v),
        view(tb_ptr, i64::from(vocab) + 1, i64::from(hidden), tb_v),
        view(row_ptr, 1, i64::from(hidden), row_v),
        lg_t_op,
        lg_load,
        vc_op,
    ];
    if let Some((seed_op, bias_gen, add_op, _)) = bias_link {
        ops.extend([seed_op, bias_gen, add_op]);
    }
    ops.extend([
        m_red, eq_op, seed_op, ramp_gen, w_op, j_red, idx32_op, idx_op, src_t_op, row_load,
        row_t_op, store,
    ]);
    let operations = a.ops(ops);
    let func = IRFunction {
        name: a.str(format!("argmax_o{row_tid}")),
        arguments: a.args(vec![
            (lg_ptr, IrType::Index),
            (tb_ptr, IrType::Index),
            (row_ptr, IrType::Index),
        ]),
        operations,
        grid: (1, 1, 1),
        return_type: None,
    };
    crate::ktir_node::KtirNode {
        func,
        program: crate::ktir_node::Program::Argmax(tied),
        bindings: vec![
            crate::ktir_node::BufferId::new(logits_tid),
            crate::ktir_node::BufferId::new(table_tid),
            crate::ktir_node::BufferId::new(row_tid),
        ],
        mask: None,
        node_out_tid: None,
    }
}
