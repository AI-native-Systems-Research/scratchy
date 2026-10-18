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
    /// The vocab axis V — the lm_head's sticked `out` axis (49152 at granite).
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
    /// INPUT — the lm_head output slot, `[rows, vocab]` vocab-sticked (the staging form).
    pub logits: &'a str,
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
#[derive(Clone, Copy, Debug)]
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
    // The typed handles. `rb` mints RowBlocked handles (annotation-only — the builders read
    // the name and the kind); the eq's `m` and the sub's `V` ride the broadcast `In` doors.
    let logits_h = rb(s.logits, rows, vocab);
    let m_h = rb(s.m, rows, Fp16::ELEMS_PER_STICK);
    let mask_h = rb(s.mask, rows, vocab);
    let mask32_h = rb(s.mask32, rows, vocab);
    let ramp_h = rb(s.ramp, 1, vocab);
    let v_h = rb(s.v_const, 1, Df::Fp32.elems_per_stick());
    let j_h = rb(s.j, rows, Df::Fp32.elems_per_stick());
    let idx32_h = rb(s.idx32, rows, Fp16::ELEMS_PER_STICK);
    let idx_u32_h = rb(s.idx_u32, rows, Fp16::ELEMS_PER_STICK);
    let mut ops = Vec::with_capacity(8);

    // 1. m = max(logits) — the fp16 stick-family max over the vocab-sticked staging.
    let max16 = reduce_opspec(OpFunc::Max, rows, vocab, s.logits, s.m, false)?;
    let folds = SdscFoldSet::new(max16.iter.cores_used());
    ops.push(
        emit_sdsc_tiled("argmax_max16", &max16, &folds, sym_id_base, layout).map_err(|e| e.0)?,
    );
    // 2. mask = (logits == m) — fp16, `m` out-broadcast (the inv_rms column form).
    ops.push(assemble_pointwise_broadcast_df(
        "argmax_eq",
        "equal",
        rows,
        vocab,
        &[In::full(&logits_h).ew(), In::col(&m_h).ew()],
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
        vocab,
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
        vocab,
        &[In::full(&mask32_h).ew(), In::mb(&ramp_h).ew()],
        s.w,
        Df::Fp32,
        sym_id_base,
        layout,
    ));
    // 5. j = max(w) — the fp32 sibling's stick-family max.
    let max32 = reduce_opspec_df(OpFunc::Max, rows, vocab, s.w, s.j, false, Df::Fp32)?;
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
