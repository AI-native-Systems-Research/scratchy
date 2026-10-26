// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE ATTENTION FAMILY'S HOST-STAGED STEP INPUTS — ONE BUILDER, TWO CALLERS.
//!
//! The contract `attn_step_slots` spells says the runtime writes four step-input
//! slots (the block-entry id tables, the repetition table, the validity mask)
//! before any spliced attention program runs. TWO runtimes must write them:
//! the worker's card session (`fill_step_input`, before every launch) and the
//! bake-time NUMERIC GATE (the emulator session, before its one judging run).
//! Before this module existed only the worker spelled the construction, so the
//! gate staged NOTHING — an unstaged id table reads as block 0 (a REAL address)
//! and an unstaged mask reads as zero (NO masking), both fluently wrong, and the
//! gate judged the branch's paged attention against garbage of its own making
//! (measured: every AttnDecode output diverging at local ≈ 1.03, e =
//! c·V_new[kv-head 0] broadcast to all heads, c ≈ GQA/(cap−1) — the signature
//! of a sweep that never left block 0's windows).
//!
//! The fix is this module: the PURE table construction — the all-or-nothing slot
//! checks, the region/batch derivation from the slot's own placement extent, the
//! id tables and the pre-replicated mask minted by the shared builders in the
//! KERNELS' read orders — lives HERE, in the crate the gate lives in, and BOTH
//! callers call it. The worker queries its session for the extents and the
//! ceiling and fills the slots; the gate queries its emulator session and stages
//! the same bytes through the raw channel. No second derivation, no gate-only
//! table builder: the staging is byte-identical because it is one function.
//!
//! ⭐ THE GEOMETRY IS THE WITNESS, NOT THREE NUMBERS. The builder is
//! const-generic over `<NQH, NKVH, HD>` and takes the
//! [`sd::AttnGeometry`] witness, so the GQA group is the const the witness's own
//! evaluation proved — there is no runtime divisibility check here to get wrong,
//! and no `.max(1)` to paper a non-dividing pair over. Both callers cross
//! `with_config_attn_geometry` (the model-geometry door) to reach the const arm;
//! a geometry no config declares is the door's own refusal, never this crate's.
//!
//! ⛔ NOTHING HERE TOUCHES A SESSION. The builder takes the four slot extents,
//! the gather ceiling, and the rows fact; it RETURNS the bytes. Which session
//! wrote them (card fill door or emulator raw staging) is the caller's half of
//! the contract — the two session types are different crates' types and neither
//! is a dependency of this law.

use ktir_superdsc::sdsc_abstract as sd;
use ktir_superdsc::sdsc_abstract::attn_step_slots;

/// ⭐⭐⭐ WHOSE ROWS THE LAUNCH RUNS — the one fact that picks the staged FORM of the attention
/// family's step inputs, because the form is the KERNEL's, and `rows_are_requests` (the routing
/// bit that chose the kernel at the splice) is exactly this. A partial overlap is not
/// expressible: a chunk's rows are token rows and a batch's are request rows, and staging one
/// kernel's layout for the other reads real block numbers from the wrong form — silently.
pub enum AttnStepRows<'a> {
    /// Consecutive positions of ONE sequence: a prompt chunk (the one-shot kernel, or the
    /// chunked ladder's fold), or the solo decode's single row (the fold at `mq == 1`). Both
    /// bodies read the SAME region form — the padded pitch convergence law — so one arm stages
    /// both. `start`/`real` are the chunk's own facts (`real == 1` is the decode row), and `map`
    /// is the sequence's logical→physical page map, the very map the launch installed. `real`
    /// is [`std::num::NonZeroU32`] because the fold's seed-safety law — window (0, 0) always
    /// holds a valid lane for every query row, so the carry's −65504 seed never meets an
    /// all-masked first window — is a THEOREM of `real ≥ 1` and `start ≥ 0`, proven by the
    /// type, never re-checked at runtime.
    Chunk {
        rows: sd::ChunkRows,
        mq: sd::RungWidth,
        start: usize,
        real: std::num::NonZeroU32,
        map: Vec<i64>,
    },
    /// Separate requests: a decode batch (`batched_decode_hm`), whose rows are the rung's launch
    /// slots — padding rows replicate live 0, maps and histories included. `maps` is the slot map's
    /// own per-row map list; `write` is the batch's ONE shared write slot, where every row's new
    /// key lands this step.
    Batch {
        rows: sd::LaunchRows<'a>,
        maps: &'a [Vec<i64>],
        write: sd::BatchSlot,
    },
}

/// The four slot extents a session's own discovery table states — the artifact's answers, not a
/// derivation. `None` = the bundle's placement declared no such slot.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnStepExtents {
    /// The validity mask's placement extent, in bytes.
    pub mask: Option<u64>,
    /// The Kᵀ-plane block-entry table's, in bytes.
    pub kt: Option<u64>,
    /// The V-plane block-entry table's, in bytes.
    pub v: Option<u64>,
    /// The repetition table's, in bytes (`None` in the three-slot chunk form).
    pub rep: Option<u64>,
}

/// The staged bytes themselves, one field per slot, in the wire form the card's fill channel
/// states: id tables and the repetition table as little-endian i32 per entry (the RawU32
/// channel), the mask as its own f16 `wire_bytes` (the IeeeF16 channel).
pub struct AttnStepPayload {
    /// The Kᵀ-plane block-entry table.
    pub kt: Vec<u8>,
    /// The V-plane block-entry table.
    pub v: Vec<u8>,
    /// The repetition table — `Some` only in the four-slot batch form.
    pub rep: Option<Vec<u8>>,
    /// The validity mask (chunk form) or pre-replicated mask (batch form).
    pub mask: Vec<u8>,
}

/// ⭐ THE CHUNK FAMILY'S REGION, FROM THE MASK SLOT'S OWN PLACEMENT EXTENT — the ladder's
/// declared region, not a host guess. Pub because the numeric gate mints its identity page map
/// over this region's pages, through this same door, before it can name the map.
pub fn chunk_region(
    what: &str,
    mask_extent_bytes: u64,
    pool: sd::PagedKvPool,
    gqa: std::num::NonZeroU32,
    mq: sd::RungWidth,
) -> Result<sd::SplicedAttnRegion, String> {
    sd::SplicedAttnRegion::of_mask_extent(mask_extent_bytes, pool, gqa, mq).ok_or_else(|| {
        format!(
            "{what}: the mask slot's placement extent {mask_extent_bytes} B is not a whole number \
             of page blocks at this geometry (mq {}, pool nkvh {} hd {}) — the \
             bake and the host disagree about the region",
            mq.count(),
            pool.nkvh(),
            pool.hd()
        )
    })
}

/// ⭐ THE CHUNK FORM'S ALL-OR-NOTHING SLOT PATTERN — `Ok(())` = the three-slot form is declared;
/// every other combination is a refusal. A PARTIAL declaration is a bake/host disagreement, and
/// so is an EMPTY one: on this branch every attention program is one of the three spliced
/// kernels and all of them declare the step inputs, so a bundle that runs attention while
/// declaring none is a bake/host desync, never a form to fall back from. Pub so the numeric gate
/// can check the pattern BEFORE it derives the region its identity page map needs — the check and
/// the builder's own arm are this one function, never two spellings.
pub fn chunk_slot_pattern(what: &str, extents: &AttnStepExtents) -> Result<(), String> {
    match (extents.mask, extents.kt, extents.v, extents.rep) {
        (Some(_), Some(_), Some(_), None) => Ok(()),
        (None, None, None, None) => Err(format!(
            "{what}: this bundle runs attention but declares NO attention step inputs — \
             every spliced attention kernel states all of them, so the bake and the host \
             disagree about what program this is (an unstaged id table reads as block 0, a \
             REAL address; an unstaged mask reads as zero, NO masking)"
        )),
        (_, _, _, Some(_)) => Err(format!(
            "{what}: this bundle declares a REPETITION table (slot {}) but the \
             launch's rows are consecutive positions of ONE sequence — the bake and \
             the host disagree about which body runs",
            attn_step_slots::REP_ROW_IDS
        )),
        _ => Err(format!(
            "{what}: this bundle declares a PARTIAL set of attention step inputs — \
             the three-slot form is all-or-nothing, and an unstaged one reads as \
             block 0 (ids) or no mask (mask), both fluently wrong"
        )),
    }
}

/// The batch form's all-or-nothing slot pattern — same law, four slots.
pub fn batch_slot_pattern(what: &str, extents: &AttnStepExtents) -> Result<(), String> {
    if extents.mask.is_none()
        || extents.kt.is_none()
        || extents.v.is_none()
        || extents.rep.is_none()
    {
        return Err(format!(
            "{what}: this bundle declares {} attention step input(s) of the four the batch \
             form states — all-or-nothing: an empty declaration is a bake/host desync (every \
             spliced attention kernel states all of them), a partial one reads as block 0 \
             (ids), a gathered Q row 0 (repetition), or no mask (mask)",
            [extents.mask, extents.kt, extents.v, extents.rep]
                .iter()
                .filter(|s| s.is_some())
                .count()
        ));
    }
    Ok(())
}

/// ⭐ THE GATHER'S ENTRY CEILING FROM A SESSION'S OWN PAGE STRIDE — the same two numbers the
/// card session's method hands [`sd::gather_entry_ceiling`] (`scratchy_spyre_bundle`'s
/// `MAX_SEGMENT_BYTES` over the stride), spelled ONCE for both callers: the card's
/// `SuperDscSession::gather_entry_ceiling` delegates here, and the numeric gate passes the
/// rolled layout's `kv_page_stride` straight through. A page stride of 0 is an unpaged bundle;
/// the 0 page ceiling hands it to the door, which refuses it.
pub fn gather_entry_ceiling_for(
    page_stride_bytes: u64,
    pool: sd::PagedKvPool,
) -> Option<sd::GatherEntryCeiling> {
    // `checked_div`'s None IS the unpaged case's 0 page ceiling (stride 0), which the
    // door below refuses — one spelling, no manual zero branch for clippy to name.
    let page_ceiling = scratchy_spyre_bundle::MAX_SEGMENT_BYTES
        .checked_div(page_stride_bytes)
        .unwrap_or(0);
    sd::gather_entry_ceiling(page_ceiling, page_stride_bytes, pool)
}

/// One id table's bytes, LE u32 per entry — the RawU32 channel's own wire form. The PLANE is
/// checked against every entry's own: [`sd::GatherEntry`] carries the plane it was minted for,
/// and a table staged into one plane's slot while naming the other plane's blocks is a
/// bake/host disagreement about which buffer the kernel reads — refused by name, never staged.
fn encode(what: &str, plane: sd::KvPlane, t: &[sd::GatherEntry]) -> Result<Vec<u8>, String> {
    t.iter()
        .try_fold(Vec::with_capacity(t.len() * 4), |mut out, e| {
            if e.plane() != plane {
                return Err(format!(
                    "{what}: an id table staged for the {plane:?} plane names a block of the \
                     {} plane — the two gathered planes number a page's slab blocks differently, \
                     so this is a table built for the other slot, never a rounding",
                    e.plane()
                ));
            }
            out.extend_from_slice(&e.as_i32().to_le_bytes());
            Ok(out)
        })
}

/// ⭐⭐⭐⭐⭐ BUILD THE ATTENTION FAMILY'S STEP INPUTS — the pure construction both runtimes
/// stage, in the KERNELS' read orders, never re-laid at a site. `Err` for every bake/host
/// disagreement — the refusal texts are the worker's own, unchanged — and an EMPTY slot
/// declaration is one of them: on this branch every attention program declares the step
/// inputs, so a bundle that runs attention while declaring none is a desync, never a form to
/// fall back from (the class the user condemned as a silent second producer).
///
/// The geometry crosses as the [`sd::AttnGeometry`] witness: the pool and the GQA group are
/// the witness's own (`of_geometry` / `gqa_nz`), so nothing here divides, defaults, or
/// re-checks a count the type already proved. The caller supplies the four slot extents (its
/// session's discovery table), the gather ceiling (its session's page stride through
/// [`gather_entry_ceiling_for`], or the card session's own method — one law either way), and
/// the rows fact ([`AttnStepRows`]). What comes back is the bytes; writing them is the
/// caller's half.
pub fn build_attn_step_inputs<const NQH: u32, const NKVH: u32, const HD: u32>(
    geom: sd::AttnGeometry<NQH, NKVH, HD>,
    what: &str,
    extents: &AttnStepExtents,
    ceiling: Option<sd::GatherEntryCeiling>,
    rows: AttnStepRows<'_>,
) -> Result<AttnStepPayload, String> {
    // The pool and the group are the witness's own facts — never re-derived here.
    let pool = sd::PagedKvPool::of_geometry(geom);
    let gqa = geom.gqa_nz();
    match rows {
        AttnStepRows::Chunk {
            rows,
            mq,
            start,
            real,
            map,
        } => {
            chunk_slot_pattern(what, extents)?;
            let region = chunk_region(what, extents.mask.unwrap_or(0), pool, gqa, mq)?;
            let ceiling = ceiling.ok_or_else(|| {
                format!(
                    "{what}: this bundle's attention gathers, but a physical page is not a whole \
                     number of SLAB BLOCKS — no integer index can name a page boundary"
                )
            })?;
            // ⭐ THE THREE STEP INPUTS, MINTED BY ONE WALK of the region — id tables and validity
            // mask as fields of ONE value (`chunk_step_inputs`), so a staging that wanted the
            // tables of one region and the mask of another has no spelling. The refusal is one
            // door naming the whole family's failure classes. The mask's seed-safety theorem
            // (window (0, 0) holds a valid lane for every query row, of `real ≥ 1` and
            // `start ≥ 0`) is proven by the `real` TYPE — it is not re-read here.
            let inputs = region
                .chunk_step_inputs(&map, rows, start, real, ceiling)
                .ok_or_else(|| {
                    format!(
                        "{what}: could not build the chunk family's step inputs from this \
                         sequence's page map and chunk — an empty map, an entry past the pool's \
                         ceiling (never a zero table: zero is block 0, a REAL address), or real \
                         {} row(s) against the rung's {} row(s) is not a chunk this region states",
                        real.get(),
                        mq.count()
                    )
                })?;
            Ok(AttnStepPayload {
                kt: encode(what, sd::KvPlane::Kt, inputs.kt_entries())?,
                v: encode(what, sd::KvPlane::V, inputs.v_entries())?,
                rep: None,
                mask: inputs.mask().wire_bytes().to_vec(),
            })
        }
        AttnStepRows::Batch { rows, maps, write } => {
            batch_slot_pattern(what, extents)?;
            let num_seqs = rows.width().rows();
            let batch =
                sd::SplicedDecodeBatch::of_ids_extent(extents.kt.unwrap_or(0), pool, gqa, num_seqs)
                    .ok_or_else(|| {
                        format!(
                            "{what}: the ids slot's placement extent {} B is not a whole number \
                             of chunk tables at {} launch row(s) — the bake and the host disagree \
                             about the batch chunking",
                            extents.kt.unwrap_or(0),
                            num_seqs.get()
                        )
                    })?;
            let ceiling = ceiling.ok_or_else(|| {
                format!(
                    "{what}: this bundle's attention gathers, but a physical page is not a whole \
                     number of SLAB BLOCKS — no integer index can name a page boundary"
                )
            })?;
            let kt = sd::spliced_decode_ids_table(sd::KvPlane::Kt, batch, maps, ceiling)
                .ok_or_else(|| {
                    format!(
                        "{what}: could not build the batch's block-entry table from {} installed \
                         page map(s) — not one map per launch row, an empty map, or an entry past \
                         int32 (never a zero table: zero is block 0, a REAL address)",
                        maps.len()
                    )
                })?;
            let v = sd::spliced_decode_ids_table(sd::KvPlane::V, batch, maps, ceiling).ok_or_else(|| {
                    format!(
                        "{what}: could not build the batch's V-plane block-entry table — same law, \
                         same refusal, the V plane's own slot"
                    )
                })?;
            let rep = sd::spliced_decode_rep_ids(batch);
            let mask = sd::spliced_decode_mask_f16(batch, &rows.mask_histories(), write)
                .ok_or_else(|| {
                    format!(
                        "{what}: could not build the batch's pre-replicated validity mask — not \
                         one history per launch row"
                    )
                })?;
            Ok(AttnStepPayload {
                kt: encode(what, sd::KvPlane::Kt, &kt)?,
                v: encode(what, sd::KvPlane::V, &v)?,
                rep: Some(rep.iter().flat_map(|&r| r.to_le_bytes()).collect()),
                mask: mask.wire_bytes().to_vec(),
            })
        }
    }
}
