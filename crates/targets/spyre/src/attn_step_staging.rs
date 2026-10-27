// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE ATTENTION FAMILY'S HOST-STAGED STEP INPUTS — ONE BUILDER, TWO CALLERS.
//!
//! The contract `attn_step_slots` spells says the runtime writes the attention family's
//! host-staged step-input slots — the three every program declares (the two planes'
//! block-entry id tables and the validity mask) plus the decode row's own repetition
//! table — before any spliced attention program runs. TWO runtimes must write them:
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
    /// Separate requests: a decode batch (`batched_decode_hm` or the batched paged-fold family),
    /// whose rows are the rung's launch slots — padding rows replicate live 0, maps and histories
    /// included. `maps` is the slot map's own per-row map list; `write` is the batch's ONE shared
    /// write slot, where every row's new key lands this step.
    Batch {
        rows: sd::LaunchRows<'a>,
        maps: &'a [Vec<i64>],
        write: sd::BatchSlot,
    },
}

/// The slot extents a session's own discovery table states — the artifact's answers, not a
/// derivation. `None` = the bundle's placement declared no such slot. The first three are the
/// family's common contract; `rep` is the decode row's own fourth slot, declared by the batched
/// decode row alone (the chunk-family rows need no bridge — their score rows ARE their q rows).
#[derive(Clone, Copy, Debug, Default)]
pub struct AttnStepExtents {
    /// The validity mask's placement extent, in bytes.
    pub mask: Option<u64>,
    /// The Kᵀ-plane block-entry table's, in bytes.
    pub kt: Option<u64>,
    /// The V-plane block-entry table's, in bytes.
    pub v: Option<u64>,
    /// The batched decode row's repetition table's, in bytes.
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
    /// The validity mask (chunk form) or pre-replicated mask (batch form).
    pub mask: Vec<u8>,
    /// The batched decode row's repetition table — empty in the chunk family, which declares
    /// no such slot.
    pub rep: Vec<u8>,
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

/// ⭐⭐⭐⭐⭐ THE ALL-OR-NOTHING SLOT PATTERN — `Ok(())` = the three-slot COMMON form is declared;
/// every other combination is a refusal. A PARTIAL declaration is a bake/host disagreement, and so is
/// an EMPTY one: on this branch every attention program is one of the spliced kernels and all of
/// them declare the three common step inputs, so a bundle that runs attention while declaring none is a
/// bake/host desync, never a form to fall back from. ONE law for BOTH row forms — the chunk and
/// batch families state the same three slots. The decode row's FOURTH slot (the repetition
/// table) is NOT checked here: it is row-form-conditional — stated by the batched decode row
/// alone, whose request-major score rows meet head-major q/o rows only through the bridge — so
/// its all-or-nothing law lives in the builder's own arms, where the row form is known (the
/// batch arm refuses an undeclared slot, the chunk arm refuses a declared one). Pub so the
/// numeric gate can check the pattern BEFORE it derives the region its identity page map
/// needs — the check and the builder's own arm are this one function, never two spellings.
pub fn slot_pattern(what: &str, extents: &AttnStepExtents) -> Result<(), String> {
    match (extents.mask, extents.kt, extents.v) {
        (Some(_), Some(_), Some(_)) => Ok(()),
        (None, None, None) => Err(format!(
            "{what}: this bundle runs attention but declares NO attention step inputs — \
             every spliced attention kernel states all of them, so the bake and the host \
             disagree about what program this is (an unstaged id table reads as block 0, a \
             REAL address; an unstaged mask reads as zero, NO masking)"
        )),
        _ => Err(format!(
            "{what}: this bundle declares a PARTIAL set of attention step inputs — \
             the three-slot form is all-or-nothing, and an unstaged one reads as \
             block 0 (ids) or no mask (mask), both fluently wrong"
        )),
    }
}

/// ⭐ THE BATCH FAMILY'S REGION, FROM THE MASK SLOT'S OWN PLACEMENT EXTENT — the ladder's
/// declared region at the batched fold stick, not a host guess. The batched twin of
/// [`chunk_region`]: same door shape, same refusal class. Pub for the same reason — the numeric
/// gate mints its identity page maps over this region's pages, through this same door, before
/// it can name them.
pub fn batch_region(
    what: &str,
    mask_extent_bytes: u64,
    pool: sd::PagedKvPool,
    gqa: std::num::NonZeroU32,
    num_seqs: std::num::NonZeroU32,
) -> Result<sd::SplicedBatchRegion, String> {
    sd::SplicedBatchRegion::of_mask_extent(mask_extent_bytes, pool, gqa, num_seqs).ok_or_else(
        || {
            format!(
                "{what}: the mask slot's placement extent {mask_extent_bytes} B is not a whole \
                 number of page blocks at the batched stick ({} launch row(s), pool nkvh {} hd \
                 {}) — the bake and the host disagree about the region",
                num_seqs.get(),
                pool.nkvh(),
                pool.hd()
            )
        },
    )
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
                     {:?} plane — the two gathered planes number a page's slab blocks differently, \
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
            slot_pattern(what, extents)?;
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
            let payload = AttnStepPayload {
                kt: encode(what, sd::KvPlane::Kt, inputs.kt_entries())?,
                v: encode(what, sd::KvPlane::V, inputs.v_entries())?,
                mask: inputs.mask().wire_bytes().to_vec(),
                // ⛔ THE CHUNK FAMILY DECLARES NO REPETITION TABLE — its score rows are its
                // q rows (token rows), so a placement that declared the bridge's slot for a
                // chunk-row bundle is a bake/host disagreement about which program this is,
                // never a table to silently leave unstaged (an unstaged i32 slot reads as
                // all-zero row indices — REAL rows, the wrong ones).
                rep: {
                    if extents.rep.is_some() {
                        return Err(format!(
                            "{what}: this chunk-row bundle declares a repetition-table slot — \
                             only the batched decode row states one, so the bake and the host \
                             disagree about which attention program this is"
                        ));
                    }
                    Vec::new()
                },
            };
            Ok(payload)
        }
        AttnStepRows::Batch { rows, maps, write } => {
            slot_pattern(what, extents)?;
            let num_seqs = rows.width().rows();
            let region =
                batch_region(what, extents.mask.unwrap_or(0), pool, gqa, num_seqs)?;
            let ceiling = ceiling.ok_or_else(|| {
                format!(
                    "{what}: this bundle's attention gathers, but a physical page is not a whole \
                     number of SLAB BLOCKS — no integer index can name a page boundary"
                )
            })?;
            // ⭐ THE THREE STEP INPUTS, MINTED BY ONE WALK of the region — id tables and validity
            // mask as fields of ONE value (`batched_step_inputs`), so a staging that wanted the
            // tables of one region and the mask of another has no spelling. The refusal is one
            // door naming the whole family's failure classes.
            let inputs = region
                .batched_step_inputs(maps, &rows.mask_histories(), write, ceiling)
                .ok_or_else(|| {
                    format!(
                        "{what}: could not build the batch's step inputs from {} installed \
                         page map(s) at {} launch row(s) — not one map and one history per \
                         launch row, an empty map, or an entry past the pool's ceiling (never \
                         a zero table: zero is block 0, a REAL address)",
                        maps.len(),
                        num_seqs.get()
                    )
                })?;
            // ⭐ THE DECODE ROW'S OWN REPETITION TABLE — the bridge from the batch's
            // request-major score rows to its head-major q/o rows. ONE `[slabs · PG]` i32
            // table, slab-major slices of `PG`: slice `σ`'s entry `r` is
            // `σ · PG + rep[r]`, where score row `r` is `(s = r / NQH, h = r % NQH)`
            // and `rep[r] = h · S + s` is its head-major row. The bytes the entry names are
            // the q/o view's own STICKS, and the sticks are laid out by the frame the bytes
            // live in: at `SLABS == 1` the view reads the token stream itself, whose
            // stick-blocked residency puts head `h` request `s` at stick `h · S + s` — the
            // head-major row IS the stick; at `SLABS > 1` the relayout wrap moves the bytes
            // into its `[nqh·R, HD]` row frame first, whose own stick law (the wrap's doc,
            // `head_relayout.py`) puts head-major row `k` slab `σ` at stick `σ · nqh·R +
            // k`, and `nqh·R == PG` — so the entry is the slab's stick base `σ · PG` plus
            // the head-major row in both cases (the stream at one slab is the `σ == 0`
            // instance of the same law). ⛔ NOT `rep[r] · slabs + σ`: that spells the frame
            // FLAT (`row·HD + slab·STK`), a different byte order than any stick-blocked
            // residency — at `SLABS == 1` the two spellings agree (both are `rep[r]`), which
            // is why every head-dim-64 width stayed coherent while granite-8b batched
            // (slabs 2) answered word salad with no fault (10-27). The table is a CONSTANT
            // of the geometry and the batch width (never of the step's contents), so it is
            // staged here with the rest rather than recomputed per fill; at `S == 1` it is
            // the identity, so the solo decode's bytes are the table's own `rep[r] = r`.
            let num_seqs = num_seqs.get();
            let slabs = geom.slabs();
            let pg = u64::from(num_seqs) * u64::from(NQH);
            let mut rep = Vec::with_capacity((slabs as u64 * pg * 4) as usize);
            for sigma in 0..slabs {
                for r in 0..pg {
                    let s = r / u64::from(NQH);
                    let h = r % u64::from(NQH);
                    let head_major = h * u64::from(num_seqs) + s;
                    let entry = u64::from(sigma) * pg + head_major;
                    rep.extend_from_slice(&(entry as i32).to_le_bytes());
                }
            }
            // ⭐⭐⭐ AND THE PAYLOAD IS STAGED WHOLE AT THE IDX2ADDR REACH — the table's own
            // entries plus a ZERO TAIL. The card's index→address conversion reads a
            // TABLE-SIZED contiguous run of index words from each gather/scatter leg's own
            // stick base (dxp's `createIdx2AddrSdsc` takes its `N_` from the OP's, which the
            // leg cuts state at the table's row count), so the slot the placement sizes — and
            // the bytes the fill's `==` law stages — are the reach, not the table alone: the
            // rep is placed at the activation segment's tail, and an un-padded table leaves
            // the last leg's run crossing the segment's end (measured: `HMI=FETCH` at the
            // second index stick, the stream dead). The tail's zeros convert to the value's
            // own base — an address no leg's work division ever dereferences. The law and its
            // C++ cites live on [`RepTableReservation::reach_bytes`]; a one-stick table's
            // reach is itself, so the solo decode's bytes are unchanged.
            let reach = ktir_superdsc::placement::RepTableReservation::reach_bytes(
                rep.len() as u64,
                std::mem::size_of::<i32>() as u64,
            );
            rep.resize(reach as usize, 0);
            // The declared extent must BE the slot's own reach — the placement sizes the slot
            // from the kernel's own view through the same law, so any other number is a
            // bake/host disagreement about the bridge itself, never a truncation decision.
            if extents.rep != Some(rep.len() as u64) {
                return Err(format!(
                    "{what}: the repetition table's declared extent {} B is not the \
                     idx2addr reach of the {slabs} slab(s) x {pg} score row(s) of i32 the \
                     bridge states ({} B) — the bake and the host disagree about the batch's \
                     row order",
                    extents.rep.map(|e| e.to_string()).unwrap_or_else(|| "NONE".into()),
                    rep.len()
                ));
            }
            Ok(AttnStepPayload {
                kt: encode(what, sd::KvPlane::Kt, inputs.kt_entries())?,
                v: encode(what, sd::KvPlane::V, inputs.v_entries())?,
                mask: inputs.mask().wire_bytes().to_vec(),
                rep,
            })
        }
    }
}
