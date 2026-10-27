// SPDX-License-Identifier: Apache-2.0
//! THE FORWARD — one prompt chunk or one decode step, as a played tape.
//!
//! Both binders build a `ForwardInputs` and play the same `ForwardShape`; the kernels live in
//! `scratchy_target_spyre::forward_tape` and compute every value, so what remains here is the I/O
//! around them plus the KV and launch-slot bookkeeping a forward needs.
//!
//! Split out of `spyre_worker` because the worker is PLUMBING: it looks the model up and invokes
//! it, and this is the invoking.

// The pool/slot vocabulary is CARD-ONLY, so its imports carry the cfg of the blocks that use it.
#[cfg(feature = "spyre-hw")]
use scratchy_subtile::sdsc_abstract::{PoolPartition, SlotCount};
// The KTIR-emulator twin below (mask/rope pad-row clamping) — the card's twin shares the laws
// through `ChunkRows` already.
#[cfg(not(feature = "spyre-hw"))]
use std::num::NonZeroU32;
// The runner type is cfg-selected: the KTIR emulator's session on the host, the sendnn one on
// silicon. Everything around it (weight load, dynamic sources, KV loop, sampling) is shared.
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::sdsc_runner::{StepLogits, SuperDscSession};

use crate::error::ExecutorResult;
#[cfg(feature = "spyre-hw")]
use crate::spyre_pool::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;

/// ⭐ #32 (c3) — A SUBMITTED DECODE STEP: the launch is on the card, the
/// idx-stick read is enqueued, nothing collected. Everything the collect half
/// needs to turn the ring slot back into tokens. The RUNG INDEX travels beside
/// it in the caller, because each rung owns its own session and only that
/// session's executor can finish the read.
#[cfg(feature = "spyre-hw")]
pub(crate) struct SubmittedDecode {
    pub(crate) read: scratchy_target_spyre::superdsc_exec::DeferredRead,
    /// The launch's row count — the seat's `rows` the submit half set.
    pub(crate) rows: u32,
    pub(crate) result_id: u32,
    pub(crate) vocab: usize,
}

/// What `run_prefill_batch` hands back: a finished step's picks, or a
/// submitted decode step whose read is deferred. A decode batch is ALWAYS the
/// latter — one decode path — and a schedule that cannot defer collects it
/// in-call through the same [`collect_decode_tokens`] door.
#[cfg(feature = "spyre-hw")]
pub(crate) enum BatchOut {
    Done(Option<Vec<u32>>),
    Deferred(SubmittedDecode),
}

/// Bind the m=N batched-prefill activations + the mq>1 attention consts into `pf` and run ONE
/// forward over prompt tokens `toks` at positions `[start..start+mq)`. This populates `pf`'s resident
/// seg2 KV (mq slots, via the mq>1 cachewr) on-card. Mirrors the per-token decode binding but for mq
/// rows + the mq>1 consts: the head-major one-hot selectors Sel_q/Sel_kv/SelT (filled from the SSOT
/// `selector_head_src_col`) and the `[mq,mq_pad]` triu causal cmask (`prefill_causal_col_valid`). RoPE
/// cos/sin are bound PER POSITION ([mq,width], row i = pos i).
///
/// `want_logits` asks for the NEXT token's logits from the bundle's lm-head tail, which the emitter
/// folds to m=1 over the LAST baked row (see `LAST_HIDDEN_TID`). The caller sets it on the FINAL chunk
/// and then has no separate decode-of-the-last-token to run — that is the whole point: TTFT becomes one
/// forward instead of two, and the second one's ~29 ms weight-stream floor disappears. Every earlier
/// chunk leaves it false and passes a NULL out-pointer, so the shim skips the ~3 MB logits D2H and the
/// sen→IEEE convert. Returns `None` exactly when `want_logits` is false.
#[cfg(feature = "spyre-hw")]
pub(crate) fn run_prefill_batch(
    // ⛔ THE NEW-BLOCK LAW IS NOT A PARAMETER ANY MORE. It was `causal_col_valid: fn(usize, usize) ->
    // bool` — the row KIND travelling as a runtime value, which is the shape `QueryRows` exists to
    // retire. The kind is now the type of `per_request` below: a decode batch has `LaunchRows` and
    // takes its DIAGONAL, a prompt chunk has `None` and takes `ChunkRows`' TRIANGLE, and neither can
    // be handed the other's.
    // `Some(rows)` when the op rows are separate requests: `rows.row(i)` carries request `i`'s rotary
    // position AND the KV slots it holds, together (see `BatchRow` — they are two different numbers and
    // every bug here has been a site using one for the other), and the prefix mask materialises one
    // validity row per op row. It holds the LIVE rows ONLY (`LiveBatch`, minted by the slot map's own
    // walk): a launch slot past them is padding, and a padding row is LIVE 0 ALL THROUGH — same token,
    // same rotation, same new-block column, same prefix history — because its one cache write lands on
    // live 0's pool cell with no barrier between them, so only identical bytes are sound. Widened to
    // `LaunchRows` below, which is where that decision is taken, once, for all four.
    // ⭐⭐ AND THE FOLD GEOMETRY THE LAUNCH DECLARED, RIDING WITH THE ROWS. The prefix mask is blocked by
    // it (`row * pages + page`) and `fold_plan::fold_pass` inverts it (`rep / pages`, `rep % pages`), so
    // the two `pages` have to be one number — and the one that governs is the DECLARED one, since that is
    // what the runtime walks. It is paired with the rows rather than passed beside them because the pair
    // is the whole condition: rows that are separate requests need a per-request mask, and a per-request
    // mask needs the geometry the fold will step it by.
    // `None` for a prompt chunk, whose rows share `start` and read one broadcast row — and whose launch
    // declares no fold, so there is no decode-shaped value for it to carry.
    // ⭐ AND WHEN THE STEP's BODY IS A BATCH CHUNK, ITS LAW INSTEAD OF THE FOLD's GRID: the mask is then
    // ONE per-row block every row block reads (`BatchPrefix::Chunk`), not a block per pass.
    per_request: Option<(
        &scratchy_subtile::sdsc_abstract::LiveBatch,
        scratchy_subtile::sdsc_abstract::BatchPrefix,
    )>,
    // ⭐⭐⭐⭐⭐ THE GATHER'S INDEX TABLE — global stick-block numbers, one per (launch row, logical page),
    // built by the caller in the SAME loop that installs the page maps.
    //
    // ⛔ IT COMES FROM THE CALLER AND CANNOT BE RECOMPUTED HERE. Which physical page holds a row's
    // logical page is the pool's live state, it CHURNS as sequences are admitted and evicted, and this
    // function sees neither the pool nor the block tables. `ForwardInputs::kv_blocks` says the same
    // thing at the other end of the hand-off.
    //
    // `None` for a launch that stages none — a prompt chunk, or a bundle that emitted no gather. The
    // pairing is checked at `kernel_values`, against the BUNDLE's own placement rather than against
    // this argument, so "table but no gather" is inert and "gather but no table" is a refusal.
    kv_blocks: Option<&[scratchy_subtile::sdsc_abstract::GatherEntry]>,
    sh: &Shared<'_>,
    pf: &mut SuperDscSession,
    b: &BundleMeta,
    // ⛔ THERE IS NO `pages` PARAMETER, AND ITS ABSENCE IS THE LOCK. It was "pages this request holds", and
    // it fed the prompt-chunk mask extent — but only through an arm that a PREFIX-FREE prefill rung never
    // took (`page_slots == 0` ⇒ `pf.cap`, a pool number). A prompt chunk's extent is now the BAKE's own
    // pmask reserve (`SuperDscSession::pmask_slots`) and a decode batch's is the DECLARED fold grid, so
    // neither can be handed a page count computed at a call site. Deleting the parameter is what stops the
    // next caller from reaching for `ctx.host.len()` or a pool size again.
    // ⛔ THE SEED LAW, NOT A TOKEN LIST. This used to be `toks: &[usize]` — one token per live
    // row, read from the token pool — which made every decode step's embed-head bind depend on
    // the PREVIOUS step's collected pick and forced a drain before every submit. The device
    // writes the seeds itself (each step's ChainStage decomposes its pick into the persistent
    // sticks), so the host binds seeds ONLY where the card cannot have: a request's first step on
    // a rung session, a churned or padded seat, a rung switch. `SeedSource::OnCard` is that
    // statement — a stable seating — and binds nothing.
    seeds: scratchy_subtile::sdsc_abstract::SeedSource<'_>,
    // The width the bundle being run was BAKED at — the launch binds exactly this many rows,
    // live or padding, and every extent below (embeddings, masks, `run_step`'s m) is sized by it.
    prefill_m: scratchy_subtile::sdsc_abstract::RungWidth,
    // ⭐⭐⭐⭐⭐ WHETHER THE BUNDLE BEING RUN GATHERS — the SECOND fact `b` cannot answer, and it rides
    // beside `prefill_m` because it has exactly `prefill_m`'s provenance: a decode batch rung is
    // addressed by FINGERPRINT, so `b` is the single-request meta borrowed for its tensor IDS and
    // describes neither this launch's row count nor its placements. The gather is emitted only by
    // the collapsed `mq > 1` bodies, so `GathersKv::of(&b.facts)` answers `false` for every rung
    // that places an index — which the card reports as `0xa35e RAS::PCI::BusFence`, a fence with no
    // `vars` raised on senlib's own monitor thread, from two gather copies resolving
    // `idx * skip_addr + base` over a tensor no step staged.
    //
    // ⛔ STILL A [`GathersKv`], SO IT STILL HAS ONE DOOR. A `bool` here would be precisely the
    // "caller answers from a phase or a rung width" hazard that type exists to forbid; every caller
    // mints this from an artifact — `BakeFacts` for a prompt chunk, the rung's own layout for a
    // batched step.
    gathers: scratchy_target_spyre::wiring::GathersKv,
    start: usize,
    want_logits: LogitsWanted,
    // ⭐ THE SEAT LAWS THIS LAUNCH'S ROWS RUN UNDER — one law per row the
    // suffix's sampler picks for, staged before anything runs. The variant must
    // match the launch's own arm: `Prompt` for a prompt-final chunk (whose rows
    // are one request's consecutive positions), `Batch` for a decode batch (one
    // law per live row). A middle prompt chunk stages nothing (its suffix never
    // runs), so the value it passes is inert — but the arm/variant match is
    // still refused wherever the seat IS staged, never silently resolved.
    seat_laws: SeatLaws<'_>,
) -> ExecutorResult<BatchOut> {
    use scratchy_subtile::sdsc_abstract::{ChunkRows, decode_prefix_col_valid};
    // ⭐ NO `hd` HERE ANY MORE. The one head-count question this function asked (`h / hd`) is now
    // the bake's own `Geometry::nqh` — the quotient was the head count only for a square Q
    // projection, and a hybrid-attention arch (gemma-4) has none. The sendnn pool sizing is a
    // KV-width question, and it reads `Loaded`'s own two numbers through the threaded
    // `PagedKvPool` — `Shared` carries no head geometry at all.
    let (h, vocab) = (sh.hidden, sh.vocab);
    // The bundle is baked for EXACTLY m=prefill_m query rows, so run at prefill_m: the `real`
    // prompt tokens fill rows [0..real), rows [real..prefill_m) are PAD. A PROMPT CHUNK's pad row
    // replicates the LAST real token — its id (`tok_at`), its rotary position, and its causal extent
    // all clamp via `ChunkRows::row_logical_pos` — so the LAST baked row, which is where the emitter
    // reads the lm-head input, always carries the last real token even when the chunk did not fill its
    // rung. The causal mask still means no real row ever attends a pad COLUMN (a real row r < real has
    // extent col <= r < real), and the pad KV slots [real..prefill_m) sit beyond the request's
    // n_computed = start+real, so nothing ever reads them — the pad stays numerically inert.
    //
    // A DECODE BATCH's pad row replicates LIVE 0 instead, all four aspects, because its cache write is
    // not its own slot's: it is live 0's cell. See `LaunchRows` below.
    // The kind of a launch WITHOUT `per_request` — a prompt chunk, whose rows are consecutive positions
    // of one prompt. Its laws (the pad-row clamp, the causal triangle) hang off this type and nothing
    // else can reach them.
    let chunk = ChunkRows::chunk(prefill_m.rows());
    // The rung width as a host-side extent, taken once — every loop and buffer below is `mq`-sized.
    let mq = prefill_m.count();
    // ⛔ A PROMPT CHUNK'S ROWS ARE THE PROMPT'S OWN TOKENS — the host owns every one of them, so
    // `OnCard` is not a spelling a chunk can take: its head kernel is the host-pushed `EmbedRow`,
    // and "the card's own ChainStage writes" is a decode-chain fact a prefill forward does not
    // have. Refused here rather than silently read as an empty token list.
    let host_toks: &[usize] = match (&per_request, seeds) {
        (_, scratchy_subtile::sdsc_abstract::SeedSource::Host(t)) => t,
        // A decode batch on a stable seating: the card's own ChainStage writes
        // are every seat's seed — no host token exists, none is read, and the
        // empty slice states exactly that (the tokens walk below builds
        // nothing for it).
        (Some(_), scratchy_subtile::sdsc_abstract::SeedSource::OnCard) => &[],
        (None, scratchy_subtile::sdsc_abstract::SeedSource::OnCard) => {
            return Err(werr(
                "run_prefill_batch: a prompt chunk cannot seed from the card — every prompt row's \
                 token is a host fact and the EmbedRow push reads it",
            ));
        }
    };
    if per_request.is_none() && (host_toks.is_empty() || host_toks.len() > mq) {
        return Err(werr(format!(
            "run_prefill_batch: real={} not in 1..={mq}",
            host_toks.len()
        )));
    }
    // ONE TOKEN PER LIVE ROW. Both come from the same walk of the slot map at the call site, so this
    // cannot fire from there — it exists so no OTHER caller can pair a rung-width token vector with
    // live-only rows and put every row past the first on another request's token.
    //
    // ⭐⭐ AND EVERY BOUND ROW'S INPUTS ARE RESOLVED HERE, ONCE, by widening the live rows to the rung.
    // A padding slot borrows live 0's PAGE MAP, and the on-card paged cachewr address has no row term
    // inside a page (a pool cell is (plane, kv-head, slot-in-page, feature)) — so the padding row's one
    // cache write and live 0's land on the SAME physical cell, and the shim elides the pipeline barrier
    // between consecutive slot writes on a disjointness premise padding breaks: the two writes are
    // UNORDERED. What is written is the pad row's WHOLE FORWARD, so every input to that forward has to
    // be live 0's — token, rotation, new-block column and prefix history alike, because a difference in
    // any one of them changes the hidden state at layer 1 and every layer after it writes different K/V
    // into live 0's cell. `LaunchRows` is the one value that answers all four, so a row cannot be
    // mirrored in some aspects and not others.
    // The LIVE rows by slot, kept past the widening below — the seat's laws are indexed by LAUNCH
    // slot, and only the live batch says which slot is a request's own (a padding slot's law is
    // LIVE 0'S OWN — [`pad_law_of`], the pad clamp: its pick is never read, and its forward is
    // live 0's by the replica law).
    let live_batch = per_request.as_ref().map(|(rows, _)| *rows);
    let per_request: Option<(
        scratchy_subtile::sdsc_abstract::LaunchRows<'_>,
        scratchy_subtile::sdsc_abstract::BatchPrefix,
    )> = match per_request {
        Some((rows, fold)) => {
            let laid = rows.launch_rows(seeds, prefill_m).ok_or_else(|| {
                werr(format!(
                    "run_prefill_batch: {} live row(s) do not lay into a {mq}-row rung with this \
                     seed source — a Host source carries exactly one token per live row",
                    rows.live()
                ))
            })?;
            Some((laid, fold))
        }
        None => None,
    };
    // `real` — how many rows carry their OWN input. A prompt chunk's is its token count (checked
    // above); a decode batch's is its LIVE count, which `launch_rows` already validated against
    // the Host token list, so the two cannot disagree here.
    let real = match &per_request {
        Some((rows, _)) => rows.live(),
        None => host_toks.len(),
    };
    // Pad rows REPLICATE — whom is a kind question, and the kind is which of these two arms exists. A
    // decode batch's pad rows take LIVE 0's token, because a pad row's forward becomes its cache write
    // and that write races live 0's at the same cell. A prompt chunk's pad rows take the LAST real
    // token: the emitter reads the lm-head input from the last baked row, which must carry the last
    // real token (and a chunk's pad KV writes append at the row's own slot, past n_computed — no
    // shared cell).
    //
    // ⭐⭐⭐ AND ON AN `OnCard` DECODE LAUNCH THERE ARE NO HOST TOKENS AT ALL — every row's seed is
    // the card's own ChainStage write, bound by nobody. The vec is EMPTY then, the tape's ChainSeed
    // steps stage nothing (the launcher's skip, below), and the gather reads the sticks as the
    // previous step's stage left them. `Host` on a decode launch carries the SEEDS: the full
    // per-row clamp `launch_rows` laid, one entry per BOUND row (live rows' own, pads replicating),
    // because the seed fns (`chain_s_seed`/`chain_l_seed`) take the launch's own width.
    let tokens: Vec<usize> = match (&per_request, seeds) {
        (Some((rows, _)), scratchy_subtile::sdsc_abstract::SeedSource::Host(_)) => (0..mq)
            .map(|i| {
                rows.row(i)
                    .expect("i < mq")
                    .token()
                    .expect("a Host-seeded launch carries every row's token")
                    .get() as usize
            })
            .collect(),
        (Some(_), scratchy_subtile::sdsc_abstract::SeedSource::OnCard) => Vec::new(),
        (None, _) => (0..mq)
            .map(|i| {
                if i < real {
                    host_toks[i]
                } else {
                    host_toks[real - 1]
                }
            })
            .collect(),
    };
    // ⭐⭐⭐⭐⭐ THE HEAD COUNT FROM THE BAKE, NOT `h / hd`. The quotient is the head count only for a
    // SQUARE Q projection; gemma-4's is not (hidden 3840 against nqh·hd 4096/8192 — and the
    // tiny-allglobal fixture's hidden 128 against 4·64). Deriving it here staged a causal mask
    // covering only `h/hd` of the baked `nqh` head rows: the bind guard refuses OVER-binds only, so
    // the shortfall was silent, and the ops read the unstaged tail of the ADDITIVE mask as ZERO —
    // i.e. those heads ran with NO causal mask. MEASURED on the tiny fixture: heads 2-3 unmasked,
    // every layer, prefill; heads 0-1 exact. `b.wiring` is the bake the tape was lowered at, so its
    // `Geometry::nqh` is the count the placements were sized by and cannot disagree with them.
    let nqh = b.wiring.geometry.nqh as usize;
    // stick-padded query count (mqp) — MUST match the emitter's `mq.div_ceil(64)*64`. The old hardcoded
    // 64 was a LIVE bug for mq>64 (e.g. prefill_m=128): the causal mask / ATTN_ZERO were bound 64-wide
    // while the emitter reads them mqp-wide, so rows≥64 read past the bound data (non-causal leakage).
    let mq_pad = mq.div_ceil(64) * 64;
    let mask_neg = -(half::f16::MAX.to_f32() / 2.0);
    // Reserved tids — MUST match lower_subtile_tape_to_superdsc.rs.

    // HOST-STAGING TIMING (SCRATCHY_SDSC_PHASE_TIME, same flag as the shim's phase breakdown and the
    // runner's bind_loop line). Everything between here and `pf.run_step` builds ~300k f32 on the
    // host per chunk -- and it all happens BEFORE the bind loop, which itself happens before
    // the executor's predict, so NONE of it appears in preamble_h2d/compute/logits_d2h. Roughly 25% of
    // those floats are INVARIANT (cmask depends only on mq; ROPE_P, IDENTITY, ATTN_ZERO and the fp8
    // clamp consts depend on nothing at all) yet are recomputed and re-bound every single forward,
    // because binding an activation is the only mechanism the session exposes for getting a
    // non-weight tensor onto the device.
    let _t_stage = std::time::Instant::now();
    let mut acts: Vec<scratchy_target_spyre::wiring::Bind> = Vec::new();
    // Activations staged in the DEVICE's format already, bound without a narrowing pass. Only worth it
    // for one that is large enough to pay for the special case: the decode batch's prefix mask, which
    // is quadratic in the request count and 93% of everything bound. See the `pmask` branch below.
    let mut acts_f16: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)> = Vec::new();
    // The gather's index table, raw LE int32 — bound through `bind_input_raw` so the shadow write
    // copies it instead of re-encoding it as fp16.
    let mut acts_i32: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)> = Vec::new();
    // Embeddings [mq, hidden] (row i = token i, pad rows replicate the last real token). The emit reads
    // the m>1 residual stream (including this host-staged embedding AND the causal cmask below) rank-2
    // STICK-MAJOR to match the always-stick-major matmul I/O, so stage the bytes stick-major: element
    // (r,c) lands at (c/64)*(mq*64) + r*64 + (c%64). This is UNCONDITIONAL for prefill (m>1) to match the
    // reference's one consistent layout; decode uses a separate path. hidden must be a whole 64-stick
    // multiple for the fold (guarded). Was gated behind SCRATCHY_SUPERDSC_STICKMAJOR, but that env is
    // forwarded on no build.rs so it never reliably reached the emit bake — made unconditional instead.
    let stickmajor_emb = true;
    if stickmajor_emb && h % 64 != 0 {
        return Err(werr(format!(
            "prefill stick-major embedding staging: hidden={h} not a multiple of 64"
        )));
    }
    let row_pos = |p: usize| -> u32 {
        match &per_request {
            // `rope_pos`, NOT the row's write slot: a request rotated at the position its keys happen to
            // be stored at, rather than at its own token count, drifts as soon as a shared write slot
            // puts a hole in its history.
            // `.get()` HERE AND NOWHERE ELSE — this is the one site that stages the rotary table, which is
            // what the type's single exit is for. A PADDING slot's rotation is live 0's own position,
            // because the pad row's K is written to live 0's pool cell unordered against live 0's write,
            // and only a byte-identical K is sound there. NOT the launch position, and not the last live
            // request's: either stores a different key in a racing write.
            Some((rows, _)) => rows.row(p).expect("p < mq").rope_pos().get(),
            None => (start + chunk.row_logical_pos(p, real)) as u32,
        }
    };
    // Triu causal cmask [mq, mq_pad]: additive 0 where col ≤ row (valid), mask_neg else. ALWAYS
    // flat/row-major (element (i,j) at i*mq_pad+j): the emitter (attn.rs, assemble_attn_block) reads
    // this via In::sliced on a Stk<FlatTag> handle, and FlatTag's addressing is row-major
    // (sdsc_abstract.rs: StickKind::Flat => Addressing::RowMajor), never stick-scattered. The old
    // stick-major staging here was byte-identical to row-major ONLY because mq_pad<=64 (one stick) in
    // every case tested so far — the "mq_pad<=64 always" claim was FALSE in general (mq_pad =
    // mq.div_ceil(64)*64, so mq>64 gives mq_pad>64, a genuine two-or-more-stick layout mismatch
    // against the always-row-major FlatTag read). Collapsing to one unconditional row-major stage
    // removes that latent mismatch for any future mq_pad>64 case; a no-op byte-for-byte change for
    // every mq_pad<=64 config tested this session (including the KTIR_PREFILL_LEN=2 bisection).
    // ⭐ WHICH new-block law these rows obey comes from WHICH KIND OF LAUNCH THIS IS, and the kind is a
    // type rather than a parameter. A prompt chunk's rows are consecutive positions of ONE prompt, so
    // row `i` takes the TRIANGLE at its clamped logical position — the last baked row, where the
    // emitter extracts the lm_head input, then attends exactly the real prompt `[0, real)` even when
    // this chunk did not fill its rung (identity when real == mq). A decode batch's rows are
    // INDEPENDENT REQUESTS, so row `i` takes the DIAGONAL at its own launch slot — and a PADDING row
    // takes LIVE 0's column, from the same `LaunchRows::row` its token and rotation came from, so its
    // whole attention is live 0's. The chunk clamp is the wrong law for a padding request row: it hands
    // it column `real - 1`, the LAST live request's new token, which makes it a different computation
    // from live 0 and therefore a different set of bytes into live 0's cell.
    //
    // ⭐ AND THE LAW IS NOW A KEY, NOT A BUFFER: the mask is a function of (rung, real) for a chunk and
    // (rung, live) for a batch and of nothing else, so `CausalKey` names it and `CausalKey::mask` is
    // its one builder. A session that holds this key already holds this mask (`keyed_holds`), and the
    // launch builds and binds nothing for it.
    let causal = match &per_request {
        Some((rows, _)) => scratchy_subtile::sdsc_abstract::CausalKey::batch(rows),
        None => scratchy_subtile::sdsc_abstract::CausalKey::chunk(prefill_m, real),
    }
    .ok_or_else(|| {
        werr(format!(
            "run_prefill_batch: no causal key for {real} of {mq} rows"
        ))
    })?;
    // TILED PER HEAD (2026-07-28, STAGE 2 of the batched-over-heads attention rewrite,
    // ir::bridge::tiled_op_sdsc_op::attn): the unified emitter now reads cmask batched over ALL
    // `nqh*mq` rows in one op (rows=nqh*mq), not per-head — unlike pmask (prefix validity), cmask
    // (causal) genuinely varies per QUERY ROW (row r attends new-block col c iff c<=r), so a single
    // mb-broadcast row is wrong for mq>1: it would apply query-row 0's causal pattern to every row.
    // The fix is a real per-head TILE, not a broadcast: repeat the same [mq,mq_pad] block `nqh` times.
    //
    // BLOCK-MAJOR over the SUB-BLOCKS the emitter folds (attn.rs `nsub = mq_pad/64`): `nsub` blocks of
    // `[nqh*mq, 64]`, block `j` holding causal columns `[64j, 64j+64)`. Required because the emitter
    // reads the mask through a FLAT handle whose row stride IS the block width (`In::sliced`, width=64),
    // so a single `[nqh*mq, mq_pad]` row-major buffer would advance 64 per row where the true row pitch
    // is `mq_pad` — every row past the first would read the wrong columns as soon as mq_pad>64.
    // At mq_pad==64 (nsub==1) this is EXACTLY the previous flat ×nqh repeat, byte for byte.

    // ⭐⭐ THE SAME FORWARD TAPE DECODE PLAYS, AT `mq` ROWS. The embedding, both rotary tables and
    // the causal mask were ~90 lines of staging here and ~90 more in the decode binder; they are
    // one set of kernels in `forward_tape` now, and this is the second caller. The regimes differ
    // ONLY in the `ForwardInputs` they hand over — row count, positions, causal block, prefix
    // predicate — which is what the three lemmas in `forward_tape`'s tests establish.
    // ⭐ ONE VALUE, Tscratchy_target_spyre::wiring::O USES. The constants and the tape both need the launch's row count, and
    // when they were two expressions they disagreed — see `LaunchRows`.
    let launch_rows = scratchy_target_spyre::wiring::StagedRows::of_launch(mq)
        .ok_or_else(|| werr("prefill chunk with zero rows".to_string()))?;
    let consts = scratchy_target_spyre::wiring::synthetic_constants(
        // ⛔ `mq`, THE LAUNCH'S ROWS — not `b.m_cap`, the bundle's baked CAPACITY. See the decode
        // site for what the difference cost.
        &b.facts.constant_env(b.wiring, mq_pad, launch_rows),
    );
    // ⛔ THE VOCAB GATE COVERS WHAT THE HOST BINDS. An `OnCard` decode launch binds no token, so
    // there is nothing to check — its seeds are the card's own draws, which the sampler's own
    // vocab-width table produced. A `Host` launch's entries are prompt tokens or collected picks,
    // and an out-of-range one would gather past the embedding table.
    for &tok in &tokens {
        if tok >= vocab {
            return Err(werr(format!("prefill token {tok} >= vocab {vocab}")));
        }
    }
    let positions: Vec<u32> = (0..mq).map(row_pos).collect();
    // ⛔ THE PREFIX MASK IS THE ONE STEP THIS REGIME DOES NOT SHARE. A decode BATCH stages it
    // directly as f16 in fold-pass blocks (`decode_batch_prefix_mask_f16`) because it is quadratic
    // in the request count; that is a different tensor shape, not a different predicate, so it
    // stays below and the tape simply has no step for it.
    // ⭐ WHICH PREFIX BUFFER THIS LAUNCH WANTS — the ONE genuinely two-regime input, and now a
    // VALUE rather than a branch in the staging. A prompt chunk wants a broadcast row over the
    // bake's reserve; a decode BATCH wants one f16 block per fold pass, because the fold takes a
    // pass per (request, page) and `fold_delta` steps the mask one block per pass.
    let histories = per_request.as_ref().map(|(rows, _)| rows.mask_histories());
    let prefix_src = match (&per_request, &histories) {
        // ⭐ A BATCH CHUNK BODY: one per-row block, from the law the emitter baked the read with.
        (Some((_, scratchy_subtile::sdsc_abstract::BatchPrefix::Chunk(bc))), Some(hs)) => {
            Some(scratchy_target_spyre::forward_tape::PrefixSource::Chunk {
                chunk: *bc,
                histories: hs,
            })
        }
        (Some((_, scratchy_subtile::sdsc_abstract::BatchPrefix::Fold(fold))), Some(hs)) => {
            use scratchy_subtile::sdsc_abstract::PrefixMaskShape;
            // THE SAME SHAPE THE EMITTER READ, from the pool's page width — the emitter blocks the
            // mask by a PAGE, so deriving it from this launch's `cap` is the drift
            // `PrefixMaskShape` exists to prevent.
            const PER_PAGE: u32 = scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
            let shape =
                PrefixMaskShape::<{ scratchy_subtile::sdsc_abstract::POOL_STICK }, PER_PAGE>::new(
                    nqh as u32, prefill_m,
                )
                .ok_or_else(|| werr("decode batch: prefix mask has zero heads".to_string()))?;
            Some(scratchy_target_spyre::forward_tape::PrefixSource::Blocks {
                shape,
                fold: *fold,
                histories: hs,
            })
        }
        // ⛔ THE EXTENT IS THE BAKE'S RESERVE, NEVER THE SESSION'S `cap`. The ops sweep the
        // placement's whole width and an additive byte the host never wrote reads as ZERO = VALID,
        // so every swept column gets an explicit value; WHICH value is the request's business.
        _ => {
            pf.pmask_slots.map(
                |cap| scratchy_target_spyre::forward_tape::PrefixSource::Broadcast {
                    cap,
                    holds: |c: usize| decode_prefix_col_valid(c, start),
                },
            )
        }
    };
    // ⛔ THE STEP EXISTS IFF THE SOURCE DOES, AND THAT IS ONE EXPRESSION, NOT TWO. This was a
    // separately-computed `pmask_slots.is_some() || per_request.is_some()` — the same disjunction
    // `prefix_src` is built from, written out a second time twenty lines above it. Two expressions
    // that must agree is the hazard the whole `PlaceId` work exists to remove, and the failure is
    // asymmetric and silent in one direction: a step with no source is a loud refusal in
    // `kernel_values` ("PrefixMask step on a bundle with no mask"), but a SOURCE WITH NO STEP is a
    // mask the host computes and never uploads — an additive buffer the device reads as all-zero,
    // i.e. "every column valid", which is fluent, wrong, and looks like a model bug.
    // ⛔ THE CAPACITY COMES FROM `prefill_m`, NOT FROM `b.wiring`, AND THAT IS THE WHOLE BATCHED-DECODE
    // FIX. `prefill_m` is documented one screen up as "the width the bundle being run was BAKED at" —
    // it is this launch's authority for every other extent here (embeddings, masks, `run_step`'s m), so
    // letting the row CHECK come from a different source is the one-quantity-computed-twice hazard,
    // and the two sources genuinely disagree on the batched path:
    //
    // A decode batch rung is the same tape baked at `B` rows, addressed by FINGERPRINT rather than
    // through a wiring, so `Wirings.decode` describes every rung's tensor ids but only the
    // single-request graph's ROW COUNT. The batched caller passes the rung's own `seqs` as `prefill_m`
    // and `model.decode.last()`'s wiring for the ids — correct, because it is the same tape — and the
    // wiring then answers 1 for a launch that is legitimately `B` rows wide.
    //
    // Before the KTIR unification the wiring slot was last-write-wins and, because rungs bake
    // ASCENDING, it happened to hold the WIDEST rung's wiring — so this check passed by accident.
    // Tightening that slot to the single-request graph was itself a real fix (`ktir_decode_cb_*` was
    // stamping `m_cap = 96`, so every single-token decode computed 96 activation rows) and it left
    // this line comparing a batched width against a bundle that no longer described it:
    // "forward tape asked for 2 row(s) from a bundle baked to hold 1", every width >= 2, 0 succeeded.
    //
    // The guard still bites: `real > mq` is refused above, and `RowCapacity` has no integer door — it
    // is mintable only from a placement or from a manifest-derived `RungWidth`.
    let shape = b.wiring.forward_shape_within(
        scratchy_target_spyre::wiring::RowCapacity::of_baked_rung(prefill_m),
        launch_rows,
        prefix_src.is_some(),
        // ⛔ THE BUNDLE'S OWN ANSWER — the `KV_BLOCK_INDEX_TID` placement, via [`GathersKv`]. Deriving
        // it from `kv_blocks.is_some()` instead would make the caller's table the authority over the
        // descriptor, so a forgotten table would silently emit no step and leave the gather reading
        // block 0 for every row.
        //
        // ⛔ AND IT IS THE PARAMETER, NOT `b.facts`: on a batched rung `b` is the borrowed
        // single-request meta, whose layout places no index — see `gathers` in the signature.
        gathers,
        consts.len(),
    );
    let inputs = scratchy_target_spyre::forward_tape::ForwardInputs {
        hidden: h,
        num_q_heads: nqh,
        mq_pad,
        // The causal mask laid out as the emitter's new-token block reads it — the SAME law, on the same
        // three facts the bake keyed it on (the rung's rows and whether rows are requests, both carried
        // by the key; the head dim), off the SAME baked geometry the head count above came from. The
        // session's load-time staging of its full-chunk mask reads this one law too.
        causal_heads_per_block: b.wiring.causal_heads_per_block(causal),
        embed_tokens: sh.embed_tokens,
        tokens: &tokens,
        // ⭐ THE SEED LAW'S WIRE SPELLING: a decode launch seeds the sticks only in the `Host`
        // arm (the caller-owned tokens above, already clamp-laid per bound row); an `OnCard`
        // launch passes `None` and the launcher's ChainSeed steps stage nothing. A prefill launch
        // has no ChainSeed step at all, so `None` there is inert by the tape's own shape.
        chain_seeds: match (&per_request, seeds) {
            (Some(_), scratchy_subtile::sdsc_abstract::SeedSource::Host(_)) => Some(&tokens[..]),
            _ => None,
        },
        positions: &positions,
        causal,
        prefix: prefix_src,
        consts: &consts,
        mask_neg,
        stick_major: stickmajor_emb,
        // ⭐⭐⭐ THE GATHER'S TABLE, from the caller's own walk of the live page maps. `None` when this
        // launch has none, which `kernel_values` turns into a REFUSAL if the bundle gathers — never
        // into an empty buffer, because an unstaged index reads as block 0 and block 0 is a real
        // address (row 0's first page), so every row would answer from row 0's keys.
        kv_blocks,
    };
    let fsteps = shape.steps();
    let fops = scratchy_target_spyre::forward_tape::operands(&fsteps);
    let ftape = scratchy_target_spyre::forward_tape::tape_steps(&fsteps, &fops);
    let mut launcher = ForwardLauncher {
        shape: &shape,
        steps: &fsteps,
        inputs: &inputs,
        staged: None,
        acts: Vec::with_capacity(fsteps.len()),
        acts_f16: Vec::new(),
        acts_i32: Vec::new(),
        causal_held: pf.keyed_holds(scratchy_target_spyre::bundle_code::PlaceId::Act(
            scratchy_target_spyre::lower_subtile_tape_to_superdsc::ATTN_CAUSAL_TID,
        )),
        causal_keyed: None,
        launched: false,
    };
    scratchy_subtile::host_tape::play(
        &scratchy_subtile::host_tape::HostTape { steps: &ftape },
        &mut launcher,
    )
    .map_err(|e| werr(format!("prefill forward tape: {e}")))?;
    debug_assert!(launcher.launched, "the tape must reach its device step");
    acts.extend(launcher.acts);
    acts_f16.extend(launcher.acts_f16);
    acts_i32.extend(launcher.acts_i32);
    // ⭐ EVERY LAUNCH WHOSE SUFFIX RUNS BINDS ITS SAMPLER SEAT, every row of it, before
    // anything runs: the mint's noise/inv_t sources are per-session seg3 placements no
    // alias covers, so an unbound seat reads ZEROS — which under the greedy law is
    // exactly right. Greedy for every row until the on-card noise kernel fills this
    // buffer. A decode batch (per_request) and a prompt-FINAL chunk (the prefill rung's
    // own minted draw) both run their suffix; a middle prompt chunk discards it, so it
    // binds nothing.
    if per_request.is_some() || matches!(want_logits, LogitsWanted::LastRow) {
        let seat = b.wiring.sample_seat.as_ref().ok_or_else(|| {
            werr(format!(
                "bundle has no sampler seat (result t{}) — a launch whose suffix runs cannot \
                 pick without the minted draw",
                b.result_id
            ))
        })?;
        // ⭐ A DECODE BATCH'S SEAT IS THE RUNG'S OWN WIDTH. `b` is the single-request meta
        // borrowed for its tensor ids, so its seat states the SOLO rung's `rows = 1` — a
        // launch at `mq` rows binds and reads `mq` rows of the per-step pair, or every row
        // past the first draws from an unstaged seat (and the readback below returns one
        // token for a `mq`-slot launch). A prompt-final chunk keeps the prefill rung's own
        // seat verbatim: its draw is the FOLDED single row.
        let mut seat = *seat;
        if per_request.is_some() {
            seat.rows = mq as u32;
        }
        // ⭐ THE ROWS' LAWS, built to the launch's own arm — the variant must
        // match `per_request`, and a mismatch is refused here, never silently
        // resolved into a sampled request served greedy or a row sampling a
        // request that is not its own.
        let seat_rows = seat.rows as usize;
        let laws: Vec<RowLaw<'_>> = match (&seat_laws, live_batch) {
            // A PROMPT-FINAL CHUNK: one request, real row `i` at its own
            // absolute position `start + i`, and a pad row CLAMPED to the last
            // real position (the `tok_at` law — its whole forward is the last
            // real token's). The consumed pick is the LAST BAKED ROW's, so an
            // unfilled rung's pad rows must carry the same draw at the same
            // salt the last real row would: a greedy pad law would silently
            // pick greedy for a sampled request whenever the chunk did not
            // fill its rung.
            (SeatLaws::Prompt { draw }, None) => (0..seat_rows)
                .map(|r| RowLaw {
                    draw: *draw,
                    step: (start + r.min(real - 1)) as u32,
                })
                .collect(),
            // A DECODE BATCH: live slot `s` runs its own request's draw at its
            // own rotary position (`rope_pos` IS that request's token count);
            // a padding slot runs LIVE 0'S LAW — [`pad_law_of`], the pad clamp.
            (SeatLaws::Batch { draws }, Some(live)) => {
                batch_seat_laws(draws, live, seat_rows)?
            }
            (SeatLaws::Batch { .. }, None) => {
                return Err(werr(
                    "run_prefill_batch: decode-batch seat laws for a launch with no \
                     per-request rows — the laws must match the launch's own arm"
                        .to_string(),
                ));
            }
            (SeatLaws::Prompt { .. }, Some(_)) => {
                return Err(werr(
                    "run_prefill_batch: prompt-chunk seat laws for a decode batch — the laws \
                     must match the launch's own arm"
                        .to_string(),
                ));
            }
        };
        acts_i32.extend(sample_binds(&seat, &laws)?);
    }
    if let Some((id, key, vals)) = launcher.causal_keyed {
        pf.bind_keyed(id, key, &vals)
            .map_err(|e| werr(format!("prefill: keyed causal bind: {e}")))?;
    }

    // Prefix validity mask [cap]: 0 on valid resident prefix [0..start), mask_neg else -- SAME shared
    // formula decode uses (decode_prefix_col_valid), generalized with `start` (the resident prefix
    // length BEFORE this chunk; 0 for a cold-start prefill). The unified emitter
    // (ir::bridge::tiled_op_sdsc_op::attn) reads this via a head-row-broadcast operand (In::mb), so
    // ONE row suffices -- no per-head duplication needed (unlike decode's own pmask staging, which
    // over-duplicates ×nqh for historical reasons but is not wrong, since every duplicate is
    // identical). This was PREVIOUSLY MISSING ENTIRELY from batched prefill staging (the old
    // selector-matmul emitter this replaced assumed prefill only ever ran from an empty prefix, so no
    // prefix mask was ever needed) -- the unified emitter always reads pmask, so its absence here
    // would have read unbound (zero) seg0 data as the mask, i.e. treating every prefix slot as VALID
    // regardless of `start`, a real correctness bug for any non-cold-start chunk.
    // MASK EXTENT: one validity row per page of this request (see the decode-side note).
    // ONE VALIDITY ROW, OR ONE PER OP ROW — matching how the emitter reads it. A prompt chunk's rows
    // share this chunk's resident prefix, so one broadcast row is exact and is what the emitter asks
    // for. A decode batch's rows are separate requests, so the emitter reads per-row and this must
    // materialise `nqh*mq` of them, repeating each request's row across heads exactly as cmask
    // already is (row `h*mq + i` is query row `i` of head `h`).
    // Hoisted so the forward below can compare it against what the DEVICE holds — see the readback.

    // seq_pos MUST be `start` (this chunk's resident-prefix length), not 0 -- the FFI adds
    // `seq_pos·slot_stride_bytes` to the cachewr's KV write offset (see lower_attn_node's
    // slot_stride_bytes comment). Hardcoding 0 here meant every chunk in a multi-chunk prefill
    // wrote its new-token KV to the SAME physical slot [0..mq_pad), clobbering every earlier
    // chunk's cache instead of appending at [start..start+mq_pad) -- only the LAST chunk's KV
    // ever survived, explaining "completely incoherent" batched prefill across every other fix
    // attempted this session (found 2026-07-28 via a KTIR_PREFILL_LEN=2 multi-chunk bisection:
    // every chunk, even start=0, decoded into one repeating token).
    if flags().phase_time {
        let elems: usize = acts.iter().map(|(_, v)| v.len()).sum();
        let f16_elems: usize = acts_f16.iter().map(|(_, v)| v.len() / 2).sum();
        eprintln!(
            "[sdsc-host] prefill_stage={:.2} ms  ({} acts, {} f32 built; {} f16 acts, {} f16 built)",
            _t_stage.elapsed().as_secs_f64() * 1e3,
            acts.len(),
            elems,
            acts_f16.len(),
            f16_elems,
        );
    }
    // NO-LOGITS for every prompt chunk: the lm-head tail's D2H is gone with the host
    // pick — a prompt chunk launches no suffix and reads nothing back
    // (`StepLogits::Discarded`). A decode batch is the one exception: its suffix IS
    // the sampler, and its answer is the idx stick read back raw
    // (`StepLogits::LeftOnDevice`).
    log_binds(
        &format!("prefill mq={mq} start={start}"),
        b.wiring,
        &acts,
        &acts_f16,
    );
    let picked = match want_logits {
        // ⭐ THE ONE DECODE PATH, SUBMITTED: stage, launch, enqueue the idx-stick read,
        // return. The minted sampler consumes the `[rows, vocab]` result on the card and
        // leaves the idx stick in its place; this arm defers the ONE raw D2H of that
        // stick — never a logits read, never a host argmax — and the caller collects it
        // through `collect_decode_tokens`, in-call when the schedule cannot defer, from
        // the drain when it can. There is no second, synchronous decode arm.
        LogitsWanted::PerRequestDeferred => {
            if b.wiring.sample_seat.is_none() {
                return Err(werr("decode bundle has no sampler seat".to_string()));
            }
            // The rung's own width, not the solo meta's (see the seat-bind site above),
            // is what `mq` is here — and it rides out in `SubmittedDecode::rows`.
            pf.stage_step_acts(&acts, &acts_f16, &acts_i32)
                .map_err(|e| {
                    werr(format!(
                        "superdsc decode batch stage (mq={mq}, start={start}): {e}"
                    ))
                })?;
            pf.submit_step(mq, start).map_err(|e| {
                werr(format!(
                    "superdsc decode batch submit (mq={mq}, start={start}): {e}"
                ))
            })?;
            // ⭐ THE COPY IS THE PLACEMENT'S TRUTH on this path: a decode-batch
            // rung is the same tape baked at `mq` rows but addressed by
            // FINGERPRINT, so `b.wiring`'s seat is the single-request meta
            // (rows=1) while the launch — the logits block, the word tables,
            // the idx stick — is `mq` rows wide (`RowCapacity::of_baked_rung`
            // takes `prefill_m` for exactly this reason, one screen up). The
            // readback must read through the SAME rows the launch laid — the
            // deferred collect below carries `rows: mq` for exactly this law.
            let read = pf
                .defer_read_tensor_ring(
                    scratchy_target_spyre::bundle_code::PlaceId::Act(b.result_id as u32),
                    mq * 64 * 4,
                )
                .map_err(|e| {
                    werr(format!(
                        "superdsc decode batch idx read (mq={mq}, start={start}): {e}"
                    ))
                })?;
            return Ok(BatchOut::Deferred(SubmittedDecode {
                read,
                rows: mq as u32,
                result_id: b.result_id as u32,
                vocab,
            }));
        }
        LogitsWanted::None => {
            pf.run_step_i32(
                mq,
                start,
                &acts,
                &acts_f16,
                &acts_i32,
                StepLogits::Discarded,
            )
            .map_err(|e| {
                werr(format!(
                    "superdsc prefill run_step (mq={mq}, start={start}): {e}"
                ))
            })?;
            None
        }
        // ⭐ THE PROMPT-FINAL CHUNK'S PICK — on the card, through the prefill rung's
        // own minted sampler (the general mint): the suffix runs, the sampler consumes
        // the folded lm-head row there, and the idx stick comes back raw, exactly like
        // decode's. One raw D2H of the idx — never a logits read, never a host argmax.
        LogitsWanted::LastRow => {
            pf.run_step_i32(
                mq,
                start,
                &acts,
                &acts_f16,
                &acts_i32,
                StepLogits::LeftOnDevice,
            )
            .map_err(|e| {
                werr(format!(
                    "superdsc prefill run_step (mq={mq}, start={start}): {e}"
                ))
            })?;
            let seat = b
                .wiring
                .sample_seat
                .as_ref()
                .ok_or_else(|| werr("prefill bundle has no sampler seat".to_string()))?;
            let picked = read_sampled_tokens(pf, seat, b.result_id as u32, vocab)?;
            log_seat_readback(
                pf,
                seat,
                b.wiring.load_consts,
                &picked,
                &format!("prefill-final mq={mq} start={start}"),
            );
            Some(picked)
        }
    };
    // PREFILL-ROW PROBE (SCRATCHY_KV_PROBE): read back the prefill session's EMBED input
    // (t{embed_src}, [mq,hidden]) + the layer-0 K cache (t9, natural [nqh,cap,hd]) and print
    // per-ROW / per-SLOT norms. If the embed has mq non-zero rows but t9 has 1 slot, the mq>1
    // attention/proj collapses 8→1 downstream; if the embed is 1 row, the binding is the bug.
    // ── COMPREHENSIVE PREFILL TRACE (SCRATCHY_PREFILL_TRACE) ── After the batched prefill forward, dump
    // RICH health stats for the PERSISTENT tensors that feed the model, so ONE run localizes a
    // FINITE-but-WRONG KV (the seam fix left the KV finite but the model still ignores the prompt). Stats
    // are layout-invariant (min/max/#NaN/#inf/#overflow/#zero), so no de-stickify is needed; the
    // per-position K/V norms expose POSITIONAL scrambling (positions permuted/duplicated/zeroed ⇒
    // attention attends the wrong tokens ⇒ off-topic ramble). |x|>256 flags fp16 x²-overflow risk,
    // |x|>65504 flags actual fp16 overflow, zero-count flags uninitialized/underflow.
    Ok(BatchOut::Done(picked))
}
#[cfg(feature = "spyre-hw")]

/// Every activation a forward binds, against what the BAKE says that tensor is — one line, at
/// `debug`, per launch.
///
/// ⛔ THIS IS THE LIST NOBODY COULD SEE. Both binders build `acts` and hand it to `run_step`, and
/// no log ever stated what was in it. A decode step that staged 96 rows where one is read looked
/// identical, from outside, to one that staged one — the tape's step LIST does not depend on the
/// row count, only the volume does. `expected` comes from the generated wiring's own
/// `tensor_shapes`, so a bind that disagrees with the bundle is visible without a card.
///
/// `<=` is legitimate and common: `pmask` reserves `[nqh*mq, cap]` and binds a single broadcast
/// row. `>` is caught and refused by the executor. A bind that is neither is the interesting case,
/// and this is what makes it readable.
fn log_binds(
    what: &str,
    w: &scratchy_target_spyre::wiring::Wiring,
    acts: &[scratchy_target_spyre::wiring::Bind],
    acts_f16: &[(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)],
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let one = |id: scratchy_target_spyre::bundle_code::PlaceId, got: usize| -> String {
        let want = w
            .tensor_shapes
            .get(id.tid() as usize)
            .map(|&(r, c)| r as usize * c as usize);
        match want {
            Some(n) if n == got => format!("{id}={got}"),
            Some(n) => format!("{id}={got}/{n}{}", if got > n { "!!" } else { "" }),
            None => format!("{id}={got}/?"),
        }
    };
    let f32s: Vec<String> = acts.iter().map(|(id, v)| one(*id, v.len())).collect();
    let f16s: Vec<String> = acts_f16
        .iter()
        .map(|(id, b)| one(*id, b.len() / 2))
        .collect();
    tracing::debug!(
        "[sdsc-binds] {what}: {} f32 [{}]{}",
        acts.len(),
        f32s.join(" "),
        if f16s.is_empty() {
            String::new()
        } else {
            format!("  + {} f16 [{}]", acts_f16.len(), f16s.join(" "))
        }
    );
}

/// The worker's launcher — the third half of the split: spyre supplies the kernels and the tape,
/// this supplies the I/O.
///
/// ⛔ IT COMPUTES NOTHING. Every value a forward binds is [`scratchy_target_spyre::forward_tape::kernel_values`]'s, in the target
/// crate, because none of it touches a session; what is left here is stage → h2d → launch. That
/// is the whole reason the split exists, and it is why one launcher serves both the decode and
/// the batched-prefill regimes — they differ only in the `ForwardInputs` they hand it.
#[cfg(feature = "spyre-hw")]
pub(crate) struct ForwardLauncher<'a, V: Fn(usize) -> bool> {
    pub(crate) shape: &'a scratchy_target_spyre::forward_tape::ForwardShape,
    pub(crate) steps: &'a [scratchy_target_spyre::forward_tape::ForwardStep],
    pub(crate) inputs: &'a scratchy_target_spyre::forward_tape::ForwardInputs<'a, V>,
    pub(crate) staged: Option<(
        scratchy_target_spyre::bundle_code::PlaceId,
        scratchy_target_spyre::forward_tape::Staged,
    )>,
    pub(crate) acts: Vec<scratchy_target_spyre::wiring::Bind>,
    /// Binds already in the DEVICE's format — the batched prefix mask, and nothing else today.
    pub(crate) acts_f16: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)>,
    /// ⭐ RAW LE int32 binds — the gather's index table, and nothing else. Kept apart from
    /// [`Self::acts_f16`] because the two need DIFFERENT shadow writes: an f16 bind is still
    /// re-encoded (`ieee_to_sen_bytes`), an index must be copied verbatim.
    pub(crate) acts_i32: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)>,
    /// ⭐ THE CAUSAL KEY THE SESSION ALREADY HOLDS — `SuperDscSession::keyed_holds`, read before the
    /// tape plays. When it equals `inputs.causal`, the causal step builds nothing and binds nothing:
    /// the device keeps that key's mask.
    pub(crate) causal_held: Option<scratchy_subtile::sdsc_abstract::CausalKey>,
    /// The causal mask to bind KEYED, when the session did not already hold this launch's key —
    /// bound through `SuperDscSession::bind_keyed`, never through the per-step [`Self::acts`].
    pub(crate) causal_keyed: Option<(
        scratchy_target_spyre::bundle_code::PlaceId,
        scratchy_subtile::sdsc_abstract::CausalKey,
        std::borrow::Cow<'static, [f32]>,
    )>,
    /// Set when the tape's device step runs; the caller reads the logits from there.
    pub(crate) launched: bool,
}

#[cfg(feature = "spyre-hw")]
impl<V: Fn(usize) -> bool> scratchy_subtile::host_tape::Launcher for ForwardLauncher<'_, V> {
    type Error = String;

    fn host_call(&mut self, k: scratchy_subtile::host_tape::KernelId) -> Result<(), Self::Error> {
        let step = self
            .steps
            .get(k.0 as usize)
            .ok_or_else(|| format!("no forward kernel {}", k.0))?;
        // ⭐ A HELD KEY BUILDS NOTHING. The mask is a function of its key alone, so a session that
        // already holds this launch's key already holds this launch's mask, and the step stages
        // nothing for `h2d` to bind.
        if step.kernel == scratchy_target_spyre::forward_tape::ForwardKernel::CausalMask
            && self.causal_held == Some(self.inputs.causal)
        {
            self.staged = None;
            return Ok(());
        }
        // ⭐⭐ THE CHAIN'S SEED SKIP — THE `OnCard` ARM, the same shape as the held causal key. A
        // stable seating's seeds are the card's own ChainStage writes (the previous step's pick,
        // decomposed into the persistent sticks at its tail), so this launch stages NO seed and
        // the device's gather reads the sticks as they were left. Binding a host seed here would
        // need the PREVIOUS step's pick before this launch's submit — the drain fence the seed
        // law exists to delete.
        if step.kernel == scratchy_target_spyre::forward_tape::ForwardKernel::ChainSeed
            && self.inputs.chain_seeds.is_none()
        {
            self.staged = None;
            return Ok(());
        }
        let staged =
            scratchy_target_spyre::forward_tape::kernel_values(step, self.shape, self.inputs)?;
        // A `Const` step's TENSOR comes from the constant list the kernel index addresses, not
        // from the step (the shape does not repeat it).
        let tensor = match step.kernel {
            scratchy_target_spyre::forward_tape::ForwardKernel::Const(i) => {
                scratchy_target_spyre::bundle_code::PlaceId::Act(
                    self.inputs
                        .consts
                        .get(i)
                        .ok_or_else(|| format!("no constant {i}"))?
                        .0,
                )
            }
            _ => step.tensor,
        };
        self.staged = Some((tensor, staged));
        Ok(())
    }

    fn h2d(&mut self, t: scratchy_subtile::host_tape::TensorId) -> Result<(), Self::Error> {
        let causal_tid = scratchy_target_spyre::lower_subtile_tape_to_superdsc::ATTN_CAUSAL_TID;
        let Some((tensor, staged)) = self.staged.take() else {
            // The two steps that stage nothing: the causal mask when the session holds its key,
            // and the chain's seed when the seeds are the card's own (`OnCard` — the two stick
            // sources the phase's chain names).
            let chain_skip = match self.shape.phase {
                scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain } => {
                    self.inputs.chain_seeds.is_none()
                        && (t.0 == chain.s_src || t.0 == chain.l_src)
                }
                scratchy_target_spyre::forward_tape::ForwardPhase::Prefill => false,
            };
            return if (t.0 == causal_tid && self.causal_held == Some(self.inputs.causal))
                || chain_skip
            {
                Ok(())
            } else {
                Err("h2d with nothing staged".to_string())
            };
        };
        // The KERNEL decided the format; the transfer just honours it.
        match staged {
            // A NEW causal key: bound keyed, so it is uploaded once and not re-sent per step.
            scratchy_target_spyre::forward_tape::Staged::F32(v) if tensor.tid() == causal_tid => {
                self.causal_keyed = Some((tensor, self.inputs.causal, v));
            }
            scratchy_target_spyre::forward_tape::Staged::F32(v) => {
                // `Staged::F32` already carries the borrow/own distinction — a computed kernel
                // output is Owned, a constant table is Borrowed — so it passes straight through.
                self.acts.push((tensor, v))
            }
            scratchy_target_spyre::forward_tape::Staged::F16(b) => self.acts_f16.push((tensor, b)),
            // ⛔ ITS OWN LIST, AND ITS OWN BIND CALL. Routing int32 entries through `acts` would
            // round every address through an f32 and then through `f32_to_f16_le`; routing them
            // through `acts_f16` skips the narrowing but NOT `ieee_to_sen_bytes`, which re-encodes
            // each 4-byte entry as two fp16 values. Both produce a valid, different block number —
            // fluent output from another row's keys. See `Executor::bind_input_raw`.
            scratchy_target_spyre::forward_tape::Staged::I32(v) => {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for e in v.iter() {
                    // ⭐ THE ONE PLACE A `GatherEntry` BECOMES A NUMBER AGAIN — the wire encoder, which
                    // is what `as_i32` exists for. Everything upstream of here carries the type.
                    bytes.extend_from_slice(&e.as_i32().to_le_bytes());
                }
                self.acts_i32.push((tensor, bytes));
            }
            // ⭐ THE CHAIN'S SEED — raw u32 lane values, LE words verbatim through
            // the same raw-integer bind the index table rides (the staging door's
            // law: LE bytes are the device encoding for raw integer words), never
            // the IEEE→SEN convert an f16 arm would run.
            scratchy_target_spyre::forward_tape::Staged::U32(v) => {
                let mut bytes = Vec::with_capacity(v.len() * 4);
                for w in v.iter() {
                    bytes.extend_from_slice(&w.to_le_bytes());
                }
                self.acts_i32.push((tensor, bytes));
            }
        }
        Ok(())
    }

    fn launch(&mut self, _k: scratchy_subtile::host_tape::KernelId) -> Result<(), Self::Error> {
        // The session call itself stays with the caller, which owns the borrow and needs the
        // logits. The tape's contract is only that every bind precedes it, which
        // `a_forward_binds_then_launches_exactly_once` pins.
        self.launched = true;
        Ok(())
    }

    fn d2h(&mut self, t: scratchy_subtile::host_tape::TensorId) -> Result<(), Self::Error> {
        Err(format!(
            "d2h t{} — the forward tape reads back via run_step",
            t.0
        ))
    }
}

// ⛔ `sendnn`-ONLY. This was lost when the decode binder was rewritten: a script inserted
// `ForwardLauncher` just above by walking back over `///` lines but NOT `#[...]`, so it landed
// between this `#[cfg]` and the fn it guarded and the attribute attached to the launcher instead.
// Invisible under `-Fsendnn`; it broke the KTIR-only build, which is exactly what `8bd5c755b` had
// just repaired.
#[cfg(feature = "spyre-hw")]
/// What a chunk body attends, for [`chunk_index_table`]: the keys resident before it (positions
/// `0..resident`, which the chunk must hold) and the slots whose pages its table names.
///
/// ⛔ EACH ITEM CARRIES ITS OWN `#[cfg]` — see the note above: an item inserted under another's
/// attribute takes it, and the item it guarded is left ungated.
struct ChunkKeys {
    resident: usize,
    mapped: u32,
    /// Whose rows the body attends with. It decides the value leg's entry order
    /// (`ChunkScratch::lx_form`), and the bake took it from the same fact.
    rows: scratchy_subtile::sdsc_abstract::ChunkRowKind,
}

#[cfg(feature = "spyre-hw")]
/// ⭐⭐⭐⭐⭐ THE SPLIT-KV CHUNK'S INDEX TABLE — for the two bodies whose rows share ONE resident
/// history: the one-request decode and a prompt chunk's prefix. The body that will RUN (`step`) names
/// its chunk by its swept extent (`pages * PAGE_SLOTS`, the rung the macro recorded), so the table is
/// built for exactly that chunk — a table for another width names blocks the copies do not read and
/// leaves the ones they do at zero. `None` when the body gathers nothing.
///
/// The keys (see [`ChunkKeys`]) must all lie inside the chunk: `select_body` chose the smallest rung
/// covering them, so a shortfall means the ladder stopped below this context — attending a prefix that
/// ends early — and it refuses.
///
/// The pages are `req.page_map(ctx, want)` — the very map the launch installed, holes served from the
/// pool's reserved range included — so the table the card reads and the map the cache write used are one
/// derivation. A chunk wider than the map repeats the LAST real page; the broadcast validity row (over
/// logical slots, exactly the order the chunk gathers) silences those slots, the reference's pad-up law.
///
/// ⛔ EVERY FAILURE REFUSES, never a zero table: an unstaged entry reads ZERO, and zero is a VALID block
/// — page 0's keys, fluently wrong.
fn chunk_index_table(
    what: &str,
    // ⭐ THE POOL'S GEOMETRY, threaded from `Loaded` — `Shared` carries no kv width BY DESIGN (a
    // hybrid-attention arch has no single one), and the pool is the one lawful model-wide use (see
    // `Loaded::kv_dim`). The same two numbers the batched-decode site reads off the model.
    pool: scratchy_subtile::sdsc_abstract::PagedKvPool,
    session: &SuperDscSession,
    step: scratchy_target_spyre::superdsc_exec::StepBody,
    req: &ReqState,
    ctx: PageMapCtx<'_>,
    keys: ChunkKeys,
) -> ExecutorResult<Option<Vec<scratchy_subtile::sdsc_abstract::GatherEntry>>> {
    let ChunkKeys {
        resident,
        mapped,
        rows,
    } = keys;
    if !step.gathers().get() {
        return Ok(None);
    }
    const PER_PAGE: u32 = scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
    let swept = step.swept().get();
    let chunk_pages = swept / PER_PAGE;
    if !swept.is_multiple_of(PER_PAGE) || (chunk_pages as usize) * (PER_PAGE as usize) < resident {
        return Err(werr(format!(
            "{what}: the chunk body sweeps {swept} slot(s) — not a whole number of {PER_PAGE}-slot \
             pages, or fewer slots than the {resident} resident key(s). The chunk ladder's top rung \
             ({} pages) is below this context.",
            scratchy_subtile::sdsc_abstract::ChunkScratch::PAGES_MAX
        )));
    }
    let per_page = session.gather_entries_per_page(pool).ok_or_else(|| {
        werr(format!(
            "{what}: this bundle emits a GATHERED KV read, but a physical page is not a whole number \
             of STICK BLOCKS — no integer index can name a page boundary"
        ))
    })?;
    let scratch = scratchy_subtile::sdsc_abstract::ChunkScratch::of_chunk(pool, chunk_pages)
        .ok_or_else(|| {
            werr(format!(
                "{what}: a {chunk_pages}-page chunk is not expressible at this geometry, but the bake \
                 emitted one — the bake and the host disagree about the chunk"
            ))
        })?;
    let want = scratchy_subtile::sdsc_abstract::RowPages::holding(
        scratchy_subtile::sdsc_abstract::SlotCount::new(mapped),
    );
    let map = req.page_map(ctx, want).ok_or_else(|| {
        werr(format!(
            "{what}: cannot map {want:?} page(s): the host granted {} block(s) — every page holding a \
             key must be a host block",
            ctx.host.len(),
        ))
    })?;
    scratchy_subtile::sdsc_abstract::chunk_gather_index_table(
        scratch,
        rows,
        &map.as_i64(),
        per_page,
        // The body's own baked op-local pin budget — the same term the bake's `lx_form` witness
        // read off the layout, so the host refuses exactly the chunks the bake refused.
        step.lx_op_local_bytes(),
        // And the chunk form's probability-part pins — the second core-local term the same
        // witness summed beside the gathers, so the two arms cannot drift.
        step.lx_prob_pin_bytes(),
    )
    .map(Some)
    .ok_or_else(|| {
        werr(format!(
            "{what}: could not build the chunk gather's index table from this request's page map \
                 — the map is empty or an entry overflows int32"
        ))
    })
}

/// ⭐ WHAT ONE CHUNK'S FORWARD PRODUCED — the per-row logits the host sampler
/// reads. One enum (not a bare `Vec<Vec<f32>>`) because the shape is a fact of the
/// step the caller must not be able to confuse with the model's other outs.
pub(crate) enum ChunkOut {
    /// A prompt-only forward: no launch in it picked, nothing to sample. The first
    /// generated token is picked by the DECODE launch that runs the prompt's last
    /// token — never by a host readback.
    Prompt,
    /// TOKENS THE CARD PICKED, one per decode launch this chunk ran, in step order.
    /// The minted sampler's idx stick, read back raw — the only pick that exists.
    Tokens(Vec<u32>),
}

/// ⭐ ONE ROW'S LAW FOR THE SEAT — what the launch's row `r` draws. `None` is
/// the greedy law: the seed quadruple `(0, 0, 0, 0)` and `inv_t = 1.0`,
/// byte-identical to the bind this path staged before any request asked for a
/// draw. `Some` is that row's request's draw, and the STEP SALT is the row's
/// ABSOLUTE position — the one input that changes every launch of one
/// generation, so the same seed draws different noise (hence, past rare ties,
/// a different token) at every position instead of repeating one pick.
#[cfg(feature = "spyre-hw")]
pub(crate) struct RowLaw<'a> {
    pub(crate) draw: Option<&'a SampledDraw>,
    /// The row's absolute position — its request's own token count at this
    /// step, or (a prompt chunk's pad row) the last real position it clamps to.
    pub(crate) step: u32,
}

/// ⭐⭐⭐ A PAD SLOT'S LAW IS LIVE 0'S OWN (the pad clamp, auditor ruling
/// 2026-10-27). A decode pad row's whole forward is live 0's — the replica
/// law: it borrows live 0's token, rotation, page map and prefix history, so
/// its LOGITS ROW is live 0's byte for byte — and the draw is a deterministic
/// function of (logits row, seed words) alone, the hash key being
/// request-owned with no row term. So the pad row's pick equals live 0's pick
/// exactly when its staged seat words are live 0's, and they must be: the
/// pad's [`crate::ktir_node::Program::ChainStage`] decompose feeds its s/l
/// stick, whose next-step embed row writes into LIVE 0'S POOL CELL (the
/// racing write — the pad's cachewr and live 0's land on the same physical
/// bytes), so only live 0's decompose keeps that write identical. The greedy
/// case falls out whole: a greedy live 0 stages the gate's 0, and the pad
/// draws greedy over live 0's replicated logits exactly as before — ONE
/// mechanism for greedy and sampled, no kernel change, no emission movement.
/// Called by every staging walk that lays pad rows (the decode batch's and
/// the solo rung's), never spelled twice.
#[cfg(feature = "spyre-hw")]
fn pad_law_of<'a>(live0: &RowLaw<'a>) -> RowLaw<'a> {
    RowLaw {
        draw: live0.draw,
        step: live0.step,
    }
}

/// ⭐ THE DECODE BATCH'S OWN LAW WALK — one law per baked row of the seat:
/// live slot `s` runs its request's draw at that request's own rotary
/// position, and a padding slot runs live 0's ([`pad_law_of`]). Extracted
/// from the staging arm so the pad clamp is pinnable without a launch.
#[cfg(feature = "spyre-hw")]
fn batch_seat_laws<'a>(
    draws: &[Option<&'a SampledDraw>],
    live: &scratchy_subtile::sdsc_abstract::LiveBatch,
    seat_rows: usize,
) -> ExecutorResult<Vec<RowLaw<'a>>> {
    if draws.len() != live.live() {
        return Err(werr(format!(
            "run_prefill_batch: {} seat law(s) for {} live row(s) — one law per live request, \
             in slot order",
            draws.len(),
            live.live()
        )));
    }
    let mut next_live = 0usize;
    let mut laws: Vec<RowLaw<'_>> = Vec::with_capacity(seat_rows);
    for s in 0..seat_rows {
        match live.row(s) {
            Some(b) => {
                laws.push(RowLaw {
                    draw: draws[next_live],
                    step: b.rope_pos.get(),
                });
                next_live += 1;
            }
            None => laws.push(match laws.first() {
                Some(live0) => pad_law_of(live0),
                // A launch with no live row at all: nothing replicates, and
                // the greedy law stands.
                None => RowLaw { draw: None, step: 0 },
            }),
        }
    }
    Ok(laws)
}

/// ⭐ THE SEAT LAWS ONE LAUNCH RUNS ITS ROWS UNDER — the two shapes a caller
/// can know them in, and ONLY those two: a prompt-final chunk serves ONE
/// request whose rows are consecutive positions of one sequence, and a decode
/// batch serves one request PER LIVE ROW. The variant must match the launch's
/// own arm (`per_request`); a mismatch is refused where the seat is staged,
/// never silently resolved.
#[cfg(feature = "spyre-hw")]
pub(crate) enum SeatLaws<'a> {
    /// A prompt-final chunk: every baked row runs `draw` — real row `i` at its
    /// own position `start + i`, and a pad row at the last real position it
    /// clamps to (the pad row replicates the last real token's entire forward,
    /// and the CONSUMED pick is the last baked row's, so the pad row must carry
    /// the same draw the last real row would, or an unfilled rung would pick
    /// greedy for a sampled request).
    Prompt {
        draw: Option<&'a SampledDraw>,
    },
    /// A decode batch: one law per LIVE row, in slot order — that row's
    /// request's draw at that request's own position. A padding slot runs
    /// LIVE 0'S LAW ([`pad_law_of`], the pad clamp): its pick is never read,
    /// but its ChainStage decompose feeds the s/l stick whose embed row races
    /// live 0's pool cell, so it must draw live 0's pick — greedy when live 0
    /// is greedy, live 0's own draw when it is sampled.
    Batch {
        draws: &'a [Option<&'a SampledDraw>],
    },
}

/// ⭐ THE SEAT BINDS EVERY LAUNCH STAGES — the seat's per-step operands at
/// their baked offsets, one law per baked row. Spelled so the greedy law is
/// the plain first-max: the seed quadruple `(0, 0, 0, 0)` — the GATE
/// table's 0 IS the greedy law, `score = xm·inv_t + g·0 = xm·inv_t` exactly
/// (the fold's clamps keep g FINITE for any seed bytes, so `g·0` is +0, never
/// NaN) — and inv_t = 1.0. A sampled row stages its request's draw instead:
/// `(seed_lo, seed_hi, step, 1)` and that draw's own inverse temperature —
/// the gate's 1 turns the noise on, and the step salt is the row's absolute
/// position. ⭐ THE KEY IS REQUEST-OWNED ALONE (the P3 localization ruling,
/// auditor 2026-10-27): the seed and the request's own step count, never the
/// slot the request happens to land in — a request draws the same noise in
/// any baked row of any batched launch (the row word the first spelling
/// carried is GONE). FOUR `[rows, 1]` i32 word tables, one per member of the
/// quadruple (the auditor's wall-4 ruling (a)): each is `invt`'s own class —
/// one 128-B stick per row, the word at the STICK HEAD (`r · seed_row_words`,
/// the seat's baked stride — the Col operand mode reads the head alone, the
/// int-bcast leg's measured law) — and the worker writes the four words of
/// row r at the four tables' stick heads and nothing else (the sticks' zero
/// padding is the reservation's own, exactly inv_t's bind). ⭐⭐ THE GATE
/// TABLE ALONE (`seed_words[3]`) carries the β reservation (front-b-noise,
/// auditor 2026-10-27): its chain crosses the i32→f32 convert, so the table
/// reserves the convert's 64-element device sweep (two sticks per row,
/// `gate_row_words`) — the gate still lands at row r's stick HEAD, the same
/// byte, with the sweep's tail zero-filled by the staging. Emitted as LE
/// words for the raw (`acts_i32`) channel, which binds verbatim — i32 words
/// for the seed tables, IEEE f32 for inv_t; the f32 `Bind` channel converts
/// to SEN169 and is NOT verbatim.
#[cfg(feature = "spyre-hw")]
fn sample_binds(
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    laws: &[RowLaw<'_>],
) -> ExecutorResult<Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)>> {
    let rows = seat.rows as usize;
    if laws.len() != rows {
        return Err(werr(format!(
            "the sampler seat holds {rows} baked row(s) but {} law(s) were staged — one law per \
             baked row, never fewer, never more",
            laws.len()
        )));
    }
    // ⭐ THE SEAT'S OWN LINE — what this launch's rows draw, stated where the
    // words are built. The chat leg's order requires the server log to SHOW
    // the seat binding the seeds (a seeded request served greedy, or two
    // requests' seeds sharing one batched launch, is invisible without it):
    // one line per launch, every row's seed, step salt, gate and inv_t.
    tracing::debug!(
        "SEAT_BIND rows={} seed_words={:?} invt=t{} laws={}",
        rows,
        seat.seed_words,
        seat.invt,
        laws.iter()
            .enumerate()
            .map(|(r, law)| match law.draw {
                Some(d) => format!(
                    "r{r}:seed={}@step{}:inv_t{}",
                    d.seed, law.step, d.inv_t
                ),
                None => format!("r{r}:greedy@step{}", law.step),
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
    // (seed_lo, seed_hi, step, greedy_gate) — the greedy row is zeros
    // (the gate word's 0 is what makes the draw greedy); a sampled row is its
    // request's seed split low/high, its step salt, and the gate's 1. One word
    // per table, at each row's stick head. ⭐ NO ROW WORD (the P3 localization
    // ruling): the key is request-owned alone, identical in every row.
    let word_of = |k: usize, r: usize| -> i32 {
        let law = &laws[r];
        match (k, law.draw) {
            (0, Some(d)) => d.seed as u32 as i32,
            (1, Some(d)) => (d.seed >> 32) as u32 as i32,
            (2, Some(_)) => law.step as i32,
            (3, Some(_)) => 1,
            // A greedy row: zeros for the seed, the step and the gate.
            (_, None) => 0,
            // The quadruple is FOUR words (the seat's own `[u32; 4]`); a fifth
            // staged word has no law anywhere — refuse rather than stage a
            // word nobody defined.
            (_, Some(_)) => panic!(
                "seed word {k}: the seat's quadruple is four words, indexed 0..=3 — a \
                 fifth is a mint/seat disagreement, never a staged guess"
            ),
        }
    };
    // ⭐ THE SEAT STAGES THROUGH THE ONE COL LAW — `stage_col_stick_heads`,
    // the same fn the bake's ten UNIFORM tables spread through
    // (`LoadConst::card_bytes`), so the seat's words and the uniform tables
    // can never disagree about where a row's word lives. The stride is the
    // placement's own reservation; `seed_row_words` is its word form, and a
    // seat that disagrees with the derivation refuses here rather than stage
    // a word at a stride nobody placed.
    let stride_bytes =
        scratchy_target_spyre::wiring::col_row_stride_bytes(scratchy_tensors::DType::I32);
    if seat.seed_row_words as usize * 4 != stride_bytes {
        return Err(werr(format!(
            "the sampler seat's seed_row_words is {} but the `[rows, 1]` reservation derives {} \
             — the seat and the placement's own law disagree, refusing to stage the seed words",
            seat.seed_row_words,
            stride_bytes / 4
        )));
    }
    // ⭐⭐⭐ THE GATE ALONE STAGES THE β RESERVATION (front-b-noise, auditor
    // ruling 2026-10-27): its chain crosses the i32→f32 convert, so its table
    // reserves the convert's 64-element device sweep — `stage_gate_sweep`
    // writes the gate at row r's stick HEAD (the same byte the one-stick law
    // ever stated) and zero-fills the sweep's second sticks. The same
    // refusal-on-disagreement contract as `seed_row_words` above.
    let gate_stride_bytes = scratchy_target_spyre::wiring::gate_row_stride_bytes();
    if seat.gate_row_words as usize * 4 != gate_stride_bytes {
        return Err(werr(format!(
            "the sampler seat's gate_row_words is {} but the gate table's `[rows, 1]` device-sweep \
             reservation derives {} — the seat and the placement's own law disagree, refusing to \
             stage the gate",
            seat.gate_row_words,
            gate_stride_bytes / 4
        )));
    }
    let mut binds = Vec::with_capacity(1 + seat.seed_words.len());
    for (k, &tid) in seat.seed_words.iter().enumerate() {
        let table = if k == 3 {
            scratchy_target_spyre::wiring::stage_gate_sweep::<4>(
                rows,
                stride_bytes,
                gate_stride_bytes,
                |r| word_of(k, r).to_le_bytes(),
            )
        } else {
            scratchy_target_spyre::wiring::stage_col_stick_heads::<4>(rows, stride_bytes, |r| {
                word_of(k, r).to_le_bytes()
            })
        };
        binds.push((scratchy_target_spyre::bundle_code::PlaceId::Act(tid), table));
    }
    // inv_t per row: the greedy law's 1.0, or the draw's own 1/T — at
    // temperature 1.0 the two are the same bytes and only the seed and gate
    // words differ. The same col law and the same stride (an f32 word shares
    // the u32 stick geometry).
    let invt = scratchy_target_spyre::wiring::stage_col_stick_heads::<4>(rows, stride_bytes, |r| {
        laws[r].draw.map_or(1.0, |d| d.inv_t).to_le_bytes()
    });
    binds.push((
        scratchy_target_spyre::bundle_code::PlaceId::Act(seat.invt),
        invt,
    ));
    Ok(binds)
}

/// Read the minted sampler's idx stick back and decode it: one i32 per row at
/// lane 0 of a 64-lane padded row. Any word at or past `vocab` is the sampler's
/// sentinel — refuse it, never sample it.
///
/// ⭐ THE STICK IS SCATTERED, NOT ROW-MAJOR. A `[rows, 64]` SENUINT32 placement
/// stores rows INTERLEAVED at the format's own 32-element stick — `dev_off_stk`
/// with `SenUint32::ELEMS_PER_STICK` — so row r's lane 0 is word `r·32`, NOT
/// `r·64`: measured on the card (probe, 2-row rung: each 64-word physical block
/// held `32×row0's idx` then `32×row1's idx`, both blocks identical), where the
/// row-major stride handed EVERY row row 0's pick. At `rows = 1` the scattered
/// offset is 0 and the solo readback is byte-identical.
#[cfg(feature = "spyre-hw")]
fn read_sampled_tokens(
    session: &mut SuperDscSession,
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    result_id: u32,
    vocab: usize,
) -> ExecutorResult<Vec<u32>> {
    let rows = seat.rows as usize;
    let bytes = session
        .read_tensor_raw(
            scratchy_target_spyre::bundle_code::PlaceId::Act(result_id),
            rows * 64 * 4,
        )
        .map_err(|e| werr(format!("sampler idx readback (t{result_id}): {e}")))?;
    decode_idx_stick(&bytes, rows, vocab)
}

/// ⭐ #32 (c3) — THE COLLECT HALF of a submitted decode step: park on the idx
/// read's own completion (the loop's ONE fence — this read's D2H, then the
/// shadow's quiet point, never a whole-stream sync), then apply the exact
/// [`read_sampled_tokens`] decode law to the bytes the ring slot holds. The
/// SAME door whether the caller collects in-call (a schedule that cannot
/// defer) or from the drain — the only difference is when, never what.
#[cfg(feature = "spyre-hw")]
pub(crate) fn collect_decode_tokens(
    session: &mut SuperDscSession,
    sub: &SubmittedDecode,
) -> ExecutorResult<Vec<u32>> {
    let bytes = session.finish_deferred_read_raw(&sub.read).map_err(|e| {
        werr(format!(
            "sampler idx readback (t{}): {e}",
            sub.result_id
        ))
    })?;
    decode_idx_stick(&bytes, sub.rows as usize, sub.vocab)
}

/// The idx stick's own decode law, shared by the sync readback and the deferred
/// collect so the two cannot drift.
#[cfg(feature = "spyre-hw")]
fn decode_idx_stick(bytes: &[u8], rows: usize, vocab: usize) -> ExecutorResult<Vec<u32>> {
    let words: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // The idx stick's own format decides the stride — the same law every device
    // tensor's host-side address takes (`dev_off_stk`), never a guessed row pitch.
    let stk = <scratchy_subtile::superdsc_opspec::SenUint32 as scratchy_subtile::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
        as usize;
    (0..rows)
        .map(|r| {
            let off = scratchy_subtile::sdsc_abstract::dev_off_stk(&[rows, 64], 1, &[r, 0], stk);
            let w = words.get(off).copied().unwrap_or(-1);
            let t = u32::try_from(w).map_err(|_| {
                werr(format!(
                    "the card's token draw for row {r} is {w} — not a token id"
                ))
            })?;
            if t as usize >= vocab {
                return Err(werr(format!(
                    "the card's token draw for row {r} is {t} >= vocab {vocab} — the sampler's \
                     sentinel, never a token"
                )));
            }
            Ok(t)
        })
        .collect()
}

/// ⭐ THE SEAT READBACK — the greedy-draw localization (auditor order
/// 2026-10-27): after a launch that drew, read the seat's OWN staged bytes
/// back off the card and state them beside the pick, so ONE served run names
/// the broken stage. Three byte-groups, one debug line per launch:
///   * THE SEAT — the four seed words and inv_t, each row at its stick head,
///     through the same scatter law `sample_binds` staged (zeros or a greedy
///     quadruple on a sampled row ⇒ the seat path: the worker binds greedy
///     whatever the params);
///   * THE HASH CHAIN'S UNIFORM TABLES — hc1/hc2/hc3/c256 (`[rows, 1]`,
///     staged at LOAD) at row 0's and row 1's stick heads, and the lane
///     table's first words. A noise CONSTANT across lanes is exactly the
///     greedy signature — a per-lane constant added to every score cannot
///     move the argmax — and the hash is the only lane-varying term in the
///     noise (the seeds and the gate enter row-wise by design). Row 1's head
///     reading 0 where row 0's holds the constant is the DENSE-STAGING
///     signature: the table was staged `rows` words dense instead of one
///     word at every row's stick head — the class law the seat's own binds
///     follow.
///   * THE PICK — the logits row's own argmax (host, f16, `dev_off_stk`)
///     beside the card's pick; pick = argmax at temp 1.0 is the failure
///     itself, and WHICH byte-group broke names the stage.
/// The card's noise intermediate g is NOT addressable from the wiring (the
/// draw's internals are lower-layer ktir synthetics, no tids), so the
/// readback brackets it: the staged inputs on one side, the pick on the
/// other. A failed read logs its tid and stops the line — the readback never
/// gates the run.
#[cfg(feature = "spyre-hw")]
fn log_seat_readback(
    session: &mut SuperDscSession,
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    consts: &[(u32, scratchy_target_spyre::wiring::LoadConst)],
    picked: &[u32],
    ctx: &str,
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    // LE words of a placement's first `words` words — the error names the tid.
    fn le_words(session: &mut SuperDscSession, tid: u32, words: usize) -> Result<Vec<i32>, String> {
        session
            .read_tensor_raw(
                scratchy_target_spyre::bundle_code::PlaceId::Act(tid),
                words * 4,
            )
            .map(|b| {
                b.chunks_exact(4)
                    .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect()
            })
            .map_err(|e| format!("t{tid}: {e}"))
    }
    // A staged table's `(tid, rows)` by its kernel operand number (1..=20,
    // the doc order in `sample_staged_tables`).
    fn uniform(
        consts: &[(u32, scratchy_target_spyre::wiring::LoadConst)],
        operand: u8,
    ) -> Option<(u32, usize)> {
        consts.iter().find_map(|&(tid, k)| match k {
            scratchy_target_spyre::wiring::LoadConst::SampleTable {
                operand: o, rows, ..
            } if o == operand => Some((tid, rows as usize)),
            _ => None,
        })
    }
    let rows = seat.rows as usize;
    // The four word tables + inv_t — the full reservation, each row's word at
    // its stick head (`sample_binds`' own law, read back through the same
    // stride it staged).
    let mut words: Vec<[i32; 4]> = vec![[0; 4]; rows];
    for k in 0..4 {
        match le_words(
            session,
            seat.seed_words[k],
            rows * seat.seed_row_words as usize,
        ) {
            Ok(ws) => {
                for r in 0..rows {
                    words[r][k] = ws
                        .get(r * seat.seed_row_words as usize)
                        .copied()
                        .unwrap_or(-1);
                }
            }
            Err(e) => {
                tracing::debug!("SEAT_READBACK {ctx}: seat word table read failed ({e})");
                return;
            }
        }
    }
    let invt: Vec<f32> = match le_words(session, seat.invt, rows * 32) {
        Ok(ws) => (0..rows)
            .map(|r| ws.get(r * 32).copied().unwrap_or(0))
            .map(|w| f32::from_bits(w as u32))
            .collect(),
        Err(e) => {
            tracing::debug!("SEAT_READBACK {ctx}: inv_t read failed ({e})");
            return;
        }
    };
    // The hash chain's uniform tables at row 0's and row 1's stick heads.
    let mut hash = String::new();
    for (name, operand) in [("hc1", 10u8), ("hc2", 11), ("hc3", 12), ("c256", 13)] {
        match uniform(consts, operand) {
            Some((tid, trows)) => {
                // ⭐ THE PLACEMENT IS THE LAUNCH'S ROWS on a batched rung: the
                // wiring's kind carries the single-request meta (rows=1)
                // while the rung's graph mints the table at the baked width,
                // and the STAGING follows the kind — so reading the seat's
                // own rows is what measures whether every stick head the
                // launch addresses actually holds the constant (a staged
                // one-word stream leaves rows past 0 reading ZERO).
                let trows = trows.max(rows);
                match le_words(session, tid, trows * 32) {
                Ok(ws) => {
                    let w0 = ws.first().copied().unwrap_or(-1);
                    let r1 = if trows > 1 {
                        format!("/r1=0x{:08X}", ws.get(32).copied().unwrap_or(-1))
                    } else {
                        String::new()
                    };
                    hash.push_str(&format!("{name}(t{tid})=0x{w0:08X}{r1} "));
                }
                Err(e) => hash.push_str(&format!("{name}: READ-FAIL({e}) ")),
                }
            }
            None => hash.push_str(&format!("{name}: NOT-IN-CONSTS ")),
        }
    }
    // The lane table's first words — DISTINCT values prove the hash's one
    // per-lane input lives; a zero or constant table flattens the noise.
    let lane = match uniform(consts, 14) {
        Some((tid, _)) => match le_words(session, tid, 4) {
            Ok(ws) => format!("lane(t{tid})={ws:?}"),
            Err(e) => format!("lane: READ-FAIL({e})"),
        },
        None => "lane: NOT-IN-CONSTS".to_string(),
    };
    // The logits row's own argmax, host-side, f16 through the same stick law
    // the idx readback uses — the REPORT rows only (every row of a wide
    // prefill rung would be minutes of nothing).
    let report: Vec<usize> = if rows > 5 {
        (0..4).chain([rows - 1]).collect()
    } else {
        (0..rows).collect()
    };
    let pad_v = seat.pad_vocab as usize;
    let vocab = seat.vocab as usize;
    let stk = <scratchy_subtile::superdsc_opspec::Fp16 as scratchy_subtile::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
        as usize;
    let mut argmax: Vec<(usize, u32)> = Vec::with_capacity(report.len());
    match session.read_tensor_raw(
        scratchy_target_spyre::bundle_code::PlaceId::Act(seat.row),
        rows * pad_v * 2,
    ) {
        Ok(b) => {
            for &r in &report {
                let mut best = (0usize, f32::NEG_INFINITY);
                for j in 0..vocab {
                    let e = scratchy_subtile::sdsc_abstract::dev_off_stk(
                        &[rows, pad_v], 1, &[r, j], stk,
                    );
                    let o = e * 2;
                    if let Some(v) = b.get(o..o + 2) {
                        let x = half::f16::from_le_bytes([v[0], v[1]]).to_f32();
                        if x > best.1 {
                            best = (j, x);
                        }
                    }
                }
                argmax.push((r, best.0 as u32));
            }
        }
        Err(e) => {
            tracing::debug!("SEAT_READBACK {ctx}: logits row t{} read failed: {e}", seat.row);
        }
    }
    let per_row: Vec<String> = report
        .iter()
        .map(|&r| {
            let pick = picked.get(r).copied();
            let am = argmax
                .iter()
                .copied()
                .find(|(ar, _)| *ar == r)
                .map(|(_, a)| a);
            format!(
                "r{r}:words={:?}:invt={}:pick={:?}:argmax={:?}:eq={}",
                words[r],
                invt[r],
                pick,
                am,
                pick == am
            )
        })
        .collect();
    tracing::debug!(
        "SEAT_READBACK {ctx} rows={} {} {} {}",
        rows,
        hash,
        lane,
        per_row.join(" ")
    );
}

#[cfg(feature = "spyre-hw")]
pub(crate) fn superdsc_forward_chunk(
    sh: &Shared<'_>,
    session: &mut SuperDscSession,
    // The prefill WIDTH LADDER, ascending by baked mq (see `SuperDscBundle::prefill_rungs`). Each
    // chunk runs on the SMALLEST rung that holds it, so a short prompt stops paying for padding rows
    // and a long one still gets the widest chunk. Empty ⇒ no batched prefill.
    prefill_rungs: &mut Vec<(usize, SuperDscSession)>,
    // The prefix-capable ladder for continuation chunks (start>0); the rungs above are prefix-free.
    // A continuation chunk's width is chosen by `PrefixLadder::rung_for` and nowhere else.
    prefill_prefix: &mut PrefixLadder,
    b: &BundleMeta,
    // The prefill meta bound by `run_prefill_batch` to `prefill_session` (its tensor ids match the prefill
    // manifest, unlike `b`, which is the decode meta the generation steps below use).
    prefill_meta: Option<&BundleMeta>,
    req: &mut ReqState,
    // ⛔ THE STORE, LIVE AND MUTABLE — the chunk loop slices this request's tokens out of it AND advances
    // its `tokens_in_pool`, because how far a chunked prefill got is known only here, after the loop. `id`
    // indexes it. Passed rather than snapshotted for the reason `b4856997` records: a snapshot taken
    // upstream of this step's writes is a stale copy, and that cost every prompt past one scheduler step.
    id: &str,
    tokens: &mut scratchy_serving_engine::input_batch::InputBatch,
    start: usize,
    toks: &[usize],
    // THE HOST'S LIST + THE POOL'S OWNER SPLIT — needed because every bind DERIVES this request's page
    // map rather than reading a stored one, and the map's fully-masked and hole pages come from the
    // reserved range.
    ctx: PageMapCtx<'_>,
    // The pool's geometry, threaded from the model — `chunk_index_table`'s law, ONE spelling (built in
    // `run_request_step` off `Loaded`'s own two numbers).
    pool: scratchy_subtile::sdsc_abstract::PagedKvPool,
) -> ExecutorResult<ChunkOut> {
    let n = toks.len();
    if n == 0 {
        return Err(werr("superdsc_forward_chunk: n=0".to_string()));
    }
    // ⭐ NO `hd` HERE EITHER — the solo-decode causal tile's head count is the bake's
    // `Geometry::nqh` (see the `num_q_heads` site below), and nothing else in this
    // function asks a head-dim question.
    let (h, vocab) = (sh.hidden, sh.vocab);
    // MASK EXTENT: one validity row PER PAGE, laid end to end. The fold is re-launched per page and
    // walks along this buffer, so page `i`'s row sits `i * page_slots` in.
    // ⛔⛔⛔ SIZED BY THE PAGES THE LAUNCH **WALKS**, NOT BY THE HOST'S BLOCK COUNT.
    //
    // Those two are the same number only for a request whose keys are still at their own token positions. A
    // row that has shared a batched step carries a HOLE, so the LOGICAL pages the fold walks — and therefore
    // the validity rows it reads — EXCEED the blocks the host granted for its tokens; the difference is
    // served by `PoolPartition`'s scratch and hole pages. Sizing this buffer by `ctx.host.len()` under-
    // allocates it and the fill runs off the end.
    //
    // MEASURED: `corrupted size vs. prev_size while consolidating`, a glibc heap abort raised while DROPPING
    // the activation `Vec<(String, Vec<f32>)>` this buffer lives in — `addr2line` on the abort's own
    // backtrace put it in THIS function. It fires after every answer is already computed and it is
    // INTERMITTENT, which is why it read as a teardown property of `scr batch` for a day.
    //
    // ⭐ THE SAME SLOT-VS-TOKEN CONFUSION AS EVERY OTHER DEFECT IN THIS PATH, in its most dangerous form:
    // the others produced a wrong token, a wrong SIZE corrupts the allocator.
    let cap = if session.page_slots > 0 {
        let walked = scratchy_subtile::sdsc_abstract::RowPages::holding(SlotCount::new(
            req.kv_hist.end().get().max((start + n) as u32),
        ));
        session.page_slots * (walked.get() as usize).max(ctx.host.len()).max(1)
    } else {
        session.cap
    };
    // ── CHUNKED BATCHED PREFILL (whole prompt, any resident prefix length). ──
    // When a distinct m=N `prefill` session exists AND there's >1 token, run the WHOLE prompt through
    // it as a sequence of ≤prefill_m batched forwards (populating its resident seg2 KV on-card via the
    // mq>1 cachewr, one ≤prefill_m chunk at a time). The prompt's LAST token IS in this
    // call, as on main — the prompt-final chunk's own minted sampler picks the first
    // generated token on the card; `run_request_step` runs no decode launch for a prompt
    // step.
    //
    // Reaching the decode loop below therefore means this call is a single generation step (n == 1):
    // a multi-token prompt returns from the batched prefill (no logits, no sample) or is refused, so
    // the loop runs on `start`/`toks` directly.
    // ── SEQUENTIAL mq=1 PREFILL IS NUKED (scripts/verify_prefill_real.sh is the SOLE authority) ──
    // A multi-token PROMPT either goes through the batched mq=N forward, or the request is REFUSED.
    // There is NO env escape, NO silent fallback, NO "walk the prompt one token at a time". Every
    // prompt prefill emits exactly one unconditional, machine-parseable truth line.
    let n_entry = n; // rows at entry: >1 ⇒ a prompt PREFILL request; ==1 ⇒ a decode step
    let prefill_top_m = prefill_rungs.last().map(|(m, _)| *m).unwrap_or(0);
    if !prefill_rungs.is_empty() {
        // CHUNKED BATCHED PREFILL: cover the WHOLE prompt (all but the last token) as a sequence of
        // ≤prefill_m batched forwards, not just one. GENERAL-mq: the unified emitter
        // (ir::bridge::tiled_op_sdsc_op::attn) handles any resident prefix length via pmask, so each
        // chunk's `start` (the resident prefix built by every PRIOR chunk, or an existing prefix from
        // a prior request continuation) just keeps advancing — there is no "only chunk 0" restriction
        // at the kernel level. The ONLY thing that was missing was this loop: the prior code called
        // run_prefill_batch ONCE and REFUSED any prompt longer than one prefill_m-sized chunk, even
        // though the exact same call already supported a nonzero `start` for continued generation.
        if n > 1 {
            let real_total = n; // every token THIS call was given (the whole prompt — no hold-back)
            // Bind with the PREFILL meta (its embed/cos/sin/ONES tensor ids match the prefill session);
            // `b` (decode meta) would SILENTLY bind the wrong ids. batched_prefill is Some whenever the
            // prefill session that got us here exists, so this never fires — but fail LOUD if that invariant
            // ever breaks, rather than corrupting the prefill bind.
            let pmeta = prefill_meta.ok_or_else(|| {
                werr(
                    "superdsc batched prefill: prefill_meta missing for a live prefill session — refusing to \
                     bind decode-graph ids into the prefill session (batched_prefill wiring bug)"
                        .to_string(),
                )
            })?;
            // ⭐⭐⭐ TWO CURSORS, BOTH ABSOLUTE: the KV SLOT this chunk writes at, and the TOKEN position
            // it reads from. Both start at `start` and advance together, so `chunk_start - tok` is 0 for
            // the whole loop — the loop cannot represent a hole, and needs none: the request's history
            // is contiguous on entry (the batched-prefill gate's precondition).
            //
            // 🛑 `tok` USED TO BE `off`, AN OFFSET INTO **THIS STEP'S** TOKEN WINDOW, and that is what made
            // the step-back unrepresentable: `chunk_write_start` moves the write slot BACKWARD to keep a
            // padded window inside its page, the same number of tokens must move with it, and an offset
            // cannot go below zero. Measured: `off -= back` wrapped to
            // `range start index 18446744073709551521 out of range for slice of length 121` and failed all
            // four requests of a batch. Absolute, the step-back is just arithmetic — the request holds its
            // whole prompt, so re-reading earlier tokens is a slice, and re-writing their keys at the same
            // slots is idempotent.
            let mut chunk_start = start;
            let mut tok = start;
            let tok_end = start + real_total;
            // The final chunk's on-card pick, carried out of the loop (a mid-prompt
            // wave's last chunk is NOT a prompt-final chunk and picks nothing).
            let mut prompt_pick: Option<Vec<u32>> = None;
            while tok < tok_end {
                // Cap the chunk at the WIDEST rung, then run it on the NARROWEST rung that holds it.
                // Both halves matter: the cap keeps chunk count minimal (each extra chunk re-pays the
                // m-independent per-chunk cost), and the rung choice keeps the padding rows
                // (`rung_m - chunk_len`) minimal within that chunk.
                // ROUTE BY `start`. The ladder rungs are baked PREFIX-FREE (no resident-prefix
                // blocks at all), which is exactly right for a chunk at start==0 and WRONG for any
                // continuation chunk, which has resident KV it must attend. So a continuation chunk
                // runs on the prefix-capable ladder instead, on the rung `PrefixLadder::rung_for`
                // picks for the rows it has left. Refuse rather than run a continuation chunk on a
                // prefix-free rung: that would silently drop every earlier token from the attention
                // and produce fluent nonsense.
                //
                // ⭐ THE RUNG IS CHOSEN ONCE, HERE, and carried: `prefix_rung` is the cap, the padded
                // write below, the session and the K/V extent's baked row count. Those used to read the ONE
                // prefix bundle's width independently; with a ladder they must be one choice, or a chunk
                // could be placed for one width and proven against another.
                let use_prefix = chunk_start > 0;
                let prefix_rung = if use_prefix {
                    Some(prefill_prefix.rung_for(chunk_start, tok_end - tok).ok_or_else(|| {
                        werr(format!(
                            "superdsc batched prefill: chunk at start={chunk_start} needs a \
                             prefix-capable bundle, and none baked. Refusing — running it on a \
                             prefix-free rung would ignore the resident KV entirely."
                        ))
                    })?)
                } else {
                    None
                };
                let cap_m = prefix_rung.map_or(prefill_top_m, |r| r.width());
                // ⛔ THE CHUNK'S **PADDED** WRITE MUST STAY INSIDE ONE PAGE.
                //
                // A continuation chunk runs on the ONE prefix-capable bundle, so its cache write covers that
                // bundle's BAKED row count — 96 — however few real rows it has. 96 does not divide the
                // 256-slot page, so the third chunk of a prompt starts at offset 192 and writes through 287:
                // into the NEXT KV HEAD's plane. That is the long-context corruption (a correct first token,
                // then fluent garbage, for any prompt needing a third chunk).
                //
                // So the HOST moves the write slot: a chunk whose padded window would leave the page starts
                // at the next page boundary instead, and the skipped slots become a HOLE in this request's
                // `KvHistory` — which `record` already represents and the prefix mask already masks, because
                // the mask asks the history WHICH SLOTS IT HOLDS, not how many.
                //
                // ⛔ THIS REPLACED `PagedKvPool::WRITE_SLACK`, which solved the same overrun by widening every
                // plane so the padded tail had dead space to land in. MEASURED: that broke hd=128 — on
                // granite-3.1-8b it turned ` Rome. Q`/` Madrid. Q` into ` Romes,`/` Madridge.`, and forcing the
                // slack to 0 restored the 2b's exact outputs. The extra slots changed the feature-stick GROUP
                // stride every reader derives; choosing the write SLOT changes one number, on the host, at
                // every head dim.
                //
                // `chunk_room` still clips the REAL rows to the chunk's WINDOW: a window starting at a page
                // boundary may span `PREFILL_WINDOW_PAGES` whole pages (its cache write is one part per page,
                // each resolving its own page), and one starting mid-page stays inside that page, because a
                // part's rows past its page end would be written where nothing reads them back. A padded
                // window wider than a page always starts at a page boundary — `chunk_write_start` steps it
                // back there.
                let padded = scratchy_subtile::sdsc_abstract::SlotCount::new(if use_prefix {
                    cap_m as u32
                } else {
                    0
                });
                let want = scratchy_subtile::sdsc_abstract::KvSlot::new(chunk_start as u32);
                let placed =
                    scratchy_subtile::sdsc_abstract::PagedKvPool::chunk_write_start(want, padded);
                if placed.slot().get() as usize != chunk_start {
                    // ⭐ STEP BACK AND RE-PROCESS THE OVERLAP — MOVING BOTH CURSORS BY THE SAME AMOUNT.
                    //
                    // `back` slots of overlap are `back` tokens, because `chunk_start - tok` is invariant:
                    // whatever hole this request carries was accumulated BEFORE this region, and both
                    // cursors advance together inside it. So the chunk re-reads `back` tokens it already
                    // wrote and re-writes their keys at the same slots — idempotent, because the same
                    // tokens at the same positions produce the same K/V.
                    //
                    // ⛔ THE ALTERNATIVE — skipping FORWARD to the page boundary — IS RECORDED IN THIS FILE
                    // AS MEASURED-BAD ("desynchronised them and produced no output at all"), and re-adding
                    // it turned clean 2167-token prompts into garbage. Stepping back keeps the two cursors
                    // in lockstep, which is the property the whole loop rests on.
                    let back = chunk_start - placed.slot().get() as usize;
                    tracing::debug!(
                        "superdsc prefill: a chunk at slot {} would write {} padded rows past its page — \
                         starting {} slot(s) earlier at {} and re-processing the overlap",
                        chunk_start,
                        padded.get(),
                        back,
                        placed.slot().get(),
                    );
                    chunk_start = placed.slot().get() as usize;
                    tok -= back;
                }
                let room = scratchy_subtile::sdsc_abstract::PagedKvPool::chunk_room(placed.slot())
                    .get() as usize;
                let chunk_len = (tok_end - tok).min(cap_m).min(room);
                let (rung_m, pf, ri, nrungs) = if let Some(r) = prefix_rung {
                    // The rung `cap_m` was read from — the same value, not a second choice.
                    (r.width(), prefill_prefix.session(r), usize::MAX, 1usize)
                } else {
                    // NOTE: pending rungs are built AFTER this chunk is served, not before it (see
                    // the end of the chunk loop). Building on first touch put the 302 ms rung build
                    // inside TTFT — measured 120 ms -> 312 ms — which is a far worse trade than the
                    // startup time it saves. Selection here only ever sees BUILT rungs, so a chunk on a
                    // not-yet-built width runs one step wider and pays a little padding instead.
                    let ri = prefill_rung_for(prefill_rungs, chunk_len).ok_or_else(|| {
                        werr(format!(
                            "superdsc batched prefill: no rung for chunk_len={chunk_len} \
                             (ladder is non-empty here, so this is a selection bug)"
                        ))
                    })?;
                    let n = prefill_rungs.len();
                    let (m, sess) = &mut prefill_rungs[ri];
                    (*m, sess, ri, n)
                };
                // ⛔ THE CHUNK'S K/V EXTENT, PROVED BEFORE ANYTHING IS WRITTEN. The launch runs at the
                // BUNDLE's `rung_m` and the history advances by the CHUNK's `chunk_len`; when those differ
                // the padding rows are written as K/V and never counted, so they stay inside the attended
                // history and the request answers with one EOS token — empty text, `status: 200`, no error.
                // See `ChunkKvExtent`: one padding row is enough, measured.
                //
                // 🛑 REFUSING IS THE POINT. Loud and incomplete beats quiet and wrong, which is already
                // this file's rule for the fold's pass count; the same reasoning applies to a chunk that
                // cannot state its own extent. A continuation chunk gets `rung_m` from the ONE prefix
                // bundle regardless of its length, so it is the case that splits first.
                // ⭐ PADDING IS SAFE **ONLY PAST THE PROMPT'S END**, and that is the whole invariant.
                //
                // `decode_prefix_col_valid(col, p) = col < p`: decode marks valid only the columns BELOW its
                // write slot. So padding rows that land at slots `>= real_total` are masked out, and decode
                // overwrites each one before it ever reaches it — self-healing. Padding rows landing INSIDE
                // `[0, real_total)` are a different animal: a later chunk counts those slots as history, the
                // mask marks them valid, and the row attends K/V computed from padding.
                //
                // Which case a chunk is in is decided by arithmetic, not luck: `chunk_write_start` steps BACK
                // so the full `cap_m`-row padded write fits inside its window (one page, or whole pages from
                // a page boundary), hence `room >= cap_m` after it, hence `chunk_len < rung_m` ONLY when
                // `real_total - off` is — the LAST chunk. So the tail pads harmlessly and every mid-prompt
                // chunk is exact.
                //
                // 🛑 A STRICTER VERSION OF THIS GUARD SHIPPED AND WAS A REGRESSION. It demanded
                // `chunk_len == rung_m` for EVERY chunk, which refused the final chunk of every long prompt
                // — turning a working 1986-token request into an empty answer, and a batch that served 2 of
                // 4 into one that served none. Being loud is only right when the thing you are loud about is
                // actually wrong.
                let is_final_chunk = tok + chunk_len == tok_end;
                if !is_final_chunk {
                    let _exact = scratchy_subtile::sdsc_abstract::ChunkKvExtent::exact(
                        chunk_len as u32,
                        rung_m as u32,
                    )
                    .ok_or_else(|| {
                        werr(format!(
                            "superdsc batched prefill: a MID-PROMPT chunk_len={chunk_len} on a \
                             prefill_m={rung_m} bundle would write {} K/V row(s) the history never counts \
                             (prefix={use_prefix}, start={chunk_start}, real_total={real_total}). A later \
                             chunk counts those slots as history and the mask marks them valid, so the \
                             request would attend K/V computed from PADDING. The step-back in \
                             `chunk_write_start` is supposed to make this unreachable — if it fires, that \
                             placement is wrong, not this chunk.",
                            scratchy_subtile::sdsc_abstract::ChunkKvExtent::unaccounted_rows(
                                chunk_len as u32,
                                rung_m as u32
                            ),
                        ))
                    })?;
                }
                // Whichever bundle this chunk runs on, it is its OWN session: give it the request's
                // pages before it writes any K/V, or it writes them into whatever its create-time
                // map points at. EVERY page the chunk writes, not only the one holding the history's
                // next slot: a window wider than a page writes one part per page, each aimed at its own
                // page through this map.
                bind_chunk_pages(
                    pf,
                    req,
                    placed.wrote(scratchy_subtile::sdsc_abstract::SlotCount::new(
                        chunk_len as u32,
                    )),
                    ctx,
                )?;
                // The chunk's tokens, at ABSOLUTE positions. `?` on the bound rather than an index panic:
                // a chunk past the request's own tokens is a wiring bug and should say so.
                let chunk_toks: Vec<usize> = (tok..tok + chunk_len)
                    .map(|p| {
                        tokens.token_at(id, p).map(|t| t as usize).ok_or_else(|| {
                            werr(format!(
                                "superdsc batched prefill: chunk wants tokens [{tok}, {}) but this \
                                 request has {} — the token cursor and the write slot have come apart",
                                tok + chunk_len,
                                tokens.token_count(id),
                            ))
                        })
                    })
                    .collect::<ExecutorResult<Vec<usize>>>()?;
                // ⭐⭐⭐⭐⭐ THE BODY THIS CHUNK RUNS, ASKED BEFORE THE LAUNCH. Past one page of prefix the
                // prompt chunk's body is a split-KV chunk — every resident page attended in ONE launch —
                // so the index table and the gather answer must be the RUNNING body's, the solo decode's
                // own law. `step_body(start)` is the selection the launch makes (`valid_len = start + 1`),
                // and the chunk must hold the `start` resident keys; this chunk's own rows, written
                // before attention, sit at slots `>= start`, which the broadcast validity row masks.
                let step = pf.step_body(chunk_start).map_err(|e| {
                    werr(format!(
                        "prompt chunk at {chunk_start}: the session could not say which body runs: {e}"
                    ))
                })?;
                let kv_blocks = chunk_index_table(
                    "prompt chunk",
                    pool,
                    pf,
                    step,
                    req,
                    ctx,
                    ChunkKeys {
                        resident: chunk_start,
                        mapped: chunk_start as u32,
                        rows: scratchy_subtile::sdsc_abstract::ChunkRowKind::Prompt,
                    },
                )?;
                // The chunk that FINISHES the prompt runs its suffix — the rung's own
                // minted sampler picks the first generated token on the card
                // (`LogitsWanted::LastRow`), and the idx stick is read back exactly
                // like a decode launch's. Every other chunk's suffix never runs
                // (`LogitsWanted::None`) and reads nothing back.
                let is_last_chunk = tok + chunk_len == tok_end;
                let is_prompt_final_chunk =
                    is_last_chunk && tok_end >= tokens.prompt_len(id);
                let chunk_pick = run_prefill_batch(
                    // A prompt chunk: these rows are consecutive positions of one sequence, so they
                    // share both the causal triangle and one resident-prefix validity row. `None` IS
                    // that statement — there are no per-request rows, so the chunk laws are the only
                    // ones reachable.
                    None,
                    // The chunk's index table — `None` for a body that gathers nothing (no prefix, or
                    // a prefix within one page on the per-page body).
                    kv_blocks.as_deref(),
                    sh,
                    pf,
                    pmeta,
                    // ⭐ FROM THE REQUEST, AT ABSOLUTE POSITIONS — not from this step's window. A chunk
                    // that stepped back needs tokens an earlier step was given, and the request holds them.
                    scratchy_subtile::sdsc_abstract::SeedSource::Host(&chunk_toks),
                    // The LADDER entry's baked width — the rung this chunk actually runs on, of
                    // which `chunk_len` rows carry real tokens.
                    scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(rung_m as u32)
                        .ok_or_else(|| {
                            werr(
                                "superdsc batched prefill: the selected rung has zero rows"
                                    .to_string(),
                            )
                        })?,
                    // ⛔ THE RUNNING BODY'S OWN ANSWER, not the bundle's: the prefix bundle's ladder
                    // mixes the per-page body (gathers nothing) with split-KV chunks (gather every
                    // page), and its `KV_BLOCK_INDEX` placement is the chunk ceiling's.
                    step.gathers(),
                    chunk_start,
                    if is_prompt_final_chunk {
                        LogitsWanted::LastRow
                    } else {
                        LogitsWanted::None
                    },
                    // ⭐ THE CHUNK'S SEAT LAW — this request's draw, every row of
                    // it (the staging arm clamps pad rows to the last real
                    // position). Only the prompt-FINAL chunk's value is ever
                    // staged; a middle chunk's suffix never runs, so its law is
                    // inert — but it is still THIS request's, never invented.
                    SeatLaws::Prompt {
                        draw: req.draw.as_ref(),
                    },
                )?;
                // A prompt chunk never defers: it asked for `LastRow`/`None`, the two arms
                // that return `Done`.
                prompt_pick = match chunk_pick {
                    BatchOut::Done(p) => p,
                    BatchOut::Deferred(_) => {
                        return Err(werr(
                            "a prompt chunk deferred its read — only a decode batch can".to_string(),
                        ))
                    }
                };
                // UNCONDITIONAL truth-signal per chunk: `chunk_len` prompt tokens went through ONE
                // mq=N forward. This is the ONLY line that lets verify_prefill_real.sh print
                // PREFILL_REAL=PASS. `prefill_m` is the rung actually RUN (not the ceiling), so the
                // padding this chunk paid for is `prefill_m - m_used`.
                // ⭐ `req_id` IS PART OF THE TRUTH-SIGNAL, not decoration. Without it these lines cannot be
                // attributed to a request, and a 50-line log of a 4-request batch then supports whatever
                // correlation the reader is hunting: I read the first four lines of one, took them for the
                // four requests' first chunks, and reported a padded-chunk correlation with the two
                // requests that collapsed. The real log was 46 exact and 4 padded. An instrument that
                // cannot say WHOSE chunk it describes is an invitation to that mistake.
                // One per PROMPT CHUNK, so `debug!` rather than stderr: diagnostic, not a
                // condition anyone must see on every request.
                tracing::debug!(
                    // ⭐ `tok`/`tok_end` ARE ON THIS LINE BECAUSE THEIR ABSENCE COST AN EVENING. A
                    // 2599-token prompt stops being prefilled at 2560 and the sampler then finds no token
                    // (`token index 2599 out of range`), and with only `start=` (a SLOT) on the line it is
                    // impossible to tell whether the loop ran out of TOKENS or stopped early — the two
                    // readings need `tok_end`, which is the one number the line did not carry.
                    "PREFILL_PATH=batched keys_to={} m_used={chunk_len} prefill_m={rung_m} \
                     rung={} top_m={prefill_top_m} prefix={} start={chunk_start} tok={tok} \
                     tok_end={tok_end} real_total={real_total}",
                    // ⛔ THIS WAS `row=`, THE KV ROW — the one identity in scope, and it is gone with the
                    // rows. `ReqState` still carries no id (the id is the map key three frames up), so what
                    // replaces it is the slot its own keys reach: distinct per request in a RAGGED batch,
                    // which is the case these traces are read for, and it also shows the shared write slot
                    // running ahead of a short request — which is what `row=` was used to spot today
                    // (`start=768 ... real_total=140`). Two requests of EQUAL length are indistinguishable
                    // here, and equal lengths cannot see the bugs this line exists for.
                    req.kv_hist.end().get(),
                    if ri == usize::MAX {
                        "PREFIX".to_string()
                    } else {
                        format!("{ri}/{nrungs}")
                    },
                    use_prefix,
                );
                // THIS CHUNK'S `chunk_len` KV SLOTS ARE NOW THIS REQUEST'S, recorded before the next
                // chunk binds — `bind_request_pages` takes the write slot from the history, so a chunk
                // that did not record would send the next one back to write over its own keys. (It used
                // to bind at the request's token count, which this function only advances after the loop,
                // so a multi-chunk prompt wrote every chunk at the same slot.)
                //
                // ⭐ AT `placed`, NOT AT THIS REQUEST'S END. A chunk that stepped back writes
                // `[placed, placed + chunk_len)`; recording `chunk_len` slots from the END instead marks
                // the last `back` of them valid although this chunk never wrote them, and the mask is
                // built from the history. MEASURED at `live=1`: 450-token prompt, history 514 vs position
                // 450; 731-token prompt, history 827 vs position 731. `ChunkStart::wrote` is now the only
                // way to name the run, so the record moves with the step-back by construction.
                req.record_chunk(
                    placed.wrote(scratchy_subtile::sdsc_abstract::SlotCount::new(
                        chunk_len as u32,
                    )),
                );
                chunk_start += chunk_len;
                tok += chunk_len;
                // ⛔⛔⛔ NO RUNG IS BUILT HERE ANY MORE, AND NONE EVER WAS.
                //
                // This block built a "pending" ladder rung after serving a chunk, so the 302 ms rung build
                // landed after the response instead of inside TTFT. It could never run: `prefill_pending`
                // was constructed `Vec::new()` at the ONE site that makes a `SuperDscBundle` and nothing
                // ever pushed to it, while the log shows the whole ladder built at load
                // (`prefill ladder: 20 rung(s) ready in 6.19s wall — built in parallel`). Dead code whose
                // DOC described the opposite of what the binary did.
                //
                // ⭐ AND IT WAS NOT HARMLESS: creating a session mid-run is exactly what `runtime::Stream::create`
                // aborts on when it races (see its ⛔ doc), so a state that cannot occur was
                // also the state a reader would blame for that abort — which is precisely what I did for an
                // hour. A dead path is a lie about what the program can do.
            }
            // NO KV HANDOFF: the prefill session's seg2 is ALIASED onto the decode session's at
            // startup (see alias_seg_from where the sessions are built), so every chunk above wrote
            // the prompt's K/V straight into the region decode reads. This used to be
            // `pf.read_seg(2)` -> `session.write_seg(2, &kv)`, a whole-segment round-trip costing
            // 188.7 MB of DMA plus two 90 MiB host memcpys and a 90 MiB zeroed Vec, all serialized
            // around a synchronize() -- the largest single item in TTFT and independent of prompt
            // length.
            // ⛔ A TOKEN COUNT, NOT A SLOT. This was `chunk_start` — the KV write slot the loop ended on —
            // which equals the token count only for a request with no hole. With one, the slot runs ahead,
            // `n_computed` claims tokens the request does not have, and the next decode step asks for a
            // token past the end (measured: `decode batch: req … has no token at 2167`).
            tokens.set_tokens_in_pool(id, tok);
            // A prompt-FINAL forward returns the final chunk's on-card pick; a
            // mid-prompt wave picks nothing — its tail belongs to a later wave —
            // and still returns Prompt.
            return Ok(match prompt_pick {
                Some(ts) => ChunkOut::Tokens(ts),
                None => ChunkOut::Prompt,
            });
        }
    }
    if n_entry > 1 {
        // Only reachable when there's no prefill bundle at all: chunking covers any prompt length, and
        // a prefill that DOES run returns its logits above rather than falling through to here.
        eprintln!(
            "PREFILL_PATH=refused reason=no-prefill-bundle n={n_entry} start={start} prefill_top_m={prefill_top_m}"
        );
        return Err(werr(format!(
            "batched prefill did not run for this {n_entry}-token prompt (no prefill bundle) — the \
             sequential mq=1 prefill path is REMOVED; refusing to serve. Real batched prefill only."
        )));
    }
    // KV-residency invariant: the device-resident cache holds exactly `start`
    // positions; this forward appends contiguously starting at `start`.
    if tokens.tokens_in_pool_for(id) != start {
        return Err(werr(format!(
            "superdsc: resident cache holds {} positions but forwarding from {start} — KV desync \
             (positions must be contiguous per request)",
            tokens.tokens_in_pool_for(id)
        )));
    }
    // PAGED: point this session at the request's pages before anything launches. Omitting it runs
    // the forward against whichever request went last.
    bind_request_pages(session, req, ctx)?;
    // The ceiling is what THIS request's pages hold — an allocator decision, not a baked constant.
    let req_cap = if session.is_paged() {
        session.context_for_pages(ctx.host.len())
    } else {
        cap
    };
    if start + n > req_cap {
        return Err(werr(format!(
            "superdsc: context {start}+{n} exceeds this request's {req_cap} positions ({} page(s) \
             x {} slots) — the scheduler must allocate it another block",
            ctx.host.len(),
            session.page_slots.max(1)
        )));
    }

    // ON-CARD WEIGHT ORACLE: read a few resident weight tids back (SEN169→f32) and print
    // rms/max — compare to the disk [host-weight] dump (load_weights). rms/max is layout-
    // invariant (re-tile is a permutation), so a mismatch ⇒ the bf16→SEN169 conversion or
    // re-tile CORRUPTS the weight magnitude (a prime wrong-content-direction suspect the
    // synthetic F32 selftest can't catch). Match ⇒ conversion magnitude OK → bug is op/perm.
    // Per-position rotary row tiled across heads to a `[1, width]` source.

    // ⭐ THE TOKENS THE CARD PICKS, one per decode launch below — the minted
    // sampler's idx stick read back raw. There is no logits row, no host argmax.
    let mut picked: Vec<u32> = Vec::with_capacity(n);
    // The step's own timing line, shared by every readback so the step is visible in
    // the same instrument — print-only (`flags().timing`), changes nothing computed.
    let step_timing = |pos: usize, t: std::time::Instant| {
        if flags().timing {
            let rss_gb = std::fs::read_to_string("/proc/self/statm")
                .ok()
                .and_then(|s| {
                    s.split_whitespace()
                        .nth(1)
                        .and_then(|p| p.parse::<f64>().ok())
                })
                .map(|pages| pages * 4096.0 / 1e9)
                .unwrap_or(0.0);
            tracing::debug!(
                "[timing] forward pos={pos} took {:.2}s rss={rss_gb:.2}GB",
                t.elapsed().as_secs_f64()
            );
        }
    };
    for (i, &tok) in toks.iter().enumerate() {
        if tok >= vocab {
            return Err(werr(format!("token {tok} >= vocab {vocab}")));
        }
        let p = start + i; // absolute position = valid prefix count this step

        // ⭐ THE FORWARD, AS A TAPE. What used to be ~110 lines of interleaved
        // value-computation and `acts.push` is now: name the shape (which
        // kernels this bundle needs), then play it. The kernels live in
        // `forward_tape`; the launcher above supplies their INPUTS and does the
        // h2d. Order is the tape's, and it is pinned by
        // `a_forward_binds_then_launches_exactly_once`.
        //
        // Note what disappeared with the `if let Some(mask_cap)` guard: a bake
        // fact no longer BRANCHES a bind, it decides whether the step exists.
        // ⭐ ONE FILL SITE, and it is the same call prefill makes. This list used to be written
        // out by hand in BOTH binders and the two drifted twice — `IDENTITY_TID` and
        // `ATTN_ZERO_TID` were each placed, gated correctly, and then never filled on the decode
        // path. Both found only on card.
        let consts = scratchy_target_spyre::wiring::synthetic_constants(
            // ⛔⛔⛔ ONE ROW — THE LAUNCH'S, NOT THE BUNDLE'S CAPACITY.
            //
            // `synthetic_constants` derives `reductions = rows > 1`, which decides whether
            // `RMS_INVCOLS` and `ONES_REDUCE` are bound at all. Passing `b.m_cap` passed the
            // bundle's BAKED row count, and granite's decode bundle is baked at 96 rows (a batched
            // decode rung) — so `rows > 1` was TRUE on the decode path, and this bound two reduce
            // constants that the pre-tape decode binder never bound.
            //
            // The commit that introduced it argued the row count was the honest replacement for a
            // hand-passed `reductions: bool` phase flag, and asserted in its own comment that
            // decode's baked count is "1 here". It is 96. The row count IS the right input; the
            // bundle's capacity is not the row count. Identical mistake to the one fixed for
            // `ForwardShape::rows` — made twice, in the same file, from the same wrong assumption.
            &b.facts
                .constant_env(b.wiring, 64, scratchy_target_spyre::wiring::StagedRows::ONE),
        );
        // ⛔ BUILT BEFORE THE SHAPE, so the step gate below can BE this value rather than a second
        // spelling of it. See the prefill path's note: a step with no source refuses loudly, but a
        // source with no step is a mask computed and never uploaded — read as all-zero, i.e. every
        // column valid, which is fluent and wrong.
        //
        // NOT `0..p`: a request that has shared a batched step carries a HOLE, and calling that
        // hole valid feeds it another request's keys.
        let prefix_src = session.pmask_slots.map(|cap| {
            scratchy_target_spyre::forward_tape::PrefixSource::Broadcast {
                cap,
                holds: |col: usize| req.kv_hist.contains(col),
            }
        });
        // The body this launch RUNS: `step_body(p)` is the same selection `run_step(1, p, ..)` makes
        // (same session, same `p`).
        let step = session.step_body(p).map_err(|e| {
            werr(format!(
                "solo decode: the session could not say which body pos {p} runs: {e}"
            ))
        })?;
        // ⭐ ONE ROW, and the SAME call prefill makes. Decode is the batched form at `rows == 1`;
        // the lemmas in `forward_tape`'s tests are what make that exact rather than approximate.
        let shape = b.wiring.forward_shape(
            scratchy_target_spyre::wiring::StagedRows::ONE,
            prefix_src.is_some(),
            // ⛔ THE RUNNING BODY'S OWN ANSWER — not "solo decode never gathers", and not the
            // bundle's. The one-request ladder mixes bodies under one bundle: the per-page bodies
            // within one page gather nothing, the split-KV chunks past it gather every page, and
            // the bundle's `KV_BLOCK_INDEX` placement is the chunk ceiling's. Asked of the bundle,
            // a one-page context demanded a table its body never reads and the step refused.
            step.gathers(),
            consts.len(),
        );
        // fp16-safe attention "-inf": half the largest finite fp16 magnitude, negated —
        // spec-derived from `half::f16::MAX`, so `exp(score + mask)` underflows to 0 and stays
        // finite.
        let mask_neg = -(half::f16::MAX.to_f32() / 2.0);
        // The new token's causal block: the one-row, one-real-row prompt-chunk key — row 0 attends
        // column 0, the same triu law a prompt chunk takes per row. One key for every solo step, so
        // the mask is bound once per session and never again.
        let causal = scratchy_subtile::sdsc_abstract::CausalKey::chunk(
            scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(1).expect("1 is a width"),
            1,
        )
        .expect("one real row of a one-row rung");
        // ⭐⭐⭐⭐⭐ THE SPLIT-KV CHUNK'S INDEX — the one-request body attends EVERY resident page in one
        // pass (see `chunk_index_table`): positions `0..p` are resident, and the map holds the new
        // token's page too.
        let kv_blocks = chunk_index_table(
            "solo decode",
            pool,
            session,
            step,
            req,
            ctx,
            ChunkKeys {
                resident: p,
                mapped: p as u32 + 1,
                rows: scratchy_subtile::sdsc_abstract::ChunkRowKind::OneRequest,
            },
        )?;
        let inputs = scratchy_target_spyre::forward_tape::ForwardInputs {
            hidden: h,
            // ⭐⭐⭐⭐⭐ THE BAKED HEAD COUNT — the causal tile is `nqh` rows at mq==1, and
            // `h / hd` is that count only for a square Q projection. Same defect class the
            // prefill path recorded: a short tile leaves the upper head rows of the ADDITIVE
            // mask unstaged, which reads as ZERO — no mask. `b` is the decode meta this step
            // launches, so its wiring's geometry is the count the placement was sized by.
            num_q_heads: b.wiring.geometry.nqh as usize,
            mq_pad: 64,
            // One row: always the sub-blocked form (`NewBlockForm::of(1, …)`), whose block is every
            // head — the SAME baked count as `num_q_heads` above, never the `h / hd` quotient (that is
            // the head count only for a square Q projection; see the site above).
            causal_heads_per_block: b.wiring.causal_heads_per_block(causal),
            embed_tokens: sh.embed_tokens,
            tokens: &[tok],
            // THE SOLO STEP SEEDS EVERY TIME, and that is the solo driver's own law: it collects
            // in-call (the pick is in hand before the next step is built), and its session is
            // chosen per CAP BUCKET — a mid-generation bucket switch lands on a session whose
            // sticks were never seeded, so "first step on this session" is not a fact the caller
            // can name cheaply. The bind is the idempotent restatement of the card's own stage
            // write when the session did chain, and the required first seed when it did not.
            chain_seeds: Some(&[tok]),
            positions: &[p as u32],
            causal,
            prefix: prefix_src,
            consts: &consts,
            mask_neg,
            // Proven to be the identity at one row, so this is decode's existing row-major
            // staging spelled as the general form.
            stick_major: true,
            // ⭐ THE GATHER'S TABLE, from this request's own installed page map — one row, one
            // page per pass, the reference's walk. `None` when the bundle places no index, which
            // `kernel_values` turns into a REFUSAL if the tape carries the step — never into an
            // empty buffer, because an unstaged index reads as block 0 and block 0 is a real
            // address (row 0's first page), so every pass would answer from row 0's keys.
            kv_blocks: kv_blocks.as_deref(),
        };
        let fsteps = shape.steps();
        let fops = scratchy_target_spyre::forward_tape::operands(&fsteps);
        let ftape = scratchy_target_spyre::forward_tape::tape_steps(&fsteps, &fops);
        let mut launcher = ForwardLauncher {
            shape: &shape,
            steps: &fsteps,
            inputs: &inputs,
            staged: None,
            acts: Vec::with_capacity(fsteps.len()),
            acts_f16: Vec::new(),
            acts_i32: Vec::new(),
            causal_held: session.keyed_holds(scratchy_target_spyre::bundle_code::PlaceId::Act(
                scratchy_target_spyre::lower_subtile_tape_to_superdsc::ATTN_CAUSAL_TID,
            )),
            causal_keyed: None,
            launched: false,
        };
        scratchy_subtile::host_tape::play(
            &scratchy_subtile::host_tape::HostTape { steps: &ftape },
            &mut launcher,
        )
        .map_err(|e| werr(format!("forward tape: {e}")))?;
        // ⛔ EVERY STAGED LIST IS NAMED, no `..`: this site took `launcher.acts` alone and called the
        // f32-only `run_step`, so the chunk gather's int32 index table (`acts_i32`) was staged and then
        // DROPPED — the device read an all-zero index and every gathered block was block 0 (measured on
        // granite-3.1-2b fp8: 64 pages of every kv head byte-identical). A new staged list is now a
        // missing field here, and an unbound one is an unused binding.
        let ForwardLauncher {
            shape: _,
            steps: _,
            inputs: _,
            staged: _,
            acts,
            acts_f16,
            mut acts_i32,
            causal_held: _,
            causal_keyed,
            launched,
        } = launcher;
        debug_assert!(launched, "the tape must reach its device step");
        if let Some((id, key, vals)) = causal_keyed {
            session
                .bind_keyed(id, key, &vals)
                .map_err(|e| werr(format!("decode: keyed causal bind: {e}")))?;
        }

        // INCREMENTAL: write 1 new token at absolute slot `p`; KV stays device-resident.
        let _t_fwd = std::time::Instant::now();
        log_binds(&format!("decode pos={p}"), b.wiring, &acts, &acts_f16);
        // ⭐ EVERY DECODE LAUNCH BINDS ITS SAMPLER SEAT — the mint's noise/inv_t
        // sources are per-session seg3 placements no alias covers, so an unbound
        // seat reads ZEROS, which under the greedy law is exactly right. This
        // launch's row 0 is the request itself: its law is the request's own
        // draw at THIS absolute position (the step salt), and any further baked
        // row would be padding this single-request session never lays — greedy,
        // like every padding row. A decode bundle with no seat cannot pick:
        // refuse, never fall back to a host readback.
        let seat = b.wiring.sample_seat.as_ref().ok_or_else(|| {
            werr(format!(
                "decode bundle has no sampler seat (result t{}) — a decode launch cannot pick \
                 without the minted draw",
                b.result_id
            ))
        })?;
        let mut solo_laws = Vec::with_capacity(seat.rows as usize);
        solo_laws.push(RowLaw {
            draw: req.draw.as_ref(),
            step: p as u32,
        });
        // The rung's further baked rows are PADDING this single-request session
        // never lays — LIVE 0'S LAW, the pad clamp ([`pad_law_of`]): their
        // forward is row 0's by the replica law, so their seat words and their
        // ChainStage decompose must be row 0's too.
        for _ in 1..seat.rows as usize {
            solo_laws.push(pad_law_of(&solo_laws[0]));
        }
        acts_i32.extend(sample_binds(seat, &solo_laws)?);
        {
            // ⭐ THE ONE DECODE PATH, SUBMITTED: the whole result stays ON THE CARD
            // (the minted sampler consumed the logits there and left the idx stick in
            // their place), the launch is submitted with no trailing drain, and the
            // ONE raw idx readback is deferred and collected through the same door the
            // batched path uses — no logits D2H, no host argmax, no oracle, and no
            // second, synchronous decode arm.
            session
                .stage_step_acts(&acts, &acts_f16, &acts_i32)
                .map_err(|e| werr(format!("superdsc decode stage (pos {p}): {e}")))?;
            session
                .submit_step(1, p)
                .map_err(|e| werr(format!("superdsc decode submit (pos {p}): {e}")))?;
            let read = session
                .defer_read_tensor_ring(
                    scratchy_target_spyre::bundle_code::PlaceId::Act(b.result_id as u32),
                    64 * 4,
                )
                .map_err(|e| werr(format!("superdsc decode idx read (pos {p}): {e}")))?;
            step_timing(p, _t_fwd);
            let sub = SubmittedDecode {
                read,
                rows: 1,
                result_id: b.result_id as u32,
                vocab,
            };
            // This launch's row is 0 (one request, its own session), so its token
            // is the idx stick's first row's lane 0.
            let toks = collect_decode_tokens(session, &sub)?;
            // ⭐ FRONT-B'S SEAT PROBE, KEPT ON THE SOLO PATH: the collect has
            // already paid this launch's one D2H fence, so the probe's own
            // debug reads add no cadence — and it never gates the run.
            log_seat_readback(
                session,
                seat,
                b.wiring.load_consts,
                &toks,
                &format!("solo pos={p}"),
            );
            picked.push(
                *toks
                    .first()
                    .ok_or_else(|| werr("the decode launch picked no token".to_string()))?,
            );
        }
    }
    tokens.set_tokens_in_pool(id, start + n);
    // The solo forward wrote its `n` keys at the slot `bind_request_pages` bound it at — this request's
    // own end, since nobody else was in the launch. Recorded here rather than per token because this
    // loop is only ever reached with n == 1 (a multi-token prompt returns from the batched path above).
    // ⭐ THE SLOT IS NAMED HERE, not derived inside a helper: this forward wrote at this request's own
    // end because nobody else was in the launch. A PREFILL chunk cannot say that (it may have stepped
    // back), which is why it goes through `record_chunk` instead.
    let at = scratchy_subtile::sdsc_abstract::BatchSlot::solo(&req.kv_hist);
    req.record_kv(
        at,
        scratchy_subtile::sdsc_abstract::SlotCount::new(n as u32),
    );
    // KV PROBE (SCRATCHY_KV_PROBE): dump decode's layer-0 K cache (t9, natural [nqh,cap,hd]) head-0
    // per-slot L2 norms for the first few slots. Compare BATCHED vs SEQUENTIAL (same prompt): the
    // sequential t9 is the golden KV. If the batched slot-0..(prompt-1) norms differ, the mq>1
    // prefill computed the wrong KV (H3); if they match, the KV is right and the bug is elsewhere.
    Ok(ChunkOut::Tokens(picked))
}

/// What a [`forward_chunk`] runs: one request's chunk, the tokens `toks` at absolute positions
/// `start..start + toks.len()`, with the request's state, its id, and the store its tokens and
/// token count live in.
pub(crate) struct Chunk<'a> {
    pub(crate) state: &'a mut ReqState,
    pub(crate) id: &'a str,
    pub(crate) tokens: &'a mut scratchy_serving_engine::input_batch::InputBatch,
    pub(crate) start: usize,
    pub(crate) toks: &'a [usize],
}

/// Run `b` ONCE for `toks` at absolute positions `start..start+toks.len()`
/// against `req`'s KV cache (m=toks.len() query rows in one forward). `start` is
/// the valid-prefix count AND the runtime `decode_position`. Appends the new
/// roped-K/V rows and returns the per-row logits `[n, vocab]`. n <= b.m_cap.
#[cfg(feature = "spyre-hw")]
pub(crate) fn forward_chunk(
    sh: &Shared<'_>,
    session: &mut SendnnSession,
    b: &BundleMeta,
    // The preserved prefill meta for the batched-prefill path, bound to the distinct prefill SESSION
    // by run_prefill_batch. None for every non-batched path (they use `b`, the decode meta).
    prefill_meta: Option<&BundleMeta>,
    Chunk {
        state: req,
        id,
        tokens,
        start,
        toks,
    }: Chunk<'_>,
    // The host's list + the pool's owner split: every bind DERIVES this request's page map, and the map's
    // fully-masked and hole pages come from the range the host is never told about.
    ctx: PageMapCtx<'_>,
    // The pool's geometry, threaded from the model — one spelling, handed to
    // `superdsc_forward_chunk`'s chunk paths (see `run_request_step`, where it is built).
    pool: scratchy_subtile::sdsc_abstract::PagedKvPool,
) -> ExecutorResult<ChunkOut> {
    {
        let SendnnSession::SuperDsc(sb) = session;
        // Split the borrow: the prompt runs through the m=N `prefill` session (if present),
        // then the last token + all generation run through the m=1 `decode` session.
        let SuperDscBundle {
            decode,
            prefill_rungs,
            prefill_prefix,
            // Named rather than `..` so adding a field keeps failing here until someone decides
            // whether this path needs it. The batch ladder is not needed: this is the ONE-request
            // path and a batched step never reaches it. The pool is not needed either: the request
            // already holds its pages and the forward only installs their map.
            decode_rungs: _,
            pool_pages: _,
            pool_rows: _,
        } = sb;
        return superdsc_forward_chunk(
            sh,
            decode,
            prefill_rungs,
            prefill_prefix,
            b,
            prefill_meta,
            req,
            id,
            tokens,
            start,
            toks,
            ctx,
            pool,
        );
    }
}

/// The KTIR-emulator twin of [`forward_chunk`]: one fused forward through the resident session,
/// with the prefix KV threaded from the host cache.
#[cfg(not(feature = "spyre-hw"))]
pub(crate) fn forward_chunk(
    sh: &Shared<'_>,
    session: &mut SpyreSession,
    b: &BundleMeta,
    prefill_meta: Option<&BundleMeta>,
    Chunk {
        state: req,
        id,
        tokens,
        start,
        toks,
    }: Chunk<'_>,
) -> ExecutorResult<ChunkOut> {
    let _ = prefill_meta; // the batched-prefill session pair is a sendnn-only shape
    let n = toks.len();
    // KTIR caps a forward at the baked m_cap. The sendnn paged prefill runs the
    // WHOLE prompt in one forward (s0 = padded bucket up to capacity), so its only
    // cap is the prefix capacity, checked below.
    {
        let mcap = b.m_cap;
        if n == 0 || n > mcap {
            return Err(werr(format!("forward_chunk: n={n} not in 1..={mcap}")));
        }
    }
    if start + n > b.capacity {
        return Err(werr(format!(
            "positions {start}..{} exceed bundle prefix capacity {} — re-emit with KTIR_PREFIX_LEN >= {}",
            start + n,
            b.capacity,
            start + n
        )));
    }
    // ⭐ NO `hd` HERE EITHER. The rotary tables are built per class (each cos/sin
    // source's own `RotaryKind`), so this function no longer reads a model-wide
    // head dim.
    let h = sh.hidden;
    // ⭐ A LAYER'S OWN WIDTH, NOT THE MODEL'S. A hybrid-attention arch (gemma-4) gives sliding
    // layers `num_kv*hd` and global layers `g_kv*g_hd` — different numbers — so every per-layer
    // quantity below (cache row count, prefix buffer size, readback count, extend length) reads
    // the layer's baked `kv_width`, never the model-wide `sh.kv_dim`.
    let have = req
        .kv_k
        .first()
        .zip(b.layers.first())
        .map_or(0, |(c, lw)| c.len() / lw.kv_width);
    if have != start {
        return Err(werr(format!(
            "KV cache desync: {have} cached rows but forwarding from position {start}"
        )));
    }
    // The number of activation ROWS the bound buffers carry: the baked `m_cap`. Each forward fills
    // n <= m_cap real rows; the remaining pad rows REPLICATE the last real token (see the staging
    // loop below — the lm-head tail reads the last BAKED row, which must carry the last real token).
    let rows = b.m_cap;

    // Build this pass's DYNAMIC sources (the static weights stay resident in the
    // session — no per-forward weight marshal). Embedding gather + per-position
    // RoPE tables, every row (real rows their own token/position, pad rows the
    // last real token's — see the staging loop).
    let mut emb = vec![0.0f32; rows * h];
    // ⭐ PER-CLASS rotary rows, computed once per position per CLASS. A hybrid-
    // attention arch (gemma-4) gives sliding and global layers DIFFERENT tables
    // (head dim, θ and partial-rotary all differ), and every cos/sin source
    // names its own class in the wiring — one table per class, then each source
    // tiles the one it carries. The wrong table is silent garbage (fluent,
    // wrong output), never a fault.
    let classes: Vec<scratchy_target_spyre::wiring::RotaryKind> = b
        .cos_srcs
        .iter()
        .map(|&(_, _, k)| k)
        .chain(b.sin_srcs.iter().map(|&(_, _, k)| k))
        .collect();
    // Per-class, per-position `[class_head_dim]` rows.
    //
    // ⭐ THE STAGING LOOP COVERS ALL `rows` (NOT JUST `toks.len()`), AND A PAD ROW
    // (i >= n) REPLICATES THE LAST REAL TOKEN — the same law the card path's
    // `ChunkRows::row_logical_pos` pins (Kani: `prefill_pad_row_holds_last_real_token`).
    // The emitter folds the vocab-wide lm-head tail to m=1 over the LAST BAKED row
    // (`selector_lastrow_col(m_cap)`), which is a PAD row whenever `n < m_cap`; if that
    // row is not numerically the last real token's forward, the first sampled token is
    // whatever the lm_head computes over a row the model never fed. The card path has
    // replicated since its own pad-row law landed; the emulator left pad rows ZERO.
    // MEASURED on gemma-4-26b-a4b-it fp8 (23-token prompt on the m=31 rung): first
    // sampled token EOS (id 106), `LAST_HIDDEN` ≈ final-norm row 0, pad-row logits 0.
    let mut cos_tab: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(rows); classes.len()];
    let mut sin_tab: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(rows); classes.len()];
    for i in 0..rows {
        // A pad row's TOKEN and ROTARY POSITION are the last real token's —
        // `min(i, n-1)` over tokens, `start + min(i, n-1)` over positions.
        let r = i.min(n - 1);
        let t = toks[r];
        if t >= sh.vocab {
            return Err(werr(format!("token {t} >= vocab {}", sh.vocab)));
        }
        emb[i * h..(i + 1) * h].copy_from_slice(&sh.embed_tokens[t * h..(t + 1) * h]);
        for (ci, &k) in classes.iter().enumerate() {
            let (c, s) = scratchy_target_spyre::manifest::rope_cos_sin_kind((start + r) as u32, k);
            cos_tab[ci].push(c);
            sin_tab[ci].push(s);
        }
    }
    // Tile a class's head_dim-wide rotary rows across heads to fill a
    // `[rows, width]` source. Every row carries its staged position (pad rows the
    // last real token's clamped one). `width` is a multiple of the class's own head dim.
    let tile =
        |rrows: &[Vec<f32>], width: usize, kind: scratchy_target_spyre::wiring::RotaryKind| {
            let chd = kind.head_dim() as usize;
            let mut buf = vec![0.0f32; rows * width];
            let reps = width / chd;
            for (i, row) in rrows.iter().enumerate() {
                for r in 0..reps {
                    let off = i * width + r * chd;
                    buf[off..off + chd].copy_from_slice(row);
                }
            }
            buf
        };
    // KTIR builds ONE dynamic-source list (embed + cos/sin + per-layer prefix-KV)
    // for the single fused forward. sendnn builds per-GROUP lists below (it threads
    // the hidden state group→group), reusing `emb`/`cos_tab`/`sin_tab` directly.
    let mut dynamic: Vec<(u64, Vec<f32>, Vec<usize>)> = {
        let mut dynamic =
            Vec::with_capacity(1 + b.cos_srcs.len() + b.sin_srcs.len() + 2 * b.layers.len());
        // ⭐ THE CHAIN REPLACES THE EMBED ROW PUSH here too — a chained wiring's
        // gather reads the two persistent sticks, so the host's contribution is
        // their seed: the LAST REAL token's decompose (the token this forward
        // feeds — the same value the card path's ChainSeed steps state, and
        // idempotent with the device's own ChainStage writes for the same
        // reason: step t feeds the token step t-1 picked). The sticks carry u32
        // lane values, so the seed rides this f32 list as BIT-PRESERVING WORDS
        // (`f32::from_bits`) — the same carrying the reference legs use — and
        // the gather's eval arm converts them back with `to_bits`.
        match b.phase {
            scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain } => {
                // ⭐ THE SEEDS ARE PER-REQUEST-ROW AT THE FORWARD'S OWN WIDTH —
                // the ONE per-row spelling (`chain_s_seed` / `chain_l_seed`,
                // the same fns the card path's ChainSeed tape arm calls): row i
                // carries ITS OWN token's decompose, pad rows repeat the last
                // real token's (the same clamp the embed fill above uses,
                // idempotent with the device's own ChainStage writes). The s
                // seed is `[rows, lanes]` — one whole pattern stick per row —
                // and the l seed is `[rows·l_sticks, lanes]`, row i's ramp
                // sticks then row i+1's, the placement shape the mint states.
                // At `rows == 1` both pushes are the solo spelling
                // byte-for-byte.
                let lanes = scratchy_subtile::subtile_ir::CHAIN_IDX_BLOCK as usize;
                let l_sticks =
                    scratchy_subtile::subtile_ir::chain_l_sticks(chain.hidden) as usize;
                let toks32: Vec<u32> = toks.iter().map(|&t| t as u32).collect();
                let s_words =
                    scratchy_subtile::subtile_ir::chain_s_seed(chain.k, &toks32, rows);
                let l_words =
                    scratchy_subtile::subtile_ir::chain_l_seed(&toks32, chain.hidden, rows);
                dynamic.push((
                    chain.s_src as u64,
                    s_words.iter().map(|w| f32::from_bits(*w)).collect(),
                    vec![rows, lanes],
                ));
                dynamic.push((
                    chain.l_src as u64,
                    l_words.iter().map(|w| f32::from_bits(*w)).collect(),
                    vec![rows * l_sticks, lanes],
                ));
            }
            scratchy_target_spyre::forward_tape::ForwardPhase::Prefill => {
                // The embedding is this chunk's rows of the hidden state.
                dynamic.push((b.embed_src as u64, emb, vec![rows, sh.hidden]));
            }
        }
        // ⭐ THE COMPILE-TIME SCALARS, AT THEIR RESERVED TIDS. The programs read each
        // model constant (a ScalarMul multiplier, an RMSNorm epsilon or divisor) and the algebraic
        // identities `0`/`1` from a bound `[1,1]` tile rather than a KTIR immediate, so that ONE
        // program serves both consumers: `dxp_standalone` has no immediate operand, and an
        // `arith.constant` the card cannot read is not a constant the card has. They are ordinary
        // `func.arguments` here, so leaving one unbound is a zero, and a zero in this position is
        // silent — `x/(1+e)` becomes `x/e` and every scaled residual vanishes.
        for (i, v) in b.scalarmul_scales.iter().enumerate() {
            let tid = scratchy_target_spyre::lower_subtile_tape_to_superdsc::scalarmul_scale_tid(i);
            dynamic.push((u64::from(tid), vec![*v], vec![1, 1]));
        }
        // Each rope table is tiled to the width its consumer reads, from THE
        // SOURCE'S OWN CLASS's rows. `classes` is cos-then-sin, so a sin
        // source's class index is offset past the cos sources.
        let n_cos = b.cos_srcs.len();
        for (ci, &(cid, w, kind)) in b.cos_srcs.iter().enumerate() {
            dynamic.push((cid as u64, tile(&cos_tab[ci], w, kind), vec![rows, w]));
        }
        for (ci, &(sid, w, kind)) in b.sin_srcs.iter().enumerate() {
            dynamic.push((
                sid as u64,
                tile(&sin_tab[n_cos + ci], w, kind),
                vec![rows, w],
            ));
        }
        dynamic
    };
    let vocab = sh.vocab;

    // KTIR: bind the per-layer prefix-KV cache sources (host-grown KV), then one
    // fused forward through the resident session.
    {
        let cap = b.capacity;
        for (li, lw) in b.layers.iter().enumerate() {
            let kvw = lw.kv_width;
            let mut kbuf = vec![0.0f32; cap * kvw];
            let mut vbuf = vec![0.0f32; cap * kvw];
            kbuf[..req.kv_k[li].len()].copy_from_slice(&req.kv_k[li]);
            vbuf[..req.kv_v[li].len()].copy_from_slice(&req.kv_v[li]);
            dynamic.push((lw.prefix_k_src as u64, kbuf, vec![cap, kvw]));
            dynamic.push((lw.prefix_v_src as u64, vbuf, vec![cap, kvw]));
        }
        // ⭐ THE LENGTH MASK IS FILLED BY WHOEVER KNOWS THE DECODE POSITION. The prefix cache tensor
        // spans the full structural capacity while only `start` of its rows are valid this step, so
        // the mask zeroes the valid columns and drives the rest to a large negative — they leave the
        // softmax through `exp(-inf) = 0`. It is a source like any other.
        if let Some(mask) = b.attn_mask_src {
            // ⭐ TWO MASKS, ONE SOURCE, TOLD APART BY THE BUNDLE'S OWN ROW COUNT. A `m_cap == 1`
            // bundle masks the resident PREFIX: `[1, capacity]`, valid up to `start`. A prompt
            // chunk (`m_cap > 1`) has no resident prefix — its bundle is baked `ActiveCap::NONE` —
            // and instead needs the `[mq, mq]` CAUSAL triangle over its own keys, which is what
            // lets the attention run all `mq` rows in one pass instead of unrolling them.
            //
            // ⭐ AND THE TRIANGLE IS TAKEN AT EACH ROW'S **CLAMPED** POSITION, NOT
            // THE PLAIN `attn_causal_mask_fill(mq)`. The emitter folds the
            // vocab-wide lm-head tail to m=1 over row `selector_lastrow_col(mq)` —
            // the LAST BAKED row, which is a PAD row whenever `n < m_cap`. A pad
            // row's forward must therefore be numerically IDENTICAL to the last
            // real token's (same token id, see the embed staging above; same rotary
            // position; same causal extent), or the first sampled token comes from
            // a row the model never fed. The card path has had this law since
            // `ChunkRows::row_logical_pos` (Kani-pinned
            // `prefill_pad_row_holds_last_real_token`); the emulator bound the RAW
            // triangle, whose pad rows attend at their PAD position's extent — a
            // row that is not any token's forward. MEASURED on gemma-4-26b-a4b-it
            // fp8 (23-token prompt on the m=31 rung): first sampled token was EOS
            // (id 106), `LAST_HIDDEN` ≈ final-norm ROW 0, pad-row logits 0.
            //
            // ⛔ NOT `CausalKey::mask`: that returns the CARD's
            // `[mq, mq_pad]` stick-padded layout (`mq_pad = mq.div_ceil(64)*64`),
            // and this view is `[mq, mq]` — a 31-row chunk would bind row stride
            // 64 into a 31-stride view and misalign every row past the first.
            // The clamp (`row_logical_pos`) is the law; the fill here composes it
            // with the same `prefill_causal_col_valid` SSOT at THIS bundle's own
            // width.
            let (fill, shape) = if b.m_cap > 1 {
                let mq = b.m_cap;
                let chunk = scratchy_subtile::sdsc_abstract::ChunkRows::chunk(
                    NonZeroU32::new(mq as u32).expect("m_cap > 1"),
                );
                let mut m = vec![0.0f32; mq * mq];
                for r in 0..mq {
                    let pos = chunk.row_logical_pos(r, n);
                    for c in 0..mq {
                        if !scratchy_subtile::sdsc_abstract::prefill_causal_col_valid(c, pos) {
                            m[r * mq + c] = -1.0e38;
                        }
                    }
                }
                (m, vec![mq, mq])
            } else {
                (
                    scratchy_target_spyre::manifest::attn_mask_fill(cap, start),
                    vec![1, cap],
                )
            };
            dynamic.push((mask as u64, fill, shape));
        }
        // ⭐ ASK FOR THE ROWS THIS FORWARD WROTE, NOT THE WHOLE RESIDENT TENSOR. Decode and prefill
        // share one session, so every tensor is resident at the WIDEST program's row count: a
        // one-row decode step had its outputs decoded at the prefill's `m`. The counts are the ones
        // the reads below already slice to — `vocab` of the result, `n * kv_width` of each layer's
        // new K/V (per LAYER: a hybrid arch's layers differ in width).
        let wanted: std::collections::HashMap<u64, usize> =
            std::iter::once((b.result_id as u64, vocab))
                .chain(b.layers.iter().flat_map(|lw| {
                    [
                        (lw.new_k_id as u64, n * lw.kv_width),
                        (lw.new_v_id as u64, n * lw.kv_width),
                    ]
                }))
                .collect();
        let outputs: Vec<(u64, usize)> = b
            .output_ids
            .iter()
            .map(|id| (*id as u64, wanted.get(&(*id as u64)).copied().unwrap_or(0)))
            .collect();
        let out = session
            .run_step(b.prog, dynamic, &outputs)
            .map_err(|e| werr(format!("run_step: {e}")))?;
        tokens.set_tokens_in_pool(id, start + n);
        for (li, lw) in b.layers.iter().enumerate() {
            let kvw = lw.kv_width;
            req.kv_k[li].extend_from_slice(&out[&(lw.new_k_id as u64)][..n * kvw]);
            req.kv_v[li].extend_from_slice(&out[&(lw.new_v_id as u64)][..n * kvw]);
        }
        // ⭐⭐⭐ ONE LOGITS ROW, AT ROW 0 — because that is what the bundle COMPUTES.
        //
        // A prefill bundle does not run its vocab-wide lm_head at `mq`. `lower_one_node`'s
        // `is_prefill_lm_head_tail` arm folds it to the m=1 tail it really is
        // (main's `lower_prefill_lm_head_at_m1`): the activation is sliced to the LAST prompt row
        // and the output is narrowed to row 0. Only that
        // row's logits are ever read, and computing all `mq` rows of a 128k-column output would be
        // `mq`× the work for one row of answer.
        //
        // ⛔ SO `(0..n)` WAS A CLAIM THE BUNDLE NEVER MADE. Rows 1..n are never written, and the
        // caller takes the LAST of what it gets (`next_back()`), so a prompt chunk with n > 1
        // sampled an unwritten row — zeros — and generation continued from a token the model never
        // chose. It coincided with row 0 only at n == 1, which is why decode looked correct and a
        // prompt that fit ONE chunk looked correct. MEASURED: the result tensor comes back with
        // ~one row of `vocab` nonzeros and every argmax below `vocab`, i.e. in row 0.
        //
        // ⭐ THE HOST PICK IS GONE, so this row is read by NOBODY: the emulator has
        // no minted sampler (the mint is a decode-bundle card fact), so an emulated
        // forward produces no token. The card is the only judge.
        Ok(ChunkOut::Prompt)
    }
}

/// Pick the smallest decode cap bucket whose capacity covers `need` positions.
/// `buckets` is ascending by capacity and non-empty; if `need` exceeds every
/// bucket, return the largest so the caller's `forward_chunk` raises the precise
/// over-capacity error (re-emit with a bigger `KTIR_PREFIX_LEN`).
pub(crate) fn select_decode(buckets: &[BundleMeta], need: usize) -> &BundleMeta {
    buckets
        .iter()
        .find(|b| b.capacity >= need)
        .unwrap_or_else(|| buckets.last().expect("at least one decode bucket"))
}

/// Forward `n` positions from `start`, then greedily sample iff this completes
/// the prompt or is a decode step. n==1 (generation) uses the cheap m=1 DECODE
/// bundle; n>1 (prompt) uses the batched m=M PREFILL bundle (chunked by m_cap),
/// falling back to the decode bundle if no prefill bundle was emitted.
pub(crate) fn run_request_step(
    model: &mut Loaded,
    req: &mut ReqState,
    // ⛔ THE TOKENS ARE READ AND WRITTEN THROUGH THE STORE, NOT THROUGH `req`. `InputBatch` is borrowed
    // live rather than snapshotted: a snapshot of a store is a COPY, and reading one taken upstream of
    // the writes for this step is exactly the defect that broke every prompt past one scheduler step
    // (`b4856997`). `id` is what indexes it.
    id: &str,
    tokens: &mut scratchy_serving_engine::input_batch::InputBatch,
    start: usize,
    n: usize,
    // The host's block list for this request, read from `InputBatch` by the caller, and the pool's owner
    // split. Passed in rather than looked up here so there is one read of the store per step.
    #[cfg(feature = "spyre-hw")] host: &[usize],
    #[cfg(feature = "spyre-hw")] part: PoolPartition,
) -> ExecutorResult<Option<u32>> {
    if n == 0 {
        return Ok(None);
    }
    // The map context for this request's whole step: the host's list (read from `InputBatch` by the
    // caller) and the pool's owner split, derived once.
    #[cfg(feature = "spyre-hw")]
    let ctx = PageMapCtx { host, part };
    let toks: Vec<usize> = (start..start + n)
        .map(|p| {
            tokens.token_at(id, p).map(|t| t as usize).ok_or_else(|| {
                werr(format!(
                    "token index {p} out of range (have {})",
                    tokens.token_count(id)
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    // Shared model state (immutable) borrowed disjointly from the chosen bundle.
    let sh = Shared {
        embed_tokens: &model.embed_tokens,
        hidden: model.hidden,
        vocab: model.vocab,
    };
    // ⭐ THE PAGED-KV POOL'S GEOMETRY, from the model's own two numbers — the same spelling the
    // batched-decode site reads (`spyre_exec`'s `PagedKvPool::new(model.kv_dim / model.head_dim, …)`),
    // threaded to the chunk paths because `Shared` deliberately carries no kv width: a hybrid-attention
    // arch has no single one, and the pool is the one lawful model-wide use (see `Loaded::kv_dim`).
    #[cfg(feature = "spyre-hw")]
    let pool = scratchy_subtile::sdsc_abstract::PagedKvPool::new(
        model.kv_dim / model.head_dim,
        model.head_dim,
    );
    // A decode step runs on the smallest cap bucket whose capacity covers the last position
    // this step reaches, so a short context pays cap-256 decode cost rather than max-cap.
    // `start + n` (not `pos + 1`) because `b` is fixed for the whole step below.
    let b: &BundleMeta = if n == 1 {
        select_decode(&model.decode, start + n)
    } else if let Some(p) = model.prefill.as_ref() {
        p
    } else {
        select_decode(&model.decode, start + n)
    };
    // OPT-IN (user-directed 2026-07-27): was unconditional on every single request -- pure log spam
    // for normal operation. Gate it behind SCRATCHY_SUPERDSC_DBG (the existing verbose-diagnostic
    // toggle this file already uses elsewhere) so it's there when actually debugging the prefill
    // path, silent otherwise.
    #[cfg(feature = "spyre-hw")]
    // KTIR forwards at most m_cap rows/chunk. The sendnn layer-group path runs the
    // whole scheduled chunk in one forward.
    #[cfg(feature = "spyre-hw")]
    let chunk_w = match &model.session {
        // SUPERDSC: with a batched-prefill session, feed the whole prompt in ONE chunk
        // (superdsc_forward_chunk runs all of it through the m=N prefill session, whose final chunk
        // also returns the first token's logits) — the TTFT win. Without a prefill session (or for the
        // generation phase, n==1), it's single-stream chunk size 1 (one token/step, device-resident
        // KV). A longer prompt splits into prefill_m+1 chunks; chunk>0 currently falls through to
        // sequential inside superdsc_forward_chunk (prefix attention is a follow-up).
        // FORCE sequential prefill (chunk_w=1): batched prefill DEADLOCKS on prompts longer than
        // prefill_m (refuses `chunk-start-nonzero-needs-prefix-attention`, then all threads futex_wait
        // forever). chunk_w is runtime prefill orchestration ONLY — it does NOT touch the SDSC emitter,
        // so the decode bundle stays byte-identical. n==1 per chunk ⇒ the batched branch never fires ⇒
        // pure token-by-token sequential, coherent for any prompt length. (The prefill bundle still
        // bakes + binds, unused — ~2× resident weight memory, harmless.)
        // Un-stubbed, but GATED: batch the prompt in one mq>1 forward ONLY when the baked prefill bundle
        // carries the matmul-by-ones reduce (model.batched_prefill.uses_ones_reduce). Without it the mq>1
        // rmsnorm infs (pod-B B6), so default builds stay chunk_w=1 = token-by-token sequential (coherent,
        // safe). The fp8-prefill bundle places ONES_REDUCE_TID (for the L2 amax) ⇒ uses_ones_reduce is set ⇒
        // this fires for fp8 automatically, no runtime env.
        SendnnSession::SuperDsc(sb) => {
            // Pass the WHOLE prompt through to superdsc_forward_chunk in ONE call whenever a
            // prefill session exists — it does its OWN internal <=prefill_m chunking now (see
            // its CHUNKED BATCHED PREFILL loop), covering any prompt length. This chunk_w used to
            // be capped at prefill_m+1, pre-splitting a longer prompt into 1-token pieces BEFORE
            // superdsc_forward_chunk ever saw more than one chunk's worth — making its internal
            // chunking loop unreachable dead code for exactly the prompts it was built to handle.
            // `sb.prefill.is_some()` IS the real "can we batch" signal (a distinct prefill session
            // exists at all). `uses_ones_reduce` used to double as a proxy for it (it happened to
            // correlate: both are true exactly when a real prefill bundle was baked), but it's an
            // accidental coupling to an unrelated flag (whether the RmsNorm all-ones weight got
            // placed) — not a real capability check, and a confusing thing to debug when it's the
            // gate that silently disables batching. Dropped; `c_prefill_session` alone is the
            // correct, direct condition.
            // ⛔⛔⛔ THERE ARE NO ENV GATES HERE ANY MORE, AND THAT IS THE POINT.
            //
            // This decision used to read TWO environment variables — `SCRATCHY_FORCE_SEQUENTIAL` (force
            // one token per forward) and `SCRATCHY_DISABLE_BATCHED_PREFILL` (opt out of batching) — each
            // added while batched prefill was unproven, and both left in after it was proven. An env var
            // that changes WHICH PROGRAM RUNS makes every measurement conditional on the environment it
            // was taken in, and this file already has the cautionary case: `SCRATCHY_LAYER_KSPLIT` was
            // read at bake AND at runtime, and setting it against a bundle baked without it emptied every
            // completion with no error line at all (`22a8f469`).
            //
            // The conditions that remain are the only real ones: a chunk is batched iff a bundle exists
            // that can run it. For a prompt STARTING at 0 that is any prefill rung; for a chunk starting
            // PAST 0 it is the prefix-capable bundle, which exists for exactly this case.
            //
            // ⛔⛔⛔ `start > 0` USED TO FORCE ONE TOKEN PER FORWARD, AND WITH PREFIX CACHING ON THAT IS
            // EVERY CACHE HIT. MEASURED 2026-08-13: a hit made the tail prefill run through the m=1 DECODE
            // bundle (`PREFILL_PATH=sequential-fallback start=768`) while the same prompt without a hit ran
            // the batched prefill — a DIFFERENT program, different geometry, different fp8 activation
            // scales. The reusing request's answer then differed from its own recompute, deterministically:
            // 3 caching-ON runs all said `Nariobi`, 2 caching-OFF runs both said `Nairobi`, and the
            // permutation test pinned it to the REUSER (put that request first, so it WRITES the shared
            // prefix instead of reusing it, and it answers `Nairobi` correctly while whoever now reuses
            // degrades instead).
            //
            // ⭐ AND THE TAIL'S PADDING IS SAFE, BY THE INVARIANT THIS FILE ALREADY RELIES ON: padding rows
            // are harmless PAST THE PROMPT'S END (decode marks valid only columns below its write slot, and
            // overwrites each padded slot before it ever reaches it), which is exactly where a last chunk's
            // padding lands. Every cold prompt's final chunk already pads that way.
            // ⛔⛔⛔ A CONTINUATION BATCHES WHENEVER THE PREFIX-CAPABLE BUNDLE EXISTS — the
            // cache-hit tail AND the middle waves of a long prompt alike. A mid-prompt wave
            // is safe on this path because:
            //
            // - mid-wave chunks are rung-exact — the placement leaves `room >= cap_m`, so
            //   `chunk_len = (tok_end - tok).min(cap_m).min(room)` is the rung width for
            //   every chunk but a wave's LAST, and that one is exempt from the extent guard;
            // - a wave-final chunk's padding stays in the window holding its real tail (a
            //   page part past its last real row has no page and is skipped), is
            //   never recorded in `kv_hist`, and is overwritten by the next wave's first
            //   chunk before any row can attend it (`chunk_write_start` is monotone, so the
            //   next chunk's padded window starts no earlier and covers it) — the same
            //   write-before-read contract every multi-row chunk already relies on.
            //
            // The chunk loop writes K/V at the TOKEN position, so it requires a request whose
            // history is contiguous (slot == token). `solo_step_is_decode` keeps unfinished
            // prompts off the shared-slot batched decode — the only writer that can put a
            // hole in a history — so a request reaching this loop has none.
            //
            // The sequential walk below remains only for the missing-bundle case.
            let c_prefill_session = if start == 0 {
                sb.prefill_top_m() > 1
            } else {
                !sb.prefill_prefix.is_empty()
            };
            if c_prefill_session {
                toks.len().max(1)
            } else {
                // The exception path: no bundle can run this chunk — a cold prompt with no prefill ladder,
                // or a continuation with no prefix-capable bundle baked. Instrument-gated to avoid one line
                // per request — printing only, never a behaviour change.
                1
            }
        }
    };
    #[cfg(not(feature = "spyre-hw"))]
    let chunk_w = b.m_cap;
    // SEQUENTIAL PREFILL (revived, user-directed 2026-07-11): a SuperDSC multi-token prompt with no
    // batched prefill session is walked one token at a time (chunk_w=1) through the decode bundle's
    // device-resident KV — each token attends the KV built from prior tokens. This is the coherence
    // baseline path while batched mq=N prefill (task #33) is brought up in parallel. Decode (n==1) and
    // non-SuperDSC paths are unaffected; a batched prefill session (prefill.is_some) still takes the
    // fast prefill_m+1 chunk above.
    // ⭐ THE PROMPT'S LAST TOKEN RUNS THROUGH PREFILL, exactly as on main. The
    // prompt-final chunk's own minted sampler (the general mint — every prefill
    // rung carries the draw) picks the first generated token ON THE CARD from
    // the folded lm-head row, and the idx stick is read back exactly like a
    // decode launch's. No decode launch runs a prompt step, so the last prompt
    // position's KV is prefill-written, as main writes it. A generation step
    // (n==1) prefills nothing and takes the DECODE pick launch below — every
    // generated token past the first is the decode bundle's own draw. A
    // mid-prompt wave (start+n still inside the prompt) forwards every token it
    // was given and picks nothing: its chunks' suffixes never run.
    let finishes = start + n >= tokens.prompt_len(id);
    let prefill_n = if finishes && n > 1 {
        n
    } else if finishes {
        0
    } else {
        n
    };
    let mut pos = start;
    let mut prompt_pick: Option<u32> = None;
    for chunk in toks[..prefill_n].chunks(chunk_w) {
        if let ChunkOut::Tokens(ts) = forward_chunk(
            &sh,
            &mut model.session,
            b,
            model.batched_prefill.as_ref(),
            Chunk {
                state: req,
                id,
                tokens,
                start: pos,
                toks: chunk,
            },
            #[cfg(feature = "spyre-hw")]
            ctx,
            #[cfg(feature = "spyre-hw")]
            pool,
        )? {
            prompt_pick = ts.last().copied();
        }
        pos += chunk.len();
    }
    if !finishes {
        return Ok(None);
    }
    let tok = if n > 1 {
        // The prompt-final chunk's on-card pick — the prefill rung's own minted
        // sampler, read back as its idx stick. A `None` here means the final
        // chunk never asked for its draw — a wiring bug, never a silent skip.
        prompt_pick.ok_or_else(|| {
            werr(
                "the prompt-final chunk picked no token — its minted draw never ran"
                    .to_string(),
            )
        })?
    } else {
        // THE PICK — on the card, through the decode bundle's sampler. No host
        // logits, no host argmax, anywhere: the idx stick is the only answer that
        // comes back.
        let db = select_decode(&model.decode, start + n);
        match forward_chunk(
            &sh,
            &mut model.session,
            db,
            model.batched_prefill.as_ref(),
            Chunk {
                state: req,
                id,
                tokens,
                start: start + n - 1,
                toks: &toks[n - 1..n],
            },
            #[cfg(feature = "spyre-hw")]
            ctx,
            #[cfg(feature = "spyre-hw")]
            pool,
        )? {
            ChunkOut::Tokens(ts) => *ts
                .last()
                .ok_or_else(|| werr("the pick step's decode launch picked no token".to_string()))?,
            ChunkOut::Prompt => {
                return Err(werr(
                    "the pick step ran no decode launch — this build has no device sampler"
                        .to_string(),
                ))
            }
        }
    };
    tokens.push_generated(id, tok);
    Ok(Some(tok))
}

/// ⭐ THE PAD CLAMP'S OWN PIN (auditor ruling 2026-10-27): a pad row's staged
/// seat words are LIVE 0'S — for a greedy live 0 and a sampled one — because
/// the pad's ChainStage decompose feeds the s/l stick whose embed row races
/// live 0's pool cell, and only live 0's law keeps those bytes identical.
#[cfg(all(test, feature = "spyre-hw"))]
mod pad_clamp_tests {
    use super::*;

    /// A seat at `rows` baked rows whose strides agree with the placement's own
    /// derivations (the staging's own refusal contract), so `sample_binds`
    /// stages for real.
    fn seat_of(rows: u32) -> scratchy_target_spyre::wiring::SampleSeat {
        scratchy_target_spyre::wiring::SampleSeat {
            seed_words: [901, 902, 903, 904],
            invt: 905,
            rows,
            row: 0,
            vocab: 100,
            pad_vocab: 128,
            seed_row_words: (scratchy_target_spyre::wiring::col_row_stride_bytes(
                scratchy_tensors::DType::I32,
            ) / 4) as u32,
            gate_row_words: (scratchy_target_spyre::wiring::gate_row_stride_bytes() / 4) as u32,
        }
    }

    /// Two live rows in a 4-row rung — slots 2 and 3 are padding — with the
    /// live batch handed to `f` while the histories it borrows are alive.
    fn with_two_of_four<R>(f: impl FnOnce(&scratchy_subtile::sdsc_abstract::LiveBatch) -> R) -> R {
        use scratchy_subtile::sdsc_abstract::{
            BatchRow, KvHistory, RungWidth, SeqPos, SlotMap,
        };
        let width = RungWidth::of_baked_rows(4).expect("a 4-row rung");
        let hists: Vec<_> = (0..2).map(|i| KvHistory::contiguous(8 + i)).collect();
        let live = SlotMap::of_live(2, width)
            .expect("2 live rows fit")
            .live_rows(|i| {
                Ok::<_, ()>(BatchRow {
                    rope_pos: SeqPos::new(10 + i as u32),
                    hist: hists[i].clone(),
                })
            })
            .expect("the builder never fails here");
        f(&live)
    }

    /// Every staged bind's pad-row words equal row 0's — the four seed tables
    /// and inv_t alike. EVERY word sits at its row's STICK HEAD
    /// (`col_row_stride_bytes`): the gate table's reservation is wider (the
    /// i32→f32 convert's device sweep, two sticks per row) but the sweep's
    /// second sticks are the reservation's own zero tail — the word the
    /// kernel reads is still the head, at the same stride as every other
    /// table.
    fn assert_pads_stage_live_0(
        seat: &scratchy_target_spyre::wiring::SampleSeat,
        laws: &[RowLaw<'_>],
        ctx: &str,
    ) {
        let binds = sample_binds(seat, laws).expect("the seat stages");
        let head_stride =
            scratchy_target_spyre::wiring::col_row_stride_bytes(scratchy_tensors::DType::I32);
        assert_eq!(binds.len(), 5, "{ctx}: four seed tables and inv_t");
        for (tid, bytes) in &binds {
            for r in [2usize, 3] {
                assert_eq!(
                    &bytes[r * head_stride..r * head_stride + 4],
                    &bytes[0..4],
                    "{ctx}: bind t{tid:?} row {r} (a padding slot) staged words row 0's own \
                     are not — the pad clamp broke",
                );
            }
        }
    }

    #[test]
    fn a_full_rung_classes_no_row_as_a_pad() {
        // ⭐ THE FULL-OCCUPANCY LAW (auditor order 2026-10-27): with live ==
        // rows, EVERY row runs its own request's draw — no row is classed as
        // a pad (the non-pipelined tree seated a 4-request launch's fourth
        // row greedy; this pin refuses that class of defect here).
        use scratchy_subtile::sdsc_abstract::{
            BatchRow, KvHistory, RungWidth, SeqPos, SlotMap,
        };
        let width = RungWidth::of_baked_rows(4).expect("a 4-row rung");
        let hists: Vec<_> = (0..4).map(|i| KvHistory::contiguous(8 + i)).collect();
        let live = SlotMap::of_live(4, width)
            .expect("4 live rows fill the rung")
            .live_rows(|i| {
                Ok::<_, ()>(BatchRow {
                    rope_pos: SeqPos::new(10 + i as u32),
                    hist: hists[i].clone(),
                })
            })
            .expect("the builder never fails here");
        let owned: Vec<SampledDraw> = (0..4)
            .map(|i| SampledDraw {
                seed: 1000 + i as u64,
                inv_t: 0.5,
            })
            .collect();
        let draws: Vec<Option<&SampledDraw>> = owned.iter().map(Some).collect();
        let laws = batch_seat_laws(&draws, &live, 4).expect("the full laws lay");
        assert_eq!(laws.len(), 4);
        for s in 0..4 {
            let law = laws[s]
                .draw
                .unwrap_or_else(|| panic!("row {s} of a FULL rung staged greedy — no row is a pad"));
            assert_eq!(law.seed, 1000 + s as u64, "row {s} runs its OWN draw");
            assert_eq!(laws[s].step, 10 + s as u32, "row {s} keeps its own step salt");
        }
    }

    #[test]
    fn a_greedy_live_0_clamps_its_pads_to_greedy() {
        let seat = seat_of(4);
        let draws: [Option<&SampledDraw>; 2] = [None, None];
        let laws = with_two_of_four(|live| batch_seat_laws(&draws, live, 4))
            .expect("the greedy laws lay");
        // The law itself: pads 2 and 3 carry live 0's law — greedy, at live
        // 0's own step salt (the word the gate's 0 makes inert).
        assert!(laws[2].draw.is_none() && laws[3].draw.is_none());
        assert_eq!(laws[2].step, laws[0].step, "the pad clamps live 0's step");
        assert_eq!(laws[3].step, laws[0].step, "every pad clamps live 0's step");
        assert_pads_stage_live_0(&seat, &laws, "greedy live 0");
    }

    #[test]
    fn a_sampled_live_0_clamps_its_pads_to_its_draw() {
        let seat = seat_of(4);
        let draw = SampledDraw {
            seed: 0x0123_4567_89AB_CDEF,
            inv_t: 0.5,
        };
        // Live 0 sampled, live 1 greedy: the pads clamp LIVE 0'S law, not the
        // last live row's and not the greedy one.
        let draws: [Option<&SampledDraw>; 2] = [Some(&draw), None];
        let laws = with_two_of_four(|live| batch_seat_laws(&draws, live, 4))
            .expect("the sampled laws lay");
        for r in [2usize, 3] {
            let pad = laws[r].draw.expect("the pad clamps live 0's draw, not greedy");
            assert_eq!(pad.seed, draw.seed, "pad {r}: live 0's seed");
            assert_eq!(
                pad.inv_t.to_bits(),
                draw.inv_t.to_bits(),
                "pad {r}: live 0's inv_t"
            );
            assert_eq!(laws[r].step, laws[0].step, "pad {r}: live 0's step salt");
        }
        // Live 1 keeps its OWN greedy law — the clamp touches pads alone.
        assert!(laws[1].draw.is_none());
        assert_pads_stage_live_0(&seat, &laws, "sampled live 0");
    }
}
