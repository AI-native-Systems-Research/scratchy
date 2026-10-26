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
use scratchy_target_spyre::manifest::argmax;
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
    // (The batch-chunk body that carried its own one-row-block law is deleted with the copy-gather
    // family; a batched rung on this branch is always the region-form fold.)
    per_request: Option<(
        &scratchy_subtile::sdsc_abstract::LiveBatch,
        scratchy_subtile::sdsc_abstract::BatchPrefix,
    )>,
    sh: &Shared<'_>,
    pf: &mut SuperDscSession,
    b: &BundleMeta,
    // ⛔ THERE IS NO `pages` PARAMETER, AND ITS ABSENCE IS THE LOCK. It was "pages this request holds", and
    // it fed the prompt-chunk mask extent — but only through an arm that a PREFIX-FREE prefill rung never
    // took (`page_slots == 0` ⇒ `pf.cap`, a pool number). A prompt chunk's extent is now the BAKE's own
    // pmask reserve (`SuperDscSession::pmask_slots`) and a decode batch's is the DECLARED fold grid, so
    // neither can be handed a page count computed at a call site. Deleting the parameter is what stops the
    // next caller from reaching for `ctx.host.len()` or a pool size again.
    toks: &[usize],
    // The width the bundle being run was BAKED at — the launch binds exactly this many rows,
    // live or padding, and every extent below (embeddings, masks, `run_step`'s m) is sized by it.
    prefill_m: scratchy_subtile::sdsc_abstract::RungWidth,
    start: usize,
    want_logits: LogitsWanted<'_>,
) -> ExecutorResult<Option<Vec<f32>>> {
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
    let real = toks.len();
    // The kind of a launch WITHOUT `per_request` — a prompt chunk, whose rows are consecutive positions
    // of one prompt. Its laws (the pad-row clamp, the causal triangle) hang off this type and nothing
    // else can reach them.
    let chunk = ChunkRows::chunk(prefill_m.rows());
    // The rung width as a host-side extent, taken once — every loop and buffer below is `mq`-sized.
    let mq = prefill_m.count();
    if real == 0 || real > mq {
        return Err(werr(format!(
            "run_prefill_batch: real={real} not in 1..={mq}"
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
    let per_request: Option<(
        scratchy_subtile::sdsc_abstract::LaunchRows<'_>,
        scratchy_subtile::sdsc_abstract::BatchPrefix,
    )> = match per_request {
        Some((rows, fold)) => {
            if rows.live() != real {
                return Err(werr(format!(
                    "run_prefill_batch: {} live row(s) but {real} token(s) — a decode batch carries \
                     exactly one token per live request",
                    rows.live()
                )));
            }
            let laid = rows.launch_rows(toks, prefill_m).ok_or_else(|| {
                werr(format!(
                    "run_prefill_batch: {} live row(s) / {} token(s) do not lay into a {mq}-row rung",
                    rows.live(),
                    toks.len()
                ))
            })?;
            Some((laid, fold))
        }
        None => None,
    };
    // Pad rows REPLICATE — whom is a kind question, and the kind is which of these two arms exists. A
    // decode batch's pad rows take LIVE 0's token, because a pad row's forward becomes its cache write
    // and that write races live 0's at the same cell. A prompt chunk's pad rows take the LAST real
    // token: the emitter reads the lm-head input from the last baked row, which must carry the last
    // real token (and a chunk's pad KV writes append at the row's own slot, past n_computed — no
    // shared cell).
    let tok_at = |i: usize| match &per_request {
        Some((rows, _)) => rows.row(i).expect("i < mq").token().get() as usize,
        None if i < real => toks[i],
        None => toks[real - 1],
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
    // flat/row-major (element (i,j) at i*mq_pad+j): the deleted hand-written emitter read
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
    let tokens: Vec<usize> = (0..mq).map(tok_at).collect();
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
        _ => pf.pmask_slots.map(|cap| {
            scratchy_target_spyre::forward_tape::PrefixSource::BroadcastPrefix {
                cap,
                holds: |c: usize| decode_prefix_col_valid(c, start),
            }
        }),
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
        positions: &positions,
        causal,
        prefix: prefix_src,
        consts: &consts,
        mask_neg,
        stick_major: stickmajor_emb,
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
    // NO-LOGITS variant for every chunk but the last: only the FINAL chunk's lm-head tail produces a
    // token, so the rest DISCARD their logits — the executor launches no suffix for them and reads
    // nothing back (`StepLogits::Discarded`). A decode batch is NOT that case: it reads its whole
    // `[rows, vocab]` result itself, so the suffix must run (`StepLogits::LeftOnDevice`).
    log_binds(
        &format!("prefill mq={mq} start={start}"),
        b.wiring,
        &acts,
        &acts_f16,
    );
    let logits = match want_logits {
        LogitsWanted::LastRow => {
            let l = pf
                .run_step_f16_logits(mq, start, &acts, &acts_f16, StepLogits::ReadBack)
                .map_err(|e| {
                    werr(format!(
                        "superdsc prefill run_step (mq={mq}, start={start}): {e}"
                    ))
                })?
                .unwrap_or_default();
            Some(l.get(..vocab).unwrap_or(&l).to_vec())
        }
        // The shim's own logits D2H converts ONE row from the placement's start (it was written for
        // the folded tail), so it cannot serve this case. Run with a null out-pointer and read the
        // whole `[rows, width]` result tensor back instead — the same single D2H, then de-interleave.
        LogitsWanted::PerRequest(geom) => {
            pf.run_step_f16_logits(
                mq,
                start,
                &acts,
                &acts_f16,
                StepLogits::LeftOnDevice,
            )
            .map_err(|e| {
                werr(format!(
                    "superdsc decode batch run_step (mq={mq}, start={start}): {e}"
                ))
            })?;
            // ⭐⭐ THE OBSERVATION, not another argument: read the mask BACK off the device and diff it
            // against the bytes we staged. Identical ⇒ the mask is exonerated and the fault is in the KV
            // path. Different ⇒ the fault is placement, the H2D window, or the staging, and the first
            // differing element names which. This is the one thing about the batched fold that has never
            // been looked at, and it is why every emitter-side guard passes while the output is wrong.
            //
            // Gated because a per-step D2H of this buffer is not free. `read_tensor` yields f32 from the
            // device's f16, so compare against the staged bytes reinterpreted the same way.
            let name = scratchy_target_spyre::bundle_code::PlaceId::Act((b.result_id) as u32);
            // TIMED, because `predict`'s own phase timer CANNOT see this. Moving the logits readback
            // out of `predict` (the shim converts one row, so it cannot serve a batch) also moved it
            // outside `logits_d2h`, which is why that column reads 0.00 for a batched decode while
            // ~47 ms of a 182 ms step at 32 requests sits unaccounted for. `read_tensor` D2Hs the
            // WHOLE segment holding the logits, and the logits are a few MB of it.
            let _t_logits = std::time::Instant::now();
            let flat = pf
                .read_tensor(name, geom.len())
                .map_err(|e| werr(format!("decode batch: read logits {name}: {e}")))?;
            if flat.len() < geom.len() {
                return Err(werr(format!(
                    "decode batch: logits {name} read back {} of {} elems ([{}, {}]) — the baked \
                     placement is smaller than the rung's row count says it is",
                    flat.len(),
                    geom.len(),
                    geom.rows,
                    geom.width
                )));
            }
            if flags().phase_time {
                let ms = _t_logits.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "[sdsc-host] batch_logits_d2h={ms:.2} ms  ({} rows x {} wide = {} elems read; \
                     {} sampled)",
                    geom.rows,
                    geom.width,
                    geom.len(),
                    mq * vocab,
                );
            }
            let gathered = geom.rows_contiguous(&flat, vocab).ok_or_else(|| {
                werr(format!(
                    "decode batch: gathering [{}, {}] logits at vocab {vocab} addressed outside the \
                     {} elems read back",
                    geom.rows,
                    geom.width,
                    flat.len()
                ))
            })?;
            // ⭐ ROW-vs-ROW LOGITS, for the one test that needs no ground truth: run N IDENTICAL prompts
            // and every row's logits must be IDENTICAL. They are not — four identical prompts produced
            // three `' B: The answer'` and one `' Madrid.'` — and with identical inputs any difference at
            // all is a defect, so this print localises WHICH rows diverge and by HOW MUCH without any
            // baseline to argue about.
            //
            // Compares every row against row 0: max |Δ|, the element where it occurs, and each row's
            // argmax. Identical argmaxes with a tiny Δ means accumulation-order noise; a different argmax
            // means the rows genuinely computed different things.
            Some(gathered)
        }
        LogitsWanted::None => {
            pf.run_step_f16_logits(
                mq,
                start,
                &acts,
                &acts_f16,
                StepLogits::Discarded,
            )
            .map_err(|e| {
                werr(format!(
                    "superdsc prefill run_step (mq={mq}, start={start}): {e}"
                ))
            })?;
            None
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
    Ok(logits)
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
            // The one step that stages nothing: the causal mask, when the session holds its key.
            return if t.0 == causal_tid && self.causal_held == Some(self.inputs.causal) {
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

/// ⭐⭐⭐ WHOSE ROWS THE LAUNCH RUNS — re-exported from the shared staging law
/// ([`scratchy_target_spyre::attn_step_staging`]): the enum is part of the ONE builder both
/// runtimes stage through, so it lives where the builder does, not beside the card's fill.
#[cfg(feature = "spyre-hw")]
pub(crate) use scratchy_target_spyre::attn_step_staging::AttnStepRows;

#[cfg(feature = "spyre-hw")]
/// ⭐⭐⭐⭐⭐ FILL THE ATTENTION FAMILY'S HOST-STAGED STEP INPUTS — the worker's half of the
/// contract `attn_step_slots` spells. ONE helper, ONE spelling: the prompt chunk, the solo
/// decode and the decode batch all stage through the session's `fill_step_input` door, slots
/// named from the contract's table, CONTENT computed by the shared builder in
/// [`scratchy_target_spyre::attn_step_staging`] — the same function the bake-time numeric
/// gate stages through, so the gate judges byte-identically what the card runs. This side is
/// the SESSION half only: query the discovery table, hand the extents and the ceiling to the
/// builder, fill what it returns, in the same order the card always filled.
///
/// ⭐ THE GEOMETRY CROSSES THE MODEL-GEOMETRY DOOR. The builder is const-generic over the
/// witness, so this fill mints the [`ModelAttnGeometry`] from the facts it holds — the wiring's
/// head count over the pool's kv geometry — and crosses `with_config_attn_geometry` into the
/// const arm. The mint is the pipeline's ONE evaluation of the GQA division (a `None` is the
/// wiring/pool disagreement the old inline division refused, same text); the door's `None` is
/// the generated contract (no config in scope declares this triple). The pool the builder
/// derives comes from the witness, so the value and the consts cannot disagree. ⚠ The mint's
/// INPUTS are the worker-side re-derivations item 1 replaces next (bake-emitted geometry
/// facts); the mint itself stays — it is the value side of the door, not a second division.
///
/// ⭐ QUERY, THEN FILL. The bundle's own baked discovery table says which slots exist; this
/// asks it (through `step_input_extent`, the artifact's own answer) and fills exactly what it
/// declares — all of them, or a refusal: on this branch every attention program is one of the
/// three spliced kernels and all of them declare the step inputs, so a bundle that runs
/// attention while declaring NONE (or a PARTIAL set) is a bake/host disagreement the builder
/// refuses, because an unstaged id table reads as block 0 (a REAL address) and an unstaged
/// mask reads as zero (NO masking), both fluently wrong.
///
/// ⛔ THERE IS NO OTHER STAGING: the door-era `KV_BLOCK_INDEX_TID` bind is deleted with the
/// copy-gather family, so a gathered KV read reaches the card only through these slots, whose
/// extents the bundle's own placement declares — that deletion is the no-host-fallback proof.
pub(crate) fn fill_attn_step_inputs(
    what: &str,
    session: &mut SuperDscSession,
    pool: scratchy_subtile::sdsc_abstract::PagedKvPool,
    nqh: u32,
    rows: AttnStepRows<'_>,
) -> ExecutorResult<()> {
    use scratchy_subtile::model_geometry::{self, HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
    // THE MINT — the one evaluation of the GQA division, from the facts this fill holds.
    let geometry = ModelAttnGeometry::mint(
        QueryHeads::new(nqh),
        KvHeads::new(pool.nkvh()),
        HeadDim::new(pool.hd()),
    )
    .ok_or_else(|| {
        werr(format!(
            "{what}: nqh {nqh} does not cover the pool's {} kv head(s) in whole GQA groups — \
             the wiring and the pool disagree about the model's geometry",
            pool.nkvh()
        ))
    })?;
    model_geometry::with_config_attn_geometry(
        geometry,
        FillAtTheGeometry {
            what,
            session,
            rows,
        },
    )
    .ok_or_else(|| {
        werr(format!(
            "{what}: geometry {geometry} is declared by no model config in scope ({}) — the fix \
             is the config naming this model, never an edit to the fill",
            model_geometry::geometry_sources()
        ))
    })?
}

/// The fill's body at the geometry the configs declare — the consumer side of the door.
#[cfg(feature = "spyre-hw")]
struct FillAtTheGeometry<'a, 'b> {
    what: &'a str,
    session: &'a mut SuperDscSession,
    rows: AttnStepRows<'b>,
}

#[cfg(feature = "spyre-hw")]
impl scratchy_subtile::model_geometry::OnAttnGeometry for FillAtTheGeometry<'_, '_> {
    type Out = ExecutorResult<()>;

    fn on_geometry<const NQH: u32, const NKVH: u32, const HD: u32>(
        self,
        geom: scratchy_subtile::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    ) -> Self::Out {
        use scratchy_subtile::sdsc_abstract::attn_step_slots;
        use scratchy_target_spyre::attn_step_staging::{self, AttnStepExtents};
        // The pool the witness states — the same facts the mint read, now as the consts' own.
        let pool = scratchy_subtile::sdsc_abstract::PagedKvPool::of_geometry(geom);
        let what = self.what;
        // Query the discovery table first: extents are the artifact's own answers, and the
        // door's `==` law re-measures every one of them at fill time.
        let extents = AttnStepExtents {
            mask: self.session.step_input_extent(attn_step_slots::MASK),
            kt: self
                .session
                .step_input_extent(attn_step_slots::KT_BLOCK_IDS),
            v: self.session.step_input_extent(attn_step_slots::V_BLOCK_IDS),
            rep: self.session.step_input_extent(attn_step_slots::REP_ROW_IDS),
        };
        // The ceiling from the session's own page stride — the ONE law, handed the card's
        // number (`gather_entry_ceiling` delegates to the same shared spelling the gate uses).
        let ceiling = self.session.gather_entry_ceiling(pool);
        // The pure construction — every refusal is the builder's own text, unchanged.
        let p = attn_step_staging::build_attn_step_inputs(geom, what, &extents, ceiling, self.rows)
            .map_err(werr)?;
        let fill = |session: &mut SuperDscSession, slot: u32, bytes: Vec<u8>| {
            session
                .fill_step_input(slot, bytes)
                .map_err(|e| werr(format!("{what}: fill step input slot {slot}: {e}")))
        };
        let session = self.session;
        fill(&mut *session, attn_step_slots::KT_BLOCK_IDS, p.kt)?;
        fill(&mut *session, attn_step_slots::V_BLOCK_IDS, p.v)?;
        if let Some(rep) = p.rep {
            fill(&mut *session, attn_step_slots::REP_ROW_IDS, rep)?;
        }
        fill(&mut *session, attn_step_slots::MASK, p.mask)?;
        Ok(())
    }
}

/// ⭐ WHAT ONE CHUNK'S FORWARD PRODUCED — the per-row logits the host sampler
/// reads. One enum (not a bare `Vec<Vec<f32>>`) because the shape is a fact of the
/// step the caller must not be able to confuse with the model's other outs.
pub(crate) enum ChunkOut {
    /// Per-row logits, the last row last — the host sampler's input.
    Rows(Vec<Vec<f32>>),
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
    // The pool's geometry, threaded from the model — the step-input fill's law, ONE spelling
    // (built in `run_request_step` off `Loaded`'s own two numbers).
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
    // mq>1 cachewr, one ≤prefill_m chunk at a time). The FINAL chunk also returns the first generated
    // token's logits: the emitter folds the vocab-wide lm-head tail to m=1 over that chunk's last row
    // (see `LAST_HIDDEN_TID`), so there is nothing left for a decode step to compute.
    //
    // This USED to stop at toks[0..n-1) and then decode toks[n-1] on the m=1 path purely to get those
    // logits — a second whole forward that re-streamed every weight (~29 ms) to compute one row the
    // prefill already had in hand. Now TTFT is ceil(n/prefill_m) forwards, not that plus one.
    //
    // Reaching the decode loop below therefore means this call is a single generation step (n == 1):
    // a multi-token prompt either returns its logits from the batched prefill or is refused, so the
    // loop runs on `start`/`toks` directly.
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
            let real_total = n; // the WHOLE prompt goes through prefill, last token included
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
            let mut prefill_logits: Option<Vec<f32>> = None;
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
                // Only the chunk that FINISHES THE PROMPT feeds the sampler — its folded tail
                // is the first generated token's distribution. Every other chunk skips the
                // logits D2H, a mid-prompt wave's last chunk included: its tail belongs to a
                // later wave, and `run_request_step`'s sample guard returns without reading it.
                let is_last_chunk = tok + chunk_len == tok_end;
                let is_prompt_final_chunk = is_last_chunk && tok_end >= tokens.prompt_len(id);
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
                // so the staged step inputs must be the RUNNING body's, the solo decode's own law.
                // `step_body(start)` is the selection the launch makes (`valid_len = start + 1`),
                // and the chunk must hold the `start` resident keys; this chunk's own rows, written
                // before attention, sit at slots `>= start`, which the broadcast validity row masks.
                let step = pf.step_body(chunk_start).map_err(|e| {
                    werr(format!(
                        "prompt chunk at {chunk_start}: the session could not say which body runs: {e}"
                    ))
                })?;
                // The LADDER entry's baked width — hoisted once: the fill below and the launch
                // below THAT both stage for the same rung, and two constructions of one width is
                // a second chance to disagree about which body runs.
                let rung = scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(rung_m as u32)
                    .ok_or_else(|| {
                        werr(
                            "superdsc batched prefill: the selected rung has zero rows".to_string(),
                        )
                    })?;
                // ⭐⭐⭐⭐⭐ THE SPLICE'S OWN STEP INPUTS — the block-entry tables and the validity
                // mask the spliced attention kernel reads, staged through the ONE fill helper in
                // the kernel's read order. The map is the CHUNK's own (`chunk_page_map`: every
                // page the write lands in backed, so a window wider than a page reads its own
                // new keys back through the gather — unlike the door-era chunk, whose own window
                // rode a segment shift). This fill is the ONE staging: the copy-gather family
                // and its `KV_BLOCK_INDEX_TID` channel are deleted, so there is no second path
                // a gathered read could reach the card through.
                fill_attn_step_inputs(
                    "prompt chunk",
                    pf,
                    pool,
                    pmeta.wiring.geometry.nqh,
                    AttnStepRows::Chunk {
                        rows: scratchy_subtile::sdsc_abstract::ChunkRows::chunk(rung.rows()),
                        mq: rung,
                        start: chunk_start,
                        real: std::num::NonZeroU32::new(chunk_len as u32).ok_or_else(|| {
                            werr(
                                "superdsc batched prefill: a prompt chunk of ZERO rows reached the \
                                 fill — the rung selection above is wrong, not this site"
                                    .to_string(),
                            )
                        })?,
                        map: req
                            .chunk_page_map(
                                ctx,
                                placed.wrote(scratchy_subtile::sdsc_abstract::SlotCount::new(
                                    chunk_len as u32,
                                )),
                            )
                            .ok_or_else(|| {
                                werr(format!(
                                    "superdsc batched prefill: cannot map the chunk's pages — \
                                     the host granted {} block(s) and a page the chunk writes \
                                     has none",
                                    ctx.host.len(),
                                ))
                            })?
                            .as_i64(),
                    },
                )?;
                prefill_logits = run_prefill_batch(
                    // A prompt chunk: these rows are consecutive positions of one sequence, so they
                    // share both the causal triangle and one resident-prefix validity row. `None` IS
                    // that statement — there are no per-request rows, so the chunk laws are the only
                    // ones reachable.
                    None,
                    sh,
                    pf,
                    pmeta,
                    // ⭐ FROM THE REQUEST, AT ABSOLUTE POSITIONS — not from this step's window. A chunk
                    // that stepped back needs tokens an earlier step was given, and the request holds them.
                    &chunk_toks,
                    // The LADDER entry's baked width — the rung this chunk actually runs on, of
                    // which `chunk_len` rows carry real tokens.
                    rung,
                    chunk_start,
                    // A prompt's tail is FOLDED to one row by the emitter, so there is one row to
                    // read however many rows this chunk ran at.
                    if is_prompt_final_chunk {
                        LogitsWanted::LastRow
                    } else {
                        LogitsWanted::None
                    },
                )?;
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
                     tok_end={tok_end} real_total={real_total} logits={is_prompt_final_chunk}",
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
            // The prompt is fully computed AND its next-token logits are in hand, so return them
            // straight to the sampler. A mid-prompt wave asks for no logits (see
            // `is_prompt_final_chunk`), so an empty hand with the prompt unfinished is the
            // expected shape — `run_request_step`'s sample guard returns before reading it.
            // `None` with the prompt DONE means the loop never ran — impossible under `n > 1`,
            // but refuse loudly rather than fall through to a decode path that would now
            // re-forward zero tokens.
            let row = match prefill_logits {
                Some(row) => row,
                None if tok_end < tokens.prompt_len(id) => return Ok(ChunkOut::Rows(Vec::new())),
                None => {
                    return Err(werr(format!(
                        "superdsc batched prefill: covered {real_total} tokens but produced no \
                         logits — the final chunk did not run its folded lm-head tail \
                         (chunk-loop wiring bug)"
                    )));
                }
            };
            return Ok(ChunkOut::Rows(vec![row]));
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

    let mut logits_rows: Vec<Vec<f32>> = Vec::with_capacity(n);
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
            scratchy_target_spyre::forward_tape::PrefixSource::BroadcastHeld {
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
        // ⭐⭐⭐⭐⭐ THE SPLICE'S OWN STEP INPUTS — the fold body reads its pages through the
        // block-entry tables and its validity through the mask, both staged per token through
        // the ONE fill helper (the same call the prompt chunk makes; the fold and the one-shot
        // kernel read the same region form). The map covers slots `0..=p` — the new token's
        // page included, because the fold gathers the whole history through the table, its own
        // new key with it. This fill is the ONE staging: the copy-gather family and its
        // `KV_BLOCK_INDEX_TID` channel are deleted, so there is no second path a gathered read
        // could reach the card through.
        fill_attn_step_inputs(
            "solo decode",
            session,
            pool,
            b.wiring.geometry.nqh,
            AttnStepRows::Chunk {
                rows: scratchy_subtile::sdsc_abstract::ChunkRows::single(),
                mq: scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(1)
                    .expect("1 is a width"),
                start: p,
                real: std::num::NonZeroU32::MIN,
                map: req
                    .page_map(
                        ctx,
                        scratchy_subtile::sdsc_abstract::RowPages::holding(
                            scratchy_subtile::sdsc_abstract::SlotCount::new(p as u32 + 1),
                        ),
                    )
                    .ok_or_else(|| {
                        werr(format!(
                            "solo decode: cannot map the page holding pos {p} — every page \
                             holding a key, the new token's included, must be a host block"
                        ))
                    })?
                    .as_i64(),
            },
        )?;
        // ⭐⭐⭐⭐⭐ THE FOLD'S OWN WALK COUNT, DECLARED — the same door the batched path takes
        // (`spyre_exec.rs`): `FoldPages::resident_before` off the solo write slot (this
        // request's own end — `BatchSlot::solo`, the constructor that cannot reach for a token
        // count), the form off the RUNNING body's own `gathers()` (the per-body door; the
        // bundle's placement answers for ANY body and is not this step's fact). The executor's
        // `n_fold_pages(p)` is the SAME derivation — `ceil(p / PAGE_SLOTS)` — so `reps`'s
        // walk-vs-declaration guard now CHECKS the solo fold where it was SKIPPED before:
        // undeclared (0), arm (i) refused nothing, and the solo path's walk ran on the
        // executor's derivation alone — the one guard the batched family has had since the
        // 2048-wall. `None` (an empty history — zero pages to fold) has no `FoldPages` by
        // type and refuses here rather than folding over a page count nobody can name.
        scratchy_subtile::sdsc_abstract::FoldPages::resident_before(
            scratchy_subtile::sdsc_abstract::BatchSlot::solo(&req.kv_hist),
            scratchy_subtile::sdsc_abstract::SlotCount::new(
                scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32,
            ),
        )
        .ok_or_else(|| {
            werr(format!(
                "solo decode at pos {p}: the request's history holds no whole page to fold — \
                 decode always follows a prefill of at least one token, so this is a population \
                 bug, not a context length"
            ))
        })?
        .grid(
            scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(1).expect("1 is a width"),
            // 📌 THE FORM'S CONSUMER SET, PINNED (the region-form flip's inertness proof):
            // the form rides `MaskPassGrid` into `MaskBlocks::of` and `declare_to`, which
            // records it on the session (`sdsc_runner.rs`'s `FoldWalker` impl:
            // `set_mask_blocks` + `set_fold_pages`). Those two records are read ONLY by the
            // fold launch path — `fold_plan::reps`'s walk-vs-declaration guard, per
            // `page_fold` group, and the fold's prefix-mask staging — and a ONE-SHOT step
            // launches no fold group, so neither consumer runs: the flip to `PerPage` that
            // the region-form stamp causes on one-shot steps is INERT (the one-shot body's
            // own mask is fill-staged by `fill_attn_step_inputs`, which reads the discovery
            // table's extents, not this declaration). A fold step (the fold twin,
            // valid_len 129-256) IS collapsed and gathered, so `PerPage` is the correct form
            // there — the flip lands it, which is the direction the fold wants.
            if step.gathers().get() {
                scratchy_subtile::sdsc_abstract::MaskBlockForm::PerPage
            } else {
                scratchy_subtile::sdsc_abstract::MaskBlockForm::PerRowPage
            },
        )
        .declare_to(session)
        .map_err(|e| werr(format!("solo decode: declare fold: {e}")))?;
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
            positions: &[p as u32],
            causal,
            prefix: prefix_src,
            consts: &consts,
            mask_neg,
            // Proven to be the identity at one row, so this is decode's existing row-major
            // staging spelled as the general form.
            stick_major: true,
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
        // ⛔ EVERY STAGED LIST IS NAMED, no `..`: a new staged list is a missing field here, and an
        // unbound one is an unused binding.
        let ForwardLauncher {
            shape: _,
            steps: _,
            inputs: _,
            staged: _,
            acts,
            acts_f16,
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
        {
            let logits = session
                .run_step_f16_logits(1, p, &acts, &acts_f16, StepLogits::ReadBack)
                .map_err(|e| werr(format!("superdsc run_step (pos {p}): {e}")))?
                .unwrap_or_default();
            step_timing(p, _t_fwd);
            let row = logits.get(..vocab).unwrap_or(&logits).to_vec();
            // ── KCACHE PROBE (SCRATCHY_KCACHE_PROBE): read the layer-0 K cache t9 (resident, Kᵀ [nqh,hd,cap])
            //    and print per-slot K-vector norms for head 0 around the 64-slot STICK BOUNDARY. If slot 64
            //    (start of stick 1) shows a discontinuity vs 62/63/65, the K WRITE/VALUE at the boundary is
            //    wrong (the pos-64 divergence); if smooth, the residue is the device READ or fp16 accumulation.
            // HOST lm_head ORACLE: recompute logits = t787 @ embed.T on the HOST from the
            // ON-CARD final-norm output (t787) and the tied embedding (= the lm_head weight).
            //   host_argmax == oncard t788 argmax ⇒ the on-card lm_head is FAITHFUL to t787,
            //     so the bug is t787 (the residual DIRECTION the body produced);
            //   host_argmax != oncard ⇒ the on-card (tiled) lm_head matmul itself is wrong.
            // Also reveals what token the on-card t787 SHOULD produce under a correct lm_head.
            // ATTENTION HOST ORACLE: verify layer-0 attention VALUES (the per-head fix made it
            // finite + context-reading but never proved correct). After this step the K/V cache
            // (t9/t10, natural [nqh,cap,hd], GQA-replicated) holds slots [0..=p] incl. the just-
            // written new token, so host attention = softmax(q·kᵀ·scale)·v over [0..=p] — no
            // separate new-token/GQA handling. cos(host, on-card t343)<1 ⇒ attention compute wrong
            // (restickify/scores/softmax); cos≈1 ⇒ attention right → bug is MLP/accumulation.
            logits_rows.push(row);
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
    Ok(ChunkOut::Rows(logits_rows))
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
        // The embedding is this chunk's rows of the hidden state.
        dynamic.push((b.embed_src as u64, emb, vec![rows, sh.hidden]));
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
        let res = &out[&(b.result_id as u64)];
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
        Ok(ChunkOut::Rows(vec![res[..vocab].to_vec()]))
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
    // ⭐ THE PAGED-KV POOL'S GEOMETRY — the model's own stored pool (the same one the
    // batched-decode site reads), threaded to the chunk paths because `Shared`
    // deliberately carries no kv width: a hybrid-attention arch has no single one, and
    // the pool is the one lawful model-wide use (see `Loaded::pool`).
    #[cfg(feature = "spyre-hw")]
    let pool = model.pool;
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
    let mut last: Vec<f32> = Vec::new();
    let mut pos = start;
    for chunk in toks.chunks(chunk_w) {
        match forward_chunk(
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
            ChunkOut::Rows(rows) => last = rows.into_iter().next_back().unwrap_or_default(),
        }
        pos += chunk.len();
    }
    let p_last = start + n - 1;
    if p_last + 1 >= tokens.prompt_len(id) {
        // The host `argmax` is the sampler.
        let tok = argmax(&last) as u32;
        // ⭐ THE SAME TOP-2 GAP AS THE BATCHED PATH, so a mixed run's steps are all visible in one log.
        //
        // Without this, `SCRATCHY_TOP2` covered only the batched sampling site — and a step with ONE live
        // request comes down HERE, so two repeat runs printed 8 and 6 lines and the difference read as
        // missing data rather than as "these steps were solo". A gap in an instrument is indistinguishable
        // from a gap in the thing it measures.
        tokens.push_generated(id, tok);
        Ok(Some(tok))
    } else {
        Ok(None)
    }
}
