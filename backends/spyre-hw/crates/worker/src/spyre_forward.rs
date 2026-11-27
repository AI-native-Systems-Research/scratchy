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
// The runner type is cfg-selected: the KTIR emulator's session on the host, the sendnn one on
// silicon. Everything around it (weight load, dynamic sources, KV loop, sampling) is shared.

#[cfg(feature = "spyre-hw")]
pub(crate) use crate::card::forward::*;
#[cfg(not(feature = "spyre-hw"))]
pub(crate) use crate::emu::forward::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;
use scratchy_serving_worker::error::ExecutorResult;

/// ⭐ WHAT ONE CHUNK'S FORWARD PRODUCED — the per-row logits the host sampler
/// reads. One enum (not a bare `Vec<Vec<f32>>`) because the shape is a fact of the
/// step the caller must not be able to confuse with the model's other outs.
pub(crate) enum ChunkOut {
    /// A forward that picked nothing: a chunk that does not finish its prompt (its
    /// suffix never runs — the finish law stated once inside each tier's forward).
    /// A chunk that DOES finish and minted no draw is the tier forward's own
    /// refusal, never this variant.
    Prompt,
    /// TOKENS THE DEVICE PICKED, one per launch this chunk ran that drew, in step
    /// order. The minted sampler's idx stick, read back raw — the only pick that
    /// exists, on the card and on the emulator alike (the bake mints the draw into
    /// both tiers' bundles). The shared readers go through
    /// [`ChunkOut::minted_pick`], so no reader names the variant.
    #[cfg(not(feature = "spyre-hw"))]
    Tokens(Vec<u32>),
    /// ⭐ THE CHUNK'S PICK IS OWED (card only): the prompt-final launch's read is
    /// deferred and its piece rides the executor's step queue — the caller's output
    /// defers on the piece's cell. The card tier collects nothing in-call; the
    /// emulator's twin has no queue and keeps its synchronous [`ChunkOut::Tokens`].
    /// Boxed: the piece IS this enum's whole weight, and a prompt pays one box at
    /// its final chunk — never per token.
    #[cfg(feature = "spyre-hw")]
    Owed(Box<crate::card::exec::InFlightStep>),
}

impl ChunkOut {
    /// ⭐ THE CHUNK'S OWN MINTED PICKS, one per launch that ran a sampler — the
    /// single door both shared pick readers go through. `None` is every chunk
    /// that picked nothing: a mid-prompt chunk (its suffix never runs — each
    /// tier's forward states the finish law once). On the card tier an owed
    /// chunk answers `None` — its pick is not IN HAND, and the chunk loop takes
    /// the piece before this door. Emulator only: the card tier's outcomes are
    /// `Prompt` and `Owed`, so nothing on it reads a collected pick here.
    #[cfg(not(feature = "spyre-hw"))]
    pub(crate) fn minted_pick(&self) -> Option<&[u32]> {
        match self {
            #[cfg(not(feature = "spyre-hw"))]
            ChunkOut::Tokens(ts) => Some(ts),
            ChunkOut::Prompt => None,
            #[cfg(feature = "spyre-hw")]
            ChunkOut::Owed(_) => None,
        }
    }
}

/// ⭐ WHAT A STEP'S PROMPT-FINAL PICK IS when the step returns — the shared door's own
/// answer shape. The emulator tier collected its pick synchronously and answers
/// [`StepPick::Token`]; the card tier owes it: the prompt-final launch's read is
/// deferred, its piece rides the executor's step queue, and the caller's output defers
/// on the piece's cell.
pub(crate) enum StepPick {
    /// The step owes no token: a mid-prompt chunk.
    None,
    /// The step's token, collected — the step waited its own pick. Emulator
    /// only: the card tier never waits a pick (it owes the piece).
    #[cfg(not(feature = "spyre-hw"))]
    Token(u32),
    /// The step's token is OWED: the piece rides the executor's queue. Boxed — the piece
    /// is the enum's whole weight, paid once per step, never per token.
    #[cfg(feature = "spyre-hw")]
    Owed(Box<crate::card::exec::InFlightStep>),
}

/// ⭐ ONE ROW'S LAW FOR THE SEAT — what the launch's row `r` draws. `None` is
/// the greedy law: the seed quadruple `(0, 0, 0, 0)` and `inv_t = 1.0`,
/// byte-identical to the bind this path staged before any request asked for a
/// draw. `Some` is that row's request's draw, and the STEP SALT is the row's
/// ABSOLUTE position — the one input that changes every launch of one
/// generation, so the same seed draws different noise (hence, past rare ties,
/// a different token) at every position instead of repeating one pick.
pub(crate) struct RowLaw<'a> {
    pub(crate) draw: Option<&'a SampledDraw>,
    /// The row's absolute position — its request's own token count at this
    /// step, or (a prompt chunk's pad row) the last real position it clamps to.
    pub(crate) step: u32,
}

/// ⭐ ONE ROW'S SEAT WORDS — the seed quadruple `(seed_lo, seed_hi, step, gate)`
/// and the inv_t this row's law stages, the ONE statement of the law both tiers
/// lay out (the card scatters these values to its stick heads, the emulator
/// packs them `[rows, 1]` at its views' own dtypes). Greedy is `(0, 0, 0, 0)`
/// and 1.0 — the gate word's 0 IS the greedy law, `score = x·inv_t + g·0`. A
/// sampled row is its request's seed split low/high, the row's step salt, the
/// gate's 1 and the draw's own 1/T. ⭐ NO ROW WORD (the P3 localization ruling):
/// the key is request-owned alone, identical in every row.
pub(crate) fn seat_words(law: &RowLaw<'_>) -> ([i32; 4], f32) {
    match law.draw {
        Some(d) => (
            [
                d.seed as u32 as i32,
                (d.seed >> 32) as u32 as i32,
                law.step as i32,
                1,
            ],
            d.inv_t,
        ),
        None => ([0; 4], 1.0),
    }
}

/// The idx stick's word for row `r`, checked into a token — both tiers' readbacks
/// end here. A negative word is not a token id, and one at or past `vocab` is the
/// sampler's sentinel (the all-NaN row's answer), never a token.
pub(crate) fn token_of_idx_word(r: usize, w: i32, vocab: usize) -> ExecutorResult<u32> {
    let t = u32::try_from(w).map_err(|_| {
        werr(format!(
            "the token draw for row {r} is {w} — not a token id"
        ))
    })?;
    if t as usize >= vocab {
        let why = if t == scratchy_subtile::subtile_ir::sample_sentinel(vocab as u32) {
            "the sampler's all-NaN sentinel: the logits row was NaN in every lane"
        } else {
            "not the all-NaN sentinel either, so no answer the sampler gives"
        };
        return Err(werr(format!(
            "the token draw for row {r} is {t} >= vocab {vocab} — {why}. \
             Refusing rather than serving a token that does not exist."
        )));
    }
    Ok(t)
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

/// Pick the smallest decode rung that reaches `need` positions ([`BundleMeta::reach`]).
/// `buckets` is ascending by reach and non-empty; if `need` exceeds every rung, return the
/// largest so the caller's `forward_chunk` raises the precise over-capacity error.
pub(crate) fn select_decode(buckets: &[BundleMeta], need: usize) -> &BundleMeta {
    buckets
        .iter()
        .find(|b| b.reach() >= need)
        .unwrap_or_else(|| buckets.last().expect("at least one decode bucket"))
}

/// Forward `n` positions from `start`, then greedily sample iff this completes
/// the prompt or is a decode step. n==1 (generation) uses the cheap m=1 DECODE
/// bundle; n>1 (prompt) uses the batched m=M PREFILL bundle (chunked by m_cap),
/// falling back to the decode bundle if no prefill bundle was emitted.
///
/// ⭐ The answer is a [`StepPick`]: the emulator tier collected its pick and answers
/// the token; the card tier owes it — the prompt-final launch's read is deferred, its
/// piece rides the executor's step queue, and the caller's output defers on the
/// piece's cell. Nothing in this call waits on the card.
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
) -> ExecutorResult<StepPick> {
    if n == 0 {
        return Ok(StepPick::None);
    }
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
    // A decode step runs on the smallest rung that reaches the last position this step reaches,
    // so a short context walks the pages it fills rather than the whole pool. A prompt runs its
    // chunks on the prefill rungs (each chunk on the smallest that reaches it, below); `b` is the
    // chunk width's source and the card's fixed meta.
    let b: &BundleMeta = if n == 1 {
        select_decode(&model.decode, start + n)
    } else if let Some(p) = model.prefill.first() {
        p
    } else {
        select_decode(&model.decode, start + n)
    };
    // ⭐ THE CARD'S PROMPT CHUNK LOOP CHUNKS FOR ITSELF, over its two ladders
    // (`superdsc_forward_chunk`), so a step's tokens go in one call.
    #[cfg(feature = "spyre-hw")]
    let chunk_w = n.max(1);
    #[cfg(not(feature = "spyre-hw"))]
    let chunk_w = b.m_cap;
    // ⭐ A LONE TOKEN IS THE DECODE BUNDLE'S STEP — main's law. The card's executor routes one
    // to the batched decode arm before this call (`solo_step_is_decode`), so on the card this
    // arm meets only a lone MID-PROMPT tail (forwarded, nothing owed); the emulator's
    // synchronous tier runs the transition token on the m=1 decode bundle and collects its
    // pick. The chunk's own outcome states whether it picked ([`ChunkOut`]) — nothing here
    // derives where the prompt ends.
    if n == 1 {
        let out = forward_chunk(
            &sh,
            &mut model.session,
            b,
            model.batched_prefill.as_ref(),
            Chunk {
                state: req,
                id,
                tokens,
                start,
                toks: &toks[..1],
            },
        )?;
        return Ok(match out {
            #[cfg(feature = "spyre-hw")]
            ChunkOut::Owed(piece) => StepPick::Owed(piece),
            #[cfg(not(feature = "spyre-hw"))]
            ChunkOut::Tokens(ts) => {
                let tok = *ts.last().ok_or_else(|| {
                    werr("the pick step's decode launch picked no token".to_string())
                })?;
                tokens.push_generated(id, tok);
                StepPick::Token(tok)
            }
            ChunkOut::Prompt => StepPick::None,
        });
    }
    // ⭐ THE PROMPT'S CHUNKS, AND THE CHUNK'S OWN OUTCOME IS THE FINISH LAW — the one statement
    // of it. A chunk that finishes the prompt returns its owed pick ([`ChunkOut::Owed`]: the
    // prompt-final chunk's minted sampler — the general mint, every prefill rung carries the
    // draw — picks the first generated token ON THE CARD from the folded lm-head row, and the
    // idx stick is read back exactly like a decode launch's, deferred onto the executor's
    // queue); a mid-prompt chunk forwards every token it was given and picks nothing (its
    // suffix never runs). This loop derives nothing about where the prompt ends — the chunk
    // that ran to it says so, so there is nothing to reconcile and no refusal to raise when
    // two derivations disagree.
    let mut pos = start;
    #[cfg(not(feature = "spyre-hw"))]
    let mut prompt_pick: Option<u32> = None;
    #[cfg(feature = "spyre-hw")]
    let mut prompt_owed: Option<Box<crate::card::exec::InFlightStep>> = None;
    for chunk in toks.chunks(chunk_w) {
        // ⭐ A PROMPT CHUNK RUNS ON THE SMALLEST PREFILL RUNG THAT REACHES IT. A rung's
        // attention walks the request's pages through the page table and the step's mask
        // stack says which keys each row attends, so a chunk over a cached prefix attends
        // that prefix exactly as the decode bundle would — as long as the prefix and the
        // chunk sit inside the blocks the rung walks (`PagedStep::reach`). The ladder tops
        // at the whole pool, so some rung reaches every chunk the pool can hold; with no
        // prefill bundle baked, the prompt walks the DECODE bundle one token at a time. The
        // staging refuses a step past its bundle's reach, so neither choice can silently
        // drop keys.
        #[cfg(not(feature = "spyre-hw"))]
        let rung = select_prefill(&model.prefill, pos + chunk.len());
        #[cfg(not(feature = "spyre-hw"))]
        if b.m_cap > 1 && rung.is_none() {
            for one in chunk.chunks(1) {
                let db = select_decode(&model.decode, pos + 1);
                if let Some(ts) = forward_chunk(
                    &sh,
                    &mut model.session,
                    db,
                    model.batched_prefill.as_ref(),
                    Chunk {
                        state: req,
                        id,
                        tokens,
                        start: pos,
                        toks: one,
                    },
                )?
                .minted_pick()
                {
                    prompt_pick = ts.last().copied();
                }
                pos += 1;
            }
            continue;
        }
        #[cfg(not(feature = "spyre-hw"))]
        let b = rung.unwrap_or(b);
        let out = forward_chunk(
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
        )?;
        // ⭐ THE CARD'S CHUNK OWES ITS PICK (the emulator's collected it): the piece
        // rides the executor's step queue and this loop keeps staging the prompt's
        // next chunk behind it — nothing here waits on the card.
        // A mid-prompt chunk (`ChunkOut::Prompt`) forwards — the card tier never
        // collects in-call, so nothing is owed for it.
        #[cfg(feature = "spyre-hw")]
        if let ChunkOut::Owed(piece) = out {
            prompt_owed = Some(piece);
        }
        #[cfg(not(feature = "spyre-hw"))]
        if let Some(ts) = out.minted_pick() {
            prompt_pick = ts.last().copied();
        }
        pos += chunk.len();
    }
    // ⭐ THE CHUNK'S OUTCOME IS THE STEP'S — the piece is the prompt-final pick, owed through
    // the executor's queue; no piece means every chunk forwarded: a mid-prompt wave, picking
    // nothing. The card tier's answer is the queue's to give, never this call's.
    #[cfg(feature = "spyre-hw")]
    {
        Ok(match prompt_owed {
            Some(piece) => StepPick::Owed(piece),
            None => StepPick::None,
        })
    }
    #[cfg(not(feature = "spyre-hw"))]
    {
        // The prompt-final chunk's collected pick — `None` is every chunk of a mid-prompt
        // step (the tier's forward answers `Prompt` for a chunk that does not finish, its
        // one statement of the finish law); a finishing chunk that minted no draw is the
        // FORWARD's own refusal, never a silent skip here.
        match prompt_pick {
            Some(tok) => {
                tokens.push_generated(id, tok);
                Ok(StepPick::Token(tok))
            }
            None => Ok(StepPick::None),
        }
    }
}

/// The idx stick's own decode law, shared by the sync readback and the deferred
/// collect so the two cannot drift. `idx_row_words` is the seat's own typed fact
/// — the stick's placed row size — so both the addressing width and (at the
/// caller) the readback byte size are the placement's own law, never a literal.
/// A word at or past the vocab is refused: the sampler answers an all-NaN logits
/// row with `sample_sentinel` (#666), and the card's draw is the only pick, so
/// every path refuses that row alike.
// The `test` arm makes the fn exist under `spyre-emu`, which has no `spyre-hw`;
// CI runs `idx_stick_tests` in that shape (the spyre-triton job).
#[cfg(any(test, feature = "spyre-hw"))]
pub(crate) fn decode_idx_stick(
    bytes: &[u8],
    rows: usize,
    idx_row_words: usize,
    vocab: usize,
) -> ExecutorResult<Vec<u32>> {
    // The idx stick's own format decides the stride — the same law every device
    // tensor's host-side address takes (`dev_off_stk`), never a guessed row pitch.
    // Words are read STRAIGHT off the landed bytes — no intermediate word buffer,
    // the same zero-alloc law the visitor door that hands these bytes in obeys.
    let stk = <ktir_superdsc::superdsc_opspec::SenUint32 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
        as usize;
    (0..rows)
        .map(|r| {
            let off =
                ktir_superdsc::sdsc_abstract::dev_off_stk(&[rows, idx_row_words], 1, &[r, 0], stk);
            let word = bytes
                .get(off * 4..off * 4 + 4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            token_of_idx_word(r, word.unwrap_or(-1), vocab)
        })
        .collect()
}

#[cfg(test)]
mod idx_stick_tests {
    use super::decode_idx_stick;
    use scratchy_subtile::subtile_ir::sample_sentinel;

    /// The idx stick's placed row width in these fixtures.
    const ROW_WORDS: usize = 64;

    /// An idx stick as the card writes it: row r's pick in every one of its `ROW_WORDS`
    /// lanes, each lane at its SenUint32 stick address (`dev_off_stk`), the layout the
    /// readback reads.
    fn stick(picks: &[u32]) -> Vec<u8> {
        let rows = picks.len();
        let stk = <ktir_superdsc::superdsc_opspec::SenUint32 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
            as usize;
        let mut words = vec![0i32; rows * ROW_WORDS];
        for (r, &p) in picks.iter().enumerate() {
            for c in 0..ROW_WORDS {
                let off =
                    ktir_superdsc::sdsc_abstract::dev_off_stk(&[rows, ROW_WORDS], 1, &[r, c], stk);
                words[off] = p as i32;
            }
        }
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The readback passes real picks and refuses the sampler's all-NaN sentinel (#666), at
    /// granite's vocab, solo and on a batched row; any other word past the vocab refuses too.
    #[test]
    fn the_readback_refuses_the_all_nan_sentinel_and_passes_a_real_pick() {
        let vocab = 49155u32;
        let v = vocab as usize;
        assert_eq!(
            decode_idx_stick(&stick(&[12358, 0]), 2, ROW_WORDS, v).expect("two real picks"),
            vec![12358, 0]
        );
        for picks in [
            vec![sample_sentinel(vocab)],
            vec![12358, sample_sentinel(vocab)],
        ] {
            let err = decode_idx_stick(&stick(&picks), picks.len(), ROW_WORDS, v)
                .expect_err("the sentinel is never a token");
            assert!(err.to_string().contains("all-NaN sentinel:"), "{err}");
        }
        let err = decode_idx_stick(&stick(&[vocab]), 1, ROW_WORDS, v).expect_err("past the vocab");
        assert!(
            err.to_string().contains("not the all-NaN sentinel"),
            "{err}"
        );
    }
}
