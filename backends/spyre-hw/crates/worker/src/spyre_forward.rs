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

/// ⭐ WHAT ONE CHUNK'S FORWARD PRODUCED — the tokens the host drew from its answer rows' logits.
/// One enum (not a bare `Vec`) because the shape is a fact of the step the caller must not be able
/// to confuse with the model's other outs.
pub(crate) enum ChunkOut {
    /// A prompt-only forward: a mid-prompt chunk, whose answer row nobody reads.
    Prompt,
    /// One drawn token per answer row of the chunk, in row order.
    Tokens(Vec<u32>),
}

impl ChunkOut {
    /// The chunk's drawn tokens — `None` for a chunk that answered nothing.
    pub(crate) fn minted_pick(&self) -> Option<&[u32]> {
        match self {
            ChunkOut::Tokens(ts) => Some(ts),
            ChunkOut::Prompt => None,
        }
    }
}

/// ⭐ ONE ROW'S DRAW LAW — what an answer row draws. `None` is the greedy law. `Some` is that row's
/// request's draw, and the STEP SALT is the row's ABSOLUTE position — the one input that changes
/// every launch of one generation, so the same seed draws afresh at every position instead of
/// repeating one pick.
pub(crate) struct RowLaw<'a> {
    pub(crate) draw: Option<&'a SampledDraw>,
    /// The row's absolute position — its request's own token count at this
    /// step, or (a prompt chunk's pad row) the last real position it clamps to.
    pub(crate) step: u32,
}

/// ⭐ THE HOST DRAW — row `r`'s token from its logits under `law`: the first arg-max at the
/// greedy law, else the Gumbel-max draw from `softmax(logits · inv_t)` — its noise a pure function
/// of the request's seed, the row's step and the lane, so a seeded request replays and the same
/// seed draws afresh at every position. A row with no logit but NaN answers nothing, and is refused
/// by name rather than served a token the model never chose.
pub(crate) fn draw_token(
    r: usize,
    logits: impl IntoIterator<Item = f32>,
    law: &RowLaw<'_>,
) -> ExecutorResult<u32> {
    let score = |lane: usize, x: f32| -> f64 {
        match law.draw {
            None => f64::from(x),
            Some(d) => f64::from(x) * f64::from(d.inv_t) + gumbel(d.seed, law.step, lane as u32),
        }
    };
    logits
        .into_iter()
        .enumerate()
        .filter(|(_, x)| !x.is_nan())
        .map(|(lane, x)| (lane, score(lane, x)))
        .fold(None, |best: Option<(usize, f64)>, (lane, s)| match best {
            Some((_, b)) if b >= s => best,
            _ => Some((lane, s)),
        })
        .map(|(lane, _)| lane as u32)
        .ok_or_else(|| {
            werr(format!(
                "row {r}'s logits are NaN in every lane — no token to draw"
            ))
        })
}

/// Gumbel noise `-ln(-ln u)` for `lane` of a row at `step` under `seed`, `u ∈ (0, 1)` the
/// splitmix64 finalizer of the three.
fn gumbel(seed: u64, step: u32, lane: u32) -> f64 {
    let mut z =
        seed ^ ((u64::from(step) << 32) | u64::from(lane)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    -(-u.ln()).ln()
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
) -> ExecutorResult<Option<u32>> {
    if n == 0 {
        return Ok(None);
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
        if let Some(ts) = forward_chunk(
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
        )?
        .minted_pick()
        {
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
            werr("the prompt-final chunk picked no token — its minted draw never ran".to_string())
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
        )?
        .minted_pick()
        {
            Some(ts) => *ts
                .last()
                .ok_or_else(|| werr("the pick step's decode launch picked no token".to_string()))?,
            None => {
                return Err(werr(
                    "the pick step ran no decode launch — this build has no device sampler"
                        .to_string(),
                ));
            }
        }
    };
    tokens.push_generated(id, tok);
    Ok(Some(tok))
}

#[cfg(test)]
mod draw_tests {
    use super::{RowLaw, draw_token};
    use crate::spyre_types::SampledDraw;

    /// The greedy law answers the FIRST arg-max, past NaN lanes; a row of NaN answers nothing.
    #[test]
    fn the_greedy_draw_is_the_first_arg_max() {
        let greedy = RowLaw {
            draw: None,
            step: 7,
        };
        assert_eq!(
            draw_token(0, [0.5, 2.0, f32::NAN, 2.0, -1.0], &greedy).unwrap(),
            1
        );
        assert_eq!(draw_token(0, [f32::NAN, -3.0], &greedy).unwrap(), 1);
        assert!(draw_token(0, [f32::NAN; 4], &greedy).is_err());
    }

    /// A sampled draw replays under its seed and step, moves with the step, and follows the
    /// logits: a lane far above the rest wins every draw, and lanes of equal logits are each
    /// drawn about as often.
    #[test]
    fn a_sampled_draw_is_a_function_of_seed_step_and_logits() {
        let d = SampledDraw {
            seed: 0x1234_5678_9abc_def0,
            inv_t: 1.0,
        };
        let law = |step| RowLaw {
            draw: Some(&d),
            step,
        };
        let flat = vec![0.0f32; 4];
        let picks: Vec<u32> = (0..4000)
            .map(|s| draw_token(0, flat.iter().copied(), &law(s)).unwrap())
            .collect();
        assert_eq!(
            picks,
            (0..4000)
                .map(|s| draw_token(0, flat.iter().copied(), &law(s)).unwrap())
                .collect::<Vec<_>>()
        );
        for lane in 0..4u32 {
            let n = picks.iter().filter(|&&p| p == lane).count();
            assert!(
                (800..1200).contains(&n),
                "lane {lane} drawn {n} of 4000 times"
            );
        }
        let peaked = [0.0f32, 40.0, 0.0, 0.0];
        assert!((0..200).all(|s| draw_token(0, peaked, &law(s)).unwrap() == 1));
    }
}
