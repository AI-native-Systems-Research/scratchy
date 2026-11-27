// SPDX-License-Identifier: Apache-2.0
//! The card's step queue: the in-flight steps and their drain.

use std::collections::HashMap;

use scratchy_serving_worker::error::ExecutorResult;

use crate::spyre_forward::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;

/// CONTINUOUS BATCHING (#45) execute_model for the sendnn PAGED session. Consumes
/// the scheduler's batched `SchedulerOutput` like a GPU target's worker:
///   • PREFILL (scheduled_new_reqs + cached chunks with n>1): each request's chunk
///     runs as its OWN multi-row `predict_perrow` over THAT request's scheduler
///     blocks (rows = consecutive prompt positions, causal self-attention).
///   • DECODE (cached reqs with n==1): ALL decode requests are PACKED into the mq
///     query rows of ONE `predict_perrow` (sub-batched at mq when B>mq) — the CB
///     win. Each row attends only its own request's blocks (PROVEN: per-row
///     block_table honored; rows don't cross-contaminate).
impl SpyreWorker {
    /// ⭐ #32 (c4) — THE DRAIN'S ONE POP: collect one in-flight step and own its whole
    /// bookkeeping. The same body runs for every caller — the knock (`drain_ready`), the
    /// in-call drain of a driver with no drain door, and the entry-drain that empties the
    /// queue before the next step builds its launch — so there is one collect path, not one
    /// per driver. The step's cell is filled here (with the step's own error if the collect
    /// failed) so a parked resolver can never hang.
    pub(crate) fn drain_one(
        infl: InFlightStep,
        model: &mut Loaded,
        requests: &mut HashMap<String, ReqState>,
        input_batch: &mut scratchy_serving_engine::input_batch::InputBatch,
    ) -> ExecutorResult<Vec<u32>> {
        let InFlightStep {
            cell,
            sub,
            seat,
            owed,
        } = infl;
        // Only the session the step was submitted on can finish the read — each rung is its
        // own executor and stream, so the piece carries the seat as a value and the bundle
        // resolves it here, never by probing.
        let SendnnSession::SuperDsc(sb) = &mut model.session;
        let sess = sb.step_session(seat);
        // ⛔ ONE FILL, AFTER EVERY EXIT. The step's resolver parks on `cell` until it is filled, so a
        // collect that returned before filling it hung that resolver for good — and the serve loop
        // with it (#1030: a request finished by EOS, its overshoot step still in flight).
        let picked = Self::collect_rows(sess, &sub, owed, requests, input_batch);
        cell.fill(match &picked {
            Ok(out) => Ok(out.clone()),
            Err(e) => Err(scratchy_serving_engine::error::EngineError::Executor(
                e.to_string(),
            )),
        });
        picked
    }

    /// The collect itself: the step's picks and the bookkeeping the drain owns — which
    /// kind of step this is decides whose tokens they are. A row whose request is gone
    /// keeps its slot and skips the bookkeeping.
    ///
    /// ⭐ A FINISHED REQUEST CAN STILL HAVE A STEP IN FLIGHT. A deferred step submits while the one
    /// before it is still on the card (depth 2), so a request that finishes by EOS on step `j` has
    /// step `j+1` in flight; the next schedule names it in `finished_req_ids`, and
    /// `execute_model` drops it before that step is drained. The engine discards the overshoot
    /// token of a finished request, so the drain skips its bookkeeping rather than failing the step.
    pub(crate) fn collect_rows(
        sess: &mut SuperDscSession,
        sub: &crate::spyre_forward::SubmittedDecode,
        owed: Owed,
        requests: &mut HashMap<String, ReqState>,
        input_batch: &mut scratchy_serving_engine::input_batch::InputBatch,
    ) -> ExecutorResult<Vec<u32>> {
        let toks = collect_decode_tokens(sess, sub)?;
        match owed {
            Owed::Decode { rows } => {
                // A REQUEST'S TOKEN COMES BACK IN THE SLOT THE LAUNCH PUT IT IN — the slot was resolved
                // at submit and rides in `rows`, so the drain cannot re-derive (and mis-derive) it.
                // The sentinel check already happened at the decode law (`collect_decode_tokens`);
                // this guard is the slice's own bound.
                let n_req = rows.len();
                let mut out = Vec::with_capacity(n_req);
                for (id, start, slot) in rows {
                    let tok = toks.get(slot).copied().ok_or_else(|| {
                        werr(format!(
                            "decode batch: {} token(s) for {n_req} request(s) — the rung's sampler did \
                             not pick one per launch slot",
                            toks.len()
                        ))
                    })?;
                    // The token stays in its slot either way: the deferred output maps answer `i` to its
                    // `req_ids[i]` and the in-call drain reads slot `i`, so a dropped row would hand every
                    // later row its neighbour's token. A gone request's overshoot pick is discarded by the
                    // engine; it only skips the pool bookkeeping of a request that no longer exists.
                    // Every pick the drain owns, one line each: request, the position the row
                    // computed, the token id. The stream a request's reply is made of, stated as
                    // the fact the numerics bar argues over — WHERE a reply diverges between two
                    // cells names the step, and the step names the rung.
                    tracing::debug!("[spyre-worker] pick {id} at {start} = {tok}");
                    if requests.contains_key(&id) {
                        input_batch.push_generated(&id, tok);
                        input_batch.set_tokens_in_pool(&id, start + 1);
                    }
                    // ⭐ THE KV HISTORY WAS RECORDED AT SUBMIT (c4b) — the write slot is a position fact
                    // the host knew when it launched, and the next step's geometry reads it before this
                    // drain on a stable seating. The drain owns the PICK; the history was never its fact.
                    out.push(tok);
                }
                Ok(out)
            }
            Owed::Prompt { id } => {
                // ⭐ A PROMPT'S FIRST GENERATED TOKEN — the mint's consumed pick is the LAST baked
                // row's (the pad clamp makes the pad rows' picks the last real row's own). The
                // prompt's pool count was set at submit (the chunk loop's own KV-history law);
                // the drain owns only the pick.
                let tok = *toks.last().ok_or_else(|| {
                    werr(
                        "the prompt-final chunk picked no token — its minted draw never ran"
                            .to_string(),
                    )
                })?;
                tracing::debug!("[spyre-worker] pick {id} (prompt-final) = {tok}");
                if requests.contains_key(&id) {
                    input_batch.push_generated(&id, tok);
                }
                Ok(vec![tok])
            }
            Owed::Wave { rows } => {
                // ⭐ A WAVE'S PICKS, one per window — request `r`'s own answer from seat row `r`.
                // Every tail's pool count is the drain's (the wave stated it at its collect before
                // steps deferred; it drains through the same door now). A mid-prompt tail's pick
                // is computed and never read — the wave's own law.
                for t in rows {
                    input_batch.set_tokens_in_pool(&t.id, t.end);
                    if let Some(slot) = t.pick {
                        let tok = toks.get(slot).copied().ok_or_else(|| {
                            werr(format!(
                                "the packed wave picked {} token(s) — tail {}'s window {} has none, \
                                 and every window's answer is the readback's to carry",
                                toks.len(),
                                t.id,
                                slot
                            ))
                        })?;
                        tracing::debug!("[spyre-worker] pick {} (wave) = {tok}", t.id);
                        if requests.contains_key(&t.id) {
                            input_batch.push_generated(&t.id, tok);
                        }
                    }
                }
                Ok(toks)
            }
        }
    }
}

/// ⭐ #32 (c4) / §3.3 — ONE SUBMITTED, UNCOLLECTED STEP. Every kind of step a schedule
/// submits rides this queue in submission order: a decode batch, a prompt's final chunk,
/// a wave of tails. The drain pops it and owns the whole collect half — the fence
/// (`collect_decode_tokens`), the bookkeeping `owed` names, and the cell fill.
pub(crate) struct InFlightStep {
    /// The cell this step's answer parks on; filled by whichever drain pops this.
    pub(crate) cell: std::sync::Arc<scratchy_serving_engine::executor::DrainCell>,
    /// The submitted step: its deferred idx-stick read, its landing, the decode law's inputs.
    pub(crate) sub: crate::spyre_forward::SubmittedDecode,
    /// Which session the step was submitted on — resolved back to the session at the drain
    /// (`SuperDscBundle::step_session`), because only the submitting session can finish
    /// the read.
    pub(crate) seat: crate::card::types::StepSeat,
    /// What the step owes, and the bookkeeping the drain owns for it.
    pub(crate) owed: Owed,
}

impl InFlightStep {
    /// ⭐ HAS THIS STEP'S CARD WORK LANDED — the non-blocking ask the knock's drain
    /// makes before it collects. The park the drain would otherwise take lives in the
    /// read's finish, so the read's own peek handle (which cannot wait, BY TYPE) is
    /// exactly the fact "collecting this step now would not park": `fired` says the
    /// card has retired the step and the D2H is complete. A step that has not landed
    /// stays QUEUED — the knock's pump then feeds the card the Step arrived behind the
    /// knock, and only a card that stays busy past this landing is harvested late.
    pub(crate) fn landed(&self) -> bool {
        self.sub.read.signal().fired()
    }
}

/// ⭐ WHAT ONE SUBMITTED STEP OWES ITS CALLERS — the drain's bookkeeping, named by the
/// kind of step that submitted. The pieces ride [`InFlightStep`] in submission order.
pub(crate) enum Owed {
    /// A DECODE BATCH: per live row, in submit order, `(request id, position, launch
    /// slot)` — the drain owns each row's pick and pool bookkeeping.
    Decode { rows: Vec<(String, usize, usize)> },
    /// A PROMPT-FINAL PICK: one request's first generated token. The prompt's pool count
    /// was set at submit (the chunk loop's KV-history law); the drain owns only the pick.
    Prompt { id: String },
    /// A WAVE'S TAILS: per tail, its request, its pool end, and its window's pick (a
    /// mid-prompt tail's is computed and never read — `None`).
    Wave { rows: Vec<OwedTail> },
}

/// One tail of an owed wave — the drain's per-request bookkeeping for it.
pub(crate) struct OwedTail {
    pub(crate) id: String,
    /// The pool count the tail's forward reached (set at the drain, the wave's own law).
    pub(crate) end: usize,
    /// The window whose pick is this tail's — `None` for a mid-prompt tail.
    pub(crate) pick: Option<usize>,
}

/// ⭐ ONE REQUEST'S OWED ANSWER within a step's output — where its token lands and how to
/// take it from its piece's cell once the drain has filled it. The step's exit owns these:
/// the deferred output's resolver waits them, and the in-call drain reads them after the
/// same pop.
pub(crate) struct OwedRow {
    /// Where the request's token lands in the step's output order (`order`/`sampled`).
    pub(crate) index: usize,
    /// The request the token belongs to — the deferred output's own id list.
    pub(crate) id: String,
    /// The piece's cell: filled by whichever drain pops the piece.
    pub(crate) cell: std::sync::Arc<scratchy_serving_engine::executor::DrainCell>,
    /// Which pick of the piece is this request's.
    pub(crate) pick: OwedPick,
}

/// Which pick of a piece's landed answer is a row's own token.
pub(crate) enum OwedPick {
    /// A decode batch's row: the launch slot the request's token comes back in.
    Slot(usize),
    /// A piece that answers ONE token (a prompt-final pick): the piece's own answer.
    One,
}

impl OwedRow {
    /// The row's token, from its piece's landed answer: waits the piece's cell (filled
    /// once the drain popped it) and takes the row's own pick of it.
    pub(crate) fn token(&self) -> Result<u32, String> {
        let toks = self.cell.wait().map_err(|e| e.to_string())?;
        match self.pick {
            OwedPick::Slot(slot) => toks.get(slot).copied().ok_or_else(|| {
                format!(
                    "the step picked {} token(s) — request {}'s launch slot {slot} has none",
                    toks.len(),
                    self.id
                )
            }),
            OwedPick::One => toks
                .last()
                .copied()
                .ok_or_else(|| format!("the step owed {} a token and picked none", self.id)),
        }
    }
}

/// ⭐ QUEUE ONE OWED PROMPT PICK and lay its row in the step's output. The token is not
/// sampled yet — its cell is — and the step's exit (the deferred output, or the in-call
/// drain through the same pop) fills it. ONE door for every site a prompt's final chunk
/// lands at.
pub(crate) fn owe_prompt(
    steps: &mut std::collections::VecDeque<InFlightStep>,
    owed: &mut Vec<OwedRow>,
    sampled: &mut Vec<Option<u32>>,
    piece: InFlightStep,
    id: &str,
) {
    let cell = piece.cell.clone();
    sampled.push(None);
    owed.push(OwedRow {
        index: sampled.len() - 1,
        id: id.to_string(),
        cell,
        pick: OwedPick::One,
    });
    steps.push_back(piece);
}
