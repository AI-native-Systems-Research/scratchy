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
    ///
    /// ⭐ THE KV HISTORY IS NEVER THE DRAIN'S FACT — every submit writes the pool count its step
    /// reached (the c4b law: a position fact the host knew when it launched, read by the next
    /// step's geometry before any drain, on a stable seating). The drain owns PICKS: the
    /// launch's typed [`LaunchPicks`] (one per slot, the bounds law stated once at the mint)
    /// and each row's bookkeeping write.
    pub(crate) fn collect_rows(
        sess: &mut SuperDscSession,
        sub: &crate::spyre_forward::SubmittedDecode,
        owed: Owed,
        requests: &mut HashMap<String, ReqState>,
        input_batch: &mut scratchy_serving_engine::input_batch::InputBatch,
    ) -> ExecutorResult<Vec<u32>> {
        let picks = collect_decode_tokens(sess, sub)?;
        match owed {
            Owed::Decode { rows } => {
                // A REQUEST'S TOKEN COMES BACK IN THE SLOT THE LAUNCH PUT IT IN — the slot was
                // minted at submit against the launch's width and rides in `rows`, so the
                // drain cannot re-derive (and mis-derive) it, and the pick is total.
                let n_req = rows.len();
                let mut out = Vec::with_capacity(n_req);
                for (id, slot) in rows {
                    let tok = picks.pick(slot);
                    // The token stays in its slot either way: the deferred output maps answer `i` to its
                    // `req_ids[i]` and the in-call drain reads slot `i`, so a dropped row would hand every
                    // later row its neighbour's token. A gone request's overshoot pick is discarded by the
                    // engine; it only skips the pool bookkeeping of a request that no longer exists.
                    // Every pick the drain owns, one line each: request, the token id. The stream a
                    // request's reply is made of, stated as the fact the numerics bar argues over —
                    // WHERE a reply diverges between two cells names the step, and the step names the rung.
                    tracing::debug!("[spyre-worker] pick {id} = {tok}");
                    if requests.contains_key(&id) {
                        input_batch.push_generated(&id, tok);
                    }
                    out.push(tok);
                }
                Ok(out)
            }
            Owed::Prompt { id } => {
                // ⭐ A PROMPT'S FIRST GENERATED TOKEN — the mint's consumed pick is the last
                // baked row's ([`prompt_final_pick`], the one rule). The prompt's pool count
                // was set at submit (the chunk loop's own KV-history law); the drain owns
                // only the pick.
                let tok = picks.prompt_final();
                tracing::debug!("[spyre-worker] pick {id} (prompt-final) = {tok}");
                if requests.contains_key(&id) {
                    input_batch.push_generated(&id, tok);
                }
                Ok(vec![tok])
            }
            Owed::Wave { rows } => {
                // ⭐ A WAVE'S PICKS, one per window — request `r`'s own answer from its
                // window's slot, minted at submit against the wave's window count. Every
                // tail's pool count was set at submit, with the wave's; a mid-prompt tail's
                // pick is computed and never read — the wave's own law.
                for t in rows {
                    if let Some(slot) = t.pick {
                        let tok = picks.pick(slot);
                        tracing::debug!("[spyre-worker] pick {} (wave) = {tok}", t.id);
                        if requests.contains_key(&t.id) {
                            input_batch.push_generated(&t.id, tok);
                        }
                    }
                }
                Ok(picks.into_tokens())
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

/// ⭐ THE LAUNCH'S OWN TYPED PICKS — the readback decoded once, ONE TOKEN PER
/// SLOT of the launch's seat width. `decode_idx_stick` states that law
/// structurally (it answers exactly one token per row it is given, and the row
/// count is the launch's own `SeatWidth`), so a [`SeatSlot`] minted against the
/// width picks TOTAL: no reader of a pick re-checks the sampler's law, and the
/// prompt-final rule lives here as the one function every prompt-final reader
/// calls.
pub(crate) struct LaunchPicks {
    toks: Vec<u32>,
}

/// A slot of a launch's seat — minted only against the width that laid it
/// ([`SeatSlot::mint`], called by the submit as it lays each row and names each
/// window), so the pick against it is total and a slot of one launch cannot be
/// spent against another's count by accident.
#[derive(Clone, Copy)]
pub(crate) struct SeatSlot {
    slot: usize,
}

impl SeatSlot {
    /// Mint against the launch's width — the ONE statement of the bounds law.
    /// `None` names a wiring bug at the caller that laid the rows.
    pub(crate) fn mint(i: usize, width: usize) -> Option<Self> {
        (i < width).then_some(Self { slot: i })
    }
}

/// ⭐ THE PROMPT-FINAL PICK IS THE LAST BAKED ROW'S — the pad clamp makes the pad
/// rows' picks the last real row's own, so the consumed pick is the last slot's.
/// ONE function: the drain's bookkeeping and the owed row's output both call it,
/// never restate it.
pub(crate) fn prompt_final_pick(toks: &[u32]) -> u32 {
    *toks
        .last()
        .expect("the prompt-final launch answered no pick — its minted draw never ran")
}

impl LaunchPicks {
    /// Mint the readback against the launch's width — `decode_idx_stick` answers
    /// one token per row it is given, and `rows` is that count (the submitted
    /// step's own [`crate::spyre_forward::SeatWidth`]), so the width is the
    /// readback's own law, never a second check.
    pub(crate) fn mint(toks: Vec<u32>, rows: usize) -> Self {
        debug_assert_eq!(
            toks.len(),
            rows,
            "decode_idx_stick answers one token per launch row"
        );
        Self { toks }
    }

    /// The slot's token — TOTAL: the slot was minted against this launch's width
    /// and the readback answered one token per row of it. The two mints are the
    /// whole law; this index cannot be out of bounds unless they disagree.
    pub(crate) fn pick(&self, s: SeatSlot) -> u32 {
        self.toks[s.slot]
    }

    /// The prompt-final pick — [`prompt_final_pick`], the one rule.
    pub(crate) fn prompt_final(&self) -> u32 {
        prompt_final_pick(&self.toks)
    }

    /// The picks as the piece's cell answer — one token per slot, in the
    /// launch's own order, so an owed row's minted slot picks its own.
    pub(crate) fn into_tokens(self) -> Vec<u32> {
        self.toks
    }
}

/// ⭐ WHAT ONE SUBMITTED STEP OWES ITS CALLERS — the drain's bookkeeping, named by the
/// kind of step that submitted. The pieces ride [`InFlightStep`] in submission order.
/// The KV history is NOT the drain's fact for any kind: every submit writes the pool
/// count it reached (the c4b law), so the drain owns only picks.
pub(crate) enum Owed {
    /// A DECODE BATCH: per live row, in submit order, `(request id, launch slot)`
    /// — the drain owns each row's pick.
    Decode { rows: Vec<(String, SeatSlot)> },
    /// A PROMPT-FINAL PICK: one request's first generated token. The prompt's pool count
    /// was set at submit (the chunk loop's KV-history law); the drain owns only the pick.
    Prompt { id: String },
    /// A WAVE'S TAILS: per tail, its request and its window's pick (a mid-prompt
    /// tail's is computed and never read — `None`). The tail's pool count was set
    /// at submit, with the wave's.
    Wave { rows: Vec<OwedTail> },
}

/// One tail of an owed wave — the drain's per-request bookkeeping for it.
pub(crate) struct OwedTail {
    pub(crate) id: String,
    /// The window whose pick is this tail's — `None` for a mid-prompt tail.
    pub(crate) pick: Option<SeatSlot>,
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
    /// A launch slot the request's token comes back in — minted at submit against
    /// the launch's width (a decode batch's row, a wave's window).
    Slot(SeatSlot),
    /// A piece that answers ONE token (a prompt-final pick): the piece's own answer.
    One,
}

impl OwedPick {
    /// The row's token out of its piece's landed answer — TOTAL, the same two mints
    /// the drain's [`LaunchPicks`] rests on: the slot was minted at submit against
    /// the piece's unit count, and the drain minted the answer one token per unit
    /// of the same count. The prompt-final rule is [`prompt_final_pick`], the one
    /// function — never restated here.
    pub(crate) fn pick(&self, answer: &[u32]) -> u32 {
        match self {
            OwedPick::Slot(s) => answer[s.slot],
            OwedPick::One => prompt_final_pick(answer),
        }
    }
}

impl OwedRow {
    /// The row's token, from its piece's landed answer: waits the piece's cell (filled
    /// once the drain popped it) and takes a share of it — the answer is every row's,
    /// so this row's read leaves the cell filled for the rows behind it — then picks
    /// this row's own token out of the share.
    pub(crate) fn token(&self) -> Result<u32, String> {
        let answer = self.cell.wait();
        let toks = match &*answer {
            Ok(toks) => toks,
            Err(e) => return Err(e.to_string()),
        };
        Ok(self.pick.pick(toks))
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
