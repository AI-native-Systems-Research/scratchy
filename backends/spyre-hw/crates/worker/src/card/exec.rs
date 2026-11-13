// SPDX-License-Identifier: Apache-2.0
//! The card's decode queue: the in-flight steps and their drain.

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
    /// bookkeeping. The same body runs for every caller — the knock (`drain_until`), the
    /// in-call drain of a schedule that could not defer, and the entry-drain that empties the
    /// queue before the next step builds its launch — so there is one collect path, not one
    /// per driver. The step's cell is filled here (with the step's own error if the collect
    /// failed) so a parked resolver can never hang.
    pub(crate) fn drain_one(
        infl: InFlightDecode,
        model: &mut Loaded,
        requests: &mut HashMap<String, ReqState>,
        input_batch: &mut scratchy_serving_engine::input_batch::InputBatch,
    ) -> ExecutorResult<Vec<u32>> {
        let InFlightDecode {
            cell,
            sub,
            rung,
            rows,
        } = infl;
        // Only the submitting rung's session can finish the read — each rung owns its own
        // executor and stream, and the W=1 seat is the owner session itself.
        let SendnnSession::SuperDsc(sb) = &mut model.session;
        let sess = sb.decode_seat(rung).sess;
        // ⛔ ONE FILL, AFTER EVERY EXIT. The step's resolver parks on `cell` until it is filled, so a
        // collect that returned before filling it hung that resolver for good — and the serve loop
        // with it (#1030: a request finished by EOS, its overshoot step still in flight).
        let picked = Self::collect_rows(sess, &sub, rows, requests, input_batch);
        cell.fill(match &picked {
            Ok(out) => Ok(out.clone()),
            Err(e) => Err(scratchy_serving_engine::error::EngineError::Executor(
                e.to_string(),
            )),
        });
        picked
    }

    /// The collect itself: the step's picks, one per launch slot, and each live row's pool
    /// bookkeeping. A row whose request is gone keeps its slot and skips the bookkeeping.
    ///
    /// ⭐ A FINISHED REQUEST CAN STILL HAVE A STEP IN FLIGHT. A deferred step submits while the one
    /// before it is still on the card (depth 2), so a request that finishes by EOS on step `j` has
    /// step `j+1` in flight; the next schedule names it in `finished_req_ids`, and
    /// `execute_model` drops it before that step is drained. The engine discards the overshoot
    /// token of a finished request, so the drain skips its bookkeeping rather than failing the step.
    pub(crate) fn collect_rows(
        sess: &mut SuperDscSession,
        sub: &crate::spyre_forward::SubmittedDecode,
        rows: Vec<(String, usize, usize)>,
        requests: &mut HashMap<String, ReqState>,
        input_batch: &mut scratchy_serving_engine::input_batch::InputBatch,
    ) -> ExecutorResult<Vec<u32>> {
        let toks = collect_decode_tokens(sess, sub)?;
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
}

/// ⭐ #32 (c4) — ONE SUBMITTED, UNCOLLECTED BATCHED DECODE STEP. The drain pops this and owns
/// the whole collect half: the fence (`collect_decode_tokens`), the per-request bookkeeping,
/// and the cell fill. `rows` is per LIVE request in `decode_ids` order — the request, its pool
/// start, and the launch slot its token comes back in — so the drain never re-derives the
/// step's slot map.
pub(crate) struct InFlightDecode {
    /// The cell this step's deferred output parks on; filled by whichever drain pops this.
    pub(crate) cell: std::sync::Arc<scratchy_serving_engine::executor::DrainCell>,
    /// The submitted step: its deferred idx-stick read and the decode law's inputs.
    pub(crate) sub: crate::spyre_forward::SubmittedDecode,
    /// Which rung's session owns this step's executor — every rung is its own session/stream,
    /// and only the submitting session can finish the read. `Owner` = the W=1 decode session.
    pub(crate) rung: crate::spyre_types::RungId,
    /// Per live row, in `decode_ids` order: `(request id, position, launch slot)`.
    pub(crate) rows: Vec<(String, usize, usize)>,
}
