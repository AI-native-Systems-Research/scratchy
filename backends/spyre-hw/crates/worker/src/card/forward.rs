// SPDX-License-Identifier: Apache-2.0
//! The card's forward: one launch on the resident session, the decode submit/collect, and the
//! minted sampler's per-row seat laws.

#[cfg(feature = "spyre-hw")]
use crate::card::pool::*;
use crate::spyre_forward::*;
#[cfg(feature = "spyre-hw")]
use crate::spyre_types::SuperDscSession;
use crate::spyre_types::*;
use crate::spyre_worker::*;
use scratchy_serving_worker::error::ExecutorResult;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::paged_step::{BatchSeq, StepRows};
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre_card::sdsc_runner::StepLogits;

/// ⭐ #32 (c3) — A SUBMITTED DECODE STEP: the launch is on the card, the idx stick's read is
/// enqueued, nothing collected. Everything the collect half needs to decode its picks. The RUNG
/// INDEX travels beside it in the caller, because each rung owns its own session and only that
/// session's executor can finish the read.
pub(crate) struct SubmittedDecode {
    pub(crate) read: scratchy_target_spyre_card::superdsc_exec::DeferredRead,
    /// The launch's answer rows ([`AnswerRows`]).
    pub(crate) rows: AnswerRows,
    /// The idx stick's row stride in words — the seat's own typed fact.
    pub(crate) idx_row_words: u32,
    pub(crate) result_id: u32,
    pub(crate) vocab: usize,
}

/// What `run_card_launch` hands back: a finished step's picks, or a
/// submitted decode step whose read is deferred. A decode batch is ALWAYS the
/// latter — one decode path — and a schedule that cannot defer collects it
/// in-call through the same [`collect_decode_tokens`] door.
pub(crate) enum BatchOut {
    Done(Option<Vec<u32>>),
    Deferred(SubmittedDecode),
}

/// ⭐ #32 (E) — THE OVERLAP PROBE'S PROCESS-WIDE STATE, in the same idiom
/// `step_segs` states the reason for: the submit half's borrows (the session,
/// `Shared`, the launcher) cannot thread a worker field, and the lock is
/// uncontended against a ~24 ms step. Holds the PREVIOUS decode step's stick
/// read as its peek handle — `fired()` is the whole of its surface, so asking
/// "is the card still busy" is a latch read, never a wait.
static PRIOR_READ: std::sync::Mutex<Option<scratchy_target_spyre_card::runtime::ReadPeek>> =
    std::sync::Mutex::new(None);

/// ⭐ THE DEPTH ROW'S OWN COUNT: decode submissions outstanding, incremented at
/// each deferred submit and decremented at each collect — the same population
/// the worker's in-flight queue holds, stated once, process-wide, so the depth
/// row counts exactly the submissions the pipeline has not yet collected.
static OUTSTANDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// ⭐ WHAT ONE CARD LAUNCH'S ROWS ARE, for the paged step: one sequence's rows — a prompt chunk, or
/// the one-request decode — or one token of each of a decode batch's sequences. Each names its KV
/// blocks as pool pages (`paged_step::page_of_block` of the scheduler's block table), which the
/// step's tables are built from and which nothing here can recompute: which page holds a block is
/// the scheduler's grant, and it churns.
pub(crate) enum LaunchRows<'a> {
    /// One sequence: `rows.real` tokens from position `rows.start`, in the pages `blocks`.
    Sequence { blocks: &'a [u32], rows: StepRows },
    /// A decode batch: one token per live sequence, in slot order; the rung's rows past them are
    /// padding, written at the null slot (`paged_step::batch_sources`).
    Batch { seqs: &'a [BatchSeq<'a>] },
    /// ⭐ A WAVE OF CONTINUATION TAILS on a packed rung — `steps` dense from window 0 (request
    /// `r` rides window `r`), each request's window `width` rows wide: the rung's own
    /// per-request width, the fact every row law below divides by. The windows past the live
    /// requests are the rung's EMPTY slots (`paged_step::card_step_sources`'s empty-slot law:
    /// their writes reach no request's cache and their answers are read by nobody).
    Packed {
        steps: &'a [scratchy_target_spyre::paged_step::RequestStep<'a>],
        width: usize,
    },
}

impl LaunchRows<'_> {
    /// THE POSITION THE BODY IS SELECTED BY — the session's `step_body(p)` picks the narrowest walk
    /// reaching `p + 1` positions, so `p` is the last position any row of the launch reaches.
    fn last_pos(&self) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => rows.end() - 1,
            LaunchRows::Batch { seqs } => seqs.iter().map(|q| q.pos).max().unwrap_or(0),
            // The deepest position ANY request reaches — the body the session selects must
            // walk to every request's keys, not one of them.
            LaunchRows::Packed { steps, .. } => {
                steps.iter().map(|s| s.rows.end()).max().unwrap_or(1) - 1
            }
        }
    }

    /// The live rows — a sequence's real tokens, a batch's sequences, or a wave's every
    /// request's real rows (the count its concatenated host tokens carry).
    fn live(&self) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => rows.real.get(),
            LaunchRows::Batch { seqs } => seqs.len(),
            LaunchRows::Packed { steps, .. } => steps.iter().map(|s| s.rows.real.get()).sum(),
        }
    }

    /// Bound row `r`'s position. A pad row's is a live row's: a sequence's pad row repeats its last
    /// real token (`StepRows::pos`), a batch's repeats live 0 (its write lands at the null slot, so
    /// what it computes is never read). A wave's row `r` is window `r / width`'s row
    /// `r % width` — its own request's position, a pad row clamped to its request's last real
    /// one; an EMPTY window's rows take live 0's last real position, the same clamp the
    /// decode's pad rows take.
    fn pos(&self, r: usize) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => rows.pos(r),
            LaunchRows::Batch { seqs } => seqs.get(r).unwrap_or(&seqs[0]).pos,
            LaunchRows::Packed { steps, width } => match steps.get(r / width) {
                Some(req) => req.rows.pos(r % width),
                None => steps[0].rows.pos(width - 1),
            },
        }
    }

    /// The live row a bound row `r` takes its token from. A wave's row takes its own
    /// window's, clamped to its request's last real row; an empty window's rows take live
    /// 0's last real row, so the seed law's `t[live_of(r)]` always names a real token.
    fn live_of(&self, r: usize) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => r.min(rows.real.get() - 1),
            LaunchRows::Batch { seqs } => {
                if r < seqs.len() {
                    r
                } else {
                    0
                }
            }
            LaunchRows::Packed { steps, width } => {
                let q = r / width;
                let mut base = 0;
                for (i, req) in steps.iter().enumerate() {
                    if i == q {
                        return base + (r % width).min(req.rows.real.get() - 1);
                    }
                    base += req.rows.real.get();
                }
                steps[0].rows.real.get() - 1
            }
        }
    }
}

/// ⭐ THE LAUNCH'S ANSWER ROWS — the result rows the host draws a token from: a decode batch's every
/// row (one per sequence), a prompt launch's folded lm-head tail (one row per request the rung
/// packs, request `r`'s at row `r`). Minted by ONE fn from the launch's own rows and step.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AnswerRows(usize);

impl AnswerRows {
    fn of_launch(
        rows_in: &LaunchRows<'_>,
        mq: usize,
        step: &scratchy_target_spyre::wiring::PagedStep,
    ) -> Self {
        Self(match rows_in {
            LaunchRows::Batch { .. } => mq,
            LaunchRows::Sequence { .. } | LaunchRows::Packed { .. } => {
                step.extent.requests().get() as usize
            }
        })
    }

    fn get(self) -> usize {
        self.0
    }
}

/// ⭐ THE SUFFIX'S MINTED SAMPLER, ONE BUNDLE: where the seeds come from, whether the suffix's
/// pick is read back, and the per-row seat laws — the three facts that describe the launch's
/// sampling half rather than its forward body.
pub(crate) struct SamplerLaws<'a> {
    /// ⛔ THE SEED LAW, NOT A TOKEN LIST. The device writes a decode step's seeds itself (each
    /// step's ChainStage decomposes its pick into the persistent sticks), so the host binds seeds
    /// ONLY where the card cannot have: a request's first step on a rung session, a churned seat, a
    /// rung switch. [`SeedSource::OnCard`] is that statement — a stable seating — and binds
    /// nothing. A prompt chunk's rows are the prompt's own tokens, always `Host`.
    pub(crate) seeds: SeedSource<'a>,
    pub(crate) want_logits: LogitsWanted,
    /// The seat laws this launch's answer rows run under, staged before anything runs. A middle
    /// prompt chunk's suffix never runs, so the value it passes is inert — but it is still that
    /// request's own.
    pub(crate) laws: SeatLaws<'a>,
}

/// ⭐ ONE CARD LAUNCH ON THE PAGED STEP — the forward tape at the rung's `width` rows, each
/// attention class's step tables for the body that runs, the minted sampler's seat, then the
/// launch and (where the caller wants it) the read of the idx stick the card's draw wrote.
///
/// ⭐ THE BODY IS ASKED FIRST. The session selects the narrowest walk that reaches the launch's last
/// position (`SuperDscSession::step_body`), and the steps the host stages are THAT body's
/// (`PagedStep::on_body` at its sweep). The launch is handed the same position, so the body staged
/// for and the body launched are one body by construction.
///
/// `b` is the wiring the rung runs under — its own for a prompt chunk and the one-request decode,
/// the one-request decode's for a decode batch rung (the same tape at `width` rows, addressed by
/// fingerprint; `width` is the rung's own baked row count, which every extent here is sized by).
pub(crate) fn run_card_launch(
    rows_in: LaunchRows<'_>,
    sh: &Shared<'_>,
    pf: &mut SuperDscSession,
    b: &BundleMeta,
    width: scratchy_spyre_bundle::rung::RungWidth,
    sampler: SamplerLaws<'_>,
) -> ExecutorResult<BatchOut> {
    let SamplerLaws {
        seeds,
        want_logits,
        laws: seat_laws,
    } = sampler;
    let (h, vocab) = (sh.hidden, sh.vocab);
    let mq = width.count();
    let live = rows_in.live();
    if live == 0 || live > mq {
        return Err(werr(format!(
            "card launch: {live} live row(s) in a {mq}-row rung"
        )));
    }
    let seq_pos = rows_in.last_pos();
    let _t_stage = std::time::Instant::now();
    // ⭐ THE BODY THAT RUNS, AND EACH CLASS'S STEP OF IT.
    let body = pf.step_body(seq_pos).map_err(|e| {
        werr(format!(
            "card launch at {seq_pos}: the session could not say which body runs: {e}"
        ))
    })?;
    let swept = body
        .swept()
        .ok_or_else(|| {
            werr(format!(
                "card launch at {seq_pos}: the session selected a body with no ladder rung — an \
                 unrolled bundle, which states no walk to stage"
            ))
        })?
        .get();
    // ⭐ THE EXTENT LAW IS THE LAUNCH'S OWN KIND — a sequence launch re-derives its pair from
    // its rows, a decode batch from its count, and a WAVE keeps the packed rung's wiring's
    // own pair (`on_body_packed`).
    let steps = b
        .wiring
        .paged
        .map(|paged| match &rows_in {
            LaunchRows::Sequence { .. } => paged.on_body(
                width.get(),
                scratchy_target_spyre::paged_step::PagedRows::Sequence,
                swept,
            ),
            LaunchRows::Batch { .. } => paged.on_body(
                width.get(),
                scratchy_target_spyre::paged_step::PagedRows::Batch,
                swept,
            ),
            LaunchRows::Packed { .. } => paged.on_body_packed(swept),
        })
        .ok_or_else(|| {
            werr(format!(
                "card launch: the body sweeps {swept} slot(s), which is no walk of this bundle's pool"
            ))
        })?;
    let first = steps
        .first()
        .ok_or_else(|| werr("this bundle states no paged step — its attention has no pools"))?;
    // Each class's tables, at the class's own tids.
    let mut card = scratchy_target_spyre::paged_step::CardSources {
        masks: Vec::new(),
        int32: Vec::new(),
    };
    for step in steps.iter() {
        let one = match &rows_in {
            LaunchRows::Sequence { blocks, rows } => {
                scratchy_target_spyre::paged_step::card_step_sources(
                    &step,
                    &[scratchy_target_spyre::paged_step::RequestStep {
                        blocks,
                        rows: *rows,
                    }],
                )
            }
            LaunchRows::Batch { seqs } => {
                scratchy_target_spyre::paged_step::card_batch_sources(&step, seqs)
            }
            LaunchRows::Packed { steps, .. } => {
                scratchy_target_spyre::paged_step::card_step_sources(&step, steps)
            }
        }
        .map_err(|e| werr(format!("card launch at {seq_pos}: paged step: {e}")))?;
        card.masks.extend(one.masks);
        card.int32.extend(one.int32);
    }
    // ⭐ ONE TOKEN PER BOUND ROW — a pad row takes its live row's (`LaunchRows::live_of`). An
    // `OnCard` launch binds none: its seeds are the card's own ChainStage writes, which only a
    // decode-phase bundle's tape reads.
    let tokens: Vec<usize> = match seeds {
        SeedSource::Host(t) => {
            if t.len() != live {
                return Err(werr(format!(
                    "card launch: {} seed token(s) for {live} live row(s) — one per live row",
                    t.len()
                )));
            }
            (0..mq).map(|r| t[rows_in.live_of(r)]).collect()
        }
        SeedSource::OnCard => {
            if matches!(
                b.wiring.phase,
                scratchy_target_spyre::forward_tape::ForwardPhase::Prefill
            ) {
                return Err(werr(
                    "card launch: a prompt chunk cannot seed from the card — every prompt row's \
                     token is a host fact and the EmbedRow push reads it",
                ));
            }
            Vec::new()
        }
    };
    for &tok in &tokens {
        if tok >= vocab {
            return Err(werr(format!("card launch: token {tok} >= vocab {vocab}")));
        }
    }
    let positions: Vec<u32> = (0..mq).map(|r| rows_in.pos(r) as u32).collect();
    let launch_rows = scratchy_target_spyre::wiring::StagedRows::of_launch(mq)
        .ok_or_else(|| werr("card launch with zero rows".to_string()))?;
    let shape = b.wiring.forward_shape_within(
        scratchy_target_spyre::wiring::RowCapacity::of_baked_rung(width),
        launch_rows,
        steps,
    );
    let inputs = scratchy_target_spyre::forward_tape::ForwardInputs {
        hidden: h,
        embed_tokens: sh.embed_tokens,
        tokens: &tokens,
        // ⭐ THE SEED LAW'S WIRE SPELLING: a decode launch seeds the sticks only in the `Host` arm;
        // an `OnCard` launch passes `None` and the launcher's ChainSeed steps stage nothing. A
        // prompt chunk's tape has no ChainSeed step, so `Some` there is inert.
        chain_seeds: match seeds {
            SeedSource::Host(_) => Some(&tokens[..]),
            SeedSource::OnCard => None,
        },
        positions: &positions,
        paged: Some(&card),
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
        acts_i32: Vec::new(),
        seeds: Vec::new(),
        card_owned_sticks: Vec::new(),
        launched: false,
    };
    scratchy_subtile::host_tape::play(
        &scratchy_subtile::host_tape::HostTape { steps: &ftape },
        &mut launcher,
    )
    .map_err(|e| werr(format!("card forward tape: {e}")))?;
    debug_assert!(launcher.launched, "the tape must reach its device step");
    // ⭐⭐⭐ THE ONCARD HAND-OFF, BEFORE THIS LAUNCH'S REFILL. The sticks the skip named are the
    // card's from this step on: dropping them from the executor's persistent maps here — ahead of
    // the launch below — is what stops THIS submit's preamble H2D from re-sending the last `Host`
    // seed over the previous step's ChainStage pick. A later churned seating re-binds its seeds
    // from the collected picks, re-inserting them.
    for id in &launcher.card_owned_sticks {
        pf.hand_to_card(*id);
    }
    let acts = launcher.acts;
    let mut acts_i32 = launcher.acts_i32;
    let seeds = launcher.seeds;
    // ⭐ THE MINTED SAMPLER'S SEAT — every launch of a bundle that mints the draw binds it, every
    // row of it, before anything runs (a MIDDLE prompt chunk included: its suffix is skipped, but
    // the seat's sources are placed all the same, and the launch's completeness check refuses a
    // placed source nobody wrote).
    let seat = b.wiring.sample_seat.ok_or_else(|| {
        werr(format!(
            "bundle has no sampler seat (result t{}) — every card bundle mints the draw",
            b.result_id
        ))
    })?;
    let answers = AnswerRows::of_launch(&rows_in, mq, &first);
    let laws: Vec<RowLaw<'_>> = match (seat_laws, &rows_in) {
        // A PROMPT CHUNK: one request, its answer at its last real position.
        (SeatLaws::Prompt { draw }, LaunchRows::Sequence { rows, .. }) => (0..answers.get())
            .map(|_| RowLaw {
                draw,
                step: (rows.end() - 1) as u32,
            })
            .collect(),
        // A DECODE: live row `s` runs its own request's draw at its own position, and a pad row
        // runs live 0's ([`pad_law_of`]).
        (SeatLaws::Batch { draws }, LaunchRows::Batch { .. }) => {
            let live_pos: Vec<u32> = (0..live).map(|s| rows_in.pos(s) as u32).collect();
            batch_seat_laws(draws, &live_pos, answers.get())?
        }
        // ⭐ A WAVE OF TAILS: answer row `r` is request `r`'s folded answer row — request `r`'s
        // draw at request `r`'s LAST position, and an empty window's runs live 0's.
        (SeatLaws::Packed { draws }, LaunchRows::Packed { steps, .. }) => {
            let last: Vec<u32> = steps.iter().map(|s| (s.rows.end() - 1) as u32).collect();
            batch_seat_laws(draws, &last, answers.get())?
        }
        // ⭐ THE LAWS AND THE LAUNCH'S ROWS ARE PAIRED — the enum's own law.
        _ => {
            return Err(werr(
                "card launch: the seat laws do not match the launch's rows — Prompt laws on a \
                 sequence, Batch laws on a decode, Packed laws on a wave"
                    .to_string(),
            ));
        }
    };
    acts_i32.extend(sample_binds(&seat, &laws));
    if flags().phase_time {
        let elems: usize = acts.iter().map(|(_, v)| v.len()).sum();
        eprintln!(
            "[sdsc-host] card_stage={:.2} ms  ({} acts, {} f32 built, {} int32 binds)",
            _t_stage.elapsed().as_secs_f64() * 1e3,
            acts.len(),
            elems,
            acts_i32.len(),
        );
    }
    log_binds(&format!("launch mq={mq} at={seq_pos}"), b.wiring, &acts);
    let picked = match want_logits {
        // ⭐ THE ONE DECODE PATH, SUBMITTED: stage, launch, enqueue the answer rows' logits read,
        // return; the caller collects it through `collect_decode_tokens`.
        LogitsWanted::PerRequestDeferred => {
            // ⭐ #32 (E) — THE OVERLAP PROBE'S ASK: had the PREVIOUS step's read already landed?
            let card_busy_at_submit = PRIOR_READ
                .lock()
                .expect("the overlap probe's own lock")
                .as_ref()
                .is_some_and(|s| !s.fired());
            let t_bind = std::time::Instant::now();
            pf.stage_step_acts_seeded(&acts, &[], &acts_i32, &seeds)
                .map_err(|e| {
                    werr(format!(
                        "superdsc decode stage (mq={mq}, at={seq_pos}): {e}"
                    ))
                })?;
            let t_submit = std::time::Instant::now();
            pf.submit_step(mq, seq_pos).map_err(|e| {
                werr(format!(
                    "superdsc decode submit (mq={mq}, at={seq_pos}): {e}"
                ))
            })?;
            let depth = OUTSTANDING.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let t_defer = std::time::Instant::now();
            // The readback's own byte size — the seat's typed fact (the idx stick's row), at the
            // launch's one typed width.
            let read = pf
                .defer_read_tensor_ring(
                    scratchy_target_spyre::bundle_code::PlaceId::Act(b.result_id as u32),
                    answers.get() * seat.idx_row_words as usize * 4,
                )
                .map_err(|e| {
                    werr(format!(
                        "superdsc decode idx read (mq={mq}, at={seq_pos}): {e}"
                    ))
                })?;
            let t_stage = std::time::Instant::now();
            *PRIOR_READ.lock().expect("the overlap probe's own lock") = Some(read.signal());
            scratchy_serving_worker::step_segs::record_submit(
                scratchy_serving_worker::step_segs::SubmitSegs {
                    bind: t_submit.duration_since(t_bind),
                    submit: t_defer.duration_since(t_submit),
                    defer: t_stage.duration_since(t_defer),
                    stage: None,
                    total: t_bind.elapsed(),
                    depth,
                    busy: card_busy_at_submit,
                },
            );
            return Ok(BatchOut::Deferred(SubmittedDecode {
                read,
                rows: answers,
                idx_row_words: seat.idx_row_words,
                result_id: b.result_id as u32,
                vocab,
            }));
        }
        // A middle prompt chunk: no suffix runs and nothing is read back.
        LogitsWanted::None => {
            pf.stage_step_acts_seeded(&acts, &[], &acts_i32, &seeds)
                .and_then(|()| pf.run_staged_step(mq, seq_pos, StepLogits::Discarded))
                .map_err(|e| {
                    werr(format!(
                        "superdsc prefill run_step (mq={mq}, at={seq_pos}): {e}"
                    ))
                })?;
            None
        }
        // ⭐ THE PROMPT-FINAL CHUNK'S PICK — on the card, through the prefill rung's own minted
        // sampler: the suffix runs, the sampler consumes the folded lm-head rows there, and the idx
        // stick comes back raw, exactly like a decode's.
        LogitsWanted::LastRow => {
            pf.stage_step_acts_seeded(&acts, &[], &acts_i32, &seeds)
                .and_then(|()| pf.run_staged_step(mq, seq_pos, StepLogits::LeftOnDevice))
                .map_err(|e| {
                    werr(format!(
                        "superdsc prefill run_step (mq={mq}, at={seq_pos}): {e}"
                    ))
                })?;
            Some(read_sampled_tokens(
                pf,
                &seat,
                answers,
                b.result_id as u32,
                vocab,
            )?)
        }
    };
    Ok(BatchOut::Done(picked))
}

/// Every activation a forward binds, against what the BAKE says that tensor is — one line, at
/// `debug`, per launch.
///
/// ⛔ THIS IS THE LIST NOBODY COULD SEE. Both binders build `acts` and hand it to `run_step`, and
/// no log ever stated what was in it. A decode step that staged 96 rows where one is read looked
/// identical, from outside, to one that staged one — the tape's step LIST does not depend on the
/// row count, only the volume does. `expected` comes from the generated wiring's own
/// `tensor_shapes`, so a bind that disagrees with the bundle is visible without a card.
///
/// `<=` is legitimate: a walk body's step stages a prefix of its pool-wide placement. `>` is caught
/// and refused by the executor.
fn log_binds(
    what: &str,
    w: &scratchy_target_spyre::wiring::Wiring,
    acts: &[scratchy_target_spyre::wiring::Bind],
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
    tracing::debug!(
        "[sdsc-binds] {what}: {} f32 [{}]",
        acts.len(),
        f32s.join(" ")
    );
}

/// The worker's launcher — the third half of the split: spyre supplies the kernels and the tape,
/// this supplies the I/O.
///
/// ⛔ IT COMPUTES NOTHING. Every value a forward binds is [`scratchy_target_spyre::forward_tape::kernel_values`]'s, in the target
/// crate, because none of it touches a session; what is left here is stage → h2d → launch. That
/// is the whole reason the split exists, and it is why one launcher serves both the decode and
/// the batched-prefill regimes — they differ only in the `ForwardInputs` they hand it.
pub(crate) struct ForwardLauncher<'a> {
    pub(crate) shape: &'a scratchy_target_spyre::forward_tape::ForwardShape,
    pub(crate) steps: &'a [scratchy_target_spyre::forward_tape::ForwardStep],
    pub(crate) inputs: &'a scratchy_target_spyre::forward_tape::ForwardInputs<'a>,
    pub(crate) staged: Option<(
        scratchy_target_spyre::bundle_code::PlaceId,
        scratchy_target_spyre::forward_tape::Staged,
    )>,
    pub(crate) acts: Vec<scratchy_target_spyre::wiring::Bind>,
    /// ⭐ RAW LE 32-bit binds — the paged step's index tables and the sampler seat's words. Kept
    /// apart from [`Self::acts`] because they need a DIFFERENT shadow write: a value is re-encoded
    /// to the card's fp16 (`ieee_to_sen_bytes`), an index must be copied verbatim.
    pub(crate) acts_i32: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)>,
    /// ⭐ THE CHAIN'S SEEDS — per rank, so not in [`Self::acts_i32`]: every card of a sharded world
    /// binds its own vocabulary block's bytes (`SuperDscSession::stage_step_acts_seeded`).
    pub(crate) seeds: Vec<(
        scratchy_target_spyre::bundle_code::PlaceId,
        scratchy_target_spyre::forward_tape::ChainSeed,
    )>,
    /// ⭐⭐⭐ THE STICKS THIS LAUNCH HANDS TO THE CARD — filled by the ChainSeed skip below. An
    /// `OnCard` launch's sticks are the CARD'S: the previous step's ChainStage writes them, and the
    /// executor's persistent bind maps must let them go, or every later submit's refill re-uploads
    /// the last `Host` seed over the card's own pick before the gather reads it. The caller walks
    /// this list through `SuperDscSession::hand_to_card` after the tape plays.
    pub(crate) card_owned_sticks: Vec<scratchy_target_spyre::bundle_code::PlaceId>,
    /// Set when the tape's device step runs.
    pub(crate) launched: bool,
}

impl scratchy_subtile::host_tape::Launcher for ForwardLauncher<'_> {
    type Error = String;

    fn host_call(&mut self, k: scratchy_subtile::host_tape::KernelId) -> Result<(), Self::Error> {
        let step = self
            .steps
            .get(k.0 as usize)
            .ok_or_else(|| format!("no forward kernel {}", k.0))?;
        // ⭐⭐ THE CHAIN'S SEED SKIP — THE `OnCard` ARM. A stable seating's seeds are the card's own
        // ChainStage writes (the previous step's pick, decomposed into the persistent sticks at
        // its tail), so this launch stages NO seed and the device's gather reads the sticks as
        // they were left. Binding a host seed here would need the PREVIOUS step's pick before
        // this launch's submit — the drain fence the seed law exists to delete.
        //
        // ⛔⛔⛔ AND THE SKIP MUST GO FURTHER THAN "STAGE NOTHING": a previous `Host` launch's seed
        // bind sits in the executor's PERSISTENT maps, and the refill re-sends every unkeyed bound
        // entry on EVERY submit — so the skip NAMES the two sticks for the caller to hand to the
        // card ([`Self::card_owned_sticks`]).
        if step.kernel == scratchy_target_spyre::forward_tape::ForwardKernel::ChainSeed
            && self.inputs.chain_seeds.is_none()
        {
            if let scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain } =
                self.shape.phase
            {
                let s = scratchy_target_spyre::bundle_code::PlaceId::Act(chain.s_src);
                let l = scratchy_target_spyre::bundle_code::PlaceId::Act(chain.l_src);
                if step.tensor != s && step.tensor != l {
                    return Err(format!(
                        "ChainSeed step on t{} — the chain names t{} (s) and t{} (l), and a seed \
                         landing anywhere else is a stick the gather never reads",
                        step.tensor.tid(),
                        chain.s_src,
                        chain.l_src
                    ));
                }
                self.card_owned_sticks.push(step.tensor);
            }
            self.staged = None;
            return Ok(());
        }
        let staged =
            scratchy_target_spyre::forward_tape::kernel_values(step, self.shape, self.inputs)?;
        self.staged = Some((step.tensor, staged));
        Ok(())
    }

    fn h2d(&mut self, t: scratchy_subtile::host_tape::TensorId) -> Result<(), Self::Error> {
        let Some((tensor, staged)) = self.staged.take() else {
            // The one step that stages nothing: the chain's seed when the seeds are the card's own
            // (`OnCard` — the two stick sources the phase's chain names).
            let chain_skip = match self.shape.phase {
                scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain } => {
                    self.inputs.chain_seeds.is_none() && (t.0 == chain.s_src || t.0 == chain.l_src)
                }
                scratchy_target_spyre::forward_tape::ForwardPhase::Prefill => false,
            };
            return if chain_skip {
                Ok(())
            } else {
                Err(format!("h2d t{} with nothing staged", t.0))
            };
        };
        // The KERNEL decided the format; the transfer just honours it.
        match staged {
            scratchy_target_spyre::forward_tape::Staged::F32(v) => {
                // `Staged::F32` already carries the borrow/own distinction — a computed kernel
                // output is Owned, a constant table is Borrowed — so it passes straight through.
                self.acts.push((tensor, v))
            }
            // ⛔ ITS OWN LIST, AND ITS OWN BIND CALL. Routing int32 entries through `acts` would
            // round every address through an f32 and then re-encode it as fp16 — a valid, different
            // page number, fluent output from another row's keys. See `Executor::bind_input_raw`.
            scratchy_target_spyre::forward_tape::Staged::I32(bytes) => {
                self.acts_i32.push((tensor, bytes));
            }
            // ⭐ THE CHAIN'S SEED — per rank: the session builds each card's bytes from its own
            // vocabulary block at the bind.
            scratchy_target_spyre::forward_tape::Staged::ChainSeed(seed) => {
                self.seeds.push((tensor, seed));
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

/// ⭐⭐⭐ A PAD SLOT'S LAW IS LIVE 0'S OWN (the pad clamp, auditor ruling
/// 2026-10-27). A decode pad row's forward takes live 0's token and position,
/// its KV write lands at the null slot (`paged_step::batch_sources`) and its
/// pick is never read — so its seat stages live 0's words, one law for greedy
/// and sampled alike, rather than a law no request asked for. Called by every
/// staging walk that lays pad rows, never spelled twice.
fn pad_law_of<'a>(live0: &RowLaw<'a>) -> RowLaw<'a> {
    RowLaw {
        draw: live0.draw,
        step: live0.step,
    }
}

/// ⭐ THE DECODE'S OWN LAW WALK — one law per baked row of the seat:
/// live slot `s` runs its request's draw at that request's own position
/// (`positions[s]`), and a padding slot runs live 0's ([`pad_law_of`]).
/// Extracted from the staging arm so the pad clamp is pinnable without a
/// launch.
fn batch_seat_laws<'a>(
    draws: &[Option<&'a SampledDraw>],
    positions: &[u32],
    seat_rows: usize,
) -> ExecutorResult<Vec<RowLaw<'a>>> {
    if draws.len() != positions.len() || draws.len() > seat_rows {
        return Err(werr(format!(
            "card launch: {} seat law(s) for {} live row(s) in a {seat_rows}-row seat — one law \
             per live request, in slot order",
            draws.len(),
            positions.len()
        )));
    }
    let mut laws: Vec<RowLaw<'_>> = Vec::with_capacity(seat_rows);
    for s in 0..seat_rows {
        match (draws.get(s), positions.get(s)) {
            (Some(&draw), Some(&step)) => laws.push(RowLaw { draw, step }),
            _ => laws.push(match laws.first() {
                Some(live0) => pad_law_of(live0),
                // A launch with no live row at all: nothing replicates, and
                // the greedy law stands.
                None => RowLaw {
                    draw: None,
                    step: 0,
                },
            }),
        }
    }
    Ok(laws)
}

/// ⭐ THE SEAT LAWS ONE LAUNCH RUNS ITS ROWS UNDER — the three shapes a caller
/// can know them in, and ONLY those three: a prompt-final chunk serves ONE
/// request whose rows are consecutive positions of one sequence, a decode
/// batch serves one request PER LIVE ROW, and a wave of tails serves one
/// request PER WINDOW. The variant must match the launch's own rows (the
/// staging match refuses every other pairing); a mismatch is refused where
/// the seat is staged, never silently resolved.
pub(crate) enum SeatLaws<'a> {
    /// A prompt-final chunk: every baked row runs `draw` — real row `i` at its
    /// own position `start + i`, and a pad row at the last real position it
    /// clamps to (the pad row replicates the last real token's entire forward,
    /// and the CONSUMED pick is the last baked row's, so the pad row must carry
    /// the same draw the last real row would, or an unfilled rung would pick
    /// greedy for a sampled request).
    Prompt { draw: Option<&'a SampledDraw> },
    /// A decode: one law per LIVE row, in slot order — that row's request's
    /// draw at that request's own position. A padding slot runs LIVE 0'S LAW
    /// ([`pad_law_of`], the pad clamp); its pick is never read.
    Batch {
        draws: &'a [Option<&'a SampledDraw>],
    },
    /// ⭐ A WAVE OF TAILS: one law per REQUEST, in window order — seat row `r`
    /// is request `r`'s folded answer row, so it runs request `r`'s draw at
    /// request `r`'s LAST position (the row the lm-head fold takes). A pad seat
    /// row — an empty window's answer — runs live 0's law ([`pad_law_of`]); its
    /// pick is never read.
    Packed {
        draws: &'a [Option<&'a SampledDraw>],
    },
}

/// ⭐ THE SEAT BINDS EVERY LAUNCH STAGES — the seat's per-step operands, one law per answer row,
/// dense at the kernel's own `[rows, 1]` views: the four seed word tables
/// `(seed_lo, seed_hi, step, gate)` as int32 words and the temperature reciprocal as an f32 word,
/// row `r`'s at word `r` ([`seat_words`], the one statement of the law). The greedy law is
/// `(0, 0, 0, 0)` and 1.0 — the gate's 0 makes `score = x·inv_t` exactly. Raw LE words through the
/// verbatim channel: the f32 `Bind` channel converts to SEN169 and is NOT verbatim.
fn sample_binds(
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    laws: &[RowLaw<'_>],
) -> Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)> {
    tracing::debug!(
        "SEAT_BIND rows={} seed_words={:?} invt=t{} laws={}",
        laws.len(),
        seat.seed_words,
        seat.invt,
        laws.iter()
            .enumerate()
            .map(|(r, law)| match law.draw {
                Some(d) => format!("r{r}:seed={}@step{}:inv_t{}", d.seed, law.step, d.inv_t),
                None => format!("r{r}:greedy@step{}", law.step),
            })
            .collect::<Vec<_>>()
            .join(" ")
    );
    let words: Vec<([i32; 4], f32)> = laws.iter().map(seat_words).collect();
    let mut binds: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)> = seat
        .seed_words
        .iter()
        .enumerate()
        .map(|(k, &tid)| {
            (
                scratchy_target_spyre::bundle_code::PlaceId::Act(tid),
                words.iter().flat_map(|(w, _)| w[k].to_le_bytes()).collect(),
            )
        })
        .collect();
    binds.push((
        scratchy_target_spyre::bundle_code::PlaceId::Act(seat.invt),
        words.iter().flat_map(|(_, t)| t.to_le_bytes()).collect(),
    ));
    binds
}

/// Read the minted sampler's idx stick back in-call and decode it: row `r`'s pick at word
/// `r · idx_row_words` ([`decode_idx_stick`]).
fn read_sampled_tokens(
    session: &mut SuperDscSession,
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    rows: AnswerRows,
    result_id: u32,
    vocab: usize,
) -> ExecutorResult<Vec<u32>> {
    let read = session
        .defer_read_tensor_raw(
            scratchy_target_spyre::bundle_code::PlaceId::Act(result_id),
            rows.get() * seat.idx_row_words as usize * 4,
        )
        .map_err(|e| werr(format!("sampler idx readback (t{result_id}): {e}")))?;
    session
        .finish_deferred_read_with(&read, |bytes| {
            decode_idx_stick(bytes, rows.get(), seat.idx_row_words as usize, vocab)
        })
        .map_err(|e| werr(format!("sampler idx readback (t{result_id}): {e}")))?
}

/// ⭐ #32 (c3) — THE COLLECT HALF of a submitted decode step: park on the idx read's own
/// completion (the loop's ONE fence — this read's D2H, then the shadow's quiet point, never a
/// whole-stream sync), then decode the picks on the landed bytes. The SAME door whether the caller
/// collects in-call (a schedule that cannot defer) or from the drain.
pub(crate) fn collect_decode_tokens(
    session: &mut SuperDscSession,
    sub: &SubmittedDecode,
) -> ExecutorResult<Vec<u32>> {
    #[cfg(feature = "alloc-census")]
    scratchy_target_spyre_card::census::collect_step();
    let t_fence = std::time::Instant::now();
    // The two `?`s keep the order: a read error returns before the outstanding decrement (the
    // submission is still uncollected on a dead read), a decode refusal decrements first.
    let toks = session
        .finish_deferred_read_with(&sub.read, |bytes| {
            decode_idx_stick(bytes, sub.rows.get(), sub.idx_row_words as usize, sub.vocab)
        })
        .map_err(|e| werr(format!("sampler idx readback (t{}): {e}", sub.result_id)))?;
    let t_pick = std::time::Instant::now();
    OUTSTANDING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    let toks = toks?;
    scratchy_serving_worker::step_segs::record_collect(
        scratchy_serving_worker::step_segs::CollectSegs {
            fence: t_pick.duration_since(t_fence),
            pick: t_pick.elapsed(),
        },
    );
    Ok(toks)
}

/// ⭐ THE PREFILL LADDERS THE CHUNK LOOP CLIMBS, ONE BUNDLE: the ladder a prompt's first chunk picks
/// from (each rung walks only its own chunk's pages), and the ladder a continuation chunk picks from
/// (each rung's walks reach up to the whole pool) — both borrowed mutably because the picked rung's
/// session is run in place.
pub(crate) struct PrefillLadders<'a> {
    /// Ascending by baked width. A chunk at position 0 runs on the smallest rung holding it.
    pub(crate) rungs: &'a mut [(usize, SuperDscSession)],
    /// The rungs a chunk past position 0 runs on — its walk reaches every key before it.
    pub(crate) prefix: &'a mut PrefixLadder,
}

/// ⭐ A PROMPT ON THE CARD — its tokens `start..start + toks.len()` as a sequence of chunks, each on
/// the narrowest prefill rung that holds it and reaches its positions (`PagedStep::reach`): a chunk at
/// position 0 on the ladder whose walk covers its own pages, a chunk past it on the ladder whose walk
/// covers the pool. Each chunk stages its paged step from the request's KV blocks, writes its keys at
/// its own positions and attends every key at or before each row's position; a chunk's pad rows
/// repeat its last real token, so their writes are the last real token's own. The chunk that finishes
/// the prompt runs its suffix: the rung's minted sampler picks the first generated token on the card.
pub(crate) fn superdsc_forward_chunk(
    sh: &Shared<'_>,
    ladders: PrefillLadders<'_>,
    // The prompt chunks' wiring — every rung of both ladders binds its tensor ids.
    prefill_meta: Option<&BundleMeta>,
    Chunk {
        state: req,
        id,
        tokens,
        start,
        toks,
    }: Chunk<'_>,
) -> ExecutorResult<ChunkOut> {
    let PrefillLadders { rungs, prefix } = ladders;
    let n = toks.len();
    if n == 0 {
        return Err(werr("superdsc_forward_chunk: n=0".to_string()));
    }
    let pmeta = prefill_meta.ok_or_else(|| {
        werr("superdsc prefill: this bundle bakes no prompt chunk — a prompt has nothing to run on")
    })?;
    // ⭐ THE REQUEST'S KV BLOCKS — the scheduler's grant, read out of the one store it was installed
    // in, as pool pages. Its pages already hold the keys of positions `0..start` (this request's
    // earlier chunks, or the request whose prefix a cache hit shares).
    let blocks: Vec<u32> = tokens
        .block_table(id)
        .map(|t| {
            t.iter()
                .map(|&b| scratchy_target_spyre::paged_step::page_of_block(b as u32))
                .collect()
        })
        .unwrap_or_default();
    let top_m = rungs.last().map(|(m, _)| *m).unwrap_or(0);
    let end = start + n;
    let mut pos = start;
    let mut pick: Option<Vec<u32>> = None;
    while pos < end {
        let left = end - pos;
        let (width, sess) = if pos == 0 {
            let ri = prefill_rung_for(rungs, left.min(top_m))
                .ok_or_else(|| werr("superdsc prefill: no prompt chunk rung baked".to_string()))?;
            let (m, sess) = &mut rungs[ri];
            (*m, sess)
        } else {
            let r = prefix.rung_for(left).ok_or_else(|| {
                werr(format!(
                    "superdsc prefill: a chunk at position {pos} needs a rung whose walk reaches the \
                     keys before it, and none is baked"
                ))
            })?;
            (r.width(), prefix.session(r))
        };
        let real = left.min(width);
        let chunk_toks: Vec<usize> = (pos..pos + real)
            .map(|p| {
                tokens.token_at(id, p).map(|t| t as usize).ok_or_else(|| {
                    werr(format!(
                        "superdsc prefill: chunk wants token {p} but this request has {}",
                        tokens.token_count(id),
                    ))
                })
            })
            .collect::<ExecutorResult<Vec<usize>>>()?;
        let last = pos + real == end;
        let finishes = last && end >= tokens.prompt_len(id);
        let rung = scratchy_spyre_bundle::rung::RungWidth::of_baked_rows(width as u32)
            .ok_or_else(|| werr("superdsc prefill: the selected rung has zero rows".to_string()))?;
        let out = run_card_launch(
            LaunchRows::Sequence {
                blocks: &blocks,
                rows: StepRows::new(pos, real).ok_or_else(|| {
                    werr(
                        "superdsc prefill: a chunk of zero real rows — the chunker staged no chunk"
                            .to_string(),
                    )
                })?,
            },
            sh,
            sess,
            pmeta,
            rung,
            SamplerLaws {
                seeds: SeedSource::Host(&chunk_toks),
                want_logits: if finishes {
                    LogitsWanted::LastRow
                } else {
                    LogitsWanted::None
                },
                laws: SeatLaws::Prompt {
                    draw: req.draw.as_ref(),
                },
            },
        )?;
        pick = match out {
            BatchOut::Done(p) => p,
            BatchOut::Deferred(_) => {
                return Err(werr(
                    "a prompt chunk deferred its read — only a decode can".to_string(),
                ));
            }
        };
        tracing::debug!(
            "PREFILL_PATH=batched rung={width} rows={real} pos={pos} end={end} top_m={top_m} \
             prefix={}",
            pos > 0
        );
        pos += real;
    }
    tokens.set_tokens_in_pool(id, end);
    Ok(match pick {
        Some(ts) => ChunkOut::Tokens(ts),
        None => ChunkOut::Prompt,
    })
}

/// ⭐ ONE REQUEST OF A PREFILL WAVE — the tokens this step forwards for it (its cache-hit
/// tail, or a fresh prompt's whole step), its KV blocks as pool pages, its draw, and whether
/// this step finishes its prompt (the request whose pick the wave's readback carries). Owned
/// whole: the gatherer builds it before any request runs, so nothing of the executor's
/// borrows rides the wave.
pub(crate) struct PackedTail {
    pub(crate) id: String,
    pub(crate) start: usize,
    pub(crate) toks: Vec<usize>,
    pub(crate) blocks: Vec<u32>,
    pub(crate) draw: Option<SampledDraw>,
    pub(crate) finishes: bool,
}

/// ⭐ A WAVE OF CONTINUATION TAILS, ONE LAUNCH — the packed rung's own front.
///
/// The scheduler hands one step several requests whose prompts share a cached prefix (or
/// several fresh prompts that fit a window each); the solo path runs each as its own chunk,
/// paying the rung's fixed cost — the weight streams and the launch — once per tail. The wave
/// batches them onto ONE packed rung `(w, s)`: request `r` rides window `r` — rows
/// `r·w .. r·w+w` of the rung's graph — staging its own three tables at its own window tids
/// ([`card_step_sources`]), seeding its rows with its own tokens, its KV writes naming its
/// own cache slots, and the rung's minted sampler picking one token per window.
///
/// The suffix runs for EVERY window — one tape, one launch, no per-window skip — so the
/// readback carries `s` picks and only the finishing requests' are kept: a mid-prompt
/// request's pick is computed and never read. The caller's selection (a rung that holds the
/// widest tail and the live count, or none) is a SELECTION, never a refusal: a wave no rung
/// holds runs the solo path.
pub(crate) fn superdsc_forward_packed(
    sh: &Shared<'_>,
    prefix: &mut PrefixLadder,
    rung: PrefixRung,
    tails: &[PackedTail],
    tokens: &mut scratchy_serving_engine::input_batch::InputBatch,
) -> ExecutorResult<Vec<Option<u32>>> {
    let (meta, sess) = prefix.packed_parts(rung);
    let (w, s) = (rung.width(), rung.seqs());
    // ⭐ REQUEST `r`'S OWN STEP — its tail's rows from its own start, at most the window's
    // width (the selection held the widest tail), in its own pages.
    let steps: Vec<scratchy_target_spyre::paged_step::RequestStep<'_>> = tails
        .iter()
        .map(|t| {
            Ok(scratchy_target_spyre::paged_step::RequestStep {
                blocks: &t.blocks,
                rows: StepRows::new(t.start, t.toks.len()).ok_or_else(|| {
                    werr("a wave tail of zero real rows — the gatherer staged no tail".to_string())
                })?,
            })
        })
        .collect::<ExecutorResult<_>>()?;
    // ⭐ THE SEEDS, CONCATENATED — one token per live row, window order: request `r`'s rows
    // take request `r`'s tokens ([`LaunchRows::live_of`]'s window law).
    let seeds: Vec<usize> = tails.iter().flat_map(|t| t.toks.iter().copied()).collect();
    let draws: Vec<Option<&SampledDraw>> = tails.iter().map(|t| t.draw.as_ref()).collect();
    let width = scratchy_spyre_bundle::rung::RungWidth::of_baked_rows((w * s) as u32)
        .ok_or_else(|| werr("the packed rung bakes zero rows".to_string()))?;
    let picked = match run_card_launch(
        LaunchRows::Packed {
            steps: &steps,
            width: w,
        },
        sh,
        sess,
        meta,
        width,
        SamplerLaws {
            seeds: SeedSource::Host(&seeds),
            want_logits: LogitsWanted::LastRow,
            laws: SeatLaws::Packed { draws: &draws },
        },
    )? {
        BatchOut::Done(picked) => picked.ok_or_else(|| {
            werr(
                "the packed launch picked no tokens — its minted draw never ran, and every \
                 window's answer is the readback's to carry"
                    .to_string(),
            )
        })?,
        BatchOut::Deferred(_) => {
            return Err(werr(
                "a packed prefill deferred its read — only a decode can".to_string(),
            ));
        }
    };
    if picked.len() != s {
        return Err(werr(format!(
            "the packed rung (w={w}, s={s}) picked {} token(s) — one per window, never another \
             count",
            picked.len()
        )));
    }
    tracing::debug!(
        "PREFILL_PATH=packed rung_w={w} rung_s={s} tails={} rows={}",
        tails.len(),
        seeds.len()
    );
    // ⭐ THE BOOKKEEPING, PER REQUEST — its pool count always, its pick iff this step
    // finished its prompt (the pick is request `r`'s own: seat row `r`, window `r`).
    let mut out = Vec::with_capacity(tails.len());
    for (r, t) in tails.iter().enumerate() {
        tokens.set_tokens_in_pool(&t.id, t.start + t.toks.len());
        if t.finishes {
            let tok = picked[r];
            tokens.push_generated(&t.id, tok);
            out.push(Some(tok));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

/// Run the prompt tokens `toks` at absolute positions `start..start+toks.len()` on the card's
/// prompt-chunk ladders ([`superdsc_forward_chunk`]).
pub(crate) fn forward_chunk(
    sh: &Shared<'_>,
    session: &mut SendnnSession,
    b: &BundleMeta,
    // The prompt chunks' wiring, which every rung of both prompt ladders binds.
    prefill_meta: Option<&BundleMeta>,
    Chunk {
        state: req,
        id,
        tokens,
        start,
        toks,
    }: Chunk<'_>,
) -> ExecutorResult<ChunkOut> {
    let SendnnSession::SuperDsc(sb) = session;
    // ⭐ ONLY THE LADDERS: the prompt runs through the prompt-chunk rungs, and a generation step
    // decodes on the decode sessions through the shared decode path in `spyre_exec`. The cap-bucket
    // meta `b` is not a fact this path reads: the ladder picks the rung.
    let SuperDscBundle {
        prefill_rungs,
        prefill_prefix,
        ..
    } = sb;
    let _ = b;
    superdsc_forward_chunk(
        sh,
        PrefillLadders {
            rungs: prefill_rungs.as_mut_slice(),
            prefix: prefill_prefix,
        },
        prefill_meta,
        Chunk {
            state: req,
            id,
            tokens,
            start,
            toks,
        },
    )
}

/// ⭐ A PAD ROW DRAWS LIVE 0'S LAW — the pad clamp: a decode rung's pad rows run live 0's draw at
/// live 0's position (their tokens are never read), and a full rung has no pad row at all.
#[cfg(all(test, feature = "spyre-hw"))]
mod pad_clamp_tests {
    use super::*;

    #[test]
    fn a_pad_row_draws_live_0s_law() {
        let draw = SampledDraw {
            seed: 7,
            inv_t: 2.0,
        };
        let draws = [Some(&draw), None];
        let laws = batch_seat_laws(&draws, &[10, 11], 4).expect("two live rows in four");
        let seen: Vec<(Option<u64>, u32)> = laws
            .iter()
            .map(|l| (l.draw.map(|d| d.seed), l.step))
            .collect();
        assert_eq!(
            seen,
            vec![(Some(7), 10), (None, 11), (Some(7), 10), (Some(7), 10)]
        );
        let full = batch_seat_laws(&draws, &[10, 11], 2).expect("two live rows in two");
        assert_eq!(full.len(), 2);
        assert!(
            batch_seat_laws(&draws, &[10, 11], 1).is_err(),
            "more laws than rows"
        );
    }
}
