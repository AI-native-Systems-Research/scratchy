// SPDX-License-Identifier: Apache-2.0
//! The card's forward: one launch on the resident session, the decode submit/collect, and the
//! sampler's seat laws.

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

/// ⭐ #32 (c3) — A SUBMITTED DECODE STEP: the launch is on the card, the
/// idx-stick read is enqueued, nothing collected. Everything the collect half
/// needs to turn the ring slot back into tokens. The RUNG INDEX travels beside
/// it in the caller, because each rung owns its own session and only that
/// session's executor can finish the read.
pub(crate) struct SubmittedDecode {
    pub(crate) read: scratchy_target_spyre_card::superdsc_exec::DeferredRead,
    /// The launch's own typed width ([`SeatWidth`]) the submit half minted.
    pub(crate) rows: SeatWidth,
    /// The idx stick's own row size in i32 words — the seat's typed readback
    /// fact, so the collect half decodes at the placement's own width.
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
}

impl LaunchRows<'_> {
    /// THE POSITION THE BODY IS SELECTED BY — the session's `step_body(p)` picks the narrowest walk
    /// reaching `p + 1` positions, so `p` is the last position any row of the launch reaches.
    fn last_pos(&self) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => rows.end() - 1,
            LaunchRows::Batch { seqs } => seqs.iter().map(|q| q.pos).max().unwrap_or(0),
        }
    }

    fn paged_rows(&self) -> scratchy_target_spyre::paged_step::PagedRows {
        match self {
            LaunchRows::Sequence { .. } => scratchy_target_spyre::paged_step::PagedRows::Sequence,
            LaunchRows::Batch { .. } => scratchy_target_spyre::paged_step::PagedRows::Batch,
        }
    }

    /// The live rows — a sequence's real tokens, or a batch's sequences.
    fn live(&self) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => rows.real.get(),
            LaunchRows::Batch { seqs } => seqs.len(),
        }
    }

    /// Bound row `r`'s position. A pad row's is a live row's: a sequence's pad row repeats its last
    /// real token (`StepRows::pos`), a batch's repeats live 0 (its write lands at the null slot, so
    /// what it computes is never read).
    fn pos(&self, r: usize) -> usize {
        match self {
            LaunchRows::Sequence { rows, .. } => rows.pos(r),
            LaunchRows::Batch { seqs } => seqs.get(r).unwrap_or(&seqs[0]).pos,
        }
    }

    /// The live row a bound row `r` takes its token from.
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
        }
    }
}

/// ⭐ THE SEAT'S LAUNCH WIDTH — the typed row count a launch binds and reads
/// the per-step seat at. `b` is the one-request decode's meta borrowed for
/// its tensor ids, so its seat states `rows = 1`; a batched launch at `mq`
/// rows binds and reads `mq` rows of the per-step seat, and a prompt keeps
/// the prefill rung's own row count verbatim (its draw is the FOLDED single
/// row). Minted by ONE fn ([`SeatWidth::of_launch`]) from the seat and the
/// launch's own rows, so no call site spells a number — the width a
/// batching launch takes is this type's value, never a field overwritten on
/// a copy of the seat.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SeatWidth(usize);

impl SeatWidth {
    /// The ONE mint — the launch's own row count over the seat.
    fn of_launch(
        seat: &scratchy_target_spyre::wiring::SampleSeat,
        rows_in: &LaunchRows<'_>,
        mq: usize,
    ) -> Self {
        Self(match rows_in {
            LaunchRows::Batch { .. } => mq,
            _ => seat.rows as usize,
        })
    }

    fn get(self) -> usize {
        self.0
    }
}

/// ⭐ THE SUFFIX'S MINTED SAMPLER, ONE BUNDLE: where the seeds come from, whether
/// the lm-head tail runs, and the per-row seat laws — the three facts that
/// describe the launch's sampling half rather than its forward body.
pub(crate) struct SamplerLaws<'a> {
    // ⛔ THE SEED LAW, NOT A TOKEN LIST. The device writes a decode step's seeds itself (each
    // step's ChainStage decomposes its pick into the persistent sticks), so the host binds seeds
    // ONLY where the card cannot have: a request's first step on a rung session, a churned seat, a
    // rung switch. `SeedSource::OnCard` is that statement — a stable seating — and binds nothing.
    // A prompt chunk's rows are the prompt's own tokens, always `Host`.
    pub(crate) seeds: ktir_superdsc::sdsc_abstract::SeedSource<'a>,
    pub(crate) want_logits: LogitsWanted,
    // ⭐ THE SEAT LAWS THIS LAUNCH'S ROWS RUN UNDER — one law per row the suffix's sampler picks
    // for, staged before anything runs. A middle prompt chunk's suffix never runs, so the value it
    // passes is inert — but it is still that request's own.
    pub(crate) laws: SeatLaws<'a>,
}

/// ⭐ ONE CARD LAUNCH ON THE PAGED STEP — the forward tape at the rung's `width` rows, the step's
/// tables for the body that runs, and the minted sampler's seat; then the launch.
///
/// ⭐ THE BODY IS ASKED FIRST. The session selects the narrowest walk that reaches the launch's last
/// position (`SuperDscSession::step_body`), and the step the host stages is THAT body's
/// (`PagedStep::on_body` at its sweep): the page table one block per trip, the mask stack one tile
/// per trip. The launch is handed the same position, so the body staged for and the body launched
/// are one body by construction.
///
/// `b` is the wiring the rung runs under — its own for a prompt chunk and the one-request decode,
/// the one-request decode's for a decode batch rung (the same tape at `width` rows, addressed by
/// fingerprint; `width` is the rung's own baked row count, which every extent here is sized by).
pub(crate) fn run_card_launch(
    rows_in: LaunchRows<'_>,
    sh: &Shared<'_>,
    pf: &mut SuperDscSession,
    b: &BundleMeta,
    width: ktir_superdsc::sdsc_abstract::RungWidth,
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
    // ⭐ THE BODY THAT RUNS, AND ITS STEP.
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
    let step = b
        .wiring
        .paged
        .ok_or_else(|| werr("this bundle states no paged step — its attention has no pools"))?
        .on_body(width.get(), rows_in.paged_rows(), swept)
        .ok_or_else(|| {
            werr(format!(
                "card launch: the body sweeps {swept} slot(s), which is no walk of this bundle's pool"
            ))
        })?;
    let card = match &rows_in {
        LaunchRows::Sequence { blocks, rows } => {
            // One request's list: a sequence launch is one request today — the packing that
            // widens this list is the concurrent-prefill rung's own step, not this front's.
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
    }
    .map_err(|e| werr(format!("card launch at {seq_pos}: paged step: {e}")))?;
    // ⭐ ONE TOKEN PER BOUND ROW — a pad row takes its live row's (`LaunchRows::live_of`). An
    // `OnCard` launch binds none: its seeds are the card's own ChainStage writes, which only a
    // decode-phase bundle's tape reads.
    let tokens: Vec<usize> = match seeds {
        ktir_superdsc::sdsc_abstract::SeedSource::Host(t) => {
            if t.len() != live {
                return Err(werr(format!(
                    "card launch: {} seed token(s) for {live} live row(s) — one per live row",
                    t.len()
                )));
            }
            (0..mq).map(|r| t[rows_in.live_of(r)]).collect()
        }
        ktir_superdsc::sdsc_abstract::SeedSource::OnCard => {
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
    // The stick-padded row count the constants' tables are laid at — whole
    // fp16 sticks, the pool's own `POOL_STICK` (the emitter's own pad law),
    // never a literal 64.
    let stick = ktir_superdsc::sdsc_abstract::POOL_STICK as usize;
    let mq_pad = mq.div_ceil(stick) * stick;
    if h % stick != 0 {
        return Err(werr(format!(
            "card launch: hidden={h} is not a whole number of {stick}-element sticks"
        )));
    }
    let launch_rows = scratchy_target_spyre::wiring::StagedRows::of_launch(mq)
        .ok_or_else(|| werr("card launch with zero rows".to_string()))?;
    // ⭐ THE SESSION'S OWN FACTS: the constants this launch's bundle placed, index for index —
    // never the wiring's bundle's (a decode batch rung runs under the one-request decode's wiring).
    let consts = scratchy_target_spyre::wiring::synthetic_constants(&pf.bake_facts().constant_env(
        b.wiring,
        mq_pad,
        launch_rows,
    ));
    let shape = b.wiring.forward_shape_within(
        scratchy_target_spyre::wiring::RowCapacity::of_baked_rung(width),
        launch_rows,
        Some(step),
        consts.len(),
    );
    let inputs = scratchy_target_spyre::forward_tape::ForwardInputs {
        hidden: h,
        embed_tokens: sh.embed_tokens,
        tokens: &tokens,
        // ⭐ THE SEED LAW'S WIRE SPELLING: a decode launch seeds the sticks only in the `Host` arm;
        // an `OnCard` launch passes `None` and the launcher's ChainSeed steps stage nothing. A
        // prompt chunk's tape has no ChainSeed step, so `Some` there is inert.
        chain_seeds: match seeds {
            ktir_superdsc::sdsc_abstract::SeedSource::Host(_) => Some(&tokens[..]),
            ktir_superdsc::sdsc_abstract::SeedSource::OnCard => None,
        },
        positions: &positions,
        consts: &consts,
        stick_major: true,
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
    // ⭐⭐⭐ THE ONCARD HAND-OFF, BEFORE THIS LAUNCH'S REFILL. The sticks the skip named are
    // the card's from this step on: dropping them from the executor's persistent maps here —
    // ahead of the launch below — is what stops THIS submit's preamble H2D from re-sending the
    // last `Host` seed over the previous step's ChainStage pick. A later churned seating
    // re-binds its seeds from the collected picks, re-inserting them.
    for id in &launcher.card_owned_sticks {
        pf.hand_to_card(*id);
    }
    let acts = launcher.acts;
    let mut acts_i32 = launcher.acts_i32;
    let seeds = launcher.seeds;
    // ⭐ EVERY LAUNCH OF A BUNDLE THAT MINTS THE DRAW BINDS ITS SAMPLER SEAT, every row of it,
    // before anything runs — a MIDDLE prompt chunk included. Its suffix is skipped, but the
    // seat's five sources are placed in this bundle all the same, and the launch's completeness
    // check (`require_sources_filled`) refuses a placed source nobody wrote. The laws a middle
    // chunk stages are this request's own, so nothing is invented.
    //
    // ⭐ THE LAUNCH'S SEAT AND WIDTH, BOUND ONCE: the width is [`SeatWidth`]'s
    // ONE mint for the whole launch — the staging block and both pick arms
    // below read this one value, never a second mint — and the seat rides
    // beside it so the pick match binds the pair as its own, never a second
    // Option walk of the wiring.
    let seat_launch = b
        .wiring
        .sample_seat
        .as_ref()
        .map(|seat| (seat, SeatWidth::of_launch(seat, &rows_in, mq)));
    // (the pair is `Copy` — a reference and a `Copy` width — so this binding
    // copies it and the pick match below binds its own copy of the same one)
    if let Some((seat, seat_width)) = seat_launch {
        let laws: Vec<RowLaw<'_>> = match seat_laws {
            // A PROMPT CHUNK: one request, row `r` at its own position, a pad row clamped to the
            // last real one (the consumed pick is the last baked row's, so the pad row carries the
            // draw the last real row would).
            SeatLaws::Prompt { draw } => (0..seat_width.get())
                .map(|r| RowLaw {
                    draw,
                    step: rows_in.pos(r) as u32,
                })
                .collect(),
            // A DECODE: live row `s` runs its own request's draw at its own position, and a pad row
            // runs live 0's ([`pad_law_of`]).
            SeatLaws::Batch { draws } => {
                let live_pos: Vec<u32> = (0..live).map(|s| rows_in.pos(s) as u32).collect();
                batch_seat_laws(draws, &live_pos, seat_width.get())?
            }
        };
        acts_i32.extend(sample_binds(seat, seat_width, &laws)?);
    }
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
    // ⭐ THE SEAT IS THE PICK'S OWN MATCH — every suffix arm takes the
    // launch's own (seat, width) pair as the match's binding, never a second
    // Option and never an unwrap over a guard established lines earlier: a
    // launch whose suffix runs and whose bundle mints no draw is refused AT
    // THE ARM, before anything stages.
    let picked = match (want_logits, seat_launch) {
        // ⭐ THE ONE DECODE PATH, SUBMITTED: stage, launch, enqueue the idx-stick read, return. The
        // minted sampler consumes the `[rows, vocab]` result on the card and leaves the idx stick in
        // its place; this arm defers the ONE raw D2H of that stick — never a logits read, never a
        // host argmax — and the caller collects it through `collect_decode_tokens`.
        (LogitsWanted::PerRequestDeferred, Some((seat, seat_width))) => {
            // ⭐ #32 (E) — THE OVERLAP PROBE'S ASK: had the PREVIOUS step's stick read already
            // landed? NOT-yet-landed is "card busy at submit".
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
            // The readback's own byte size — the seat's typed fact (the idx
            // stick's placed row, minted once at the bake), never a
            // recomputed `mq · 64 · 4`, at the launch's one typed width.
            let read = pf
                .defer_read_tensor_ring(
                    scratchy_target_spyre::bundle_code::PlaceId::Act(b.result_id as u32),
                    seat_width.get() * seat.idx_row_words as usize * 4,
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
                rows: seat_width,
                // The collect half's own typed readback fact — the idx
                // stick's row size, carried beside the launch's typed width
                // so the decode law never re-derives it.
                idx_row_words: seat.idx_row_words,
                result_id: b.result_id as u32,
                vocab,
            }));
        }
        // A deferred decode whose bundle mints no draw — refused at the arm,
        // before anything stages (the match's own binding, no second Option).
        (LogitsWanted::PerRequestDeferred, None) => {
            return Err(werr(format!(
                "bundle has no sampler seat (result t{}) — a deferred decode cannot pick \
                 without the minted draw",
                b.result_id
            )));
        }
        // A middle prompt chunk: no suffix runs and nothing is read back.
        (LogitsWanted::None, _) => {
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
        // sampler: the suffix runs, the sampler consumes the folded lm-head row there, and the idx
        // stick comes back raw, exactly like a decode's.
        (LogitsWanted::LastRow, Some((seat, seat_width))) => {
            pf.stage_step_acts_seeded(&acts, &[], &acts_i32, &seeds)
                .and_then(|()| pf.run_staged_step(mq, seq_pos, StepLogits::LeftOnDevice))
                .map_err(|e| {
                    werr(format!(
                        "superdsc prefill run_step (mq={mq}, at={seq_pos}): {e}"
                    ))
                })?;
            // The prompt-final width is the launch's one typed width — the
            // mint's verbatim prompt case, the same value the staging block
            // bound the seat's rows at.
            let picked = read_sampled_tokens(pf, seat, seat_width, b.result_id as u32, vocab)?;
            log_seat_readback(
                pf,
                seat,
                b.wiring.load_consts,
                &picked,
                &format!("prefill-final mq={mq} at={seq_pos}"),
            );
            Some(picked)
        }
        // A prompt-final pick whose bundle mints no draw — refused at the arm.
        (LogitsWanted::LastRow, None) => {
            return Err(werr(format!(
                "bundle has no sampler seat (result t{}) — a prompt-final pick cannot run \
                 without the minted draw",
                b.result_id
            )));
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
    /// ⭐ RAW LE 32-bit binds — the paged step's index tables and the chain's seed words. Kept apart
    /// from [`Self::acts`] because they need a DIFFERENT shadow write: a value is re-encoded to the
    /// card's fp16 (`ieee_to_sen_bytes`), an index must be copied verbatim.
    pub(crate) acts_i32: Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)>,
    /// ⭐ THE CHAIN'S SEEDS — per rank, so not in [`Self::acts_i32`]: every card of a sharded world
    /// binds its own vocabulary block's words (`SuperDscSession::stage_step_acts_seeded`).
    pub(crate) seeds: Vec<(
        scratchy_target_spyre::bundle_code::PlaceId,
        scratchy_target_spyre::forward_tape::ChainSeed,
    )>,
    /// ⭐⭐⭐ THE STICKS THIS LAUNCH HANDS TO THE CARD — filled by the ChainSeed skip below.
    /// An `OnCard` launch's sticks are the CARD'S: the previous step's ChainStage writes them,
    /// and the executor's persistent bind maps must let them go, or every later submit's
    /// refill re-uploads the last `Host` seed over the card's own pick before the gather
    /// reads it (the stuck-seed repetition, pinned on card 2026-10-27). The caller walks
    /// this list through [`SuperDscSession::hand_to_card`] after the tape plays.
    pub(crate) card_owned_sticks: Vec<scratchy_target_spyre::bundle_code::PlaceId>,
    /// Set when the tape's device step runs; the caller reads the logits from there.
    pub(crate) launched: bool,
}

impl scratchy_subtile::host_tape::Launcher for ForwardLauncher<'_> {
    type Error = String;

    fn host_call(&mut self, k: scratchy_subtile::host_tape::KernelId) -> Result<(), Self::Error> {
        let step = self
            .steps
            .get(k.0 as usize)
            .ok_or_else(|| format!("no forward kernel {}", k.0))?;
        // ⭐⭐ THE CHAIN'S SEED SKIP — THE `OnCard` ARM. A
        // stable seating's seeds are the card's own ChainStage writes (the previous step's pick,
        // decomposed into the persistent sticks at its tail), so this launch stages NO seed and
        // the device's gather reads the sticks as they were left. Binding a host seed here would
        // need the PREVIOUS step's pick before this launch's submit — the drain fence the seed
        // law exists to delete.
        //
        // ⛔⛔⛔ AND THE SKIP MUST GO FURTHER THAN "STAGE NOTHING". A previous `Host` launch's
        // seed bind sits in the executor's PERSISTENT maps (`bind_input_raw` lands in `bound`),
        // and the refill re-sends every unkeyed `bound` entry on EVERY submit — so an `OnCard`
        // launch that only skips staging would have its preamble H2D overwrite the sticks with
        // the LAST host seed every step, erasing the card's own ChainStage pick before the
        // gather reads it. The skip therefore NAMES the two sticks for the caller to hand to
        // the card ([`Self::card_owned_sticks`]).
        if step.kernel == scratchy_target_spyre::forward_tape::ForwardKernel::ChainSeed
            && self.inputs.chain_seeds.is_none()
        {
            match self.shape.phase {
                scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain } => {
                    let s = scratchy_target_spyre::bundle_code::PlaceId::Act(chain.s_src);
                    let l = scratchy_target_spyre::bundle_code::PlaceId::Act(chain.l_src);
                    // THIS STEP'S OWN STICK, once — the s step and the l step each name theirs.
                    if step.tensor == s || step.tensor == l {
                        self.card_owned_sticks.push(step.tensor);
                    } else {
                        return Err(format!(
                            "ChainSeed step on t{} — the chain names t{} (s) and t{} (l), and a \
                             seed landing anywhere else is a stick the gather never reads",
                            step.tensor.tid(),
                            chain.s_src,
                            chain.l_src
                        ));
                    }
                }
                scratchy_target_spyre::forward_tape::ForwardPhase::Prefill => {}
            }
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
                Err("h2d with nothing staged".to_string())
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
            // ⭐ THE CHAIN'S SEED — per rank: the session builds each card's words from its own
            // vocabulary block at the bind, raw u32 LE through the same raw-integer channel the
            // index table rides, never the IEEE→SEN convert an f16 arm would run.
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

/// ⭐ THE SEAT LAWS ONE LAUNCH RUNS ITS ROWS UNDER — the two shapes a caller
/// can know them in, and ONLY those two: a prompt-final chunk serves ONE
/// request whose rows are consecutive positions of one sequence, and a decode
/// batch serves one request PER LIVE ROW. The variant must match the launch's
/// own arm (`per_request`); a mismatch is refused where the seat is staged,
/// never silently resolved.
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
fn sample_binds(
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    width: SeatWidth,
    laws: &[RowLaw<'_>],
) -> ExecutorResult<Vec<(scratchy_target_spyre::bundle_code::PlaceId, Vec<u8>)>> {
    let rows = width.get();
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
                Some(d) => format!("r{r}:seed={}@step{}:inv_t{}", d.seed, law.step, d.inv_t),
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
    let word_of = |k: usize, r: usize| -> i32 { seat_words(&laws[r]).0[k] };
    // ⭐ THE SEAT STAGES THROUGH THE ONE COL LAW — `stage_col_stick_heads`,
    // the same fn the bake's ten UNIFORM tables spread through
    // (`LoadConst::card_bytes`), so the seat's words and the uniform tables
    // can never disagree about where a row's word lives. The strides are the
    // SEAT'S OWN typed facts — `seed_row_words` and `gate_row_words`, minted
    // once at the bake from the placement's own reservation laws — so the
    // staging and the placement cannot drift by construction, never by a
    // runtime comparison.
    let stride_bytes = seat.seed_row_words as usize * 4;
    // ⭐⭐⭐ THE GATE ALONE STAGES THE β RESERVATION (front-b-noise, auditor
    // ruling 2026-10-27): its chain crosses the i32→f32 convert, so its table
    // reserves the convert's 64-element device sweep — `stage_gate_sweep`
    // writes the gate at row r's stick HEAD (the same byte the one-stick law
    // ever stated) and zero-fills the sweep's second sticks.
    let gate_stride_bytes = seat.gate_row_words as usize * 4;
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
        seat_words(&laws[r]).1.to_le_bytes()
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
/// ⭐ THE STICK IS SCATTERED, NOT ROW-MAJOR. A `[rows, W]` SENUINT32 placement
/// (W the seat's minted `idx_row_words`) stores rows INTERLEAVED at the
/// format's own 32-element stick — `dev_off_stk`
/// with `SenUint32::ELEMS_PER_STICK` — so row r's lane 0 is word `r·32`, NOT
/// `r·W`: measured on the card (probe, 2-row rung: each 64-word physical block
/// held `32×row0's idx` then `32×row1's idx`, both blocks identical), where the
/// row-major stride handed EVERY row row 0's pick. At `rows = 1` the scattered
/// offset is 0 and the solo readback is byte-identical.
fn read_sampled_tokens(
    session: &mut SuperDscSession,
    seat: &scratchy_target_spyre::wiring::SampleSeat,
    width: SeatWidth,
    result_id: u32,
    vocab: usize,
) -> ExecutorResult<Vec<u32>> {
    let bytes = session
        .read_tensor_raw(
            scratchy_target_spyre::bundle_code::PlaceId::Act(result_id),
            width.get() * seat.idx_row_words as usize * 4,
        )
        .map_err(|e| werr(format!("sampler idx readback (t{result_id}): {e}")))?;
    decode_idx_stick(&bytes, width.get(), seat.idx_row_words as usize, vocab)
}

/// ⭐ #32 (c3) — THE COLLECT HALF of a submitted decode step: park on the idx
/// read's own completion (the loop's ONE fence — this read's D2H, then the
/// shadow's quiet point, never a whole-stream sync), then apply the exact
/// [`read_sampled_tokens`] decode law to the bytes the ring slot holds. The
/// SAME door whether the caller collects in-call (a schedule that cannot
/// defer) or from the drain — the only difference is when, never what.
pub(crate) fn collect_decode_tokens(
    session: &mut SuperDscSession,
    sub: &SubmittedDecode,
) -> ExecutorResult<Vec<u32>> {
    // The probe's collect-half clocks: the fence is the wait (at depth ≥ 2 it waits
    // everything enqueued behind the read too, which is the pipeline's own cadence,
    // not an artifact), the pick is the decode. And this collect is the outstanding
    // count's other half — the submit incremented it, this decrements it, so the
    // depth row counts exactly the submissions still uncollected.
    let t_fence = std::time::Instant::now();
    let bytes = session
        .finish_deferred_read_raw(&sub.read)
        .map_err(|e| werr(format!("sampler idx readback (t{}): {e}", sub.result_id)))?;
    let t_pick = std::time::Instant::now();
    let toks = decode_idx_stick(
        &bytes,
        sub.rows.get(),
        sub.idx_row_words as usize,
        sub.vocab,
    );
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
///     The card's noise intermediate g is NOT addressable from the wiring (the
///     draw's internals are lower-layer ktir synthetics, no tids), so the
///     readback brackets it: the staged inputs on one side, the pick on the
///     other. A failed read logs its tid and stops the line — the readback never
///     gates the run.
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
                b.as_chunks::<4>()
                    .0
                    .iter()
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
    for (k, seed_tid) in seat.seed_words.iter().enumerate() {
        match le_words(session, *seed_tid, rows * seat.seed_row_words as usize) {
            Ok(ws) => {
                for (r, row) in words.iter_mut().enumerate() {
                    row[k] = ws
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
    // The f32 stick width — the launch's readback stride for the seat's f32 word
    // tables (inv_t): each row's word at its stick head. Derived from the
    // format's own stick (the ladder's one derivation), never spelled here.
    const FP32_ELEMS_PER_STICK: usize =
        <ktir_superdsc::superdsc_opspec::Fp32 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
            as usize;
    let invt: Vec<f32> = match le_words(session, seat.invt, rows * FP32_ELEMS_PER_STICK) {
        Ok(ws) => (0..rows)
            .map(|r| ws.get(r * FP32_ELEMS_PER_STICK).copied().unwrap_or(0))
            .map(|w| f32::from_bits(w as u32))
            .collect(),
        Err(e) => {
            tracing::debug!("SEAT_READBACK {ctx}: inv_t read failed ({e})");
            return;
        }
    };
    // The hash chain's uniform tables at row 0's and row 1's stick heads.
    // The u32 stick width is the same readback stride for the seat's I32 word
    // tables. The u32 and f32 sticks are both 32 lanes today, but they are TWO
    // laws (one per format), each read from its own format — a stick widening
    // that moved one and not the other must move its own reader, not both by
    // accident of a shared literal.
    const U32_ELEMS_PER_STICK: usize =
        <ktir_superdsc::superdsc_opspec::SenUint32 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
            as usize;
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
                match le_words(session, tid, trows * U32_ELEMS_PER_STICK) {
                    Ok(ws) => {
                        let w0 = ws.first().copied().unwrap_or(-1);
                        let r1 = if trows > 1 {
                            format!(
                                "/r1=0x{:08X}",
                                ws.get(U32_ELEMS_PER_STICK).copied().unwrap_or(-1)
                            )
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
    let stk = <ktir_superdsc::superdsc_opspec::Fp16 as ktir_superdsc::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
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
                    let e =
                        ktir_superdsc::sdsc_abstract::dev_off_stk(&[rows, pad_v], 1, &[r, j], stk);
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
            tracing::debug!(
                "SEAT_READBACK {ctx}: logits row t{} read failed: {e}",
                seat.row
            );
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
        let rung = ktir_superdsc::sdsc_abstract::RungWidth::of_baked_rows(width as u32)
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
                seeds: ktir_superdsc::sdsc_abstract::SeedSource::Host(&chunk_toks),
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

/// ⭐ THE PAD CLAMP'S OWN PIN (auditor ruling 2026-10-27): a pad row's staged
/// seat words are LIVE 0'S — for a greedy live 0 and a sampled one.
#[cfg(all(test, feature = "spyre-hw"))]
mod pad_clamp_tests {
    use super::*;

    /// The seat the BAKE'S OWN MINT mints at `rows` baked rows — built the way
    /// the expansion builds it (`mint_sample` over a one-request lm-head graph,
    /// read back by `sample_seat_of_ir`), never spelled field by field, so the
    /// staging this module pins runs against the real seat's own strides: the
    /// seat cannot drift from the law because it IS the law's own output.
    fn seat_of(rows: u32) -> scratchy_target_spyre::wiring::SampleSeat {
        use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
        use scratchy_subtile::subtile_ir::{
            SampleRows, SourceShape, SubOp, lower_region, mint_sample,
        };
        // K=1, vocab 5 — the one-request lm-head shape the mint admits, the
        // same fixture the spyre crate's own mint tests pin.
        let input = LoweringInput {
            exported: None,
            sources: vec![SourceShape { rows, cols: 1 }, SourceShape { rows, cols: 5 }],
            ops: vec![OpDesc {
                op: SubOp::MatmulTile {
                    n: 5,
                    weight: GemmWeight::Dense,
                },
                m: rows,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
        };
        let nz = std::num::NonZeroU32::new(64).expect("64 is a whole stick");
        let mut minted = lower_region(&input, nz);
        mint_sample(&mut minted, SampleRows::All);
        scratchy_target_spyre::lower_subtile_tape_to_ktir::sample_seat_of_ir(&minted)
            .expect("the minted graph's seat derives")
            .expect("the mint appends exactly one Sample node")
    }

    /// Two live rows in a 4-row rung — slots 2 and 3 are padding — at positions 10 and 11.
    const TWO_OF_FOUR: [u32; 2] = [10, 11];

    /// Every staged bind's pad-row words equal row 0's — the four seed tables
    /// and inv_t alike. EVERY word sits at its row's STICK HEAD
    /// (the seat's own `seed_row_words` stride): the gate table's reservation is wider (the
    /// i32→f32 convert's device sweep, two sticks per row) but the sweep's
    /// second sticks are the reservation's own zero tail — the word the
    /// kernel reads is still the head, at the same stride as every other
    /// table.
    fn assert_pads_stage_live_0(
        seat: &scratchy_target_spyre::wiring::SampleSeat,
        laws: &[RowLaw<'_>],
        ctx: &str,
    ) {
        // The width comes through the REAL DOOR — `of_launch` over a real
        // `LaunchRows` (a sequence of four real rows at positions 10..13, the
        // TWO_OF_FOUR rung's own shape), never a spelled number: the staging
        // this pin checks is the same mint every launch reads.
        let rows_in = LaunchRows::Sequence {
            blocks: &[],
            rows: scratchy_target_spyre::paged_step::StepRows::new(10, 4).expect("four real rows"),
        };
        let binds = sample_binds(
            seat,
            SeatWidth::of_launch(seat, &rows_in, seat.rows as usize),
            laws,
        )
        .expect("the seat stages");
        let head_stride = seat.seed_row_words as usize * 4;
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
        let positions: Vec<u32> = (0..4).map(|i| 10 + i).collect();
        let owned: Vec<SampledDraw> = (0..4)
            .map(|i| SampledDraw {
                seed: 1000 + i as u64,
                inv_t: 0.5,
            })
            .collect();
        let draws: Vec<Option<&SampledDraw>> = owned.iter().map(Some).collect();
        let laws = batch_seat_laws(&draws, &positions, 4).expect("the full laws lay");
        assert_eq!(laws.len(), 4);
        for (s, row) in laws.iter().enumerate() {
            let law = row.draw.unwrap_or_else(|| {
                panic!("row {s} of a FULL rung staged greedy — no row is a pad")
            });
            assert_eq!(law.seed, 1000 + s as u64, "row {s} runs its OWN draw");
            assert_eq!(row.step, 10 + s as u32, "row {s} keeps its own step salt");
        }
    }

    #[test]
    fn a_greedy_live_0_clamps_its_pads_to_greedy() {
        let seat = seat_of(4);
        let draws: [Option<&SampledDraw>; 2] = [None, None];
        let laws = batch_seat_laws(&draws, &TWO_OF_FOUR, 4).expect("the greedy laws lay");
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
        let laws = batch_seat_laws(&draws, &TWO_OF_FOUR, 4).expect("the sampled laws lay");
        for r in [2usize, 3] {
            let pad = laws[r]
                .draw
                .expect("the pad clamps live 0's draw, not greedy");
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

    /// ⭐⭐⭐ THE ONCARD HAND-OFF'S OWN PIN (card-pinned 2026-10-27): an `OnCard` decode
    /// launch must NAME its two sticks for the caller to hand to the card, because the
    /// executor's refill re-sends every unkeyed `bound` entry on every submit — a
    /// previous `Host` launch's seed bind left in the maps would overwrite the card's own
    /// ChainStage pick before the gather reads it (the stuck-seed repetition: every
    /// batched text from the third token on was the seed token re-emitted, while solo —
    /// which re-seeds `Host` every step — stayed byte-identical). The skip is the launch's
    /// ONE fact about the sticks, exactly as the held causal key is its one fact about
    /// the mask.
    #[test]
    fn an_oncard_launch_names_its_sticks_and_a_host_launch_names_none() {
        use scratchy_subtile::host_tape::{HostTape, play};
        use scratchy_target_spyre::forward_tape::{
            ForwardInputs, ForwardShape, operands, tape_steps,
        };

        let chain = scratchy_target_spyre::wiring::ChainWiring {
            s_src: 41,
            l_src: 43,
            k: 3,
            hidden: 2048,
            replaces: 7,
            block: 65536,
        };
        let shape = ForwardShape {
            rows: 1,
            embed_src: 1,
            cos_srcs: &[],
            sin_srcs: &[],
            paged: None,
            n_consts: 0,
            phase: scratchy_target_spyre::forward_tape::ForwardPhase::Decode { chain },
        };
        let steps = shape.steps();
        let ops = operands(&steps);
        let ftape = tape_steps(&steps, &ops);

        // (the hand-off list, whether the seeds staged, and the tape's own launch flag)
        let play_with = |chain_seeds: Option<&'static [usize]>| {
            let inputs = ForwardInputs {
                hidden: 2048,
                embed_tokens: &[],
                tokens: &[],
                chain_seeds,
                positions: &[],
                consts: &[],
                stick_major: true,
                paged: None,
            };
            let mut launcher = ForwardLauncher {
                shape: &shape,
                steps: &steps,
                inputs: &inputs,
                staged: None,
                acts: Vec::new(),
                acts_i32: Vec::new(),
                seeds: Vec::new(),
                card_owned_sticks: Vec::new(),
                launched: false,
            };
            play(&HostTape { steps: &ftape }, &mut launcher).expect("the decode tape plays");
            assert!(launcher.launched, "the tape must reach its device step");
            (launcher.card_owned_sticks, launcher.seeds)
        };

        // OnCard: the two ChainSeed steps stage nothing and the skip names EXACTLY the
        // chain's own sticks — the list the caller walks through `hand_to_card`.
        let (oncard, oncard_i32) = play_with(None);
        assert_eq!(
            oncard,
            vec![
                scratchy_target_spyre::bundle_code::PlaceId::Act(chain.s_src),
                scratchy_target_spyre::bundle_code::PlaceId::Act(chain.l_src),
            ],
            "the OnCard skip names the chain's own two sticks, once each"
        );
        assert!(
            oncard_i32.is_empty(),
            "an OnCard launch stages no seed words"
        );

        // Host: the seeds stage (per-row words through the raw channel) and NOTHING is
        // handed to the card — a Host launch owns its sticks this step.
        let (host, host_i32) = play_with(Some(&[1318]));
        assert!(
            host.is_empty(),
            "a Host-seeded launch names no card-owned stick"
        );
        assert_eq!(
            host_i32.len(),
            2,
            "the s and the l seed both stage through the raw channel"
        );
        // ⭐ EACH RANK'S SEED IS ITS OWN BLOCK'S. With the table cut into two 32768-column blocks,
        // token 1318 is rank 0's: rank 0 seeds its slab and lane, rank 1 the sentinel's zero row.
        type W2 = scratchy_target_spyre::bundle_code::collective::TpWorld<2>;
        let (r0, r1) = (
            scratchy_target_spyre::bundle_code::collective::Rank::<W2>::of::<0>(),
            scratchy_target_spyre::bundle_code::collective::Rank::<W2>::of::<1>(),
        );
        let mut split = host_i32[1].1.clone();
        split.chain.block = 32768;
        let alone = host_i32[1].1.words::<W2>(None);
        assert_eq!(
            split.words(Some(r0)),
            alone,
            "the owner seeds the token's own ramp"
        );
        assert_eq!(
            split.words(Some(r1)),
            scratchy_subtile::subtile_ir::chain_l_sentinel_ramp(chain.hidden),
            "a rank that does not hold the token seeds the zero row"
        );
    }
}
