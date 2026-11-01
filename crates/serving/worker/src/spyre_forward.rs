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
#[cfg(not(feature = "spyre-hw"))]
use scratchy_target_spyre::runner::SpyreSession;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::sdsc_runner::{StepLogits, SuperDscSession};

use crate::error::ExecutorResult;
#[cfg(feature = "spyre-hw")]
use crate::spyre_pool::*;
use crate::spyre_types::*;
use crate::spyre_worker::*;
#[cfg(feature = "spyre-hw")]
use scratchy_target_spyre::paged_step::{BatchSeq, StepRows};

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

/// What `run_card_launch` hands back: a finished step's picks, or a
/// submitted decode step whose read is deferred. A decode batch is ALWAYS the
/// latter — one decode path — and a schedule that cannot defer collects it
/// in-call through the same [`collect_decode_tokens`] door.
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
static PRIOR_READ: std::sync::Mutex<Option<scratchy_target_spyre::runtime::ReadPeek>> =
    std::sync::Mutex::new(None);

/// ⭐ THE DEPTH ROW'S OWN COUNT: decode submissions outstanding, incremented at
/// each deferred submit and decremented at each collect — the same population
/// the worker's in-flight queue holds, stated once, process-wide, so the depth
/// row counts exactly the submissions the pipeline has not yet collected.
#[cfg(feature = "spyre-hw")]
static OUTSTANDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// ⭐ WHAT ONE CARD LAUNCH'S ROWS ARE, for the paged step: one sequence's rows — a prompt chunk, or
/// the one-request decode — or one token of each of a decode batch's sequences. Each names its KV
/// blocks as pool pages (`paged_step::page_of_block` of the scheduler's block table), which the
/// step's tables are built from and which nothing here can recompute: which page holds a block is
/// the scheduler's grant, and it churns.
#[cfg(feature = "spyre-hw")]
pub(crate) enum LaunchRows<'a> {
    /// One sequence: `rows.real` tokens from position `rows.start`, in the pages `blocks`.
    Sequence { blocks: &'a [u32], rows: StepRows },
    /// A decode batch: one token per live sequence, in slot order; the rung's rows past them are
    /// padding, written at the null slot (`paged_step::batch_sources`).
    Batch { seqs: &'a [BatchSeq<'a>] },
}

#[cfg(feature = "spyre-hw")]
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
            LaunchRows::Sequence { rows, .. } => rows.real,
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
            LaunchRows::Sequence { rows, .. } => r.min(rows.real.max(1) - 1),
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

/// ⭐ THE SUFFIX'S MINTED SAMPLER, ONE BUNDLE: where the seeds come from, whether
/// the lm-head tail runs, and the per-row seat laws — the three facts that
/// describe the launch's sampling half rather than its forward body.
#[cfg(feature = "spyre-hw")]
pub(crate) struct SamplerLaws<'a> {
    // ⛔ THE SEED LAW, NOT A TOKEN LIST. The device writes a decode step's seeds itself (each
    // step's ChainStage decomposes its pick into the persistent sticks), so the host binds seeds
    // ONLY where the card cannot have: a request's first step on a rung session, a churned seat, a
    // rung switch. `SeedSource::OnCard` is that statement — a stable seating — and binds nothing.
    // A prompt chunk's rows are the prompt's own tokens, always `Host`.
    pub(crate) seeds: scratchy_subtile::sdsc_abstract::SeedSource<'a>,
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
#[cfg(feature = "spyre-hw")]
pub(crate) fn run_card_launch(
    rows_in: LaunchRows<'_>,
    sh: &Shared<'_>,
    pf: &mut SuperDscSession,
    b: &BundleMeta,
    width: scratchy_subtile::sdsc_abstract::RungWidth,
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
            scratchy_target_spyre::paged_step::card_step_sources(&step, blocks, rows)
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
        scratchy_subtile::sdsc_abstract::SeedSource::Host(t) => {
            if t.len() != live {
                return Err(werr(format!(
                    "card launch: {} seed token(s) for {live} live row(s) — one per live row",
                    t.len()
                )));
            }
            (0..mq).map(|r| t[rows_in.live_of(r)]).collect()
        }
        scratchy_subtile::sdsc_abstract::SeedSource::OnCard => {
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
    // The stick-padded row count the constants' tables are laid at (`mq.div_ceil(64)*64`, the
    // emitter's own pad).
    let mq_pad = mq.div_ceil(64) * 64;
    if h % 64 != 0 {
        return Err(werr(format!(
            "card launch: hidden={h} is not a whole number of 64-element sticks"
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
            scratchy_subtile::sdsc_abstract::SeedSource::Host(_) => Some(&tokens[..]),
            scratchy_subtile::sdsc_abstract::SeedSource::OnCard => None,
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
    // ⭐ EVERY LAUNCH OF A BUNDLE THAT MINTS THE DRAW BINDS ITS SAMPLER SEAT, every row of it,
    // before anything runs — a MIDDLE prompt chunk included. Its suffix is skipped, but the
    // seat's five sources are placed in this bundle all the same, and the launch's completeness
    // check (`require_sources_filled`) refuses a placed source nobody wrote. The laws a middle
    // chunk stages are this request's own, so nothing is invented.
    let decodes = matches!(want_logits, LogitsWanted::PerRequestDeferred);
    let suffix_runs = decodes || matches!(want_logits, LogitsWanted::LastRow);
    if suffix_runs && b.wiring.sample_seat.is_none() {
        return Err(werr(format!(
            "bundle has no sampler seat (result t{}) — a launch whose suffix runs cannot \
             pick without the minted draw",
            b.result_id
        )));
    }
    if let Some(seat) = b.wiring.sample_seat.as_ref() {
        // ⭐ A DECODE BATCH'S SEAT IS THE RUNG'S OWN WIDTH. `b` is the one-request decode's meta
        // borrowed for its tensor ids, so its seat states `rows = 1` — a launch at `mq` rows binds
        // and reads `mq` rows of the per-step seat. A prompt-final chunk keeps the prefill rung's
        // own seat verbatim: its draw is the FOLDED single row.
        let mut seat = *seat;
        if matches!(rows_in, LaunchRows::Batch { .. }) {
            seat.rows = mq as u32;
        }
        let seat_rows = seat.rows as usize;
        let laws: Vec<RowLaw<'_>> = match seat_laws {
            // A PROMPT CHUNK: one request, row `r` at its own position, a pad row clamped to the
            // last real one (the consumed pick is the last baked row's, so the pad row carries the
            // draw the last real row would).
            SeatLaws::Prompt { draw } => (0..seat_rows)
                .map(|r| RowLaw {
                    draw,
                    step: rows_in.pos(r) as u32,
                })
                .collect(),
            // A DECODE: live row `s` runs its own request's draw at its own position, and a pad row
            // runs live 0's ([`pad_law_of`]).
            SeatLaws::Batch { draws } => {
                let live_pos: Vec<u32> = (0..live).map(|s| rows_in.pos(s) as u32).collect();
                batch_seat_laws(draws, &live_pos, seat_rows)?
            }
        };
        acts_i32.extend(sample_binds(&seat, &laws)?);
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
    let picked = match want_logits {
        // ⭐ THE ONE DECODE PATH, SUBMITTED: stage, launch, enqueue the idx-stick read, return. The
        // minted sampler consumes the `[rows, vocab]` result on the card and leaves the idx stick in
        // its place; this arm defers the ONE raw D2H of that stick — never a logits read, never a
        // host argmax — and the caller collects it through `collect_decode_tokens`.
        LogitsWanted::PerRequestDeferred => {
            // ⭐ #32 (E) — THE OVERLAP PROBE'S ASK: had the PREVIOUS step's stick read already
            // landed? NOT-yet-landed is "card busy at submit".
            let card_busy_at_submit = PRIOR_READ
                .lock()
                .expect("the overlap probe's own lock")
                .as_ref()
                .is_some_and(|s| !s.fired());
            let t_bind = std::time::Instant::now();
            pf.stage_step_acts(&acts, &[], &acts_i32).map_err(|e| {
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
            let read = pf
                .defer_read_tensor_ring(
                    scratchy_target_spyre::bundle_code::PlaceId::Act(b.result_id as u32),
                    mq * 64 * 4,
                )
                .map_err(|e| {
                    werr(format!(
                        "superdsc decode idx read (mq={mq}, at={seq_pos}): {e}"
                    ))
                })?;
            let t_stage = std::time::Instant::now();
            *PRIOR_READ.lock().expect("the overlap probe's own lock") = Some(read.signal());
            crate::step_segs::record_submit(crate::step_segs::SubmitSegs {
                bind: t_submit.duration_since(t_bind),
                submit: t_defer.duration_since(t_submit),
                defer: t_stage.duration_since(t_defer),
                stage: None,
                total: t_bind.elapsed(),
                depth,
                busy: card_busy_at_submit,
            });
            return Ok(BatchOut::Deferred(SubmittedDecode {
                read,
                rows: mq as u32,
                result_id: b.result_id as u32,
                vocab,
            }));
        }
        // A middle prompt chunk: no suffix runs and nothing is read back.
        LogitsWanted::None => {
            pf.run_step_i32(mq, seq_pos, &acts, &[], &acts_i32, StepLogits::Discarded)
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
        LogitsWanted::LastRow => {
            pf.run_step_i32(mq, seq_pos, &acts, &[], &acts_i32, StepLogits::LeftOnDevice)
                .map_err(|e| {
                    werr(format!(
                        "superdsc prefill run_step (mq={mq}, at={seq_pos}): {e}"
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
                &format!("prefill-final mq={mq} at={seq_pos}"),
            );
            Some(picked)
        }
    };
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
#[cfg(feature = "spyre-hw")]
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

#[cfg(feature = "spyre-hw")]
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

/// ⭐ WHAT ONE CHUNK'S FORWARD PRODUCED — the per-row logits the host sampler
/// reads. One enum (not a bare `Vec<Vec<f32>>`) because the shape is a fact of the
/// step the caller must not be able to confuse with the model's other outs.
pub(crate) enum ChunkOut {
    /// A prompt-only forward: no launch in it picked, nothing to sample. The first
    /// generated token is picked by the DECODE launch that runs the prompt's last
    /// token — never by a host readback.
    Prompt,
    /// TOKENS THE DEVICE PICKED, one per launch this chunk ran that drew, in step
    /// order. The minted sampler's idx stick, read back raw — the only pick that
    /// exists, on the card and on the emulator alike (the bake mints the draw into
    /// both tiers' bundles). The shared readers go through
    /// [`ChunkOut::minted_pick`], so no reader names the variant.
    Tokens(Vec<u32>),
}

impl ChunkOut {
    /// ⭐ THE CHUNK'S OWN MINTED PICKS, one per launch that ran a sampler — the
    /// single door both shared pick readers go through. `None` is every chunk
    /// that picked nothing: a mid-prompt chunk (its suffix never runs), or any
    /// chunk of a build with no minted sampler, where the pick step's own
    /// refusal — not a silent `Prompt` match — is what the caller meets.
    pub(crate) fn minted_pick(&self) -> Option<&[u32]> {
        match self {
            ChunkOut::Tokens(ts) => Some(ts),
            ChunkOut::Prompt => None,
        }
    }
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
fn token_of_idx_word(r: usize, w: i32, vocab: usize) -> ExecutorResult<u32> {
    let t = u32::try_from(w).map_err(|_| {
        werr(format!(
            "the token draw for row {r} is {w} — not a token id"
        ))
    })?;
    if t as usize >= vocab {
        return Err(werr(format!(
            "the token draw for row {r} is {t} >= vocab {vocab} — the sampler's sentinel, never \
             a token"
        )));
    }
    Ok(t)
}

/// ⭐⭐⭐ A PAD SLOT'S LAW IS LIVE 0'S OWN (the pad clamp, auditor ruling
/// 2026-10-27). A decode pad row's forward takes live 0's token and position,
/// its KV write lands at the null slot (`paged_step::batch_sources`) and its
/// pick is never read — so its seat stages live 0's words, one law for greedy
/// and sampled alike, rather than a law no request asked for. Called by every
/// staging walk that lays pad rows, never spelled twice.
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
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
    let toks = decode_idx_stick(&bytes, sub.rows as usize, sub.vocab);
    OUTSTANDING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    let toks = toks?;
    crate::step_segs::record_collect(crate::step_segs::CollectSegs {
        fence: t_pick.duration_since(t_fence),
        pick: t_pick.elapsed(),
    });
    Ok(toks)
}

/// The idx stick's own decode law, shared by the sync readback and the deferred
/// collect so the two cannot drift.
#[cfg(feature = "spyre-hw")]
fn decode_idx_stick(bytes: &[u8], rows: usize, vocab: usize) -> ExecutorResult<Vec<u32>> {
    let words: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    // The idx stick's own format decides the stride — the same law every device
    // tensor's host-side address takes (`dev_off_stk`), never a guessed row pitch.
    let stk = <scratchy_subtile::superdsc_opspec::SenUint32 as scratchy_subtile::superdsc_opspec::DataFormat>::ELEMS_PER_STICK
        as usize;
    (0..rows)
        .map(|r| {
            let off = scratchy_subtile::sdsc_abstract::dev_off_stk(&[rows, 64], 1, &[r, 0], stk);
            token_of_idx_word(r, words.get(off).copied().unwrap_or(-1), vocab)
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
///     The card's noise intermediate g is NOT addressable from the wiring (the
///     draw's internals are lower-layer ktir synthetics, no tids), so the
///     readback brackets it: the staged inputs on one side, the pick on the
///     other. A failed read logs its tid and stops the line — the readback never
///     gates the run.
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
                        &[rows, pad_v],
                        1,
                        &[r, j],
                        stk,
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
#[cfg(feature = "spyre-hw")]
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
#[cfg(feature = "spyre-hw")]
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
        let rung = scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(width as u32)
            .ok_or_else(|| werr("superdsc prefill: the selected rung has zero rows".to_string()))?;
        let out = run_card_launch(
            LaunchRows::Sequence {
                blocks: &blocks,
                rows: StepRows { start: pos, real },
            },
            sh,
            sess,
            pmeta,
            rung,
            SamplerLaws {
                seeds: scratchy_subtile::sdsc_abstract::SeedSource::Host(&chunk_toks),
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

/// Run the prompt tokens `toks` at absolute positions `start..start+toks.len()` on the card's
/// prompt-chunk ladders ([`superdsc_forward_chunk`]).
#[cfg(feature = "spyre-hw")]
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
            "positions {start}..{} exceed the bundle's KV capacity {} — the model's own context",
            start + n,
            b.capacity,
        )));
    }
    // ⭐ NO `hd` HERE EITHER. The rotary tables are built per class (each cos/sin
    // source's own `RotaryKind`), so this function no longer reads a model-wide
    // head dim.
    let h = sh.hidden;
    // ⭐ THE REQUEST'S KV BLOCKS — the scheduler's grant, read out of the one store it was installed
    // in. Its pages already hold the keys of positions `0..start` (this request's earlier steps, or
    // the request whose prefix a cache hit shares), and this forward's write scatters positions
    // `start..start + n` into them; the paged step below checks the grant reaches that far.
    // Block `b` is pool page `b + 1`: page 0 is the null page a padded row writes
    // (`paged_step::page_of_block`).
    let blocks: Vec<u32> = tokens
        .block_table(id)
        .map(|t| {
            t.iter()
                .map(|&b| scratchy_target_spyre::paged_step::page_of_block(b as u32))
                .collect()
        })
        .unwrap_or_default();
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
        // reason: step t feeds the token step t-1 picked). ⭐ THE WORDS RIDE AS
        // THEIR VALUES, not their bits: the gather's eval arm reads them as
        // numbers (`s_stick[..] as usize / k`, `l_stick[..] as usize`) and so do
        // the emulator's doors, which write the same sticks numerically at the
        // model dtype. The words the gather reads are f16-exact: the s stick's
        // lane 0 is `k·s` (granite-3.1-2b: k = 4, s ≤ 768 — a multiple of 4
        // below 4096) and the l stick's word 0 is `l < 64`. A `f32::from_bits`
        // word is a denormal the f16 channel flushes to zero, which seeded every
        // decode step with token 0's row.
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
                let l_sticks = scratchy_subtile::subtile_ir::chain_l_sticks(chain.hidden) as usize;
                let toks32: Vec<u32> = toks.iter().map(|&t| t as u32).collect();
                let s_words = scratchy_subtile::subtile_ir::chain_s_seed(chain.k, &toks32, rows);
                let l_words =
                    scratchy_subtile::subtile_ir::chain_l_seed(&toks32, chain.hidden, rows);
                dynamic.push((
                    chain.s_src as u64,
                    s_words.iter().map(|&w| w as f32).collect(),
                    vec![rows, lanes],
                ));
                dynamic.push((
                    chain.l_src as u64,
                    l_words.iter().map(|&w| w as f32).collect(),
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

    // ⭐ THE PAGED STEP — the page table, the mask stack and the write rows, staged from the
    // request's blocks by `scratchy_target_spyre::paged_step` (the law the bake's numeric gate stages
    // by too), then one fused forward through the resident session. The pools themselves are not
    // staged: every layer's write scatters this step's K/V into them and its walk gathers from them,
    // on the device, step after step.
    {
        let step = b
            .paged
            .ok_or_else(|| werr("this bundle states no paged step — its attention has no pools"))?;
        let rows = scratchy_target_spyre::paged_step::StepRows { start, real: n };
        let srcs = scratchy_target_spyre::paged_step::step_sources(&step, &blocks, &rows)
            .map_err(|e| werr(format!("request {id}: paged step: {e}")))?;
        session
            .set_typed_sources(
                srcs.int32
                    .into_iter()
                    .map(|(tid, bytes, shape)| {
                        (tid as usize, bytes, scratchy_tensors::DType::I32, shape)
                    })
                    .collect(),
            )
            .map_err(|e| werr(format!("paged step: {e}")))?;
        let (mask_tid, mask, mask_shape) = srcs.mask;
        dynamic.push((u64::from(mask_tid), mask, mask_shape));
        // ⭐ ASK FOR THE ROWS THIS FORWARD WROTE, NOT THE WHOLE RESIDENT TENSOR. Decode and prefill
        // share one session, so every tensor is resident at the WIDEST program's row count: a
        // one-row decode step had its outputs decoded at the prefill's `m`. The result is the one
        // readback — `vocab` of the logits row, or, for a rung that draws (its result re-pointed at
        // the idx stick, `[rows, 64]`), the whole stick (`0`).
        let result_elems = if b.sample_seat.is_some() { 0 } else { vocab };
        let wanted: std::collections::HashMap<u64, usize> =
            std::iter::once((b.result_id as u64, result_elems)).collect();
        let outputs: Vec<(u64, usize)> = b
            .output_ids
            .iter()
            .map(|id| (*id as u64, wanted.get(&(*id as u64)).copied().unwrap_or(0)))
            .collect();
        // ⭐ THE MINTED DRAW'S SEAT, STAGED FRESH EVERY STEP. The bake mints the token draw into
        // this rung (`Wiring::sample_seat`), and its five operands go stale the moment a step
        // completes — the step word IS the step's freshness, a held one is a held token. One law
        // per seat row, by the card's own prompt law: row `r` runs this request's draw at
        // `start + min(r, n - 1)`, a pad row clamped to the last real position (at decode `n = 1`
        // and the one row sits at `start`). The words are `seat_words`' — the one statement of the
        // law both tiers lay out — packed `[rows, 1]` at the views' own dtypes (i32 words, f32
        // inv_t), the memref-faithful layout the bake's numeric gate stages them in; the f16
        // channel `run_step` takes would encode them into half the bytes their views span.
        if let Some(seat) = b.sample_seat {
            let rows = seat.rows as usize;
            let draw = req.draw;
            let words: Vec<([i32; 4], f32)> = (0..rows)
                .map(|r| {
                    seat_words(&RowLaw {
                        draw: draw.as_ref(),
                        step: (start + r.min(n - 1)) as u32,
                    })
                })
                .collect();
            let mut typed: Vec<(usize, Vec<u8>, scratchy_tensors::DType, Vec<usize>)> = seat
                .seed_words
                .iter()
                .enumerate()
                .map(|(k, &tid)| {
                    (
                        tid as usize,
                        words.iter().flat_map(|(w, _)| w[k].to_le_bytes()).collect(),
                        scratchy_tensors::DType::I32,
                        vec![rows, 1],
                    )
                })
                .collect();
            typed.push((
                seat.invt as usize,
                words
                    .iter()
                    .flat_map(|(_, inv_t)| inv_t.to_le_bytes())
                    .collect(),
                scratchy_tensors::DType::F32,
                vec![rows, 1],
            ));
            session
                .set_typed_sources(typed)
                .map_err(|e| werr(format!("sampler seat: {e}")))?;
        }
        let out = session
            .run_step(b.prog, dynamic, &outputs)
            .map_err(|e| werr(format!("run_step: {e}")))?;
        tokens.set_tokens_in_pool(id, start + n);
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
        // ⭐ THE PICK IS THE MINTED SAMPLER'S, READ BACK LIKE THE CARD'S. The bake mints the draw
        // into this rung and re-points its result at the idx stick — one word per seat row at lane
        // 0, every lane equal (the store is a whole stick per row) — so no host argmax runs: the
        // emulator's readback is memref-faithful, row `r`'s lane 0 is element `r·width`, and each
        // word passes the same token law the card's decode applies (`token_of_idx_word`). The
        // caller consumes the LAST row's pick, the folded tail's: the last real token's forward.
        match b.sample_seat {
            Some(seat) => {
                let idx = out.get(&(b.result_id as u64)).ok_or_else(|| {
                    werr(format!(
                        "the minted draw's idx stick t{} came back unread",
                        b.result_id
                    ))
                })?;
                let rows = seat.rows as usize;
                let width = idx.len() / rows.max(1);
                tracing::debug!(
                    "EMU_PICK t{} rows={rows} width={width} words={:?} seat={:?}",
                    b.result_id,
                    &idx[..idx.len().min(8)],
                    seat
                );
                let picks = (0..rows)
                    .map(|r| token_of_idx_word(r, idx[r * width] as i32, seat.vocab as usize))
                    .collect::<ExecutorResult<Vec<u32>>>()?;
                Ok(ChunkOut::Tokens(picks))
            }
            None => Ok(ChunkOut::Prompt),
        }
    }
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

/// The smallest prompt-chunk rung that reaches `need` positions, if any does. `rungs` is
/// ascending by reach.
#[cfg(not(feature = "spyre-hw"))]
fn select_prefill(rungs: &[BundleMeta], need: usize) -> Option<&BundleMeta> {
    rungs.iter().find(|b| b.reach() >= need)
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

/// ⭐ THE PAD CLAMP'S OWN PIN (auditor ruling 2026-10-27): a pad row's staged
/// seat words are LIVE 0'S — for a greedy live 0 and a sampled one.
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

    /// Two live rows in a 4-row rung — slots 2 and 3 are padding — at positions 10 and 11.
    const TWO_OF_FOUR: [u32; 2] = [10, 11];

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
                card_owned_sticks: Vec::new(),
                launched: false,
            };
            play(&HostTape { steps: &ftape }, &mut launcher).expect("the decode tape plays");
            assert!(launcher.launched, "the tape must reach its device step");
            (launcher.card_owned_sticks, launcher.acts_i32)
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
    }
}
