// SPDX-License-Identifier: Apache-2.0
//! #32 tuning, step 1 — WHERE ONE PIPELINED DECODE STEP'S HOST TIME GOES, AND HOW
//! DEEP THE PIPELINE ACTUALLY IS.
//!
//! The submit half of a pipelined decode step (`run_prefill_batch`'s deferred decode
//! arm) clocks its host segments and flushes them here, one lock per step: the act bind
//! (keyed causal bind + act diff + seed build), the submit doorbell, the stick D2H
//! defer, the j+1 staging, and the whole submit body, so the residual (total − the
//! four) is visible. The collect half (`collect_decode_tokens`) clocks the fence wait
//! and the pick decode the same way. And the depth row — the ruling's own demand —
//! counts the submissions outstanding at each submit: ≥ 2 there is the ONLY evidence
//! the pipeline is real, and 1 everywhere is the measurement that proved it never was.
//!
//! ⛔ AN INSTRUMENT MAY PRINT; IT MUST NOT COST. Accumulation is unconditional
//! — a few `Instant` reads and one uncontended mutex lock against a ~24 ms
//! step — because the worker cannot know bench-ness and a `Duration` add is
//! noise. The PRINT is the only gated thing, and its gate (`--bench`) never
//! leaves the CLI: [`step_segment_table`] hands the formatted table to whoever
//! asks, `None` when no step ever recorded (every backend that records no steps,
//! every non-decode run). No env toggles, no numerics touched.
//!
//! The `Timers`/`Diag` idiom (a process-wide static behind a mutex) because
//! the step loop holds the session and `Shared` borrows a worker-field
//! accumulator cannot thread through. Cumulative by design: a `--bench` run's
//! N includes the warmup's single decode step and the first-play cold
//! outliers — they land in `max` where they belong; the steady-state read is
//! the mean.
//!
//! ⭐ THE PROMPT-CHUNK TABLE, the same instrument over the prefill's own
//! launches: every prompt chunk the card runs — a middle chunk (no suffix),
//! the prompt-final chunk (its pick is read), a wave of tails on a packed
//! rung — clocks its own submit half and flushes it through
//! [`record_chunk`], printed by [`chunk_segment_table`] as one table per
//! chunk kind beside the decode table. The depth row counts CHUNK
//! submissions outstanding (its own population, never the decode table's);
//! the busy row reads the same prior-read latch the decode arm's does. On
//! the tree this was typed under, a chunk's door waits the card out before
//! the host stages the next chunk, so the table's honest reading is depth 1
//! and busy 0 — the BEFORE the pipelining front is measured against.

use std::sync::Mutex;
use std::time::Duration;

/// One step's SUBMIT-half segments, clocked by the pipelined loop and flushed here in
/// one lock, with the pipeline depth at the submit. `stage` is `None` on a batch's
/// last step (nothing to stage) — absent from that segment's mean, never a recorded
/// zero.
///
/// `pub`, like the module's whole surface, and for a reason the tree states:
/// `record_submit`'s only caller is a target's worker crate, which a build may not
/// link — a `pub(crate)` surface would be dead code in every clippy leg of this
/// crate. `pub` items of a lib crate are reachable by
/// type in every feature combination, which is the law this instrument is built
/// under: no cfg, no allow, nothing dead.
pub struct SubmitSegs {
    pub bind: Duration,
    pub submit: Duration,
    pub defer: Duration,
    pub stage: Option<Duration>,
    pub total: Duration,
    /// Submissions outstanding at this submit — the in-flight depth, THIS step
    /// included. The row that proves or refutes the pipeline: 1 everywhere is
    /// serial with a fence's name on it; ≥ 2 is steps genuinely in flight.
    pub depth: usize,
    /// ⭐ #32 (E) — CARD BUSY AT SUBMIT: the PREVIOUS step's stick read had
    /// NOT yet landed when this step's submit work began (a non-blocking
    /// peek of the prior read's completion signal). The overlap fraction —
    /// the depth row's honest twin: depth counted 1.99 even in the build
    /// whose whole-stream fence idled the card at every submit (the
    /// executor's pre-schedule inflated the count without device overlap),
    /// while this row reads `false` exactly when the card had nothing left
    /// to execute. `false` on the first step ever submitted (no prior read
    /// exists — nothing was in flight).
    pub busy: bool,
}

/// ⭐ THE SERIALIZATION HUNT'S OWN ROWS — what each `execute_model` found on entry and
/// where each step's collect ran, recorded by the worker's own executor arm. The depth
/// and busy rows read the CARD's state at submit; these read the WORKER's, and the two
/// sides of that pair are what the hunt needed: a pipeline that pre-schedules lands a
/// `Step` message while the previous step is still uncollected (entry ≥ 1), while a
/// starved one finds the queue empty at every entry (entry 0) and the knock — not the
/// pre-schedule — is what retires each step. `exit` says which world the exit itself
/// chose (the door's deferral, or the no-door in-call drain); `drain` says who actually
/// collected (a knock, a seating churn, the in-call exit, shutdown).
///
/// `pub` for the reason [`SubmitSegs`] states: the only caller is a target's worker
/// crate, which a build may not link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitSite {
    /// The door was set: the owed rows travelled out in a resolver for the caller
    /// to knock and wait on.
    Deferred,
    /// No door: the exit drained every in-flight step itself, in-call.
    InCall,
    /// Nothing was owed — the schedule's rows all answered in the output (a chunk
    /// wave, a readless step).
    NothingOwed,
}

/// Who drained each in-flight step — the collect's caller, recorded once per step
/// popped from the worker's in-flight queue. `pub` for the reason [`SubmitSegs`]
/// states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrainSite {
    /// A knock: the executor's `drain_ready` (fed first by `drain_knock`'s pump), sent
    /// by a deferred output's resolver.
    Knock,
    /// A seating churn (a rung or seating change) draining the queue it replaces.
    Churn,
    /// The no-door in-call exit.
    Exit,
    /// Shutdown — no deferred step outlives the worker.
    Shutdown,
}

/// One step's COLLECT-half segments — the fence wait (at depth ≥ 2 it waits
/// everything enqueued behind the read too, which is the pipeline's own cadence, not
/// an artifact) and the pick decode. Recorded where the reads are fenced, which under
/// deferral is a later drain, not the submitting step's own call.
///
/// `kind` routes the row: a DECODE step's collect lands in the decode table; a
/// CHUNK's lands in its own kind's chunk table (the prompt-final pick's fence — the
/// one host wait a prompt meets — files under `Chunk`, where its submit did).
pub struct CollectSegs {
    pub kind: StepKind,
    pub fence: Duration,
    pub pick: Duration,
}

/// ⭐ THE PROMPT-CHUNK TABLE'S KIND — keyed by the launch's own facts, never by a
/// model or a geometry: a chunk that runs no suffix and reads nothing back, the
/// chunk that finishes its prompt (the suffix runs and its pick is read), and a
/// wave of tails on a packed rung. The table prints one block per kind, so a
/// kind that never ran is absent from the output by type, never as a zero row.
///
/// `pub` for the same reason [`SubmitSegs`] states: the only caller is a
/// target's worker crate, which a build may not link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkKind {
    /// A middle chunk: no suffix runs, nothing is read back.
    Middle,
    /// The prompt-final chunk: the suffix runs and its pick is read in-call.
    Final,
    /// A wave of continuation tails on a packed rung: one launch, `s` windows.
    Packed,
}

/// ⭐ WHICH TABLE A SUBMITTED STEP'S SEGMENTS FILE UNDER — the routing fact the owed
/// pick carries from its submit to its collect, so the two land in the same table:
/// `Decode` is the decode table, `Chunk(k)` the chunk table's kind `k`. Keyed by the
/// launch's own rows and end, never by a model or a geometry.
///
/// `pub` for the same reason [`SubmitSegs`] states: the only caller is a target's
/// worker crate, which a build may not link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// A decode batch — the decode table's own population.
    Decode,
    /// A prompt chunk or a wave of tails — the chunk table's, at its kind.
    Chunk(ChunkKind),
}

impl ChunkKind {
    /// Every kind, in the order the table prints them — the accumulator's own
    /// index law, stated once here and consumed by `record_chunk` alone.
    const ALL: [ChunkKind; 3] = [ChunkKind::Middle, ChunkKind::Final, ChunkKind::Packed];

    /// The kind's own words for its table's header line.
    const fn label(self) -> &'static str {
        match self {
            ChunkKind::Middle => "middle — no suffix runs, nothing read",
            ChunkKind::Final => "prompt-final — the suffix runs, its pick read",
            ChunkKind::Packed => "a wave of tails on a packed rung",
        }
    }

    /// The kind's slot in the accumulator — the ONE place a kind becomes an
    /// index, so no caller can misfile one.
    const fn slot(self) -> usize {
        match self {
            ChunkKind::Middle => 0,
            ChunkKind::Final => 1,
            ChunkKind::Packed => 2,
        }
    }
}

/// One prompt-chunk launch's SUBMIT-half segments, clocked by the chunk's submit
/// and flushed here in one lock: `stage` (everything the host staged before the
/// door), `door` (the submit — every card's binds, then its doorbell, no wait),
/// and `fence` (the prompt-final pick's collect, recorded through
/// [`record_collect`] where the one collect door runs it — `None` at the submit,
/// absent from that segment's mean, never a recorded zero). `depth` counts the
/// card's submissions outstanding at the submit, this launch included — the lead
/// card's own step-queue depth, every step of the engine's alike (a chunk queued
/// behind a decode owes that decode's work, and the reverse). `busy` is the
/// last-armed landing's peek — the previous step's, whatever kind it was: not yet
/// fired is the card owing work when this chunk queued.
///
/// `pub` for the reason [`SubmitSegs`] states — the caller is a target's worker
/// crate a build may not link.
pub struct ChunkSegs {
    pub kind: ChunkKind,
    pub stage: Duration,
    pub door: Duration,
    pub fence: Option<Duration>,
    pub depth: usize,
    pub busy: bool,
}

/// One segment's running mean/max material: its own count (`stage` records one step
/// fewer per batch than the others; the collect segments record only where a drain
/// ran), its nanosecond total, its max.
struct Acc {
    n: u64,
    ns: u128,
    max_ns: u64,
}

impl Acc {
    /// The const initializer — a static cannot call `Default::default`, and the
    /// zero this spells is the only starting state an accumulator has.
    const ZERO: Acc = Acc {
        n: 0,
        ns: 0,
        max_ns: 0,
    };

    fn add(&mut self, d: Duration) {
        let ns = d.as_nanos();
        self.n += 1;
        self.ns += ns;
        if ns > u128::from(self.max_ns) {
            self.max_ns = ns as u64;
        }
    }
}

/// The depth row's own accumulator: the mean and max of in-flight submissions at each
/// submit, plus the count of submits that ran at each depth (the distribution, not
/// just the average — a pipeline that mostly runs 1 with one lucky 2 says so here).
struct DepthStats {
    n: u64,
    sum: u128,
    max: usize,
    at_depth: [u64; 8],
}

impl DepthStats {
    const ZERO: DepthStats = DepthStats {
        n: 0,
        sum: 0,
        max: 0,
        at_depth: [0; 8],
    };

    fn add(&mut self, depth: usize) {
        self.n += 1;
        self.sum += depth as u128;
        if depth > self.max {
            self.max = depth;
        }
        if depth < self.at_depth.len() {
            self.at_depth[depth] += 1;
        }
    }
}

/// ⭐ #32 (E) — THE BUSY ROW'S OWN ACCUMULATOR: how many submits began with
/// the previous step's stick read still un-landed (the card owed work), over
/// how many submits ran at all. The overlap fraction in one count each —
/// no mean to dilute, because the row's question is a fraction, not a
/// duration.
struct BusyStats {
    n: u64,
    busy: u64,
}

impl BusyStats {
    const ZERO: BusyStats = BusyStats { n: 0, busy: 0 };

    fn add(&mut self, busy: bool) {
        self.n += 1;
        if busy {
            self.busy += 1;
        }
    }
}

/// One chunk kind's own accumulator — the same segment/depth/busy shape the
/// decode table's `Stats` states, per kind, with the chunk table's own span
/// names. `n` is the kind's launch count (a middle chunk records no fence, so
/// the fence accumulator's own count is the only one that can lag `n`).
struct ChunkAcc {
    n: u64,
    stage: Acc,
    door: Acc,
    fence: Acc,
    depth: DepthStats,
    busy: BusyStats,
}

impl ChunkAcc {
    /// The const initializer, same law as [`Acc::ZERO`].
    const ZERO: ChunkAcc = ChunkAcc {
        n: 0,
        stage: Acc::ZERO,
        door: Acc::ZERO,
        fence: Acc::ZERO,
        depth: DepthStats::ZERO,
        busy: BusyStats::ZERO,
    };
}

struct Stats {
    steps: u64,
    bind: Acc,
    submit: Acc,
    defer: Acc,
    stage: Acc,
    total: Acc,
    depth: DepthStats,
    busy: BusyStats,
    fence: Acc,
    pick: Acc,
    /// ⭐ The serialization hunt's rows: what each `execute_model` found on entry
    /// (steps still uncollected), which exit it took, and who drained each step.
    entry: DepthStats,
    exit_deferred: u64,
    exit_in_call: u64,
    exit_nothing_owed: u64,
    drain_knock: u64,
    drain_churn: u64,
    drain_exit: u64,
    drain_shutdown: u64,
    /// The prompt-chunk table's accumulators, one slot per [`ChunkKind`].
    chunks: [ChunkAcc; 3],
}

impl Stats {
    /// The const initializer, same law as [`Acc::ZERO`].
    const ZERO: Stats = Stats {
        steps: 0,
        bind: Acc::ZERO,
        submit: Acc::ZERO,
        defer: Acc::ZERO,
        stage: Acc::ZERO,
        total: Acc::ZERO,
        depth: DepthStats::ZERO,
        busy: BusyStats::ZERO,
        fence: Acc::ZERO,
        pick: Acc::ZERO,
        entry: DepthStats::ZERO,
        exit_deferred: 0,
        exit_in_call: 0,
        exit_nothing_owed: 0,
        drain_knock: 0,
        drain_churn: 0,
        drain_exit: 0,
        drain_shutdown: 0,
        chunks: [ChunkAcc::ZERO, ChunkAcc::ZERO, ChunkAcc::ZERO],
    };
}

fn stats() -> std::sync::MutexGuard<'static, Stats> {
    static S: Mutex<Stats> = Mutex::new(Stats::ZERO);
    S.lock()
        .expect("step_segs: the probe's own lock, never held across a panic")
}

/// Flush one step's submit-half segments. Called once per pipelined step, after the
/// stick read is deferred. (`pub` for the reason `SubmitSegs`'s own doc states — the
/// cfg-gated caller must not make this dead code in a plain clippy leg.)
pub fn record_submit(s: SubmitSegs) {
    let mut t = stats();
    t.steps += 1;
    t.bind.add(s.bind);
    t.submit.add(s.submit);
    t.defer.add(s.defer);
    if let Some(d) = s.stage {
        t.stage.add(d);
    }
    t.total.add(s.total);
    t.depth.add(s.depth);
    t.busy.add(s.busy);
}

/// Flush one step's collect-half segments — the fence and the pick decode, recorded
/// where the reads are fenced (the sync composition, or a drain under deferral). The
/// step's kind routes the row: a decode step's into the decode table, a chunk's into
/// its own kind's chunk table (the fence row its submit left for the collect to fill).
pub fn record_collect(s: CollectSegs) {
    let mut t = stats();
    match s.kind {
        StepKind::Decode => {
            t.fence.add(s.fence);
            t.pick.add(s.pick);
        }
        StepKind::Chunk(k) => {
            t.chunks[k.slot()].fence.add(s.fence);
        }
    }
}

/// ⭐ Flush one `execute_model`'s entry fact: how many steps were still uncollected
/// when this schedule arrived. The pre-schedule's own proof on the worker's side of
/// the channel — entry ≥ 1 means a `Step` landed while the previous step was still
/// in flight; entry 0 at every decode step is the starved pipeline the hunt named.
/// (`pub` for the reason [`SubmitSegs`]'s doc states.)
pub fn record_exec_entry(in_flight: usize) {
    stats().entry.add(in_flight);
}

/// ⭐ Flush the exit's own choice for one `execute_model` — the door's deferral, the
/// no-door in-call drain, or nothing owed. (`pub` for the reason [`SubmitSegs`]'s
/// doc states.)
pub fn record_exit(site: ExitSite) {
    let mut t = stats();
    match site {
        ExitSite::Deferred => t.exit_deferred += 1,
        ExitSite::InCall => t.exit_in_call += 1,
        ExitSite::NothingOwed => t.exit_nothing_owed += 1,
    }
}

/// ⭐ Flush who drained one in-flight step. (`pub` for the reason [`SubmitSegs`]'s
/// doc states.)
pub fn record_drain(site: DrainSite) {
    let mut t = stats();
    match site {
        DrainSite::Knock => t.drain_knock += 1,
        DrainSite::Churn => t.drain_churn += 1,
        DrainSite::Exit => t.drain_exit += 1,
        DrainSite::Shutdown => t.drain_shutdown += 1,
    }
}

/// Flush one prompt-chunk launch's segments (`pub` for the reason [`ChunkSegs`]'s
/// own doc states — the caller is a target's worker crate a build may not link).
pub fn record_chunk(s: ChunkSegs) {
    let mut t = stats();
    let c = &mut t.chunks[s.kind.slot()];
    c.n += 1;
    c.stage.add(s.stage);
    c.door.add(s.door);
    if let Some(d) = s.fence {
        c.fence.add(d);
    }
    c.depth.add(s.depth);
    c.busy.add(s.busy);
}

/// The accumulated per-segment table, for the CLI to print under `--bench`.
/// `None` when no step ever recorded — the answer every backend that records no steps
/// and every non-decode run must give, by type, not by empty string.
pub fn step_segment_table() -> Option<String> {
    let t = stats();
    if t.steps == 0 {
        return None;
    }
    let mut out = format!(
        "[step-segs] {} pipelined decode step(s): host time per segment \
         (mean / max / total, ms)\n",
        t.steps
    );
    let rows = [
        ("bind", "keyed causal bind + act diff + seed build", &t.bind),
        ("submit", "submit_step (the doorbell + refill)", &t.submit),
        (
            "defer",
            "stick D2H defer (the enqueue, ring slot)",
            &t.defer,
        ),
        (
            "stage",
            "j+1 staging (pure host, overlaps step j)",
            &t.stage,
        ),
        (
            "total",
            "whole submit body (residual = total - the four)",
            &t.total,
        ),
        (
            "fence",
            "THE fence (collect half: wait + stick D2H)",
            &t.fence,
        ),
        ("pick", "pick decode (stick word to token)", &t.pick),
    ];
    for (name, what, a) in rows {
        let mean = a.ns as f64 / 1e6 / a.n.max(1) as f64;
        out.push_str(&format!(
            "[step-segs]   {name:<6} ({}): {:>8.3} / {:>8.3} / {:>9.1}  over {} step(s)\n",
            what,
            mean,
            a.max_ns as f64 / 1e6,
            a.ns as f64 / 1e6,
            a.n,
        ));
    }
    // ⭐ THE DEPTH ROW — the ruling's own demand: the pipeline's proof or its
    // refutation, in submits-outstanding at each submit. The distribution prints
    // whole (a mostly-1 pipeline with one lucky 2 says so), and ≥ 2 is the only
    // line any claim of pipelining may stand on.
    let dist = t
        .depth
        .at_depth
        .iter()
        .enumerate()
        .filter(|&(d, &c)| d >= 1 && c > 0)
        .map(|(d, &c)| format!("{d}:{c}"))
        .collect::<Vec<_>>()
        .join(" ");
    out.push_str(&format!(
        "[step-segs]   depth  (submissions outstanding at each submit): mean {:>5.2}, max {}, \
         distribution {}\n",
        t.depth.sum as f64 / t.depth.n.max(1) as f64,
        t.depth.max,
        dist,
    ));
    // ⭐ #32 (E) — THE BUSY ROW, the depth row's honest twin: the fraction of
    // submits that began with the previous step's read still un-landed (the
    // card owed work). ~100% is real overlap; ~0% is a serial pipeline
    // wearing depth-2's clothes — the pre-schedule can inflate the depth row,
    // never this one.
    out.push_str(&format!(
        "[step-segs]   busy   (card owed the previous read at submit): {}/{} = {:.0}%\n",
        t.busy.busy,
        t.busy.n,
        100.0 * t.busy.busy as f64 / t.busy.n.max(1) as f64,
    ));
    // ⭐ THE ENTRY ROW — the pre-schedule's proof on the worker's side: steps still
    // uncollected when each schedule arrived. The chunk population pipes this to 19;
    // the decode tail is the question — entry 0 there at every step is the starved
    // pipeline (the knock retires each step, the pre-schedule sends nothing ahead),
    // entry ≥ 1 is a Step landing on an uncollected one.
    let entry_dist = t
        .entry
        .at_depth
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(d, &c)| format!("{d}:{c}"))
        .collect::<Vec<_>>()
        .join(" ");
    out.push_str(&format!(
        "[step-segs]   entry  (steps uncollected at each execute_model entry): mean {:>5.2}, max {}, \
         distribution {}\n",
        t.entry.sum as f64 / t.entry.n.max(1) as f64,
        t.entry.max,
        entry_dist,
    ));
    // ⭐ THE EXIT ROW — which world the exit itself chose, per schedule.
    out.push_str(&format!(
        "[step-segs]   exit   (the exit's choice): deferred {} / in-call {} / nothing-owed {} over {} \
         schedule(s)\n",
        t.exit_deferred, t.exit_in_call, t.exit_nothing_owed, t.entry.n,
    ));
    // ⭐ THE DRAIN ROW — who collected each in-flight step.
    out.push_str(&format!(
        "[step-segs]   drain  (who drained each in-flight step): knock {} / churn {} / exit {} / \
         shutdown {}\n",
        t.drain_knock, t.drain_churn, t.drain_exit, t.drain_shutdown,
    ));
    Some(out)
}

/// The accumulated prompt-chunk tables, one block per chunk kind that ran, for
/// the CLI to print under `--bench` beside the decode table. `None` when no
/// chunk ever recorded — the answer every backend that records no chunks (the
/// emulator, every non-spyre target) and every decode-only run must give, by
/// type, not by empty string.
pub fn chunk_segment_table() -> Option<String> {
    let t = stats();
    let mut out = String::new();
    for kind in ChunkKind::ALL {
        let c = &t.chunks[kind.slot()];
        if c.n == 0 {
            continue;
        }
        out.push_str(&format!(
            "[step-segs] {} prompt chunk(s) ({}): host time per segment \
             (mean / max / total, ms)\n",
            c.n,
            kind.label(),
        ));
        let rows = [
            (
                "stage",
                "the host staging before the door (the tape, the tables, the seat binds)",
                &c.stage,
            ),
            (
                "door",
                "the submit (every card's binds, then its doorbell — no wait)",
                &c.door,
            ),
            (
                "fence",
                "the prompt-final pick's collect (the read's park, in the one collect door)",
                &c.fence,
            ),
        ];
        for (name, what, a) in rows {
            let mean = a.ns as f64 / 1e6 / a.n.max(1) as f64;
            out.push_str(&format!(
                "[step-segs]   {name:<6} ({}): {:>8.3} / {:>8.3} / {:>9.1}  over {} step(s)\n",
                what,
                mean,
                a.max_ns as f64 / 1e6,
                a.ns as f64 / 1e6,
                a.n,
            ));
        }
        // The depth row, the chunk table's own: chunk submissions outstanding at
        // each submit — 1 everywhere is the serial BEFORE; ≥ 2 is chunks genuinely
        // in flight. The distribution prints whole, same law as the decode table's.
        let dist = c
            .depth
            .at_depth
            .iter()
            .enumerate()
            .filter(|&(d, &n)| d >= 1 && n > 0)
            .map(|(d, &n)| format!("{d}:{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&format!(
            "[step-segs]   depth  (chunk submissions outstanding at each submit): mean {:>5.2}, \
             max {}, distribution {}\n",
            c.depth.sum as f64 / c.depth.n.max(1) as f64,
            c.depth.max,
            dist,
        ));
        // The busy row, the depth row's honest twin over the same prior-read latch
        // the decode arm's reads: the fraction of chunk submits that began with the
        // previous decode read still un-landed. On a solo prefill this is 0 — the
        // card owed nothing when each chunk queued, the measurement this table
        // exists to state.
        out.push_str(&format!(
            "[step-segs]   busy   (card owed earlier work as the chunk queued): {}/{} = {:.0}%\n",
            c.busy.busy,
            c.busy.n,
            100.0 * c.busy.busy as f64 / c.busy.n.max(1) as f64,
        ));
    }
    (!out.is_empty()).then_some(out)
}
