// SPDX-License-Identifier: Apache-2.0
//! #32 tuning, step 1 — WHERE ONE PIPELINED STEP'S HOST TIME GOES, AND HOW DEEP THE
//! PIPELINE ACTUALLY IS — for every kind of step a target submits ([`StepKind`]): a
//! decode step and a prompt's chunk, each its own table.
//!
//! The submit half of a pipelined step clocks its host segments and flushes them here,
//! one lock per step: the act bind (the shadow writes, with the shadow's own quiet
//! point), the submit doorbell, the pick's D2H defer (a step that picks), the j+1
//! staging, and the whole submit body, so the residual (total − the four) is visible.
//! The collect half clocks the fence wait and the pick decode the same way. And the
//! depth row — the ruling's own demand — counts the submissions in flight at each
//! submit: ≥ 2 there is the ONLY evidence the pipeline is real, and 1 everywhere is
//! the measurement that proved it never was.
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

use std::sync::Mutex;
use std::time::Duration;

/// WHAT A STEP IS — the table a record lands in. A prompt's chunks and the decode steps
/// around them ride one pipeline, and each kind's segments are its own story (a chunk's
/// bind is a whole chunk's shadow writes, a decode step's a few rows'), so each kind is
/// its own table and no mean mixes the two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepKind {
    /// One chunk of a prompt (or one wave of prompt tails).
    PromptChunk,
    /// One decode step (a batch of one generated token per live request).
    Decode,
}

impl StepKind {
    /// Every kind, in the order the table prints them.
    const ALL: [StepKind; 2] = [StepKind::PromptChunk, StepKind::Decode];

    /// The kind's slot in the per-kind accumulators.
    const fn slot(self) -> usize {
        match self {
            StepKind::PromptChunk => 0,
            StepKind::Decode => 1,
        }
    }

    /// How the table names a kind's steps.
    const fn plural(self) -> &'static str {
        match self {
            StepKind::PromptChunk => "pipelined prompt chunk(s)",
            StepKind::Decode => "pipelined decode step(s)",
        }
    }
}

/// One step's SUBMIT-half segments, clocked by the pipelined loop and flushed here in
/// one lock, with the pipeline depth at the submit. `stage` is `None` on a batch's
/// last step (nothing to stage), and `defer` on a step that picks nothing (nothing to
/// read back) — absent from that segment's mean, never a recorded zero.
///
/// `pub`, like the module's whole surface, and for a reason the tree states:
/// `record_submit`'s only caller is a target's worker crate, which a build may not
/// link — a `pub(crate)` surface would be dead code in every clippy leg of this
/// crate. `pub` items of a lib crate are reachable by
/// type in every feature combination, which is the law this instrument is built
/// under: no cfg, no allow, nothing dead.
pub struct SubmitSegs {
    /// Which table this step lands in.
    pub kind: StepKind,
    pub bind: Duration,
    pub submit: Duration,
    pub defer: Option<Duration>,
    pub stage: Option<Duration>,
    pub total: Duration,
    /// Submissions in flight as this step starts queuing on the card — the depth,
    /// THIS step included, of every kind: a decode step queued behind a prompt chunk
    /// counts it. The row that proves or refutes the pipeline: 1 everywhere is serial
    /// with a fence's name on it; ≥ 2 is steps genuinely in flight.
    pub depth: usize,
    /// ⭐ #32 (E) — CARD BUSY AS THE STEP QUEUES: the card had NOT yet retired the
    /// work ahead of this step when this step's first control block was about to
    /// join the stream (a non-blocking read of the earlier submissions' completion
    /// latches) — so this step queued before the card ran dry and runs back to back
    /// behind that work. The overlap fraction — the depth row's honest twin: depth
    /// counted 1.99 even in the build whose whole-stream fence idled the card at
    /// every submit (the executor's pre-schedule inflated the count without device
    /// overlap), while this row reads `false` exactly when the card had nothing left
    /// to execute. `false` on the first step ever submitted (nothing was in flight).
    pub busy: bool,
}

/// One step's COLLECT-half segments — the fence wait (at depth ≥ 2 it waits
/// everything enqueued behind the read too, which is the pipeline's own cadence, not
/// an artifact) and the pick decode. Recorded where the reads are fenced, which under
/// deferral is a later drain, not the submitting step's own call.
pub struct CollectSegs {
    /// Which table this collect lands in — its step's.
    pub kind: StepKind,
    pub fence: Duration,
    pub pick: Duration,
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
    };
}

/// One accumulator set per [`StepKind`], indexed by [`StepKind::slot`].
fn stats() -> std::sync::MutexGuard<'static, [Stats; 2]> {
    static S: Mutex<[Stats; 2]> = Mutex::new([Stats::ZERO, Stats::ZERO]);
    S.lock()
        .expect("step_segs: the probe's own lock, never held across a panic")
}

/// Flush one step's submit-half segments. Called once per pipelined step, after its
/// doorbell (and its pick's defer, when it picks). (`pub` for the reason `SubmitSegs`'s
/// own doc states — the cfg-gated caller must not make this dead code in a plain clippy
/// leg.)
pub fn record_submit(s: SubmitSegs) {
    // ⭐ ONE LINE PER STEP, at `debug` under its own `step_segs` target — the table's facts, step
    // by step, for a driver that prints no table (`serve`): `RUST_LOG=info,step_segs=debug`
    // streams every submission's kind, depth and busy bit and turns nothing else on.
    tracing::debug!(
        target: "step_segs",
        "[step-seg] {:?} depth={} busy={} bind={:.3} submit={:.3} total={:.3} ms",
        s.kind,
        s.depth,
        s.busy,
        s.bind.as_secs_f64() * 1e3,
        s.submit.as_secs_f64() * 1e3,
        s.total.as_secs_f64() * 1e3,
    );
    let mut all = stats();
    let t = &mut all[s.kind.slot()];
    t.steps += 1;
    t.bind.add(s.bind);
    t.submit.add(s.submit);
    if let Some(d) = s.defer {
        t.defer.add(d);
    }
    if let Some(d) = s.stage {
        t.stage.add(d);
    }
    t.total.add(s.total);
    t.depth.add(s.depth);
    t.busy.add(s.busy);
}

/// Flush one step's collect-half segments — the fence and the pick decode, recorded
/// where the reads are fenced (the sync composition, or a drain under deferral).
pub fn record_collect(s: CollectSegs) {
    let mut all = stats();
    let t = &mut all[s.kind.slot()];
    t.fence.add(s.fence);
    t.pick.add(s.pick);
}

/// The accumulated per-segment tables, one per [`StepKind`] that recorded a step, for the
/// CLI to print under `--bench`. `None` when no step ever recorded — the answer every
/// backend that records no steps must give, by type, not by empty string.
pub fn step_segment_table() -> Option<String> {
    let all = stats();
    let tables: Vec<String> = StepKind::ALL
        .iter()
        .map(|&k| (k, &all[k.slot()]))
        .filter(|(_, t)| t.steps > 0)
        .map(|(k, t)| kind_table(k, t))
        .collect();
    (!tables.is_empty()).then(|| tables.concat())
}

/// One kind's table: every segment row, then the depth and busy rows.
fn kind_table(kind: StepKind, t: &Stats) -> String {
    let mut out = format!(
        "[step-segs] {} {}: host time per segment (mean / max / total, ms)\n",
        t.steps,
        kind.plural()
    );
    let rows = [
        (
            "bind",
            "the shadow writes (with the shadow's quiet point)",
            &t.bind,
        ),
        ("submit", "submit_step (the doorbell + refill)", &t.submit),
        (
            "defer",
            "the pick's D2H defer (the enqueue, ring slot)",
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
    // refutation, in submissions in flight as each step queues. The distribution prints
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
        "[step-segs]   depth  (submissions in flight as each step queues): mean {:>5.2}, max {}, \
         distribution {}\n",
        t.depth.sum as f64 / t.depth.n.max(1) as f64,
        t.depth.max,
        dist,
    ));
    // ⭐ #32 (E) — THE BUSY ROW, the depth row's honest twin: the fraction of
    // steps that queued with the card still owing earlier work (its completion
    // latches un-fired). ~100% is real overlap — every step queued before the card
    // ran dry; ~0% is a serial pipeline wearing depth-2's clothes — the
    // pre-schedule can inflate the depth row, never this one.
    out.push_str(&format!(
        "[step-segs]   busy   (card owed earlier work as the step queued): {}/{} = {:.0}%\n",
        t.busy.busy,
        t.busy.n,
        100.0 * t.busy.busy as f64 / t.busy.n.max(1) as f64,
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ⭐ EVERY KIND IS ITS OWN TABLE: a prompt's chunks and the decode steps around them ride one
    /// pipeline, and each kind's rows count its own steps alone — its depth distribution, its busy
    /// fraction, and a defer row that counts only the steps that deferred a pick (a chunk that
    /// picks nothing is absent from it, never a recorded zero). The only test in this binary that
    /// records, so the process-wide accumulators hold exactly these steps.
    #[test]
    fn each_step_kind_is_its_own_table() {
        let ms = Duration::from_millis;
        for (kind, depth, busy, defer) in [
            (StepKind::PromptChunk, 1, false, None),
            (StepKind::PromptChunk, 2, true, None),
            (StepKind::PromptChunk, 2, true, Some(ms(1))),
            (StepKind::Decode, 2, true, Some(ms(1))),
        ] {
            record_submit(SubmitSegs {
                kind,
                bind: ms(2),
                submit: ms(3),
                defer,
                stage: None,
                total: ms(6),
                depth,
                busy,
            });
        }
        record_collect(CollectSegs {
            kind: StepKind::Decode,
            fence: ms(30),
            pick: ms(0),
        });
        let table = step_segment_table().expect("steps were recorded");
        let (chunks, decode) = table
            .split_once("[step-segs] 1 pipelined decode step(s)")
            .expect("the decode table follows the chunk table");
        assert!(
            chunks.starts_with("[step-segs] 3 pipelined prompt chunk(s)"),
            "{table}"
        );
        assert!(chunks.contains("distribution 1:1 2:2"), "{table}");
        assert!(chunks.contains("2/3 = 67%"), "{table}");
        let defer = chunks
            .lines()
            .find(|l| l.contains("the pick's D2H defer"))
            .expect("the chunk table states its defer row");
        assert!(defer.ends_with("over 1 step(s)"), "{defer}");
        assert!(decode.contains("distribution 2:1"), "{table}");
        assert!(decode.contains("1/1 = 100%"), "{table}");
        let fence = decode
            .lines()
            .find(|l| l.contains("THE fence"))
            .expect("the decode table states its fence row");
        assert!(fence.ends_with("over 1 step(s)"), "{fence}");
    }
}
