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
//! asks, `None` when no step ever recorded (every non-spyre backend, every
//! non-decode run). No env toggles, no numerics touched.
//!
//! The `Timers`/`Diag` idiom (a process-wide static behind a mutex) because
//! the step loop holds the session and `Shared` borrows a worker-field
//! accumulator cannot thread through. Cumulative by design: a `--bench` run's
//! N includes the warmup's single decode step and the first-play cold
//! outliers — they land in `max` where they belong; the steady-state read is
//! the mean.

use std::sync::Mutex;
use std::time::Duration;

/// One step's SUBMIT-half segments, clocked by the pipelined loop and flushed here in
/// one lock, with the pipeline depth at the submit. `stage` is `None` on a batch's
/// last step (nothing to stage) — absent from that segment's mean, never a recorded
/// zero.
///
/// `pub`, like the module's whole surface, and for a reason the tree states:
/// the worker's spyre mods are cfg-gated in lib.rs, so `record_submit`'s only caller
/// exists in spyre builds — a `pub(crate)` surface would be dead code in every plain
/// (non-spyre) clippy leg of this crate. `pub` items of a lib crate are reachable by
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

/// One step's COLLECT-half segments — the fence wait (at depth ≥ 2 it waits
/// everything enqueued behind the read too, which is the pipeline's own cadence, not
/// an artifact) and the pick decode. Recorded where the reads are fenced, which under
/// deferral is a later drain, not the submitting step's own call.
pub struct CollectSegs {
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

fn stats() -> std::sync::MutexGuard<'static, Stats> {
    static S: Mutex<Stats> = Mutex::new(Stats::ZERO);
    S.lock().expect("step_segs: the probe's own lock, never held across a panic")
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
/// where the reads are fenced (the sync composition, or a drain under deferral).
pub fn record_collect(s: CollectSegs) {
    let mut t = stats();
    t.fence.add(s.fence);
    t.pick.add(s.pick);
}

/// The accumulated per-segment table, for the CLI to print under `--bench`.
/// `None` when no step ever recorded — the answer every non-spyre backend
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
        (
            "bind",
            "keyed causal bind + act diff + seed build",
            &t.bind,
        ),
        ("submit", "submit_step (the doorbell + refill)", &t.submit),
        ("defer", "stick D2H defer (the enqueue, ring slot)", &t.defer),
        ("stage", "j+1 staging (pure host, overlaps step j)", &t.stage),
        ("total", "whole submit body (residual = total - the four)", &t.total),
        ("fence", "THE fence (collect half: wait + stick D2H)", &t.fence),
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
    Some(out)
}
