// SPDX-License-Identifier: Apache-2.0
//! ⭐ COMPILE EACH GROUP AS IT IS EMITTED, THEN RECLAIM ITS JSON — a bounded builder work queue.
//!
//! ## The shape of the problem
//!
//! `dxp_standalone` takes a DIRECTORY and requires one json per device op — merging a group's 64 ops
//! into one file is `DtException: Expected empty FoldManager when importing from json` — so its input
//! has to be materialised on a filesystem. For gemma-4-12b that is ~470,000 files (54 bundles x ~8,700
//! device ops) if they all exist at once.
//!
//! Two facts bound it without touching the file COUNT, which is dxp's to choose:
//!
//!   * dxp's OUTPUT is 6.6% of its input and 2 files per GROUP rather than per op (measured: a 64-op
//!     group is 1.1 MB of json in 64 files, compiling to 73 KB in `spyreCodeDir/{init_binary.bin,
//!     spyrecode.json}`).
//!   * **Nothing but dxp ever reads the input.** So a group's json only has to exist between being
//!     written and being compiled.
//!
//! This queue exploits both:
//!
//!   * **STAGED on local disk** under [`StageRoot`] (`$SCRATCHY_SUPERDSC_STAGE`, else the temp dir):
//!     20,341 files/s measured, against ~220/s on the build pod's NFS-mounted `$HOME`.
//!   * **BOUNDED**: staged bytes are capped at [`MAX_STAGED_BYTES`] because [`BakeQueue::reserve`]
//!     BLOCKS the emitter when the compilers fall behind, and [`read_compiled`] reclaims a group's
//!     staging dir the moment dxp is done with it. Measured on a granite-3.1-2b bake: 1047 groups
//!     compiled, peak staging 85 MB.
//!   * **MEMOIZED** on a hash of exactly what dxp will read, so a group whose input recurs is not
//!     recompiled — 470 of 1517 on that same bake (31%), since the rung ladder emits the same
//!     projections at many widths and each bundle also has a fused twin.
//!
//! Compiled output is returned IN MEMORY, keyed by [`GroupId`], for the emitter to bake into the
//! binary; nothing is written anywhere durable.
//!
//! It also makes the build FAIL FAST. The dxp wall that cost a full emission to discover
//! (`L3DlOpsScheduler` refusing gemma-4's 32-core matmuls) is hit on the FIRST group of the first
//! bundle, seconds in, because [`BakeQueue::submit`] returns the first error it sees.
//!
//! ## No dxp
//!
//! On a cardless host (a Mac, a plain `cargo check`) [`DxpTool::resolve`] yields `None`, [`global`] is
//! `None`, and the emitter stages nothing and writes no file at all — the bundle's metadata is still
//! baked, and `bundle::have_device_code()` reports the absence of the programs. That is a CAPABILITY
//! probe, not a behaviour flag: there is one code path and it is taken whenever the tool exists.

use std::collections::{HashMap, HashSet};
// Only `pre_exec` needs it now that `process_group(0)` is gone (see `ChildProc::spawn`), and that call
// is Linux-only — so on a Mac this import would be dead and `-D warnings` would fail on it.
#[cfg(target_os = "linux")]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use scratchy_spyre_bundle::correction;

/// ⭐ THE BUILD'S FAILURE AND EVERY LIVE COMPILER PROCESS, UNDER ONE LOCK.
///
/// ⛔ WHY THESE TWO FACTS SHARE A MUTEX: a worker that found a failure used to just record it and
/// return; the other `COMPILE_WIDTH - 1` workers' `dxp_standalone` children kept running with nothing
/// tracking them. When the build then exits on that failure (the normal `finish()` -> `Err` path), Unix
/// does not kill a process's children for it — they are orphaned onto whatever reparents them, which on
/// a pod is often a bare `sleep 1` that never calls `wait()`. Orphan now, zombie forever.
///
/// So the failure has to KILL them, which means the failing worker needs a registry of children it did
/// not spawn. And that registry cannot be a second lock beside the error: "has the build failed?" and
/// "spawn and record a child" would then be separately ordered, and a sweep could slip between a
/// worker's check and its insert, leaving exactly the untracked child this exists to prevent. One lock
/// makes [`Self::register`] and [`Self::fail`] mutually exclusive, so a child is either registered
/// before the sweep (and killed by it) or refused after it (and killed by its own guard) — never
/// neither.
#[derive(Debug, Default)]
struct Reaper {
    inner: Mutex<ReaperInner>,
}

#[derive(Debug, Default)]
struct ReaperInner {
    /// First failure, which is also the "the build is over" flag — ONE fact, not a message beside a
    /// bool that has to be kept in step with it.
    err: Option<String>,
    /// pid of every `dxp_standalone` currently running.
    ///
    /// ⛔ A PID, AND DELIBERATELY NOT A PGID. These children are spawned into THIS PROCESS'S process
    /// group — see [`ChildProc::spawn`] — so `kill(-x)` is not available to us and must not be: the
    /// group contains cargo, rustc and every sibling compile. One pid per child is both precise enough
    /// (dxp forks nothing) and the only safe target.
    live: HashSet<i32>,
}

impl Reaper {
    /// Every critical section here is a `HashSet` operation and `kill(2)`, neither of which can panic
    /// or leave a half-updated invariant, so a poisoned lock cannot mean broken state — recovering
    /// beats propagating a spurious failure through every call site.
    fn inner(&self) -> std::sync::MutexGuard<'_, ReaperInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record `pid` as live, or REFUSE because the build has already failed — in which case the
    /// caller's guard kills it immediately rather than adding a child nothing will reap.
    fn register(&self, pid: i32) -> bool {
        let mut g = self.inner();
        if g.err.is_some() {
            return false;
        }
        g.live.insert(pid);
        true
    }

    fn deregister(&self, pid: i32) {
        self.inner().live.remove(&pid);
    }

    /// Record the first failure and SIGKILL every live compiler. Later failures only report: the sweep
    /// has happened, and everything it could still kill is already dying.
    fn fail(&self, e: String) {
        let mut g = self.inner();
        if g.err.is_some() {
            return;
        }
        g.err = Some(e);
        for &pid in g.live.iter() {
            teardown::sigkill_and_leave_the_reap_to_the_owner(pid);
        }
    }

    fn first_error(&self) -> Option<String> {
        self.inner().err.clone()
    }

    fn has_failed(&self) -> bool {
        self.inner().err.is_some()
    }

    /// ⭐ KILL AND REAP EVERY LIVE COMPILER — the teardown for the exits NO destructor sees.
    ///
    /// ⛔ THIS IS THE PATH THAT LEAKED, AND [`Self::fail`] IS NOT IT. `fail` runs only when a dxp compile
    /// fails, and every child it kills is reaped by the worker that owns it. But the emit dies other ways:
    /// `audit_layout_addresses` refusing a layout, any other panic in the lowering, the `panic!` that
    /// turns the first bake error into the build's error. Each of those unwinds the MAIN thread of the
    /// build script, and no destructor on a WORKER thread ever runs — so `COMPILE_WIDTH` children are
    /// still running when the process exits. MEASURED on the pod, granite-3.1-2b, staging turned read-only
    /// mid-emit so the emitter's own write failed and no dxp compile did: **30 permanent `dxp_standalone`
    /// zombies, one per live compiler, and 0 with this sweep in place.** The build failed identically both
    /// times (`panicked at codegen.rs:9534`, zero `dxp refused` lines), so the sweep is the only variable.
    ///
    /// `PR_SET_PDEATHSIG` (see [`DxpTool::compile`]) covers the KILL for that case — the kernel signals
    /// each child when its spawning thread dies — but nothing covered the REAP, and a killed-but-unreaped
    /// child re-parented to a `sleep infinity` PID 1 is a zombie forever. This is the reap.
    ///
    /// ⛔ CLOSE THE DOOR BEFORE COUNTING. The workers keep running all through `atexit`, so a ONE-SHOT
    /// snapshot of `live` is a sample, not a set: a child registered after it is never swept. Marking the
    /// build over FIRST — under the same lock [`Self::register`] takes — is what turns the sample into a
    /// set. From that moment every later spawn is REFUSED, and a refused spawn is killed and reaped by its
    /// own guard's `Drop`, so `live` can only shrink and a bounded number of rounds drains it. The rounds
    /// catch a worker that was between `spawn` and `register` when the door closed: it appears in `live` a
    /// moment later and the next round takes it.
    ///
    /// ⚠️ HONEST SCOPE: the leak this whole sweep fixes was measured at 30 permanent `dxp_standalone`
    /// zombies per failed emit — one per live compiler — going to 0. The door-closing above is a race
    /// closed BY CONSTRUCTION, not by measurement: a snapshot-only sweep also measured 0, because the
    /// window between it and process exit is short. It is here because "short" is not "empty" and the
    /// window widens with anything that slows teardown.
    fn kill_and_reap_all(&self) {
        {
            let mut g = self.inner();
            if g.err.is_none() {
                g.err = Some("the build script is exiting".to_string());
            }
        }
        // Bounded: `live` only shrinks now, so this converges. The sleep gives a worker caught mid-`spawn`
        // time to reach its refusal, which is where its own teardown happens.
        for round in 0..32 {
            // Snapshot under the lock, then kill and reap OUTSIDE it: reaping sleeps, and holding the
            // registry across it would stall the `deregister` of every worker still trying to exit.
            let live: Vec<i32> = self.inner().live.iter().copied().collect();
            if live.is_empty() {
                break;
            }
            teardown::kill_and_reap_each(&live);
            if round > 0 {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }
}

/// The process-wide [`Reaper`], published for [`reap_on_exit`].
///
/// A `static` beside the queue rather than a reach into [`global`], because an `atexit` handler must
/// never be the thing that CREATES the queue: `global()` would spawn `COMPILE_WIDTH` worker threads
/// during process teardown on a build that never baked anything at all.
static EXIT_REAPER: std::sync::OnceLock<Arc<Reaper>> = std::sync::OnceLock::new();

/// `atexit` hook: kill and reap every compiler still live at process exit.
///
/// Covers a panic unwind, a `main` that returns, and `process::exit` — every way the build script can end
/// short of being SIGKILLed itself, which nothing in userspace can cover. Registered rather than called
/// from a destructor because the queue lives in a `static` ([`global`]), and statics are never dropped.
extern "C" fn reap_on_exit() {
    if let Some(reaper) = EXIT_REAPER.get() {
        reaper.kill_and_reap_all();
        // Last resort for a child forked but never registered: the registry cannot name it, so nothing
        // else can reap it. ⛔ ONLY HERE, never inside `kill_and_reap_all`: this reaps by `-1`, so any
        // caller that is not the dying process would steal an exit status from unrelated code — in the
        // unit tests, from a sibling test's own child.
        teardown::drain_dead_children();
    }
}

/// Arm BOTH teardowns — ONCE per process, before the first child can exist.
///
/// ⛔ TWO HOOKS, BECAUSE THEY COVER DISJOINT EXITS AND NEITHER IS A SUPERSET OF THE OTHER:
///
///   * `atexit` covers every exit that runs C++/Rust teardown — a panic unwind, `main` returning,
///     `process::exit`. It does NOT run when a signal terminates the process.
///   * [`signals`] covers the terminating signals, which is `^C` on a `cargo build`, `^\`, a plain
///     `kill`, and the SIGHUP an `oc rsh` sends when its connection drops. Those run NO destructor and
///     NO `atexit` hook at all, which is why the first three attempts at this bug — all of them aimed
///     at `Drop`, [`Reaper::fail`] and `atexit` — never touched the failure being reported.
///
/// Only a SIGKILL of this process is left, and nothing in userspace can cover that: `PR_SET_PDEATHSIG`
/// still kills the children, and their corpses are then at the mercy of whatever PID 1 is.
fn arm_teardown(reaper: &Arc<Reaper>) {
    if EXIT_REAPER.set(Arc::clone(reaper)).is_err() {
        // Already armed. One queue serves the whole process ([`global`]), so the hook registered by the
        // first arming already points at the reaper that owns every live child.
        return;
    }
    // SAFETY: `atexit` stores a plain `extern "C"` function pointer, which has static lifetime here. The
    // handler only reads a `OnceLock` and calls `kill`/`waitpid`, so it is safe to run during teardown
    // while worker threads are still live.
    unsafe {
        libc::atexit(reap_on_exit);
    }
    signals::arm();
}

/// ⛔⛔ KILL AND REAP ARE ONE OPERATION — AND THIS MODULE IS WHY THAT IS NOT MERELY A COMMENT.
///
/// 🛑 SIGKILL ENDS A PROCESS; ONLY `wait` CLEARS ITS TASK-TABLE ENTRY. `libc::kill` and `libc::waitpid`
/// are unrelated FFI calls and nothing in the type system ties them together, which is precisely how the
/// previous fix shipped a `kill` with no `wait`: the guard, the `Drop`, the whole RAII shape was right,
/// and it still leaked, because `std::process::Child` is the one std type that deliberately has NO `Drop`
/// — dropping it neither waits nor kills. So the language will not catch this for you. A MODULE BOUNDARY
/// will: [`kill`] and [`reap`] are private here, and the only teardown spellable from outside is one that
/// reaps. The single caller that legitimately does not reap has to say so in the function's name.
///
/// What a leaked zombie costs: it is re-parented to PID 1 on exit, and PID 1 in a dev pod is
/// `sleep infinity`, which never calls `wait()`. So it is PERMANENT and unclearable without restarting
/// the pod — 227 counted across one session.
mod teardown {
    /// End one child we own: SIGKILL, then reap. The only way out of this module for a child we hold.
    pub(super) fn kill_and_reap(pid: i32) {
        kill(pid);
        reap(pid);
    }

    /// End many. Kills EVERY child before reaping any, so the deaths overlap and the reaps almost all
    /// return on their first poll — at `COMPILE_WIDTH` children that is the difference between
    /// milliseconds and a visible stall on the way out.
    pub(super) fn kill_and_reap_each(pids: &[i32]) {
        for &pid in pids {
            kill(pid);
        }
        for &pid in pids {
            reap(pid);
        }
    }

    /// ⚠️ SIGKILL WITH NO REAP — the one legitimate caller, and it is NOT teardown.
    ///
    /// [`super::Reaper::fail`] hurries along a child whose `Child` another worker still holds, and that
    /// worker's `wait_with_output` is what reaps it. Reaping here would race the owner and turn a
    /// diagnostic `dxp refused …` into `wait: No child processes`. The reap obligation travels with the
    /// [`super::ChildProc`], never with the killer — spelled out in the name so this cannot be mistaken
    /// for the functions above.
    pub(super) fn sigkill_and_leave_the_reap_to_the_owner(pid: i32) {
        kill(pid);
    }

    /// Collect any child of ours that is ALREADY dead, without blocking and without knowing its pid.
    ///
    /// The backstop for a child that was forked but not yet registered when the door closed: the
    /// registry cannot name it, so nothing else can reap it. Non-blocking, so it can never hang the
    /// exit; it only ever clears corpses.
    pub(super) fn drain_dead_children() {
        for _ in 0..1024 {
            let mut status: libc::c_int = 0;
            // SAFETY: -1 waits on any child of this process; `status` is a live local. Only reached at
            // process exit, where stealing a status from code that is itself about to die is harmless.
            let got = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
            if got <= 0 {
                return;
            }
        }
    }

    /// ⚠️ Only sound while the child is known to be UNREAPED: a pid is free for reuse the instant it is
    /// reaped, so a kill sent after that could land on an unrelated process.
    ///
    /// ⛔⛔ POSITIVE, NEVER `-pid`. These children live in THIS PROCESS'S process group — see
    /// [`super::ChildProc::spawn`], which is what makes Ctrl+C reach them at all — so a pid here is a
    /// pid and nothing else. `kill(-pid)` would signal whatever process GROUP happens to bear that
    /// number, which is either nothing or something entirely unrelated to this build.
    fn kill(pid: i32) {
        // SAFETY: a positive pid targets exactly that one child. ESRCH (it is already gone) is the
        // expected outcome of a race with normal exit, not an error to surface.
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }

    /// BOUNDED, and polling rather than blocking, on purpose. SIGKILL is asynchronous — the leader is
    /// not reaped the instant `kill` returns — but a blocking `waitpid` on a leader wedged in
    /// uninterruptible I/O would hang the build, and a hung build is worse than the zombie this exists
    /// to prevent. The first or second poll succeeds in practice; the budget only bounds the
    /// pathological case.
    ///
    /// One pid, one status: a single successful `waitpid` is terminal here, unlike the group form this
    /// replaces. dxp forks nothing — measured on the pod, 0 children on every live `dxp_standalone`
    /// sampled — so the child is a leaf and there is nothing under it to leave behind.
    fn reap(pid: i32) {
        const BUDGET: std::time::Duration = std::time::Duration::from_secs(1);
        const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_millis(50);
        let mut waited = std::time::Duration::ZERO;
        let mut backoff = std::time::Duration::from_micros(200);
        loop {
            let mut status: libc::c_int = 0;
            // SAFETY: a positive pid waits on exactly that one child, and `status` is a live local.
            // Naming the pid is what keeps this from stealing a sibling compile's exit status — which
            // `waitpid(-1, …)` would, now that every child shares one process group.
            let reaped = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
            if reaped > 0 {
                // Collected, and there is only ever one status per pid.
                return;
            }
            if reaped < 0 {
                // ECHILD — not ours to reap any more, which IS the success condition: the owning
                // worker's `wait_with_output` got there first. EINTR is the only retryable outcome.
                if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return;
            }
            // 0: a member is alive but not yet reapable — the SIGKILL is still in flight.
            if waited >= BUDGET {
                return;
            }
            std::thread::sleep(backoff);
            waited += backoff;
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }
}

/// ⭐⭐ THE TEARDOWN CTRL+C TAKES — the exit NO destructor and NO `atexit` hook can see.
///
/// ⛔⛔ THIS IS THE HOLE THE PREVIOUS THREE ATTEMPTS LEFT OPEN, and it is the one that actually gets hit:
/// `^C` on a `cargo build`. A default-disposition SIGINT, SIGTERM, SIGHUP or SIGQUIT terminates a process
/// from inside the kernel — there is no unwind, so no `Drop` runs, and no call to `exit(3)`, so no
/// `atexit` hook runs either. EVERY userspace teardown in this file is unreachable on that path, which is
/// why three fixes aimed at `Drop`, at [`Reaper::fail`] and at `atexit` all left the symptom exactly
/// where it was. The children were then killed by `PR_SET_PDEATHSIG` — which works — and that is still
/// not enough, because by then their parent is gone, so each corpse re-parents to PID 1, and PID 1 in a
/// dev pod is `sleep infinity`, which never calls `wait()`. The corpse is a PERMANENT zombie.
///
/// [`ChildProc::spawn`] fixes the other half by keeping the children in this process's process group, so
/// the tty's signal reaches them directly. This module fixes THIS half: something has to still be alive
/// to call `wait()` on the corpses, and that means handling the signal instead of dying on it. We outlive
/// our own children by the few hundred microseconds a reap takes, then die exactly as we would have.
///
/// ## Why a lock-free table instead of the [`Reaper`]
///
/// A signal handler may call only async-signal-safe functions. `kill`, `waitpid`, `nanosleep`,
/// `sigaction`, `pthread_sigmask`, `raise` and `_exit` are all on that list, so the teardown itself is
/// perfectly legal in a handler — but a `Mutex` is NOT, and `Reaper`'s registry sits behind one. Locking
/// it here would deadlock the build outright whenever the signal happened to land on a thread already
/// holding it, which is worse than the zombie. So the handler reads a fixed array of `AtomicI32`: no
/// allocation, no lock, no `HashSet`, every operation a single lock-free atomic. The `Reaper` stays the
/// source of truth for every other path; this is its shadow, written beside it by [`ChildProc`].
mod signals {
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

    /// Slots for pids the handler may reap. At most [`super::COMPILE_WIDTH`] children are live at once
    /// (one per worker); double that so a slot is always free even while a refused spawn is still
    /// tearing its own child down.
    const SLOTS: usize = super::COMPILE_WIDTH * 2;

    /// Every live child's pid, or 0 for a free slot. 0 is a safe sentinel: no child is ever pid 0.
    static LIVE: [AtomicI32; SLOTS] = [const { AtomicI32::new(0) }; SLOTS];

    /// Set BEFORE the handler sweeps, so a worker that forks concurrently tears its own child down
    /// instead of escaping the sweep. See the ordering argument in [`super::ChildProc::spawn`].
    static TEARING_DOWN: AtomicBool = AtomicBool::new(false);

    /// ⛔⛔ ONE THREAD SWEEPS. `sa_mask` BLOCKS SIGNALS ONLY IN THE HANDLING THREAD, so a second signal —
    /// a SIGHUP as the terminal goes away, a SIGTERM from cargo, a second `^C` — is delivered to a
    /// DIFFERENT thread, which enters this same handler concurrently.
    ///
    /// MEASURED, and it is why the first version of this fix only got two thirds of the way: the second
    /// entrant found the pid table already emptied by the first, so it swept nothing, fell straight
    /// through to [`restore_and_reraise`], and KILLED THE PROCESS OUT FROM UNDER the first thread's reap
    /// loop. 21 children claimed, 7 reaped, the loop never reached its end — and exactly 14 zombies, the
    /// 14 it had not got to yet. The instrument that showed it was the handler firing TWICE.
    static SWEEPING: AtomicBool = AtomicBool::new(false);

    /// The ways a terminal ends a build: `^C`, `^\`, a `kill`, and the hangup an `oc rsh`/`kubectl exec`
    /// dropping its connection delivers. All four terminate by default, so all four skip every
    /// destructor.
    const FATAL: [libc::c_int; 4] = [libc::SIGINT, libc::SIGTERM, libc::SIGHUP, libc::SIGQUIT];

    /// What was installed before us, one per [`FATAL`] entry, so the process still dies the way it would
    /// have. Read in the handler: `OnceLock::get` on an initialised cell is an atomic load and a deref —
    /// no lock, no allocation — which is the same property `EXIT_REAPER` relies on inside `atexit`.
    static PREVIOUS: std::sync::OnceLock<[libc::sigaction; FATAL.len()]> =
        std::sync::OnceLock::new();

    /// Publish `pid` where the handler can see it.
    ///
    /// A full table is a silent no-op rather than an error: `Drop` and the `atexit` sweep still cover
    /// that child, and only the signal path would miss it. Failing a compile over a bookkeeping slot
    /// would trade a rare leaked zombie for a broken build.
    pub(super) fn track(pid: i32) {
        for slot in LIVE.iter() {
            if slot
                .compare_exchange(0, pid, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Release `pid`'s slot. Already-cleared is the normal case when the handler swept it first.
    pub(super) fn untrack(pid: i32) {
        for slot in LIVE.iter() {
            if slot
                .compare_exchange(pid, 0, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return;
            }
        }
    }

    /// Has a signal teardown begun? Checked by [`super::ChildProc::spawn`] AFTER its [`track`], which is
    /// what makes the two orderings exhaustive.
    pub(super) fn tearing_down() -> bool {
        TEARING_DOWN.load(Ordering::SeqCst)
    }

    /// Install the handler for every signal in [`FATAL`] — once per process, before the first child.
    ///
    /// ⛔ NEVER OVERRIDE `SIG_IGN`. A shell sets SIGINT and SIGQUIT to `SIG_IGN` for a background job,
    /// and `SIG_IGN` is inherited across BOTH fork and exec — so a build started with `&`, or under
    /// `nohup`, is deliberately ignoring `^C`, and installing a handler over that would make it die on a
    /// signal it was meant to survive. Not hypothetical: that inheritance is exactly what made the first
    /// attempt to MEASURE this bug print four identical rows, because the harness's own children had
    /// inherited `SIG_IGN` and no arrangement of them could ever have died.
    pub(super) fn arm() {
        // SAFETY: `sigaction` is POD — an integer handler slot, a signal set, flags, and on Linux a
        // nullable restorer pointer whose zero value is its `None`. Zeroing is the documented way to
        // build one before filling the fields that matter.
        let mut previous = [unsafe { std::mem::zeroed::<libc::sigaction>() }; FATAL.len()];
        for (i, &sig) in FATAL.iter().enumerate() {
            // SAFETY: a null `act` makes this a pure query of the current disposition into `old`.
            let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
            if unsafe { libc::sigaction(sig, std::ptr::null(), &mut old) } != 0 {
                continue;
            }
            if old.sa_sigaction == libc::SIG_IGN {
                // Leave it ignored, and record that so a re-raise cannot resurrect it either.
                previous[i] = old;
                continue;
            }
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction =
                on_fatal_signal as extern "C" fn(libc::c_int) as libc::sighandler_t;
            action.sa_flags = 0;
            // SAFETY: fills `sa_mask`, blocking every signal for the handler's duration — a second `^C`
            // must not re-enter a sweep that is halfway through the table.
            unsafe { libc::sigfillset(&mut action.sa_mask) };
            // SAFETY: `action` outlives the call and the handler is an `extern "C"` fn with static
            // lifetime.
            if unsafe { libc::sigaction(sig, &action, &mut old) } == 0 {
                previous[i] = old;
            }
        }
        let _ = PREVIOUS.set(previous);
    }

    /// ⛔ ASYNC-SIGNAL-SAFE ONLY BELOW THIS LINE. No allocation, no `Mutex`, no `format!`, no `println!`,
    /// no `std::thread::sleep` — every call here is on POSIX's async-signal-safe list.
    ///
    /// ⭐ THE STORE COMES BEFORE THE SWEEP, and [`super::ChildProc::spawn`]'s matching load comes AFTER
    /// its [`track`]. That pairing is what makes the two exhaustive: a child forked concurrently with
    /// this handler is either already in the table when [`sweep`] reads it, or its spawner sees this flag
    /// and tears it down itself. It cannot be neither.
    extern "C" fn on_fatal_signal(sig: libc::c_int) {
        TEARING_DOWN.store(true, Ordering::SeqCst);
        // ⛔ SECOND ENTRANT RETURNS, AND MUST NOT RE-RAISE. Another thread is already sweeping and will
        // end this process when it is done; re-raising here ends it EARLY, mid-sweep, which is precisely
        // the 14-zombie failure documented on [`SWEEPING`]. Returning resumes a thread that is about to
        // be terminated anyway, and [`TEARING_DOWN`] is already set, so it cannot start new work.
        if SWEEPING.swap(true, Ordering::SeqCst) {
            return;
        }
        sweep();
        restore_and_reraise(sig);
    }

    /// Kill and reap every tracked child. Split out of [`on_fatal_signal`] so it is REACHABLE FROM A TEST
    /// — the handler itself ends in `_exit`, so a test could otherwise only observe it by dying.
    ///
    /// ⛔ Does NOT touch [`TEARING_DOWN`]: that flag is process-wide, and setting it here would leave
    /// every later `spawn` in the same test binary refusing.
    pub(super) fn sweep() {
        // Claim the whole set first so a concurrent `Drop` cannot also reap these, then kill, then reap.
        // Killing all before reaping any is what keeps the deaths overlapping.
        let mut pids = [0i32; SLOTS];
        for (slot, out) in LIVE.iter().zip(pids.iter_mut()) {
            *out = slot.swap(0, Ordering::SeqCst);
        }
        for &pid in pids.iter() {
            if pid > 0 {
                // SAFETY: a positive pid, and ours until reaped. ESRCH just means the tty's own signal
                // already finished it, which is the common case now that it shares our process group.
                unsafe { libc::kill(pid, libc::SIGKILL) };
            }
        }
        // ONE budget shared across every child, not one per child: at `COMPILE_WIDTH` children a
        // per-child budget turns `^C` into a multi-second stall. ~500 x 1 ms, and in practice the first
        // round takes them all.
        for _ in 0..500 {
            let mut remaining = false;
            for pid in pids.iter_mut() {
                if *pid <= 0 {
                    continue;
                }
                let mut status: libc::c_int = 0;
                // SAFETY: a positive pid waits on exactly that child; `status` is a live local.
                let got = unsafe { libc::waitpid(*pid, &mut status, libc::WNOHANG) };
                // >0 reaped it; <0 is ECHILD, i.e. the owning worker's own wait got there first. Both
                // mean "no corpse left", which is the whole job.
                if got != 0 {
                    *pid = 0;
                } else {
                    remaining = true;
                }
            }
            if !remaining {
                break;
            }
            let ts = libc::timespec {
                tv_sec: 0,
                tv_nsec: 1_000_000,
            };
            // SAFETY: a live local, and `nanosleep` is async-signal-safe.
            unsafe { libc::nanosleep(&ts, std::ptr::null_mut()) };
        }
    }

    /// Put back the disposition we replaced and re-raise, so this process dies EXACTLY as it would have
    /// without us — same signal, and `WIFSIGNALED` still true, which is what cargo and the shell read to
    /// decide whether to report a failure or stay quiet about a deliberate interrupt.
    fn restore_and_reraise(sig: libc::c_int) -> ! {
        if let (Some(prev), Some(i)) = (PREVIOUS.get(), FATAL.iter().position(|&s| s == sig)) {
            // SAFETY: `prev[i]` is the exact disposition read back at install time for this signal.
            unsafe { libc::sigaction(sig, &prev[i], std::ptr::null_mut()) };
        }
        // SAFETY: all async-signal-safe. The kernel masked `sig` on entry to the handler, so it has to
        // be unblocked or the re-raise would sit pending until we returned — and we would then exit by
        // code rather than by signal.
        unsafe {
            let mut set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut set);
            libc::sigaddset(&mut set, sig);
            libc::pthread_sigmask(libc::SIG_UNBLOCK, &set, std::ptr::null_mut());
            libc::raise(sig);
            // Reached only if the restored disposition was `SIG_IGN` or a handler that returned. 128+n
            // is the shell's own encoding for it.
            libc::_exit(128 + sig);
        }
    }
}

/// ⭐ ONE `dxp_standalone` CHILD, OWNED — in THIS PROCESS'S process group, registered in the [`Reaper`]
/// for as long as `Self` is alive.
///
/// `Drop`, not the order of statements in [`DxpTool::compile`], is what guarantees the registry entry
/// is cleared and an unreaped child is killed: true today (the only path is spawn then wait), and still
/// true of whatever `compile` grows into later — an early `?`, a timeout, a panic on this thread. A bare
/// insert-then-remove around the wait call gets that right only until someone edits the function
/// between the two lines.
struct ChildProc<'a> {
    /// Where the leader is in its lifecycle — which is precisely what `Drop` has to decide from.
    leader: Leader,
    pid: i32,
    reaper: &'a Reaper,
}

/// The leader's place in its lifecycle: spawned, being waited on, or reaped.
///
/// ⛔ THE MIDDLE STATE IS THE POINT, and an `Option<Child>` could not express it. With two states —
/// "holding a `Child`" and "not" — a panic inside [`ChildProc::wait_with_output`], which has already
/// MOVED the `Child` out by the time anything in it can panic, left `Drop` reading the "already reaped"
/// case: it neither killed nor reaped, and the child ran on to be orphaned. `Waiting` says "the leader
/// is still ours and still unreaped, but the `Child` is gone" — killable and reapable, and reachable
/// only by unwinding.
enum Leader {
    /// Spawned, not yet waited on. `Drop` must kill the child and reap it.
    Running(std::process::Child),
    /// [`ChildProc::wait_with_output`] has taken the `Child` and is waiting on it. Seen by `Drop` only
    /// if that wait unwound, in which case the leader is unreaped and must be killed and reaped by
    /// pid — there is no `Child` left to wait on.
    Waiting,
    /// Reaped. `Drop` must NOT kill: the pid is free for reuse from this moment on, so a kill could
    /// land on an unrelated process.
    Reaped,
}

impl<'a> ChildProc<'a> {
    /// Spawn `cmd` INTO THIS PROCESS'S OWN PROCESS GROUP — stock `Command` behaviour, deliberately
    /// unmodified — and register it.
    ///
    /// ⛔⛔⛔ DO NOT ADD `process_group(0)` BACK. THAT ONE CALL IS WHAT BROKE CTRL+C, and it is the third
    /// failed attempt at this bug, not a missing fourth safeguard.
    ///
    /// A tty delivers SIGINT (Ctrl+C), SIGQUIT (Ctrl+\) and SIGHUP (the `oc rsh` connection dropping) to
    /// its FOREGROUND PROCESS GROUP. A child inherits its parent's group, so by default every
    /// `dxp_standalone` is in that group and dies on Ctrl+C for free, with no code of ours involved —
    /// which is exactly the behaviour a build should have. `process_group(0)` put each child in a fresh
    /// group of its own, and a group of its own is BY DEFINITION not the terminal's foreground group, so
    /// the keystroke stopped reaching them. They then ran on until `PR_SET_PDEATHSIG` killed them at
    /// rustc's death — by which time their parent was gone, so each corpse re-parented to PID 1, which in
    /// a dev pod is `sleep infinity` and never calls `wait()`. MEASURED: 17 `dxp_standalone` in state `Z`,
    /// every one `PPID 1` with `PGID == its own PID` — that last equality being the fingerprint of the
    /// `process_group(0)` this removes.
    ///
    /// It was added so [`Reaper::fail`] could `kill(-pgid)` a child's whole subtree. dxp has no subtree
    /// (measured: 0 child processes on every live sample), so [`teardown::kill`] names the pid instead
    /// and loses nothing. Sharing our group is in fact STRICTLY better on that point: anything dxp ever
    /// did fork would inherit the group too, and so would take the tty's signal along with everyone else.
    ///
    /// `Err` once the build has already failed: the child is spawned but immediately torn down by the
    /// guard's own `Drop`, so losing the registration race cannot leave it running.
    fn spawn(cmd: &mut std::process::Command, reaper: &'a Reaper) -> Result<Self, String> {
        let child = cmd.spawn().map_err(|e| format!("spawn: {e}"))?;
        let pid = child.id() as i32;
        // ⛔ PUBLISH TO THE SIGNAL TABLE BEFORE ASKING WHETHER TEARDOWN HAS BEGUN, and never the other
        // way round. The signal handler cannot take the `Reaper`'s mutex (see [`signals`]), so this
        // lock-free table is the only registry it can read, and this ORDER is the whole race argument:
        // the handler marks teardown and then sweeps, so if our store lands before its sweep it kills
        // this child, and if it does not, then its mark preceded our load and we tear the child down
        // ourselves below. One of the two always holds; neither can be skipped.
        signals::track(pid);
        // Construct the guard BEFORE registering, so the refusal path below tears the child down
        // through the same `Drop` as every other exit.
        let guard = ChildProc {
            leader: Leader::Running(child),
            pid,
            reaper,
        };
        if signals::tearing_down() {
            return Err("the build is being torn down by a signal".to_string());
        }
        if !reaper.register(pid) {
            return Err("the bake already failed in another group".to_string());
        }
        Ok(guard)
    }

    /// Wait for the leader and collect its output, consuming the guard so `Drop` runs immediately
    /// after — deregistering either way.
    ///
    /// Advances to [`Leader::Reaped`] when the leader is gone — on success, and equally on `ECHILD`,
    /// which says something else reaped it first. ⚠️ THAT SECOND CASE IS A SAFETY CONDITION, NOT
    /// TIDINESS: a reaped pid is free for reuse, so treating `ECHILD` as "still ours" would send `Drop`
    /// on to `kill(pid)` and it could land the SIGKILL on an unrelated process. Only
    /// [`Reaper::kill_and_reap_all`] and [`signals`] can get there first, and only during teardown.
    ///
    /// Any OTHER error leaves the leader's fate unknown, so the state stays [`Leader::Waiting`] and
    /// `Drop` kills and reaps — the child is still ours in that case.
    fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        let child = match std::mem::replace(&mut self.leader, Leader::Waiting) {
            Leader::Running(child) => child,
            // Unreachable: `spawn` is the only constructor, it always sets `Running`, and this method
            // consumes `self` so it cannot run twice. Reported rather than panicked so an impossible
            // state costs a build error instead of a crash — and the state is put BACK, so `Drop` still
            // makes the right kill/reap decision.
            already => {
                self.leader = already;
                return Err(std::io::Error::other(
                    "ChildProc: the leader was already taken",
                ));
            }
        };
        let out = child.wait_with_output();
        let gone = match &out {
            Ok(_) => true,
            Err(e) => e.raw_os_error() == Some(libc::ECHILD),
        };
        if gone {
            self.leader = Leader::Reaped;
        }
        out
    }
}

impl Drop for ChildProc<'_> {
    fn drop(&mut self) {
        self.reaper.deregister(self.pid);
        signals::untrack(self.pid);
        // KILL AND REAP AS A PAIR, and only while the leader is still unreaped. Once `wait_with_output`
        // has reaped it the pid is already free for recycling, so a kill here could hit an unrelated
        // process; until then the child is guaranteed to be ours. The reap is what keeps the SIGKILL from
        // leaving a zombie nothing will ever collect — see [`teardown`].
        if !matches!(self.leader, Leader::Reaped) {
            teardown::kill_and_reap(self.pid);
        }
    }
}

/// ⭐ THE DISK BOUND: staged json bytes that may exist at once, across every bundle.
///
/// Distinct from [`COMPILE_WIDTH`] on purpose. These were ONE constant, which made them impossible to
/// set: raising it to get dxp parallelism raised peak scratch by the same factor, and lowering it to
/// bound scratch throttled the compile. They limit different resources — this one disk, that one CPU
/// — so they are two numbers.
///
/// Counted in BYTES rather than groups because a group is 64 ops in one place and 512 in another, so
/// a group count bounds nothing in particular. The emitter knows a group's exact size before it
/// writes it (it already holds the rendered json), so the reservation is exact, not an estimate.
pub const MAX_STAGED_BYTES: usize = 512 * 1024 * 1024;

/// ⭐ THE CPU BOUND: concurrent `dxp_standalone` processes.
///
/// dxp is single-threaded per group, so this is the compile width. Capped rather than unbounded so a
/// 192-core host does not fork 900 compilers at once; the disk bound above is what stops the emitter
/// running ahead of them.
pub const COMPILE_WIDTH: usize = 32;

/// Kept as the queue's const-generic parameter — the channel depth, which only has to exceed the
/// compile width so a finished worker never waits for the producer.
pub const IN_FLIGHT: usize = COMPILE_WIDTH * 2;

/// ⭐ A BYTE BUDGET FOR STAGED JSON, with the blocking on `reserve`.
///
/// `reserve` waits until the request fits under [`MAX_STAGED_BYTES`]; `release` wakes a waiter. The
/// emitter reserves a group's exact size BEFORE writing it and the compiler releases it when the
/// staging dir is deleted, so the amount of json on disk at any instant is bounded by construction
/// rather than by how fast dxp happens to be.
#[derive(Debug)]
pub struct StageBudget {
    used: Mutex<usize>,
    freed: Condvar,
    peak: AtomicUsize,
}

impl StageBudget {
    fn new() -> StageBudget {
        StageBudget {
            used: Mutex::new(0),
            freed: Condvar::new(),
            peak: AtomicUsize::new(0),
        }
    }

    /// Block until `n` bytes fit, then claim them.
    ///
    /// A single group larger than the whole budget would never fit, so it is allowed through alone
    /// (once nothing else is staged) rather than deadlocking the emit — the budget is a throttle, not
    /// a correctness property.
    fn reserve(&self, n: usize) {
        let Ok(mut used) = self.used.lock() else {
            return;
        };
        while *used + n > MAX_STAGED_BYTES && *used > 0 {
            used = match self.freed.wait(used) {
                Ok(g) => g,
                Err(_) => return,
            };
        }
        *used += n;
        self.peak.fetch_max(*used, Ordering::Relaxed);
    }

    fn release(&self, n: usize) {
        if let Ok(mut used) = self.used.lock() {
            *used = used.saturating_sub(n);
            self.freed.notify_one();
        }
    }

    /// High-water mark, so a build log can state what it actually used.
    pub fn peak_bytes(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }
}

/// ⭐ THE OUTSTANDING SET, AS A COUNTDOWN LATCH — what makes [`BakeQueue::finish`] a BARRIER and not a
/// shutdown.
///
/// `enter` on submit, `leave` on completion (compiled, memo-hit or failed), `wait_empty` blocks until the
/// count is zero. One number rather than a submitted/settled pair: two cumulative counters have to be
/// compared, and "are they equal yet" is a question a failed send can make un-answerable.
#[derive(Debug, Default)]
pub struct Latch {
    outstanding: Mutex<usize>,
    empty: Condvar,
}

impl Latch {
    fn enter(&self) {
        if let Ok(mut n) = self.outstanding.lock() {
            *n += 1;
        }
    }

    /// Undo an `enter` whose work never reached a worker, so a failed send cannot strand the latch
    /// above zero forever.
    fn cancel(&self) {
        self.leave();
    }

    fn leave(&self) {
        if let Ok(mut n) = self.outstanding.lock() {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.empty.notify_all();
            }
        }
    }

    /// Block until nothing is outstanding. `abort` lets a failure end the wait — the workers drain the
    /// rest without compiling, so the count still falls, but there is no reason to wait for it.
    fn wait_empty(&self, abort: &dyn Fn() -> bool) {
        let Ok(mut n) = self.outstanding.lock() else {
            return;
        };
        while *n > 0 && !abort() {
            n = match self.empty.wait(n) {
                Ok(g) => g,
                Err(_) => return,
            };
        }
    }
}

/// The queue as the emitter names it — the const is bound here so the emitter does not become generic
/// over it.
pub type Bake = BakeQueue<IN_FLIGHT>;

/// Where the per-op json is STAGED for dxp.
///
/// `$SCRATCHY_SUPERDSC_STAGE` overrides it; the default is the system temp dir. This is a PATH, not a
/// behaviour switch: the pipeline is identical wherever it points, and the only reason to move it is
/// that the default temp dir is not local (or not big enough for [`MAX_STAGED_BYTES`]).
#[derive(Clone, Debug)]
pub struct StageRoot(PathBuf);

impl StageRoot {
    pub fn resolve() -> StageRoot {
        let base = match std::env::var_os("SCRATCHY_SUPERDSC_STAGE") {
            Some(p) => PathBuf::from(p),
            None => std::env::temp_dir(),
        };
        // Per-process, so two concurrent cargo units running this expansion never share a staging dir.
        StageRoot(base.join(format!("superdsc-stage-{}", std::process::id())))
    }

    /// The staging dir for one group of one bundle — `<root>/<fp>/group_<i>`, so a dxp error message
    /// names the bundle and group it refused.
    pub fn group_dir(&self, fp: &str, gi: usize) -> PathBuf {
        self.0.join(fp).join(format!("group_{gi}"))
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

/// WHICH bundle and group a compiled program belongs to — the key the emitter reads its results back
/// under.
///
/// ⛔ AN IDENTITY, NOT A DIRECTORY, because the identity is the return channel: a result is either
/// filed under the id that was submitted or the build fails naming it. A directory as the channel
/// cannot report a group nobody looked for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GroupId {
    /// The emitting bundle's content fingerprint.
    pub fp: String,
    /// Launch order within that bundle.
    pub group: u32,
}

impl std::fmt::Display for GroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/group_{}", self.fp, self.group)
    }
}

/// One dxp-compiled launch group, IN MEMORY — what the emitter hands the macro to bake.
///
/// ⛔ dxp's `spyrecode.json` IS PARSED HERE AND NOWHERE ELSE. `job_bin_ptr` and `correction` are its
/// whole contribution, and they are extracted at the one point where the compiler that produced them is
/// in scope.
#[derive(Clone, Debug, Default)]
pub struct CompiledGroup {
    /// dxp's device image (`init_binary.bin`). Empty for a group dxp compiled to a job plan alone.
    pub init_binary: Vec<u8>,
    /// `ComputeOnDevice.job_bin_ptr` — where execution starts.
    pub job_bin_ptr: u64,
    /// The finished program-correction flits, or empty when this program needs no correction.
    pub correction: Vec<u8>,
}

/// A group whose json is fully written in STAGING and which is therefore ready to compile, together
/// with the identity its compiled output is reported under.
///
/// A newtype rather than a path and a name because submitting a HALF-WRITTEN group is the one mistake
/// that yields a corrupt device binary instead of an error, and the only constructor is
/// [`Self::sealed`], called at the single site that has just written the group's last file.
#[derive(Clone, Debug)]
pub struct SealedGroup {
    stage: PathBuf,
    id: GroupId,
    /// Bytes of json this group staged — carried so the budget release matches the reserve exactly
    /// rather than being re-measured off a directory that is about to be deleted.
    staged_bytes: usize,
    /// ⭐ CONTENT KEY: a hash of exactly what dxp will read. Two groups with the same key compile to
    /// the same bytes, so the second one copies instead of running dxp.
    ///
    /// MEASURED on a gemma-4 emit mid-flight: 96 groups staged, **56 distinct**, 40 redundant. The
    /// rung ladder emits the same projections at many query-row counts and each bundle also has a
    /// fused twin, so identical group content recurs constantly — 42% of dxp invocations were
    /// recompiling input they had already seen.
    key: u64,
}

impl SealedGroup {
    /// The caller asserts every `sdsc_*.json` AND `bundle.mlir` for this group is in `stage`, that
    /// `staged_bytes` is what it reserved from the [`StageBudget`], and that `key` hashes exactly the
    /// per-op json dxp will read (so equal keys really do mean equal compiler input).
    /// `id` is what the compiled output is reported under.
    pub fn sealed(stage: PathBuf, id: GroupId, staged_bytes: usize, key: u64) -> SealedGroup {
        SealedGroup {
            stage,
            id,
            staged_bytes,
            key,
        }
    }

    pub fn path(&self) -> &Path {
        &self.stage
    }
}

/// The dxp compiler, RESOLVED. Constructible only when both the binary and the SDK share dir it
/// needs are present, so "can this build compile a bundle?" is a `Option<DxpTool>` rather than a
/// pair of strings someone checks at the call site.
#[derive(Clone, Debug)]
pub struct DxpTool {
    bin: PathBuf,
    deeptools: PathBuf,
}

impl DxpTool {
    /// `$DXP_STANDALONE`, else the `bin` sibling of `$DEEPTOOLS_PATH`'s `share` dir, else the
    /// on-pod default — the SAME resolution order `scratchy-builder-spyre`'s `build.rs` uses, so
    /// the two cannot disagree about which compiler ran. `None` when either piece is missing, which
    /// is every cardless build.
    pub fn resolve() -> Option<DxpTool> {
        let deeptools = PathBuf::from(std::env::var("DEEPTOOLS_PATH").ok()?);
        if !deeptools.exists() {
            return None;
        }
        let bin = match std::env::var("DXP_STANDALONE") {
            Ok(p) => PathBuf::from(p),
            Err(_) => deeptools
                .parent()
                .map(|sdk| sdk.join("bin").join("dxp_standalone"))
                .unwrap_or_else(|| PathBuf::from("/opt/ibm/spyre/deeptools/bin/dxp_standalone")),
        };
        bin.exists().then_some(DxpTool { bin, deeptools })
    }

    /// Compile ONE group dir in place. `Ok(())` leaves `spyreCodeDir/{init_binary.bin,
    /// spyrecode.json}` beside the json; `Err` carries dxp's own message, which is the only useful
    /// thing about a scheduler refusal.
    ///
    /// Runs under a [`ChildProc`] — this process's own process group, torn down by `Drop` — so a
    /// DIFFERENT worker's failure can reach and kill this child (via [`Reaper::fail`]) instead of leaving
    /// it to be orphaned when the build exits.
    fn compile(&self, group: &Path, reaper: &Reaper) -> Result<(), String> {
        // DUMP_SPYRE_CODE=1 is what makes dxp emit `spyreCodeDir/` — the artifact the runtime reads
        // and the marker `build.rs` skips on. Mirrors build.rs's invocation exactly.
        let mut cmd = std::process::Command::new(&self.bin);
        cmd.arg("--bundle")
            .arg("-d")
            .arg(group)
            .arg("-b")
            .arg("sentient")
            .env("DEEPTOOLS_PATH", &self.deeptools)
            .env("DUMP_SPYRE_CODE", "1")
            // ⛔⛔⛔ CAP dxp's OWN THREAD POOL, or `COMPILE_WIDTH` children is a thread bomb.
            //
            // dxp sizes its pool from `hardware_concurrency` (`dscglobal.h:56`
            // `parallelThreads = std::thread::hardware_concurrency()`), which reports the HOST's core
            // count and ignores the cgroup quota. MEASURED with `ps -eo nlwp=,pcpu=,rss=` across a real
            // bake: each `dxp_standalone` is **193 threads, ~1.9 GB RSS**, on a pod that advertises
            // `nproc` 192 while `cpu.max` is `2000000 100000` = **20 CPUs**. At `COMPILE_WIDTH` = 32 that
            // is ~6,200 threads, and the bake dies part-way through a group with
            // `LLVM ERROR: pthread_create failed: Resource temporarily unavailable`.
            //
            // `DT_PARALLEL_THREADS` is deeptools' own knob (`util/utils.cpp:18 parseDtParallelThreads`:
            // absolute count, `N%` of hardware_concurrency, or negative for all-minus-N, clamped >= 1).
            // `dxp_standalone` exposes no equivalent flag (`-d`/`-b`/`--dump-bundle-module`/`--use-dxp`
            // only), so the environment is the only seam.
            //
            // ⭐ AND IT IS FASTER THAN THROTTLING THE WIDTH, which is the fix this replaces. Measured on
            // granite-3.1-8b fp16, one build at a time on an otherwise idle pod:
            //
            //   | COMPILE_WIDTH | DT_PARALLEL_THREADS | result                    |
            //   |---------------|---------------------|---------------------------|
            //   | 32            | unset               | ✗ pthread_create (121 s)  |
            //   | 20            | unset               | ✗ pthread_create ( 88 s)  |
            //   | 12            | unset               | ✓ 680 s                   |
            //   | 32            | 1                   | ✓ **448 s**               |
            //
            // ⛔ THE WIDTH IS THE WRONG LEVER, and not merely the slower one: width 32 with no cap
            // SUCCEEDS on one pod (471 s) and fails on another, so any width constant is tuned to one
            // host's quota. A per-child cap is host-independent.
            //
            // ⚠️ NOT CLAIMED: that those 192 threads do no work. `pcpu` sampled ~105 % per dxp, but that
            // is instantaneous and a bursty pool would look the same. The 448 s vs 680 s above is the
            // evidence that 1 is not slower here — not the thread count.
            .env("DT_PARALLEL_THREADS", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // What `Command::output()` did implicitly, and spawning by hand does NOT: dxp gets EOF
            // rather than the build's own stdin. Inheriting it would hand the same descriptor to all
            // `COMPILE_WIDTH` compilers at once.
            .stdin(Stdio::null());
        // ⭐ THE KERNEL-SIDE BACKSTOP for the teardown a userspace sweep CANNOT see. `Reaper::fail`
        // only runs when a dxp compile fails; if the build dies any other way — a panic elsewhere in
        // the emit, an OOM kill, Ctrl-C on cargo — no destructor on these worker threads ever runs, and
        // in-flight children orphan exactly as before. PDEATHSIG makes the kernel SIGKILL the child
        // when the thread that spawned it dies, which covers all of those without our cooperation.
        #[cfg(target_os = "linux")]
        // SAFETY: `pre_exec` runs between fork and exec, where only async-signal-safe calls are
        // permitted. `prctl` is a bare syscall — it allocates nothing and takes no lock.
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let guard = ChildProc::spawn(&mut cmd, reaper)
            .map_err(|e| format!("{}: {e}", self.bin.display()))?;
        let out = guard
            .wait_with_output()
            .map_err(|e| format!("wait {}: {e}", self.bin.display()))?;
        let marker = group.join("spyreCodeDir").join("spyrecode.json");
        if out.status.success() && marker.exists() {
            return Ok(());
        }
        // dxp reports a scheduler refusal on stderr and can still exit 0 while producing nothing,
        // so the marker is part of the verdict — a silent no-output is a failure, not a pass.
        // The REASON is on the `what():` line. `terminate called after throwing an instance of
        // 'DtException'` comes FIRST and contains the word DtException, so matching on that alone
        // reported the abort and threw away the diagnosis — which is what hid
        // `L3DlOpsScheduler.cpp:1375 There must be at least one valid candidate` behind a useless
        // "terminate called" for a whole build.
        let stderr = String::from_utf8_lossy(&out.stderr);
        let dt = stderr
            .lines()
            .find(|l| l.contains("what():"))
            .or_else(|| stderr.lines().find(|l| l.contains("DtException")))
            .unwrap_or_else(|| stderr.lines().last().unwrap_or("(no stderr)"));
        Err(format!(
            "dxp refused {} (status {:?}): {}",
            group.display(),
            out.status.code(),
            dt.trim()
        ))
    }
}

/// READ a compiled group out of staging, then drop the staging dir entirely.
///
/// Nothing is published: dxp's two output files answer two questions ([`CompiledGroup`]) and they are
/// answered here, in the process that ran the compiler. The ~513 `sdsc_*.json` that fed dxp go with the
/// staging dir — they are its INPUT and nothing else reads them, which is what makes staging bounded.
fn read_compiled(stage: &Path, id: &GroupId) -> Result<CompiledGroup, String> {
    let code = stage.join("spyreCodeDir");
    // `compile` has already verified spyrecode.json exists — that IS its success marker.
    let plan = std::fs::read_to_string(code.join("spyrecode.json"))
        .map_err(|e| format!("{id}: read spyrecode.json: {e}"))?;
    let program = correction::parse_spyrecode(&id.to_string(), &plan).map_err(|e| e.to_string())?;
    // An absent image is legitimate: dxp can compile a group to a job plan alone.
    let init_binary = std::fs::read(code.join("init_binary.bin")).unwrap_or_default();
    let _ = std::fs::remove_dir_all(stage);
    Ok(CompiledGroup {
        init_binary,
        job_bin_ptr: program.job_bin_ptr,
        correction: program.correction,
    })
}

/// What a content key is doing right now: being compiled by someone, or already done.
///
/// Two workers can hold groups with the SAME key at once, which is common — the fused twin of a
/// bundle is submitted moments after the split one. The second must WAIT rather than compile: doing
/// the work twice is what this exists to avoid.
enum MemoState {
    InProgress,
    /// The compiled program, shared — a memo hit is an `Arc` clone.
    Done(Arc<CompiledGroup>),
    /// It failed; do not retry it, and do not report a second consequence of the same cause.
    Failed,
}

/// A BOUNDED builder work queue: `IN_FLIGHT` groups may be outstanding, `IN_FLIGHT` workers drain
/// them, and the first dxp failure stops the build.
///
/// The bound is the point. An unbounded queue would let the emitter run ahead and materialise the
/// whole ladder again — which is the problem this exists to solve — so `submit` blocks rather than
/// buffering.
pub struct BakeQueue<const N: usize> {
    stage: StageRoot,
    /// The DISK bound. Shared with the workers, which release a group's bytes when its staging dir
    /// is deleted — that release is what unblocks the emitter's next `reserve`.
    budget: Arc<StageBudget>,
    /// Content key -> what happened to it. Skips dxp for a group whose exact input was already
    /// compiled this build (42% of them, measured). Kept alive here even though only the workers'
    /// cloned `Arc`s read it: this is the handle that outlives the queue itself.
    #[allow(dead_code)]
    memo: Arc<(Mutex<std::collections::HashMap<u64, MemoState>>, Condvar)>,
    /// dxp runs actually skipped, for the build log.
    memo_hits: Arc<AtomicUsize>,
    /// `Mutex<Option<..>>` rather than `Option<..>` because ONE queue serves the whole emit from a
    /// `static` (see [`global`]), so closing it happens through a shared reference.
    tx: Mutex<Option<std::sync::mpsc::SyncSender<SealedGroup>>>,
    /// The pool's threads. RETAINED, not read: the pool is process-lived now that [`Self::finish`] is a
    /// barrier rather than a shutdown, and these are what own it. Dropping the handles would only detach
    /// the threads; keeping them is what leaves a real shutdown available if one is ever wanted.
    #[allow(dead_code)]
    workers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// First failure — kept so `submit` can refuse further work and the caller can surface it — TOGETHER
    /// with every live `dxp_standalone`, because recording that failure is what kills them. See
    /// [`Reaper`].
    reaper: Arc<Reaper>,
    compiled: Arc<AtomicUsize>,
    /// Device-image bytes compiled, for the build log.
    device_bytes: Arc<AtomicUsize>,
    /// ⭐ EVERY COMPILED GROUP, BY IDENTITY. The emit's output, in memory, drained by [`Self::finish`].
    results: Arc<Mutex<HashMap<GroupId, Arc<CompiledGroup>>>>,
    /// Groups handed to the workers and not yet accounted for — see [`Latch`].
    inflight: Arc<Latch>,
}

impl<const N: usize> BakeQueue<N> {
    const _BOUNDED: () = assert!(
        N > 0,
        "a bounded queue with no slots cannot make progress — the emitter would block forever"
    );

    /// Start the workers. `None` when this build has no dxp (cardless): the caller then writes json
    /// and leaves it for `build.rs`, exactly as before.
    pub fn start() -> Option<Bake> {
        let () = Self::_BOUNDED;
        let tool = DxpTool::resolve()?;
        // A SyncSender IS the bound: `send` blocks while `N` items are unclaimed.
        let (tx, rx) = std::sync::mpsc::sync_channel::<SealedGroup>(N);
        let rx = Arc::new(Mutex::new(rx));
        let reaper: Arc<Reaper> = Arc::new(Reaper::default());
        // Arm both teardowns before the first child can exist: the emit's own panics unwind only the
        // main thread, and a `^C` unwinds nothing at all, so these hooks are the only things that reap a
        // worker's in-flight child.
        arm_teardown(&reaper);
        let compiled = Arc::new(AtomicUsize::new(0));
        let device_bytes = Arc::new(AtomicUsize::new(0));
        let results: Arc<Mutex<HashMap<GroupId, Arc<CompiledGroup>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let budget = Arc::new(StageBudget::new());
        let memo: Arc<(Mutex<std::collections::HashMap<u64, MemoState>>, Condvar)> =
            Arc::new((Mutex::new(std::collections::HashMap::new()), Condvar::new()));
        let memo_hits = Arc::new(AtomicUsize::new(0));
        // COMPILE_WIDTH workers, not N: N is only the channel depth. Conflating them is what made
        // the disk bound and the compile width the same number.
        let inflight: Arc<Latch> = Arc::new(Latch::default());
        let mut workers = Vec::with_capacity(COMPILE_WIDTH);
        for _ in 0..COMPILE_WIDTH {
            let (rx, reaper, tool) = (Arc::clone(&rx), Arc::clone(&reaper), tool.clone());
            let (compiled, device_bytes) = (Arc::clone(&compiled), Arc::clone(&device_bytes));
            let results = Arc::clone(&results);
            let budget = Arc::clone(&budget);
            let (memo, memo_hits) = (Arc::clone(&memo), Arc::clone(&memo_hits));
            let inflight = Arc::clone(&inflight);
            workers.push(std::thread::spawn(move || {
                loop {
                    // Hold the receiver lock only for the recv, never across the compile — otherwise
                    // the `N` workers would serialise into one.
                    let job = match rx.lock() {
                        Ok(g) => g.recv(),
                        Err(_) => return,
                    };
                    let Ok(job) = job else { return };
                    // Already failed? Drain without working, so the emitter's `submit` error is the
                    // one that surfaces rather than a pile of consequences.
                    if reaper.has_failed() {
                        // Still release: the emitter may be blocked in `reserve` and has to be able
                        // to reach its own `submit` error rather than deadlocking behind a drain.
                        let _ = std::fs::remove_dir_all(&job.stage);
                        budget.release(job.staged_bytes);
                        inflight.leave();
                        continue;
                    }
                    // ⭐ HAS THIS EXACT INPUT ALREADY BEEN COMPILED? Claim the key, or wait for
                    // whoever holds it and copy their result. Waiting rather than racing matters:
                    // a bundle's fused twin is submitted moments after the split one with identical
                    // group content, so without the wait both would run dxp on the same bytes.
                    let mut memoized: Option<Arc<CompiledGroup>> = None;
                    let mut mine = false;
                    {
                        let (lock, cv) = &*memo;
                        let mut m = match lock.lock() {
                            Ok(g) => g,
                            Err(_) => return,
                        };
                        loop {
                            match m.get(&job.key) {
                                None => {
                                    m.insert(job.key, MemoState::InProgress);
                                    mine = true;
                                    break;
                                }
                                Some(MemoState::Done(g)) => {
                                    memoized = Some(Arc::clone(g));
                                    break;
                                }
                                // Someone else's identical group already failed — its error is the
                                // one that surfaces; do not compile it again to say the same thing.
                                Some(MemoState::Failed) => break,
                                Some(MemoState::InProgress) => {
                                    m = match cv.wait(m) {
                                        Ok(g) => g,
                                        Err(_) => return,
                                    };
                                }
                            }
                        }
                    }
                    let outcome: Result<Arc<CompiledGroup>, String> = match (memoized, mine) {
                        (Some(done), _) => {
                            memo_hits.fetch_add(1, Ordering::Relaxed);
                            // The staged json is this group's only footprint; its compiled twin is
                            // already in hand.
                            let _ = std::fs::remove_dir_all(&job.stage);
                            Ok(done)
                        }
                        (None, false) => Err(format!(
                            "an identical group ({:016x}) already failed to compile",
                            job.key
                        )),
                        (None, true) => tool
                            .compile(job.path(), &reaper)
                            .and_then(|()| read_compiled(&job.stage, &job.id))
                            .map(Arc::new),
                    };
                    if mine {
                        let (lock, cv) = &*memo;
                        if let Ok(mut m) = lock.lock() {
                            m.insert(
                                job.key,
                                match &outcome {
                                    Ok(g) => MemoState::Done(Arc::clone(g)),
                                    Err(_) => MemoState::Failed,
                                },
                            );
                        }
                        cv.notify_all();
                    }
                    // The staging dir is gone by now on success, and on failure it is kept for
                    // diagnosis — either way these bytes are no longer the emitter's problem.
                    budget.release(job.staged_bytes);
                    match outcome {
                        Ok(g) => {
                            compiled.fetch_add(1, Ordering::Relaxed);
                            device_bytes.fetch_add(g.init_binary.len(), Ordering::Relaxed);
                            // ⭐ THE RETURN CHANNEL, keyed by identity: the group is either here for
                            // the emitter to bake, or the build fails naming it.
                            if let Ok(mut r) = results.lock() {
                                r.insert(job.id.clone(), g);
                            }
                        }
                        // Records the failure AND, if it is the first, SIGKILLs every other live
                        // compiler — one call, because they are one decision under one lock.
                        Err(e) => reaper.fail(e),
                    }
                    inflight.leave();
                }
            }));
        }
        Some(BakeQueue {
            stage: StageRoot::resolve(),
            budget,
            memo,
            memo_hits,
            tx: Mutex::new(Some(tx)),
            workers: Mutex::new(workers),
            reaper,
            compiled,
            device_bytes,
            results,
            inflight,
        })
    }

    /// Where this queue stages per-op json — the emitter writes group dirs under it.
    pub fn stage(&self) -> &StageRoot {
        &self.stage
    }

    /// Claim `bytes` of the staging budget, BLOCKING until they fit. Call before writing a group's
    /// json; the worker releases the same count once that group's staging dir is gone. This is the
    /// disk bound — without it the emitter runs ahead of dxp and stages the whole ladder.
    pub fn reserve(&self, bytes: usize) {
        self.budget.reserve(bytes);
    }

    /// High-water staging use, for the build log.
    pub fn peak_staged_bytes(&self) -> usize {
        self.budget.peak_bytes()
    }

    /// Hand one finished group to the compilers. BLOCKS while `N` are outstanding — that block is
    /// the disk bound. `Err` as soon as any group has failed, so the emitter stops instead of
    /// writing the rest of a ladder that cannot compile.
    pub fn submit(&self, group: SealedGroup) -> Result<(), String> {
        if let Some(e) = self.reaper.first_error() {
            return Err(e);
        }
        // Clone the sender out from under the lock: `send` BLOCKS when the queue is full, and
        // holding the lock across it would serialise every producer behind one blocked send.
        let tx = match self.tx.lock() {
            Ok(g) => g.clone(),
            Err(_) => None,
        };
        match tx {
            // A closed channel means every worker is gone; the stored error explains why.
            None => Err("bake queue already finished".to_string()),
            Some(tx) => {
                // Entered BEFORE the send: a job must be outstanding before any worker can account for
                // it, or `finish` could return through a gap and read an incomplete `results`.
                self.inflight.enter();
                tx.send(group).map_err(|_| {
                    self.inflight.cancel();
                    self.reaper
                        .first_error()
                        .unwrap_or_else(|| "bake workers exited".to_string())
                })
            }
        }
    }

    /// ⛔⛔⛔ A BARRIER, NOT A SHUTDOWN — AND THAT DISTINCTION WAS A BUILD FAILURE.
    ///
    /// Waits until nothing is outstanding ([`Latch`]), then reports the first failure. The channel stays
    /// open and the workers stay alive.
    ///
    /// 🛑 IT USED TO DROP THE SENDER AND JOIN THE WORKERS, on the reasoning that it is "called ONCE per
    /// emit". It is called once per `#[forward]` EXPANSION, and a build with more than one model expands
    /// more than once against ONE process-wide queue ([`global`]) — so the first model's finish killed the
    /// pool and every bundle of the next model died in `submit`:
    ///
    /// ```text
    /// ktir_decode_granite_3_1_8b_...: 3-bundle write failed — prefix_err=Some("bake queue already
    /// finished") body_err=Some(...) suffix_err=Some(...)
    /// ```
    ///
    /// The latch gives the same guarantee the join gave — every submitted group is in `results` before
    /// this returns — without ending the pool, so the next expansion keeps these workers, this memo table
    /// (which is what lets a projection shared between two models compile once) and this staging budget.
    /// Still one barrier per expansion, so dxp runs `COMPILE_WIDTH` wide across bundle boundaries.
    ///
    /// ⭐ THE COUNTS ARE CUMULATIVE across expansions, deliberately: they describe what the BUILD
    /// compiled, which is what the log line is for.
    pub fn finish(&self) -> Result<BakeStats, String> {
        self.inflight.wait_empty(&|| self.reaper.has_failed());
        if let Some(e) = self.reaper.first_error() {
            return Err(e);
        }
        Ok(BakeStats {
            groups: self.compiled.load(Ordering::Relaxed),
            device_bytes: self.device_bytes.load(Ordering::Relaxed),
            peak_staged_bytes: self.budget.peak_bytes(),
            memo_hits: self.memo_hits.load(Ordering::Relaxed),
        })
    }

    /// The compiled program for one group, once [`Self::finish`] has returned.
    ///
    /// `None` means dxp never produced it — which after a successful `finish` can only be a group that
    /// was never submitted, so the caller refuses naming the id rather than baking a bundle with a hole
    /// in its launch sequence.
    pub fn compiled_group(&self, id: &GroupId) -> Option<Arc<CompiledGroup>> {
        self.results.lock().ok()?.get(id).cloned()
    }

    /// Has anything been submitted? Lets the caller skip a "0 groups" log line.
    pub fn any(&self) -> bool {
        self.compiled.load(Ordering::Relaxed) > 0
    }
}

/// ⭐ ONE QUEUE FOR THE WHOLE EMIT.
///
/// The queue used to be created and drained per BUNDLE, which put a join barrier at every bundle
/// boundary: the compile width could never exceed one bundle's group count, and each boundary wound
/// down to a single running `dxp_standalone` before the next wound up. Across a 27-bundle ladder that
/// is 27 serialisation points, and it shows up exactly as "sometimes 8, sometimes 3, sometimes 1".
///
/// Process-wide, so group N of bundle 3 compiles while bundle 4 is being written. `None` on a
/// cardless build (no dxp), which is what keeps `cargo check` working.
pub fn global() -> Option<&'static Bake> {
    static Q: std::sync::OnceLock<Option<Bake>> = std::sync::OnceLock::new();
    let q = Q.get_or_init(Bake::start).as_ref();
    if q.is_none() {
        no_dxp_or_die();
    }
    q
}

/// ⛔⛔⛔ A PLAN-ONLY BAKE IS NOT A BUILD — IT IS A BINARY THAT CANNOT COMPUTE, AND IT USED TO EXIT 0.
///
/// `DxpTool::resolve()` yields `None` on any host without the deeptools compiler, and the emit then wrote
/// a bundle's memory PLAN with NO device programs. Nothing failed: the plan is pure Rust, so the crate
/// compiled, the binary linked, and it SERVED — every session reported itself ready off the plan, no launch
/// ever happened, forwards returned in microseconds, the logits buffer stayed zero, argmax landed on a
/// constant special id, and completions came back `""` with `finish_reason: "length"` and no error. A
/// container image built this way ran for hours looking healthy.
///
/// 🛑 WHY IT WAS SILENT, TWICE OVER. The `None` arm was DESIGNED for the cardless laptop (`cargo check`),
/// so it is the default rather than a stated intent; and a build script's stderr is swallowed by cargo, so
/// even a warning would not have reached the build log. The image build then set `DEEPTOOLS_PATH` in its
/// RUNTIME stage but not in the stage that runs `cargo build`, and there was nothing anywhere to say so.
///
/// So: FAIL, naming the variable and the stage. A host that genuinely has no card must say so on purpose
/// via `SCRATCHY_PLAN_ONLY_BAKE=1` — which is a claim about the machine, not a fallback the build picks by
/// itself. Type-checking without a card stays possible; shipping a model that cannot compute does not.
fn no_dxp_or_die() {
    if std::env::var_os("SCRATCHY_PLAN_ONLY_BAKE").is_some() {
        return;
    }
    panic!(
        "SuperDSC bake: no device compiler — `DEEPTOOLS_PATH` is unset or `dxp_standalone` is missing, \
         so this bundle would be emitted as a memory PLAN WITH NO DEVICE PROGRAMS. That binary links and \
         serves: every session reports ready, nothing is ever launched, and every completion comes back \
         EMPTY (`finish_reason: \"length\"`, ~0.6 ms/token) with no error anywhere. Refusing to build it.\n\
         \n\
         • Set `DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share` (and the deeptools `LD_LIBRARY_PATH`) in \
         the stage that runs `cargo build` — in a Dockerfile that is the BUILD stage, not the runtime \
         stage. Setting it only at runtime is exactly this failure.\n\
         • Or, on a machine that truly has no card and only needs a type-check, state it: \
         `SCRATCHY_PLAN_ONLY_BAKE=1`."
    );
}

/// Drain the process-wide queue. Call once, after the last bundle is written and before its compiled
/// artifacts are read back.
pub fn finish_global() -> Result<Option<BakeStats>, String> {
    match global() {
        None => Ok(None),
        Some(q) => q.finish().map(Some),
    }
}

/// What one bundle's inline bake did — reported so a build log says how much it never left on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BakeStats {
    pub groups: usize,
    /// Device-image bytes compiled — how much device code the emit produced, which is what reaches
    /// the binary.
    pub device_bytes: usize,
    /// High-water staging use. Bounded by `MAX_STAGED_BYTES` by construction; reported so a build
    /// log states what it actually took rather than what it was allowed.
    pub peak_staged_bytes: usize,
    /// dxp invocations SKIPPED because an identical group had already been compiled.
    pub memo_hits: usize,
}

/// ⛔ THE ZOMBIE REGRESSION TESTS — a leak that is COUNTED, so the test counts it.
///
/// Each asks the one question the process table answers: after the teardown under test, is the child
/// still THIS process's child? `waitpid` says `ECHILD` only once a child has been reaped, so `ECHILD` is
/// the pass condition and BOTH other answers are the bug — `0` means it is still running, and a positive
/// return means it was sitting there as a zombie and the test itself just collected it.
///
/// These run anywhere `dxp` does not have to exist (a Mac included): the leak is about this process's own
/// children, so `sleep` and `true` stand in for the compiler exactly.
#[cfg(test)]
mod tests {
    use super::*;

    /// ⛔ EVERY TEST THAT SPAWNS TAKES THIS. [`signals::sweep`] claims the WHOLE process-wide table, so a
    /// sweep running beside another test's live child would reap that child out from under it and turn
    /// `a_waited_guard_leaves_nothing_behind` into a flake. `cargo test` runs these on threads of ONE
    /// process, so the table is shared whether or not the tests are written as if it is.
    static SPAWNING: Mutex<()> = Mutex::new(());

    fn spawning() -> std::sync::MutexGuard<'static, ()> {
        SPAWNING.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Is `pid` still ours — alive or a zombie? `false` (i.e. `ECHILD`) is the only answer that means
    /// REAPED.
    fn still_ours(pid: i32) -> bool {
        let mut status: libc::c_int = 0;
        // SAFETY: a positive pid waits on exactly that child, and `status` is a live local.
        unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) >= 0 }
    }

    /// A child that outlives the test if the reap is missing, so a leak is visible rather than racing
    /// with normal exit.
    fn sleeper() -> std::process::Command {
        let mut cmd = std::process::Command::new("sleep");
        cmd.arg("30")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .stdin(Stdio::null());
        cmd
    }

    /// ⛔⭐ THE TEST THAT WOULD HAVE CAUGHT THIS ON DAY ONE, and the one no previous attempt wrote.
    ///
    /// A tty delivers `^C` to its FOREGROUND PROCESS GROUP and nowhere else, so "the child is in our
    /// process group" IS the property that makes Ctrl+C work. `process_group(0)` silently traded it away
    /// for a subtree kill dxp never needed, and every test still passed: the four zombie tests all drive
    /// teardown from inside this process, where a private group looks identical to a shared one. Only the
    /// tty can tell the difference, and nothing asked it.
    ///
    /// Deterministic, no signals, no `/proc` — one `getpgid` against another.
    #[test]
    fn a_spawned_child_is_in_our_own_process_group() {
        let _serial = spawning();
        let reaper = Reaper::default();
        let guard = ChildProc::spawn(&mut sleeper(), &reaper).expect("spawn");
        let pid = guard.pid;
        // SAFETY: `getpgid` reads a pid's process group; 0 means this process. Both are plain integers.
        let (child_pgrp, our_pgrp) = unsafe { (libc::getpgid(pid), libc::getpgid(0)) };
        assert!(child_pgrp > 0, "getpgid({pid}) failed");
        assert_eq!(
            child_pgrp, our_pgrp,
            "a dxp child must share OUR process group ({our_pgrp}), not sit in its own ({child_pgrp}) \
             where a terminal's ^C/^\\/SIGHUP can never reach it"
        );
    }

    /// ⭐ THE SIGNAL TEARDOWN ITSELF: a child that only the lock-free table knows about is killed AND
    /// reaped.
    ///
    /// This is the path a `^C` takes, minus the dying. [`signals::sweep`] is split out of the handler
    /// precisely so a test can reach it — the handler ends in `_exit`, so the only other way to observe it
    /// would be to kill the test binary and go looking in `/proc`, which reports "reaped" and "collected
    /// by a reaping PID 1" identically and so cannot tell a pass from a failure.
    #[test]
    fn the_signal_sweep_kills_and_reaps_a_tracked_child() {
        let _serial = spawning();
        let child = sleeper().spawn().expect("spawn");
        let pid = child.id() as i32;
        signals::track(pid);
        // Dropping a `std::process::Child` neither kills nor reaps — that fact is the whole bug — so this
        // is exactly the shape of a worker thread frozen mid-compile by a signal.
        drop(child);
        signals::sweep();
        assert!(
            !still_ours(pid),
            "pid {pid} survived the signal sweep unreaped — this is the ^C leak"
        );
    }

    /// ⛔ THE NEGATIVE CONTROL for the sweep test above. If `untrack` were a no-op the sweep would still
    /// reap this child, that test would pass for the wrong reason, and the leaked slots would fill the
    /// 64-entry table until real children silently stopped being tracked at all.
    ///
    /// ⚠️ Uses a REAL child, never a made-up pid: a table that failed to release its slot makes the sweep
    /// send SIGKILL to whatever it still holds, and a fabricated number in there is a live stranger's pid.
    #[test]
    fn untrack_removes_a_child_from_the_signal_sweep() {
        let _serial = spawning();
        let mut child = sleeper().spawn().expect("spawn");
        let pid = child.id() as i32;
        signals::track(pid);
        signals::untrack(pid);
        signals::sweep();
        assert!(
            still_ours(pid),
            "untrack did not release the slot — the sweep reaped a child it no longer tracked"
        );
        // Ours to clean up, precisely because the sweep correctly left it alone.
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Dropping a guard that never waited must leave NOTHING behind. This is the path a refused
    /// registration takes (`spawn` returns `Err` and only `Drop` will ever see that child), and the path
    /// an unwound `wait_with_output` takes. Before the reap, the SIGKILL alone left a zombie this process
    /// still owned — which on exit re-parented to a PID 1 that never calls `wait()`.
    #[test]
    fn dropping_an_unwaited_guard_reaps_the_child() {
        let _serial = spawning();
        let reaper = Reaper::default();
        let guard = ChildProc::spawn(&mut sleeper(), &reaper).expect("spawn");
        let pid = guard.pid;
        drop(guard);
        assert!(!still_ours(pid), "pid {pid} survived Drop unreaped");
        assert!(reaper.inner().live.is_empty(), "Drop must deregister");
    }

    /// The exit sweep reaps a child whose owning worker will NEVER run again — the `atexit` path, and the
    /// one [`Reaper::fail`] cannot cover.
    #[test]
    fn the_exit_sweep_reaps_a_child_no_destructor_will_see() {
        let _serial = spawning();
        let reaper = Reaper::default();
        let child = sleeper().spawn().expect("spawn");
        let pid = child.id() as i32;
        assert!(reaper.register(pid), "a fresh Reaper must accept a child");
        // Dropping a `std::process::Child` does NOT reap it — that fact is the whole bug — so this is
        // exactly the shape of a worker thread frozen mid-compile by process exit.
        drop(child);
        reaper.kill_and_reap_all();
        assert!(
            !still_ours(pid),
            "pid {pid} survived the exit sweep unreaped"
        );
    }

    /// ⛔ THE SWEEP MUST CLOSE THE DOOR BEFORE IT COUNTS, or its snapshot of `live` is a sample rather than
    /// a set — the workers are still running during `atexit` and a child registered after the snapshot is
    /// never swept. Refusing every later spawn is what makes `live` monotonically shrink, and so what makes
    /// the sweep's bounded rounds converge.
    #[test]
    fn the_exit_sweep_refuses_every_later_spawn() {
        let _serial = spawning();
        let reaper = Reaper::default();
        reaper.kill_and_reap_all();
        assert!(
            !reaper.register(4242),
            "a registration after the sweep must be refused, or it escapes the sweep"
        );
        // And a refused spawn is torn down by its own guard, which is the path that makes the refusal safe.
        let err = ChildProc::spawn(&mut sleeper(), &reaper)
            .err()
            .expect("spawn must be refused once the sweep has run");
        assert!(err.contains("already failed"), "unexpected refusal: {err}");
    }

    /// The SUCCESS path must not regress: a guard that DID wait has nothing left to kill, and killing
    /// there would target a pid already free for reuse.
    #[test]
    fn a_waited_guard_leaves_nothing_behind() {
        let _serial = spawning();
        let reaper = Reaper::default();
        let mut cmd = std::process::Command::new("true");
        cmd.stdout(Stdio::null())
            .stderr(Stdio::null())
            .stdin(Stdio::null());
        let guard = ChildProc::spawn(&mut cmd, &reaper).expect("spawn");
        let pid = guard.pid;
        let out = guard.wait_with_output().expect("wait");
        assert!(out.status.success(), "`true` must exit 0");
        assert!(!still_ours(pid), "a waited leader must already be reaped");
        assert!(reaper.inner().live.is_empty(), "the guard must deregister");
    }
}
