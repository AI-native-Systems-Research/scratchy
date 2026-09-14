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
    /// pgid of every `dxp_standalone` currently running. Each is its own process group, so a kill
    /// reaches its descendants without touching this process or a sibling compile.
    live: HashSet<i32>,
}

impl Reaper {
    /// Every critical section here is a `HashSet` operation and `kill(2)`, neither of which can panic
    /// or leave a half-updated invariant, so a poisoned lock cannot mean broken state — recovering
    /// beats propagating a spurious failure through every call site.
    fn inner(&self) -> std::sync::MutexGuard<'_, ReaperInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record `pgid` as live, or REFUSE because the build has already failed — in which case the
    /// caller's guard kills it immediately rather than adding a child nothing will reap.
    fn register(&self, pgid: i32) -> bool {
        let mut g = self.inner();
        if g.err.is_some() {
            return false;
        }
        g.live.insert(pgid);
        true
    }

    fn deregister(&self, pgid: i32) {
        self.inner().live.remove(&pgid);
    }

    /// Record the first failure and SIGKILL every live compiler. Later failures only report: the sweep
    /// has happened, and everything it could still kill is already dying.
    fn fail(&self, e: String) {
        let mut g = self.inner();
        if g.err.is_some() {
            return;
        }
        g.err = Some(e);
        for &pgid in g.live.iter() {
            kill_group(pgid);
        }
    }

    fn first_error(&self) -> Option<String> {
        self.inner().err.clone()
    }

    fn has_failed(&self) -> bool {
        self.inner().err.is_some()
    }
}

/// SIGKILL a whole process group.
///
/// ⚠️ Only sound while the group is known to have a live member: a pgid is reusable once its last
/// member is reaped, so a kill sent after that could in principle land on an unrelated new group. Every
/// caller here sends it to a group it is still holding a `Child` for, or has just refused to reap.
fn kill_group(pgid: i32) {
    // SAFETY: a negative pid targets the process group rather than one pid. ESRCH (the group is
    // already gone) is the expected outcome of a race with normal exit, not an error to surface.
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
}

/// ⭐ ONE `dxp_standalone` CHILD, OWNED — its own process group, registered in the [`Reaper`] for as
/// long as `Self` is alive.
///
/// `Drop`, not the order of statements in [`DxpTool::compile`], is what guarantees the registry entry
/// is cleared and an unreaped group is killed: true today (the only path is spawn then wait), and still
/// true of whatever `compile` grows into later — an early `?`, a timeout, a panic on this thread. A bare
/// insert-then-remove around the wait call gets that right only until someone edits the function
/// between the two lines.
struct ChildGroup<'a> {
    /// `None` once [`Self::wait_with_output`] has reaped the leader — which is what tells `Drop` the
    /// group must NOT be killed, its pgid being free for reuse from that moment on.
    child: Option<std::process::Child>,
    pgid: i32,
    reaper: &'a Reaper,
}

impl<'a> ChildGroup<'a> {
    /// Spawn `cmd` into a FRESH process group (pgid == its own pid, via `process_group(0)`) — detached
    /// from this process's group and every sibling's — and register it.
    ///
    /// `Err` once the build has already failed: the child is spawned but immediately torn down by the
    /// guard's own `Drop`, so losing the registration race cannot leave it running.
    fn spawn(cmd: &mut std::process::Command, reaper: &'a Reaper) -> Result<Self, String> {
        let child = cmd
            .process_group(0)
            .spawn()
            .map_err(|e| format!("spawn: {e}"))?;
        let pgid = child.id() as i32;
        // Construct the guard BEFORE registering, so the refusal path below tears the child down
        // through the same `Drop` as every other exit.
        let guard = ChildGroup {
            child: Some(child),
            pgid,
            reaper,
        };
        if !reaper.register(pgid) {
            return Err("the bake already failed in another group".to_string());
        }
        Ok(guard)
    }

    /// Wait for the leader and collect its output, consuming the guard so `Drop` runs immediately
    /// after — deregistering either way.
    fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        self.child
            .take()
            .expect("ChildGroup::spawn is the only constructor and it always fills child")
            .wait_with_output()
    }
}

impl Drop for ChildGroup<'_> {
    fn drop(&mut self) {
        self.reaper.deregister(self.pgid);
        // Kill ONLY while the leader is still unreaped. Once `wait_with_output` has reaped it the group
        // may be empty and its pgid already recycled, so a kill here could hit an unrelated process
        // group; while we still hold the `Child`, the group is guaranteed to be ours.
        if self.child.is_some() {
            kill_group(self.pgid);
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
    /// Runs under a [`ChildGroup`] — its own process group, torn down by `Drop` — so a DIFFERENT
    /// worker's failure can reach and kill this child (via [`Reaper::fail`]) instead of leaving it to be
    /// orphaned when the build exits.
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
        let guard = ChildGroup::spawn(&mut cmd, reaper)
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
