// SPDX-License-Identifier: Apache-2.0
//! `SCRATCHY_LOAD_TIME`: sub-phase timing of the generated weight load
//! (`try_load` — the pass the `#[forward]`-emitted `Weights::load` makes over
//! every tensor of the checkpoint).
//!
//! The instrument for the boot layer between "Using cached model" and
//! "superdsc_weights returned": that window is `try_load` almost whole
//! (measured 2b fp8, nickm3, 2026-11-10: 1.01s of the 1.13s window), and every
//! per-byte pass the load makes lives in shared code — the cast in
//! [`crate::weights::GpuWeights`]'s `maybe_cast_cpu`, the `.to_vec()` of
//! `take_cpu`, the transpose of `load_raw`, and the backend allocator's
//! host-copy. Each site accumulates into the counters here; the worker that
//! ran the load prints [`summary`] next to its own whole-pass timing, so the
//! residual (fingerprint match, store ops, struct building) is the difference
//! the two numbers leave.
//!
//! Off unless `SCRATCHY_LOAD_TIME` is set; the recorders are one relaxed
//! atomic add each, so the on-cost is noise against a per-tensor copy.

use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// One counter per per-byte pass the load makes, plus the whole-pass total the
/// printing site supplies. `bytes` and `elapsed` are accumulated separately so
/// a summary can state a rate, not just a time.
struct Counters {
    calls: AtomicU64,
    bytes: AtomicU64,
    elapsed_ns: AtomicU64,
}

impl Counters {
    fn record(&self, bytes: usize, elapsed_ns: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.elapsed_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
    }
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static CAST: Counters = Counters {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    elapsed_ns: AtomicU64::new(0),
};
static ALLOC_ZERO: Counters = Counters {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    elapsed_ns: AtomicU64::new(0),
};
static ALLOC_COPY: Counters = Counters {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    elapsed_ns: AtomicU64::new(0),
};
static ALLOC_VIEW: Counters = Counters {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    elapsed_ns: AtomicU64::new(0),
};
static TO_VEC: Counters = Counters {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    elapsed_ns: AtomicU64::new(0),
};
static TRANSPOSE: Counters = Counters {
    calls: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    elapsed_ns: AtomicU64::new(0),
};

/// Whether `SCRATCHY_LOAD_TIME` is set. Checked once per process into
/// `ENABLED`; the recorders read that flag.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Arm the instrument (`SCRATCHY_LOAD_TIME` set). Called by the recording
/// sites' crate init path — the worker's load — so the env var is read once,
/// not per tensor.
pub fn arm() {
    if std::env::var_os("SCRATCHY_LOAD_TIME").is_some() {
        ENABLED.store(true, Ordering::Relaxed);
    }
}

/// The cast pass of `maybe_cast_cpu`: one dtype conversion of `bytes` through
/// the cast scratch (the no-cast early return records nothing — it is not a
/// pass over the bytes).
pub fn record_cast(bytes: usize, start: Instant) {
    if enabled() {
        CAST.record(bytes, start.elapsed().as_nanos() as u64);
    }
}

/// The backend allocator's zero-fill of a fresh host buffer (`vec![0u8; n]`
/// before the copy that overwrites it). Counted apart from the copy because
/// the two have different fixes: the zero is removable without touching the
/// copy's contract.
pub fn record_alloc_zero(bytes: usize, start: Instant) {
    if enabled() {
        ALLOC_ZERO.record(bytes, start.elapsed().as_nanos() as u64);
    }
}

/// The backend allocator's host copy (`src_host` → the fresh buffer).
pub fn record_alloc_copy(bytes: usize, start: Instant) {
    if enabled() {
        ALLOC_COPY.record(bytes, start.elapsed().as_nanos() as u64);
    }
}

/// The backend allocator's ALIAS of its source instead of a copy — the bytes
/// already live where the backend wants them (a registered mmap under a
/// host-memory allocator), so the "allocation" is a keepalive handle and this
/// records the bytes that DID NOT copy. A load where every weight views is a
/// load whose `alloc-copy` line reads zero calls.
pub fn record_alloc_view(bytes: usize) {
    if enabled() {
        ALLOC_VIEW.record(bytes, 0);
    }
}

/// `take_cpu`/`copy_cpu`'s `.to_vec()` — a whole-tensor copy out of the store
/// or the cast scratch.
pub fn record_to_vec(bytes: usize, start: Instant) {
    if enabled() {
        TO_VEC.record(bytes, start.elapsed().as_nanos() as u64);
    }
}

/// `transpose_2d_in_place` — the whole-tensor transpose `load_raw` runs.
pub fn record_transpose(bytes: usize, start: Instant) {
    if enabled() {
        TRANSPOSE.record(bytes, start.elapsed().as_nanos() as u64);
    }
}

fn push_line(out: &mut String, name: &str, c: &Counters) {
    let ns = c.elapsed_ns.load(Ordering::Relaxed);
    let bytes = c.bytes.load(Ordering::Relaxed);
    let calls = c.calls.load(Ordering::Relaxed);
    let secs = ns as f64 / 1e9;
    let rate = if bytes > 0 {
        format!(" ({:.0} MB/s)", bytes as f64 / 1e6 / secs)
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "[load-time] {name:<10} {secs:6.3}s  {calls:>4} call(s)  {bytes:>12} B{rate}"
    );
}

/// One block naming every counted pass and the residual the whole-pass total
/// leaves unexplained. `total` is the caller's own timing of the whole load
/// (`try_load`); the residual is fingerprint matching, store ops, and struct
/// building — the work no per-byte pass accounts for.
pub fn summary(total: std::time::Duration) -> String {
    let counted = CAST.elapsed_ns.load(Ordering::Relaxed)
        + ALLOC_ZERO.elapsed_ns.load(Ordering::Relaxed)
        + ALLOC_COPY.elapsed_ns.load(Ordering::Relaxed)
        + TO_VEC.elapsed_ns.load(Ordering::Relaxed)
        + TRANSPOSE.elapsed_ns.load(Ordering::Relaxed);
    let total_ns = total.as_nanos() as u64;
    let residual = total_ns.saturating_sub(counted);
    let mut out = format!("[load-time] try_load   {:6.3}s\n", total.as_secs_f64());
    push_line(&mut out, "cast", &CAST);
    push_line(&mut out, "alloc-zero", &ALLOC_ZERO);
    push_line(&mut out, "alloc-copy", &ALLOC_COPY);
    push_line(&mut out, "alloc-view", &ALLOC_VIEW);
    push_line(&mut out, "to-vec", &TO_VEC);
    push_line(&mut out, "transpose", &TRANSPOSE);
    let _ = writeln!(
        out,
        "[load-time] {:<10} {:6.3}s  (fingerprint match, store ops, struct building)",
        "residual",
        residual as f64 / 1e9
    );
    out
}
