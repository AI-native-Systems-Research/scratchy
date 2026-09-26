// SPDX-License-Identifier: Apache-2.0
//! Metal implementation of the [`DeviceAllocator`] trait.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::residency::Pinned;
use scratchy_tensors::DeviceAllocator;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Send wrapper so a detached background thread can hold a Metal buffer
/// alive past the spawning scope. `Retained<…MTLBuffer>` is `!Send`, but
/// a post-copy weight buffer is immutable and `StorageModeShared`, so
/// retaining/reading it from another thread is sound. Used to keep the
/// sidecar-writer's source allocation mapped until the write finishes
/// even if the owning `MmapRegion` is torn down first. (regression: c54955a3)
struct BufKeepAlive(#[allow(dead_code)] Buffer);
// SAFETY: see doc comment — immutable shared-storage buffer, read-only access.
unsafe impl Send for BufKeepAlive {}

const DEFAULT_CHUNK_BYTES: usize = 256 * 1024 * 1024;

struct MetalArena {
    buffer: Pinned,
    base: *mut u8,
    capacity: usize,
    used: usize,
}

unsafe impl Send for MetalArena {}
unsafe impl Sync for MetalArena {}

/// Fast non-cryptographic content hash over a byte region —
/// u64-chunked FNV-1a variant (~RAM-bandwidth in release). Integrity
/// bit for the aligned sidecar: computed during the build write,
/// re-verified in the background after every cache-hit launch.
fn content_hash64(base: *const u8, len: usize) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let words = len / 8;
    // SAFETY: caller guarantees `base..base+len` readable.
    let w = unsafe { std::slice::from_raw_parts(base as *const u64, words) };
    for &x in w {
        h = (h ^ x).wrapping_mul(PRIME);
    }
    let tail = unsafe { std::slice::from_raw_parts(base.add(words * 8), len - words * 8) };
    for &b in tail {
        h = (h ^ b as u64).wrapping_mul(PRIME);
    }
    h
}

/// Latch flipped by the executor once model load + warmup complete.
/// Background sidecar writers wait on it so the one-time cache build
/// never contends with the load itself (observed: writers racing the
/// realign-copy turned an 8-9.5 s miss launch into 18.6 s — three I/O
/// streams + ~19 GiB of dirtied page cache on a 32 GiB box).
static WEIGHTS_LOAD_COMPLETE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Called by the worker after warmup. Idempotent.
pub fn signal_weights_load_complete() {
    WEIGHTS_LOAD_COMPLETE.store(true, std::sync::atomic::Ordering::Release);
}

/// Last-resort cancel for the background weight-cache writer threads. `join_sidecar_writers` sets
/// it only when a build overruns the drain budget — a genuinely wedged
/// write (full disk / dead device). Writers poll it between chunks, drop
/// their temp file, and bail, so the one-time cache build can never hold
/// process exit hostage.
static CANCEL_CACHE_WRITE: AtomicBool = AtomicBool::new(false);

/// Cheap `Relaxed` read for the writer's copy loop.
fn cache_write_cancelled() -> bool {
    CANCEL_CACHE_WRITE.load(Ordering::Relaxed)
}

/// Registry of in-flight background sidecar writers. A short-lived
/// process (the `scr` CLI) MUST drain this before exit via
/// `join_sidecar_writers`: the writers are detached, so on a fast
/// `scr chat -q …` the process exits before the multi-GB write
/// finishes, the temp file is discarded, and the cache never persists —
/// so every launch re-pays the cold realign-copy (~1.7 s on M1 Max)
/// instead of taking the warm zero-copy path (~0.3 s). (regression: c54955a3)
static SIDECAR_WRITERS: Mutex<Vec<std::thread::JoinHandle<()>>> = Mutex::new(Vec::new());

/// Block until every pending aligned-sidecar writer has finished, so the
/// one-time cache build survives process exit. Silent no-op on the warm
/// path (nothing building). Prints a hysteresis notice only if the flush
/// actually takes a beat — a fast build stays quiet, so steady-state
/// launches show nothing. Idempotent; safe from any teardown path.
///
/// Two hazards this must not hit:
///   * Writers park up to 180 s on [`WEIGHTS_LOAD_COMPLETE`] (they wait so
///     the build never contends with load). By the time we drain, load is
///     over by definition — and a load that *fails* after some shards
///     spawned writers drops the worker straight into this join while the
///     latch is still unset, so every writer would burn its full 180 s
///     timeout here and the process looks wedged on "Completing initial
///     model caching…". Release the latch first.
///   * A genuinely stuck write (full disk / dead device) must never wedge
///     exit forever: after a generous budget, cancel and detach.
pub fn join_sidecar_writers() {
    // Release any writer still parked on the load-complete latch: at drain
    // time load is finished (or aborted), so waiting longer only stalls
    // exit. This is what turns the failed-load teardown from a 180 s stall
    // into an immediate return.
    signal_weights_load_complete();

    let handles: Vec<std::thread::JoinHandle<()>> = match SIDECAR_WRITERS.lock() {
        Ok(mut v) => std::mem::take(&mut *v),
        Err(_) => return,
    };
    if handles.is_empty() {
        return;
    }
    let done = Arc::new(AtomicBool::new(false));
    let watchdog = {
        let done = Arc::clone(&done);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if !done.load(Ordering::Acquire) {
                eprintln!(
                    "Completing initial model caching… (first run only; later launches load instantly)"
                );
            }
        })
    };

    // Generous budget: a healthy first-run build (~19 GiB for the largest
    // shard) finishes well under a minute even on a slow disk, so this
    // never fires on a good build — it only rescues exit from a wedged
    // write. `JoinHandle` has no timed join, so poll `is_finished`.
    const DRAIN_BUDGET_SECS: u64 = 300;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(DRAIN_BUDGET_SECS);
    while std::time::Instant::now() < deadline {
        if handles.iter().all(|h| h.is_finished()) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    if handles.iter().any(|h| !h.is_finished()) {
        // Overran the budget — cancel, give writers a beat to drop their
        // temp files, then detach whatever's still stuck in a syscall.
        CANCEL_CACHE_WRITE.store(true, Ordering::Release);
        eprintln!(
            "aligned-cache: build exceeded {DRAIN_BUDGET_SECS}s and was cancelled so exit \
             isn't blocked; it will rebuild on the next launch"
        );
        let hard = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < hard && handles.iter().any(|h| !h.is_finished()) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    // Reap the finished writers; drop (detach) any still stuck after the
    // cancel window rather than block on their join.
    for h in handles {
        if h.is_finished() {
            let _ = h.join();
        }
    }
    done.store(true, Ordering::Release);
    let _ = watchdog.join();
}

/// Bump when the packed layout rule changes (MIN_BIND_ALIGN, packing
/// order, …) — invalidates every existing sidecar.
const ALIGNED_CACHE_LAYOUT_VERSION: u32 = 1;

/// Identity + validity data for one shard's aligned sidecar.
#[derive(Clone, Debug)]
struct AlignedCacheMeta {
    src_size: u64,
    src_mtime_ns: u128,
    aligned_capacity: usize,
    /// `~/.cache/scratchy/metal-aligned-weights/<hash>-<shard>.bin`
    bin: std::path::PathBuf,
    /// The original (un-canonicalized) source weight path. Recorded only
    /// so tooling (`scr model cache inspect`) can label a sidecar with the
    /// model it came from — including local-path loads that can't be
    /// reverse-mapped from the HF cache. Not part of the validity check.
    src_path: String,
    /// `content_hash64` of the aligned blob, filled by the builder.
    content_hash: std::cell::Cell<u64>,
}

impl AlignedCacheMeta {
    fn cache_bin_path(&self) -> std::path::PathBuf {
        self.bin.clone()
    }
    fn meta_json_path(&self) -> std::path::PathBuf {
        self.bin.with_extension("meta.json")
    }
    fn to_json(&self) -> String {
        // `src_path` is JSON-escaped via serde (paths may contain quotes,
        // backslashes, unicode); the rest are numbers.
        let src_path = serde_json::to_string(&self.src_path).unwrap_or_else(|_| "\"\"".to_string());
        format!(
            "{{\"layout_version\":{},\"src_size\":{},\"src_mtime_ns\":{},\"aligned_capacity\":{},\"content_hash\":{},\"src_path\":{}}}",
            ALIGNED_CACHE_LAYOUT_VERSION,
            self.src_size,
            self.src_mtime_ns,
            self.aligned_capacity,
            self.content_hash.get(),
            src_path,
        )
    }
    /// The expected blob hash from the on-disk meta (None for metas
    /// written before the integrity bit existed — treated as invalid
    /// by `is_valid_on_disk`).
    fn disk_content_hash(&self) -> Option<u64> {
        let j: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(self.meta_json_path()).ok()?).ok()?;
        j.get("content_hash").and_then(|v| v.as_u64())
    }
    /// True iff bin + meta exist and match this source + layout.
    fn is_valid_on_disk(&self) -> bool {
        let Ok(meta_str) = std::fs::read_to_string(self.meta_json_path()) else {
            return false;
        };
        let Ok(j) = serde_json::from_str::<serde_json::Value>(&meta_str) else {
            return false;
        };
        let ok = j.get("layout_version").and_then(|v| v.as_u64())
            == Some(ALIGNED_CACHE_LAYOUT_VERSION as u64)
            && j.get("src_size").and_then(|v| v.as_u64()) == Some(self.src_size)
            && j.get("src_mtime_ns").and_then(|v| v.as_u128_lossy()) == Some(self.src_mtime_ns)
            && j.get("aligned_capacity").and_then(|v| v.as_u64())
                == Some(self.aligned_capacity as u64)
            && j.get("content_hash").and_then(|v| v.as_u64()).is_some();
        ok && std::fs::metadata(&self.bin)
            .map(|m| m.len() as usize == self.aligned_capacity)
            .unwrap_or(false)
    }
}

/// `as_u128` polyfill — serde_json numbers cap at u64/f64; mtime_ns is
/// stored as a JSON number that fits u64 in practice (year 2554).
trait U128Lossy {
    fn as_u128_lossy(&self) -> Option<u128>;
}
impl U128Lossy for serde_json::Value {
    fn as_u128_lossy(&self) -> Option<u128> {
        self.as_u64().map(|v| v as u128)
    }
}

/// Per-tensor record stored in the parent `MmapRegion`. Each tensor's
/// bytes live at `aligned_buffer.contents() + dst_offset` (the
/// `dst_offset` is 16-aligned by construction so kernel bindings
/// pass the strict alignment gate). `ready` is signalled by the
/// background loader thread that ran the `pread` for this tensor;
/// `alloc_and_copy_host` joins on it before handing the pointer out.
struct MmapTensor {
    /// Byte offset within the mmap (i.e. file) where the tensor
    /// starts. Used as the binary-search key when classifying an
    /// incoming `src` pointer.
    src_offset: usize,
    len: usize,
    /// 16-aligned offset within `aligned_buffer.contents()`.
    dst_offset: usize,
    ready: Arc<TensorReady>,
}

/// One-shot ready signal. Set by the background pread worker(s) for
/// the tensor; `take()` waits on it before returning the
/// destination pointer. Multiple chunks may share a single tensor's
/// `TensorReady` via `chunks_remaining`.
struct TensorReady {
    done: AtomicBool,
    chunks_remaining: AtomicUsize,
    waiter: Mutex<()>,
    cv: Condvar,
}

impl TensorReady {
    fn new(n_chunks: usize) -> Self {
        Self {
            done: AtomicBool::new(n_chunks == 0),
            chunks_remaining: AtomicUsize::new(n_chunks),
            waiter: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Mark one chunk as done. The last chunk to complete flips
    /// `done` and notifies any waiters.
    fn signal_chunk(&self) {
        if self.chunks_remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = self.waiter.lock().expect("TensorReady waiter mutex");
            self.done.store(true, Ordering::Release);
            self.cv.notify_all();
        }
    }

    fn wait(&self) {
        if self.done.load(Ordering::Acquire) {
            return;
        }
        let mut g = self.waiter.lock().expect("TensorReady waiter mutex");
        while !self.done.load(Ordering::Acquire) {
            g = self.cv.wait(g).expect("TensorReady cv");
        }
    }
}

struct MmapRegion {
    /// Original mmap base pointer + length. The mmap is kept alive
    /// for **pointer identity only** — callers compute
    /// `src = mmap.as_ptr() + data_offset` from the safetensors
    /// header parse and we look up which `MmapTensor` corresponds
    /// to that address. Tensor *data* pages of the mmap are never
    /// touched after construction; the bytes live in
    /// `aligned_buffer` after a background `pread` from the file.
    base: *const u8,
    len: usize,
    /// **Pre-aligned destination buffer.** One `MTLBuffer`
    /// (storageModeShared) sized to fit every tensor in the shard
    /// laid out at 16-aligned offsets. The CPU-mappable
    /// `.contents()` pointer is what the loader threads `pread`
    /// into. All `alloc_and_copy_host{,_aligned}` zero-copy returns
    /// and `buffer_for` lookups resolve into this buffer.
    aligned_buffer: Pinned,
    aligned_base: *mut u8,
    aligned_capacity: usize,
    /// Per-tensor records, sorted by `src_offset` for binary-search
    /// lookup from the `src` pointer the caller passes to
    /// `alloc_and_copy_host`.
    tensors: Vec<MmapTensor>,
    _mmap: Arc<memmap2::Mmap>,
    /// When the region is served from the aligned sidecar cache,
    /// `aligned_buffer` is a bytesNoCopy wrap of THIS mapping —
    /// keep it alive for the buffer's lifetime.
    _cache_mmap: Option<Arc<memmap2::Mmap>>,
}

unsafe impl Send for MmapRegion {}
unsafe impl Sync for MmapRegion {}

#[derive(Clone, Copy)]
enum MmapClassify {
    /// Source `src` lies within a registered mmap and its shifted
    /// offset (`mmap_offset + region.shift`) is `min_align`-aligned.
    /// `aligned_ptr` is the address inside the per-region pre-aligned
    /// destination buffer where the bulk-copied bytes live — i.e.
    /// what `alloc_and_copy_host{,_aligned}` returns to the caller.
    Aligned { aligned_ptr: *mut u8 },
    /// Source `src` is in a registered mmap but the shifted offset
    /// isn't `min_align`-aligned — falls through to the arena memcpy
    /// path.
    Unaligned,
    /// Source `src` is outside every registered mmap (e.g. the bytes
    /// were heap-allocated by `maybe_cast_cpu` after a CPU cast).
    Outside,
}

pub type ArenaHook = Arc<dyn Fn(&Buffer) + Send + Sync>;

pub struct MetalAllocator {
    device: Device,
    /// Scratch / activation arenas — the mutable per-forward working set.
    /// Inserted into the wired `residency` set.
    arenas: Arc<Mutex<Vec<MetalArena>>>,
    /// Weight-memcpy arenas: the fraction of weights that miss the zero-copy
    /// mmap path (unaligned / outside a registered mmap) and get copied by
    /// `alloc_and_copy_host`. Immutable once written, so — like the big
    /// mmap/copy weight buffers — inserted into `weights_residency` and left
    /// pageable by default. Kept in a separate pool from `arenas` because a
    /// shared arena buffer holds interleaved scratch + weight bytes and can't
    /// be un-wired per-allocation.
    weight_arenas: Arc<Mutex<Vec<MetalArena>>>,
    chunk_bytes: usize,
    on_new_arena: Arc<Mutex<Option<ArenaHook>>>,
    /// Wired set: activation arenas + KV cache. `requestResidency`'d, so
    /// the transient working set stays pinned across command buffers.
    residency: crate::residency::MetalResidencySet,
    /// Weights set. By default ([`WeightResidency::Unwired`]) a distinct
    /// *un-wired* set: weights are declared to each command buffer (MTL4
    /// requires it) but never `requestResidency`'d, so the OS can page the
    /// immutable, mmap-backed weight pages under memory pressure instead of
    /// OOMing. Under [`WeightResidency::Wired`] it is instead a CLONE of
    /// `residency` (weights pinned alongside the working set — legacy).
    weights_residency: crate::residency::MetalResidencySet,
    mmaps: Arc<Mutex<Vec<MmapRegion>>>,
    /// Diagnostic counters for `alloc_and_copy_host` routing. Bumped
    /// once per call so the worker can print a one-shot "zero-copy
    /// vs memcpy" breakdown after `try_load` completes. Atomics are
    /// `Relaxed` — these aren't synchronization, just stats.
    load_stats: Arc<LoadStats>,
}

/// Histogram of tensor-offset trailing-zero counts. Index `i` counts
/// tensors whose offset within their mmap has exactly `i` trailing
/// zero bits (i.e. is aligned to `2^i` but not `2^(i+1)`). Capped at
/// 16; anything ≥ 16 lands in bucket 16. Wired through `LoadStats`.
pub const ALIGNMENT_HISTOGRAM_BUCKETS: usize = 17;

#[derive(Default)]
pub struct LoadStats {
    /// `alloc_and_copy_host` call returned an mmap-aliased pointer
    /// (no copy, ≤ a few hundred ns).
    pub zero_copy_calls: AtomicU64,
    pub zero_copy_bytes: AtomicU64,
    /// Subset of `zero_copy_calls` that only succeeded because the
    /// caller passed `min_align < MIN_BIND_ALIGN` via
    /// `alloc_and_copy_host_aligned` (e.g. F16 scales/biases at
    /// `mod 4 == 2` taking the 2-byte-aligned fast path). Bumps
    /// only when the offset would have FAILED the strict 16-byte
    /// gate but PASSED the relaxed gate — pure visibility into how
    /// much the dtype-aware relaxation actually saves.
    pub zero_copy_relaxed_calls: AtomicU64,
    pub zero_copy_relaxed_bytes: AtomicU64,
    /// `alloc_and_copy_host` call fell through to arena memcpy
    /// (~5 GB/s on Apple Silicon). High-volume miss here is the
    /// startup-time bottleneck.
    pub memcpy_calls: AtomicU64,
    pub memcpy_bytes: AtomicU64,
    /// Per-prefix breakdown: keyed by a coarse category derived
    /// from the byte count (so we can tell scales/biases apart
    /// from packed weight blobs without threading prefix strings
    /// down to the allocator).
    pub memcpy_small_calls: AtomicU64, // < 1 MiB
    pub memcpy_med_calls: AtomicU64,   // 1 MiB ≤ ... < 16 MiB
    pub memcpy_large_calls: AtomicU64, // ≥ 16 MiB
    /// Why zero-copy failed: source pointer wasn't in any
    /// registered mmap region (e.g. tensor was already heap-copied
    /// upstream, or mmap was never registered).
    pub memcpy_outside_mmap: AtomicU64,
    /// Why zero-copy failed: pointer was inside a registered mmap,
    /// but the offset wasn't 16-byte aligned. This is the
    /// safetensors-data-section-base alignment problem.
    pub memcpy_unaligned: AtomicU64,
    /// Histogram of observed offset trailing-zero counts (0..=16).
    pub alignment_hist: [AtomicU64; ALIGNMENT_HISTOGRAM_BUCKETS],
}

impl LoadStats {
    fn observe_offset_alignment(&self, tz: u32) {
        let idx = (tz as usize).min(ALIGNMENT_HISTOGRAM_BUCKETS - 1);
        self.alignment_hist[idx].fetch_add(1, Ordering::Relaxed);
    }
}

unsafe impl Send for MetalAllocator {}
unsafe impl Sync for MetalAllocator {}

impl Clone for MetalAllocator {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            arenas: Arc::clone(&self.arenas),
            weight_arenas: Arc::clone(&self.weight_arenas),
            chunk_bytes: self.chunk_bytes,
            on_new_arena: Arc::clone(&self.on_new_arena),
            residency: self.residency.clone(),
            weights_residency: self.weights_residency.clone(),
            mmaps: Arc::clone(&self.mmaps),
            load_stats: Arc::clone(&self.load_stats),
        }
    }
}

/// Whether weight buffers are permanently wired or only declared resident
/// per command buffer.
///
/// [`WeightResidency::Unwired`] is the default: weights are immutable and
/// mmap-backed, so pinning multi-GB of them for the whole process lifetime
/// (`requestResidency`) is what drives Apple-Silicon wired-memory pressure
/// and the OOM / beachball on large models. Un-wired weights are still
/// declared to every command buffer via `useResidencySet:` — resident while
/// a forward executes, pageable between forwards — which satisfies MTL4's
/// explicit-residency requirement without pinning them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightResidency {
    /// Weights declared per-command-buffer but never `requestResidency`-
    /// pinned → reclaimable under memory pressure. The default.
    #[default]
    Unwired,
    /// Weights pinned alongside the working set (the historical behavior).
    /// Escape hatch for regression isolation / debugging.
    Wired,
}

/// Build the weights residency set for a device. [`WeightResidency::Unwired`]
/// yields a distinct set that is never `requestResidency`'d; [`Wired`] clones
/// the working-set residency so weights are pinned alongside it.
///
/// [`Wired`]: WeightResidency::Wired
fn make_weights_residency(
    device: &Device,
    wired: &crate::residency::MetalResidencySet,
    mode: WeightResidency,
) -> crate::residency::MetalResidencySet {
    match mode {
        WeightResidency::Unwired => crate::residency::MetalResidencySet::new_unwired(device),
        WeightResidency::Wired => wired.clone(),
    }
}

impl MetalAllocator {
    pub fn new(device: Device) -> Self {
        Self::build(device, DEFAULT_CHUNK_BYTES, WeightResidency::default())
    }

    pub fn with_chunk_bytes(device: Device, chunk_bytes: usize) -> Self {
        Self::build(device, chunk_bytes, WeightResidency::default())
    }

    /// Construct with an explicit weight-residency mode. Prefer [`Self::new`]
    /// (defaults to [`WeightResidency::Unwired`]); use this only to force
    /// [`WeightResidency::Wired`] for regression isolation.
    pub fn with_weight_residency(device: Device, mode: WeightResidency) -> Self {
        Self::build(device, DEFAULT_CHUNK_BYTES, mode)
    }

    fn build(device: Device, chunk_bytes: usize, weight_mode: WeightResidency) -> Self {
        let residency = crate::residency::MetalResidencySet::new(&device);
        let weights_residency = make_weights_residency(&device, &residency, weight_mode);
        Self {
            device,
            arenas: Arc::new(Mutex::new(Vec::new())),
            weight_arenas: Arc::new(Mutex::new(Vec::new())),
            chunk_bytes,
            on_new_arena: Arc::new(Mutex::new(None)),
            residency,
            weights_residency,
            mmaps: Arc::new(Mutex::new(Vec::new())),
            load_stats: Arc::new(LoadStats::default()),
        }
    }

    /// Snapshot the load-time routing counters. Caller is expected
    /// to log them once (after `try_load` completes) — the atomics
    /// are not reset.
    pub fn load_stats(&self) -> &LoadStats {
        &self.load_stats
    }

    pub fn residency(&self) -> &crate::residency::MetalResidencySet {
        &self.residency
    }

    /// The residency set covering weight buffers. A distinct un-wired set
    /// by default ([`WeightResidency::Unwired`]); equal to [`Self::residency`]
    /// under [`WeightResidency::Wired`]. Command buffers must declare this
    /// set (via `useResidencySet:`) in addition to [`Self::residency`] so
    /// weights are resident while a forward executes.
    pub fn weights_residency(&self) -> &crate::residency::MetalResidencySet {
        &self.weights_residency
    }

    /// Allocation breakdown for budget diagnostics: (mmap regions
    /// total bytes, arena capacity bytes, arena used bytes).
    pub fn allocation_breakdown(&self) -> (usize, usize, usize) {
        let regions: usize = self
            .mmaps
            .lock()
            .expect("MetalAllocator mmaps Mutex")
            .iter()
            .map(|r| r.aligned_capacity)
            .sum();
        // Both scratch and weight-memcpy arenas count toward the arena
        // budget, but they are NOT the same thing and a caller that
        // prints their sum invites the reading that weights were
        // copied when the copy was scratch. `weight_arena_bytes`
        // below splits them for exactly that reason.
        let (cap, used) = {
            let mut cap = 0usize;
            let mut used = 0usize;
            for pool in [&self.arenas, &self.weight_arenas] {
                let arenas = pool.lock().expect("MetalAllocator arenas Mutex");
                cap += arenas.iter().map(|a| a.capacity).sum::<usize>();
                used += arenas.iter().map(|a| a.used).sum::<usize>();
            }
            (cap, used)
        };
        (regions, cap, used)
    }

    /// `(scratch_cap, scratch_used, weight_cap, weight_used)` — the
    /// two arena pools SEPARATELY.
    ///
    /// [`Self::allocation_breakdown`] sums them, which reads as "the
    /// loader copied this much" when most of it can be the activation
    /// arena. On mixtral the summed figure said 23.63 GiB used while
    /// the loader's own routing counters said it memcpy'd 10 MiB; the
    /// number was not wrong, the LABEL was.
    pub fn arena_breakdown_by_pool(&self) -> (usize, usize, usize, usize) {
        let sum = |pool: &Arc<Mutex<Vec<MetalArena>>>| {
            let arenas = pool.lock().expect("MetalAllocator arenas Mutex");
            (
                arenas.iter().map(|a| a.capacity).sum::<usize>(),
                arenas.iter().map(|a| a.used).sum::<usize>(),
            )
        };
        let (sc, su) = sum(&self.arenas);
        let (wc, wu) = sum(&self.weight_arenas);
        (sc, su, wc, wu)
    }

    pub fn set_arena_hook(&self, hook: ArenaHook) {
        for pool in [&self.arenas, &self.weight_arenas] {
            let arenas = pool.lock().expect("MetalAllocator arenas Mutex");
            for arena in arenas.iter() {
                (hook)(&arena.buffer);
            }
        }
        *self.on_new_arena.lock().expect("arena hook mutex") = Some(hook);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn buffer_for(&self, ptr: *const u8) -> Option<(Buffer, u64)> {
        let p = ptr as usize;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            for region in mmaps.iter() {
                // Zero-copy returns from `alloc_and_copy_host_aligned`
                // point into the per-region pre-aligned MTLBuffer that
                // covers the packed-aligned tensor layout, valid range
                // `[aligned_base, aligned_base + aligned_capacity)`.
                let a_start = region.aligned_base as usize;
                let a_end = a_start + region.aligned_capacity;
                if p >= a_start && p < a_end {
                    return Some((region.aligned_buffer.clone(), (p - a_start) as u64));
                }
            }
        }
        // A returned pointer lives in either the scratch arenas (alloc_uninit)
        // or the weight-memcpy arenas (alloc_and_copy_host) — search both.
        for pool in [&self.arenas, &self.weight_arenas] {
            let arenas = pool.lock().expect("MetalAllocator arenas Mutex");
            for arena in arenas.iter() {
                let start = arena.base as usize;
                let end = start + arena.used;
                if p >= start && p < end {
                    return Some((arena.buffer.clone(), (p - start) as u64));
                }
            }
        }
        None
    }

    /// Parse a safetensors header (`[u64 header_size_le][JSON header]
    /// [data section]`) and return the list of tensors with their
    /// **file-relative** byte offsets and sizes. Returns `None` if
    /// the header doesn't parse — the caller treats that as an error
    /// for safetensors shards (the load path only feeds safetensors
    /// in).
    fn parse_safetensors_tensors(base: *const u8, len: usize) -> Option<Vec<(usize, usize)>> {
        if len < 8 {
            return None;
        }
        // SAFETY: `base` points to at least 8 mapped bytes.
        let header_size = unsafe { std::ptr::read_unaligned(base as *const u64).to_le() } as usize;
        if header_size == 0 || header_size > len.saturating_sub(8) {
            return None;
        }
        let header_bytes = unsafe { std::slice::from_raw_parts(base.wrapping_add(8), header_size) };
        let json: serde_json::Value = serde_json::from_slice(header_bytes).ok()?;
        let obj = json.as_object()?;
        let data_section_start = 8 + header_size;
        let mut tensors = Vec::with_capacity(obj.len());
        for (key, val) in obj {
            if key == "__metadata__" {
                continue;
            }
            let offs = val.get("data_offsets")?.as_array()?;
            let lo = offs.first()?.as_u64()? as usize;
            let hi = offs.get(1)?.as_u64()? as usize;
            if hi < lo {
                return None;
            }
            // Reject a tensor whose data extends past the mapped file: a
            // partial/interrupted download leaves a truncated shard whose
            // header still names the full (untruncated) offsets. Without
            // this, the copy loop reads past the mmap (SIGBUS) or bakes the
            // short blob into a "valid" — but corrupt — aligned cache.
            let abs_end = data_section_start.checked_add(hi)?;
            if abs_end > len {
                return None;
            }
            tensors.push((data_section_start + lo, hi - lo));
        }
        tensors.sort_by_key(|&(off, _)| off);
        Some(tensors)
    }

    /// Register a safetensors shard for zero-copy weight binding.
    ///
    /// Lays out every tensor at a 16-aligned `dst_offset` in a single
    /// shared-storage `MTLBuffer`, then dispatches `pread` tasks to
    /// the rayon global pool that read each tensor's bytes straight
    /// from disk into `buffer.contents() + dst_offset`. This function
    /// **returns before any tensor data has been read**: each
    /// per-tensor `TensorReady` is signalled by the background
    /// worker(s) once that tensor's bytes are in the destination
    /// buffer. `take()`-side callers join on the corresponding
    /// `TensorReady` before consuming the pointer.
    ///
    /// The mmap is retained only as a pointer-identity device: the
    /// caller's `CpuTensorRef` holds `mmap.as_ptr() + data_offset`
    /// values that we look up via the per-tensor `src_offset` table.
    /// **Tensor-data pages of the mmap are never faulted in** — the
    /// bytes come from `pread(fd, …)` straight into the destination.
    pub fn register_mmap(&self, path: &Path, mmap: Arc<memmap2::Mmap>) -> Result<()> {
        let base = mmap.as_ptr();
        let len = mmap.len();
        if len == 0 {
            return Ok(());
        }

        let tensors_src = Self::parse_safetensors_tensors(base, len).ok_or_else(|| {
            anyhow::anyhow!(
                "MetalAllocator::register_mmap: {} is not a complete safetensors shard \
                 (header failed to parse, or a tensor extends past the {len}-byte file — an \
                 interrupted or partial download leaves a truncated shard). Delete the file and \
                 re-download the model.",
                path.display()
            )
        })?;

        // Pack each tensor at the next 16-aligned offset.
        let mut packed: Vec<(usize, usize, usize)> = Vec::with_capacity(tensors_src.len());
        let mut running = 0usize;
        for &(src_off, sz) in &tensors_src {
            let dst_off = running.div_ceil(Self::MIN_BIND_ALIGN) * Self::MIN_BIND_ALIGN;
            packed.push((src_off, sz, dst_off));
            running = dst_off + sz;
        }
        let aligned_capacity = running
            .div_ceil(Self::MIN_BIND_ALIGN)
            .saturating_mul(Self::MIN_BIND_ALIGN)
            .max(Self::MIN_BIND_ALIGN);

        // ── Aligned sidecar cache (zero-copy relaunch) ────────────────
        //
        // The realign-copy below is a pure function of (source file,
        // packing layout). Persist its output once, then on later
        // launches mmap the cached aligned blob and wrap it as a
        // bytesNoCopy MTLBuffer — no copy, single memory copy total
        // (the file pages ARE the buffer; load-time footprint halves).
        // Measured on Qwen3.5-35B (18.99 GiB): warm relaunch pays only
        // the residency wiring (~3.3-3.7 s) instead of the ~8 s copy.
        //
        // Validity: source (size, mtime_ns) + layout version +
        // aligned_capacity, stored in a sidecar .meta.json.
        //
        let cache_enabled = true;
        let meta = Self::aligned_cache_meta(path, aligned_capacity);
        let cache_file = meta
            .as_ref()
            .filter(|_| cache_enabled)
            .map(|m| m.cache_bin_path());
        if let (Some(cf), Some(m)) = (cache_file.as_ref(), meta.as_ref())
            && m.is_valid_on_disk()
        {
            match Self::register_from_aligned_cache(
                self,
                cf,
                base,
                len,
                aligned_capacity,
                &packed,
                mmap,
                m.disk_content_hash(),
            ) {
                Ok(()) => {
                    tracing::info!(
                        "aligned-cache HIT: {} served zero-copy from {}",
                        path.display(),
                        cf.display()
                    );
                    return Ok(());
                }
                Err((e, mmap_back)) => {
                    tracing::warn!(
                        "aligned-cache: rejected {} ({e}); deleting and falling back to copy",
                        cf.display()
                    );
                    // Self-heal: a cache that fails to map or fails the
                    // integrity check must not be retried forever.
                    if let Some(m) = meta.as_ref() {
                        let _ = std::fs::remove_file(m.meta_json_path());
                    }
                    let _ = std::fs::remove_file(cf);
                    return self.register_mmap_copy_path(
                        path,
                        mmap_back,
                        base,
                        len,
                        packed,
                        aligned_capacity,
                        meta,
                        cache_enabled,
                    );
                }
            }
        }
        self.register_mmap_copy_path(
            path,
            mmap,
            base,
            len,
            packed,
            aligned_capacity,
            meta,
            cache_enabled,
        )
    }

    /// Compute the sidecar identity for `path`, or None when source
    /// metadata is unavailable. mtime_ns truncates to u64 range (fine
    /// until year 2554); cache dir is created lazily by the writer.
    fn aligned_cache_meta(path: &Path, aligned_capacity: usize) -> Option<AlignedCacheMeta> {
        let md = std::fs::metadata(path).ok()?;
        let mtime = md.modified().ok()?;
        let src_mtime_ns =
            mtime.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos() & (u64::MAX as u128);
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        canon.hash(&mut h);
        let dir = scratchy_core_common::cache::metal_aligned_weights_dir();
        let stem = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| "shard".into());
        let bin = dir.join(format!("{:016x}-{stem}.bin", h.finish()));
        Some(AlignedCacheMeta {
            src_size: md.len(),
            src_mtime_ns,
            aligned_capacity,
            bin,
            src_path: path.to_string_lossy().into_owned(),
            content_hash: std::cell::Cell::new(0),
        })
    }

    /// Cache-hit path: mmap the aligned sidecar and wrap it as a
    /// bytesNoCopy MTLBuffer — no copy; the file pages are the
    /// buffer. On failure returns the source mmap back so the caller
    /// can fall through to the copy path.
    #[allow(clippy::result_large_err)]
    fn register_from_aligned_cache(
        &self,
        cache_bin: &Path,
        base: *const u8,
        len: usize,
        aligned_capacity: usize,
        packed: &[(usize, usize, usize)],
        mmap: Arc<memmap2::Mmap>,
        expected_hash: Option<u64>,
    ) -> std::result::Result<(), (anyhow::Error, Arc<memmap2::Mmap>)> {
        let file = match std::fs::File::open(cache_bin) {
            Ok(f) => f,
            Err(e) => return Err((e.into(), mmap)),
        };
        let cache_mmap = match unsafe { memmap2::Mmap::map(&file) } {
            Ok(m) => Arc::new(m),
            Err(e) => return Err((e.into(), mmap)),
        };
        if cache_mmap.len() != aligned_capacity {
            return Err((
                anyhow::anyhow!(
                    "sidecar size {} != expected aligned_capacity {}",
                    cache_mmap.len(),
                    aligned_capacity
                ),
                mmap,
            ));
        }
        let page: usize = 16384;
        let rounded = aligned_capacity.div_ceil(page) * page;
        let ptr = cache_mmap.as_ptr() as *mut std::ffi::c_void;
        let Some(nn) = std::ptr::NonNull::new(ptr) else {
            return Err((anyhow::anyhow!("null mmap base"), mmap));
        };
        let Some(buffer) = (unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    nn,
                    rounded,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }) else {
            return Err((
                anyhow::anyhow!("newBufferWithBytesNoCopy returned nil"),
                mmap,
            ));
        };
        // Weight buffer (mmap cache-hit, bytesNoCopy). Goes in the weights
        // set so it is left pageable (default WeightResidency::Unwired).
        // Commit here so the add takes effect on a distinct un-wired set
        // (the wired set is also committed per-forward in the pool; a
        // double-commit when the two alias is harmless). Weight loading is
        // one-time at startup, before any forward's command buffer exists.
        let buffer = self.weights_residency.pin(buffer);
        self.weights_residency.commit();

        let tensors: Vec<MmapTensor> = packed
            .iter()
            .map(|&(src_off, sz, dst_off)| MmapTensor {
                src_offset: src_off,
                len: sz,
                dst_offset: dst_off,
                // Bytes are already on disk — every tensor is ready.
                ready: Arc::new(TensorReady::new(0)),
            })
            .collect();

        // ALWAYS-ON integrity bit: spot-check the head window (4 KiB)
        // of every tensor against the SOURCE bytes before serving from
        // the sidecar (~8 MiB of scattered source reads). Catches torn
        // writes, truncation surviving the size check, and
        // wrong-file/bit-rot with high probability; mismatch is
        // self-healing — the caller falls back to the copy path and
        // deletes the bad cache.
        {
            let src_base = base as usize;
            let dst_base = cache_mmap.as_ptr() as usize;
            for (i, t) in tensors.iter().enumerate() {
                let w = t.len.min(4096);
                if w == 0 {
                    continue;
                }
                let a = unsafe {
                    std::slice::from_raw_parts((src_base + t.src_offset) as *const u8, w)
                };
                let b = unsafe {
                    std::slice::from_raw_parts((dst_base + t.dst_offset) as *const u8, w)
                };
                if a != b {
                    return Err((
                        anyhow::anyhow!(
                            "sidecar integrity check failed at tensor #{i} \
                             (src_off={}, dst_off={}) — torn or stale cache",
                            t.src_offset,
                            t.dst_offset
                        ),
                        mmap,
                    ));
                }
            }
        }

        // Integrity bit (full coverage): the head-window spot-check
        // above misses corruption away from tensor heads (proven by a
        // byte-flip test at +2.5 GB). Verify the FULL blob against the
        // build-time content hash in the BACKGROUND — zero startup
        // cost; runs after the load-complete latch so it never
        // contends with launch. On mismatch the process ABORTS loudly
        // (it is already serving from these pages — continuing means
        // silently corrupt weights, this week's nightmare class) and
        // deletes the cache so relaunch self-heals via the copy path.
        if let Some(expected) = expected_hash {
            // Hold an Arc clone of the sidecar mmap for the lifetime of
            // this detached hash. It outlives the caller's teardown on a
            // short-lived run (e.g. `scr chat -q …` finishes inference,
            // prints, and unwinds while the latch has already fired).
            // Capturing only a raw `usize` let `_cache_mmap` (the sole
            // owner) drop and munmap mid-hash during teardown — the hash
            // loop then read unmapped pages → EXC_BAD_ACCESS, a ~50%
            // teardown segfault. The Arc defers the munmap until the hash
            // returns; it never blocks exit. (regression: c54955a3)
            let hash_mmap = Arc::clone(&cache_mmap);
            let hash_len = aligned_capacity;
            let bin = cache_bin.to_path_buf();
            let meta_json = bin.with_extension("meta.json");
            std::thread::spawn(move || {
                let t_wait = std::time::Instant::now();
                while !WEIGHTS_LOAD_COMPLETE.load(std::sync::atomic::Ordering::Acquire)
                    && t_wait.elapsed() < std::time::Duration::from_secs(180)
                {
                    std::thread::sleep(std::time::Duration::from_millis(250));
                }
                let t0 = std::time::Instant::now();
                let got = content_hash64(hash_mmap.as_ptr(), hash_len);
                if got != expected {
                    let _ = std::fs::remove_file(&meta_json);
                    let _ = std::fs::remove_file(&bin);
                    eprintln!(
                        "FATAL: aligned-cache integrity verification FAILED for {} \
                         (content_hash {got:#x} != recorded {expected:#x}). The cache has \
                         been deleted; relaunch will rebuild it from the checkpoint. \
                         Aborting rather than serve corrupt weights.",
                        bin.display()
                    );
                    std::process::abort();
                }
                tracing::info!(
                    "aligned-cache: background integrity verify OK for {} in {:?}",
                    bin.display(),
                    t0.elapsed()
                );
            });
        }

        let aligned_base = cache_mmap.as_ptr() as *mut u8;
        self.mmaps
            .lock()
            .expect("MetalAllocator mmaps Mutex")
            .push(MmapRegion {
                base,
                len,
                aligned_buffer: buffer,
                aligned_base,
                aligned_capacity,
                tensors,
                _mmap: mmap,
                _cache_mmap: Some(cache_mmap),
            });
        Ok(())
    }

    /// Background sidecar writer: temp file + atomic rename, meta json
    /// last (a crash leaves an invalid/incomplete cache that the
    /// validity check rejects). Weights are immutable after the load
    /// copy completes. `keepalive` retains the source `dst_buffer` for
    /// the lifetime of this detached write: on a short run the caller
    /// can tear down `self.mmaps` (the buffer's other owner) while the
    /// write is still in flight — the guard defers that deallocation
    /// until the write returns, instead of reading freed pages. It
    /// never blocks exit. (regression: c54955a3)
    fn spawn_aligned_cache_writer(
        cache_bin: std::path::PathBuf,
        aligned_base: usize,
        len: usize,
        meta: AlignedCacheMeta,
        keepalive: BufKeepAlive,
    ) {
        let handle = std::thread::spawn(move || {
            let _keepalive = keepalive;
            // Stay out of the load's way: wait for warmup (bounded so
            // short-lived tools still build their cache eventually). Bail
            // early if teardown cancelled us mid-wait.
            let t_wait = std::time::Instant::now();
            while !WEIGHTS_LOAD_COMPLETE.load(std::sync::atomic::Ordering::Acquire)
                && t_wait.elapsed() < std::time::Duration::from_secs(180)
            {
                if cache_write_cancelled() {
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            if cache_write_cancelled() {
                return;
            }
            let t0 = std::time::Instant::now();
            if let Some(dir) = cache_bin.parent()
                && let Err(e) = std::fs::create_dir_all(dir)
            {
                tracing::warn!("aligned-cache: create_dir_all failed: {e}");
                return;
            }
            // Single-builder exclusion across PROCESSES: flock on a
            // sidecar lock file. The kernel releases the lock on any
            // process death (including SIGKILL), so there is no stale-
            // lock protocol. A second `scr` launched in parallel
            // skips the build — the winner produces the cache, and the
            // loser has already loaded via the copy path anyway
            // (waiting would be strictly slower than the copy it
            // already did).
            let lock_path = cache_bin.with_extension("lock");
            let Ok(lock_file) = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)
            else {
                tracing::warn!("aligned-cache: cannot open lock file; skipping build");
                return;
            };
            {
                use std::os::fd::AsRawFd;
                let rc =
                    unsafe { libc::flock(lock_file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                if rc != 0 {
                    tracing::info!(
                        "aligned-cache: another process is building {}; skipping",
                        cache_bin.display()
                    );
                    return;
                }
            }
            // Holding the lock: re-check (the other process may have
            // finished the build while we waited on the latch) and
            // sweep any orphaned tmp from a killed builder.
            if meta.is_valid_on_disk() {
                tracing::info!(
                    "aligned-cache: {} already built; skipping",
                    cache_bin.display()
                );
                return;
            }
            // Pid-suffixed tmp: even if exclusion is ever bypassed,
            // two builders can't truncate each other's stream; rename
            // is atomic and both write identical bytes.
            let tmp = cache_bin.with_extension(format!("tmp.{}", std::process::id()));
            if let Some(dir) = cache_bin.parent()
                && let Some(stem) = cache_bin
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                && let Ok(rd) = std::fs::read_dir(dir)
            {
                let prefix = stem.strip_suffix(".bin").unwrap_or(&stem);
                for e in rd.flatten() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if name.starts_with(prefix) && name.contains(".tmp") {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
            let write = || -> std::io::Result<()> {
                use std::io::Write;
                let file = std::fs::File::create(&tmp)?;
                // F_NOCACHE: don't dirty ~19 GiB of page cache for a
                // write-once blob — keeps the source/weight pages (and
                // everything else) resident.
                #[cfg(target_os = "macos")]
                unsafe {
                    use std::os::fd::AsRawFd;
                    libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1);
                }
                let mut f = std::io::BufWriter::with_capacity(8 << 20, file);
                const CHUNK: usize = 64 << 20;
                let mut off = 0usize;
                while off < len {
                    // Teardown cancelled the build (overran the drain
                    // budget): stop now — the caller removes the temp file.
                    if cache_write_cancelled() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "weight-cache write cancelled during teardown",
                        ));
                    }
                    let n = (len - off).min(CHUNK);
                    let s =
                        unsafe { std::slice::from_raw_parts((aligned_base + off) as *const u8, n) };
                    f.write_all(s)?;
                    off += n;
                }
                f.flush()?;
                f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
                Ok(())
            };
            if let Err(e) = write() {
                tracing::warn!("aligned-cache: build failed ({e}); removing temp");
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            // Integrity bit: hash the blob we just wrote (from the
            // in-memory buffer — RAM-bandwidth, no re-read) and stamp
            // it into the meta. Verified in the background after every
            // cache-hit launch.
            meta.content_hash
                .set(content_hash64(aligned_base as *const u8, len));
            if let Err(e) = std::fs::rename(&tmp, &cache_bin) {
                tracing::warn!("aligned-cache: rename failed: {e}");
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            let meta_tmp = meta
                .meta_json_path()
                .with_extension(format!("json.tmp.{}", std::process::id()));
            if let Err(e) = std::fs::write(&meta_tmp, meta.to_json())
                .and_then(|()| std::fs::rename(&meta_tmp, meta.meta_json_path()))
            {
                tracing::warn!("aligned-cache: meta write failed: {e}");
                let _ = std::fs::remove_file(&meta_tmp);
                return;
            }
            tracing::info!(
                "aligned-cache: built {} ({:.2} GiB) in {:?} — next launch loads zero-copy",
                cache_bin.display(),
                len as f64 / (1 << 30) as f64,
                t0.elapsed()
            );
        });
        // Register so a short-lived process can drain the build before
        // exit (see join_sidecar_writers); otherwise the cache never
        // persists and every launch re-pays the cold copy.
        if let Ok(mut writers) = SIDECAR_WRITERS.lock() {
            writers.push(handle);
        }
    }

    /// The pre-sidecar `register_mmap` tail: allocate the aligned
    /// buffer, copy every tensor into it (page-fault bound; WILLNEED
    /// hinted), then optionally build the sidecar in the background.
    #[allow(clippy::too_many_arguments)]
    fn register_mmap_copy_path(
        &self,
        path: &Path,
        mmap: Arc<memmap2::Mmap>,
        base: *const u8,
        len: usize,
        packed: Vec<(usize, usize, usize)>,
        aligned_capacity: usize,
        meta: Option<AlignedCacheMeta>,
        cache_enabled: bool,
    ) -> Result<()> {
        let dst_buffer = self
            .device
            .newBufferWithLength_options(aligned_capacity, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MetalAllocator::register_mmap: newBufferWithLength({} bytes) returned nil",
                    aligned_capacity
                )
            })?;
        let aligned_base = dst_buffer.contents().as_ptr() as *mut u8;
        anyhow::ensure!(
            !aligned_base.is_null(),
            "MetalAllocator::register_mmap: destination buffer.contents() is null"
        );
        // Validate that the file exists / is openable; per-task we
        // re-open by path so each worker has its own fd (concurrent
        // `pread` on a shared fd on macOS appears to interleave reads
        // in practice — verified by bisect against sync-chunked).
        std::fs::File::open(path)
            .with_context(|| format!("MetalAllocator::register_mmap: open {}", path.display()))?;

        // Wrap the aligned destination base as a Send/Sync usize for
        // closure capture. Each worker writes into a disjoint chunk
        // of the buffer; no two workers race on the same byte.
        let dst_base_usize = aligned_base as usize;

        // Chunk size targets ~16 MiB per copy so large tensors
        // parallelize across rayon workers and small tensors remain
        // a single dispatch.
        const READ_CHUNK: usize = 16 * 1024 * 1024;

        // mmap source base — usize for Send across rayon closures. The
        // backing `Arc<Mmap>` is held alive in the `MmapRegion` below
        // for the lifetime of every captured pointer.
        let src_base_usize = base as usize;

        // Build per-tensor entries + a flat work-list of chunk copies.
        //
        // Userspace memcpy from the mmap'd safetensors region into the
        // StorageModeShared MTLBuffer.
        //
        // Why not `pread`: on macOS, `pread` into the `contents()` of a
        // large StorageModeShared MTLBuffer fails with EFAULT — the
        // kernel can't DMA into GPU-mapped pages. Empirically this hits
        // any shard above ~2 GiB; the standalone
        // `large_buffer_offset_probe_test` confirmed userspace memcpy
        // into the same buffer at 4.68 GiB offset DOES work.
        //
        // Why PARALLEL: the copy is page-fault bound — every source
        // page is a cold file page the kernel reads from NVMe on
        // first touch. A single thread faults sequentially at queue
        // depth ~1 and measured 1.33 GiB/s (13.9 s for the 18.6 GiB
        // Qwen3.5-35B). Scoped worker threads fault disjoint 16 MiB
        // chunks concurrently, restoring NVMe queue depth. The old
        // `rayon::spawn` version was removed over a 'static-capture
        // lifetime puzzle; `std::thread::scope` borrows instead, so
        // the buffer pointer never needs 'static.
        //
        // SAFETY: every chunk's dst is a disjoint range inside the
        // freshly-allocated destination MTLBuffer (alive in this
        // scope, moved into MmapRegion below); src points into the
        // mmap'd file region (Arc'd into MmapRegion's `_mmap`). No
        // two chunks overlap.
        let mut tensors: Vec<MmapTensor> = Vec::with_capacity(packed.len());
        // (src_off, dst_off, len, per-tensor ready handle)
        let mut chunk_jobs: Vec<(usize, usize, usize, Arc<TensorReady>)> = Vec::new();
        for (src_off, sz, dst_off) in packed {
            let n_chunks = if sz == 0 { 0 } else { sz.div_ceil(READ_CHUNK) };
            let ready = Arc::new(TensorReady::new(n_chunks));
            for chunk_idx in 0..n_chunks {
                let off_in_tensor = chunk_idx * READ_CHUNK;
                let chunk_sz = (sz - off_in_tensor).min(READ_CHUNK);
                chunk_jobs.push((
                    src_off + off_in_tensor,
                    dst_off + off_in_tensor,
                    chunk_sz,
                    Arc::clone(&ready),
                ));
            }
            tensors.push(MmapTensor {
                src_offset: src_off,
                len: sz,
                dst_offset: dst_off,
                ready,
            });
        }
        {
            // Async readahead for the whole source mapping before the
            // copy loop touches it. Measured on Qwen3.5-35B (18.6 GiB,
            // 4 shards): baseline sequential copy 13.9 s; with
            // MADV_WILLNEED 7.4 s — the kernel streams file pages in
            // ahead of the copier instead of demand-faulting at queue
            // depth 1. (`GpuWeights::from_dir` issues the same hint
            // per shard at mmap time, so later shards prefetch while
            // earlier ones copy.)
            unsafe {
                libc::madvise(base as *mut libc::c_void, len, libc::MADV_WILLNEED);
            }
            // 1 worker: the copy is page-fault bound and macOS
            // serializes fault handling on the VM object lock — extra
            // threads measured NEUTRAL-to-WORSE (2w: 27 s, 4w: 22 s,
            // 8w: 14.5 s vs 1w+WILLNEED: 7.4 s, 4w+WILLNEED: 9.7 s).
            let n_workers = 1;
            let next = std::sync::atomic::AtomicUsize::new(0);
            let jobs = &chunk_jobs;
            std::thread::scope(|scope| {
                for _ in 0..n_workers {
                    scope.spawn(|| {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some((chunk_src, chunk_dst, chunk_sz, ready)) = jobs.get(i) else {
                                break;
                            };
                            let dst = (dst_base_usize + chunk_dst) as *mut u8;
                            let src = (src_base_usize + chunk_src) as *const u8;
                            unsafe {
                                std::ptr::copy_nonoverlapping(src, dst, *chunk_sz);
                            }
                            ready.signal_chunk();
                        }
                    });
                }
            });
        }

        // Weight buffer (aligned copy path). Weights set → pageable by
        // default (WeightResidency::Unwired). Commit so a distinct un-wired
        // set registers the add (see the cache-hit path above).
        let dst_buffer = self.weights_residency.pin(dst_buffer);
        self.weights_residency.commit();

        let cache_path = if cache_enabled {
            meta.as_ref().map(|m| m.cache_bin_path())
        } else {
            None
        };
        let aligned_base_for_writer = aligned_base as usize;
        // Clone before the buffer is moved into the MmapRegion: the
        // writer thread holds this to keep the allocation alive past a
        // short-run teardown (see spawn_aligned_cache_writer).
        let writer_keepalive = BufKeepAlive(dst_buffer.clone());
        self.mmaps
            .lock()
            .expect("MetalAllocator mmaps Mutex")
            .push(MmapRegion {
                base,
                len,
                aligned_buffer: dst_buffer,
                aligned_base,
                aligned_capacity,
                tensors,
                _mmap: mmap,
                _cache_mmap: None,
            });

        // Build the sidecar in the background so the NEXT launch takes
        // the zero-copy path. Weights are immutable post-copy, so the
        // writer reads a stable buffer.
        if let (Some(cp), Some(m)) = (cache_path, meta) {
            Self::spawn_aligned_cache_writer(
                cp,
                aligned_base_for_writer,
                aligned_capacity,
                m,
                writer_keepalive,
            );
        } else {
            drop(writer_keepalive);
        }
        Ok(())
    }

    /// Largest alignment the qmv / qmm_t / NAX kernels demand on a
    /// buffer offset bound via `setBuffer:offset:atIndex:`. The packed
    /// int4 weight binding is declared `device const uint32_t*`
    /// (`shaders/quantized_qmv.metal:829`, `quantized_qmm.metal:388`),
    /// which requires 4-byte aligned offsets — and Apple's M-series
    /// driver does NOT silently tolerate misalignment (see
    /// `tests/quantized_qmv_test.rs::affine_qmv_fast_b4_bf16_unaligned_packed_offset_1_byte_prefix`,
    /// which reproduces the live divergence by binding at offset
    /// %4 = 1 and gets `worst abs_err = 19.07` vs an allowed 0.39).
    /// 16 covers u32 + simdgroup_float4 + any future SIMD-wide types,
    /// and is cheap enough to gate the mmap-alias short-circuit on.
    const MIN_BIND_ALIGN: usize = 16;

    /// `Some(aligned_ptr)` if `src` matches the start of a tensor in
    /// a registered mmap. The returned pointer points into the
    /// per-region pre-aligned destination buffer at the tensor's
    /// 16-aligned `dst_offset`; blocks (per-tensor `Condvar`) until
    /// the background `pread` for that tensor has completed.
    ///
    /// `min_align` is honored for visibility only — `dst_offset` is
    /// always 16-aligned by construction, so any sane request passes
    /// (the diagnostic counters still observe whether the relaxation
    /// vs strict gate would have mattered).
    ///
    /// `None` if `src` is outside every registered mmap, or doesn't
    /// match a tensor head (e.g. a CPU-cast scratch buffer).
    fn aligned_mmap_offset(
        &self,
        src: *const u8,
        bytes: usize,
        min_align: usize,
    ) -> Option<*mut u8> {
        let p = src as usize;
        let end = p.saturating_add(bytes);
        let ready;
        let aligned_ptr;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            let region = mmaps
                .iter()
                .find(|r| p >= r.base as usize && end <= r.base as usize + r.len)?;
            let mmap_offset = p - region.base as usize;
            let tensor = match region
                .tensors
                .binary_search_by_key(&mmap_offset, |t| t.src_offset)
            {
                Ok(idx) => &region.tensors[idx],
                Err(_) => return None,
            };
            if tensor.len != bytes {
                return None;
            }
            if min_align != 0 && tensor.dst_offset % min_align != 0 {
                return None;
            }
            aligned_ptr = unsafe { region.aligned_base.add(tensor.dst_offset) };
            ready = Arc::clone(&tensor.ready);
        }
        ready.wait();
        Some(aligned_ptr)
    }

    /// As [`aligned_mmap_offset`] but also bumps the histogram /
    /// classification counters and returns `Unaligned` vs `Outside`
    /// distinctly for the diagnostic log.
    fn classify_mmap_offset(&self, src: *const u8, bytes: usize, min_align: usize) -> MmapClassify {
        let p = src as usize;
        let end = p.saturating_add(bytes);
        let ready;
        let aligned_ptr;
        {
            let mmaps = self.mmaps.lock().expect("MetalAllocator mmaps Mutex");
            let region = match mmaps
                .iter()
                .find(|r| p >= r.base as usize && end <= r.base as usize + r.len)
            {
                Some(r) => r,
                None => return MmapClassify::Outside,
            };
            let mmap_offset = p - region.base as usize;
            // Histogram still buckets by the raw intra-mmap offset:
            // it characterizes the safetensors file layout, not our
            // routing. With packed-aligned dst, every match becomes
            // `Aligned` regardless of the source's trailing zeros —
            // the histogram just documents that file-level fact.
            let tz = mmap_offset.trailing_zeros();
            self.load_stats.observe_offset_alignment(tz);
            let tensor = match region
                .tensors
                .binary_search_by_key(&mmap_offset, |t| t.src_offset)
            {
                Ok(idx) => &region.tensors[idx],
                Err(_) => return MmapClassify::Outside,
            };
            if tensor.len != bytes {
                return MmapClassify::Outside;
            }
            if min_align != 0 && tensor.dst_offset % min_align != 0 {
                return MmapClassify::Unaligned;
            }
            aligned_ptr = unsafe { region.aligned_base.add(tensor.dst_offset) };
            ready = Arc::clone(&tensor.ready);
        }
        ready.wait();
        MmapClassify::Aligned { aligned_ptr }
    }

    pub fn arena_count(&self) -> usize {
        self.arenas
            .lock()
            .expect("MetalAllocator arenas Mutex")
            .len()
            + self
                .weight_arenas
                .lock()
                .expect("MetalAllocator weight_arenas Mutex")
                .len()
    }

    pub fn used_bytes(&self) -> usize {
        let mut used = 0usize;
        for pool in [&self.arenas, &self.weight_arenas] {
            used += pool
                .lock()
                .expect("MetalAllocator arenas Mutex")
                .iter()
                .map(|a| a.used)
                .sum::<usize>();
        }
        used
    }

    fn push_arena_locked(
        device: &Device,
        arenas: &mut Vec<MetalArena>,
        chunk_bytes: usize,
        min_bytes: usize,
        hook: &Arc<Mutex<Option<ArenaHook>>>,
        residency: &crate::residency::MetalResidencySet,
    ) -> Result<usize> {
        let capacity = min_bytes.max(chunk_bytes);
        let buffer = device
            .newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "MetalAllocator: newBufferWithLength_options({} bytes) returned nil",
                    capacity
                )
            })?;
        let base = buffer.contents().as_ptr() as *mut u8;
        if base.is_null() {
            anyhow::bail!(
                "MetalAllocator: buffer contents() returned null pointer (capacity={})",
                capacity
            );
        }
        let buffer = residency.pin(buffer);
        if let Some(cb) = hook.lock().expect("arena hook mutex").as_ref() {
            (cb)(&buffer);
        }
        arenas.push(MetalArena {
            buffer,
            base,
            capacity,
            used: 0,
        });
        Ok(arenas.len() - 1)
    }

    /// Uninitialised space in the WEIGHT pool — the twin of
    /// [`Self::alloc_uninit`] for immutable model data.
    ///
    /// The two pools are not interchangeable. `arenas` joins the WIRED
    /// residency set (pinned across command buffers, which is what the
    /// per-forward working set needs); `weight_arenas` joins
    /// `weights_residency`, which under the default
    /// `WeightResidency::Unwired` is deliberately NOT wired so the OS
    /// can page immutable weight pages instead of OOMing the process.
    ///
    /// MoE expert stacking used `alloc_uninit`, so every MoE model put
    /// its experts in the wired pool AND counted them as activation
    /// arena — mixtral reported 23.62 GiB of "scratch" that was
    /// entirely experts.
    pub fn alloc_uninit_weights(&self, bytes: usize) -> Result<*mut u8> {
        // POOL changes, residency does NOT: experts join the weight
        // pool so the budget attributes them correctly, but stay in
        // the same residency set they were already in. Moving them to
        // `weights_residency` as well costs ~3% TTFT on
        // gemma-4-26b-a4b (181.5 -> 187.2 ms median, ITL unchanged),
        // measured before/after with three and two samples — a real
        // regression against PLAN's perf gate, and a separate
        // decision from fixing the accounting.
        self.alloc_uninit_in(&self.weight_arenas, &self.residency, bytes)
    }

    pub fn alloc_uninit(&self, bytes: usize) -> Result<*mut u8> {
        self.alloc_uninit_in(&self.arenas, &self.residency, bytes)
    }

    /// Shared body: identical bump-allocation, different pool and
    /// residency set. Written once so the two cannot drift in their
    /// alignment or arena-growth rules.
    fn alloc_uninit_in(
        &self,
        pool: &Arc<Mutex<Vec<MetalArena>>>,
        residency: &crate::residency::MetalResidencySet,
        bytes: usize,
    ) -> Result<*mut u8> {
        let mut arenas = pool.lock().expect("MetalAllocator arenas Mutex");
        if bytes == 0 {
            if arenas.is_empty() {
                Self::push_arena_locked(
                    &self.device,
                    &mut arenas,
                    self.chunk_bytes,
                    0,
                    &self.on_new_arena,
                    residency,
                )?;
            }
            return Ok(arenas[0].base);
        }
        // Pad `used` up to MIN_BIND_ALIGN so every subsequent
        // allocation lands at a properly-aligned offset. Without this,
        // a small tensor whose size isn't a multiple of 16 (e.g. an
        // F16 vector with an odd element count) would shift every
        // following allocation off-alignment, and an int4 U32 weight
        // bound there would hit the same unaligned-binding UB we
        // gate against on the mmap-alias path. Wastes at most 15
        // bytes per allocation; negligible against tensor sizes.
        let aligned_bytes = bytes.div_ceil(Self::MIN_BIND_ALIGN) * Self::MIN_BIND_ALIGN;
        let idx = if let Some(idx) = arenas
            .iter()
            .rposition(|a| a.capacity - a.used >= aligned_bytes)
        {
            idx
        } else {
            Self::push_arena_locked(
                &self.device,
                &mut arenas,
                self.chunk_bytes,
                aligned_bytes,
                &self.on_new_arena,
                residency,
            )?
        };
        let arena = &mut arenas[idx];
        let offset = arena.used;
        debug_assert_eq!(
            offset % Self::MIN_BIND_ALIGN,
            0,
            "arena.used is not {}-aligned on entry; previous alloc didn't pad",
            Self::MIN_BIND_ALIGN
        );
        let dst = unsafe { arena.base.add(offset) };
        arena.used = offset + aligned_bytes;
        Ok(dst)
    }
}

impl DeviceAllocator for MetalAllocator {
    type Stream = ();
    type Mem = crate::MetalMem;

    fn stream(&self) {}

    fn register_mmap(&self, path: &Path, mmap: Arc<memmap2::Mmap>) -> Result<()> {
        MetalAllocator::register_mmap(self, path, mmap)
    }
    // push_alloc / take_allocations / unrecord_alloc: trait defaults (no-op) —
    // metal arenas own their memory; nothing to track for ownership transfer.

    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        unsafe { self.alloc_and_copy_host_aligned(src_host, bytes, Self::MIN_BIND_ALIGN) }
    }

    unsafe fn alloc_and_copy_host_aligned(
        &mut self,
        src_host: *const u8,
        bytes: usize,
        min_align: usize,
    ) -> Result<*mut u8> {
        // Zero-copy fast path: the source already lives in a
        // registered safetensors mmap AND lands at a `min_align`-aligned
        // offset. `min_align` is clamped above by `MIN_BIND_ALIGN` since
        // any pointer we return is also reachable via the trait
        // `alloc_and_copy_host` (no dtype guarantee), and may be bound
        // to arbitrary kernels later. The clamp below keeps the
        // SIMD-wide-safe floor; per-dtype relaxation below 16 only
        // helps for offsets in `(MIN_BIND_ALIGN, dtype_size]`.
        //
        // For `mlx-community` 4bit safetensors the data section lands
        // at file-offset `mod 16 = 2`, so the 16-byte gate rejects
        // every tensor.
        // Relaxing to `min_align = 2` lets F16/BF16 scales/biases/
        // RMSNorm-gain tensors take this path — those kernel bindings
        // read scalar (`sl[0]`, `weight[i]`) and 2-byte alignment is
        // safe.
        let effective_min_align = min_align.clamp(1, Self::MIN_BIND_ALIGN);
        if bytes > 0 {
            match self.classify_mmap_offset(src_host, bytes, effective_min_align) {
                MmapClassify::Aligned { aligned_ptr } => {
                    self.load_stats
                        .zero_copy_calls
                        .fetch_add(1, Ordering::Relaxed);
                    self.load_stats
                        .zero_copy_bytes
                        .fetch_add(bytes as u64, Ordering::Relaxed);
                    // Visibility bookkeeping: did the relaxation
                    // actually do anything? Re-check at the strict
                    // 16-byte gate; if THAT would have failed, the
                    // relaxation is responsible for this zero-copy.
                    // After the register-time bulk-copy lands the
                    // shift, this counter should approach 0 (every
                    // canonical-layout tensor passes the strict gate
                    // already).
                    if effective_min_align < Self::MIN_BIND_ALIGN {
                        let strict =
                            self.aligned_mmap_offset(src_host, bytes, Self::MIN_BIND_ALIGN);
                        if strict.is_none() {
                            self.load_stats
                                .zero_copy_relaxed_calls
                                .fetch_add(1, Ordering::Relaxed);
                            self.load_stats
                                .zero_copy_relaxed_bytes
                                .fetch_add(bytes as u64, Ordering::Relaxed);
                        }
                    }
                    return Ok(aligned_ptr);
                }
                MmapClassify::Unaligned => {
                    self.load_stats
                        .memcpy_unaligned
                        .fetch_add(1, Ordering::Relaxed);
                }
                MmapClassify::Outside => {
                    self.load_stats
                        .memcpy_outside_mmap
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let aligned_bytes = if bytes == 0 {
            0
        } else {
            bytes.div_ceil(Self::MIN_BIND_ALIGN) * Self::MIN_BIND_ALIGN
        };
        // Weight-memcpy fallback: immutable weight bytes → the weight-arena
        // pool, inserted into `weights_residency` (pageable by default),
        // NOT the wired scratch `arenas`.
        let mut arenas = self
            .weight_arenas
            .lock()
            .expect("MetalAllocator weight_arenas Mutex");
        if bytes == 0 {
            if arenas.is_empty() {
                Self::push_arena_locked(
                    &self.device,
                    &mut arenas,
                    self.chunk_bytes,
                    0,
                    &self.on_new_arena,
                    &self.weights_residency,
                )?;
            }
            return Ok(arenas[0].base);
        }

        let idx = if let Some(idx) = arenas
            .iter()
            .rposition(|a| a.capacity - a.used >= aligned_bytes)
        {
            idx
        } else {
            Self::push_arena_locked(
                &self.device,
                &mut arenas,
                self.chunk_bytes,
                aligned_bytes,
                &self.on_new_arena,
                &self.weights_residency,
            )?
        };
        let arena = &mut arenas[idx];
        let offset = arena.used;
        debug_assert_eq!(
            offset % Self::MIN_BIND_ALIGN,
            0,
            "arena.used is not {}-aligned on entry",
            Self::MIN_BIND_ALIGN
        );
        let dst = unsafe { arena.base.add(offset) };
        unsafe { std::ptr::copy_nonoverlapping(src_host, dst, bytes) };
        arena.used = offset + aligned_bytes;
        self.load_stats.memcpy_calls.fetch_add(1, Ordering::Relaxed);
        self.load_stats
            .memcpy_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        let bucket = if bytes < 1 << 20 {
            &self.load_stats.memcpy_small_calls
        } else if bytes < 16 << 20 {
            &self.load_stats.memcpy_med_calls
        } else {
            &self.load_stats.memcpy_large_calls
        };
        bucket.fetch_add(1, Ordering::Relaxed);
        Ok(dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn try_device() -> Option<Device> {
        MTLCreateSystemDefaultDevice()
    }

    #[test]
    fn aligned_meta_json_records_escaped_src_path() {
        // `src_path` lets `scr model cache inspect` label a sidecar with its
        // model — including local-path loads. It must be emitted and
        // JSON-escaped (paths can contain quotes/backslashes).
        let meta = AlignedCacheMeta {
            src_size: 123,
            src_mtime_ns: 456,
            aligned_capacity: 789,
            bin: std::path::PathBuf::from("/tmp/x.bin"),
            src_path: "/weird/pa\"th/model.safetensors".to_string(),
            content_hash: std::cell::Cell::new(0xdead),
        };
        let json = meta.to_json();
        let v: serde_json::Value = serde_json::from_str(&json).expect("to_json emits valid JSON");
        assert_eq!(v["src_path"], "/weird/pa\"th/model.safetensors");
        assert_eq!(v["content_hash"], 0xdead_u64);
        assert_eq!(v["layout_version"], ALIGNED_CACHE_LAYOUT_VERSION);
    }

    #[test]
    fn parse_safetensors_rejects_truncated_shard() {
        // Minimal safetensors: [u64 header_len][json][data]. One tensor
        // "w" of `data_len` bytes at data_offsets [0, data_len].
        let data_len = 32usize;
        let header = format!(
            "{{\"w\":{{\"dtype\":\"F16\",\"shape\":[16],\"data_offsets\":[0,{data_len}]}}}}"
        );
        let hb = header.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        buf.extend_from_slice(hb);
        let data_section_start = buf.len();
        buf.extend_from_slice(&vec![0u8; data_len]);

        // Full, well-formed buffer parses to one tensor at the right offset.
        let got = MetalAllocator::parse_safetensors_tensors(buf.as_ptr(), buf.len())
            .expect("well-formed shard parses");
        assert_eq!(got, vec![(data_section_start, data_len)]);

        // Truncated: the header still names the full 32-byte tensor, but
        // the file is 1 byte short — exactly what an interrupted download
        // mmaps. Must reject rather than hand back an out-of-bounds range
        // (which the copy loop would SIGBUS on, or bake into a corrupt
        // aligned cache).
        assert!(
            MetalAllocator::parse_safetensors_tensors(buf.as_ptr(), buf.len() - 1).is_none(),
            "a tensor extending past the mapped file must be rejected"
        );
    }

    #[test]
    fn weight_residency_modes_construct() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        // Default is un-wired weights.
        assert_eq!(WeightResidency::default(), WeightResidency::Unwired);
        // Both modes construct and expose a weights residency set whose
        // active-ness matches the (always-wired) working-set residency.
        for mode in [WeightResidency::Unwired, WeightResidency::Wired] {
            let alloc = MetalAllocator::with_weight_residency(device.clone(), mode);
            assert_eq!(
                alloc.weights_residency().is_active(),
                alloc.residency().is_active(),
                "weights set activeness should track the working-set residency ({mode:?})"
            );
        }
    }

    #[test]
    fn alloc_returns_pointer_into_arena() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::new(device);
        let src = b"hello, metal";
        let ptr = unsafe {
            alloc
                .alloc_and_copy_host(src.as_ptr(), src.len())
                .expect("alloc")
        };
        assert!(!ptr.is_null());
        let read = unsafe { std::slice::from_raw_parts(ptr, src.len()) };
        assert_eq!(read, src);
        // `used_bytes` reports the padded request (rounded up to
        // `MIN_BIND_ALIGN` so the *next* allocation lands at an
        // aligned offset). 12 bytes round up to 16.
        let expected_used =
            src.len().div_ceil(MetalAllocator::MIN_BIND_ALIGN) * MetalAllocator::MIN_BIND_ALIGN;
        assert_eq!(alloc.used_bytes(), expected_used);
        assert_eq!(alloc.arena_count(), 1);

        let (buf, off) = alloc.buffer_for(ptr).expect("buffer_for");
        assert_eq!(off, 0);
        assert_eq!(buf.length(), DEFAULT_CHUNK_BYTES);
    }

    #[test]
    fn multiple_allocs_share_arena_until_full() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        let buf_a = vec![0xAAu8; 1024];
        let buf_b = vec![0xBBu8; 1024];
        let buf_c = vec![0xCCu8; 3072];

        let pa = unsafe {
            alloc
                .alloc_and_copy_host(buf_a.as_ptr(), buf_a.len())
                .unwrap()
        };
        let pb = unsafe {
            alloc
                .alloc_and_copy_host(buf_b.as_ptr(), buf_b.len())
                .unwrap()
        };
        let pc = unsafe {
            alloc
                .alloc_and_copy_host(buf_c.as_ptr(), buf_c.len())
                .unwrap()
        };

        let (ba, oa) = alloc.buffer_for(pa).unwrap();
        let (bb, ob) = alloc.buffer_for(pb).unwrap();
        let (bc, oc) = alloc.buffer_for(pc).unwrap();
        assert_eq!(Retained::as_ptr(&ba), Retained::as_ptr(&bb));
        assert_ne!(Retained::as_ptr(&ba), Retained::as_ptr(&bc));
        assert_eq!(oa, 0);
        assert_eq!(ob, 1024);
        assert_eq!(oc, 0);

        let read_a = unsafe { std::slice::from_raw_parts(pa, buf_a.len()) };
        let read_c = unsafe { std::slice::from_raw_parts(pc, buf_c.len()) };
        assert!(read_a.iter().all(|&b| b == 0xAA));
        assert!(read_c.iter().all(|&b| b == 0xCC));

        assert_eq!(alloc.arena_count(), 2);
        // Each allocation rounds up to a multiple of MIN_BIND_ALIGN
        // (16). 1024, 1024, 3072 are already multiples of 16 so the
        // total is unchanged here — this test serves as a guardrail
        // that aligned-size inputs don't grow.
        assert_eq!(alloc.used_bytes(), 1024 + 1024 + 3072);
    }

    #[test]
    fn weight_memcpy_and_scratch_use_separate_arena_pools() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        // Scratch (alloc_uninit) → wired `arenas`; weight copy
        // (alloc_and_copy_host) → un-wired `weight_arenas`. They must land
        // in DIFFERENT arena buffers so the weight pool can be un-wired
        // without dragging the scratch pool with it.
        let scratch = alloc.alloc_uninit(64).expect("scratch alloc");
        let w = vec![0x9u8; 64];
        let weight = unsafe {
            alloc
                .alloc_and_copy_host(w.as_ptr(), w.len())
                .expect("weight alloc")
        };
        let (sb, _) = alloc.buffer_for(scratch).expect("scratch buffer_for");
        let (wb, _) = alloc.buffer_for(weight).expect("weight buffer_for");
        assert_ne!(
            Retained::as_ptr(&sb),
            Retained::as_ptr(&wb),
            "scratch and weight-memcpy must not share an arena buffer"
        );
        // One arena in each pool.
        assert_eq!(alloc.arena_count(), 2);
    }

    #[test]
    fn oversized_request_gets_dedicated_arena() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::with_chunk_bytes(device, 4096);
        let big = vec![0x42u8; 16 * 1024];
        let p = unsafe { alloc.alloc_and_copy_host(big.as_ptr(), big.len()).unwrap() };
        let (buf, off) = alloc.buffer_for(p).unwrap();
        assert_eq!(off, 0);
        assert!(buf.length() >= big.len());
        assert_eq!(alloc.arena_count(), 1);
    }

    #[test]
    fn zero_byte_alloc_returns_nonnull_sentinel() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let mut alloc = MetalAllocator::new(device);
        let p = unsafe { alloc.alloc_and_copy_host(std::ptr::null(), 0).unwrap() };
        assert!(!p.is_null());
        assert_eq!(alloc.used_bytes(), 0);
    }

    #[test]
    fn buffer_for_returns_none_for_foreign_pointer() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let alloc = MetalAllocator::new(device);
        let stack = 0u8;
        assert!(alloc.buffer_for(&stack).is_none());
    }
}
