// SPDX-License-Identifier: Apache-2.0
//! CUDA-only `GpuWeights` surface: the weight pre-stage ("precast") pipeline
//! and the `CudaWeightsExt` extension trait (stream uploads, TP shard copies,
//! alloc tracking) plus the `from_path` / `from_gguf_file` constructors.
//!
//! The neutral `GpuWeights` struct + host loader live in `scratchy-layers`;
//! everything that issues CUDA `driver` calls or mints `RawGpuMem` rides here,
//! reaching the struct through its public accessors.

use anyhow::{Result, bail};
use cudarc::driver::sys::CUstream;
use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::driver;
use crate::dtype::DType;
use scratchy_layers::weights::{GpuWeights, UploadSrc};
use scratchy_tensors::{DeviceAllocator, PrecastEntry, PrecastPipeline};

// ---- pre-stage tuning constants ---------------------------------------------

/// Starting active reader count for the pre-stage ramp.
const PRESTAGE_WORKERS_START: usize = 16;
/// Ceiling for the active reader count.
const PRESTAGE_WORKERS_MAX: usize = 64;
/// Ramp controller observation window.
const RAMP_WINDOW_MS: u64 = 200;
/// Readers added per ramp step.
const RAMP_STEP: usize = 8;
/// Minimum fractional throughput gain a window must show to keep ramping.
const RAMP_GAIN_MIN: f64 = 0.10;
/// Minimum sustained observation before a settled QD is persisted.
const RAMP_MIN_OBSERVE_MS: u64 = 1000;
/// Floor / ceiling for the per-worker pinned staging slot.
const STAGE_SLOT_BYTES_MIN: usize = 2 << 20;
const STAGE_SLOT_BYTES_MAX: usize = 16 << 20;
/// Target wall-time for one chunk's H2D copy.
const STAGE_CHUNK_TARGET_US: usize = 200;
/// Fallback H2D link bandwidth (GB/s) when the GPU has no profile.
const DEFAULT_PCIE_H2D_GBPS: f64 = 20.0;

/// Per-worker staging slot size, derived from the GPU profile's host-link
/// bandwidth (one chunk ≈ [`STAGE_CHUNK_TARGET_US`]), rounded to a power of two
/// and clamped to [[`STAGE_SLOT_BYTES_MIN`], [`STAGE_SLOT_BYTES_MAX`]].
fn resolve_stage_slot_bytes() -> usize {
    const US_PER_SEC: f64 = 1e6;
    const BYTES_PER_GB: f64 = 1e9;
    let gbps = match crate::targets::detect() {
        Ok(profile) => profile.pcie_h2d_gbps,
        Err(e) => {
            tracing::info!(
                "pre-stage: no GPU profile ({e}); assuming {DEFAULT_PCIE_H2D_GBPS} GB/s H2D"
            );
            DEFAULT_PCIE_H2D_GBPS
        }
    };
    let target = gbps * BYTES_PER_GB * (STAGE_CHUNK_TARGET_US as f64 / US_PER_SEC);
    (target as usize)
        .next_power_of_two()
        .clamp(STAGE_SLOT_BYTES_MIN, STAGE_SLOT_BYTES_MAX)
}

/// Sidecar path persisting the settled reader QD for the FS device backing
/// `model_dir` (keyed by `st_dev` — QD is a property of the disk).
fn qd_sidecar_path(model_dir: &Path) -> Option<std::path::PathBuf> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let dev = std::fs::metadata(model_dir).ok()?.dev();
        let cache = std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache"))
            })?;
        Some(cache.join("scratchy").join(format!("io-qd-{dev}")))
    }
    #[cfg(not(unix))]
    {
        let _ = model_dir;
        None
    }
}

/// Read the persisted reader QD for this disk, clamped to the ramp range.
fn read_qd_sidecar(model_dir: &Path) -> Option<usize> {
    let s = std::fs::read_to_string(qd_sidecar_path(model_dir)?).ok()?;
    let qd = s.trim().parse::<usize>().ok()?;
    Some(qd.clamp(PRESTAGE_WORKERS_START, PRESTAGE_WORKERS_MAX))
}

/// Persist the settled reader QD (best effort).
fn write_qd_sidecar(model_dir: &Path, qd: usize) {
    let Some(path) = qd_sidecar_path(model_dir) else {
        return;
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, qd.to_string()).ok();
}

// ---- pre-stage pipeline state ------------------------------------------------

type PrecastKey = (usize, usize, usize);

fn precast_key(mmap: &Arc<memmap2::Mmap>, data_offset: usize, size_bytes: usize) -> PrecastKey {
    (mmap.as_ptr() as usize, data_offset, size_bytes)
}

/// One unit of pre-stage work: (key, mmap, data_offset, size_bytes, dtype).
type PrecastWorkItem = (PrecastKey, Arc<memmap2::Mmap>, usize, usize, DType);

#[derive(Default)]
struct PrecastShared {
    ready: HashMap<PrecastKey, PrecastEntry>,
    in_progress: std::collections::HashSet<PrecastKey>,
    consumed: std::collections::HashSet<PrecastKey>,
}

struct PrecastState {
    shared: Mutex<PrecastShared>,
    cv: std::sync::Condvar,
    active_workers: std::sync::atomic::AtomicUsize,
    shutdown: AtomicBool,
}

impl PrecastPipeline for PrecastState {
    fn take(
        &self,
        mmap_base: usize,
        data_offset: usize,
        size_bytes: usize,
    ) -> Option<PrecastEntry> {
        let key: PrecastKey = (mmap_base, data_offset, size_bytes);
        let mut s = self.shared.lock().ok()?;
        loop {
            if let Some(entry) = s.ready.remove(&key) {
                return Some(entry);
            }
            if s.in_progress.contains(&key) && !self.shutdown.load(Ordering::Relaxed) {
                s = self.cv.wait(s).ok()?;
                continue;
            }
            s.consumed.insert(key);
            return None;
        }
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.cv.notify_all();
    }
}

struct PrecastStats {
    cast: std::sync::atomic::AtomicUsize,
    staged: std::sync::atomic::AtomicUsize,
    bytes: std::sync::atomic::AtomicUsize,
    live_workers: std::sync::atomic::AtomicUsize,
    t0: std::time::Instant,
}

/// QD ramp controller — observes windows of staged bytes and activates more
/// reader lanes while that improves throughput; persists the settle per-disk.
fn prestage_ramp_controller(
    state: Arc<PrecastState>,
    stats: Arc<PrecastStats>,
    work_len: usize,
    next: Arc<std::sync::atomic::AtomicUsize>,
    model_dir: Option<std::path::PathBuf>,
) {
    let window = std::time::Duration::from_millis(RAMP_WINDOW_MS);
    let mut last_bytes = 0usize;
    let measure = |last: &mut usize| -> Option<f64> {
        std::thread::sleep(window);
        if state.shutdown.load(Ordering::Relaxed) || next.load(Ordering::Relaxed) >= work_len {
            return None;
        }
        let bytes = stats.bytes.load(Ordering::Relaxed);
        let rate = bytes.saturating_sub(*last) as f64 / window.as_secs_f64();
        *last = bytes;
        Some(rate)
    };

    let Some(mut best_rate) = measure(&mut last_bytes) else {
        return;
    };
    loop {
        let prev = state.active_workers.load(Ordering::Relaxed);
        if prev >= PRESTAGE_WORKERS_MAX {
            break;
        }
        state.active_workers.store(
            (prev + RAMP_STEP).min(PRESTAGE_WORKERS_MAX),
            Ordering::Relaxed,
        );
        state.cv.notify_all();
        let Some(rate) = measure(&mut last_bytes) else {
            return;
        };
        if rate > best_rate * (1.0 + RAMP_GAIN_MIN) {
            best_rate = rate;
        } else {
            state.active_workers.store(prev, Ordering::Relaxed);
            break;
        }
    }
    let settled = state.active_workers.load(Ordering::Relaxed);
    let observed_ms = stats.t0.elapsed().as_millis();
    if observed_ms >= u128::from(RAMP_MIN_OBSERVE_MS) {
        tracing::info!("pre-stage QD ramp settled at {settled} readers (persisted)");
        if let Some(dir) = model_dir {
            write_qd_sidecar(&dir, settled);
        }
    } else {
        tracing::info!(
            "pre-stage QD ramp settled at {settled} readers \
             ({observed_ms}ms observed — too brief to persist)"
        );
    }
}

/// Background worker: pulls tensors off the shared list (largest first), faults
/// their mmap pages, and streams the bytes (casting floats when needed) through
/// a pinned slot into a fresh device buffer. Results land in `state.shared.ready`.
fn precast_worker(
    idx: usize,
    state: Arc<PrecastState>,
    target_dtype: Option<DType>,
    work: Arc<Vec<PrecastWorkItem>>,
    next: Arc<std::sync::atomic::AtomicUsize>,
    ctx: usize,
    slot_bytes: usize,
    stats: Arc<PrecastStats>,
) {
    {
        let mut s = state.shared.lock().unwrap();
        while idx >= state.active_workers.load(Ordering::Relaxed)
            && !state.shutdown.load(Ordering::Relaxed)
            && next.load(Ordering::Relaxed) < work.len()
        {
            s = state.cv.wait(s).unwrap();
        }
        drop(s);
    }
    if state.shutdown.load(Ordering::Relaxed) || next.load(Ordering::Relaxed) >= work.len() {
        precast_worker_finish(&state, &stats);
        return;
    }

    if let Err(e) = unsafe { driver::ctx_set_current(ctx as cudarc::driver::sys::CUcontext) } {
        tracing::warn!("pre-stage worker: ctx_set_current failed, worker idle: {e}");
        precast_worker_finish(&state, &stats);
        return;
    }
    let slot = match unsafe { driver::mem_alloc_host(slot_bytes) } {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("pre-stage worker: pinned slot alloc failed, worker idle: {e}");
            precast_worker_finish(&state, &stats);
            return;
        }
    };

    loop {
        {
            let mut s = state.shared.lock().unwrap();
            while idx >= state.active_workers.load(Ordering::Relaxed)
                && !state.shutdown.load(Ordering::Relaxed)
                && next.load(Ordering::Relaxed) < work.len()
            {
                s = state.cv.wait(s).unwrap();
            }
        }
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some((key, mmap, data_offset, size_bytes, dtype)) = work.get(i) else {
            break;
        };

        {
            let mut s = state.shared.lock().unwrap();
            if s.consumed.contains(key) || s.ready.contains_key(key) || s.in_progress.contains(key)
            {
                continue;
            }
            s.in_progress.insert(*key);
        }

        let data = &mmap[*data_offset..*data_offset + *size_bytes];
        let result =
            unsafe { stage_to_device(data, *dtype, target_dtype, slot, slot_bytes, &stats.bytes) };

        let mut s = state.shared.lock().unwrap();
        s.in_progress.remove(key);
        match result {
            Ok(entry) => {
                if entry.dtype == *dtype {
                    stats.staged.fetch_add(1, Ordering::Relaxed);
                } else {
                    stats.cast.fetch_add(1, Ordering::Relaxed);
                }
                s.ready.insert(*key, entry);
            }
            Err(e) => {
                tracing::warn!("pre-stage failed (falling back to pageable upload): {e}");
            }
        }
        drop(s);
        state.cv.notify_all();
    }

    unsafe {
        driver::mem_free_host(slot).ok();
    }
    precast_worker_finish(&state, &stats);
}

/// Stream one tensor from its mmap bytes into a fresh device buffer through a
/// pinned `slot`, casting chunks to `target_dtype` when needed.
///
/// # Safety
/// `slot` must be a valid pinned allocation of `slot_bytes`; the thread must
/// have a current CUDA context.
unsafe fn stage_to_device(
    data: &[u8],
    dtype: DType,
    target_dtype: Option<DType>,
    slot: NonNull<u8>,
    slot_bytes: usize,
    progress_bytes: &std::sync::atomic::AtomicUsize,
) -> Result<PrecastEntry> {
    let needs_cast = match target_dtype {
        Some(target) => matches!(dtype, DType::F32 | DType::F16 | DType::BF16) && dtype != target,
        None => false,
    };
    let target = if needs_cast {
        target_dtype.unwrap_or(dtype)
    } else {
        dtype
    };

    let src_elem = dtype.size_bytes();
    let dst_elem = target.size_bytes();
    let numel = data.len() / src_elem;
    let staged_bytes = numel * dst_elem;

    let dev_ptr = unsafe { driver::mem_alloc(staged_bytes)? };
    let gpu = unsafe { crate::raw_cuda(dev_ptr, staged_bytes) };

    let chunk_elems = slot_bytes / dst_elem.max(src_elem);
    anyhow::ensure!(chunk_elems > 0, "stage slot smaller than one element");

    let mut done = 0usize;
    while done < numel {
        let n = chunk_elems.min(numel - done);
        let src = &data[done * src_elem..(done + n) * src_elem];
        if needs_cast {
            cast_slice_into(slot, src, dtype, target, n)?;
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), slot.as_ptr(), n * src_elem);
            }
        }
        unsafe {
            driver::memcpy_htod(gpu.ptr().add(done * dst_elem), slot.as_ptr(), n * dst_elem)?;
        }
        progress_bytes.fetch_add(n * dst_elem, Ordering::Relaxed);
        done += n;
    }

    Ok(PrecastEntry {
        gpu,
        size_bytes: staged_bytes,
        dtype: target,
    })
}

/// Scalar cast of `numel` elements from `src` (`src_dtype`) into `dst` (`target`).
fn cast_slice_into(
    dst: NonNull<u8>,
    src: &[u8],
    src_dtype: DType,
    target: DType,
    numel: usize,
) -> Result<()> {
    match (src_dtype, target) {
        (DType::F32, DType::BF16) => {
            let s = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
            let d = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut u16, numel) };
            for (s, d) in s.iter().zip(d.iter_mut()) {
                *d = half::bf16::from_f32(*s).to_bits();
            }
        }
        (DType::F32, DType::F16) => {
            let s = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
            let d = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut u16, numel) };
            for (s, d) in s.iter().zip(d.iter_mut()) {
                *d = half::f16::from_f32(*s).to_bits();
            }
        }
        (DType::F16, DType::BF16) => {
            let s = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
            let d = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut u16, numel) };
            for (s, d) in s.iter().zip(d.iter_mut()) {
                *d = half::bf16::from_f32(half::f16::from_bits(*s).to_f32()).to_bits();
            }
        }
        (DType::BF16, DType::F16) => {
            let s = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
            let d = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut u16, numel) };
            for (s, d) in s.iter().zip(d.iter_mut()) {
                *d = half::f16::from_f32(half::bf16::from_bits(*s).to_f32()).to_bits();
            }
        }
        (DType::BF16, DType::F32) => {
            let s = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
            let d = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut f32, numel) };
            for (s, d) in s.iter().zip(d.iter_mut()) {
                *d = half::bf16::from_bits(*s).to_f32();
            }
        }
        (DType::F16, DType::F32) => {
            let s = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
            let d = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut f32, numel) };
            for (s, d) in s.iter().zip(d.iter_mut()) {
                *d = half::f16::from_bits(*s).to_f32();
            }
        }
        _ => bail!("unhandled cast: {src_dtype:?} → {target:?}"),
    }
    Ok(())
}

/// Last worker out logs the pipeline totals.
fn precast_worker_finish(state: &PrecastState, stats: &PrecastStats) {
    state.cv.notify_all();
    if stats.live_workers.fetch_sub(1, Ordering::AcqRel) == 1 {
        let secs = stats.t0.elapsed().as_secs_f64();
        let bytes = stats.bytes.load(Ordering::Relaxed);
        let gib = bytes as f64 / (1u64 << 30) as f64;
        tracing::info!(
            "Pre-stage pipeline done: {} cast + {} copied to device, \
             {gib:.2} GiB in {secs:.2}s ({:.2} GiB/s)",
            stats.cast.load(Ordering::Relaxed),
            stats.staged.load(Ordering::Relaxed),
            gib / secs.max(f64::EPSILON),
        );
    }
}

// ---- constructors (free fns — only one external caller) ----------------------

/// Unified entry point: load a safetensors directory or a `.gguf` file (format
/// detected from the path). For safetensors, `target_dtype` is stored so
/// `take()` casts; for GGUF the registered loader consumes everything eagerly.
///
/// # Safety
/// Caller must hold a valid CUDA context + stream.
pub unsafe fn from_path(
    path: impl AsRef<Path>,
    stream: CUstream,
    alloc: &mut crate::CachingAllocator,
    target_dtype: DType,
    tp_rank: usize,
    tp_world_size: usize,
) -> Result<GpuWeights<crate::CudaAllocator>> {
    let path = path.as_ref();
    let is_gguf_file = path.is_file() && path.extension().is_some_and(|e| e == "gguf");
    let gguf_in_dir = path.is_dir() && {
        std::fs::read_dir(path)
            .ok()
            .and_then(|mut it| {
                it.find_map(|entry| {
                    let p = entry.ok()?.path();
                    (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
                })
            })
            .is_some()
    };
    if is_gguf_file {
        return unsafe {
            from_gguf_file(path, target_dtype, alloc, stream, tp_rank, tp_world_size)
        };
    }
    if gguf_in_dir {
        let gguf_path = std::fs::read_dir(path)?
            .find_map(|entry| {
                let p = entry.ok()?.path();
                (p.extension().is_some_and(|e| e == "gguf")).then_some(p)
            })
            .ok_or_else(|| anyhow::anyhow!("no .gguf file in directory"))?;
        return unsafe {
            from_gguf_file(
                &gguf_path,
                target_dtype,
                alloc,
                stream,
                tp_rank,
                tp_world_size,
            )
        };
    }
    let mut gw = GpuWeights::from_dir(path, crate::CudaAllocator::new(stream))?;
    gw.set_target_dtype(target_dtype);
    Ok(gw)
}

/// Load a single `.gguf` file via the inventory-registered loader.
///
/// # Safety
/// Caller must hold a valid CUDA context + stream.
pub unsafe fn from_gguf_file(
    path: impl AsRef<Path>,
    model_dtype: DType,
    alloc: &mut crate::CachingAllocator,
    stream: CUstream,
    tp_rank: usize,
    tp_world_size: usize,
) -> Result<GpuWeights<crate::CudaAllocator>> {
    let path = path.as_ref();
    let reg = crate::gguf_loader::registered_gguf_loader().ok_or_else(|| {
        anyhow::anyhow!(
            "GGUF loader not registered — link `scratchy-target-cuda` (which submits a \
             `GgufLoaderRegistration` via inventory) into the consuming binary"
        )
    })?;
    unsafe { (reg.load)(path, model_dtype, alloc, stream, tp_rank, tp_world_size) }
}

// ---- CudaWeightsExt: stream uploads + alloc tracking + precast ----------------

/// CUDA-only `GpuWeights` operations that issue `driver` DMAs / mint
/// `RawGpuMem`. Implemented for the cuda-allocator pool; callers `use` it.
pub trait CudaWeightsExt {
    /// Track an externally-allocated `RawGpuMem` (GGUF dequant path).
    fn push_gpu_alloc(&mut self, alloc: crate::alloc::RawGpuMem);
    /// Track a raw `(ptr, size)` GPU allocation (quant loaders).
    fn record_alloc(&mut self, ptr: *mut u8, size_bytes: usize);
    /// Untrack one allocation by pointer, leaking it for the caller to free.
    fn unrecord_alloc(&mut self, ptr: *mut u8);
    /// Drain all tracked GPU allocations (load-then-detach).
    fn take_gpu_allocs(&mut self) -> Vec<crate::alloc::RawGpuMem>;
    /// The H2D stream.
    fn stream(&self) -> CUstream;
    /// Start the background pre-stage pipeline (no-op if already running).
    fn start_precast(&mut self);
    /// Remove `name` and copy it directly into `dst` on `stream`.
    ///
    /// # Safety
    /// Valid CUDA context + stream; `dst` sized for the (post-cast) tensor.
    unsafe fn take_into(&mut self, name: &str, dst: *mut u8, stream: CUstream) -> Result<usize>;
    /// TP-shard variant of [`Self::take_into`].
    ///
    /// # Safety
    /// As [`Self::take_into`].
    unsafe fn take_shard_into(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
        dst: *mut u8,
        stream: CUstream,
    ) -> Result<usize>;
}

impl CudaWeightsExt for GpuWeights<crate::CudaAllocator> {
    fn push_gpu_alloc(&mut self, alloc: crate::alloc::RawGpuMem) {
        self.allocator_mut().adopt_raw(alloc);
    }

    fn record_alloc(&mut self, ptr: *mut u8, size_bytes: usize) {
        self.allocator_mut()
            .adopt_raw(unsafe { crate::raw_cuda(ptr, size_bytes) });
    }

    fn unrecord_alloc(&mut self, ptr: *mut u8) {
        self.allocator_mut().unrecord_alloc(ptr);
    }

    fn take_gpu_allocs(&mut self) -> Vec<crate::alloc::RawGpuMem> {
        self.allocator_mut().take_allocations()
    }

    fn stream(&self) -> CUstream {
        self.allocator().stream()
    }

    fn start_precast(&mut self) {
        if self.precast_started() {
            return;
        }
        let target_dtype = self.target_dtype();
        let ctx = match unsafe { driver::ctx_get_current() } {
            Ok(ctx) if !ctx.is_null() => ctx as usize,
            _ => {
                tracing::warn!("start_precast: no current CUDA context; pre-stage disabled");
                return;
            }
        };

        let mut seen = std::collections::HashSet::new();
        let mut work: Vec<PrecastWorkItem> = self
            .mmap_tensor_refs()
            .into_iter()
            .filter_map(|(mmap, data_offset, size_bytes, dtype)| {
                let key = precast_key(&mmap, data_offset, size_bytes);
                seen.insert(key)
                    .then_some((key, mmap, data_offset, size_bytes, dtype))
            })
            .collect();
        work.sort_by_key(|b| std::cmp::Reverse(b.3));
        let work = Arc::new(work);

        let start_qd = self
            .source_dir()
            .and_then(read_qd_sidecar)
            .unwrap_or(PRESTAGE_WORKERS_START);

        let state = Arc::new(PrecastState {
            shared: Mutex::new(PrecastShared::default()),
            cv: std::sync::Condvar::new(),
            active_workers: std::sync::atomic::AtomicUsize::new(start_qd),
            shutdown: AtomicBool::new(false),
        });
        self.set_precast(Arc::clone(&state) as Arc<dyn PrecastPipeline>);

        let slot_bytes = resolve_stage_slot_bytes();
        tracing::info!(
            "Pre-stage pipeline: {start_qd} readers (QD ramp to ≤{PRESTAGE_WORKERS_MAX}) × {} MiB pinned slots",
            slot_bytes >> 20,
        );

        let stats = Arc::new(PrecastStats {
            cast: std::sync::atomic::AtomicUsize::new(0),
            staged: std::sync::atomic::AtomicUsize::new(0),
            bytes: std::sync::atomic::AtomicUsize::new(0),
            live_workers: std::sync::atomic::AtomicUsize::new(PRESTAGE_WORKERS_MAX),
            t0: std::time::Instant::now(),
        });
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        for i in 0..PRESTAGE_WORKERS_MAX {
            let state = Arc::clone(&state);
            let work = Arc::clone(&work);
            let next = Arc::clone(&next);
            let stats = Arc::clone(&stats);
            let handle = std::thread::Builder::new()
                .name(format!("weight-prestage-{i}"))
                .spawn(move || {
                    precast_worker(i, state, target_dtype, work, next, ctx, slot_bytes, stats);
                })
                .expect("failed to spawn pre-stage worker");
            self.push_precast_handle(handle);
        }

        {
            let state = Arc::clone(&state);
            let stats = Arc::clone(&stats);
            let next = Arc::clone(&next);
            let work_len = work.len();
            let model_dir = self.source_dir().map(|p| p.to_path_buf());
            let handle = std::thread::Builder::new()
                .name("weight-prestage-qd".into())
                .spawn(move || {
                    prestage_ramp_controller(state, stats, work_len, next, model_dir);
                })
                .expect("failed to spawn pre-stage QD controller");
            self.push_precast_handle(handle);
        }
    }

    unsafe fn take_into(&mut self, name: &str, dst: *mut u8, stream: CUstream) -> Result<usize> {
        match self.take_upload_src(name)? {
            UploadSrc::Device(entry) => {
                driver::memcpy_dtod_async(dst, entry.gpu.ptr(), entry.size_bytes, stream)?;
                let size = entry.size_bytes;
                driver::stream_synchronize(stream)?;
                drop(entry);
                Ok(size)
            }
            UploadSrc::DeviceSlice { src, bytes, _keep } => {
                driver::memcpy_dtod_async(dst, src, bytes, stream)?;
                driver::stream_synchronize(stream)?;
                drop(_keep);
                Ok(bytes)
            }
            UploadSrc::Host {
                ptr,
                bytes,
                from_scratch,
                ..
            } => {
                driver::memcpy_htod_async(dst, ptr, bytes, stream)?;
                if from_scratch {
                    driver::stream_synchronize(stream)?;
                }
                Ok(bytes)
            }
        }
    }

    unsafe fn take_shard_into(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
        dst: *mut u8,
        stream: CUstream,
    ) -> Result<usize> {
        match self.take_shard_upload_src(name, dim, rank, world_size)? {
            UploadSrc::DeviceSlice { src, bytes, _keep } => {
                driver::memcpy_dtod_async(dst, src, bytes, stream)?;
                driver::stream_synchronize(stream)?;
                drop(_keep);
                Ok(bytes)
            }
            UploadSrc::Device(entry) => {
                driver::memcpy_dtod_async(dst, entry.gpu.ptr(), entry.size_bytes, stream)?;
                let size = entry.size_bytes;
                driver::stream_synchronize(stream)?;
                drop(entry);
                Ok(size)
            }
            UploadSrc::Host { ptr, bytes, .. } => {
                driver::memcpy_htod_async(dst, ptr, bytes, stream)?;
                Ok(bytes)
            }
        }
    }
}
