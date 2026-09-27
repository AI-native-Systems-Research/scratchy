// SPDX-License-Identifier: Apache-2.0
//! MTL4 bake artifacts and step type.
//!
//! One `Mtl4Step` is produced per `BucketStep::Dispatch` from a bucket's
//! plan, holding the pipeline state and a pre-built `MTL4ArgumentTable`
//! per coalesced sub-command (one per entry in `direct_bindings`).
//!
//! Every bucket step is a `BucketStep::Dispatch`; `bake_mtl4_steps`
//! returns `None` only when a kernel exceeds the 31-entry
//! argument-table bind cap, and the pool asserts eligibility at forward
//! time. (f16 and bf16 dense GEMM both run as `gemm_{f16,bf16}_specialized`
//! Dispatch steps, `gemv_{f16,bf16}_specialized` at one row — there is no
//! MPS / classic command-buffer path.)

use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};

use super::__re::{ComputePipelineState, Device, MTL4ArgumentTableDescriptor, MTLDevice, MTLSize};
use super::lowered::Fence;
use super::worker::BucketStep;

/// MTL4 argument-table buffer-binding slot cap. The Metal runtime
/// enforces 31; we exit baking with `None` if any kernel asks for
/// more — such a bucket has no MTL4 plan and isn't executable.
pub const MTL4_MAX_BUFFER_BINDS: usize = 31;

/// One MTL4 dispatch segment, mirroring a `BucketStep::Dispatch`.
///
/// `pipeline` is a `dyn MTLComputePipelineState`; MTL4's
/// `setComputePipelineState` accepts it unchanged. `tables[i]` +
/// `dispatches[i]` together encode one
/// dispatch — `tables.len() == dispatches.len()` and matches the
/// inner length of the corresponding `direct_bindings` /
/// `direct_dispatch` in the source `BucketStep::Dispatch`.
pub struct Mtl4Step {
    /// Kernel id of this step's dispatches (one kernel per Dispatch step;
    /// coalescing requires same pipeline = same kernel). Used for
    /// the per-dispatch timing print label.
    pub kernel: super::lowered::KernelId,
    pub pipeline: ComputePipelineState,
    pub tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    pub dispatches: Vec<(MTLSize, MTLSize)>,
    /// Parallel to `dispatches`: optional per-dispatch m-axis scaling
    /// hint. When `Some`, the runtime patches the named axis of
    /// `dispatches[i].0` proportionally with actual `num_tokens`
    /// (see [`super::lowered::MScaling`]), shrinking the grid from
    /// the `bucket_m`-baked baseline down to the actual M of this
    /// forward.
    pub m_scaling: Vec<Option<super::lowered::MScaling>>,
    /// The fence before each sub-dispatch (parallel to `tables` /
    /// `dispatches`). Sourced from the macro-emitted
    /// `LoweredMetalTape::barrier_before` — no runtime analysis. The
    /// hazard class picks the MTL4 barrier's visibility: `Coherent`
    /// (RAW) → `Device`, `Ordered` (WAR/WAW) → `None`.
    pub barrier_before: Vec<Fence>,
    /// One runtime-gate flag per sub-dispatch (parallel to
    /// `tables` / `dispatches`). `None` (the common case) =
    /// always dispatch. `Some(OnlyIfSingleSeq)` /
    /// `Some(OnlyIfMultiSeq)` = the worker skips this dispatch
    /// when the live `num_seqs` doesn't match — used by the
    /// lm_head slice + fallback pair so the right path fires
    /// based on whether the bucket holds one sequence or many.
    pub runtime_gate: Vec<Option<super::lowered::RuntimeGate>>,
}

// Mtl4Step holds `Retained<ProtocolObject<dyn MTL4ArgumentTable>>`s,
// which aren't `Send`/`Sync` by default (Apple objc objects). The pool
// checks each worker out under a mutex (see `MetalWorkerPool::checkout`)
// and the argument tables outlive every forward via Apple ARC.
unsafe impl Send for Mtl4Step {}
unsafe impl Sync for Mtl4Step {}

/// Per-dispatch GPU timestamp probe (`SCRATCHY_METAL_KERNEL_TIME`).
///
/// One `MTL4CounterHeap` of timestamp entries plus the per-write
/// sidecar the printout needs (kernel id + grid of the dispatch each
/// timestamp followed). `write_timestamp` is called from the dispatch
/// loop after every kernel; `print_tape` is called after the host wait
/// of a forward, resolves the heap on the CPU timeline (safe — the
/// event wait has already drained the GPU), and prints each kernel's
/// duration as the delta to the previous timestamp, aggregated per
/// kernel id. `RefCell` because the dispatch loop takes `&self` while
/// the write index must advance; the pool's checkout mutex guarantees
/// single-threaded access per worker.
pub struct KernelTimeHeap {
    heap: Retained<ProtocolObject<dyn ::objc2_metal::MTL4CounterHeap>>,
    next_index: std::cell::RefCell<usize>,
    capacity: usize,
    /// Ticks per second of the GPU timestamp clock, read once at heap
    /// creation via `queryTimestampFrequency` — the deltas the printout
    /// divides by this, not by a nanosecond assumption.
    tick_hz: u64,
    /// (kernel id, grid width × height × depth, threadgroup size) per
    /// written timestamp, parallel to the heap indices.
    entries: std::cell::RefCell<Vec<(super::lowered::KernelId, String)>>,
}

// Same ARC-handles story as `Mtl4Step` above.
unsafe impl Send for KernelTimeHeap {}
unsafe impl Sync for KernelTimeHeap {}

impl KernelTimeHeap {
    /// Heap capacity. The driver caps timestamp heaps at 4096 entries
    /// (measured: NSError "Requested Heap size is too large (requested
    /// 16384 max is 4096)" on M1 Max / macOS 26) — the probe requests
    /// the cap, saturates silently past it, and a decode bucket's few
    /// hundred dispatches stay well inside.
    const CAPACITY: usize = 4096;

    pub fn new(device: &super::__re::Device) -> Self {
        use ::objc2_metal::MTL4CounterHeapType;
        let desc = ::objc2_metal::MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        unsafe { desc.setCount(Self::CAPACITY) };
        let heap = device
            .newCounterHeapWithDescriptor_error(&desc)
            .expect("newCounterHeapWithDescriptor returned nil (SCRATCHY_METAL_KERNEL_TIME)");
        let tick_hz = {
            use ::objc2_metal::MTLDevice as _;
            device.queryTimestampFrequency()
        };
        Self {
            heap,
            next_index: std::cell::RefCell::new(0),
            capacity: Self::CAPACITY,
            tick_hz,
            entries: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Write one relaxed-granularity timestamp after the dispatch the
    /// caller just encoded, remembering its kernel id + grid for the
    /// printout. No-op (silently) if the heap is full — the probe must
    /// never fail a forward.
    pub fn write_timestamp(
        &self,
        enc: &ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        kernel: super::lowered::KernelId,
        grid: MTLSize,
    ) {
        let idx = *self.next_index.borrow();
        if idx >= self.capacity {
            return;
        }
        unsafe {
            use ::objc2_metal::MTL4ComputeCommandEncoder as _;
            enc.writeTimestampWithGranularity_intoHeap_atIndex(
                // Precise, not Relaxed: measured on M1 Max / macOS 26,
                // Relaxed resolves an all-zero heap (the driver batches
                // per-pass and apparently writes nothing for a
                // dispatch-heavy compute encoder). The probe accepts
                // the perturbation cost; it is opt-in.
                ::objc2_metal::MTL4TimestampGranularity::Precise,
                &self.heap,
                idx,
            );
        }
        *self.next_index.borrow_mut() += 1;
        let grid_str = format!("{}x{}x{}", grid.width, grid.height, grid.depth);
        self.entries.borrow_mut().push((kernel, grid_str));
    }

    /// Resolve the heap and print per-kernel deltas + aggregates for
    /// the timestamps written so far, then reset the write cursor so
    /// the next forward starts fresh. Caller must have drained the GPU
    /// (the pool's event wait) — `resolveCounterRange` is only correct
    /// on the CPU timeline after the writers finish.
    pub fn print_tape(&self, header: &str) {
        use ::objc2_foundation::NSRange;
        let n = (*self.next_index.borrow()).min(self.capacity);
        if n == 0 {
            return;
        }
        let data = unsafe {
            use ::objc2_metal::MTL4CounterHeap as _;
            self.heap.resolveCounterRange(NSRange::new(0, n))
        }
        .expect("resolveCounterRange returned nil");
        let len = data.length() / std::mem::size_of::<u64>();
        let mut stamps = Vec::with_capacity(len);
        let mut raw = vec![0u64; len];
        if len > 0 {
            unsafe {
                data.getBytes_length(
                    std::ptr::NonNull::new_unchecked(raw.as_mut_ptr() as *mut _),
                    raw.len() * std::mem::size_of::<u64>(),
                );
            }
        }
        stamps.append(&mut raw);
        // Convert ticks → ms via the queried clock frequency. A zero
        // stamp (unwritten entry) reads as 0 and is skipped by the
        // saturating delta below, collapsing to a 0-duration row
        // rather than a bogus huge one.
        let ms = |ticks: u64| ticks as f64 / self.tick_hz as f64 * 1e3;
        eprintln!(
            "[kernel-time] {header}: {} dispatches (resolved {} stamps)",
            n,
            stamps.len()
        );
        // Per-kernel aggregates: (count, total ns, max ns).
        use std::collections::BTreeMap;
        let mut agg: BTreeMap<String, (u64, u64, u64)> = BTreeMap::new();
        // Slowest single dispatches, for the grid-shape question: (ticks,
        // kernel, grid string), printed top-first.
        let mut slowest: Vec<(u64, super::lowered::KernelId, String)> = Vec::new();
        {
            let entries = self.entries.borrow();
            for i in 1..n.min(stamps.len()) {
                let dt = stamps[i].saturating_sub(stamps[i - 1]);
                // Stamp i is written AFTER dispatch i in the encoder stream,
                // so the delta between stamp i-1 and stamp i measures the
                // kernel dispatched AT i (it started once i-1's work drained).
                let (kernel, grid) = &entries[i];
                let key = format!("{kernel:?}");
                let e = agg.entry(key).or_insert((0, 0, 0));
                e.0 += 1;
                e.1 += dt;
                e.2 = e.2.max(dt);
                slowest.push((dt, *kernel, grid.clone()));
            }
        }
        let mut total: u64 = 0;
        for (kernel, (count, sum_ticks, max_ticks)) in &agg {
            total += sum_ticks;
            eprintln!(
                "[kernel-time]   {kernel:<42} n={count:<4} sum={:>9.3} ms  max={:>9.3} ms",
                ms(*sum_ticks),
                ms(*max_ticks),
            );
        }
        eprintln!("[kernel-time]   TOTAL {:>9.3} ms", ms(total));
        slowest.sort_unstable_by_key(|s| std::cmp::Reverse(s.0));
        for (ticks, kernel, grid) in slowest.iter().take(12) {
            if *ticks == 0 {
                break;
            }
            eprintln!(
                "[kernel-time]   slow {kernel:?} {grid} {:>9.3} ms",
                ms(*ticks)
            );
        }
        *self.next_index.borrow_mut() = 0;
        self.entries.borrow_mut().clear();
    }
}

/// Build one `Mtl4Step` per `BucketStep::Dispatch`. Returns `None` if any
/// step is a `Gemm` (MPS) or any kernel exceeds the argument-table
/// binding cap — such a bucket has no MTL4 plan (the pool asserts
/// `mtl4_steps.is_some()` at forward time).
pub fn bake_mtl4_steps(steps: &[BucketStep], device: &Device) -> Option<Vec<Mtl4Step>> {
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            BucketStep::Dispatch {
                kernel,
                pipeline,
                direct_bindings,
                direct_dispatch,
                direct_m_scaling,
                barrier_before,
                runtime_gate,
                ..
            } => {
                debug_assert_eq!(direct_bindings.len(), direct_dispatch.len());
                debug_assert_eq!(direct_bindings.len(), direct_m_scaling.len());
                debug_assert_eq!(direct_bindings.len(), barrier_before.len());
                debug_assert_eq!(direct_bindings.len(), runtime_gate.len());
                // Fail-fast occupancy guard (model-load time). A dispatch that
                // requests more threads/threadgroup than the pipeline's
                // maxTotalThreadsPerThreadgroup is illegal: the driver silently
                // under-launches and returns WRONG results with no error — this
                // exact bug shipped for the hd256/512 `attention_via_cache_v2`
                // decode kernel (needs 1024 threads; M1's register budget capped
                // its pipeline at 640, so 12 of 32 simdgroups never ran and the
                // softmax combine read uninitialized threadgroup memory). We can
                // never make this a rustc/metal *compile*-time error because
                // maxTotalThreadsPerThreadgroup is a per-GPU-family RUNTIME
                // property (the offline `xcrun metal` emits family-agnostic AIR),
                // but we surface it here, LOUDLY, at bake — never mid-generation
                // silently. A kernel that legitimately needs a large launch must
                // carry `[[max_total_threads_per_threadgroup(N)]]` so the driver
                // guarantees the launch (spilling registers if needed) instead of
                // silently truncating it.
                {
                    use objc2_metal::MTLComputePipelineState as _;
                    let cap = pipeline.maxTotalThreadsPerThreadgroup();
                    for (_, tpt) in direct_dispatch {
                        let req = tpt.width * tpt.height * tpt.depth;
                        assert!(
                            req <= cap,
                            "kernel {kernel:?}: dispatch requests {req} threads/threadgroup \
                             ({}x{}x{}) but its pipeline maxTotalThreadsPerThreadgroup is {cap} \
                             — the GPU would under-launch and silently corrupt output. Add \
                             [[max_total_threads_per_threadgroup({req})]] to this kernel (or \
                             reduce its threadgroup size).",
                            tpt.width,
                            tpt.height,
                            tpt.depth,
                        );
                    }
                }
                let mut tables = Vec::with_capacity(direct_bindings.len());
                for cmd_bindings in direct_bindings {
                    let max_idx = cmd_bindings
                        .iter()
                        .map(|(_, _, idx)| *idx as usize)
                        .max()
                        .unwrap_or(0);
                    if max_idx + 1 > MTL4_MAX_BUFFER_BINDS {
                        return None;
                    }
                    let desc = MTL4ArgumentTableDescriptor::new();
                    desc.setMaxBufferBindCount(max_idx + 1);
                    let table = device.newArgumentTableWithDescriptor_error(&desc).ok()?;
                    for (buf, off, idx) in cmd_bindings {
                        // GPU virtual address + caller-supplied byte
                        // offset, bound directly into the argument
                        // table.
                        let addr = buf.gpuAddress() + *off;
                        unsafe {
                            table.setAddress_atIndex(addr, *idx as usize);
                        }
                    }
                    tables.push(table);
                }
                out.push(Mtl4Step {
                    kernel: *kernel,
                    pipeline: pipeline.clone(),
                    tables,
                    dispatches: direct_dispatch.clone(),
                    m_scaling: direct_m_scaling.clone(),
                    barrier_before: barrier_before.clone(),
                    runtime_gate: runtime_gate.clone(),
                });
            }
        }
    }
    Some(out)
}
