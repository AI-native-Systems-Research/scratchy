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
//! Dispatch steps — there is no MPS / classic command-buffer path.)

use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{MTL4ArgumentTable, MTLBuffer};

use super::__re::{ComputePipelineState, Device, MTL4ArgumentTableDescriptor, MTLDevice, MTLSize};
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
    /// One barrier-before flag per sub-dispatch (parallel to
    /// `tables` / `dispatches`). Sourced from the macro-emitted
    /// `LoweredMetalTape::barrier_before` — no runtime analysis.
    /// `true` means the runtime must emit a `Dispatch→Dispatch`
    /// MTL4 encoder barrier before this sub-dispatch.
    pub barrier_before: Vec<bool>,
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
