# src/interpreter/metal/

## OVERVIEW
Runtime execution: `MetalWorkerPool` checks out workers, each with its own arena and baked MTL4 steps, and drives one forward per checkout. Consumers import it as `scratchy_target_metal::interpreter::metal::*`, and the macro codegen emits those paths verbatim. Score ~17 (12k LOC, high centrality).

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Pool, rung picking, fold bands, MTL4 probe | `pool.rs` (`MetalWorkerPool`, `MetalBucketSpec`, `MetalRungs`, `pick_rung`, `FoldBand`, `RuntimeFactory`, `host_guest`) |
| Worker arena, bucket baking, production dispatch | `worker.rs` (`MetalWorker`, `HostArena`/`GuestArena`, `BucketStep`, `run_bucket_mtl4` ~l.1153) |
| Function-constant index contract per kernel, GEMM/GEMV selection | `pipelines.rs` module doc; `SpecializedPipelines`, `GemmBody`, `GEMV_ROWS`, `GEMM_BLOCKED_ROWS`, `NAX_GEMM_MIN_M` |
| Forward surface, async delivery | `forward.rs` (`ForwardInputs`, `ForwardReads`, `InFlight`, `Deferral`, `CommitFeedback`) |
| MTL4 step baking, kernel timing heap | `mtl4.rs` (`bake_mtl4_steps`, `MTL4_MAX_BUFFER_BINDS = 31`) |
| Runtime-bound buffers (turboquant scratch, padding) | `runtime.rs` `RuntimeBindings` |
| objc2 type aliases for emitted code | `mod.rs` `#[doc(hidden)] pub mod __re` |

## CONVENTIONS
- The remaining small files (`lowered`, `ids`, `kernel_*`, `continuation_witness`) re-export `crate::tape`. Edit the source there.
- Every per-layer scalar (eps, attn_scale, strides) is a `CanonicalParams` constant from the model JSON. There's no runtime extras struct.
- Selection thresholds are named consts with a measured rationale in their doc comments. Change the const and its doc together.
- The pool is deliberately sync (`Mutex`+`Condvar`): the host side is sync, so async buys nothing.
- The memory budget (`max_workers`) is computed by the caller/macro, not the pool.
- `FoldBand::{Narrow,Wide}` is keyed by silicon `GenClass` through `op_abi::METAL_FUSIONS{,_NARROW}`.
- Tests are inline (pipelines/pool/worker hold ~75 `#[test]`s). `forward-telemetry` gates extra worker paths.

## ANTI-PATTERNS
- One forward per checked-out worker. Workers, arenas and execution plans are never shared.
- A kernel needing more than 31 argument-table binds makes its bucket MTL4-ineligible. Design bindings under the cap.
- `Send`/`Sync` impls on objc2 handles need a `// SAFETY:` comment (see forward.rs, pool.rs). Closures capture only `Retained` handles + POD.
- Never add a second dispatch path. `run_bucket_mtl4` is production, and `pool.rs` panics if `newMTL4CommandQueue()` is nil.
