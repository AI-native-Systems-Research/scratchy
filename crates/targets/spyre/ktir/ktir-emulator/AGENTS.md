# ktir-emulator — KTIR interpreter (CPU + Metal)

## OVERVIEW
Rust port of Python `ktir_cpu`. Executes KTIR modules with an LX/HBM memory model, a latency model, and GPU/BLAS offload. Score ~20 (largest crate, own build.rs, shaders, features, ignored timing suites).

## WHERE TO LOOK
| Need | Location |
|------|----------|
| Run an op/region/function | `src/interpreter.rs` (`execute_op`, `execute_region`, `execute_function_*`) |
| Turnkey fused/resident run | `src/program.rs` (`execute`, `Session`); `resident.rs`, `segmented.rs`, `resident_runner.rs` (all behind `optimizer`) |
| Add/modify an op handler | `src/dialects/<dialect>.rs`, registered through `register(&mut Dispatch)` in `dialects/mod.rs` |
| Cross-core comm | `src/comm.rs` (seam), `src/comm_sched.rs`, `dialects/ktdp_comm.rs` |
| LX / HBM / core state | `src/machine_state/{memory,context}.rs` (`SpyreMemoryHierarchy`, `CoreContext`, `LxOptions`) |
| Latency categories | `src/latency.rs`; register them in lockstep with handlers |
| MSL emit + GEMM tiers | `src/metal.rs` (~7.3k LOC; `MatmulTier` Naive/Simdgroup/Nax, `choose_matmul_backend`) |
| MTL4 dispatch | `src/metal/mtl4.rs`, a deliberate copy of `scratchy_target_metal::mtl4_dispatch` (target crates must not depend on each other) |
| AOT shaders | `shaders/*.metal` + `build.rs` (14 metallibs) |
| Test op builder | `src/test_support.rs`; its twin is `tests/common/mod.rs`, so keep both in sync |

## CONVENTIONS
- Root re-exports every ktir-core module plus `ktir_optimizer` (feature `optimizer`, default on). Module names mirror the Python files to keep them diffable.
- Handler sig: `(op, &mut CoreContext, &ExecutionEnv) -> Result<Option<Value>, String>`. Registration is an explicit table built once (`Dispatch::shared()` OnceLock), not macros.
- Nested regions go through `execute_region` and never suspend; comm ops are top-level only.
- SSA is a flat slot table with an undo `trail` for region scoping. Shared state is `Rc<UnsafeShared<..>>`, not Arc/Mutex.
- `build.rs` emits `cfg(metal)` on macOS automatically and `cfg(metal_aot)` only after a probe compile succeeds. The `metal` module builds everywhere; its objc2 halves gate on `cfg(metal)`.
- BLAS: Accelerate on macOS; features `openblas|openblas-system|mkl|blis`. Naive GEMM is the oracle, and `blas_src` is referenced once at the crate root.

## ANTI-PATTERNS
- Dropping `-mmacosx-version-min=26.2` from the shader compile. SDK 26.5 miscompiles MPP `matmul2d` to half-K (MLX #3622).
- Renaming a shader variant without updating the `include_bytes!` stems in metal.rs. AOT output must be byte-equivalent to JIT.
- Unwrapping a missing AOT variant. It returns Err and falls back to JIT.
- Caching pipelines, or re-marshaling resident weights. Weights are never re-marshaled once resident, so a stale one can't be served.
- m>1 GEMM silently staying on CPU: it always offloads, and an Err there is fatal.
- MTL4: `setBuffer`/`setBytes` (use the residency set plus gpuAddress argument tables); `MTLCommandQueue::commandBuffer` (clippy-banned); a missing `barrier()` between dependent encodes; `useResidencySet` before the set is committed.
- Changing the LX cache away from the modelled 2 MB (`LX_CAPACITY_MB`).
- Using `UnsafeShared` beyond disjoint or read-only access.
- Matching more than one tile group in `find_tile_group`. That breaks the disjointness invariant.

## COMMANDS
```bash
cargo test -p ktir-emulator                       # in-file dialect unit tests + port_* suites
cargo test --release -p ktir-emulator --features metal --test metal_conformance -- --ignored --nocapture
KTIR_DIFF_INJECT_DIVERGENCE=0.05 ...metal_conformance   # negative control: MUST fail
```
- Ignored suites: `metal_conformance` (needs `uv`), `gpu_dispatch_floor`, `flash_attn_timing`, `bench_amx_vs_metal`.
- Debug knobs: `KTIR_DIFF_RESIDENT`, `KTIR_DIFF_GPU`, `KTIR_FORCE_GPU_GEMM`, `KTIR_FORCE_GPU_MAP`, `KTIR_MAP_GPU_MIN_ELEMS`, `KTIR_FORCE_FUSE_ATTN`.
- Examples: `bench`, `ktir_diff_run`, `mtl4_smoke`. Bench `smollm2_bench` needs `~/.cache/cudaforge/ktir/<MODEL>` and is stale: it imports the nonexistent `ktir_emulator::parser::parse_module`.
