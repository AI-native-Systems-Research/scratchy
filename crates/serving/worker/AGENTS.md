# scratchy-serving-worker

## OVERVIEW
Device workers + executors: one `MetalWorker`/`CudaWorker` in `gpu_worker.rs`, the Spyre worker split across `spyre_*.rs`, uniproc/threadpool/multinode executors, TP groups. Score ~20 (18 files, ~18.5k LOC, backend-gated).

## WHERE TO LOOK
| Task | File |
|------|------|
| Metal/CUDA step, KV/GDN pools, chain layout, spec-decode backend | `src/gpu_worker.rs` (8.3k LOC; `MetalWorkerFactory`, `SeqMmInfo`) |
| Pre-init Metal sizing guard (used by api `init`) | `src/metal_info.rs` |
| Executors | `src/uniproc.rs`, `src/threadpool.rs`, `src/multinode.rs` (`MultiNodeExecutor`, `ControlMessage`) |
| TP/PP group resolution | `src/parallel.rs` (`ParallelGroup`, `ResolvedParallelConfig`) |
| Spyre | `spyre_worker.rs` (lookup), `spyre_types.rs` (nouns), `spyre_load.rs` (staging), `spyre_forward.rs` (tape play), `spyre_pool.rs` (paged KV), `spyre_exec.rs` (step loop) |
| `error`/`worker`/`worker_factory`/`input_batch` | re-export shims over serving-engine; edit engine instead |

## CONVENTIONS
- Module gates: `gpu_worker`/`gpu_worker_base` = `any(cuda, metal)`; `metal_info` = `metal`; `spyre_*` = `any(spyre, spyre-hw)`. `gpu_worker.rs` is cfg-mutexed: nearly every item carries `#[cfg(feature = "metal")]`/cuda, and imports follow the cfg of the code that uses them.
- `build.rs` emits `cfg(fa3_built)` when `CUDA_ARCH>=90` or nvidia-smi reports sm_90+. It mirrors `scratchy-builder-cuda`'s `detect_cuda_arch`, so change both together.
- `extern crate scratchy_builder_cuda as _` exists only to run its build.rs (`.a` link). Not dead code.
- Spyre rides the neutral seam (scratchy-layers `GpuWeights`, scratchy-tensors `DType`, scratchy-subtile `DeviceWidth`) and NEVER touches target-cuda.
- `spyre-hw` must not pull `spyre-emu`/ktir-emulator (objc2 breaks Linux card builds); gate per path, "NOT `cfg(not(spyre-hw))`".
- Every raw MTLBuffer/arena pointer use carries a `// SAFETY:` comment; keep that up when adding unsafe.

## ANTI-PATTERNS
- Spyre: NOTHING ANALYSES A MODEL. Wiring is a macro-emitted `static` and weights are generated `BoundWeight`s; the worker is not where a model lives (inventory lookup only).
- Device regions are never freed (`DevAddr`). The KV extent is the bake's reserve, never the session's `cap`.
- A missing factor is a refusal, never a zero table. Finish order in `spyre_exec` is a capacity invariant, not tidiness.

## COMMANDS
```bash
cargo test -p scratchy-serving-worker                  # CI (no backend)
cargo test -p scratchy-serving-worker -F metal         # macOS gpu_worker tests
cargo test -p scratchy-serving-worker -F spyre-emu     # spyre_load/exec tests on host
```
