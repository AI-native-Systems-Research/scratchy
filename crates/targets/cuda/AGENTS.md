# crates/targets/cuda — `scratchy-target-cuda` (+ builder, cost-sweep)

## OVERVIEW
CUDA runtime: GPU tensor/device types, kernel FFI wrappers, layers, the `InstructionEval`
interpreter, serving worker, and the GPU cost profiles the cuda ISel solver reads. It earned its
own file because it's the largest single-crate domain (~58k LOC in `src/`, 3 workspace crates here).

## STRUCTURE
```
cuda/
├── build.rs      # LINK only: emits cfg(fa3_built) + rustc-link-lib for the kernel .a files
├── builder/      # scratchy-builder-cuda: COMPILE only (nvcc via cudaforge); no runtime code
├── cost-sweep/   # scratchy-cost-sweep-cuda: bin gpu_cost_sweep -> profiles/*.csv
├── csrc/         # CUDA C++ kernels (see csrc/AGENTS.md)
├── profiles/     # cost_{h100_sm90,l40s_sm89,l4_sm89}.csv (solver CostTable input)
├── src/          # crate source; lib.rs is the only entry
└── tests/        # nvidia_smi_round_trip.rs, spyre_weight_load.rs (self-skip without hw/cache)
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| New op's cuda arm | `src/eval.rs` (`InstructionEval` match) |
| Kernel FFI wrapper | `src/kernels.rs` (13.6k LOC, one wrapper per `extern "C"` launcher) |
| Graph capture/replay, bucket pick, SWA/GDN | `src/cuda_worker.rs` (7.5k) |
| MoE / quant layers | `src/layers_moe.rs`, `src/layers_quant.rs`, `src/weights_quant.rs` |
| GPU profile select (`SCRATCHY_GPU`) | `src/targets.rs` (`ALL`, `for_name`, `detect`) |
| Arch registration rows | `src/arch_registry.rs` (inventory, re-exported `inventory`) |
| FlashInfer tuple set | `builder/src/flashinfer_config.rs` (`FLASHINFER_CONFIG_SET`) |
| nvcc flags, CUTLASS/FlashInfer pins | `builder/build.rs` |
| Cost CSV format | `cost-sweep/src/main.rs` |

## CONVENTIONS
- `cuda` feature is off by default; feature-less mode = core types + `targets::*` profiles,
  because the forward-compiler proc-macro reads profiles at expansion. Don't gate the crate on `cuda`.
- Metal builds never pull this crate (`cargo tree -p scratchy-cli --features metal -i scratchy-target-cuda` is empty).
- Dep edge points UP to `scratchy-forward-compiler`, which keeps the compiler cuda-free.
- Root re-exports (`run`, `run_backbone`, `view`, `tile_ref`, `take_owned`, `load_layered_*`,
  `GpuTensor`, ...) are load-bearing: the generated forward emits `::scratchy_target_cuda::...` paths.
- Kernel libs live in the shared cache `~/.cache/cudaforge/scratchy-serving-cuda/` (the legacy
  name is kept on purpose). It's split from the target crate so Docker caches the nvcc layer.
- `build.rs` reuses `builder/src/flashinfer_config.rs` via `#[path]`, so it's the one source of
  truth for FlashInfer symbol names.
- Arch detection: `CUDA_ARCH` env, then `nvidia-smi compute_cap`, then default sm89.
- Profile key = GPU name (`SCRATCHY_GPU=l4|l40s|h100`, else nvidia-smi name), never compute capability.
- Cost CSV header `kernel,M,N,K,cost_us`; first row is `launch_overhead,...`; all costs are
  compute-only. New families encode extra dims in the kernel name.
- `tests/spyre_weight_load.rs` is `#![cfg(feature = "spyre")]` but this crate defines no `spyre`
  feature, so it's currently compiled out (dead test).
- Tests self-skip (`eprintln!` + return) when hw/cache is missing, never `#[ignore]`.
  `layers_tests.rs` is `#[path]`-included under `cfg(all(feature="cuda", test))`.

## ANTI-PATTERNS
- `SCRATCHY_SKIP_CUDA_KERNELS=1` is for check/clippy ONLY. Anything that links fails on undefined symbols.
- `cfg(fa3_built)` and the FA3 link directive must share `cuda_arch_ge_90()`. Never probe for
  the `.a` on disk (cargo caching desyncs them). Only cost-sweep may probe on disk.
- Adding to `FLASHINFER_CONFIG_SET`: each tuple adds one CUDA TU. An Fp16 tuple needs matching
  extern decls + dispatch in `src/flashinfer.rs` in the same change.
- Bumping `FLASHINFER_COMMIT` means reconciling `builder/templates/*.j2` against upstream jinja.
- A new FA3 shape needs its instantiation file AND the matching `FLASHATTENTION_DISABLE_*` flag cleared.
- The builder must not depend on `scratchy-models` (bootstrap cycle).
- Graph capture: no `cudaFree` before replay, no `fi_plan_new` during capture, no per-call
  `cudaMalloc`/`Free` on the hot path (use preallocated workspace). FA3 workspace + flashinfer plans are never freed.
- Metal-only ops (`AffineEmbed`, `SynthGateUpSiluMul`, `AffineQuant`, `Nvfp4`) stay `unreachable!` here.
- MoE: never double-apply routed scaling. GPTQ/Marlin zero-points are consume-and-drop (vLLM parity).
- `take_to_cpu_f32` removes the tensor, so take each key once. Tied `lm_head` maps to `model.embed_tokens.weight`.
- Cost-sweep timing uses raw cudarc FFI, not crate wrappers, so plan/RAII overhead stays out of the numbers.
- The crate-level `#![allow(...)]` in `lib.rs` predates the no-`#[allow]` rule. Don't add more.

## COMMANDS
```bash
cargo test -p scratchy-target-cuda -F cuda                       # layers_tests + quant tests
cargo test -p scratchy-target-cuda --test nvidia_smi_round_trip
cargo run -p scratchy-target-cuda -F cuda --release --example q5_0_dequant_check [-- --gguf F --tensor T]
CUDA_PATH=/usr/local/cuda-12.9 cargo run -p scratchy-cost-sweep-cuda --features cuda --release \
  --bin gpu_cost_sweep > profiles/cost_<gpu>.csv                 # ATTENTION_SWEEP_ONLY=1 skips GEMM
```
