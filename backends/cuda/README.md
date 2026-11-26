# The CUDA backend

scratchy on an NVIDIA H100: the `cuda` CLI, the same commands as `scr` and `spyre`, served through
the CUDA worker. This directory is its own Cargo workspace (its own `Cargo.lock` and `target/`),
reaching the shared crates by path into the root's `crates/`; nothing in the root workspace depends
on it.

```
kernels/           every kernel (`*.cu`), the op header (`scratchy_ops.cuh`), and ThunderKittens
                   vendored at kernels/thunderkittens/ (its LICENSE and PROVENANCE.md: the pinned
                   commit)
crates/cuda/       scratchy-target-cuda: the ABI tables the shared passes take as input, the step
                   and tape types, the weight table, and under `nvcc` the kernels, their launchers
                   and the runtime that plays a tape
crates/compiler/   scratchy-target-cuda-compiler: CUDA's emitter (`Cuda`, the shared pipeline's
                   `TargetEmitter`) and its opcode lowering, run at build time
crates/worker/     scratchy-target-cuda-worker: the `WorkerFactory` the engine serves through
crates/models/     scratchy-target-cuda-models: the selected models, compiled with CUDA's emitter
crates/cli/        scratchy-target-cuda-cli: the `cuda` binary
scripts/           the H100 drivers (scripts/README.md)
```

## Build

CUDA builds FP8 models only: name `quant/fp8-dynamic-per-channel` with one of these models, each an
FP8 dynamic per-channel checkpoint. Any other scope panics at build time, naming why.

| feature | checkpoint |
|---|---|
| `model/qwen3.8-27b-text-only` | `axteam/Qwen3.8-27B-FP8-dynamic` |
| `model/qwen3.6-35b-a3b` | `RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic` |
| `model/gemma-4-26b-a4b-it` | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` |

```bash
cd backends/cuda
cargo build --release --features nvcc,serve,model/qwen3.8-27b-text-only,quant/fp8-dynamic-per-channel
target/release/cuda serve /path/to/Qwen3.8-27B-FP8-dynamic --kv-cache-dtype fp8
```

The `nvcc` feature compiles `kernels/*.cu` for `sm_90a` (WGMMA and TMA) with `-std=c++20` and
links them. It needs CUDA 12.8+ `nvcc` on `PATH`, and the binary an H100. Without it the workspace
builds on any host with no CUDA toolchain, and `cuda` refuses to start a worker, naming the
feature. The worker samples greedily only, and refuses any other sampling request when it is
submitted.

## Check

What CI runs (the `linux-workspace` and `linux-cuda` jobs), with no toolchain and no GPU:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets \
  --features scratchy-models/qwen3.8-27b-text-only,scratchy-quantizations/fp8-dynamic-per-channel -- -D warnings
cargo test -p scratchy-target-cuda -p scratchy-target-cuda-compiler
cargo test -p scratchy-target-cuda --test kernels_use_thunderkittens
cargo test -p scratchy-target-cuda-models \
  --features scratchy-models/qwen3.8-27b-text-only,scratchy-quantizations/fp8-dynamic-per-channel \
  --test cuda_tapes_27b_fp8    # and cuda_tapes_35b_fp8, cuda_tapes_gemma4_fp8 at their models
```

The kernels' own tests (`--features nvcc`) need an H100, so CI never builds them: the H100 run
(`scripts/kernels.sh`, `scripts/chain.sh`) is their only gate.
