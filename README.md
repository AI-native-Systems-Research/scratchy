# scratchy

A Rust LLM inference stack built around the **scratchy whole-forward
[DSL](https://en.wikipedia.org/wiki/Domain-specific_language)
compiler** — a procedural-macro compiler that turns a model's forward
pass, written as ~50 lines of [declarative
pseudocode](crates/models/arch/dsl), into specialized, fused,
workload-aware GPU dispatch code for NVIDIA CUDA, Apple Metal, and
[IBM Spyre AIU](https://research.ibm.com/blog/spyre-for-z). In
scratchy, everything is a constant. This turns complex analysis into
arithmetic, greatly simplifying our code. It also can enable more
constant-prop optimizations, reducing register pressure in key
kernels.

The repository contains three ecosystems under `crates/`:

- **Whole-forward DSLs** for [many modern
    architectures](crates/models/arch/dsl).
- **A [whole-forward DSL compiler](docs/COMPILER.md)** that
    macro-expands a DSL into a target-specific instruction tape and
    target-specific weight loaders.
- **An [inference serving stack](crates/serving)** — engine,
  scheduler, executor, OpenAI-compatible server, etc.

## Building

Install a recent [Rust toolchain](https://rustup.rs/). Then, pick your
target via [feature
flag](https://doc.rust-lang.org/cargo/reference/features.html):
`-Fcuda` requires the CUDA build toolkit; `-Fmetal` requires macOS;
`-Fspyre` requires the Spyre build toolkit.  For example, you can
compile support for Apple Silicon, the Llama 3.2 3B models, and the
MLX 4-bit quant:

```bash
cargo build -F metal,model/llama-3.2-3b,quant/mlx --release
```

This builds a binary in `target/release/scr`. By default, only the
simple `scr chat` CLI command is compiled in. Provide `-Fserve` or
`-Fbench` to bring more features into the binary.

Note the convention for selecting models and quants:
- `model/<stem>` — one checked-in model config, e.g. `model/qwen2.5-7b`
  (`model/<arch>` for every config in an arch, `model/all` for everything).
- `quant/<preset>` — compiles that quantization instead of dense/bf16
  (e.g. `quant/mlx` for every MLX affine int4 preset at once).

See [`docs/BUILD.md`](docs/BUILD.md) for the full feature-scoping
mechanics.

## Repository layout

```
crates/
  # ── whole-forward DSL compiler ──────────────────────────────
  compiler/                scratchy-forward-compiler: shared #[forward]/
                            #[vision_forward] pipeline (op tape, reroll,
                            layer classes, codegen) — target-neutral
  compiler/macros/         scratchy-forward-compiler-macro: the pipeline
                            driver scratchy-models' build script calls per arch
  compiler/ir/             scratchy-ir: backend-neutral Instruction enum +
                            WeightAccessors dispatch trait
  compiler/subtile/        scratchy-subtile: SubtileIR — the shared substrate
                            metal and spyre both lower from
  compiler/sdsc/           scratchy-sdsc: SuperDSC (Spyre work-divided)
                            lowering support
  layers/                  scratchy-layers: layer types (RmsNorm,
                            LinearLayer, …), GpuWeights
  tensors/                 scratchy-tensors: GpuTensor/OwnedTensor, dtypes,
                            device handles

  models/arch/             scratchy-models: ONE crate holding every model
                            architecture (llama, qwen2/3, gemma2/3/4,
                            mistral, mixtral, phi3, granite, commandr,
                            deepseek-v2/v3, …), gated by arch-<name> Cargo
                            features; each config gated by its own <stem>
  models/vision/           scratchy-vision: vision-tower host glue (RoPE
                            tables, cu_seqlens, patch flatten)
  models/quantization/     scratchy-quantizations: shared quantization
                            preset definitions, each gated by its own
                            <preset> Cargo feature

  targets/cuda/            scratchy-target-cuda: CUDA GPU runtime, kernel
                            FFI wrappers, layer impls, empirical cost tables
  targets/cuda/builder/    scratchy-builder-cuda: build-time .cu generation
                            + nvcc compilation
  targets/cuda/cost-sweep/ scratchy-cost-sweep-cuda: empirical CUDA
                            cost-table generation
  targets/metal/           scratchy-target-metal: Metal kernels, impl lib,
                            target profiles for Apple M1-M4
  targets/spyre/           scratchy-target-spyre: IBM Spyre/KTIR host runtime
  targets/spyre/builder/   scratchy-builder-spyre: build-time SuperDSC/
                            sendnn bake
  targets/spyre/bundle/    scratchy-spyre-bundle: the baked-bundle type
                            family (SuperDSC/KTIR)

  # ── Rust vLLM serving stack ─────────────────────────────────
  serving/scheduler/       scratchy-serving-scheduler
  serving/engine/          scratchy-serving-engine
  serving/worker/          scratchy-serving-worker: the GPU worker
                            (metal/cuda/spyre)
  serving/api/             scratchy-serving-api: OpenAI/Anthropic HTTP
                            (axum) server + the in-process engine
  serving/transport/       scratchy-serving-transport

  # ── CLI ─────────────────────────────────
  cli/scr/                 scratchy-cli: the `scr` binary
  cli/tui/                 scratchy-tui: `scr-tui`, a standalone interactive
                            terminal UI binary (own workspace — pulls goose
                            from git; not part of the root workspace build)

  # ── Common ─────────────────────────────────
  core/common/ core/config/ core/model/    shared types, config parsing,
                            weight/config plumbing
  core/hf-hub-downloader/  hf-hub-downloader: HF Hub model download/cache
  e2e/                     scratchy-e2e: correctness goldens vs. Python vLLM
  benches/                 benchmarking tools
```
