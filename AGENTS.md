# PROJECT KNOWLEDGE BASE

**Generated:** 2026-10-27
**Commit:** acc4046b
**Branch:** docs/agents-md

## OVERVIEW
Rust LLM inference engine. A Python-DSL model + weights compile at build time into static instruction tapes. These are played on metal (Apple, Metal 4), cuda, or spyre (IBM). Serving is a vLLM port.
Invariants, model-crate features, build and CI gates: see `CLAUDE.md`. This file doesn't repeat them.

## STRUCTURE
```
./
├── crates/compiler/     # build-time forward pipeline: macros (NOT a proc-macro), ir, subtile
├── crates/models/       # arch (DSL + configs/<arch>/<stem>.json), quantization, spec, vision
├── crates/targets/      # metal | cuda | spyre (+ spyre/ktir, spyre/flex-rs)
├── crates/triton/       # Triton .py -> TTIR -> KTIR -> SuperDSC ladder (re-hosted from triton-spyre)
├── crates/serving/      # scheduler, transport -> engine -> worker -> api
├── crates/core/         # common, config, hf-hub-downloader, model (weights, baked tokenizer)
├── crates/layers/       # backend-neutral layer types, KV/GDN pools, TurboQuant
├── crates/tensors/      # GpuTensor / DeviceAllocator seams + cfg-neutrality guard
├── crates/cli/          # `scr` binary (scr/) + out-of-workspace TUI (tui/)
├── crates/benches/      # scratchy-bench (`scr bench`)
├── crates/e2e/          # GPU e2e harness
├── scripts/             # bench, CI, golden, emit-arch tooling
├── site/                # benchmark site (own Cargo.lock, out of workspace)
├── docs/                # BUILD, MODELS, TESTING (authoritative docs)
└── third_party/, tools/ # vendored flash-attn shims; small leaf utilities
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add model / arch | `crates/models/arch/`, `docs/MODELS.md` | then the checklist in `crates/models/arch/dsl/CLAUDE.md` |
| Arch build pipeline / feature tables | `crates/models/arch/scratchy-forwards.rs` | `CUDA_ONLY`/`METAL_ONLY`/`SPYRE_CAPABLE` |
| Quant preset | `crates/models/quantization/presets/`, `configs/<arch>/quantizations.json` | |
| Forward codegen | `crates/compiler/macros/src/` | quick check: `scripts/emit-arch.sh` |
| New opcode | `crates/compiler/ir/src/lib.rs` `Instruction` | + one emitter arm per target |
| Shared passes (reroll, colouring, KV codec) | `crates/compiler/subtile/` | target ABI = const tables |
| Metal lowering / runtime / shaders | `crates/targets/metal/src/tape/`, `src/interpreter/metal/`, `shaders/` | |
| Metal macro-time tapes | `crates/targets/metal/compiler/` | |
| CUDA kernels / profiles | `crates/targets/cuda/csrc/`, `crates/targets/cuda/profiles/` | |
| Spyre lowering / bake | `crates/targets/spyre/src/` | KTIR -> SuperDSC in `ktir/ktir-superdsc` |
| Spyre hardware FFI | `crates/targets/spyre/flex-rs/` | sole `extern "C"` boundary |
| HTTP / chat API | `crates/serving/api/` | `serve` feature gates axum |
| Scheduling / KV cache | `crates/serving/scheduler/` | |
| CLI args / subcommands | `crates/cli/scr/src/args.rs`, `crates/cli/scr/src/commands/` | |
| CI jobs | `.github/workflows/rust.yml` | |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `compile_carrier` | fn | crates/compiler/macros/src/lib.rs | pipeline entry, called from the models build script |
| `Instruction` | enum | crates/compiler/ir/src/lib.rs | closed opcode set shared by every target |
| `try_load` / `ScratchyArchRegistration` | fn/struct | crates/compiler/src/arch_registry.rs | inventory-based arch loading |
| `reroll_subtile_tape` | fn | crates/compiler/subtile/src/subtile_tape.rs | shared layer reroll |
| `lower_subtile_tape_to_metal` | fn | crates/targets/metal/src/tape/lowering.rs | SubtileTape -> LoweredMetalTape |
| `METAL_FUSIONS`, `METAL_COLOUR_FACTS` | const | crates/targets/metal/src/op_abi.rs | metal ABI tables for the shared passes |
| `MetalWorker::run_bucket_mtl4` | fn | crates/targets/metal/src/interpreter/metal/worker.rs | production metal dispatch |
| `InstructionEval` / `run` | trait/fn | crates/targets/cuda/src/eval.rs | cuda tape interpreter |
| `lower_graph_to_ktir` | fn | crates/targets/spyre/src/lower_subtile_tape_to_ktir.rs | spyre producer |
| `lower_function` | fn | crates/targets/spyre/ktir/ktir-superdsc/src/emit/whole_function.rs | KTIR -> SuperDSC |
| `make_ktir` | fn | crates/triton/triton-ktir/src/lib.rs | 7-pass KTIR pipeline |
| `EngineCore` | struct | crates/serving/engine/src/engine_core.rs | engine step loop |
| `Scheduler` | struct | crates/serving/scheduler/src/scheduler/core.rs | admission + prefix cache |
| `AsyncEngine` / `initialize_stack` | struct/fn | crates/serving/api/src/{engine,init}.rs | HTTP<->engine bridge, stack wiring |
| `GpuTensor`, `DeviceAllocator` | type/trait | crates/tensors/src/ | device seams |

## CONVENTIONS
- Dep direction: macro -> forward-compiler -> ir. Never the reverse. Target crates never depend on each other: `ktir-emulator/src/metal/mtl4.rs` copies `mtl4_dispatch`, so fix bugs in both.
- Feature exclusivity: `cuda`/`metal`/`spyre` give a `compile_error!`. `spyre-hw` (card) and `spyre-emu` (emulator) are siblings, and neither is a fallback for the other.
- Workers register via `inventory::submit!` and are kept alive by `extern crate ... as _` anchors. Removing an anchor leaves an empty registry.
- Generated code names `scratchy_target_*` module paths verbatim. Renaming them breaks codegen.
- Serving and `crates/triton` are ports: keep the SPDX header + "Port of:" docs and `PROVENANCE.md`.
- KTIR is constructed, never parsed. Each Triton ladder stage passes values. Refusals fail closed.

## ANTI-PATTERNS (THIS PROJECT)
- `.commandBuffer()`: Metal 4 only (banned in `clippy.toml` and by the metal build.rs).
- `ktir-superdsc` must not gain deps/features beyond ktir-core, half, serde, serde_json. flex-rs or objc2 in a cuda/metal/emit-only build breaks CI.
- Never compile spyre/triton crates in macOS jobs (`scripts/ci-macos-scope.sh lock`).
- Never build the `tool-spans` api feature (known broken). Never hand-edit `site/data` JSON, marlin generated kernels, or `crates/serving/api/src/chat_template/hf_renders.rs`.
- No per-model tokenizer lists or benchmark hacks (see dsl CLAUDE.md).
- Existing crate-level `#![allow]`s (cuda, tensors, metal lib, macros/subtile `dead_code`) are legacy and set no precedent.
- A missing golden is a failure, never a skip.

## COMMANDS
```bash
scripts/emit-arch.sh <arch> <stem>              # emit any forward, no backend needed
cargo test -p scratchy-forward-compiler --lib   # --tests is stale (removed #[forward])
cargo test -p scratchy-forward-compiler-macro --features metal --lib
cargo test -p scratchy-subtile [--features spyre]
cargo test -p scratchy-target-metal             # self-skips without a Metal 4 GPU
cargo test -p scratchy-target-spyre --features spyre-emu --lib --tests
SCRATCHY_SKIP_SENDNN_CXX=1 cargo check -p flex-rs
cargo fmt --all -- --check; cargo deny check bans
python3 -m unittest discover -s scripts/tests
```

## NOTES
- Adding an arch DSL: bump the count in `crates/compiler/macros/tests/arch_declarations.rs`.
- `SCRATCHY_SKIP_CUDA_KERNELS=1` / `SCRATCHY_SKIP_SENDNN_CXX=1` are for check/clippy only. Linking fails with them.
- Spyre bake on a host without dxp needs `SCRATCHY_PLAN_ONLY_BAKE=1`.
- macOS SDK >= 26.2 is required (metal build panics otherwise).
- Kani for subtile needs `--features spyre`. Without it, Kani silently finds zero harnesses.

## CHILD AGENTS.md
- CI/docs/tooling: `.github/`, `docs/`, `scripts/`, `site/`
- crates: `benches/`, `cli/`, `core/`, `e2e/`, `layers/`, `tensors/`, `models/`, `models/arch/`
- compiler: `crates/compiler/`, `compiler/macros/`, `compiler/subtile/`
- serving: `crates/serving/`, `serving/{api,engine,scheduler,worker}/`
- metal: `crates/targets/metal/`, `metal/compiler/`, `metal/shaders/`, `metal/src/tape/`, `metal/src/interpreter/metal/`
- cuda: `crates/targets/cuda/`, `cuda/csrc/`
- spyre: `crates/targets/spyre/`, `spyre/flex-rs/`, `spyre/ktir/`, `ktir/{ktir-emulator,ktir-optimizer,ktir-superdsc}/`
- triton: `crates/triton/`, `triton/{splice,triton-frontend,triton-ktir,triton-ktir-superdsc,triton-numeric}/`
