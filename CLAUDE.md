# Building & verifying scratchy

Rust LLM inference engine. A Python-DSL model + weights compile at build time into static instruction tapes. These are played on metal (Apple, Metal 4), cuda, or spyre (IBM). Serving is a vLLM port.

## Architecture invariants

`dsl_math + weights → weight_loader_fn() + forward_fn() + instruction_tape`

- `forward_fn()` **plays the tape**, injecting the loaded weights. Nothing else.
- **Everything is a compile-time constant.** `#[forward]` runs the whole
  pipeline at expansion — op tape, reroll, layer classes, slot coloring,
  lifetimes, barriers, arena sizes, kernel selection — and emits per-bucket
  `static` tapes of each runtime's own types. No runtime lowering, no runtime
  analysis, no mirror types.
- **Everything common is shared.** One implementation of every target-neutral
  pass, consumed by both spyre and metal. Target-ABI facts (in-place kernels,
  write bindings, symbol pickers, barrier classes) are const tables the shared
  passes take as *input* — never logic, never scattered match arms.
- **Per-target surface = opcode lowering only.** A new arch touches
  metal/spyre only if it introduces a new opcode, and then only one emitter
  arm per target plus one registry row.
- **No instruction selection** — the tape comes straight from the fused
  forward.
- No new env-var behavior toggles; no `#[allow]`; no weakening tests
  ([`docs/TESTING.md`](docs/TESTING.md) says which tests stay tests, and what weakening one means).

## The model crate
All model architectures compile into ONE crate, **`scratchy-models`**
(`crates/models/arch/`); quant presets live on a separate crate,
**`scratchy-quantizations`** (`crates/models/quantization/`) — two crates
because Cargo won't alias one crate under two names, and `scratchy-cli`
needs both `model` and `quant` as short CLI aliases. Speculative-decoding
drafters are a third alias on a third crate, **`scratchy-spec`**
(`crates/models/spec/`): `metal` already compiles the MTP head of every
selected model whose checkpoint carries one, and its default drafter published
as a model of its own (Gemma 4 26B-A4B's assistant at MLX 4-bit g32, under
`quant/mlx`) (`mtp-heads`), and `spec/mtp` adds the other published drafters
(the bf16 and QAT assistants). None has
a default scope: naming zero models is
a **build-time panic**, not a silent empty binary. Every checked-in `configs/<arch>/<stem>.json` gets its own `<stem>`
feature (`model/<stem>` from the CLI, `model/<arch>`/`model/all` to widen);
every quant preset gets its own `<preset>` feature (`quant/<preset>`,
`quant/mlx` for every MLX affine preset at once) — a selected preset
**replaces** a model's dense/bf16 emission, it doesn't add to it. See
[`docs/BUILD.md`](docs/BUILD.md) for the full mechanics.

## Build
```bash
# The CLI you run. Pick ONE backend: metal (macOS) or spyre. Add serve for HTTP.
# Name at least one model — this fails loudly (build-time panic) otherwise.
cargo build --release -p scratchy-cli --features metal,model/llama-3.2-3b
```
- No default model/quant scope, so every build names its own — no
  `--no-default-features` dance needed to narrow to one specific model, just
  name that one model and nothing else. Widen with e.g.
  `--features metal,model/qwen2.5-7b` or `--features metal,model/all`
  (CI's scope).
- **When iterating/verifying a change, scope to exactly the one model you
  need to keep the build fast**: `-Fmodel/<stem>` (e.g. `-Fmodel/llama-3.2-1b`
  or `-Fmodel/smollm2-135m` for the smallest/quickest option), plus
  `-Fquant/<preset>` only if the change is quant-specific. Never build
  `model/<arch>` or `model/all` just to test one thing — that forward-expands
  every config in scope (minutes-to-hours, OOM risk on a laptop).
- `metal` and `spyre` are mutually exclusive.
- Multimodal (vision arches + the `image`/`scratchy-vision` decode stack) is
  opt-in via the `multimodal` feature on `scratchy-cli` (on by default
  there) — a plain `chat`-only build never compiles image/png/zune-jpeg/etc.
- List cached models with `scr model ls`; run a model in-process with
  `scr chat -m <model> --device metal -q "..."` (no image arg — images need the
  HTTP server, i.e. the `serve` feature + `scr serve ... --kv-cache-dtype ...`).

## Verify like CI before opening a build / deps / codegen PR
`.github/workflows/rust.yml`: the **linux-workspace** job runs
`cargo clippy --workspace --features spyre,scratchy-models/granite-3.1-8b-instruct
--exclude scratchy-target-metal --exclude scratchy-target-metal-compiler
-- -D warnings` with `SCRATCHY_SKIP_SENDNN_CXX=1` (the runner has no Spyre SDK;
check/clippy don't link the C++ shim). Both metal crates are excluded because
they depend unconditionally on Apple-only `objc2` — the
compiler crate via its own edge to scratchy-target-metal — and `--workspace`
builds every member as a root regardless of `--features`, so excluding only
one of the two still drags `objc2` onto Linux); the **macOS** job
runs `cargo clippy -p scratchy-models --features metal,all,mtp,turboquant -- -D warnings`
+ kernel tests (`mtp` adds the multi-token-prediction head of every model whose
checkpoint carries one: heads have no configs of their own, so `all` alone
compiles none). The macOS `all` run is the full-scope case —
expect it to be slow (every config, every arch) and RAM-heavy; that's expected
CI-scale cost, not a regression.

Reproduce the **x86_64** linux-workspace gate in a container (Apple Silicon
runs it amd64-emulated — slow but faithful; a native aarch64 run is faster but
not the CI arch):
```bash
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  -e SCRATCHY_SKIP_SENDNN_CXX=1 rust:1-bookworm bash -c '
    rustup component add clippy >/dev/null 2>&1
    cargo clippy --workspace --features spyre,scratchy-models/granite-3.1-8b-instruct \
      --exclude scratchy-target-metal \
      --exclude scratchy-target-metal-compiler -- -D warnings'
```
The macOS metal CI job runs locally via `scripts/act-local.sh` when present.

See [`README.md`](README.md) (Building), [`docs/BUILD.md`](docs/BUILD.md), and
[`docs/MODELS.md`](docs/MODELS.md) for detail.

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
| New tape op (metal/spyre) | `crates/compiler/subtile/src/subtile_ir.rs` `SubOp` + one row in `ops.rs` | then the exhaustive matches in metal `compiler/src/steps_from_tape.rs`, `src/op_abi.rs` and spyre `lower_subtile_tape_to_{ktir,superdsc}.rs`; the build lists any you missed |
| New CUDA interpreter op | `crates/compiler/ir/src/lib.rs` `Instruction` | + its arm in `crates/targets/cuda/src/eval.rs`, emitted by `crates/compiler/macros/src/impl_lib.rs` |
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
- No per-model tokenizer lists or benchmark hacks (see `crates/models/arch/dsl/CLAUDE.md`).
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

## Child CLAUDE.md files
Claude Code loads these when it works in that directory.
- CI/docs/tooling: `.github/`, `docs/`, `scripts/`, `site/`
- crates: `benches/`, `cli/`, `core/`, `e2e/`, `layers/`, `tensors/`, `models/`, `models/arch/`
- compiler: `crates/compiler/`, `compiler/macros/`, `compiler/subtile/`
- serving: `crates/serving/`, `serving/{api,engine,scheduler,worker}/`
- metal: `crates/targets/metal/`, `metal/compiler/`, `metal/shaders/`, `metal/src/tape/`, `metal/src/interpreter/metal/`
- cuda: `crates/targets/cuda/`, `cuda/csrc/`
- spyre: `crates/targets/spyre/`, `spyre/flex-rs/`, `spyre/ktir/`, `ktir/{ktir-emulator,ktir-optimizer,ktir-superdsc}/`
- triton: `crates/triton/`, `triton/{splice,triton-frontend,triton-ktir,triton-ktir-superdsc,triton-numeric}/`

## Commits & PR titles
Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
both, scoped to the crate or area touched (`fix(metal): ...`).

## Agent skills

### Issue tracker

GitHub Issues on `AI-native-Systems-Research/scratchy`; always pass `gh -R`. Specs and plans are files under `docs/superpowers/`, not issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

`CONTRIBUTING.md`, `docs/*.md`, and specs/plans under `docs/superpowers/`; root `GLOSSARY.md` + `docs/adr/` once `/domain-modeling` creates them. See `docs/agents/domain.md`.
