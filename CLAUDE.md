# Building & verifying scratchy

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
- No new env-var behavior toggles; no `#[allow]`; no weakening tests.

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
- `cuda`, `metal` and `spyre` are mutually exclusive. `cuda` builds no model
  yet: `scratchy-models`' build script refuses every arch until one lowers.
- `scratchy-target-cuda`'s `nvcc` feature compiles `kernels/*.cu`
  (ThunderKittens vendored at `kernels/thunderkittens/`) for `sm_90a` and links
  them. It needs CUDA 12.8+ `nvcc` on `PATH`, and its tests need an H100. CI
  never builds it, so the GKE H100 run is its only gate.
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
one of the two still drags `objc2` onto Linux). Its "CUDA scaffold" step then
lints `scratchy-target-cuda`/`scratchy-target-cuda-compiler` (`--all-targets`)
and the macro under `--features cuda`, and runs the two CUDA crates' tests — no
nvcc, no GPU. The **macOS** job
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

## Commits & PR titles
Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
both, scoped to the crate or area touched (`fix(metal): ...`).
