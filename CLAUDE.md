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
- **No instruction selection** for spyre or metal — the tape comes straight
  from the fused forward. ISel (`solver`, `cost`, the `impl Implementation`
  blocks) is cuda-gated and stays that way.
- No net line growth in target crates (declared-data tables exempt); no new
  env-var behavior toggles; no `#[allow]`; no weakening tests.

## The model crate
All model architectures compile into ONE crate, **`scratchy-models`**
(`crates/models/arch/`); quant presets live on a separate crate,
**`scratchy-quantizations`** (`crates/models/quantization/`) — two crates
because Cargo won't alias one crate under two names, and `scratchy-cli`
needs both `model` and `quant` as short CLI aliases. Neither has a default
scope: naming zero models is a **build-time panic**, not a silent empty
binary. Every checked-in `configs/<arch>/<stem>.json` gets its own `<stem>`
feature (`model/<stem>` from the CLI, `model/<arch>`/`model/all` to widen);
every quant preset gets its own `<preset>` feature (`quant/<preset>`,
`quant/mlx` for every MLX affine preset at once) — a selected preset
**replaces** a model's dense/bf16 emission, it doesn't add to it. See
[`docs/BUILD.md`](docs/BUILD.md) for the full mechanics.

## Build
```bash
# The CLI you run. Pick ONE backend: metal (macOS) or cuda. Add serve for HTTP.
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
- `cuda` and `metal` are mutually exclusive.
- Multimodal (vision arches + the `image`/`scratchy-vision` decode stack) is
  opt-in via the `multimodal` feature on `scratchy-cli` (on by default
  there) — a plain `chat`-only build never compiles image/png/zune-jpeg/etc.
- List cached models with `scr model ls`; run a model in-process with
  `scr chat -m <model> --device metal -q "..."` (no image arg — images need the
  HTTP server, i.e. the `serve` feature + `scr serve ... --kv-cache-dtype ...`).

## Verify like CI before opening a build / deps / codegen PR
`.github/workflows/rust.yml`: the **linux-cuda** job runs
`cargo clippy --workspace --features cuda,scratchy-models/all
--exclude scratchy-target-metal --exclude scratchy-target-metal-compiler
-- -D warnings` (both depend unconditionally on Apple-only `objc2` — the
compiler crate via its own edge to scratchy-target-metal — and `--workspace`
builds every member as a root regardless of `--features`, so excluding only
one of the two still drags `objc2` onto Linux); the **macOS** job
runs `cargo clippy -p scratchy-models --features metal,all -- -D warnings`
+ kernel tests. It sets `SCRATCHY_GPU=h100`, `CUDA_COMPUTE_CAP=90`,
`SCRATCHY_SKIP_CUDA_KERNELS=1` (so `scratchy-builder-cuda`'s build script doesn't
need a GPU/nvcc). Both `all` runs are the full-scope case —
expect it to be slow (every config, every arch) and RAM-heavy; that's expected
CI-scale cost, not a regression.

Reproduce the **x86_64** linux-cuda gate in a container (Apple Silicon runs it
amd64-emulated — slow but faithful; a native aarch64 run is faster but not the
CI arch):
```bash
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  -e SCRATCHY_GPU=h100 -e CUDA_COMPUTE_CAP=90 -e SCRATCHY_SKIP_CUDA_KERNELS=1 \
  nvidia/cuda:12.9.1-devel-ubuntu22.04 bash -c '
    apt-get update -qq && apt-get install -y -qq curl git pkg-config libssl-dev ca-certificates >/dev/null 2>&1
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -c clippy >/dev/null 2>&1
    export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
    cargo clippy --workspace --features cuda,scratchy-models/all \
      --exclude scratchy-target-metal \
      --exclude scratchy-target-metal-compiler -- -D warnings'
```
The macOS metal CI job runs locally via `scripts/act-local.sh` when present.

See [`README.md`](README.md) (Building), [`docs/BUILD.md`](docs/BUILD.md), and
[`docs/MODELS.md`](docs/MODELS.md) for detail.

## Commits & PR titles
Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
both, scoped to the crate or area touched (`fix(metal): ...`).
