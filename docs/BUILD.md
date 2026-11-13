# Building scratchy

How to build `scratchy-cli` (binary `scr`) — backend selection, and the
`-Fmodel/`/`-Fquant/` feature landscape that scopes which models and
quantization variants actually compile in. For adding a *new* model
architecture (DSL carrier, `configs/`, weight shapes), see
[`MODELS.md`](MODELS.md) instead — this doc is about building what already
exists, not adding to it.

## Quickstart

```bash
cargo build -p scratchy-cli --release --features metal,model/llama-3.2-3b
```

`metal` and `spyre` are mutually exclusive (and `cuda`, which builds FP8 models
only: add `quant/fp8-dynamic-per-channel`; see
[`cuda` and `nvcc`](#cuda-and-nvcc--the-nvidia-h100-backend)). Add
`serve` for the HTTP server
(`chat` — the in-process engine — is on by default), `multimodal` for the
image-decode stack, `bench` for `scr bench serve`. Tab completion over real
HuggingFace model ids is on by default (`hf-completions`); pass
`--no-default-features` to build with no network access at all. TurboQuant KV
compression is on with `metal` (and off for spyre), fixed at build time — see
[`turboquant`](#turboquant--kv-cache-compression-fixed-at-build-time).

## There is no default model, quant or drafter scope

`scratchy-models`, `scratchy-quantizations` and `scratchy-spec` all ship
`default = []`.
Naming zero models is a **build-time panic**, not a silently-empty binary —
`scratchy-forwards.rs` checks `total_models_emitted()` after every arch is
processed and fails loudly, telling you to name at least one. This means
every build — dev, CI, or deployment — names its own scope explicitly; there
is no `--no-default-features` dance to first strip away a default you don't
want.

## Model, quant and drafter scope: three crates, three CLI aliases

Model *scope* (`crates/models/arch/`, crate `scratchy-models`) and quant
*scope* (`crates/models/quantization/`, crate `scratchy-quantizations`) are
Cargo features on two **different** crates — not because picking a model and
picking a quantization are conceptually different (both are just "pick a
JSON file by name": model configs from `configs/<arch>/*.json`, quant
presets from `quantization/presets/*.json`), but because **Cargo forbids
aliasing one crate under two different local names**, and
`crates/cli/scr/Cargo.toml` needs two distinct short aliases reachable from
the CLI build line: `model` and `quant`. Speculative-decoding drafters are a
third alias, `spec`, on a third crate (`crates/models/spec/`, crate
`scratchy-spec`) for the same reason; its features forward to
`scratchy-models`, whose build script compiles the drafters.

```toml
# crates/cli/scr/Cargo.toml
model = { package = "scratchy-models", ... }
spec = { package = "scratchy-spec", ... }
quant = { package = "scratchy-quantizations", ... }
```

Cargo's `pkg/feature` CLI syntax resolves against these local names and
activates the (optional) dependency itself, so every feature below is
reachable directly from `scr`'s own build line as `model/<feature>` or
`quant/<feature>` — no hand-maintained passthrough feature list to keep in
sync (a guard test, `crates/cli/scr/tests/arch_feature_passthrough.rs`,
fails loudly if one ever creeps back in).

## `-Fmodel/<...>` — which models compile in

One Cargo feature per checked-in `configs/<arch>/<stem>.json`, named
`<stem>` — no `model-` text prefix; the `model` alias already says that.

```bash
cargo build -p scratchy-cli --features metal,model/qwen2.5-7b
```

- **`model/<stem>`** — exactly one model + variant, e.g.
  `model/granite-3.1-2b-instruct`. Disabled means the file is never even
  opened (the gate runs on the filename alone, before the JSON is parsed),
  so this is both the selection mechanism and a compile-time win — an
  unselected model costs nothing, not even a parse.
- **`model/<arch>`** — every config in one arch, e.g. `model/granite` is
  shorthand for "all granite models".
- **`model/all`** — every checked-in config, every arch. This is CI's
  full-scope case: slow (minutes) and RAM-heavy. Don't reach for it to test
  one thing.
- A handful of stems are byte-identical files checked into two arch dirs at
  once (the same checkpoint, loadable via a vision arch or its non-vision
  base arch — e.g. `qwen3.5-9b` under both `qwen3-5-vl/` and `qwen3-5/`).
  The vision arch keeps the bare stem; the base arch's copy is
  `<stem>-text-only`.

`arch-<name>` (which architecture's Rust module + DSL carrier compiles at
all) is implied automatically by any of that arch's `<stem>` features — you
essentially never name it directly.

## `-Fquant/<...>` — which quantization a model compiles as

One Cargo feature per preset, named `<preset>` — no `quant-` text prefix,
same reasoning as `<stem>`.

```bash
cargo build -p scratchy-cli --features metal,model/granite-3.1-2b-instruct,quant/mlx
# → granite-3.1-2b-instruct compiles as its MLX affine variants, not dense
```

**A selected quant preset REPLACES a model's dense/bf16 emission — it does
not add to it.** If any preset both survives Cargo-feature scoping and is
claimable on the active backend (metal only claims `mlx-affine-*`/`nvfp4`;
cuda requires `fp8-dynamic-per-channel` and refuses every other preset —
see `quant_preset_active` in `crates/compiler/macros/src/config.rs`), every selected model in that arch
gets its quant variant(s) *instead of* dense. A preset that wouldn't
actually synthesize on the active backend can never suppress dense, so a
model never ends up with zero compiled variants.

- **`quant/<preset>`** — one specific preset, e.g. `quant/fp8-dynamic-per-channel`.
- **`quant/mlx`** — shorthand for every MLX affine (int4) preset at once —
  the dominant use case on metal, but still opt-in (no backend
  defaults any preset on; a default-on preset would silently kill
  dense on every build using that backend).

No `quant/*` feature named at all = every selected model compiles dense only.

## `-Fspec/<...>` — which drafters compile with the selected models

```bash
cargo build -p scratchy-cli --features metal,serve,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
# → Qwen3.6-35B-A3B plus its multi-token-prediction head; serving it drafts with the head
cargo build -p scratchy-cli --features metal,serve,model/gemma-4-26b-a4b-it,quant/mlx
# → Gemma 4 26B-A4B plus its assistant at MLX 4-bit g32 (`quant/mlx` selects mlx-affine-b4-g32);
#   serving it drafts with guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit
cargo build -p scratchy-cli --features metal,serve,model/gemma-4-26b-a4b-it,spec/mtp,quant/mlx
# → also the bf16 and QAT assistants; serving drafts with one when `--speculative-model` names it
```

- **`metal`** turns on scratchy-models' **`mtp-heads`**: every selected model
  whose checkpoint carries a multi-token-prediction head compiles that head
  too. Such a head has no config of its own: its `arch.json` names the arches
  whose checkpoints carry it (`checkpoint_of`), and it compiles on each
  selected config of theirs that declares its layers, that config verbatim
  (`crates/compiler/macros/src/config.rs`). Serving drafts with the head from
  the target's own checkpoint when that carries it (its weight map holds
  `language_model.mtp.*`: `scratchery/Qwen3.5-122B-A10B-OptiQ-2bit-mtp`) or
  names the sidecar file an MLX conversion stripped it into (`mtp_file`:
  Nemotron-3 Super OptiQ keeps its head in `optiq/mtp.safetensors`, under the
  same `mtp.*` prefix), else from the repo it is published under when that
  resolves (`mlx-community/Qwen3.6-35B-A3B-MTP-4bit`), else it serves without
  drafting and says why. `--num-speculative-tokens 0` serves without drafting.
  It also compiles each selected model's default drafter published as a model
  of its own: a drafter with a config of its own names its target
  (`head_targets`), and the one its target drafts with by default names the
  repo each compiled variant of it is published as (`published_as`; a drafter
  checkpoint loads only into the variant of its own precision) — Gemma 4
  26B-A4B's assistant at MLX 4-bit g32,
  `guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit` (Google ships no
  Gemma 4 checkpoint that carries its drafter). Serving a checkpoint of that
  target's shape (`hidden_size` + `num_hidden_layers`) drafts with it when the
  repo resolves (downloading it on first use), else serves without drafting
  and says why. A build that compiles no variant a repo is named for (one
  without `mlx-affine-b4-g32`, which `quant/mlx` selects) drafts only when
  `--speculative-model` names one.
- **`spec/mtp`** — adds the other drafters published as models of their own,
  which draft only when `--speculative-model` names them (Gemma 4's bf16 and
  QAT assistants).

No `spec/*` feature named = only the heads inside their targets' checkpoints
and the targets' default published drafters compile (with `metal`); a spyre
build compiles none.

## `cuda` and `nvcc` — the NVIDIA H100 backend

`scratchy-cli` has two CUDA features:

- **`cuda`** — the CUDA engine with no toolchain. It expands the selected
  models to their CUDA tapes and type-checks everything, on any host (a Mac
  included), but links no kernel: `--device cuda` is claimed and refused with
  the command that rebuilds with kernels.
- **`nvcc`** — `cuda` plus the H100 kernels and the worker that runs them. It
  needs Linux with `nvcc` from CUDA 12.8 or newer on `PATH`; the build stops
  with a clear message when `nvcc` is missing or older. The kernels compile
  for `sm_90a` only (Hopper with its architecture-specific instructions: the
  warpgroup tensor-core MMA, WGMMA, and the tensor memory accelerator, TMA),
  so the binary runs on an H100 and refuses any other GPU at start.

`cuda` builds FP8 models only: name `quant/fp8-dynamic-per-channel` with one
of `model/qwen3.8-27b-text-only`, `model/qwen3.6-35b-a3b` or
`model/gemma-4-26b-a4b-it`. Any other `cuda` build panics at build time.
[`MODELS.md`](MODELS.md#cuda-the-three-fp8-models) lists their checkpoints
and what the backend does not support yet.

```bash
cargo build --release -p scratchy-cli \
    --features nvcc,serve,model/qwen3.6-35b-a3b,quant/fp8-dynamic-per-channel
scr serve RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic --device cuda --kv-cache-dtype fp8
```

### The kernels and ThunderKittens

The kernels are `crates/targets/cuda/kernels/*.cu`, the way metal keeps its
shaders under `shaders/`. Every one is written with
[ThunderKittens](https://github.com/HazyResearch/ThunderKittens),
HazyResearch's header-only CUDA C++20 tile library (MIT license). It is
vendored, unmodified, at one pinned commit in
`crates/targets/cuda/kernels/thunderkittens/`; its `PROVENANCE.md` gives the
commit, the license, the commands that check the copy against upstream, and
how to re-pin. Nothing fetches it at build time.

Under `nvcc`, `crates/targets/cuda/build.rs` compiles each kernel with the
flags of ThunderKittens' own `kernels/common.mk` (`-std=c++20 -O3
--use_fast_math -DKITTENS_SM90`, `-gencode arch=compute_90a,code=sm_90a`, and
the vendored `include/` and `prototype/` as `-I` paths), and archives the
objects into one static library the crate links. Without `nvcc` the build
script spawns nothing and links nothing.

### What CI checks without a GPU

CI never builds `nvcc`. The linux-workspace job's "CUDA scaffold" step runs,
with no CUDA toolkit and no GPU:

- clippy (`-D warnings`) on `scratchy-target-cuda` and
  `scratchy-target-cuda-compiler`, on the macro under `cuda`, and on
  `scratchy-models` with each of the three FP8 models;
- the tests of `scratchy-target-cuda` and `scratchy-target-cuda-compiler`
  (the lowering and the tape checks);
- `kernels_use_thunderkittens`, a check of the kernels' source text: every
  kernel takes its data as ThunderKittens views and moves and computes it with
  ThunderKittens operations (no raw pointers, no hand-written warp shuffles
  or atomics, inline PTX only from a short allowlist), and the few operations
  ThunderKittens lacks live in `kernels/scratchy_ops.cuh`;
- `cuda_tapes_27b_fp8`, `cuda_tapes_35b_fp8` and `cuda_tapes_gemma4_fp8`,
  which expand each model to its CUDA tapes and check them against what the
  CUDA compiler lowers.

So CI proves the tapes and the kernel source rules, not the kernels. A kernel
change is verified only by running the GPU tests below on an H100.

### Running the GPU tests

Each GPU test is a `[[test]]` of `crates/targets/cuda/Cargo.toml` (and of the
worker crate's) with `required-features = ["nvcc"]`, so a build without
`nvcc` never compiles it. On an H100 host with CUDA 12.8+:

```bash
# One suite: a kernel's output against a CPU reference or a bit-exact twin.
cargo test --release -p scratchy-target-cuda --features nvcc --test gemm_fp8 -- --test-threads=1
# Every GPU suite of the target crate, then the worker's.
cargo test --release -p scratchy-target-cuda --features nvcc -- --test-threads=1
cargo test --release -p scratchy-target-cuda-worker --features nvcc -- --test-threads=1
```

The H100 runs pass `--test-threads=1`, so one test has the GPU at a time.

To run a suite under NVIDIA's `compute-sanitizer` (it ships with the CUDA
toolkit), build the test binary without running it, take its path from
cargo's `Executable` line (newer cargo puts it under `target/release/build/`,
not `deps/`), and run it under each tool:

```bash
cargo test --release -p scratchy-target-cuda --features nvcc --test moe --no-run 2>&1 | grep Executable
compute-sanitizer --tool memcheck  <that path> --test-threads=1   # out-of-bounds and misaligned accesses
compute-sanitizer --tool racecheck <that path> --test-threads=1   # shared-memory races
compute-sanitizer --tool synccheck <that path> --test-threads=1   # barrier misuse
```

The sanitizers make a kernel much slower, so run them on one suite or one
test at a time.

## `turboquant` — KV cache compression, fixed at build time

On metal, a model's KV cache is either dense (the model's own dtype) or
TurboQuant codes (3-bit for the Llama family, 4-bit otherwise), and the build
decides which: nothing chooses at runtime. `scratchy-cli`'s `metal` feature
turns `turboquant` **on**; a spyre build does not have it. To build
metal with a dense KV cache, name `metal-base` instead of `metal`: the same
engine without scratchy-models' `turboquant` or `mtp-heads` (add
`model/mtp-heads` back to keep MTP drafting):

```bash
cargo build --release -p scratchy-cli --features metal-base,model/llama-3.2-1b
```

With it on, every selected model whose geometry the codec takes is emitted with
`KV_CODEC = TurboQuant`; the rest stay dense and the build names each one and
why:

```
warning: scratchy-models@0.1.0: turboquant: qwen2.5-0.5b keeps a dense KV cache: its KV row is 12 KiB/token, below the 24 KiB TurboQuant threshold
```

A model stays dense when it has no KV cache, when its KV cache is an MLA
latent, when a head_dim is not a power of two no wider than 512, or when its
fp16 KV row is under 24 KiB/token — too small for the capacity to be worth the
fidelity. The rule is `scratchy_layers::turboquant::codec_for`. The
server logs the loaded model's codec at start (`KV cache codec: TurboQuant
3-bit (built in)`).

`--kv-cache-dtype` on metal takes the codec by default (`auto`) and can only
*assert* it: `fp16` or `turboquant` refuses to start if the binary stores the
model's KV cache the other way.

## `hf-completions` — shell tab completion over real model ids

`scr completions bash` / `scr completions zsh` print a completion script for
this binary. Install it once, either by hand:

```bash
scr completions zsh  > ~/.zsh/completions/_scr        # dir must be on $fpath
scr completions bash > /usr/local/etc/bash_completion.d/scr
```

or let `scr` do it — `--install` writes the script to the standard per-shell
location AND wires up the shell rc file so it actually loads:

```bash
scr completions bash --install   # writes ~/.local/share/scr/completions.bash,
                                  # sources it from ~/.bashrc
scr completions zsh  --install   # writes ~/.zsh/completions/_scr,
                                  # adds it to $fpath in ~/.zshrc
```

`--install` is idempotent (marked block, safe to re-run after a rebuild) and
never clobbers the rest of the rc file. For zsh specifically, it inserts the
`fpath` entry *before* the first existing `compinit` call rather than
appending — zsh's completion system only picks up an autoloadable `#compdef`
function from a directory that was already on `$fpath` when `compinit` ran, so
appending after an existing `compinit` call would silently install a script
that never actually completes anything. If no `compinit` call exists yet,
`--install` adds one.

Subcommands, flags and enum values in that script come from the binary's own
clap command tree, so they always match the feature set it was built with —
a `chat`-only build never offers `serve`. Re-run `--install` (or regenerate by
hand) after rebuilding with different features.

**Model names come from `hf-completions`, which is on by default.** The build
asks huggingface.co which repos carry each compiled arch's family
tag, then fetches each candidate's `config.json` and keeps only those whose
shape (`hidden_size`, `num_hidden_layers`, expert count) matches a compiled
config AND whose `quantization_config` matches a compiled quant preset. So the
completions obey `-Fmodel/<stem>` and `-Fquant/<preset>` exactly: an
`mlx-affine-b4-g64` build completes 4-bit g64 MLX repos and not their 8-bit
siblings, and a dense build completes only unquantized repos. The result is
baked into the binary as a `&'static [&'static str]` — **tab completion itself
never touches the network.**

```bash
cargo build -p scratchy-cli --features metal,model/llama-3.2-1b
scr model names --source compiled      # what this build resolved to
```

- **On by default, opt out for an air-gapped build.** Verifying candidates
  costs one HTTP request each — ~160 for a single small arch, and hundreds per
  arch at `model/all` scope, which a clean build pays in full. To make no
  network requests at all, drop the feature:

  ```bash
  cargo build -p scratchy-cli --no-default-features \
      --features chat,metal,model/llama-3.2-1b
  ```

  `ureq` is then not even linked into the build script, so the build cannot
  reach the Hub by accident. The feature rides `scratchy-cli`'s default
  features via a weak `model?/hf-completions` edge, so it activates only when
  a `-Fmodel/<stem>` is named; building `-p scratchy-models` directly does not
  turn it on (its own `default` stays empty, which is what keeps CI's
  `--workspace` and `metal,all` runs network-free).
- **The names come most downloaded first**, the Hub's own ranking, so the
  canonical checkpoint leads its fine-tunes. `scr model names --source compiled`
  prints them in that order. zsh shows them in that order too. bash ≥ 4.4 does
  as well (`compopt -o nosort`), but bash 3.2, the one macOS ships, has no
  `compopt` and sorts them alphabetically.
- **`HF_TOKEN` at BUILD time widens the list.** Gated repos (`meta-llama/*`)
  return 401 without it. Their shape can't be verified, so they are left out,
  and the same model scope resolves differently with and without a token. The
  build says so: a `cargo:warning` counts the gated repos left out and names a
  few.
- Results are cached under `<target-dir>/hf-registry-cache/` with a 7-day TTL,
  so repeat builds don't re-hit the Hub. `cargo clean` discards it.

**With no registry there is no model-name completion at all** — subcommands and
flags still complete, but `scr chat <TAB>` offers nothing. That is deliberate.
The obvious fallback, completing from the local hf-hub cache, is wrong: the
cache is whatever has ever been pulled on this machine, not what this binary can
run, so it offers other architectures, other sizes and other quantizations —
and being build-independent it produces the *same* list for every `-Fmodel`
scope, which reads exactly like a broken scope filter. Offering nothing is more
honest than offering ids that fail at load. Filtering the cache by compiled
arch/shape/quant would make it a valid source and is tracked in
[#26](https://github.com/AI-native-Systems-Research/scratchy/issues/26).

## Fast iteration: scope to one small model

When verifying a change, **scope the build to exactly the one model you
need** — `-Fmodel/<stem>` (e.g. `-Fmodel/llama-3.2-1b` or
`-Fmodel/smollm2-135m` for the smallest/quickest option), plus
`-Fquant/<preset>` only if the change is quant-specific:

```bash
cargo build -p scratchy-cli --features metal,model/smollm2-135m
```

**Never build `model/<arch>` or `model/all` just to test one thing** — the
`#[forward]` macro is the bulk of per-arch compile time, and forward-expanding
every config in scope (rather than the one you actually need) costs
minutes-to-hours and risks OOM on a laptop. Model and quant selection is the
feature flags, full stop: to pick one specific model, name that model's
feature directly.

## Direct crate builds

`-p scratchy-cli` reaches `scratchy-models`/`scratchy-quantizations`/`scratchy-spec`
via the `model`/`quant`/`spec` aliases. Building `-p scratchy-models` or
`-p scratchy-quantizations` directly uses their real crate names instead
(the aliases only exist on `crates/cli/scr/Cargo.toml`'s dependency edges);
`scratchy-models` names the drafters itself (`mtp`, which `spec/mtp` forwards to):

```bash
cargo build -p scratchy-models --no-default-features \
    --features metal,llama-3.2-3b
# → only llama-3.2-3b's forward expands; its config.json is the only one
#   even read out of crates/models/arch/configs/llama/
```
