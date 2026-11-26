# Building scratchy

How to build the scratchy CLI — `scr` on metal, `spyre` on IBM Spyre, `cuda` on an NVIDIA H100,
each its target's binary
over the common CLI core (`scratchy-cli`) — and the
`-Fmodel/`/`-Fquant/` feature landscape that scopes which models and
quantization variants actually compile in. For adding a *new* model
architecture (DSL carrier, `configs/`, weight shapes), see
[`MODELS.md`](MODELS.md) instead — this doc is about building what already
exists, not adding to it.

## Quickstart

```bash
cargo build -p scratchy-target-metal-cli --release --features model/llama-3.2-3b      # scr
cargo build -p scratchy-target-spyre-cli --release --features spyre-emu,model/granite-3.1-2b-instruct,snt1p0  # spyre
(cd backends/cuda && cargo build --release --features nvcc,model/qwen3.8-27b-text-only,quant/fp8-dynamic-per-channel)  # cuda
```

`cuda` builds from its own workspace, `backends/cuda`, and FP8 models only:
`quant/fp8-dynamic-per-channel` with `model/qwen3.8-27b-text-only`,
`model/qwen3.6-35b-a3b` or `model/gemma-4-26b-a4b-it`. Its `nvcc` feature needs
CUDA 12.8+ on `PATH`; without it the binary builds anywhere and refuses to start
a worker. See [`backends/cuda/README.md`](https://github.com/AI-native-Systems-Research/scratchy/blob/main/backends/cuda/README.md).

Each binary links its one target. Add
`serve` for the HTTP server
(`chat` — the in-process engine — is on by default), `multimodal` for the
image-decode stack, `bench` for `scr bench serve`. Tab completion over real
HuggingFace model ids is on by default in `scr` (`model/hf-completions`); pass
`--no-default-features` to build with no network access at all. The metal KV
cache is int8, fixed at build time — see
[int8 KV](#int8-kv--the-metal-kv-cache-fixed-at-build-time); the spyre KV cache
is float8 by default on card (`spyre-hw`) builds — see
[fp8 KV](#fp8-kv--the-spyre-kv-cache-float8-by-default-on-card-builds).

## Every spyre build names its chip

A spyre build names the chip it bakes for: `snt1p0`, Spyre generation 1 (the card whose
geometry `ktir-core`'s `DeviceGen` table states). Naming none is a **compile error** in
`ktir-core` — the same law the model scope holds (no silent default) — so every spyre
command names it: `-F snt1p0` on the CLI/worker/models crates, which forward it down to
`ktir-core` the way `tp-<W>` is forwarded, or `-F ktir-core/snt1p0` for the ktir and
Triton crates that only depend on `ktir-core` itself.

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
cargo build -p scratchy-target-metal-cli --features model/qwen2.5-7b
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
cargo build -p scratchy-target-metal-cli --features model/granite-3.1-2b-instruct,quant/mlx
# → granite-3.1-2b-instruct compiles as its MLX affine variants, not dense
```

**A selected quant preset REPLACES a model's dense/bf16 emission — it does
not add to it.** If any preset both survives Cargo-feature scoping and is
claimable on the active backend (metal only claims `mlx-affine-*`/`nvfp4` —
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
cargo build -p scratchy-target-metal-cli --features serve,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
# → Qwen3.6-35B-A3B plus its multi-token-prediction head; serving it drafts with the head
cargo build -p scratchy-target-metal-cli --features serve,model/gemma-4-26b-a4b-it,quant/mlx
# → Gemma 4 26B-A4B plus its assistant at MLX 4-bit g32 (`quant/mlx` selects mlx-affine-b4-g32);
#   serving it drafts with guardiangate1775/gemma-4-26B-A4B-it-assistant-4bit
cargo build -p scratchy-target-metal-cli --features serve,model/gemma-4-26b-a4b-it,spec/mtp,quant/mlx
# → also the bf16 and QAT assistants; serving drafts with one when `--speculative-model` names it
```

- **`scr`** turns on scratchy-models' **`mtp-heads`** by default: every selected model
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
and the targets' default published drafters compile (in `scr`); a spyre
build compiles none.

## int8 KV — the metal KV cache, fixed at build time

On metal, every model whose geometry the codec takes stores its KV cache as int8
(`scratchy_layers::kv_int8`, oMLX's `affine8`): each head row of K and V is
rotated — a fixed ±1 sign per element, then an orthonormal Walsh–Hadamard
transform — and stored as 8-bit codes against one f32 scale for the row. The
writers encode as they write; decode attention reads the codes directly in the
rotated domain; prefill reads the cached rows staged back into a dense scratch.
Nothing chooses at runtime. A spyre build does not have it.

The rest stay dense and the build names each one and why:

```
warning: scratchy-target-metal-models@0.1.0: int8 KV: phi-3.5-mini keeps a dense KV cache: head_dim 96 is not a power of two from 64 to 512
```

A model stays dense when it has no KV cache, when its KV cache is an MLA latent,
or when a head_dim is not a power of two from 64 to 512 (the rotation is a
Walsh–Hadamard transform; the decode kernel splits a head over whole 32-lane
simdgroups, two at least). The rule is `scratchy_layers::kv_int8::codec_for`.
The server logs the loaded model's codec at start (`KV cache codec: int8
(built in)`).

`--kv-cache-dtype` on metal takes the codec by default (`auto`) and can only
*assert* it: `fp16` or `int8` refuses to start if the binary stores the model's
KV cache the other way.

## fp8 KV — the spyre KV cache, float8 by default on card builds

On a card build (`spyre-hw`), `spyre` stores every model's KV cache as float8
e4m3 when each of its KV head dims is whole float8 sticks of 128 elements: the
card feature turns on `kv/fp8`. Each new K and V row is saturated at ±448 and
cast as it is written, and the paged attention widens each page to 16 bits as it
gathers it. Nothing chooses at runtime. An emulator (`spyre-emu`) build keeps a
16-bit cache unless it names `kv/fp8` itself.

The rest stay dense and the build names each one and why:

```
warning: scratchy-target-spyre-models@0.1.0: fp8 KV: granite-3.1-2b-instruct keeps a dense KV cache: head_dim 64 is not whole float8 sticks of 128
```

A model stays dense when it has no KV cache, when its KV cache is an MLA latent,
when a head_dim is not a multiple of 128, or when its config states a
`kv_cache_scheme` (per-layer scales only the checkpoint's tensors hold). The rule
is `scratchy_layers::kv_int8::codec_for` at spyre's `Fp8Heads`.

`--features spyre-hw,kv/dense` keeps every model's KV cache 16-bit on a card
build — the build for an fp16-KV A/B, with no `--no-default-features`.

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
cargo build -p scratchy-target-metal-cli --features model/llama-3.2-1b
scr model names --source compiled      # what this build resolved to
```

- **On by default, opt out for an air-gapped build.** Verifying candidates
  costs one HTTP request each — ~160 for a single small arch, and hundreds per
  arch at `model/all` scope, which a clean build pays in full. To make no
  network requests at all, drop the feature:

  ```bash
  cargo build -p scratchy-target-metal-cli --no-default-features \
      --features chat,model/mtp-heads,model/llama-3.2-1b
  ```

  `ureq` is then not even linked into the build script, so the build cannot
  reach the Hub by accident. The feature rides `scr`'s default features
  (`model/hf-completions`); building `-p scratchy-models` or a models crate
  directly does not turn it on (scratchy-models' own `default` stays empty,
  which is what keeps CI's `--workspace` and `all` runs network-free). `spyre`
  has it off by default (`hf-completions`): a card host's build is air-gapped.
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
cargo build -p scratchy-target-metal-cli --features model/smollm2-135m
```

**Never build `model/<arch>` or `model/all` just to test one thing** — the
`#[forward]` macro is the bulk of per-arch compile time, and forward-expanding
every config in scope (rather than the one you actually need) costs
minutes-to-hours and risks OOM on a laptop. Model and quant selection is the
feature flags, full stop: to pick one specific model, name that model's
feature directly.

## Direct crate builds

A target's CLI (`-p scratchy-target-metal-cli`, `-p scratchy-target-spyre-cli`) reaches
`scratchy-models`/`scratchy-quantizations`/`scratchy-spec` via the `model`/`quant`/`spec` aliases. Building a target's models crate
directly (`scratchy-target-metal-models`, `scratchy-target-spyre-models`) names
the scope crates by their real names instead (the aliases only exist on the
CLIs' dependency edges); `scratchy-models` names the drafters itself (`mtp`,
which `spec/mtp` forwards to):

```bash
cargo build -p scratchy-target-metal-models --features scratchy-models/llama-3.2-3b
# → only llama-3.2-3b's forward expands; its config.json is the only one
#   even read out of crates/models/arch/configs/llama/
```
