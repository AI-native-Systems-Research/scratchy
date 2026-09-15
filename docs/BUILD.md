# Building scratchy

How to build `scratchy-cli` (binary `scr`) — backend selection, and the
`-Fmodel/`/`-Fquant/` feature landscape that scopes which models and
quantization variants actually compile in. For adding a *new* model
architecture (DSL carrier, `configs/`, weight shapes), see
[`MODELS.md`](MODELS.md) instead — this doc is about building what already
exists, not adding to it.

## Quickstart

```bash
cargo build -p scratchy-cli --release --features metal,model/llama-3.2-3b   # or: cuda
```

`cuda` and `metal` are mutually exclusive. Add `serve` for the HTTP server
(`chat` — the in-process engine — is on by default), `multimodal` for the
image-decode stack, `bench` for `scr bench serve`. Tab completion over real
HuggingFace model ids is on by default (`hf-completions`); pass
`--no-default-features` to build with no network access at all.

## There is no default model or quant scope

Both `scratchy-models` and `scratchy-quantizations` ship `default = []`.
Naming zero models is a **build-time panic**, not a silently-empty binary —
`scratchy-forwards.rs` checks `total_models_emitted()` after every arch is
processed and fails loudly, telling you to name at least one. This means
every build — dev, CI, or deployment — names its own scope explicitly; there
is no `--no-default-features` dance to first strip away a default you don't
want.

## Two axes, two crates, two CLI aliases

Model *scope* (`crates/models/arch/`, crate `scratchy-models`) and quant
*scope* (`crates/models/quantization/`, crate `scratchy-quantizations`) are
Cargo features on two **different** crates — not because picking a model and
picking a quantization are conceptually different (both are just "pick a
JSON file by name": model configs from `configs/<arch>/*.json`, quant
presets from `quantization/presets/*.json`), but because **Cargo forbids
aliasing one crate under two different local names**, and
`crates/cli/scr/Cargo.toml` needs two distinct short aliases reachable from
the CLI build line: `model` and `quant`.

```toml
# crates/cli/scr/Cargo.toml
model = { package = "scratchy-models", ... }
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
  base arch — e.g. `gemma-3-12b-it` under both `gemma3/` and `gemma3-mm/`).
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
cuda is the mirror — see `quant_preset_active` in
`crates/compiler/macros/src/config.rs`), every selected model in that arch
gets its quant variant(s) *instead of* dense. A preset that wouldn't
actually synthesize on the active backend can never suppress dense, so a
model never ends up with zero compiled variants.

- **`quant/<preset>`** — one specific preset, e.g. `quant/fp8-dynamic-per-channel`.
- **`quant/mlx`** — shorthand for every MLX affine (int4) preset at once —
  the dominant use case on metal, but still opt-in (neither `metal` nor
  `cuda` defaults any preset on; a default-on preset would silently kill
  dense on every build using that backend).

No `quant/*` feature named at all = every selected model compiles dense only.

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
- **`HF_TOKEN` at BUILD time widens the list.** Gated repos (`meta-llama/*`)
  return 401 without it and are dropped, so the same model scope resolves
  differently with and without a token.
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
minutes-to-hours and risks OOM on a laptop. `SCRATCHY_BUILD_FILTER=<tag>,...`
(comma-separated substrings, OR'd, matched against the stem) still exists as
an orthogonal cross-cutting filter on top of whichever `model/*` features are
enabled — useful for narrowing a wide `model/<arch>` build without listing
every excluded model individually — but for picking one specific model it's
no longer necessary; just name that model's feature directly.

## Direct crate builds

`-p scratchy-cli` reaches `scratchy-models`/`scratchy-quantizations` via
the `model`/`quant` aliases. Building `-p scratchy-models` or
`-p scratchy-quantizations` directly uses their real crate names instead
(the aliases only exist on `crates/cli/scr/Cargo.toml`'s dependency edges):

```bash
cargo build -p scratchy-models --no-default-features \
    --features cuda,llama-3.2-3b
# → only llama-3.2-3b's solver runs; its config.json is the only one
#   even read out of crates/models/arch/configs/llama/
```
