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
image-decode stack, `bench` for `scr bench serve`.

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
