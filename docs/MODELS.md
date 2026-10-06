# Model architectures

How to add a *new* model architecture — DSL carrier, `configs/`, weight
shapes, quantization presets. For building what already exists (backend
selection, the `-Fmodel/`/`-Fquant/` feature landscape, fast-iteration
scoping), see [`BUILD.md`](BUILD.md) instead.

Every architecture the `#[forward]` codegen compiles lives in ONE crate,
`scratchy-models` (`crates/models/arch/`). Each arch is a DSL
carrier (`dsl/<arch>.py`) paired with a `configs/<arch>/` dir, gated by a
per-arch cargo feature (`arch-llama`, `arch-qwen3`, ...), implied
automatically by any of that arch's `<stem>` features. There are
no per-arch crates — everything below lives under `crates/models/arch/`.

## Layout

```
crates/models/arch/
  Cargo.toml               # arch-<name> + <stem> features (quant presets live on scratchy-quantizations instead); default = []
  scratchy-forwards.rs     # build script — runs the #[forward] pipeline per arch
  src/lib.rs               # #[cfg(feature="arch-<name>")] pub mod <arch>;  (includes OUT_DIR emit)
  dsl/<arch>.py            # one @forward / @vision_forward(...) def — the math
  configs/<arch>/
    <size1>.json           # per-model hyperparameters (HF config.json verbatim)
    <size2>.json
    ...
    weights.json           # per-arch weight shape formulas (shared across sizes)
    arch.json              # optional — per-arch declarations (everything about the
                           #   arch that isn't derivable from a verbatim config.json)
    quantizations.json     # optional — preset names this arch supports
    <size>-<preset>.overrides.json  # optional — per-(size, preset) drift
```

- **`<size>.json`** — the upstream HuggingFace `config.json` for one
  model (e.g. Qwen3-0.6B). Contains hyperparameters the compiler uses
  as shape bounds: `hidden_size`, `num_attention_heads`, `head_dim`,
  `intermediate_size`, `num_hidden_layers`, `vocab_size`, etc. Commit
  verbatim — don't edit by hand. Different model sizes under the same
  architecture get one file each.

- **`weights.json`** — shape of every weight that the compiler's
  dataflow-driven shape inference cannot pin (or would pin
  incorrectly). Expressed as products of bound names from the
  config.json vocabulary. Example for Qwen3, where `q_norm` / `k_norm`
  are per-head RMS norms of shape `[head_dim]`:

  ```json
  {
    "self_attn.q_norm": ["head_dim"],
    "self_attn.k_norm": ["head_dim"]
  }
  ```

  Only weights whose shape can't be derived from op-sig dataflow +
  `<size>.json` bounds belong here. For most Llama / Qwen2 / Granite
  entries, dataflow pins everything and `weights.json` is empty or
  absent.

- **`arch.json`** (optional) — the arch's own declarations: everything
  about it that is **not** derivable from a verbatim `config.json` plus
  the DSL body, and that is the same across every size. The DSL file
  declares the arch's *math* and nothing else, so these facts live here,
  next to the checkpoints they describe.

  Precedence, widest to narrowest — each wins field-by-field over the
  one before it:

  ```
  <size>.json  →  arch.json  →  <size>-<preset>.overrides.json
  (per size)      (per arch)    (per checkpoint)
  ```

  Keys reuse the same names the per-checkpoint overrides already use, so
  there is one vocabulary for both tiers: `scale_dtype`,
  `decoder_safetensors_prefix`, `tie_default`, `vision_norm_eps`,
  `vision_rope_style`, `vision_pos_emb_interp`, `vision_pos_embed_key`,
  `vision_safetensors_layout`, `vision_d_model_fingerprint`,
  `vision_patch_embed_flatten`, plus the `{name: value}` maps
  `bound_defaults`, `scalar_defaults`, `config_aliases`,
  `weight_leaf_renames`. An unknown key is a build **error**, not a
  silent no-op — a typo'd declaration would otherwise mis-emit the arch.

  ```json
  {
    "scale_dtype": "bf16",
    "bound_defaults": { "rms_norm_zero_centered": 1 },
    "decoder_safetensors_prefix": "language_model"
  }
  ```

  `params` (optional) is the bound schema vision towers need, where the
  flat top-level harvest isn't enough — one entry per bound, each with
  exactly one source: `from` (a dotted path into the verbatim config;
  a numeric segment indexes an array, and `default` covers configs that
  omit the key), `value` (a literal), or `expr` (arithmetic over bounds
  declared *earlier*, with `*` `/` `+` `-`, parens, and `sqrt()`).

  It is an **array, not an object**: `expr` entries read bounds that
  earlier entries defined, and only a list preserves that order.

  ```json
  "params": [
    { "name": "vision_embed_dim", "from": "vision_config.hidden_size" },
    { "name": "vision_num_heads", "from": "vision_config.num_heads" },
    { "name": "vision_head_dim",  "expr": "vision_embed_dim / vision_num_heads" },
    { "name": "vision_rope_half_dim", "expr": "vision_head_dim / 2" }
  ]
  ```

- **`quantizations.json`** (optional) — flat list of preset names
  this arch supports. Each preset cross-multiplies with every dense
  base in `configs/` to produce one compiled variant per
  `(size, preset)` pair. The preset definitions themselves live in
  the shared `scratchy-quantizations` crate (see below); this file
  just declares which ones apply to this arch.

  ```json
  { "quantizations": ["awq-gemm", "gptq-sym", "fp8-block-128x128"] }
  ```

- **`<size>-<preset>.overrides.json`** (optional) — per-HF-repo
  drift. Some HF quantization checkpoints diverge from their dense
  base in non-quantization fields (e.g. TinyLlama-GPTQ has
  `vocab_size=32003` vs. the dense base's `32000`). The overlay
  deep-merges last on top of the dense base + preset.

## Shared quantization presets

Preset definitions are JSON fragments that deep-merge onto each dense
base. They live once in `crates/models/quantization/presets/`:

```
crates/models/quantization/presets/
  awq-gemm.json
  bnb-nf4-dq.json
  ct-int4-sym.json
  fp8-block-128x128.json
  fp8-dynamic-per-tensor.json
  fp8-static-per-tensor.json
  gptq-sym.json
  gptq-sym-desc_act.json
  ...
```

The `#[forward]` macro discovers `presets/*.json` content by walking up
from `crates/models/arch/configs/` to the workspace root — that part
doesn't go through Rust code. But `scratchy-models` DOES depend on
`scratchy-quantizations` at the Rust level (both as a regular dependency,
for runtime preset loading, and as a build-dependency, so
`scratchy-forwards.rs` can call `scratchy_quantizations::enabled_presets()`
to learn which preset Cargo features are active — see
[`BUILD.md`](BUILD.md) for why quant scope lives on that crate and not this
one). Adding a
new preset means: dropping a JSON in `presets/`, listing it in the
relevant arch's `quantizations.json`, AND adding a matching `<preset> = []`
feature (no `quant-` prefix; plus a `preset!(v, "<preset>")` line in
`enabled_presets()`) to
`crates/models/quantization/Cargo.toml`/`src/lib.rs` — without that feature,
the preset is defined but never selectable.

## Multi-token-prediction heads

A multi-token-prediction (MTP) head drafts tokens for speculative decoding
from its target model's final hidden states. It is an arch of its own —
`qwen3-5-mtp` (`dsl/qwen3-5-mtp.py`) for Qwen3.5/3.6 — whose forward reads
the `target_hidden` extern; that alone makes it a head (the macro registers
it as one, and the serving stack drafts with the MTP proposer instead of a
draft model). What its `arch.json` declares:

- **`hf_architectures`** — the identity it registers under: an MLX drafter
  repo carries only `model_type: "qwen3_5_mtp"`. Its config's own
  `architectures` (its target's) name the targets it drafts for.
- **`drafter_repo_infix: "-MTP"`** — an MLX conversion strips the head's
  tensors from the target and publishes them apart, as the target's repo with
  this before its last `-` token (`mlx-community/Qwen3.6-35B-A3B-4bit` →
  `mlx-community/Qwen3.6-35B-A3B-MTP-4bit`).
- **`decoder_safetensors_prefix: ""`** — the drafter stores its tensors bare
  (`fc.*`, `layers.0.*`, `norm.*`, `pre_fc_norm_*`).
- **`params`** — its configs are the target's verbatim HF config, so
  `num_hidden_layers` *replaces* the target's with `mtp_num_hidden_layers`;
  `vocab_size` replaces the target's with the head's draft vocabulary (a
  prefix of the target's token ids, 65536 for Qwen3.6), while
  `embed_vocab_size` keeps every token for the embedding table.

The head carries no `embed_tokens` / `lm_head`: the worker lends it the
target's (the same mmap, nothing uploaded twice), the lm_head cut to the
head's `vocab_size` rows. `spec/mtp` compiles the head of every selected
model whose checkpoint carries one (a head config is selected when it is the
same file as a selected target's), and a build that compiles the head drafts
with it whenever it serves the target. How many tokens it drafts a step is a
compile-time constant (`spec_drafts` in the head's `arch.json`, 2): the
target's verify steps carry exactly that many rows per sequence, and both
models' kernels bake it. To serve without it, build without `spec/mtp`:

```bash
cargo build --release -p scratchy-cli --features metal,serve,model/qwen3.6-35b-a3b,\
spec/mtp,quant/mlx-affine-b4-g64-qembed
scr serve mlx-community/Qwen3.6-35B-A3B-4bit
```

The head drafts only for a step of at most `spec_max_seqs` sequences (its
`arch.json`, 1); a larger step runs as its target alone, at the target's
cost. The gate exists because a verify step carries `spec_drafts + 1` rows
a sequence, and on a MoE target those rows read more distinct experts: on
the base M5, Qwen3.6-35B-A3B's verify step costs 1.6 plain steps at 1
sequence and 2.65 at 8, against about 2.3 tokens a verify step. One
sequence is the point measured to pay on both GPUs measured (base M5, M5
Max); the base M5 also paid at up to 4.

Open work (each item, with its measurements, in the module docs of
`crates/serving/engine/src/spec_decode/mtp.rs`):

- **A gate per device and target**, derived at expansion from the verify
  and plain tapes' costs on each declared device profile, in place of one
  count for every GPU and every target of the head arch. Blocked on a metal
  tape cost model.
- **Stale requests never draft again**: a request that shares a step past
  the gate misses head KV for that step's rows, so it drafts no more for
  the rest of its life — at a gate of 1, once a second request arrives,
  neither drafts again. A prefix-cache hit on such blocks reuses head KV
  that was never written (fewer drafts accepted, output unaffected).
- **Verify cost past one sequence**: the experts its rows pick, and, past 2
  sequences, the 64-row prefill-shaped tape (verify-sized rungs should
  derive from the gate).
- **The scan's replay at prefill**: a build with a head runs its
  Gated-DeltaNet scan 24% slower per prompt token, the price of a bit-exact
  replay in one loop body; without a head it compiles out.
- **Time to first token** pays the head's pass over the prompt (about
  0.19 s on a 5.4k-token prompt, base M5).
- **`k` per bucket**, and re-measuring `k = 3` since the fold fix.
- **The M5 Max at the current build**, and `--num-speculative-tokens`,
  which a head ignores.

## Recipe for a new model architecture

All arches share the one `scratchy-models` crate (`crates/models/arch/`) —
adding an arch means adding files and feature-gates inside it, not creating a
new crate.

### 1. Add the `arch-<name>` feature

In `crates/models/arch/Cargo.toml`, add `arch-<name> = []` (or with
dependency passthroughs, mirroring an existing vision arch if applicable)
next to the other `arch-*` entries, and list it under `all-arches`. In
`crates/models/arch/src/lib.rs`, add the matching
`#[cfg(feature = "arch-<name>")] pub mod <name>;`.

### 2. Commit one config.json per supported size

Download the upstream HuggingFace `config.json` for each model size:

```
curl -L https://huggingface.co/<org>/<model-size>/resolve/main/config.json \
    > crates/models/arch/configs/<arch>/<size>.json
```

The filename is free-form but should match the HF model ID (e.g.
`qwen3-0.6b.json`). The compiler reads the `architectures` field of
each config to sanity-check it matches the DSL body's arch name. Every
checked-in config's file stem needs a matching bare `<stem>` Cargo feature
declared in `crates/models/arch/Cargo.toml` (only if the same file is
checked in under two arch dirs — the vision-arch/base-arch checkpoint-reuse
case — does it need disambiguating: `<stem>-text-only` for the base arch's
copy, plain `<arch>-<stem>` for any other collision; added to that arch's
bare-`<arch>` list and to `all`) — nothing else makes the config
reachable. No passthrough is needed on `crates/cli/scr/Cargo.toml`'s side —
it reaches every scratchy-models feature via the `model` dependency alias
(`model = { package = "scratchy-models", ... }`).

### 3. Generate weights.json

Run the probe binary against any one model size — the shape formulas
don't change across sizes, so one probe suffices:

```
cargo run -p scratchy-forward-compiler --bin probe_weights --features probe -- \
    --arch <arch> <hf-org>/<size> [<hf-org>/<size> ...]
```

The probe walks up to the workspace root, resolves
`crates/models/arch/configs/<arch>/` (with `_` → `-` on the arch
name), fetches the safetensors header, rewrites numeric dims as
bound-name products from the config.json vocabulary, and writes
`weights.json` next to the size configs. Review the output before
committing — literal integers it couldn't rewrite as known bounds are
flagged for review.

### 4. Write the DSL body

Add `crates/models/arch/dsl/<arch>.py`:

```python
import torch
import torch.nn.functional as F


@forward
def <arch>():
    # math of the forward pass, in bound-name shapes
    ...
```

`@forward` takes optional `workloads=[...]` / `sk_buckets=[...]` overrides of
the default bucket ladders. The file is a strict subset of Python: the
compiler parses it and never runs it, and anything outside the dialect
(documented in `crates/compiler/macros/src/parse_python.rs`) is a build
error with a `line:col`.

The DSL body expresses the forward pass as pure math — `embed`,
`rmsnorm`, `gemm`, `rope_append`, `attention`, `silu`, `add`, `mul`,
etc. Every tensor is a local; weights are referenced by dotted path
(e.g. `self_attn.q_proj[layer]`). Loop over `num_hidden_layers` using
a bound name from config.json. See `dsl/llama.py` as the canonical
reference.

### 5. Commit a correctness golden

Add `testdata/<arch>_<size>.json` generated from Python vLLM on the
same prompts the existing goldens use. Add a
`test_cuda_correctness_<arch>_<size>` in
`crates/e2e/tests/e_correctness.rs`. Verify the test passes on
CUDA before claiming the arch is landed:

```
cargo test -p scratchy-e2e --features e2e,cuda --release \
    --test e_correctness test_cuda_correctness_<arch>_<size> \
    -- --ignored --test-threads=1
```

No edits to `scratchy-serving-worker/src/cuda_worker.rs` are needed — the
arch auto-registers via `inventory::submit!` from the umbrella.

## Building (which models/quants actually compile in)

Which `<stem>`/`<preset>` Cargo features are active controls which of the
configs and presets described above actually forward-expand — see
[`BUILD.md`](BUILD.md) for the full `-Fmodel/`/`-Fquant/` feature landscape,
including fast-iteration scoping and why they're two separate crate
aliases.

## What NOT to do

- **Don't** hand-edit `<size>.json` — keep it verbatim with upstream.
- **Don't** add entries to `weights.json` for weights dataflow already
  pins correctly. The file is for exceptions only (per-head norms,
  latent projections, etc.) — every entry is a claim that dataflow
  can't derive this one.
- **Don't** put cross-arch weight shapes in
  `crates/compiler/macros/src/weights_manifest.rs`. It's reserved for
  truly universal conventions (e.g. `embed_tokens`, `lm_head`) —
  everything arch-specific lives in the arch's `weights.json`.
- **Don't** edit `scratchy-serving-worker/src/cuda_worker.rs` for a new arch.
  Auto-registration covers it; arch-specific state lives on the
  emitted `Weights` type.
