# Claude Code on scratchy vs ollama — M5 Max

Part of the Claude Code serving comparison: three models, scratchy against
ollama, every applicable optimization ablated. **Nothing here is measured yet.**
The contract lands before the numbers, so the rung pairing and the disclosures
cannot be chosen after seeing which of them flatters us — the same ethic that
fixes the metric list before phase 4 runs, and that lands the leaderboard schema
and its validator while there is still nothing to put in them.

**Every empty section below names the task that fills it.** A silent blank here
is a defect: a later reader cannot tell silence from a tested claim.

| section | state |
|---|---|
| [Headline](#headline) | empty — phase 4 measures, phase 5 writes it |
| [Machine and toolchain](#machine-and-toolchain) | recorded; nothing measured on it yet |
| [Scope and deferrals](#scope-and-deferrals) | **recorded — T0.8**; the KV ablation is in scope, prediction on record |
| [Method and metrics](#method-and-metrics) | owned by the agreed method, not restated here |
| [Exact commands](#exact-commands) | scratchy side recorded; ollama side **T0.4** |
| [Results](#results) | empty — phase 4 |
| [Quantization rungs](#quantization-rungs--r1-and-r2) | **recorded — T0.5** |
| [Blocked](#blocked) | recorded |
| [Artifacts](#artifacts) | empty — phase 4 |

Task ids are the work breakdown's, not this document's: `T0.x` are phase 0
(parity plumbing) and `T4.x` phase 4 (the runs and ablations). Each is named
where it is first cited; the lists they come from live in the work breakdown,
so a `T4.x` here is a pointer out of this file rather than to a table in it.

---

## Headline

Empty on purpose. The acceptance criteria put the fairness disclosures *before*
the headline, and allow the headline to be "ollama wins here". Writing one now
would be writing it before the measurement.

## Machine and toolchain

The host the pins below were taken on. Phase 4 re-states this per run; nothing
in this file has been measured on it.

| | |
|---|---|
| chip | Apple M5 Max, 18 cores (`hw.perflevel0.logicalcpu` 6, `hw.perflevel1.logicalcpu` 12) |
| GPU | 40-core (`system_profiler SPDisplaysDataType`), Metal 4 |
| memory | 64 GiB unified |
| OS | macOS 26.7.1 (build 25G241) |
| scratchy | `fc2c3332` — the commit this document branched from |
| rustc | 1.97.1 (8bab26f4f 2026-07-14) |
| ollama | client 0.34.2; **no daemon was running.** The served version, context length, parallelism and keep-alive are **T0.4** (pinning the ollama side) |
| claude | 2.1.289 |

**`claude` has already moved.** The epic was written against 2.1.285; this host
runs 2.1.289. The first fairness rule exists for exactly this — a Claude Code
release rewrites the system prompt and the tool schemas, so it invalidates a
Lane A trace corpus. The drift is recorded rather than silently updated: a
corpus captured under one version is not replayable under another, and phase 1's
capture stamps whichever version it ran.

## Scope and deferrals

**Recorded — T0.8.** This section says what phase 4 measures and what it does
not, so a missing row reads as a decision rather than as a result nobody
noticed. **Nothing here is measured.** The numbers below are arithmetic over the
checked-in configs and the code that reads them, and they are written down
*before* the run on purpose.

### TurboQuant KV cache: measured, with the prediction on record

TurboQuant stores a model's KV cache — the per-token keys and values a model
keeps so it does not have to re-read the conversation for each new word — as
4-bit codes instead of fp16. T0.8 originally deferred this ablation on the
grounds below. **It is now in scope:** a measured near-zero is worth more than
an argued one, and if the number is near zero we want to know why from data.
The arithmetic stays here as a prediction rather than as a reason to skip.

#### It is already on in every build we ship

Worth stating first, because it is the opposite of what "add TurboQuant to the
comparison" sounds like. On `scratchy-cli` the `metal` feature includes
`turboquant` (`crates/cli/scr/Cargo.toml:45`), and all three models clear the
codec's threshold, so **every build in [Exact commands](#exact-commands)
already runs 4-bit KV.** There is nothing to switch on. What the ablation needs
is the *dense* arm.

#### What the dense arm costs

Cargo features only add, so `--features metal` cannot ask for metal without
`turboquant`, and `--kv-cache-dtype fp16` only asserts what the binary already
does — it refuses to start on a TurboQuant build rather than changing it
(`crates/serving/worker/src/gpu_worker.rs:3322-3337`). The dense arm therefore
needs `"turboquant"` dropped from that one feature list and named explicitly
instead, plus a second compile of each model.

That is a change to the build graph, not to the codec, the threshold, or the
policy — so it still measures TurboQuant exactly as implemented today. Phase 4
should confirm that reading before spending the compiles.

#### A first data point that costs nothing

Four lines the server already prints at startup say what is compressed and how
big the pool is, on the builds we are shipping, with no second compile and no
A/B:

| line | from | gives |
|---|---|---|
| `KV cache codec: … (built in)` | `gpu_worker.rs:3338` | dense or TurboQuant, and the bit width |
| `TurboQuant KV: N-bit codebook (… n/m global layers compressed)` | `targets/metal/src/turboquant.rs:76` | how many layers the codec actually covers |
| `KvCachePool: … = N MB` | `layers/src/kv_cache.rs:247`, `:396` | the pool's real size |
| `GdnStatePool: n/m linear layers × k slots` | `layers/src/gdn_state.rs:144` | how much of Qwen3.6 holds recurrent state instead of KV |

Recording those three sets of lines is the cheapest way to confirm or break
every prediction below, and it should happen before the dense builds are
queued.

#### The prediction

Two reasons to expect a small effect. If the measurement disagrees, the model of
the mechanism below is wrong, which is itself worth having.

**1. The codec covers a minority of layers.** It compresses full-context layers
only; on a mixed-attention model each sliding layer stays fp16 and gets a
16-byte placeholder the tape never binds
(`crates/compiler/macros/src/codegen.rs:12637-12646`,
`crates/targets/metal/src/turboquant.rs:61-140`).

| model | layers | keep a cache that grows with the session | the rest |
|---|---|---|---|
| `gemma-4-12b-it` | 48 | **8** full-attention | 40 capped at a 1024-token window — 8 MiB each, fp16, flat in session length |
| `gemma-4-26b-a4b-it` | 30 | **5** full-attention | 25 windowed, same |
| `qwen3.6-35b-a3b` | 40 | **10** full-attention | 30 hold GDN recurrent state, not a KV cache |

**2. What is left is small, because the architecture already shrank it.**
gemma-4's full-context layers carry 1 and 2 KV heads where its sliding layers
carry 8 (`num_global_key_value_heads`; per-layer geometry in
`crates/core/model/src/weight.rs:446-468`), and Qwen3.6's carry 2. At the 65536
tokens `scr launch claude` pins:

| model | the compressed layers' cost | KV at 65536 tokens | 4-bit saves | of a 64 GiB box |
|---|---|---|---|---|
| `gemma-4-12b-it` | 8 × 2 × 1 × 512 × 2 B = **16 KiB/token** | 1.00 GiB coded + 320 MiB fp16 windows | ~0.75 GiB | ~1.2% |
| `gemma-4-26b-a4b-it` | 5 × 2 × 2 × 512 × 2 B = **20 KiB/token** | 1.25 GiB coded + 200 MiB fp16 windows | ~0.94 GiB | ~1.5% |
| `qwen3.6-35b-a3b` | 10 × 2 × 2 × 256 × 2 B = **20 KiB/token** | 1.25 GiB coded | ~0.94 GiB | ~1.5% |

(`2 ×` is K and V; the codec also stores an f32 norm per head-vector and one
layer of fp16 staging scratch, so the ratio is a little under 4× —
`kv_bytes_per_token` and `bytes_per_vec`,
`crates/layers/src/turboquant.rs:391` and `:228`.)

scratchy's own threshold agrees with that reading. `codec_for` keeps a model
dense below 24 KiB/token because *"TurboQuant trades fidelity for KV CAPACITY.
Below this, the capacity is not the constraint and the trade is a bad one"*
(`crates/layers/src/turboquant.rs:276-291`). All three models sit at
16–20 KiB/token of growing cache, i.e. under that bar. They are built with the
codec anyway because the threshold is measured over all layers — correct for a
model where every layer keeps a cache, 4× to 24× high for these three:

| model | all-layers figure the threshold saw | the part that grows | apart by |
|---|---|---|---|
| `gemma-4-12b-it` | 384 KiB/token | 16 KiB/token | **24×** |
| `gemma-4-26b-a4b-it` | 240 KiB/token | 20 KiB/token | **12×** |
| `qwen3.6-35b-a3b` | 80 KiB/token | 20 KiB/token | **4×** |

#### Where to look for the effect

The codec's own rationale is **capacity, not speed** — it trades fidelity for
how much KV fits. So the run most likely to show something is the
context-ceiling one (**T4.4**: how far each engine gets on 64 GB and where it
dies), not per-turn TTFT and TPOT at a 65536-token cap where the pool is about a
GiB either way. Report peak RSS, the pool size from the log lines above, and the
ceiling; report TTFT and TPOT too, with the expectation that they move inside
run-to-run noise.

One thing to settle while measuring, since it changes what the memory numbers
mean: **output fidelity.** The codec is lossy, so the dense and 4-bit arms are
not guaranteed to produce the same text. A capacity win paid for in output
quality is the same trap as the tool-span arm in [point
3](#what-a-rung-does-not-pin) below, and the same disclosure applies.

#### What a near-zero result would and would not say

If the numbers come back flat, that is a result **about these three models**,
not about KV quantization. All three were chosen for the epic's other questions
— a dense control, an MoE, an MoE with GDN — and all three happen to keep very
little long-lived KV. The publishable sentence is the narrow one: *on these
three models, at this context length, there is almost nothing to compress.*

A model where the codec has real work to do is already in the tree:
**`qwen3.6-27b`** (`model/qwen3.6-27b`,
`configs/qwen3-5/qwen3.6-27b.json`) keeps full-context KV in 16 of 64 layers at
4 KV heads × 256 — **64 KiB/token**, three to four times these three, and about
4 GiB at 65536 tokens. T0.7 already proposes adding it to the runner's tables.
If the ablation is flat on the three, running it there is what turns "flat on
our set" into something general.

#### Two things found establishing this, both filed

Neither is a bench result. Both change what the measurement means, so they are
recorded here and tracked in **#280** and **#281**.

1. **`qwen3.6-35b-a3b` provisions a KV cache for all 40 layers; 10 hold one.**
   The per-layer sizing override is emitted only for arches with
   `OpKind::SlidingAttention` tiles
   (`crates/compiler/macros/src/codegen.rs:7827-7886`), and this arch's
   non-full layers are `GatedDeltaNet`, so it returns `None`, the pool takes the
   uniform path at `model.num_hidden_layers()` = 40
   (`crates/serving/worker/src/gpu_worker.rs:3400-3410`), and the 30 GDN layers
   each get K and V buffers the forward never writes. TurboQuant follows: a
   uniform arch's `is_global` map is all-true, so packed stores and norms are
   provisioned for those 30 layers too — the codec is compressing 30 layers of
   cache nothing uses. The pages are never touched, so RSS is largely spared,
   but the **block count is not**: `compute_num_blocks` divides the budget by 40
   layers' worth of cache (`crates/serving/api/src/init.rs:3900-3927`), so the
   context that fits is computed ~4× too conservatively. The fix shape already
   exists ~200 lines below the KV pool's own construction, in the same
   `initialize_cache` (`crates/serving/worker/src/gpu_worker.rs:3594-3626`):
   `GdnStatePool::new` takes a `linear_layers` mask and allocates only the
   layers that need state, and says so in its own words: *"Unlike
   `KvCachePool`, which allocates every layer…"*
   (`crates/layers/src/gdn_state.rs:20-24`). The KV pool wants that mask's
   complement. Filed as **#280**, and **worth fixing before T4.4 reads a
   context ceiling on this model**, since it is the ceiling it would be
   reading.

2. **The 24 KiB/token threshold has no mixed-attention case.** It is defined
   over all layers and was calibrated on two models where every layer keeps a
   cache (`qwen2.5-0.5b` at 12 KiB/token → dense, `llama-3.2-1b` at 32 →
   coded), so hybrid-window and hybrid-recurrent arches are the first to meet it
   on a figure that is not the part of their cache that grows. Not a mistake in
   the constant; a case it was not set on. Whether 16–20 KiB/token *should* be
   coded stays a policy call, and the ablation is now the thing that answers it.
   Filed as **#281**.

## Method and metrics

Owned by the agreed method and deliberately not restated here: the metric list,
the per-turn / per-task / per-session split, the seven fairness rules, and the
lane rule — every latency claim comes from Lane A (trace replay), every outcome
claim from Lane B (live Claude Code), neither ever quoted as the other. Phase 5
promotes that content into [`BENCHMARKING.md`](BENCHMARKING.md) as a §7; this
file carries only what is specific to a run.

### Exact commands

**scratchy — identical on both rungs**, because a rung changes only what ollama
loads. `serve_argv` (`crates/cli/scr/src/commands/launch.rs:226-295`) is what
turns these into the spawned server's argv, so what is written here is what the
server gets:

```bash
# the dense control — the model the harness is built against
cargo build --release -p scratchy-cli \
  --features metal,serve,model/gemma-4-12b-it,quant/mlx-affine-b4-g64
scr launch claude -m mlx-community/gemma-4-12B-it-4bit --device metal

# the MoE carrying the prefix-cache and tool-span headline
cargo build --release -p scratchy-cli \
  --features metal,serve,model/gemma-4-26b-a4b-it,quant/mlx-affine-b4-g64
scr launch claude -m mlx-community/gemma-4-26b-a4b-it-4bit --device metal

# MoE + GDN: prefix caching is off by design, so this one measures the cost
cargo build --release -p scratchy-cli \
  --features metal,serve,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
scr launch claude -m mlx-community/Qwen3.6-35B-A3B-4bit --device metal
```

One model per binary on purpose: naming `model/all` forward-expands every config
in scope ([`BUILD.md`](BUILD.md)), which is minutes-to-hours and an OOM risk on a
laptop.

**ollama — T0.4.** The tags are named per rung below; the environment that must
accompany them (`OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL`,
`OLLAMA_KEEP_ALIVE`, the pinned version, and what `ollama launch claude` puts in
Claude Code's environment) is T0.4's one-page note and is not guessed here.

## Results

Empty — phase 4 measures. Lane A and Lane B tables land separately and are never
quoted as each other.

## Fairness disclosures

The agreed method owns the list that travels with every published number: KV
dtype unmatched across engines, differing quantizers on R1, build-time forward
compilation flattering scratchy's startup, daemon versus spawned server, and
powermetrics reporting estimates rather than meter readings. What phase 0
resolves to concrete artifacts is the rung matrix.

### Quantization rungs — R1 and R2

Both engines are asked to run "the same model". But the file each one loads was
**quantized by a different tool**, and quantization changes both speed and
output. One comparison therefore cannot separate the engine from the quantizer,
so the comparison runs two rungs:

| rung | the question it answers | what differs between the engines |
|---|---|---|
| **R1** "what a user gets" | the real-world one — install either engine, pull what its own users pull | engine **and** quantizer |
| **R2** "matched weights" | how much of R1 was the engine | engine only: the same mlx 4-bit weights on both sides |

R1 is the headline. R2 is what stops every R1 number from being dismissed as a
quantizer comparison.

#### The scratchy side, pinned — the same on both rungs

| model | feature | checkpoint @ revision | declared preset | per-tensor exceptions, **from the checkpoint** |
|---|---|---|---|---|
| `gemma-4-12b-it` (dense control) | `model/gemma-4-12b-it`, arch `gemma4` | `mlx-community/gemma-4-12B-it-4bit` @ `73bcf09092aa277861d5a191b989b666f7f32e8f` | `mlx-affine-b4-g64` — the only one the arch declares (`configs/gemma4/quantizations.json`) | none |
| `gemma-4-26b-a4b-it` | `model/gemma-4-26b-a4b-it`, arch `gemma4-moe` | `mlx-community/gemma-4-26b-a4b-it-4bit` @ `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` | `mlx-affine-b4-g64`, **pinned** (`configs/gemma4-moe/quantizations.json`) | `router.proj` at 8 bits — 30 entries, one per layer, declared in `gemma-4-26b-a4b-it-mlx-affine-b4-g64.overrides.json` |
| `qwen3.6-35b-a3b` | `model/qwen3.6-35b-a3b`, arch `qwen3-5-moe` | `mlx-community/Qwen3.6-35B-A3B-4bit` @ `38740b847e4cb78f352aba30aa41c76e08e6eb46` | `mlx-affine-b4-g64-qembed` (`configs/qwen3-5-moe/quantizations.json`) | `mlp.gate` + `mlp.shared_expert_gate` at 8 bits — 80 entries, 40 layers × 2 roles, declared in `qwen3.6-35b-a3b-mlx-affine-b4-g64-qembed.overrides.json` |

Revisions are each repo's HuggingFace `sha`, resolved 2026-10-05. A run that
loads a different revision is a different cell: the acceptance criteria name
"checkpoint revision" among the fields every published row must carry.

#### The preset is not the whole specification

Worth stating next to the table, because this architecture has already produced
two unloadable builds and the rung matrix is where a reader looks for what was
actually loaded. **Per-tensor quantization detail comes from the checkpoint; the
preset supplies only the global default.** The precedence chain is
[`MODELS.md`](MODELS.md)'s, narrowest last:

```
<size>.json   →   arch.json   →   <size>-<preset>.overrides.json
(per size)        (per arch)      (per checkpoint)  ← wins field-by-field
```

- `gemma4-moe` declares `mlx-affine-b4-g64` and **deliberately does not declare**
  `mlx-affine-b4-g64-mlp8-router8`: that preset also claims
  `mlp.{gate,up,down}_proj` at 8 bits, which this checkpoint's mlp is not, so it
  compiles in 26.5 s and then cannot open the weights
  (`configs/gemma4-moe/quantizations.json`).
- `qwen3-5-moe` serves two checkpoints differing **only** in the router gates'
  width — 4-bit on 3.5, 8-bit on 3.6. Its arch-wide preset is the one true of
  both (`mlx-affine-b4-g64-qembed`), and 3.6's 8-bit gates are declared in the
  per-(stem, preset) overlay. Because the two variants are byte-identical on
  every other fingerprint axis, what keeps them apart at load time is the
  per-role affine bit map in the macro-emitted `fingerprint_matches` — the binary
  reads the packed widths rather than trusting a preset name
  (`crates/models/arch/tests/metal_o2_logits.rs:648-658`).

  **This supersedes the preset T0.1 names.** The task text says this arch
  declares `mlx-affine-b4-g64-gate8-qembed`; it no longer does, and the string
  survives in the tree only inside the comment explaining why it was removed.
  Building from that name fails feature resolution outright, which is why every
  preset above is quoted from the config.

#### R1 — "what a user gets"

| model | scratchy | ollama |
|---|---|---|
| `gemma-4-12b-it` | as pinned above | `gemma4:12b` — manifest digest and the quantization the tag actually carries: **T0.4** |
| `gemma-4-26b-a4b-it` | as pinned above | `gemma4:26b` — same: **T0.4** |
| `qwen3.6-35b-a3b` | as pinned above | `qwen3.6:35b` — same: **T0.4** |

The two sides load checkpoints produced by **different quantizers**
(mlx-community affine 4-bit against ollama's default GGUF). That is disclosed on
every R1 row rather than normalised away: R1 answers "what does a user get", and
only R2 isolates the engine.

#### R2 — "matched weights"

| model | scratchy | ollama |
|---|---|---|
| `gemma-4-12b-it` | **unchanged from R1** — that is the point | `gemma4:12b-mlx` — same checkpoint? **T0.6** |
| `gemma-4-26b-a4b-it` | unchanged from R1 | `gemma4:26b-mlx` — same checkpoint? **T0.6** |
| `qwen3.6-35b-a3b` | unchanged from R1 | `qwen3.6:35b-mlx` — same checkpoint? **T0.6** |

**R2 is gated, not assumed.** `gemma4:26b-mlx` reports 18 GB against the
mlx-community 4-bit's ~15.4 GB, so the tags are not obviously the same artifact.
T0.6 — the `-mlx` tag audit — pulls them and reads the manifest and config, and
R2 exists only once it names a concrete pair.

**The fallback, written down now so it is not invented later.** If T0.6 shows the
`-mlx` tags are a different checkpoint, R2 falls back to **bf16 against f16-GGUF
on `gemma-4-12b-it` only** — the smallest model in the set, so the one where a
conversion-based control is affordable. On the scratchy side that is a build with
no quant preset named (the dense/bf16 emission a preset would otherwise replace,
[`BUILD.md`](BUILD.md)), at 2 bytes per parameter rather than 4 bits — several
times the footprint, which is why the fallback is scoped to one model and not the
set.

#### What a rung does not pin

A rung fixes the **weights**. Four things it does not fix, each of which must
still appear on every published row:

1. **KV-cache dtype — not matchable at runtime on metal.** The `metal` feature
   implies `turboquant` (`crates/cli/scr/Cargo.toml:45`), and the codec is fixed
   when the model is built, from that feature plus the model's attention
   geometry. `--kv-cache-dtype` *asserts* the built-in codec and is refused when
   it disagrees (`crates/serving/worker/src/gpu_worker.rs:3322-3337`). So the
   match-or-disclose rule is satisfied here by **disclosure**: matching it would
   mean a second build without the feature, not a flag — the same point
   [`BENCHMARKING.md`](BENCHMARKING.md) §4 already makes against mlx-lm. How much
   KV each model even has is architectural:

   | model | layers | holds KV |
   |---|---|---|
   | `gemma-4-12b-it` | 48 | 8 full-attention; the other 40 are capped at a 1024-token sliding window, whose KV scratchy sizes to the window (`crates/serving/api/src/init.rs:3900-3955`) |
   | `gemma-4-26b-a4b-it` | 30 | 5 full-attention, 25 windowed |
   | `qwen3.6-35b-a3b` | 40 | 10 full-attention; the other 30 hold recurrent conv/ssm state, not KV |

   Those counts are also what the codec reaches — it compresses the
   full-context layers only. That is a disclosure about the engines being
   unmatched on KV dtype, which is separate from scratchy's own dense-vs-4-bit
   ablation; the latter is in phase 4's scope with its expected direction
   written down in [Scope and deferrals](#scope-and-deferrals) (**T0.8**).

2. **Context length.** `scr launch claude` emits `--max-model-len 65536`
   (`crates/cli/scr/src/commands/launch.rs:215`), well under these checkpoints'
   262144 ceiling. The matched-context rule requires ollama's
   `OLLAMA_CONTEXT_LENGTH` set to the same value and **no truncation on either
   side, proven** by per-turn token accounting — T0.4 for the ollama number.

3. **Which binary.** Per-tool relocatable spans are a Cargo feature, off by
   default (`crates/serving/api/Cargo.toml:47`, forwarded at
   `crates/cli/scr/Cargo.toml:100`), and the two arms do **not** produce identical
   text: the span renderer writes tool schemas as its own plain text instead of
   through the model's chat template, so the tool parser never fires and a request
   that asked for a tool call can come back as prose. The span arm is therefore a
   cross-build axis — owned by phase 4's ablations and by the separate work making
   that path output-neutral — and is orthogonal to the rung. A published row names
   its rung *and* its build.

4. **Concurrency.** `scr launch claude` defaults `--max-num-seqs` to 1, unlike
   `scr serve` — one interactive session needs one in-flight sequence, and
   batch 1 keeps the GDN recurrent-state pool small (61 MiB per slot on this MoE,
   so 256 slots reserve 15.7 GiB and OOM a 32 GiB box before the first token;
   `crates/cli/scr/src/args.rs:126-139`). It is a default, not a hard-code —
   raising it is one of phase 4's ablations.

## Blocked

Every cell this document could not fill, with what resolves it. Nothing here is
an estimate.

| cell | blocked on | resolved by |
|---|---|---|
| R1, ollama side — tag digest and the quantization each default tag carries | **T0.4** | pulling `gemma4:12b` / `gemma4:26b` / `qwen3.6:35b` and reading the manifest |
| R2, both sides — are the `-mlx` tags the same checkpoint as mlx-community's 4-bit? | **T0.6** | manifest + config of `gemma4:{12b,26b}-mlx` and `qwen3.6:35b-mlx`; R2 names a concrete pair, or the fallback above takes effect |
| ollama context length, parallelism, keep-alive, pinned version | **T0.4** | the one-page "ollama configuration as tested" note |
| no-truncation proof on both sides | **T0.4** + the phase 2 harness | per-turn token accounting in the replay client and the live driver |
| KV ablation: dense-vs-4-bit numbers | phase 4 | the dense arm needs `turboquant` split out of `metal` and a second compile per model; the prediction and the free startup-log measurement are recorded (**T0.8**) |
| server-side proof that the engine under test served the traffic | **T0.9** + phase 2's provenance check | request count and token totals recorded from the server, not from the client |
| every number | phase 4 | — |

## Artifacts

Empty — phase 4 produces them. Each run's JSON, the trace corpus revision, and
the `claude` version it was captured under land here.
