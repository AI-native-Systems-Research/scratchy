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
| [Scope and deferrals](#scope-and-deferrals) | **T0.8**, the deferral record |
| [Method and metrics](#method-and-metrics) | owned by the agreed method, not restated here |
| [Exact commands](#exact-commands) | scratchy side recorded; ollama side **T0.4** |
| [Results](#results) | empty — phase 4 |
| [Quantization rungs](#quantization-rungs--r1-and-r2) | **recorded — T0.5**; R2's pair audited, pinned and **verified by load** — **T0.6** |
| [Blocked](#blocked) | recorded |
| [Artifacts](#artifacts) | empty — phase 4 |

Task ids are the work breakdown's: `T0.x` are phase 0 (parity plumbing), and
each one is named where it is first cited.

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

**T0.8, the deferral record, fills this.** The TurboQuant KV-quant ablation is
deferred — not measured, not claimed — and the layer-coverage numbers behind
that decision belong here, so a later reader does not read silence as an
untested claim. The mechanism they rest on is recorded below, under [What a rung
does not pin](#what-a-rung-does-not-pin).

## Method and metrics

Owned by the agreed method and deliberately not restated here: the metric list,
the per-turn / per-task / per-session split, the seven fairness rules, and the
lane rule — every latency claim comes from Lane A (trace replay), every outcome
claim from Lane B (live Claude Code), neither ever quoted as the other. Phase 5
promotes that content into [`BENCHMARKING.md`](BENCHMARKING.md) as a §7; this
file carries only what is specific to a run.

### Exact commands

**scratchy — identical on both rungs**, because a rung changes the *comparator*
and the weights it loads, never scratchy's side.
`serve_argv` (`crates/cli/scr/src/commands/launch.rs:226-295`) is what
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

**ollama — R1's comparator, T0.4.** The tags are named per rung below; the
environment that must accompany them (`OLLAMA_CONTEXT_LENGTH`,
`OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE`, the pinned version, and what
`ollama launch claude` puts in Claude Code's environment) is T0.4's one-page note
and is not guessed here.

**Both rungs are ollama**, so there is no third engine to pin. R2 swaps only
the weights each side loads — see
[R2 as it will run](#r2-as-it-will-run--the-concrete-pair) for its two artifacts
and the one extra scratchy build (no quant preset) that rung needs.

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

| rung | the question it answers | engines | what differs | scope |
|---|---|---|---|---|
| **R1** "what a user gets" | the real-world one — install either engine, pull what its own users pull | scratchy vs ollama | engine **and** quantizer | all three models, Lane A **and** B |
| **R2** "matched weights" | how much of R1 was the engine | scratchy vs ollama | engine only — **the same 4-bit files**, imported into ollama | `gemma-4-12b-it` only ([why](#what-r2-does-not-cover-and-why-it-is-one-model)); every metric readable |

R1 is the headline. R2 keeps it from being dismissed as a quantizer comparison.

**Both rungs are ollama, and both are on `/v1/messages`.** The comparator set is
deliberately limited to servers exposing the Anthropic Messages API (@starpit's
call), because that is the only endpoint Claude Code speaks: a server that cannot
answer it cannot be measured on this workload at all. That rules out **mlx-lm**,
which was considered as R2's comparator precisely because it loads the *same*
mlx-community 4-bit files scratchy does — its `do_POST` registers
`/v1/completions`, `/v1/chat/completions` and `/chat/completions` only, and
`/v1/messages` 404s. Recorded because it is the road not taken for a structural
reason, not an oversight:

| candidate for R2 | matched weights? | serves `/v1/messages`? | verdict |
|---|---|---|---|
| ollama `-mlx` tags, 4-bit | **no** — ollama's own NVFP4/MXFP8 ([audit](#r2--matched-weights)) | yes | ruled out: not the same checkpoint |
| ollama `gemma4:12b-mlx-bf16` | yes, byte-verified | yes | ruled out: bf16 only, bandwidth-bound, one model |
| **the mlx-community 4-bit checkpoint, `ollama create`d** | **yes, byte-verified** | **yes** | **R2** — same files, 4-bit, every metric readable |
| mlx-lm on the 4-bit checkpoint | yes — literally the same files | **no** | ruled out: cannot serve this workload |

**What R2 costs, stated plainly.** Not regime — `ollama create` imports the
mlx-community checkpoint *preserving its quantization*, so R2 runs the same
4-bit weights users run, and every metric is readable. What it costs is
**coverage**: ollama's importer discards per-tensor bit-width overrides, which
only the 12b has none of, so R2 reaches **one model of three** and the two MoE
models' R1 rows inherit their quantizer control by analogy from the dense sibling
rather than directly. Measured per model under
[What R2 does not cover](#what-r2-does-not-cover-and-why-it-is-one-model).

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

| model | scratchy | ollama | tag carries (read 2026-10-06, **T0.6**) |
|---|---|---|---|
| `gemma-4-12b-it` | as pinned above | `gemma4:12b` | `gguf` / `Q4_K_M`, 8.02 GB, + a 0.47 GB GGUF draft model, **`draft_num_predict 3`** |
| `gemma-4-26b-a4b-it` | as pinned above | `gemma4:26b` | `gguf` / `Q4_K_M`, 18.73 GB, + a 0.46 GB GGUF draft model and a 1.19 GB vision projector, **`draft_num_predict 3`** |
| `qwen3.6-35b-a3b` | as pinned above | `qwen3.6:35b` | `gguf` / `Q4_K_M`, 22.62 GB, native MTP head inside the GGUF, **`draft_num_predict 2`** |

Digests for all three are in
[R2's table below](#how-this-was-established-and-how-to-re-verify-it). **T0.4**
still owns the rest of the ollama side (context length, parallelism, keep-alive,
pinned engine version, and the no-truncation proof).

The two sides load checkpoints produced by **different quantizers**
(mlx-community affine 4-bit against ollama's default GGUF `Q4_K_M`). That is
disclosed on every R1 row rather than normalised away: R1 answers "what does a
user get", and only R2 isolates the engine. **All three default tags are also
speculative-decoding builds with the feature switched on**, which scratchy's side
has no equivalent of — see
[Speculative decoding on the ollama side](#speculative-decoding-on-the-ollama-side).
It is the largest single fairness item this audit found, and it bears on R1's
rows, not just R2's.

#### R2 — "matched weights"

**Audited — T0.6, 2026-10-06. The `-mlx` tags are not mlx-community's
checkpoint, which is why R2 imports that checkpoint into ollama instead of
pulling a tag.** The tags
are ollama's *own* requantization of the same base models into
**microscaling float** formats — NVFP4 and MXFP8 — not MLX affine int4. Read from
each tag's registry manifest and its per-tensor safetensors `__metadata__`
(method below; no full pull required):

| ollama tag | ollama's own label | what the tensors actually are | main-model bytes vs scratchy's 4-bit |
|---|---|---|---|
| `gemma4:12b-mlx` | `safetensors`, `file_type` **empty** | **NVFP4** `g16` on every language-model matmul, `embed_tokens` included; **bf16** vision tower, audio tower and the bundled draft model | 6.804 GB vs 6.741 GB — **+0.93%** |
| `gemma4:26b-mlx` | `safetensors`, `file_type` `nvfp4` | **mixed**: NVFP4 `g16` on `q/k/o_proj`, `mlp.{gate,up}_proj`, `experts.gate_up_proj`; **MXFP8 `g32`** on `v_proj`, `mlp.down_proj`, `experts.down_proj` and `embed_tokens`; **bf16** `router.proj` and the whole vision tower | 17.431 GB vs 15.341 GB — **+13.62%** |
| `qwen3.6:35b-mlx` | `safetensors`, `file_type` `nvfp4` | **NVFP4** `g16` on attention, the GDN `linear_attn` projections, experts and shared expert; **bf16** `embed_tokens`, `lm_head`, `mlp.gate`, the whole visual tower, and a 1.690 GB `mtp` multi-token-prediction head | 21.890 GB vs 20.402 GB — **+7.29%** |

`gemma4:12b-mlx` is byte-identical to `gemma4:12b-nvfp4`, and `qwen3.6:35b-mlx`
to `qwen3.6:35b-a3b-nvfp4` (same manifest digest) — on those two, `-mlx` *is* the
pure-NVFP4 build. On the 26b it is not: `gemma4:26b-mlx` and `gemma4:26b-nvfp4`
are different manifests, and **both** are mixed-precision, differing in which
roles get 8 bits. There is no uniform-4-bit ollama artifact for the 26b at all.

**Different storage would not matter; different dequantization math does.** The
same tensor, `…layers.0.self_attn.k_proj`, on the 12b:

| | scratchy — `mlx-community/gemma-4-12B-it-4bit` | ollama — `gemma4:12b-mlx` |
|---|---|---|
| packed weight | `U32 [2048, 480]` | `U32 [2048, 480]` |
| companions | `.scales` `BF16 [2048, 60]`, `.biases` `BF16 [2048, 60]` | `.scale` `U8 [2048, 240]`, `.global_scale` `F32 []` |
| scheme | affine int4, scale **and zero-point**, group **64** (`3840/64 = 60`) | NVFP4 E2M1, E4M3 block scale, **no** zero-point, group **16** (`3840/16 = 240`), plus one tensor-wide FP32 scale |

The packed weight shape is **identical**, which is exactly why the 12b's +0.93%
size delta would have passed a footprint check and still been the wrong artifact.
The 26b's 18-vs-15.4 GB gap that prompted this task was the easy case; the 12b was
the trap. **Footprint does not identify a checkpoint — per-tensor metadata does.**

**ollama's own labels do not identify it either.** `gemma4:26b-mlx` declares
`file_type: nvfp4` while four of its weight roles — `v_proj`, `mlp.down_proj`,
`experts.down_proj`, `embed_tokens` — are MXFP8 at twice the bit width;
`gemma4:12b-mlx` declares no `file_type` at all; and the `config.json` these
tags ship carries **no `quantization` block** and says `dtype: bfloat16`,
describing the *base* model rather than the artifact. Only the per-tensor
`__metadata__` is truthful.

##### R2 as it will run — the concrete pair

**One model, `gemma-4-12b-it`, the *same 4-bit files* on both sides, on
`/v1/messages`.** scratchy's side is its R1 build, unchanged. ollama's side is the
mlx-community checkpoint imported into ollama with `ollama create`, which
**preserves the source quantization** rather than requantizing — so both engines
read the identical affine-int4 weights at the width users actually run.

| side | artifact | invocation |
|---|---|---|
| scratchy | `mlx-community/gemma-4-12B-it-4bit` @ `73bcf09092aa277861d5a191b989b666f7f32e8f` — **its R1 build and preset, unchanged** | `-F metal,serve,model/gemma-4-12b-it,quant/mlx-affine-b4-g64` |
| ollama | the **same HF snapshot directory**, imported | `printf 'FROM <snapshot-dir>\n' > Modelfile && ollama create cc-g12-mlx4 -f Modelfile` |

The snapshot path carries the revision, so provenance is structural rather than
recorded by hand:
`~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-4bit/snapshots/73bcf09092aa…/`.
This is **better provenance than a published tag**, which floats (see
[the republication finding](#how-this-was-established-and-how-to-re-verify-it)).

**Verified end to end on this host, 2026-10-06, ollama 0.34.2** — a load and a
tool call, which is T0.1's bar, not a size comparison:

| check | result |
|---|---|
| import | `1341 tensors, preserving source quantization` → `file_type: int4`, per-tensor `__metadata__: {"group_size": "64", "quant_type": "int4"}` |
| total bytes | 6,773,236,528 B — the HF repo's own 6.773 GB |
| weights byte-identical | `.weight`, **`.scale` and `.bias`** all SHA-256 equal to the source on every tensor sampled (ollama renames `.scales`/`.biases` → `<name>.weight.scale`/`.bias`; bytes unchanged) |
| generation | `"What is the capital of France?"` → `"The capital of France is Paris."` — dequant is correct, not merely non-crashing |
| `/v1/messages` + tools | real `tool_use` block, `stop_reason: "tool_use"`, with `renderer`/`parser` `gemma4` |

**Three things this pairing fixes outright**, rather than caveating:

1. **Regime.** 6.741 GB per token on both sides, not 23.920 GB. The
   bandwidth-bound objection to a bf16 rung does not apply, so **every metric is
   readable — TPOT and ITL included.**
2. **No speculative-decoding asymmetry.** The import carries no `image.draft`
   layer, no `draft.*` tensors and **no `params` blob**, so no `draft_num_predict`
   — unlike the published `gemma4:12b`, which sets it to 3
   ([below](#speculative-decoding-on-the-ollama-side)). On R2 both engines decode
   one token at a time, so R2 isolates the engine *more* cleanly than R1 does.
3. **bf16 is not needed at all.** The byte-identical bf16 pair this audit also
   found (`gemma4:12b-mlx-bf16` against `mlx-community/gemma-4-12B-it-bf16`, equal
   to 551 B on 23.92 GB) is kept only as a recorded alternative; R2 does not use
   it, because a 4-bit match on the same files is strictly better.

Two disclosures remain:

- **Storage layout differs, contents do not.** ollama stores one single-tensor
  safetensors file per tensor; HF ships five shards. Same bytes, different file
  granularity, which can touch load/mmap behaviour — so it is named on
  startup-time rows, not on steady-state ones.
- **KV-cache dtype stays unmatched.** `metal` implies `turboquant` and the codec
  is fixed at build time; matching it would mean a second build, not a flag. Same
  match-or-disclose resolution as [item 1 below](#what-a-rung-does-not-pin).

###### What R2 does not cover, and why it is one model

**`ollama create` drops per-tensor bit-width overrides.** It copies tensor bytes
faithfully but writes the checkpoint's *global* width onto every quantized
tensor, discarding the per-tensor exceptions the
[scratchy table](#the-scratchy-side-pinned--the-same-on-both-rungs) lists. That is
survivable only for a checkpoint with no exceptions — which, of the three, is the
12b alone. Measured per model:

| checkpoint | per-tensor overrides | `ollama create` | result |
|---|---|---|---|
| `gemma-4-12B-it-4bit` | **none** | clean | ✅ **R2** |
| `Qwen3.6-35B-A3B-4bit` | 80 — `mlp.gate`, `mlp.shared_expert_gate` at 8 bits | imports, **relabels them `int4`** | ❌ load panics |
| `gemma-4-26b-a4b-it-4bit` | 30 — `router.proj` at 8 bits | **prints success, creates nothing** | ❌ no model |

The qwen failure is worth quoting, because it is this epic's own footgun seen from
the other side. `…layers.0.mlp.gate.weight` is `U32 [256, 512]` with scales
`[256, 32]`; at `hidden 2048` and `group_size 64` the scales fix `in_features` at
`32 × 64 = 2048`, and `2048 / 512 = 4` values per `u32` — **8 bits**, exactly as
the checkpoint's override declares. ollama labels it `int4` and its runtime then
refuses the tensor it was handed:

```
mlx: [quantized_matmul] The shapes of the weight and scales are incompatible
based on bits and group_size. w.shape() == (256,512) and scales.shape() == (256,32)
with group_size=64 and bits=4
```

Compare T0.1's scratchy-side failure on the same role,
`affine_dequant_b4: scales shape [256, 32] != [256, 16]`. Same tensor, same
arithmetic, same lesson in both engines: **per-tensor quantization detail comes
from the checkpoint; a global default is not a substitute.** It fails loudly,
which is the good outcome — a mislabel that *ran* would be far worse.

The 26b's failure mode is worse than the qwen's and should be treated as a trap:
`ollama create` prints `writing manifest` and
`successfully imported mlxtest-g26 with 1048 layers`, **exits 0, never contacts
the daemon, and creates no model** — `ollama list` does not show it afterwards.
Any automation that trusts that exit code will record a model it does not have.
**The runner must assert the model exists after an import, not trust `create`.**

**So R2 is the dense 12b, and the two MoE models' R1 rows inherit their quantizer
control by analogy** from the dense sibling rather than directly. If an R1 result
on the 26b or qwen is challenged as a quantizer artifact, the strict rebuttal
covers the 12b only. That is weaker than silence would imply, so it is stated
rather than left for a reader to notice.

Worth separating, because it changes what the limitation *is*: this is now an
**ollama importer bug**, not a physical constraint. The earlier bf16 framing was
bounded by this host's 64 GiB; this one would lift the moment ollama preserves
per-tensor widths, or if the two MoE checkpoints were republished without
exceptions. It is listed under [Blocked](#blocked) as upstream-fixable, not closed.

One non-option, recorded so it is not proposed later: stripping the overrides from
the checkpoint's `config.json` to force a uniform 4-bit import would make the
labels agree with nothing — the gate *data* is still 8-bit — and would corrupt
exactly the way scratchy's `-mlp8-router8` preset does. The fix is upstream, not
in the config.

###### One gotcha for any future import: `mode` must be declared

An older mlx-community checkpoint whose `config.json` gives
`quantization: {group_size, bits}` **without** `"mode": "affine"` imports with
empty per-tensor metadata and then panics at load:

```
mlx: [dequantize] Invalid quantization mode ''.
```

Measured on `mlx-community/SmolLM-135M-Instruct-4bit`, which lacks the field;
adding `"mode": "affine"` to the config makes the same import emit
`{"group_size": "64", "quant_type": "int4"}` and run. All three checkpoints in
this comparison declare `mode: affine` already, so none needs patching — but a
fourth model might, and the symptom names neither the cause nor the fix.

Speculative decoding is a disclosure on **R1**; it is written out once below.

##### Speculative decoding on the ollama side

This belongs to T0.6 because it is a property of the *artifacts*, and it lands on
**R1 as much as R2**. **Every default tag in this comparison is a
speculative-decoding build with the feature switched on**, by two different
mechanisms, and scratchy's side has no equivalent enabled.

The switch is `draft_num_predict` in each tag's `params` blob; the machinery is
either a bundled draft model or the model's own multi-token-prediction head:

| ollama tag | rung | machinery the artifact carries | drafter architecture | `draft_num_predict` |
|---|---|---|---|---|
| `gemma4:12b` | R1 | GGUF draft model, 0.47 GB | `gemma4-assistant` | **3** |
| `gemma4:26b` | R1 | GGUF draft model, 0.46 GB | `gemma4-assistant` | **3** |
| `qwen3.6:35b` | R1 | native **MTP** head, inside the GGUF | — (in-model) | **2** |
| `gemma4:12b-mlx` | — | bf16 draft model, 0.85 GB | `Gemma4UnifiedAssistantForCausalLM` | unset |
| `gemma4:26b-mlx` | — | bf16 draft model, 0.84 GB | `Gemma4AssistantForCausalLM` | unset |
| `qwen3.6:35b-mlx` | — | native **MTP** head, 1.69 GB bf16 | — (in-model) | unset |
| `gemma4:12b-mlx-bf16` | recorded alternative | bf16 draft model, 0.85 GB | `Gemma4UnifiedAssistantForCausalLM` | unset |
| **R2's import** (`ollama create`) | **R2** | **none — no draft layer, no `params` blob** | — | **absent** |

**The drafter is not one architecture across tags.** Both GGUF tags declare
`gemma4-assistant`, but on the MLX side the 12b ships
`Gemma4UnifiedAssistantForCausalLM` (`model_type: gemma4_unified_assistant`) and
the 26b ships `Gemma4AssistantForCausalLM` (`model_type: gemma4_assistant`) —
read from each tag's config blob and confirmed against its bundled
`draft/config.json`. A row that names "ollama's draft model" generically is
describing three different things, so each is named per tag above.

Two things follow, and they differ by rung:

- **On R1 it is on, explicitly.** `draft_num_predict` is set on all three default
  tags. A phase-4 row that compares scratchy with n-gram spec decode *off*
  against these tags is comparing against an engine drafting 2–3 tokens a step
  with a trained drafter. That inverts the sign of the spec-decode ablation.
- **On the `-mlx` tags the weights ship but the knob is unset** — the bf16
  control included. Unset is not the same as off — it may fall through to an engine
  default — so **this is a runtime question T0.4 must answer by measuring, not by
  reading the manifest.** T0.6's claim stops at what the artifact contains.

**`-mtp-` is a tag-naming convention, not a separate option, and the defaults
resolve to it.** `qwen3.6:35b` and `qwen3.6:35b-a3b-mtp-q4_K_M` share config
digest `sha256:99afa5bbf1e7…` with byte-identical layers; `gemma4:26b` and
`gemma4:26b-a4b-it-mtp-q4_K_M` share `sha256:cd16db7156ed…`. **The default tag
*is* the MTP build in both cases.** Non-spec-decode builds do exist —
`gemma4:26b-a4b-it-q4_K_M` (`draft: null`, no draft layer) and
`qwen3.6:35b-a3b-q4_K_M` — but they are *different and older* builds
(`model_type` 25.8B vs 25.2B, `requires 0.20.0` vs `0.30.9`), so swapping to them
changes more than the one variable. Prefer setting `draft_num_predict` on the
default tag to switching tags.

**scratchy cannot match qwen's MTP on the checkpoint it loads, and not because of
the engine.** Upstream `Qwen/Qwen3.6-35B-A3B` ships the head — 19 tensors under
`mtp.{fc,layers,norm,pre_fc_norm_embedding,pre_fc_norm_hidden}` — and ollama's
`-mlx` tag preserves all of it. **mlx-community's conversions drop it: 0 of 19 in
both `Qwen3.6-35B-A3B-4bit` and `-bf16`**, while the `config.json` they ship still
declares `mtp_num_hidden_layers: 1`. So the config advertises a head whose weights
are not in the file. Any qwen MTP comparison needs a checkpoint that retains them;
that is a checkpoint problem before it is a scratchy feature request.

**What it would take to put *ollama* on a matched-weights rung, recorded so it is
not re-derived.** R2 no longer needs this — `ollama create` supplies matched
4-bit weights on the 12b. It is what it would take to compare against ollama's
*own published* `-mlx` artifacts instead. scratchy already has an `nvfp4` preset
at `group_size 16`, matching ollama's, but only the `llama` arch declares it, and
there is no MXFP8 support in the tree at all
(`crates/models/quantization/presets/nvfp4.json`,
`crates/layers/src/layers.rs:237-240`). It would need NVFP4 declared on
`gemma4`/`gemma4-moe`/`qwen3-5-moe`, MXFP8 added for the 26b, **and** a way to
feed scratchy ollama's per-tensor blob store, since these weights are published
nowhere else. The converse — importing mlx-community's checkpoint into ollama via
`ollama create` — is cheaper to test and
[listed as open](#blocked). Either is an epic, not a phase-0 task.

##### How this was established, and how to re-verify it

No full pull was needed, and none was made: ollama's registry serves manifests and
blobs over plain HTTPS, and a tensor's quantization is in the first ~300 bytes of
its blob.

```bash
# the manifest: every tensor as its own named layer
curl -s https://registry.ollama.ai/v2/library/gemma4/manifests/26b-mlx | jq .

# the pinnable identity (tag -> digest) and when it was pushed
curl -sI -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  https://registry.ollama.ai/v2/library/gemma4/manifests/26b-mlx \
  | grep -i 'ollama-content-digest\|ollama-push-time'

# one tensor's quantization, from its safetensors header
curl -sL -H 'Range: bytes=0-4095' \
  https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:<tensor-digest> \
  | python3 -c 'import sys,struct,json; d=sys.stdin.buffer.read(); \
n=struct.unpack("<Q",d[:8])[0]; print(json.dumps(json.loads(d[8:8+n]),indent=1))'
```

**Pin ollama tags by manifest digest, not by tag name — they float.** Measured
here: the `gemma4:26b` manifest pulled to this machine on 2026-09-29 names config
`sha256:62b183484ba7…` with a 16,947,541,728 B model layer, while the registry on
2026-10-06 serves config `sha256:cd16db7156ed…` with a 17,074,419,072 B one. The
tag was republished 2026-09-30 02:35 UTC and the weights moved by 127 MB under a
fixed name. Digests for every tag named in this document, as of 2026-10-06:

| tag | pushed (UTC) | manifest digest |
|---|---|---|
| `gemma4:12b` | 2026-09-30 02:34 | `sha256:6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b` |
| `gemma4:12b-mlx` | 2026-08-27 18:23 | `sha256:ded7a27350032202d9e9b2a6071e8aa89959ab156771b5228f30863c741c4970` |
| `gemma4:12b-mlx-bf16` | 2026-08-14 18:22 | `sha256:ae28af21156f7155ac3608617f0516c7a8acd8c9553f4192df9c2b5105770179` |
| `gemma4:26b` | 2026-09-30 02:35 | `sha256:001e5dafc3c77684c2307ebc6ab8e336e10c9b18eca52acf547d72fc83c3ca8c` |
| `gemma4:26b-mlx` | 2026-08-27 18:23 | `sha256:f0fc7e0ae4947d382989b1f57db3d098a86b87c5b6e9523b9561ba81a2a64879` |
| `qwen3.6:35b` | 2026-09-30 19:53 | `sha256:a7eb95c53bcf96b4bdd008d0fab4a5dac88047d9c1a7a9ab88ed453423fbd87c` |
| `qwen3.6:35b-mlx` | 2026-08-27 18:23 | `sha256:e92a3e94bbca90a85491dc34e9257bfee2318cedaa16360828c4d8edf14295b9` |

A load is still the acceptance test, the same bar T0.1 sets: these digests say
what the artifacts *are*, not that any engine opens them. Every artifact a rung
names is loaded and served before a number is quoted from it.

#### What a rung does not pin

A rung fixes the **weights**. Four things it does not fix, each of which must
still appear on every published row:

1. **KV-cache dtype — not matchable at runtime on metal.** The `metal` feature
   implies `turboquant` (`crates/cli/scr/Cargo.toml:45`), and the codec is fixed
   when the model is built, from that feature plus the model's attention
   geometry. `--kv-cache-dtype` *asserts* the built-in codec and is refused when
   it disagrees (`crates/serving/worker/src/gpu_worker.rs:3066-3081`). So the
   match-or-disclose rule is satisfied here by **disclosure**: matching it would
   mean a second build without the feature, not a flag — the same point
   [`BENCHMARKING.md`](BENCHMARKING.md) §4 already makes against mlx-lm. How much
   KV each model even has is architectural, and T0.8 records the consequence:

   | model | layers | holds KV |
   |---|---|---|
   | `gemma-4-12b-it` | 48 | 8 full-attention; the other 40 are capped at a 1024-token sliding window, whose KV scratchy sizes to the window (`crates/serving/api/src/init.rs:3841-3907`) |
   | `gemma-4-26b-a4b-it` | 30 | 5 full-attention, 25 windowed |
   | `qwen3.6-35b-a3b` | 40 | 10 full-attention; the other 30 hold recurrent conv/ssm state, not KV |

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
| ~~R1, ollama side — tag digest and the quantization each default tag carries~~ | — | **resolved, T0.6**: all three are `gguf`/`Q4_K_M`; digests and sizes recorded in [R1](#r1--what-a-user-gets) and [R2](#how-this-was-established-and-how-to-re-verify-it) |
| ~~R2, both sides — are the `-mlx` tags the same checkpoint as mlx-community's 4-bit?~~ | — | **resolved, T0.6**: **no** — ollama's own NVFP4/MXFP8 requantization. But `ollama create` imports mlx-community's checkpoint *preserving its quantization*, so R2 runs the **same 4-bit files** on both engines on `gemma-4-12b-it`, verified by load and a `tool_use` round-trip |
| ~~whether either engine loads R2's pair~~ | — | **resolved, T0.6**: verified on this host — import preserves the quantization byte-for-byte, generates correctly, and round-trips a `tool_use` on `/v1/messages` |
| per-tensor bit-width overrides dropped by `ollama create` | **upstream ollama** | blocks R2 on the 26b and qwen; fails loudly on load (qwen) or silently at import (26b). Not a host limit — it lifts if ollama preserves per-tensor widths |
| `ollama create` can exit 0 having created nothing | **T0.7 runner** | assert the model is in `ollama list` after an import; never trust the exit code |
| a matched-weights control for the **two MoE models** | **upstream ollama** (was: this host's RAM) | the importer drops their 8-bit router/gate overrides; the MoE R1 rows inherit the 12b's control by analogy, disclosed per row |
| ~~does `ollama create` preserve an mlx-community affine-int4 checkpoint?~~ | — | **resolved, T0.6**: **yes** for a uniformly-quantized checkpoint — that is what R2 now rests on. No for one with per-tensor overrides (row above) |
| does ollama's MLX engine draft when `draft_num_predict` is **unset**? | **T0.4** | every `-mlx` tag ships draft weights with the knob unset; unset is not off. Measure it — a manifest cannot answer it |
| ollama spec decode — disabled, or measured and disclosed, on **R1**? | **T0.4** + phase 4's spec-decode ablation | all three default tags set `draft_num_predict` 2–3; prefer overriding it on the default tag over switching to the older non-MTP builds |
| ollama context length, parallelism, keep-alive, pinned version | **T0.4** | the one-page "ollama configuration as tested" note |
| no-truncation proof on both sides | **T0.4** + the phase 2 harness | per-turn token accounting in the replay client and the live driver |
| KV-quant deferral and its layer-coverage evidence | **T0.8** | the [Scope and deferrals](#scope-and-deferrals) section above |
| server-side proof that the engine under test served the traffic | **T0.9** + phase 2's provenance check | request count and token totals recorded from the server, not from the client |
| every number | phase 4 | — |

## Artifacts

Empty — phase 4 produces them. Each run's JSON, the trace corpus revision, and
the `claude` version it was captured under land here.
