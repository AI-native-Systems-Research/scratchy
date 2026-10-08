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
| [Quantization rungs](#quantization-rungs--r1-and-r2) | **recorded — T0.5** |
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

**scratchy — identical on both rungs**, because a rung changes only what ollama
loads. `serve_argv` (`crates/cli/scr/src/commands/launch.rs:226-295`) is what
turns these into the spawned server's argv, so what is written here is what the
server gets:

```bash
# the dense control — the model the harness is built against
cargo build --release -p scratchy-cli \
  --features metal,claude,model/gemma-4-12b-it,quant/mlx-affine-b4-g64
scr launch claude -m mlx-community/gemma-4-12B-it-4bit --device metal

# the MoE carrying the prefix-cache and tool-span headline
cargo build --release -p scratchy-cli \
  --features metal,claude,model/gemma-4-26b-a4b-it,quant/mlx-affine-b4-g64
scr launch claude -m mlx-community/gemma-4-26b-a4b-it-4bit --device metal

# MoE + GDN: prefix caching is off by design, so this one measures the cost
cargo build --release -p scratchy-cli \
  --features metal,claude,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed
scr launch claude -m mlx-community/Qwen3.6-35B-A3B-4bit --device metal
```

One model per binary on purpose: naming `model/all` forward-expands every config
in scope ([`BUILD.md`](BUILD.md)), which is minutes-to-hours and an OOM risk on a
laptop.

`claude`, not `serve`, is the feature these builds need: `scr launch claude` is
gated on `claude` (`crates/cli/scr/src/main.rs:106`), and `claude = ["serve"]`
is one-directional — a `serve` build has the server but not the subcommand that
drives it.

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

### Proving the engine under test served the traffic

**T0.9.** `scr launch claude` points Claude Code at the local server with
`ANTHROPIC_BASE_URL` and the model-tier variables. A settings file's `env` block
*overwrites* the process environment Claude Code inherits, so a developer with
`ANTHROPIC_BASE_URL` in `~/.claude/settings.json` would measure a different
endpoint than the one under test — and nothing would fail. Launch now sets those
values twice: in the environment, and through `claude --settings`, which
outranks the user and both project files.

The same applies to **provider selection**, which bypasses `ANTHROPIC_BASE_URL`
rather than competing with it. `CLAUDE_CODE_USE_BEDROCK`, `…_VERTEX`,
`…_FOUNDRY`, `…_MANTLE` and `…_ANTHROPIC_AWS` each route to a provider with its
own endpoint variable, so one of them left set in a settings file sends the run
there and our base URL is never consulted. Claude Code's `/setup-bedrock`
wizard writes one into `~/.claude/settings.json`, so this is a configuration a
developer gets by following the documented setup. Launch blanks all five.

**Neither of those is the guarantee.** Managed settings rank above
`--settings`, and three of the five managed delivery mechanisms are not files at
all — a macOS configuration profile, the Windows registry, and server-managed
policy held by Anthropic or a gateway. Launch deliberately does **not** scan the
settings files to warn about what it cannot override: a scan can only ever see
the file-based sources, so it would reassure in exactly the cases it cannot
detect, and the request count below catches all of them without reading anyone's
configuration. Keys launch does not set — a per-provider
`ANTHROPIC_BEDROCK_BASE_URL`, an `ANTHROPIC_CUSTOM_HEADERS` — are likewise
caught by the count rather than by inspection.

What makes a run trustworthy is therefore the server's own count:
`GET /server_info` reports `requests_served`, incremented once per arriving
request excluding the liveness and introspection routes
(`server::COUNTED_EXCLUDED_ROUTES`). Launch reads it before and after a session
and reports the delta. In `-p`/`--print` mode — how a harness drives this — a
zero count **fails the command**, because a run given a prompt that served no
requests sent that prompt somewhere else; a harness therefore learns it from the
exit status rather than by scraping stderr. Interactively a zero count is only
warned about, since exiting without sending anything legitimately serves none.

An unreadable count is reported as such and never fails, and is not the same
claim as zero: `requests_served` is `Option<u64>`, so a server too old to report
it reads as "cannot say" rather than as a definite zero.

Two properties matter for fairness. The counter is **always compiled**, unlike
everything behind the non-default `metrics` feature, so a measured build carries
no instrumentation a user's build would not — and it is **one atomic add per
request**, nothing per token, so it is not on the decode path. A published
number needs no disclosure about its own measurement apparatus.

**Request counts only.** Server-side token totals would cross-check the
client's own accounting, which is what the no-truncation row above needs, but
they answer a different question from "did the traffic arrive here" and are
deliberately not part of T0.9. They remain open in the table below.

## Blocked

Every cell this document could not fill, with what resolves it. Nothing here is
an estimate.

| cell | blocked on | resolved by |
|---|---|---|
| R1, ollama side — tag digest and the quantization each default tag carries | **T0.4** | pulling `gemma4:12b` / `gemma4:26b` / `qwen3.6:35b` and reading the manifest |
| R2, both sides — are the `-mlx` tags the same checkpoint as mlx-community's 4-bit? | **T0.6** | manifest + config of `gemma4:{12b,26b}-mlx` and `qwen3.6:35b-mlx`; R2 names a concrete pair, or the fallback above takes effect |
| ollama context length, parallelism, keep-alive, pinned version | **T0.4** | the one-page "ollama configuration as tested" note |
| no-truncation proof on both sides | **T0.4** + the phase 2 harness | per-turn token accounting in the replay client and the live driver |
| KV-quant deferral and its layer-coverage evidence | **T0.8** | the [Scope and deferrals](#scope-and-deferrals) section above |
| per-run provenance record (which config served how many requests) | phase 2's provenance check | `GET /server_info` carries both `scratchy_core_config` and `requests_served`; **T0.9 landed the count** (see [Proving the engine under test served the traffic](#proving-the-engine-under-test-served-the-traffic)) |
| server-side **token totals**, as a cross-check on the client's own accounting | still open — **not** delivered by T0.9 | a server-side counter alongside `requests_served`, or the `metrics` feature's `prompt_tokens_total` / `output_tokens_total` if a build is willing to carry it |
| every number | phase 4 | — |

## Artifacts

Empty — phase 4 produces them. Each run's JSON, the trace corpus revision, and
the `claude` version it was captured under land here.
