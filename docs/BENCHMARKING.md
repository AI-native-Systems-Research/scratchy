# Benchmarking scratchy against other frameworks

Methodology for the head-to-head comparisons, and the metric definitions they
report.

| harness | question | scenarios |
|---|---|---|
| [`scripts/bench_serve_compare.sh`](../scripts/bench_serve_compare.sh) | steady-state serving throughput and latency under load | input × output × concurrency sweep |
| `bench_serve_compare.sh --multi-turn-cells` | how much prefix reuse cuts TTFT on a conversation's later turns | system prompt × turns × concurrency ([multi-turn cells](#multi-turn-prefix-reuse-cells)) |
| `scr bench startup --exec` | how long from `exec` until the user sees a word | frozen / cold / warm cache ladder |
| `scr bench startup` (no `--exec`) | in-process engine construction cost | cold / warm iterations |

## `--exec` vs plain `bench startup` — two different measurements

`scr bench startup` without `--exec` mirrors
[vLLM's `bench startup`](https://docs.vllm.ai/en/latest/cli/bench/startup/) down to
`--num-iters-cold` / `--num-iters-warmup` / `--num-iters-warm`. The two modes are
not interchangeable:

| | plain `bench startup` | `bench startup --exec` |
|---|---|---|
| timed region | `LLMBuilder::build()`, in-process (`crates/benches/src/startup.rs`) | `exec` → first content byte of the first token |
| process | one process, N engine constructions | fresh `fork`/`exec` per measurement |
| generates tokens | no — never sends a request, so **there is no TTFT to report** | yes; the clock stops on the first token byte |
| what "cold" means | a fresh engine object; the HF cache is explicitly *not* wiped and the page cache is untouched | the FROZEN/COLD rungs below, with the eviction *verified* |
| sees `exec`, dyld, first-touch faults | no, by construction | yes — these dominate a real first launch |
| other frameworks | no; it constructs scratchy's own `LLM` | yes, via `--child-cmd` |

They are complementary. Plain `bench startup` is the cheap, repeatable way to
watch engine-init cost for regressions — no sudo, no second framework,
percentiles over iterations in one process. `--exec` is what a *user-perceived*
or *cross-framework* claim requires.

**Do not put their numbers in the same table.** Plain `bench startup` will always
look faster, for the uninteresting reason that it is measuring less.

---

## 1. The cache ladder

"Cold start" is not one thing — it is a stack of caches, each independently warm.
Naming a single "cold" number without saying which were populated is how startup
benchmarks become unfalsifiable. So the ladder is explicit:

| surface | FROZEN | COLD | WARM |
|---|---|---|---|
| OS page cache (weights, binary, dylibs) | evicted | warm | warm |
| derived on-disk caches (`--remove-path`) | **removed** | present | present |
| checkpoint on disk | present | present | present |
| process | fresh `exec` | fresh `exec` | resident, ≥1 request served |
| pipelines / KV pool / warmup | rebuilt | rebuilt | done |

- **FROZEN** — first run ever on this machine, short of downloading weights.
- **COLD** — the honest everyday case: you ran it before, the machine has been
  busy but not enough to evict the weights, and you launch again.
- **WARM** — a server already up and serving. Isolates request latency from all
  startup cost.

COLD and WARM say "you ran this before", so when nothing has run yet the harness
performs **one throwaway priming launch** and discards it. Without that, the
first repetition of a fresh run measures FROZEN while being labelled COLD — on a
Metal run before priming existed, COLD rep 0 took 7.935 s with ~31,570 major
faults and rep 1 took 1.218 s with ~0, and the rung's median averaged the two. A
FROZEN repetition earlier in the list already populates the caches, so priming is
skipped in that case.

### Eviction is platform-specific, and one obvious choice is wrong

`--evict purge` (macOS) drops the whole unified buffer cache. It is symmetric: it
evicts CPython and a framework's dylibs exactly as it evicts the scratchy binary,
which is what makes a cross-framework FROZEN fair. **`purge` requires root**, so
it runs as `sudo -n purge` — cache the credential with `sudo -v` before starting.
The `-n` is deliberate: a benchmark must not block on a password prompt partway
through, and a FROZEN run is checked for the credential up front rather than
failing after it has already disturbed the cache state it needed.

`--evict fadvise` (Linux) calls `posix_fadvise(POSIX_FADV_DONTNEED)` over
`--evict-path`. It is deliberately **not** `drop_caches`: that file is not
namespaced, so writing it from a container evicts the *host's* entire page cache
and perturbs every other workload on a shared node. fadvise is unprivileged and
surgical, and read-only mmap'd weight shards are exactly the clean-page case
where `DONTNEED` is reliable. It needs `--evict-path`, and says so rather than
silently evicting nothing.

`--evict none` leaves the page cache alone; FROZEN then collapses toward COLD.

## 2. Metrics

One external stopwatch, held by the benchmark process, with **the same code for
every backend**. That single shared implementation is the fairness guarantee:
nothing is self-reported.

- **`ttft_exec`** — seconds from `exec` to the first token byte. The headline.
  Spans process init, weight load, pipeline compile, KV allocation, warmup,
  prefill and the first sample as one measured span, because that is what a user
  experiences.
- **`t_ready`** — `exec` until `GET /v1/models` answers 200. Splits `ttft_exec`
  into "getting ready" and "doing the work". Note this is an HTTP 200, not a TCP
  accept: a listener can bind before the model is resident.
  `--poll-interval-ms` (default 20) is the only quantization and is reported
  alongside.
- **`TTFT`** — first token measured from **request send** against a running
  server: TTFT in the usual sense, and the startup-independent half of
  `ttft_exec`.
- **`tpot`** — per-token decode interval over N−1 intervals, matching
  `crates/benches/src/serve.rs`. `decode tok/s` is `1000/tpot`.
- **`peak_rss`** / **`major_faults`** — the child's own `ru_maxrss` /
  `ru_majflt`, read via `wait4(2)` when it is reaped, same call for every
  backend. It has to be `wait4` rather than `getrusage(RUSAGE_CHILDREN)`: the
  latter's `ru_maxrss` is a high-water mark over *every* child the process has
  reaped, so per-repetition figures silently break after the first one — a Metal
  run reported a live WARM server at 1 MiB because an earlier COLD repetition had
  already pushed the mark to ~270 MiB.
- **`evicted`** — KiB that left the page cache when a FROZEN repetition's
  eviction ran, from `/proc/meminfo` `Cached` sampled either side of it.

### Proving the eviction happened

A FROZEN number is worthless unless the eviction demonstrably took effect, so the
run asserts it and exits non-zero on failure. **Where `evicted` is available it is
the evidence, and `major_faults` is not gated on.** That ordering is deliberate:

`ru_majflt` only counts faults on *memory-mapped* pages, and on Linux readahead
plus fault-around will satisfy a sequential scan of a fully-evicted mmap'd file
from within a single major fault. Measured on an H100 node: a `Cached` drop of
511,560 kB for a 512,000 kB file — a 99.9% eviction — produced `majflt=1`, against
`majflt=0` warm. Gating on `1 > 0` would be gating on noise, and tightening the
threshold instead would reject correctly-evicted Linux runs.

On macOS there is no `/proc/meminfo`, so the fault delta carries the argument —
and there it is strong, because `purge` drops everything and a real loader's
access pattern defeats readahead: 11,426 faults against 0 on a real checkpoint.
`read()`-based children never produce major faults at all, whatever the cache
state, so a FROZEN rung measured with `cat`/`dd`/`wc` is not measuring anything.

Cells are reported as `median (p10–p90) ×reps`, never a bare mean — a mean hid a
bimodal ITL distribution in this repo for a week
(`crates/cli/scr/src/commands/chat.rs`).

## 3. Fairness rules

1. **Unique prompt per request.** A shared prefix lets a prefix cache serve the
   repeat and TTFT collapses. Prompts are seeded: identical across backends,
   unique per rep. This is not hypothetical — a `bench serve` run against a
   shared-prefix dataset reported a 98% prefix-cache hit rate, which inflated
   throughput 1.84× and understated TTFT 11× before it was caught. Pass
   `--no-prefix-caching` to the server under test as well. `scr bench serve`
   sends its pre-flight and warmup requests throwaway prompts from a seed
   disjoint from `--seed`, for every dataset, so no measured prompt is cached
   before it is timed.

   **Exception: [multi-turn cells](#multi-turn-prefix-reuse-cells).** There
   the reuse *is* the measurement, so prefix caching stays **on** for every
   backend (`scratchy-nocache` is the deliberate baseline). Sharing is still
   confined: a request repeats only its own conversation's earlier turns, plus
   the one system prompt every conversation shares; pre-flight and warmups use
   disjoint throwaway conversations with a system prompt of their own. The
   backends do not reuse at the same granularity, and the published numbers
   must say so:

   - **scratchy** reuses whole 16-token KV blocks. On Gated-DeltaNet models
     (Qwen3.5/3.6) a hit can resume only from a recurrent-state snapshot,
     taken at `floor((N-8)/16)*16` of an earlier N-token prompt — up to ~23
     tokens short of the shared prefix — and only when it saves at least 256
     tokens.
   - **mlx-lm** reuses through its own prompt cache, on its own rules. Read
     the `cached_tokens` it reports rather than assuming it matches scratchy.
2. **Pin the sampling params.** Every request is greedy (`temperature 0`).
   Leaving temperature *unset* is the trap: the field is omitted, each **server**
   applies its own default, and the backends get timed on different sampling
   paths. Measured on scratchy:

   | | TTFT p50 | TPOT p50 |
   |---|---|---|
   | `temperature 0` (argmax) | 44.3 ms | 12.96 ms |
   | `temperature 1.0` | 47.0 ms | 15.81 ms |
   | unset (server default) | 71.2 ms | 15.81 ms |

   That spread is larger than most differences anyone would report, and it is
   pure measurement artifact. Greedy also matches what the parity gate verifies,
   so what is timed is what was checked.
3. **Parity gate, blocking** (`--parity-cmd`). A broken dequant path can be
   *fast*, so timing a wrong computation is worse than not timing at all. The
   gate enforces exact agreement on short high-confidence prompts and does not
   gate on drift deep inside open-ended generations: greedy argmax legitimately
   splits at near-ties between two kernel stacks. For real numerical fidelity
   work use the golden-reference tooling (`scripts/generate_mlx_goldens.py`,
   `tools/vision_parity/`) instead. The gate only means "same load path" if
   both sides render the *same* prompt: `scr chat` follows the checkpoint
   template's declared defaults, while `mlx_lm.generate` forces
   `enable_thinking` on for any vocab with think tokens — gemma-4's template
   defaults it off and prefills an empty thought channel — so the matrix gives
   the mlx-lm side a per-model `--chat-template-config`
   (`mlx_parity_config` in `scripts/bench_metal_matrix.sh`) wherever the two
   disagree.
4. **Same weights, matching preset.** scratchy must be built with the preset
   matching the checkpoint's `config.json`.
5. **One model resident at a time**, and `HF_HUB_OFFLINE=1` for both — left
   online, some frameworks make a hub round trip per launch and network jitter
   lands inside the measurement.
6. **A settle delay before WARM** (`--settle-s`), so lazy post-ready
   initialization does not land in the steady-state sample.

## 4. Known asymmetries — disclose, do not hide

These favour one side or the other and must travel with any published number.

- **scratchy compiles the model at build time.** `#[forward]` expands the whole
  forward pass during `cargo build`, so work other engines do at runtime is
  already in the binary. A real engineering advantage, and also why startup
  numbers flatter scratchy. The build cost is a one-off; record it as a footnote
  rather than folding it into a scenario.
- **Derived on-disk caches have no cross-framework equivalent.** scratchy can
  keep an aligned-weights sidecar; `--remove-path` puts it on the FROZEN rung so
  it is neither charged per launch nor hidden. Measured caveat: for
  `Llama-3.2-3B-Instruct-4bit` the sidecar is neither needed nor rebuilt —
  scratchy writes it only from the realign-*copy* path
  (`crates/targets/metal/src/metal_allocator.rs`) and that checkpoint loads
  648/648 tensors zero-copy from the HF mmap. For that model FROZEN→COLD is a
  page-cache delta only. It is *not* inert for checkpoints needing realignment
  (the code cites Qwen3.5-35B at 18.99 GiB). **Re-check per model.**
- **KV-cache dtype is not matched by default.** A default metal build stores
  KV as TurboQuant codes (3-bit for the Llama family, 4-bit otherwise; the
  `turboquant` feature) while mlx-lm uses an uncompressed cache — a *quality*
  difference as well as a perf one. It is fixed at build time, so match it with
  a build without the feature, not a runtime flag.
- **Process shape differs.** scratchy is one static binary; mlx-lm is a CPython
  interpreter plus imports. FROZEN evicts both, which is the real-world cost, but
  it is not a like-for-like measurement of model loading alone.
- **Weight download is out of scope.** The checkpoint is on disk in all three
  scenarios.
- **An agentic workload adds asymmetries of its own**, and resolves the
  quantization rungs to concrete per-engine artifacts: see
  [`CLAUDE_CODE_BENCH.md`](CLAUDE_CODE_BENCH.md), which carries the Claude Code
  comparison's pins, disclosures and blocked cells. Its method is promoted into
  this file as a §7 once that epic's phase 5 lands.

## 5. Running it

Any backend is named by `--child-cmd`, using the same shell-words convention as
`scr sweep --serve-cmd`, so nothing here is scratchy-specific.

```bash
# scratchy, cold + warm, 3 reps each
scr bench startup --exec -m "$MODEL" \
    --child-cmd "scr serve $MODEL --device cuda:0 --port 8731 --no-prefix-caching" \
    --scenarios cold,warm --reps 3 --output-json out/scratchy.json

# the same measurement against vLLM — no new code, just a different child
scr bench startup --exec -m "$MODEL" \
    --child-cmd "python -m vllm.entrypoints.openai.api_server --model $MODEL --port 8731 --no-enable-prefix-caching" \
    --scenarios cold,warm --reps 3 --output-json out/vllm.json

# FROZEN on Linux: evict the weights and the binary, and prove it happened
scr bench startup --exec -m "$MODEL" \
    --child-cmd "scr serve $MODEL --device cuda:0 --port 8731" \
    --scenarios frozen,cold --reps 3 \
    --evict fadvise \
    --evict-path "$HF_HOME/hub/models--org--name/snapshots" \
    --evict-path target/release/scr

# FROZEN on macOS: `sudo -v` first, or the up-front check refuses to start
sudo -v
scr bench startup --exec -m "$MODEL" \
    --child-cmd "target/release/scr serve $MODEL --device metal --port 8731 --no-prefix-caching" \
    --scenarios frozen,cold --reps 2 --evict purge

# CLI mode needs {prompt}; add a blocking parity gate against mlx-lm.
# Thinking models: pin mlx-lm to the checkpoint template's own default
# (scr chat renders it; mlx_lm.generate turns thinking on) — e.g. gemma-4
# needs --chat-template-config '{"enable_thinking":false}'.
sudo -v
scr bench startup --exec -m "$MODEL" --mode cli \
    --child-cmd "target/release/scr chat -m $MODEL --device metal -q {prompt} --max-tokens {output_len}" \
    --parity-cmd "python -m mlx_lm.generate --model $MODEL --prompt {prompt} --max-tokens {output_len}" \
    --scenarios frozen,cold --evict purge
```

A run whose validity checks fail prints them and exits non-zero. That is
intentional: a FROZEN rep that faulted no more than COLD did not measure a frozen
start, and reporting it would be worse than reporting nothing.

### Multi-turn prefix-reuse cells

`scr bench serve --dataset-name multi-turn` runs closed-loop conversations: each
request is a non-streaming POST to `/v1/chat/completions` (unless `--endpoint`
says otherwise) whose `messages` are the system prompt, user 1, assistant 1, …,
user k. The server renders its own chat template; the client never does.

| flag | default | meaning |
|---|---|---|
| `--num-prompts` | 1000 | conversations |
| `--multi-turn-system-len` | 2048 | tokens (approx.) in the one system prompt every conversation shares |
| `--multi-turn-turns` | 3 | requests per conversation; turn k+1 is sent once turn k has answered |
| `--multi-turn-user-len` | 128 | tokens (approx.) per user message |
| `--multi-turn-reply-len` | 256 | tokens (approx.) per assistant reply in the history |
| `--chat-template-kwargs` | not sent | JSON object sent as `chat_template_kwargs` |
| `--max-concurrency` | unbounded | conversations in flight |
| `--request-rate` | `inf` | paces conversation starts |
| `--num-warmups` | 0 | warmup requests, walking throwaway conversations turn by turn |

- Messages are whole words from a fixed list, cut to length with the model's
  tokenizer — never decoded token ids, which can spell out special tokens. The
  assistant replies in the history are seeded text, not what the model said,
  so every backend is sent byte-identical requests on every turn.
- Nothing streams, so the request latency is reported as the TTFT. It is one
  only at `--output-len 1`, which the compare script always uses.
- The results JSON adds, per turn k (1-based): `completed_turn{k}`,
  `median_ttft_ms_turn{k}`, `p{P}_ttft_ms_turn{k}` for each
  `--metric-percentiles` value, `mean_prompt_tokens_turn{k}`
  (`usage.prompt_tokens`) and `mean_cached_tokens_turn{k}`
  (`usage.prompt_tokens_details.cached_tokens`). The cached count is null when
  the server reported none in the whole run; scratchy leaves the field out on a
  full miss, so once a run has reported one, a missing count reads as 0.
- Turn 1 is not uniformly cold: the first `--max-concurrency` conversations
  prefill the system prompt, and later ones can reuse it.
  `mean_cached_tokens_turn1` shows how much they did.

```bash
# scratchy against itself without reuse, and against mlx-lm. Qwen3.5/3.6: pin
# thinking off so both servers render the same prompt (mlx_lm.server reads a
# request's chat_template_kwargs; scr serve merges them over its defaults).
scripts/bench_serve_compare.sh --model mlx-community/Qwen3.5-9B-4bit \
    --backends scratchy,scratchy-nocache,mlx-lm \
    --multi-turn-cells 2048x3x1,2048x3x4 \
    --chat-template-kwargs '{"enable_thinking":false}'
```

Each cell (`SYSxTURNSxCONC`) gets its own seed, shared across backends, and
each of the script's `--warmups` is one whole throwaway conversation. The
summary reports each turn's median TTFT and cached/prompt tokens, plus every
later turn's TTFT as a ratio of turn 1's.

## 6. Reproducing without a GPU

Everything above assumes real hardware, but the harness itself is process
management, not inference: it spawns a child, waits for a port or a line of
stdout, samples `getrusage`, and tears the process group down afterwards. None of
that needs a backend, and the crate is built so that it doesn't — `default = []`,
`cuda`/`metal` are pure pass-throughs to `scratchy-serving-api`, and there is no
backend `cfg` in the crate. So the full test suite and the `scr` binary carrying
`bench startup --exec` both build on any Linux or macOS box:

```bash
cargo test -p scratchy-bench --features datasets   # the whole suite; no GPU
cargo build -p scratchy-cli --no-default-features --features bench
./target/debug/scr bench startup --help   # renders the `--exec` section
```

`datasets` is the optional feature behind the parquet-backed
`hotpotqa`/`multihop`/`msmarco` subcommands, and it needs no GPU either.
Featureless (`cargo test -p scratchy-bench`) drops only the tests of their
parquet readers — nothing that touches the startup harness, whose process
handling these tests are the only executable coverage of. `linux-cuda` in
`.github/workflows/rust.yml` runs the same test command and lints the crate
both with and without `datasets`, so reviewing the harness on a laptop runs the
same gate CI does.
Don't add `--locked`; the featureless resolution differs from the committed
lockfile.

**What this does not give you is a measurement.** Reviewing the harness's
behaviour and publishing a number are different activities:

- **A fake child measures nothing.** Per [Proving the eviction
  happened](#proving-the-eviction-happened), a `read()`-based child never
  produces major faults whatever the cache state — so a FROZEN rung driven by
  `cat`/`dd`/`wc` is not measuring page-cache eviction, it is measuring `cat`.
  Standing up a stub OpenAI server would validate the plumbing and nothing past
  it.
- **`--evict fadvise` is Linux-only** (`crates/benches/src/startup_exec/mod.rs`).
  On macOS, FROZEN needs `--evict purge` and a live `sudo` ticket.
- **Which rungs a GPU-less box may publish: none.** COLD and WARM still want a
  real loader touching real weights, and FROZEN additionally wants the eviction
  to have demonstrably taken effect. The reference figures stay the runs recorded
  on real hardware — an H100 node for CUDA, an M5 Max for Metal.
