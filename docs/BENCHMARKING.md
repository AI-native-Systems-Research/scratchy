# Benchmarking scratchy against other frameworks

Methodology for the head-to-head comparisons, and the metric definitions they
report.

| harness | question | scenarios |
|---|---|---|
| [`scripts/bench_serve_compare.sh`](../scripts/bench_serve_compare.sh) | steady-state serving throughput and latency under load | input × output × concurrency sweep |
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

### Eviction is platform-specific, and one obvious choice is wrong

`--evict purge` (macOS) drops the whole unified buffer cache. It is symmetric: it
evicts CPython and a framework's dylibs exactly as it evicts the scratchy binary,
which is what makes a cross-framework FROZEN fair.

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
- **`peak_rss`** / **`major_faults`** — child `ru_maxrss` / `ru_majflt`, same
  call for every backend. `major_faults` is *evidence the eviction worked*: if
  FROZEN does not fault far more than COLD, the cache control failed and the run
  is void. The run asserts this and exits non-zero.

Cells are reported as `median (p10–p90) ×reps`, never a bare mean — a mean hid a
bimodal ITL distribution in this repo for a week
(`crates/cli/scr/src/commands/chat.rs`).

## 3. Fairness rules

1. **Unique prompt per request.** A shared prefix lets a prefix cache serve the
   repeat and TTFT collapses. Prompts are seeded: identical across backends,
   unique per rep. This is not hypothetical — a `bench serve` run against a
   shared-prefix dataset reported a 98% prefix-cache hit rate, which inflated
   throughput 1.84× and understated TTFT 11× before it was caught. Pass
   `--no-prefix-caching` to the server under test as well.
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
   `tools/vision_parity/`) instead.
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
- **KV-cache dtype is not matched by default.** On metal scratchy enables
  TurboQuant 3-bit KV compression while mlx-lm uses an uncompressed cache — a
  *quality* difference as well as a perf one. Match it explicitly.
- **A background integrity hash can overlap early decode.** On an aligned-cache
  *hit* scratchy content-hashes the cached blob on a background thread
  (0.39–2.44 s observed for a ~1.7 GiB sidecar), competing with the first
  requests. It does not happen when no sidecar exists, so it is present in some
  runs and absent from others — hence `--settle-s`, and a reason to read
  `t_ready` and the server log together rather than trusting one rep.
- **Process shape differs.** scratchy is one static binary; mlx-lm is a CPython
  interpreter plus imports. FROZEN evicts both, which is the real-world cost, but
  it is not a like-for-like measurement of model loading alone.
- **Weight download is out of scope.** The checkpoint is on disk in all three
  scenarios.

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

# FROZEN on macOS, plus a blocking parity gate against mlx-lm
sudo -v && scr bench startup --exec -m "$MODEL" --mode cli \
    --child-cmd "target/release/scr chat -m $MODEL --device metal" \
    --parity-cmd "python -m mlx_lm.generate --model $MODEL" \
    --scenarios frozen,cold --evict purge
```

A run whose validity checks fail prints them and exits non-zero. That is
intentional: a FROZEN rep that faulted no more than COLD did not measure a frozen
start, and reporting it would be worse than reporting nothing.
