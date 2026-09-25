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

# FROZEN on macOS: `sudo -v` first, or the up-front check refuses to start
sudo -v
scr bench startup --exec -m "$MODEL" \
    --child-cmd "target/release/scr serve $MODEL --device metal --port 8731 --no-prefix-caching" \
    --scenarios frozen,cold --reps 2 --evict purge

# CLI mode needs {prompt}; add a blocking parity gate against mlx-lm
sudo -v
scr bench startup --exec -m "$MODEL" --mode cli \
    --child-cmd "target/release/scr chat -m $MODEL --device metal -q {prompt} --max-tokens {output_len}" \
    --parity-cmd "python -m mlx_lm.generate --model $MODEL --prompt {prompt} --max-tokens {output_len}" \
    --scenarios frozen,cold --evict purge
```

A run whose validity checks fail prints them and exits non-zero. That is
intentional: a FROZEN rep that faulted no more than COLD did not measure a frozen
start, and reporting it would be worse than reporting nothing.

## 6. Reproducing without a GPU

Everything above assumes real hardware, but the harness itself is process
management, not inference: it spawns a child, waits for a port or a line of
stdout, samples `getrusage`, and tears the process group down afterwards. None of
that needs a backend, and the crate is built so that it doesn't — `default = []`,
`cuda`/`metal` are pure pass-throughs to `scratchy-serving-api`, and there is no
backend `cfg` in the crate. So the full test suite and the `scr` binary carrying
`bench startup --exec` both build on any Linux or macOS box:

```bash
cargo test -p scratchy-bench          # the whole suite; no features, no GPU
cargo build -p scratchy-cli --no-default-features --features bench
./target/debug/scr bench startup --help   # renders the `--exec` section
```

Featureless drops only the tests behind the optional `datasets` feature (the
parquet-backed `hotpotqa`/`multihop`/`msmarco` subcommands) — nothing that
touches the startup harness, whose process handling those tests are the only
executable coverage of. `linux-cuda` in `.github/workflows/rust.yml` runs these
same commands, so reviewing the harness on a laptop runs the same gate CI does.
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
