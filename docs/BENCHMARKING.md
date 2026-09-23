# Benchmarking scratchy against other frameworks

Methodology for the head-to-head comparisons, and the metric definitions they
report. Until now this knowledge lived only in shell-script header comments;
this file is the reference.

The comparisons are split by what they measure, because they are different
questions with different confounds:

| harness | question | scenarios |
|---|---|---|
| [`scripts/bench_serve_compare.sh`](../scripts/bench_serve_compare.sh) | steady-state serving throughput and latency under load | input × output × concurrency sweep |
| [`scripts/bench_startup_compare.sh`](../scripts/bench_startup_compare.sh) | how long from `exec` until the user sees a word | frozen / cold / warm cache ladder |

Both currently target **mlx-lm** on Apple Silicon.

---

## 1. Startup: the cache ladder

"Cold start" is not one thing — it is a stack of caches, each of which can be
independently warm. Naming a single "cold" number without saying which of them
were populated is how startup benchmarks become unfalsifiable. So the ladder is
explicit:

| surface | FROZEN | COLD | WARM |
|---|---|---|---|
| OS page cache (weights, binary, dylibs) | purged | warm | warm |
| `~/.cache/scratchy/metal-aligned-weights` (if present; see caveat) | **removed** | present | present |
| mlx-lm `__pycache__` | **removed** | present | present |
| HF snapshot on disk | present | present | present |
| process | fresh `exec` | fresh `exec` | resident, ≥1 request served |
| Metal pipelines / KV pool / warmup | rebuilt | rebuilt | done |

- **FROZEN** — first run ever on this machine, short of downloading weights.
  Every page comes off SSD, and scratchy's aligned-weights sidecar is removed.
- **COLD** — the honest everyday case: you ran it before, the machine has been
  doing other things but not enough to evict 2 GiB, and you launch it again.
- **WARM** — a server that is already up and has served traffic. Isolates
  request latency from all startup cost.

**Both FROZEN and COLD are always reported.** scratchy can keep a derived
on-disk sidecar that MLX has no equivalent of; publishing only FROZEN would
charge it a one-time cost on every launch, and publishing only COLD would hide
that cost entirely.

> **Measured caveat, and it contradicts the obvious assumption.** For
> `Llama-3.2-3B-Instruct-4bit` the sidecar is neither needed nor rebuilt.
> scratchy writes it only from the realign-*copy* path
> (`crates/targets/metal/src/metal_allocator.rs:1372`), and this checkpoint's
> tensors already satisfy the bind alignment, so it loads 648/648 tensors
> zero-copy directly from the HF mmap — with or without the sidecar. For this
> model FROZEN→COLD is therefore a page-cache delta for *both* backends, and
> the sidecar rung is inert (~0.17 s, inside the run-to-run spread).
>
> The rung stays because it is *not* inert for checkpoints that do need
> realignment: the code cites Qwen3.5-35B (18.99 GiB), where a warm relaunch
> pays ~3.5 s of residency wiring instead of an ~8 s copy. **Re-check this per
> model rather than assuming either way** — and note that once removed, the
> sidecar will not come back for a zero-copy checkpoint, so COLD reps before
> and after a FROZEN rep can take different load paths.

`sudo purge` is the only faithful page-cache drop on macOS; it is symmetric, in
that it evicts CPython and the MLX dylibs exactly as it evicts the scratchy
binary. Run with `--no-purge` and FROZEN collapses toward COLD.

## 2. Metrics

### Primary — one external stopwatch

Every headline number is taken by [`scripts/startup_probe.py`](../scripts/startup_probe.py),
which holds the clock itself and runs **the same code for both backends**. That
single shared implementation is the core fairness guarantee: nothing is
self-reported.

- **`ttft_exec`** — seconds from `exec` to the first token byte. The headline.
  Spans process init, weight load, pipeline compile, KV allocation, warmup,
  prefill and the first sample as one measured span, because that is what a
  user experiences.
- **`t_ready`** — `exec` until the server answers `GET /v1/models`. Splits
  `ttft_exec` into "getting ready" and "doing the work". Poll granularity is
  20 ms and is reported next to the number.
- **`TTFT`** — time to first token measured from **request send** against an
  already-running server. This is TTFT in the usual sense, and it is the
  startup-independent half of `ttft_exec`: reported for every scenario, so
  FROZEN, COLD and WARM are directly comparable on the same axis. In
  frozen/cold it is the first request a fresh process ever serves, so it still
  carries lazy pipeline compilation; in WARM it is steady state. p50 and p99
  both reported.
- **`tpot`** — per-token decode interval, over N−1 intervals, matching
  `crates/benches/src/serve.rs`. `decode tok/s` is `1000/tpot`.
- **`E2E*`** = `ttft_exec + (out_len − 1) · tpot` — the work-normalized total.
  Required because mlx-lm ignores `ignore_eos` and stops early; a raw
  wall-clock total would reward it for generating less.

### Secondary — diagnostics

- **`peak_rss`** — child `ru_maxrss`, same call both sides. On Apple Silicon
  unified memory this includes GPU buffers.
- **`major_faults`** — child `ru_majflt`. This is *evidence the purge worked*:
  if FROZEN does not fault far more than COLD, the cache control failed and
  the run is void. The summary asserts this and prints PASS/FAIL.

### Why not the built-in numbers

Neither framework's self-reported timings compose into `ttft_exec`:

- `scr bench startup` rebuilds an `LLM` **in-process** and calls it "cold";
  its own comment (`crates/benches/src/startup.rs:80`) notes the HF cache is
  never wiped. It cannot observe `exec`, dyld, or per-process Metal pipeline
  compilation. **Do not compare its output with `ttft_exec`.**
- `scr chat --bench` reports `startup` and then a TTFT measured *from after
  startup*, so the two never add up to user-perceived latency — and real work
  lands after "startup" is declared done (a warmup generation, a background
  integrity hash, lazy TurboQuant codebook selection).
- `mlx_lm.generate --verbose` reports prompt/generation tok/s but never import
  or load time.

## 3. Fairness rules

Rules 1–3 were learned the hard way by `bench_serve_compare.sh`; 4–8 are
specific to startup.

1. **Unique prompt per request.** A shared prompt lets scratchy's prefix cache
   (or mlx-lm's prompt cache) serve the repeat, and TTFT collapses to ~0.
   Prompts are seeded: identical across backends, unique per rep.
2. **mlx-lm ignores `ignore_eos`.** Compare TTFT and TPOT directly; use `E2E*`
   for totals. Generated tokens per request are always printed so
   under-generation stays visible.
3. **One model resident at a time.** Backends run serially; the probe refuses
   to start against an already-serving port.
4. **Same weights.** Both read the same `mlx-community` 4-bit checkpoint.
   scratchy must be built with the preset matching the checkpoint's
   `config.json` — for `Llama-3.2-3B-Instruct-4bit`
   (`{"group_size": 64, "bits": 4}`) that is `quant/mlx-affine-b4-g64`.
5. **Parity gate, blocking** ([`scripts/startup_parity.py`](../scripts/startup_parity.py)).
   A broken dequant path can be *fast*, so timing a wrong computation is worse
   than not timing at all. The gate enforces exact agreement on short
   high-confidence prompts, and merely records drift on open-ended ones —
   greedy argmax legitimately splits at near-ties between two different int4
   kernel stacks, so gating on that would block on floating-point noise.
   For real numerical fidelity work use the golden-reference tooling
   (`scripts/generate_mlx_goldens.py`, `tools/vision_parity/`) instead.
6. **ABBA interleaving.** Backend order reverses on odd reps so thermal drift
   does not accrue to whichever backend always runs second. Thermal state and
   power source are recorded, and throttling warns.
7. **`HF_HUB_OFFLINE=1` for both.** Left online, mlx-lm makes a hub round trip
   on every launch (`Fetching 6 files`) and network jitter lands inside the
   measurement.
8. **Median + p10/p90, never a bare mean**, with the rep count in every cell.
   `crates/cli/scr/src/commands/chat.rs:90` records what a mean costs: it hid
   a bimodal ITL distribution for a week.
9. **Pin the sampling params explicitly.** Every request is greedy
   (`temperature 0`) on both backends and from both clients. Leaving
   temperature *unset* is the trap: `scr bench serve` then omits the field
   entirely and each **server** applies its own default, so the two backends
   get timed on different sampling paths. Measured on scratchy, which has
   separate argmax and sampler kernels:

   | | TTFT p50 | TPOT p50 |
   |---|---|---|
   | `--temperature 0` (argmax) | 44.3 ms | 12.96 ms |
   | `--temperature 1.0` | 47.0 ms | 15.81 ms |
   | unset (server default) | 71.2 ms | 15.81 ms |

   That spread is larger than most differences anyone would report, and it is
   pure measurement artifact. Greedy also matches the parity gate, so what is
   timed is what was verified.

## 4. Known asymmetries — disclose, do not hide

These favour one side or the other and must travel with any published number.

- **scratchy compiles the model at build time.** `#[forward]` expands the whole
  forward pass during `cargo build`, so work MLX does at runtime is already in
  the binary. A real engineering advantage, and also why startup numbers
  flatter scratchy versus a from-source comparison. The build cost is a
  one-off; record it as a footnote rather than folding it into a scenario.
- **scratchy's aligned-weights sidecar** has no MLX equivalent — hence the
  FROZEN rung — but see the measured caveat above: it is inert for this
  checkpoint, which loads zero-copy without it.
- **KV-cache dtype is not matched by default.** On metal scratchy enables
  TurboQuant 3-bit KV compression while mlx-lm uses an uncompressed cache.
  That is a *quality* difference as well as a perf one. Use
  `--kv-cache-dtype fp16` for a like-for-like cache.
- **A background integrity hash overlaps early decode.** On an aligned-cache
  *hit* scratchy content-hashes the cached blob on a background thread
  (`crates/targets/metal/src/metal_allocator.rs:1002`) — observed between
  0.39 s and 2.44 s for this model's ~1.7 GiB sidecar, competing with the
  first requests. It does not occur when there is no sidecar, so it is present
  in some COLD/WARM runs and absent in others: another reason WARM samples
  begin only after a settle delay, and a reason to read `t_ready` and the
  server log together rather than trusting a single rep.
- **Process shape differs.** scratchy is one static binary; mlx-lm is a CPython
  interpreter plus imports. Under FROZEN both are evicted, which is the
  real-world cost, but it is not a like-for-like measurement of model loading.

## 5. Two clients, one server

WARM steady state is measured by **`scr bench serve`** — the repo's own load
generator. It is base-URL driven and backend-agnostic, so the same binary
drives scratchy and mlx-lm alike (this is exactly how `bench_serve_compare.sh`
compares them), and the warm numbers come from the repo's existing TTFT/ITL
logic and numpy-linear percentiles rather than a second implementation of the
same statistics.

It cannot serve FROZEN or COLD: those need the clock to start *before* `exec`,
and `bench serve` can only measure from request send against a server that is
already up. That is why `startup_probe.py` exists, and why it also issues its
own warm requests — two independent clients against one server should agree,
and the summary prints both. When they disagreed (scratchy 15.8 vs 12.9 ms
TPOT) the cause was rule 9, not noise.

## 6. Running it

```bash
# Build with serve + bench + the preset matching the checkpoint.
cargo build --release -p scratchy-cli \
  --features metal,serve,bench,model/llama-3.2-3b,quant/mlx-affine-b4-g64

# mlx-lm in its own pinned venv; the version is recorded in the summary.
uv venv /tmp/mlxbench --python 3.12
uv pip install --python /tmp/mlxbench/bin/python mlx-lm

# Smoke first (no sudo, ~1 min), then the full ladder.
scripts/bench_startup_compare.sh --mlx-python /tmp/mlxbench/bin/python \
    --scenarios cold --modes server --reps 1 --no-long --no-purge
scripts/bench_startup_compare.sh --mlx-python /tmp/mlxbench/bin/python
```

FROZEN needs `sudo purge`; the script authorizes once up front and fails early
if sudo is unavailable, rather than silently producing a fake frozen number.

Results land in `bench_results/startup_compare/<timestamp>_<label>/`:
per-rep JSON, backend logs, `run_meta.txt` (host, OS, git SHA, versions,
thermal state), `parity/`, and `summary.md`. Re-summarize a finished directory
without re-running it:

```bash
python3 scripts/startup_summary.py bench_results/startup_compare/<dir>
```
