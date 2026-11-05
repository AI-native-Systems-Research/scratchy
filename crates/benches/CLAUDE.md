# crates/benches — `scratchy-bench` (library behind `scr bench`)

## OVERVIEW
All benchmark subcommands; lib only, consumed by `crates/cli/scr` under its `bench` feature. Earns a file: measurement-validity rules are dense and easy to break.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Dispatch | `src/lib.rs` `run_bench(BenchCommand)` |
| CLI surface (all subcommands) | `src/args.rs` |
| Online serving bench, SSE stats, percentiles | `src/serve.rs` (`percentile` shared) |
| Workloads (sharegpt, random, multi-turn) | `src/datasets.rs` |
| HTTP client + concurrency bound | `src/http.rs` |
| In-process startup | `src/startup.rs`; `--exec` → `src/startup_exec/` |
| Cross-process startup harness | `src/startup_exec/mod.rs`, args `startup_exec/args.rs` (`Scenario` FROZEN/COLD/WARM, `Backend`, `Evict`) |
| Long-context / RAG benches | `niah.rs ruler.rs ragcsv.rs musique.rs longbench.rs spans.rs` |
| Parquet benches (`datasets` feature) | `hotpotqa.rs multihop.rs msmarco.rs` |

## CONVENTIONS
- One module per subcommand; only clap arg structs + `run_bench` are public.
- Blocking benches run in `tokio::task::spawn_blocking` — `LLM` builds its own runtime.
- HTTP: `ureq` (not reqwest), `http_status_as_error(false)` (4xx/5xx is data), 1h timeout for cold servers.
- `datasets` feature gates arrow/parquet; without it the three subcommands parse then bail with a rebuild hint — keep that behavior.
- `rand` 0.9 pairs with `rand_distr` 0.5; bump together.
- External backends (mlx-lm, vLLM) via free-form shell-words commands (`--child-cmd`, `--serve-cmd`) with `{prompt}`/`{output_len}`; backend identity is a closed `ValueEnum`.
- startup_exec: per-child `libc::wait4` (not RUSAGE_CHILDREN); child in own process group, SIGTERM→SIGKILL after `KILL_GRACE` 300s (lets metal sidecar flush caches); readiness = 200 on `/v1/models`.
- Stats: median with p10–p90.

## ANTI-PATTERNS
- Never a bare mean (hid a bimodal ITL once).
- Never self-reported timings: one stopwatch for every backend.
- Never `drop_caches` on Linux (evicts a shared node); use `posix_fadvise(DONTNEED)`; macOS `sudo -n purge`.
- A FROZEN rep must prove eviction (`/proc/meminfo` Cached drop; major faults only weak fallback) or it's void.
- Never reuse prompts across reps/cells (`throwaway_seed`) — prefix cache inflated throughput 1.84x once.
- Parity gate stays exact-match; on mismatch refuse to benchmark, never loosen.
- No samples → `null`, never 0 ms. Client never renders chat templates.
- Group-kill before reap; report measurement error before reap error.

## COMMANDS
```bash
cargo test -p scratchy-bench
cargo build -p scratchy-bench -F metal,datasets
scr bench startup --exec ...   # cross-process; see args.rs
```
