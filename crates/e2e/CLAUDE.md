# crates/e2e — `scratchy-e2e` end-to-end GPU tests

## OVERVIEW
In-process server + HTTP client harness and ~40 protected (T5) e2e test files run against real cached checkpoints on a GPU.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| In-process server | `src/server.rs` (`TestServer`, builder) |
| HTTP client helpers | `src/client.rs` |
| Golden/shape assertions | `src/assertions.rs` |
| Model tiers 1–4, GPU skip | `src/lib.rs` (`TestModels`, `skip_if_no_gpu!`, `gpu_backend_available`) |
| Tests | `tests/*.rs` |
| Fixtures/goldens | `testdata/` |

## CONVENTIONS
- Every test file starts `#![cfg(feature = "e2e")]`; every test is `#[ignore]` — opt in with `--ignored`.
- Use `skip_if_no_gpu!` at test start; pick models via `TestModels` tiers (small tier first).
- Names: `e<n>_*` = phase-numbered suites, `e_<topic>` = topic suites. Follow the neighbor you extend.
- Start the server through `TestServer` builder, never a spawned `scr` process.
- Default features: guided-decoding, metrics, multimodal, tls; opt-in `otel`, `rag`, `multiproc`, `nccl`. `metal` pulls scratchy-target-metal only to probe Metal 4 availability.
- Some doc comments still say `-p vllm-e2e`; the package is `scratchy-e2e`.

## ANTI-PATTERNS
- Protected tier (docs/TESTING.md): don't delete, relax, or loosen tolerances to go green.
- Don't run GPU-heavy files in parallel when they share a device: `--test-threads=1`.

## COMMANDS
```bash
cargo test -p scratchy-e2e --features e2e,metal --test <file> -- --ignored --test-threads=1
cargo test -p scratchy-e2e --features e2e,cuda  --test <file> -- --ignored
```
