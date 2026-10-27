# crates/core — backend-neutral foundation crates

## OVERVIEW
Four crates every other layer depends on: shared types (`common`), configs (`config`), Hub downloader (`hf-hub-downloader`), checkpoint/tokenizer loading (`model`).

## WHERE TO LOOK
| Crate (dir) | Role | Key files |
|-------------|------|-----------|
| `scratchy-core-common` (common/) | flat re-exports: engine IO, KV, request, sampling, telemetry | `src/kv.rs`, `src/engine_io.rs`, `src/request.rs`, `src/sampling.rs`, `src/telemetry*` |
| `scratchy-core-config` (config/) | `CacheConfig`, `KVCacheConfig`, `compute_hybrid_kv_layout`, `SchedulerConfig`, `CudaGraphConfig`, `SpansConfig` | `src/lib.rs` |
| `hf-hub-downloader` (hf-hub-downloader/) | replaces the `hf-hub` crate; HF cache layout | `src/`, `tests/live_hub.rs` |
| `scratchy-core-model` (model/) | `SafeTensorsIndex`, `HfModelConfig`, baked tokenizer, hub resolve, embedding, LoRA | `src/weight.rs`, `src/baked_tokenizer.rs`, `src/hub.rs`, `src/lora.rs` |

- Quant-config parsing is NOT here: `scratchy-forward-compiler-macro`. GGUF: `scratchy-quantizations::gguf`.

## CONVENTIONS
- common features: `forward-telemetry`, `sampler-telemetry`, `otel`; CI clippies `otel` separately.
- `kv.rs` Kani proofs under `cfg(kani)` (declared via check-cfg). Run: `KANI_PKG=scratchy-core-common ~/.claude/bin/kani-safe.sh <harness>`; check harness and "undetermined" counts, not just "SUCCESS".
- Downloader: shipped HTTP is `ureq`; no dev-deps by design. Probe agent `max_redirects(0)`, download agent follows 8. Content lands via temp + verify + atomic rename.
- `hub.rs` also renames-in atomically; `baked_tokenizer.rs` unsafe casts carry alignment SAFETY notes — keep them true.

## ANTI-PATTERNS
- `kv.rs`: `KvBlockTokens` and `CacheableTokens` travel together; `KvAddressing` is declared, never inferred; `KvExtent` span is a required arg, never defaulted.
- Never leave unverified content in the HF cache; the retry budget is not refilled between attempts.
- Don't add `hf-hub` or reqwest as a shipped dep.
- No per-model tokenizer special cases in `baked_tokenizer` — extend the general bake (see `crates/models/arch/dsl/CLAUDE.md`).

## COMMANDS
```bash
cargo test -p scratchy-core-common -p scratchy-core-config -p scratchy-core-model
cargo test -p hf-hub-downloader                      # unit
cargo test -p hf-hub-downloader --test live_hub      # live network
cargo test --release -p scratchy-core-model --lib bakes_every_cached_tokenizer -- --ignored --nocapture
```
