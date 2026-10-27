# scratchy-serving-engine

## OVERVIEW
Target-neutral engine core: `EngineCore` step loop (scheduler + executor), `EngineCoreClient`, the `Executor`/`Worker`/`WorkerFactory` traits, `InputBatch`, GPU-budget defaults, grammar, spec decode. Score ~18 (16 files, ~14k LOC; traits implemented by worker + targets).

## WHERE TO LOOK
| Task | File |
|------|------|
| Inner loop, add/abort, `StepOutcome` | `src/engine_core.rs` (`EngineCore`, `EngineCoreConfig`, `HybridKvConfig`) |
| In-proc client used by api `AsyncEngine` | `src/core_client.rs` (`InprocClient`, `EmbedSender`) |
| Executor contract, deferred D2H token resolve | `src/executor.rs` (`Executor`, `ModelRunnerOutput`, `D2hResolve`, `NoopExecutor`) |
| Per-slot batch state (SoA, zero-alloc decode) | `src/input_batch.rs` (70 tests) |
| Default max_num_seqs / served context / KV bytes / prefill bucket | `src/gpu_budget.rs` |
| Worker contract, rank alignment | `src/worker.rs`; factory + model path: `src/worker_factory.rs` |
| Constrained decoding (llguidance/toktrie) | `src/grammar.rs` (`guided-decoding` only) |
| Drafting | `src/spec_decode/`: `ngram.rs` (CPU), `proposer.rs` (draft model), `mtp.rs` (MTP head), `verify.rs` (`greedy_rejection_sample`), `backend.rs` (`SpecDecodeBackend`) |

## CONVENTIONS
- Nothing here names a backend: execution goes through `Box<dyn Executor>` / `Box<dyn Worker>`. Backend code belongs in serving-worker or `targets/*`.
- `spec_decode` is always compiled; only `grammar` is gated (`guided-decoding`). `multiproc` only forwards to transport.
- Drafts flow worker -> `ModelRunnerOutput.draft_token_ids` -> engine -> scheduler. The engine never runs draft GPU work itself.
- `ProposerConfig::{Ngram, DraftModel, Mtp}` is the single selector; add proposers there.

## ANTI-PATTERNS
- `input_batch.rs` `per_slot_fields!`: the ONLY place a per-slot field may be introduced, since it generates push/swap/pop together. Never hand-write the three motions.
- `gpu_budget`: an explicit ask NEVER comes through here; the worker honours it verbatim. Only unset values resolve.
- `align_all_max_num_seqs`: min across ranks, one number per process.

## COMMANDS
```bash
cargo test -p scratchy-serving-engine --features guided-decoding   # CI covers grammar
```
Consumers' tests (`InprocClient` + `NoopExecutor`) live in api `engine.rs`/`llm.rs`/`server.rs` and worker `uniproc.rs`.
