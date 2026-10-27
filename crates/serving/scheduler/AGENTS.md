# scratchy-serving-scheduler

## OVERVIEW
Port of `vllm/v1/core/sched/*` + `block_pool.py`: `Scheduler` state machine and `SimpleBlockTracker` KV manager (uniform, prefix-cache, hybrid/SWA, recurrent-state modes). Score ~16 (distinct domain; `core.rs` is 9.2k LOC and the file to understand).

## WHERE TO LOOK
| Task | File |
|------|------|
| Scheduling, prefix cache, span hashing, seal, snapshot planning | `src/scheduler/core.rs` (`Scheduler`, `SimpleBlockTracker`, `KVCacheManagerOps`, `RecurrentPrefixPlan`) |
| Step output wire types | `src/scheduler/output.rs` (`SchedulerOutput`, `StepId`, `SpecPipeline`, KV/snapshot ops) |
| Trait / pause | `src/scheduler/interface.rs` (`SchedulerInterface`, `PauseState`) |
| FCFS / priority queues | `src/scheduler/request_queue.rs` |
| GDN/Mamba snapshot index, LRU pinning | `src/scheduler/recurrent_state.rs` (private; only `SnapshotBudget` exported) |
| KV vs snapshot shared budget | `src/scheduler/shared_memory.rs` (private) |
| Batched-admission starvation regression (#694) | `src/scheduler/batched_admission_tests.rs` (in lib so `--lib` runs it) |
| Worker<->allocator KV-extent seam | `tests/zz_kv_extent_spans_the_worker_and_the_allocator.rs` |

## CONVENTIONS
- No features. Deps: core-common, core-config, serde, tracing only.
- Backend facts are DECLARED through `SimpleBlockTracker` builders (`enable_hybrid`, `sharing_one_write_slot`, `with_kv_backing`, `with_state_snapshots`), never inferred from whether a report arrived.
- Wire types use serde; add new fields with `#[serde(default)]` + `skip_serializing_if`.
- Python-parity details are intentional (ref counting, lazy free-queue eviction, null block, reversed free order, `num_tokens - 1` reuse cap). Don't "fix" them.
- Bugs that live on a seam get a cross-crate test in `tests/`, because each side alone looks fine and the pair is a wrong-output bug.

## ANTI-PATTERNS
- A FAILED STEP'S BLOCKS ARE NO PREFIX-CACHE HITS (`forget_unwritten`); its KV may not exist.
- `step_reach` is SET ONCE PER `schedule()`, NOT DERIVED PER REQUEST.
- A snapshot is ready only once its step's output has come back; a reservation is not a hit. Recency alone does not pin.
- Hybrid prefix-cache keys are namespaced by group id; full and sliding hashes MUST NOT collide. Prefix caching stays ON on the hybrid path.
- Span hashing: only the span's first block resets the parent chain (the old per-block reset produced false KV hits and garbage output).
- A request resumes at `plan.tokens`, never at a KV hit past it.

## COMMANDS
```bash
cargo test -p scratchy-serving-scheduler --lib     # CI invocation
cargo test -p scratchy-serving-scheduler --test zz_kv_extent_spans_the_worker_and_the_allocator
```
