# crates/serving — vLLM-port serving stack

## OVERVIEW
Rust port of vLLM's v1 serving path: HTTP/in-process API -> engine core loop -> scheduler -> worker -> target backend. Earned its file as the boundary of 5 crates (~75k LOC) sharing one layering contract.

## STRUCTURE
```
serving/
├── api/        # scratchy-serving-api: axum OpenAI/Anthropic server + in-process LLM, tokenize/detok, parsers
├── engine/     # scratchy-serving-engine: EngineCore loop, Executor/Worker traits, InputBatch, spec_decode
├── scheduler/  # scratchy-serving-scheduler: Scheduler + SimpleBlockTracker (KV/prefix cache)
├── worker/     # scratchy-serving-worker: Metal/CUDA/Spyre workers, executors, TP/multinode
└── transport/  # scratchy-serving-transport: msgpack codec, handshake types, TcpControlChannel, ZMQ (multiproc)
```
Dependency order (bottom-up): scheduler, transport -> engine -> worker -> api. scratchy-cli depends on api.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| New HTTP route / request field | `api/src/server.rs`, `api/src/protocol.rs` (Anthropic: `anthropic.rs`, Responses: `responses.rs`) |
| Startup wiring, KV/max-num-seqs budget | `api/src/init.rs` + `engine/src/gpu_budget.rs` |
| Step loop, request lifecycle | `engine/src/engine_core.rs` (inner), `api/src/engine.rs` (AsyncEngine, outer) |
| Admission, preemption, prefix cache, hybrid/SWA, GDN snapshots | `scheduler/src/scheduler/core.rs` |
| Device step / KV pools | `worker/src/gpu_worker.rs` (metal+cuda), `worker/src/spyre_*.rs` |
| Speculative decoding | `engine/src/spec_decode/` (ngram, draft-model, MTP) |
| Multi-process / multi-node wire | `transport/src/` (`messages.rs`, `tcp_store.rs`, `transport.rs`) |

## CONVENTIONS
- Every file: `SPDX-License-Identifier: Apache-2.0` + "Copyright contributors to the vLLM project"; module doc names the Python file ported (`vllm/v1/...`). Keep it when adding files; mirror the Python layout.
- Neutral traits (`Worker`, `WorkerFactory`, `Executor`, `InputBatch`, errors) live in **engine**, below compiler/targets, so `targets/*` implement them without cycles; worker re-exports them wholesale (`pub use scratchy_serving_engine::worker::*`).
- Workers register through `inventory::submit!` WorkerFactory. Registrations survive linker GC only via `extern crate ... as _` anchors (`api/src/init.rs` -> worker; `worker/src/gpu_worker.rs`/`spyre_worker.rs` -> scratchy_models). Never delete an anchor as "unused".
- Backend features (`cuda`, `metal`, `spyre`, `spyre-hw`, `spyre-emu`, `nccl`, `guided-decoding`, telemetry seams) are forwarding trees api -> worker -> engine/targets. Add a feature at every hop or it silently no-ops.
- Hard rules are written as `⛔` shouting-case doc comments; `⭐` marks key design. Read the nearby ⛔ before editing a function.
- Transport: `multiproc` is the only ZMQ gate; the default build is serde/msgpack/TCP only.

## ANTI-PATTERNS
- Explicit user caps (max-num-seqs, KV) are CAPPED, NEVER REPLACED; only unset values get resolved defaults (`gpu_budget.rs`, `init.rs`).
- max_num_seqs is ONE number per process: the MIN across ranks (`worker.rs` `align_all_max_num_seqs`).
- No model knowledge in serving: model facts arrive generated from `#[forward]`; never read checkpoints/configs to infer layout here.

## COMMANDS
```bash
cargo test -p scratchy-serving-scheduler --lib                      # CI
cargo test -p scratchy-serving-transport -p scratchy-serving-worker   # CI
cargo test -p scratchy-serving-engine --features scratchy-serving-engine/guided-decoding   # CI
cargo clippy -p scratchy-serving-api -p scratchy-serving-worker --all-targets \
  --features scratchy-serving-api/nccl,scratchy-models/smollm2-135m -- -D warnings   # CI nccl gate
```
