# scratchy-serving-api

## OVERVIEW
Front door: axum OpenAI/Anthropic/Responses server (`serve`), in-process `LLM` for `scr chat`, stack init, tokenizer/detokenizer, chat templates, tool/reasoning parsers. Score ~25 (29 files, ~33k LOC, the highest centrality in serving).

## WHERE TO LOOK
| Task | File |
|------|------|
| Routes (/v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models, /health) | `server.rs` (`build_router`, `AppState`) |
| /v1/messages (Anthropic) | `anthropic.rs`; /v1/responses: `responses.rs` |
| Request lifecycle validate->tokenize->submit->poll->detokenize | `engine.rs` (`AsyncEngine`, 8.4k LOC) |
| Stack wiring worker->executor->engine, budget triage, KV dtype | `init.rs` (`initialize_stack`, `VllmConfig`) |
| Offline/in-process API (mirrors `vllm.LLM`) | `llm.rs` |
| Per-model tool-call stream parsing | `tool_parser.rs` (5.2k LOC, 116 tests) |
| `<think>` splitting | `reasoning_parser.rs` |
| HF Jinja compat on minijinja | `chat_template.rs` + `chat_template/` (`tojson.rs` = CPython json.encoder port) |
| RAG (`rag` feature) | `augment/` (leann-core HNSW, spnl-core Query IR) |
| Prometheus / ORCA headers | `metrics.rs`, `orca.rs` (`metrics`) |
| No-HTTP engine loop | `headless.rs` |

## CONVENTIONS
- `serve` is what pulls axum/tower-http/http. The engine path (`llm`/`init`/`engine`/`spans`) must compile without it, so `scr chat` never links axum. Put handler code in serve-gated modules (`server`, `query`, `anthropic`, `responses`); `spans.rs` stays HTTP-free.
- `tls` = the axum-server listener only; plain HTTP is tokio `TcpListener` + `axum::serve`.
- Dep hygiene is deliberate: serde_json WITHOUT `preserve_order` (client key order goes through `protocol::OrderedJson` / minijinja maps), `http` without default features, `ureq` rather than reqwest. A feature flipped on one side compiles shared crates twice.
- `tower` is a dev-dep: unit tests drive the router with `ServiceExt::oneshot`. Tests are inline (555 markers); run gated modules with features, e.g. `cargo test -p scratchy-serving-api --features serve,metrics`.
- Anthropic `cache_control` becomes `engine::CACHE_BREAKPOINT` ('\u{F8FF}') in rendered text, then is extracted to byte offsets in `EngineCoreRequest::cache_breakpoints` before tokenizing.
- Tool-call JSON is serialized `json.dumps`-style (`", "` / `": "`) to match the model's training format.
- spyre-api cfgs must key on `spyre` (enabled by both `spyre-hw` and `spyre-emu`), never on the card tier alone.

## ANTI-PATTERNS
- `chat_template/hf_renders.rs` is @generated: regenerate with `scripts/generate_chat_template_goldens.py` (transformers goldens) instead of editing it.
- `tokenizer.rs`: truncation STAYS DISABLED (tokenizer.json may carry a truncating post-processor).
- Over-context prompts are rejected in one place only (`engine.rs`); don't add a second check.
- Tokens that render to nothing are a mute failure: log the ids, never swallow them (`engine.rs`).
- `tool_choice` must name a declared tool; a backend refusal must name the backend property it lacks, not a generic error.
- `tool-spans` feature is KNOWN BROKEN (it bypasses the chat template). Don't build it or turn it into a runtime flag.
