# crates/cli — `scr` binary (scr/) and standalone TUI (tui/)

## OVERVIEW
`scr/` = crate `scratchy-cli`, bin `scr`, the workspace default member; `tui/` = `scratchy-tui`, its own workspace (excluded from root, own `Cargo.lock`).

## WHERE TO LOOK
| Task | Location |
|------|----------|
| All clap args | `scr/src/args.rs` |
| Entry, signal/exit handling | `scr/src/main.rs` |
| Subcommands | `scr/src/commands/{batch,bench,bundle,cache,chat,collect_env,completions,convert,launch,model,model_info,pull,serve}.rs` |
| Remote server client | `scr/src/http.rs` `RemoteClient` |
| Feature passthrough guard | `scr/tests/arch_feature_passthrough.rs` |
| TUI app / provider | `tui/src/lib.rs`, `tui/src/provider.rs` (`ScratchyProvider`), bin `scr-tui` |

## CONVENTIONS
- Dep aliases: `model` = scratchy-models, `quant` = scratchy-quantizations (build-dep), `spec` = scratchy-spec → `--features model/<stem>,quant/<preset>,spec/mtp`.
- Feature bundles: default = `chat` + `hf-completions`; `serve` adds axum + guided-decoding; `claude` = `serve`; `metal` = `metal-base` + `turboquant` + `mtp-heads`; `bench` pulls `scratchy-bench`.
- New backend/arch feature on scratchy-models → forward it here; `arch_feature_passthrough.rs` fails otherwise.
- `dotenvy` loads `.env` before clap parses.
- SIGINT exits 130.
- `launch claude` runs `max_num_seqs=1`; `--host/--port` are reserved by launch.
- `RemoteClient::with_timeout` for metadata GETs; generation streams have no short timeout.
- TUI: goose is a git dep pinned by rev; bump rev deliberately. `DownloadTracker` uses std `Mutex` with `try_lock`.

## ANTI-PATTERNS
- `spyre-hw` builds must exit via `libc::_exit` — senlib atexit handlers segfault on normal exit.
- Never hold the TUI std `Mutex` across `.await`.
- Don't add `crates/cli/tui` to the root workspace.
- Don't put engine logic in commands; they call serving/bench crates.

## COMMANDS
```bash
cargo test -p scratchy-cli --features metal,model/smollm2-135m
cargo build --release -p scratchy-cli --features metal,serve,model/smollm2-135m
cargo build --manifest-path crates/cli/tui/Cargo.toml
```
