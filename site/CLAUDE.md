# site/ — benchmark site (crate `scratchy-site`)

## OVERVIEW
Static site generator for the Metal/Spyre benchmark pages and docs; OUTSIDE the workspace (own `Cargo.lock`), dioxus 0.7 SSR, no wasm, per-page JS sidecars.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Site entry, asset hashing | `src/main.rs` (`Site`, `hashed_name` FNV-1a → `<stem>.<16hex>.<ext>`) |
| Chart components | `src/charts.rs` |
| Carbon design components | `src/carbon.rs` |
| Metal page | `src/metal/mod.rs`; schema `src/metal/data.rs` (schema 2); `src/metal/numbers.rs` (`Engine`, `Metric`) |
| Spyre page | `src/spyre/mod.rs`; schema `src/spyre/data.rs` (schema 1) |
| Metal data | `data/metal/<machine>-<YYYYMMDDTHHMMSSZ>-<sha8>.json` |
| Spyre data | `data/spyre/<date>-<sha8>.json`, `data/spyre/progress/<pr>.json` |
| Docs theme | `theme/` |
| Test fixtures | `testdata/metal/` (cfg(test) only) |
| CI | `.github/workflows/pages.yml` (build on PR, deploy on main) |

## CONVENTIONS
- Metal JSON comes from `scripts/bench_metal_matrix.sh` / metal-round bot; copy files in unedited (see `.claude/skills/metal-bench`).
- Spyre: one new sweep file per sweep (`.claude/skills/spyre-bench`); `progress/` may be absent.
- All spyre serde types `#[serde(deny_unknown_fields)]` — misspelt key fails the build.
- Loaders collect ALL bad files and report them together, not fail-fast.
- A new model on the metal page needs its weight size in `src/metal/data.rs` (`WEIGHTS`).
- Spyre `BATCHES = [1, 4, 8]`: batch >=16 corrupts on that stack.

## ANTI-PATTERNS
- Never hand-edit or "fix" committed data JSON; re-run the bench.
- torch-spyre is never re-measured for a PR — only triton-scratchy values change.
- Charts never get a scroll box; fit the layout.
- Don't add it to the root workspace or pull wasm/hydration in.

## COMMANDS
```bash
cargo run --release --manifest-path site/Cargo.toml                       # build
cargo run --release --manifest-path site/Cargo.toml -- serve --port 8080 --no-open
cargo test --manifest-path site/Cargo.toml
site/open.sh
```
