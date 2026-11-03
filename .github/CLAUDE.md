# .github/ — CI workflows, PR/issue templates, dependabot

## OVERVIEW
Defines every gate a PR must pass; earns a file because job scope and per-feature steps are non-obvious and easy to under-cover.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Main CI | `workflows/rust.yml` |
| Bot benchmark round → `site/data/metal` | `workflows/metal-round.yml` (`collect`, `commit`; driven by `scripts/metal_round.py`) |
| Site build/deploy | `workflows/pages.yml` (build on PR, deploy on main) |
| PR body contract | `PULL_REQUEST_TEMPLATE.md` |
| Version groups | `dependabot.yml` |

## rust.yml JOBS
| Job | Runner | Does |
|-----|--------|------|
| `linux-cuda` | ubuntu | workspace clippy + extra per-feature steps; repro `scripts/ci-linux-cuda-clippy.sh` |
| `macos-scope` | ubuntu | `scripts/ci-macos-scope.sh changed` → skip macOS jobs when nothing relevant changed |
| `macos-metal-clippy` / `macos-metal-tests` | macos-26 (Metal 4 required) | models clippy, kernel + crate tests; `ci-macos-scope.sh lock` |
| `linux-spyre-emu` / `linux-spyre-triton` | ubuntu | spyre emulator + triton paths |
| `fmt` | ubuntu | `cargo fmt --all -- --check` |
| `scripts` | ubuntu | `python3 -m unittest discover -s scripts/tests` |

## CONVENTIONS
- Workspace clippy covers lib+bins only, default features: `nccl`, `otel`, `datasets`, `guided-decoding`, `tool-spans` each have a dedicated clippy/test step. A new opt-in feature needs its own step.
- Every step carries a rationale comment; keep that when adding one.
- Also gated: `cargo deny check bans`, `cargo test -p scratchy-serving-scheduler --lib`.
- dependabot groups lock-step families: arrow/parquet, opentelemetry*, llguidance/toktrie*, minijinja*. Don't split them.
- PR template: input→output proof with the exact feature scope, no-regression proof, invariant checklist.

## ANTI-PATTERNS
- Don't add a dep edge from macOS-built crates into spyre/triton crates (`MACOS_UNBUILT`) — `lock` fails.
- Don't widen macOS jobs to cuda or linux jobs to metal crates (objc2).
- metal-round commits data files; never hand-edit them.

## COMMANDS
```bash
scripts/act-local.sh                 # run macOS jobs locally
scripts/ci-linux-cuda-clippy.sh
scripts/ci-macos-scope.sh changed origin/main HEAD
```
