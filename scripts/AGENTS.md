# scripts/ — dev, CI and benchmark tooling (not a crate)

## OVERVIEW
~28 bash/Python tools that CI, benchmarks and PR proofs run; earns a file because each script encodes a measurement contract you must not bypass.

## WHERE TO LOOK
| Task | File |
|------|------|
| Metal bench matrix vs mlx-lm/ollama | `bench_metal_matrix.sh` (runner, not measurer; see `.claude/skills/metal-bench`) |
| Claude Code trace record/replay | `cc_trace.py record\|replay` (corpus in gitignored `bench_results/`) |
| Bot benchmark round (metal-round.yml) | `metal_round.py check\|collect\|commit-input` |
| Spyre timing / perf PR gate | `spyre_card_timing.sh`, `spyre_pr_cells.sh` |
| Expand one arch's `#[forward]` without cargo build | `emit-arch.sh <arch> <stem> [--rerooted]` |
| macOS CI locally | `act-local.sh` |
| macOS job skip / lock gate | `ci-macos-scope.sh changed <base> <head>` / `lock` |
| linux-cuda clippy gate | `ci-linux-cuda-clippy.sh` |
| Batched-decode correctness | `metal-batch-gate.sh` (granite-3.3-8b) |
| Batched prefill is real? | `verify_prefill_real.sh` |
| Goldens | `generate_golden_refs.py`, `generate_mlx_goldens.py`, `generate_chat_template_goldens.py` |
| Tiny synthetic checkpoints | `make_tiny_deepseek_v3.py`, `make_tiny_kimi_k2.py` |
| Near-tie parity scoring | `parity_ties.py` |
| Script unit tests | `tests/test_*.py` (stdlib-only) |

## CONVENTIONS
- Every script opens with a "WHY THIS EXISTS" header naming the false conclusion it prevents. Keep it current when changing behavior.
- Bash: `set -euo pipefail`; exceptions `metal-batch-gate.sh`, `verify_prefill_real.sh` use `set -uo pipefail` and fail closed by hand.
- Python: SPDX Apache-2.0 + vLLM-contributors header; stdlib-only unless the script is a torch/mlx golden generator.
- Bench scripts double as PR repro records (e.g. `bench_prefill_ab.sh` = PR #597, `bench_glm45_air_m2_fold.sh` = PR #295) — don't repurpose; add a new one.
- `emit-arch.sh` mirrors Cargo's feature→`CARGO_FEATURE_*` transform exactly (exported via `env`, dotted names); default backend `spyre` (no nvcc/macOS needed); `SCRATCHY_QUANTS` empty = plain `-Fmodel/<stem>`; extra features via `SCRATCHY_EMIT_FEATURES`.
- `act-local.sh` sandboxes `RUSTUP_HOME` under `$XDG_CACHE_HOME/scratchy/act-rustup`; macOS jobs native, linux emulated amd64.
- `ci-macos-scope.sh` hardcodes `MACOS_UNBUILT` (spyre target + triton crates); `lock` fails if a macOS job compiled one — a new dep edge into them turns macOS CI red.

## ANTI-PATTERNS
- Never accept a pasted result of `verify_prefill_real.sh`; run it yourself.
- `metal-batch-gate.sh`: a substring oracle is not an oracle; prefix caching and one-request-per-step can hide faults; if the `--max-num-seqs 1` control isn't 12/12, ignore the batched number.
- `bench_metal_matrix.sh` frozen scenario needs `sudo purge`; without sudo pass `--scenarios cold,warm`. `--offline` for no network.
- `cc_trace.py` relay rewrites nothing — don't add request mutation.
- Don't add third-party deps to `scripts/tests`.

## COMMANDS
```bash
python3 -m unittest discover -s scripts/tests   # CI "scripts" job
scripts/emit-arch.sh llama llama-3.2-1b
scripts/act-local.sh
```
