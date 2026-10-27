# docs/ — human docs and methodology contracts

## OVERVIEW
Reference docs agents must read before touching builds, models, tests or benchmarks; earns a file as the index of which doc is authoritative for what.

## WHERE TO LOOK
| Question | Doc |
|----------|-----|
| Feature scoping, cuda two-step build, air-gap, metal-base | `BUILD.md` |
| `#[forward]` pipeline, passes, tapes | `COMPILER.md` |
| New arch layout contract / recipe | `MODELS.md` (then `crates/models/arch/dsl/CLAUDE.md` checklist) |
| Test tiers T1–T5, protected tests, what "weakening" means | `TESTING.md` |
| Bench rungs, fairness rules | `BENCHMARKING.md` |
| Claude Code workload bench | `CLAUDE_CODE_BENCH.md` |
| Spyre ops issues, k8s | `spyre/KNOWN-ISSUES.md`, `spyre/KUBERNETES.md` |
| In-flight plans | `nemotron-super-streaming-plan.md`, `spyre-paged-kv-elimination-todo.md` |
| Specs/plans per CONTRIBUTING (dirs created on first use) | `superpowers/specs/YYYY-MM-DD-<slug>-design.md`, `superpowers/plans/YYYY-MM-DD-<slug>.md` |
| Essays | `blogs/` |

## FACTS NOT IN ROOT CLAUDE.md
- cuda = two cargo invocations: build `scratchy-builder-cuda` kernels first (`SCRATCHY_GPU`, `CUDA_COMPUTE_CAP`), then the binary.
- `hf-completions` is default; air-gapped builds use `--no-default-features`.
- TurboQuant KV is fixed at build time (on with `metal`); `metal-base` = dense KV, no MTP.
- Protected T5 zones: GPU numerics, wire surface, scheduler dynamics, e2e, spyre flex-rs allocator.
- Prefer moving a check to a lower tier (const assert, newtype, compile_error!, lint) over adding a runtime test.

## CONVENTIONS
- Specs/plans are live: update them in the same PR that changes behavior.
- Docs state rules with the reason; keep the "why" when editing.
- `CLAUDE_CODE_BENCH.md` empty result sections are intentional (each names the task that fills it). Don't delete or fill with guesses.

## ANTI-PATTERNS
- Don't duplicate BUILD.md mechanics into READMEs or AGENTS.md; link.
- Don't report numbers in docs without the methodology from `BENCHMARKING.md`.
