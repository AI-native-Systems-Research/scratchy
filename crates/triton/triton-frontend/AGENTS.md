# triton-frontend - bridge one: Triton .py -> TTIR value

## OVERVIEW
Owned, dependency-free compile of `@triton.jit` source to `ttir::Module`, diffed against real Triton's `make_ir`.
Earned its file: ~10k LOC, the most-imported crate on the ladder (score ~22). `README.md` holds the status table and the "What is left" list.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Public entry | `codegen::{compile, compile_ast, KernelSpec, ArgSpec, parse_dtype}` (`src/codegen.rs`, 2.9k LOC) |
| Typing / promotion rules | `src/semantic.rs` (`Val`, `Semantic`) |
| Target-conditional policy | `src/target.rs` (`Target::div_promotes_narrow_floats`) |
| Pass pipeline (mirrors Triton `make_ir` order) | `src/opt.rs` (`make_ttir`, folds, `inline_calls`, `cse`) |
| Python lexer/parser | `src/py/parser.rs` (only entry `parse`) |
| Scope boundary | `src/py/census.rs` (`check`, `classify_call`, `TL_FUNCTIONS`) |
| Shared vocabulary with triton-ktir | `src/ttir/mod.rs` (the ONLY shared surface) |
| Golden reader / printer / diff | `src/ttir/{parse,print}.rs`, `src/diff.rs` (tests only) |
| Symbol mangling | `src/mangle.rs` |
| Fixture status / stale pins | `tests/fixture_status.rs`, `tests/golden_diff.rs`, `tests/divergence.rs` |

## CONVENTIONS
- `py` never names a TTIR type and `ttir` never names a Python one; they meet only in `codegen`.
- Swapping the TTIR vocabulary must stay a change contained to `src/ttir/mod.rs`.
- The AST is census-bounded, not a CPython mirror. Out-of-census = error naming construct + line:col.
- Positions follow CPython: 1-based line, 0-based col. MLIR `loc` columns are 1-based.
- Bridge-one goldens are `<name>.ttir_raw.mlir` (`make_ir` raw). `<name>.ttir.mlir` (post `make_ttir`) is bridge two's input; never diff bridge one against it.
- Golden diff compares values structurally, never printed strings.
- opt passes return `bool` (changed?). CSE keep-list is fail-closed: unlisted ops are never merged.
- Floats are NOT constant-folded here (golden fidelity).
- `mangle.rs` works around Rust's f64 `Display` (`1e-05` prints as `0.00001`).
- Host code in fixtures (`dict(...)`, f-strings, asserts) is skipped structurally, not parsed.

## ANTI-PATTERNS
- f16 / f16 MUST NOT promote to f32 on Spyre: Triton's rule is a PTX fact and makes kernels illegal at the next contraction.
- f16 -> int is `extf` to f32 THEN `fptosi`, never a direct `fptosi`.
- Never reintroduce a third-party parser (ruff pulled 44 lockfile packages), behind a feature or otherwise.
- Parser refuses by name: `while`, `with`, `try`, `class`, walrus, f-strings, string prefixes, `;`, `\\` continuations.
- Compiling a kernel Triton refuses is a silent-wrong-answer risk: `bias_add_f32` is an ASSERTED XFAIL.
- A golden-parse test must fail on zero functions / lost ops; a subset parser would let every diff pass.
- Invariant (`ttir/mod.rs`): a result's location IS its op's location; maintained by hand, not by structure.

## COMMANDS
```bash
cargo test -p triton-frontend            # offline-safe; tests/no_parser.rs covers the parser-free half
PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python3 tools/gen_ttir_goldens.py [outdir]  # needs real Triton
```
New fixture: add `.py` to `../test-fixtures`, then regen goldens, then add an entry to `tests/fixture_status.rs`.
