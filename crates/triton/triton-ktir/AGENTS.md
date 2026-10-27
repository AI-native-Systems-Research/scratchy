# triton-ktir - bridge two: TTIR value -> KTIR value

## OVERVIEW
Behavioural port of the C++ KTDP/KTDF passes. Takes frontend's `ttir::Module` via `from_ttir::convert` and emits `ktir_core` IR.
Earned its file: ~17k LOC, 12 passes, 141 tests (score ~24).

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Pipeline entry | `src/lib.rs` `make_ktir(&mut Module, grid)` (the 7 passes in C++ order) |
| Mutable IR (deliberate divergence from ktir-core's arena `Copy`) | `src/ir.rs` (`Op`, `OpKind`, `AttrKey`, `Attr`, `IrType`) |
| TTIR -> our IR | `src/from_ttir.rs` (`OP_SET`, `ATTR_KEYS`, `convert`) |
| Consumer-contract audit (`ktir_to_superdsc`) | `src/consumer.rs` (`WALKED_OPS`, `READ_ATTRS`, `check`) |
| Largest pass | `src/passes/to_ktir.rs` (3.7k LOC); `to_ktir::lower` is the door |
| ktir_core emission | `src/passes/to_ktir_emit.rs` (only callable from `to_ktir::lower`) |
| Matmul lowering + const-int helpers | `src/passes/dot_to_linalg.rs` |
| Shared traversal/edit toolkit | `src/passes/walk.rs` |
| Golden text tools (tests only) | `src/text/{parse,print,diff}.rs` |
| Golden plumbing | `tests/common/mod.rs` (reads `../test-goldens/ktir/<config>/<stage>.mlir` + `grid.txt`) |

## CONVENTIONS
- Each pass names the C++ file it ports and reproduces its DIAGNOSTICS too; `vector_add`/`mul` prove the refusals.
- `ir` mirrors `ktir::ir` shapes (same `Ssa`, matching AttrKey names) so a swap is a rename table.
- One IR model for ttir and KTIR; the dialect is the op-name prefix. One block per region.
- `Module::hints` (SSA print names) are diagnostic-only; no pass reads them.
- `text` parser is line-oriented over machine-printed MLIR only; an unaccountable line = error naming the line.
- Printer emits custom assembly formats (the consumer refuses generic form) and `#map`/`#set` aliases.
- `text::diff` is structural on the SSA graph: renames are not differences, rewires are. `Difference.field` names the field.

## ANTI-PATTERNS
- `crate::passes` must NEVER use `crate::text`. Wanting to parse in a pass means `ir::Op` is missing a field.
- `from_ttir` has NO fallback arm. The deliberate `Other`s (`negf`, `rsqrt`, `cmpf`, `sitofp`, `fptosi`, `absf`) must stay distinguishable from "unmodelled".
- Three C++ passes are deliberately NOT ported and must not be: `LaneMajorLayout`, `CarriedValuesToMemory`, `GridIndexChains`.
- `ktdp.load` stays out of `is_pure`: C++ DCE keeps the dead read, and censuses must match.
- Parser must never skip an op: a skipped op shows up as a phantom census diff blamed on a pass.
- Tests fail closed on a missing golden (never SKIP).

## COMMANDS
```bash
cargo test -p triton-ktir                       # offline-safe
cargo test -p triton-ktir to_ktir::             # one pass's inline tests
cargo test -p triton-ktir --test golden_ktir    # C++ op-census parity
cargo run --offline -p triton-ktir --example emit_ktir -- <config> /tmp/out.ktir.mlir
cargo run --offline -p triton-ktir --example census_sched -- attention_flash_noncausal
cargo run --offline -p triton-ktir --example handable -- <config> <grid-csv>
```
Golden regen: the command printed by `tests/common/mod.rs`'s panic (triton-spyre side).
