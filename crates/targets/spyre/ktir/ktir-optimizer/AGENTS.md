# ktir-optimizer — KTIR -> KTIR rewrite passes

## OVERVIEW
Pure IR passes over `ktir-core` (no features, no other deps). Its only direct dependent is ktir-emulator (feature `optimizer`). Outside crates reach it through the re-export `ktir_emulator::ktir_optimizer` (e.g. `crates/targets/spyre/src/runner.rs` uses `fusion::{Binding, NodeSpec, ProgramSpec}`). Score ~11 (distinct domain, ~9k LOC).

## WHERE TO LOOK
| Pass | File | Entry points |
|------|------|--------------|
| Flash attention | `src/flash_attn.rs` (2.9k) | `apply_flash_attention`, `recognize_attention`, `tile_attention`, rerolled variants |
| Segment fusion | `src/fusion.rs` (1.9k) | `fuse_program`, `plan_segments(_budgeted)`, `ProgramSpec`, `NodeSpec`, `Segment` |
| Head rewrite / decode islands | `src/head_rewrite.rs` (2.3k) | `apply_head_rewrite` |
| Matmul tiling | `src/matmul_tile.rs` | `apply_matmul_tiling` |
| Tile coalescing | `src/tile_coalesce.rs` | `apply_tile_coalesce(.., budget)` |
| Splat sinking | `src/splat_sink.rs` | `apply_splat_sink` |
| Op construction helpers | `src/ir_builder.rs` | |

## CONVENTIONS
- Pass sig: `(a: &Arena, module: &mut IRModule<'a>, ...) -> usize`. The return value is the rewrite count, so 0 means a no-op.
- New ops are allocated in the caller's `Arena`; never hold borrows across arena growth beyond what `Arena` permits.
- Tests are inline `#[cfg(test)]` modules; there is no tests/ dir.
- Fusion budgets are LX byte budgets (`plan_segments_budgeted`). Take the number from the emulator/hardware model, not a local constant.

## ANTI-PATTERNS
- Adding a dependency on ktir-emulator (that would be a cycle) or on any device crate. Keep it host-pure.
- New env-var toggles or I/O inside a pass. Passes are deterministic IR -> IR.
- Changing a pass's semantics without checking `ktir-emulator` (`program.rs`, `resident.rs`, `segmented.rs`) and `spyre/src/runner.rs`.

## COMMANDS
```bash
cargo test -p ktir-optimizer
cargo test -p ktir-emulator      # exercises passes end-to-end through program::execute
```
