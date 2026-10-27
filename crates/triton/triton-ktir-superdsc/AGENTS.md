# triton-ktir-superdsc - KTIR value -> ktir-superdsc door

## OVERVIEW
Adapter only. It reshapes our KTIR into `ktir-superdsc`'s `KtirNode` and calls its entry points; every descriptor decision belongs to `ktir-superdsc`.
Earned its file: distinct domain and the config table three crates share (score ~15).

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Request node from KTIR | `src/lib.rs` `node_for` (parameter position IS the `BufferId`) |
| Per-kind emission | `emit_node`, `emit_node_with(node, None)` = vendored unit-test arm |
| Whole-function door (attention) | `emit_whole` |
| Rope | `drive_rope`, `RopeRows` (head dims 64/128/256 only) |
| Device memory plan | `src/layout.rs` (`for_node`, `for_regions`, `reserved_consts_for`) |
| Test configurations (fixture + constexpr + grid) | `src/cases.rs` (`ALL`, `case`, `root`) |
| dxp bundle serializer | `src/bake.rs` -> `bundle.mlir` + `sdsc_N.json` + `init_binary.bin` |
| Refusal debugging | `dump(&KtirNode)` (one line per op) |

## CONVENTIONS
- Configurations are stated ONCE in `cases.rs`: consumed by `bake_py`, `triton-ktir`'s `pure_rust_ktir`, `triton-numeric`, and mirrored by `test-numeric/gen_numeric_data.py`. The grid is part of the configuration.
- Layouts are derived from the program's own `regions`, never guessed.
- `Error { stage, message }` with named stages: `handoff`, `regions`, `b-orientation`, `program`, `geometry`, `ktir-superdsc`, `whole-function`.
- `serde_json` is a direct, declared edge (the documented dependency exception).
- Contract rules cite the shipped fixture or the producer's file:line. Keep the citations when you edit.

## ANTI-PATTERNS
- NEVER re-add a `Program::Attn` per-node door (`drive_attn` was deleted): attention goes only through `emit_whole`.
- NEVER default geometry: `HEAD_DIM`/`H`/`GQA` are read from constexprs. `GQA` is the grouping (NKVH = H/GQA), not the kv head count.
- Matmul B orientation is read from `indexing_maps` (unrecoverable at k == n); a node with disagreeing matmuls is refused.
- `RopeRows::HeadMajor` is refused by name: same shapes, different bytes.
- No `_` arms on `Program`: a new variant must be an E0004.
- `layout.rs` refuses BEFORE placing if a reserved constant's extent is unstated; a half layout is worse than `None`.
- bake: top-level key `{idx}_{op_name}` (the index orders ModuleStitcher); `time > 1` ops are pre-unrolled, never `scf.for`; no `manifest.json`.
- `headmajor: None` on this door; head-major o_proj is spyre-bundle-only.
- Tests build defect inputs in-test (assert the premise first) and never read a fixture that is meant to change.
- `ktir-superdsc` must NEVER depend on `ktir-emulator`; that edge lives only in triton-numeric.

## COMMANDS
```bash
cargo test -p triton-ktir-superdsc
cargo run --offline -p triton-ktir-superdsc --example drive -- rmsnorm_granite 1 rmsnorm   # KTIR_NO_LAYOUT=1: no-layout arm
cargo run -p triton-ktir-superdsc --example bake_py     # whole .py -> dxp chain; KTIR_WHOLE=1: whole-function door
```
On-pod gate: `dxp_standalone -d <dir> -b sentient` with `SENARCH=MPW4` (pod is `cargo --offline`).
