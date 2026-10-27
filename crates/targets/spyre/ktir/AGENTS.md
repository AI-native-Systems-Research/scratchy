# ktir/ — KTIR toolchain (4 crates)

## OVERVIEW
KTIR = Spyre's tile IR: built as const data by `#[forward]`, then interpreted (emulator), rewritten (optimizer), or lowered to SuperDSC JSON (superdsc). Earned its file because of the 4-crate dependency graph plus ktir-core (score ~12; core is too small for its own file).

## STRUCTURE
```
ktir-core/       zero-dep IR + codecs; every other crate builds on it
ktir-optimizer/  IR->IR passes            (deps: ktir-core)
ktir-emulator/   CPU/Metal interpreter     (deps: ktir-core, ktir-optimizer[default], objc2[metal])
ktir-superdsc/   KTIR -> SuperDSC lowering (deps: ktir-core, serde, serde_json, half)
```
Dependency direction is load-bearing: superdsc and optimizer must never reach the emulator (objc2 is Apple-only and breaks linux-cuda).

## WHERE TO LOOK
| Need | Location |
|------|----------|
| Op / function / module types | `ktir-core/src/ir.rs` (`Operation<'a>`, `IRFunction`, `IRModule`, `Value`, `Ssa`) |
| Op allocation | `ktir-core/src/arena.rs` (`Arena`; read its SAFETY contracts before touching it) |
| f16/bf16/e4m3/dlf16 rounding | `ktir-core/src/codec.rs` |
| MemRef / TileRef, `STICK_BYTES=128` | `ktir-core/src/memref.rs` |
| Affine maps/sets, symbolic bounds | `ktir-core/src/affine.rs` |
| Op kind enum, attr keys | `ktir-core/src/opkind.rs`, `attrkey.rs` |
| Fast hashing | `ktir-core/src/fxhash.rs` (`FxHashMap`, used everywhere instead of std HashMap) |
| Per-crate detail | `ktir-emulator/AGENTS.md`, `ktir-optimizer/AGENTS.md`, `ktir-superdsc/AGENTS.md` |

## CONVENTIONS
- KTIR is **constructed, never parsed**. The affine text parser was deleted from ktir-core; don't bring back a text front door.
- Ops are `Operation<'static>` baked const data; runtime structures borrow the op's affine data instead of copying it.
- Unknown symbols are refused by name, never defaulted to 0.
- Cargo.tomls are vendored/normalized style (`autotests/autobenches/autoexamples = false` in the emulator), so declare every new target explicitly.
- Module docs cite their Python (`ktir_cpu`) or torch-spyre source plus behavior deltas; keep the citations when editing.
- Edition 2024; Apache-2.0 SPDX header on every file.

## ANTI-PATTERNS
- Adding a dependency to ktir-core. It stays dependency-free.
- Adding any `ktir-emulator` edge from ktir-superdsc, ktir-optimizer, or a non-Apple consumer.
- Re-implementing codecs or `STICK_BYTES` locally; import them from ktir-core.

## COMMANDS
```bash
cargo test -p ktir-core
cargo test -p ktir-optimizer
cargo test -p ktir-superdsc
cargo test --release -p ktir-emulator --features metal --test port_ops
```
