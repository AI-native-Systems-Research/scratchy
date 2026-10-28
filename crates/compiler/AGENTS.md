# crates/compiler

## OVERVIEW
Four crates: compile-time forward pipeline (`macros/`), its subtile IR (`subtile/`), the cfg-free op enum (`ir/`), and runtime support that emitted code links against (`.` = `scratchy-forward-compiler`). Earned its file: crate boundary, score ~20.

## STRUCTURE
```
compiler/
├── src/        # scratchy-forward-compiler: runtime-side types emitted code names
├── src/bin/    # probe-weights (feature `probe`)
├── ir/         # scratchy-ir: `Instruction` enum + `WeightAccessors`, one file
├── macros/     # scratchy-forward-compiler-macro: the pipeline (own AGENTS.md)
├── subtile/    # scratchy-subtile: target-neutral IR/tape passes (own AGENTS.md)
└── tests/      # gemma2/phase7 end-to-end - STALE, see NOTES
```

## DEPENDENCY DIRECTION (load-bearing - Cargo.toml comments document each)
- macro -> forward-compiler -> ir. NEVER forward-compiler -> macro (cycle; the re-export was dropped for this).
- forward-compiler has NO edge to `scratchy-target-metal` (metal depends UP on compiler); MTL4 types come via `objc2`/`objc2-metal` directly.
- `subtile` -> forward-compiler is dev-dep only (cpu_golden). Non-test subtile code never names it.
- `nccl` here forwards only `scratchy-ir/nccl`: keeps CUDA C ABI symbols out of the macro's host .so.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| New CUDA interpreter op | `ir/src/lib.rs` `Instruction` (closed, scalar-only fields). Metal/spyre tape ops are `SubOp` in `subtile/` |
| KV offsets / layer stride | `ir/src/lib.rs` `KvOffsets` - stride is NOT always 1 |
| Weight trait emitted code impls | `src/dispatcher.rs` `ScratchyWeights`, `BoundWeight`, `HfFingerprint` |
| Metal chain/followup surface | `src/dispatcher.rs` `MetalForwardOnto`, `MetalChainBody` (cfg metal) |
| Arch registration / `try_load` | `src/arch_registry.rs` (`inventory`-based) |
| Bucket lookup at runtime | `src/lib.rs` `BucketEntry`, `find_bucket` |
| `scr model info` dumps | `src/info.rs` `normalize_slice`, `BucketDump` |
| Drafter / MTP heads | `src/heads.rs` `HeadRegistration`, `draft_head` |
| GDN (linear-attn) state slots | `src/gdn_slot_allocator.rs`, `src/gdn_state_layout.rs` |
| Vision forward | `src/mm.rs` (cfg `vision`) |
| Spyre bundle lookup | `src/arch_registry.rs` `resolve_ktir_bundle` (cfg spyre) |

## CONVENTIONS
- `Instruction` lives in `ir` but stays re-exported here: emitted code spells `::scratchy_forward_compiler::Instruction`.
- Features select the backend (`cuda`/`metal`/`spyre`/`spyre-hw`); `cuda` implies `vision`. `default = []`.
- `ir/`, `macros/`, `.` build.rs files are identical: emit `cfg(fa3_built)` when `CUDA_ARCH` or nvidia-smi says sm >= 90.
- `is_gemm` is CARRIED, never inferred (dispatcher.rs). No one names a weight by string - use `BoundWeight`.
- Unsafe forward methods carry Safety docblocks (live CUDA device, arena-slot invariants); keep them current.

## ANTI-PATTERNS
- Do not add a dep from here to the macro crate or to `scratchy-target-metal`.
- Do not widen `Instruction` fields beyond scalars (it must compile under every feature set, and the macro emits it as const data).

## COMMANDS
```bash
cargo test -p scratchy-forward-compiler --lib            # CI gate (rust.yml)
cargo run -p scratchy-forward-compiler --features probe --bin probe-weights
```

## NOTES
- `tests/*.rs` import `scratchy_forward_compiler_macro::forward`, which no longer exists (macro crate is a plain lib, no attribute macro) and isn't a dev-dep. CI runs `--lib` only, so `cargo test -p scratchy-forward-compiler --tests` is expected to fail. Delete or port them; don't "fix" by re-adding a dev-dep cycle.
