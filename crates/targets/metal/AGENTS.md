# scratchy-target-metal

## OVERVIEW
Apple-Silicon runtime + tape construction for the metal backend (crate `scratchy-target-metal`); sibling crate `compiler/` (`scratchy-target-metal-compiler`) is the macro-time half. Score ~25 (crate root, 40+ modules, build.rs codegen).

## STRUCTURE
```
metal/
├── build.rs        # AoT .metallib per shader, steel-paged codegen, MTL3 + function-constant guards
├── compiler/       # proc-macro-time lowering: SubtileTape -> step tape -> static per-bucket tapes
├── shaders/        # MSL kernels; mlx_* = vendored MLX headers
├── src/tape/       # pure, objc-free tape model + lowering (macro runs it at expansion)
├── src/interpreter/metal/  # worker pool, arenas, MTL4 encoding, forward()
├── src/*.rs        # device, allocator, op_abi, aot bake, sampling, loaders, cpu oracles
└── tests/          # one integration-test binary per *_test.rs
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Target-ABI const tables (fusions, colour rules, elidable ops) | `src/op_abi.rs` (`METAL_FUSIONS{,_NARROW}`, `METAL_COLOUR_FACTS`, `METAL_ELIDABLE`) |
| SubtileTape -> metal entry (refusal contract) | `src/from_tape.rs` |
| Baked-shader compile per constant set | `src/aot.rs` (`bake`, `baked_build`, `baked_pipeline`) |
| Single-op MTL4 dispatch (tests, side paths) | `src/mtl4_dispatch.rs` |
| Buffers, weight load, shader sidecar cache | `src/metal_allocator.rs`, `src/metal_mem.rs` (always `StorageModeShared`) |
| CPU oracles for kernel parity | `src/cpu_reference.rs`, `src/cpu_golden.rs` |
| Steel-paged symbol table | `src/lib.rs::steel_paged` (includes `OUT_DIR/steel_paged_kernels_generated.rs`) |
| Add a paged-attention head-dim | `build.rs` `STEEL_PAGED_HEAD_DIMS` / `STEEL_PAGED_DTYPES` (one-line edit) |

## CONVENTIONS
- `lib.rs` re-exports tensor/core types as `__gpu` scaffolding: macro-emitted code names `crate::__gpu::*` paths. Tiny shim modules (`kv_cache`, `gdn_state`, `quantized`, `targets`, `mm_dispatch`) exist only for path compatibility. Don't delete them.
- `BLOCKS_PER_CHUNK = 128` (lib.rs) is the single source of truth and gets baked into synthesized MSL.
- Features: `forward-telemetry`, `sampler-telemetry`, `vision` (gates `mm_dispatch`, `vision_arch`). All off by default.
- Toolchain: `MIN_MACOS = (26, 2)`, and build.rs panics if the SDK is older. Shaders that reach `<MetalPerformancePrimitives/>` are detected automatically and compiled with `-fno-fast-math -std=metal4.0`, to dodge the SDK `matmul2d` miscompile.
- The crate-level `#![allow(clippy::too_many_arguments, clippy::type_complexity)]` in lib.rs predates the rule. It's not a precedent for new allows.

## ANTI-PATTERNS
- 🛑 METAL 4 ONLY. Never call `.commandBuffer()`. `build.rs::guard_no_classic_mtl3` scans `src/` and `tests/` and panics, and root `clippy.toml` `disallowed-methods` is its twin. Dispatch only via `mtl4_dispatch` or `run_bucket_mtl4`.
- A lowering error is a REFUSAL, NEVER a FALLBACK (from_tape.rs). Slots and sources are not interchangeable. Don't run `fuse_silu_mul` here, because it renumbers slots.
- Never hand-list head dims at a call site. Use `steel_paged_symbol` and the generated table.
- `op_abi` entries are data, never logic. The colourer and the emitter arm must agree, or the output is garbage.
- MTL4 encoders don't auto-serialize same-encoder dispatches. A producer->consumer pair needs an explicit barrier.
- Shared-storage buffers are read-only across threads (metal_allocator SAFETY notes). The cache build never holds a lock and never persists partial state.
- RmsNorm gains are never pinned at load: loaders cast the gain dtype themselves.

## TESTS (`tests/`)
- Each `<kernel>_test.rs` is its own binary: `cargo test -p scratchy-target-metal --test <stem>`.
- No-GPU skip idiom: `let Some(dev) = detect_device() else { eprintln!("skipping: no Metal 4 GPU"); return; }`. NAX-only tests also gate on `is_nax_capable`.
- Parity means identical bits against a `cpu_reference`/`cpu_golden` oracle, or against the unfused kernel chain it replaces.
- Inputs are deterministic: `kernel_inputs::{pseudo, build_cos_sin}` (xorshift), never `rand`. Shapes are real decode shapes.
- `mod common;` re-exports the production `mtl4_dispatch::*`. Don't hand-roll encoders.
- `*_probe_test` / `*_bench_test` are `#[ignore]`d measurements: `-- --ignored --nocapture`.
- To reach mixed-bit expert paths, build mixed-bit slabs: uniform 4-bit never exercises them (affine_gather_qmv_test).

## COMMANDS
```bash
cargo test -p scratchy-target-metal                 # macOS + Metal 4; tests self-skip otherwise
cargo test -p scratchy-target-metal --test gemm_crossover_probe_test -- --ignored --nocapture
```
