# crates/compiler/macros (scratchy-forward-compiler-macro)

## OVERVIEW
The whole forward pipeline, run from `scratchy-models`' build.rs. Despite the name it is a NORMAL lib: no attribute macro, no `proc-macro = true`. Score ~28 (31 files, ~104k LOC, highest centrality).

## PIPELINE (lib.rs `compile_carrier`, :572)
`parse_python_file` -> `classify` -> `shape::infer` -> `cfg::build_cfg` -> `fuf::unroll` -> [`solver::solve`, cuda only] -> `schedule` -> `codegen::emit_model` / `interpreter_codegen::lower_bucket` -> `render::render_tokens` (build.rs writes the file).
spyre/metal skip `solver`/`cost`/`impl_lib` matching; they lower via `to_wavefront` -> `scratchy_subtile::lower` -> `interpreter_codegen`.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Build-script entry points | `lib.rs` `compile_carrier`, `check_head_ladders` (:380) | public API is the `pub use` block :53-79 |
| HF config / arch.json parsing | `config.rs` `ModelParams`, `load_dir` | `selected_config_paths` drives feature scoping |
| Quant presets / storage formats | `quantization.rs` `QuantizationConfig`, `StorageFormat` | consumed by `hf_registry_build.rs` |
| DSL parsing | `parse_python.rs` `PythonCarrier` | |
| Per-op kernels (cuda ISel) | `impl_lib.rs` `Implementation`, `ImplementationLibrary` | 19.9k lines; new kernels go HERE, not in codegen |
| Weights struct + forward fns | `codegen.rs` `emit_model` (:13626) | 18.5k lines |
| Tape tokens, slot coloring, buckets | `interpreter_codegen.rs` `lower_bucket`, `colored_slot_map` | metal+spyre emission |
| Weight accessor set (metal/spyre) | `weight_bindings.rs` | the ONE producer; cuda derives via ISel |
| TP all-reduce/all-gather insertion | `tp_lowering.rs` | cfg `nccl`; TP set {1,2,4,8} |
| Spyre KTIR programs | `ktir_tokens.rs` `program_tokens` | cfg spyre |
| Vision variants | `vision_glue.rs` `emit_per_variant` | |
| Metal static-tape bake | `metal_bake_module` re-export | from `scratchy-target-metal-compiler` |

## CONVENTIONS
- `cuda`/`metal`/`spyre` mutually exclusive via `compile_error!` (lib.rs:27-34). `spyre-hw` = card; `spyre-emu` = emulator plus bake-time numeric gate vs dsl_math (divergence past quant floor fails the build).
- Without `nccl`, only tp=1 compiles, byte-identical to single-rank.
- `turboquant`: metal-only KV codec, fixed per model at build time.
- Depends on `scratchy-target-metal-compiler`, never the reverse.
- Unit tests live inline at module bottoms; `tests/arch_declarations.rs` guards "DSL files are math only, arch facts in `configs/<arch>/arch.json`" (expects an exact DSL-file count - update it when adding an arch).

## ANTI-PATTERNS
- Pre-existing `#![allow(dead_code)]` in impl_lib/solver/schedule/quantization/to_wavefront/shape/target: legacy. Never add more (root rule).
- `#[cfg(test)]` imports must not widen the production cfg under non-cuda backends.
- `protected` slot color must never appear on another slot (interpreter_codegen.rs:4483).
- KTIR program is BUNDLE-INVARIANT; only bound addresses vary per launch (ktir_tokens.rs:319).
- Wiring must not describe a program that did not ship (codegen.rs:10958); the dispatcher fn is always emitted and refuses, never cfg'd out (:13779).
- Distinct vision layouts must not share a canonical (lib.rs:1016).
- CSV match is strict: softcap decode must not accept a nosoftcap-only CSV (impl_lib.rs:17345).

## COMMANDS
```bash
cargo test -p scratchy-forward-compiler-macro --features metal --lib   # CI variant-selection gate
scripts/emit-arch.sh <arch> <stem> [--rerooted] > /tmp/x.rs            # any arch, any host, no backend
cargo run -p scratchy-forward-compiler-macro --example emit_arch -- <arch>
```
`emit-arch.sh` matches build.rs emission byte-for-byte (modulo `reroot_crate`); use it to diff codegen changes without a full model build.
