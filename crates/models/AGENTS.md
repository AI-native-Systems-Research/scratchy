# crates/models — model crates: arch, quantization, spec, vision

## OVERVIEW
Four crates; `arch/` has its own AGENTS.md. This file covers the map plus `quantization/`, `spec/`, `vision/`.

## WHERE TO LOOK
| Crate (dir) | Role |
|-------------|------|
| `scratchy-models` (arch/) | see `arch/AGENTS.md` |
| `scratchy-quantizations` (quantization/) | presets + quant layer types + GGUF |
| `scratchy-spec` (spec/) | doc-only lib; feature `mtp = ["scratchy-models/mtp"]` |
| `scratchy-vision` (vision/) | pure-CPU vision-tower host glue (rope, pos-embed interp, preprocess) |

| Task | Location |
|------|----------|
| Add/inspect a preset | `quantization/presets/<name>.json` + same-named Cargo feature |
| Active preset list for the build script | `quantization/src/lib.rs` `enabled_presets()` (`preset!` macro, cfg only) |
| MLX affine dequant / layers | `quantization/src/affine.rs`, `layers_quant.rs` (`AffineQuantLinear`, `Nvfp4Linear`) |
| GGML block types | `quantization/src/ggml_quant.rs` (`GgmlDType`, `GgmlStorage`) |
| GGUF parse, tokenizer, config map, mmproj | `quantization/src/gguf/{mod,format,spec}.rs` |
| Vision config / rope / window dispatch | `vision/src/lib.rs` (`VisionConfig`, `build_qwen2_5_window_dispatch`) |
| Per-arch processor metadata | `vision/src/mm_meta.rs` (`MmMetadata`, `PlaceholderPolicy`, `SizePolicy`) |
| Image decode / resize / placeholder expansion | `vision/src/preprocess.rs` |

## CONVENTIONS
- Preset name == Cargo feature == `quantizations.json` key, verbatim, no `quant-` prefix.
- `#[forward]` finds `presets/*.json` by walking up from the consuming manifest to the workspace root.
- scratchy-models depends on quantizations as both dep and build-dep; feature refs unify across both edges.
- Quant scope is compile-time cfg only (`enabled_presets`), never env reads.
- Vision: arch names never appear in compiler/serving; each arch declares `pub const PROCESSOR: MmMetadata`, baked via `#[vision_forward(processor = ...)]`.
- Vision host fns cite the exact upstream Python they mirror (HF / mlx-vlm); keep that when adding.
- `scratchy-vision` `cuda/metal/nccl` features are inert symmetry toggles; device helpers live in `scratchy-target-cuda::vision_device`.
- `image` crate: jpeg/png/webp only, default-features off.

## ANTI-PATTERNS
- Never bundle `nvfp4` into `mlx` (codegen panics `UnsupportedWeight` on the SDSC/wavefront path).
- Never give quantizations a default preset (would kill dense emission on every metal build).
- Wrong `mrope_positions` silently corrupts attention — verify against HF.
- No backend deps in `scratchy-vision`.

## COMMANDS
```bash
cargo test -p scratchy-quantizations
cargo test -p scratchy-vision
cargo build --release -p scratchy-cli --features metal,model/gemma-4-26b-a4b-it,spec/mtp
```
