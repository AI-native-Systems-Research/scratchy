# crates/models/arch — `scratchy-models` (every arch, one crate)

## OVERVIEW
Build script expands each enabled `dsl/<arch>.py` against `configs/<arch>/` into `$OUT_DIR/<mod>.rs`; `src/lib.rs` includes them. Highest-churn crate when adding models. DSL/config/tokenizer checklist: `dsl/CLAUDE.md`.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Build entry (`main`, `emit_arch`, `reroot_crate`, `bake_tokenizers`) | `scratchy-forwards.rs` (named build script → `build-script-scratchy-forwards`) |
| Hub registry for shell completions | `hf_registry_build.rs` (only build-time network I/O) |
| Feature tables (per stem/arch/`all`) | `Cargo.toml` |
| Arch module cfg gates, `compiled_hf_registry()` | `src/lib.rs` |
| Backend availability tables | `CUDA_ONLY`, `METAL_ONLY`, `SPYRE_CAPABLE` in `scratchy-forwards.rs` |
| Universal carrier-vs-tape gate | `tests/carrier_parity.rs` + `tests/goldens/<arch>-<stem>/` |
| Metal bit-exact logits oracle | `tests/metal_o2_logits.rs` (`#[ignore]`) |
| Other gates | `qwen3_py_parity.rs`, `gemma4_router_affine_fingerprint.rs`, `spyre_weight_folds.rs` |
| Golden generators (torch) | `tests/golden_gen_*.py` |

## CONVENTIONS
- Pipeline per arch: `parse_python_file` → `compile_carrier` → `reroot_crate` (`crate::` → `crate::<mod>::`; leaves `pub(crate)`, `::` paths, `crate::__metal_bake`) → `write_if_changed`. Arches fan out on rayon.
- `SCRATCHY_QUANTS` set once before the rayon fan-out; never mutate env inside it.
- `rerun-if-changed`: watch `quantizations.json` only if it exists; tracked env: `SCRATCHY_GPU`, `KTIR_*`, `SCRATCHY_BUILD_FILTER`, `HF_TOKEN`, `CARGO_FEATURE_HF_COMPLETIONS`.
- Feature names carry no `model-` prefix. Stem collision: vision arch keeps the bare stem, text base gets `-text-only`, else `<arch>-<stem>`.
- Backend tables must match `src/lib.rs` cfg gates; edit both.
- Runtime discovery is `inventory::submit!`; consumers `extern crate scratchy_models as _;`, no per-arch force-links.
- `hf_registry.rs` is always written (empty without `hf-completions`); 7-day cache in `<target>/hf-registry-cache/`; ordered by downloads.
- Tests are data-driven: model facts live in `harness.json`/`config.json`, never per-arch test code; loads go through `try_load`.
- Parity threshold: cosine > 0.99 per logits row; below that is a real divergence.

## ANTI-PATTERNS
- `src/lib.rs` `#![allow]`s are scoped to generated code; don't extend.
- Don't emit an arch for a backend that can't compile it (bad metal code / ISel panic) — gate via the tables.
- Unmatched `SCRATCHY_BUILD_FILTER` tag panics by design.
- Expert-pruned checkpoints (REAP) are dropped from the registry deliberately.
- carrier_parity must run with NO quant features (unless testing a preset).
- `spyre_weight_folds.rs`: f16 floor refuses down-scale folds; granite pins zero folds.
- Fingerprints follow the merged config; never default a width (`unwrap_or(8)` was the bug).
- `metal_o2_logits`: one case per process (pools never torn down).

## COMMANDS
```bash
cargo test -p scratchy-models --features metal,qwen3-0.6b-parity-tiny --test carrier_parity -- --nocapture
python tests/golden_gen_carrier.py --arch qwen3 --stem qwen3-0.6b --tiny [--preset P]
cargo test -p scratchy-models --features metal,qwen3-0.6b --test qwen3_py_parity
cargo test --release -p scratchy-models --features metal,<stem> --test metal_o2_logits -- --ignored --exact --nocapture <case>
```
