# crates/layers — `scratchy-layers` backend-neutral layer types

## OVERVIEW
Layer structs, weight loading, KV/GDN pools, rotary math and TurboQuant codec — types + pure accessors only; used by compiler, models/arch and every target.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Linear/norm/embedding types, neutral loaders | `src/layers.rs` (`LinearLayer` enum, `linear_load`, `embedding_load`, `try_synthesize_packed_slice`) |
| Weight loading, mmap, pre-cast, LoRA merge, affine gate | `src/weights.rs` (`GpuWeights`, `TensorRefs`, `UploadSrc`) |
| Paged KV pool, spans, metal chunk allocator | `src/kv_cache.rs` (`KvCachePool`, `drop_chunks`) |
| GatedDeltaNet state pool | `src/gdn_state.rs` (`GdnStatePool`) |
| MoE layer structs, routing | `src/layers_moe.rs` (`FusedMoELayer`, `select_moe_block_m`) |
| RoPE / YaRN / LongRoPE host math | `src/rotary.rs` (`RotaryCache`) |
| TurboQuant KV codec | `src/turboquant.rs` (`PolarQuantizer`, `TurboQuantKvStore`, `codec_for`) |
| Layered vision/text loaders | `src/loaders.rs` |

## CONVENTIONS
- This crate names no backend. Kernel-calling `forward` and backend-bound `load` live in target crates as `*Ops` / `*Ext` extension traits (e.g. `CudaRotaryExt`).
- Allocation through a caller-supplied `alloc_buffer` closure.
- Re-exports `scratchy_tensors::{device_allocator, dtype, tensor}` so it can serve as the backend-less `__gpu` alias (module mirror).
- Pools are generic over `M: PoolMemory`, hold raw GPU pointers; every view has a `// SAFETY:` (pool owns memory, view borrows `&self`).
- `weights.rs` load pipeline: `madvise(WILLNEED)`, rayon shard-header parse, background F32→BF16/F16 pre-cast ahead of `take()`.
- `turboquant.rs` is a faithful port of turboquant-mlx (codebook, bit layout) — change both sides or neither.
- GDN pool: one slot per resident sequence, not token blocks.

## ANTI-PATTERNS
- No backend runtime methods or cfg(cuda/metal) here (guarded by `crates/tensors/tests/cfg_neutrality.rs`).
- KV: co-resident layers never write the same physical block; never read stale chunk-table entries after `drop_chunks`; always keep chunk 0 (`keep.max(1)`); vision codegen never emits paged-attention reads.
- GDN: non-GDN codegen never emits a `GatedDeltaNet` read (panics by contract).
- weights: never cast integer weights; never examine unlisted `TensorRefs` rows (test `an_unlisted_tensor_is_never_examined`); don't resolve lm_head prefix aliasing (variants bake the prefix).
- rotary: keep the fill NaN-branch-free so it vectorizes.

## COMMANDS
```bash
cargo test -p scratchy-layers     # no features, no build.rs
```
