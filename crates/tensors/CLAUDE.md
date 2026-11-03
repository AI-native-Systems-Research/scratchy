# crates/tensors — `scratchy-tensors` backend-neutral tensor core

## OVERVIEW
`GpuTensor`/`TensorView`, `DType`, `DeviceAllocator` and weight-source seams every backend builds on; also hosts the workspace-wide cfg-neutrality guard test.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Tensor descriptor | `src/tensor.rs` (`GpuTensor` `#[repr(C)]` 32B `Copy`, `MAX_DIMS = 4`; `TensorView<'a>`) |
| Allocator seam | `src/device_allocator.rs` (`DeviceAllocator`, `PoolMemory`) |
| Dtypes | `src/dtype.rs` (`DType`), `src/lowering_dtype.rs` (`MetalDtype`, `ScaleDtype`) |
| Weight source seam | `src/weight_source.rs` (`WeightSource`) |
| Owned / raw memory | `src/owned_tensor.rs`, `src/raw_mem.rs` (`RawGpuMem`) |
| Pre-cast pipeline seam | `src/precast.rs` |
| Weight naming | `src/weight_path.rs` |
| Opaque handles, `LoadStream` | `src/lib.rs` (`ForwardCtxHandle`, `ForwardDeviceHandle`, `GpuWeightsHandle`, `MetalRuntimeHandle`) |
| Architecture guard | `tests/cfg_neutrality.rs` |

## CONVENTIONS
- No CUDA/Metal types in signatures. Backend facts enter via trait seams or the opaque lifetime-carrying handles, recovered with `unsafe as_ref/as_mut`.
- Only cfg gate: `cuda` feature switches `LoadStream` (`CUstream` vs `()`).
- `cudarc` features must match every consumer (`dynamic-linking` + `cuda-version-from-build-system`); mixing in `dynamic-loading` panics cudarc's build script.
- `memmap2` is here only for `register_mmap`'s signature.
- `GpuTensor` is contiguous row-major, owns nothing; lifetime belongs to the region (weights/arena/KV pool).
- `DeviceAllocator` is used monomorphized, never `dyn`.
- Crate-level `#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]` is pre-existing; don't add more.

## ANTI-PATTERNS
- `cfg_neutrality.rs`: no `cfg(feature = "cuda"/"metal")` in neutral zones (incl. `crates/models`, `crates/tensors`); neutral code must not name `scratchy_target_cuda`, `scratchy_serving_cuda`, `CudaGraph`. Dispatch via trait/registry seams.
- `WeightSource` is frozen: object-safe, no allocator/precast/streaming methods.
- `adopt_raw` / `alloc_and_copy_device` are cuda-only hooks (`unreachable!` default); calling them on metal is a bug.
- Handle `as_ref/as_mut`: `T` must be the exact concrete type the handle was built from; no aliasing the `&mut`.
- Don't cast bf16 disk weights to fp16 (exponent clipping → garbage, e.g. Llama-3.x on metal).

## COMMANDS
```bash
cargo test -p scratchy-tensors                     # incl. cfg_neutrality guard
cargo build -p scratchy-tensors --features cuda
```
