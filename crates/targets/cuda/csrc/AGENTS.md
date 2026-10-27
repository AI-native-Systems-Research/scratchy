# csrc — CUDA C++ kernels for scratchy-target-cuda

## OVERVIEW
CUDA C++ kernel sources, mostly vLLM/llama.cpp-derived with torch stripped out. The public surface
is `extern "C"` launchers bound in `../src/kernels.rs`. It's a distinct domain (C++, its own build
path, about 30k LOC) and the Rust parent doesn't cover it.

## STRUCTURE
```
csrc/
├── *_kernels.cu, quantized.cu, ...  # -> libvllm_kernels.a (quantized.cu alone -> libggml_kernels.a)
├── cutlass_standalone_gemm.cu       # macro-generated GEMM tile zoo for the solver
├── cutlass_moe_grouped.cu, cutlass_gemm_{silu_mul,bias}*  # own .a each
├── core/scalar_type.hpp             # vllm::ScalarType (only TORCH_CHECK replaced)
├── cutlass_scaled_mm/               # fp8 scaled_mm: c2x_sm89 + c3x_sm90
├── marlin/                          # W4A16 dense Marlin + GPTQ/AWQ repack
└── marlin_moe/                      # Marlin MoE, reusing marlin's header layout
```

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Add/change a launcher | the `.cu` file's trailing `extern "C"` block + the wrapper in `../src/kernels.rs` |
| Compile flags, file lists, lib names | `../builder/build.rs` |
| Link names | `../build.rs` (and `../cost-sweep/build.rs` for its trimmed subset) |
| GEMM configs for the cost model | `cutlass_standalone_gemm.cu` `CUTLASS_GEMM_CONFIG` families |
| fp8 tile dispatch | `cutlass_scaled_mm/*_fp8_dispatch.cuh` |
| Marlin kernel body | `marlin/marlin_template.h` (2k LOC; changes reach marlin_moe too) |
| GGML K-quant dequant | `quantized.cu` (6k LOC, large grid tables, skim) |

## CONVENTIONS
- Launchers take raw pointers + dims + `cudaStream_t` and return int status codes. Native ones use the `scratchy_` prefix.
- Arch gating happens at compile time (`__CUDA_ARCH__`, `common.hpp` `enable_smXX_to_smYY`). Per-arch
  variants are separate symbols/files (`_sm80`/`_sm89`/`_sm90`, `c2x`/`c3x`). Rust picks the variant at runtime.
- Every `extern "C"` entry is ALWAYS defined, so FFI links on any arch. On the wrong arch it's a
  stub, or `abort()` for c3x without `ENABLE_SCALED_MM_SM90`. Only sm_90+ build hosts get that define.
- Keep upstream SPDX headers and lineage comments. Edit vendored files sparingly and diff them against upstream.
- `MARLIN_NAMESPACE_NAME` (`marlin` / `marlin_moe_wna16`) avoids ODR clashes between the two Marlin copies.
- Marlin `b_type_id`: 1 = kU4 (AWQ, zero-point), 0 = kU4B8 (GPTQ).
- Scaled-mm epilogues must expose `EVTCompute` + static `prepare_args`. c3x small-M configs use
  `swap_ab` with the scale args swapped `(b_scales, a_scales)`.
- CUTLASS workspace uses `cudaMallocAsync`/`cudaFreeAsync` (graph-capture safe).

## ANTI-PATTERNS
- Never hand-edit `marlin*/kernel_selector.h` or `sm80_kernel_*.cu`. Regenerate them with
  `python3 marlin/generate_w4a16_kernels.py` (W4A16, SM80+, stages=4 only).
- The marlin selector has no default branch, so an unsupported config fails loud. Don't add a fallback.
- Only `half`/`nv_bfloat16` compute types are allowed (`STATIC_ASSERT_SCALAR_TYPE_VALID`).
- Pre-SM89 fp8 MoE GEMM is a never-called stub (Rust routes it to dequant). Don't make it reachable.
- Failures abort (`MARLIN_CHECK`, `CUTLASS_CHECK`, trap on a wrong SM). Never swallow them.
- Marlin workspace locks are zeroed to `sms*4` before each launch. Don't rely on prior zero state.
- dequant.h load-time zero/scale fold "may have accuracy issue". Don't enable it by default.
- c3x azp/grouped epilogues are intentionally omitted.
- Affine dequant assumes `group_size` divides `in` (no row-straddling groups).

## COMMANDS
```bash
cargo build -p scratchy-target-cuda --features cuda   # nvcc: tens of minutes, GBs of RAM
CUDA_ARCH=90 cargo build -p scratchy-target-cuda --features cuda   # force sm_90 (FA3, c3x)
python3 crates/targets/cuda/csrc/marlin/generate_w4a16_kernels.py
```
FFI correctness tests live on the Rust side (`../src/quant.rs`, `../src/layers_tests.rs`,
`crates/e2e/tests/e_correctness.rs` for fp8 scaled_mm). There are no tests in csrc.
