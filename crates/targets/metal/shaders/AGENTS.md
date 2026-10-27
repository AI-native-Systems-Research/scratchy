# shaders/

## OVERVIEW
MSL kernels for the metal backend: 39 `.metal` + 11 `.h` (~21k LOC), plus vendored MLX headers in `mlx_quantized/`, `mlx_steel_attn/`, `mlx_steel_gemm/`. Score ~20 (50 files, its own toolchain and two compile regimes).

## TWO COMPILE REGIMES (split by `#include "baked.h"`)
| Regime | Compiled by | Specialized by |
|--------|-------------|----------------|
| Function-constant | `../build.rs`, one `.metallib` per `.metal` into `OUT_DIR` | `[[function_constant(N)]]` at pipeline build; indices documented in `src/interpreter/metal/pipelines.rs` module doc |
| Baked | `src/aot.rs` bake, once per distinct constant set | `SCRATCHY_CONSTANT_<slot>` macros; each kernel of a batch in its own `SCRATCHY_NS` namespace |

- Baked shaders must NOT declare `[[function_constant]]`. `build.rs::refuse_function_constants` asserts this.
- `baked.h` is the contract: `SCRATCHY_CONSTANT`, `SCRATCHY_CONSTANT_OPTIONAL`, `SCRATCHY_COMPILES`, `SCRATCHY_KERNEL(sym, fn)`.
- Headers that declare constants carry NO include guard (they're re-read per kernel in a batch). Every other header is `#pragma once`.

## WHERE TO LOOK
| Family | Files |
|--------|-------|
| Attention: decode + paged prefill, rope fold, sinks | `attention.metal` (3.2k), `attention_steel_*paged.metal` -> `mlx_steel_attn/*_kernel.h` |
| Quant GEMM/GEMV | `quantized_qmv.metal`, `quantized_qmm{,_nax}.metal`, `quantized_qvm.metal`, `quantized_dequantize.metal`, `metal_kittens.h` |
| NAX (M5 MMA) tiles | `metal_nax.h` (vendored, `static_assert`-dense) |
| GDN / Mamba2 | `gdn_*.metal`, `gdn_state.h` (prefix-cache snapshot ring) |
| MoE | `moe_*.metal`, `moe_route.h` |
| Sampling | `sampling.metal`, `argmax.metal`, `argpartition.metal` |
| KV write / turboquant | `kv_writer.h`, `turboquant*.{metal,h}` |

## CONVENTIONS
- Baked slot numbers in each file header MUST match the typed `*Constants` structs in `src/tape/kernel_constants.rs`. Slot drift is the top hazard.
- Rust names each kernel through a `KernelId` / `MetalKernel` ZST (`src/tape/kernel_identity.rs`). A new kernel needs both.
- Vendored MLX code stays verbatim, with upstream file:line provenance in comments. Deviations are documented in the header doc, never edited silently.
- The paged steel kernels take model params as baked constants 0..5 (`ATTN_PAGED_*`), not a params struct.

## ANTI-PATTERNS
- V is never roped and never mutated, and is always read from the cache. The paged cache is a position-independent reuse artifact.
- Rope K once into scratch (`rope_once_*`). Never re-rope inside the per-q-tile flash loop. With ROPE_ON_READ unset, buffer 8 is unbound and the output must stay byte-identical.
- gpt-oss sinks are added UNSCALED (never times `sm_scale`). Sinkless builds stay byte-identical.
- Tail tiles never read past the last row or deref unallocated pages. Bounds are baked, not runtime-checked. A loader row never straddles a page (`TROWS == BROWS`).
- `gated_act.h`: an unknown activation tag is a build-time refusal, never a fallback.
- `activation.metal`: the two scale kinds are distinct. Never conflate them.
- GDN checkpoints advance only their own rows. A replayed (negative) row never stores.
