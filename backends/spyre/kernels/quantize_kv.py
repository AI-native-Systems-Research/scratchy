# SPDX-License-Identifier: Apache-2.0
"""spyre-inference's `quantize_kv` (spyre-inference 4a7d21d, `ops/reshape_and_cache.py:20-29`), in
Triton: a float8 KV cache's new K (or V) saturated and cast, ONE elementwise pass over the whole
tensor, ahead of the per-head stores (`reshape_and_cache_head_major.py`).

Their function, statement for statement: `/ scale` (the model's `KvCacheScales` — `Unscaled` when
its config states no `kv_cache_scheme`, and then no divide
— their `if scale != 1.0`), `x.clamp(-fmax, fmax)` with `fmax = torch.finfo(dtype).max`, and
`.to(dtype)`. torch-spyre lowers the clamp as its two `clip`s (`clamp_min`, `clamp_max`) and the
cast as `spyre.qfp8ch`; here they are `arith.maxnumf` and `arith.minnumf` against splatted bounds
and `arith.truncf` f16 → f8E4M3FN, each lowered by the walk on its own.

THE SPLICE'S OWN CONTRACT (`scratchy_triton_splice::kv_quantize_kernel`):

* PARAMETERS, IN ORDER: `desc_x` (the f16 rows), `desc_o` (the float8 rows).
* CONSTEXPRS: `M`, `N`, `FMAX` — `M` rows of `N` columns, `FMAX` the float8 format's own saturation
  bound.
* GRID: `[1]`.

ONE whole `[M, N]` tile, as their one pass is: the program states no row corner for any door to
address, and the lowering's own work division splits the tensor across the cores, as it does every
elementwise op's.
"""

import triton
import triton.language as tl


@triton.jit
def quantize_kv_fwd(desc_x, desc_o,  #
                    M: tl.constexpr, N: tl.constexpr,  #
                    FMAX: tl.constexpr,  #
                    ):
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N], strides=[N, 1], block_shape=[M, N])
    x = x_desc.load([0, 0])
    # x = x.clamp(-fmax, fmax).to(dtype)
    o_desc.store([0, 0], tl.minimum(tl.maximum(x, -FMAX), FMAX).to(o_desc.dtype))
