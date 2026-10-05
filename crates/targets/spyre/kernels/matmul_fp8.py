# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton fp8 W8A8 matmul kernel — the weight-side fp8 half.

BODY PROVENANCE: `crates/triton/test-fixtures/matmul_fp8.py` VERBATIM — the
contract-verified fixture (`triton-ktir-superdsc/tests/matmul_fp8_contract.rs`
pins its bake to exactly {1 tt.dot, 3 tt.descriptor_load, 1 arith.mulf, 1
tt.descriptor_store} with the fused `fq_afp8_op`/`fq_dqw_op` chain present).
Only this header is new.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `desc_x`, `desc_w`, `desc_ws`, `desc_o` — the node's
  operand order (activation, fp8 weight, per-channel scale) then the output,
  arity 3. The registry row matches `GemmWeight::Fp8Dynamic`; the DENSE
  arity-2 form is `matmul.py`'s row.
* CONSTEXPRS: `M`, `K`, `N`, `BLOCK_M`, `BLOCK_K`, `BLOCK_N` — stated by the
  splice from the node's own regions, all blocks the whole extents (ONE tile;
  `verify_canonical_fp8_matmul_kernel` refuses `BLOCK_K < K` and `BLOCK_N < N`).
* GRID: `[1]`.

WHAT THE DEVICE COMPUTES vs WHAT THE PROGRAM COMPUTES: the program spells the
weight-side dequant (`p * ws`) and the emulator executes exactly that —
`o[m, n] = Σ_k a[m,k] · e4m3(w[n,k]) · ws[n]` (the fp8 view widens on read).
The CARD's W8A8 additionally quantizes the activation per token and applies
`a_scale[m] · w_scale[n]`; that chain is emitted by the DOOR
(`ktir_matmul_fp8::matmul_fp8_descriptors`, unchanged) from the weight view's
`is_fp8` + arity-3 — downstream of both producers — and its cross-node dedup
(the `quantized` set the door's caller threads, bundle-wide) is likewise the
door's, not the splice's. The spelled `mulf` is therefore what the ladder's
fp8 verifier REQUIRES of the kernel, so a kernel that forgot it is refused at
compile time rather than silently reinterpreted.

THE ORIENTATION: same law as `matmul.py` — the weight descriptor is the
checkpoint's own on-disk `[n, k]` buffer (fp8 bytes stay 1-byte verbatim on
the emulator path, `spyre_load.rs`), loaded and widened `.to(tl.float16)`
then `.T`, which `dot_to_linalg`'s fp8 arm folds into the transpose-B
indexing maps — the exact maps `KtirFunc::matmul_fp8` states over the same
bytes.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_fp8_fwd(desc_x, desc_w, desc_ws, desc_o,  #
                   M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,  #
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
                   M_TOTAL: tl.constexpr):
    start_m = tl.program_id(0)
    # Descriptors, not pointer blocks (swiglu delta 3). The weight's descriptor
    # elem is fp8e4nv (delta 1); its shape is the checkpoint's `[out, in]`.
    # ⛔ THE DESCRIPTOR NAMES THE STORAGE, THE STORE NAMES THE WINDOW — see
    # `matmul.py`'s `M_TOTAL` law: the out descriptor's shape is the output
    # TENSOR's `[M_TOTAL, N]`, while the store takes the `[M, N]` tile at row 0
    # (the prefill lm-head fold's m=1 tail over the LAST_HIDDEN staging).
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, K],
                                       strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[N, K],
                                       strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    ws_desc = tl.make_tensor_descriptor(desc_ws, shape=[1, N],
                                        strides=[N, 1],
                                        block_shape=[1, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M_TOTAL, N],
                                       strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
    # delta 1: the widening is spelled because the frontend refuses a mixed
    # dot; at the KTIR level it is a no-op folded away before the emitter.
    # delta 3: `.T` presents the [k, n] operand; the fold states transpose-B.
    w = w_desc.load([0, 0]).to(tl.float16)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)  # delta 2: f16 acc
    p = tl.dot(x, w.T, acc)

    # delta 4: the per-channel scale, one row broadcast over m product rows.
    ws = ws_desc.load([0, 0])
    o_desc.store([offs_m, 0], p * ws)
