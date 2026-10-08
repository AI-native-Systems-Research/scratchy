"""The fp8 W8A8 matmul for Spyre -- ONE projection, weight-side fp8, derived from
`swiglu_mlp.py` (this repo's own already-validated Granite MLP kernel) and from
scratchy's card-proven W8A8 chain.

WHAT THIS KERNEL COMPUTES
=========================
`o[m, n] = (x[m, k] quantized to fp8) @ w_fp8[n, k] * a_scale[m] * w_scale[n]`

The device's fp8 is W8A8: `matmulfp8` multiplies fp8 x fp8 into an f16
accumulator, and the dequant multiplies back `a_scale` (per activation row --
computed ON DEVICE by the quantize chain abs -> amax -> scale -> clamp ->
qfp8ch) and `w_scale` (per output channel -- a checkpoint constant, presented
by the host). The whole chain is scratchy's `ktir_matmul_fp8.rs`, unchanged,
reached through the arity-3 `matmul_oriented` door: a program that binds THREE
inputs (A, W_fp8, w_scale) over a weight view whose `Dtype` is `Fp8E4m3`.

WHY THE KERNEL SPELLS THE DEQUANT AT ALL
========================================
The per-`Program` door lowers from the REGION LIST -- it never inspects the
compute ops (only the B orientation, `matmul_b_orientation`). So the
`* w_scale` in the kernel body is not what makes the device multiply by
w_scale; `matmul_fp8_descriptors` does that whatever the kernel says. What the
spelled mulf buys is that `spyre-dot-to-linalg`'s fp8 contract can VERIFY the
kernel computes what the door assumes: the verifier requires the dot's result
to feed exactly one `arith.mulf` by the w_scale load, which feeds the store. A
kernel that forgot the multiply would be REFUSED there, not silently lowered
with the door's dequant bolted on. (It would also be refused earlier: an
unloaded `desc_ws` parameter is dropped by DCE and
`every_parameter_states_its_width` red-stops on the hole.)

THE DELTA, and the justification for each item
==============================================
Every item is measured or sourced to a definition, in the manner of
`swiglu_mlp.py`'s own delta list.

1. THE WEIGHT POINTER IS `*fp8e4nv` AND THE WIDENING IS SPELLED. OCP E4M3,
   Triton's `fp8e4nv`, the one fp8 variant the device reads. The frontend
   refuses a mixed `tl.dot(f16, fp8)` outright ("Both operands must be same
   dtype", `semantic.py`'s dot), so the weight tile is widened with
   `.to(tl.float16)` before the dot -- the SAME widening every fp8 Triton
   kernel on every backend spells. On this backend the cast is a NO-OP AT THE
   KTIR LEVEL and is folded away: the device widens on load
   (`ktdp.load` over an fp8 view reads f16 values; `TileStorage::Fp8E4m3`),
   and scratchy's own producer types even the fp8 weight load's RESULT F16
   (`KTIR_ELEM`). fp8-ness lives only in the view's `Dtype` attribute, which
   is what the arity-3 door keys on -- so the cast never survives to matter.

2. THE DOT IS f16 x f16 WITH AN f16 ACCUMULATOR, exactly as `swiglu_mlp.py`
   delta 2: device floats are DL16-f16, and the f16 guard in
   `spyre-dot-to-linalg` passes BECAUSE of the spelled widening (operand B is
   the `.to(tl.float16)` result, f16-typed).

3. THE WEIGHT IS n-MAJOR AND THE DOT OPERAND CARRIES `.T` (swiglu delta 13).
   `w` is declared `[N, K]` -- `torch.nn.Linear.weight`'s `[out_features,
   in_features]`, the checkpoint's own layout -- and `w.T` presents the `[k,
   n]` operand the dot wants. `spyre-dot-to-linalg` folds the transpose into
   the indexing maps (transpose-B), and the host stages the bytes into the
   kernel slot's `[k, n]` device order as it does for every presented weight.

4. w_scale IS A THIRD POINTER, NOT A constexpr. It is a real `[1, N]` f16 row
   (per output channel) that the host presents; a constexpr cannot carry a
   per-channel vector. The kernel loads it and multiplies -- one row
   broadcast over the m rows of the product, which is exactly the `In::mb`
   broadcast `matmul_fp8_descriptors`' dequant states for m > 1. The mulf's
   operands are `(dot_result, broadcast(load(ws)))` or the mirror order; both
   spellings appear in Triton kernels and the verifier accepts either.

5. TILES ARRIVE THROUGH DESCRIPTORS, NOT POINTER ARITHMETIC (swiglu delta 3),
   and M/N/K/BLOCK_* are constexpr (swiglu delta 4). Same grounds, same
   refusals.

6. ONE DOT, ONE LOAD PER OPERAND, ONE STORE -- the whole-kernel contract
   `verify_canonical_matmul_kernel` demands (review P1.0): the multicore
   matmul emitter is a shape-keyed TEMPLATE that derives everything from the
   descriptor M/N/K and silently drops anything else, so the fp8 arm admits
   EXACTLY {1 tt.dot, 3 tt.descriptor_load (x, w, ws), 1 arith.mulf, 1
   tt.descriptor_store} and refuses every deviation by name. No K-loop here:
   BLOCK_K == K makes the contraction single-tile, matching the smallest
   swiglu configuration. A K-looped fp8 form is a follow-on, not this
   fixture's claim.

THE HOST CONTRACT
=================
Four buffers, positionally bound (A=arg0, W=arg1, ws=arg2, O=arg3):
  * `x`  : f16, `[M, K]` row-major.
  * `w`  : fp8e4m3 PACKED (1 byte/elem), `[N, K]` = the checkpoint weight,
           staged to the kernel slot's 128-stick fp8 device order.
  * `ws` : f16, `[1, N]` (one row).
  * `o`  : f16, `[M, N]` row-major.
The torch reference for scoring is torch-spyre's `quantize_fp8_with_scale`
(`decompositions.py:966`): THAT function is the spec -- the device's
activation quantize (abs -> amax -> scale -> clamp -> qfp8ch) is its
decomposition, and the score tolerance must be derived, not assumed.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_fp8_fwd(desc_x, desc_w, desc_ws, desc_o,  #
                   M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,  #
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    # Descriptors, not pointer blocks (swiglu delta 3). The weight's descriptor
    # elem is fp8e4nv (delta 1); its shape is the checkpoint's `[out, in]`.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, K],
                                       strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[N, K],
                                       strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    ws_desc = tl.make_tensor_descriptor(desc_ws, shape=[1, N],
                                        strides=[N, 1],
                                        block_shape=[1, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, N],
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


# --- the Spyre configuration ---------------------------------------------------

SIGNATURE = {
    "desc_x": "*fp16", "desc_w": "*fp8e4nv", "desc_ws": "*fp16",
    "desc_o": "*fp16",
    "M": "constexpr", "K": "constexpr", "N": "constexpr",
    "BLOCK_M": "constexpr", "BLOCK_K": "constexpr", "BLOCK_N": "constexpr",
}


def constexprs(m=64, k=128, n=128, block_m=64, block_k=None, block_n=128):
    """One configuration. `block_k=None` means "do not tile K" (delta 6).

    `n == block_n` by default because the single-tile contract requires it: the
    W8A8 door emits ONE M x N program and discards the block size, so a
    multi-block N (or M) is refused by the verifier rather than collapsed.
    The small twin mirrors `swiglu_mlp_small`'s shape family (d_model=128,
    d_ff=256) so the two fixtures' extents are comparable. Granite's qkv
    projection is `k=4096, n=12288` (2x128+4096 fused qkv), its output
    projection `k=4096, n=4096`.
    """
    return {"M": m, "K": k, "N": n,
            "BLOCK_M": block_m, "BLOCK_K": k if block_k is None else block_k,
            "BLOCK_N": block_n}
