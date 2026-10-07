# SPDX-License-Identifier: Apache-2.0
"""Granite's RMSNorm -- NOT LayerNorm.

    x * rsqrt(mean(x**2) + eps) * weight

`GraniteRMSNorm` is `LlamaRMSNorm`, whose forward is exactly:

    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states

THERE IS NO MEAN SUBTRACTION AND NO BIAS. That is the whole difference from LayerNorm and
it is the reason this is three ops rather than five: no `mean(x)`, no `x - mean`, no `+ b`.
`rms_norm_eps = 1e-05` in `ibm-granite/granite-3.3-8b-instruct`'s config.json (read from
`crates/models/arch/configs/granite/granite-3.3-8b-instruct.json`, not recalled).

THE DECOMPOSITION IS DELIBERATELY THE OBVIOUS ONE. The backend carries RMSNorm as three
templated ops -- `EXX2_ZEROMEAN` (the mean of squares, zero-mean being exactly the "no
subtraction" above) then `RSQRT` then a multiply -- so a clever in-kernel rearrangement
would be a DIFFERENT algorithm for the emitter to recognise and a different one for a
tolerance to be derived against. One reduce, one reciprocal-sqrt, two multiplies, in that
order.

THE DELTA, and the justification for each item
==============================================

1. THE REDUCTION AND THE TRANSCENDENTAL ARE f16, WITH AN f32 ISLAND AROUND THE `rsqrt`
   ONLY. This is `swiglu_mlp.py`'s delta 5 verbatim, for the same reason and with the same
   shape: `tl.rsqrt` is decorated `@_check_dtype(dtypes=["fp32", "fp64"])`, so calling it on
   an f16 tensor does not lower at all -- MEASURED, the refusal is

       ValueError: Expected dtype ['fp32', 'fp64'] but got fp16

   so the widen / call / truncate island is written out. `LegalizeTypes` collapses that
   shape; genuine f32 compute it red-stops (delta 2 of `attention_flash.py`).

2. THE MEAN IS A SUM TIMES A CONSTEXPR RECIPROCAL, not a divide. `tl.sum(...) / D_MODEL`
   would be a bare f16 divide, which Triton's `computation_type_impl` promotes to f32 for a
   reason that is a PTX fact and false on this device (see `src/target.rs` in the Rust front
   end). `1.0 / D_MODEL` is constexpr-folded in Python before any IR exists, so `INV_D`
   reaches the kernel as one number and the mean is one multiply. `tl.fdiv` would also
   dodge the promotion; a compile-time reciprocal is one op instead of two.

3. THE WEIGHT IS A 1D TILE BROADCAST ALONG THE ROWS (`w[None, :]`), which is HF's own
   shape: `nn.Parameter(torch.ones(hidden_size))`. The broadcast is LANE-INVARIANT per row
   and stick-wide along D_MODEL, so on the device it is the same kind of operand as
   `attention_flash.py`'s delta 8 splat sources rather than a direct read. That is a
   LOWERING and HOST-LAYOUT question, decided by the fold law recorded there, and it is NOT
   re-derived here: this fixture's contract is the TTIR.

4. M AND D_MODEL ARE constexpr (SPYRE-SPECIFIC), for `attention_flash.py`'s delta 7 reason.

5. THE SUM OF D_MODEL SQUARES ACCUMULATES IN f32, AND THE ISLAND IS LOAD-BEARING — it was
   an f16 reduce when this fixture landed, which the splice review correctly rejected as a
   correctness bug, not a headroom footnote: f16's maximum is 65504, so 4096 squares whose
   RMS is 4 already sum to ~65500, and ONE element ≥ 256 overflows `x * x` on its own at
   any width. Residual-stream outliers make both reachable in a real model. The shipped
   kernel (`crates/targets/spyre/kernels/rmsnorm.py`, the splice's row) widens the squares
   before the reduce (`(x * x).to(tl.float32)`), matching the accumulator precision the
   builder's own program states; this fixture records the same shape so the ladder's
   TTIR contract and the splice's kernel agree. The transcendental stays an f16-in/f16-out
   island around `tl.rsqrt` (delta 1) — that one is `@_check_dtype`'s requirement, not a
   range decision.

   The two mitigations recorded when the sum was f16 (reciprocal folded into the reduce;
   tile pre-scale) are both SUPERSEDED by the f32 accumulator — the range question they
   answered no longer exists.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd(desc_x, desc_w, desc_o,  #
                M: tl.constexpr, D_MODEL: tl.constexpr,  # SPYRE: delta 4
                BLOCK_M: tl.constexpr,  #
                EPS: tl.constexpr,  #
                INV_D: tl.constexpr,  # SPYRE: delta 2
                ):
    start_m = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])
    # HF's weight is 1D (`nn.Parameter(torch.ones(hidden_size))`), so the descriptor is too.
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[D_MODEL], strides=[1],
                                       block_shape=[D_MODEL])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])

    offs_m = start_m * BLOCK_M
    x = x_desc.load([offs_m, 0])
    # EXX2_ZEROMEAN: the mean of squares, with NO mean subtracted. One reduce along the
    # hidden axis, in the f32 accumulator (delta 5), one multiply by the constexpr
    # reciprocal (delta 2).
    ms = tl.sum((x * x).to(tl.float32), 1) * INV_D
    # RSQRT, inside the f32 island `tl.rsqrt`'s own dtype check forces (delta 1).
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    w = w_desc.load([0])
    # `x * r * weight`. HF spells the last multiply `self.weight * hidden_states`; the same
    # two operands in the other order, which for one IEEE multiply is the same result.
    o_desc.store([offs_m, 0], x * r[:, None] * w[None, :])


# --- the Spyre configuration ---------------------------------------------------
# Granite-3.3 8B, from its config.json: hidden_size 4096, rms_norm_eps 1e-05.
GRANITE = dict(d_model=4096)
RMS_NORM_EPS = 1e-05
BLOCK_M = 64

SIGNATURE = {
    "desc_x": "*fp16", "desc_w": "*fp16", "desc_o": "*fp16",
    "M": "constexpr", "D_MODEL": "constexpr", "BLOCK_M": "constexpr",
    "EPS": "constexpr", "INV_D": "constexpr",
}


def constexprs(m=64, d_model=128, eps=RMS_NORM_EPS):
    """One configuration. `INV_D` is folded here, on the host, so the kernel never divides."""
    return {"M": m, "D_MODEL": d_model, "BLOCK_M": BLOCK_M,
            "EPS": eps, "INV_D": 1.0 / d_model}


def inputs(seed: int = 0, m=64, d_model=128):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, d_model, generator=g, dtype=torch.float16)
    w = torch.randn(d_model, generator=g, dtype=torch.float16)
    return x, w


def reference(x: torch.Tensor, w: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """Native PyTorch reference: `LlamaRMSNorm.forward`, which is `GraniteRMSNorm`'s.

    Computed in f32 and cast back exactly as HF does -- so the comparison against the
    device's f16 reduce also measures delta 5's dynamic-range question rather than hiding it.
    """
    dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (w.to(torch.float32) * h).to(dtype)
