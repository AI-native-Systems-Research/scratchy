"""Scaled dot-product attention, PORTED FROM TORCH-SPYRE'S SDPA DECOMPOSITION.

Source: torch-spyre `torch_spyre/_inductor/decompositions.py` at a9fa997b (2026-10-29),
`spyre__sdpa_overrideable` (:1960) -- the decomposition torch-spyre's Inductor backend
lowers every `aten._scaled_dot_product_fused_attention_overrideable` to. Nothing here is
derived from Triton's fused-attention tutorial (that is `attention_flash.py`) or from
scratchy's hand-built builder arms (that is `crates/targets/spyre/kernels/attn.py`). Each
construct below names the torch-spyre line it restates.

THE ALGORITHM, as torch-spyre states it
=======================================

torch-spyre nests five `for_each_tile` levels: batch -> KV head -> query head within the
GQA group -> query rows (all MAPS: independent tiles, `out_dim` set) -> KV rows (a SCAN:
`init` set, carry threaded). Per (batch, head, query tile):

  1. q_scaled = q * scale                                    (:2175)
     ONCE, before the KV loop -- "Q is invariant across the counted Lk loop". The score
     tile is never scaled; Q is. `scale` defaults to 1/sqrt(head_dim) (:1989).
  2a. ONE KV block (:2178-2200) -- the stable softmax, no carry:
        s = q_scaled @ k^T + mask ; m = amax(s) ; p = exp(s - m)
        out = (p @ v) / sum(p)
  2b. SEVERAL KV blocks (:2204-2270) -- the online softmax. Carry
      (running max = -inf, denominator = 0, output = 0) and per block:
        s          = q_scaled @ k_blk^T + mask_blk
        block_max  = amax(s)
        correction = exp(min(running_max - block_max, 0))    (:2246)
        new_max    = maximum(running_max, block_max)
        p          = exp(s - new_max)
        denom      = denom * correction + sum(p)
        output     = output * correction + p @ v_blk          (:2256, matmul THEN add)
      and after the last block: output / denom. `correction` is computed BEFORE
      `new_max` -- it equals exp(running_max - new_max), but spelled with the clamp it
      reads only the OLD max and the BLOCK max, so the running max updates in place.
  3. Masks are ADDITIVE tiles sliced with the scores (:2191, :2240). `is_causal`
     becomes `spyre::causal_mask` (customops.py:1130): a host-built [Lq, Lk] tile,
     tril -> 0, above the diagonal -> -inf, transferred once. Nothing is compared on
     the device.
  4. GQA keeps K/V at Hkv (:2012-2025): "Keep the GQA relationship explicit instead of
     repeating K/V up to Hq." No K/V copy exists; the query head reads its KV head.
  5. Every intermediate is in the query's dtype -- f16 scores, f16 exp, f16 running
     max/denominator/output. No f32 accumulator anywhere in the decomposition.

THE TRITON SPELLING, and the only places it differs in form
===========================================================

* THE MAP LEVELS ARE THE GRID. torch-spyre's batch/head/group/query maps are
  independent tiles; a Triton program is one independent tile. pid 0 = query block,
  pid 1 = query head, pid 2 = batch. A program holds ONE query head (torch-spyre's
  `group_tile_size == 1`, the tiling its planner picks for decode; see `inputs()`).
* THE SCAN IS `tl.range`, with (m_i, l_i, acc) as its loop carry -- the scan's own
  carry triple, initialised as torch-spyre initialises it (-inf, 0, 0).
* THE ONE-BLOCK BODY IS A constexpr ARM. `num_kv_tiles == 1` is a trace-time fact in
  torch-spyre (the tiling is static), and `SKV == BLOCK_N` is one here.
* `tl.exp` TAKES f32 (Triton's `@_check_dtype`), so each exp is widened and truncated
  straight back: torch-spyre's f16 `torch.exp`, spelled as the island LegalizeTypes
  collapses. `tl.sum` likewise comes back to f16 explicitly. No other widening.
* THE LAYOUT IS SCRATCHY'S: q/o are `[Z*SQ, NQH*HD]` (token-major, heads in columns),
  k/v are `[Z*SKV, NKVH*HD]` -- the physical [B, S, H, D] storage torch-spyre's own
  comment says SDPA "routinely receives" (:1994), read through [BLOCK, HD] descriptor
  tiles at column `head * HD`. No transposed plane, no splat-wide staging.
* `SCALE` IS A constexpr, because a compute operand must be a compile-time constant on
  this target (`attention_flash.py` delta 9). torch-spyre's `query_scale` is a Python
  float at trace time too.

WHAT IS NOT PORTED HERE: torch-spyre's tiling COST MODEL (`_select_sdpa_tiling`, :764),
which chooses `kv_block_size` and the map tile sizes. BLOCK_M / BLOCK_N arrive as
constexprs; `inputs()` records the plan torch-spyre's planner chooses for each
configuration's shape, so each configuration runs a tiling torch-spyre itself would run.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sdpa_fwd(desc_q, desc_k, desc_v, desc_o, desc_mask,  #
             Z: tl.constexpr, NQH: tl.constexpr, NKVH: tl.constexpr,  #
             HD: tl.constexpr, SQ: tl.constexpr, SKV: tl.constexpr,  #
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
             SCALE: tl.constexpr, HAS_MASK: tl.constexpr):
    # The map levels (batch -> head -> group -> query, :2272-2317), as the grid.
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    # GQA without repeating K/V (:2012): the query head reads KV head h // G.
    G: tl.constexpr = NQH // NKVH
    kvh = h // G

    q_width: tl.constexpr = NQH * HD
    kv_width: tl.constexpr = NKVH * HD
    q_desc = tl.make_tensor_descriptor(desc_q, shape=[Z * SQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[BLOCK_M, HD])
    k_desc = tl.make_tensor_descriptor(desc_k, shape=[Z * SKV, kv_width],
                                       strides=[kv_width, 1],
                                       block_shape=[BLOCK_N, HD])
    v_desc = tl.make_tensor_descriptor(desc_v, shape=[Z * SKV, kv_width],
                                       strides=[kv_width, 1],
                                       block_shape=[BLOCK_N, HD])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[Z * SQ, q_width],
                                       strides=[q_width, 1],
                                       block_shape=[BLOCK_M, HD])
    # The additive mask, [SQ, SKV], broadcast over batch and heads exactly as
    # `spyre::causal_mask`'s [1, 1, Lq, Lk] is (:2032). Built only when a load reads it:
    # an unread descriptor leaves its parameter addressed nowhere (attention_flash.py
    # delta 11), and the no-mask entry point below passes a placeholder in this slot.
    if HAS_MASK:
        mask_desc = tl.make_tensor_descriptor(desc_mask, shape=[SQ, SKV],
                                              strides=[SKV, 1],
                                              block_shape=[BLOCK_M, BLOCK_N])

    q_row = z * SQ + start_m * BLOCK_M
    kv_row = z * SKV
    q = q_desc.load([q_row, h * HD])  # [BLOCK_M, HD]
    # :2175 -- scale Q once, outside the KV loop.
    q_scaled = q * SCALE

    if SKV == BLOCK_N:
        # :2178-2200 -- one KV block: the stable softmax, no carry.
        k = k_desc.load([kv_row, kvh * HD])  # [BLOCK_N, HD]
        scores = tl.dot(q_scaled, k.T, out_dtype=tl.float16)  # [BLOCK_M, BLOCK_N]
        if HAS_MASK:
            scores = scores + mask_desc.load([start_m * BLOCK_M, 0])
        block_max = tl.max(scores, 1)
        exp_scores = tl.exp((scores - block_max[:, None]).to(tl.float32)).to(tl.float16)
        denominator = tl.sum(exp_scores, 1).to(tl.float16)
        v = v_desc.load([kv_row, kvh * HD])  # [BLOCK_N, HD]
        output = tl.dot(exp_scores, v, out_dtype=tl.float16)  # [BLOCK_M, HD]
        output = output / denominator[:, None]
    else:
        # :2204-2218 -- the scan's carry: running max -inf, denominator 0, output 0.
        running_max = tl.full([BLOCK_M], float("-inf"), tl.float16)
        denominator = tl.full([BLOCK_M], 0.0, tl.float16)
        output = tl.full([BLOCK_M, HD], 0.0, tl.float16)
        for start_n in tl.range(0, SKV, BLOCK_N):
            # :2220 `sdpa_lk_body`, line for line.
            k = k_desc.load([kv_row + start_n, kvh * HD])  # [BLOCK_N, HD]
            scores = tl.dot(q_scaled, k.T, out_dtype=tl.float16)  # [BLOCK_M, BLOCK_N]
            if HAS_MASK:
                scores = scores + mask_desc.load([start_m * BLOCK_M, start_n])
            block_max = tl.max(scores, 1)
            # :2246 -- the correction reads the OLD max and the BLOCK max only.
            correction = tl.exp(
                tl.minimum(running_max - block_max, 0.0).to(tl.float32)).to(tl.float16)
            new_max = tl.maximum(running_max, block_max)
            exp_scores = tl.exp((scores - new_max[:, None]).to(tl.float32)).to(tl.float16)
            denominator = denominator * correction + tl.sum(exp_scores, 1).to(tl.float16)
            v = v_desc.load([kv_row + start_n, kvh * HD])  # [BLOCK_N, HD]
            weighted = tl.dot(exp_scores, v, out_dtype=tl.float16)  # [BLOCK_M, HD]
            # :2256 -- the matmul, THEN the rescaled add (not an accumulating dot).
            output = output * correction[:, None] + weighted
            running_max = new_max
        # :2270
        output = output / denominator[:, None]
    o_desc.store([q_row, h * HD], output)


@triton.jit
def sdpa_fwd_nomask(desc_q, desc_k, desc_v, desc_o,  #
                    Z: tl.constexpr, NQH: tl.constexpr, NKVH: tl.constexpr,  #
                    HD: tl.constexpr, SQ: tl.constexpr, SKV: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    SCALE: tl.constexpr):
    """The FOUR-pointer entry point: no `attn_bias`, not causal -- torch-spyre's
    `masks == ()` (:2066), the decode shape. A launch binds four buffers, so this is the
    kernel whose signature says so; `desc_q` stands in the mask slot and `HAS_MASK=False`
    never turns it into a descriptor (`attention_flash.py` delta 11). `inline_calls`
    inlines the call, so the ops are `sdpa_fwd`'s own."""
    sdpa_fwd(desc_q, desc_k, desc_v, desc_o, desc_q,  #
             Z, NQH, NKVH, HD, SQ, SKV, BLOCK_M, BLOCK_N, SCALE, False)


# ---------------------------------------------------------------------------------------
# torch side: the stimulus, the mask torch-spyre builds, and the reference.


def causal_mask(sq, skv):
    """`spyre::causal_mask` (customops.py:1130), verbatim: tril -> 0.0 (keep), the rest
    -> -inf, in the query dtype. Query i attends keys 0..i (upper-left aligned)."""
    keep = torch.tril(torch.ones(sq, skv, dtype=torch.bool))
    mask = torch.zeros(sq, skv, dtype=torch.float16)
    mask.masked_fill_(~keep, float("-inf"))
    return mask


def inputs(seed=0, z=1, nqh=8, nkvh=2, hd=64, sq=256, skv=256, causal=True):
    """Unit-variance f16 q, k, v in the kernel's layout ([Z*S, H*HD], token-major), plus
    the causal mask when `causal`. Returns (q, k, v, mask-or-None)."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(z * sq, nqh * hd, generator=g, dtype=torch.float32).to(torch.float16)
    k = torch.randn(z * skv, nkvh * hd, generator=g, dtype=torch.float32).to(torch.float16)
    v = torch.randn(z * skv, nkvh * hd, generator=g, dtype=torch.float32).to(torch.float16)
    return q, k, v, (causal_mask(sq, skv) if causal else None)


def _bhsd(x, z, s, h, hd):
    """[Z*S, H*HD] -> torch SDPA's logical [Z, H, S, HD]."""
    return x.reshape(z, s, h, hd).permute(0, 2, 1, 3)


def reference(q, k, v, z, nqh, nkvh, hd, sq, skv, scale, causal):
    """torch's own `scaled_dot_product_attention`, in f32 on the f16 inputs, cast once to
    f16 -- the function torch-spyre's decomposition implements, NOT a restatement of it.
    `is_causal` is torch's (upper-left tril), so the reference never reads the mask bytes
    the kernel is handed. GQA via `enable_gqa` (torch broadcasts K/V over each group)."""
    qf = _bhsd(q.float(), z, sq, nqh, hd)
    kf = _bhsd(k.float(), z, skv, nkvh, hd)
    vf = _bhsd(v.float(), z, skv, nkvh, hd)
    o = torch.nn.functional.scaled_dot_product_attention(
        qf, kf, vf, is_causal=causal, scale=scale, enable_gqa=(nqh != nkvh))
    return o.permute(0, 2, 1, 3).reshape(z * sq, nqh * hd).to(torch.float16)


def torch_spyre_decomposition(q, k, v, mask, z, nqh, nkvh, hd, sq, skv, scale, block_n):
    """torch-spyre's `kv_level` (decompositions.py:2173-2270) transliterated onto CPU
    tensors, in f16, at the kernel's KV block size: the SAME ALGORITHM the kernel runs,
    executed by torch, so `_selfcheck` can show the port's arithmetic and the reference
    agree before any compiler is involved."""
    g = nqh // nkvh
    qb = _bhsd(q, z, sq, nqh, hd)
    kb = _bhsd(k, z, skv, nkvh, hd).repeat_interleave(g, dim=1)
    vb = _bhsd(v, z, skv, nkvh, hd).repeat_interleave(g, dim=1)
    q_scaled = qb * scale
    num_kv_tiles = skv // block_n

    def scores_of(kt, t0):
        s = (q_scaled.float() @ kt.transpose(-1, -2).float()).to(torch.float16)
        if mask is not None:
            s = s + mask[:, t0:t0 + kt.shape[-2]]
        return s

    if num_kv_tiles == 1:
        s = scores_of(kb, 0)
        p = torch.exp(s - torch.amax(s, dim=-1).unsqueeze(-1))
        out = (p.float() @ vb.float()).to(torch.float16) / p.sum(dim=-1).unsqueeze(-1)
    else:
        running_max = torch.full(qb.shape[:-1], float("-inf"), dtype=torch.float16)
        denominator = torch.zeros(qb.shape[:-1], dtype=torch.float16)
        out = torch.zeros_like(qb)
        for t in range(num_kv_tiles):
            t0 = t * block_n
            s = scores_of(kb[..., t0:t0 + block_n, :], t0)
            block_max = torch.amax(s, dim=-1)
            correction = torch.exp(torch.clamp_max(running_max - block_max, 0.0))
            new_max = torch.maximum(running_max, block_max)
            p = torch.exp(s - new_max.unsqueeze(-1))
            denominator = denominator * correction + p.sum(dim=-1)
            weighted = (p.float() @ vb[..., t0:t0 + block_n, :].float()).to(torch.float16)
            out = out * correction.unsqueeze(-1) + weighted
            running_max = new_max
        out = out / denominator.unsqueeze(-1)
    return out.permute(0, 2, 1, 3).reshape(z * sq, nqh * hd)


def check_decomposition(q, k, v, mask, ref, z, nqh, nkvh, hd, sq, skv, scale, block_n):
    """The port's own arithmetic against torch's SDPA at one configuration: the transliterated
    decomposition (f16 throughout, at the configuration's KV block) must land within a few f16
    rounds of the f32 reference, or the reference -- not the compiler -- is wrong. The 2e-2
    ceiling is ~40 f16 half-ULPs at unit scale: a sanity gate for the torch side, not the
    harness's derived bound. Returns max|decomposition - reference|."""
    dec = torch_spyre_decomposition(q, k, v, mask, z, nqh, nkvh, hd, sq, skv, scale, block_n)
    err = (dec.float() - ref.float()).abs().max().item()
    assert err < 2e-2, (z, nqh, nkvh, hd, sq, skv, block_n, mask is not None, err)
    return err


def _selfcheck():
    """`check_decomposition` at the three configurations the numeric harness stages (the
    generator runs it per configuration as it writes the data)."""
    for (z, nqh, nkvh, hd, sq, skv, block_n, causal) in [
        (1, 8, 2, 64, 256, 256, 256, True),
        (1, 8, 2, 64, 256, 256, 128, True),
        (2, 8, 2, 64, 1, 512, 256, False),
    ]:
        scale = hd ** -0.5
        q, k, v, mask = inputs(0, z, nqh, nkvh, hd, sq, skv, causal)
        ref = reference(q, k, v, z, nqh, nkvh, hd, sq, skv, scale, causal)
        err = check_decomposition(q, k, v, mask, ref, z, nqh, nkvh, hd, sq, skv, scale,
                                  block_n)
        print(f"selfcheck z={z} nqh={nqh} nkvh={nkvh} sq={sq} skv={skv} "
              f"block_n={block_n} causal={causal}: max|dec - ref| = {err:.3e}")


if __name__ == "__main__":
    _selfcheck()
