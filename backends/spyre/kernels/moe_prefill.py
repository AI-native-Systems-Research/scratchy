# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton mixture-of-experts kernel for a PROMPT — gemma-4's whole sparse block
over T tokens at once, router included, every expert swept densely.

    out[t] = Σ_e  w[t, e] · down_e( act_e( x_e[t] ) )        w[t, e] = 0 off the top-K

THE PREFILL TWIN OF `moe.py`. The decode kernel (`moe.py`) reads one token's K routed
experts by id — a slab gather per pair, the address depending on data. A prompt routes T
tokens to up to T·K distinct experts, so this kernel does what IBM spyre-inference's
`_moe_persistent` does: it builds a DENSE route `[T, E]` (each token's K route weights at
their expert lanes, zero elsewhere) and sweeps ALL E experts, each a plain matmul of the
whole `[T, H]` token tile with that expert's slab, scaled per token by its dense weight. No
address depends on data; every slab is a direct load at a trace-time constant expert.

THE COMPUTATION, IN PROGRAM ORDER:

1. ROUTER — `moe.py`'s router verbatim over T rows: rmsnorm (plain gain), the f16 logits
   `[T, E]`, the round-form top-K (descending, ties to the higher id), a softmax over the K
   at `H^-0.5`, × each selected expert's `per_expert_scale`.
2. DENSE ROUTE — the K route weights scattered to their expert lanes: round j adds its weight
   to lane `ids[:, j]` by a one-hot select. Lanes off the top-K stay exactly 0.
3. EXPERTS — for every expert e: gate/up W8A8 over the `[T, H]` tile, gelu-tanh(gate)·up
   `[T, I]`, down W8A8 back to `[T, H]`, × `dense[:, e]`, accumulated in f16.

THE NUMERICS ARE `moe.py`'s, AT THE SAME GRAIN, so a token computes the same numbers on
either path (a prefix cached by one is reusable by the other):

* gate/up: one fp8 scale per TOKEN row `[1, H]` — the input is the token's own, quantized once
  and read by every expert's two gemms;
* down: one scale per (token, expert) row `[1, I]` — `moe.py`'s per-pair grain (#639): at a
  routed expert the row is the pair's own activation.

For a token NOT routed to e, its row's product is multiplied by an exact 0 and adds nothing.
⛔ THE f16 ACCUMULATION ORDER IS EXPERT-ID ORDER here, slot order in `moe.py`; the two are the
same sum in a different association, so they agree to f16 rounding, not bit for bit.

THE SPLICE'S OWN CONTRACT:

* PARAMETERS, IN ORDER: `moe.py`'s first twelve (router input, gain, projection, per-expert
  scale; expert input; the three fp8 banks and their scales; the output), then the SCRATCH:
  `desc_dense` `[T, E]` the dense route.
* CONSTEXPRS: `T` (the prompt's token rows), `H`, `I`, `E`, `K`; `EPS`, `INV_H`, `TEMP`;
  `INV_QMAX`, `SCALE_FLOOR` — `moe.py`'s, with `T` for `M`.
* GRID: `[1]`.

⛔ EVERY LOOP IS `tl.static_range` (the ladder's law), and the expert sweep is the one that
matters: E iterations of three fp8 gemms each. That is the bake cost this kernel brings, and
whether the card door keeps it unrolled or rolls it into one expert body is the door's
decision, measured, not this file's.
"""

import triton
import triton.language as tl


@triton.jit
def _fp8_scale(amax, INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr, ROWS: tl.constexpr):
    """`quantscalepertokenfp8`'s `FMIN(FMAX(amax · mulConst, clipMin), clipMax)`, one scale per
    row (`amax` is `[ROWS]`). f16 bounds from birth: a bare Python float in `tl.minimum`
    becomes an f32 tensor and widens the scale."""
    lo = tl.full([ROWS], SCALE_FLOOR, tl.float16)
    hi = tl.full([ROWS], 448.0, tl.float16)
    return tl.minimum(tl.maximum(amax * INV_QMAX, lo), hi)


@triton.jit
def _fp8_codes(a, inv, ROWS: tl.constexpr, N: tl.constexpr):
    """`clip(a · (1/s), ±448)` rounded to e4m3, each row at its own scale; widened back to
    f16 for the dot (the codes are the values)."""
    lo = tl.full([ROWS, N], -448.0, tl.float16)
    hi = tl.full([ROWS, N], 448.0, tl.float16)
    q = tl.minimum(tl.maximum(a * inv[:, None], lo), hi)
    return q.to(tl.float8e4nv).to(tl.float16)


@triton.jit
def _w8a8(q, s, w_codes, w_scale, ROWS: tl.constexpr, N: tl.constexpr):
    """`(q @ W.T · s) · w_scale` — the matmulfp8 and its two dequant legs, `s` per row."""
    raw = tl.dot(q, w_codes.T, tl.zeros([ROWS, N], dtype=tl.float16))
    return (raw * s[:, None]) * w_scale


@triton.jit
def _gelu_tanh(g, ROWS: tl.constexpr, N: tl.constexpr):
    """`moe.py`'s gelu: the tanh form through the sigmoid identity, the cubic in the f32
    island; the `1` an f16 tile."""
    g32 = g.to(tl.float32)
    v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
    e = tl.exp(-v).to(tl.float16)
    return tl.fdiv(g, tl.full([ROWS, N], 1.0, tl.float16) + e)


@triton.jit
def moe_prefill_fwd(desc_xr, desc_rg, desc_rw, desc_pes,  #
                    desc_xe, desc_wg, desc_sg, desc_wu, desc_su, desc_wd, desc_sd,  #
                    desc_o,  #
                    desc_dense,  #
                    T: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
                    E: tl.constexpr, K: tl.constexpr,  #
                    EPS: tl.constexpr, INV_H: tl.constexpr, TEMP: tl.constexpr,  #
                    INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    # ── the router: `moe.py`'s, over T rows ───────────────────────────────────
    xr_desc = tl.make_tensor_descriptor(desc_xr, shape=[T, H], strides=[H, 1],
                                        block_shape=[T, H])
    rg_desc = tl.make_tensor_descriptor(desc_rg, shape=[H], strides=[1], block_shape=[H])
    rw_desc = tl.make_tensor_descriptor(desc_rw, shape=[E, H], strides=[H, 1],
                                        block_shape=[E, H])
    pes_desc = tl.make_tensor_descriptor(desc_pes, shape=[E], strides=[1], block_shape=[E])
    xr = xr_desc.load([0, 0])
    ms = tl.sum(xr * xr, 1) * INV_H
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    xn = xr * r[:, None] * rg_desc.load([0])[None, :]
    lg = tl.dot(xn, rw_desc.load([0, 0]).T, tl.zeros([T, E], dtype=tl.float16))

    offs = tl.arange(0, E).to(tl.float16)
    koffs = tl.arange(0, K).to(tl.float16)
    pes = pes_desc.load([0])
    vals = tl.zeros([T, K], dtype=tl.float16)
    idsf = tl.zeros([T, K], dtype=tl.float16)
    psel = tl.zeros([T, K], dtype=tl.float16)
    neg_inf = tl.full([T, E], float("-inf"), tl.float16)
    no_lane = tl.full([T, E], -1.0, tl.float16)
    zero_e = tl.zeros([T, E], dtype=tl.float16)
    # A per-row or per-lane value enters a `[T, E]` select as itself: `tl.where` broadcasts
    # both sides to the condition's shape. Not `x + zero_e` (`moe.py`'s decode spelling,
    # where every tile is one row): at T rows that add of a zero splat is the one shape
    # upstream's `reorder-broadcast` rewrites, and staying out of that pass's reach keeps our
    # make_ttir and theirs one form.
    one_k = tl.full([K], 1.0, tl.float16)
    zero_k = tl.zeros([K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        mx = tl.max(lg, 1).to(tl.float16)
        lanes = tl.where(lg == mx[:, None], offs[None, :], no_lane)
        idx = tl.max(lanes, 1).to(tl.float16)
        sel = offs[None, :] == idx[:, None]
        pj = tl.sum(tl.where(sel, pes[None, :], zero_e), 1)
        oh = tl.where(koffs == j, one_k, zero_k)[None, :]
        vals = vals + mx[:, None] * oh
        idsf = idsf + idx[:, None] * oh
        psel = psel + pj[:, None] * oh
        lg = tl.where(sel, neg_inf, lg)

    # The route weights, `moe.py`'s chain: softmax over the K at `H^-0.5`, × the scales.
    sv = (vals * TEMP).to(tl.float32)
    ex = tl.exp(sv - tl.max(sv, 1)[:, None])
    rw = tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel

    # ── the dense route `[T, E]`: slot j's weight at lane ids[:, j] ──────────
    # Each lane is hit at most once per row (the top-K ids are distinct), so adding a
    # one-hot column into zeros places every weight exactly and leaves every other lane 0.
    dense = tl.zeros([T, E], dtype=tl.float16)
    zero_tk = tl.zeros([T, K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        idj = tl.sum(tl.where(koffs[None, :] == j, idsf, zero_tk), 1)
        wj = tl.sum(tl.where(koffs[None, :] == j, rw, zero_tk), 1)
        dense = dense + tl.where(offs[None, :] == idj[:, None], wj[:, None], zero_e)
    dense_desc = tl.make_tensor_descriptor(desc_dense, shape=[T, E], strides=[E, 1],
                                           block_shape=[T, E])
    dense_desc.store([0, 0], dense)

    # ── the experts, every one, over the whole token tile ─────────────────────
    xe_desc = tl.make_tensor_descriptor(desc_xe, shape=[T, H], strides=[H, 1],
                                        block_shape=[T, H])
    wg_desc = tl.make_tensor_descriptor(desc_wg, shape=[E, I, H], strides=[I * H, H, 1],
                                        block_shape=[1, I, H])
    wu_desc = tl.make_tensor_descriptor(desc_wu, shape=[E, I, H], strides=[I * H, H, 1],
                                        block_shape=[1, I, H])
    wd_desc = tl.make_tensor_descriptor(desc_wd, shape=[E, H, I], strides=[H * I, I, 1],
                                        block_shape=[1, H, I])
    sg_desc = tl.make_tensor_descriptor(desc_sg, shape=[E, I], strides=[I, 1],
                                        block_shape=[1, I])
    su_desc = tl.make_tensor_descriptor(desc_su, shape=[E, I], strides=[I, 1],
                                        block_shape=[1, I])
    sd_desc = tl.make_tensor_descriptor(desc_sd, shape=[E, H], strides=[H, 1],
                                        block_shape=[1, H])
    # The route's column e, one expert's weight for every token: a `[T, 1]` window of the
    # dense route would be 2 bytes wide (under the 16-byte block floor), so each column is
    # read off the whole `[T, E]` tile by a one-hot reduce instead.
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[T, H], strides=[H, 1],
                                       block_shape=[T, H])
    xe = xe_desc.load([0, 0])
    # gate/up: one scale per TOKEN row, quantized once for every expert.
    sx = _fp8_scale(tl.max(tl.abs(xe), 1).to(tl.float16), INV_QMAX, SCALE_FLOOR, T)
    qx = _fp8_codes(xe, tl.fdiv(tl.full([T], 1.0, tl.float16), sx), T, H)
    acc = tl.zeros([T, H], dtype=tl.float16)
    for e in tl.static_range(0, E):
        wg = tl.reshape(wg_desc.load([e, 0, 0]), [I, H]).to(tl.float16)
        wu = tl.reshape(wu_desc.load([e, 0, 0]), [I, H]).to(tl.float16)
        gate = _w8a8(qx, sx, wg, sg_desc.load([e, 0]), T, I)
        up = _w8a8(qx, sx, wu, su_desc.load([e, 0]), T, I)
        act = _gelu_tanh(gate, T, I) * up
        # down: one scale per (token, e) row — `moe.py`'s per-pair grain.
        sa = _fp8_scale(tl.max(tl.abs(act), 1).to(tl.float16), INV_QMAX, SCALE_FLOOR, T)
        qa = _fp8_codes(act, tl.fdiv(tl.full([T], 1.0, tl.float16), sa), T, I)
        wd = tl.reshape(wd_desc.load([e, 0, 0]), [H, I]).to(tl.float16)
        down = _w8a8(qa, sa, wd, sd_desc.load([e, 0]), T, H)
        we = tl.sum(tl.where(offs[None, :] == e, dense, zero_e), 1)
        acc = acc + down * we[:, None]
    o_desc.store([0, 0], acc)
