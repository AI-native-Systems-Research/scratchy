# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton mixture-of-experts kernel — gemma-4's whole sparse block, router
included, as ONE program.

    out[m] = Σ_j  w[m, j] · down_e( act_e( x_e[m] ) )        e = ids[m, j]

THE COMPUTATION, IN PROGRAM ORDER:

1. ROUTER NORM — `rmsnorm(x_r, router.scale)`, gemma's plain-gain convention (no `+1`),
   over the UN-normed residual (the DSL's `gemma_moe(hidden_states, h2, …)` routes off its
   first argument).
2. ROUTER LOGITS — `xn @ router.proj.T`, the dense `[E, H]` projection, f16.
3. TOP-K — K rounds over the whole `[M, E]` row: the row maximum, the highest expert id
   among the lanes equal to it, then that lane masked out. The selection `topk.ddl`'s
   ripple computes (`routetopk.py`): descending by value, and among equal values the later
   scan position first. Slot 0 the maximum.
4. ROUTE WEIGHTS — the K selected logits × `H^-0.5`, a softmax over the K, then × each
   selected expert's `router.per_expert_scale`, read BY ITS ID.
5. EXPERTS — per routed pair, the gate/up/down slabs selected BY ID out of the stacked
   `[E, ·, ·]` fp8 banks (`W.load([e, 0, 0])`, the `_p_matmul.py:423` slab gather), at
   the card's W8A8 numerics, `gelu_tanh(gate) · up` between.
6. COMBINE — `Σ_j w[m, j] · down_j`, accumulated in f16 (the card's combine legs).

THE NUMERICS ARE THE CARD'S, SPELLED HERE. A W8A8 projection on the card is
`fp8_quantize_descriptors` + `matmulfp8` + two dequant legs
(`ktir_matmul_fp8.rs`):

    s     = min(max(amax(|a|) · 1/448, 1/448²), 448)    one scale per token row
    q     = fp8(min(max(a · (1/s), -448), 448))         the activation's codes
    raw   = q @ W_codes.T                                f16
    out   = (raw · s) · w_scale                          two f16 roundings

and the card's expert door quantizes a token's PAIR ROWS with ONE scale over all K
pairs' lanes (`fp8_quantize_descriptors(rows, k·in, pairs = k)`): the gate/up rows are the
token's input replicated per pair, so their scale is the input's own; the down rows are
the K activations side by side, so the down scale is the amax over ALL K activations. The
down projection therefore runs in two passes over the slots — the activations first, then
the shared scale, then the projections. Writing the quantize in the program is what makes
the emulator compute what the card computes; before this kernel the emulator ran these
projections W8A16.

THE SPLICE'S OWN CONTRACT:

* PARAMETERS, IN ORDER: the router input `desc_xr` `[M, H]`, its gain `desc_rg` `[H]`, its
  projection `desc_rw` `[E, H]`, its per-expert scale `desc_pes` `[E]`; the expert input
  `desc_xe` `[M, H]`; the three banks and their per-channel scales `desc_wg`/`desc_sg`
  `[E, I, H]`/`[E, I]`, `desc_wu`/`desc_su` the same, `desc_wd`/`desc_sd` `[E, H, I]`/`[E, H]`;
  the output `desc_o` `[M, H]`; then the SCRATCH the program round-trips through memory —
  `desc_ids` `[M, K]` the routed ids as i32, `desc_rs` `[M, K]` the route weights,
  `desc_act` `[M, K·I]` the K activations.
* CONSTEXPRS: `M`, `H`, `I`, `E`, `K`; `EPS`, `INV_H` (the router norm); `TEMP` (`H^-0.5`);
  `INV_QMAX` (`1/448`) and `SCALE_FLOOR` (`1/448²`), the per-token scale's `mulConst` and
  `clipMin` (`OpFunc::op_consts`).
* GRID: `[1]`.

⛔ WHY THE SCRATCH IS MEMORY. The slab gather reads its index through a POINTER (the
indirect tile's index view is a kernel argument — `convert_ttir_to_ktdp::slab_gather`), so
the routed ids are stored as i32 before any slab is read; the route weights ride with them;
and the down projection's shared scale needs every pair's activation before any is
quantized, so the activations are held between the two passes.

⛔ WHY THE TOP-K IS ROUND-BASED, NOT `routetopk.py`'s RIPPLE. The ripple reads the logits
one `[M, 1]` column at a time, and a descriptor block must carry at least 16 bytes in its
last dimension (upstream's own check — real Triton refuses the 2-byte column). Rounds over
the whole row select the same experts in the same slot order with no column access.

⛔ EVERY LOOP IS `tl.static_range` — the ladder's law (`expertmatmul.py`'s header): every
offset but the routed expert id is a trace-time constant, and the id is the ONLY data-
dependent address in the program.
"""

import triton
import triton.language as tl


@triton.jit
def _fp8_scale(amax, INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    """The per-token activation scale — `quantscalepertokenfp8`'s
    `FMIN(FMAX(amax · mulConst, clipMin), clipMax)` (`OpFunc::op_consts`), for one token
    row (`amax` is `[1]`). The bounds are f16 tiles from birth: `tl.minimum`/`tl.maximum`
    turn a bare Python float into an f32 tensor, which would widen the scale to f32."""
    lo = tl.full([1], SCALE_FLOOR, tl.float16)
    hi = tl.full([1], 448.0, tl.float16)
    return tl.minimum(tl.maximum(amax * INV_QMAX, lo), hi)


@triton.jit
def _fp8_codes(a, inv, N: tl.constexpr):
    """`clip(a · (1/s), ±448)` rounded to e4m3 — the card's `fq_sc`/`fq_clip`/`qfp8ch` legs,
    for one `[1, N]` row. Widened back to f16 for the dot: the codes are the values."""
    lo = tl.full([1, N], -448.0, tl.float16)
    hi = tl.full([1, N], 448.0, tl.float16)
    q = tl.minimum(tl.maximum(a * inv[:, None], lo), hi)
    return q.to(tl.float8e4nv).to(tl.float16)


@triton.jit
def _w8a8(q, s, w_codes, w_scale, ROWS: tl.constexpr, N: tl.constexpr):
    """`(q @ W.T · s) · w_scale` — the matmulfp8 and its two dequant legs."""
    raw = tl.dot(q, w_codes.T, tl.zeros([ROWS, N], dtype=tl.float16))
    return (raw * s[:, None]) * w_scale


@triton.jit
def _gelu_tanh(g, N: tl.constexpr):
    """`gatedact.py`'s gelu: the tanh form through the sigmoid identity, the cubic in the
    f32 island (an f16 x³ overflows at |x| ≥ 41). The `1` is an f16 tile, `swiglu_mlp.py`'s
    spelling: `1.0 + e` would make the divide an f16-by-f32 `fdiv`, which upstream refuses."""
    g32 = g.to(tl.float32)
    v = 1.5957691216057308 * (g32 + 0.044715 * g32 * g32 * g32)
    e = tl.exp(-v).to(tl.float16)
    return tl.fdiv(g, tl.full([1, N], 1.0, tl.float16) + e)


@triton.jit
def moe_fwd(desc_xr, desc_rg, desc_rw, desc_pes,  #
            desc_xe, desc_wg, desc_sg, desc_wu, desc_su, desc_wd, desc_sd,  #
            desc_o,  #
            desc_ids, desc_rs, desc_act,  #
            M: tl.constexpr, H: tl.constexpr, I: tl.constexpr,  #
            E: tl.constexpr, K: tl.constexpr,  #
            EPS: tl.constexpr, INV_H: tl.constexpr, TEMP: tl.constexpr,  #
            INV_QMAX: tl.constexpr, SCALE_FLOOR: tl.constexpr):
    # ── the router ────────────────────────────────────────────────────────────
    xr_desc = tl.make_tensor_descriptor(desc_xr, shape=[M, H], strides=[H, 1],
                                        block_shape=[M, H])
    rg_desc = tl.make_tensor_descriptor(desc_rg, shape=[H], strides=[1], block_shape=[H])
    rw_desc = tl.make_tensor_descriptor(desc_rw, shape=[E, H], strides=[H, 1],
                                        block_shape=[E, H])
    pes_desc = tl.make_tensor_descriptor(desc_pes, shape=[E], strides=[1], block_shape=[E])
    xr = xr_desc.load([0, 0])
    ms = tl.sum(xr * xr, 1) * INV_H
    r = tl.rsqrt((ms + EPS).to(tl.float32)).to(tl.float16)
    xn = xr * r[:, None] * rg_desc.load([0])[None, :]
    lg = tl.dot(xn, rw_desc.load([0, 0]).T, tl.zeros([M, E], dtype=tl.float16))

    # THE TOP-K, K rounds over the whole `[M, E]` row: each round takes the row maximum
    # and the HIGHEST expert id among the lanes equal to it, then masks that lane out.
    # This is `topk.ddl`'s ripple order exactly — descending by value, and among equal
    # values the later scan position first (the ripple's strict `>` lets the newcomer take
    # the slot). The lane ids ride f16 (exact to 2048; the real router's `E = 128`).
    offs = tl.arange(0, E).to(tl.float16)
    koffs = tl.arange(0, K).to(tl.float16)
    pes = pes_desc.load([0])
    vals = tl.zeros([M, K], dtype=tl.float16)
    idsf = tl.zeros([M, K], dtype=tl.float16)
    psel = tl.zeros([M, K], dtype=tl.float16)
    # Every select is tile-on-tile f16 (`routetopk.py`'s law): a bare Python float in a
    # `tl.where` becomes an f32 tensor and widens the whole select.
    neg_inf = tl.full([M, E], float("-inf"), tl.float16)
    no_lane = tl.full([M, E], -1.0, tl.float16)
    zero_e = tl.zeros([M, E], dtype=tl.float16)
    one_k = tl.full([K], 1.0, tl.float16)
    zero_k = tl.zeros([K], dtype=tl.float16)
    for j in tl.static_range(0, K):
        # `tl.max` reduces an f16 tile in f32 (upstream's own promotion); each reduce is
        # narrowed straight back, the widening island LegalizeTypes collapses.
        mx = tl.max(lg, 1).to(tl.float16)
        lanes = tl.where(lg == mx[:, None], offs[None, :] + zero_e, no_lane)
        idx = tl.max(lanes, 1).to(tl.float16)
        sel = offs[None, :] == idx[:, None]
        # The selected expert's per-expert scale, by its id (a one-hot over the E lanes).
        pj = tl.sum(tl.where(sel, pes[None, :] + zero_e, zero_e), 1)
        # Slot j of the three `[M, K]` results: each slot is written exactly once, so
        # accumulating a one-hot column into zeros places it exactly. The one-hot is a
        # select of f16 tiles, not a bool cast (`.to` of an i1 is `arith.uitofp`).
        oh = tl.where(koffs == j, one_k, zero_k)[None, :]
        vals = vals + mx[:, None] * oh
        idsf = idsf + idx[:, None] * oh
        psel = psel + pj[:, None] * oh
        lg = tl.where(sel, neg_inf, lg)

    # The route weights: the K selected logits at the `H^-0.5` temperature, a softmax over
    # the K (`routesoftmax.py`'s chain), × each selected expert's scale. The ids land as i32:
    # the slab gather reads them back through a pointer (its index view).
    ids_desc = tl.make_tensor_descriptor(desc_ids, shape=[M, K], strides=[K, 1],
                                         block_shape=[M, K])
    rs_desc = tl.make_tensor_descriptor(desc_rs, shape=[M, K], strides=[K, 1],
                                        block_shape=[M, K])
    ids_desc.store([0, 0], idsf.to(tl.int32))
    sv = (vals * TEMP).to(tl.float32)
    ex = tl.exp(sv - tl.max(sv, 1)[:, None])
    rs_desc.store([0, 0], tl.fdiv(ex, tl.sum(ex, 1)[:, None]).to(tl.float16) * psel)

    # ── the experts ───────────────────────────────────────────────────────────
    xe_desc = tl.make_tensor_descriptor(desc_xe, shape=[M, H], strides=[H, 1],
                                        block_shape=[1, H])
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
    rw1_desc = tl.make_tensor_descriptor(desc_rs, shape=[M, K], strides=[K, 1],
                                         block_shape=[1, K])
    act_desc = tl.make_tensor_descriptor(desc_act, shape=[M, K * I], strides=[K * I, 1],
                                         block_shape=[1, I])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, H], strides=[H, 1],
                                       block_shape=[1, H])
    for m in tl.static_range(0, M):
        # This token's K route weights, one row; slot j's weight is its one-hot sum.
        wrow = rw1_desc.load([m, 0])
        # The gate/up activation: the token's input, one scale for all K pairs.
        xe = xe_desc.load([m, 0])
        sx = _fp8_scale(tl.max(tl.abs(xe), 1).to(tl.float16), INV_QMAX, SCALE_FLOOR)
        qx = _fp8_codes(xe, tl.fdiv(tl.full([1], 1.0, tl.float16), sx), H)
        # Pass 1: every pair's activation, and their shared amax.
        amax = tl.zeros([1], dtype=tl.float16)
        for j in tl.static_range(0, K):
            e = tl.load(desc_ids + (m * K + j))
            wg = tl.reshape(wg_desc.load([e, 0, 0]), [I, H]).to(tl.float16)
            wu = tl.reshape(wu_desc.load([e, 0, 0]), [I, H]).to(tl.float16)
            gate = _w8a8(qx, sx, wg, sg_desc.load([e, 0]), 1, I)
            up = _w8a8(qx, sx, wu, su_desc.load([e, 0]), 1, I)
            act = _gelu_tanh(gate, I) * up
            act_desc.store([m, j * I], act)
            amax = tl.maximum(amax, tl.max(tl.abs(act), 1).to(tl.float16))
        # Pass 2: the down projections at the shared scale, weighted and summed.
        sa = _fp8_scale(amax, INV_QMAX, SCALE_FLOOR)
        inv_a = tl.fdiv(tl.full([1], 1.0, tl.float16), sa)
        acc = tl.zeros([1, H], dtype=tl.float16)
        for j in tl.static_range(0, K):
            e = tl.load(desc_ids + (m * K + j))
            qa = _fp8_codes(act_desc.load([m, j * I]), inv_a, I)
            wd = tl.reshape(wd_desc.load([e, 0, 0]), [H, I]).to(tl.float16)
            down = _w8a8(qa, sa, wd, sd_desc.load([e, 0]), 1, H)
            w = tl.sum(tl.where(koffs[None, :] == j, wrow, tl.zeros([1, K], tl.float16)), 1)
            acc = acc + down * w[:, None]
        o_desc.store([m, 0], acc)
