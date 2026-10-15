# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton expert-bank matmul kernel — the MoE pair projection.

BODY PROVENANCE: the ladder's own one-hot-over-`E` form, designed against the
door's contract (`lower_ktir_to_superdsc::expert_matmul`) and the e2e golden
(`moe_expert_chain_e2e`), not ported from a fixture. The door never walks
these ops (it re-derives its five-leg chain from the minted views), so this
body serves the EMULATOR alone — and the ladder has no indirect slab select
(`tt.descriptor_gather` is bound to its five vendor asserts), so the expert
`e` slab is selected arithmetically: every expert's projection is computed
and masked by the pair's one-hot `ids == float(e)`, the sum over experts
leaving only the routed slab's term.

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this
kernel):

* PARAMETERS, IN ORDER: `desc_w`, `desc_s`, `desc_idx`, `desc_x`, `desc_o` —
  ⭐ THE MINT ORDER, NOT THE NODE'S OPERAND ORDER. `KtirFunc::expert_matmul`
  mints `w`/`s` first (their views open the bank), then `idx`, then `rows`,
  so the program's parameters arrive as `[w, s, idx, rows]` and the door
  resolves each by extent (the rank-1 scale bank, its row-mate the code
  bank, the two `m`-rowed inputs split wide/narrow). The splice therefore
  binds `inputs[2]` (w), `inputs[3]` (s), the SORT'S INDEX INPUT (see
  below), and `inputs[0]` (the pair rows), then the output — arity 4.
* ⭐ THE INDICES ARE NOT THE MATMUL'S OWN INPUT. The node's `inputs[1]` is
  the SORT's output (the pair routing), wired only so the splice can walk
  one producer up and bind the sort's OTHER input — the `[m, k]` expert-id
  tensor — as this program's `desc_idx`, exactly the producer walk the
  branch's own builder performed (`producer_of` on the pairs, refusing a
  producer that is not a two-input `ExpertSort` by name).
* CONSTEXPRS: `M`, `K`, `E`, `IN`, `N` — stated by the splice from the
  node's own regions: `M` the output's rows, `K` the node's top-k payload,
  `IN = rows.v_cols / K`, `N = out.v_cols / K`, `E = w.v_rows / N` — the
  same extent laws the door re-derives from the minted views, so a
  disagreement is the door's own refusal, not a silent second reading.
* GRID: `[1]`.

NUMERICS, THE GOLDEN'S OWN: `proj = f16(f16(f32-dot) · f16(scale))` per
pair. The dot is the f16-accumulator form (`matmul_fp8`'s own spelling —
the ladder's `dot_to_linalg` refuses any non-f16 dot result, the
f32-accumulator folding being a named follow-on), and the emulator's
`linalg.matmul` computes in f32 and rounds ONCE to the f16 out, which IS
the golden's `f16(f32-dot)`; the scale mulf is the ONE f16 rounding of the
product; the one-hot `tl.where` adds f16 terms and exact zeros in f32, so
the routed slab's term passes to the store bit-identically and every other
expert's term is exactly zero.

⛔ EVERY LOOP IS `tl.static_range` — the ladder's own law. A `tl.range`
induction variable is a runtime scalar, and a runtime scalar cannot splat
(`tt.splat` is argument-only in the census); `static_range` unrolls at trace
time with the induction variable a PYTHON INT, so `float(e)` is a
trace-time constant (a legal float splat) and every load corner
(`[e * N, 0]`, `[0, j * IN]`) is constant at trace. The cost is `E`× the
door's per-pair work — an emulator-only cost the card never pays, because
the card runs the door's gathered five-leg chain over the same views.
"""

import triton
import triton.language as tl


@triton.jit
def expertmatmul_fwd(desc_w, desc_s, desc_idx, desc_x, desc_o,  #
                     M: tl.constexpr, K: tl.constexpr, E: tl.constexpr,  #
                     IN: tl.constexpr, N: tl.constexpr):
    # Descriptors name STORAGE (the door's own views): the stacked
    # `[E*N, IN]` fp8 code bank, the flat `[E*N]` scale row (rank-1, the
    # loader's bf16 scales staged f16), the `[M, K]` expert ids, the
    # `[M, K*IN]` pair rows, the `[M, K*N]` pair outputs. The blocks are the
    # windows the loops read: one expert slab, one slot's id column, one
    # slot's activation window, one slot's output window.
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[E * N, IN],
                                       strides=[IN, 1],
                                       block_shape=[N, IN])
    s_desc = tl.make_tensor_descriptor(desc_s, shape=[E * N],
                                       strides=[1],
                                       block_shape=[N])
    idx_desc = tl.make_tensor_descriptor(desc_idx, shape=[M, K],
                                         strides=[K, 1],
                                         block_shape=[M, 1])
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, K * IN],
                                       strides=[K * IN, 1],
                                       block_shape=[M, IN])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, K * N],
                                       strides=[K * N, 1],
                                       block_shape=[M, N])

    for j in tl.static_range(0, K):
        # Slot j's activation window `[M, IN]` and id column `[M, 1]` — the
        # ids are f16 expert numbers (exact to 2048), compared against the
        # trace-time constant `float(e)`.
        xs = x_desc.load([0, j * IN])
        ids = idx_desc.load([0, j])
        acc = tl.zeros([M, N], dtype=tl.float32)
        for e in tl.static_range(0, E):
            # Expert e's fp8 slab, widened for the dot (the frontend refuses
            # a mixed dot), and its `[N]` scale window.
            w = w_desc.load([e * N, 0]).to(tl.float16)
            sc = s_desc.load([e * N])
            # The f16-accumulator dot — the f32 code-dot rounded ONCE to
            # f16 (the emulator computes `linalg.matmul` in f32 and rounds
            # to the out dtype) — then the ONE f16 scale rounding: the
            # golden's own two roundings.
            p16 = tl.zeros([M, N], dtype=tl.float16)
            p = tl.dot(xs, w.T, p16)
            t = p * sc[None, :]
            # The one-hot: the routed expert's term passes, every other
            # expert contributes an exact zero to the f32 accumulator.
            acc += tl.where(ids == float(e), t, 0.0).to(tl.float32)
        o_desc.store([0, j * N], acc.to(tl.float16))
