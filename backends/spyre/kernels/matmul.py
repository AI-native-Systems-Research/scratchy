# SPDX-License-Identifier: Apache-2.0
"""Scratchy's Triton dense fp16 matmul kernel.

BODY PROVENANCE: derived from `crates/targets/spyre/triton/test-fixtures/swiglu_mlp.py`'s projection
structure (deltas 2/3/4 there) — one `tl.dot` over descriptor-loaded tiles, which is
the form `spyre-dot-to-linalg` rewrites to `linalg.matmul`. The fixture's OWN three-dot
fused MLP is NOT this kernel: `SubOp::MatmulTile` is ONE projection, and the activation
is its own node (`silumul.py`).

THE SPLICE'S OWN CONTRACT (what `scratchy-triton-splice` states about this kernel):

* PARAMETERS, IN ORDER: `desc_a`, `desc_w`, `desc_o` — the node's operand order
  (activation, weight) then the output. The registry does not permute. ⛔ This row is
  DENSE-ONLY (arity 2): the fp8 arity-3 W8A8 form is its OWN row over
  `matmul_fp8.py`.
* CONSTEXPRS: `M`, `K`, `N`, `BLOCK_M`, `BLOCK_K`, `BLOCK_N`, `M_TOTAL` — stated by
  the splice from the node's own regions (`A is [M, K]`, the out tile is `[M, N]`,
  all blocks the whole extents — ONE tile).
* GRID: `[1]`.

THE ORIENTATION: the weight descriptor is the PRESENTED weight's own on-disk `[n, k]`
buffer (the region the executor binds verbatim), and the `.T` is the single-dot
canonical verifier's transposed form — `verify_canonical_matmul_kernel` admits `tt.trans`
directly over the weight load, and `dot_to_linalg` folds that trans into the transpose-B
`indexing_maps = [[0,2],[1,2],[0,1]]`, so the weight is contracted in its on-disk
orientation on every executor.

⛔⛔ `N` IS THE **DEVICE** WIDTH, NOT THE LOGICAL ONE — whole sticks (`rung::row_pitch`):
the output's rows are held whole sticks wide, so a view stating the logical width would
understate the buffer every consumer addresses. The weight descriptor's shape is `[N, K]`
at the same device width — the zero-padded staging the worker binds (the checkpoint's
leading `n_logical` rows, then zero pad rows feeding only the output's pad columns).
"""

import triton
import triton.language as tl


@triton.jit
def matmul_fwd(desc_a, desc_w, desc_o,  #
               M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,  #
               BLOCK_M: tl.constexpr,  #
               BLOCK_K: tl.constexpr,  #
               BLOCK_N: tl.constexpr,  #
               M_TOTAL: tl.constexpr,  #
               ):
    start_m = tl.program_id(0)
    # Descriptors, not pointer blocks (swiglu delta 3). The W descriptor is the
    # presented weight's own `[N, K]` on-disk region at the DEVICE width (the
    # zero-padded staging); the `.T` below is the transposed canonical form
    # (`tt.trans` directly over the load), which `dot_to_linalg` folds into the
    # transpose-B indexing maps — the same maps main's `KtirFunc::matmul` states over the
    # same bytes.
    # ⛔ THE DESCRIPTOR NAMES THE STORAGE, THE STORE NAMES THE WINDOW — the out
    # descriptor's shape is the OUTPUT TENSOR's `[M_TOTAL, N]` (the storage the
    # layout reserved, N the device width), while the store takes the `[M, N]`
    # tile at row 0: the prefill lm-head fold re-lowers the tail at m=S over the
    # LAST_HIDDEN staging, and its output is the `[S, vocab]` prefix of the
    # `[mq, vocab]` logits storage (row 0 alone at S = 1). A whole-region node
    # states `M_TOTAL = M`, so nothing changes for it.
    a_desc = tl.make_tensor_descriptor(desc_a, shape=[M, K],
                                       strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    w_desc = tl.make_tensor_descriptor(desc_w, shape=[N, K],
                                       strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M_TOTAL, N],
                                       strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    # The canonical single-dot offsets, framed by the orientation: A loads at
    # `[pid, k]`, the `.T` B loads its `[N, K]` slice at `[pid, k]` (an n ROW by pid,
    # k in place), the store at `[pid, pid]` — with every k index 0 at single-tile
    # (grid [1], BLOCK_K == K), which is what the splice states. At grid [1] `start_m`
    # is 0, and a `start_m * BLOCK_M` multiply would be an op the shape-keyed template
    # refuses ("does not model").
    a = a_desc.load([start_m, 0])
    w = w_desc.load([start_m, 0]).T
    # f16 accumulator (swiglu delta 2): device floats are DL16-f16; Triton defaults
    # `tl.dot` to f32 but the device MAC rounds to f16 at every step, so there is no
    # wide accumulator to keep.
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
    # ONE dot, no K-loop: BLOCK_K == K, the same single-tile whole-region law
    # `KtirFunc::matmul` from main states (one `linalg.matmul`, no scf.for).
    p = tl.dot(a, w, acc)
    o_desc.store([start_m, start_m], p)
