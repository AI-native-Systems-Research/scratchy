"""Flash attention for Spyre, DERIVED FROM TRITON'S OWN KERNEL.

Source: python/tutorials/06-fused-attention.py in this tree (`_attn_fwd` +
`_attn_fwd_inner`). That kernel is the canonical, widely-exercised Triton flash
attention. This file is it, with a MINIMAL and individually-justified delta for the
Spyre target. Every difference is listed below so the diff against the tutorial is
auditable; if you are tempted to add a fifth kind of change, that is a signal the
change belongs in the compiler instead.

WHY DERIVE RATHER THAN WRITE ONE: an earlier hand-rolled fixture expressed both
matmuls as `tl.sum(q[:, None, :] * k[None, :, :], axis=2)` -- a broadcast-multiply
plus reduce. That emits ZERO `tt.dot`, so `spyre-dot-to-linalg` had nothing to
rewrite and the multicore matmul builder (which works, and is HW-validated) was
unreachable; the body arrived at the emitter as four reductions, the one shape it
has no builder for. The tutorial uses `tl.dot` for both legs, which is what the
backend can already lower. Do not re-derive the score matmul by hand.

THE DELTA, and the justification for each item
==============================================

1. GPU-ONLY MACHINERY REMOVED. `@triton.autotune` + its config list + `keep` +
   `prune_invalid_configs` + `_host_descriptor_pre_hook`; `warp_specialize`;
   `num_stages`; `IS_HOPPER` / `is_blackwell()`; the Hopper-only
   `acc.reshape/permute/split/join` accumulator path; the `FP8_OUTPUT` /
   `tl.float8e5` variants. None of these describe the computation -- they are a
   CUDA search space and CUDA codegen hints. Spyre compiles one configuration.

2. ACCUMULATORS ARE f16, NOT f32 (`m_i`, `l_i`, `acc`). This is a HARDWARE fact,
   not a concession:
     - device floats are DL16-f16; `LegalizeTypes` RED-STOPS genuine f32 compute by
       design, and demoting is an owner decision recorded here at the source rather
       than worked around in a pass.
     - `arch.elemsPerStick("f32")` is 32 vs 64 for f16, so an f32 score tile spans
       TWO sticks -- and reduce-MAX mis-combines partial maxima past ONE stick of
       columns (a real, documented hardware defect). f32 would walk straight into it.
     - the hardware-proven reference (scratchy's SuperDSC attention) is
       `Df::Fp16` throughout.
   Consequence to keep in mind: every reduce and transcendental must be truncated
   back to f16 so any widening stays a local extf/op/truncf ISLAND -- that shape
   `LegalizeTypes` collapses; genuine f32 it refuses.

3. THE ADDITIVE MASK CONSTANT IS FINITE (-1.0e4, was -1.0e6). At f16 the tutorial's
   -1.0e6 overflows (f16 max is 65504) to -inf. An all-masked block then leaves
   `m_ij = -inf`, and `exp2(m_i - m_ij)` is `-inf - -inf` = NaN one line below.
   That is exactly the trap the reference implementation documents for its seed/fold
   roles. A finite -1.0e4 keeps `m_ij` finite while `exp2(qk - m_ij)` still
   underflows to 0, so masked columns contribute nothing AND no NaN can appear. It
   also removes the need for the `finite` / `m_safe` select dance a -inf mask forces.
   NOTE the tutorial's mask is ADDITIVE, which is what the proven Spyre path wants:
   there it is a fused `StridedAdd` epilogue on the score matmul.

4. GQA. The tutorial has one head index for q and k/v. Granite is GQA, so a query
   head reads kv plane `off_h // GQA`. The cache is nkvh-DEDUPED (verified against
   the reference's live code: both the resident paged pool and the per-step staged
   buffer are kv-head indexed via `qh / gqa`; the comment in that file claiming
   nqh-expansion is stale). Set GQA=1 for a non-GQA model.

5. THE LOGSUMEXP STORE IS DROPPED (`tl.store(m_ptrs, m_i)`). `M` exists to feed the
   backward pass; this is inference-only. Dropping it also keeps the kernel to one
   output, so the emitter sees one store.

6. NOT DONE, AND RECORDED SO IT IS NOT RE-ATTEMPTED: reading K from a separate
   pre-transposed cache plane. It removes the `tt.trans` that dot-to-linalg objects to
   ("A and B must be direct tt.descriptor_load results"), and the hardware-proven
   reference does keep such a plane (its `KvPlane::Kt`). But it makes the host hold a
   SECOND FULL COPY OF K -- doubling the KV cache, which is the dominant memory
   consumer in inference and precisely what paging exists to bound -- in order to avoid
   an operation that is FREE. KTDP's `construct_access_tile` carries an
   `access_tile_order` affine map, and that op's definition states "an identity map
   keeps the order, a transpose map swaps the two dimensions"; the verifier requires
   only that it be a permutation. So the transpose belongs in the LOWERING, folded into
   that order map, and the kernel keeps `.T` exactly as Triton wrote it.

7. Z / H / N_CTX ARE constexpr, NOT RUNTIME ARGS (SPYRE-SPECIFIC). The tutorial
   passes them as i32 arguments because a CUDA kernel is shape-generic. Spyre is not:
   affine loop bounds are static, and the emitter derives the corelet work division
   from the descriptor extents, so a runtime extent has no faithful lowering ("A/B
   descriptor M/N/K are not compile-time constants"). Spyre therefore compiles one
   kernel per shape, which is already true of every other fixture in this backend.

8. THE LAUNCH CONTRACT: Q, K, V AND O ARE PRESENTED IN THE LAYOUT THE FOLD LAW
   FORCES, and THE KERNEL SOURCE DOES NOT CHANGE FOR IT. This is a HOST-SIDE
   obligation, so it is a delta on the caller and not on the code below.

   WHY. A reduction wider than the compute unit's vector becomes a TREE of folds,
   and a fold's second tile must start on a stick boundary, so the reduced axis has
   to sit DIRECTLY ABOVE THE LANE AXIS (measured; R8 of
   --spyre-carried-values-to-memory, probes p55/p57/p59/p60/p63/p63b). The two row
   reductions here produce ONE VALUE PER QUERY ROW, so the query axis must be the
   LANE axis of the score tile -- which transposes it, and transposes the
   contraction that produces it. `--spyre-lane-major-layout` takes that decision
   for the whole body; what it cannot do is move data, because a transposed READ is
   refused downstream (a transposing `access_tile_order` is rejected in
   ConstructThreeStagePipeline, probe p02, and it permutes the base as well as the
   walk). So the host presents:

     desc_q  Q^T, [HEAD_DIM, Z*H*N_CTX] row-major. The DIRECT operand of the score
             contraction: it varies along the lanes, so it cannot be a splat. A
             transpose only, 8192 elements per block, no widening.
     desc_k  K STICK WIDE, [Z*(H//GQA)*N_CTX, HEAD_DIM, 64] with K[row, d] in LANE
             0 of stick (row, d). The score contraction's SPLAT source: K is
             invariant along the lanes, so it cannot be read directly. A splat's
             source must be stick wide -- a [K, 1] layout is refused BY NAME by the
             senprog emitter ("Extent in load/store set is larger than from the
             layout", p57), because the 64-lane load set is checked against the
             SOURCE layout's extent.
     desc_v  V^T STICK WIDE, [HEAD_DIM, Z*(H//GQA)*N_CTX, 64] with V[row, d] in
             LANE 0 of stick (d, row). P.V's SPLAT source, same reasoning; the
             transpose is there because P.V contracts over the KV axis, so V's head
             axis is the one that must be outermost.
     desc_o  O^T, [HEAD_DIM, Z*H*N_CTX] row-major, written by the kernel and read
             back transposed by the host. run_o is [HEAD_DIM, BLOCK_M].

   PAD THE UNUSED LANES WITH A DIFFERENT VALUE. Lanes 1..63 of a stick-wide splat
   source are never read by a correct program, so filling them with the lane-0
   value would make a wrong-lane read return the right answer. Fill them with
   something else and a wrong-lane read is a wrong NUMBER -- that is the `k_pad`
   control in p65 (457 against a correct 457-free reference), and it is the only
   reason a lane-indexing mistake shows up at all.

   WHAT IS NOT ASKED OF THE HOST: nothing for P or for the score tile. Both are
   in-kernel values, and this layout reads them DIRECTLY -- which is what makes it
   viable at all, since replicating an in-kernel value along a non-outermost
   parallel dimension is upstream defect 7 and writing one lane of a stick is
   defect 8, both measured and both closed
   (scheduler_probes/DESIGN_QUESTION_pv_splat_operand.md).

   COST: K and V are read 64x wider than the data they carry. That is the price of
   the splat form on this device, it falls on the KV cache, and it is the one item
   in this list worth revisiting if the scheduler ever grows a broadcast that
   widens a narrow source itself (scheduler_probes/UPSTREAM_DEFECTS.md).

9. sm_scale IS constexpr, NOT A RUNTIME ARGUMENT (SPYRE-SPECIFIC), and this one is a
   HARD constraint rather than a convenience. `qk_scale = sm_scale * 1/log(2)` multiplies
   the score tile, so it reaches a COMPUTE OP AS AN OPERAND -- and on this device:

     * a compute group can only read values it LOADS. The scheduler builds one FIFO slot
       per LOADED operand and none for anything else, so a tensor operand that is not a
       load is used from inside a `ktdf.stage` region its definition does not dominate.
       MEASURED, twice: probe p41, and again on this very fixture -- four
       `tensor.splat qk_scale` inputs, four `operand #1 does not dominate this use`.
     * and dcc's constant-operand encoding is an ISA IMMEDIATE FIELD with FOUR legal
       values (`constValToField`, VectorOperands.cpp:250 -- only 0, 1, 2, 3), so an
       arbitrary scalar cannot be an operand at all, by encoding.

   So the scale has to arrive through MEMORY, as a caller-seeded splat buffer -- which is
   exactly what `--spyre-carried-values-to-memory` already does for the `ln2` its exp2
   decomposition needs, and now does for any splat that reaches a compute op as an INPUT
   (`materializeSplatInputs`). That routing needs the VALUE at compile time to put it in
   the `spyre.constant_buffers` contract, and a runtime argument does not have one.

   *** THIS DELTA IS A NAMED COMPILER LIMITATION, NOT A FIXTURE TIDY-UP. READ IT AS ONE. ***

   A REAL CALLER PASSES sm_scale AT RUNTIME, and that STILL REFUSES -- by name, from
   `materializeSplatInputs`: "this compute op reads a tensor.splat as an INPUT whose scalar
   is NOT a compile-time constant". The refusal is deliberate and fail-closed (the
   alternative was to fabricate a value, which is a wrong answer with no diagnostic), but
   it IS a boundary of what this backend compiles today, and a reader of any result
   obtained with this fixture should see it here rather than discover it.

   WHAT CLOSING IT TAKES, so the size of the gap is not guessed: the routing machinery
   already exists and is exercised -- a buffer argument, a `spyre.constant_buffers`-shaped
   contract entry, a fresh view and tile per group, a broadcast read that
   ScalarBroadcastLegalization turns into a splat transfer. ONLY THE VALUE'S PROVENANCE
   DIFFERS. A runtime scale needs a contract kind that names the SCALAR ARGUMENT the
   LAUNCHER must broadcast into the buffer (`spyre.scalar_buffers = [{arg = N, from =
   <kernel argument>, type = ...}]`) instead of a compile-time `value`, and the launcher
   filling it at dispatch instead of the compiler baking it. No new emission shape, no new
   device capability -- one contract kind and its launcher side.

   Same ground as delta 7 -- Spyre compiles one kernel per shape, so it compiles one per
   scale too -- with the added and stronger reason that a runtime scalar has no lowering
   here at all YET.

NOT CHANGED, deliberately: both `tl.dot` legs, `exp2` with the `1/log(2)` scale fold,
the additive mask, the STAGE 1/2/3 off-band/on-band split, BOTH tiling axes
(BLOCK_M over query rows and BLOCK_N over keys), and descriptor loads.

PAGING IS NOT IN THIS KERNEL, BY DESIGN. Paged KV addressing is host-resolved:
`agen.indirect_vector_load` is LXLU-only by construction and LX cannot hold the
cache, so the host walks the page map and presents a contiguous view. There is no
block table here and there should not be one -- see the project note on host-resolved
paging. That makes this the same kernel for contiguous and paged KV.

10. THE CAUSAL MASK IS A LOADED f16 TILE, NOT AN IN-KERNEL INTEGER COMPARE
    (SPYRE-SPECIFIC, STAGE 2 ONLY). The tutorial writes

        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, keep_v, drop_v)

    and it cannot lower here: `offs_m` and `offs_n` are i32 `tl.arange` vectors and
    THERE IS NO LANE-WISE INTEGER COMPARE ON THIS DEVICE. The compute units are f16
    SIMD; the predicate would need a comparison instruction that does not exist, and
    the `tl.where` a per-lane select on its result.

    SO THE PREDICATE IS PRE-COMPUTED ON THE HOST AND THE MASK ARRIVES ADDITIVELY --
    which is what the tutorial's form already wanted (delta 3: additive is the proven
    Spyre path, a fused StridedAdd epilogue on the score matmul in the reference).
    `qk + mask_tile` is a plain f16 elementwise add, the one thing the device does
    natively, and it replaces a compare, a select and two materialised operands.

    *** AND THE TILE IS POSITION-INDEPENDENT, WHICH IS WHY IT IS ONE BUFFER FOR THE
    WHOLE LAUNCH AND NOT ONE PER BLOCK. *** On the on-band stage
    `lo = start_m * BLOCK_M`, and with BLOCK_M == BLOCK_N the loop runs EXACTLY ONCE,
    so `start_n == start_m * BLOCK_M` and

        offs_m - start_n == tl.arange(0, BLOCK_M)

    leaving the condition as `arange(BLOCK_M)[:, None] >= arange(BLOCK_N)[None, :]` --
    the same lower-triangular pattern for every query block. The tile is therefore a
    COMPILE-TIME CONSTANT: 0 on and below the diagonal, -1.0e4 above it, in f16, at
    BLOCK_M x BLOCK_N. It carries no grid position, so it does NOT interact with the
    per-core address question at all.

    THE HOST OWES ONE MORE REGION: 64 x 64 f16 = 8 KiB, ONCE, for the whole launch.
    Alongside delta 8's four, and negligible against them.

    WHAT THE PIN BUYS, AND IT IS NOT THE MASK. Causal STAGE 1 (off-band) runs
    `lo, hi = 0, start_m * BLOCK_M`, so ITS TRIP COUNT IS THE QUERY BLOCK INDEX and
    varies per grid position. The emission unrolls that sweep at compile time, so a
    position-dependent trip count has no single form. WITH THE POSITION PINNED THE
    TRIP COUNT IS A COMPILE-TIME CONSTANT, which is the real and specific reason
    causal is tractable at a pinned position. It is NOT luck, and it is NOT the mask:
    the mask tile above is position-independent either way. And the pin must stay a
    single constant substitution at the last step -- the per-core address is always
    emitted in its correct `uniform.query_map` form, and that substitution is the ONLY
    difference between the one-core and the multi-core build.

"""

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v, desc_mask,  # SPYRE: delta 10
                    offset_kv_y, start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        # off-band: strictly before the diagonal, so NO mask is needed at all.
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # on-band: straddles the diagonal, so this is the only masked stage.
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # causal = False: every key is in range.
        lo, hi = 0, N_CTX
    offsetk_y = offset_kv_y + lo
    offsetv_y = offset_kv_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # The transpose STAYS HERE, as Triton wrote it. It is FREE on Spyre: KTDP's
        # construct_access_tile takes an `access_tile_order` affine map whose own op
        # definition says "a transpose map swaps the two dimensions", and the verifier
        # requires only that it be a permutation. So the lowering must fold tt.trans of
        # a descriptor load into that map. Keeping a second, pre-transposed K plane in
        # the cache instead would DOUBLE the KV cache -- the dominant memory consumer
        # in inference, and the very thing paging exists to bound -- to avoid an
        # operation that costs nothing.
        k = desc_k.load([offsetk_y, 0]).T
        # out_dtype=f16 (delta 2): Triton defaults tl.dot to an f32 accumulator, but
        # the device MAC rounds to DL16 at every step, so there is no wide accumulator
        # to keep. The second dot infers f16 from `acc`.
        qk = tl.dot(q, k, out_dtype=tl.float16)
        if STAGE == 2:
            # SPYRE: delta 10 -- THE MASK IS A LOADED f16 TILE, NOT AN IN-KERNEL
            # INTEGER COMPARE. The upstream form is
            #     mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            #     qk = qk * qk_scale + tl.where(mask, keep_v, drop_v)
            # and it cannot lower: `offs_m`/`offs_n` are i32 `tl.arange` vectors and
            # THERE IS NO LANE-WISE INTEGER COMPARE ON THIS DEVICE. The compute units
            # are f16 SIMD; a per-lane predicate would have to come from a comparison
            # instruction that does not exist, and the `tl.where` would then need a
            # per-lane select on it.
            #
            # SO THE PREDICATE IS PRE-COMPUTED AND THE MASK ARRIVES ADDITIVELY, which
            # is what the tutorial's own form already wanted (see the note at delta 3
            # -- ADDITIVE is the proven Spyre path). `qk + mask_tile` is a plain f16
            # elementwise add, which is the one thing the device does natively.
            #
            # AND THE TILE IS POSITION-INDEPENDENT, which is why one buffer serves the
            # whole launch. On the on-band stage `lo = start_m * BLOCK_M`, and with
            # BLOCK_M == BLOCK_N there is exactly ONE iteration, so
            # `start_n == start_m * BLOCK_M` and
            #     offs_m - start_n == tl.arange(0, BLOCK_M)
            # leaving the condition as `arange(BLOCK_M)[:, None] >= arange(BLOCK_N)[None, :]`
            # -- the same lower-triangular pattern for every query block. So this is a
            # COMPILE-TIME CONSTANT [BLOCK_M, BLOCK_N] tile: 0 on and below the
            # diagonal, -1.0e4 above it. It needs no per-core address and therefore
            # does not interact with the grid-position question at all.
            #
            # THE VALUES ARE -1.0e4 AND 0, IN f16, FOR THE REASONS AT DELTA 3: -1e6
            # overflows f16 to -inf and an all-masked block then yields -inf - -inf =
            # NaN one line below; and building the operands in the compute dtype keeps
            # the whole mask-add in f16, where Python literals would produce an f32
            # select that LegalizeTypes rightly refuses as genuine f32 compute
            # (SP-E2-03).
            #
            # THE HOST OWES ONE MORE REGION, and it is 8 KiB once for the whole launch,
            # not per block: see delta 8's launch contract.
            mask_tile = desc_mask.load([0, 0])
            qk = qk * qk_scale + mask_tile
            m_ij = tl.maximum(m_i, tl.max(qk, 1).to(tl.float16))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, (tl.max(qk, 1) * qk_scale).to(tl.float16))
            qk = qk * qk_scale - m_ij[:, None]
        # exp2, not exp: qk_scale carries the 1/log(2) so the base change is free.
        # Widen for the transcendental and truncate straight back -- that island is
        # what LegalizeTypes collapses.
        p = tl.math.exp2(qk.to(tl.float32)).to(tl.float16)
        # -- compute correction factor
        alpha = tl.math.exp2((m_i - m_ij).to(tl.float32)).to(tl.float16)
        l_ij = tl.sum(p, 1).to(tl.float16)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        v = desc_v.load([offsetv_y, 0])
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def attn_fwd(sm_scale: tl.constexpr,  # SPYRE: delta 9

             desc_q, desc_k, desc_v, desc_o,  #
             desc_mask,  # SPYRE: delta 10 -- the prepared additive causal mask tile
             Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr,  # SPYRE: delta 7
             HEAD_DIM: tl.constexpr,  #
             BLOCK_M: tl.constexpr,  #
             BLOCK_N: tl.constexpr,  #
             GQA: tl.constexpr,  #
             STAGE: tl.constexpr,  #
             ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    # The kv planes are nkvh-deduped, so their descriptor extent is H // GQA planes.
    kv_y_dim = Z * (H // GQA) * N_CTX
    desc_q = tl.make_tensor_descriptor(desc_q, shape=[y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(desc_k, shape=[kv_y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(desc_v, shape=[kv_y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(desc_o, shape=[y_dim, HEAD_DIM],
                                       strides=[HEAD_DIM, 1],
                                       block_shape=[BLOCK_M, HEAD_DIM])
    # SPYRE: delta 10. ONE [BLOCK_M, BLOCK_N] f16 tile for the whole launch -- the
    # on-band mask is position-independent (see the note at its use), so its extent is
    # the block, not the sequence, and its offset is always [0, 0].
    #
    # SPYRE: delta 11 -- AND IT IS BUILT ONLY WHEN STAGE 2 RUNS. `STAGE & 2` is a
    # `tl.constexpr` predicate, so exactly one arm is generated. Without the guard the
    # non-causal build emits a descriptor no load consumes, DCE deletes it, and the
    # parameter it addressed is left with no `construct_memory_view` -- which is the
    # handoff refusal `attn_fwd_noncausal` exists to answer. Guarding it means the
    # placeholder that entry point passes in this slot contributes NO op at all, rather
    # than one that only survives to be deleted.
    if STAGE & 2:
        desc_mask = tl.make_tensor_descriptor(desc_mask, shape=[BLOCK_M, BLOCK_N],
                                              strides=[BLOCK_N, 1],
                                              block_shape=[BLOCK_M, BLOCK_N])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # GQA: a query head reads the kv plane it shares, off_h // GQA.
    offset_kv_y = off_z * (N_CTX * (H // GQA)) + (off_h // GQA) * N_CTX
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # f16 accumulators (delta 2). l_i seeds at 1.0, not 0, so the epilogue divide
    # cannot be by zero even if every block is masked.
    m_i = tl.zeros([BLOCK_M], dtype=tl.float16) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float16) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2), so exp2 above needs no extra multiply
    # load q: it stays resident for the whole KV sweep
    q = desc_q.load([qo_offset_y, 0])
    # For causal=True, STAGE=3 and the inner gets 1 (off-band) then 2 (on-band).
    # For causal=False, STAGE=1 and the inner gets 3 (everything, unmasked).
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v, desc_mask,  # SPYRE: delta 10
                                        offset_kv_y, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v, desc_mask,  # SPYRE: delta 10
                                        offset_kv_y, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX)
    # epilogue. The tutorial also stores m_i + log2(l_i) as the logsumexp for the
    # backward pass; dropped here (delta 5) -- inference needs only the output.
    acc = acc / l_i[:, None]
    desc_o.store([qo_offset_y, 0], acc)


@triton.jit
def attn_fwd_noncausal(sm_scale: tl.constexpr,  # SPYRE: delta 9

                       desc_q, desc_k, desc_v, desc_o,  #
                       Z: tl.constexpr, H: tl.constexpr, N_CTX: tl.constexpr,
                       HEAD_DIM: tl.constexpr,  #
                       BLOCK_M: tl.constexpr,  #
                       BLOCK_N: tl.constexpr,  #
                       GQA: tl.constexpr,  #
                       STAGE: tl.constexpr,  #
                       ):
    """SPYRE: delta 11 -- THE FOUR-POINTER ENTRY POINT FOR THE NON-CAUSAL CASE.

    `attn_fwd` takes FIVE pointers because STAGE 2 needs the prepared additive mask
    (delta 10). The non-causal configuration passes `4 - STAGE == 3` to the inner, which
    takes the `else` branch and NEVER LOADS THE MASK -- so nothing addresses `desc_mask`
    and no `ktdp.construct_memory_view` is left taking it.

    THAT IS A REFUSAL, NOT A TIDY-UP, and it is OURS -- `spyre-ktir-handoff`'s
    `every_parameter_states_its_width`:

        parameter 4 (%desc_mask) is addressed NOWHERE: no `ktdp.construct_memory_view`
        anywhere in the body takes it, so at grid [16] nothing states how wide the
        buffer it addresses is, and this crate will not invent one.

    and it is RIGHT to refuse. The parameter still occupies binding slot 4 and the
    consumer numbers its buffers BY PARAMETER POSITION, so narrowing the guard to
    tolerate a trailing hole would make the launcher's buffer numbering depend on which
    ops DCE happened to delete. Narrowing it is also not sufficient: scratchy's
    `regions()` independently hard-errors on a viewless parameter, so the refusal would
    only move one crate downstream.

    So THE FIXTURE STATES THE CONTRACT instead: a non-causal launch binds FOUR buffers,
    and this is the kernel whose signature says so. The body is NOT duplicated -- this is
    a `@triton.jit` call, which `make_ttir`'s `inline_calls` inlines exactly as it already
    inlines `_attn_fwd_inner`, so what reaches the emitter is `attn_fwd`'s own ops and
    nothing else. `desc_q` stands in the mask slot as a placeholder that delta 11's
    `if STAGE & 2` never turns into a descriptor, so it contributes no op whatsoever.
    """
    tl.static_assert(STAGE == 1)
    attn_fwd(sm_scale, desc_q, desc_k, desc_v, desc_o, desc_q,  #
             Z, H, N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N, GQA, STAGE)


# --- the Spyre configuration ---------------------------------------------------
# BLOCK_N is one f16 stick (64) because the cross-key reductions must stay inside a
# single stick. It is a correctness constraint from the hardware, not a tuning knob.
#
# ⛔⛔⛔ THE DEFECT IS A CONJUNCTION, AND EVERY SINGLE-AXIS READING OF IT IS WRONG.
# reduce-MAX mis-combines partial maxima when `rows > 1` AND the reduced extent
# exceeds one stick of COLUMNS -- neither condition alone. This kernel's softmax max
# is over BLOCK_M query rows, so `rows > 1` already holds and the COLUMN half is what
# binds here; that is why 64 is load-bearing in THIS kernel.
#
# Each half-reading has a SHIPPING, HARDWARE-PROVEN counter-example, so neither can be
# re-derived from the diagnostic alone:
#   * "rows > 1 is broken" -- refuted by `attn_bmax` (rows = nqh > 1, width = one
#     stick), real and hardware-verified at 31 tok/s. An earlier per-row split built on
#     this reading was tried and REVERTED; see the note in
#     `ir/bridge/tiled_op_sdsc_op/attn.rs`.
#   * "past one stick of columns is broken" -- refuted by the fp8 activation
#     quantiser's `fq_amax_op` (rows == 1, cols == k, many sticks), on every fp8 matmul
#     of every layer, at 41 tok/s. See `emit/ktir_matmul_fp8.rs`.
#
# The one measurement everyone read half of is the attention diagnostic `mxp = 0 over
# [nqh, cap]`: it moved BOTH axes off their safe values AT ONCE, so it cannot attribute
# the failure to either. The conjunction is what `reduce.rs`'s own `stickmajor` branch
# fires on, and it is enforced as such at `emit/lower_ktir_to_superdsc.rs::reduce`.
BLOCK_N = 64

SIGNATURE = {
    "sm_scale": "constexpr",  # SPYRE: delta 9
    "desc_q": "*fp16", "desc_k": "*fp16", "desc_v": "*fp16", "desc_o": "*fp16",
    "desc_mask": "*fp16",  # SPYRE: delta 10
    "Z": "constexpr", "H": "constexpr", "N_CTX": "constexpr",
    "HEAD_DIM": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
    "GQA": "constexpr", "STAGE": "constexpr",
}


def constexprs(z=1, h=4, n_ctx=256, head_dim=128, block_m=64, gqa=4, causal=True,
               sm_scale=1.0):
    # STAGE 3 = causal (off-band then on-band); STAGE 1 = full, unmasked.
    # Z/H/N_CTX are constexpr on Spyre (delta 7), so a shape change is a recompile.
    return {"Z": z, "H": h, "N_CTX": n_ctx,
            "HEAD_DIM": head_dim, "BLOCK_M": block_m, "BLOCK_N": BLOCK_N,
            "GQA": gqa, "STAGE": 3 if causal else 1,
            "sm_scale": sm_scale}
