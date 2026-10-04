"""The Granite SwiGLU MLP for Spyre, DERIVED FROM TRITON'S OWN KERNELS.

Sources, all in the Triton tree at `~/git/triton`:

  * `python/tutorials/03-matrix-multiplication.py::matmul_kernel` -- the canonical
    Triton matmul, and the one that already carries a FUSED ACTIVATION hook
    (`ACTIVATION: tl.constexpr` ... `accumulator = leaky_relu(accumulator)`). That is
    the projection structure.
  * `python/triton_kernels/triton_kernels/swiglu_details/_swiglu.py::compute_swiglu`
    -- the SwiGLU activation, whose body is
    `s = gelu / (1 + exp_ftz(-alpha * gelu)); return tl.fma(s, linear, s)`. Note the
    activation is spelled as a DIVIDE, not as a multiply by a reciprocal; that is kept.
  * `python/triton_kernels/bench/bench_mlp.py::run_mlp` -- the arrangement. Its own
    comments read "first matmul + swiglu" (`fused_activation=act1`) then "second
    matmul". This file is that arrangement fused into ONE kernel.

WHY DERIVE RATHER THAN WRITE ONE, and it is the same lesson as
`attention_flash.py`: both projections and the down projection must reach the emitter
as `tl.dot`, because that is what `spyre-dot-to-linalg` rewrites to `linalg.matmul` and
what the emitter has a `batchmatmul` builder for. Expressing a contraction as a
broadcast-multiply plus `tl.sum` emits ZERO `tt.dot` and lands on the one shape there is
no builder for. All three contractions here are `tl.dot`.

THE DELTA, and the justification for each item
==============================================
Every item is either MEASURED (a diagnostic is quoted) or sourced to a definition. The
two candidate deltas that turned out NOT to be needed are recorded as such, at 6, so
nobody re-adds them.

1. GPU-ONLY MACHINERY REMOVED. From the tutorial: `@triton.autotune` + its config
   list, the `GROUP_SIZE_M` L2-reuse swizzle (`num_pid_in_group` / `group_id` /
   `first_pid_m`), the eight `tl.assume` bound hints, `num_stages` / `num_warps`. From
   `_swiglu`: the persistent grid-stride `for pid in tl.range(tl.program_id(0),
   M_BLOCKS * N_BLOCKS, tl.num_programs(0), num_stages=2)` loop, the whole flexpoint
   layer (`load_scale` / `float_to_flex` / `update_scale` / `thread_local_absmax` /
   `OutExpectedScale` / `OutActualScale` / `OutChecksumScale` /
   `flexpoint_saturate_inf`), the `tl.inline_asm_elementwise("ex2.approx.ftz.f32")`
   fast path and its `tl.target_info.is_cuda()` guard, and
   `tl.extra.cuda.num_threads()`. None of it describes the computation: it is a CUDA
   search space, a CUDA numerics scheme and CUDA codegen hints. Spyre compiles one
   configuration.

2. ACCUMULATORS ARE f16, NOT f32. The tutorial says in as many words "We accumulate
   into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of fp32 values for higher accuracy", and
   `compute_swiglu` opens by widening BOTH branches with `.to(tl.float32)`. Neither has
   a form here, for the reasons recorded at length in `attention_flash.py`'s delta 2:
   device floats are DL16-f16, `LegalizeTypes` RED-STOPS genuine f32 compute by design,
   and `arch.elemsPerStick("f32")` is 32 against 64 for f16 so an f32 tile spans two
   sticks. `out_dtype=tl.float16` on the two projections; the down projection infers
   f16 from `acc`.

3. TILES ARRIVE THROUGH DESCRIPTORS, NOT THROUGH POINTER ARITHMETIC AND MASKS. This is
   the tutorial's biggest single delta and it is MEASURED, not argued. The tutorial
   builds a block of pointers and loads it under a boundary mask:

       a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
       a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)

   Run through this backend's own pipeline, that form is REFUSED BY NAME, and by the
   very first pass:

       error: spyre-dot-to-linalg: matmul is not the canonical faithfully-lowerable
       kernel, so it is refused rather than silently miscompiled (review P1.0):
       A and B must be direct tt.descriptor_load results

   (Reproduced on exactly the tutorial's load form reduced to one tile with constexpr
   shapes, so nothing else can be the cause -- see the findings file.) Two further
   reasons the pointer form cannot work even if that guard were relaxed: the boundary
   mask is a LANE-WISE INTEGER COMPARE on `tl.arange` vectors, which this device has
   not got (`attention_flash.py` delta 10 measured the same thing and precomputes its
   mask on the host); and the tutorial's `% M` / `% N` wraparound on the offsets is a
   non-affine address, so it has no bundle address expression. `tl.make_tensor_descriptor`
   lowers to `ktdp.construct_memory_view` + `construct_access_tile` + `ktdp.load`,
   which is what the emitter reads.

4. M / D_MODEL / D_FF AND ALL SIX STRIDES ARE constexpr, NOT RUNTIME ARGS
   (SPYRE-SPECIFIC). The tutorial passes them as runtime i32 because a CUDA kernel is
   shape-generic. Spyre is not: affine loop bounds are static and the emitter derives
   the corelet work division from the descriptor extents. Same ground as
   `attention_flash.py` delta 7 -- one kernel per shape, which is already true of every
   other fixture here. (The strides go with them: a runtime stride would be ignored by
   a shape-keyed emitter, which `test_stages.py::test_compile_matmul_bad_stride_is_refused`
   exists to prevent.)

5. THE SIGMOID IS OPEN-CODED, BECAUSE `tl.sigmoid` REFUSES AN f16 TENSOR OUTRIGHT.
   MEASURED:

       tl.sigmoid(tl.zeros([64, 64], tl.float16))
       -> ValueError: Expected dtype ['fp32', 'fp64'] but got fp16

   `tl.sigmoid` is a Python-level `@triton.jit` helper -- `1 / (1 + math.exp(-x))`,
   `triton/language/standard.py:49` -- and `math.exp` is decorated
   `@_check_dtype(dtypes=["fp32", "fp64"])` (`triton/language/math.py:96`). So on a
   device whose compute dtype is f16, Triton's own sigmoid is UNUSABLE as written, and
   the widen-transcendental-truncate island has to be spelled by hand:
   `tl.exp((-g).to(tl.float32)).to(tl.float16)`. That is the same island
   `attention_flash.py` spells for `exp2`, and `LegalizeTypes` collapses it: the emitted
   KTIR carries `math.exp ... : tensor<64x64xf16>` and NO `arith.extf` / `arith.truncf`
   at all (measured on the checked-in KTIR).

6. THE DIVIDE IS `tl.fdiv`, NOT `/`. MEASURED, and this one is a compiler fact worth
   knowing beyond this kernel:

       f16 / f16          -> fp32
       tl.fdiv(f16, f16)  -> fp16

   The promotion is `computation_type_impl` rule 2 (`triton/language/semantic.py:83-86`)
   and its stated reason is "div and mod are not supported for floats with bitwidth
   < 32". That is a PTX fact and it is FALSE on this device: `OpFuncs::REALDIV` is bound
   by `broadcast_ops.ddl`, and `arith.divf` already maps to `"realdiv"` in
   `triton-superdsc-lower/src/opmap.rs`. Left alone the promotion is not merely
   wasteful, it makes the kernel ILLEGAL -- the f32 quotient reaches the down
   projection's `tl.dot` beside an f16 weight and the FRONTEND rejects it:
   `Both operands must be same dtype. Got fp32 and fp16`. `tl.fdiv` is the same
   `create_fdiv` with `arithmetic_check` off.

   *** NOTE FOR THE ATTENTION FIXTURE: its epilogue `acc = acc / l_i[:, None]` has the
   SAME latent promotion, so its divide is an f32 island where a `realdiv` would do. ***

   AND TWO THINGS THAT ARE NOT DELTAS, measured so they are not re-added:
     * `tl.zeros([...], tl.float16) + 1.0` stays **f16** -- a scalar of equal kind does
       not participate in promotion (`computation_type_impl` rule 0). `tl.full(...,
       tl.float16)` below is clarity, NOT necessity.
     * `-g` on an f16 tensor stays f16, and reaches KTIR as `arith.negf`.

7. `alpha` AND `limit` DROPPED. `compute_swiglu` takes a runtime `alpha` that multiplies
   the sigmoid argument and a runtime `limit` that clamps both branches. Granite's
   SwiGLU is alpha = 1 with no clamp, and `alpha` would not lower anyway: it is a
   runtime scalar reaching a compute op as an OPERAND, which is exactly
   `attention_flash.py` delta 9 -- refused BY NAME by `materializeSplatInputs`, because
   a compute group can only read values it LOADS and dcc's constant-operand encoding has
   four legal immediate values. `limit` WOULD map (`MAXIMUM`/`MINIMUM` are
   `broadcast_ops.ddl`); it is dropped because it is not in the model, not because it
   cannot lower.

8. THE `+1` ON THE UP BRANCH IS DROPPED. Upstream returns `tl.fma(s, linear, s)`, whose
   own comment reads `(s * (linear + 1))` -- the gpt-oss "SwiGLU with a unit bias on the
   linear branch" variant. Granite and Llama compute `silu(gate) * up`, and so does this
   repo's ALREADY-VALIDATED reference: `test/pod/swiglu_oracle.py` states the op as
   `h_i = silu(g_i) * u_i`. Keeping the `+1` would put this kernel outside the oracle we
   already own, which is the opposite of the point.

9. GATE AND UP ARE TWO WEIGHT TENSORS, NOT ONE INTERLEAVED PROJECTION. Upstream's
   activation receives `A` at width `2 * BLOCK_N` with the two branches interleaved and
   unpacks them with `tl.split(tl.reshape(a_packed, (BLOCK_M, BLOCK_N, 2)))`. Two
   reasons that cannot come along: the projections are IN this kernel, so there is no
   packed activation to unpack; and the emitter's only rank changes are
   `tensor.expand_shape` / `collapse_shape` with a DEGENERATE dimension (its
   `expand_shape` requires `dims.iter().position(|d| *d == 1)`), so a rank-3
   `[BLOCK_M, BLOCK_N, 2]` reshape has no SuperDSC form at all.

10. THE TWO PROJECTIONS, THE ACTIVATION AND THE DOWN PROJECTION ARE ONE LOOP OVER d_ff.
    Upstream is three separate launches (`run_mlp`: matmul+swiglu, then matmul), each
    tiling both of its own axes and materialising the `[M, d_ff]` intermediate in HBM.
    Fused as below, that intermediate NEVER EXISTS: only `[BLOCK_M, BLOCK_N]` does. At
    Granite width that is the difference between 1.6 MiB and 8 KiB per token block. This
    is a Spyre-motivated restructure, and it is why the kernel reads as one loop rather
    than as three kernels.

11. THE LAUNCH CONTRACT IS PLAIN ROW-MAJOR -- AND THAT IS THE HEADLINE, BY CONTRAST
    WITH ATTENTION. All five tensors are presented `[rows, cols]` contiguous row-major.
    NO transposition, NO stick-wide splat source, NO lane-padding discipline, NO
    64x-widened operand.

    WHY, and it is structural rather than lucky: attention's delta 8 layout is forced by
    its ROW REDUCTIONS. A reduction wider than the compute unit's vector becomes a TREE
    of folds and a fold's second tile must start on a stick boundary, so the reduced axis
    has to sit directly above the lane axis -- which transposes the score tile and
    everything feeding it. **THIS KERNEL HAS NO REDUCTION AT ALL**: no `tl.max`, no
    `tl.sum`, no `tt.reduce`. Only contractions, and a contraction's layout is
    `bmm.ddl`'s, which the emitter reads off the DDL rather than choosing. Measured
    consequence: every `access_tile_order` in the emitted KTIR is the identity
    `affine_map<(d0, d1) -> (d0, d1)>` on all six tiles, so attention's delta 6 (the
    transposing order map, which is still an unsettled host obligation there) does not
    arise here either.

    THE ONE HOST OBLIGATION, and `manifest.json` names it: the down-projection
    accumulator is loop-carried, so it is a caller-seeded buffer of `[BLOCK_M, D_MODEL]`
    f16 zeros. That is the whole caller contract beyond the five pointers.

    ONE NOTE ON WEIGHT ORIENTATION -- SEE DELTA 13, WHICH CORRECTS WHAT THIS PARAGRAPH
    USED TO SAY. All three weights are presented `[out, in]`, which is exactly
    `torch.nn.Linear.weight`, so the host owes NO transpose at all.

12. BLOCK_K IS A REAL KNOB AND BOTH SETTINGS LOWER (measured, both to a program):
      * `BLOCK_K == D_MODEL` -- the inner loop is single-trip and `canonicalize`
        deletes it, so the emission is identical to writing one `tl.dot` per projection.
      * `BLOCK_K < D_MODEL` -- a genuinely NESTED `scf.for`, which is the first nested
        loop this emitter has been asked for. It found a defect (two loops both naming
        their carried buffer `carried0`, which MLIR rejects as a repeated region entry
        argument); that is fixed, with a fail-closed guard, and recorded in the findings
        file. Both settings are kept as configurations rather than as two kernels, so
        there is one source of truth for the shapes.

13. ALL THREE WEIGHTS ARE n-MAJOR AND ALL THREE DOT OPERANDS CARRY `.T`. The down
    projection was the odd one out and it contradicted THIS FILE'S OWN delta 11: `wg`/`wu`
    are declared `[D_FF, D_MODEL]` and loaded `.T`, while `wd` was declared
    `[D_FF, D_MODEL]` and loaded WITHOUT `.T` -- which for the down projection is `[k, n]`,
    the opposite orientation. It is now `[D_MODEL, D_FF]` loaded `[0, n]` with `.T`.

    WHY THIS IS THE ORIENTATION, and it is device evidence rather than a preference. The
    two spellings reach `ktir-superdsc` as two different `linalg.matmul` forms -- an absent
    `indexing_maps` IS MLIR's default `[(d0,d2), (d2,d1), (d0,d1)]`, i.e. W as `[k, n]`,
    against the explicit `[(d0,d2), (d1,d2), (d0,d1)]` that says W as `[n, k]`. Nothing in
    that crate reads the maps, so BOTH forms get the same descriptor and one of them must
    therefore be computing the transposed contraction. The DDLs cannot settle which:
    `bmm.ddl`'s layout is unreachable from V1 (the dataflow scheduler has no contraction op
    at all -- `KTIRLegalityCheck.cpp:103`, "V1 only supports add/mul/sub compute ops"), what
    is emitted instead is a multiply plus a fold tree governed by the ELEMENTWISE templates,
    and `broadcast_ops.ddl:6-7` fixes no axis order. So the only pairing with device
    evidence behind it is scratchy's -- `[n, k]` bytes read with transpose-B maps, Granite
    at 41 tok/s -- and every weight here is presented to match it.

    THE HOST CONTRACT, corrected: `Wg`/`Wu` are `[D_FF, D_MODEL]` and `Wd` is
    `[D_MODEL, D_FF]`, i.e. all three are `[out_features, in_features]` -- precisely how
    `torch.nn.Linear.weight` is already stored. Delta 11 used to claim the opposite
    (`[in, out]`, "the host transposes once at weight-load time") and that was wrong in both
    halves: it named the orientation the code does not use, and it charged the host for a
    transpose it does not owe.

NOT CHANGED, deliberately: all three `tl.dot` legs, the activation written as a DIVIDE
exactly as `compute_swiglu` writes it, the accumulate-into-`acc` third operand of the
down projection, and descriptor loads.

MoE IS NOT IN THIS KERNEL, BY DESIGN, on the same grounds as paging in
`attention_flash.py`: expert routing is a gather, `agen.indirect_vector_load` is
LXLU-only by construction, and the host resolves the expert assignment and presents the
weights it selected. There is no `expt_map` here and there should not be one. That makes
this the same kernel for a dense MLP and for one expert of an MoE.
"""

import triton
import triton.language as tl


@triton.jit
def swiglu_mlp_fwd(desc_x, desc_wg, desc_wu, desc_wd, desc_o,  #
                   M: tl.constexpr, D_MODEL: tl.constexpr,  # SPYRE: delta 4
                   D_FF: tl.constexpr,  #
                   BLOCK_M: tl.constexpr,  #
                   BLOCK_N: tl.constexpr,  #
                   BLOCK_K: tl.constexpr,  # SPYRE: delta 12
                   ):
    start_m = tl.program_id(0)
    # SPYRE: delta 3 -- descriptors, not pointer blocks. The tutorial's masked
    # `tl.load` of `a_ptr + offs[:, None] * stride + ...` is refused by
    # `spyre-dot-to-linalg` ("A and B must be direct tt.descriptor_load results").
    # SPYRE: delta 11 -- every one of these is plain contiguous row-major. The weights
    # are [in, out]; a torch `[out, in]` weight is transposed once at load time.
    x_desc = tl.make_tensor_descriptor(desc_x, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    wg_desc = tl.make_tensor_descriptor(desc_wg, shape=[D_FF, D_MODEL],
                                        strides=[D_MODEL, 1],
                                        block_shape=[BLOCK_N, BLOCK_K])
    wu_desc = tl.make_tensor_descriptor(desc_wu, shape=[D_FF, D_MODEL],
                                        strides=[D_MODEL, 1],
                                        block_shape=[BLOCK_N, BLOCK_K])
    # SPYRE: delta 13 -- n-major like wg/wu, so its `tl.dot` operand carries `.T` too.
    wd_desc = tl.make_tensor_descriptor(desc_wd, shape=[D_MODEL, D_FF],
                                        strides=[D_FF, 1],
                                        block_shape=[D_MODEL, BLOCK_N])
    o_desc = tl.make_tensor_descriptor(desc_o, shape=[M, D_MODEL],
                                       strides=[D_MODEL, 1],
                                       block_shape=[BLOCK_M, D_MODEL])

    offs_m = start_m * BLOCK_M
    # f16 accumulators (delta 2). The down projection's accumulator is the kernel's one
    # loop-carried value, so it is the one thing the caller must seed (delta 11).
    acc = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float16)
    # The sigmoid's `1`, in the compute dtype. `tl.zeros(...) + 1.0` would also stay
    # f16 (delta 6); this is just less to check when reading.
    one = tl.full([BLOCK_M, BLOCK_N], 1.0, tl.float16)

    # SPYRE: delta 10 -- ONE loop over d_ff does gate, up, activation and the down
    # projection's accumulation, so the [BLOCK_M, D_FF] intermediate never exists.
    for n in tl.range(0, D_FF, BLOCK_N):
        # gate and up projections. `out_dtype` is f16 (delta 2): Triton defaults
        # `tl.dot` to an f32 accumulator, but the device MAC rounds to DL16 at every
        # step, so there is no wide accumulator to keep.
        g = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        u = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        for k in tl.range(0, D_MODEL, BLOCK_K):  # SPYRE: delta 12
            x = x_desc.load([offs_m, k])
            g = tl.dot(x, wg_desc.load([n, k]).T, g)
            u = tl.dot(x, wu_desc.load([n, k]).T, u)

        # SwiGLU. `compute_swiglu`'s own form -- a DIVIDE, not a reciprocal multiply --
        # with three Spyre deltas on it and nothing else:
        #   delta 5: `tl.sigmoid` cannot be called on an f16 tensor at all
        #            (`math.exp` is @_check_dtype(["fp32","fp64"])), so the widen /
        #            truncate island is written out. LegalizeTypes collapses it.
        #   delta 6: `tl.fdiv`, because `/` upcasts f16 to f32 for a reason that is a
        #            PTX fact and false here -- and the upcast then makes the kernel
        #            illegal at the down projection below.
        #   delta 7: no `alpha` (a runtime scalar has no lowering) and no `limit` clamp.
        # The emitter RECOGNISES this four-op chain and emits ONE templated `silu`
        # SuperDSC for it (OpFuncs::SILU_FWD -> unary_parallel.ddl), whose device body
        # is the same sigmoid opaque over the same six constants this repo's SwiGLU
        # reference already validated. Writing the activation any other way still
        # lowers, but as four plain SuperDSCs computing a DIFFERENT algorithm, to which
        # the derived tolerance would not apply.
        e = tl.exp((-g).to(tl.float32)).to(tl.float16)
        s = tl.fdiv(g, one + e)
        # delta 8: `silu(gate) * up`, NOT upstream's `s * (up + 1)`.
        h = s * u

        # down projection, accumulating over d_ff blocks. f16 is inferred from `acc`.
        # SPYRE: delta 13 -- `.T`, as on both projections above. The tile is `[D_MODEL, BLOCK_N]`
        # = `[n, k]` and `.T` makes the dot operand `[BLOCK_N, D_MODEL]` = `[k, n]`.
        acc = tl.dot(h, wd_desc.load([0, n]).T, acc)

    o_desc.store([offs_m, 0], acc)


# --- the Spyre configuration ---------------------------------------------------
# BLOCK_N is one f16 stick (64). Unlike `attention_flash.py`, where 64 is a
# CORRECTNESS constraint (reduce-MAX mis-combines partial maxima when `rows > 1` AND
# the reduced extent passes one stick of columns -- a CONJUNCTION; see that file for
# the two shipping counter-examples that refute each single-axis reading), here it is
# only the tile width of an elementwise activation -- there is no reduction to keep
# inside a stick at all (delta 11). It stays 64 because the stick-axis extent
# must be a multiple of 64, which the emitter checks and refuses by name.
BLOCK_N = 64

SIGNATURE = {
    "desc_x": "*fp16", "desc_wg": "*fp16", "desc_wu": "*fp16",
    "desc_wd": "*fp16", "desc_o": "*fp16",
    "M": "constexpr", "D_MODEL": "constexpr", "D_FF": "constexpr",
    "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
}


def constexprs(m=64, d_model=128, d_ff=256, block_m=64, block_k=None):
    """One configuration. `block_k=None` means "do not tile d_model" (delta 12).

    Granite is `d_model=4096, d_ff=12800` -- 200 sticks of d_ff. Shapes are constexpr
    (delta 4), so a shape change is a recompile.
    """
    return {"M": m, "D_MODEL": d_model, "D_FF": d_ff,
            "BLOCK_M": block_m, "BLOCK_N": BLOCK_N,
            "BLOCK_K": d_model if block_k is None else block_k}


# Granite-3 8B: d_model 4096, d_ff 12800.
GRANITE = dict(d_model=4096, d_ff=12800)
