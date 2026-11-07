// SPDX-License-Identifier: Apache-2.0
//! CUDA's step: one or two kernel launches of a lowered tape, the counterpart of metal's
//! `MetalStep`.

/// Re-exported so a static tape names it beside the steps that carry it
/// (`scratchy_target_cuda::step::GainConvention`).
pub use scratchy_subtile::handoff::RopePairing;
pub use scratchy_subtile::subtile_ir::GainConvention;
use serde::Serialize;

use crate::tape::{ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, WeightIx};

/// One launch-emitting step of a lowered tape, its operands already resolved: one kernel launch,
/// or two stream-ordered launches (a `RopeAppend` rotates K, then writes the cache).
///
/// It gains one variant per opcode CUDA lowers, each added by the issue that writes that opcode's
/// kernel, beside that kernel's registry row and its arm of [`CudaStep::extents`]. The tape
/// player matches it exhaustively, so a new variant does not compile until the player launches
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum CudaStep {
    /// The token-embedding row gather, the forward's first step when its input is token ids
    /// (`kernels/embed_bf16.cu`): metal's `MetalStep::Embed`. Not lowered from an op of the
    /// shared tape: it fills the embedded hidden, colour 0, which the tape's first ops read.
    Embed(EmbedBf16),
    /// The bf16 GEMM, lowered from `SubOp::MatmulTile` when `n` is a multiple of its tile's
    /// (`kernels/gemm_bf16.cu`).
    Gemm(GemmBf16),
    /// The narrow bf16 GEMM, lowered from `SubOp::MatmulTile` when `n` is at most
    /// [`GEMM_NARROW_BF16_MAX_N`] (`kernels/gemm_narrow_bf16.cu`): the operands of
    /// [`Self::Gemm`], another kernel. The lowering's `MatmulTile` arm chooses it at expansion.
    GemmNarrow(GemmBf16),
    /// The bf16 GEMV, a one-row GEMM, lowered from `SubOp::MatmulTile` of a one-row bucket whose
    /// `n` the narrow GEMM does not take (`kernels/gemv_bf16.cu`): the operands of [`Self::Gemm`]
    /// with `m = 1`, another kernel. The lowering's `MatmulTile` arm chooses it at expansion.
    Gemv(GemmBf16),
    /// The bf16 RMSNorm, lowered from `SubOp::RmsNorm` (`kernels/rms_norm_bf16.cu`).
    RmsNorm(RmsNormBf16),
    /// The bf16 elementwise add, lowered from `Elementwise(Add)` (`kernels/add_bf16.cu`).
    Add(AddBf16),
    /// `silu(gate) · up` in one launch, lowered from the `SiluMul` fold of an `Elementwise(Silu)`
    /// into the `Elementwise(Mul)` it gates (`kernels/silu_mul_bf16.cu`).
    SiluMul(SiluMulBf16),
    /// The bf16 partial rotary embedding, lowered from `SubOp::RopeRotate`
    /// (`kernels/rope_bf16.cu`).
    Rope(RopeBf16),
    /// The per-head split of a doubled `q_proj` output into query and gate, lowered from
    /// `SubOp::GateSplit` (`kernels/gate_split_bf16.cu`).
    GateSplit(GateSplitBf16),
    /// `attn · sigmoid(gate)`, lowered from `SubOp::GateApply` (`kernels/gate_apply_bf16.cu`).
    GateApply(GateApplyBf16),
    /// K's rope, then K and V written into the paged KV cache, lowered from `SubOp::RopeAppend`:
    /// two launches, `kernels/rope_bf16.cu` then `kernels/kv_append_bf16.cu`.
    RopeAppend(RopeAppendBf16),
    /// Causal grouped-query attention of a prefill bucket over the paged KV cache, lowered from
    /// `SubOp::AttnDecode` when the bucket has more than one row (`kernels/attn_prefill_bf16.cu`).
    AttnPrefill(AttnPrefillBf16),
    /// One query row's grouped-query attention over the paged KV cache, lowered from
    /// `SubOp::AttnDecode` when the bucket has one row (`kernels/attn_decode_bf16.cu`): the
    /// operands of [`Self::AttnPrefill`] with `rows = 1`, another kernel.
    AttnDecode(AttnPrefillBf16),
    /// One prefill chunk of a Gated-DeltaNet linear-attention layer, lowered from
    /// `SubOp::GatedDeltaNet`: four stream-ordered launches, `kernels/gdn_conv1d.cu`,
    /// `kernels/gdn_gating.cu`, `kernels/gdn_scan.cu` then `kernels/gdn_rms_norm_gated.cu`.
    GatedDeltaNet(GatedDeltaNetBf16),
    /// A MoE block's router, `[m, experts]` logits into the MoE scratch, lowered from
    /// `SubOp::RouterLogits`: the tiled GEMM or the GEMV (`kernels/gemm_bf16.cu`,
    /// `kernels/gemv_bf16.cu`, chosen at expansion) over the router at byte 0 of the bundle.
    RouterLogits(RouterLogitsBf16),
    /// A MoE block's routing, lowered from the `MoeRoute` fold driven by `SubOp::RouteTopK`:
    /// softmax, top-k and renorm in one launch (`kernels/moe_route_bf16.cu`).
    MoeRoute(MoeRouteBf16),
    /// The pairs sorted by expert, lowered from `SubOp::ExpertSort` at
    /// `op_abi::CUDA_SORTED_PAIRS` pairs or more: three launches of a stable counting sort
    /// (`kernels/moe_sort.cu`).
    ExpertSort(ExpertSortBf16),
    /// One projection of every sorted pair, lowered from `SubOp::ExpertMatmul` of a sorted bucket:
    /// one grouped GEMM over the expert-padded tiles (`kernels/expert_gemm_bf16.cu`).
    ExpertGemm(ExpertGemmBf16),
    /// One projection of every pair, each from its own token row, lowered from
    /// `SubOp::ExpertMatmul` of a gathered bucket (`kernels/expert_gemv_bf16.cu`).
    ExpertGemv(ExpertGemvBf16),
    /// Each token's pair rows summed by its routing weights, lowered from `SubOp::ExpertCombine`
    /// (`kernels/moe_combine_bf16.cu`).
    ExpertCombine(ExpertCombineBf16),
    /// `routed + shared · sigmoid(g)`, `g` broadcast across the row, lowered from
    /// `SubOp::GateScale` (`kernels/gate_scale_bf16.cu`).
    GateScale(GateScaleBf16),
    /// `x · scale`, a compile-time scalar (Gemma's embedding scale), lowered from
    /// `SubOp::ScalarMul` (`kernels/pointwise_bf16.cu`, `scratchy_cuda_scale_bf16`).
    Scale(ScaleBf16),
    /// The RMSNorm with no gain (Gemma's `v_norm`), lowered from `SubOp::RmsNormUnit`
    /// (`kernels/rms_norm_bf16.cu`'s unit instantiation, `scratchy_cuda_rms_norm_unit_bf16`).
    RmsNormUnit(RmsNormUnitBf16),
    /// `gelu_tanh(gate) · up` in one launch, lowered from the `GeluMul` fold of an
    /// `Elementwise(Gelu)` into the `Elementwise(Mul)` it gates, and from Gemma's
    /// `ExpertGatedAct { act: Gelu }`: the operands of [`Self::SiluMul`]
    /// (`kernels/silu_mul_bf16.cu`'s GELU instantiation, `scratchy_cuda_gelu_mul_bf16`).
    GeluMul(SiluMulBf16),
    /// `x · w[0]`, `w` a `[1]` bf16 weight (Gemma's `layer_scalar`), lowered from
    /// `SubOp::ScalarWeightMul` (`kernels/pointwise_bf16.cu`, `scratchy_cuda_layer_scale_bf16`).
    LayerScale(LayerScaleBf16),
    /// `cap · tanh(x / cap)` (Gemma's final logits), lowered from `SubOp::TanhSoftCap`
    /// (`kernels/pointwise_bf16.cu`, `scratchy_cuda_soft_cap_bf16`).
    SoftCap(SoftCapBf16),
    /// Gemma's router pre-norm, `x̂ · router.scale` into the MoE scratch's router-norm region,
    /// lowered from `SubOp::RouterNorm`: the operands of [`Self::RmsNorm`], its gain the bundle's
    /// `router.scale` (`kernels/rms_norm_bf16.cu`, `scratchy_cuda_rms_norm_bf16`).
    RouterNorm(RmsNormBf16),
    /// The bf16 GEMM on 64-column tiles, lowered from `SubOp::MatmulTile` when `n` is a multiple
    /// of [`GEMM_COL64_TILE`]'s and not of [`GEMM_BF16_TILE`]'s, above
    /// [`GEMM_NARROW_BF16_MAX_N`]: the operands of [`Self::Gemm`], `kernels/gemm_bf16.cu`'s
    /// template at one 64-column B block (`scratchy_cuda_gemm_col64_bf16_abt`, #924).
    GemmCol64(GemmBf16),
    /// [`Self::Gemm`] of up to three weights off one input in one launch, lowered from the
    /// `SiblingMatmuls` fold (#1049): the q/k/v projections, or a GDN layer's `in_proj_qkv` and
    /// `in_proj_z` (`kernels/gemm_bf16.cu`'s grouped template, `scratchy_cuda_gemm_group_bf16_abt`).
    /// Each member's output is the bits of its own [`Self::Gemm`].
    GemmGroup(GemmGroupBf16),
    /// [`Self::GemmNarrow`] of up to three weights off one input in one launch, lowered from the
    /// `SiblingMatmuls` fold: a GDN layer's `in_proj_a` and `in_proj_b`
    /// (`kernels/gemm_narrow_bf16.cu`, `scratchy_cuda_gemm_narrow_group_bf16_abt`). Each member's
    /// output is the bits of its own [`Self::GemmNarrow`].
    GemmNarrowGroup(GemmGroupBf16),
    /// [`Self::Gemv`] of up to three weights off one input in one launch, lowered from the
    /// `SiblingMatmuls` fold of a one-row bucket (`kernels/gemv_bf16.cu`,
    /// `scratchy_cuda_gemv_group_bf16_abt`). Each member takes its own K split
    /// ([`gemv_splits`]), its partials at its own place in the GEMV scratch, so its output is the
    /// bits of its own [`Self::Gemv`].
    GemvGroup(GemmGroupBf16),
    /// `sum = a + b`, then `y = rmsnorm(sum)`, in one launch, lowered from the `AddRmsNorm` fold
    /// of an `Elementwise(Add)` into the `RmsNorm` reading it (#1049, Qwen's residual add and the
    /// next norm; `kernels/rms_norm_bf16.cu`, `scratchy_cuda_add_rms_norm_bf16`). Both outputs
    /// are the bits of today's [`Self::Add`] then [`Self::RmsNorm`].
    AddRmsNorm(AddRmsNormBf16),
    /// `out = residual + rmsnorm(x)` in one launch, lowered from the `NormAdd` fold of an
    /// `RmsNorm` into the `Elementwise(Add)` it alone feeds (#1049, Gemma's post-norms;
    /// `kernels/rms_norm_bf16.cu`, `scratchy_cuda_norm_add_bf16`): the bits of today's
    /// [`Self::RmsNorm`] then [`Self::Add`].
    NormAdd(NormAddBf16),
    /// `out = act(bf16(a · gateᵀ), bf16(a · upᵀ))` in one launch, lowered from the `Gated` fold
    /// with its projections folded in (#1049): a dense SwiGLU or GeGLU MLP's gate and up GEMMs
    /// and their activation (`kernels/gemv_bf16.cu`'s gate/up GEMV at one row,
    /// `kernels/gemm_bf16.cu`'s gate/up GEMM otherwise). The bits of the two GEMMs (or GEMVs)
    /// then [`Self::SiluMul`] (or [`Self::GeluMul`]).
    GateUpAct(GateUpActBf16),
    /// The routed experts' gate and up GEMVs and their activation in one launch, lowered from the
    /// `ExpertActivation` fold of a one-row bucket (#1049, `kernels/gemv_bf16.cu`'s expert gate/up
    /// GEMV): the bits of the two [`Self::ExpertGemv`]s then the activation's step.
    ExpertGateUpAct(ExpertGateUpActBf16),
    /// A one-row bucket's [`Self::GatedDeltaNet`] in one launch (#1049,
    /// `kernels/gdn_decode.cu`): its operands, its conv, gating, scan and gated norm as the four
    /// kernels compute them at one token, so its output, conv ring and state are their bits. The
    /// intermediates stay in shared memory: it reads and writes no GDN scratch.
    GdnDecode(GatedDeltaNetBf16),
    /// The per-token dynamic e4m3 quantization of an fp8 GEMM's bf16 input (FP8 S2, its kernel
    /// S3's): each of `rows` rows of `x` to e4m3 by `max(amax / 448, 1 / (448·512))`, into the
    /// tape's quant scratch. Inserted by CUDA's lowering before every fp8 consumer of more than
    /// one row, once per group; a one-row GEMV quantizes in its prologue.
    QuantRowsFp8(QuantRowsFp8),
    /// [`Self::Gemm`] over an fp8 weight ([`Fp8Weight`]): `a` is the quant scratch, the e4m3 rows
    /// [`Self::QuantRowsFp8`] wrote, each row's f32 scale after them; `C = (A·Bᵀ) · w_scale[n] ·
    /// a_scale[m]`, rounded once to bf16. Its kernel is S3's.
    GemmFp8(GemmBf16),
    /// [`Self::GemmCol64`] over an fp8 weight, as [`Self::GemmFp8`].
    GemmCol64Fp8(GemmBf16),
    /// [`Self::GemmNarrow`] over an fp8 weight (the 27B's `in_proj_a`/`b`), as [`Self::GemmFp8`].
    GemmNarrowFp8(GemmBf16),
    /// [`Self::Gemv`] over an fp8 weight: `a` is the bf16 row, which the GEMV quantizes in its
    /// prologue (`kernels/gemv_fp8.cu`).
    GemvFp8(GemmBf16),
    /// [`Self::GemmGroup`] over fp8 weights, as [`Self::GemmFp8`] each.
    GemmGroupFp8(GemmGroupBf16),
    /// [`Self::GemmNarrowGroup`] over fp8 weights (the 27B's `in_proj_a` + `in_proj_b`), as
    /// [`Self::GemmFp8`] each.
    GemmNarrowGroupFp8(GemmGroupBf16),
    /// [`Self::GemvGroup`] over fp8 weights, as [`Self::GemvFp8`] each.
    GemvGroupFp8(GemmGroupBf16),
}

/// An fp8 GEMM's or GEMV's `k` is a non-zero multiple of this: its K step (64 e4m3 bytes, one
/// 64-byte-swizzled row, two `k32` WGMMAs). Every fp8 `k` of the three FP8 checkpoints is one
/// (Gemma's down projections, 2112 and 704, are not multiples of 128).
pub const GEMM_FP8_K_MULTIPLE: u32 = 64;

/// One fp8 weight's device buffer ([`crate::weights::CudaWeight::Fp8`]): the checkpoint's e4m3
/// `[n, k]` bytes verbatim, then its per-channel scale as `[n]` f32 at the next
/// [`crate::tape::CUDA_ARENA_ALIGN`]-byte boundary. `None` on overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8Weight {
    pub scale: u64,
    pub bytes: u64,
}

/// [`Fp8Weight`] of an `[n, k]` projection.
pub const fn fp8_weight(n: u32, k: u32) -> Option<Fp8Weight> {
    let Some(scale) = (n as u64 * k as u64).checked_next_multiple_of(crate::tape::CUDA_ARENA_ALIGN)
    else {
        return None;
    };
    let Some(bytes) = scale.checked_add(n as u64 * 4) else {
        return None;
    };
    Some(Fp8Weight { scale, bytes })
}

/// `q [rows, cols]` e4m3 then `scale [rows]` f32 (at [`QuantRowsFp8::scale_at`]) from `x [rows,
/// cols]` bf16, in the tape's quant scratch, whose start `q` is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct QuantRowsFp8 {
    pub x: ArenaOffset,
    pub q: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

impl QuantRowsFp8 {
    /// The scales' byte offset from `q`: the e4m3 rows, rounded up to
    /// [`crate::tape::CUDA_ARENA_ALIGN`].
    pub const fn scale_at(&self) -> Option<u64> {
        (self.rows as u64 * self.cols as u64)
            .checked_next_multiple_of(crate::tape::CUDA_ARENA_ALIGN)
    }

    /// The quant scratch bytes it writes: the rows, then their scales.
    pub const fn bytes(&self) -> Option<u64> {
        match self.scale_at() {
            Some(at) => at.checked_add(self.rows as u64 * 4),
            None => None,
        }
    }
}

/// Why a tape's quantized activations are not each read by the consumer they were written for:
/// `step` reads the quant scratch with no `QuantRowsFp8` of its `(x, rows, cols)` before it since
/// the last quant write, or a `QuantRowsFp8` at `step` overwrites one no fp8 GEMM read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantChainError {
    /// An fp8 GEMM of more than one row whose input is not the live quantized rows.
    Unquantized { step: usize },
    /// A quant write that no fp8 GEMM reads before the next one overwrites it.
    Overwritten { step: usize },
}

/// The quant scratch's liveness over `steps`, in order: every multi-row fp8 GEMM step (or group,
/// or gate/up pair) reads exactly the rows the last [`CudaStep::QuantRowsFp8`] before it wrote
/// (at its `q`, of its `rows` and `cols`), with no other quant write between them; and every
/// quant write is read before the next. The one scratch is reused, the way the GEMV scratch is.
/// That the quant write read the consumer's own bf16 input is the lowering's to state: the
/// consumer names only the scratch.
pub fn quant_chain(steps: &[CudaStep]) -> Result<(), QuantChainError> {
    let mut live: Option<(usize, QuantRowsFp8, bool)> = None;
    for (step, s) in steps.iter().enumerate() {
        if let CudaStep::QuantRowsFp8(q) = *s {
            if let Some((at, _, false)) = live {
                return Err(QuantChainError::Overwritten { step: at });
            }
            live = Some((step, q, false));
            continue;
        }
        let Some((x, m, k)) = s.quantized_input() else {
            continue;
        };
        match live {
            Some((at, q, _)) if (q.q, q.rows, q.cols) == (x, m, k) => live = Some((at, q, true)),
            _ => return Err(QuantChainError::Unquantized { step }),
        }
    }
    match live {
        Some((at, _, false)) => Err(QuantChainError::Overwritten { step: at }),
        _ => Ok(()),
    }
}

/// The value heads per key head the one-launch GDN decode takes (`kernels/gdn_decode.cu`): one CTA
/// per key head owns its value heads, `nv / nk` of them; qwen3.5's are 1 (0.8B) and 2 (9B, 35B).
pub const GDN_DECODE_MAX_HEADS: u32 = 4;

/// The activation a gate/up step applies (`kernels/scratchy_ops.cuh`'s op struct): SwiGLU's
/// `silu_mul` or GeGLU's `gelu_tanh_mul`, the op of [`CudaStep::SiluMul`] or [`CudaStep::GeluMul`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GatedOp {
    Silu,
    Gelu,
}

impl GatedOp {
    /// The launchers' code for it (`kernels/gemv_bf16.cu`'s `ACT_SILU`, `ACT_GELU`).
    pub const fn code(self) -> u32 {
        match self {
            Self::Silu => 0,
            Self::Gelu => 1,
        }
    }

    /// The activation step of `m` on its own.
    pub const fn step(self, m: SiluMulBf16) -> CudaStep {
        match self {
            Self::Silu => CudaStep::SiluMul(m),
            Self::Gelu => CudaStep::GeluMul(m),
        }
    }
}

/// `out [m, n] = act(bf16(a · gateᵀ), bf16(a · upᵀ))`: `gate` and `up` are the GEMMs it folds,
/// each a [`GemmBf16`] over the same `a`, `m`, `n` and `k` (their `c` the colours today's two
/// launches write, which this one never writes); `kernel` the step each would be on its own, the
/// same for both ([`CudaStep::Gemm`], [`CudaStep::GemmCol64`] or [`CudaStep::Gemv`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateUpActBf16 {
    pub gate: GemmBf16,
    pub up: GemmBf16,
    pub out: ArenaOffset,
    pub kernel: GateUpKernel,
    pub act: GatedOp,
}

/// Which GEMM kernel a gate/up step's projections are on their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GateUpKernel {
    Gemm,
    GemmCol64,
    Gemv,
    /// The fp8 kernels (FP8 S2): each projection [`CudaStep::GemmFp8`], [`CudaStep::GemmCol64Fp8`]
    /// or [`CudaStep::GemvFp8`] on its own, each scaled by its own `w_scale` before the
    /// activation, rounded once.
    GemmFp8,
    GemmCol64Fp8,
    GemvFp8,
}

impl GateUpKernel {
    /// The step `g` is on its own.
    pub const fn step(self, g: GemmBf16) -> CudaStep {
        match self {
            Self::Gemm => CudaStep::Gemm(g),
            Self::GemmCol64 => CudaStep::GemmCol64(g),
            Self::Gemv => CudaStep::Gemv(g),
            Self::GemmFp8 => CudaStep::GemmFp8(g),
            Self::GemmCol64Fp8 => CudaStep::GemmCol64Fp8(g),
            Self::GemvFp8 => CudaStep::GemvFp8(g),
        }
    }
}

impl GateUpActBf16 {
    /// The activation it folds, over the projections' colours.
    pub const fn activation(&self) -> SiluMulBf16 {
        SiluMulBf16 {
            gate: self.gate.c,
            up: self.up.c,
            out: self.out,
            rows: self.gate.m,
            cols: self.gate.n,
        }
    }

    /// The bytes of GEMV scratch the fused GEMV writes: the gate's partials, then the up's, each
    /// at its own split ([`gemv_scratch_bytes`]); 0 for the GEMMs. `None` on overflow.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        let (GateUpKernel::Gemv | GateUpKernel::GemvFp8) = self.kernel else {
            return Some(0);
        };
        let (n, k) = (self.gate.n, self.gate.k);
        let Some(one) = gemv_scratch_bytes(n, 1, gemv_splits(n, k, 1)) else {
            return None;
        };
        one.checked_mul(2)
    }
}

/// The routed experts' gate and up GEMVs and their activation: `gate` and `up` the two
/// [`ExpertGemvBf16`]s it folds (the MoE scratch's gate and up regions, which this launch never
/// writes), `out` the activation rows, `act` the activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGateUpActBf16 {
    pub gate: ExpertGemvBf16,
    pub up: ExpertGemvBf16,
    pub out: ArenaOffset,
    pub act: GatedOp,
}

impl ExpertGateUpActBf16 {
    /// The activation it folds, over the gate and up regions.
    pub const fn activation(&self) -> SiluMulBf16 {
        SiluMulBf16 {
            gate: self.gate.c,
            up: self.up.c,
            out: self.out,
            rows: self.gate.m * self.gate.top_k,
            cols: self.gate.inter,
        }
    }

    /// The GEMV scratch the fused launch writes: the gate's partials, then the up's.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        let (n, k) = self.gate.n_k();
        let Some(pairs) = self.gate.m.checked_mul(self.gate.top_k) else {
            return None;
        };
        let Some(one) = gemv_scratch_bytes(n, pairs, gemv_splits(n, k, pairs)) else {
            return None;
        };
        one.checked_mul(2)
    }
}

/// `sum = a + b` (one f32 add, rounded once), then `y` the [`RmsNormBf16`] of `sum`: `a`, `b`,
/// `sum` and `y` `[rows, cols]` arena slots, `gain` the norm's `[cols]` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AddRmsNormBf16 {
    pub a: ArenaOffset,
    pub b: ArenaOffset,
    pub sum: ArenaOffset,
    pub norm: RmsNormBf16,
}

impl AddRmsNormBf16 {
    /// The add the fold absorbed.
    pub const fn add(&self) -> AddBf16 {
        AddBf16 {
            a: self.a,
            b: self.b,
            out: self.sum,
            rows: self.norm.rows,
            cols: self.norm.cols,
        }
    }
}

/// `out = residual + bf16(`[`RmsNormBf16`]`(x))`, the sum in f32 and rounded once: `norm.y` is
/// the add's output (the norm's own is never written), `residual` the add's other operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct NormAddBf16 {
    pub norm: RmsNormBf16,
    pub residual: ArenaOffset,
}

/// Which kernel a grouped GEMM groups (#1049): the step each member is on its own, and the
/// grouped step of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupKind {
    Gemm,
    GemmNarrow,
    Gemv,
    GemmFp8,
    GemmNarrowFp8,
    GemvFp8,
}

impl GroupKind {
    /// A member as the step it is on its own.
    pub const fn step(self, g: GemmBf16) -> CudaStep {
        match self {
            Self::Gemm => CudaStep::Gemm(g),
            Self::GemmNarrow => CudaStep::GemmNarrow(g),
            Self::Gemv => CudaStep::Gemv(g),
            Self::GemmFp8 => CudaStep::GemmFp8(g),
            Self::GemmNarrowFp8 => CudaStep::GemmNarrowFp8(g),
            Self::GemvFp8 => CudaStep::GemvFp8(g),
        }
    }

    /// The grouped step of `g`.
    pub const fn grouped(self, g: GemmGroupBf16) -> CudaStep {
        match self {
            Self::Gemm => CudaStep::GemmGroup(g),
            Self::GemmNarrow => CudaStep::GemmNarrowGroup(g),
            Self::Gemv => CudaStep::GemvGroup(g),
            Self::GemmFp8 => CudaStep::GemmGroupFp8(g),
            Self::GemmNarrowFp8 => CudaStep::GemmNarrowGroupFp8(g),
            Self::GemvFp8 => CudaStep::GemvGroupFp8(g),
        }
    }
}

/// One weight of a grouped GEMM ([`GemmGroupBf16`]): `B [n, k]` and its output `C [m, n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GroupMember {
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub n: u32,
}

/// Up to three GEMMs `C_i[m, n_i] = A[m, k] · B_i[n_i, k]ᵀ` that share `A`, in one launch: the
/// `first` member is the fold's driver, the others its siblings in tape order. Each member is the
/// [`GemmBf16`] [`Self::member`] names, whose kernel shape rule it meets on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemmGroupBf16 {
    pub a: ArenaOffset,
    pub m: u32,
    pub k: u32,
    pub first: GroupMember,
    pub second: GroupMember,
    pub third: Option<GroupMember>,
}

impl GemmGroupBf16 {
    /// The members in order: `first`, `second`, then `third` when there is one.
    pub const fn members(&self) -> [Option<GroupMember>; 3] {
        [Some(self.first), Some(self.second), self.third]
    }

    /// Member `g` as the GEMM it computes.
    pub const fn member(&self, g: GroupMember) -> GemmBf16 {
        GemmBf16 {
            a: self.a,
            b: g.b,
            c: g.c,
            m: self.m,
            n: g.n,
            k: self.k,
        }
    }

    /// The GEMV scratch bytes of the members before `upto` (`upto` 3: all of them), each at its
    /// own split ([`gemv_scratch_bytes`]): member `i`'s partials start at the bytes before it.
    /// `None` on overflow.
    pub const fn gemv_scratch_before(&self, upto: usize) -> Option<u64> {
        let members = self.members();
        let mut total = 0u64;
        let mut i = 0;
        while i < upto && i < 3 {
            if let Some(g) = members[i] {
                let Some(bytes) = gemv_scratch_bytes(g.n, 1, gemv_splits(g.n, self.k, 1)) else {
                    return None;
                };
                let Some(sum) = total.checked_add(bytes) else {
                    return None;
                };
                total = sum;
            }
            i += 1;
        }
        Some(total)
    }
}

/// The 64-column GEMM's tile `(m, n, k)`: `gemm_bf16.cu`'s template at one 64-column B tile.
/// Every `m`, `n` and `k` it takes is a non-zero multiple of these; no 64-column box crosses an
/// array's column edge (a TMA box past it needs a plain tile, TK-1's R1). Gemma's 128, 704 and
/// 2112 are multiples of 64, not of 256.
pub const GEMM_COL64_TILE: (u32, u32, u32) = (128, 64, 64);

/// The expert GEMM's column tiles: 256 (`kernels/expert_gemm_bf16.cu`, Qwen's 512 and 2048) or 64
/// (Gemma's 704, #924), chosen at expansion by the projection's `n`: `expert_gemm_bf16.cu`'s
/// template at four or one 64-column B blocks.
pub const EXPERT_GEMM_N_TILES: &[u32] = &[64, 256];

/// The head dims the attention, rope-append and KV-append kernels take: 256 (every Qwen layer,
/// Gemma's sliding layers) and 512 (Gemma's global layers, #885's 13c).
pub const ATTN_BF16_HEAD_DIMS: &[u32] = &[256, 512];

/// A sliding window is a non-zero multiple of this many keys, [`CUDA_KV_BLOCK_SIZE`]: a KV tile
/// is then wholly inside or wholly below the window but at the window's lower edge. Gemma's 1024
/// is 16 blocks.
pub const ATTN_WINDOW_MULTIPLE: u32 = CUDA_KV_BLOCK_SIZE;

/// The scale's and the layer scale's `cols` is a non-zero multiple of this (one 16-byte vector of
/// 8 bf16), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const SCALE_BF16_COLS: u32 = 8;

/// The soft-cap's `cols` is a non-zero multiple of this (one 16-byte vector of 8 bf16), `rows`
/// non-zero, and `rows · cols` at most `i32::MAX`.
pub const SOFT_CAP_BF16_COLS: u32 = 8;

/// `out = x · scale` over `[rows, cols]`, bf16 in and out, computed in f32 and rounded once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ScaleBf16 {
    pub x: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    /// `scale` as its bits (`f32::to_bits`), so the step is `Eq`.
    pub scale_bits: u32,
}

/// `y[r, c] = x[r, c] / rms(x[r, :])`, `rms(x) = sqrt(mean(x²) + eps)`, bf16 in and out, computed
/// in f32 and rounded once: [`RmsNormBf16`] with a unit gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RmsNormUnitBf16 {
    pub x: ArenaOffset,
    pub y: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    /// `eps` as its bits (`f32::to_bits`).
    pub eps_bits: u32,
}

/// `out = x · scale[0]` over `[rows, cols]`, `scale` a `[1]` bf16 weight, computed in f32 and
/// rounded once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LayerScaleBf16 {
    pub x: ArenaOffset,
    pub scale: WeightIx,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// `out = cap · tanh(x / cap)` over `[rows, cols]`, bf16 in and out, computed in f32 and rounded
/// once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SoftCapBf16 {
    pub x: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    /// `cap` as its bits (`f32::to_bits`).
    pub cap_bits: u32,
}

/// [`RopePairing`] as the unit variant it is: `scratchy_subtile` has no serde.
fn serialize_rope_pairing<S: serde::Serializer>(
    pairing: &RopePairing,
    s: S,
) -> Result<S::Ok, S::Error> {
    let (index, variant) = match pairing {
        RopePairing::HalfRotated => (0, "HalfRotated"),
        RopePairing::HalfHead => (1, "HalfHead"),
    };
    s.serialize_unit_variant("RopePairing", index, variant)
}

/// The embed's `hidden` is a non-zero multiple of this (`kernels/embed_bf16.cu`: one 16-byte
/// vector of 8 bf16 per thread and step); `rows`, `hidden` and `vocab` are non-zero, none above
/// `i32::MAX`, and `rows · hidden` at most `i32::MAX`.
pub const EMBED_BF16_COLS: u32 = 8;

/// `out[r, :] = table[tokens[r], :]` for every one of the bucket's `rows`, padding rows included,
/// bf16 copied bit for bit. `tokens` is the tape's token region (`rows` u32,
/// [`crate::tape::TapeInput::TokenIds`]); `table` the `[vocab, hidden]` bf16 embedding table;
/// `out` the embedded hidden, `[rows, hidden]`. An id at or above `vocab` writes a `+0.0` row and
/// reads nothing: the worker refuses such an id before any copy, so this is defence in depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct EmbedBf16 {
    pub tokens: ArenaOffset,
    pub table: WeightIx,
    pub out: ArenaOffset,
    pub rows: u32,
    pub hidden: u32,
    pub vocab: u32,
}

/// The bf16 GEMM's tile `(m, n, k)`: every `m`, `n` and `k` it takes is a non-zero multiple of
/// these (`kernels/gemm_bf16.cu`: two 64-row consumer warpgroups, four 64-column B tiles, one
/// 64-wide K step).
pub const GEMM_BF16_TILE: (u32, u32, u32) = (128, 256, 64);

/// `C[m, n] = A[m, k] · B[n, k]ᵀ`, bf16 in and out, accumulated in f32.
///
/// `A` and `C` are arena slots; `B` is a weight buffer in the checkpoint's `[N, K]` storage order
/// (the IR states it as `[K, N]`; the bytes are the checkpoint's). `C` overlaps neither operand:
/// CUDA's colour rule for a matmul reads its inputs across the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemmBf16 {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// The narrow GEMM's largest `n`: all of `B`'s rows are one 64-row ThunderKittens tile, and every
/// column of `C` one 64-column output tile (`kernels/gemm_narrow_bf16.cu`).
pub const GEMM_NARROW_BF16_MAX_N: u32 = 64;
/// The narrow GEMM's `k`, and its `n` but for [`GEMM_NARROW_BF16_ONE`], are non-zero multiples of
/// this: 8 bf16 is 16 bytes, so every row stride of `A`, `B` and `C` is a multiple of 16 bytes, as
/// their TMA tensor maps need.
pub const GEMM_NARROW_BF16_MULTIPLE: u32 = 8;
/// The one `n` the narrow GEMM takes off [`GEMM_NARROW_BF16_MULTIPLE`]: a one-column projection
/// (a shared expert's gate), stored one bf16 per row.
pub const GEMM_NARROW_BF16_ONE: u32 = 1;
/// At `n` = [`GEMM_NARROW_BF16_ONE`] the narrow GEMM's `k` is a non-zero multiple of this: its
/// warp-vector kernel walks each row in chunks of one naive `rv_fl<256>`
/// (`kernels/gemm_narrow_bf16.cu`). Checked by the launcher only; qwen3-5-moe's shared expert gate
/// has `k` = 2048.
pub const GEMM_NARROW_BF16_ONE_K_MULTIPLE: u32 = 256;

/// The GEMV's `k` is a non-zero multiple of this: 8 bf16 is 16 bytes, so every row stride of `a`
/// and `B` is a multiple of 16 bytes, as their TMA tensor maps need (`kernels/gemv_bf16.cu`). `n`
/// is non-zero (any value: `c` is written without a tensor map), and `n` and `k` are at most
/// `i32::MAX`; `m` is 1.
pub const GEMV_BF16_MULTIPLE: u32 = 8;

/// The decode GEMV's chunk along `k` (`kernels/gemv_bf16.cu`): one 512-byte vector TMA load per
/// row and stage. A K split is a run of whole chunks.
pub const GEMV_CHUNK: u32 = 256;
/// The decode GEMV's output rows per CTA: 8 warps of 4 rows. The scratch holds each pair's `n`
/// rounded up to it.
pub const GEMV_BLOCK_ROWS: u32 = 32;
/// The most K splits the decode GEMV takes (its launcher's bound).
pub const GEMV_MAX_SPLITS: u32 = 16;
/// The CTAs a decode GEMV aims for: two per SM of the H100's 132.
pub const GEMV_TARGET_CTAS: u32 = 264;
/// The column blocks (`⌈n / 32⌉ · pairs`) from which a decode GEMV runs at one split: half of
/// [`GEMV_TARGET_CTAS`]'s resident CTAs, 66. At or above it one split already keeps the GPU busy,
/// and splitting only adds the reduce launch and a round trip of the partials (#1031's diagnosis:
/// one split is 10–26% faster on every such shape).
pub const GEMV_SPLIT_BELOW_BLOCKS: u32 = GEMV_TARGET_CTAS / 4;

/// #1031's split table (`GEMV_SPLITS`), as a function of a decode GEMV's `(n, k)` and its pairs
/// (1 for the dense GEMV and the router, `m · top_k` for the expert GEMV): one split from
/// [`GEMV_SPLIT_BELOW_BLOCKS`] column blocks; below it, the fewest K splits that put
/// [`GEMV_TARGET_CTAS`] CTAs on the GPU, each split at least 2 chunks long (so its 4-stage ring
/// has loads to overlap), at most [`GEMV_MAX_SPLITS`]. A pure function of the step's constant
/// shape, so every launch of a step adds its products in the same order. The dense and the expert
/// GEMV run one row routine and take this one table; at one pair and the same `(n, k)` they take
/// the same split, so their rows are the same bits.
///
/// | `(n, k)`, pairs | blocks | split |
/// |---|---|---|
/// | 9B `(4096, 4096)`, `(4096, 12288)`, `(8192, 4096)` | 128, 256 | 1 |
/// | 9B `(1024, 4096)` | 32 | 4 |
/// | 0.8B `(4096, 1024)`, `(6144, 1024)` | 128, 192 | 1 |
/// | 0.8B `(1024, 3584)` | 32 | 5 |
/// | the lm_heads, `n` ≥ 8448 | ≥ 264 | 1 |
/// | 35B experts `(512, 2048)` × 8 | 128 | 1 |
/// | Gemma experts `(704, 2816)` × 8 | 176 | 1 |
/// | the routers, `(256, 2048)` and `(128, 2816)` | 8, 4 | 4 |
pub const fn gemv_splits(n: u32, k: u32, pairs: u32) -> u32 {
    let blocks = n.div_ceil(GEMV_BLOCK_ROWS) as u64 * pairs as u64;
    if blocks >= GEMV_SPLIT_BELOW_BLOCKS as u64 {
        return 1;
    }
    let chunks = k.div_ceil(GEMV_CHUNK);
    let mut splits = 1u32;
    while blocks * (splits as u64) < GEMV_TARGET_CTAS as u64
        && splits < GEMV_MAX_SPLITS
        && chunks / (splits + 1) >= 2
        && gemv_split_chunks(chunks, splits + 1) != 0
    {
        splits += 1;
    }
    splits
}

/// The chunks of one of `splits` K runs of a `chunks`-chunk row, `⌈chunks / splits⌉`, or 0 when
/// that leaves a run empty or `splits` is out of `1..=GEMV_MAX_SPLITS`: the launcher's rule
/// (`kernels/gemv_bf16.cu`, `split_chunks`), so every split [`gemv_splits`] picks is one it takes.
pub const fn gemv_split_chunks(chunks: u32, splits: u32) -> u32 {
    if splits == 0 || splits > GEMV_MAX_SPLITS || splits > chunks {
        return 0;
    }
    let each = chunks.div_ceil(splits);
    if (splits - 1) * each < chunks {
        each
    } else {
        0
    }
}

/// The f32 scratch bytes a decode GEMV at `splits` writes: `splits · pairs · n_pad` f32, `n_pad`
/// `n` rounded up to [`GEMV_BLOCK_ROWS`]; 0 at one split (each CTA stores its rows directly).
/// `None` on overflow.
pub const fn gemv_scratch_bytes(n: u32, pairs: u32, splits: u32) -> Option<u64> {
    if splits <= 1 {
        return Some(0);
    }
    let rows = n.div_ceil(GEMV_BLOCK_ROWS) as u64 * GEMV_BLOCK_ROWS as u64;
    let Some(per) = rows.checked_mul(pairs as u64) else {
        return None;
    };
    let Some(all) = per.checked_mul(splits as u64) else {
        return None;
    };
    all.checked_mul(4)
}

/// The RMSNorm's `cols` is a non-zero multiple of this: 8 bf16 is 16 bytes, so every row stride is a multiple of 16 bytes, as the ThunderKittens 16-row tile's TMA tensor maps need (`kernels/rms_norm_bf16.cu`). `rows`
/// is non-zero and `rows · cols` at most `i32::MAX`.
pub const RMS_NORM_BF16_COLS: u32 = 8;

/// `y[r, c] = x[r, c] / rms(x[r, :]) · (offset + gain[c])`, `rms(x) = sqrt(mean(x²) + eps)`,
/// bf16 in and out, computed in f32 and rounded once.
///
/// `x` and `y` are `[rows, cols]` arena slots; `gain` is a `[cols]` bf16 weight. `offset` is the
/// convention's ([`GainConvention::offset`]): 0 for a stored gain, 1 for a zero-centred one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RmsNormBf16 {
    pub x: ArenaOffset,
    pub gain: WeightIx,
    pub y: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    /// `eps` as its bits (`f32::to_bits`), so the step is `Eq`; [`Self::eps`] reads it back.
    pub eps_bits: u32,
    #[serde(serialize_with = "serialize_gain_convention")]
    pub convention: GainConvention,
}

/// [`GainConvention`] as the unit variant it is: `scratchy_subtile` has no serde, so the static
/// tape names it `GainConvention::Scale` or `GainConvention::OnePlusScale` through this.
fn serialize_gain_convention<S: serde::Serializer>(
    convention: &GainConvention,
    s: S,
) -> Result<S::Ok, S::Error> {
    let (index, variant) = match convention {
        GainConvention::Scale => (0, "Scale"),
        GainConvention::OnePlusScale => (1, "OnePlusScale"),
    };
    s.serialize_unit_variant("GainConvention", index, variant)
}

impl RmsNormBf16 {
    /// The norm's `eps`.
    pub const fn eps(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }
}

/// The add's `cols` is a non-zero multiple of this (8 bf16 is 16 bytes, so every row stride is a multiple of 16 bytes, as the ThunderKittens 16-row tile's TMA tensor maps need, `kernels/add_bf16.cu`), `rows`
/// non-zero, and `rows · cols` at most `i32::MAX`.
pub const ADD_BF16_COLS: u32 = 8;

/// `out = a + b` over `[rows, cols]`, bf16 in and out: one f32 add, one rounding. No broadcast:
/// all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AddBf16 {
    pub a: ArenaOffset,
    pub b: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// The SiluMul's `cols` is a non-zero multiple of this (8 bf16 is 16 bytes, so every row stride is a multiple of 16 bytes, as the ThunderKittens 16-row tile's TMA tensor maps need, `kernels/silu_mul_bf16.cu`),
/// `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const SILU_MUL_BF16_COLS: u32 = 8;

/// `out = gate / (1 + e^(−gate)) · up` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SiluMulBf16 {
    pub gate: ArenaOffset,
    pub up: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// The rope's `rot_dim` is a non-zero multiple of this, at most `head_dim`: each head's `R/2`
/// rotated pairs are whole 8-wide vectors on both halves (`kernels/rope_bf16.cu`).
pub const ROPE_BF16_ROT_MULTIPLE: u32 = 16;
/// The rope's `head_dim` is a non-zero multiple of this (the pass-through lanes are whole 16-byte
/// vectors), and its `cols` a multiple of `head_dim`. `rows` is non-zero and `rows · cols` at most
/// `i32::MAX`.
pub const ROPE_BF16_HEAD_MULTIPLE: u32 = 8;
/// The `(rot_dim, pairing)`s the rope kernel is instantiated for (`kernels/rope_bf16.cu`): 64
/// half-rotated, one plain `st_bf<16, 64>` box, each half two 16-column base tiles (qwen3.5's
/// partial rope); 256 half-rotated, four boxes, each half two whole boxes (Gemma 4's sliding
/// layers, a full rope on a 256-lane head); and 128 half-head on a 512-lane head
/// ([`ROPE_BF16_HALF_HEAD_DIM`]), box 0 paired with box 4 (Gemma 4's global layers, proportional).
/// Its launcher refuses any other, so the kernel's rule is tighter than the lowering's
/// [`ROPE_BF16_ROT_MULTIPLE`].
pub const ROPE_BF16_KERNEL_SHAPES: &[(u32, RopePairing)] = &[
    (64, RopePairing::HalfRotated),
    (256, RopePairing::HalfRotated),
    (128, RopePairing::HalfHead),
];
/// The one head dim of the rope kernel's half-head instantiation: a whole 512-lane head is one
/// unit of 8 boxes (`kernels/rope_bf16.cu`).
pub const ROPE_BF16_HALF_HEAD_DIM: u32 = 512;
/// The rope kernel's `head_dim` is a non-zero multiple of this: whole 64-column boxes, so every
/// head starts at a box (`kernels/rope_bf16.cu`). Tighter than the lowering's
/// [`ROPE_BF16_HEAD_MULTIPLE`], and checked by the launcher only.
pub const ROPE_BF16_KERNEL_HEAD_MULTIPLE: u32 = 64;

/// Partial rotary embedding over `[rows, cols]`, `cols = heads · head_dim`, bf16 in and out.
///
/// For token row `r` and each head, pair `d < rot_dim/2` of lanes `(d, d + rot_dim/2)` (NeoX
/// pairing within the rotated width) becomes `(x0·c − x1·s, x1·c + x0·s)` in f32, rounded once,
/// with `c = table[r, d]` and `s = table[r, rot_dim/2 + d]`. Lanes `rot_dim..head_dim` are copied
/// bit for bit. `table` is the tape's rotary table ([`crate::tape::RotaryTable`]), `[rows,
/// rot_dim]` bf16; `out` overlaps neither input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RopeBf16 {
    pub x: ArenaOffset,
    pub table: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    pub head_dim: u32,
    pub rot_dim: u32,
    /// Which lane a rotated lane pairs with: `HalfRotated` (the kernel's), or `HalfHead`
    /// (Gemma's proportional rope, #885's 13b-3). `table` is the tape's table of the step's class.
    #[serde(serialize_with = "serialize_rope_pairing")]
    pub pairing: RopePairing,
}

/// The gate split's `head_dim` is a non-zero multiple of this (`kernels/gate_split_bf16.cu`: one
/// 16-byte vector per thread, never straddling a head), and `half_cols` a non-zero multiple of
/// `head_dim`. `rows` is non-zero and `rows · 2·half_cols` at most `i32::MAX`.
pub const GATE_SPLIT_BF16_HEAD_MULTIPLE: u32 = 8;
/// The one `head_dim` the gate split is instantiated for (`kernels/gate_split_bf16.cu`): one plain
/// `st_bf<16, 256>` tile per head. Checked after [`GATE_SPLIT_BF16_HEAD_MULTIPLE`] and the
/// whole-heads rule, so a shape those refuse keeps its refusal.
pub const GATE_SPLIT_BF16_HEAD_DIM: u32 = 256;

/// Per head `h` and lane `d`: `q[r, h·hd + d] = qg[r, 2h·hd + d]` and `gate[r, h·hd + d] =
/// qg[r, 2h·hd + hd + d]`, bf16 copies. `qg` is `[rows, 2·half_cols]`, `q` and `gate` are `[rows,
/// half_cols]`, and neither output overlaps `qg` or the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateSplitBf16 {
    pub qg: ArenaOffset,
    pub q: ArenaOffset,
    pub gate: ArenaOffset,
    pub rows: u32,
    pub half_cols: u32,
    pub head_dim: u32,
}

/// The gate apply's `cols` is a non-zero multiple of this (8 bf16 is 16 bytes, so every row stride is a multiple of 16 bytes, as the ThunderKittens 16-row tile's TMA tensor maps need, `kernels/gate_apply_bf16.cu`),
/// `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const GATE_APPLY_BF16_COLS: u32 = 8;

/// `out = attn / (1 + e^(−gate))` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateApplyBf16 {
    pub attn: ArenaOffset,
    pub gate: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// Token slots per KV block = the attention kernel's K/V tile height, so one paged
/// (block, kv head) tile is exactly one TMA box (`kernels/attn_prefill_bf16.cu`).
pub const CUDA_KV_BLOCK_SIZE: u32 = 64;
/// Block-table entries per sequence: 128 · 64 = 8192 tokens, codegen's prefix capacity.
pub const CUDA_MAX_BLOCKS_PER_SEQ: u32 = 128;
/// Query rows per attention CTA (two consumer warpgroups of 64): `rows` is a multiple.
pub const ATTN_PREFILL_BF16_ROWS: u32 = 128;
/// The one head dim the attention kernel is instantiated for.
pub const ATTN_PREFILL_BF16_HEAD_DIM: u32 = 256;
/// The head dim of the column-split attention kernels (`kernels/attn_prefill_d512_bf16.cu`,
/// `kernels/attn_decode_d512_bf16.cu`): Gemma 4's global layers. Each of a CTA's two consumer
/// warpgroups keeps one 256-column half of the output; the prefill takes [`ATTN_PREFILL_BF16_ROWS`]
/// and the decode [`ATTN_DECODE_BF16_MAX_GROUP`] as the head-dim-256 kernels do.
pub const ATTN_D512_BF16_HEAD_DIM: u32 = 512;
// The worker picks the attention kernel by these names: they are exactly the head dims a KV step
// may take.
const _: () = assert!(
    ATTN_BF16_HEAD_DIMS.len() == 2
        && ATTN_BF16_HEAD_DIMS[0] == ATTN_PREFILL_BF16_HEAD_DIM
        && ATTN_BF16_HEAD_DIMS[0] == ATTN_DECODE_BF16_HEAD_DIM
        && ATTN_BF16_HEAD_DIMS[1] == ATTN_D512_BF16_HEAD_DIM
);
/// The one head dim the decode attention kernel is instantiated for: its query, K and V tiles are
/// `st_bf<64, 256>`, the prefill's (`kernels/attn_decode_bf16.cu`). It takes one query row,
/// `q_heads` a non-zero multiple of `kv_heads` by at most [`ATTN_DECODE_BF16_MAX_GROUP`], and
/// `block_size` [`CUDA_KV_BLOCK_SIZE`].
pub const ATTN_DECODE_BF16_HEAD_DIM: u32 = 256;
/// The decode attention's largest GQA group `q_heads / kv_heads`: a kv head's query heads are
/// stacked as the rows of one query tile, and only the 16 rows of each consumer warpgroup's warp 0
/// are merged and stored (`kernels/attn_decode_bf16.cu`). The target models' groups are 2 to 12.
pub const ATTN_DECODE_BF16_MAX_GROUP: u32 = 16;
/// The head dims the KV append is instantiated for (`kernels/kv_append_bf16.cu`): a head's row is
/// one `sv_bf<256>`, one 512-byte box of the cache's vector tensor map, or one `sv_bf<512>`, two
/// such boxes (Gemma 4's global layers). The lowering already refuses any other head dim for a
/// `RopeAppend` (`CanonicalRefusal::KvHeadDim`).
pub const KV_APPEND_BF16_HEAD_DIMS: &[u32] = &[256, 512];

/// K's partial rope, then the paged-cache write of the rotated K and the unrotated V.
///
/// Launch 1 is [`RopeBf16`]'s kernel, unchanged: `k [rows, cols]` rotated by `table` into
/// `k_out`, `cols = kv_heads · head_dim`. Launch 2 copies, for each token row `t` whose
/// `slot_mapping[t]` is not `u32::MAX`, row `t` of `k_out` and of `v` into layer `layer`'s K and
/// V cache at that slot, bit for bit (`kernels/kv_append_bf16.cu`); its `head_dim` is
/// one of [`KV_APPEND_BF16_HEAD_DIMS`]. `slot_mapping` is the tape's
/// region ([`crate::tape::KvTape::slot_mapping`], `rows` u32); `layer` is an index into
/// [`crate::tape::KvTape::layers`]. `k_out` overlaps no input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RopeAppendBf16 {
    pub k: ArenaOffset,
    pub v: ArenaOffset,
    pub table: ArenaOffset,
    pub slot_mapping: ArenaOffset,
    pub k_out: ArenaOffset,
    pub layer: u32,
    pub rows: u32,
    pub cols: u32,
    pub head_dim: u32,
    pub rot_dim: u32,
    /// As [`RopeBf16::pairing`].
    #[serde(serialize_with = "serialize_rope_pairing")]
    pub pairing: RopePairing,
}

/// Causal GQA attention of `q [rows, q_heads · head_dim]` over layer `layer`'s paged K/V cache,
/// into `out` of the same shape, bf16 in and out, the softmax in f32.
///
/// The keys are the sequence's `P + L` cached tokens, read through the tape's block table; query
/// row `i` attends to keys `0 ..= P + i` (`P`, `L` read from the tape's `sequence` region). Query
/// head `g` reads kv head `g / (q_heads / kv_heads)`. `block_table` and `sequence` are the tape's
/// regions ([`crate::tape::KvTape`]); `out` overlaps no input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AttnPrefillBf16 {
    pub q: ArenaOffset,
    pub block_table: ArenaOffset,
    pub sequence: ArenaOffset,
    pub out: ArenaOffset,
    pub layer: u32,
    pub rows: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// `scale` as its bits (`f32::to_bits`), so the step is `Eq`; [`Self::scale`] reads it back.
    /// The kernel folds it into `log2(e)`.
    pub scale_bits: u32,
    /// The keys a query row sees, its last `window` positions (a non-zero multiple of
    /// [`ATTN_WINDOW_MULTIPLE`]); `None` for full causal attention. The head-dim-256 kernels take
    /// it as a launch argument (0 for `None`); the head-dim-512 kernels are #885's 13b-3.
    pub window: Option<u32>,
}

impl AttnPrefillBf16 {
    /// The softmax scale: `head_dim^-0.5` for qwen3.5.
    pub const fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
}

/// The one key-head dim the GDN scan takes: each value row of `S` is 4 lanes of 32 f32 registers
/// (`kernels/gdn_scan.cu`).
pub const GDN_HEAD_K: u32 = 128;
/// The GDN value-head dim is a non-zero multiple of this, the value rows of one scan CTA
/// (`kernels/gdn_scan.cu`), and at most [`GDN_HEAD_V_MAX`].
pub const GDN_HEAD_V_MULTIPLE: u32 = 32;
/// The largest GDN value-head dim: the gated norm's one warp per row is a naive `rv_fl<head_v>`,
/// instantiated for 32, 64, …, 256 (`kernels/gdn_rms_norm_gated.cu`).
pub const GDN_HEAD_V_MAX: u32 = 256;
/// The causal conv's kernel width is in this inclusive range: its window lives in registers
/// (`kernels/gdn_conv1d.cu`; metal's `GDN_CONV_KMAX` is 8).
pub const GDN_CONV_KERNEL_RANGE: (u32, u32) = (2, 8);
/// The tokens one conv warp walks. At least `conv_kernel − 1`, so only block 0 reads or writes a
/// slot's conv ring.
pub const GDN_CONV_TOKEN_BLOCK: u32 = 64;
/// The causal conv's `conv_dim` is a non-zero multiple of this: one warp's channels, a naive
/// `rv_fl<128>` (`kernels/gdn_conv1d.cu`). Checked by the launcher only; qwen3.5's `conv_dim`
/// (6144 at the 0.8b, 8192 at the 9b) is a multiple, so no lowered tape is refused.
pub const GDN_CONV_CHANNEL_GROUP: u32 = 128;
/// The GDN gating's `nv` is a non-zero multiple of this: one warp per row walks the row in naive
/// `rv_fl<16>` chunks (`kernels/gdn_gating.cu`). Checked by the launcher only; qwen3.5's `nv` is 16
/// at the 0.8b and 32 at the 9b, 48 at qwen3.6-27b and 64 at qwen3.5-122b-a10b.
pub const GDN_GATING_NV_MULTIPLE: u32 = 16;
/// The tokens the scan stages in shared memory at once.
pub const GDN_SCAN_TOKEN_BLOCK: u32 = 32;

/// Bytes of one f32: the GDN scratch, bundle and state are f32.
const F32_BYTES: u64 = 4;
/// Every region of the GDN scratch and bundle starts at a multiple of this.
const GDN_ALIGN: u64 = 256;

/// The causal conv's channels, `2 · num_k_heads · head_k + num_v_heads · head_v` (`q | k | v`), or
/// `None` on overflow.
pub const fn gdn_conv_dim(
    num_k_heads: u32,
    head_k: u32,
    num_v_heads: u32,
    head_v: u32,
) -> Option<u32> {
    let (Some(key), Some(value)) = (
        num_k_heads.checked_mul(head_k),
        num_v_heads.checked_mul(head_v),
    ) else {
        return None;
    };
    let Some(qk) = key.checked_mul(2) else {
        return None;
    };
    qk.checked_add(value)
}

/// The f32 scratch a GDN step's four launches pass their intermediates through: the byte offsets
/// of `conv_out [rows, conv_dim]`, `g [rows, nv]`, `beta [rows, nv]` and `o [rows, value_dim]`,
/// each a multiple of 256, and the total. Metal's `GdnScratchLayout`: the same regions in the same
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnScratch {
    pub conv_out: u64,
    pub g: u64,
    pub beta: u64,
    pub o: u64,
    pub bytes: u64,
}

/// [`GdnScratch`] for `rows` tokens, or `None` when a byte count overflows `u64`.
pub const fn gdn_scratch(rows: u32, conv_dim: u32, nv: u32, value_dim: u32) -> Option<GdnScratch> {
    let rows = rows as u64;
    let Some(g) = f32_region(0, rows * conv_dim as u64) else {
        return None;
    };
    let Some(beta) = f32_region(g, rows * nv as u64) else {
        return None;
    };
    let Some(o) = f32_region(beta, rows * nv as u64) else {
        return None;
    };
    let Some(bytes) = f32_region(o, rows * value_dim as u64) else {
        return None;
    };
    Some(GdnScratch {
        conv_out: 0,
        g,
        beta,
        o,
        bytes,
    })
}

/// A GDN layer's packed weight bundle, all f32: the byte offsets of `conv1d [conv_dim, K]` (row
/// `c` channel `c`'s taps, oldest first, as the checkpoint's `[conv_dim, 1, K]`), `a_log [nv]`,
/// `dt_bias [nv]` and `norm [head_v]` (the gated norm's plain gain), each a multiple of 256, and
/// the total. f32 throughout, so no on-disk dtype reaches a kernel: bf16 to f32 is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnBundle {
    pub conv1d: u64,
    pub a_log: u64,
    pub dt_bias: u64,
    pub norm: u64,
    pub bytes: u64,
}

/// [`GdnBundle`] for one layer's geometry, or `None` when a byte count overflows `u64`.
pub const fn gdn_bundle(
    conv_dim: u32,
    conv_kernel: u32,
    nv: u32,
    head_v: u32,
) -> Option<GdnBundle> {
    let Some(a_log) = f32_region(0, conv_dim as u64 * conv_kernel as u64) else {
        return None;
    };
    let Some(dt_bias) = f32_region(a_log, nv as u64) else {
        return None;
    };
    let Some(norm) = f32_region(dt_bias, nv as u64) else {
        return None;
    };
    let Some(bytes) = f32_region(norm, head_v as u64) else {
        return None;
    };
    Some(GdnBundle {
        conv1d: 0,
        a_log,
        dt_bias,
        norm,
        bytes,
    })
}

/// The 256-aligned end of `elems` f32 placed at `at`, or `None` on overflow.
const fn f32_region(at: u64, elems: u64) -> Option<u64> {
    let Some(bytes) = elems.checked_mul(F32_BYTES) else {
        return None;
    };
    let Some(end) = at.checked_add(bytes) else {
        return None;
    };
    end.checked_next_multiple_of(GDN_ALIGN)
}

/// The bytes of the GDN sequence region every GDN launch reads: `[slot, q_len, start, 0]` u32.
pub const GDN_SEQUENCE_BYTES: u64 = 16;

/// The scan's query scale, `head_k^-½`, in f64 by Newton's iteration (a `const fn` has no `sqrt`)
/// and rounded once to f32: exactly `2^-3.5` rounded for the one `head_k` the scan takes, 128.
pub const fn gdn_scale(head_k: u32) -> f32 {
    if head_k == 0 {
        return f32::INFINITY;
    }
    let x = head_k as f64;
    // y ← y·(3 − x·y²)/2 converges to x^-½ from any start in (0, √(3/x)); 1/x is in it for x ≥ 1.
    let mut y = 1.0 / x;
    let mut i = 0;
    while i < 64 {
        y = y * (3.0 - x * y * y) / 2.0;
        i += 1;
    }
    y as f32
}

/// One prefill chunk of a Gated-DeltaNet layer: `1 ≤ q_len ≤ rows` real tokens of one sequence.
///
/// Four launches, in order on the stream, each reading `[slot, q_len, start, 0]` u32 from the
/// tape's `sequence` region (`start` is 1 on a sequence's first chunk, 0 to continue from its
/// slot: `scratchy_layers::gdn_state::GdnStep::encode`'s bits 0-7):
///
/// 1. the causal depthwise conv of `qkv [rows, conv_dim]` after the slot's conv ring (zero, never
///    read, when fresh), then SiLU, into the scratch's f32 `conv_out`; the ring updated;
/// 2. the gating, `g = −exp(a_log)·softplus(a + dt_bias)` and `β = sigmoid(b)`, into `g`, `beta`;
/// 3. the recurrent gated delta-rule scan from the slot's f32 state `S` (zero, never read, when
///    fresh), into the f32 `o`; `S` written back once;
/// 4. the gated RMSNorm `o / rms(o) · norm · SiLU(z)` per value head, into `out [rows, value_dim]`
///    bf16; rows `q_len..rows` written `+0.0`.
///
/// `bundle` is the packed f32 weight bundle ([`gdn_bundle`]), `scratch` the tape's GDN scratch
/// ([`gdn_scratch`] at `rows`), `layer` an index into [`crate::tape::GdnTape::layers`]. Value head
/// `h` reads key head `h / (num_v_heads / num_k_heads)`. No launch reads a padding row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GatedDeltaNetBf16 {
    /// bf16 `[rows, conv_dim]`: `q | k | v`.
    pub qkv: ArenaOffset,
    /// bf16 `[rows, value_dim]`: the gated norm's gate.
    pub z: ArenaOffset,
    /// bf16 `[rows, num_v_heads]`: the decay's input.
    pub a: ArenaOffset,
    /// bf16 `[rows, num_v_heads]`: the write strength's input.
    pub b: ArenaOffset,
    pub bundle: WeightIx,
    /// bf16 `[rows, value_dim]`.
    pub out: ArenaOffset,
    pub scratch: ArenaOffset,
    pub sequence: ArenaOffset,
    pub layer: u32,
    pub rows: u32,
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub head_k: u32,
    pub head_v: u32,
    pub conv_kernel: u32,
    /// The gated norm's `eps` (`rms_norm_eps`) as its bits (`f32::to_bits`), so the step is `Eq`;
    /// [`Self::eps`] reads it back.
    pub eps_bits: u32,
}

impl GatedDeltaNetBf16 {
    /// The gated norm's `eps`.
    pub const fn eps(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }

    /// The query scale, `head_k^-½`.
    pub const fn scale(&self) -> f32 {
        gdn_scale(self.head_k)
    }

    /// The conv's channels, or `None` on overflow.
    pub const fn conv_dim(&self) -> Option<u32> {
        gdn_conv_dim(self.num_k_heads, self.head_k, self.num_v_heads, self.head_v)
    }

    /// The value width, `num_v_heads · head_v`, or `None` on overflow.
    pub const fn value_dim(&self) -> Option<u32> {
        self.num_v_heads.checked_mul(self.head_v)
    }

    /// The scratch's regions at this step's rows, or `None` on overflow.
    pub const fn scratch_layout(&self) -> Option<GdnScratch> {
        let (Some(conv_dim), Some(value_dim)) = (self.conv_dim(), self.value_dim()) else {
            return None;
        };
        gdn_scratch(self.rows, conv_dim, self.num_v_heads, value_dim)
    }

    /// The bundle's regions for this step's geometry, or `None` on overflow.
    pub const fn bundle_layout(&self) -> Option<GdnBundle> {
        let Some(conv_dim) = self.conv_dim() else {
            return None;
        };
        gdn_bundle(conv_dim, self.conv_kernel, self.num_v_heads, self.head_v)
    }
}

/// Which kernel a GEMM-shaped MoE step runs, chosen at expansion by the GEMM's shape rule: the
/// tiled GEMM for a bucket of rows, the GEMV for one row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MatmulKernel {
    Tiled,
    Gemv,
    /// The 64-column GEMM ([`GEMM_COL64_TILE`]): Gemma's 128-expert router at m ≥ 2
    /// (`scratchy_cuda_gemm_col64_bf16_abt`, #924).
    Col64,
}

/// Which buffer a MoE step's weight index names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MoeWeight {
    /// Qwen's packed bundle ([`moe_bundle`]): the router at byte 0, then every expert's fused gate
    /// and up, then every expert's down.
    Bundle,
    /// Gemma's own tensors: the router step's `[experts, hidden]` projection, an expert step's
    /// expert bank ([`expert_bank`]: every expert's fused gate and up, then every expert's down,
    /// with no router).
    Own,
    /// Qwen's fp8 bundle ([`fp8_moe_bundle`] with the router): [`Self::Bundle`]'s bf16 router,
    /// then every expert's e4m3 gate and up and down, then their f32 per-channel scales.
    Fp8Bundle,
    /// Gemma's fp8 expert bank ([`fp8_moe_bundle`] with no router); its router step reads its own
    /// bf16 projection, as [`Self::Own`]'s.
    Fp8Own,
}

impl MoeWeight {
    /// The layout an expert step reads its slab from: the bundle's, or the bank's (its gate and
    /// up at byte 0). `None` on overflow.
    pub const fn experts(self, experts: u32, hidden: u32, inter: u32) -> Option<MoeBundle> {
        match self {
            Self::Bundle => moe_bundle(experts, hidden, inter),
            Self::Own => expert_bank(experts, hidden, inter),
            Self::Fp8Bundle => fp8_moe_bundle(experts, hidden, inter, true),
            Self::Fp8Own => fp8_moe_bundle(experts, hidden, inter, false),
        }
    }

    /// The bytes a router step reads its weight as: the whole bundle, or the projection.
    pub const fn router_bytes(self, experts: u32, hidden: u32, inter: u32) -> Option<u64> {
        match self {
            Self::Bundle | Self::Fp8Bundle => match self.experts(experts, hidden, inter) {
                Some(b) => Some(b.bytes),
                None => None,
            },
            Self::Own | Self::Fp8Own => bf16_bytes(experts, hidden),
        }
    }
}

/// `logits[m, experts] = x[m, hidden] · routerᵀ`, the router `[experts, hidden]` bf16 at byte 0
/// of `bundle` (the MoE bundle, [`moe_bundle`], or Gemma's own projection, as `weight` says), into
/// the tape's MoE logits region. Gemma's `x` is the MoE scratch's router-norm rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RouterLogitsBf16 {
    pub x: ArenaOffset,
    pub bundle: WeightIx,
    pub weight: MoeWeight,
    pub logits: ArenaOffset,
    pub m: u32,
    pub experts: u32,
    pub hidden: u32,
    /// The bundle's expert width, so every step naming the bundle states its geometry.
    pub inter: u32,
    pub kernel: MatmulKernel,
}

/// The one expert count the routing kernel's Qwen program takes: one warp per token row, 8 logits
/// per lane (`kernels/moe_route_bf16.cu`). Its Gemma program takes each of
/// `op_abi::MOE_EXPERT_COUNTS`.
pub const MOE_ROUTE_EXPERTS: u32 = 256;
/// The routing kernel's largest top-k: one pick per round of its warp reductions
/// (`kernels/moe_route_bf16.cu`).
pub const MOE_ROUTE_MAX_TOP_K: u32 = 32;
/// The combine's `hidden` is a non-zero multiple of this: one naive `rv_fl<256>` per column chunk
/// (`kernels/moe_combine_bf16.cu`). Checked by the launcher only; qwen3-5-moe's `hidden` is 2048.
pub const MOE_COMBINE_BF16_CHUNK: u32 = 256;

/// The routing of `m` tokens over `experts` (one warp per token row, `experts` = 256): `p =
/// softmax(logits)` in f32, its `top_k` largest picked (equal probabilities by the higher expert
/// index; a NaN lowest), stored ascending (slot `top_k − 1` the best) as i32 into `topk_idx`, and
/// their probabilities, divided by their sum when `renorm`, as f32 into `scores`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MoeRouteBf16 {
    pub logits: ArenaOffset,
    pub topk_idx: ArenaOffset,
    pub scores: ArenaOffset,
    pub m: u32,
    pub experts: u32,
    pub top_k: u32,
    pub renorm: bool,
    pub program: RouteProgram,
    /// Gemma's `router.per_expert_scale` (`[experts]` bf16), each pick's weight multiplied by its
    /// expert's entry last; `None` for Qwen.
    pub expert_scale: Option<WeightIx>,
}

/// Which routing a [`MoeRouteBf16`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RouteProgram {
    /// Qwen: a softmax over every expert, the top-k, renormalised when `renorm`.
    QwenSoftmax,
    /// Gemma: the top-k of the raw logits, each scaled by `scale` (`hidden^-0.5`), a softmax over
    /// the picks, then the per-expert scale (`kernels/moe_route_bf16.cu`'s Gemma program).
    GemmaScaledSoftmax {
        /// `scale` as its bits (`f32::to_bits`).
        scale_bits: u32,
    },
}

/// The `P = m · top_k` (token, pick) pairs sorted by expert, stably: every live tile of the
/// sorted rows written (a pair's token row, a padding row `+0.0`), and the index regions the
/// grouped GEMM and the combine read. The regions are the tape's [`crate::tape::MoeTape`]'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertSortBf16 {
    /// `[m, hidden]` bf16, the token rows.
    pub x: ArenaOffset,
    pub topk_idx: ArenaOffset,
    pub chunk_hist: ArenaOffset,
    pub counts: ArenaOffset,
    pub offsets: ArenaOffset,
    pub tile_expert: ArenaOffset,
    pub pos: ArenaOffset,
    pub rows: ArenaOffset,
    pub m: u32,
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub max_tiles: u32,
}

/// Which projection of a MoE bundle an expert step computes; its weights for expert `e` are
/// `[n, k]` bf16 at [`ExpertProjection::slab`]'s bytes into the bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ExpertProjection {
    /// `[inter, hidden]`: the first `inter` rows of the expert's fused gate and up.
    Gate,
    /// `[inter, hidden]`: the next `inter` rows.
    Up,
    /// `[hidden, inter]`.
    Down,
}

impl ExpertProjection {
    /// Whether the gathered expert GEMV reads each pair's own row (the down projection, over the
    /// activation rows) rather than its token's row (the gate and up, over the token rows).
    pub const fn reads_pairs(self) -> bool {
        matches!(self, Self::Down)
    }

    /// `(n, k)`: the projection's output and input widths.
    pub const fn n_k(self, hidden: u32, inter: u32) -> (u32, u32) {
        match self {
            Self::Gate | Self::Up => (inter, hidden),
            Self::Down => (hidden, inter),
        }
    }

    /// `(part, expert_stride)`: expert `e`'s `[n, k]` starts `part + e · expert_stride` bytes into
    /// the bundle of `bundle`'s layout. `None` on overflow.
    pub const fn slab(self, bundle: MoeBundle, hidden: u32, inter: u32) -> Option<(u64, u64)> {
        let (h, i) = (hidden as u64, inter as u64);
        let Some(one) = (h * i).checked_mul(CUDA_ACT_BYTES) else {
            return None;
        };
        match self {
            Self::Gate => Some((bundle.gate_up, 2 * one)),
            Self::Up => Some((bundle.gate_up + one, 2 * one)),
            Self::Down => Some((bundle.down, one)),
        }
    }
}

/// `c[128t .. 128t + 128, :] = a[128t .. 128t + 128, :] · B_eᵀ` for every live tile `t` of the
/// sorted rows, `e = tile_expert[t]`, `B_e` expert `e`'s `[n, k]` of the bundle's `proj`. A tile
/// at or past `offsets[experts] / 128` is never read or written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGemmBf16 {
    pub a: ArenaOffset,
    pub bundle: WeightIx,
    /// The bundle's layout: Qwen's MoE bundle, or Gemma's expert bank.
    pub weight: MoeWeight,
    pub proj: ExpertProjection,
    pub offsets: ArenaOffset,
    pub tile_expert: ArenaOffset,
    pub c: ArenaOffset,
    pub max_tiles: u32,
    pub experts: u32,
    pub hidden: u32,
    pub inter: u32,
    /// The column tile, one of [`EXPERT_GEMM_N_TILES`]: 256 for every Qwen tape.
    pub n_tile: u32,
}

impl ExpertGemmBf16 {
    /// `(n, k)` of the projection.
    pub const fn n_k(&self) -> (u32, u32) {
        self.proj.n_k(self.hidden, self.inter)
    }
}

/// Which row of `a` pair `p = (token t, slot j)` reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PairRows {
    /// Its token's row, `a[t]` (the gate and up projections).
    Token,
    /// Its own row, `a[p]` (the down projection).
    Pair,
}

/// `c[p, :] = a[row(p), :] · B_eᵀ` for each of the `m · top_k` pairs, `e = topk_idx[p]`, `B_e`
/// expert `e`'s `[n, k]` of the bundle's `proj`; `row(p)` is the pair's token row for the gate and
/// up projections, its own row for the down ([`PairRows`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGemvBf16 {
    pub a: ArenaOffset,
    pub rows: PairRows,
    pub bundle: WeightIx,
    /// As [`ExpertGemmBf16::weight`].
    pub weight: MoeWeight,
    pub proj: ExpertProjection,
    pub topk_idx: ArenaOffset,
    pub c: ArenaOffset,
    pub m: u32,
    pub top_k: u32,
    pub experts: u32,
    pub hidden: u32,
    pub inter: u32,
}

impl ExpertGemvBf16 {
    /// `(n, k)` of the projection.
    pub const fn n_k(&self) -> (u32, u32) {
        self.proj.n_k(self.hidden, self.inter)
    }
}

/// `out[t, :] = Σ_j scores[t, j] · down[row(t, j), :]` over the `top_k` slots in order, in f32,
/// rounded once; `row(t, j)` is `pos[t · top_k + j]` when `pos` is given (sorted), else
/// `t · top_k + j` (gathered).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertCombineBf16 {
    pub down: ArenaOffset,
    pub pos: Option<ArenaOffset>,
    pub scores: ArenaOffset,
    pub out: ArenaOffset,
    pub m: u32,
    pub top_k: u32,
    pub hidden: u32,
    /// The down region's rows: `R` sorted, `P` gathered.
    pub down_rows: u32,
}

/// The gate scale's `cols` is a non-zero multiple of this (8 bf16 is 16 bytes, so every row stride
/// is a multiple of 16 bytes, as the ThunderKittens 16-row tile's TMA tensor maps need,
/// `kernels/gate_scale_bf16.cu`), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const GATE_SCALE_BF16_COLS: u32 = 8;

/// `out = routed + shared · (1 / (1 + e^(−g)))` over `[rows, cols]`, `g` `[rows, 1]` broadcast
/// across the row, computed in f32 and rounded once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateScaleBf16 {
    pub routed: ArenaOffset,
    pub shared: ArenaOffset,
    pub g: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// A MoE layer's dense bf16 bundle, its three checkpoint tensors back to back: the byte offsets
/// of the router `[experts, hidden]`, the fused gate and up `[experts, 2·inter, hidden]` (per
/// expert the first `inter` rows gate, the next up) and the down `[experts, hidden, inter]`, and
/// the total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeBundle {
    pub router: u64,
    pub gate_up: u64,
    pub down: u64,
    pub bytes: u64,
}

/// A MoE layer's fp8 bundle ([`fp8_moe_bundle`]): [`MoeBundle`]'s regions over e4m3 experts, and
/// the f32 per-channel scales after them, `[experts, 2·inter]` (each expert's gate then up rows)
/// and `[experts, hidden]`, each region at a [`crate::tape::CUDA_ARENA_ALIGN`]-byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8MoeBundle {
    pub experts: MoeBundle,
    pub gate_up_scale: u64,
    pub down_scale: u64,
}

/// The fp8 bundle of one MoE layer: the bf16 router `[experts, hidden]` at byte 0 when `router`
/// (Qwen's; Gemma's bank has none), then the e4m3 fused gate and up `[experts, 2·inter, hidden]`
/// and down `[experts, hidden, inter]`, gathered from the checkpoint's per-expert tensors, then
/// their f32 scales. `None` when a byte count overflows `u64`.
pub const fn fp8_moe_layout(
    experts: u32,
    hidden: u32,
    inter: u32,
    router: bool,
) -> Option<Fp8MoeBundle> {
    let (e, h, i) = (experts as u64, hidden as u64, inter as u64);
    let align = crate::tape::CUDA_ARENA_ALIGN;
    let gate_up = match router {
        true => match (e * h).checked_mul(CUDA_ACT_BYTES) {
            Some(b) => b,
            None => return None,
        },
        false => 0,
    };
    let regions = [e * 2 * i * h, e * h * i, e * 2 * i * 4, e * h * 4];
    let mut at = [0u64; 4];
    let mut end = gate_up;
    let mut r = 0;
    while r < 4 {
        let Some(start) = end.checked_next_multiple_of(align) else {
            return None;
        };
        at[r] = start;
        let Some(next) = start.checked_add(regions[r]) else {
            return None;
        };
        end = next;
        r += 1;
    }
    Some(Fp8MoeBundle {
        experts: MoeBundle {
            router: 0,
            gate_up: at[0],
            down: at[1],
            bytes: end,
        },
        gate_up_scale: at[2],
        down_scale: at[3],
    })
}

/// [`fp8_moe_layout`]'s experts' regions, as an expert step reads its slab.
pub const fn fp8_moe_bundle(
    experts: u32,
    hidden: u32,
    inter: u32,
    router: bool,
) -> Option<MoeBundle> {
    match fp8_moe_layout(experts, hidden, inter, router) {
        Some(b) => Some(b.experts),
        None => None,
    }
}

/// [`MoeBundle`] for one layer's geometry, or `None` when a byte count overflows `u64`.
pub const fn moe_bundle(experts: u32, hidden: u32, inter: u32) -> Option<MoeBundle> {
    let (e, h, i) = (experts as u64, hidden as u64, inter as u64);
    let Some(gate_up) = (e * h).checked_mul(CUDA_ACT_BYTES) else {
        return None;
    };
    let Some(gu) = (e * 2 * i).checked_mul(h) else {
        return None;
    };
    let Some(gu) = gu.checked_mul(CUDA_ACT_BYTES) else {
        return None;
    };
    let Some(down) = gate_up.checked_add(gu) else {
        return None;
    };
    let Some(d) = (e * h).checked_mul(i) else {
        return None;
    };
    let Some(d) = d.checked_mul(CUDA_ACT_BYTES) else {
        return None;
    };
    let Some(bytes) = down.checked_add(d) else {
        return None;
    };
    Some(MoeBundle {
        router: 0,
        gate_up,
        down,
        bytes,
    })
}

/// Gemma's expert bank, [`moe_bundle`] with no router: the fused gate and up `[experts, 2·inter,
/// hidden]` at byte 0, then the down `[experts, hidden, inter]`. Its `router` is 0 and empty.
/// `None` when a byte count overflows `u64`.
pub const fn expert_bank(experts: u32, hidden: u32, inter: u32) -> Option<MoeBundle> {
    let Some(b) = moe_bundle(experts, hidden, inter) else {
        return None;
    };
    Some(MoeBundle {
        router: 0,
        gate_up: 0,
        down: b.down - b.gate_up,
        bytes: b.bytes - b.gate_up,
    })
}

/// How a step uses a layer's paged KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvUse {
    /// It writes K and V at the forward's slots.
    Write,
    /// It reads the sequence's cached K and V.
    Read,
}

/// A weight a step reads, and the bytes it reads of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightRead {
    pub ix: WeightIx,
    pub bytes: u64,
}

/// What one step touches: the arena spans it reads, in operand order, the span it writes, a
/// second span it writes (a two-output kernel's), the weight it reads, the KV-cache layer it
/// writes or reads (an index into [`crate::tape::KvTape::layers`]), and the GDN state layer it
/// reads and writes (an index into [`crate::tape::GdnTape::layers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepExtents {
    pub reads: [Option<ArenaSpan>; 4],
    pub writes: ArenaSpan,
    pub second_write: Option<ArenaSpan>,
    pub weight: Option<WeightRead>,
    pub kv: Option<(u32, KvUse)>,
    pub gdn: Option<u32>,
    /// The write may equal read 0 exactly, the kernel's `out == x` contract.
    pub in_place: bool,
}

impl CudaStep {
    /// A decode GEMV's K split and its pairs (`gemv_splits`): the dense GEMV and a GEMV router at
    /// one pair, the expert GEMV at `m · top_k`; `None` for every other step. Every variant is
    /// named, so a new step states whether it is one.
    pub const fn gemv_split(&self) -> Option<(u32, u32, u32)> {
        match *self {
            Self::Gemv(GemmBf16 { n, k, .. }) | Self::GemvFp8(GemmBf16 { n, k, .. }) => {
                Some((n, 1, gemv_splits(n, k, 1)))
            }
            Self::RouterLogits(RouterLogitsBf16 {
                experts,
                hidden,
                kernel: MatmulKernel::Gemv,
                ..
            }) => Some((experts, 1, gemv_splits(experts, hidden, 1))),
            Self::ExpertGemv(g) => {
                let (n, k) = g.n_k();
                let Some(pairs) = g.m.checked_mul(g.top_k) else {
                    return None;
                };
                Some((n, pairs, gemv_splits(n, k, pairs)))
            }
            Self::RouterLogits(_)
            | Self::Embed(_)
            | Self::Gemm(_)
            | Self::GemmNarrow(_)
            | Self::RmsNorm(_)
            | Self::Add(_)
            | Self::SiluMul(_)
            | Self::Rope(_)
            | Self::GateSplit(_)
            | Self::GateApply(_)
            | Self::RopeAppend(_)
            | Self::AttnPrefill(_)
            | Self::AttnDecode(_)
            | Self::GatedDeltaNet(_)
            | Self::MoeRoute(_)
            | Self::ExpertSort(_)
            | Self::ExpertGemm(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::LayerScale(_)
            | Self::SoftCap(_)
            | Self::RouterNorm(_)
            | Self::GemmCol64(_)
            // A grouped GEMV's members each take their own split, at their own place in the
            // scratch ([`GemmGroupBf16::gemv_scratch_before`]); [`Self::gemv_scratch`] sums them.
            | Self::GemmGroup(_)
            | Self::GemmNarrowGroup(_)
            | Self::GemvGroup(_)
            | Self::AddRmsNorm(_)
            | Self::NormAdd(_)
            | Self::GateUpAct(_)
            | Self::ExpertGateUpAct(_)
            | Self::GdnDecode(_)
            | Self::QuantRowsFp8(_)
            | Self::GemmFp8(_)
            | Self::GemmCol64Fp8(_)
            | Self::GemmNarrowFp8(_)
            | Self::GemmGroupFp8(_)
            | Self::GemmNarrowGroupFp8(_)
            | Self::GemvGroupFp8(_) => None,
        }
    }

    /// The grouped kernel a GEMM step's kernel has (#1049), and the GEMM: `Some` for the tiled
    /// GEMM, the narrow GEMM and the GEMV, the three a `SiblingMatmuls` fold groups; `None` for
    /// every other step. Every variant is named, so a new step states whether it groups.
    pub const fn group_kind(&self) -> Option<(GroupKind, GemmBf16)> {
        match *self {
            Self::Gemm(g) => Some((GroupKind::Gemm, g)),
            Self::GemmNarrow(g) => Some((GroupKind::GemmNarrow, g)),
            Self::Gemv(g) => Some((GroupKind::Gemv, g)),
            Self::GemmFp8(g) => Some((GroupKind::GemmFp8, g)),
            Self::GemmNarrowFp8(g) => Some((GroupKind::GemmNarrowFp8, g)),
            Self::GemvFp8(g) => Some((GroupKind::GemvFp8, g)),
            Self::QuantRowsFp8(_)
            | Self::GemmCol64Fp8(_)
            | Self::GemmGroupFp8(_)
            | Self::GemmNarrowGroupFp8(_)
            | Self::GemvGroupFp8(_)
            | Self::Embed(_)
            | Self::RmsNorm(_)
            | Self::Add(_)
            | Self::SiluMul(_)
            | Self::Rope(_)
            | Self::GateSplit(_)
            | Self::GateApply(_)
            | Self::RopeAppend(_)
            | Self::AttnPrefill(_)
            | Self::AttnDecode(_)
            | Self::GatedDeltaNet(_)
            | Self::RouterLogits(_)
            | Self::MoeRoute(_)
            | Self::ExpertSort(_)
            | Self::ExpertGemm(_)
            | Self::ExpertGemv(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::LayerScale(_)
            | Self::SoftCap(_)
            | Self::RouterNorm(_)
            | Self::GemmCol64(_)
            | Self::GemmGroup(_)
            | Self::GemmNarrowGroup(_)
            | Self::GemvGroup(_)
            | Self::AddRmsNorm(_)
            | Self::NormAdd(_)
            | Self::GateUpAct(_)
            | Self::ExpertGateUpAct(_)
            | Self::GdnDecode(_) => None,
        }
    }

    /// A grouped GEMM step's kind and group; `None` for every other step.
    pub const fn group(&self) -> Option<(GroupKind, GemmGroupBf16)> {
        match *self {
            Self::GemmGroup(g) => Some((GroupKind::Gemm, g)),
            Self::GemmNarrowGroup(g) => Some((GroupKind::GemmNarrow, g)),
            Self::GemvGroup(g) => Some((GroupKind::Gemv, g)),
            Self::GemmGroupFp8(g) => Some((GroupKind::GemmFp8, g)),
            Self::GemmNarrowGroupFp8(g) => Some((GroupKind::GemmNarrowFp8, g)),
            Self::GemvGroupFp8(g) => Some((GroupKind::GemvFp8, g)),
            Self::QuantRowsFp8(_)
            | Self::GemmFp8(_)
            | Self::GemmCol64Fp8(_)
            | Self::GemmNarrowFp8(_)
            | Self::GemvFp8(_)
            | Self::Embed(_)
            | Self::Gemm(_)
            | Self::GemmNarrow(_)
            | Self::Gemv(_)
            | Self::RmsNorm(_)
            | Self::Add(_)
            | Self::SiluMul(_)
            | Self::Rope(_)
            | Self::GateSplit(_)
            | Self::GateApply(_)
            | Self::RopeAppend(_)
            | Self::AttnPrefill(_)
            | Self::AttnDecode(_)
            | Self::GatedDeltaNet(_)
            | Self::RouterLogits(_)
            | Self::MoeRoute(_)
            | Self::ExpertSort(_)
            | Self::ExpertGemm(_)
            | Self::ExpertGemv(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::LayerScale(_)
            | Self::SoftCap(_)
            | Self::RouterNorm(_)
            | Self::GemmCol64(_)
            | Self::AddRmsNorm(_)
            | Self::NormAdd(_)
            | Self::GateUpAct(_)
            | Self::ExpertGateUpAct(_)
            | Self::GdnDecode(_) => None,
        }
    }

    /// The input an fp8 GEMM of more than one row (or a group, or a gate/up pair of them, or an
    /// expert GEMM over an fp8 bank: its sorted or activation rows) reads quantized, `(a, m, k)`:
    /// lowered, `a` is its bf16 input, which the lowering quantizes before it; on a tape, `a` is
    /// the quant scratch the [`Self::QuantRowsFp8`] before it wrote, which [`quant_chain`] checks.
    /// `None` for every other step, the one-row GEMV included.
    pub const fn quantized_input(&self) -> Option<(ArenaOffset, u32, u32)> {
        if let Self::ExpertGemm(
            g @ ExpertGemmBf16 {
                weight: MoeWeight::Fp8Bundle | MoeWeight::Fp8Own,
                ..
            },
        ) = *self
        {
            let rows = g
                .max_tiles
                .checked_mul(crate::op_abi::CUDA_EXPERT_TILE_ROWS);
            return match rows {
                Some(rows) => Some((g.a, rows, g.n_k().1)),
                None => None,
            };
        }
        let g = match *self {
            Self::GemmFp8(g) | Self::GemmCol64Fp8(g) | Self::GemmNarrowFp8(g) => g,
            Self::GemmGroupFp8(g) | Self::GemmNarrowGroupFp8(g) => g.member(g.first),
            Self::GateUpAct(GateUpActBf16 {
                gate,
                kernel: GateUpKernel::GemmFp8 | GateUpKernel::GemmCol64Fp8,
                ..
            }) => gate,
            _ => return None,
        };
        Some((g.a, g.m, g.k))
    }

    /// The kernel steps the step's launch computes: a grouped GEMM's members, each as the GEMM
    /// whose bits it writes, in order; every other step itself. `CudaTape::validate` checks each
    /// part as a step of its own, and a grouped step's members against each other.
    pub fn parts(&self) -> [Option<CudaStep>; 3] {
        if let Some((kind, g)) = self.group() {
            return g.members().map(|m| m.map(|m| kind.step(g.member(m))));
        }
        // A gate/up step's GEMMs, each its own kernel step, then the activation over their
        // colours: what the chain it folds computes, each part checked as that step.
        if let Self::GateUpAct(g) = self {
            return [
                Some(g.kernel.step(g.gate)),
                Some(g.kernel.step(g.up)),
                Some(g.act.step(g.activation())),
            ];
        }
        if let Self::ExpertGateUpAct(g) = self {
            return [
                Some(Self::ExpertGemv(g.gate)),
                Some(Self::ExpertGemv(g.up)),
                Some(g.act.step(g.activation())),
            ];
        }
        [Some(*self), None, None]
    }

    /// The GEMV scratch bytes the step writes: [`gemv_scratch_bytes`] of [`Self::gemv_split`], 0
    /// for a step that is no decode GEMV or runs at one split; a grouped GEMV's, the sum of its
    /// members' (each holds its partials at once); `None` on overflow.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        if let Self::GemvGroup(g) | Self::GemvGroupFp8(g) = self {
            return g.gemv_scratch_before(3);
        }
        if let Self::GateUpAct(g) = self {
            return g.gemv_scratch();
        }
        if let Self::ExpertGateUpAct(g) = self {
            return g.gemv_scratch();
        }
        match self.gemv_split() {
            Some((n, pairs, splits)) => gemv_scratch_bytes(n, pairs, splits),
            None => Some(0),
        }
    }

    /// What the step's kernel reads and writes, in bytes; `None` when a byte count overflows
    /// `u64`. Each arm states its own kernel's operands, next to that kernel's registry row:
    /// `CudaTape::validate` checks a tape against it. A grouped GEMM's are its first member's:
    /// `validate` checks every member's through [`Self::parts`].
    pub const fn extents(&self) -> Option<StepExtents> {
        match *self {
            Self::GemmGroup(g) | Self::GemmNarrowGroup(g) | Self::GemvGroup(g) => {
                Self::Gemm(g.member(g.first)).extents()
            }
            Self::GemmGroupFp8(g) | Self::GemmNarrowGroupFp8(g) | Self::GemvGroupFp8(g) => {
                match self.group() {
                    Some((kind, _)) => kind.step(g.member(g.first)).extents(),
                    None => None,
                }
            }
            // The bf16 rows; the e4m3 rows and their scales, the quant scratch.
            Self::QuantRowsFp8(q) => {
                let (Some(xb), Some(qb)) = (bf16_bytes(q.rows, q.cols), q.bytes()) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(q.x, xb)), None, None, None],
                    writes: span(q.q, qb),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            // The quantized rows (the GEMV's: the bf16 row) and the fp8 weight with its scales.
            Self::GemmFp8(GemmBf16 { a, b, c, m, n, k })
            | Self::GemmCol64Fp8(GemmBf16 { a, b, c, m, n, k })
            | Self::GemmNarrowFp8(GemmBf16 { a, b, c, m, n, k })
            | Self::GemvFp8(GemmBf16 { a, b, c, m, n, k }) => {
                let ab = match self {
                    Self::GemvFp8(_) => bf16_bytes(m, k),
                    _ => QuantRowsFp8 {
                        x: a,
                        q: a,
                        rows: m,
                        cols: k,
                    }
                    .bytes(),
                };
                let (Some(ab), Some(cb), Some(w)) = (ab, bf16_bytes(m, n), fp8_weight(n, k)) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(a, ab)), None, None, None],
                    writes: span(c, cb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: b,
                        bytes: w.bytes,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::Embed(EmbedBf16 {
                tokens,
                table,
                out,
                rows,
                hidden,
                vocab,
            }) => {
                let (Some(ob), Some(tb)) = (bf16_bytes(rows, hidden), bf16_bytes(vocab, hidden))
                else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(tokens, u32_bytes(rows))), None, None, None],
                    writes: span(out, ob),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: table,
                        bytes: tb,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::Gemm(GemmBf16 { a, b, c, m, n, k })
            | Self::GemmCol64(GemmBf16 { a, b, c, m, n, k })
            | Self::GemmNarrow(GemmBf16 { a, b, c, m, n, k })
            | Self::Gemv(GemmBf16 { a, b, c, m, n, k }) => {
                let (Some(ab), Some(bb), Some(cb)) =
                    (bf16_bytes(m, k), bf16_bytes(n, k), bf16_bytes(m, n))
                else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(a, ab)), None, None, None],
                    writes: span(c, cb),
                    second_write: None,
                    weight: Some(WeightRead { ix: b, bytes: bb }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::RmsNorm(RmsNormBf16 {
                x,
                gain,
                y,
                rows,
                cols,
                ..
            })
            | Self::RouterNorm(RmsNormBf16 {
                x,
                gain,
                y,
                rows,
                cols,
                ..
            }) => {
                let (Some(xb), Some(gb)) = (bf16_bytes(rows, cols), bf16_bytes(1, cols)) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, xb)), None, None, None],
                    writes: span(y, xb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: gain,
                        bytes: gb,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            // The projections' input; the activation's output. The weights are each part's
            // (`validate` checks every part's through [`Self::parts`]), the first one here.
            Self::GateUpAct(g) => {
                // An fp8 pair's are its first projection's on its own: the quantized rows.
                let (ab, bb) = match g.kernel.step(g.gate) {
                    s @ (Self::GemmFp8(_) | Self::GemmCol64Fp8(_) | Self::GemvFp8(_)) => {
                        match s.extents() {
                            Some(StepExtents {
                                reads: [Some(a), ..],
                                weight: Some(w),
                                ..
                            }) => (Some(a.bytes), Some(w.bytes)),
                            _ => return None,
                        }
                    }
                    _ => (
                        bf16_bytes(g.gate.m, g.gate.k),
                        bf16_bytes(g.gate.n, g.gate.k),
                    ),
                };
                let (Some(ab), Some(cb), Some(bb)) = (ab, bf16_bytes(g.gate.m, g.gate.n), bb)
                else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(g.gate.a, ab)), None, None, None],
                    writes: span(g.out, cb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: g.gate.b,
                        bytes: bb,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            // The token rows and the picks; the activation rows.
            Self::ExpertGateUpAct(g) => {
                let Some(e) = Self::ExpertGemv(g.gate).extents() else {
                    return None;
                };
                Some(StepExtents {
                    writes: span(g.out, e.writes.bytes),
                    ..e
                })
            }
            // The norm's gain and both inputs; the sum, then the norm.
            Self::AddRmsNorm(f) => {
                let RmsNormBf16 {
                    gain,
                    y,
                    rows,
                    cols,
                    ..
                } = f.norm;
                let (Some(xb), Some(gb)) = (bf16_bytes(rows, cols), bf16_bytes(1, cols)) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(f.a, xb)), Some(span(f.b, xb)), None, None],
                    writes: span(f.sum, xb),
                    second_write: Some(span(y, xb)),
                    weight: Some(WeightRead {
                        ix: gain,
                        bytes: gb,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            // The norm's input and gain and the residual; the add's output, the norm's `y`.
            Self::NormAdd(NormAddBf16 { norm, residual }) => {
                let RmsNormBf16 {
                    x,
                    gain,
                    y,
                    rows,
                    cols,
                    ..
                } = norm;
                let (Some(xb), Some(gb)) = (bf16_bytes(rows, cols), bf16_bytes(1, cols)) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, xb)), Some(span(residual, xb)), None, None],
                    writes: span(y, xb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: gain,
                        bytes: gb,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::Add(AddBf16 {
                a,
                b,
                out,
                rows,
                cols,
            })
            | Self::SiluMul(SiluMulBf16 {
                gate: a,
                up: b,
                out,
                rows,
                cols,
            })
            | Self::GeluMul(SiluMulBf16 {
                gate: a,
                up: b,
                out,
                rows,
                cols,
            })
            | Self::GateApply(GateApplyBf16 {
                attn: a,
                gate: b,
                out,
                rows,
                cols,
            }) => {
                let Some(bytes) = bf16_bytes(rows, cols) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(a, bytes)), Some(span(b, bytes)), None, None],
                    writes: span(out, bytes),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::Rope(RopeBf16 {
                x,
                table,
                out,
                rows,
                cols,
                rot_dim,
                ..
            }) => {
                let (Some(xb), Some(tb)) = (bf16_bytes(rows, cols), bf16_bytes(rows, rot_dim))
                else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, xb)), Some(span(table, tb)), None, None],
                    writes: span(out, xb),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::GateSplit(GateSplitBf16 {
                qg,
                q,
                gate,
                rows,
                half_cols,
                ..
            }) => {
                let Some(hb) = bf16_bytes(rows, half_cols) else {
                    return None;
                };
                let Some(qgb) = hb.checked_mul(2) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(qg, qgb)), None, None, None],
                    writes: span(q, hb),
                    second_write: Some(span(gate, hb)),
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::RopeAppend(RopeAppendBf16 {
                k,
                v,
                table,
                slot_mapping,
                k_out,
                layer,
                rows,
                cols,
                rot_dim,
                ..
            }) => {
                let (Some(kb), Some(tb)) = (bf16_bytes(rows, cols), bf16_bytes(rows, rot_dim))
                else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(k, kb)),
                        Some(span(v, kb)),
                        Some(span(table, tb)),
                        Some(span(slot_mapping, u32_bytes(rows))),
                    ],
                    writes: span(k_out, kb),
                    second_write: None,
                    weight: None,
                    kv: Some((layer, KvUse::Write)),
                    gdn: None,
                    in_place: false,
                })
            }
            Self::AttnPrefill(AttnPrefillBf16 {
                q,
                block_table,
                sequence,
                out,
                layer,
                rows,
                q_heads,
                head_dim,
                ..
            })
            | Self::AttnDecode(AttnPrefillBf16 {
                q,
                block_table,
                sequence,
                out,
                layer,
                rows,
                q_heads,
                head_dim,
                ..
            }) => {
                let Some(cols) = q_heads.checked_mul(head_dim) else {
                    return None;
                };
                let Some(qb) = bf16_bytes(rows, cols) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(q, qb)),
                        Some(span(block_table, u32_bytes(CUDA_MAX_BLOCKS_PER_SEQ))),
                        Some(span(sequence, u32_bytes(2))),
                        None,
                    ],
                    writes: span(out, qb),
                    second_write: None,
                    weight: None,
                    kv: Some((layer, KvUse::Read)),
                    gdn: None,
                    in_place: false,
                })
            }
            // The GDN step's reads and its output; no scratch (`kernels/gdn_decode.cu`).
            Self::GdnDecode(g) => {
                let Some(e) = Self::GatedDeltaNet(g).extents() else {
                    return None;
                };
                Some(StepExtents {
                    second_write: None,
                    ..e
                })
            }
            Self::GatedDeltaNet(
                g @ GatedDeltaNetBf16 {
                    qkv,
                    z,
                    a,
                    b,
                    bundle,
                    out,
                    scratch,
                    layer,
                    rows,
                    num_v_heads,
                    ..
                },
            ) => {
                let (Some(conv_dim), Some(value_dim)) = (g.conv_dim(), g.value_dim()) else {
                    return None;
                };
                let (Some(qb), Some(vb), Some(hb)) = (
                    bf16_bytes(rows, conv_dim),
                    bf16_bytes(rows, value_dim),
                    bf16_bytes(rows, num_v_heads),
                ) else {
                    return None;
                };
                let (Some(sc), Some(bu)) = (g.scratch_layout(), g.bundle_layout()) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(qkv, qb)),
                        Some(span(z, vb)),
                        Some(span(a, hb)),
                        Some(span(b, hb)),
                    ],
                    writes: span(out, vb),
                    second_write: Some(span(scratch, sc.bytes)),
                    weight: Some(WeightRead {
                        ix: bundle,
                        bytes: bu.bytes,
                    }),
                    kv: None,
                    gdn: Some(layer),
                    in_place: false,
                })
            }
            Self::RouterLogits(RouterLogitsBf16 {
                x,
                bundle,
                weight,
                logits,
                m,
                experts,
                hidden,
                inter,
                ..
            }) => {
                let (Some(xb), Some(lb), Some(bu)) = (
                    bf16_bytes(m, hidden),
                    bf16_bytes(m, experts),
                    weight.router_bytes(experts, hidden, inter),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, xb)), None, None, None],
                    writes: span(logits, lb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: bundle,
                        bytes: bu,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::MoeRoute(MoeRouteBf16 {
                logits,
                topk_idx,
                scores,
                m,
                experts,
                top_k,
                expert_scale,
                ..
            }) => {
                let (Some(lb), Some(pairs)) = (bf16_bytes(m, experts), m.checked_mul(top_k)) else {
                    return None;
                };
                let weight = match expert_scale {
                    Some(ix) => Some(WeightRead {
                        ix,
                        bytes: experts as u64 * CUDA_ACT_BYTES,
                    }),
                    None => None,
                };
                Some(StepExtents {
                    reads: [Some(span(logits, lb)), None, None, None],
                    writes: span(topk_idx, u32_bytes(pairs)),
                    second_write: Some(span(scores, u32_bytes(pairs))),
                    weight,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::ExpertSort(s) => {
                let Some(l) = s.layout() else {
                    return None;
                };
                let Some(xb) = bf16_bytes(s.m, s.hidden) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(s.x, xb)),
                        Some(span(s.topk_idx, u32_bytes(l.pairs))),
                        None,
                        None,
                    ],
                    writes: span(s.rows, l.rows_bytes),
                    // The index regions, `chunk_hist` to the end of `pos`, one span:
                    // `MoeTape::validate` checks each region's own offset.
                    second_write: Some(span(s.chunk_hist, l.index_bytes)),
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::ExpertGemm(g) => {
                let rows = g.max_tiles as u64 * crate::op_abi::CUDA_EXPERT_TILE_ROWS as u64;
                let (n, k) = g.n_k();
                // An fp8 bank's rows are the quantized ones, e4m3 then their scales.
                let ab = match self.quantized_input() {
                    Some((a, rows, cols)) => QuantRowsFp8 {
                        x: a,
                        q: a,
                        rows,
                        cols,
                    }
                    .bytes(),
                    None => (rows * k as u64).checked_mul(CUDA_ACT_BYTES),
                };
                let (Some(ab), Some(cb), Some(bu)) = (
                    ab,
                    (rows * n as u64).checked_mul(CUDA_ACT_BYTES),
                    g.weight.experts(g.experts, g.hidden, g.inter),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(g.a, ab)),
                        Some(span(g.offsets, u32_bytes(g.experts + 1))),
                        Some(span(g.tile_expert, u32_bytes(g.max_tiles))),
                        None,
                    ],
                    writes: span(g.c, cb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: g.bundle,
                        bytes: bu.bytes,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::ExpertGemv(g) => {
                let Some(pairs) = g.m.checked_mul(g.top_k) else {
                    return None;
                };
                let a_rows = match g.rows {
                    PairRows::Token => g.m,
                    PairRows::Pair => pairs,
                };
                let (n, k) = g.n_k();
                let (Some(ab), Some(cb), Some(bu)) = (
                    bf16_bytes(a_rows, k),
                    bf16_bytes(pairs, n),
                    g.weight.experts(g.experts, g.hidden, g.inter),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(g.a, ab)),
                        Some(span(g.topk_idx, u32_bytes(pairs))),
                        None,
                        None,
                    ],
                    writes: span(g.c, cb),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: g.bundle,
                        bytes: bu.bytes,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::ExpertCombine(ExpertCombineBf16 {
                down,
                pos,
                scores,
                out,
                m,
                top_k,
                hidden,
                down_rows,
            }) => {
                let (Some(pairs), Some(db), Some(ob)) = (
                    m.checked_mul(top_k),
                    bf16_bytes(down_rows, hidden),
                    bf16_bytes(m, hidden),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(down, db)),
                        Some(span(scores, u32_bytes(pairs))),
                        match pos {
                            Some(p) => Some(span(p, u32_bytes(pairs))),
                            None => None,
                        },
                        None,
                    ],
                    writes: span(out, ob),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::GateScale(GateScaleBf16 {
                routed,
                shared,
                g,
                out,
                rows,
                cols,
            }) => {
                let (Some(bytes), Some(gb)) = (bf16_bytes(rows, cols), bf16_bytes(rows, 1)) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(routed, bytes)),
                        Some(span(shared, bytes)),
                        Some(span(g, gb)),
                        None,
                    ],
                    writes: span(out, bytes),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::Scale(ScaleBf16 {
                x, out, rows, cols, ..
            })
            | Self::RmsNormUnit(RmsNormUnitBf16 {
                x,
                y: out,
                rows,
                cols,
                ..
            }) => {
                let Some(bytes) = bf16_bytes(rows, cols) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, bytes)), None, None, None],
                    writes: span(out, bytes),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            // `pointwise_bf16.cu` reads each element before the same thread writes it, so the
            // soft-cap may write over its input.
            Self::SoftCap(SoftCapBf16 {
                x, out, rows, cols, ..
            }) => {
                let Some(bytes) = bf16_bytes(rows, cols) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, bytes)), None, None, None],
                    writes: span(out, bytes),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: true,
                })
            }
            Self::LayerScale(LayerScaleBf16 {
                x,
                scale,
                out,
                rows,
                cols,
            }) => {
                let Some(bytes) = bf16_bytes(rows, cols) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(x, bytes)), None, None, None],
                    writes: span(out, bytes),
                    second_write: None,
                    weight: Some(WeightRead {
                        ix: scale,
                        bytes: CUDA_ACT_BYTES,
                    }),
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
        }
    }
}

/// The sort's regions for its geometry: the pairs `P = m · top_k`, the sorted rows' bytes
/// (`max_tiles · 128` rows of `hidden` bf16), and the index regions' bytes from `chunk_hist` to
/// the end of `pos`, each region at a multiple of 256: `chunk_hist` `[⌈P/256⌉, experts]` u32,
/// `counts` `[experts]` u32, `offsets` `[experts + 1]` u32, `tile_expert` `[max_tiles]` i32, `pos`
/// `[P]` u32, in that order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortLayout {
    pub pairs: u32,
    pub chunk_hist: u64,
    pub counts: u64,
    pub offsets: u64,
    pub tile_expert: u64,
    pub pos: u64,
    pub index_bytes: u64,
    pub rows_bytes: u64,
}

/// Pairs per `chunk_hist` row: one sort CTA's chunk (`kernels/moe_sort.cu`).
pub const MOE_SORT_CHUNK: u32 = 256;

/// The byte offsets, from `chunk_hist`, of the sort's index regions for `pairs` pairs over
/// `experts` experts and `max_tiles` tiles, each 256-aligned, and their total: what
/// [`SortLayout`] states. `None` on overflow.
pub const fn sort_index_regions(pairs: u32, experts: u32, max_tiles: u32) -> Option<[u64; 6]> {
    let chunks = pairs.div_ceil(MOE_SORT_CHUNK) as u64;
    let Some(counts) = aligned_after(0, chunks * experts as u64 * 4) else {
        return None;
    };
    let Some(offsets) = aligned_after(counts, experts as u64 * 4) else {
        return None;
    };
    let Some(tile_expert) = aligned_after(offsets, (experts as u64 + 1) * 4) else {
        return None;
    };
    let Some(pos) = aligned_after(tile_expert, max_tiles as u64 * 4) else {
        return None;
    };
    let index_bytes = pos + pairs as u64 * 4;
    Some([0, counts, offsets, tile_expert, pos, index_bytes])
}

/// The 256-aligned end of `bytes` placed at `at`, or `None` on overflow.
const fn aligned_after(at: u64, bytes: u64) -> Option<u64> {
    let Some(end) = at.checked_add(bytes) else {
        return None;
    };
    end.checked_next_multiple_of(GDN_ALIGN)
}

impl ExpertSortBf16 {
    /// The sort's region layout ([`SortLayout`]), or `None` on overflow.
    pub const fn layout(&self) -> Option<SortLayout> {
        let Some(pairs) = self.m.checked_mul(self.top_k) else {
            return None;
        };
        let Some([chunk_hist, counts, offsets, tile_expert, pos, index_bytes]) =
            sort_index_regions(pairs, self.experts, self.max_tiles)
        else {
            return None;
        };
        let rows = self.max_tiles as u64 * crate::op_abi::CUDA_EXPERT_TILE_ROWS as u64;
        let Some(rows_bytes) = (rows * self.hidden as u64).checked_mul(CUDA_ACT_BYTES) else {
            return None;
        };
        Some(SortLayout {
            pairs,
            chunk_hist,
            counts,
            offsets,
            tile_expert,
            pos,
            index_bytes,
            rows_bytes,
        })
    }
}

impl CudaStep {
    /// The weight index the step reads, for the compiler to rewrite (two indices naming one
    /// checkpoint tensor become one). Each arm is [`Self::extents`]' `weight`.
    pub fn weight_mut(&mut self) -> Option<&mut WeightIx> {
        match self {
            Self::Embed(EmbedBf16 { table, .. }) => Some(table),
            Self::Gemm(GemmBf16 { b, .. })
            | Self::GemmCol64(GemmBf16 { b, .. })
            | Self::GemmNarrow(GemmBf16 { b, .. })
            | Self::Gemv(GemmBf16 { b, .. })
            | Self::GemmFp8(GemmBf16 { b, .. })
            | Self::GemmCol64Fp8(GemmBf16 { b, .. })
            | Self::GemmNarrowFp8(GemmBf16 { b, .. })
            | Self::GemvFp8(GemmBf16 { b, .. }) => Some(b),
            Self::RmsNorm(RmsNormBf16 { gain, .. })
            | Self::RouterNorm(RmsNormBf16 { gain, .. })
            | Self::AddRmsNorm(AddRmsNormBf16 {
                norm: RmsNormBf16 { gain, .. },
                ..
            })
            | Self::NormAdd(NormAddBf16 {
                norm: RmsNormBf16 { gain, .. },
                ..
            }) => Some(gain),
            Self::LayerScale(LayerScaleBf16 { scale, .. }) => Some(scale),
            Self::MoeRoute(MoeRouteBf16 { expert_scale, .. }) => expert_scale.as_mut(),
            Self::GatedDeltaNet(GatedDeltaNetBf16 { bundle, .. })
            | Self::GdnDecode(GatedDeltaNetBf16 { bundle, .. }) => Some(bundle),
            Self::RouterLogits(RouterLogitsBf16 { bundle, .. }) => Some(bundle),
            Self::ExpertGemm(ExpertGemmBf16 { bundle, .. })
            | Self::ExpertGemv(ExpertGemvBf16 { bundle, .. })
            | Self::ExpertGateUpAct(ExpertGateUpActBf16 {
                gate: ExpertGemvBf16 { bundle, .. },
                ..
            }) => Some(bundle),
            // The gate's; [`Self::weights_mut`] gives the up's too.
            Self::GateUpAct(GateUpActBf16 {
                gate: GemmBf16 { b, .. },
                ..
            }) => Some(b),
            // A grouped GEMM's first member's; [`Self::weights_mut`] gives every member's.
            Self::GemmGroup(GemmGroupBf16 { first, .. })
            | Self::GemmNarrowGroup(GemmGroupBf16 { first, .. })
            | Self::GemvGroup(GemmGroupBf16 { first, .. })
            | Self::GemmGroupFp8(GemmGroupBf16 { first, .. })
            | Self::GemmNarrowGroupFp8(GemmGroupBf16 { first, .. })
            | Self::GemvGroupFp8(GemmGroupBf16 { first, .. }) => Some(&mut first.b),
            Self::Add(_)
            | Self::SiluMul(_)
            | Self::Rope(_)
            | Self::GateSplit(_)
            | Self::GateApply(_)
            | Self::RopeAppend(_)
            | Self::AttnPrefill(_)
            | Self::AttnDecode(_)
            | Self::ExpertSort(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::SoftCap(_)
            | Self::QuantRowsFp8(_) => None,
        }
    }

    /// Every weight index the step reads, for the compiler's rewrite: a grouped GEMM's members'
    /// in order, every other step's [`Self::weight_mut`].
    pub fn weights_mut(&mut self) -> Vec<&mut WeightIx> {
        match self {
            Self::GemmGroup(g)
            | Self::GemmNarrowGroup(g)
            | Self::GemvGroup(g)
            | Self::GemmGroupFp8(g)
            | Self::GemmNarrowGroupFp8(g)
            | Self::GemvGroupFp8(g) => {
                let GemmGroupBf16 {
                    first,
                    second,
                    third,
                    ..
                } = g;
                let mut all = vec![&mut first.b, &mut second.b];
                all.extend(third.as_mut().map(|t| &mut t.b));
                all
            }
            Self::GateUpAct(GateUpActBf16 { gate, up, .. }) => vec![&mut gate.b, &mut up.b],
            Self::ExpertGateUpAct(ExpertGateUpActBf16 { gate, up, .. }) => {
                vec![&mut gate.bundle, &mut up.bundle]
            }
            other => other.weight_mut().into_iter().collect(),
        }
    }
}

/// `rows × cols` bf16 elements' bytes, or `None` on overflow.
const fn bf16_bytes(rows: u32, cols: u32) -> Option<u64> {
    (rows as u64 * cols as u64).checked_mul(CUDA_ACT_BYTES)
}

/// `n` u32 values' bytes.
const fn u32_bytes(n: u32) -> u64 {
    n as u64 * 4
}

const fn span(offset: ArenaOffset, bytes: u64) -> ArenaSpan {
    ArenaSpan { offset, bytes }
}
