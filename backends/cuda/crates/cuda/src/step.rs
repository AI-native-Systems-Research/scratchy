// SPDX-License-Identifier: Apache-2.0
//! CUDA's step: one or two kernel launches of a lowered tape, the counterpart of metal's
//! `MetalStep`.

/// Re-exported so a static tape names it beside the steps that carry it
/// (`scratchy_target_cuda::step::GainConvention`).
pub use scratchy_ir::model_facts::RopePairing;
pub use scratchy_subtile::subtile_ir::GainConvention;
use serde::Serialize;

/// The constants the kernels share ([`crate::kernel_consts`]), under the paths they always had.
pub use crate::kernel_consts::{
    ACT_QUANT_MAX_COLS, ADD_BF16_COLS, ARGMAX_CHUNK, ARGMAX_MAX_VOCAB, ATTN_D512_BF16_HEAD_DIM,
    ATTN_DECODE_BF16_HEAD_DIM, ATTN_DECODE_BF16_MAX_GROUP, ATTN_DECODE_MAX_SPLITS,
    ATTN_PREFILL_BF16_HEAD_DIM, ATTN_PREFILL_BF16_ROWS, CUDA_KV_BLOCK_SIZE,
    CUDA_MAX_BLOCKS_PER_SEQ, EMBED_BF16_COLS, EXPERT_GATE_UP_FP8_INTER_MULTIPLE,
    EXPERT_GATE_UP_FP8_K_MULTIPLE, EXPERT_GEMM_N_TILES, GATE_APPLY_BF16_COLS, GATE_SCALE_BF16_COLS,
    GATE_SPLIT_BF16_HEAD_DIM, GATED_ACT_GELU, GATED_ACT_SILU, GATHER_LAST_ROW_BF16_COLS,
    GDN_CONV_CHANNEL_GROUP, GDN_CONV_KERNEL_RANGE, GDN_CONV_TOKEN_BLOCK, GDN_DECODE_HEAD_VS,
    GDN_DECODE_MAX_HEADS, GDN_GATING_NV_MULTIPLE, GDN_HEAD_K, GDN_HEAD_V_MAX, GDN_HEAD_V_MULTIPLE,
    GDN_SCAN_TOKEN_BLOCK, GEMM_BF16_TILE, GEMM_COL64_TILE, GEMM_FP8_K_MULTIPLE,
    GEMM_NARROW_BF16_MAX_N, GEMM_NARROW_BF16_MULTIPLE, GEMM_NARROW_BF16_ONE_K_MULTIPLE,
    GEMV_BF16_MULTIPLE, GEMV_BLOCK_ROWS, GEMV_CHUNK, GEMV_MAX_SPLITS, GEMV_TARGET_CTAS,
    GEMV_TC_MAX_PAIRS, GEMV_TC_N_MULTIPLE, H100_SXM_SMS, KV_APPEND_BF16_HEAD_DIMS,
    MOE_COMBINE_BF16_CHUNK, MOE_ROUTE_EXPERTS, MOE_ROUTE_MAX_TOP_K, MOE_SORT_CHUNK, NORM_ROW_COLS,
    NORM_ROW_FORM_ADD_NORM, NORM_ROW_FORM_NORM, NORM_ROW_FORM_NORM_ADD, NORM_ROW_MAX_COLS,
    RMS_NORM_BF16_COLS, ROPE_BF16_HALF_HEAD_DIM, ROPE_BF16_KERNEL_HEAD_MULTIPLE,
    ROPE_PAIRING_HALF_HEAD, ROPE_PAIRING_HALF_ROTATED, ROW_MAX_COLS, SCALE_BF16_COLS,
    SILU_MUL_BF16_COLS, SOFT_CAP_BF16_COLS,
};
use crate::tape::{ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, WeightIx};

/// The `f32`s a step carries, each as its exact bits: a step is `Eq` (a tape is compared by value),
/// so a float field holds its bits and its type reads them back as the float.
macro_rules! f32_bits_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
        pub struct $name(pub u32);
        impl $name {
            /// `v`, held as its bits.
            pub const fn new(v: f32) -> Self {
                Self(v.to_bits())
            }
            /// The float.
            pub const fn get(self) -> f32 {
                f32::from_bits(self.0)
            }
        }
    )*};
}

f32_bits_newtype! {
    /// An RMSNorm's `eps`.
    RmsNormEps,
    /// The attention softmax's scale (`head_dim^-0.5` for qwen3.5).
    AttnScale,
    /// A multiplier: a pointwise scale's, and the scaled routing's `hidden^-0.5`.
    Scale,
    /// A tanh soft cap on logits.
    SoftCap,
}

/// The dimensions and counts a step or tape carries, each its own type: a GEMM's `n` cannot be
/// passed where its `k` is meant. `get()` reads the number; `Display` and `Debug` print it as the
/// bare number, so every message reads as before.
macro_rules! u32_newtype {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
        pub struct $name(pub u32);
        impl $name {
            /// The number.
            pub const fn get(self) -> u32 {
                self.0
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }
        impl From<$name> for u32 {
            fn from(v: $name) -> u32 {
                v.0
            }
        }
        impl From<$name> for u64 {
            fn from(v: $name) -> u64 {
                u64::from(v.0)
            }
        }
    )*};
}

u32_newtype! {
    /// The rows a step runs: a GEMM's `m`, a row op's rows, a bucket's tokens or pairs.
    Rows,
    /// A row op's columns (a gate split's `half_cols`).
    Cols,
    /// A GEMM's output columns, its weight's rows.
    NDim,
    /// A GEMM's inner dimension, its input's columns.
    KDim,
    /// Per-head feature width.
    HeadDim,
    /// The leading lanes of a head that a rope rotates.
    RotDim,
    /// A rotary table's frequency denominator: the rotated width, or the head dim (proportional).
    FreqDim,
    /// Query heads.
    NumQHeads,
    /// Key/value heads.
    NumKvHeads,
    /// A model layer id.
    LayerId,
    /// The residual stream's width.
    HiddenSize,
    /// A feed-forward's (an expert's) inner width.
    IntermediateSize,
    /// The token embedding's rows (the lm_head's, the argmax's row width).
    VocabSize,
    /// Routed experts of a MoE layer.
    NumExperts,
    /// Experts each token routes to.
    TopK,
    /// The sorted expert rows' tiles.
    MaxTiles,
    /// An expert GEMM's column tile.
    NTile,
    /// The rows of one expert tile.
    TileRows,
    /// A Gated-DeltaNet layer's key or value heads.
    GdnHeads,
    /// A Gated-DeltaNet key or value head's width.
    GdnHeadDim,
    /// A Gated-DeltaNet conv's taps.
    ConvKernel,
    /// The slices a split-K GEMV or a split decode attention cuts its work into.
    Splits,
    /// The fewest key tiles a decode attention split takes.
    MinTiles,
    /// The last keys a sliding-window query sees.
    AttnWindow,
    /// Token slots per paged-KV block.
    BlockSize,
    /// Block-table entries per sequence.
    MaxBlocksPerSeq,
    /// A tape's rows: its bucket's.
    BucketM,
    ConvDim,
}

/// A rope's base frequency (`rope_theta`), as its exact `f64` bits, so a rotary table is `Eq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RopeTheta(pub u64);

impl RopeTheta {
    /// `v`, held as its bits.
    pub const fn new(v: f64) -> Self {
        Self(v.to_bits())
    }
    /// The base frequency.
    pub const fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

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
    GemmNarrow(GemmNarrowBf16),
    /// The bf16 GEMV, a one-row GEMM, lowered from `SubOp::MatmulTile` of a one-row bucket whose
    /// `n` the narrow GEMM does not take (`kernels/gemv_bf16.cu`): the operands of [`Self::Gemm`]
    /// with `m = 1`, another kernel. The lowering's `MatmulTile` arm chooses it at expansion.
    Gemv(GemvBf16),
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
    /// two launches, `kernels/rope_bf16.cu` then `kernels/kv_append_fp8.cu` (the cache is e4m3).
    RopeAppend(RopeAppendBf16),
    /// Causal grouped-query attention of a prefill bucket over the paged KV cache, lowered from
    /// `SubOp::AttnDecode` when the bucket has more than one row (`kernels/attn_prefill_fp8kv.cu`).
    AttnPrefill(AttnPrefillBf16),
    /// One query row's grouped-query attention over the paged KV cache, lowered from
    /// `SubOp::AttnDecode` when the bucket has one row (`kernels/attn_decode_fp8kv.cu`): the
    /// operands of [`Self::AttnPrefill`] with `rows = 1`, another kernel.
    AttnDecode(AttnDecodeBf16),
    /// One prefill chunk of a Gated-DeltaNet linear-attention layer, lowered from
    /// `SubOp::GatedDeltaNet`: four stream-ordered launches, `kernels/gdn_conv1d.cu`,
    /// `kernels/gdn_gating.cu`, `kernels/gdn_chunk_scan.cu` (FP8 S8 F6a: the scan 64 tokens at a
    /// time) then `kernels/gdn_rms_norm_gated.cu`.
    GatedDeltaNet(GdnPrefillBf16),
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
    /// An fp8 bank's sort (FP8 S8 P1): [`Self::ExpertSort`] on the token rows' e4m3 quantization
    /// (`x`, in the quant scratch, its [`QuantRowsFp8`]), each pair's sorted row its token's e4m3
    /// row and scale (`rows`, e4m3 then the scales, `QuantRowsFp8`'s layout over the capacity rows),
    /// which the bank's gate and up read as their quantized input (`kernels/moe_sort.cu`,
    /// `scratchy_cuda_moe_sort_fp8`).
    ExpertSortFp8(ExpertSortBf16),
    /// An fp8 bank's act and its down's input quantization over the live rows in one launch (FP8 S8
    /// P1), in place of the act over every sorted row and a [`Self::QuantRowsFp8`] of them
    /// (`kernels/act_quant_fp8.cu`, `scratchy_cuda_expert_act_quant_fp8`).
    ExpertActQuant(ExpertActQuantFp8),
    /// An fp8 bank's prefill gate, up and act + quantize (FP8 S8 P3(b)) in two launches: the fused
    /// grouped GEMM (`kernels/gemm_fp8.cu`'s `expert_gate_up_template`), which writes the act into
    /// the gate's region, then the live rows' quantize of it into the act's `q_act`
    /// (`kernels/act_quant_fp8.cu`). The bits of its parts, [`Self::ExpertGemm`] of the gate and of
    /// the up then [`Self::ExpertActQuant`]; the up's region is not written.
    ExpertGateUpGemmFp8(ExpertGateUpGemmFp8),
    /// One projection of every sorted pair, lowered from `SubOp::ExpertMatmul` of a sorted bucket:
    /// one grouped GEMM over the expert-padded tiles (`kernels/expert_gemm_bf16.cu`).
    ExpertGemm(ExpertGemmBf16),
    /// One projection of every pair, each from its own token row, lowered from
    /// `SubOp::ExpertMatmul` of a gathered bucket (`kernels/expert_gemv_bf16.cu`).
    ExpertGemv(ExpertGemvSplitBf16),
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
    /// The sequence's last row moved to the front of a decoder's result-matmul input, in place
    /// (FP8 S8 F2), lowered from `SubOp::SampleRowsGather`: row `q_len − 1` of `x [rows, cols]`
    /// copied over row 0, `q_len` read from the KV sequence region on the device
    /// (`kernels/gather_last_row_bf16.cu`, `scratchy_cuda_gather_last_row_bf16`).
    GatherLastRow(GatherLastRowBf16),
    /// Gemma's router pre-norm, `x̂ · router.scale` into the MoE scratch's router-norm region,
    /// lowered from `SubOp::RouterNorm`: the operands of [`Self::RmsNorm`], its gain the bundle's
    /// `router.scale` (`kernels/rms_norm_bf16.cu`, `scratchy_cuda_rms_norm_bf16`).
    RouterNorm(RmsNormBf16),
    /// The bf16 GEMM on 64-column tiles, lowered from `SubOp::MatmulTile` when `n` is a multiple
    /// of [`GEMM_COL64_TILE`]'s and not of [`GEMM_BF16_TILE`]'s, above
    /// [`GEMM_NARROW_BF16_MAX_N`]: the operands of [`Self::Gemm`], `kernels/gemm_bf16.cu`'s
    /// template at one 64-column B block (`scratchy_cuda_gemm_col64_bf16_abt`, #924).
    GemmCol64(GemmCol64Bf16),
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
    GemvGroup(GemvGroupBf16),
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
    ExpertGateUpAct(ExpertGateUpGemvBf16),
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
    GemmFp8(GemmFp8),
    /// [`Self::GemmCol64`] over an fp8 weight, as [`Self::GemmFp8`].
    GemmCol64Fp8(GemmCol64Fp8),
    /// [`Self::GemmNarrow`] over an fp8 weight (the 27B's `in_proj_a`/`b`), as [`Self::GemmFp8`].
    GemmNarrowFp8(GemmNarrowFp8),
    /// [`Self::Gemv`] over an fp8 weight: `a` is the bf16 row, which the GEMV quantizes in its
    /// prologue (`kernels/gemv_fp8.cu`).
    GemvFp8(GemvFp8),
    /// [`Self::GemmGroup`] over fp8 weights, as [`Self::GemmFp8`] each.
    GemmGroupFp8(GemmGroupBf16),
    /// [`Self::GemmNarrowGroup`] over fp8 weights (the 27B's `in_proj_a` + `in_proj_b`), as
    /// [`Self::GemmFp8`] each.
    GemmNarrowGroupFp8(GemmGroupBf16),
    /// [`Self::GemvGroup`] over fp8 weights, as [`Self::GemvFp8`] each.
    GemvGroupFp8(GemvGroupBf16),
    /// [`Self::GemvFp8`] on the tensor cores (FP8 S8 T2, `kernels/gemv_fp8_tc.cu`'s B8), the
    /// shapes [`gemv_tc_takes`] takes: `a` is the quant scratch, the one e4m3 row the step before
    /// wrote and its scale after it, as [`Self::GemmFp8`]'s; `c = bf16((a·Bᵀ) · w_scale · a_scale)`
    /// (with the residual's `with_residual` when it has one), WGMMA's fp8 accumulation.
    GemvTcFp8(GemvTcFp8),
    /// [`Self::GemvGroupFp8`] on the tensor cores, each member as [`Self::GemvTcFp8`].
    GemvGroupTcFp8(GemmGroupBf16),
    /// [`Self::GateUpAct`] of one row on the tensor cores: the gate and the up as B8's f32
    /// members off the quantized row (its GEMV scratch, `2 · n` f32), then the act + quantize
    /// (`kernels/act_quant_fp8.cu`): `out = bf16(act(gate, up))`, and that row's e4m3 bytes and
    /// scale at `q_act`, the quant scratch's second slot, which the down reads.
    GateUpTcFp8(GateUpTcFp8),
    /// [`Self::ExpertGemv`] over an fp8 bank on the tensor cores (B8's expert form): `a` the
    /// quant scratch's rows, as [`Self::GemvTcFp8`]'s.
    ExpertGemvTcFp8(ExpertGemvBf16),
    /// [`Self::ExpertGateUpAct`] over an fp8 bank on the tensor cores, as [`Self::GateUpTcFp8`]:
    /// B8's expert form for the gate and the up (f32), then the act + quantize of the pairs' rows.
    ExpertGateUpTcFp8(ExpertGateUpTcFp8),
    /// A one-row [`Self::RmsNorm`] and the [`Self::QuantRowsFp8`] of its `y` right after it, in
    /// one launch (FP8 S8 N3, `kernels/rms_norm_quant_row.cu`): `y` as N3's norm computes it, then
    /// its e4m3 row and scale at `q`, the quant scratch's first slot, `quant_rows_fp8` of `y` bit
    /// for bit.
    RmsNormQuantFp8(RmsNormQuantFp8),
    /// [`Self::AddRmsNorm`] of one row and the quantization of its `y`, as [`Self::RmsNormQuantFp8`].
    AddRmsNormQuantFp8(AddRmsNormQuantFp8),
}

/// [`CudaStep::RmsNormQuantFp8`]: the norm, and where its quantization goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RmsNormQuantFp8 {
    pub norm: RmsNormBf16,
    pub q: ArenaOffset,
}

impl RmsNormQuantFp8 {
    /// The quantization's write, as the [`QuantRowsFp8`] of `y` it equals.
    pub const fn quant_write(&self) -> QuantRowsFp8 {
        QuantRowsFp8 {
            x: self.norm.y,
            q: self.q,
            rows: self.norm.rows,
            cols: self.norm.cols,
        }
    }
}

/// [`CudaStep::AddRmsNormQuantFp8`]: the add-norm, and where its quantization goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AddRmsNormQuantFp8 {
    pub fold: AddRmsNormBf16,
    pub q: ArenaOffset,
}

impl AddRmsNormQuantFp8 {
    /// As [`RmsNormQuantFp8::quant_write`].
    pub const fn quant_write(&self) -> QuantRowsFp8 {
        RmsNormQuantFp8 {
            norm: self.fold.norm,
            q: self.q,
        }
        .quant_write()
    }
}

/// Whether a norm, add-norm or norm-add of `rows` × `cols` plays the one-row norm (FP8 S8 N3):
/// the decode bucket's one token row, at a width its kernel takes. Every other norm plays the
/// row kernels (`kernels/rms_norm_bf16.cu`).
pub const fn norm_row_takes(rows: u32, cols: u32) -> bool {
    rows == 1 && cols.is_multiple_of(NORM_ROW_COLS) && cols <= NORM_ROW_MAX_COLS
}

/// [`CudaStep::GateUpTcFp8`]: the gate/up fold it computes (its `kernel` the CUDA-core one,
/// [`GateUpKernel::GemvFp8`], whose bits it does not keep), and where its act + quantize writes the
/// activation's e4m3 row and scale (the quant scratch's second slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateUpTcFp8 {
    pub fold: GateUpActBf16,
    pub q_act: ArenaOffset,
}

/// [`CudaStep::ExpertGateUpTcFp8`], as [`GateUpTcFp8`] for the experts' fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGateUpTcFp8 {
    pub fold: ExpertGateUpActBf16,
    pub q_act: ArenaOffset,
}

impl GateUpTcFp8 {
    /// The act + quantize's write, as the [`QuantRowsFp8`] of `out` it equals.
    pub const fn quant_write(&self) -> QuantRowsFp8 {
        QuantRowsFp8 {
            x: self.fold.out,
            q: self.q_act,
            rows: self.fold.gate.m,
            cols: Cols(self.fold.gate.n.get()),
        }
    }

    /// The GEMV scratch it writes: the gate's f32 rows, then the up's ([`gemv_tc_f32_bytes`]).
    pub const fn gemv_scratch(&self) -> Option<u64> {
        gemv_tc_f32_bytes(self.fold.gate.m.get(), self.fold.gate.n.get())
    }
}

impl ExpertGateUpTcFp8 {
    /// As [`GateUpTcFp8::quant_write`], over the pairs' rows.
    pub const fn quant_write(&self) -> Option<QuantRowsFp8> {
        let Some(rows) = self
            .fold
            .gate
            .m
            .get()
            .checked_mul(self.fold.gate.top_k.get())
        else {
            return None;
        };
        Some(QuantRowsFp8 {
            x: self.fold.out,
            q: self.q_act,
            rows: Rows(rows),
            cols: Cols(self.fold.gate.inter.get()),
        })
    }

    /// As [`GateUpTcFp8::gemv_scratch`], over the pairs' rows.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        let Some(rows) = self
            .fold
            .gate
            .m
            .get()
            .checked_mul(self.fold.gate.top_k.get())
        else {
            return None;
        };
        gemv_tc_f32_bytes(rows, self.fold.gate.inter.get())
    }
}

/// The f32 gate and up rows a tensor-core gate/up writes into the GEMV scratch: `rows · n` f32
/// each, the up's at the gate's bytes rounded up to [`crate::tape::CUDA_ARENA_ALIGN`]. `None` on
/// overflow.
pub const fn gemv_tc_f32_bytes(rows: u32, n: u32) -> Option<u64> {
    let Some(one) =
        (rows as u64 * n as u64 * 4).checked_next_multiple_of(crate::tape::CUDA_ARENA_ALIGN)
    else {
        return None;
    };
    one.checked_mul(2)
}

/// The weight bytes a decode fp8 GEMV launch reads at or above which it runs on the tensor cores
/// ([`gemv_tc_takes`]). shortcut: 1, so every shape the kernel takes switches, until T1's
/// measured table of the tensor-core GEMV against the CUDA-core one sets it.
pub const GEMV_TC_MIN_BYTES: u64 = 1;

/// Whether a decode fp8 GEMV of members `ns` (0 for an absent one) over `k`, at `pairs` pairs,
/// runs on the tensor cores (FP8 S8): every member's `n` and `k` a non-zero
/// multiple of [`GEMV_TC_N_MULTIPLE`], at most [`GEMV_TC_MAX_PAIRS`] pairs, and its weight bytes
/// per launch (`pairs · Σ n · k`) at least [`GEMV_TC_MIN_BYTES`]. A function of the shape alone,
/// beside [`gemv_splits`]: the lowering picks the kernel with it at expansion.
pub const fn gemv_tc_takes(ns: [u32; 3], k: u32, pairs: u32) -> bool {
    if k == 0 || !k.is_multiple_of(GEMV_TC_N_MULTIPLE) || pairs == 0 || pairs > GEMV_TC_MAX_PAIRS {
        return false;
    }
    let mut rows = 0u64;
    let mut i = 0;
    while i < ns.len() {
        if i == 0 && ns[i] == 0 {
            return false;
        }
        if !ns[i].is_multiple_of(GEMV_TC_N_MULTIPLE) {
            return false;
        }
        rows += ns[i] as u64;
        i += 1;
    }
    rows * k as u64 * pairs as u64 >= GEMV_TC_MIN_BYTES
}

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
    pub rows: Rows,
    pub cols: Cols,
}

impl QuantRowsFp8 {
    /// The scales' byte offset from `q`: the e4m3 rows, rounded up to
    /// [`crate::tape::CUDA_ARENA_ALIGN`].
    pub const fn scale_at(&self) -> Option<u64> {
        (self.rows.get() as u64 * self.cols.get() as u64)
            .checked_next_multiple_of(crate::tape::CUDA_ARENA_ALIGN)
    }

    /// The quant scratch bytes it writes: the rows, then their scales.
    pub const fn bytes(&self) -> Option<u64> {
        match self.scale_at() {
            Some(at) => at.checked_add(self.rows.get() as u64 * 4),
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

/// The quant scratch's liveness over `steps`, in order. The scratch has two slots, each its own
/// `q`: the first holds what a [`CudaStep::QuantRowsFp8`] wrote, the second what a tensor-core
/// gate/up's act + quantize wrote (FP8 S8 T2, [`CudaStep::quant_write`]). Every fp8 step that reads
/// a quantized input ([`CudaStep::quantized_input`]) reads exactly the rows the last write to its
/// slot wrote (at that `q`, of its `rows` and `cols`), the reads of a step before its writes; and
/// every [`CudaStep::QuantRowsFp8`] is read before the next write to its slot (an act + quantize
/// is not: its down may read the bf16 activation). That the quant write read the consumer's own
/// bf16 input is the lowering's to state: the consumer names only the scratch.
pub fn quant_chain(steps: &[CudaStep]) -> Result<(), QuantChainError> {
    // Per slot: the writing step, its write, whether it was read, whether it must be.
    let mut live: [Option<(usize, QuantRowsFp8, bool, bool)>; 2] = [None, None];
    for (step, s) in steps.iter().enumerate() {
        if let Some((x, m, k)) = s.quantized_input() {
            let slot = live.iter_mut().flatten().find(|(_, q, _, _)| q.q == x);
            match slot {
                Some((_, q, read, _)) if (q.rows, q.cols) == (Rows(m), Cols(k)) => *read = true,
                _ => return Err(QuantChainError::Unquantized { step }),
            }
        }
        let Some(w) = s.quant_write() else {
            continue;
        };
        let required = matches!(
            s,
            CudaStep::QuantRowsFp8(_)
                | CudaStep::RmsNormQuantFp8(_)
                | CudaStep::AddRmsNormQuantFp8(_)
        );
        let at = live
            .iter()
            .position(|l| l.is_some_and(|(_, q, _, _)| q.q == w.q));
        let at = match at.or_else(|| live.iter().position(Option::is_none)) {
            Some(at) => at,
            // A third slot: no lowering writes one.
            None => return Err(QuantChainError::Overwritten { step }),
        };
        if let Some((was, _, false, true)) = live[at] {
            return Err(QuantChainError::Overwritten { step: was });
        }
        live[at] = Some((step, w, false, required));
    }
    match live
        .iter()
        .flatten()
        .find(|(_, _, read, required)| *required && !read)
    {
        Some((at, ..)) => Err(QuantChainError::Overwritten { step: *at }),
        None => Ok(()),
    }
}

/// Whether a `GatedDeltaNet` step of `rows` rows scans with the recurrent kernel
/// (`kernels/gdn_scan.cu`) rather than the chunked one (`kernels/gdn_chunk_scan.cu`, FP8 S8 F6a): at
/// a one-row bucket, a decode the one-launch `GdnDecode` does not take (the 27B's three value heads
/// a key head), where the chunked scan would pay a 64-token chunk for one token; every prefill
/// bucket's is the chunked scan. A constant of each step (its `rows`), chosen at expansion.
pub const fn gdn_scan_is_recurrent(rows: u32) -> bool {
    rows == 1
}

/// Which scan a [`CudaStep::GatedDeltaNet`] plays, chosen when the step is emitted
/// ([`gdn_scan_is_recurrent`] of its rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GdnScan {
    /// The recurrent scan (`kernels/gdn_scan.cu`).
    Recurrent,
    /// The chunked scan (`kernels/gdn_chunk_scan.cu`).
    Chunked,
}

impl GdnScan {
    /// The scan of a step of `rows` rows.
    pub const fn of(rows: u32) -> Self {
        if gdn_scan_is_recurrent(rows) {
            Self::Recurrent
        } else {
            Self::Chunked
        }
    }
}

/// A Gated-DeltaNet bucket's step ([`CudaStep::GatedDeltaNet`]): its layer, and its scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GdnPrefillBf16 {
    pub gdn: GatedDeltaNetBf16,
    pub scan: GdnScan,
}

impl GdnPrefillBf16 {
    /// `gdn` at its rows' scan.
    pub const fn of(gdn: GatedDeltaNetBf16) -> Self {
        Self {
            gdn,
            scan: GdnScan::of(gdn.rows.get()),
        }
    }
}

/// The activation a gate/up step applies (`kernels/scratchy_ops.cuh`'s op struct): SwiGLU's
/// `silu_mul` or GeGLU's `gelu_tanh_mul`, the op of [`CudaStep::SiluMul`] or [`CudaStep::GeluMul`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum GatedOp {
    Silu,
    Gelu,
}

impl GatedOp {
    /// The launchers' code for it ([`GATED_ACT_SILU`], [`GATED_ACT_GELU`]).
    pub const fn code(self) -> u32 {
        match self {
            Self::Silu => GATED_ACT_SILU,
            Self::Gelu => GATED_ACT_GELU,
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
    /// The GEMV, each projection at this K split.
    Gemv(GemvSplit),
    /// The fp8 kernels (FP8 S2): each projection [`CudaStep::GemmFp8`], [`CudaStep::GemmCol64Fp8`]
    /// or [`CudaStep::GemvFp8`] on its own, each scaled by its own `w_scale` before the
    /// activation, rounded once.
    GemmFp8,
    GemmCol64Fp8,
    /// The fp8 GEMV, each projection at this K split.
    GemvFp8(GemvSplit),
}

impl GateUpKernel {
    /// The step `g` is on its own.
    pub const fn step(self, g: GemmBf16) -> CudaStep {
        let g = g.matmul();
        match self {
            Self::Gemm => CudaStep::Gemm(GemmBf16::of(g)),
            Self::GemmCol64 => CudaStep::GemmCol64(GemmCol64Bf16::of(g)),
            Self::Gemv(split) => CudaStep::Gemv(GemvBf16 {
                split,
                ..GemvBf16::of(g)
            }),
            Self::GemmFp8 => CudaStep::GemmFp8(GemmFp8::of(g)),
            Self::GemmCol64Fp8 => CudaStep::GemmCol64Fp8(GemmCol64Fp8::of(g)),
            Self::GemvFp8(split) => CudaStep::GemvFp8(GemvFp8 {
                split,
                ..GemvFp8::of(g)
            }),
        }
    }

    /// Whether it is a GEMV, and its split.
    pub const fn gemv_split(self) -> Option<GemvSplit> {
        match self {
            Self::Gemv(split) | Self::GemvFp8(split) => Some(split),
            Self::Gemm | Self::GemmCol64 | Self::GemmFp8 | Self::GemmCol64Fp8 => None,
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
            cols: Cols(self.gate.n.get()),
        }
    }

    /// The bytes of GEMV scratch the fused GEMV writes: the gate's partials, then the up's, each
    /// at its own split ([`gemv_scratch_bytes`]); 0 for the GEMMs. `None` on overflow.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        let Some(split) = self.kernel.gemv_split() else {
            return Some(0);
        };
        let Some(one) = gemv_scratch_bytes(self.gate.n.get(), 1, split.splits.get()) else {
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
            rows: Rows(self.gate.m.get() * self.gate.top_k.get()),
            cols: Cols(self.gate.inter.get()),
        }
    }
}

/// The routed experts' gate and up GEMVs in one launch ([`CudaStep::ExpertGateUpAct`]): the fold,
/// and the K split each projection takes, chosen when the step is emitted ([`GemvSplit::of`] of
/// the gate's shape at `m · top_k` pairs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGateUpGemvBf16 {
    pub fold: ExpertGateUpActBf16,
    pub split: GemvSplit,
}

impl ExpertGateUpGemvBf16 {
    /// `fold` at the split its gate's shape takes.
    pub const fn of(fold: ExpertGateUpActBf16) -> Self {
        Self {
            fold,
            split: ExpertGemvSplitBf16::split_of(&fold.gate),
        }
    }

    /// The GEMV scratch the fused launch writes: the gate's partials, then the up's.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        let (n, _) = self.fold.gate.n_k();
        let Some(pairs) = self
            .fold
            .gate
            .m
            .get()
            .checked_mul(self.fold.gate.top_k.get())
        else {
            return None;
        };
        let Some(one) = gemv_scratch_bytes(n, pairs, self.split.splits.get()) else {
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
    GemvTcFp8,
}

impl GroupKind {
    /// A member as the step it is on its own (a GEMV at its shape's split, [`GemvBf16::of`]).
    pub const fn step(self, g: Matmul) -> CudaStep {
        match self {
            Self::Gemm => CudaStep::Gemm(GemmBf16::of(g)),
            Self::GemmNarrow => CudaStep::GemmNarrow(GemmNarrowBf16::of(g)),
            Self::Gemv => CudaStep::Gemv(GemvBf16::of(g)),
            Self::GemmFp8 => CudaStep::GemmFp8(GemmFp8::of(g)),
            Self::GemmNarrowFp8 => CudaStep::GemmNarrowFp8(GemmNarrowFp8::of(g)),
            Self::GemvFp8 => CudaStep::GemvFp8(GemvFp8::of(g)),
            Self::GemvTcFp8 => CudaStep::GemvTcFp8(GemvTcFp8::of(GemvFp8::of(g))),
        }
    }

    /// The grouped step of `g` (a GEMV group at each member's split, [`GemvGroupBf16::of`]).
    pub const fn grouped(self, g: GemmGroupBf16) -> CudaStep {
        match self {
            Self::Gemm => CudaStep::GemmGroup(g),
            Self::GemmNarrow => CudaStep::GemmNarrowGroup(g),
            Self::Gemv => CudaStep::GemvGroup(GemvGroupBf16::of(g)),
            Self::GemmFp8 => CudaStep::GemmGroupFp8(g),
            Self::GemmNarrowFp8 => CudaStep::GemmNarrowGroupFp8(g),
            Self::GemvFp8 => CudaStep::GemvGroupFp8(GemvGroupBf16::of(g)),
            Self::GemvTcFp8 => CudaStep::GemvGroupTcFp8(g),
        }
    }
}

/// One weight of a grouped GEMM ([`GemmGroupBf16`]): `B [n, k]` and its output `C [m, n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GroupMember {
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub n: NDim,
}

/// Up to three GEMMs `C_i[m, n_i] = A[m, k] · B_i[n_i, k]ᵀ` that share `A`, in one launch: the
/// `first` member is the fold's driver, the others its siblings in tape order. Each member is the
/// [`GemmBf16`] [`Self::member`] names, whose kernel shape rule it meets on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemmGroupBf16 {
    pub a: ArenaOffset,
    pub m: Rows,
    pub k: KDim,
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
    pub const fn member(&self, g: GroupMember) -> Matmul {
        Matmul {
            a: self.a,
            b: g.b,
            c: g.c,
            m: self.m,
            n: g.n,
            k: self.k,
        }
    }
}

/// The head dims the attention, rope-append and KV-append kernels take: 256 (every Qwen layer,
/// Gemma's sliding layers) and 512 (Gemma's global layers, #885's 13c).
pub const ATTN_BF16_HEAD_DIMS: &[u32] = &[256, 512];

/// A sliding window is a non-zero multiple of this many keys, [`CUDA_KV_BLOCK_SIZE`]: a KV tile
/// is then wholly inside or wholly below the window but at the window's lower edge. Gemma's 1024
/// is 16 blocks.
pub const ATTN_WINDOW_MULTIPLE: u32 = CUDA_KV_BLOCK_SIZE;

/// Row `q_len − 1` of `x [rows, cols]` bf16 copied over its row 0, in place; `q_len` is entry 1
/// of the KV sequence region (`[prefix_len, q_len]` u32), written by the forward. A `q_len` of 0,
/// 1 or above `rows` copies nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GatherLastRowBf16 {
    pub x: ArenaOffset,
    pub sequence: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
}

/// `out = x · scale` over `[rows, cols]`, bf16 in and out, computed in f32 and rounded once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ScaleBf16 {
    pub x: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
    pub scale: Scale,
}

/// `y[r, c] = x[r, c] / rms(x[r, :])`, `rms(x) = sqrt(mean(x²) + eps)`, bf16 in and out, computed
/// in f32 and rounded once: [`RmsNormBf16`] with a unit gain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RmsNormUnitBf16 {
    pub x: ArenaOffset,
    pub y: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
    pub eps: RmsNormEps,
}

/// `out = x · scale[0]` over `[rows, cols]`, `scale` a `[1]` bf16 weight, computed in f32 and
/// rounded once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LayerScaleBf16 {
    pub x: ArenaOffset,
    pub scale: WeightIx,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
}

/// `out = cap · tanh(x / cap)` over `[rows, cols]`, bf16 in and out, computed in f32 and rounded
/// once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SoftCapBf16 {
    pub x: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
    pub cap: SoftCap,
}

/// [`RopePairing`] as the unit variant it is: `scratchy_ir` has no serde.
fn serialize_rope_pairing<S: serde::Serializer>(
    pairing: &RopePairing,
    s: S,
) -> Result<S::Ok, S::Error> {
    let (index, variant) = match pairing {
        RopePairing::Standard => (0, "Standard"),
        RopePairing::Proportional => (1, "Proportional"),
    };
    s.serialize_unit_variant("RopePairing", index, variant)
}

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
    pub rows: Rows,
    pub hidden: HiddenSize,
    pub vocab: VocabSize,
}

/// `C[m, n] = A[m, k] · B[n, k]ᵀ` before its kernel is chosen: what a matmul's lowering builds and
/// a fold's members are, and every GEMM payload's operands ([`CudaStep::gemm_view`]).
///
/// `A` and `C` are arena slots; `B` is a weight buffer in the checkpoint's `[N, K]` storage order
/// (the IR states it as `[K, N]`; the bytes are the checkpoint's). `C` overlaps neither operand:
/// CUDA's colour rule for a matmul reads its inputs across the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matmul {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub m: Rows,
    pub n: NDim,
    pub k: KDim,
}

/// One payload per tiled GEMM kernel, each the [`Matmul`] it computes: `m` rows, no epilogue.
macro_rules! gemm_payload {
    ($($(#[$doc:meta])* $name:ident;)*) => {$(
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
        pub struct $name {
            pub a: ArenaOffset,
            pub b: WeightIx,
            pub c: ArenaOffset,
            pub m: Rows,
            pub n: NDim,
            pub k: KDim,
        }

        impl $name {
            /// `g` on this kernel.
            pub const fn of(g: Matmul) -> Self {
                Self { a: g.a, b: g.b, c: g.c, m: g.m, n: g.n, k: g.k }
            }

            /// The matmul it computes.
            pub const fn matmul(&self) -> Matmul {
                Matmul { a: self.a, b: self.b, c: self.c, m: self.m, n: self.n, k: self.k }
            }
        }
    )*};
}

gemm_payload! {
    /// [`CudaStep::Gemm`]: bf16 `A` and `B`, `C` rounded once from f32, on [`GEMM_BF16_TILE`]s.
    GemmBf16;
    /// [`CudaStep::GemmCol64`]: [`GemmBf16`]'s operands on [`GEMM_COL64_TILE`]s.
    GemmCol64Bf16;
    /// [`CudaStep::GemmNarrow`]: [`GemmBf16`]'s operands, `n` at most [`GEMM_NARROW_BF16_MAX_N`].
    GemmNarrowBf16;
    /// [`CudaStep::GemmFp8`]: `a` the quant scratch (the e4m3 rows, each row's f32 scale after
    /// them), `b` an fp8 weight ([`Fp8Weight`]); `C = (A·Bᵀ) · w_scale[n] · a_scale[m]`, rounded
    /// once to bf16, on 128-column tiles.
    GemmFp8;
    /// [`CudaStep::GemmCol64Fp8`]: [`GemmFp8`]'s operands on 64-column tiles.
    GemmCol64Fp8;
    /// [`CudaStep::GemmNarrowFp8`]: [`GemmFp8`]'s operands on the narrow kernel.
    GemmNarrowFp8;
}

/// [`CudaStep::Gemv`], a one-row bf16 GEMM: `c [1, n] = a [1, k] · Bᵀ`, at its K split. `residual`
/// is the residual it adds its output to (FP8 S8 W4a, the one-row bucket's `MatvecEpilogue`): `c =
/// bf16(bf16(a · Bᵀ) + residual)`, `add_bf16.cu`'s add, `residual` a `[1, n]` arena slot and `c`
/// the add's own colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemvBf16 {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub n: NDim,
    pub k: KDim,
    pub residual: Option<ArenaOffset>,
    pub split: GemvSplit,
}

/// [`CudaStep::GemvFp8`]: [`GemvBf16`] over an fp8 weight ([`Fp8Weight`]); `a` is the bf16 row,
/// which the GEMV quantizes in its prologue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemvFp8 {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub n: NDim,
    pub k: KDim,
    pub residual: Option<ArenaOffset>,
    pub split: GemvSplit,
}

/// [`CudaStep::GemvTcFp8`]: [`GemvFp8`] on the tensor cores, no K split; `a` is the quant scratch,
/// the one e4m3 row the step before wrote and its scale after it, as [`GemmFp8`]'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemvTcFp8 {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub n: NDim,
    pub k: KDim,
    pub residual: Option<ArenaOffset>,
}

/// `g`'s one row: a GEMV's kernel rule chose it for one row only.
const fn one_row(g: Matmul) -> Matmul {
    assert!(g.m.get() == 1, "a GEMV computes one row");
    g
}

impl GemvBf16 {
    /// `g`, one row, at its shape's split (one pair), no residual.
    pub const fn of(g: Matmul) -> Self {
        let g = one_row(g);
        Self {
            a: g.a,
            b: g.b,
            c: g.c,
            n: g.n,
            k: g.k,
            residual: None,
            split: GemvSplit::of(g.n.get(), g.k.get(), 1),
        }
    }

    /// The matmul it computes, without its residual.
    pub const fn matmul(&self) -> Matmul {
        Matmul {
            a: self.a,
            b: self.b,
            c: self.c,
            m: Rows(1),
            n: self.n,
            k: self.k,
        }
    }
}

impl GemvFp8 {
    /// As [`GemvBf16::of`].
    pub const fn of(g: Matmul) -> Self {
        let g = one_row(g);
        Self {
            a: g.a,
            b: g.b,
            c: g.c,
            n: g.n,
            k: g.k,
            residual: None,
            split: GemvSplit::of(g.n.get(), g.k.get(), 1),
        }
    }

    /// As [`GemvBf16::matmul`].
    pub const fn matmul(&self) -> Matmul {
        Matmul {
            a: self.a,
            b: self.b,
            c: self.c,
            m: Rows(1),
            n: self.n,
            k: self.k,
        }
    }
}

impl GemvTcFp8 {
    /// `g` on the tensor cores, its residual kept.
    pub const fn of(g: GemvFp8) -> Self {
        Self {
            a: g.a,
            b: g.b,
            c: g.c,
            n: g.n,
            k: g.k,
            residual: g.residual,
        }
    }

    /// As [`GemvBf16::matmul`].
    pub const fn matmul(&self) -> Matmul {
        Matmul {
            a: self.a,
            b: self.b,
            c: self.c,
            m: Rows(1),
            n: self.n,
            k: self.k,
        }
    }
}

/// The one `n` the narrow GEMM takes off [`GEMM_NARROW_BF16_MULTIPLE`]: a one-column projection
/// (a shared expert's gate), stored one bf16 per row.
pub const GEMM_NARROW_BF16_ONE: u32 = 1;

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

/// A decode GEMV's K split, written into its step when the step is emitted ([`gemv_splits`] of the
/// step's shape and pairs). The player launches it as stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemvSplit {
    pub splits: Splits,
}

impl GemvSplit {
    /// The split of an `n` × `k` GEMV over `pairs` slices ([`gemv_splits`]).
    pub const fn of(n: u32, k: u32, pairs: u32) -> Self {
        Self {
            splits: Splits(gemv_splits(n, k, pairs)),
        }
    }
}

/// A grouped decode GEMV ([`CudaStep::GemvGroup`], [`CudaStep::GemvGroupFp8`]): the group, and
/// each member's K split, as [`GemmGroupBf16`]'s members (`third` exactly when it has one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GemvGroupBf16 {
    pub group: GemmGroupBf16,
    pub first: GemvSplit,
    pub second: GemvSplit,
    pub third: Option<GemvSplit>,
}

impl GemvGroupBf16 {
    /// `group` at each member's split, one pair.
    pub const fn of(group: GemmGroupBf16) -> Self {
        let k = group.k;
        Self {
            group,
            first: GemvSplit::of(group.first.n.get(), k.get(), 1),
            second: GemvSplit::of(group.second.n.get(), k.get(), 1),
            third: match group.third {
                Some(m) => Some(GemvSplit::of(m.n.get(), k.get(), 1)),
                None => None,
            },
        }
    }

    /// Each member's split, in [`GemmGroupBf16::members`]' order.
    pub const fn splits(&self) -> [Option<GemvSplit>; 3] {
        [Some(self.first), Some(self.second), self.third]
    }

    /// The GEMV scratch bytes of the members before `upto` (`upto` 3: all of them), each at its
    /// stored split ([`gemv_scratch_bytes`]): member `i`'s partials start at the bytes before it.
    /// `None` on overflow.
    pub const fn gemv_scratch_before(&self, upto: usize) -> Option<u64> {
        let (members, splits) = (self.group.members(), self.splits());
        let mut total = 0u64;
        let mut i = 0;
        while i < upto && i < 3 {
            if let (Some(g), Some(split)) = (members[i], splits[i]) {
                let Some(bytes) = gemv_scratch_bytes(g.n.get(), 1, split.splits.get()) else {
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
    pub rows: Rows,
    pub cols: Cols,
    pub eps: RmsNormEps,
    #[serde(serialize_with = "serialize_gain_convention")]
    pub convention: GainConvention,
    /// The kernel the player launches: chosen when the step is emitted ([`NormKernel::of`]).
    pub kernel: NormKernel,
}

/// Which kernel a norm, add-norm or norm-add plays (FP8 S8 N3), chosen when the step is emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum NormKernel {
    /// The one-row norm (`kernels/rms_norm_quant_row.cu`).
    Row,
    /// The row kernels (`kernels/rms_norm_bf16.cu`).
    Tiled,
}

impl NormKernel {
    /// The kernel of a `rows` × `cols` norm ([`norm_row_takes`]).
    pub const fn of(rows: u32, cols: u32) -> Self {
        if norm_row_takes(rows, cols) {
            Self::Row
        } else {
            Self::Tiled
        }
    }
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

/// `out = a + b` over `[rows, cols]`, bf16 in and out: one f32 add, one rounding. No broadcast:
/// all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AddBf16 {
    pub a: ArenaOffset,
    pub b: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
}

/// `out = gate / (1 + e^(−gate)) · up` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SiluMulBf16 {
    pub gate: ArenaOffset,
    pub up: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
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
    (64, RopePairing::Standard),
    (256, RopePairing::Standard),
    (128, RopePairing::Proportional),
];

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
    pub rows: Rows,
    pub cols: Cols,
    pub head_dim: HeadDim,
    pub rot_dim: RotDim,
    /// Which lane a rotated lane pairs with: `Standard` (the kernel's), or `Proportional`
    /// (Gemma's proportional rope, #885's 13b-3). `table` is the tape's table of the step's class.
    #[serde(serialize_with = "serialize_rope_pairing")]
    pub pairing: RopePairing,
}

/// The gate split's `head_dim` is a non-zero multiple of this (`kernels/gate_split_bf16.cu`: one
/// 16-byte vector per thread, never straddling a head), and `half_cols` a non-zero multiple of
/// `head_dim`. `rows` is non-zero and `rows · 2·half_cols` at most `i32::MAX`.
pub const GATE_SPLIT_BF16_HEAD_MULTIPLE: u32 = 8;

/// Per head `h` and lane `d`: `q[r, h·hd + d] = qg[r, 2h·hd + d]` and `gate[r, h·hd + d] =
/// qg[r, 2h·hd + hd + d]`, bf16 copies. `qg` is `[rows, 2·half_cols]`, `q` and `gate` are `[rows,
/// half_cols]`, and neither output overlaps `qg` or the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateSplitBf16 {
    pub qg: ArenaOffset,
    pub q: ArenaOffset,
    pub gate: ArenaOffset,
    pub rows: Rows,
    pub half_cols: Cols,
    pub head_dim: HeadDim,
}

/// `out = attn / (1 + e^(−gate))` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateApplyBf16 {
    pub attn: ArenaOffset,
    pub gate: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
}

// The worker picks the attention kernel by these names: they are exactly the head dims a KV step
// may take.
const _: () = assert!(
    ATTN_BF16_HEAD_DIMS.len() == 2
        && ATTN_BF16_HEAD_DIMS[0] == ATTN_PREFILL_BF16_HEAD_DIM
        && ATTN_BF16_HEAD_DIMS[0] == ATTN_DECODE_BF16_HEAD_DIM
        && ATTN_BF16_HEAD_DIMS[1] == ATTN_D512_BF16_HEAD_DIM
);
/// The decode attention's split over keys per head dim (FP8 S8 W6b): `(head_dim, splits,
/// min_tiles)`. Of the window's tiles of 64 keys a split takes at least `min_tiles`, so a sequence
/// of `2 · 64` keys or fewer is one split, the unsplit kernel's bits; longer ones take up to
/// `splits` CTAs a kv head, merged in split order (`kernels/attn_decode_merge.cu`). The H100 pod at
/// a8ce5637 chose 16 × 2 at both head dims: faster than one split at every kv_len 192 to 8192.
pub const ATTN_DECODE_SPLITS: [(u32, u32, u32); 2] = [
    (ATTN_DECODE_BF16_HEAD_DIM, 16, 2),
    (ATTN_D512_BF16_HEAD_DIM, 16, 2),
];
const _: () = assert!(
    ATTN_DECODE_SPLITS[0].1 <= ATTN_DECODE_MAX_SPLITS
        && ATTN_DECODE_SPLITS[1].1 <= ATTN_DECODE_MAX_SPLITS
);

/// `(splits, min_tiles)` of the decode attention at `head_dim` ([`ATTN_DECODE_SPLITS`]); `None`
/// at a head dim with no decode kernel.
pub const fn attn_decode_split(head_dim: u32) -> Option<(u32, u32)> {
    match attn_decode_split_at(head_dim) {
        Some(s) => Some((s.splits.get(), s.min_tiles.get())),
        None => None,
    }
}

/// The decode attention's split over keys, written into its step when the step is emitted
/// ([`ATTN_DECODE_SPLITS`] at its head dim): up to `splits` CTAs a kv head, each at least
/// `min_tiles` tiles of 64 keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AttnDecodeSplit {
    pub splits: Splits,
    pub min_tiles: MinTiles,
}

/// A decode bucket's attention ([`CudaStep::AttnDecode`]): the attention, its split over keys, and
/// the split scratch its partials take at that split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AttnDecodeBf16 {
    pub attn: AttnPrefillBf16,
    pub split: AttnDecodeSplit,
    pub scratch: AttnDecodeScratch,
}

/// Where a decode attention's split partials lie in the tape's split scratch
/// ([`attn_decode_split_scratch_bytes`] at its stored split), written into its step when the step
/// is emitted: the f32 rows from the scratch's start, each row's max and sum from `ml_offset`, and
/// `bytes` in all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AttnDecodeScratch {
    pub ml_offset: u64,
    pub bytes: u64,
}

impl AttnDecodeBf16 {
    /// `attn` at its head dim's split and that split's scratch; `None` at a head dim with no decode
    /// kernel, or when the scratch overflows `u64`.
    pub const fn of(attn: AttnPrefillBf16) -> Option<Self> {
        let Some(split) = attn_decode_split_at(attn.head_dim.get()) else {
            return None;
        };
        match attn_decode_split_scratch_bytes(
            split.splits.get(),
            attn.kv_heads.get(),
            attn.head_dim.get(),
        ) {
            Some((ml_offset, bytes)) => Some(Self {
                attn,
                split,
                scratch: AttnDecodeScratch { ml_offset, bytes },
            }),
            None => None,
        }
    }
}

/// [`attn_decode_split`] as the stored value.
const fn attn_decode_split_at(head_dim: u32) -> Option<AttnDecodeSplit> {
    let mut i = 0;
    while i < ATTN_DECODE_SPLITS.len() {
        let (hd, splits, min_tiles) = ATTN_DECODE_SPLITS[i];
        if hd == head_dim {
            return Some(AttnDecodeSplit {
                splits: Splits(splits),
                min_tiles: MinTiles(min_tiles),
            });
        }
        i += 1;
    }
    None
}

/// The split decode's f32 partials at `kv_heads` × `head_dim` ([`attn_decode_split`]'s splits):
/// the rows `[splits, kv_heads, 16, head_dim]`, then each row's max and sum `[splits, kv_heads, 2,
/// 16]`, in this order in the tape's split scratch. `(rows bytes, all bytes)`; `None` at a head dim
/// with no decode kernel, or on overflow.
pub const fn attn_decode_scratch_bytes(kv_heads: u32, head_dim: u32) -> Option<(u64, u64)> {
    let Some((splits, _)) = attn_decode_split(head_dim) else {
        return None;
    };
    attn_decode_split_scratch_bytes(splits, kv_heads, head_dim)
}

/// [`attn_decode_scratch_bytes`] at `splits`: what a decode step at its stored split
/// ([`AttnDecodeSplit`]) writes. `None` on overflow.
pub const fn attn_decode_split_scratch_bytes(
    splits: u32,
    kv_heads: u32,
    head_dim: u32,
) -> Option<(u64, u64)> {
    let heads = splits as u64 * kv_heads as u64 * ATTN_DECODE_BF16_MAX_GROUP as u64;
    let Some(rows) = heads.checked_mul(head_dim as u64 * 4) else {
        return None;
    };
    let Some(all) = rows.checked_add(heads * 2 * 4) else {
        return None;
    };
    Some((rows, all))
}

/// K's partial rope, then the paged-cache write of the rotated K and the unrotated V.
///
/// Launch 1 is [`RopeBf16`]'s kernel, unchanged: `k [rows, cols]` rotated by `table` into
/// `k_out`, `cols = kv_heads · head_dim`. Launch 2 writes, for each token row `t` whose
/// `slot_mapping[t]` is not `u32::MAX`, row `t` of `k_out` and of `v` into layer `layer`'s e4m3 K
/// and V cache at that slot, each element `e4m3_satfinite` at scale 1.0
/// (`kernels/kv_append_fp8.cu`); its `head_dim` is
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
    pub layer: LayerId,
    pub rows: Rows,
    pub cols: Cols,
    pub head_dim: HeadDim,
    pub rot_dim: RotDim,
    /// As [`RopeBf16::pairing`].
    #[serde(serialize_with = "serialize_rope_pairing")]
    pub pairing: RopePairing,
    /// The query rope the step also computes (FP8 S8 W4c, the one-row bucket's `Sibling` fold):
    /// `q.out = rope(q.x)` over `[rows, q.cols]` at this step's table, head dim, rotated width and
    /// pairing. `None`: the query rope is its own [`CudaStep::Rope`].
    pub q: Option<RopeQ>,
    /// The per-head norms the step applies before the rope (FP8 S8 W4c, Gemma's `NormedRope`
    /// fold): `q.x`'s and `k`'s heads by their RMSNorm (`q_gain`, `k_gain`), `v`'s by the unit
    /// norm, each at `eps`; `k` and `v` (and `q.x`) are then the raw projections. `None` on every
    /// other step.
    pub norms: Option<HeadNorms>,
}

/// A query rope folded into its layer's [`RopeAppendBf16`] (FP8 S8 W4c).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RopeQ {
    pub x: ArenaOffset,
    pub out: ArenaOffset,
    pub cols: Cols,
}

/// The per-head norms a [`RopeAppendBf16`] applies (FP8 S8 W4c, Gemma's): q's and k's RMSNorm
/// gains (`[head_dim]` bf16 weights, the stored-gain convention, offset 0) and V's unit norm, all
/// at `eps`. The normed V is appended to the cache only: the decode attention reads the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct HeadNorms {
    pub q_gain: WeightIx,
    pub k_gain: WeightIx,
    pub eps: RmsNormEps,
}

/// Causal GQA attention of `q [rows, q_heads · head_dim]` over layer `layer`'s paged K/V cache,
/// into `out` of the same shape, bf16 in and out, the softmax in f32; the cache is e4m3, widened
/// to bf16 on read (`kernels/attn_prefill_fp8kv.cu`).
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
    pub layer: LayerId,
    pub rows: Rows,
    pub q_heads: NumQHeads,
    pub kv_heads: NumKvHeads,
    pub head_dim: HeadDim,
    /// The softmax scale, which the kernel folds into `log2(e)`.
    pub scale: AttnScale,
    /// The keys a query row sees, its last `window` positions (a non-zero multiple of
    /// [`ATTN_WINDOW_MULTIPLE`]); `None` for full causal attention. The head-dim-256 kernels take
    /// it as a launch argument (0 for `None`); the head-dim-512 kernels are #885's 13b-3.
    pub window: Option<AttnWindow>,
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
    pub layer: LayerId,
    pub rows: Rows,
    pub num_k_heads: GdnHeads,
    pub num_v_heads: GdnHeads,
    pub head_k: GdnHeadDim,
    pub head_v: GdnHeadDim,
    pub conv_kernel: ConvKernel,
    /// The gated norm's `eps` (`rms_norm_eps`).
    pub eps: RmsNormEps,
    pub layout: GdnLayout,
}

/// A GDN step's layouts, written into its step when the step is emitted: the conv's channels
/// ([`gdn_conv_dim`]), the scratch's regions at its rows (the tape's, [`gdn_scratch`]) and its
/// bundle's ([`gdn_bundle`]). The player places its launches' operands by them as stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GdnLayout {
    pub conv_dim: ConvDim,
    pub scratch: GdnScratch,
    pub bundle: GdnBundle,
}

impl GatedDeltaNetBf16 {
    /// The query scale, `head_k^-½`.
    pub const fn scale(&self) -> f32 {
        gdn_scale(self.head_k.get())
    }

    /// The value width, `num_v_heads · head_v`, or `None` on overflow.
    pub const fn value_dim(&self) -> Option<u32> {
        self.num_v_heads.get().checked_mul(self.head_v.get())
    }
}

/// Which kernel a GEMM-shaped MoE step runs, chosen at expansion by the GEMM's shape rule: the
/// tiled GEMM for a bucket of rows, the GEMV for one row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MatmulKernel {
    Tiled,
    /// The GEMV, at its K split.
    Gemv(GemvSplit),
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
    pub m: Rows,
    pub experts: NumExperts,
    pub hidden: HiddenSize,
    /// The bundle's expert width, so every step naming the bundle states its geometry.
    pub inter: IntermediateSize,
    pub kernel: MatmulKernel,
}

/// The routing of `m` tokens over `experts` (one warp per token row) by `program`: its `top_k`
/// picks (equal scores by the higher expert index; a NaN lowest), stored ascending (slot
/// `top_k − 1` the best) as i32 into `topk_idx`, and their weights as f32 into `scores`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MoeRouteBf16 {
    pub logits: ArenaOffset,
    pub topk_idx: ArenaOffset,
    pub scores: ArenaOffset,
    pub m: Rows,
    pub experts: NumExperts,
    pub top_k: TopK,
    pub program: RouteProgram,
}

/// Which routing a [`MoeRouteBf16`] runs, each with only its own facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RouteProgram {
    /// `p = softmax(logits)` over every expert in f32 (at [`MOE_ROUTE_EXPERTS`]), its `top_k`
    /// largest picked, their probabilities divided by their sum when `renorm` (qwen3-5-moe).
    SoftmaxTopK { renorm: bool },
    /// The `top_k` largest raw logits picked, each times `scale` (`hidden^-0.5`), a softmax over
    /// the picks, then each times its expert's `expert_scale` entry (Gemma 4;
    /// `kernels/moe_route_bf16.cu`'s scaled program).
    TopKScaledSoftmax {
        scale: Scale,
        /// `router.per_expert_scale`, `[experts]` bf16.
        expert_scale: WeightIx,
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
    pub m: Rows,
    pub experts: NumExperts,
    pub top_k: TopK,
    pub hidden: HiddenSize,
    pub max_tiles: MaxTiles,
}

/// The act of an fp8 bank's gate and up over its live rows (`offsets[experts]`, the sort's) and the
/// quantization of the result into the quant scratch at `q_act` (FP8 S8 P1): the bits of `act`'s
/// [`GatedOp`] step then a [`QuantRowsFp8`] of its `out`, on every live row; `act.out` itself is
/// not written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertActQuantFp8 {
    pub act: SiluMulBf16,
    pub op: GatedOp,
    pub offsets: ArenaOffset,
    pub experts: NumExperts,
    pub q_act: ArenaOffset,
}

impl ExpertActQuantFp8 {
    /// The quant write it equals: `act.out`'s rows.
    pub const fn quant_write(&self) -> QuantRowsFp8 {
        QuantRowsFp8 {
            x: self.act.out,
            q: self.q_act,
            rows: self.act.rows,
            cols: self.act.cols,
        }
    }
}

/// [`CudaStep::ExpertGateUpGemmFp8`]: the chain it folds, each step as lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGateUpGemmFp8 {
    pub gate: ExpertGemmBf16,
    pub up: ExpertGemmBf16,
    pub act: ExpertActQuantFp8,
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
    pub max_tiles: MaxTiles,
    pub experts: NumExperts,
    pub hidden: HiddenSize,
    pub inter: IntermediateSize,
    /// The column tile, one of [`EXPERT_GEMM_N_TILES`]: 256 for every Qwen tape.
    pub n_tile: NTile,
}

impl ExpertGemmBf16 {
    /// `(n, k)` of the projection.
    pub const fn n_k(&self) -> (u32, u32) {
        self.proj.n_k(self.hidden.get(), self.inter.get())
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
    pub m: Rows,
    pub top_k: TopK,
    pub experts: NumExperts,
    pub hidden: HiddenSize,
    pub inter: IntermediateSize,
}

/// The routed experts' decode GEMV ([`CudaStep::ExpertGemv`]): the projection, at its K split over
/// `m · top_k` pairs, chosen when the step is emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGemvSplitBf16 {
    pub gemv: ExpertGemvBf16,
    pub split: GemvSplit,
}

impl ExpertGemvSplitBf16 {
    /// `gemv` at its shape's split.
    pub const fn of(gemv: ExpertGemvBf16) -> Self {
        Self {
            gemv,
            split: Self::split_of(&gemv),
        }
    }

    /// The split of `g`'s shape at its `m · top_k` pairs ([`GemvSplit::of`]).
    pub const fn split_of(g: &ExpertGemvBf16) -> GemvSplit {
        let (n, k) = g.n_k();
        GemvSplit::of(n, k, g.m.get().saturating_mul(g.top_k.get()))
    }
}

impl ExpertGemvBf16 {
    /// `(n, k)` of the projection.
    pub const fn n_k(&self) -> (u32, u32) {
        self.proj.n_k(self.hidden.get(), self.inter.get())
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
    pub m: Rows,
    pub top_k: TopK,
    pub hidden: HiddenSize,
    /// The down region's rows: `R` sorted, `P` gathered.
    pub down_rows: Rows,
}

/// `out = routed + shared · (1 / (1 + e^(−g)))` over `[rows, cols]`, `g` `[rows, 1]` broadcast
/// across the row, computed in f32 and rounded once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct GateScaleBf16 {
    pub routed: ArenaOffset,
    pub shared: ArenaOffset,
    pub g: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: Rows,
    pub cols: Cols,
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
    /// A one-weight GEMM step's matmul (a GEMV's at its one row) and the residual it adds (a
    /// one-row GEMV's, FP8 S8 W4a); `None` for every other step.
    pub const fn gemm_view(&self) -> Option<(Matmul, Option<ArenaOffset>)> {
        match self {
            Self::Gemm(g) => Some((g.matmul(), None)),
            Self::GemmCol64(g) => Some((g.matmul(), None)),
            Self::GemmNarrow(g) => Some((g.matmul(), None)),
            Self::GemmFp8(g) => Some((g.matmul(), None)),
            Self::GemmCol64Fp8(g) => Some((g.matmul(), None)),
            Self::GemmNarrowFp8(g) => Some((g.matmul(), None)),
            Self::Gemv(g) => Some((g.matmul(), g.residual)),
            Self::GemvFp8(g) => Some((g.matmul(), g.residual)),
            Self::GemvTcFp8(g) => Some((g.matmul(), g.residual)),
            _ => None,
        }
    }

    /// A decode GEMV's K split and its pairs (`gemv_splits`): the dense GEMV and a GEMV router at
    /// one pair, the expert GEMV at `m · top_k`; `None` for every other step. Every variant is
    /// named, so a new step states whether it is one.
    pub const fn gemv_split(&self) -> Option<(u32, u32, u32)> {
        match *self {
            Self::Gemv(GemvBf16 { n, split, .. }) | Self::GemvFp8(GemvFp8 { n, split, .. }) => {
                Some((n.get(), 1, split.splits.get()))
            }
            Self::RouterLogits(RouterLogitsBf16 {
                experts,
                kernel: MatmulKernel::Gemv(split),
                ..
            }) => Some((experts.get(), 1, split.splits.get())),
            Self::ExpertGemv(ExpertGemvSplitBf16 { gemv: g, split }) => {
                let (n, _) = g.n_k();
                let Some(pairs) = g.m.get().checked_mul(g.top_k.get()) else {
                    return None;
                };
                Some((n, pairs, split.splits.get()))
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
            | Self::ExpertSortFp8(_)
            | Self::ExpertActQuant(_)
            | Self::ExpertGateUpGemmFp8(_)
            | Self::ExpertGemm(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::LayerScale(_)
            | Self::SoftCap(_)
            | Self::GatherLastRow(_)
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
            | Self::GemvGroupFp8(_)
            // B8 takes no K split; a gate/up's f32 rows are its own scratch (`gemv_scratch`).
            | Self::GemvTcFp8(_)
            | Self::GemvGroupTcFp8(_)
            | Self::GateUpTcFp8(_)
            | Self::ExpertGemvTcFp8(_)
            | Self::ExpertGateUpTcFp8(_)
            | Self::RmsNormQuantFp8(_)
            | Self::AddRmsNormQuantFp8(_) => None,
        }
    }

    /// The grouped kernel a GEMM step's kernel has (#1049), and the GEMM: `Some` for the tiled
    /// GEMM, the narrow GEMM and the GEMV, the three a `SiblingMatmuls` fold groups; `None` for
    /// every other step. Every variant is named, so a new step states whether it groups.
    pub const fn group_kind(&self) -> Option<(GroupKind, Matmul)> {
        match *self {
            Self::Gemm(g) => Some((GroupKind::Gemm, g.matmul())),
            Self::GemmNarrow(g) => Some((GroupKind::GemmNarrow, g.matmul())),
            Self::Gemv(g) => Some((GroupKind::Gemv, g.matmul())),
            Self::GemmFp8(g) => Some((GroupKind::GemmFp8, g.matmul())),
            Self::GemmNarrowFp8(g) => Some((GroupKind::GemmNarrowFp8, g.matmul())),
            Self::GemvFp8(g) => Some((GroupKind::GemvFp8, g.matmul())),
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
            | Self::ExpertSortFp8(_)
            | Self::ExpertActQuant(_)
            | Self::ExpertGateUpGemmFp8(_)
            | Self::ExpertGemm(_)
            | Self::ExpertGemv(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::LayerScale(_)
            | Self::SoftCap(_)
            | Self::GatherLastRow(_)
            | Self::RouterNorm(_)
            | Self::GemmCol64(_)
            | Self::GemmGroup(_)
            | Self::GemmNarrowGroup(_)
            | Self::GemvGroup(_)
            | Self::AddRmsNorm(_)
            | Self::NormAdd(_)
            | Self::GateUpAct(_)
            | Self::ExpertGateUpAct(_)
            | Self::GdnDecode(_)
            // The tensor-core kinds are picked after the fold groups (`gemv_tc_takes`).
            | Self::GemvTcFp8(_)
            | Self::GemvGroupTcFp8(_)
            | Self::GateUpTcFp8(_)
            | Self::ExpertGemvTcFp8(_)
            | Self::ExpertGateUpTcFp8(_)
            | Self::RmsNormQuantFp8(_)
            | Self::AddRmsNormQuantFp8(_) => None,
        }
    }

    /// A grouped GEMM step's kind and group; `None` for every other step.
    pub const fn group(&self) -> Option<(GroupKind, GemmGroupBf16)> {
        match *self {
            Self::GemmGroup(g) => Some((GroupKind::Gemm, g)),
            Self::GemmNarrowGroup(g) => Some((GroupKind::GemmNarrow, g)),
            Self::GemvGroup(GemvGroupBf16 { group: g, .. }) => Some((GroupKind::Gemv, g)),
            Self::GemmGroupFp8(g) => Some((GroupKind::GemmFp8, g)),
            Self::GemmNarrowGroupFp8(g) => Some((GroupKind::GemmNarrowFp8, g)),
            Self::GemvGroupFp8(GemvGroupBf16 { group: g, .. }) => Some((GroupKind::GemvFp8, g)),
            Self::GemvGroupTcFp8(g) => Some((GroupKind::GemvTcFp8, g)),
            Self::GemvTcFp8(_)
            | Self::GateUpTcFp8(_)
            | Self::ExpertGemvTcFp8(_)
            | Self::ExpertGateUpTcFp8(_)
            | Self::QuantRowsFp8(_)
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
            | Self::ExpertSortFp8(_)
            | Self::ExpertActQuant(_)
            | Self::ExpertGateUpGemmFp8(_)
            | Self::ExpertGemm(_)
            | Self::ExpertGemv(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::LayerScale(_)
            | Self::SoftCap(_)
            | Self::GatherLastRow(_)
            | Self::RouterNorm(_)
            | Self::GemmCol64(_)
            | Self::AddRmsNorm(_)
            | Self::NormAdd(_)
            | Self::GateUpAct(_)
            | Self::ExpertGateUpAct(_)
            | Self::GdnDecode(_)
            | Self::RmsNormQuantFp8(_)
            | Self::AddRmsNormQuantFp8(_) => None,
        }
    }

    /// The input an fp8 GEMM of more than one row (or a group, or a gate/up pair of them, or an
    /// expert GEMM over an fp8 bank: its sorted or activation rows) reads quantized, `(a, m, k)`:
    /// lowered, `a` is its bf16 input, which the lowering quantizes before it; on a tape, `a` is
    /// the quant scratch the [`Self::QuantRowsFp8`] before it wrote, which [`quant_chain`] checks.
    /// `None` for every other step, the one-row GEMV included.
    pub const fn quantized_input(&self) -> Option<(ArenaOffset, u32, u32)> {
        // An fp8 bank's down reads the quant scratch; its gate and up read the e4m3 sorted rows
        // (FP8 S8 P1, [`Self::ExpertSortFp8`]), which its sort reads the quant scratch for.
        if let Self::ExpertGemm(ExpertGemmBf16 {
            proj: ExpertProjection::Down,
            ..
        }) = *self
        {
            return self.fp8_expert_rows();
        }
        if let Self::ExpertSortFp8(s) = *self {
            return Some((s.x, s.m.get(), s.hidden.get()));
        }
        // The tensor-core decode GEMVs (FP8 S8 T2): the expert form's token row or pairs' rows.
        if let Self::ExpertGemvTcFp8(g)
        | Self::ExpertGateUpTcFp8(ExpertGateUpTcFp8 {
            fold: ExpertGateUpActBf16 { gate: g, .. },
            ..
        }) = *self
        {
            let rows = match g.rows {
                PairRows::Token => Some(g.m.get()),
                PairRows::Pair => g.m.get().checked_mul(g.top_k.get()),
            };
            return match rows {
                Some(rows) => Some((g.a, rows, g.n_k().1)),
                None => None,
            };
        }
        let g = match *self {
            Self::GemmFp8(g) => g.matmul(),
            Self::GemmCol64Fp8(g) => g.matmul(),
            Self::GemmNarrowFp8(g) => g.matmul(),
            Self::GemvTcFp8(g) => g.matmul(),
            Self::GemmGroupFp8(g) | Self::GemmNarrowGroupFp8(g) | Self::GemvGroupTcFp8(g) => {
                g.member(g.first)
            }
            Self::GateUpTcFp8(t) => t.fold.gate.matmul(),
            Self::GateUpAct(GateUpActBf16 {
                gate,
                kernel: GateUpKernel::GemmFp8 | GateUpKernel::GemmCol64Fp8,
                ..
            }) => gate.matmul(),
            _ => return None,
        };
        Some((g.a, g.m.get(), g.k.get()))
    }

    /// An fp8 bank's grouped expert GEMM's e4m3 rows `(a, rows, k)`: the sorted capacity rows (its
    /// gate's and up's, the sort's) or the activation rows (its down's, the quant scratch's), e4m3
    /// then their scales; `None` for every other step.
    pub const fn fp8_expert_rows(&self) -> Option<(ArenaOffset, u32, u32)> {
        if let Self::ExpertGemm(
            g @ ExpertGemmBf16 {
                weight: MoeWeight::Fp8Bundle | MoeWeight::Fp8Own,
                ..
            },
        ) = *self
        {
            // An fp8 bank's row tile (FP8 S8 P3(a)).
            let rows = g
                .max_tiles
                .get()
                .checked_mul(crate::op_abi::CUDA_FP8_EXPERT_TILE_ROWS);
            return match rows {
                Some(rows) => Some((g.a, rows, g.n_k().1)),
                None => None,
            };
        }
        None
    }

    /// The quantization the step writes into the quant scratch, as the [`QuantRowsFp8`] it is or
    /// equals: a quant write's own, a tensor-core gate/up's act + quantize of its activation (at
    /// the scratch's second slot); `None` for every other step.
    pub const fn quant_write(&self) -> Option<QuantRowsFp8> {
        match *self {
            Self::QuantRowsFp8(q) => Some(q),
            Self::ExpertActQuant(t) => Some(t.quant_write()),
            Self::ExpertGateUpGemmFp8(t) => Some(t.act.quant_write()),
            Self::GateUpTcFp8(t) => Some(t.quant_write()),
            Self::ExpertGateUpTcFp8(t) => t.quant_write(),
            Self::RmsNormQuantFp8(t) => Some(t.quant_write()),
            Self::AddRmsNormQuantFp8(t) => Some(t.quant_write()),
            _ => None,
        }
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
        // The prefill's fused expert gate + up + act (FP8 S8 P3(b)): the chain it folds.
        if let Self::ExpertGateUpGemmFp8(t) = self {
            return [
                Some(Self::ExpertGemm(t.gate)),
                Some(Self::ExpertGemm(t.up)),
                Some(Self::ExpertActQuant(t.act)),
            ];
        }
        if let Self::ExpertGateUpAct(ExpertGateUpGemvBf16 { fold: g, split }) = self {
            return [
                Some(Self::ExpertGemv(ExpertGemvSplitBf16 {
                    gemv: g.gate,
                    split: *split,
                })),
                Some(Self::ExpertGemv(ExpertGemvSplitBf16 {
                    gemv: g.up,
                    split: *split,
                })),
                Some(g.act.step(g.activation())),
            ];
        }
        // A tensor-core gate/up: itself (the quantized row, the gate's weight, the activation and
        // its quantization), then its up as the GEMV it is on its own (its weight).
        if let Self::GateUpTcFp8(t) = self {
            return [
                Some(*self),
                Some(Self::GemvTcFp8(GemvTcFp8::of(GemvFp8::of(
                    t.fold.up.matmul(),
                )))),
                None,
            ];
        }
        if let Self::ExpertGateUpTcFp8(t) = self {
            return [Some(*self), Some(Self::ExpertGemvTcFp8(t.fold.up)), None];
        }
        // A norm with its quantization (FP8 S8 N3): itself (the norm's operands), then the
        // quantization as the quant write it equals.
        if let Self::RmsNormQuantFp8(t) = self {
            return [Some(*self), Some(Self::QuantRowsFp8(t.quant_write())), None];
        }
        if let Self::AddRmsNormQuantFp8(t) = self {
            return [Some(*self), Some(Self::QuantRowsFp8(t.quant_write())), None];
        }
        // A rope append with its query rope folded in (FP8 S8 W4c): the append, then the query rope
        // as the step it is on its own, so `validate` checks its spans too.
        if let Self::RopeAppend(r) = self
            && let Some(q) = r.q
        {
            return [
                Some(*self),
                Some(Self::Rope(RopeBf16 {
                    x: q.x,
                    table: r.table,
                    out: q.out,
                    rows: r.rows,
                    cols: q.cols,
                    head_dim: r.head_dim,
                    rot_dim: r.rot_dim,
                    pairing: r.pairing,
                })),
                None,
            ];
        }
        [Some(*self), None, None]
    }

    /// The split scratch bytes the step writes: [`gemv_scratch_bytes`] of [`Self::gemv_split`], 0
    /// for a step that is no decode GEMV or runs at one split; a grouped GEMV's, the sum of its
    /// members' (each holds its partials at once); a decode attention's, its split partials
    /// ([`AttnDecodeScratch`], written by its split launch and read by its merge, the
    /// same step's second launch); `None` on overflow.
    pub const fn gemv_scratch(&self) -> Option<u64> {
        if let Self::AttnDecode(AttnDecodeBf16 { scratch, .. }) = self {
            return Some(scratch.bytes);
        }
        if let Self::GemvGroup(g) | Self::GemvGroupFp8(g) = self {
            return g.gemv_scratch_before(3);
        }
        if let Self::GateUpAct(g) = self {
            return g.gemv_scratch();
        }
        if let Self::ExpertGateUpAct(g) = self {
            return g.gemv_scratch();
        }
        if let Self::GateUpTcFp8(t) = self {
            return t.gemv_scratch();
        }
        if let Self::ExpertGateUpTcFp8(t) = self {
            return t.gemv_scratch();
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
            Self::GemmGroup(g)
            | Self::GemmNarrowGroup(g)
            | Self::GemvGroup(GemvGroupBf16 { group: g, .. }) => {
                Self::Gemm(GemmBf16::of(g.member(g.first))).extents()
            }
            // Its parts' ([`Self::parts`]), each checked as the step it is; here the gate's.
            Self::ExpertGateUpGemmFp8(t) => Self::ExpertGemm(t.gate).extents(),
            Self::GemmGroupFp8(g)
            | Self::GemmNarrowGroupFp8(g)
            | Self::GemvGroupFp8(GemvGroupBf16 { group: g, .. })
            | Self::GemvGroupTcFp8(g) => match self.group() {
                Some((kind, _)) => kind.step(g.member(g.first)).extents(),
                None => None,
            },
            // The bf16 rows; the e4m3 rows and their scales, the quant scratch.
            Self::QuantRowsFp8(q) => {
                let (Some(xb), Some(qb)) = (bf16_bytes(q.rows.get(), q.cols.get()), q.bytes())
                else {
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
            Self::GemmFp8(_)
            | Self::GemmCol64Fp8(_)
            | Self::GemmNarrowFp8(_)
            | Self::GemvFp8(_)
            | Self::GemvTcFp8(_) => {
                let Some((Matmul { a, b, c, m, n, k }, residual)) = self.gemm_view() else {
                    return None;
                };
                let ab = match self {
                    Self::GemvFp8(_) => bf16_bytes(m.get(), k.get()),
                    _ => QuantRowsFp8 {
                        x: a,
                        q: a,
                        rows: m,
                        cols: Cols(k.get()),
                    }
                    .bytes(),
                };
                let (Some(ab), Some(cb), Some(w)) = (
                    ab,
                    bf16_bytes(m.get(), n.get()),
                    fp8_weight(n.get(), k.get()),
                ) else {
                    return None;
                };
                // FP8 S8 W4a: the residual a one-row GEMV adds, `[1, n]` as its output.
                let r = match residual {
                    Some(r) => Some(span(r, cb)),
                    None => None,
                };
                Some(StepExtents {
                    reads: [Some(span(a, ab)), r, None, None],
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
                let (Some(ob), Some(tb)) = (
                    bf16_bytes(rows.get(), hidden.get()),
                    bf16_bytes(vocab.get(), hidden.get()),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [Some(span(tokens, u32_bytes(rows.get()))), None, None, None],
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
            Self::Gemm(_) | Self::GemmCol64(_) | Self::GemmNarrow(_) | Self::Gemv(_) => {
                let Some((Matmul { a, b, c, m, n, k }, residual)) = self.gemm_view() else {
                    return None;
                };
                let (Some(ab), Some(bb), Some(cb)) = (
                    bf16_bytes(m.get(), k.get()),
                    bf16_bytes(n.get(), k.get()),
                    bf16_bytes(m.get(), n.get()),
                ) else {
                    return None;
                };
                // FP8 S8 W4a: the residual a one-row GEMV adds, `[1, n]` as its output.
                let r = match residual {
                    Some(r) => Some(span(r, cb)),
                    None => None,
                };
                Some(StepExtents {
                    reads: [Some(span(a, ab)), r, None, None],
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
                let (Some(xb), Some(gb)) = (
                    bf16_bytes(rows.get(), cols.get()),
                    bf16_bytes(1, cols.get()),
                ) else {
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
                        bf16_bytes(g.gate.m.get(), g.gate.k.get()),
                        bf16_bytes(g.gate.n.get(), g.gate.k.get()),
                    ),
                };
                let (Some(ab), Some(cb), Some(bb)) =
                    (ab, bf16_bytes(g.gate.m.get(), g.gate.n.get()), bb)
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
            // The quantized row and the gate's weight; the activation, then its quantization in
            // the quant scratch's second slot (the up's weight is its part's: `parts`).
            Self::GateUpTcFp8(t) => {
                let (Some(e), Some(qb)) = (
                    Self::GemvTcFp8(GemvTcFp8::of(GemvFp8::of(t.fold.gate.matmul()))).extents(),
                    t.quant_write().bytes(),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    writes: span(t.fold.out, e.writes.bytes),
                    second_write: Some(span(t.q_act, qb)),
                    ..e
                })
            }
            // As `GateUpTcFp8`, over the experts' token row and picks.
            Self::ExpertGateUpTcFp8(t) => {
                let (Some(e), Some(q)) = (
                    Self::ExpertGemvTcFp8(t.fold.gate).extents(),
                    t.quant_write(),
                ) else {
                    return None;
                };
                let Some(qb) = q.bytes() else {
                    return None;
                };
                Some(StepExtents {
                    writes: span(t.fold.out, e.writes.bytes),
                    second_write: Some(span(t.q_act, qb)),
                    ..e
                })
            }
            // The quantized rows (the token row, or the pairs' own), as `ExpertGemv`'s otherwise.
            Self::ExpertGemvTcFp8(g) => {
                let (Some(e), Some((_, rows, k))) = (
                    Self::ExpertGemv(ExpertGemvSplitBf16::of(g)).extents(),
                    self.quantized_input(),
                ) else {
                    return None;
                };
                let Some(ab) = (QuantRowsFp8 {
                    x: g.a,
                    q: g.a,
                    rows: Rows(rows),
                    cols: Cols(k),
                })
                .bytes() else {
                    return None;
                };
                let [_, picks, ..] = e.reads;
                Some(StepExtents {
                    reads: [Some(span(g.a, ab)), picks, None, None],
                    ..e
                })
            }
            // The token rows and the picks; the activation rows.
            Self::ExpertGateUpAct(ExpertGateUpGemvBf16 { fold: g, split }) => {
                let Some(e) = Self::ExpertGemv(ExpertGemvSplitBf16 {
                    gemv: g.gate,
                    split,
                })
                .extents() else {
                    return None;
                };
                Some(StepExtents {
                    writes: span(g.out, e.writes.bytes),
                    ..e
                })
            }
            // The norm's; its quantization is its second part's ([`Self::parts`]).
            Self::RmsNormQuantFp8(t) => Self::RmsNorm(t.norm).extents(),
            Self::AddRmsNormQuantFp8(t) => Self::AddRmsNorm(t.fold).extents(),
            // The norm's gain and both inputs; the sum, then the norm.
            Self::AddRmsNorm(f) => {
                let RmsNormBf16 {
                    gain,
                    y,
                    rows,
                    cols,
                    ..
                } = f.norm;
                let (Some(xb), Some(gb)) = (
                    bf16_bytes(rows.get(), cols.get()),
                    bf16_bytes(1, cols.get()),
                ) else {
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
                let (Some(xb), Some(gb)) = (
                    bf16_bytes(rows.get(), cols.get()),
                    bf16_bytes(1, cols.get()),
                ) else {
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
                let Some(bytes) = bf16_bytes(rows.get(), cols.get()) else {
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
                let (Some(xb), Some(tb)) = (
                    bf16_bytes(rows.get(), cols.get()),
                    bf16_bytes(rows.get(), rot_dim.get()),
                ) else {
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
                let Some(hb) = bf16_bytes(rows.get(), half_cols.get()) else {
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
                let (Some(kb), Some(tb)) = (
                    bf16_bytes(rows.get(), cols.get()),
                    bf16_bytes(rows.get(), rot_dim.get()),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(k, kb)),
                        Some(span(v, kb)),
                        Some(span(table, tb)),
                        Some(span(slot_mapping, u32_bytes(rows.get()))),
                    ],
                    writes: span(k_out, kb),
                    second_write: None,
                    weight: None,
                    kv: Some((layer.get(), KvUse::Write)),
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
            | Self::AttnDecode(AttnDecodeBf16 {
                attn:
                    AttnPrefillBf16 {
                        q,
                        block_table,
                        sequence,
                        out,
                        layer,
                        rows,
                        q_heads,
                        head_dim,
                        ..
                    },
                ..
            }) => {
                let Some(cols) = q_heads.get().checked_mul(head_dim.get()) else {
                    return None;
                };
                let Some(qb) = bf16_bytes(rows.get(), cols) else {
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
                    kv: Some((layer.get(), KvUse::Read)),
                    gdn: None,
                    in_place: false,
                })
            }
            // The GDN step's reads and its output; no scratch (`kernels/gdn_decode.cu`).
            Self::GdnDecode(g) => {
                let Some(e) = Self::GatedDeltaNet(GdnPrefillBf16::of(g)).extents() else {
                    return None;
                };
                Some(StepExtents {
                    second_write: None,
                    ..e
                })
            }
            Self::GatedDeltaNet(GdnPrefillBf16 {
                gdn:
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
                ..
            }) => {
                let Some(value_dim) = g.value_dim() else {
                    return None;
                };
                let conv_dim = g.layout.conv_dim.get();
                let (Some(qb), Some(vb), Some(hb)) = (
                    bf16_bytes(rows.get(), conv_dim),
                    bf16_bytes(rows.get(), value_dim),
                    bf16_bytes(rows.get(), num_v_heads.get()),
                ) else {
                    return None;
                };
                let (sc, bu) = (g.layout.scratch, g.layout.bundle);
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
                    gdn: Some(layer.get()),
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
                    bf16_bytes(m.get(), hidden.get()),
                    bf16_bytes(m.get(), experts.get()),
                    weight.router_bytes(experts.get(), hidden.get(), inter.get()),
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
                program,
            }) => {
                let (Some(lb), Some(pairs)) = (
                    bf16_bytes(m.get(), experts.get()),
                    m.get().checked_mul(top_k.get()),
                ) else {
                    return None;
                };
                let weight = match program {
                    RouteProgram::TopKScaledSoftmax { expert_scale, .. } => Some(WeightRead {
                        ix: expert_scale,
                        bytes: experts.get() as u64 * CUDA_ACT_BYTES,
                    }),
                    RouteProgram::SoftmaxTopK { .. } => None,
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
                let Some(xb) = bf16_bytes(s.m.get(), s.hidden.get()) else {
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
            // The e4m3 sort: the quantized token rows and the picks; the sorted e4m3 rows and
            // their scales, then the index regions.
            Self::ExpertSortFp8(s) => {
                let Some(l) = s.layout() else {
                    return None;
                };
                let Some(capacity) = s
                    .max_tiles
                    .get()
                    .checked_mul(crate::op_abi::CUDA_FP8_EXPERT_TILE_ROWS)
                else {
                    return None;
                };
                let xq = QuantRowsFp8 {
                    x: s.x,
                    q: s.x,
                    rows: s.m,
                    cols: Cols(s.hidden.get()),
                };
                let rq = QuantRowsFp8 {
                    x: s.rows,
                    q: s.rows,
                    rows: Rows(capacity),
                    cols: Cols(s.hidden.get()),
                };
                let (Some(xb), Some(rb)) = (xq.bytes(), rq.bytes()) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(s.x, xb)),
                        Some(span(s.topk_idx, u32_bytes(l.pairs))),
                        None,
                        None,
                    ],
                    writes: span(s.rows, rb),
                    second_write: Some(span(s.chunk_hist, l.index_bytes)),
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            // The gate and up rows and the sort's offsets; the quant write's span.
            Self::ExpertActQuant(t) => {
                let (Some(ab), Some(qb)) = (
                    bf16_bytes(t.act.rows.get(), t.act.cols.get()),
                    t.quant_write().bytes(),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(t.act.gate, ab)),
                        Some(span(t.act.up, ab)),
                        Some(span(t.offsets, u32_bytes(t.experts.get() + 1))),
                        None,
                    ],
                    writes: span(t.q_act, qb),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: false,
                })
            }
            Self::ExpertGemm(g) => {
                let fp8 = matches!(g.weight, MoeWeight::Fp8Bundle | MoeWeight::Fp8Own);
                let rows = g.max_tiles.get() as u64 * crate::op_abi::expert_tile_rows(fp8) as u64;
                let (n, k) = g.n_k();
                // An fp8 bank's rows are the quantized ones, e4m3 then their scales.
                let ab = match self.fp8_expert_rows() {
                    Some((a, rows, cols)) => QuantRowsFp8 {
                        x: a,
                        q: a,
                        rows: Rows(rows),
                        cols: Cols(cols),
                    }
                    .bytes(),
                    None => (rows * k as u64).checked_mul(CUDA_ACT_BYTES),
                };
                let (Some(ab), Some(cb), Some(bu)) = (
                    ab,
                    (rows * n as u64).checked_mul(CUDA_ACT_BYTES),
                    g.weight
                        .experts(g.experts.get(), g.hidden.get(), g.inter.get()),
                ) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(g.a, ab)),
                        Some(span(g.offsets, u32_bytes(g.experts.get() + 1))),
                        Some(span(g.tile_expert, u32_bytes(g.max_tiles.get()))),
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
            Self::ExpertGemv(ExpertGemvSplitBf16 { gemv: g, .. }) => {
                let Some(pairs) = g.m.get().checked_mul(g.top_k.get()) else {
                    return None;
                };
                let a_rows = match g.rows {
                    PairRows::Token => g.m,
                    PairRows::Pair => Rows(pairs),
                };
                let (n, k) = g.n_k();
                let (Some(ab), Some(cb), Some(bu)) = (
                    bf16_bytes(a_rows.get(), k),
                    bf16_bytes(pairs, n),
                    g.weight
                        .experts(g.experts.get(), g.hidden.get(), g.inter.get()),
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
                    m.get().checked_mul(top_k.get()),
                    bf16_bytes(down_rows.get(), hidden.get()),
                    bf16_bytes(m.get(), hidden.get()),
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
                let (Some(bytes), Some(gb)) = (
                    bf16_bytes(rows.get(), cols.get()),
                    bf16_bytes(rows.get(), 1),
                ) else {
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
                let Some(bytes) = bf16_bytes(rows.get(), cols.get()) else {
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
            // The rows read (row `q_len − 1`, any of them) and the sequence region's two entries;
            // the write is row 0 of the same rows, declared as the whole tensor it is in place over.
            Self::GatherLastRow(GatherLastRowBf16 {
                x,
                sequence,
                rows,
                cols,
            }) => {
                let Some(bytes) = bf16_bytes(rows.get(), cols.get()) else {
                    return None;
                };
                Some(StepExtents {
                    reads: [
                        Some(span(x, bytes)),
                        Some(span(sequence, u32_bytes(2))),
                        None,
                        None,
                    ],
                    writes: span(x, bytes),
                    second_write: None,
                    weight: None,
                    kv: None,
                    gdn: None,
                    in_place: true,
                })
            }
            // `pointwise_bf16.cu` reads each element before the same thread writes it, so the
            // soft-cap may write over its input.
            Self::SoftCap(SoftCapBf16 {
                x, out, rows, cols, ..
            }) => {
                let Some(bytes) = bf16_bytes(rows.get(), cols.get()) else {
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
                let Some(bytes) = bf16_bytes(rows.get(), cols.get()) else {
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
        let Some(pairs) = self.m.get().checked_mul(self.top_k.get()) else {
            return None;
        };
        let Some([chunk_hist, counts, offsets, tile_expert, pos, index_bytes]) =
            sort_index_regions(pairs, self.experts.get(), self.max_tiles.get())
        else {
            return None;
        };
        let rows = self.max_tiles.get() as u64 * crate::op_abi::CUDA_EXPERT_TILE_ROWS as u64;
        let Some(rows_bytes) = (rows * self.hidden.get() as u64).checked_mul(CUDA_ACT_BYTES) else {
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
            | Self::GemmCol64(GemmCol64Bf16 { b, .. })
            | Self::GemmNarrow(GemmNarrowBf16 { b, .. })
            | Self::Gemv(GemvBf16 { b, .. })
            | Self::GemmFp8(GemmFp8 { b, .. })
            | Self::GemmCol64Fp8(GemmCol64Fp8 { b, .. })
            | Self::GemmNarrowFp8(GemmNarrowFp8 { b, .. })
            | Self::GemvFp8(GemvFp8 { b, .. })
            | Self::GemvTcFp8(GemvTcFp8 { b, .. }) => Some(b),
            Self::RmsNorm(RmsNormBf16 { gain, .. })
            | Self::RouterNorm(RmsNormBf16 { gain, .. })
            | Self::AddRmsNorm(AddRmsNormBf16 {
                norm: RmsNormBf16 { gain, .. },
                ..
            })
            | Self::NormAdd(NormAddBf16 {
                norm: RmsNormBf16 { gain, .. },
                ..
            })
            | Self::RmsNormQuantFp8(RmsNormQuantFp8 {
                norm: RmsNormBf16 { gain, .. },
                ..
            })
            | Self::AddRmsNormQuantFp8(AddRmsNormQuantFp8 {
                fold:
                    AddRmsNormBf16 {
                        norm: RmsNormBf16 { gain, .. },
                        ..
                    },
                ..
            }) => Some(gain),
            Self::LayerScale(LayerScaleBf16 { scale, .. }) => Some(scale),
            Self::MoeRoute(MoeRouteBf16 { program, .. }) => match program {
                RouteProgram::TopKScaledSoftmax { expert_scale, .. } => Some(expert_scale),
                RouteProgram::SoftmaxTopK { .. } => None,
            },
            Self::GatedDeltaNet(GdnPrefillBf16 {
                gdn: GatedDeltaNetBf16 { bundle, .. },
                ..
            })
            | Self::GdnDecode(GatedDeltaNetBf16 { bundle, .. }) => Some(bundle),
            Self::RouterLogits(RouterLogitsBf16 { bundle, .. }) => Some(bundle),
            Self::ExpertGemm(ExpertGemmBf16 { bundle, .. })
            | Self::ExpertGateUpGemmFp8(ExpertGateUpGemmFp8 {
                gate: ExpertGemmBf16 { bundle, .. },
                ..
            })
            | Self::ExpertGemv(ExpertGemvSplitBf16 {
                gemv: ExpertGemvBf16 { bundle, .. },
                ..
            })
            | Self::ExpertGemvTcFp8(ExpertGemvBf16 { bundle, .. })
            | Self::ExpertGateUpAct(ExpertGateUpGemvBf16 {
                fold:
                    ExpertGateUpActBf16 {
                        gate: ExpertGemvBf16 { bundle, .. },
                        ..
                    },
                ..
            })
            | Self::ExpertGateUpTcFp8(ExpertGateUpTcFp8 {
                fold:
                    ExpertGateUpActBf16 {
                        gate: ExpertGemvBf16 { bundle, .. },
                        ..
                    },
                ..
            }) => Some(bundle),
            // The gate's; [`Self::weights_mut`] gives the up's too.
            Self::GateUpAct(GateUpActBf16 {
                gate: GemmBf16 { b, .. },
                ..
            })
            | Self::GateUpTcFp8(GateUpTcFp8 {
                fold:
                    GateUpActBf16 {
                        gate: GemmBf16 { b, .. },
                        ..
                    },
                ..
            }) => Some(b),
            // A grouped GEMM's first member's; [`Self::weights_mut`] gives every member's.
            Self::GemmGroup(GemmGroupBf16 { first, .. })
            | Self::GemmNarrowGroup(GemmGroupBf16 { first, .. })
            | Self::GemvGroup(GemvGroupBf16 {
                group: GemmGroupBf16 { first, .. },
                ..
            })
            | Self::GemmGroupFp8(GemmGroupBf16 { first, .. })
            | Self::GemmNarrowGroupFp8(GemmGroupBf16 { first, .. })
            | Self::GemvGroupFp8(GemvGroupBf16 {
                group: GemmGroupBf16 { first, .. },
                ..
            })
            | Self::GemvGroupTcFp8(GemmGroupBf16 { first, .. }) => Some(&mut first.b),
            Self::Add(_)
            | Self::SiluMul(_)
            | Self::Rope(_)
            | Self::GateSplit(_)
            | Self::GateApply(_)
            | Self::RopeAppend(_)
            | Self::AttnPrefill(_)
            | Self::AttnDecode(_)
            | Self::ExpertSort(_)
            | Self::ExpertSortFp8(_)
            | Self::ExpertActQuant(_)
            | Self::ExpertCombine(_)
            | Self::GateScale(_)
            | Self::Scale(_)
            | Self::RmsNormUnit(_)
            | Self::GeluMul(_)
            | Self::SoftCap(_)
            | Self::GatherLastRow(_)
            | Self::QuantRowsFp8(_) => None,
        }
    }

    /// Every weight index the step reads, for the compiler's rewrite: a grouped GEMM's members'
    /// in order, every other step's [`Self::weight_mut`].
    pub fn weights_mut(&mut self) -> Vec<&mut WeightIx> {
        match self {
            Self::GemmGroup(g)
            | Self::GemmNarrowGroup(g)
            | Self::GemvGroup(GemvGroupBf16 { group: g, .. })
            | Self::GemmGroupFp8(g)
            | Self::GemmNarrowGroupFp8(g)
            | Self::GemvGroupFp8(GemvGroupBf16 { group: g, .. })
            | Self::GemvGroupTcFp8(g) => {
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
            Self::GateUpAct(GateUpActBf16 { gate, up, .. })
            | Self::GateUpTcFp8(GateUpTcFp8 {
                fold: GateUpActBf16 { gate, up, .. },
                ..
            }) => vec![&mut gate.b, &mut up.b],
            Self::ExpertGateUpGemmFp8(ExpertGateUpGemmFp8 { gate, up, .. }) => {
                vec![&mut gate.bundle, &mut up.bundle]
            }
            Self::ExpertGateUpAct(ExpertGateUpGemvBf16 {
                fold: ExpertGateUpActBf16 { gate, up, .. },
                ..
            })
            | Self::ExpertGateUpTcFp8(ExpertGateUpTcFp8 {
                fold: ExpertGateUpActBf16 { gate, up, .. },
                ..
            }) => vec![&mut gate.bundle, &mut up.bundle],
            // The head norms' gains a rope append reads (FP8 S8 W4c).
            Self::RopeAppend(RopeAppendBf16 {
                norms: Some(HeadNorms { q_gain, k_gain, .. }),
                ..
            }) => vec![q_gain, k_gain],
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
