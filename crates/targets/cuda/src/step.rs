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
    /// [`GEMM_NARROW_BF16_MAX_N`]: the operands of [`Self::Gemm`], another kernel (#885's 13b).
    GemmCol64(GemmBf16),
}

/// The 64-column GEMM's tile `(m, n, k)`: `gemm_bf16.cu`'s template at one 64-column B tile.
/// Every `m`, `n` and `k` it takes is a non-zero multiple of these; no 64-column box crosses an
/// array's column edge (a TMA box past it needs a plain tile, TK-1's R1). Gemma's 128, 704 and
/// 2112 are multiples of 64, not of 256.
pub const GEMM_COL64_TILE: (u32, u32, u32) = (128, 64, 64);

/// The expert GEMM's column tiles: 256 (`kernels/expert_gemm_bf16.cu`, Qwen's 512 and 2048) or 64
/// (Gemma's 704, #885's 13b), chosen at expansion by the projection's `n`.
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
/// The `rot_dim`s the rope kernel is instantiated for (`kernels/rope_bf16.cu`): 64, one plain
/// `st_bf<16, 64>` box, each half two 16-column base tiles (qwen3.5's partial rope); and 256, four
/// boxes, each half two whole boxes (Gemma 4's sliding layers, a full rope on a 256-lane head). Its
/// launcher refuses any other, so the kernel's rule is tighter than the lowering's
/// [`ROPE_BF16_ROT_MULTIPLE`]; Gemma 4's global layers' 128 is #885's 13b-3.
pub const ROPE_BF16_KERNEL_ROT_DIMS: &[u32] = &[64, 256];
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
/// The one head dim the decode attention kernel is instantiated for: its query, K and V tiles are
/// `st_bf<64, 256>`, the prefill's (`kernels/attn_decode_bf16.cu`). It takes one query row,
/// `q_heads` a non-zero multiple of `kv_heads` by at most [`ATTN_DECODE_BF16_MAX_GROUP`], and
/// `block_size` [`CUDA_KV_BLOCK_SIZE`].
pub const ATTN_DECODE_BF16_HEAD_DIM: u32 = 256;
/// The decode attention's largest GQA group `q_heads / kv_heads`: a kv head's query heads are
/// stacked as the rows of one query tile, and only the 16 rows of each consumer warpgroup's warp 0
/// are merged and stored (`kernels/attn_decode_bf16.cu`). The target models' groups are 2 to 12.
pub const ATTN_DECODE_BF16_MAX_GROUP: u32 = 16;
/// The one head dim the KV append is instantiated for (`kernels/kv_append_bf16.cu`): a head's row
/// is one `sv_bf<256>`, one 512-byte box of the cache's vector tensor map. The lowering already
/// refuses any other head dim for a `RopeAppend` (`CanonicalRefusal::KvHeadDim`).
pub const KV_APPEND_BF16_HEAD_DIM: u32 = 256;

/// K's partial rope, then the paged-cache write of the rotated K and the unrotated V.
///
/// Launch 1 is [`RopeBf16`]'s kernel, unchanged: `k [rows, cols]` rotated by `table` into
/// `k_out`, `cols = kv_heads · head_dim`. Launch 2 copies, for each token row `t` whose
/// `slot_mapping[t]` is not `u32::MAX`, row `t` of `k_out` and of `v` into layer `layer`'s K and
/// V cache at that slot, bit for bit (`kernels/kv_append_bf16.cu`); its `head_dim` is
/// [`KV_APPEND_BF16_HEAD_DIM`]. `slot_mapping` is the tape's
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
/// The GDN gating's `nv` is a multiple of this, at most twice it: one warp per row holds the row as
/// a naive `rv_fl<nv>`, instantiated for 16 and 32 (`kernels/gdn_gating.cu`). Checked by the
/// launcher only; qwen3.5's `nv` is 16 at the 0.8b and 32 at the 9b.
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
    /// The 64-column GEMM ([`GEMM_COL64_TILE`]): Gemma's 128-expert router at m ≥ 2. #885's 13b.
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
}

impl MoeWeight {
    /// The layout an expert step reads its slab from: the bundle's, or the bank's (its gate and
    /// up at byte 0). `None` on overflow.
    pub const fn experts(self, experts: u32, hidden: u32, inter: u32) -> Option<MoeBundle> {
        match self {
            Self::Bundle => moe_bundle(experts, hidden, inter),
            Self::Own => expert_bank(experts, hidden, inter),
        }
    }

    /// The bytes a router step reads its weight as: the whole bundle, or the projection.
    pub const fn router_bytes(self, experts: u32, hidden: u32, inter: u32) -> Option<u64> {
        match self {
            Self::Bundle => match moe_bundle(experts, hidden, inter) {
                Some(b) => Some(b.bytes),
                None => None,
            },
            Self::Own => bf16_bytes(experts, hidden),
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
    /// What the step's kernel reads and writes, in bytes; `None` when a byte count overflows
    /// `u64`. Each arm states its own kernel's operands, next to that kernel's registry row:
    /// `CudaTape::validate` checks a tape against it.
    pub const fn extents(&self) -> Option<StepExtents> {
        match *self {
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
                let (Some(ab), Some(cb), Some(bu)) = (
                    (rows * k as u64).checked_mul(CUDA_ACT_BYTES),
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
            | Self::Gemv(GemmBf16 { b, .. }) => Some(b),
            Self::RmsNorm(RmsNormBf16 { gain, .. })
            | Self::RouterNorm(RmsNormBf16 { gain, .. }) => Some(gain),
            Self::LayerScale(LayerScaleBf16 { scale, .. }) => Some(scale),
            Self::MoeRoute(MoeRouteBf16 { expert_scale, .. }) => expert_scale.as_mut(),
            Self::GatedDeltaNet(GatedDeltaNetBf16 { bundle, .. }) => Some(bundle),
            Self::RouterLogits(RouterLogitsBf16 { bundle, .. }) => Some(bundle),
            Self::ExpertGemm(ExpertGemmBf16 { bundle, .. })
            | Self::ExpertGemv(ExpertGemvBf16 { bundle, .. }) => Some(bundle),
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
            | Self::SoftCap(_) => None,
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
