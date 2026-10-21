// SPDX-License-Identifier: Apache-2.0
//! CUDA's step: one or two kernel launches of a lowered tape, the counterpart of metal's
//! `MetalStep`.

use scratchy_subtile::subtile_ir::GainConvention;

use crate::tape::{ArenaOffset, ArenaSpan, CUDA_ACT_BYTES, WeightIx};

/// One launch-emitting step of a lowered tape, its operands already resolved: one kernel launch,
/// or two stream-ordered launches (a `RopeAppend` rotates K, then writes the cache).
///
/// It gains one variant per opcode CUDA lowers, each added by the issue that writes that opcode's
/// kernel, beside that kernel's registry row and its arm of [`CudaStep::extents`]. The tape
/// player matches it exhaustively, so a new variant does not compile until the player launches
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaStep {
    /// The bf16 GEMM, lowered from `SubOp::MatmulTile` (`kernels/gemm_bf16.cu`).
    Gemm(GemmBf16),
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmBf16 {
    pub a: ArenaOffset,
    pub b: WeightIx,
    pub c: ArenaOffset,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// The RMSNorm's `cols` is a non-zero multiple of this: each thread loads and stores 16-byte
/// vectors of 8 bf16 (`kernels/rms_norm_bf16.cu`). `rows` is non-zero and `rows · cols` at most
/// `i32::MAX`.
pub const RMS_NORM_BF16_COLS: u32 = 8;

/// `y[r, c] = x[r, c] / rms(x[r, :]) · (offset + gain[c])`, `rms(x) = sqrt(mean(x²) + eps)`,
/// bf16 in and out, computed in f32 and rounded once.
///
/// `x` and `y` are `[rows, cols]` arena slots; `gain` is a `[cols]` bf16 weight. `offset` is the
/// convention's ([`GainConvention::offset`]): 0 for a stored gain, 1 for a zero-centred one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormBf16 {
    pub x: ArenaOffset,
    pub gain: WeightIx,
    pub y: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    /// `eps` as its bits (`f32::to_bits`), so the step is `Eq`; [`Self::eps`] reads it back.
    pub eps_bits: u32,
    pub convention: GainConvention,
}

impl RmsNormBf16 {
    /// The norm's `eps`.
    pub const fn eps(&self) -> f32 {
        f32::from_bits(self.eps_bits)
    }
}

/// The add's `cols` is a non-zero multiple of this (`kernels/add_bf16.cu`: one 8-wide bf16
/// vector per thread), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const ADD_BF16_COLS: u32 = 8;

/// `out = a + b` over `[rows, cols]`, bf16 in and out: one f32 add, one rounding. No broadcast:
/// all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddBf16 {
    pub a: ArenaOffset,
    pub b: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
}

/// The SiluMul's `cols` is a non-zero multiple of this (`kernels/silu_mul_bf16.cu`: one 8-wide
/// bf16 vector per thread), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const SILU_MUL_BF16_COLS: u32 = 8;

/// `out = gate / (1 + e^(−gate)) · up` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Partial rotary embedding over `[rows, cols]`, `cols = heads · head_dim`, bf16 in and out.
///
/// For token row `r` and each head, pair `d < rot_dim/2` of lanes `(d, d + rot_dim/2)` (NeoX
/// pairing within the rotated width) becomes `(x0·c − x1·s, x1·c + x0·s)` in f32, rounded once,
/// with `c = table[r, d]` and `s = table[r, rot_dim/2 + d]`. Lanes `rot_dim..head_dim` are copied
/// bit for bit. `table` is the tape's rotary table ([`crate::tape::RotaryTable`]), `[rows,
/// rot_dim]` bf16; `out` overlaps neither input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopeBf16 {
    pub x: ArenaOffset,
    pub table: ArenaOffset,
    pub out: ArenaOffset,
    pub rows: u32,
    pub cols: u32,
    pub head_dim: u32,
    pub rot_dim: u32,
}

/// The gate split's `head_dim` is a non-zero multiple of this (`kernels/gate_split_bf16.cu`: one
/// 16-byte vector per thread, never straddling a head), and `half_cols` a non-zero multiple of
/// `head_dim`. `rows` is non-zero and `rows · 2·half_cols` at most `i32::MAX`.
pub const GATE_SPLIT_BF16_HEAD_MULTIPLE: u32 = 8;

/// Per head `h` and lane `d`: `q[r, h·hd + d] = qg[r, 2h·hd + d]` and `gate[r, h·hd + d] =
/// qg[r, 2h·hd + hd + d]`, bf16 copies. `qg` is `[rows, 2·half_cols]`, `q` and `gate` are `[rows,
/// half_cols]`, and neither output overlaps `qg` or the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateSplitBf16 {
    pub qg: ArenaOffset,
    pub q: ArenaOffset,
    pub gate: ArenaOffset,
    pub rows: u32,
    pub half_cols: u32,
    pub head_dim: u32,
}

/// The gate apply's `cols` is a non-zero multiple of this (`kernels/gate_apply_bf16.cu`: one
/// 8-wide bf16 vector per thread), `rows` non-zero, and `rows · cols` at most `i32::MAX`.
pub const GATE_APPLY_BF16_COLS: u32 = 8;

/// `out = attn / (1 + e^(−gate))` over `[rows, cols]`, bf16 in and out, computed in f32 and
/// rounded once. No broadcast: all three are arena slots of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// K's partial rope, then the paged-cache write of the rotated K and the unrotated V.
///
/// Launch 1 is [`RopeBf16`]'s kernel, unchanged: `k [rows, cols]` rotated by `table` into
/// `k_out`, `cols = kv_heads · head_dim`. Launch 2 copies, for each token row `t` whose
/// `slot_mapping[t]` is not `u32::MAX`, row `t` of `k_out` and of `v` into layer `layer`'s K and
/// V cache at that slot, bit for bit (`kernels/kv_append_bf16.cu`). `slot_mapping` is the tape's
/// region ([`crate::tape::KvTape::slot_mapping`], `rows` u32); `layer` is an index into
/// [`crate::tape::KvTape::layers`]. `k_out` overlaps no input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

/// Causal GQA attention of `q [rows, q_heads · head_dim]` over layer `layer`'s paged K/V cache,
/// into `out` of the same shape, bf16 in and out, the softmax in f32.
///
/// The keys are the sequence's `P + L` cached tokens, read through the tape's block table; query
/// row `i` attends to keys `0 ..= P + i` (`P`, `L` read from the tape's `sequence` region). Query
/// head `g` reads kv head `g / (q_heads / kv_heads)`. `block_table` and `sequence` are the tape's
/// regions ([`crate::tape::KvTape`]); `out` overlaps no input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

impl AttnPrefillBf16 {
    /// The softmax scale: `head_dim^-0.5` for qwen3.5.
    pub const fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
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
/// second span it writes (a two-output kernel's), the weight it reads, and the KV-cache layer it
/// writes or reads (an index into [`crate::tape::KvTape::layers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepExtents {
    pub reads: [Option<ArenaSpan>; 4],
    pub writes: ArenaSpan,
    pub second_write: Option<ArenaSpan>,
    pub weight: Option<WeightRead>,
    pub kv: Option<(u32, KvUse)>,
}

impl CudaStep {
    /// What the step's kernel reads and writes, in bytes; `None` when a byte count overflows
    /// `u64`. Each arm states its own kernel's operands, next to that kernel's registry row:
    /// `CudaTape::validate` checks a tape against it.
    pub const fn extents(&self) -> Option<StepExtents> {
        match *self {
            Self::Gemm(GemmBf16 { a, b, c, m, n, k }) => {
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
                })
            }
            Self::RmsNorm(RmsNormBf16 {
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
                })
            }
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
