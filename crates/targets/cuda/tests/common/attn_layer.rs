// SPDX-License-Identifier: Apache-2.0
//! One qwen3.5-0.8b full-attention layer: the canonical the attention-layer tape tests lower, and
//! its `static` tapes written out by hand from the issue's numbers, which `tape_layout` checks
//! against what the compiler lowers.
//!
//! The ops, in the order the macro emits them (`crates/compiler/macros/src/to_wavefront.rs`: the
//! DSL's `gate_split`, the q/k norms with the per-head Reshape pair shape inference puts around
//! each, `rope_append` as `RopeRotate(q)` then `RopeAppend(k, ·, ·, v, pk, pv)` at `:981-1071`, and
//! `attention` as `AttnDecode(qr, pk, pv, kr, v)` at `:1093-1240`):
//!
//! ```text
//!  0 n  = RmsNorm(x, g_in)              6 k  = n · Wk   [M, 512]          11 qr = RopeRotate(qf, cos_q, sin_q)
//!  1 qg = n · Wq   [M, 4096]            7 kh = Reshape(k) [2M, 256]       12 kr = RopeAppend(kf, cos_k, sin_k, v, pk, pv)
//!  2 q (+gate) = GateSplit(qg)          8 kn = RmsNorm(kh, k_norm)        13 attn = AttnDecode(qr, pk, pv, kr, v)
//!  3 qh = Reshape(q) [8M, 256]          9 kf = Reshape(kn) [M, 512]       14 ga = GateApply(attn, gate)
//!  4 qn = RmsNorm(qh, q_norm)          10 v  = n · Wv   [M, 512]          15 o  = ga · Wo
//!  5 qf = Reshape(qn) [M, 2048]                                           16 y  = o + x   (result)
//! ```
//!
//! Shapes: hidden `H = 1024`, 8 query heads over 2 kv heads of `HD = 256`, rotated width `R = 64`,
//! `rms_norm_eps` 1e-6, the zero-centred gain, `scale = 256^-0.5 = 0.0625`. Model layer 3 (the
//! first full-attention layer) is KV pool index 0. Every weight is declared in the IR as `[K, N]`;
//! its bytes on the device are `[N, K]`.
//!
//! Sources: 0 `x` (the embedded hidden), 1 `g_in`, 2 `Wq`, 3 `q_norm`, 4 `Wk`, 5 `k_norm`, 6 `Wv`,
//! 7, 8 `cos_q`, `sin_q` and 9, 10 `cos_k`, `sin_k` (the global rotary table), 11 `pk` and 12 `pv`
//! (`PrefixK { 3 }` and `PrefixV { 3 }`, `[8192, 512]`, codegen's prefix capacity) and 13 `Wo`.
//!
//! Colours: 0 `x`; 1 `n`, then `o`; 2 `qg`; 3 `q` (and `qh`); 4 `qn` (and `qf`); 5 `k` (and `kh`),
//! then `v`; 6 `kn` (and `kf`); 7 `qr`, then `ga`; 8 `kr`; 9 `attn`; 10 `y`; 11 the gate. After
//! the colours: the rotary table, the block table, the slot mapping and the sequence.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::{
    AttnMask, EwKind, GainConvention, RopeFormTag, RowScale, SourceShape, SubOp,
};
use scratchy_target_cuda::step::{
    AddBf16, AttnPrefillBf16, CUDA_KV_BLOCK_SIZE, CUDA_MAX_BLOCKS_PER_SEQ, CudaStep, GateApplyBf16,
    GateSplitBf16, GemmBf16, RmsNormBf16, RopeAppendBf16, RopeBf16, RopePairing,
};
use scratchy_target_cuda::tape::{
    ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, KvClass, KvTape, RotaryTable, TapeInput,
    WeightIx,
};

use super::chains::with_hidden;

/// qwen3.5-0.8b's hidden size.
pub const H: u32 = 1024;
/// qwen3.5's head dim.
pub const HD: u32 = 256;
/// qwen3.5-0.8b's query heads.
pub const HEADS: u32 = 8;
/// qwen3.5-0.8b's kv heads.
pub const KV_HEADS: u32 = 2;
/// The query width, `HEADS · HD`.
pub const Q: u32 = HEADS * HD;
/// The K and V width, `KV_HEADS · HD`.
pub const KV: u32 = KV_HEADS * HD;
/// qwen3.5's rotated width, `0.25 · HD`.
pub const R: u32 = 64;
/// qwen3.5's `rope_theta`.
pub const THETA: f64 = 1e7;
/// qwen3.5's `rms_norm_eps`.
pub const EPS: f32 = 1e-6;
/// `HD^-0.5`.
pub const SCALE: f32 = 0.0625;
/// The model layer whose K/V the fixture writes and reads: qwen3.5's first full-attention layer.
pub const LAYER: u32 = 3;
/// Codegen's prefix capacity: the `PrefixK` / `PrefixV` sources' rows.
pub const PREFIX_CAPACITY: u32 = 8192;

/// The layer's ops at `m` rows (the module's table), over the module's sources.
pub fn attn_layer_input(m: u32) -> LoweringInput {
    use InputRef::{Ext, Op};
    let norm = SubOp::RmsNorm {
        eps: EPS,
        gain: GainConvention::OnePlusScale,
    };
    let matmul = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    let op = |op, inputs| OpDesc { op, m, inputs };
    let shape = |rows, cols| SourceShape { rows, cols };
    let view = |heads: u32, cols| SubOp::Reshape {
        rows: RowScale::Times(NonZeroU32::new(heads).expect("a non-zero head count")),
        cols,
    };
    let geom = ModelAttnGeometry::mint(
        QueryHeads::new(HEADS),
        KvHeads::new(KV_HEADS),
        HeadDim::new(HD),
    )
    .expect("8 query heads over 2 kv heads");
    LoweringInput {
        exported: None,
        sources: vec![
            shape(m, H),
            shape(1, H),
            shape(H, 2 * Q),
            shape(1, HD),
            shape(H, KV),
            shape(1, HD),
            shape(H, KV),
            shape(m, Q),
            shape(m, Q),
            shape(m, KV),
            shape(m, KV),
            shape(PREFIX_CAPACITY, KV),
            shape(PREFIX_CAPACITY, KV),
            shape(Q, H),
        ],
        ops: vec![
            // 0: n = RmsNorm(x, g_in)
            op(norm, vec![Ext(0), Ext(1)]),
            // 1: qg = n · Wq
            op(matmul(2 * Q), vec![Op(0), Ext(2)]),
            // 2: q = GateSplit(qg), and the gate, its second output
            op(SubOp::GateSplit { half_cols: Q }, vec![Op(1)]),
            // 3: qh = the per-head view [8m, 256]
            op(view(HEADS, HD), vec![Op(2)]),
            // 4: qn = RmsNorm(qh, q_norm)
            op(norm, vec![Op(3), Ext(3)]),
            // 5: qf = the flatten-back view [m, 2048]
            op(view(1, Q), vec![Op(4)]),
            // 6: k = n · Wk
            op(matmul(KV), vec![Op(0), Ext(4)]),
            // 7: kh = the per-head view [2m, 256]
            op(view(KV_HEADS, HD), vec![Op(6)]),
            // 8: kn = RmsNorm(kh, k_norm)
            op(norm, vec![Op(7), Ext(5)]),
            // 9: kf = the flatten-back view [m, 512]
            op(view(1, KV), vec![Op(8)]),
            // 10: v = n · Wv
            op(matmul(KV), vec![Op(0), Ext(6)]),
            // 11: qr = RopeRotate(qf, cos_q, sin_q)
            op(
                SubOp::rope_rotate(HeadDim::new(HD)),
                vec![Op(5), Ext(7), Ext(8)],
            ),
            // 12: kr = RopeAppend(kf, cos_k, sin_k, v, pk, pv)
            op(
                SubOp::rope_append(HeadDim::new(HD), LAYER, AttnMask::Causal, RopeFormTag::NeoX),
                vec![Op(9), Ext(9), Ext(10), Op(10), Ext(11), Ext(12)],
            ),
            // 13: attn = AttnDecode(qr, pk, pv, kr, v)
            op(
                SubOp::attn_decode(geom, SCALE, PREFIX_CAPACITY, AttnMask::Causal),
                vec![Op(11), Ext(11), Ext(12), Op(12), Op(10)],
            ),
            // 14: ga = GateApply(attn, gate): operand 1 names op 2 and means its second output
            op(SubOp::GateApply, vec![Op(13), Op(2)]),
            // 15: o = ga · Wo
            op(matmul(H), vec![Op(14), Ext(13)]),
            // 16: y = o + x, the result
            op(SubOp::Elementwise(EwKind::Add), vec![Op(15), Ext(0)]),
        ],
        result: 16,
    }
}

/// [`attn_layer_input`] as the front end binds it: `x` the embedded hidden, both rope pairs the
/// global rotary table, `pk` and `pv` layer 3's prefix cache.
pub fn attn_layer(m: u32) -> LoweredDecode {
    let mut l = with_hidden(attn_layer_input(m));
    for (ix, binding) in [
        (7, SourceBinding::Cos { local: false }),
        (8, SourceBinding::Sin { local: false }),
        (9, SourceBinding::Cos { local: false }),
        (10, SourceBinding::Sin { local: false }),
        (
            11,
            SourceBinding::PrefixK {
                layer: u64::from(LAYER),
            },
        ),
        (
            12,
            SourceBinding::PrefixV {
                layer: u64::from(LAYER),
            },
        ),
    ] {
        l.bindings[ix] = binding;
    }
    l
}

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

const fn gemm(a: u64, b: u32, c: u64, m: u32, n: u32, k: u32) -> CudaStep {
    CudaStep::Gemm(GemmBf16 {
        a: ArenaOffset(a),
        b: WeightIx(b),
        c: ArenaOffset(c),
        m,
        n,
        k,
    })
}

const fn norm(x: u64, gain: u32, y: u64, rows: u32, cols: u32) -> CudaStep {
    CudaStep::RmsNorm(RmsNormBf16 {
        x: ArenaOffset(x),
        gain: WeightIx(gain),
        y: ArenaOffset(y),
        rows,
        cols,
        eps_bits: EPS.to_bits(),
        convention: GainConvention::OnePlusScale,
    })
}

const fn gate_split(qg: u64, q: u64, gate: u64, rows: u32) -> CudaStep {
    CudaStep::GateSplit(GateSplitBf16 {
        qg: ArenaOffset(qg),
        q: ArenaOffset(q),
        gate: ArenaOffset(gate),
        rows,
        half_cols: Q,
        head_dim: HD,
    })
}

const fn rope(x: u64, table: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::Rope(RopeBf16 {
        x: ArenaOffset(x),
        table: ArenaOffset(table),
        out: ArenaOffset(out),
        rows,
        cols: Q,
        head_dim: HD,
        rot_dim: R,
        pairing: RopePairing::HalfRotated,
    })
}

const fn rope_append(
    (k, v): (u64, u64),
    table: u64,
    slot_mapping: u64,
    k_out: u64,
    rows: u32,
) -> CudaStep {
    CudaStep::RopeAppend(RopeAppendBf16 {
        k: ArenaOffset(k),
        v: ArenaOffset(v),
        table: ArenaOffset(table),
        slot_mapping: ArenaOffset(slot_mapping),
        k_out: ArenaOffset(k_out),
        layer: 0,
        rows,
        cols: KV,
        head_dim: HD,
        rot_dim: R,
        pairing: RopePairing::HalfRotated,
    })
}

const fn attn(q: u64, block_table: u64, sequence: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::AttnPrefill(AttnPrefillBf16 {
        q: ArenaOffset(q),
        block_table: ArenaOffset(block_table),
        sequence: ArenaOffset(sequence),
        out: ArenaOffset(out),
        layer: 0,
        rows,
        q_heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HD,
        scale_bits: SCALE.to_bits(),
        window: None,
    })
}

const fn gate_apply(attn: u64, gate: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::GateApply(GateApplyBf16 {
        attn: ArenaOffset(attn),
        gate: ArenaOffset(gate),
        out: ArenaOffset(out),
        rows,
        cols: Q,
    })
}

const fn add(a: u64, b: u64, out: u64, rows: u32) -> CudaStep {
    CudaStep::Add(AddBf16 {
        a: ArenaOffset(a),
        b: ArenaOffset(b),
        out: ArenaOffset(out),
        rows,
        cols: H,
    })
}

const fn rotary(offset: u64, bytes: u64) -> Option<RotaryTable> {
    Some(RotaryTable {
        span: span(offset, bytes),
        rot_dim: R,
        theta_bits: THETA.to_bits(),
    })
}

/// Layer 3's cache, with its three regions at `table`, `slots` (`slot_bytes` of them) and `seq`.
const fn kv(table: u64, (slots, slot_bytes): (u64, u64), seq: u64) -> Option<KvTape> {
    Some(KvTape {
        layers: &[LAYER],
        kv_heads: KV_HEADS,
        head_dim: HD,
        block_size: CUDA_KV_BLOCK_SIZE,
        max_blocks_per_seq: CUDA_MAX_BLOCKS_PER_SEQ,
        block_table: span(table, 512),
        slot_mapping: span(slots, slot_bytes),
        sequence: span(seq, 8),
        classes: &[KvClass {
            kv_heads: KV_HEADS,
            head_dim: HD,
            window: None,
        }],
        layer_class: &[0],
    })
}

/// `g_in` is `H` bf16, `Wq` `2Q × H`, `q_norm` and `k_norm` `HD`, `Wk` and `Wv` `KV × H`, `Wo`
/// `H × Q`; `x`, the rotary pairs and the prefix caches are not uploaded as weights.
const ATTN_LAYER_WEIGHT_BYTES: &[u64] = &[
    0, 2048, 8388608, 512, 1048576, 512, 1048576, 0, 0, 0, 0, 0, 0, 4194304,
];

/// [`attn_layer`]`(128)`: every colour offset and size of [`ATTN_LAYER_M256`] halved, and the
/// rotary table and slot mapping with them; the block table (512) and the sequence (its 256-byte
/// slot) are fixed. Colours end at 4849664; the table at 4849664, the block table at 4866048, the
/// slot mapping at 4866560, the sequence at 4867072.
pub static ATTN_LAYER_M128: CudaTape = CudaTape {
    bucket_m: 128,
    arena: ArenaLayout {
        colour_bytes: &[
            262144, 262144, 1048576, 524288, 524288, 131072, 131072, 524288, 131072, 524288,
            262144, 524288,
        ],
        colour_offsets: &[
            0, 262144, 524288, 1572864, 2097152, 2621440, 2752512, 2883584, 3407872, 3538944,
            4063232, 4325376,
        ],
        total_bytes: 4867328,
    },
    input: TapeInput::Hidden(span(0, 262144)),
    result: span(4063232, 262144),
    weight_bytes: ATTN_LAYER_WEIGHT_BYTES,
    steps: &[
        norm(0, 1, 262144, 128, H),
        gemm(262144, 2, 524288, 128, 2 * Q, H),
        gate_split(524288, 1572864, 4325376, 128),
        norm(1572864, 3, 2097152, 1024, HD),
        gemm(262144, 4, 2621440, 128, KV, H),
        norm(2621440, 5, 2752512, 256, HD),
        gemm(262144, 6, 2621440, 128, KV, H),
        rope(2097152, 4849664, 2883584, 128),
        rope_append((2752512, 2621440), 4849664, 4866560, 3407872, 128),
        attn(2883584, 4866048, 4867072, 3538944, 128),
        gate_apply(3538944, 4325376, 2883584, 128),
        gemm(2883584, 13, 262144, 128, H, Q),
        add(262144, 0, 4063232, 128),
    ],
    rotary: rotary(4849664, 16384),
    kv: kv(4866048, (4866560, 512), 4867072),
    gdn: None,
    moe: None,
    rotary_local: None,
};

/// [`attn_layer`]`(256)`: `x` 0, `n` (then `o`) 524288, `qg` 1048576, `q` 3145728, `qn` 4194304,
/// `k` (then `v`) 5242880, `kn` 5505024, `qr` (then `ga`) 5767168, `kr` 6815744, `attn` 7077888,
/// `y` 8126464, the gate 8650752; colours end at 9699328. The rotary table (32768 bytes) at
/// 9699328, the block table (512) at 9732096, the slot mapping (1024) at 9732608, the sequence (8,
/// in a 256-byte slot) at 9733632.
pub static ATTN_LAYER_M256: CudaTape = CudaTape {
    bucket_m: 256,
    arena: ArenaLayout {
        colour_bytes: &[
            524288, 524288, 2097152, 1048576, 1048576, 262144, 262144, 1048576, 262144, 1048576,
            524288, 1048576,
        ],
        colour_offsets: &[
            0, 524288, 1048576, 3145728, 4194304, 5242880, 5505024, 5767168, 6815744, 7077888,
            8126464, 8650752,
        ],
        total_bytes: 9733888,
    },
    input: TapeInput::Hidden(span(0, 524288)),
    result: span(8126464, 524288),
    weight_bytes: ATTN_LAYER_WEIGHT_BYTES,
    steps: &[
        norm(0, 1, 524288, 256, H),
        gemm(524288, 2, 1048576, 256, 2 * Q, H),
        gate_split(1048576, 3145728, 8650752, 256),
        norm(3145728, 3, 4194304, 2048, HD),
        gemm(524288, 4, 5242880, 256, KV, H),
        norm(5242880, 5, 5505024, 512, HD),
        gemm(524288, 6, 5242880, 256, KV, H),
        rope(4194304, 9699328, 5767168, 256),
        rope_append((5505024, 5242880), 9699328, 9732608, 6815744, 256),
        attn(5767168, 9732096, 9733632, 7077888, 256),
        gate_apply(7077888, 8650752, 5767168, 256),
        gemm(5767168, 13, 524288, 256, H, Q),
        add(524288, 0, 8126464, 256),
    ],
    rotary: rotary(9699328, 32768),
    kv: kv(9732096, (9732608, 1024), 9733632),
    gdn: None,
    moe: None,
    rotary_local: None,
};
