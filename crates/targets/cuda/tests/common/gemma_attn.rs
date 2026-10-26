// SPDX-License-Identifier: Apache-2.0
//! One Gemma 4 26B-A4B sliding attention layer at its real geometry (#922), as `lower_canonical`
//! lowers it under the model's two attention classes. The ops, in the order the DSL's sliding
//! branch emits them (`crates/models/arch/dsl/gemma4-moe.py`):
//!
//! ```text
//!  0 n  = RmsNorm(x, g_in)              6 kh = Reshape(k) [8M, 256]       12 qr = RopeRotate(qf, cos, sin)
//!  1 q  = n · Wq   [M, 4096]            7 kn = RmsNorm(kh, k_norm)        13 kr = RopeAppend(kf, cos, sin, vn, pk, pv)
//!  2 qh = Reshape(q) [16M, 256]         8 kf = Reshape(kn) [M, 2048]      14 attn = AttnDecode(qr, pk, pv, kr, vn)
//!  3 qn = RmsNorm(qh, q_norm)           9 v  = n · Wv   [M, 2048]         15 o  = attn · Wo  (result)
//!  4 qf = Reshape(qn) [M, 4096]        10 vn = RmsNormUnit(v)
//!  5 k  = n · Wk   [M, 2048]           11 (the global table's unread binding, below)
//! ```
//!
//! Shapes: hidden `H = 2816`, 16 query heads over 8 kv heads of `HD = 256` (a GQA group of 2),
//! rotated in full (`R = 256`, theta 1e4, the half-rotated pairing: lane `d` with `d + 128`), a
//! window of 1024 keys, `rms_norm_eps` 1e-6 with the stored gain (`GainConvention::Scale`), and a
//! softmax scale of 1 (the q and k norms size the scores). The v norm is per KV head
//! (`RmsNormUnit` at `(8M, 256)`, #921). Model layer 0 (a sliding layer) is KV pool index 0. Every
//! weight is declared in the IR as `[K, N]`; its bytes on the device are `[N, K]`. The q, k and v
//! projections are 4096, 2048 and 2048 columns (the 256-column tile) and `o` 2816 columns of a
//! 4096 `K`: at m ≥ 2 a GEMM on the 256-column tile only if 2816 is a multiple of 256, which it is
//! (11 tiles); at m = 1 every projection is a GEMV. So the layer launches completely.
//!
//! Sources: 0 `x` (the embedded hidden), 1 `g_in`, 2 `Wq`, 3 `q_norm`, 4 `Wk`, 5 `k_norm`, 6 `Wv`,
//! 7, 8 `cos_q`, `sin_q` and 9, 10 `cos_k`, `sin_k` (the local rotary table), 11 `pk` and 12 `pv`
//! (`PrefixK { 0 }` and `PrefixV { 0 }`, `[8192, 2048]`, codegen's prefix capacity), 13 `Wo`, and
//! 14 a `[1, 1]` source bound to `Cos { local: false }` that no op reads: the canonical places a
//! rotary table exactly when its source is bound (`compiler/src/canonical.rs`, `table`), and a
//! tape with a local table needs the global one (`CudaTape::validate`, `TapeError::KvClasses`), as
//! every Gemma tape has both.

use std::num::NonZeroU32;

use scratchy_subtile::handoff::{LoweredDecode, RopePairing, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::{
    AttnMask, GainConvention, RopeFormTag, RowScale, SourceShape, SubOp,
};
use scratchy_target_cuda_compiler::canonical::{CudaAttnClass, CudaAttnClasses, CudaModelFacts};

use super::chains::{GEMMA4_FACTS, with_hidden};

/// Gemma 4 26B-A4B's hidden size.
pub const H: u32 = 2816;
/// Its sliding layers' head dim.
pub const HD: u32 = 256;
/// Its query heads.
pub const HEADS: u32 = 16;
/// Its sliding layers' kv heads.
pub const KV_HEADS: u32 = 8;
/// The query width, `HEADS · HD`.
pub const Q: u32 = HEADS * HD;
/// The K and V width, `KV_HEADS · HD`.
pub const KV: u32 = KV_HEADS * HD;
/// The sliding layers' rotated width: the whole head.
pub const R: u32 = 256;
/// The sliding layers' `rope_theta`.
pub const THETA: f64 = 1e4;
/// The global layers' `rope_theta`, which the unread global table is built at.
pub const THETA_GLOBAL: f64 = 1e6;
/// The global layers' rotated width (`partial_rotary_factor` 0.25 of 512).
pub const R_GLOBAL: u32 = 128;
/// The sliding window, in keys.
pub const WINDOW: u32 = 1024;
/// Its `rms_norm_eps`.
pub const EPS: f32 = 1e-6;
/// Its softmax scale.
pub const SCALE: f32 = 1.0;
/// The model layer whose K/V the fixture writes and reads: a sliding layer.
pub const LAYER: u32 = 0;
/// Codegen's prefix capacity: the `PrefixK` / `PrefixV` sources' rows.
pub const PREFIX_CAPACITY: u32 = 8192;

/// Gemma 4's two attention classes over [`GEMMA4_FACTS`]: the sliding (base) class, 8 kv heads of
/// 256 rotated in full at theta 1e4 under a window of 1024, and the global class, 2 kv heads of
/// 512, 128 of each rotated (proportional, the half-head pairing) at theta 1e6, with no window.
pub const GEMMA_ATTN_FACTS: CudaModelFacts = CudaModelFacts {
    head_dim: HD,
    rot_dim: R,
    rope_theta: THETA,
    classes: Some(CudaAttnClasses {
        base: CudaAttnClass {
            head_dim: HD,
            kv_heads: KV_HEADS,
            rot_dim: R,
            rope_theta: THETA,
            pairing: RopePairing::HalfRotated,
            window: Some(WINDOW),
        },
        global: CudaAttnClass {
            head_dim: 512,
            kv_heads: 2,
            rot_dim: R_GLOBAL,
            rope_theta: THETA_GLOBAL,
            pairing: RopePairing::HalfHead,
            window: None,
        },
    }),
    ..GEMMA4_FACTS
};

/// The layer's ops at `m` rows (the module's table), over the module's sources.
pub fn gemma_attn_input(m: u32) -> LoweringInput {
    use InputRef::{Ext, Op};
    let norm = SubOp::RmsNorm {
        eps: EPS,
        gain: GainConvention::Scale,
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
    .expect("16 query heads over 8 kv heads");
    let mask = AttnMask::SlidingWindow {
        window: NonZeroU32::new(WINDOW).expect("a window"),
    };
    LoweringInput {
        exported: None,
        sources: vec![
            shape(m, H),
            shape(1, H),
            shape(H, Q),
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
            shape(1, 1),
        ],
        ops: vec![
            // 0: n = RmsNorm(x, g_in)
            op(norm, vec![Ext(0), Ext(1)]),
            // 1: q = n · Wq
            op(matmul(Q), vec![Op(0), Ext(2)]),
            // 2: qh = the per-head view [16m, 256]
            op(view(HEADS, HD), vec![Op(1)]),
            // 3: qn = RmsNorm(qh, q_norm)
            op(norm, vec![Op(2), Ext(3)]),
            // 4: qf = the flatten-back view [m, 4096]
            op(view(1, Q), vec![Op(3)]),
            // 5: k = n · Wk
            op(matmul(KV), vec![Op(0), Ext(4)]),
            // 6: kh = the per-head view [8m, 256]
            op(view(KV_HEADS, HD), vec![Op(5)]),
            // 7: kn = RmsNorm(kh, k_norm)
            op(norm, vec![Op(6), Ext(5)]),
            // 8: kf = the flatten-back view [m, 2048]
            op(view(1, KV), vec![Op(7)]),
            // 9: v = n · Wv
            op(matmul(KV), vec![Op(0), Ext(6)]),
            // 10: vn = RmsNormUnit(v), per kv head
            op(SubOp::RmsNormUnit { eps: EPS }, vec![Op(9)]),
            // 11: qr = RopeRotate(qf, cos_q, sin_q)
            op(
                SubOp::rope_rotate(HeadDim::new(HD)),
                vec![Op(4), Ext(7), Ext(8)],
            ),
            // 12: kr = RopeAppend(kf, cos_k, sin_k, vn, pk, pv)
            op(
                SubOp::rope_append(HeadDim::new(HD), LAYER, mask, RopeFormTag::NeoX),
                vec![Op(8), Ext(9), Ext(10), Op(10), Ext(11), Ext(12)],
            ),
            // 13: attn = AttnDecode(qr, pk, pv, kr, vn), windowed
            op(
                SubOp::attn_decode(geom, SCALE, PREFIX_CAPACITY, mask),
                vec![Op(11), Ext(11), Ext(12), Op(12), Op(10)],
            ),
            // 14: o = attn · Wo, the result
            op(matmul(H), vec![Op(13), Ext(13)]),
        ],
        result: 14,
    }
}

/// [`gemma_attn_input`] as the front end binds it: `x` the embedded hidden, both rope pairs the
/// local rotary table, `pk` and `pv` layer 0's prefix cache, and source 14 the global table's
/// cos, which no op reads.
pub fn gemma_attn(m: u32) -> LoweredDecode {
    let mut l = with_hidden(gemma_attn_input(m));
    for (ix, binding) in [
        (7, SourceBinding::Cos { local: true }),
        (8, SourceBinding::Sin { local: true }),
        (9, SourceBinding::Cos { local: true }),
        (10, SourceBinding::Sin { local: true }),
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
        (14, SourceBinding::Cos { local: false }),
    ] {
        l.bindings[ix] = binding;
    }
    l
}
