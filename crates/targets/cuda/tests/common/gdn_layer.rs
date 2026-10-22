// SPDX-License-Identifier: Apache-2.0
//! One qwen3.5-0.8b linear-attention layer: the canonical the GDN tape tests lower, and its
//! `static` tapes written out by hand from the issue's numbers, which `tape_layout` checks
//! against what the compiler lowers.
//!
//! The ops, in the order the macro emits them (`crates/models/arch/dsl/qwen3-5.py:17-23`, and
//! `gated_delta_net` as `GatedDeltaNet(qkv, z, a, b, bundle)` at
//! `crates/compiler/macros/src/to_wavefront.rs:1530-1556`):
//!
//! ```text
//! 0 n    = RmsNorm(x, g_in)                    [M, 1024]
//! 1 qkv  = n · Wqkv                            [M, 6144]   Gemm
//! 2 z    = n · Wz                              [M, 2048]   Gemm
//! 3 a    = n · Wa                              [M, 16]     GemmNarrow
//! 4 b    = n · Wb                              [M, 16]     GemmNarrow
//! 5 core = GatedDeltaNet(qkv, z, a, b, bundle) [M, 2048]   conv1d, gating, scan, gated norm
//! 6 o    = core · Wo                           [M, 1024]   Gemm
//! 7 y    = o + x                               [M, 1024]   the result
//! ```
//!
//! Shapes are qwen3.5-0.8b's: hidden `H = 1024`, 16 key heads and 16 value heads of 128 (`key_dim
//! = value_dim = 2048`, `conv_dim = 6144`), a 4-tap conv, `rms_norm_eps` 1e-6, the zero-centred
//! input gain. The bundle (source 6) is the macro's `[1, 1]` source bound to model layer 0's
//! `linear_attn` weight; on the device it is the packed f32 `gdn_bundle`. Every other weight is
//! declared in the IR as `[K, N]`; its bytes on the device are `[N, K]`.
//!
//! Sources: 0 `x` (the embedded hidden), 1 `g_in`, 2 `Wqkv`, 3 `Wz`, 4 `Wa`, 5 `Wb`, 6 the
//! bundle, 7 `Wo`.
//!
//! Colours: 0 `x`; 1 `n`, then `o` (`n` is dead after step 4); 2 `qkv`; 3 `z`; 4 `a`; 5 `b`; 6
//! `core` (not `z`'s colour 3: the GDN op reads its inputs across the step); 7 `y`. After the
//! colours: the GDN sequence region (16 bytes in a 256-byte slot), then the GDN scratch.

use scratchy_subtile::handoff::{LoweredDecode, SourceBinding, UnrollIndex};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SourceShape, SubOp};
use scratchy_target_cuda::step::{
    AddBf16, CudaStep, GatedDeltaNetBf16, GemmBf16, RmsNormBf16, gdn_bundle,
};
use scratchy_target_cuda::tape::{
    ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, GdnTape, TapeInput, WeightIx,
};

use super::chains::with_hidden;

/// qwen3.5-0.8b's hidden size.
pub const H: u32 = 1024;
/// qwen3.5-0.8b's linear-attention key heads.
pub const NK: u32 = 16;
/// qwen3.5-0.8b's linear-attention value heads.
pub const NV: u32 = 16;
/// qwen3.5's linear-attention key and value head dims.
pub const HEAD_K: u32 = 128;
pub const HEAD_V: u32 = 128;
/// `NK · HEAD_K`.
pub const KEY_DIM: u32 = NK * HEAD_K;
/// `NV · HEAD_V`.
pub const VALUE_DIM: u32 = NV * HEAD_V;
/// The conv's channels, `2 · KEY_DIM + VALUE_DIM`.
pub const CONV_DIM: u32 = 2 * KEY_DIM + VALUE_DIM;
/// qwen3.5's `linear_conv_kernel_dim`.
pub const CONV_KERNEL: u32 = 4;
/// qwen3.5's `rms_norm_eps`.
pub const EPS: f32 = 1e-6;
/// The model layer the fixture's GDN state is: qwen3.5's first linear-attention layer.
pub const LAYER: u32 = 0;
/// The bundle's source index.
pub const BUNDLE: u32 = 6;

/// The layer's ops at `m` rows (the module's table), over the module's sources.
pub fn gdn_layer_input(m: u32) -> LoweringInput {
    use InputRef::{Ext, Op};
    let shape = |rows, cols| SourceShape { rows, cols };
    let matmul = |n| SubOp::MatmulTile {
        n,
        weight: GemmWeight::Dense,
    };
    let op = |op, inputs| OpDesc { op, m, inputs };
    LoweringInput {
        exported: None,
        sources: vec![
            shape(m, H),
            shape(1, H),
            shape(H, CONV_DIM),
            shape(H, VALUE_DIM),
            shape(H, NV),
            shape(H, NV),
            shape(1, 1),
            shape(VALUE_DIM, H),
        ],
        ops: vec![
            // 0: n = RmsNorm(x, g_in)
            op(
                SubOp::RmsNorm {
                    eps: EPS,
                    gain: GainConvention::OnePlusScale,
                },
                vec![Ext(0), Ext(1)],
            ),
            // 1-4: the four input projections
            op(matmul(CONV_DIM), vec![Op(0), Ext(2)]),
            op(matmul(VALUE_DIM), vec![Op(0), Ext(3)]),
            op(matmul(NV), vec![Op(0), Ext(4)]),
            op(matmul(NV), vec![Op(0), Ext(5)]),
            // 5: core = GatedDeltaNet(qkv, z, a, b, bundle)
            op(
                SubOp::GatedDeltaNet,
                vec![Op(1), Op(2), Op(3), Op(4), Ext(BUNDLE as usize)],
            ),
            // 6: o = core · Wo
            op(matmul(H), vec![Op(5), Ext(7)]),
            // 7: y = o + x, the result
            op(SubOp::Elementwise(EwKind::Add), vec![Op(6), Ext(0)]),
        ],
        result: 7,
    }
}

/// [`gdn_layer_input`] as the front end binds it: `x` the embedded hidden, the bundle model layer
/// 0's `linear_attn` weight.
pub fn gdn_layer(m: u32) -> LoweredDecode {
    let mut l = with_hidden(gdn_layer_input(m));
    l.bindings[BUNDLE as usize] = SourceBinding::Weight {
        id: BUNDLE,
        index: Some(UnrollIndex(u64::from(LAYER))),
    };
    l
}

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

const fn gemm(a: u64, b: u32, c: u64, (m, n, k): (u32, u32, u32)) -> GemmBf16 {
    GemmBf16 {
        a: ArenaOffset(a),
        b: WeightIx(b),
        c: ArenaOffset(c),
        m,
        n,
        k,
    }
}

const fn norm(x: u64, y: u64, rows: u32) -> CudaStep {
    CudaStep::RmsNorm(RmsNormBf16 {
        x: ArenaOffset(x),
        gain: WeightIx(1),
        y: ArenaOffset(y),
        rows,
        cols: H,
        eps_bits: EPS.to_bits(),
        convention: GainConvention::OnePlusScale,
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

/// The GDN step over `qkv`, `z`, `a`, `b` into `out`, with the tape's `sequence` and `scratch`.
pub const fn gdn_step(
    (qkv, z, a, b): (u64, u64, u64, u64),
    out: u64,
    (sequence, scratch): (u64, u64),
    rows: u32,
) -> GatedDeltaNetBf16 {
    GatedDeltaNetBf16 {
        qkv: ArenaOffset(qkv),
        z: ArenaOffset(z),
        a: ArenaOffset(a),
        b: ArenaOffset(b),
        bundle: WeightIx(BUNDLE),
        out: ArenaOffset(out),
        scratch: ArenaOffset(scratch),
        sequence: ArenaOffset(sequence),
        layer: 0,
        rows,
        num_k_heads: NK,
        num_v_heads: NV,
        head_k: HEAD_K,
        head_v: HEAD_V,
        conv_kernel: CONV_KERNEL,
        eps_bits: EPS.to_bits(),
    }
}

/// Layer 0's GDN state, its sequence region at `sequence` and its scratch (`bytes`) at `scratch`.
const fn gdn_tape(sequence: u64, (scratch, bytes): (u64, u64)) -> Option<GdnTape> {
    Some(GdnTape {
        layers: &[LAYER],
        num_k_heads: NK,
        num_v_heads: NV,
        head_k: HEAD_K,
        head_v: HEAD_V,
        conv_kernel: CONV_KERNEL,
        sequence: span(sequence, 16),
        scratch: span(scratch, bytes),
    })
}

/// The packed bundle's bytes: `conv1d [6144, 4]` f32 (98304) at 0, `a_log [16]` at 98304,
/// `dt_bias [16]` at 98560, `norm [128]` at 98816, each in a 256-byte slot: 99328.
pub const BUNDLE_BYTES: u64 = 99328;

const _: () = {
    let Some(b) = gdn_bundle(CONV_DIM, CONV_KERNEL, NV, HEAD_V) else {
        panic!("the bundle's bytes fit u64");
    };
    assert!(b.conv1d == 0 && b.a_log == 98304 && b.dt_bias == 98560 && b.norm == 98816);
    assert!(b.bytes == BUNDLE_BYTES);
};

/// `g_in` is `H` bf16, `Wqkv` `CONV_DIM × H`, `Wz` `VALUE_DIM × H`, `Wa` and `Wb` `NV × H`, the
/// bundle [`BUNDLE_BYTES`], `Wo` `H × VALUE_DIM`; `x` is not uploaded as a weight.
const GDN_LAYER_WEIGHT_BYTES: &[u64] = &[
    0,
    2048,
    12582912,
    4194304,
    32768,
    32768,
    BUNDLE_BYTES,
    4194304,
];

/// [`gdn_layer`]`(128)`: `x` 0, `n` (then `o`) 262144, `qkv` 524288, `z` 2097152, `a` 2621440, `b`
/// 2625536, `core` 2629632, `y` 3153920; colours end at 3416064. The GDN sequence (16 bytes, in a
/// 256-byte slot) at 3416064, the scratch (4210688 bytes: `conv_out` 3145728, `g` 8192, `beta`
/// 8192, `o` 1048576) at 3416320. Arena 7627008.
pub static GDN_LAYER_M128: CudaTape = CudaTape {
    bucket_m: 128,
    arena: ArenaLayout {
        colour_bytes: &[262144, 262144, 1572864, 524288, 4096, 4096, 524288, 262144],
        colour_offsets: &[
            0, 262144, 524288, 2097152, 2621440, 2625536, 2629632, 3153920,
        ],
        total_bytes: 7627008,
    },
    input: TapeInput::Hidden(span(0, 262144)),
    result: span(3153920, 262144),
    weight_bytes: GDN_LAYER_WEIGHT_BYTES,
    steps: &[
        norm(0, 262144, 128),
        CudaStep::Gemm(gemm(262144, 2, 524288, (128, CONV_DIM, H))),
        CudaStep::Gemm(gemm(262144, 3, 2097152, (128, VALUE_DIM, H))),
        CudaStep::GemmNarrow(gemm(262144, 4, 2621440, (128, NV, H))),
        CudaStep::GemmNarrow(gemm(262144, 5, 2625536, (128, NV, H))),
        CudaStep::GatedDeltaNet(gdn_step(
            (524288, 2097152, 2621440, 2625536),
            2629632,
            (3416064, 3416320),
            128,
        )),
        CudaStep::Gemm(gemm(2629632, 7, 262144, (128, H, VALUE_DIM))),
        add(262144, 0, 3153920, 128),
    ],
    rotary: None,
    kv: None,
    gdn: gdn_tape(3416064, (3416320, 4210688)),
};

/// [`gdn_layer`]`(256)`: every colour of [`GDN_LAYER_M128`] doubled. `x` 0, `n` (then `o`) 524288,
/// `qkv` 1048576, `z` 4194304, `a` 5242880, `b` 5251072, `core` 5259264, `y` 6307840; colours end
/// at 6832128. The GDN sequence at 6832128, the scratch (8421376 bytes: 6291456, 16384, 16384,
/// 2097152) at 6832384. Arena 15253760.
pub static GDN_LAYER_M256: CudaTape = CudaTape {
    bucket_m: 256,
    arena: ArenaLayout {
        colour_bytes: &[
            524288, 524288, 3145728, 1048576, 8192, 8192, 1048576, 524288,
        ],
        colour_offsets: &[
            0, 524288, 1048576, 4194304, 5242880, 5251072, 5259264, 6307840,
        ],
        total_bytes: 15253760,
    },
    input: TapeInput::Hidden(span(0, 524288)),
    result: span(6307840, 524288),
    weight_bytes: GDN_LAYER_WEIGHT_BYTES,
    steps: &[
        norm(0, 524288, 256),
        CudaStep::Gemm(gemm(524288, 2, 1048576, (256, CONV_DIM, H))),
        CudaStep::Gemm(gemm(524288, 3, 4194304, (256, VALUE_DIM, H))),
        CudaStep::GemmNarrow(gemm(524288, 4, 5242880, (256, NV, H))),
        CudaStep::GemmNarrow(gemm(524288, 5, 5251072, (256, NV, H))),
        CudaStep::GatedDeltaNet(gdn_step(
            (1048576, 4194304, 5242880, 5251072),
            5259264,
            (6832128, 6832384),
            256,
        )),
        CudaStep::Gemm(gemm(5259264, 7, 524288, (256, H, VALUE_DIM))),
        add(524288, 0, 6307840, 256),
    ],
    rotary: None,
    kv: None,
    gdn: gdn_tape(6832128, (6832384, 8421376)),
};
