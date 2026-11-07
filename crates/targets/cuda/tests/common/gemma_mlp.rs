// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 26B-A4B's dense MLP block at its real geometry (#921): `y = x + down(gelu_tanh(gate(n))
//! · up(n))`, `n = RmsNorm(x, g)` with the stored gain (`GainConvention::Scale`). At m = 1 its
//! GEMMs are GEMVs; m ≥ 2 needs the 64-column GEMM (13b-4).
//!
//! Shapes: hidden `H = 2816`, intermediate `I = 2112`, `rms_norm_eps` 1e-6. Every weight is
//! declared in the IR as `[K, N]`; its bytes on the device are `[N, K]`. Sources: 0 `x [m, H]`
//! (rebound to the embedded hidden), 1 `g [1, H]`, 2 `Wg [H, I]`, 3 `Wu [H, I]`, 4 `Wd [I, H]`.

use scratchy_subtile::handoff::LoweredDecode;
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{EwKind, GainConvention, SourceShape, SubOp};
use scratchy_target_cuda_compiler::canonical::CudaModelFacts;

use super::chains::{GEMMA4_FACTS, with_hidden};

/// Gemma 4 26B-A4B's hidden size.
pub const H: u32 = 2816;
/// Its dense MLP's intermediate size.
pub const I: u32 = 2112;
/// Its `rms_norm_eps`.
pub const EPS: f32 = 1e-6;

/// The block lowers with Gemma 4's facts.
pub const GEMMA_MLP_FACTS: CudaModelFacts = GEMMA4_FACTS;

/// The block's ops as the macro emits them (`Gelu` and `Mul` apart, the residual add as
/// `add(delta, residual)`), at `m` rows.
pub fn gemma_mlp_input(m: u32) -> LoweringInput {
    let matmul = |n, act, w| OpDesc {
        op: SubOp::MatmulTile {
            n,
            weight: GemmWeight::Dense,
        },
        m,
        inputs: vec![act, InputRef::Ext(w)],
    };
    let ew = |kind, inputs| OpDesc {
        op: SubOp::Elementwise(kind),
        m,
        inputs,
    };
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            SourceShape { rows: 1, cols: H },
            SourceShape { rows: H, cols: I },
            SourceShape { rows: H, cols: I },
            SourceShape { rows: I, cols: H },
        ],
        ops: vec![
            // 0: n = RmsNorm(x, g)
            OpDesc {
                op: SubOp::RmsNorm {
                    eps: EPS,
                    gain: GainConvention::Scale,
                },
                m,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            // 1: gate = n · Wg
            matmul(I, InputRef::Op(0), 2),
            // 2: up = n · Wu
            matmul(I, InputRef::Op(0), 3),
            // 3: s = gelu(gate), absorbed into op 4
            ew(EwKind::Gelu, vec![InputRef::Op(1)]),
            // 4: a = s · up, the GeluMul fold's driver
            ew(EwKind::Mul, vec![InputRef::Op(3), InputRef::Op(2)]),
            // 5: d = a · Wd
            matmul(H, InputRef::Op(4), 4),
            // 6: y = d + x, the result
            ew(EwKind::Add, vec![InputRef::Op(5), InputRef::Ext(0)]),
        ],
        result: 6,
    }
}

/// [`gemma_mlp_input`] as the front end binds it, with `x` rebound to the embedded hidden.
pub fn gemma_mlp(m: u32) -> LoweredDecode {
    with_hidden(gemma_mlp_input(m))
}
