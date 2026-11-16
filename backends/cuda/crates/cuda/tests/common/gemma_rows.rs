// SPDX-License-Identifier: Apache-2.0
//! Gemma 4's row maps as one chain (#921): `x [m, 2816] → ScalarMul(√2816) → RmsNormUnit(1e-6) →
//! ScalarWeightMul(w) → TanhSoftCap(30)`, each op one step of its own kernel. Not a Gemma layer:
//! the four steps in one tape, so the tape player runs each against its standalone launcher.
//!
//! Sources: 0 `x` (rebound to the embedded hidden, colour 0, the forward's input), 1 `w [1, 1]`
//! (a `layer_scalar`).

use scratchy_subtile::handoff::LoweredDecode;
use scratchy_subtile::lower::{InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{SourceShape, SubOp};
use scratchy_target_cuda_compiler::canonical::CudaModelFacts;

use super::chains::{GEMMA4_FACTS, with_hidden};

/// Gemma 4 26B-A4B's hidden size.
pub const H: u32 = 2816;
/// Its `rms_norm_eps`.
pub const EPS: f32 = 1e-6;
/// Its `final_logit_softcapping`.
pub const CAP: f32 = 30.0;

/// The chain lowers with Gemma 4's facts.
pub const GEMMA_ROWS_FACTS: CudaModelFacts = GEMMA4_FACTS;

/// The embedding scale, `√H`.
pub fn embed_scale() -> f32 {
    (H as f32).sqrt()
}

/// The chain's ops at `m` rows.
pub fn gemma_rows_input(m: u32) -> LoweringInput {
    let op = |op, inputs| OpDesc { op, m, inputs };
    use InputRef::{Ext, Op};
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            SourceShape { rows: 1, cols: 1 },
        ],
        ops: vec![
            op(
                SubOp::ScalarMul {
                    scale: embed_scale(),
                },
                vec![Ext(0)],
            ),
            op(SubOp::RmsNormUnit { eps: EPS }, vec![Op(0)]),
            op(SubOp::ScalarWeightMul, vec![Op(1), Ext(1)]),
            op(SubOp::TanhSoftCap { cap: CAP }, vec![Op(2)]),
        ],
        result: 3,
    }
}

/// [`gemma_rows_input`] as the front end binds it, with `x` rebound to the embedded hidden.
pub fn gemma_rows(m: u32) -> LoweredDecode {
    with_hidden(gemma_rows_input(m))
}
