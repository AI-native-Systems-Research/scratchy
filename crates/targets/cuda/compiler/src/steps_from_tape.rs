// SPDX-License-Identifier: Apache-2.0
//! CUDA'S OPCODE LOWERING: one tape step's op to one [`CudaStep`].
//!
//! The match lists every `SubOp` variant, with no wildcard, so a new `SubOp` fails to compile here
//! (E0004) as it does in the colour rule. No opcode lowers yet: every variant sits in the one
//! refusing arm, and each opcode issue moves exactly one variant out of it into an arm of its own.

use scratchy_subtile::subtile_ir::SubOp;
use scratchy_target_cuda::step::CudaStep;

use crate::canonical::CanonicalRefusal;

/// Lower the op of the `step`-th `Compute` on the tape to its CUDA step.
pub fn lower_step(step: usize, op: &SubOp) -> Result<CudaStep, CanonicalRefusal> {
    use SubOp as L;
    match op {
        L::MatmulTile { .. }
        | L::SumReduce { .. }
        | L::Reshape { .. }
        | L::Elementwise(_)
        | L::SiluMul
        | L::ScalarMul { .. }
        | L::RmsNorm { .. }
        | L::RmsNormReduce { .. }
        | L::RmsNormApply { .. }
        | L::RopeRotate { .. }
        | L::RopeAppend { .. }
        | L::KvAppend { .. }
        | L::AttnDecode { .. }
        | L::TanhSoftCap { .. }
        | L::RmsNormUnit { .. }
        | L::ScalarWeightMul
        | L::GateSplit { .. }
        | L::GateApply
        | L::Concat { .. }
        | L::GateScale
        | L::LoadRows { .. }
        | L::EmbeddingGather { .. }
        | L::VisionRope
        | L::VarlenAttention { .. }
        | L::EncoderAttn { .. }
        | L::GatedDeltaNet
        | L::Mamba2 { .. }
        | L::Mean
        | L::RouterNorm { .. }
        | L::RouterLogits { .. }
        | L::RouteSoftmax
        | L::RouteSigmoidBias { .. }
        | L::RouteBias { .. }
        | L::RouteArgsort
        | L::RouteTopK { .. }
        | L::RouteGatherScores
        | L::RouteScale { .. }
        | L::RouteRenorm
        | L::RouteExpertScale { .. }
        | L::ExpertSort { .. }
        | L::ExpertMatmul { .. }
        | L::ExpertGatedAct { .. }
        | L::ExpertAct { .. }
        | L::ExpertUnsort
        | L::ExpertCombine { .. }
        | L::KvEncode { .. }
        | L::KvStage { .. }
        | L::RotateRows { .. }
        | L::AttnPackedKv
        | L::SampleRowsGather
        | L::SampleRowsScatter
        | L::AllRowsMatmul => Err(CanonicalRefusal::NotLowered {
            step,
            op: op.name(),
        }),
    }
}
