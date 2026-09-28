// SPDX-License-Identifier: Apache-2.0
//! Which bindings each kernel WRITES, as declared data.
//!
//! The tape already names every buffer a command binds; what it does not say is which of them
//! the kernel modifies. The megakernel's dataflow analysis needs exactly that, and nothing else,
//! from metal.
//!
//! A binding is written when the kernel stores to the buffer it names OR through the addresses
//! that buffer holds: the paged KV cache is bound as `const uint64_t*` page tables, and the
//! kernels that append to it (`rope_append*`, `tq_compress_paged`'s write-back) store into the
//! pages. Those must be declared here — no reflection can see them.
//!
//! `None` means not declared yet. The plan never analyses such a command: it ends a megakernel
//! run there and the command is dispatched on its own, so an undeclared kernel costs a boundary,
//! never correctness.
//!
//! The match is exhaustive: a new `KernelId` does not compile until its author takes a position.

use crate::tape::lowered::KernelId;

pub fn kernel_writes(k: KernelId) -> Option<&'static [u8]> {
    use KernelId as K;
    match k {
        // out @ 0
        K::RmsNorm
        | K::RmsNormUnit
        | K::ScalarWeightMul
        | K::SiluMul
        | K::GeluMul
        | K::Gemm
        | K::MmEmbedSplice
        | K::ScalarMul
        | K::AttentionViaCache
        | K::AttentionViaCacheTq
        // in place @ 0
        | K::Add
        | K::MoePerExpertScale
        | K::GatherLastToken
        | K::ScatterFirstToLastRow => Some(&[0]),
        // normed out @ 0, residual in place @ 1
        K::FusedAddRmsNorm => Some(&[0, 1]),
        // out @ 1
        K::ArgPartitionTopK | K::SliceTrailingColsU32 | K::Softmax | K::TanhSoftCap => Some(&[1]),
        // out @ 2
        K::TakeAlongAxis | K::MoeWeightedSum => Some(&[2]),
        // y @ 4
        K::AffineQmvQuad | K::AffineQmvFast | K::AffineQmv | K::AffineEmbed => Some(&[4]),
        // y @ 5
        K::AffineGatherQmvFast | K::AffineGatherQmv => Some(&[5]),
        // q in place @ 0; K/V pages through the page tables @ 6, 7
        K::RopeAppendNormed => Some(&[0, 6, 7]),
        // q, k, v in place @ 0..2; K/V pages through the page tables @ 6, 7
        K::RopeAppend => Some(&[0, 1, 2, 6, 7]),
        // raw pages written back through the chunk table @ 0; packed store @ 5; norms @ 6
        K::TqQuantizeToPacked => Some(&[0, 5, 6]),
        K::Embed
        | K::NormAddScalarMul
        | K::FusedGateUpSiluMul
        | K::FusedQkvRopeCache
        | K::FusedAffineQkvRopeCache
        | K::AttentionPrefillSdpaPaged
        | K::RopeOnceNax
        | K::RopeOnceSteel
        | K::RopeOnceGqaShared
        | K::AttnGatherKRope
        | K::AttnGatherVCopyT
        | K::AttnQConvert
        | K::AttnOConvert
        | K::AttnCausalSoftmax
        | K::AttnGemmQk
        | K::AttnGemmPv
        | K::BiasAdd
        | K::Reshape
        | K::AffineQmmT
        | K::AffineGatherQmmT
        | K::AffineGatherQmmTNax
        | K::AffineQmmTSplitK
        | K::AffineQmmTNax
        | K::AffineQmmSmallM
        | K::AffineW4a8Quant
        | K::AffineQmmW4a8
        | K::AffineGatherW4a8Quant
        | K::AffineGatherQmmW4a8
        | K::Nvfp4Qmv
        | K::Nvfp4QmmT
        | K::Nvfp4QmmTNax
        | K::GateApply
        | K::GateScale
        | K::GateSplit
        | K::GatedDeltaNet
        | K::SplitKReduceSum
        | K::SynthPreAttn
        | K::SynthMlpPreDown
        | K::SynthGateUpSiluMul
        | K::MoeGroupOffsets
        | K::MoeGroupInit
        | K::MoeGroupScatter
        | K::MoeGroupScatterQ8
        | K::MoeGroupGather
        | K::VisionLayerNorm
        | K::VisionRope
        | K::VisionVarlenAttn
        | K::EmbeddingGather
        | K::AvgPool2d
        | K::VisionGelu
        | K::VisionLoadPixels
        | K::TqStageRotated
        | K::TqRotateRows => None,
    }
}
