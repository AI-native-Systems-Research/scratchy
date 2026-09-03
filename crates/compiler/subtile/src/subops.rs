// SPDX-License-Identifier: Apache-2.0
//! THE registry for [`crate::subtile_ir::SubOp`] — the shared IR's own
//! op vocabulary.
//!
//! `ops.rs` has done this for `LoweredOp` since `90f18ac53`: one
//! X-macro row per op, and every consumer generates its site from the
//! same rows instead of hand-writing a parallel match. `SubOp` — the
//! form BOTH backends' emitters consume — had no such registry, so
//! its facts were restated per consumer.
//!
//! A probe that added one throwaway `SubOp` and read the compiler's
//! own errors measured the cost: five compiler-forced sites for a new
//! `SubOp`, two of them in this crate. One of those
//! two is the arity validator below, which is pure declared data of
//! exactly the kind `ops.rs` already derives. Moving it here removes
//! that site.
//!
//! Row grammar (bracketed so each stays one token tree):
//! `[<pattern>] arity = (|n| <bool over n>);`
//! The pattern is a real match pattern, so nested enums
//! (`Elementwise(EwKind::Silu)`) are rows in their own right rather
//! than a special case the consumer has to unpack.

/// X-macro over every `SubOp`. Callbacks receive the full row list.
#[macro_export]
macro_rules! for_each_subtile_op {
    ($cb:ident) => {
        $cb! {
            // 2 = [act, W] fp16; 3 = [act, W, w_scale] fp8 W8A8.
            [SubOp::MatmulTile { .. }] arity = (|n| n == 2 || n == 3);
            [SubOp::SumReduce] arity = (|n| n >= 1);
            [SubOp::Reshape] arity = (|n| n == 1);
            [SubOp::Elementwise(
                EwKind::Silu | EwKind::Gelu | EwKind::QuickGelu | EwKind::GeluErf
            )] arity = (|n| n == 1);
            [SubOp::Elementwise(EwKind::Mul | EwKind::Add | EwKind::Sub)] arity = (|n| n == 2);
            [SubOp::ScalarMul { .. }] arity = (|n| n == 1);
            [SubOp::SiluMul] arity = (|n| n == 2);
            [SubOp::RmsNorm { .. }] arity = (|n| n == 2);
            [SubOp::RmsNormReduce { .. }] arity = (|n| n == 1);
            [SubOp::RmsNormApply { .. }] arity = (|n| n == 3);
            [SubOp::RopeRotate { .. }] arity = (|n| n == 3);
            // E.12: RopeAppend takes [K, cos, sin, V, K_cache, V_cache]
            // — the caches are per-layer PrefixK / PrefixV sources used
            // as TMA-store destinations for the new decode token's K/V.
            [SubOp::RopeAppend { .. }] arity = (|n| n == 6);
            [SubOp::AttnDecode { .. }] arity = (|n| n >= 3 && n % 2 == 1);

            // The rest of the arch vocabulary. Arities are transcribed from the
            // `LoweredOp` doc comments that state each op's operand list — the
            // same contract `validate_arity` checks on the wavefront side, so a
            // disagreement between the two lists is a disagreement about ONE
            // documented fact rather than a new one invented here.
            [SubOp::TanhSoftCap] arity = (|n| n == 1);
            [SubOp::RmsNormUnit { .. }] arity = (|n| n == 1);
            [SubOp::ScalarWeightMul] arity = (|n| n == 2);
            [SubOp::GateSplit { .. }] arity = (|n| n == 1);
            [SubOp::GateApply] arity = (|n| n == 2);
            [SubOp::GateScale] arity = (|n| n == 3);
            // Host-staged: the buffer is delivered by the runtime, not by an operand.
            [SubOp::LoadPixels { .. }] arity = (|n| n == 0);
            [SubOp::LoadPosEmbeds { .. }] arity = (|n| n == 0);
            // The index table is a runtime input, not an operand — hence ONE.
            [SubOp::EmbeddingGather { .. }] arity = (|n| n == 1);
            [SubOp::VisionRope] arity = (|n| n == 2);
            [SubOp::VarlenAttention { .. }] arity = (|n| n == 3);
            [SubOp::EncoderAttn { .. }] arity = (|n| n == 3);
            [SubOp::GatedDeltaNet] arity = (|n| n == 5);
            [SubOp::GemmaMoe { .. }] arity = (|n| n == 4);
            [SubOp::Moe { .. }] arity = (|n| n == 2);
            [SubOp::Mean] arity = (|n| n == 1);
        }
    };
}

macro_rules! __derive_subop_arity {
    ($( [$pat:pat] arity = (|$an:ident| $arity:expr); )*) => {
        /// Whether `n_operands` is legal for `op` — derived from the
        /// registry, used by the IR validator.
        ///
        /// The generated match is exhaustive over `SubOp`, so adding a
        /// variant without a row fails HERE at compile time.
        pub fn subop_arity_ok<F: crate::subtile_ir::RopeForm>(
            op: &crate::subtile_ir::SubOp<F>,
            n_operands: usize,
        ) -> bool {
            use crate::subtile_ir::{EwKind, SubOp};
            match op {
                $( $pat => { let $an = n_operands; $arity } )*
            }
        }
    };
}
for_each_subtile_op!(__derive_subop_arity);
