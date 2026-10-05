//! THE op registry — the single declaration site for the facts of THE op vocabulary,
//! [`crate::subtile_ir::SubOp`].
//!
//! Every fact a consumer needs about an op beyond its prose docs — its kind, its name, its
//! operand arity and its output-width rule — is declared ONCE here as an X-macro row, for the op
//! as the front end states it and as a lowered node performs it alike (the two are one enum, at
//! two [`crate::subtile_ir::OpStage`]s). Consumers (the IR validator, `lower_region`'s output
//! widths, the fixtures, per-target fusion tables keyed on [`SubOpKind`], refusal messages) each
//! invoke [`crate::for_each_subop!`] with their own callback and generate their site from the
//! same rows.
//!
//! Adding an op = adding its variant (with its prose docs) and ONE row here. Every generated
//! match is exhaustive over the enum, so a variant without a row is a compile error at the
//! generated site, not a silently-unhandled case nine files away. This exists because a previous
//! attempt at the metal/spyre unification wrote each of these facts by hand at every site: one op
//! touched nine files, and every slot bug of that branch traced to one site disagreeing with
//! another.
//!
//! Row grammar: `Kind [<pattern>] arity = (|n| <bool over n>), cols = <class>;`
//! - `Kind` names the op in [`SubOpKind`] and in [`crate::subtile_ir::SubOp::name`]; for a
//!   variant with fields it is the variant's own name.
//! - The pattern is a real match pattern, so an elementwise kind (`Elementwise(EwKind::Silu)`) is
//!   a row in its own right rather than a special case the consumer has to unpack.
//! - `arity` is a closure-form predicate over the operand count (the binder is declared IN the
//!   row so macro hygiene ties it to the body).
//! - `cols` is one of (bracketed so it stays one macro token tree):
//!   - `[in0]` — shape-preserving: output width = `inputs[0]` width;
//!   - `[in1]` — output width = `inputs[1]` width (GDN's z operand);
//!   - `[one]` — one column per row (a row reduction);
//!   - `[field f]` — output width is the op's own field `f`;
//!   - `[nz f]` — the op's own non-zero count field `f` (`.get()`);
//!   - `[pairs in0 k]` / `[pairs f k]` — `(token, expert)` pair rows laid out `[m, k·w]`, `w` the
//!     width of operand 0 or of the op's field `f`, `k` the op's top-k field;
//!   - `[q_width g]` — the query width of the op's head geometry field `g`.
//!
//! Rows after `@expansion` are the ops one construct expands to (an arch-level MoE block, or the KV
//! codec or sampled rows a target's facts insert); they also generate `expansion_ops!()`, the
//! pattern a target without them refuses by.

/// X-macro over every op of [`crate::subtile_ir::SubOp`]. Callbacks receive the full row list.
#[macro_export]
macro_rules! for_each_subop {
    ($cb:ident) => {
        $cb! {
            // 2 = [act, W] fp16 / affine; 3 = [act, W, w_scale] fp8 W8A8.
            MatmulTile [SubOp::MatmulTile { .. }] arity = (|n| n == 2 || n == 3), cols = [field n];
            SumReduce [SubOp::SumReduce { .. }] arity = (|n| n >= 1), cols = [in0];
            Reshape [SubOp::Reshape { .. }] arity = (|n| n == 1), cols = [field cols];
            Silu [SubOp::Elementwise(EwKind::Silu)] arity = (|n| n == 1), cols = [in0];
            Gelu [SubOp::Elementwise(EwKind::Gelu)] arity = (|n| n == 1), cols = [in0];
            QuickGelu [SubOp::Elementwise(EwKind::QuickGelu)] arity = (|n| n == 1), cols = [in0];
            GeluErf [SubOp::Elementwise(EwKind::GeluErf)] arity = (|n| n == 1), cols = [in0];
            Mul [SubOp::Elementwise(EwKind::Mul)] arity = (|n| n == 2), cols = [in0];
            Add [SubOp::Elementwise(EwKind::Add)] arity = (|n| n == 2), cols = [in0];
            Sub [SubOp::Elementwise(EwKind::Sub)] arity = (|n| n == 2), cols = [in0];
            BiasAdd [SubOp::Elementwise(EwKind::BiasAdd)] arity = (|n| n == 2), cols = [in0];
            ScalarMul [SubOp::ScalarMul { .. }] arity = (|n| n == 1), cols = [in0];
            SiluMul [SubOp::SiluMul] arity = (|n| n == 2), cols = [in0];
            RmsNorm [SubOp::RmsNorm { .. }] arity = (|n| n == 2), cols = [in0];
            RmsNormReduce [SubOp::RmsNormReduce { .. }] arity = (|n| n == 1), cols = [one];
            RmsNormApply [SubOp::RmsNormApply { .. }] arity = (|n| n == 3), cols = [in0];
            RopeRotate [SubOp::RopeRotate { .. }] arity = (|n| n == 3), cols = [in0];
            // E.12: RopeAppend takes [K, cos, sin, V, K_cache, V_cache]
            // — the caches are per-layer PrefixK / PrefixV sources used
            // as TMA-store destinations for the new decode token's K/V.
            RopeAppend [SubOp::RopeAppend { .. }] arity = (|n| n == 6), cols = [in0];
            // `[Q, (K_seg, V_seg)...]`, optionally with a trailing
            // `[num_heads]` sinks weight source (gpt-oss attention
            // sinks — a 6th input read as a weight, not a K/V pair).
            AttnDecode [SubOp::AttnDecode { .. }] arity = (|n| (n >= 3 && n % 2 == 1) || n == 6),
                cols = [q_width geom];
            TanhSoftCap [SubOp::TanhSoftCap] arity = (|n| n == 1), cols = [in0];
            RmsNormUnit [SubOp::RmsNormUnit { .. }] arity = (|n| n == 1), cols = [in0];
            ScalarWeightMul [SubOp::ScalarWeightMul] arity = (|n| n == 2), cols = [in0];
            GateSplit [SubOp::GateSplit { .. }] arity = (|n| n == 1), cols = [field half_cols];
            GateApply [SubOp::GateApply] arity = (|n| n == 2), cols = [in0];
            Concat [SubOp::Concat { .. }] arity = (|n| n == 2), cols = [field cols];
            GateScale [SubOp::GateScale] arity = (|n| n == 3), cols = [in0];
            // Host-staged: the buffer is delivered by the runtime, not by an operand.
            LoadRows [SubOp::LoadRows { .. }] arity = (|n| n == 0), cols = [field width];
            // The index table is a runtime input, not an operand — hence ONE.
            EmbeddingGather [SubOp::EmbeddingGather { .. }] arity = (|n| n == 1), cols = [in0];
            VisionRope [SubOp::VisionRope] arity = (|n| n == 2), cols = [in0];
            VarlenAttention [SubOp::VarlenAttention { .. }] arity = (|n| n == 3), cols = [in0];
            EncoderAttn [SubOp::EncoderAttn { .. }] arity = (|n| n == 3), cols = [q_width geom];
            GatedDeltaNet [SubOp::GatedDeltaNet] arity = (|n| n == 5), cols = [in1];
            Mean [SubOp::Mean] arity = (|n| n == 1), cols = [one];
            // The ops one construct expands to (a MoE block, a KV codec's steps, sampled rows) —
            // also generating `expansion_ops!()`, the pattern a target without them refuses by.
            @expansion
            KvEncode [SubOp::KvEncode { .. }] arity = (|n| n == 2 || n == 3), cols = [one];
            KvStage [SubOp::KvStage { .. }] arity = (|n| n == 1), cols = [one];
            RotateRows [SubOp::RotateRows { .. }] arity = (|n| n == 1), cols = [in0];
            // `[q, out, packed_k, packed_v]`, optionally with the attention's
            // trailing `[num_heads]` sinks weight source (gpt-oss attention
            // sinks — the 5th input read as a weight, as the dense form's 6th).
            AttnPackedKv [SubOp::AttnPackedKv] arity = (|n| n == 4 || n == 5), cols = [in1];
            RouterNorm [SubOp::RouterNorm { .. }] arity = (|n| n == 2), cols = [in0];
            RouterLogits [SubOp::RouterLogits { .. }] arity = (|n| n == 2), cols = [nz experts];
            RouteSoftmax [SubOp::RouteSoftmax] arity = (|n| n == 1), cols = [in0];
            RouteSigmoidBias [SubOp::RouteSigmoidBias { .. }] arity = (|n| n == 2), cols = [in0];
            RouteBias [SubOp::RouteBias { .. }] arity = (|n| n == 2), cols = [in0];
            RouteArgsort [SubOp::RouteArgsort] arity = (|n| n == 1), cols = [in0];
            RouteTopK [SubOp::RouteTopK { .. }] arity = (|n| n == 1), cols = [nz k];
            RouteGatherScores [SubOp::RouteGatherScores] arity = (|n| n == 2), cols = [in1];
            RouteScale [SubOp::RouteScale { .. }] arity = (|n| n == 1), cols = [in0];
            RouteRenorm [SubOp::RouteRenorm] arity = (|n| n == 1), cols = [in0];
            RouteExpertScale [SubOp::RouteExpertScale { .. }] arity = (|n| n == 3), cols = [in0];
            ExpertSort [SubOp::ExpertSort { .. }] arity = (|n| n == 2), cols = [pairs in0 k];
            ExpertMatmul [SubOp::ExpertMatmul { .. }] arity = (|n| n == 3), cols = [pairs n k];
            ExpertGatedAct [SubOp::ExpertGatedAct { .. }] arity = (|n| n == 2), cols = [in0];
            ExpertUnsort [SubOp::ExpertUnsort] arity = (|n| n == 2), cols = [in0];
            ExpertCombine [SubOp::ExpertCombine { .. }] arity = (|n| n == 2), cols = [field hidden];
            SampleRowsGather [SubOp::SampleRowsGather] arity = (|n| n == 1), cols = [in0];
            SampleRowsScatter [SubOp::SampleRowsScatter] arity = (|n| n == 1), cols = [in0];
            // `[rows, over, weight]`, or `[.., weight, w_scale]` for an fp8 matmul.
            AllRowsMatmul [SubOp::AllRowsMatmul] arity = (|n| n == 3 || n == 4), cols = [in1];
        }
    };
}

// ── Derived: output width ───────────────────────────────────────────
// The arm's pattern and body come from one row's `cols` class; a field-reading class binds the
// row's own field name in the pattern, so no arm needs a nested match.
macro_rules! __out_cols_pat {
    ($kind:ident, $pat:pat, [field $f:ident]) => {
        $crate::subtile_ir::SubOp::$kind { $f, .. }
    };
    ($kind:ident, $pat:pat, [q_width $g:ident]) => {
        $crate::subtile_ir::SubOp::$kind { $g, .. }
    };
    ($kind:ident, $pat:pat, [nz $f:ident]) => {
        $crate::subtile_ir::SubOp::$kind { $f, .. }
    };
    ($kind:ident, $pat:pat, [pairs in0 $k:ident]) => {
        $crate::subtile_ir::SubOp::$kind { $k, .. }
    };
    ($kind:ident, $pat:pat, [pairs $f:ident $k:ident]) => {
        $crate::subtile_ir::SubOp::$kind { $f, $k, .. }
    };
    ($kind:ident, $pat:pat, $cols:tt) => {
        $pat
    };
}
macro_rules! __out_cols_body {
    ($w:ident, [in0]) => {
        $w(0)
    };
    ($w:ident, [in1]) => {
        $w(1)
    };
    ($w:ident, [one]) => {
        1u32
    };
    ($w:ident, [field $f:ident]) => {
        *$f
    };
    ($w:ident, [q_width $g:ident]) => {
        $g.q_width()
    };
    ($w:ident, [nz $f:ident]) => {
        $f.get()
    };
    ($w:ident, [pairs in0 $k:ident]) => {
        $w(0) * $k.get()
    };
    ($w:ident, [pairs $f:ident $k:ident]) => {
        *$f * $k.get()
    };
}

macro_rules! __derive_registry {
    ($( $kind:ident [$pat:pat] arity = (|$an:ident| $arity:expr), cols = $cols:tt; )*
     @expansion $( $ek:ident [$ep:pat] arity = (|$ean:ident| $ea:expr), cols = $ec:tt; )*) => {
        __derive_registry! {
            $( $kind [$pat] arity = (|$an| $arity), cols = $cols; )*
            $( $ek [$ep] arity = (|$ean| $ea), cols = $ec; )*
        }

        /// Every op an arch-level construct expands to, as ONE match pattern (use it with
        /// `SubOp` in scope): the arm a target that realizes none of them refuses them by.
        #[macro_export]
        macro_rules! expansion_ops {
            () => {
                $( $ep )|*
            };
        }
    };
    ($( $kind:ident [$pat:pat] arity = (|$an:ident| $arity:expr), cols = $cols:tt; )*) => {
        /// An op without its fields: how a declared target table names an op. Derived from the
        /// registry, so no table keeps a parallel list.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum SubOpKind {
            $( $kind, )*
        }

        impl<F: $crate::subtile_ir::RopeForm, S: $crate::subtile_ir::OpStage>
            $crate::subtile_ir::SubOp<F, S>
        {
            /// The op's kind.
            pub fn kind(&self) -> SubOpKind {
                use $crate::subtile_ir::{EwKind, SubOp};
                match self {
                    $( $pat => SubOpKind::$kind, )*
                }
            }

            /// The op's registry name — for dumps and refusal messages, so no consumer
            /// hand-writes a parallel string table.
            pub fn name(&self) -> &'static str {
                use $crate::subtile_ir::{EwKind, SubOp};
                match self {
                    $( $pat => stringify!($kind), )*
                }
            }

            /// Whether `n_operands` is legal for the op — used by the IR validator.
            pub fn arity_ok(&self, n_operands: usize) -> bool {
                use $crate::subtile_ir::{EwKind, SubOp};
                match self {
                    $( $pat => { let $an = n_operands; $arity } )*
                }
            }

            /// The op's output column count. `operand_cols(k)` yields the width of operand
            /// `k`; a row declaring `[in0]`/`[in1]` calls it, the other classes never do.
            /// Taking a LOOKUP rather than just `in0_cols` is what lets this serve every
            /// consumer: GDN's width comes from operand 1.
            pub fn out_cols(&self, operand_cols: impl Fn(usize) -> u32) -> u32 {
                use $crate::subtile_ir::{EwKind, SubOp};
                match self {
                    $( __out_cols_pat!($kind, $pat, $cols) =>
                        __out_cols_body!(operand_cols, $cols), )*
                }
            }
        }
    };
}
for_each_subop!(__derive_registry);
