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
//! - `rows` is the same shape of fact for the output ROW count — which op row law a step obeys:
//!   - `[in0]` — shape-preserving: output rows = `inputs[0]`'s REGISTERED rows. The token count
//!     `m` and the true row count diverge exactly once a per-head `Reshape` view enters the chain
//!     (`[m, heads·hd] → [m·heads, hd]` multiplies rows through every consumer until the
//!     flatten-back divides them away), so the row scale is carried by the TENSORS and inherited
//!     through operand 0 — never re-read from `m`, which the bridge stamps with the token count
//!     for every op alike.
//!   - `[m]` — the op's output rows ARE the token count: the host-staged loads (no operand to
//!     inherit), the KV codec's packed stores (operand 0 is a cache/prefix source whose rows are
//!     capacity, not tokens), and the MoE expansion ops (their pair rows are the target's to
//!     derive; the registry's own `[pairs]` cols class lays them out `[m, k·w]`).
//!   - Reshape needs no `rows` class: its rows follow from its own payload relative to the token
//!     count (`m · mult / div`), the one op the front end states a scale for.
//!
//!   - `[in k]` — the op's operand `k` is its GEMM WEIGHT (the operand a target re-tiles/stages
//!     as a weight buffer): MatmulTile/RouterLogits at 1, the gathered pair gemms at 2;
//!   - `[none]` — the op has no gemm-weight operand (a norm's gain, a router bias, an attention
//!     sink and every non-gemm weight reader are NOT gemm weights — they stage by their own
//!     paths, so classifying them here would make the bake refuse them).
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
            MatmulTile [SubOp::MatmulTile { .. }] arity = (|n| n == 2 || n == 3),
                cols = [field n], rows = [in0], gemm_w = [in 1];
            SumReduce [SubOp::SumReduce { .. }] arity = (|n| n >= 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            // The one op with its own row scale: `m · mult / div`, stated by its payload.
            Reshape [SubOp::Reshape { .. }] arity = (|n| n == 1), cols = [field cols],
                rows = [scale rows], gemm_w = [none];
            Silu [SubOp::Elementwise(EwKind::Silu)] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            Gelu [SubOp::Elementwise(EwKind::Gelu)] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            QuickGelu [SubOp::Elementwise(EwKind::QuickGelu)] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            GeluErf [SubOp::Elementwise(EwKind::GeluErf)] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            Relu2 [SubOp::Elementwise(EwKind::Relu2)] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            Mul [SubOp::Elementwise(EwKind::Mul)] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            Add [SubOp::Elementwise(EwKind::Add)] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            Sub [SubOp::Elementwise(EwKind::Sub)] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            BiasAdd [SubOp::Elementwise(EwKind::BiasAdd)] arity = (|n| n == 2), cols = [in0],
                rows = [in0], gemm_w = [none];
            ScalarMul [SubOp::ScalarMul { .. }] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            SiluMul [SubOp::SiluMul] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            RmsNorm [SubOp::RmsNorm { .. }] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            RmsNormReduce [SubOp::RmsNormReduce { .. }] arity = (|n| n == 1), cols = [one],
                rows = [in0], gemm_w = [none];
            RmsNormApply [SubOp::RmsNormApply { .. }] arity = (|n| n == 3), cols = [in0],
                rows = [in0], gemm_w = [none];
            RopeRotate [SubOp::RopeRotate { .. }] arity = (|n| n == 3), cols = [in0], rows = [in0],
                gemm_w = [none];
            // E.12: RopeAppend takes [K, cos, sin, V, K_cache, V_cache]
            // — the caches are per-layer PrefixK / PrefixV sources used
            // as TMA-store destinations for the new decode token's K/V.
            RopeAppend [SubOp::RopeAppend { .. }] arity = (|n| n == 6), cols = [in0], rows = [in0],
                gemm_w = [none];
            // `[K, V, K_cache, V_cache]`: the un-rotated writer.
            KvAppend [SubOp::KvAppend { .. }] arity = (|n| n == 4), cols = [in0], rows = [in0],
                gemm_w = [none];
            // `[Q, (K_seg, V_seg)...]`, optionally with a trailing
            // `[num_heads]` sinks weight source (gpt-oss attention
            // sinks — a 6th input read as a weight, not a K/V pair).
            AttnDecode [SubOp::AttnDecode { .. }] arity = (|n| (n >= 3 && n % 2 == 1) || n == 6),
                cols = [q_width geom], rows = [in0], gemm_w = [none];
            TanhSoftCap [SubOp::TanhSoftCap { .. }] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            RmsNormUnit [SubOp::RmsNormUnit { .. }] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            ScalarWeightMul [SubOp::ScalarWeightMul] arity = (|n| n == 2), cols = [in0],
                rows = [in0], gemm_w = [none];
            GateSplit [SubOp::GateSplit { .. }] arity = (|n| n == 1), cols = [field half_cols],
                rows = [in0], gemm_w = [none];
            GateApply [SubOp::GateApply] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            Concat [SubOp::Concat { .. }] arity = (|n| n == 2), cols = [field cols],
                rows = [in0], gemm_w = [none];
            GateScale [SubOp::GateScale] arity = (|n| n == 3), cols = [in0], rows = [in0],
                gemm_w = [none];
            // Host-staged: the buffer is delivered by the runtime, not by an operand — so there is
            // no operand to inherit rows from.
            LoadRows [SubOp::LoadRows { .. }] arity = (|n| n == 0), cols = [field width],
                rows = [m], gemm_w = [none];
            // The index table is a runtime input, not an operand — hence ONE.
            EmbeddingGather [SubOp::EmbeddingGather { .. }] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            VisionRope [SubOp::VisionRope] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            VarlenAttention [SubOp::VarlenAttention { .. }] arity = (|n| n == 3), cols = [in0],
                rows = [in0], gemm_w = [none];
            EncoderAttn [SubOp::EncoderAttn { .. }] arity = (|n| n == 3), cols = [q_width geom],
                rows = [in0], gemm_w = [none];
            GatedDeltaNet [SubOp::GatedDeltaNet] arity = (|n| n == 5), cols = [in1], rows = [in0],
                gemm_w = [none];
            Mamba2 [SubOp::Mamba2 { .. }] arity = (|n| n == 2), cols = [field inner],
                rows = [in0], gemm_w = [none];
            Mean [SubOp::Mean] arity = (|n| n == 1), cols = [one], rows = [in0], gemm_w = [none];
            // The greedy token pick: one `[1, PAD_V]` logits row plus the seven staged
            // tables (functions of the vocab alone — the seventh carries the pad law's
            // offset row), out ONE `[1, 64]` idx stick — the
            // output width is the device's own stick, never a shape read (the whole row
            // is one tile in, one stick out, every lane equal).
            ArgMax [SubOp::ArgMax { .. }] arity = (|n| n == 8), cols = [stick],
                rows = [in0], gemm_w = [none];
            // The ops one construct expands to (a MoE block, a KV codec's steps, sampled rows) —
            // also generating `expansion_ops!()`, the pattern a target without them refuses by.
            @expansion
            // Operand 0 is the cache/prefix source the writer filled — its rows are capacity, not
            // this step's tokens. An un-rotated writer (`kv_append`, Nemotron-H) has no rotary
            // row, so its encode reads the cache alone.
            KvEncode [SubOp::KvEncode { .. }] arity = (|n| (1..=3).contains(&n)), cols = [one],
                rows = [m], gemm_w = [none];
            KvStage [SubOp::KvStage { .. }] arity = (|n| n == 1), cols = [one], rows = [m],
                gemm_w = [none];
            RotateRows [SubOp::RotateRows { .. }] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            // `[q, out, packed_k, packed_v]`, optionally with the attention's
            // trailing `[num_heads]` sinks weight source (gpt-oss attention
            // sinks — the 5th input read as a weight, as the dense form's 6th).
            AttnPackedKv [SubOp::AttnPackedKv] arity = (|n| n == 4 || n == 5), cols = [in1],
                rows = [in0], gemm_w = [none];
            RouterNorm [SubOp::RouterNorm { .. }] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            RouterLogits [SubOp::RouterLogits { .. }] arity = (|n| n == 2), cols = [nz experts],
                rows = [in0], gemm_w = [in 1];
            RouteSoftmax [SubOp::RouteSoftmax] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            RouteSigmoidBias [SubOp::RouteSigmoidBias { .. }] arity = (|n| n == 2), cols = [in0],
                rows = [in0], gemm_w = [none];
            RouteBias [SubOp::RouteBias { .. }] arity = (|n| n == 2), cols = [in0], rows = [in0],
                gemm_w = [none];
            RouteArgsort [SubOp::RouteArgsort] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            RouteTopK [SubOp::RouteTopK { .. }] arity = (|n| n == 1), cols = [nz k], rows = [in0],
                gemm_w = [none];
            RouteGatherScores [SubOp::RouteGatherScores] arity = (|n| n == 2), cols = [in1],
                rows = [in0], gemm_w = [none];
            RouteScale [SubOp::RouteScale { .. }] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            RouteRenorm [SubOp::RouteRenorm] arity = (|n| n == 1), cols = [in0], rows = [in0],
                gemm_w = [none];
            RouteExpertScale [SubOp::RouteExpertScale { .. }] arity = (|n| n == 3), cols = [in0],
                rows = [in0], gemm_w = [none];
            // The MoE pair-row ops: their `[pairs]` cols class lays the rows out `[m, k·w]`, so the
            // registered row count is the token count and the pair structure is the target's to
            // derive. ExpertUnsort restores TOKEN order from pair rows.
            ExpertSort [SubOp::ExpertSort { .. }] arity = (|n| n == 2), cols = [pairs in0 k],
                rows = [m], gemm_w = [none];
            ExpertMatmul [SubOp::ExpertMatmul { .. }] arity = (|n| n == 3 || n == 4), cols = [pairs n k],
                rows = [m], gemm_w = [in 2];
            ExpertGatedAct [SubOp::ExpertGatedAct { .. }] arity = (|n| n == 2), cols = [in0],
                rows = [m], gemm_w = [none];
            ExpertAct [SubOp::ExpertAct { .. }] arity = (|n| n == 1), cols = [in0],
                rows = [m], gemm_w = [none];
            ExpertUnsort [SubOp::ExpertUnsort] arity = (|n| n == 2), cols = [in0], rows = [m],
                gemm_w = [none];
            ExpertCombine [SubOp::ExpertCombine { .. }] arity = (|n| n == 2), cols = [field hidden],
                rows = [in0], gemm_w = [none];
            SampleRowsGather [SubOp::SampleRowsGather] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            SampleRowsScatter [SubOp::SampleRowsScatter] arity = (|n| n == 1), cols = [in0],
                rows = [in0], gemm_w = [none];
            // `[rows, over, weight]`, or `[.., weight, w_scale]` for an fp8 matmul.
            AllRowsMatmul [SubOp::AllRowsMatmul] arity = (|n| n == 3 || n == 4), cols = [in1],
                rows = [in0], gemm_w = [in 2];
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
    ($w:ident, [stick]) => {
        crate::subtile_ir::ARGMAX_BLOCK
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

// ── Derived: output rows ────────────────────────────────────────────
// The same shape of derivation as `out_cols`, over the `rows` class. The two classes:
// - `[in0]` — the op preserves its operand 0's row count (the row scale a per-head Reshape
//   view introduced is carried by the tensors and inherited through operand 0);
// - `[m]` — the op's output rows are the TOKEN count `m`;
// - `[scale f]` — Reshape's own `m · mult / div`, the one op the front end states a scale for.
// A field-reading class binds the row's own field name in the pattern.
macro_rules! __out_rows_pat {
    ($kind:ident, $pat:pat, [scale $f:ident]) => {
        $crate::subtile_ir::SubOp::$kind { $f, .. }
    };
    ($kind:ident, $pat:pat, $rows:tt) => {
        $pat
    };
}
macro_rules! __out_rows_body {
    ($w:ident, $m:ident, [in0]) => {
        $w(0)
    };
    ($w:ident, $m:ident, [m]) => {
        $m
    };
    ($w:ident, $m:ident, [scale $f:ident]) => {{
        let r = u64::from($m) * u64::from($f.mult());
        let d = u64::from($f.div());
        assert!(
            r % d == 0,
            "a reshape over [{} rows] with scale {:?} is not a whole row count — the \
             bridge only mints whole multiples",
            $m,
            $f
        );
        (r / d) as u32
    }};
}

// ── Derived: the gemm-weight operand ───────────────────────────────
// One class pair: `[in k]` names operand k as the op's GEMM WEIGHT, `[none]` says it has none.
// The arm's pattern is the row's own `$pat` — no field of the op keys the fact.
macro_rules! __gemm_w_body {
    ([none]) => {
        None
    };
    ([in $i:literal]) => {
        Some($i)
    };
}

macro_rules! __derive_registry {
    ($( $kind:ident [$pat:pat] arity = (|$an:ident| $arity:expr), cols = $cols:tt,
        rows = $rows:tt, gemm_w = $gw:tt; )*
     @expansion $( $ek:ident [$ep:pat] arity = (|$ean:ident| $ea:expr), cols = $ec:tt,
        rows = $erc:tt, gemm_w = $egw:tt; )*) => {
        __derive_registry! {
            $( $kind [$pat] arity = (|$an| $arity), cols = $cols, rows = $rows, gemm_w = $gw; )*
            $( $ek [$ep] arity = (|$ean| $ea), cols = $ec, rows = $erc, gemm_w = $egw; )*
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
    ($( $kind:ident [$pat:pat] arity = (|$an:ident| $arity:expr), cols = $cols:tt,
        rows = $rows:tt, gemm_w = $gw:tt; )*) => {
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

            /// The op's output ROW count. `operand_rows(k)` yields the REGISTERED row count of
            /// operand `k` (the row scale a per-head view introduced, carried by the tensors);
            /// `m` is the token count. A row declaring `[in0]` calls the lookup, `[m]` never
            /// does, and Reshape's `[scale f]` reads its own payload — the one op the front
            /// end states a row scale for.
            pub fn out_rows(&self, m: u32, operand_rows: impl Fn(usize) -> u32) -> u32 {
                use $crate::subtile_ir::{EwKind, SubOp};
                match self {
                    $( __out_rows_pat!($kind, $pat, $rows) =>
                        __out_rows_body!(operand_rows, m, $rows), )*
                }
            }

            /// ⭐ THE OP'S GEMM-WEIGHT OPERAND POSITION — the operand a target must re-tile /
            /// stage as a WEIGHT BUFFER, when the op has one. This is the exhaustiveness
            /// authority for every "which weights does the device retile?" collection: a
            /// gemm kind the collector does not name is a weight staged by the WRONG law
            /// (the 26b router defect — `RouterLogits`' `[E,hidden]` projection excluded
            /// from the kernel-weight manifest, staged verbatim into a stick-tiled slot,
            /// `A @ perm(W)`, junk output), so collectors REFUSE an op that carries a gemm
            /// weight they do not handle rather than letting it pass silently. `None` is
            /// NOT "no weight at all": a norm's gain, a router bias, an attention sink and
            /// a GDN bundle are weight READERS that stage by their own paths — classifying
            /// them here would make those collectors refuse them. A new op that reads a
            /// gemm weight MUST answer this row (`gemm_w = [in k]`) or every gemm-weight
            /// collector refuses it at bake.
            pub fn gemm_weight_input(&self) -> Option<usize> {
                use $crate::subtile_ir::{EwKind, SubOp};
                match self {
                    $( $pat => __gemm_w_body!($gw), )*
                }
            }
        }
    };
}
for_each_subop!(__derive_registry);
