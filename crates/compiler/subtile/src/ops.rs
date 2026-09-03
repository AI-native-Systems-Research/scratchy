//! THE op registry — the single declaration site for the tape vocabulary.
//!
//! Every fact a consumer needs about a [`crate::lower::LoweredOp`] beyond
//! its prose docs — its fields, operand arity, and output-width rule — is
//! declared ONCE here as an X-macro row. Consumers (the enum lock guard,
//! `op_out_cols`, the emitter in the `#[forward]` macro crate, per-target
//! support tables, refusal lists) each invoke [`for_each_lowered_op!`]
//! with their own callback and generate their site from the same rows.
//!
//! Adding an op = adding ONE row here (plus the enum variant with its
//! prose docs, which the lock guard below forces to stay in sync — a
//! missing or drifted variant is a compile error at the guard match, not
//! a silently-unhandled case nine files away). This exists because the
//! previous attempt at the metal/spyre unification wrote each of these
//! facts by hand at every site: one op touched nine files, and every slot
//! bug of that branch traced to one site disagreeing with another.
//!
//! Row grammar:
//! `Name { fields } , arity = (|n| <bool over n>), cols = <class>;`
//! - `arity` is a closure-form predicate over the operand count (the
//!   binder is declared IN the row so macro hygiene ties it to the body).
//! - `cols` is one of (bracketed so it stays one macro token tree):
//!   - `[in0]` — shape-preserving: output width = `inputs[0]` width;
//!   - `[in1]` — output width = `inputs[1]` width (GDN's z operand);
//!   - `[field f]` — output width is the op's own field `f`;
//!   - `[expr |op| ...]` — computed from the op's fields.

/// X-macro over every [`crate::lower::LoweredOp`]. Callbacks receive the
/// full row list and pick what they need.
#[macro_export]
macro_rules! for_each_lowered_op {
    ($cb:ident) => {
        $cb! {
            Gemm { n: u32, weight: GemmWeight },
                arity = (|n| n == 2 || n == 3),
                cols = [field n];
            RmsNorm { eps: f32, gain_offset: f32 },
                arity = (|n| n == 2),
                cols = [in0];
            Silu {},
                arity = (|n| n == 1),
                cols = [in0];
            Gelu {},
                arity = (|n| n == 1),
                cols = [in0];
            TanhSoftCap {},
                arity = (|n| n == 1),
                cols = [in0];
            RmsNormUnit { eps: f32 },
                arity = (|n| n == 1),
                cols = [in0];
            ScalarWeightMul {},
                arity = (|n| n == 2),
                cols = [in0];
            Moe { qwen_shared: bool, num_experts: u32, top_k: u32, moe_inter: u32, shared_inter: u32, norm_topk: bool, group_size: u32, bits: u32 },
                arity = (|n| n == 2),
                cols = [in0];
            GemmaMoe { num_experts: u32, top_k: u32, moe_inter: u32, group_size: u32, bits: u32 },
                arity = (|n| n == 4),
                cols = [in0];
            LoadPixels { in_features: u32 },
                arity = (|n| n == 0),
                cols = [field in_features];
            LoadPosEmbeds { width: u32 },
                arity = (|n| n == 0),
                cols = [field width];
            EmbeddingGather { indices_kind: u8 },
                arity = (|n| n == 1),
                cols = [in0];
            VisionRope {},
                arity = (|n| n == 2),
                cols = [in0];
            QuickGelu {},
                arity = (|n| n == 1),
                cols = [in0];
            GeluErf {},
                arity = (|n| n == 1),
                cols = [in0];
            VarlenAttention { cu_kind: u8 },
                arity = (|n| n == 3),
                cols = [in0];
            EncoderAttn { geom: ModelAttnGeometry, scale: f32 },
                arity = (|n| n == 3),
                cols = [expr |op| op_encoder_q_width(op)];
            GatedDeltaNet {},
                arity = (|n| n == 5),
                cols = [in1];
            GateSplit { half_cols: u32 },
                arity = (|n| n == 1),
                cols = [field half_cols];
            GateApply {},
                arity = (|n| n == 2),
                cols = [in0];
            GateScale {},
                arity = (|n| n == 3),
                cols = [in0];
            Mul {},
                arity = (|n| n == 2),
                cols = [in0];
            ScalarMul { scale: f32 },
                arity = (|n| n == 1),
                cols = [in0];
            SiluMul {},
                arity = (|n| n == 2),
                cols = [in0];
            Add {},
                arity = (|n| n == 2),
                cols = [in0];
            Mean {},
                arity = (|n| n == 1),
                cols = [expr |_op| 1u32];
            Sub {},
                arity = (|n| n == 2),
                cols = [in0];
            BiasAdd {},
                arity = (|n| n == 2),
                cols = [in0];
            Reshape { rows_mult: u32, rows_div: u32, cols: u32 },
                arity = (|n| n == 1),
                cols = [field cols];
            RopeRotate { head_dim: HeadDim },
                arity = (|n| n == 3),
                cols = [in0];
            RopeAppend { head_dim: HeadDim, layer: u32, is_global: bool, interleaved: bool },
                arity = (|n| n == 4),
                cols = [in0];
            AttnDecode { geom: ModelAttnGeometry, scale: f32, valid_len: u32, sliding: bool },
                arity = (|n| n >= 3),
                cols = [expr |op| op_geom_q_width(op)];
        }
    };
}

/// `cols = expr(...)` helper for [`AttnDecode`]: the registry row can't
/// name `geom` positionally, so the expression routes through this fn.
pub(crate) fn op_encoder_q_width(op: &crate::lower::LoweredOp) -> u32 {
    match op {
        crate::lower::LoweredOp::EncoderAttn { geom, .. } => geom.q_width(),
        _ => unreachable!("registry routes only EncoderAttn here"),
    }
}

pub(crate) fn op_geom_q_width(op: &crate::lower::LoweredOp) -> u32 {
    match op {
        crate::lower::LoweredOp::AttnDecode { geom, .. } => geom.q_width(),
        _ => unreachable!("op_geom_q_width called on a non-AttnDecode row"),
    }
}

// ── Lock guard ──────────────────────────────────────────────────────
// A match generated from the registry, over the hand-written enum, with
// every declared field bound by name. If the enum gains a variant the
// registry lacks: non-exhaustive match → compile error HERE. If the
// registry names a field the enum lacks (or drops one): pattern error
// HERE. The prose docs stay on the enum; the facts stay here; the
// compiler holds them together.
macro_rules! __lock_registry_to_enum {
    ($( $name:ident { $($f:ident : $t:ty),* $(,)? },
         arity = (|$an:ident| $arity:expr),
         cols = $cols:tt; )*) => {
        #[allow(unused_variables, dead_code)]
        pub(crate) fn __registry_lock(op: &crate::lower::LoweredOp) {
            use crate::lower::LoweredOp;
            match op {
                $( LoweredOp::$name { $($f),* } => {} )*
            }
        }
    };
}
for_each_lowered_op!(__lock_registry_to_enum);

// ── Derived: operand-arity check ────────────────────────────────────
macro_rules! __derive_arity {
    ($( $name:ident { $($f:ident : $t:ty),* $(,)? },
         arity = (|$an:ident| $arity:expr),
         cols = $cols:tt; )*) => {
        /// Whether `n` operands is legal for `op` — derived from the
        /// registry, used by the IR validator.
        pub fn lowered_op_arity_ok(op: &crate::lower::LoweredOp, n_operands: usize) -> bool {
            use crate::lower::LoweredOp;
            match op {
                $( LoweredOp::$name { .. } => {
                    let $an = n_operands;
                    $arity
                } )*
            }
        }
    };
}
for_each_lowered_op!(__derive_arity);

// ── Derived: output width ───────────────────────────────────────────
macro_rules! __derive_out_cols_arm {
    (@arm $op:ident, $w:ident, $name:ident, [in0]) => {
        $w(0)
    };
    (@arm $op:ident, $w:ident, $name:ident, [in1]) => {
        $w(1)
    };
    (@arm $op:ident, $w:ident, $name:ident, [field $f:ident]) => {
        match $op {
            crate::lower::LoweredOp::$name { $f, .. } => *$f,
            _ => unreachable!(),
        }
    };
    (@arm $op:ident, $w:ident, $name:ident, [expr |$p:ident| $e:expr]) => {{
        let $p = $op;
        $e
    }};
}
macro_rules! __derive_out_cols {
    ($( $name:ident { $($f:ident : $t:ty),* $(,)? },
         arity = (|$an:ident| $arity:expr),
         cols = $cols:tt; )*) => {
        /// Output column count of `op` — derived from the registry.
        ///
        /// `operand_cols(k)` yields the width of operand `k`; a row
        /// declaring `[in0]`/`[in1]` calls it, the other classes never
        /// do. Taking a LOOKUP rather than just `in0_cols` is what lets
        /// this serve every consumer: the fixtures path used to keep a
        /// hand-written copy of this match purely because GDN's width
        /// comes from operand 1, and that copy is the kind of second
        /// site the registry exists to prevent.
        pub fn lowered_op_out_cols(
            op: &crate::lower::LoweredOp,
            operand_cols: impl Fn(usize) -> u32,
        ) -> u32 {
            use crate::lower::LoweredOp;
            match op {
                $( LoweredOp::$name { .. } =>
                    __derive_out_cols_arm!(@arm op, operand_cols, $name, $cols), )*
            }
        }
    };
}
for_each_lowered_op!(__derive_out_cols);

// ── Derived: stable name strings (histograms, audit notes, refusals) ─
macro_rules! __derive_names {
    ($( $name:ident { $($f:ident : $t:ty),* $(,)? },
         arity = (|$an:ident| $arity:expr),
         cols = $cols:tt; )*) => {
        /// The op's registry name — derived; for dumps and refusal
        /// messages, so no consumer hand-writes a parallel string table.
        pub fn lowered_op_name(op: &crate::lower::LoweredOp) -> &'static str {
            use crate::lower::LoweredOp;
            match op {
                $( LoweredOp::$name { .. } => stringify!($name), )*
            }
        }
    };
}
for_each_lowered_op!(__derive_names);
