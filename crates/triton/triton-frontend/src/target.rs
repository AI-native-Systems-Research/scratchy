//! Target-conditional front-end policy.
//!
//! Triton's front end bakes in a handful of rules that are NOT semantic necessities but
//! properties of the GPU it was written for. Porting those faithfully would make correct
//! Spyre kernels illegal, so they live here as switches with the measurement that decides
//! each one.

/// Which front-end policies apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// # THE DIVERGENCE. `f16 / f16` MUST NOT PROMOTE TO f32 ON SPYRE.
    ///
    /// Triton's `computation_type_impl` (`python/triton/language/semantic.py:68`) promotes
    /// the result of `/` and `%` to f32 whenever either operand is f16 or bf16. Its rule 3
    /// states the reason in the source:
    ///
    /// ```text
    /// # 3 ) if one operand is half, the other is implicitly converted to half
    /// #     unless we're doing / or %, which do not exist natively in PTX for fp16.
    /// ```
    ///
    /// **That is a PTX fact and it is FALSE on this device.** Divide is a templated op
    /// here: `OpFuncs::REALDIV` is bound by `broadcast_ops.ddl`, and `arith.divf` already
    /// maps to `"realdiv"` in `../triton-superdsc/triton-superdsc-lower/src/opmap.rs`.
    ///
    /// Porting the promotion faithfully is not merely wasteful, it makes kernels ILLEGAL
    /// at the next contraction: the f32 quotient reaches a `tl.dot` beside an f16 weight
    /// and Triton's own frontend then refuses it with
    /// `Both operands must be same dtype. Got fp32 and fp16`.
    ///
    /// THE EVIDENCE THAT IT IS A FRONT-END POLICY AND NOT A SEMANTIC REQUIREMENT:
    /// `tl.fdiv` is the same `create_fdiv` with `arithmetic_check` turned off, and it
    /// yields f16 from f16 operands. One operation, two answers, decided by a flag -- so
    /// the flag is policy. `swiglu_mlp.py`'s delta 6 records the measurement:
    ///
    /// ```text
    ///     f16 / f16          -> fp32
    ///     tl.fdiv(f16, f16)  -> fp16
    /// ```
    ///
    /// WHEN `false` (the Spyre setting), `/` on two f16 tensors stays f16 and emits a
    /// single `arith.divf`, which is exactly what `tl.fdiv` already produces. Set it
    /// `true` only to reproduce upstream Triton's output for a comparison.
    ///
    /// CONSEQUENCE FOR THE GOLDEN DIFF, stated so it is not mistaken for a bug: the
    /// fixtures were written to work around the promotion (they call `tl.fdiv`), so the
    /// goldens contain no bare `f16 / f16` and this switch changes nothing about them.
    /// `tests/divergence.rs` therefore exercises it directly rather than relying on a
    /// fixture to cover it.
    pub div_promotes_narrow_floats: bool,

    /// Emit the integer-overflow check `binary_op_sanitize_overflow_impl` builds around
    /// every `+`, `-`, `*` on integers narrower than 64 bits
    /// (`semantic.py:212`): widen both operands to i64, redo the op, compare against the
    /// type's min and max, and `and` the two predicates.
    ///
    /// Triton's default is ON, and the goldens were generated with it on -- it is the
    /// single largest source of ops in the raw TTIR (68 `arith.extsi`, 35 `arith.cmpi sle`
    /// + 35 `sge`, 35 `arith.andi` across the seven configurations). Kept as a switch
    /// because it must be ON to match the oracle but is pure dead code the canonicalizer
    /// deletes, so a future direct-to-KTIR path will want it off.
    pub sanitize_overflow: bool,
}

impl Target {
    /// The Spyre front end.
    pub fn spyre() -> Target {
        Target {
            div_promotes_narrow_floats: false,
            sanitize_overflow: true,
        }
    }

    /// Upstream Triton's rules, for differential testing only.
    pub fn upstream_gpu() -> Target {
        Target {
            div_promotes_narrow_floats: true,
            sanitize_overflow: true,
        }
    }
}

impl Default for Target {
    fn default() -> Self {
        Target::spyre()
    }
}
