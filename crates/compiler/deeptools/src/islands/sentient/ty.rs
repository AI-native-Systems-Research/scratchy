//! THE MLIR TYPES THIS RUNG WRITES — re-exported from the rung below.
//!
//! ⛔⛔ RE-EXPORTED, NOT RESTATED. A `memref<1x2048xf16>`, a `vector<64xf16>`, an `affine_map` and an
//! `affine_set` are MLIR's own type system, not a dialect's and not a rung's. The dump in
//! [`crate::islands::sentient::dialects`] shows the very same `#map`/`#set` declarations surviving
//! from DataflowIR into SentientIR on one real granite program, so two definitions of them would be
//! two things to keep in step for no gain.
//!
//! ⭐ WHAT WOULD JUSTIFY A TYPE OF ITS OWN HERE: a type this rung introduces and the one below cannot
//! express. The dialect's own vocabulary — ports, register classes, precisions, fold and unroll modes
//! — is exactly that, and it lives with the ops in
//! [`crate::islands::sentient::dialects::sentient`] because it is dialect-specific rather than part
//! of MLIR's type system. That is the same split the rung below uses: `Precision` and `LocalUnit` sit
//! in `dialects::dataflow`, while `MemRef` and `Vector` sit in its `ty`.

pub use crate::islands::dataflow_ir::ty::{
    AffineExpr, AffineMap, ElemType, IntegerSet, MemRef, Vector,
};
