//! Bridge 3: `TiledOp -> SdscOp`. This is where composite-op decomposition legitimately belongs
//! (RmsNorm's 6-op form, RoPE's rotate+xc+rs+add, AttnDecode's fused attention): the schedule must
//! be decided FIRST against the whole composite `TileOp` (Bridge 2), and only THEN does
//! decomposition into concrete `SdscOp`s happen, informed by that schedule.
//!
//! Split into small, ISOLATED per-op-family modules rather than one growing file — each op family
//! (pointwise/broadcast, reduce, matmul) has its own opspec builders + `assemble_*` wrappers and
//! essentially no cross-family dependency, so each gets its own file:
//!   - [`pointwise`] — the pointwise/broadcast family.
//!   - [`reduce`] — the reduce family.
//!   - [`matmul`] — the matmul family (plus its dim-vocabulary/cost-model-splitter helpers).
//!
//! All PRIMITIVE (non-decomposing: one `TileOp`/tiling call in, one `OpSpec`/`EmittedOp` out)
//! TileIR→SdscOp entry points for these families live here for real — moved out of the live
//! emitter (`crate::lower_subtile_tape_to_superdsc`), which now `use`s them by name so every
//! existing unqualified call site there is unaffected. Re-exported flat at this module's own path
//! (`crate::ir::bridge::tiled_op_sdsc_op::foo`) so no caller needed to change when this split from
//! one file into three.
//!
//! The COMPOSITE-op decomposition itself (RoPE's rotate+xc+rs+add, AttnDecode's fused attention,
//! SiluMul's internal multi-op sequence, and the `lower_*_node` dispatch functions) still lives in
//! the live emitter: each of those already calls through to these families' functions as its own
//! OWN primitive-op building blocks. RmsNorm's decomposition ([`rmsnorm`]) has moved here —
//! it is the first composite decomposer ported, matching torch-spyre's `spyre_rms_norm` exactly
//! (see that module's doc). Migrating the *remaining* composite decomposers here too is real,
//! ongoing work, not yet complete for RoPE/AttnDecode/SiluMul.
//!   - [`rmsnorm`] — torch-spyre `spyre_rms_norm`, one formula for every row count.

pub mod attn;
pub mod matmul;
pub mod pointwise;
pub mod reduce;
pub mod rmsnorm;

pub use attn::assemble_attn;
pub use matmul::{
    SharedKernelBmmForm, assemble_matmul, assemble_matmul_batched_off,
    assemble_matmul_batched_seeded, assemble_matmul_off, assemble_matmul_off_phys_m,
    assemble_matmul_placed, assemble_matmul_seeded, assemble_matmul_split, matmul_opspec,
    matmul_opspec_batched, matmul_opspec_batched_off, matmul_opspec_off,
    matmul_opspec_off_operands, matmul_opspec_split, try_assemble_matmul_seeded,
};
pub use pointwise::{
    assemble_gather_copy, assemble_pointwise_broadcast_off_from_tile,
    assemble_pointwise_seeded_from_tile, gather_copy_opspec, pointwise_broadcast_opspec_from_tile,
    pointwise_opspec_from_tile,
};
pub use reduce::{
    assemble_reduce, assemble_reduce_df, assemble_reduce_off, assemble_reduce_seeded,
    reduce_opspec, reduce_opspec_df, reduce_opspec_off,
};
pub use rmsnorm::assemble_rmsnorm;
