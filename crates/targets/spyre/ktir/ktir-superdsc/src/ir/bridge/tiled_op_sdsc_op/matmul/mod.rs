//! Bridge 3 (`TiledOp -> SdscOp`), matmul family — split into small isolated files:
//!   - [`dims`] — `matmul_dims`/`matmul_split_map`, the matmul-specific TileOp dim-vocabulary and
//!     cost-model-splitter adapter (the matmul analogue of the shared `distribute_cores` the
//!     pointwise/reduce families use). `matmul_split_plan`/`matmul_cost_split`/`core_split` (the
//!     actual cost-model search) are genuinely shared infrastructure, not TileIR→SdscOp
//!     translation, so they stay in the live emitter and are imported by name, same pattern as
//!     `distribute_cores` for the other families.
//!   - [`opspec`] — the PRIMITIVE (one `TileOp`/tiling call in, one `OpSpec` out, no internal
//!     decomposition) opspec builders: `matmul_opspec`, `matmul_opspec_off`, `matmul_opspec_split`,
//!     `matmul_opspec_batched`.
//!   - [`assemble`] — the `assemble_matmul*` wrappers (opspec + `emit_sdsc_tiled`).
//!   - [`walk`] — the NAMED walk axes (`MbAxis`/`YAxis`/`InAxis`/`OutAxis`), the sealed
//!     [`walk::Walk2`]/[`walk::Walk3`] declared-walk constructors, each named for the stride
//!     assignment its axis order encodes, and the sealed rung regimes ([`walk::RungRegime`])
//!     whose associated types fix which order a bundle's batched matmuls declare.

pub mod assemble;
pub mod dims;
pub mod opspec;
pub(crate) mod walk;

pub use assemble::{
    assemble_matmul, assemble_matmul_batched_off, assemble_matmul_batched_seeded,
    assemble_matmul_off, assemble_matmul_off_maybe_epilogue, assemble_matmul_off_phys_m,
    assemble_matmul_off_phys_m_maybe_epilogue, assemble_matmul_off_phys_m_with_epilogue,
    assemble_matmul_off_with_epilogue, assemble_matmul_placed, assemble_matmul_seeded,
    assemble_matmul_split,
};
pub use opspec::{
    matmul_opspec, matmul_opspec_batched, matmul_opspec_batched_off, matmul_opspec_off,
    matmul_opspec_off_operands, matmul_opspec_off_operands_phys, matmul_opspec_split,
};
pub use walk::SharedKernelBmmForm;

pub use dims::set_split_mb_forbidden;
