//! The three translations between adjacent islands. See `crate::ir` for the full pipeline map and
//! the purity contract each bridge is held to.
//!
//! `tiled_op_sdsc_op` (Bridge 3) holds the four PRIMITIVE (one `TileOp` in, one concrete op out,
//! no internal decomposition) TileIR→SdscOp entry points. Composite-op decomposition (RmsNorm's
//! 6-op form, RoPE's rotate+xc+rs+add, AttnDecode's fused attention, every `lower_*_node`
//! function) still lives in `crate::lower_subtile_tape_to_superdsc` (the live emitter), which
//! calls through to this module's functions as its own primitive-op building blocks — migrating
//! the composite decomposers too would mean relocating the ~13k-line emitter wholesale, not
//! completing the Bridge 3 boundary itself.
//!
//! `span_overflow` is a Bridge-2-adjacent utility (the span-overflow half of the tiler, ported
//! from torch-spyre — see its own module doc) rather than a fourth bridge stage.

pub mod subtile_tape_tile_op;
// ⭐ BRIDGE 3, THE TILER AND THE SPAN REFUSAL NOW LIVE IN `ktir-superdsc`, beside the island they
// translate. Re-exported as MODULES so every
// `crate::ir::bridge::{tiled_op_sdsc_op,tile_op_tiled_op,span_overflow}::…` path in this crate
// resolves unchanged — `subtile_tape_tile_op` (the SubtileIR-shaped syntax bridge) is the only one
// left here, because its input is a `SubtileTape` node.
pub use ktir_superdsc::ir::bridge::{span_overflow, tile_op_tiled_op, tiled_op_sdsc_op};
