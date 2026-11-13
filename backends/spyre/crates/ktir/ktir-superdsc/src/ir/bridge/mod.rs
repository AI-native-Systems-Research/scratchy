// SPDX-License-Identifier: Apache-2.0
//! The translations between adjacent islands. See [`super`] for the pipeline map and the purity
//! contract each bridge is held to.
//!
//! `span_overflow` is a Bridge-2-adjacent utility (the span-overflow half of the tiler, ported from
//! torch-spyre — see its own module doc) rather than a bridge stage of its own.

/// The span-overflow refusal: a real build-time `Err` on the attention path when one core's
/// physical span would exceed what the device can address in one go.
pub mod span_overflow;
/// THE TILER: core-split + LX-fit time-tiling. Not pure syntax, and that is correct — scheduling is
/// a real decision, CONFINED to this one bridge and parameterized by an explicit `splitter`.
pub mod tile_op_tiled_op;
/// Bridge 3: `TiledOp -> SdscOp` — the per-op-family assemblers (pointwise/broadcast, reduce,
/// matmul, rmsnorm) and the fused attention emitter. See its own header.
pub mod tiled_op_sdsc_op;
