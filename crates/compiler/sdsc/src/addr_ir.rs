// SPDX-License-Identifier: Apache-2.0
//! AddrIR — the SEVENTH island, one lowering pass below `TiledIR` and the first rung of the WIRE.
//!
//! TiledIR proved WHICH region each core owns (disjoint + covering). AddrIR turns each owned region into
//! the concrete on-device ELEMENT OFFSET the wire's `SdscOp` needs: for core `(cr,cs)` of an op whose
//! output is `[R,C]` (stick-last on the last dim), the per-core start offset is `dev_off([R,C], 1, corner)`
//! where `corner = (r0, c0)` is the region's top-left (from `CoreSplit::region`). This is pure INTEGER
//! address math — no floats — so it is 100% Kani-closeable, which is the whole point: the monolith computed
//! these offsets UNVERIFIED (and #50-collided); here the collision is *unconstructable* by proof.
//!
//! ISLAND obligation (proven in [`kani_proofs`], composing the already-closed `plan_regions_disjoint_*` +
//! `dev_off_general_injective`/`_in_footprint`): for one op's `CoreSplit::plan`, DISTINCT cores map to
//! DISTINCT output offsets (no aliased write ⇒ no #50) that all lie within the tensor footprint
//! `R * pad64(C)` (no clobber of the next tensor). Segment BASES (adding `seg_base` to place tensors in
//! HBM) are a SEPARATE island — this one owns only the intra-tensor per-core offset.

use crate::flat_ir::Shape;
use crate::tiled_ir::{CoreSplit, TiledIR};
use scratchy_subtile::sdsc_abstract::dev_off;

/// Per-op wire addressing: the output tensor's per-core start OFFSET (in elements, from the tensor base),
/// one entry per used core in `(cr, cs)` row-major order. Parallel to `TiledIR::outs`/`splits`.
#[derive(Clone, Debug)]
pub struct OpAddr {
    /// The op's output tensor id (into `AddrIR::tensors`).
    pub out: usize,
    /// The op's output shape `[rows, cols]`.
    pub shape: Shape,
    /// The 32-core split (carried from TiledIR).
    pub split: CoreSplit,
    /// `offsets[cr * stick_cores + cs]` = element offset of core `(cr,cs)`'s output tile start.
    pub offsets: Vec<usize>,
}

/// A distinct IR: TiledIR's op graph annotated with each op's concrete per-core output offsets.
#[derive(Clone, Debug)]
pub struct AddrIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    pub result: usize,
    pub ops: Vec<OpAddr>,
}

/// The element offset of core `(cr, cs)`'s output-tile TOP-LEFT, for an `[R,C]` stick-last (stick = last
/// dim) output. This is the SOLE address formula — the same `dev_off` the Kani proofs close over.
pub fn core_out_offset(split: CoreSplit, rows: u32, cols: u32, cr: u32, cs: u32) -> usize {
    let (r0, _r1, c0, _c1) = split.region(cr, cs, rows, cols);
    dev_off(
        &[rows as usize, cols as usize],
        1,
        &[r0 as usize, c0 as usize],
    )
}

/// THE PASS: `TiledIR → AddrIR`. Compute every op's per-core output offset via `core_out_offset`
/// (`dev_off ∘ region-corner`). No decisions — TiledIR already decided the split; this only addresses it.
pub fn lower_tiled_to_addr(t: &TiledIR) -> AddrIR {
    let mut ops = Vec::with_capacity(t.outs.len());
    for (k, &out) in t.outs.iter().enumerate() {
        let shape = t.tensors[out];
        let split = t.splits[k];
        let mut offsets = Vec::with_capacity(split.ncores() as usize);
        for cr in 0..split.row_cores {
            for cs in 0..split.stick_cores {
                offsets.push(core_out_offset(split, shape.rows, shape.cols, cr, cs));
            }
        }
        ops.push(OpAddr {
            out,
            shape,
            split,
            offsets,
        });
    }
    AddrIR {
        tensors: t.tensors.clone(),
        num_sources: t.num_sources,
        result: t.result,
        ops,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiled_ir::CoreSplit;

    /// Runtime mirror of the Kani obligation (a fast regression on the concrete granite shapes): each op's
    /// per-core output offsets are DISTINCT (no #50) and inside the footprint `rows * pad64(cols)`.
    #[test]
    fn per_core_offsets_distinct_and_in_footprint() {
        let pad64 = |c: u32| ((c + 63) / 64) * 64;
        for &(r, c) in &[
            (1u32, 128u32),
            (96, 576),
            (256, 128),
            (1, 49159),
            (384, 384),
            (17, 200),
        ] {
            let split = CoreSplit::plan(r, c);
            let mut seen = std::collections::BTreeSet::new();
            let foot = (r * pad64(c)) as usize;
            for cr in 0..split.row_cores {
                for cs in 0..split.stick_cores {
                    let off = core_out_offset(split, r, c, cr, cs);
                    assert!(
                        off < foot,
                        "shape [{r},{c}] core ({cr},{cs}) offset {off} >= footprint {foot}"
                    );
                    assert!(
                        seen.insert(off),
                        "shape [{r},{c}] core ({cr},{cs}) offset {off} COLLIDES (#50)"
                    );
                }
            }
        }
    }
}
