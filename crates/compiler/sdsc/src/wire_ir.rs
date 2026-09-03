// SPDX-License-Identifier: Apache-2.0
//! WireIR — the ELEVENTH island: the STRUCTURAL assembly of one op's wire node from the already-proven
//! parts (CoreMap linear ids, OpDims per-dim splits, AddrIR per-core offsets). This is the last IR STATE
//! before serialization to the card's `SdscOp` JSON (that serialize step is trivial and not a pass). WireIR
//! adds NO new decisions — it just gathers the proven pieces into the per-core work table the wire needs.
//!
//! ISLAND obligation (Kani, concrete shapes, <2s): the assembled table is CONSISTENT — `num_cores ==
//! wk_mb · wk_out` (the per-dim split counts multiply to the core count); the table has exactly `num_cores`
//! entries in linear-id order; each core's `(mb_idx, out_idx)` equals `CoreMap::wk_slice(id)` (so the
//! `coreIdToWkSlice_` the card reads matches the grid position that produced the address); and the per-core
//! output offsets are DISTINCT (the #50 property, re-checked at the assembled node). All structural/integer.

use crate::addr_ir::core_out_offset;
use crate::core_map::{core_id, wk_slice};
use crate::op_dims::matmul_output_dims;
use crate::tiled_ir::CoreSplit;

/// One core's work in the assembled node: its grid slice indices + its output tile start offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreWork {
    pub mb_idx: u32,
    pub out_idx: u32,
    pub out_offset: usize,
}

/// The structural wire node for one op (pre-serialization). `cores[id]` is the work of linear core `id`
/// (`id = cr·stick_cores + cs`, CoreMap order — the `coreIdToWkSlice_` key order). `wk_mb`/`wk_out` are
/// `numWkSlicesPerDim_`; `num_cores` is `numCoresUsed_`.
#[derive(Clone, Debug)]
pub struct WireOp {
    pub num_cores: u32,
    pub wk_mb: u32,
    pub wk_out: u32,
    pub cores: Vec<CoreWork>,
}

/// Assemble the wire node for a matmul OUTPUT `[m, n]` (n 64-padded). Pure gather of the proven pieces:
/// CoreMap linear order, OpDims split counts, AddrIR per-core offsets. No new decisions.
pub fn assemble_matmul_wire(m: u32, n: u32, split: CoreSplit) -> WireOp {
    let (mb, out) = matmul_output_dims(m, n, split);
    let mut cores = Vec::with_capacity(split.ncores() as usize);
    // linear-id order: cr outer, cs inner — matches `core_id(cr,cs,stick_cores)`.
    for cr in 0..split.row_cores {
        for cs in 0..split.stick_cores {
            cores.push(CoreWork {
                mb_idx: cr,
                out_idx: cs,
                out_offset: core_out_offset(split, m, n, cr, cs),
            });
        }
    }
    WireOp {
        num_cores: split.ncores(),
        wk_mb: mb.nsplits,
        wk_out: out.nsplits,
        cores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn matmul_wire_is_consistent() {
        for &(m, n) in &[
            (1u32, 4096u32),
            (1, 1024),
            (1, 49664),
            (256, 128),
            (384, 384),
        ] {
            let split = CoreSplit::plan(m, n);
            let w = assemble_matmul_wire(m, n, split);
            assert_eq!(
                w.num_cores,
                w.wk_mb * w.wk_out,
                "[{m},{n}] core count ≠ split product"
            );
            assert_eq!(
                w.cores.len() as u32,
                w.num_cores,
                "[{m},{n}] table size ≠ num_cores"
            );
            let mut offs = BTreeSet::new();
            for cr in 0..split.row_cores {
                for cs in 0..split.stick_cores {
                    let id = core_id(cr, cs, split.stick_cores) as usize;
                    let cw = w.cores[id];
                    // coreIdToWkSlice_ consistency: linear id decodes to this core's grid slice.
                    assert_eq!(
                        (cw.mb_idx, cw.out_idx),
                        wk_slice(split, id as u32),
                        "[{m},{n}] id {id} slice"
                    );
                    assert!(
                        offs.insert(cw.out_offset),
                        "[{m},{n}] id {id} DUPLICATE offset (#50)"
                    );
                }
            }
        }
    }
}

// ── Kani: the assembled node is structurally consistent (concrete shapes ⇒ the Vec is concrete; ~<2s). ──
#[cfg(kani)]
mod kani_wire {
    use super::*;

    fn matmul_wire_consistent(m: u32, n: u32) {
        let split = CoreSplit::plan_capped(m, n, crate::tiled_ir::MAX_CORES);
        let w = assemble_matmul_wire(m, n, split);
        // split counts multiply to the core count; table sized to it.
        assert!(w.num_cores == w.wk_mb * w.wk_out);
        assert!(w.cores.len() == w.num_cores as usize);
        // symbolic linear id in range: its stored grid slice matches CoreMap's decode (coreIdToWkSlice_ ok).
        let id: u32 = kani::any();
        kani::assume(id < w.num_cores);
        let cw = w.cores[id as usize];
        assert!((cw.mb_idx, cw.out_idx) == wk_slice(split, id));
    }
    #[kani::proof]
    fn wire_hidden() {
        matmul_wire_consistent(1, 4096); // m=1, row_cores=1, stick_cores=32
    }
    #[kani::proof]
    fn wire_lmhead() {
        matmul_wire_consistent(1, 49664); // m=1 lm_head, stick_cores=8
    }
    #[kani::proof]
    fn wire_rope() {
        matmul_wire_consistent(256, 128); // multi-row, row_cores=32, stick_cores=1
    }
}
