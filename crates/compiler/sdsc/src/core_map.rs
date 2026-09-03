// SPDX-License-Identifier: Apache-2.0
//! CoreMap — the TENTH island: the linear core id ↔ `(row-core, stick-core)` grid mapping that keys the
//! wire's `SdscOp::coreIdToWkSlice_` / `coreIdToDsc_`. Row-major: `core_id = cr·stick_cores + cs`. This is
//! the "silently-wrong core-index mapping" hazard called out for the monolith emission — if the linear id
//! and the grid position disagree, cores do each other's work and the build guard never notices. Here it is
//! proven a BIJECTION over `[0, ncores)`, so that class of bug is unconstructable.
//!
//! ISLAND obligation (Kani, <1s): (a) distinct grid cores map to DISTINCT linear ids (each core distinct
//! work — proof uses only ·/+, symbolic-cheap); (b) `decode(core_id(cr,cs)) == (cr,cs)` round-trips (proven
//! per concrete `stick_cores`, so the ÷/% are concrete-divisor = cheap, dodging the `core_split` ÷-hostility).

use crate::tiled_ir::CoreSplit;

/// Row-major linear id of grid core `(cr, cs)` — the string key (`id.to_string()`) of `coreIdToWkSlice_`.
pub fn core_id(cr: u32, cs: u32, stick_cores: u32) -> u32 {
    cr * stick_cores + cs
}

/// Inverse of [`core_id`]: linear id → `(row-core, stick-core)`.
pub fn decode(core_id: u32, stick_cores: u32) -> (u32, u32) {
    (core_id / stick_cores, core_id % stick_cores)
}

/// The per-core work-slice indices `{mb: cr, out: cs}` (what `coreIdToWkSlice_[core_id]` holds for a
/// matmul: which row-chunk and which stick-chunk this core owns).
pub fn wk_slice(split: CoreSplit, core_id_v: u32) -> (u32, u32) {
    decode(core_id_v, split.stick_cores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_id_roundtrips_all_grids() {
        // every (row_cores, stick_cores) a CoreSplit::plan can yield (product ≤ 32).
        for row_cores in 1..=32u32 {
            for stick_cores in 1..=32u32 {
                if row_cores * stick_cores > 32 {
                    continue;
                }
                let mut seen = std::collections::BTreeSet::new();
                for cr in 0..row_cores {
                    for cs in 0..stick_cores {
                        let id = core_id(cr, cs, stick_cores);
                        assert!(id < row_cores * stick_cores, "id {id} ≥ ncores");
                        assert_eq!(decode(id, stick_cores), (cr, cs), "roundtrip {cr},{cs}");
                        assert!(seen.insert(id), "id {id} not unique");
                    }
                }
                assert_eq!(
                    seen.len() as u32,
                    row_cores * stick_cores,
                    "not a bijection onto [0,ncores)"
                );
            }
        }
    }
}

// ── Kani (<1s each): the core-index mapping is a bijection ⇒ the monolith's silently-wrong-core hazard is
//    unconstructable. Injectivity is symbolic (·/+ only); round-trip is per-concrete-stick_cores (cheap ÷/%).
#[cfg(kani)]
mod kani_coremap {
    use super::*;

    /// DISTINCT grid cores → DISTINCT linear ids (each core does its own work). Only ·/+, so symbolic
    /// `stick_cores` is fine (no ÷/%). Bounds: `cr ≤ 31`, `cs < stick_cores ≤ 32` (any plan fits).
    #[kani::proof]
    fn core_id_injective() {
        let sc: u32 = kani::any();
        kani::assume(sc >= 1 && sc <= 32);
        let (cr1, cs1): (u32, u32) = (kani::any(), kani::any());
        let (cr2, cs2): (u32, u32) = (kani::any(), kani::any());
        kani::assume(cr1 <= 31 && cr2 <= 31 && cs1 < sc && cs2 < sc);
        kani::assume(!(cr1 == cr2 && cs1 == cs2)); // distinct grid cores
        assert!(core_id(cr1, cs1, sc) != core_id(cr2, cs2, sc));
    }

    /// `decode(core_id(cr,cs)) == (cr,cs)` — concrete `stick_cores` so the ÷/% are concrete-divisor (cheap).
    fn roundtrip(sc: u32) {
        let (cr, cs): (u32, u32) = (kani::any(), kani::any());
        kani::assume(cr <= 31 && cs < sc);
        let id = core_id(cr, cs, sc);
        assert!(decode(id, sc) == (cr, cs));
    }
    #[kani::proof]
    fn roundtrip_sc1() {
        roundtrip(1); // stick_cores=1 (all cores split rows — the RoPE 256×128 plan)
    }
    #[kani::proof]
    fn roundtrip_sc8() {
        roundtrip(8); // lm_head padded-vocab plan
    }
    #[kani::proof]
    fn roundtrip_sc32() {
        roundtrip(32); // m=1 wide projection (hidden 4096: row_cores=1, stick_cores=32)
    }
}
