// SPDX-License-Identifier: Apache-2.0
//! emit — the SERIALIZATION bridge: the Kani-verified tower produces the on-card `SdscOp` JSON by DRIVING
//! the monolith's ABI serde (`assemble_matmul_split` → `emit_sdsc_tiled` → `write_bundle`) with the tower's
//! PROVEN work-division (`CoreSplit`). Nothing here decides layout/addresses/splits — those come from the
//! Kani-closed islands (CoreSplit disjoint+covering, AddrIR `dev_off`, CoreMap bijection). This module only
//! (a) hands the tower's split to the ABI emit via the `WorkPlan::divide` splitter seam, and (b) writes the
//! bundle. It is the boundary the Kani ladder verifies UP TO; the emitted JSON's on-card acceptance is the
//! `scr chat` gate (serde/ABI is not Kani-provable — see the handoff's HONEST BOUNDARY note).
//!
//! Island-split plan: as each reused monolith emit-piece (coordInfo, per-core address, fold props) is
//! shown to need it, replace it with the corresponding proven tower island (FoldIR, AddrIR, …) — the
//! splitter seam is the first such replacement (split now comes from CoreSplit, not `matmul_cost_split`).

use crate::tiled_ir::CoreSplit;
use scratchy_subtile::superdsc_opspec::ItDim;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc as mono;
use std::collections::BTreeMap;

/// The tower's Kani-proven 32-core partition expressed as a `WorkPlan::divide` splitter: `CoreSplit::plan`
/// on the matmul OUTPUT `[mb=m, out=n]` → `{mb: row_cores, out: stick_cores}` (>1 entries only). This is
/// the seam that makes the emitted SDSC use the PROVEN split (disjoint+covering, #50-free) instead of the
/// monolith's `matmul_cost_split`.
pub fn tower_matmul_splitter(dims: &[ItDim], _max: u32) -> BTreeMap<&'static str, u32> {
    let sz = |name: &str| {
        dims.iter()
            .find(|d| d.name == name)
            .map(|d| d.size)
            .unwrap_or(1)
    };
    let sp = CoreSplit::plan(sz("mb"), sz("out"));
    let mut map = BTreeMap::new();
    if sp.row_cores > 1 {
        map.insert("mb", sp.row_cores);
    }
    if sp.stick_cores > 1 {
        map.insert("out", sp.stick_cores);
    }
    map
}

/// Emit one matmul's wire node (`EmittedOp`) via the tower's proven split + the ABI serde. `n` (OUTPUT
/// cols) must be a 64-multiple. Serialize the returned op(s) with `mono::write_bundle`.
pub fn emit_matmul(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
) -> Result<mono::EmittedOp, mono::SuperDscError> {
    let mut sym = 0i64;
    mono::assemble_matmul_split(
        op_name,
        m,
        n,
        k,
        1,
        "a",
        "w",
        "o",
        &mut sym,
        None,
        tower_matmul_splitter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// END-TO-END serialization THROUGH THE TOWER: emit a matmul via the tower's proven split, write the
    /// dxp bundle, and confirm the on-card `SdscOp` JSON was produced and carries the op. (Structural — the
    /// on-card numeric acceptance is the `scr chat` gate, not this test.)
    #[test]
    fn matmul_serializes_through_tower() {
        let op = emit_matmul("matmul_tower0", 1, 4096, 4096).expect("tower emit_matmul");
        let dir = std::env::temp_dir().join("scratchy_tower_emit_test");
        let _ = std::fs::remove_dir_all(&dir);
        mono::write_bundle(&dir, std::slice::from_ref(&op)).expect("write_bundle");
        let bytes =
            std::fs::read(dir.join("op_0/sdsc_0.json")).expect("per-op sdsc_0.json written");
        assert!(!bytes.is_empty(), "emitted SdscOp JSON is empty");
        let text = String::from_utf8_lossy(&bytes);
        assert!(
            text.contains("matmul_tower0"),
            "op key missing in emitted JSON"
        );
        assert!(
            text.contains("dscs_") && text.contains("numCoresUsed_"),
            "SdscOp shape missing"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
