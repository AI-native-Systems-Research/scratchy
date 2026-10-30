// SPDX-License-Identifier: Apache-2.0
//! Write the dxp input of a tensor-parallel forward's collective SENDS and CONSUMES, for compiling
//! by hand with `dxp_standalone`.
//!
//! A two-rank MLP sublayer (`scratchy_subtile::fixtures::mlp_sublayer`, sharded by the shared
//! analysis, which column-splits gate/up, row-splits down and weaves ONE all-reduce between down and
//! the residual add) is lowered through the producer and the SuperDSC door exactly as a bake lowers
//! it, and only the launch groups of a collective are written: one `group_{gi}/` per send or
//! consume under the directory given as the only argument. The send stores the operand into the
//! receive window (launched against the peer's segment 1, a device-to-device store); the consume
//! reads the window, the one operand no other program in the tree has placed in the weight segment.
//!
//! ```text
//! cargo run -p scratchy-target-spyre --features spyre --example dump_tp_consume_sdsc -- <dir>
//! ```

use std::num::NonZeroU32;

use ktir_superdsc::ktir_node::ActiveCap;
use scratchy_subtile::fixtures::mlp_sublayer;
use scratchy_subtile::subtile_ir::lower_region;
use scratchy_target_spyre::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
use scratchy_target_spyre::lower_subtile_tape_to_superdsc::{FoldGrouping, write_dxp_input};
use scratchy_target_spyre::tp_plan::{SpyreWorld, tp_bake};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or("usage: dump_tp_consume_sdsc <output dir>")?;
    let bake = tp_bake(&mlp_sublayer(2048, 8192), SpyreWorld::of::<2>())
        .map_err(|e| format!("the two-rank sublayer does not shard: {e:?}"))?;
    let ir = lower_region(&bake.sharded.local, NonZeroU32::new(128).ok_or("128 != 0")?);
    // The sublayer's gemm weights: gate, up, down.
    let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
    let (ops, layout) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
        .map_err(|e| format!("the two-rank sublayer does not lower: {e:?}"))?;
    let consumes: Vec<_> = ops
        .iter()
        .filter(|e| e.tp_consume.is_some() || e.tp_send.is_some())
        .map(|e| e.shallow_copy())
        .collect();
    let window = layout
        .placements
        .get(&ktir_superdsc::reserved_tids::TP_RECV_TID)
        .ok_or("the sharded layout carries no receive window")?;
    eprintln!(
        "{} op(s), {} collective op(s); window seg{} @ {} ({} B)",
        ops.len(),
        consumes.len(),
        window.segment,
        window.offset,
        window.size
    );
    write_dxp_input(&dir, &consumes, FoldGrouping::Split)?;
    eprintln!("wrote the all-reduce's send and consume dxp input under {}", dir.display());

    // And the ALL-GATHER's send and consumes — one consume group per rank, from a two-rank
    // vocabulary projection.
    let gather = tp_bake(
        &scratchy_subtile::fixtures::vocab_projection(2048, 8192),
        SpyreWorld::of::<2>(),
    )
    .map_err(|e| format!("the two-rank projection does not shard: {e:?}"))?;
    let ir = lower_region(&gather.sharded.local, NonZeroU32::new(128).ok_or("128 != 0")?);
    let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
    let (ops, _) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
        .map_err(|e| format!("the two-rank projection does not lower: {e:?}"))?;
    let consumes: Vec<_> = ops
        .iter()
        .filter(|e| e.tp_consume.is_some() || e.tp_send.is_some())
        .map(|e| e.shallow_copy())
        .collect();
    let gather_dir = dir.join("all_gather");
    write_dxp_input(&gather_dir, &consumes, FoldGrouping::Split)?;
    eprintln!(
        "wrote the all-gather's {} collective op(s) under {}",
        consumes.len(),
        gather_dir.display()
    );
    Ok(())
}
