// SPDX-License-Identifier: Apache-2.0
//! Spyre's half of tensor parallelism: the target facts the shared analysis takes as INPUT, and the
//! bake-time collective facts derived from its OUTPUT.
//!
//! The analysis (`scratchy_subtile::tensor_parallel`) is target-neutral and decides where every
//! collective goes. Spyre contributes one fact to it — the column granule it can split at
//! ([`SPYRE_SPLIT_GRANULE`], one fp16 stick) — and reads back the sharded forward: each collective
//! becomes consume launches over stick-column fragments of its operand ([`consume_fragments`]), and
//! each sender's block of the receive window is as wide as the widest of them
//! ([`TpBake::window_block`]). Nothing here decides a placement.
//!
//! ⭐ THE WORLD IS THE TYPE `T` — one world, named once by the build's `tp/N`, for the analysis that
//! sharded the forward and for every bundle that runs it. Nothing here reads a world back.

use scratchy_subtile::lower::LoweringInput;
use scratchy_subtile::superdsc_opspec::{DataFormat, Fp16};
use scratchy_subtile::tensor_parallel::{self, ShardRefusal, ShardedForward, SplitGranule};

use crate::bundle_code::collective::{
    Fragment, FragmentRefusal, Payload, RecvWindow, Tp, fragments,
};

/// The narrowest column block spyre splits a tensor at: one fp16 stick. A per-rank extent that is
/// not whole sticks is not a layout the device's tiles can address, so the analysis replicates such
/// an op instead of splitting it.
pub const SPYRE_SPLIT_GRANULE: SplitGranule =
    SplitGranule::of::<{ <Fp16 as DataFormat>::ELEMS_PER_STICK }>();

/// ⭐ THE ONE CUT OF A COLLECTIVE'S OPERAND — the consume launches the emitter mints for a
/// `[rows, cols]` operand, and the fragments the window is sized over, are this function's output
/// and nothing else's, so no fragment can be wider than the window it lands in.
///
/// ⛔ CUT AT ONE WINDOW BLOCK (`RecvWindow::block_bytes_for` — `WINDOW_ROWS` rows of the operand),
/// never at the wire: a send moves any payload as its run of messages (`Payload::messages`), but
/// two fragments of one collective cannot share its cut — the peer's next fragment can land while
/// this rank still consumes the last (a wdone says "landed", never "read"). Every operand a bundle
/// hands a collective has at most `WINDOW_ROWS` rows, so the cut is always ONE fragment.
pub fn consume_fragments<T: Tp>(rows: u32, cols: u32) -> Result<Vec<Fragment>, FragmentRefusal> {
    match RecvWindow::<T>::block_bytes_for(cols) {
        Some(block) => fragments(rows, cols, block),
        None => Err(FragmentRefusal::RaggedCols { cols }),
    }
}

/// Why a tensor-parallel bake cannot be planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TpBakeRefusal {
    Shard(ShardRefusal),
    Fragment {
        collective: usize,
        why: FragmentRefusal,
    },
}

/// A forward sharded for world `T`, with the receive window its collectives need.
#[derive(Clone, Debug)]
pub struct TpBake<T: Tp> {
    pub sharded: ShardedForward<T>,
    /// One sender's block of the receive window the forward's collectives need —
    /// [`RecvWindow::block_bytes_for`] its widest collective operand, the block the layout places
    /// one of per (cut, sender). `None` exactly when the forward has no collective (a world of one).
    pub window_block: Option<Payload>,
}

/// Shard `input` for world `T` and size the receive window its collectives need.
pub fn tp_bake<T: Tp>(input: &LoweringInput) -> Result<TpBake<T>, TpBakeRefusal> {
    let () = T::CHECKED;
    let sharded = tensor_parallel::shard::<T>(input, SPYRE_SPLIT_GRANULE, &[])
        .map_err(TpBakeRefusal::Shard)?;
    let shapes = tensor_parallel::op_shapes(&sharded.local);
    let mut widest_cols: Option<u32> = None;
    for (collective, c) in sharded.collectives.iter().enumerate() {
        let (rows, cols) = match c.reads {
            scratchy_subtile::lower::InputRef::Op(r) => shapes[r],
            scratchy_subtile::lower::InputRef::Ext(e) => {
                let s = sharded.local.sources[e];
                (s.rows, s.cols)
            }
        };
        consume_fragments::<T>(rows, cols)
            .map_err(|why| TpBakeRefusal::Fragment { collective, why })?;
        widest_cols = Some(widest_cols.map_or(cols, |w| w.max(cols)));
    }
    Ok(TpBake {
        sharded,
        window_block: widest_cols.and_then(RecvWindow::<T>::block_bytes_for),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle_code::collective::{Rank, TpWorld, ranks};
    use scratchy_subtile::fixtures::one_layer_input;

    /// Run `check` at every world width spyre's tests cover: no law here may hold at one width
    /// only.
    macro_rules! at_every_world {
        ($check:ident) => {
            $check::<TpWorld<2>>();
            $check::<TpWorld<4>>();
            $check::<TpWorld<8>>();
        };
    }

    /// The extent pair every `lower_graph_*` call below states for its graph: these fixtures are
    /// ONE request of ONE row (m=1, no AttnDecode — the pair still must describe the graph it is
    /// handed, and it does: one row, one request).
    fn one_row_one_request() -> scratchy_subtile::sdsc_abstract::QueryExtent {
        scratchy_subtile::sdsc_abstract::QueryExtent::of_baked_rung(
            scratchy_subtile::sdsc_abstract::RungWidth::of_baked_rows(1)
                .expect("the fixtures' row count is nonzero"),
            scratchy_subtile::sdsc_abstract::RungCap::ONE,
        )
        .expect("one row at one request fits a u32")
    }

    /// One rank: no collective, no window — the tp=1 identity at the plan's level.
    #[test]
    fn one_rank_needs_no_window() {
        let b = tp_bake::<TpWorld<1>>(&one_layer_input()).unwrap();
        assert!(b.sharded.collectives.is_empty());
        assert!(b.window_block.is_none());
    }

    /// Any world on the decode layer: each all-reduce sends one rank's `[1, hidden]` fp16 partial
    /// in one fragment, and a window block holds `WINDOW_ROWS` such rows — the widest prompt
    /// chunk's operand, so the decode bundle and every prefill rung place the same window.
    #[test]
    fn a_world_sizes_its_window_blocks_to_the_widest_chunk_of_hidden_rows() {
        use crate::bundle_code::collective::WINDOW_ROWS;
        fn check<T: Tp>() {
            let b = tp_bake::<T>(&one_layer_input()).unwrap();
            assert_eq!(b.sharded.collectives.len(), 2, "world {}", T::RANKS);
            let hidden_bytes = 2048 * u64::from(<Fp16 as DataFormat>::WORD_LENGTH);
            assert_eq!(
                b.window_block.unwrap().get(),
                hidden_bytes * u64::from(WINDOW_ROWS)
            );
            assert_eq!(consume_fragments::<T>(1, 2048).unwrap().len(), 1);
            assert_eq!(consume_fragments::<T>(WINDOW_ROWS, 2048).unwrap().len(), 1);
        }
        at_every_world!(check);
    }

    /// ⭐ THE EMISSION, END TO END ON THE HOST, AT EVERY WORLD: the analysis's decode layer,
    /// lowered to SubtileIR, through the producer's own node lowering and the launch index. Each
    /// all-reduce becomes, PER RANK, `W - 1` device programs — the rank-order chain of adds over
    /// every rank's share — all carrying ONE consume marker that plays at that rank. Its sends are
    /// no program at all: F-send control blocks the CB pipeline issues from what the consume names.
    /// Each rank's consume is its own launch group.
    #[test]
    fn an_all_reduce_lowers_to_one_consume_group_per_rank_naming_its_sends() {
        use crate::lower_subtile_tape_to_ktir::{NodeLowering, lower_one_node};
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::collective::{CollectiveKind, WindowCut};
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::place::PlaceId;
        use scratchy_subtile::subtile_ir::{SubOp, lower_region};

        fn check<T: Tp>() {
            let n = T::RANKS as usize;
            let b = tp_bake::<T>(&one_layer_input()).unwrap();
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let mut ops = Vec::new();
            let mut operands = Vec::new();
            for node in ir.nodes.iter().filter(|n| matches!(n.op, SubOp::AllReduce)) {
                operands.push(PlaceId::Act(node.inputs[0].tensor.index() as u32));
                match lower_one_node::<T, _>(
                    node,
                    &ir,
                    ActiveCap::FULL,
                    false,
                    one_row_one_request(),
                ) {
                    NodeLowering::Ops(v) => ops.extend(v),
                    NodeLowering::Unhandled(why) => panic!("the collective did not lower: {why}"),
                    NodeLowering::HostRouted(k) => panic!("the collective was host-routed: {k}"),
                }
            }
            assert_eq!(
                ops.len(),
                operands.len() * n * (n - 1),
                "world {n}: per all-reduce, per rank, W - 1 adds — and no send program"
            );
            for (i, e) in ops.iter().enumerate() {
                let c = e.tp_consume.expect("every program is a consume");
                let k = i / (n * (n - 1));
                assert_eq!(
                    (c.kind, c.operand),
                    (CollectiveKind::AllReduce, operands[k])
                );
                assert_eq!(
                    c.cut,
                    WindowCut::of_collective(k),
                    "collectives alternate cuts"
                );
                assert_eq!(
                    c.fragment,
                    consume_fragments::<T>(1, 2048).unwrap()[0],
                    "the whole partial"
                );
                assert_eq!(
                    c.plays_at.get() as usize,
                    (i % (n * (n - 1))) / (n - 1),
                    "each rank's chain plays at that rank"
                );
            }
            // Every rank's chain reads every PEER's block of the cut, and never its own.
            for (k, chain) in ops.chunks(n * (n - 1)).enumerate() {
                for me in ranks::<T>() {
                    let mine = &chain[me.get() as usize * (n - 1)..][..n - 1];
                    let read: std::collections::BTreeSet<u32> = mine
                        .iter()
                        .flat_map(|e| e.ktir.as_ref().expect("a program").bindings.iter())
                        .map(|b| b.get())
                        .collect();
                    let cut = WindowCut::of_collective(k);
                    for s in ranks::<T>() {
                        assert_eq!(
                            read.contains(&cut.block_tid(s)),
                            s != me,
                            "world {n}: rank {} reads rank {}'s block iff it is a peer",
                            me.get(),
                            s.get()
                        );
                    }
                }
            }
            let shifts = launch_index(&ops, FoldGrouping::Split);
            assert_eq!(
                shifts.len(),
                operands.len() * n,
                "one launch group per (collective, rank)"
            );
        }
        at_every_world!(check);
    }

    /// ⭐ THROUGH THE DOOR, AS A BAKE LOWERS IT: the consume marker survives the SuperDSC door, and
    /// the launch index makes ONE group per (collective, rank) whose shifts carry the marker.
    #[test]
    fn the_consume_survives_the_door_as_one_group_per_rank() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::ktir_node::ActiveCap;
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::lower_region;

        fn check<T: Tp>() {
            let b = tp_bake::<T>(&mlp_sublayer(2048, 8192)).unwrap();
            assert_eq!(b.sharded.collectives.len(), 1, "one all-reduce, after down");
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
            let (ops, _layout) = lower_graph_to_superdsc::<T, _>(
                &ir,
                &weight_ids,
                ActiveCap::FULL,
                false,
                one_row_one_request(),
            )
            .unwrap();
            let shifts = launch_index(&ops, FoldGrouping::Split);
            let consume_groups: Vec<_> = shifts.iter().filter_map(|s| s.tp).collect();
            assert_eq!(
                consume_groups.len(),
                T::RANKS as usize,
                "one launch group per rank's consume"
            );
            let mut ranks: Vec<u32> = consume_groups.iter().map(|c| c.plays_at.get()).collect();
            ranks.sort_unstable();
            assert_eq!(ranks, (0..T::RANKS).collect::<Vec<_>>());
        }
        at_every_world!(check);
    }

    /// ⭐ THE ALL-GATHER, THROUGH THE DOOR, AT EVERY WORLD: the vocabulary projection's gather is
    /// ONE consume group PER RANK — `W` block copies, every rank's block to its columns — and each
    /// rank's pipeline plays its own group alone: a send to every peer, a wait on every peer.
    #[test]
    fn an_all_gather_is_one_consume_group_per_rank() {
        use crate::bundle_code::collective::{CbStep, CollectiveKind, cb_pipeline};
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::ktir_node::ActiveCap;
        use scratchy_subtile::fixtures::vocab_projection;
        use scratchy_subtile::subtile_ir::lower_region;
        use scratchy_subtile::tensor_parallel::CollectiveKind as Shared;

        fn check<T: Tp>() {
            let n = T::RANKS as usize;
            let b = tp_bake::<T>(&vocab_projection(2048, 8192)).unwrap();
            let kinds: Vec<Shared> = b.sharded.collectives.iter().map(|c| c.kind).collect();
            assert_eq!(kinds, vec![Shared::AllGather], "the result is gathered");
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
            let (ops, _) = lower_graph_to_superdsc::<T, _>(
                &ir,
                &weight_ids,
                ActiveCap::FULL,
                false,
                one_row_one_request(),
            )
            .unwrap();
            let shifts = launch_index(&ops, FoldGrouping::Split);
            let consumes: Vec<_> = shifts.iter().filter_map(|s| s.tp).collect();
            assert_eq!(consumes.len(), n, "one consume group per rank");
            assert!(consumes.iter().all(|c| c.kind == CollectiveKind::AllGather));
            let groups: Vec<crate::bundle_code::LaunchGroup<'static, T>> = shifts
                .iter()
                .map(|kv| crate::bundle_code::LaunchGroup {
                    kv: *kv,
                    ..Default::default()
                })
                .collect();
            for id in ranks::<T>() {
                let steps: Vec<CbStep<T>> = cb_pipeline(&groups, id).collect();
                let count = |f: fn(&CbStep<T>) -> bool| steps.iter().filter(|s| f(s)).count();
                assert_eq!(
                    (
                        count(|s| matches!(s, CbStep::Send { .. })),
                        count(|s| matches!(s, CbStep::Wait { .. }))
                    ),
                    (n - 1, n - 1),
                    "world {n}: rank {} sends to and waits on every peer once",
                    id.get()
                );
                assert_eq!(
                    steps.len(),
                    groups.len() - (n - 1) + 2 * (n - 1),
                    "every other rank's consume is skipped"
                );
            }
        }
        at_every_world!(check);
    }

    /// Core 0's start address of every tensor a descriptor states, in its `scheduleTree_` order.
    fn core0_starts(v: &serde_json::Value, out: &mut Vec<u64>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, x) in m {
                    if k == "startAddressCoreCorelet_" {
                        if let Some(s) = x
                            .get("data_")
                            .and_then(|d| d.get("[0, 0, 0]"))
                            .and_then(|s| s.as_str())
                            .and_then(|s| s.parse::<u64>().ok())
                        {
                            out.push(s);
                        }
                    } else {
                        core0_starts(x, out);
                    }
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| core0_starts(x, out)),
            _ => {}
        }
    }

    /// Every `startAddressCoreCorelet_.data_` value under `v`.
    fn collect_starts(v: &serde_json::Value, out: &mut Vec<u64>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, x) in m {
                    if k == "startAddressCoreCorelet_" {
                        if let Some(d) = x.get("data_").and_then(|d| d.as_object()) {
                            out.extend(d.values().filter_map(|s| s.as_str()?.parse::<u64>().ok()));
                        }
                    } else {
                        collect_starts(x, out);
                    }
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| collect_starts(x, out)),
            _ => {}
        }
    }

    /// Every core-0 start address of the descriptors one rank's consume group renders to.
    fn rank_group_starts<T: Tp>(
        ops: &[ktir_superdsc::emit::EmittedOp<T>],
        rank: u32,
        each: bool,
    ) -> Vec<Vec<u64>> {
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, render_dxp_input};
        let group: Vec<_> = ops
            .iter()
            .filter(|e| e.tp_consume.is_some_and(|c| c.plays_at.get() == rank))
            .map(|e| e.shallow_copy())
            .collect();
        let rendered = render_dxp_input(&group, FoldGrouping::Split).unwrap();
        let mut out = Vec::new();
        for g in &rendered {
            for (name, body) in &g.files {
                if !name.ends_with(".json") {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(body).unwrap();
                let mut firsts = Vec::new();
                if each {
                    collect_starts(&v, &mut firsts);
                } else {
                    core0_starts(&v, &mut firsts);
                }
                out.push(firsts);
            }
        }
        out
    }

    /// ⭐ THE STATIC AUDIT (#1007 §5), AT EVERY WORLD: every peer's block a rank's consume reads is
    /// addressed, in the very input dxp compiles, at its TYPED placement (`SEGMENT_OFFSETS[segment]`
    /// plus the offset, the emitter's own segment law), one stick per core, and the rank's OWN block
    /// is never read. (Compiled by hand with `dxp_standalone` through the `dump_tp_consume_sdsc`
    /// example.)
    #[test]
    fn each_consume_reads_every_peer_block_at_its_typed_placement() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::wire::SEGMENT_OFFSETS;
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::lower_region;

        fn check<T: Tp>() {
            let b = tp_bake::<T>(&mlp_sublayer(2048, 8192)).unwrap();
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
            let (ops, layout) = lower_graph_to_superdsc::<T, _>(
                &ir,
                &weight_ids,
                ActiveCap::FULL,
                false,
                one_row_one_request(),
            )
            .unwrap();
            let cut = ops.iter().find_map(|e| e.tp_consume).unwrap().cut;
            let at = |tid: u32| {
                let p = layout.placements[&tid];
                (SEGMENT_OFFSETS[p.segment] + p.offset, p.size)
            };
            for me in ranks::<T>() {
                let starts: Vec<u64> = rank_group_starts(&ops, me.get(), true)
                    .into_iter()
                    .flatten()
                    .collect();
                assert!(
                    !starts.is_empty(),
                    "rank {}'s consume states no address",
                    me.get()
                );
                for s in ranks::<T>() {
                    let (want, size) = at(cut.block_tid(s));
                    let reads: Vec<u64> = starts
                        .iter()
                        .copied()
                        .filter(|a| (want..want + size).contains(a))
                        .collect();
                    if s == me {
                        assert!(reads.is_empty(), "rank {} reads its own block", me.get());
                    } else {
                        assert!(
                            reads.contains(&want),
                            "world {}: rank {} reads rank {}'s block from its first byte",
                            T::RANKS,
                            me.get(),
                            s.get()
                        );
                        assert_eq!(reads.len(), 32, "the block's 32 sticks, one per core");
                    }
                }
            }
        }
        at_every_world!(check);
    }

    /// ⭐ THE GATHER'S STATIC AUDIT, AT EVERY WORLD: in the input dxp compiles, rank `me`'s consume
    /// group writes EVERY rank's block at the output's columns `s · lc` — its own from its operand,
    /// each peer's from the first byte of that peer's window block — every address at its typed
    /// placement (a column block of a one-row fp16 tensor starts `col · 2` bytes in).
    #[test]
    fn each_ranks_gather_writes_every_block_at_its_columns() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::wire::SEGMENT_OFFSETS;
        use scratchy_subtile::fixtures::vocab_projection;
        use scratchy_subtile::subtile_ir::{SubOp, lower_region};

        fn check<T: Tp>() {
            let v = 8192u64;
            let lc = v / u64::from(T::RANKS);
            let b = tp_bake::<T>(&vocab_projection(2048, v as u32)).unwrap();
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
            let (ops, layout) = lower_graph_to_superdsc::<T, _>(
                &ir,
                &weight_ids,
                ActiveCap::FULL,
                false,
                one_row_one_request(),
            )
            .unwrap();
            let at = |tid: u32| {
                let p = layout.placements[&tid];
                SEGMENT_OFFSETS[p.segment] + p.offset
            };
            let cut = ops.iter().find_map(|e| e.tp_consume).unwrap().cut;
            let out = at(ir.result.index() as u32);
            let local = at(ir
                .nodes
                .iter()
                .find(|n| matches!(n.op, SubOp::AllGather { .. }))
                .unwrap()
                .inputs[0]
                .tensor
                .index() as u32);
            for me in ranks::<T>() {
                let mut copies: Vec<(u64, u64)> = rank_group_starts(&ops, me.get(), false)
                    .into_iter()
                    .filter_map(|f| match f[..] {
                        [src, dst] => Some((src, dst)),
                        _ => None,
                    })
                    .collect();
                copies.sort_unstable();
                let mut want: Vec<(u64, u64)> = ranks::<T>()
                    .map(|s| {
                        let src = if s == me { local } else { at(cut.block_tid(s)) };
                        (src, out + u64::from(s.get()) * lc * 2)
                    })
                    .collect();
                want.sort_unstable();
                assert_eq!(
                    copies,
                    want,
                    "world {}: rank {} must place every rank's block at that rank's columns",
                    T::RANKS,
                    me.get()
                );
            }
        }
        at_every_world!(check);
    }

    /// ⭐ THE SEND'S STATIC AUDIT, AT EVERY WORLD: the F-sends move what the consume names — the
    /// partial, as one fragment from its first byte, all of it — and every rank's consume, in the
    /// very input dxp compiles, reads that partial at the SAME typed placement. So the bytes a rank's
    /// sends take from its card are the bytes its own consume adds.
    #[test]
    fn the_sends_move_the_partial_the_consume_reads() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::wire::SEGMENT_OFFSETS;
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::{SubOp, lower_region};

        fn check<T: Tp>() {
            let b = tp_bake::<T>(&mlp_sublayer(2048, 8192)).unwrap();
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
            let (ops, layout) = lower_graph_to_superdsc::<T, _>(
                &ir,
                &weight_ids,
                ActiveCap::FULL,
                false,
                one_row_one_request(),
            )
            .unwrap();
            let partial = ir
                .nodes
                .iter()
                .find(|n| matches!(n.op, SubOp::AllReduce))
                .unwrap()
                .inputs[0]
                .tensor
                .index() as u32;
            let p = layout.placements[&partial];
            let at_partial = SEGMENT_OFFSETS[p.segment] + p.offset;
            for me in ranks::<T>() {
                let c = ops
                    .iter()
                    .filter_map(|e| e.tp_consume)
                    .find(|c| c.plays_at == me)
                    .unwrap();
                assert_eq!(c.operand, ktir_superdsc::place::PlaceId::Act(partial));
                assert_eq!(
                    c.fragment.byte_offset(),
                    0,
                    "one fragment, from its first byte"
                );
                assert_eq!(c.fragment.bytes().get(), p.size, "the whole partial");
                let firsts: Vec<u64> = rank_group_starts(&ops, me.get(), false)
                    .into_iter()
                    .flatten()
                    .collect();
                assert!(
                    firsts.contains(&at_partial),
                    "world {}: rank {}'s consume reads its partial at its typed placement",
                    T::RANKS,
                    me.get()
                );
            }
        }
        at_every_world!(check);
    }

    /// ⭐ THE KTIR PATH GROUPS A CONSUME THE SAME WAY, AT EVERY WORLD: `ktir_groups` makes one group
    /// per program, except a consume's programs, which are ONE group per rank — a rank's gather is
    /// every block's copy, and splitting them would put a send and a wait per peer in front of each
    /// (the extra waits would never be answered). So each rank still sends to and waits on each
    /// peer once.
    #[test]
    fn the_ktir_path_makes_one_group_per_rank_of_a_gather() {
        use crate::bundle_code::collective::{CbStep, cb_pipeline};
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_ktir;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, ktir_groups};
        use ktir_superdsc::ktir_node::ActiveCap;
        use scratchy_subtile::fixtures::vocab_projection;
        use scratchy_subtile::subtile_ir::lower_region;

        fn check<T: Tp>() {
            let n = T::RANKS as usize;
            let b = tp_bake::<T>(&vocab_projection(2048, 8192)).unwrap();
            let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
            let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
            let (ops, _) = lower_graph_to_ktir::<T, _>(
                &ir,
                &weight_ids,
                ActiveCap::FULL,
                false,
                one_row_one_request(),
            )
            .unwrap();
            let groups = ktir_groups(&ops, FoldGrouping::Split, None).unwrap();
            let consumes: Vec<_> = groups.iter().filter(|g| g.kv.tp.is_some()).collect();
            assert_eq!(consumes.len(), n, "one consume group per rank");
            assert!(
                consumes.iter().all(|g| g.programs.len() == n),
                "each holds its rank's {n} block copies"
            );
            for id in ranks::<T>() {
                let steps: Vec<CbStep<T>> = cb_pipeline(&groups, id).collect();
                let count = |f: fn(&CbStep<T>) -> bool| steps.iter().filter(|s| f(s)).count();
                assert_eq!(
                    (
                        count(|s| matches!(s, CbStep::Send { .. })),
                        count(|s| matches!(s, CbStep::Wait { .. }))
                    ),
                    (n - 1, n - 1),
                    "rank {} sends to and waits on every peer once",
                    id.get()
                );
            }
        }
        at_every_world!(check);
    }

    /// The window's placement, at every world: a sharded forward's layout carries one block per
    /// (cut, sender) in the weight segment's base region — equal, adjacent, in order, sized by the
    /// plan's own law — above every weight; an unsharded forward's layout has none.
    #[test]
    fn a_sharded_layout_places_one_block_per_sender_above_the_weights() {
        use crate::bundle_code::collective::{WINDOW_CUTS, WindowCut};
        use crate::lower_subtile_tape_to_superdsc::compute_bundle_layout;
        use ktir_superdsc::placement::SegRole;
        use scratchy_subtile::subtile_ir::lower_region;

        // The MLP sublayer: one all-reduce after down, and no attention — the window's placement is
        // the collective's, so a fixture whose cache the paged layout would refuse proves nothing
        // more about it. Its gemm weights: gate, up, down.
        use scratchy_subtile::fixtures::mlp_sublayer;
        let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
        fn layout_of<T: Tp>(
            weight_ids: &std::collections::HashSet<u32>,
        ) -> ktir_superdsc::placement::BundleLayout {
            let nb = std::num::NonZeroU32::new(128).unwrap();
            let b = tp_bake::<T>(&mlp_sublayer(2048, 8192)).unwrap();
            compute_bundle_layout::<T, _>(
                &lower_region(&b.sharded.local, nb),
                weight_ids,
                scratchy_triton_splice::PagedRows::Sequence,
                &[],
                one_row_one_request(),
            )
            .expect("the sublayer lays out")
        }
        let rank0 = Rank::<TpWorld<1>>::of::<0>();
        assert!(
            !layout_of::<TpWorld<1>>(&weight_ids)
                .placements
                .contains_key(&WindowCut::of_collective(0).block_tid(rank0))
        );
        fn check<T: Tp>(weight_ids: &std::collections::HashSet<u32>) {
            let layout = layout_of::<T>(weight_ids);
            let blocks: Vec<u32> = (0..WINDOW_CUTS)
                .flat_map(|c| ranks::<T>().map(move |r| WindowCut::of_collective(c).block_tid(r)))
                .collect();
            let first = layout.placements[&blocks[0]];
            assert_eq!(first.segment, SegRole::Weight.segment());
            assert_eq!(first.offset % 128, 0);
            let planned = tp_bake::<T>(&mlp_sublayer(2048, 8192)).unwrap();
            assert_eq!(Some(first.size), planned.window_block.map(|b| b.get()));
            for (i, tid) in blocks.iter().enumerate() {
                let p = layout.placements[tid];
                assert_eq!((p.segment, p.bank, p.size), (first.segment, 0, first.size));
                assert_eq!(p.offset, first.offset + i as u64 * first.size, "block {i}");
            }
            let weights_end = layout
                .placements
                .values()
                .filter(|p| !blocks.contains(&p.tid) && p.segment == first.segment)
                .map(|p| p.offset + p.size)
                .max()
                .unwrap_or(0);
            assert!(first.offset >= weights_end, "the window overlaps a weight");
        }
        check::<TpWorld<2>>(&weight_ids);
        check::<TpWorld<4>>(&weight_ids);
        check::<TpWorld<8>>(&weight_ids);
    }
}
