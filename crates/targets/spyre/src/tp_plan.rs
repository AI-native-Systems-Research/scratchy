// SPDX-License-Identifier: Apache-2.0
//! Spyre's half of tensor parallelism: the target facts the shared analysis takes as INPUT, and the
//! bake-time collective facts derived from its OUTPUT.
//!
//! The analysis (`scratchy_subtile::tensor_parallel`) is target-neutral and decides where every
//! collective goes. Spyre contributes exactly two facts to it — which worlds it bakes
//! ([`SpyreWorld`], from the `SPYRE_TP_WORLDS` row) and the column granule it can split at
//! ([`SPYRE_SPLIT_GRANULE`], one fp16 stick) — and reads back the sharded forward: each collective
//! becomes consume launches over stick-column fragments of its operand ([`consume_fragments`]), and
//! the receive window is as wide as the widest of them ([`TpBake::window_bytes`]). Nothing here
//! decides a placement.

use scratchy_subtile::lower::LoweringInput;
use scratchy_subtile::superdsc_opspec::{DataFormat, Fp16};
use scratchy_subtile::tensor_parallel::{
    self, ShardRefusal, ShardedForward, SplitGranule, WorldSize,
};

use crate::bundle_code::collective::{Fragment, FragmentRefusal, TpWorld, WireBytes, fragments};

/// THE WORLD, AS BOTH HALVES NAME IT — the analysis's [`WorldSize`] and the bundle's [`TpWorld`],
/// minted together from ONE const `W`, so the forward that was sharded and the bundle that runs it
/// can never be for two different worlds. A `W` spyre does not bake does not compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpyreWorld {
    plan: TpWorld,
    analysis: WorldSize,
}

impl SpyreWorld {
    pub const fn of<const W: u32>() -> Self {
        Self {
            plan: TpWorld::of::<W>(),
            analysis: WorldSize::of::<W>(),
        }
    }

    pub const fn plan(self) -> TpWorld {
        self.plan
    }

    pub const fn analysis(self) -> WorldSize {
        self.analysis
    }
}

/// The narrowest column block spyre splits a tensor at: one fp16 stick. A per-rank extent that is
/// not whole sticks is not a layout the device's tiles can address, so the analysis replicates such
/// an op instead of splitting it.
pub const SPYRE_SPLIT_GRANULE: SplitGranule =
    SplitGranule::of::<{ <Fp16 as DataFormat>::ELEMS_PER_STICK }>();

/// ⭐ THE ONE CUT OF A COLLECTIVE'S OPERAND — the consume launches the emitter mints for a
/// `[rows, cols]` operand, and the fragments the window is sized over, are this function's output
/// and nothing else's, so no fragment can be wider than the window it lands in.
pub fn consume_fragments(rows: u32, cols: u32) -> Result<Vec<Fragment>, FragmentRefusal> {
    fragments(rows, cols, WireBytes::CAP)
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

/// A forward sharded for one spyre world, with the receive window its collectives need.
#[derive(Clone, Debug)]
pub struct TpBake {
    pub world: SpyreWorld,
    pub sharded: ShardedForward,
    /// The widest fragment any collective of the forward sends — the window's size. `None` exactly
    /// when the forward has no collective (a world of one).
    pub window_bytes: Option<WireBytes>,
}

/// Shard `input` for `world` and size the receive window its collectives need.
pub fn tp_bake(input: &LoweringInput, world: SpyreWorld) -> Result<TpBake, TpBakeRefusal> {
    let sharded = tensor_parallel::shard(input, world.analysis, SPYRE_SPLIT_GRANULE)
        .map_err(TpBakeRefusal::Shard)?;
    let shapes = tensor_parallel::op_shapes(&sharded.local);
    let mut window_bytes: Option<WireBytes> = None;
    for (collective, c) in sharded.collectives.iter().enumerate() {
        let (rows, cols) = shapes[c.reads];
        let frags = consume_fragments(rows, cols)
            .map_err(|why| TpBakeRefusal::Fragment { collective, why })?;
        for f in frags {
            if window_bytes.is_none_or(|w| f.bytes().get() > w.get()) {
                window_bytes = Some(f.bytes());
            }
        }
    }
    Ok(TpBake {
        world,
        sharded,
        window_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_subtile::fixtures::one_layer_input;

    /// One rank: no collective, no window — the tp=1 identity at the plan's level.
    #[test]
    fn one_rank_needs_no_window() {
        let b = tp_bake(&one_layer_input(), SpyreWorld::of::<1>()).unwrap();
        assert!(b.sharded.collectives.is_empty());
        assert!(b.window_bytes.is_none());
    }

    /// Two ranks on the decode layer: each all-reduce sends one rank's `[1, hidden]` fp16 partial
    /// in one fragment, so the window is exactly that wide.
    #[test]
    fn two_ranks_size_the_window_to_one_hidden_row() {
        let b = tp_bake(&one_layer_input(), SpyreWorld::of::<2>()).unwrap();
        assert_eq!(b.sharded.collectives.len(), 2);
        let hidden_bytes = 2048 * u64::from(<Fp16 as DataFormat>::WORD_LENGTH);
        assert_eq!(b.window_bytes.unwrap().get(), hidden_bytes);
        assert_eq!(consume_fragments(1, 2048).unwrap().len(), 1);
    }

    /// ⭐ THE EMISSION, END TO END ON THE HOST: the analysis's two-rank decode layer, lowered to
    /// SubtileIR, through the producer's own node lowering and the launch index. Each all-reduce
    /// becomes TWO device programs (one fragment at decode): the SEND, a block copy of the partial
    /// it reduces into the receive window (launched against the PEER's weight segment — device
    /// p2p), then the CONSUME, the sum over the partial and the window. Each is its own launch
    /// group, and the groups' shifts carry the markers the CB pipeline turns into
    /// `send, wait, launch`.
    #[test]
    fn an_all_reduce_lowers_to_a_send_group_then_a_consume_group() {
        use crate::lower_subtile_tape_to_ktir::{NodeLowering, lower_one_node};
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::collective::CollectiveKind;
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::place::PlaceId;
        use ktir_superdsc::reserved_tids::TP_RECV_TID;
        use scratchy_subtile::subtile_ir::{SubOp, lower_region};

        let b = tp_bake(&one_layer_input(), SpyreWorld::of::<2>()).unwrap();
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let mut ops = Vec::new();
        let mut operands = Vec::new();
        for node in ir.nodes.iter().filter(|n| matches!(n.op, SubOp::AllReduce)) {
            operands.push(PlaceId::Act(node.inputs[0].tensor.index() as u32));
            match lower_one_node(node, &ir, ActiveCap::FULL, false) {
                NodeLowering::Ops(v) => ops.extend(v),
                NodeLowering::Unhandled(why) => panic!("the collective did not lower: {why}"),
                NodeLowering::HostRouted(k) => panic!("the collective was host-routed: {k}"),
            }
        }
        assert_eq!(ops.len(), 4, "a send and a consume per all-reduce at decode");
        for (pair, operand) in ops.chunks(2).zip(&operands) {
            let [send, consume] = pair else { unreachable!() };
            let s = send.tp_send.expect("the first program is the send");
            assert_eq!((s.kind, s.operand), (CollectiveKind::AllReduce, *operand));
            assert!(send.tp_consume.is_none());
            let c = consume.tp_consume.expect("the second program is the consume");
            assert_eq!((c.kind, c.operand), (CollectiveKind::AllReduce, *operand));
            for (e, what) in [(send, "the send writes"), (consume, "the consume reads")] {
                let k = e.ktir.as_ref().expect("a program");
                assert!(
                    k.bindings.iter().any(|b| b.get() == TP_RECV_TID),
                    "{what} the receive window"
                );
            }
        }
        let shifts = launch_index(&ops, FoldGrouping::Split);
        assert_eq!(shifts.len(), 4, "each send and each consume is its own launch group");
        for (s, e) in shifts.iter().zip(&ops) {
            assert_eq!((s.send, s.tp), (e.tp_send, e.tp_consume));
        }
    }

    /// ⭐ THROUGH THE DOOR, AS A BAKE LOWERS IT — the case the producer-side test cannot see. The
    /// SuperDSC door mints fresh descriptor ops from each program, so the consume marker must be
    /// carried onto every op a consume program lowers to (`ktir_superdsc_door::lower_op`), and the
    /// launch index must make ONE group of them per consume whose shifts carry the marker. Before
    /// the door carried it, this forward lowered to zero consume groups: every consume a fusable
    /// `Pure` launch with no send and no wait in front of it.
    #[test]
    fn the_consume_survives_the_door_as_one_group_per_consume() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::ktir_node::ActiveCap;
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::lower_region;

        let b = tp_bake(&mlp_sublayer(2048, 8192), SpyreWorld::of::<2>()).unwrap();
        assert_eq!(b.sharded.collectives.len(), 1, "one all-reduce, after down");
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
        let (ops, _layout) =
            lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        assert!(
            ops.iter().any(|e| e.tp_consume.is_some()),
            "the consume marker did not survive the door"
        );
        let shifts = launch_index(&ops, FoldGrouping::Split);
        let consume_groups: Vec<_> = shifts.iter().filter_map(|s| s.tp).collect();
        assert_eq!(consume_groups.len(), 1, "one launch group per consume");
        let group = shifts.iter().position(|s| s.tp.is_some()).unwrap();
        let after = &shifts[group + 1..];
        assert!(
            after.iter().all(|s| s.tp.is_none()),
            "no second group sends the same fragment"
        );
    }

    /// ⭐ THE STATIC AUDIT (#1007 §5): the consume's window operand is addressed, in the very input
    /// dxp compiles, at its TYPED placement — `SEGMENT_OFFSETS[segment] + offset`, the emitter's own
    /// 16 GiB segment law — and at no other. Read off the rendered descriptor's per-core start addresses, so a consume
    /// whose window read was re-derived anywhere between the layout and the wire fails here instead
    /// of on a card. (Compiled by hand with `dxp_standalone` through the
    /// `dump_tp_consume_sdsc` example: rc 0, and the same address in its schedule tree.)
    #[test]
    fn the_consume_reads_the_window_at_its_typed_placement() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, render_dxp_input};
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::wire::SEGMENT_OFFSETS;
        use ktir_superdsc::reserved_tids::TP_RECV_TID;
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::lower_region;

        let b = tp_bake(&mlp_sublayer(2048, 8192), SpyreWorld::of::<2>()).unwrap();
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
        let (ops, layout) =
            lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        let window = layout.placements[&TP_RECV_TID];
        let want = SEGMENT_OFFSETS[window.segment] + window.offset;
        let consumes: Vec<_> = ops
            .iter()
            .filter(|e| e.tp_consume.is_some())
            .map(|e| e.shallow_copy())
            .collect();
        let groups = render_dxp_input(&consumes, FoldGrouping::Split).unwrap();
        assert_eq!(groups.len(), 1, "one consume group");
        // Every per-core start address the descriptor states, across all of its tensors.
        let mut starts: Vec<u64> = Vec::new();
        for (name, body) in &groups[0].files {
            if !name.ends_with(".json") {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            collect_starts(&v, &mut starts);
        }
        assert!(!starts.is_empty(), "the descriptor states no start address");
        // The window's first stick is core 0's start, and its 2048 columns are 32 sticks, one per core.
        let window_reads: Vec<u64> = starts
            .iter()
            .copied()
            .filter(|a| (want..want + window.size).contains(a))
            .collect();
        assert!(window_reads.contains(&want), "no core starts at the window's first byte");
        assert_eq!(window_reads.len(), 32, "the window's 32 sticks, one per core");
    }

    /// ⭐ THE GATHER'S STATIC AUDIT: in the input dxp compiles, rank `r`'s consume group writes its
    /// own block at the output's columns `r · lc` and the peer's block — read from the receive
    /// window's first byte — at columns `peer · lc`, every address at its typed placement
    /// (`SEGMENT_OFFSETS[segment] + offset`, in bytes; a column block of a one-row fp16 tensor
    /// starts `col · 2` bytes in).
    #[test]
    fn each_ranks_gather_writes_its_blocks_at_their_columns() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, render_dxp_input};
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::reserved_tids::TP_RECV_TID;
        use ktir_superdsc::wire::SEGMENT_OFFSETS;
        use scratchy_subtile::fixtures::vocab_projection;
        use scratchy_subtile::subtile_ir::lower_region;

        let (v, lc) = (8192u64, 4096u64);
        let b = tp_bake(&vocab_projection(2048, v as u32), SpyreWorld::of::<2>()).unwrap();
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
        let (ops, layout) =
            lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        let at = |tid: u32| {
            let p = layout.placements[&tid];
            SEGMENT_OFFSETS[p.segment] + p.offset
        };
        let (out, window) = (at(ir.result.index() as u32), at(TP_RECV_TID));
        for rank in 0..2u64 {
            let peer = 1 - rank;
            let group: Vec<_> = ops
                .iter()
                .filter(|e| {
                    e.tp_consume
                        .and_then(|c| c.plays_at)
                        .is_some_and(|p| u64::from(p.rank().rank()) == rank)
                })
                .map(|e| e.shallow_copy())
                .collect();
            assert_eq!(group.len(), 2, "rank {rank}'s consume is two block copies");
            let rendered = render_dxp_input(&group, FoldGrouping::Split).unwrap();
            // Each copy as its (source, destination) pair: core 0's start of each tensor, in the
            // descriptor's tensor order — the input, then the output.
            let mut copies: Vec<(u64, u64)> = Vec::new();
            for g in &rendered {
                for (name, body) in &g.files {
                    if !name.ends_with(".json") {
                        continue;
                    }
                    let mut firsts = Vec::new();
                    core0_starts(&serde_json::from_str(body).unwrap(), &mut firsts);
                    if let [src, dst] = firsts[..] {
                        copies.push((src, dst));
                    }
                }
            }
            copies.sort_unstable();
            let local = at(ir.nodes.iter().find(|n| matches!(n.op, scratchy_subtile::subtile_ir::SubOp::AllGather { .. })).unwrap().inputs[0].tensor.index() as u32);
            let mut want = vec![(local, out + rank * lc * 2), (window, out + peer * lc * 2)];
            want.sort_unstable();
            assert_eq!(
                copies, want,
                "rank {rank}: its own block must land at its columns and the peer's, out of the \
                 window, at the peer's"
            );
        }
    }

    /// ⭐ THE SEND'S STATIC AUDIT: in the input dxp compiles, the send is ONE copy from the partial's
    /// own placement to the receive window's — `SEGMENT_OFFSETS[1] + window offset` — the address the
    /// send's launch translates to the PEER's weight segment, so the store lands in the peer's
    /// window. Read as the descriptor's (source, destination) pair.
    #[test]
    fn the_send_copies_the_partial_to_the_window() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, render_dxp_input};
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::reserved_tids::TP_RECV_TID;
        use ktir_superdsc::wire::SEGMENT_OFFSETS;
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::{SubOp, lower_region};

        let b = tp_bake(&mlp_sublayer(2048, 8192), SpyreWorld::of::<2>()).unwrap();
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
        let (ops, layout) =
            lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        let at = |tid: u32| {
            let p = layout.placements[&tid];
            SEGMENT_OFFSETS[p.segment] + p.offset
        };
        let partial = ir
            .nodes
            .iter()
            .find(|n| matches!(n.op, SubOp::AllReduce))
            .unwrap()
            .inputs[0]
            .tensor
            .index() as u32;
        let sends: Vec<_> = ops
            .iter()
            .filter(|e| e.tp_send.is_some())
            .map(|e| e.shallow_copy())
            .collect();
        assert_eq!(sends.len(), 1, "one send per all-reduce");
        let rendered = render_dxp_input(&sends, FoldGrouping::Split).unwrap();
        let mut copies = Vec::new();
        for g in &rendered {
            for (name, body) in &g.files {
                if name.ends_with(".json") {
                    let mut firsts = Vec::new();
                    core0_starts(&serde_json::from_str(body).unwrap(), &mut firsts);
                    if let [src, dst] = firsts[..] {
                        copies.push((src, dst));
                    }
                }
            }
        }
        assert_eq!(copies, vec![(at(partial), at(TP_RECV_TID))]);
        assert_eq!(layout.placements[&TP_RECV_TID].segment, 1, "the window is in segment 1");
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

    /// ⭐ THE ALL-GATHER, THROUGH THE DOOR: a two-rank vocabulary projection gathers its result, and
    /// the gather lowers to ONE consume group PER RANK — this rank's block and the peer's, as two
    /// block copies — and each rank's pipeline plays its own group alone, with one send and one
    /// wait. The copies place rank `r`'s block at columns `r · v/2` on rank `r`'s own consume and
    /// at the same columns of the peer's, so both ranks assemble the same `[1, v]` row.
    #[test]
    fn an_all_gather_is_one_consume_group_per_rank() {
        use crate::bundle_code::collective::{CbStep, CollectiveKind, TpWorld, cb_pipeline};
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::ktir_node::ActiveCap;
        use scratchy_subtile::fixtures::vocab_projection;
        use scratchy_subtile::subtile_ir::lower_region;
        use scratchy_subtile::tensor_parallel::CollectiveKind as Shared;

        let b = tp_bake(&vocab_projection(2048, 8192), SpyreWorld::of::<2>()).unwrap();
        let kinds: Vec<Shared> = b.sharded.collectives.iter().map(|c| c.kind).collect();
        assert_eq!(kinds, vec![Shared::AllGather], "the projection's result is gathered");
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
        let (ops, _) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        let shifts = launch_index(&ops, FoldGrouping::Split);
        let consumes: Vec<_> = shifts.iter().filter_map(|s| s.tp).collect();
        assert_eq!(consumes.len(), 2, "one consume group per rank");
        assert!(consumes.iter().all(|c| c.kind == CollectiveKind::AllGather));
        let groups: Vec<crate::bundle_code::LaunchGroup<'static>> = shifts
            .iter()
            .map(|kv| crate::bundle_code::LaunchGroup {
                kv: *kv,
                ..Default::default()
            })
            .collect();
        for id in TpWorld::of::<2>().ranks() {
            let steps: Vec<CbStep> = cb_pipeline(&groups, id).collect();
            let sends = steps.iter().filter(|s| matches!(s, CbStep::Send { .. })).count();
            let waits = steps.iter().filter(|s| matches!(s, CbStep::Wait { .. })).count();
            assert_eq!((sends, waits), (1, 1), "rank {} sends once and waits once", id.rank());
            // One step per group it plays (the other rank's consume skipped), plus its consume's
            // wait.
            assert_eq!(steps.len(), groups.len() - 1 + 1, "the other rank's consume is skipped");
        }
    }

    /// The window's placement: a sharded forward's layout carries it in the weight segment's base
    /// region, one message wide, above every weight; an unsharded forward's layout has none.
    #[test]
    fn a_sharded_layout_places_the_window_above_the_weights() {
        use crate::bundle_code::collective::RDMA_MESSAGE_CAP_BYTES;
        use crate::lower_subtile_tape_to_superdsc::compute_bundle_layout;
        use ktir_superdsc::placement::SegRole;
        use ktir_superdsc::reserved_tids::TP_RECV_TID;
        use scratchy_subtile::subtile_ir::lower_region;

        // The fixture's gemm weights: q, k, v, o, gate, up, down.
        let weight_ids: std::collections::HashSet<u32> =
            [2u32, 3, 4, 9, 11, 12, 13].into_iter().collect();
        let nb = std::num::NonZeroU32::new(128).unwrap();
        let layout_of = |w: SpyreWorld| {
            let b = tp_bake(&one_layer_input(), w).unwrap();
            compute_bundle_layout(&lower_region(&b.sharded.local, nb), &weight_ids, false, false, &[])
                .expect("the decode layer lays out")
        };
        assert!(!layout_of(SpyreWorld::of::<1>()).placements.contains_key(&TP_RECV_TID));
        let two = layout_of(SpyreWorld::of::<2>());
        let win = two.placements[&TP_RECV_TID];
        assert_eq!(win.segment, SegRole::Weight.segment());
        assert_eq!(win.bank, 0);
        assert_eq!(win.size, RDMA_MESSAGE_CAP_BYTES);
        assert_eq!(win.offset % 128, 0);
        let weights_end = two
            .placements
            .values()
            .filter(|p| p.tid != TP_RECV_TID && p.segment == win.segment)
            .map(|p| p.offset + p.size)
            .max()
            .unwrap_or(0);
        assert!(win.offset >= weights_end, "the window overlaps a weight");
    }
}
