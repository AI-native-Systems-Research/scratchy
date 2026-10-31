// SPDX-License-Identifier: Apache-2.0
//! Spyre's half of tensor parallelism: the target facts the shared analysis takes as INPUT, and the
//! bake-time collective facts derived from its OUTPUT.
//!
//! The analysis (`scratchy_subtile::tensor_parallel`) is target-neutral and decides where every
//! collective goes. Spyre contributes exactly two facts to it — which worlds it bakes
//! ([`SpyreWorld`], from the `SPYRE_TP_WORLDS` row) and the column granule it can split at
//! ([`SPYRE_SPLIT_GRANULE`], one fp16 stick) — and reads back the sharded forward: each collective
//! becomes consume launches over stick-column fragments of its operand ([`consume_fragments`]), and
//! each cut of the receive window is as wide as the widest of them ([`TpBake::window_cut`]). Nothing here
//! decides a placement.

use scratchy_subtile::lower::LoweringInput;
use scratchy_subtile::superdsc_opspec::{DataFormat, Fp16};
use scratchy_subtile::tensor_parallel::{
    self, ShardRefusal, ShardedForward, SplitGranule, WorldSize,
};

use crate::bundle_code::collective::{
    Fragment, FragmentRefusal, Payload, RecvWindow, TpWorld, fragments,
};

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
///
/// ⛔ CUT AT ONE WINDOW CUT (`RecvWindow::cut_bytes_for` — `WINDOW_ROWS` rows of the operand),
/// never at the wire: a send moves any payload as its run of messages (`Payload::messages`), but
/// two fragments of one collective cannot share its cut — the peer's next fragment can land while
/// this rank still consumes the last (a wdone says "landed", never "read"). Every operand a bundle
/// hands a collective has at most `WINDOW_ROWS` rows, so the cut is always ONE fragment.
pub fn consume_fragments(rows: u32, cols: u32) -> Result<Vec<Fragment>, FragmentRefusal> {
    match RecvWindow::cut_bytes_for(cols) {
        Some(window) => fragments(rows, cols, window),
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

/// A forward sharded for one spyre world, with the receive window its collectives need.
#[derive(Clone, Debug)]
pub struct TpBake {
    pub world: SpyreWorld,
    pub sharded: ShardedForward,
    /// One cut of the receive window the forward's collectives need —
    /// [`RecvWindow::cut_bytes_for`] its widest collective operand, the cut the layout places
    /// `WINDOW_CUTS` of. `None` exactly when the forward has no collective (a world of one).
    pub window_cut: Option<Payload>,
}

/// Shard `input` for `world` and size the receive window its collectives need.
pub fn tp_bake(input: &LoweringInput, world: SpyreWorld) -> Result<TpBake, TpBakeRefusal> {
    let sharded = tensor_parallel::shard(input, world.analysis, SPYRE_SPLIT_GRANULE)
        .map_err(TpBakeRefusal::Shard)?;
    let shapes = tensor_parallel::op_shapes(&sharded.local);
    let mut widest_cols: Option<u32> = None;
    for (collective, c) in sharded.collectives.iter().enumerate() {
        let (rows, cols) = shapes[c.reads];
        consume_fragments(rows, cols).map_err(|why| TpBakeRefusal::Fragment { collective, why })?;
        widest_cols = Some(widest_cols.map_or(cols, |w| w.max(cols)));
    }
    Ok(TpBake {
        world,
        sharded,
        window_cut: widest_cols.and_then(RecvWindow::cut_bytes_for),
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
        assert!(b.window_cut.is_none());
    }

    /// Two ranks on the decode layer: each all-reduce sends one rank's `[1, hidden]` fp16 partial
    /// in one fragment, and the window holds `WINDOW_ROWS` such rows — the widest prompt chunk's
    /// operand, so the decode bundle and every prefill rung place the same window. A chunk at the
    /// window's full row count is still one fragment; its send is several messages once it
    /// passes the cap.
    #[test]
    fn two_ranks_size_the_window_to_the_widest_chunk_of_hidden_rows() {
        use crate::bundle_code::collective::{RDMA_MESSAGE_CAP_BYTES, WINDOW_ROWS};
        let b = tp_bake(&one_layer_input(), SpyreWorld::of::<2>()).unwrap();
        assert_eq!(b.sharded.collectives.len(), 2);
        let hidden_bytes = 2048 * u64::from(<Fp16 as DataFormat>::WORD_LENGTH);
        assert_eq!(
            b.window_cut.unwrap().get(),
            hidden_bytes * u64::from(WINDOW_ROWS)
        );
        assert_eq!(consume_fragments(1, 2048).unwrap().len(), 1);
        assert_eq!(consume_fragments(WINDOW_ROWS, 2048).unwrap().len(), 1);
        // An 8192-wide hidden at the full window: 8 MiB, one fragment, sent as two messages.
        let wide = consume_fragments(WINDOW_ROWS, 8192).unwrap();
        assert_eq!(wide.len(), 1);
        assert!(wide[0].bytes().get() > RDMA_MESSAGE_CAP_BYTES);
        assert_eq!(wide[0].bytes().message_count(), 2);
    }

    /// ⭐ THE EMISSION, END TO END ON THE HOST: the analysis's two-rank decode layer, lowered to
    /// SubtileIR, through the producer's own node lowering and the launch index. Each all-reduce
    /// becomes ONE device program (one fragment at decode): the CONSUME, the sum over the partial
    /// and the receive window. Its send is no program at all — an F-send control block (device p2p
    /// RDMA) the CB pipeline issues from what the consume names: the partial, as one fragment. Each
    /// consume is its own launch group, whose shifts carry the marker the CB pipeline turns into
    /// `send, wait, launch`.
    #[test]
    fn an_all_reduce_lowers_to_one_consume_group_naming_its_send() {
        use crate::lower_subtile_tape_to_ktir::{NodeLowering, lower_one_node};
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, launch_index};
        use ktir_superdsc::collective::{CollectiveKind, WindowCut};
        use ktir_superdsc::ktir_node::ActiveCap;
        use ktir_superdsc::place::PlaceId;
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
        assert_eq!(
            ops.len(),
            2,
            "one consume per all-reduce at decode, and no send program"
        );
        for (k, (consume, operand)) in ops.iter().zip(&operands).enumerate() {
            let c = consume.tp_consume.expect("the program is the consume");
            assert_eq!((c.kind, c.operand), (CollectiveKind::AllReduce, *operand));
            assert_eq!(
                c.cut,
                WindowCut::of_collective(k),
                "consecutive collectives alternate cuts"
            );
            assert_eq!(
                c.fragment,
                consume_fragments(1, 2048).unwrap()[0],
                "the whole partial"
            );
            let k = consume.ktir.as_ref().expect("a program");
            assert!(
                k.bindings.iter().any(|b| b.get() == c.cut.tid()),
                "the consume reads its cut of the receive window"
            );
        }
        let shifts = launch_index(&ops, FoldGrouping::Split);
        assert_eq!(shifts.len(), 2, "each consume is its own launch group");
        for (s, e) in shifts.iter().zip(&ops) {
            assert_eq!(s.tp, e.tp_consume);
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
        use scratchy_subtile::fixtures::mlp_sublayer;
        use scratchy_subtile::subtile_ir::lower_region;

        let b = tp_bake(&mlp_sublayer(2048, 8192), SpyreWorld::of::<2>()).unwrap();
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32, 3, 4].into_iter().collect();
        let (ops, layout) =
            lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        let cut = ops.iter().find_map(|e| e.tp_consume).unwrap().cut;
        let window = layout.placements[&cut.tid()];
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
        assert!(
            window_reads.contains(&want),
            "no core starts at the window's first byte"
        );
        assert_eq!(
            window_reads.len(),
            32,
            "the window's 32 sticks, one per core"
        );
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
        let cut = ops.iter().find_map(|e| e.tp_consume).unwrap().cut;
        let (out, window) = (at(ir.result.index() as u32), at(cut.tid()));
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
            let local = at(ir
                .nodes
                .iter()
                .find(|n| matches!(n.op, scratchy_subtile::subtile_ir::SubOp::AllGather { .. }))
                .unwrap()
                .inputs[0]
                .tensor
                .index() as u32);
            let mut want = vec![(local, out + rank * lc * 2), (window, out + peer * lc * 2)];
            want.sort_unstable();
            assert_eq!(
                copies, want,
                "rank {rank}: its own block must land at its columns and the peer's, out of the \
                 window, at the peer's"
            );
        }
    }

    /// ⭐ THE SEND'S STATIC AUDIT: the F-send moves what the consume names — the partial, as one
    /// fragment from its first byte, all of it — and the consume, in the very input dxp compiles,
    /// reads that partial at the SAME typed placement (`SEGMENT_OFFSETS[segment] + offset`) and the
    /// window at its own. So the bytes the send takes from this card are the bytes this card's
    /// consume adds, and the peer's send lands where this consume reads.
    #[test]
    fn the_send_moves_the_partial_the_consume_reads() {
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_superdsc;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, render_dxp_input};
        use ktir_superdsc::ktir_node::ActiveCap;
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
        let consumes: Vec<_> = ops
            .iter()
            .filter(|e| e.tp_consume.is_some())
            .map(|e| e.shallow_copy())
            .collect();
        let c = consumes[0].tp_consume.unwrap();
        assert_eq!(
            c.operand,
            ktir_superdsc::place::PlaceId::Act(partial),
            "the send's operand"
        );
        assert_eq!(
            c.fragment.byte_offset(),
            0,
            "one fragment, from the partial's first byte"
        );
        assert_eq!(
            c.fragment.bytes().get(),
            layout.placements[&partial].size,
            "the whole partial"
        );
        let rendered = render_dxp_input(&consumes, FoldGrouping::Split).unwrap();
        let mut firsts = Vec::new();
        for g in &rendered {
            for (name, body) in &g.files {
                if name.ends_with(".json") {
                    core0_starts(&serde_json::from_str(body).unwrap(), &mut firsts);
                }
            }
        }
        assert!(
            firsts.contains(&at(partial)) && firsts.contains(&at(c.cut.tid())),
            "the consume reads the partial and the window at their typed placements: {firsts:?}"
        );
        assert_eq!(
            layout.placements[&c.cut.tid()].segment,
            1,
            "the window is in segment 1"
        );
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
        assert_eq!(
            kinds,
            vec![Shared::AllGather],
            "the projection's result is gathered"
        );
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
            let sends = steps
                .iter()
                .filter(|s| matches!(s, CbStep::Send { .. }))
                .count();
            let waits = steps
                .iter()
                .filter(|s| matches!(s, CbStep::Wait { .. }))
                .count();
            assert_eq!(
                (sends, waits),
                (1, 1),
                "rank {} sends once and waits once",
                id.rank()
            );
            // One launch per group it plays (the other rank's consume skipped), plus its consume's
            // send and wait.
            assert_eq!(
                steps.len(),
                groups.len() - 1 + 2,
                "the other rank's consume is skipped"
            );
        }
    }

    /// ⭐ THE KTIR PATH GROUPS A CONSUME THE SAME WAY: `ktir_groups` makes one group per program,
    /// except a consume's programs, which are ONE group — a rank's gather is its own block's copy and
    /// the peer's, and two groups would put two sends and two waits in front of them (one wait
    /// would never be answered). So each rank still sends once and waits once.
    #[test]
    fn the_ktir_path_makes_one_group_per_rank_of_a_gather() {
        use crate::bundle_code::collective::{CbStep, TpWorld, cb_pipeline};
        use crate::lower_subtile_tape_to_ktir::lower_graph_to_ktir;
        use crate::lower_subtile_tape_to_superdsc::{FoldGrouping, ktir_groups};
        use ktir_superdsc::ktir_node::ActiveCap;
        use scratchy_subtile::fixtures::vocab_projection;
        use scratchy_subtile::subtile_ir::lower_region;

        let b = tp_bake(&vocab_projection(2048, 8192), SpyreWorld::of::<2>()).unwrap();
        let ir = lower_region(&b.sharded.local, std::num::NonZeroU32::new(128).unwrap());
        let weight_ids: std::collections::HashSet<u32> = [2u32].into_iter().collect();
        let (ops, _) = lower_graph_to_ktir(&ir, &weight_ids, ActiveCap::FULL, false).unwrap();
        let groups = ktir_groups(&ops, FoldGrouping::Split, None).unwrap();
        let consumes: Vec<_> = groups.iter().filter(|g| g.kv.tp.is_some()).collect();
        assert_eq!(consumes.len(), 2, "one consume group per rank");
        assert!(
            consumes.iter().all(|g| g.programs.len() == 2),
            "each holds its rank's two block copies"
        );
        for id in TpWorld::of::<2>().ranks() {
            let steps: Vec<CbStep> = cb_pipeline(&groups, id).collect();
            let sends = steps
                .iter()
                .filter(|s| matches!(s, CbStep::Send { .. }))
                .count();
            let waits = steps
                .iter()
                .filter(|s| matches!(s, CbStep::Wait { .. }))
                .count();
            assert_eq!(
                (sends, waits),
                (1, 1),
                "rank {} sends once and waits once",
                id.rank()
            );
        }
    }

    /// The window's placement: a sharded forward's layout carries it in the weight segment's base
    /// region, sized by the plan's own law (the same bytes `tp_bake` reports), above every weight;
    /// an unsharded forward's layout has none.
    #[test]
    fn a_sharded_layout_places_the_window_above_the_weights() {
        use crate::lower_subtile_tape_to_superdsc::compute_bundle_layout;
        use ktir_superdsc::placement::SegRole;
        use ktir_superdsc::reserved_tids::TP_RECV_CUT_TIDS;
        use scratchy_subtile::subtile_ir::lower_region;

        // The fixture's gemm weights: q, k, v, o, gate, up, down.
        let weight_ids: std::collections::HashSet<u32> =
            [2u32, 3, 4, 9, 11, 12, 13].into_iter().collect();
        let nb = std::num::NonZeroU32::new(128).unwrap();
        let layout_of = |w: SpyreWorld| {
            let b = tp_bake(&one_layer_input(), w).unwrap();
            compute_bundle_layout(
                &lower_region(&b.sharded.local, nb),
                &weight_ids,
                false,
                false,
                &[],
            )
            .expect("the decode layer lays out")
        };
        assert!(
            !layout_of(SpyreWorld::of::<1>())
                .placements
                .contains_key(&TP_RECV_CUT_TIDS[0])
        );
        let two = layout_of(SpyreWorld::of::<2>());
        let win = two.placements[&TP_RECV_CUT_TIDS[0]];
        assert_eq!(win.segment, SegRole::Weight.segment());
        assert_eq!(win.bank, 0);
        let planned = tp_bake(&one_layer_input(), SpyreWorld::of::<2>()).unwrap();
        assert_eq!(Some(win.size), planned.window_cut.map(|w| w.get()));
        assert_eq!(win.offset % 128, 0);
        // The cuts: equal, adjacent, in order — one window.
        for (i, &tid) in TP_RECV_CUT_TIDS.iter().enumerate() {
            let cut = two.placements[&tid];
            assert_eq!(
                (cut.segment, cut.bank, cut.size),
                (win.segment, 0, win.size)
            );
            assert_eq!(cut.offset, win.offset + i as u64 * win.size, "cut {i}");
        }
        let weights_end = two
            .placements
            .values()
            .filter(|p| !TP_RECV_CUT_TIDS.contains(&p.tid) && p.segment == win.segment)
            .map(|p| p.offset + p.size)
            .max()
            .unwrap_or(0);
        assert!(win.offset >= weights_end, "the window overlaps a weight");
    }
}
