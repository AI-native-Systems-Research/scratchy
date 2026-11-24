//! #603 PR2 — one launch's compute control blocks, as DATA.
//!
//! The submit walk (`superdsc_exec::launch_ops_inner`) stages a launch sequence whose only
//! per-launch inputs are the launch row's segment shifts, weight bank and request facts. This
//! module is that sequence's ONE data form — the staged batch ([`walk_items`]) and the replay
//! canvas ([`WalkCanvas`]) — both built from the ENUMERATION one function states
//! (`fold_plan::launches`) and the per-launch arithmetic one ([`launch_site`]), the same calls the
//! submit walk makes. There is no second construction to check against: the canvas type's only
//! constructor runs the walk, so what the player replays IS the walk by type, never by
//! comparison.
//!
//! ⭐ A STEP'S IMAGE IS ITS CANVAS. Nothing a launch addresses moves with the step's position: a
//! walk's body shifts by its own trip, every other group launches where the bake placed it. So the
//! replay player copies a rendered canvas verbatim ([`WalkCanvas`]) and stamps only the tags.

use anyhow::{Result, anyhow, bail};
use flex_rs::control_block_wire::ControlBlockWire;
use flex_rs::scheduler::RunPlace;

use crate::runtime::{
    BatchedCb, BatchedCompute, ByteSize, DevAddr, RdmaEnd, RdmaSend, Stream, prog_offset_base,
};
use crate::superdsc_exec::{LaunchWhere, Ops, SEG_INTERMEDIATE, SEG_KV, SEG_MASK, SEG_WEIGHT};
use scratchy_target_spyre::bundle_code::collective::{
    CbStep, Rank, Sid, TpConsume, TpSend, WINDOW_SEGMENT, group_steps,
};
use scratchy_target_spyre::bundle_code::{KvBank, NUM_SEGMENTS};
use scratchy_target_spyre::fold_plan;

/// ⭐ THE SESSION'S REGIONS, AS ONE VALUE — the positional segments, the weight segment's extra
/// banks and the KV segment's (one per layer past bank 0). A launch resolves every segment's base
/// through these three together ([`segment_base`]).
pub(crate) struct SegTables<'a> {
    /// The positional segments (`Executor::seg_addr`).
    pub(crate) seg: &'a [Option<DevAddr>; NUM_SEGMENTS],
    /// The weight segment's extra banks (`Executor::weight_bank_addr`), bank `b ≥ 1` at `b - 1`.
    pub(crate) banks: &'a [Option<DevAddr>],
    /// The KV segment's extra banks (`Executor::kv_bank_addr`), bank `b ≥ 1` at `b - 1`.
    pub(crate) kv_banks: &'a [DevAddr],
}

/// The base one segment's tensors resolve against for this launch — the weight
/// BANK's region for the weight segment (bank 0 IS `seg_addr[WEIGHT]`; bank
/// `b ≥ 1` is `banks[b - 1]`), the launch's own layer region for the KV segment
/// ([`KvBank`]), the segment's own region otherwise. Shared by
/// the walk and the canvas construction so the bank selection cannot drift
/// between them. Refusals are the walk's own, verbatim — except that a bank slot which
/// exists but holds no region yet, which the walk's old code only reached as
/// the later "seg1 has no region", is named HERE as the bank it is (louder,
/// same unreachable-at-launch corner: prepare refuses a bank with no region
/// before any launch can name it).
fn segment_base<'a>(
    tables: &SegTables<'a>,
    weight_bank: usize,
    kv_bank: KvBank,
    i: usize,
) -> Result<&'a DevAddr> {
    let (seg, banks) = (tables.seg, tables.banks);
    if i == SEG_KV.get()
        && let Some(b) = kv_bank.extra()
    {
        return Ok(&tables.kv_banks[b]);
    }
    if i == SEG_WEIGHT.get() {
        match weight_bank.checked_sub(1) {
            None => seg[i]
                .as_ref()
                .ok_or_else(|| anyhow!("seg{i} has no region")),
            Some(b) => banks.get(b).and_then(|a| a.as_ref()).ok_or_else(|| {
                anyhow!(
                    "launch: weight bank {weight_bank} has no region ({} bank(s))",
                    banks.len() + 1
                )
            }),
        }
    } else {
        seg[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))
    }
}

/// ONE (op, fold pass)'s launch arithmetic — the pure half of the submit walk,
/// factored out so the walk, the diag printers it feeds, and this module's
/// canvas construction all read ONE computation. Every input arrives as a
/// parameter — the fold state and the segment bases included — so the canvas
/// can be derived without touching live state, and so
/// the borrow of `Executor`'s fields lasts exactly one call (the walk's loop
/// interleaves `&mut self` diag hooks with this arithmetic).
pub(crate) struct LaunchSite {
    /// The per-segment byte offsets after the relaunch's walk shift lands on the
    /// launch row's static shifts.
    pub(crate) off: [u64; NUM_SEGMENTS],
    /// POSITIONAL per segment, owned: the resolved tensor bases with the
    /// combined shift applied — exactly what `compute`/`compute_batch` take.
    pub(crate) ta: Vec<DevAddr>,
    /// The program allocation (segment 7) this launch runs.
    pub(crate) prog: DevAddr,
    /// Offset within the program allocation where kernel execution begins.
    pub(crate) bootstrap: u64,
}

impl LaunchSite {
    /// The staged-batch item this site becomes — ONE constructor shared by the
    /// submit walk's collect branch and [`walk_items`], so the item's shape
    /// cannot drift between the real walk and the replay canvas (a field
    /// added or dropped in one would be a compile error in the
    /// other, not a silent divergence).
    pub(crate) fn batched_compute(&self) -> BatchedCompute {
        BatchedCompute {
            prog: self.prog.clone(),
            tensor_allocs: self.ta.clone(),
            bootstrap_offset: self.bootstrap,
        }
    }
}

/// One (op, request, relaunch)'s launch: the request's tables and the walk's body shift for
/// relaunch `rep` on the launch row's static shifts, the resolved bases, the program and its
/// bootstrap.
pub(crate) fn launch_site<T: ktir_superdsc::collective::Tp>(
    ops: &Ops<T>,
    oi: usize,
    request: u32,
    rep: i64,
    tables: &SegTables<'_>,
    where_: &LaunchWhere,
) -> Result<LaunchSite> {
    let op = &ops[oi];
    let mut off = where_.byte_off;
    let delta = fold_plan::seg_deltas(&op.kv, request, rep, where_.requests.pitch);
    off[SEG_KV.get()] += delta.kv.0;
    off[SEG_MASK.get()] += delta.mask.0;
    off[SEG_INTERMEDIATE.get()] += delta.intermediate.0;

    // PER-LAYER ADDRESS ADVANCE (re-roll weight/KV threading). dxp does NOT apply
    // `ComputeParams::tensor_byte_offsets` to per-op compute addresses (MEASURED: every
    // layer read layer-0's weights). So bake the per-segment offset into a SHIFTED,
    // non-owning base per segment: all of that segment's tensors then resolve to
    // seg_base + v·stride = layer v's copy. Layer 0 (off=0) is unshifted.
    // ⭐ WHICH REGION BACKS THE WEIGHT SLOT: this launch's BANK. Positional slot 1 is
    // "the weights" to the program, and a `DevAddr` carries no segment identity, so
    // handing a different region here is the whole mechanism that lets one segment's
    // worth of addresses span several regions. Bank 0 (`weight_bank` = 0, every unbanked
    // bundle) is `seg_addr[SEG_WEIGHT]`, exactly as before.
    // ⭐ AND A COLLECTIVE'S CONSUME TAKES NEITHER: its weight-segment operand is the receive window,
    // one placement in the base region, so `weight_slot` pins it to shift 0 in bank 0 — the one law
    // the walk and the canvas construction both read through this function.
    let (weight_shift, weight_bank) = op
        .code
        .kv
        .weight_slot(off[SEG_WEIGHT.get()], where_.weight_bank);
    off[SEG_WEIGHT.get()] = weight_shift;
    let wbase = segment_base(tables, weight_bank, where_.kv_bank, SEG_WEIGHT.get())?;
    let mut ta = Vec::with_capacity(NUM_SEGMENTS);
    for (i, &sh) in off.iter().enumerate() {
        let base = if i == SEG_WEIGHT.get() {
            wbase
        } else {
            segment_base(tables, weight_bank, where_.kv_bank, i)?
        };
        ta.push(
            if sh > 0
                && base.is_single_chunk()
                && let Some(s) = base.shifted(ByteSize(sh))
            {
                s
            } else {
                base.clone()
            },
        );
    }

    let prog = op
        .addr
        .clone()
        .ok_or_else(|| anyhow!("op {oi} has no program allocation"))?;
    // #2814: the bootstrap argument is the OFFSET, not the full VA (flex bounds the
    // seg7 translation to the program allocation's size).
    // ⛔ THE SAME VALUE `alloc_ops_at` UPLOADED AT, not a second subtraction. This line was
    // `op.code.job_bin_ptr - prog_offset_base()` — bare, so an underflow wrapped here while
    // the allocation path refused it, and the two could only agree by both being written
    // correctly. `ProgImage` is now the one door: see `LaunchGroup::prog_image`.
    let bootstrap = op
        .code
        .prog_image(prog_offset_base())
        .ok_or_else(|| {
            anyhow!(
                "op {oi}: job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x} — it names no \
                 bootstrap offset at all",
                op.code.job_bin_ptr,
                prog_offset_base()
            )
        })?
        .bootstrap();
    Ok(LaunchSite {
        off,
        ta,
        prog,
        bootstrap,
    })
}

/// A collective's consume as a session BOUND it: the send its baked placements name, for the rank
/// that plays it. Minted only at bind (`superdsc_exec::bind_ops`), for the consumes the session
/// plays.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundCollective<T: ktir_superdsc::collective::Tp> {
    pub(crate) send: TpSend<T>,
    pub(crate) me: Rank<T>,
}

/// What [`site_cbs`] reads off a group: its launch order, its consume (the bundle's), and the
/// session's binding of it.
#[derive(Clone, Copy)]
pub(crate) struct SiteTp<T: ktir_superdsc::collective::Tp> {
    pub(crate) index: usize,
    pub(crate) consume: Option<TpConsume<T>>,
    pub(crate) bound: Option<BoundCollective<T>>,
}

/// ⭐ ONE (op, PASS)'S CONTROL BLOCKS — its launch, and in front of a collective's consume a send to
/// and a wait on each of its exchange's peers, in the bundle's own order ([`group_steps`]). The submit walk and
/// [`walk_items`] both stage through this, so the walk and the replay canvas cannot disagree about a
/// collective's CBs any more than about a launch's.
///
/// A send's two endpoints are addressed through THE CONSUME LAUNCH'S OWN SEGMENT BASES: the
/// operand's segment as the consume reads it (every layer shift the launch carries applies) and the
/// window's as the consume reads it (pinned to bank 0, unshifted, by `weight_slot`). The window
/// end is this rank's block, which is the same block on every peer (the layout is card-symmetric),
/// so one landing address serves every send; the destination is the peer's base-address-table
/// entry, its rank.
pub(crate) fn site_cbs<T: ktir_superdsc::collective::Tp>(
    tp: SiteTp<T>,
    launch: BatchedCompute,
) -> Result<Vec<BatchedCb>> {
    let SiteTp {
        index,
        consume,
        bound,
    } = tp;
    // A consume the session never bound is a group bind refused — unreachable from `bind_ops`,
    // which binds every consume it keeps; refused here rather than staged without its send.
    let bound = match (consume, bound) {
        (Some(_), None) => bail!("group {index} is a collective's consume the session never bound"),
        (_, b) => b,
    };
    let end = |l: &BatchedCompute, segment: u8, offset: u64| -> Result<RdmaEnd> {
        let base = l.tensor_allocs.get(usize::from(segment)).ok_or_else(|| {
            anyhow!(
                "group {index}: the send names segment {segment}, but the consume launch binds \
                 only {} segment(s)",
                l.tensor_allocs.len()
            )
        })?;
        Ok(RdmaEnd {
            base: base.clone(),
            segment,
            offset,
        })
    };
    let mut launch = Some(launch);
    let mut out = Vec::new();
    // Only a group's first send waits on the compute ahead of it: every send reads the same
    // operand, which the first one's barrier already saw written.
    let mut first_send = true;
    for step in group_steps(index, consume) {
        match step {
            // ⭐ THE CUT: a send is its run of messages (`TpSend::messages`, the vendor's windowed
            // loop), one F-send each, to peer `to`, signalling this rank's sid for each slot.
            CbStep::Send { to, .. } => {
                let (Some(BoundCollective { send, me }), Some(l)) = (bound, launch.as_ref()) else {
                    bail!("group {index} sends with no binding or after its launch");
                };
                for m in send.messages() {
                    out.push(BatchedCb::Send(RdmaSend {
                        sid: Sid::of(me, m.slot).get(),
                        dest_tid: base_addr_entry(to),
                        bytes: m.bytes,
                        source: end(l, send.source_segment, m.source_offset)?,
                        window: end(l, WINDOW_SEGMENT as u8, m.window_offset)?,
                        place: RunPlace {
                            first: m.first,
                            last: m.last,
                        },
                        barrier: std::mem::replace(&mut first_send, false),
                    }));
                }
            }
            // ⭐ AND THE PAIRING: every one of peer `from`'s messages increments its slot's counter
            // on `from`'s sid once and a wait consumes exactly one count, so `from`'s run of N
            // messages is waited for N times, once per slot. Every rank's run is this rank's own —
            // the forward is SPMD and the fragment is the payload's — so this rank's messages name
            // `from`'s slots too.
            // ⭐ ONE WAIT CB PER MESSAGE SLOT, OVER EVERY PEER OF THE EXCHANGE AT ONCE. The wdone
            // barrier waits on a SET of sids (its list bitmask) and consumes one count from each —
            // measured on four cards (`cards_all_gather`, `ONE_WAIT=1`: byte-exact, every wdone
            // back to 0) — so the exchange's peers' landings into slot `s` are one CB, not one per
            // peer. The pipeline names every peer's wait step in order (`TpConsume::peers`: every
            // peer for an all-gather, the round's one partner for an all-reduce round); the CB for
            // the slot is staged at the first and covers them all, so the later steps stage nothing.
            CbStep::Wait { from, consume, .. } => {
                let Some(BoundCollective { send, .. }) = bound else {
                    bail!("group {index} waits with no binding");
                };
                if consume.peers().next() != Some(from) {
                    continue;
                }
                for m in send.messages() {
                    out.push(BatchedCb::Wait {
                        sids: consume.peers().map(|p| Sid::of(p, m.slot).get()).collect(),
                    });
                }
            }
            CbStep::Launch { .. } => {
                let Some(l) = launch.take() else {
                    bail!("group {index} launches twice");
                };
                out.push(if consume.is_some() {
                    BatchedCb::Consume(l)
                } else {
                    BatchedCb::Compute(l)
                });
            }
        }
    }
    Ok(out)
}

/// The base-address-table entry a send to `to` names: the peer's rank, the entry
/// `runtime::link_peers` installs that card's region-2 address at on every other card.
pub(crate) fn base_addr_entry<T: ktir_superdsc::collective::Tp>(to: Rank<T>) -> u8 {
    // A rank is below `collective::MAX_WORLD`, far below a byte.
    to.get() as u8
}

/// The walk's staged batch for one launch call — the OLD way, built from the
/// same `fold_plan::launches`/`launch_site` the submit walk reads. This is the parity
/// gate's reference side. The submit walk keeps its own loop body (its diag knobs
/// interleave with the arithmetic), which is exactly why the gate exists: if
/// the walk and its twin ever disagree in addressing, the rendered CBs
/// differ and prepare refuses. The enumeration itself is ONE function both read, so
/// order cannot drift — only addressing can, and the gate catches that. Returns the
/// items and the walk's host fence: set whenever the call launches anything, since
/// every launch fences.
pub(crate) fn walk_items<T: ktir_superdsc::collective::Tp>(
    ops: &Ops<T>,
    tables: &SegTables<'_>,
    where_: &LaunchWhere,
) -> Result<(Vec<BatchedCb>, bool)> {
    let mut items = Vec::new();
    let mut barrier = false;
    // ⭐ THE ONE ENUMERATION — `fold_plan::launches`, the same sequence the submit walk and the
    // plan build enumerate: a walking bundle's span repeats once per request of the launch,
    // request-outer, and every other op launches once.
    let kvs: Vec<fold_plan::OpKv> = ops.iter().map(|o| o.kv).collect();
    for l in fold_plan::launches(&kvs, where_.requests.count) {
        let site = launch_site(ops, l.op, l.request, l.rep, tables, where_)?;
        barrier = true;
        items.extend(site_cbs(ops[l.op].site_tp(), site.batched_compute())?);
    }
    Ok((items, barrier))
}

/// ⭐ THE REPLAY CANVAS, AS A TYPE — the walk's own staged batch, rendered
/// through the one render door. The field is private to this module and the
/// constructors below are its only producers: a canvas that is not a rendered
/// walk cannot be written, so there is no second construction to compare
/// against and no parity left to prove at load — the plan twin this type
/// replaced was a second statement of the walk, and the boot-time parity gate
/// that checked it was the compensation for that second statement.
///
/// Nothing a launch addresses moves with the step's position (a walk's body
/// shifts by its own trip; every other group launches where the bake placed
/// it), so one canvas per launch row IS every step's CBs for that row.
pub(crate) struct WalkCanvas {
    cbs: std::sync::Arc<Vec<ControlBlockWire>>,
    /// The walk's host fence — the OR over every row merged into this canvas.
    barrier: bool,
}

impl WalkCanvas {
    /// THE constructor: the walk's own enumeration (`fold_plan::launches`),
    /// staged by [`walk_items`] and rendered by the same door the submit path
    /// renders through. A non-empty ops list stages at least one launch, so a
    /// canvas from this door is never empty.
    pub(crate) fn of_walk<T: ktir_superdsc::collective::Tp>(
        ops: &Ops<T>,
        tables: &SegTables<'_>,
        where_: &LaunchWhere,
        stream: &Stream<'_>,
    ) -> Result<Self> {
        let (items, barrier) = walk_items(ops, tables, where_)?;
        Ok(Self {
            cbs: std::sync::Arc::new(stream.render_batch(&items, 1)?),
            barrier,
        })
    }

    /// One derivation's rows, concatenated. The merge preserves the invariant
    /// — a concatenation of rendered walks is a rendered walk — and stamps the
    /// canvas's first CB with the same EDEP1 the submit path stamps
    /// (`ResponseWorker::prepare_cbs`, the vendor `Finalize` port at
    /// control_block_stream.cpp:423): the replay patch rewrites tags in place
    /// and never touches the CTRL mode bits, so what derivation stores is what
    /// every replay of this image submits — a cached image must not hold the
    /// EDEP0 its rows were rendered with. The submit-side stamp makes the two
    /// agree rather than this one being load-bearing (a double-set of the same
    /// value is fine; a miss is the defect).
    pub(crate) fn merged(mut rows: Vec<Self>) -> Self {
        let mut cbs = Vec::new();
        let mut barrier = false;
        for row in rows.drain(..) {
            cbs.extend(row.cbs.iter().copied());
            barrier |= row.barrier;
        }
        if let Some(first) = cbs.first_mut() {
            first.set_edep(1);
        }
        Self {
            cbs: std::sync::Arc::new(cbs),
            barrier,
        }
    }

    /// The degenerate zero-row canvas. A derivation never produces one (a
    /// prepared session's prefix and suffix lists are allocated
    /// unconditionally, so a derivation merges at least one row); only the
    /// cardless cache-accounting test holds one, for a payload it never
    /// replays.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            cbs: std::sync::Arc::new(Vec::new()),
            barrier: false,
        }
    }

    /// The rendered CBs — shared by every replay's submission batch, which
    /// clones them out to stamp pool-leased tags. Read-only: a caller can
    /// submit this canvas but can never construct one.
    pub(crate) fn cbs(&self) -> &std::sync::Arc<Vec<ControlBlockWire>> {
        &self.cbs
    }

    /// The walk's host fence over the whole canvas.
    pub(crate) fn barrier(&self) -> bool {
        self.barrier
    }

    /// How many CBs the canvas renders — the player's accounting line.
    pub(crate) fn len(&self) -> usize {
        self.cbs.len()
    }

    /// The zero-row degenerate — true only of [`WalkCanvas::empty`].
    pub(crate) fn is_empty(&self) -> bool {
        self.cbs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Chunk0;
    use flex_rs::address::{ByteOffset, RegionId};
    use flex_rs::memory_region::DomainId;
    use scratchy_target_spyre::bundle_code::collective::{
        CollectiveKind, CutPayload, Exchange, LandingSlot, Payload, Rank, Tp, TpWorld, WindowCut,
        WireBytes, World, fragments, pairwise_rounds, ranks,
    };

    /// One launch over seven distinct segment bases, region `10 + segment`.
    fn launch() -> BatchedCompute {
        let at = |region: u64| {
            DevAddr::from_chunk(Chunk0 {
                region: RegionId(region),
                offset: ByteOffset(0),
                size: ByteSize(1 << 26),
                domain: DomainId(0),
            })
        };
        BatchedCompute {
            prog: at(99),
            tensor_allocs: (0..NUM_SEGMENTS as u64).map(|s| at(10 + s)).collect(),
            bootstrap_offset: 0,
        }
    }

    fn consume<T: Tp>(me: Rank<T>) -> TpConsume<T> {
        TpConsume {
            kind: CollectiveKind::AllReduce,
            operand: scratchy_target_spyre::bundle_code::PlaceId::Act(7),
            fragment: fragments(1, 2048, Payload::from(WireBytes::CAP)).unwrap()[0],
            plays_at: me,
            cut: WindowCut::of_collective(0),
            exchange: Exchange::EveryPeer,
        }
    }

    fn region(e: &RdmaEnd) -> u64 {
        e.base.chunk0().unwrap().region.0
    }

    /// A consume site, at every world: a send to every peer — each reading its operand through the
    /// consume launch's own base for that segment, landing at this rank's block through its
    /// weight-segment base, signalling on this rank's sid, naming the peer's BAT entry — then a
    /// wait on every peer's sid, then the barriered consume.
    #[test]
    fn a_consume_site_sends_to_and_waits_on_every_peer() {
        fn check<T: Tp>() {
            let cut = WindowCut::of_collective(1);
            for me in ranks::<T>() {
                let send = TpSend {
                    source_segment: 3,
                    source_offset: 4096,
                    window_offset: 1 << 20,
                    bytes: CutPayload::new(Payload::new(4096).unwrap()).unwrap(),
                    cut,
                };
                let slot = LandingSlot::of(cut, 0).unwrap();
                let cbs = site_cbs(
                    SiteTp {
                        index: 5,
                        consume: Some(consume(me)),
                        bound: Some(BoundCollective { send, me }),
                    },
                    launch(),
                )
                .unwrap();
                let peers: Vec<Rank<T>> = me.peers().collect();
                let n = peers.len();
                // A send to every peer, ONE wait over every peer's sid, the consume.
                assert_eq!(cbs.len(), n + 2, "world {}", <T as World>::RANKS);
                for (i, to) in peers.iter().enumerate() {
                    let BatchedCb::Send(s) = &cbs[i] else {
                        panic!("the sends lead");
                    };
                    assert_eq!(
                        (s.sid, s.dest_tid, s.bytes.get()),
                        (Sid::of(me, slot).get(), to.get() as u8, 4096)
                    );
                    assert_eq!(s.place, RunPlace::ALONE);
                    assert_eq!(
                        s.barrier,
                        i == 0,
                        "only the first send waits on the compute"
                    );
                    assert_eq!(
                        (region(&s.source), s.source.segment, s.source.offset),
                        (13, 3, 4096)
                    );
                    assert_eq!(
                        (region(&s.window), s.window.segment, s.window.offset),
                        (11, 1, 1 << 20)
                    );
                }
                let want: Vec<u8> = peers.iter().map(|&p| Sid::of(p, slot).get()).collect();
                assert!(matches!(&cbs[n], BatchedCb::Wait { sids } if *sids == want));
                assert!(matches!(cbs[n + 1], BatchedCb::Consume(_)));
            }
        }
        check::<TpWorld<2>>();
        check::<TpWorld<3>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
    }

    /// An all-reduce round's site, at every power-of-two world: ONE send, to the round's partner
    /// alone (its BAT entry), barriered behind the compute that wrote the running sum; ONE wait,
    /// on the partner's sid alone; then the barriered consume.
    #[test]
    fn a_pairwise_round_site_sends_to_and_waits_on_its_partner_alone() {
        fn check<T: Tp>() {
            let cut = WindowCut::of_collective(0);
            let slot = LandingSlot::of(cut, 0).unwrap();
            for me in ranks::<T>() {
                for round in pairwise_rounds::<T>() {
                    let send = TpSend {
                        source_segment: 3,
                        source_offset: 4096,
                        window_offset: 1 << 20,
                        bytes: CutPayload::new(Payload::new(4096).unwrap()).unwrap(),
                        cut,
                    };
                    let cbs = site_cbs(
                        SiteTp {
                            index: 5,
                            consume: Some(TpConsume {
                                exchange: Exchange::Pairwise(round),
                                ..consume(me)
                            }),
                            bound: Some(BoundCollective { send, me }),
                        },
                        launch(),
                    )
                    .unwrap();
                    let partner = me.partner(round);
                    assert_eq!(cbs.len(), 3, "world {} round {}", T::RANKS, round.get());
                    let BatchedCb::Send(s) = &cbs[0] else {
                        panic!("the send leads");
                    };
                    assert_eq!(
                        (s.sid, s.dest_tid, s.barrier),
                        (Sid::of(me, slot).get(), partner.get() as u8, true)
                    );
                    assert!(
                        matches!(&cbs[1], BatchedCb::Wait { sids } if *sids == [Sid::of(partner, slot).get()])
                    );
                    assert!(matches!(cbs[2], BatchedCb::Consume(_)));
                }
            }
        }
        check::<TpWorld<2>>();
        check::<TpWorld<4>>();
        check::<TpWorld<8>>();
    }

    /// A send wider than one message stages as its run — every message an F-send at the matching
    /// offset into a slot of its own, `first` / `last` on its ends — to every peer, then one wait
    /// per (peer, message) on that slot's sid, then the consume: the write-done counts a peer's run
    /// raises are exactly the counts this rank's waits on it take, slot by slot.
    #[test]
    fn a_wide_send_stages_its_messages_then_as_many_waits() {
        type W4 = TpWorld<4>;
        let me = Rank::<W4>::of::<0>();
        let cap = WireBytes::CAP.get();
        let cut = WindowCut::of_collective(0);
        let send = TpSend {
            source_segment: 0,
            source_offset: 0,
            window_offset: 1 << 20,
            bytes: CutPayload::new(Payload::new(cap + 4096).unwrap()).unwrap(),
            cut,
        };
        let slot = |m| LandingSlot::<W4>::of(cut, m).unwrap();
        let cbs = site_cbs(
            SiteTp {
                index: 1,
                consume: Some(consume(me)),
                bound: Some(BoundCollective { send, me }),
            },
            launch(),
        )
        .unwrap();
        let peers: Vec<Rank<W4>> = me.peers().collect();
        // Two messages to each peer, one wait per message slot over every peer, the consume.
        assert_eq!(cbs.len(), peers.len() * 2 + 2 + 1);
        for (p, to) in peers.iter().enumerate() {
            let sends: Vec<&RdmaSend> = cbs[2 * p..2 * p + 2]
                .iter()
                .map(|c| match c {
                    BatchedCb::Send(s) => s,
                    _ => panic!("the runs' sends lead"),
                })
                .collect();
            assert_eq!(
                sends
                    .iter()
                    .map(|s| (s.source.offset, s.window.offset, s.bytes.get(), s.dest_tid))
                    .collect::<Vec<_>>(),
                [
                    (0, 1 << 20, cap, to.get() as u8),
                    (cap, (1 << 20) + cap, 4096, to.get() as u8)
                ]
            );
            assert_eq!(
                sends.iter().map(|s| s.place).collect::<Vec<_>>(),
                [
                    RunPlace {
                        first: true,
                        last: false
                    },
                    RunPlace {
                        first: false,
                        last: true
                    }
                ]
            );
            assert_eq!(
                sends.iter().map(|s| s.sid).collect::<Vec<_>>(),
                [Sid::of(me, slot(0)).get(), Sid::of(me, slot(1)).get()]
            );
        }
        let waits = &cbs[2 * peers.len()..2 * peers.len() + 2];
        for m in 0..2 {
            let want: Vec<u8> = peers.iter().map(|&p| Sid::of(p, slot(m)).get()).collect();
            assert!(matches!(&waits[m as usize], BatchedCb::Wait { sids } if *sids == want));
        }
        assert!(matches!(cbs.last(), Some(BatchedCb::Consume(_))));
    }

    /// A plain group is its one launch; a consume the session never bound refuses rather than
    /// stage a consume with no send in front of it.
    #[test]
    fn a_plain_site_is_one_launch_and_an_unbound_consume_refuses() {
        let plain = site_cbs(
            SiteTp::<TpWorld<1>> {
                index: 0,
                consume: None,
                bound: None,
            },
            launch(),
        )
        .unwrap();
        assert!(matches!(plain.as_slice(), [BatchedCb::Compute(_)]));
        let why = site_cbs(
            SiteTp {
                index: 2,
                consume: Some(consume(Rank::<TpWorld<2>>::of::<0>())),
                bound: None,
            },
            launch(),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(why.contains("never bound"), "{why}");
    }
}
