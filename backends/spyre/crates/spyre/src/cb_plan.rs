//! #603 PR2 — one launch's compute control blocks, as DATA.
//!
//! The submit walk (`superdsc_exec::launch_ops_inner`) stages a launch sequence whose only
//! per-launch inputs are the launch row's segment shifts, weight bank and request facts. This
//! module is that sequence as data — a [`CbPlan`] (one entry per (op, request, relaunch), carrying
//! exactly `build_compute_cb`'s inputs) plus a render. The ENUMERATION is one function
//! (`fold_plan::launches`) and the per-launch arithmetic one ([`launch_site`]), both called by the
//! submit walk, the reference walk and the plan build alike, and the gate below proves the data
//! twin against the walk anyway.
//!
//! ⭐ THE PARITY GATE IS A PREPARE-TIME REFUSAL, NOT A RUNTIME CHECK. A step never pays for it:
//! `Executor::prepare` renders every allocated launch list two ways — the walk ([`walk_items`]) and
//! the plan ([`build_cb_plan`] + [`CbPlan::render`]) — through ONE render door
//! (`crate::runtime::Stream::render_batch`, the same `build_compute_cb` the submit path runs) and
//! REFUSES TO LOAD on the first differing byte.
//!
//! ⭐ A STEP'S IMAGE IS ITS CANVAS. Nothing a launch addresses moves with the step's position: a
//! walk's body shifts by its own trip, every other group launches where the bake placed it. So the
//! replay player copies a rendered canvas verbatim ([`render_canvas`]) and stamps only the tags.

use anyhow::{Result, anyhow, bail};
use flex_rs::control_block_wire::ControlBlockWire;
use flex_rs::scheduler::RunPlace;

use crate::bundle_code::NUM_SEGMENTS;
use crate::bundle_code::collective::{
    CbStep, Rank, Sid, TpConsume, TpSend, WINDOW_SEGMENT, group_steps,
};
use crate::fold_plan::{self, OpKv};
use crate::runtime::{
    BatchedCb, BatchedCompute, ByteSize, DevAddr, RdmaEnd, RdmaSend, Stream, prog_offset_base,
};
use crate::superdsc_exec::{LaunchWhere, Ops, SEG_INTERMEDIATE, SEG_KV, SEG_MASK, SEG_WEIGHT};

/// The base one segment's tensors resolve against for this launch — the weight
/// BANK's region for the weight segment (bank 0 IS `seg_addr[WEIGHT]`; bank
/// `b ≥ 1` is `banks[b - 1]`), the segment's own region otherwise. Shared by
/// the walk and the plan build so the bank selection cannot drift between
/// them. Refusals are the walk's own, verbatim — except that a bank slot which
/// exists but holds no region yet, which the walk's old code only reached as
/// the later "seg1 has no region", is named HERE as the bank it is (louder,
/// same unreachable-at-launch corner: prepare refuses a bank with no region
/// before any launch can name it).
fn segment_base<'a>(
    seg: &'a [Option<DevAddr>; NUM_SEGMENTS],
    banks: &'a [Option<DevAddr>],
    weight_bank: usize,
    i: usize,
) -> Result<&'a DevAddr> {
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
/// parity gate all read ONE computation. Every input arrives as a parameter —
/// the fold state and the segment bases included — so the prepare-time gate
/// can drive it with a synthetic session without touching live state, and so
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
    /// cannot drift between the real walk and the parity gate's reference
    /// side (a field added or dropped in one would be a compile error in the
    /// other, not a silent divergence the gate cannot see).
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
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
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
    // the walk and the plan twin both read through this function.
    let (weight_shift, weight_bank) = op
        .code
        .kv
        .weight_slot(off[SEG_WEIGHT.get()], where_.weight_bank);
    off[SEG_WEIGHT.get()] = weight_shift;
    let wbase = segment_base(seg, banks, weight_bank, SEG_WEIGHT.get())?;
    let mut ta = Vec::with_capacity(NUM_SEGMENTS);
    for (i, &sh) in off.iter().enumerate() {
        let base = if i == SEG_WEIGHT.get() {
            wbase
        } else {
            segment_base(seg, banks, weight_bank, i)?
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
/// every peer and a wait on every peer, in the bundle's own order ([`group_steps`]). The submit walk, [`walk_items`] and
/// [`CbPlan::render`] all stage through this, so the walk and its twin cannot disagree about a
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
            // ⭐ ONE WAIT CB PER MESSAGE SLOT, OVER EVERY PEER AT ONCE. The wdone barrier waits on a
            // SET of sids (its list bitmask) and consumes one count from each — measured on four
            // cards (`cards_all_gather`, `ONE_WAIT=1`: byte-exact, every wdone back to 0) — so the
            // `W - 1` peers' landings into slot `s` are one CB, not `W - 1`. The pipeline names
            // every peer's wait step in order; the CB for the slot is staged at the first and covers
            // them all, so the later steps stage nothing.
            CbStep::Wait { from, .. } => {
                let Some(BoundCollective { send, me }) = bound else {
                    bail!("group {index} waits with no binding");
                };
                if me.peers().next() != Some(from) {
                    continue;
                }
                for m in send.messages() {
                    out.push(BatchedCb::Wait {
                        sids: me.peers().map(|p| Sid::of(p, m.slot).get()).collect(),
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
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<(Vec<BatchedCb>, bool)> {
    let mut items = Vec::new();
    let mut barrier = false;
    // ⭐ THE ONE ENUMERATION — `fold_plan::launches`, the same sequence the submit walk and the
    // plan build enumerate: a walking bundle's span repeats once per request of the launch,
    // request-outer, and every other op launches once.
    let kvs: Vec<fold_plan::OpKv> = ops.iter().map(|o| o.kv).collect();
    for l in fold_plan::launches(&kvs, where_.requests.count) {
        let site = launch_site(ops, l.op, l.request, l.rep, seg, banks, where_)?;
        barrier = true;
        items.extend(site_cbs(ops[l.op].site_tp(), site.batched_compute())?);
    }
    Ok((items, barrier))
}

/// ONE (op, relaunch) of a launch sequence — `build_compute_cb`'s inputs as
/// data. Everything here is a constant of the (list, launch row) the plan was
/// built for.
pub(crate) struct CbPlanEntry<T: ktir_superdsc::collective::Tp> {
    /// Diagnostic identity — the label the walk's own traces print plus the
    /// request and relaunch — so a parity refusal names its CB.
    label: String,
    prog: DevAddr,
    bootstrap_offset: u64,
    /// The op's own launch-group facts, carried whole: the render calls the SAME
    /// `fold_plan::seg_deltas` the walk calls, never a re-derivation.
    kv: OpKv,
    request: u32,
    rep: i64,
    /// POSITIONAL, one per segment: the base the walk would resolve (the
    /// weight BANK's region for the weight segment, the segment's own
    /// otherwise) and the launch row's static shift — stored SEPARATELY, not
    /// pre-folded, so the render's one combined `shifted(static + delta)` is
    /// the walk's own arithmetic to the byte (a pre-shifted base would clamp
    /// differently at the region's end).
    bases: Vec<(DevAddr, u64)>,
    /// The collective CBs in front of this entry, staged by the walk's own [`site_cbs`].
    tp: SiteTp<T>,
}

/// One launch's CB constructor list — the walk's sequence as data, keyed by
/// (list, launch row) at build time.
pub(crate) struct CbPlan<T: ktir_superdsc::collective::Tp> {
    entries: Vec<CbPlanEntry<T>>,
    /// The launch's request pitch, recorded at build from the same `where_` the walk reads — the
    /// render composes each entry's request shift with it, exactly as the walk does live.
    pitch: u64,
}

impl<T: ktir_superdsc::collective::Tp> CbPlan<T> {
    /// The staged batch the plan renders to, with the walk's host fence: the same `seg_deltas`
    /// the walk calls, applied to the entry's recorded (base, static shift) with ONE combined
    /// `shifted` per segment, exactly as the walk computes it.
    pub(crate) fn render(&self) -> Result<(Vec<BatchedCb>, bool)> {
        let mut items = Vec::with_capacity(self.entries.len());
        for e in &self.entries {
            let delta = fold_plan::seg_deltas(&e.kv, e.request, e.rep, self.pitch);
            let mut dyn_off = [0u64; NUM_SEGMENTS];
            dyn_off[SEG_KV.get()] += delta.kv.0;
            dyn_off[SEG_MASK.get()] += delta.mask.0;
            dyn_off[SEG_INTERMEDIATE.get()] += delta.intermediate.0;
            let mut tensor_allocs = Vec::with_capacity(NUM_SEGMENTS);
            for (i, (base, static_off)) in e.bases.iter().enumerate() {
                let sh = *static_off + dyn_off[i];
                tensor_allocs.push(
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
            items.extend(site_cbs(
                e.tp,
                BatchedCompute {
                    prog: e.prog.clone(),
                    tensor_allocs,
                    bootstrap_offset: e.bootstrap_offset,
                },
            )?);
        }
        Ok((items, !self.entries.is_empty()))
    }

    /// Which entry, and which of its CBs, the plan's `i`-th CB is — the CBs an entry stages are
    /// [`entry_cbs`]'s (one per message of each send, one wait per message slot, the launch), so
    /// a refusal names the send or wait, not just its group.
    fn cb_label(&self, i: usize) -> String {
        self.entries
            .iter()
            .flat_map(|e| entry_cbs(&e.tp).into_iter().map(move |k| (e, k)))
            .nth(i)
            .map_or_else(|| format!("CB {i}"), |(e, k)| format!("{} {k}", e.label))
    }
}

/// The kind of every CB [`site_cbs`] stages for `tp`, in order — the same walk, without the
/// addresses: each send is one CB per message, the waits one CB per message slot over every
/// peer, then the launch.
fn entry_cbs<T: ktir_superdsc::collective::Tp>(tp: &SiteTp<T>) -> Vec<&'static str> {
    let messages = tp.bound.map_or(0, |b| b.send.messages().count());
    let mut out = Vec::new();
    let mut waited = false;
    for step in group_steps(tp.index, tp.consume) {
        match step {
            CbStep::Send { .. } => out.extend(std::iter::repeat_n("send", messages)),
            CbStep::Wait { .. } if !waited => {
                waited = true;
                out.extend(std::iter::repeat_n("wait", messages));
            }
            CbStep::Wait { .. } => {}
            CbStep::Launch { .. } => out.push("launch"),
        }
    }
    out
}

/// Assemble the plan for one launch call: the (op, request, relaunch) enumeration and every per-CB
/// input. Built from the same `fold_plan::launches` the walk calls; the bases come from the same
/// `segment_base` resolution the walk's `launch_site` performs.
pub(crate) fn build_cb_plan<T: ktir_superdsc::collective::Tp>(
    ops: &Ops<T>,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<CbPlan<T>> {
    let mut entries = Vec::new();
    let kvs: Vec<fold_plan::OpKv> = ops.iter().map(|o| o.kv).collect();
    for l in fold_plan::launches(&kvs, where_.requests.count) {
        let (oi, op) = (l.op, &ops[l.op]);
        // WHICH REGION BACKS THE WEIGHT SLOT, AND AT WHAT SHIFT — `weight_slot`, the one law the
        // walk's `launch_site` reads: the launch's bank and layer shift, except for a collective's
        // consume, pinned to bank 0 at shift 0 because its weight-segment operand is the window.
        let (weight_shift, weight_bank) = op
            .code
            .kv
            .weight_slot(where_.byte_off[SEG_WEIGHT.get()], where_.weight_bank);
        let wbase = segment_base(seg, banks, weight_bank, SEG_WEIGHT.get())?;
        let mut bases = Vec::with_capacity(NUM_SEGMENTS);
        for (i, byte_off) in where_.byte_off.iter().enumerate() {
            bases.push(if i == SEG_WEIGHT.get() {
                (wbase.clone(), weight_shift)
            } else {
                (segment_base(seg, banks, weight_bank, i)?.clone(), *byte_off)
            });
        }
        entries.push(CbPlanEntry {
            label: format!("{}/req{}/rep{}", op.label(), l.request, l.rep),
            prog: op
                .addr
                .clone()
                .ok_or_else(|| anyhow!("op {oi} has no program allocation"))?,
            bootstrap_offset: op
                .code
                .prog_image(prog_offset_base())
                .ok_or_else(|| {
                    anyhow!(
                        "op {oi}: job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x} — it \
                         names no bootstrap offset at all",
                        op.code.job_bin_ptr,
                        prog_offset_base()
                    )
                })?
                .bootstrap(),
            kv: op.kv,
            request: l.request,
            rep: l.rep,
            bases,
            tp: op.site_tp(),
        });
    }
    Ok(CbPlan {
        entries,
        pitch: where_.requests.pitch,
    })
}

/// ⭐ THE 2a GATE. Render one launch both ways — the walk's staged batch and
/// the CbPlan's — through one render door, and REFUSE on the first differing
/// byte. Called only from `Executor::prepare` (and the split ladder's lazy
/// allocation): a mismatch means the plan's data twin of the walk is wrong,
/// and the player that will replay these images must not load. Returns the
/// number of CBs proven byte-identical.
pub(crate) fn parity_check<T: ktir_superdsc::collective::Tp>(
    ops: &Ops<T>,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    label: &str,
    stream: &Stream,
) -> Result<usize> {
    let (walked, walked_barrier) = walk_items(ops, seg, banks, where_)?;
    let plan = build_cb_plan(ops, seg, banks, where_)?;
    let (rendered, rendered_barrier) = plan.render()?;
    if walked.len() != rendered.len() {
        bail!(
            "[cbplan] PARITY REFUSED at {label}: the walk stages {} launch(es) but the plan \
             renders {} — the (op, relaunch) ENUMERATION itself disagrees. The plan is the \
             walk's data twin; refusing to load rather than replay a wrong image.",
            walked.len(),
            rendered.len()
        );
    }
    if walked_barrier != rendered_barrier {
        bail!(
            "[cbplan] PARITY REFUSED at {label}: the walk's host-fence OR is \
             {walked_barrier} but the plan's is {rendered_barrier}"
        );
    }
    // Both sides through the SAME door at the SAME tags: the tag field is a
    // patch site (2b stamps pool-leased tags), so it cancels in the compare
    // and what remains — tensor XLAT paddrs and lengths, program, bootstrap —
    // is everything the device would be handed.
    let old = stream.render_batch(&walked, 1)?;
    let new = stream.render_batch(&rendered, 1)?;
    for (i, (a, b)) in old.iter().zip(new.iter()).enumerate() {
        if a != b {
            let (mut at, mut wb, mut pb) = (0, 0u8, 0u8);
            for (j, (x, y)) in a.bytes().iter().zip(b.bytes().iter()).enumerate() {
                if x != y {
                    (at, wb, pb) = (j, *x, *y);
                    break;
                }
            }
            bail!(
                "[cbplan] PARITY REFUSED at {label}: CB {i} of {} ({}) differs — first \
                 differing wire byte at +{at}: walk {wb:#04x} vs plan {pb:#04x}. The plan's \
                 data twin of the walk disagrees; refusing to load rather than replay a \
                 wrong image.",
                old.len(),
                plan.cb_label(i)
            );
        }
    }
    Ok(old.len())
}

/// One launch row's replay canvas — what the player copies verbatim into a step image. Nothing a
/// launch addresses moves with the step's position (a walk's body shifts by its own trip; every
/// other group launches where the bake placed it), so the canvas proven by [`parity_check`] at the
/// row's `where_` IS every step's CBs for that row. Returns the canvas and the row's host fence.
pub(crate) fn render_canvas<T: ktir_superdsc::collective::Tp>(
    ops: &Ops<T>,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    stream: &Stream,
) -> Result<(Vec<ControlBlockWire>, bool)> {
    let (items, barrier) = build_cb_plan(ops, seg, banks, where_)?.render()?;
    Ok((stream.render_batch(&items, 1)?, barrier))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle_code::collective::{
        CollectiveKind, CutPayload, LandingSlot, Payload, Rank, Tp, TpWorld, WindowCut, WireBytes,
        World, fragments, ranks,
    };
    use crate::runtime::Chunk0;
    use flex_rs::address::{ByteOffset, RegionId};
    use flex_rs::memory_region::DomainId;

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
            operand: crate::bundle_code::PlaceId::Act(7),
            fragment: fragments(1, 2048, Payload::from(WireBytes::CAP)).unwrap()[0],
            plays_at: me,
            cut: WindowCut::of_collective(0),
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
