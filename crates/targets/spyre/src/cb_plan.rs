//! #603 PR2 — one launch's compute control blocks, as DATA (stage 2a).
//!
//! The submit walk (`superdsc_exec::launch_ops_inner`) re-derives, every step,
//! a launch sequence whose only per-step inputs are a handful of scalars: the
//! fold page count, the write slot, the write page's base. This module is that
//! sequence as data — a [`CbPlan`] (one entry per (op, fold pass), carrying
//! exactly `build_compute_cb`'s inputs) plus a render that re-applies the
//! per-step scalars through the SAME `fold_plan` vocabulary the walk uses.
//! There is no second analysis to drift: the enumeration and the per-(op, rep)
//! arithmetic are single-sourced here ([`op_reps`], [`launch_site`]) and called
//! by BOTH the submit walk and the plan build, and the gate below proves the
//! data twin against the walk anyway.
//!
//! ⭐ THE PARITY GATE IS A PREPARE-TIME REFUSAL, NOT A RUNTIME CHECK. A step
//! never pays for it: `Executor::prepare` renders every allocated launch list
//! two ways — the walk ([`walk_items`]) and the plan
//! ([`build_cb_plan`] + [`CbPlan::render`]) — through ONE render door
//! (`crate::runtime::Stream::render_compute_batch`, the same
//! `build_compute_cb` the submit path runs) and REFUSES TO LOAD on the first
//! differing byte. A lazily-allocated split ladder is checked at its allocation
//! (`ensure_split_bodies`), the one event that makes it exist — never per step.
//!
//! The images this feeds (2c's player) are keyed (list × launch row ×
//! fold-page count × fold shape): the fold shape — live request count, declared
//! `fold_pages`, the installed page maps — changes what `reps` enumerates and
//! what `seg_deltas` computes, so an image is valid only under the shape it was
//! rendered for. 2a renders nothing cached; it only proves the machinery, and
//! the gate's synthetic session (in `superdsc_exec`, next to the hook)
//! deliberately exercises the batched fold (reps > 1), the slot write, the
//! non-fold page resolution and the per-request intermediate shift.
//!
//! ⭐ 2b — THE PATCH TABLE. Within one image key the ONLY per-step inputs are
//! the write position's SLOT IN ITS PAGE and its SLAB, and every address term
//! they move is affine in those two scalars ([`StepScalars`]) — the fold
//! terms are per-rep constants baked into the image, and the write PAGE never
//! moves inside an image's replayed span: every step between two page
//! boundaries writes the page its canvas was rendered on, and the boundary
//! step itself (slot 0 of the next page) is not replayed at all — it takes
//! the walk (see `try_replay_forward`; a moved write base is not affine in
//! any per-byte scalar, because a shifted write target's extent is END-PINNED
//! — `DevAddr::shifted` keeps the allocation's end and gives back the bytes
//! remaining, so the XLAT word's paddr and length fields move together and no
//! per-byte coefficient can express the pair). A replayed step is therefore
//! canvas + patches: copy the image, add each moving word's affine addend
//! ([`PatchPlan::apply`]). The coefficients are not read off the wire layout
//! by hand; they are MEASURED at prepare by finite differences through the
//! same render door ([`build_patch_plan`]) and then proven off-sample
//! ([`patch_check`]): at steps neither the canvas nor the differentials
//! visited, the patched canvas must equal the walk's fresh render byte for
//! byte, or the load refuses. A bundle whose movement is not affine in the
//! two scalars refuses there — loudly, at prepare, never as a wrong token.

use anyhow::{Result, anyhow, bail};
use flex_rs::control_block_wire::{CB_NUM_BYTES, ControlBlockWire};

use crate::bundle_code::NUM_SEGMENTS;
use crate::fold_plan::{self, OpKv, SessionKv, SlotPos, UniformPages};
use crate::runtime::{BatchedCompute, ByteSize, DevAddr, Stream, prog_offset_base};
use crate::superdsc_exec::{
    LaunchWhere, Ops, SEG_INTERMEDIATE, SEG_KV, SEG_MASK, SEG_WEIGHT, SeqPos,
};

/// The fold pass count for one op — `fold_plan::reps` plus the walk's refusal,
/// single-sourced so the submit walk and the plan build cannot disagree about
/// how many passes an op takes (a disagreement is a different CB COUNT, which
/// the parity gate would catch — but refusing at the source is louder).
pub(crate) fn op_reps(kv: &OpKv, fold: &SessionKv, n_fold_pages: i64, oi: usize) -> Result<i64> {
    let reps = fold_plan::reps(kv, fold, n_fold_pages);
    if reps < 0 {
        // A NEGATIVE COUNT IS A REFUSAL, not a loop bound: the fold would take more passes
        // than the mask has blocks, and the bytes past the last block are unstaged zeros
        // that an additive mask reads as VALID. Abandon rather than fold in whatever the
        // pool holds — loud and incomplete beats quiet and wrong.
        // ⛔ NAME WHICH REFUSAL FIRED. `reps` returns -1 for TWO unrelated reasons and the
        // walk's message used to describe only the second, so a page-count DISAGREEMENT read
        // as a capacity shortfall and sent the reader off to enlarge a mask that was big
        // enough.
        let (declared_pages, blocks) = (fold.fold_pages, fold.mask_blocks);
        let why = if declared_pages > 0 && n_fold_pages != declared_pages as i64 {
            format!(
                "the launch walks {n_fold_pages} page(s) but the host STAGED the mask for \
                 {declared_pages} — a disagreement, not a shortfall; the mask is fine and \
                 one of the two counts is wrong"
            )
        } else {
            format!(
                "the staged mask has only {blocks} block(s) — a genuine capacity shortfall; \
                 mask bytes go as pages*width^2*nqh*512, so declare more blocks or fold \
                 fewer pages"
            )
        };
        bail!(
            "fold: {n_fold_pages} pass(es) REFUSED — {why}. ABANDONING the launch at op \
             {oi}. A pass past the last block reads bytes the host never staged; they are \
             zero, and zero is VALID in an additive mask, so the extra passes would fold in \
             whatever the pool holds."
        );
    }
    Ok(reps)
}

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
    /// The per-segment byte offsets after the fold pass's deltas land on the
    /// launch row's static shifts.
    pub(crate) off: [u64; NUM_SEGMENTS],
    /// POSITIONAL per segment, owned: the resolved tensor bases with the
    /// combined shift applied — exactly what `compute`/`compute_batch` take.
    pub(crate) ta: Vec<DevAddr>,
    /// The program allocation (segment 7) this launch runs.
    pub(crate) prog: DevAddr,
    /// Offset within the program allocation where kernel execution begins.
    pub(crate) bootstrap: u64,
    /// The HOST fence flag — see [`CbPlanEntry::barrier`]; never a wire bit.
    pub(crate) barrier: bool,
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

/// `Ok(None)` = THIS (op, pass) DOES NOT LAUNCH — a page part ([`OpKv::part`]) whose
/// page the request does not hold, a short last chunk's padding rows past the request's pages
/// (#620). Its rows are never read, and the only base left to aim it at is the cursor's own
/// page, which is part 0's keys. Every caller — the submit walk, `walk_items`, the render —
/// skips the entry, so the plan twin and the walk can never disagree about the enumeration.
pub(crate) fn launch_site(
    ops: &Ops,
    oi: usize,
    rep: i64,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<Option<LaunchSite>> {
    let op = &ops[oi];
    let mut off = where_.byte_off;
    // ⭐ THE WITNESS IS MINTED HERE (was: inside the walk's rep loop).
    // `UniformPages::sweeping` checks the sweep against the pool that is
    // actually installed; a disagreement is a REFUSAL, never an address — the
    // alternative is a pass naming a page its row does not own.
    let Some(per) = UniformPages::sweeping(where_.n_fold_pages, fold) else {
        bail!(
            "fold_plan refused op {oi} (rep {rep}): the installed page maps do not \
             admit a uniform {}-page sweep",
            where_.n_fold_pages
        );
    };
    // ⭐ THE AGREED PAGE GEOMETRY IS MINTED HERE, beside the sweep witness — the op's
    // baked `kv_page_slots` and the session's installed size must AGREE, or the launch
    // refuses naming both spellings. The old law was the comment "they agree in every
    // bundle" (the deltas each read their own side) plus a canary in the patch gate
    // below; `PageGeometry::agree` is now the one door both deltas decompose behind.
    let geo = fold_plan::PageGeometry::agree(&op.kv, fold)
        .map_err(|why| anyhow!("fold_plan refused op {oi} (rep {rep}): {why}"))?;
    // WHERE THIS OP READS AND WRITES KV — resolved in `fold_plan`. It composes, in this
    // order: the fold pass's page and validity row, the write slot within its page, the
    // page a non-fold op works in, and the current slab.
    let aim = fold_plan::seg_deltas(
        &op.kv,
        fold,
        geo,
        rep,
        per,
        SlotPos::of_launch(where_.slot_pos.0),
        where_.kv_page_base,
    );
    // ⛔ A PAGE PART WITH NO PAGE DOES NOT LAUNCH — a short last chunk's padding past the
    // request's pages. Its rows are never read, and the only base left to aim it at is the
    // cursor's own page, which is part 0's keys.
    let Some(delta) = aim.deltas() else {
        return Ok(None);
    };
    off[SEG_KV.get()] += delta.kv.0;
    // The mask has its own segment precisely so a fold can shift it without disturbing
    // the running softmax state or any other activation.
    off[SEG_MASK.get()] += delta.mask.0;
    // AND THE PASS'S OWN ROW BLOCK. A fold pass emitted for ONE request's rows has to
    // be aimed at them, and `qs`, the score block, the whole online-softmax state and
    // `out` are all Intermediates, so one shift of seg0 moves every operand it reads
    // and writes. Zero for every bundle whose fold spans the batch.
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
    let wbase = segment_base(seg, banks, where_.weight_bank, SEG_WEIGHT.get())?;
    let mut ta = Vec::with_capacity(NUM_SEGMENTS);
    for (i, &sh) in off.iter().enumerate() {
        let base = if i == SEG_WEIGHT.get() {
            wbase
        } else {
            segment_base(seg, banks, where_.weight_bank, i)?
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
    // A BARRIER AFTER EVERY GROUP IS WHAT MAKES A 2 KB COPY COST A LAUNCH. It is a
    // device-wide settle, so consecutive groups cannot overlap however small they are —
    // and a batched decode layer is B per-request cache writes in a row, each moving
    // 2 KB into a DIFFERENT request's page. They are disjoint by construction, so
    // ordering them against EACH OTHER buys nothing; only the last has to settle before
    // the Kᵀ restickify and the next layer read what they wrote. Drop the barrier ONLY
    // between consecutive slot-writes; STRICT_ORDERING still holds the stream in order.
    // (The flag itself is a HOST fence call — `ComputeParams::pipeline_barrier` =
    // `issue_barrier` — never a wire bit: compute CBs are always built with the CTRL
    // barrier bit clear, pod-flex @ed96487 audit. The player ORs these into its one
    // doorbell.)
    // ⛔ AND ONLY WHEN THE NEXT ONE LAUNCHES (#620). A page part skipped for want of a
    // page leaves THIS write as the last before the readers, so it must settle like any
    // other last one.
    let next_is_slot_write = op.kv.slot_write
        && ops.get(oi + 1).is_some_and(|n| {
            n.kv.slot_write
                && fold_plan::PageGeometry::agree(&n.kv, fold).is_ok_and(|geo_n| {
                    fold_plan::seg_deltas(
                        &n.kv,
                        fold,
                        geo_n,
                        0,
                        per,
                        SlotPos::of_launch(where_.slot_pos.0),
                        where_.kv_page_base,
                    )
                    .deltas()
                    .is_some()
                })
        });
    let barrier = !next_is_slot_write;
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
    Ok(Some(LaunchSite {
        off,
        ta,
        prog,
        bootstrap,
        barrier,
    }))
}

/// The walk's staged batch for one launch call — the OLD way, built from the
/// same `op_reps`/`launch_site` the submit walk reads. This is the parity
/// gate's reference side. The submit walk keeps its own loop (its diag knobs
/// interleave with the arithmetic), which is exactly why the gate exists: if
/// the two loops ever disagree in enumeration or addressing, the rendered CBs
/// differ and prepare refuses. Returns the items and the walk's host-fence OR.
pub(crate) fn walk_items(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<(Vec<BatchedCompute>, bool)> {
    let mut items = Vec::new();
    let mut barrier = false;
    for oi in 0..ops.len() {
        let reps = op_reps(&ops[oi].kv, fold, where_.n_fold_pages, oi)?;
        for rep in 0..reps {
            // A page part with no page contributes neither an item nor a barrier — the
            // walk's own `continue`, in the twin.
            let Some(site) = launch_site(ops, oi, rep, fold, seg, banks, where_)? else {
                continue;
            };
            barrier |= site.barrier;
            items.push(site.batched_compute());
        }
    }
    Ok((items, barrier))
}

/// ONE (op, fold pass) of a launch sequence — `build_compute_cb`'s inputs as
/// data. Everything here is a constant of the (list, launch row, fold shape)
/// the plan was built for; the per-step scalars enter only at
/// [`CbPlan::render`].
pub(crate) struct CbPlanEntry {
    /// Diagnostic identity — the label the walk's own traces print plus the
    /// pass — so a parity refusal names its CB.
    label: String,
    prog: DevAddr,
    bootstrap_offset: u64,
    /// The op's own KV descriptor, carried whole: the render calls the SAME
    /// `fold_plan::seg_deltas` the walk calls, never a re-derivation.
    kv: OpKv,
    rep: i64,
    /// The HOST fence flag for this entry (`!next_is_slot_write`, computed at
    /// build against the enumeration's next op). A host fence call, never a
    /// wire bit — compute CBs are built with the CTRL barrier bit clear
    /// (`CtrlSection::new(tag, false)`; pod-flex @ed96487 audit). The player
    /// submits the OR of these with the image's one doorbell.
    barrier: bool,
    /// POSITIONAL, one per segment: the base the walk would resolve (the
    /// weight BANK's region for the weight segment, the segment's own
    /// otherwise) and the launch row's static shift — stored SEPARATELY, not
    /// pre-folded, so the render's one combined `shifted(static + delta)` is
    /// the walk's own arithmetic to the byte (a pre-shifted base would clamp
    /// differently at the region's end).
    bases: Vec<(DevAddr, u64)>,
}

/// One launch's CB constructor list — the walk's sequence as data, keyed by
/// (list, launch row, fold shape) at build time.
pub(crate) struct CbPlan {
    entries: Vec<CbPlanEntry>,
}

impl CbPlan {
    /// The staged batch the plan renders to, with the walk's host-fence OR.
    ///
    /// The fold state arrives per RENDER, not per plan — it is the per-step
    /// input: the same `seg_deltas` the walk calls, applied to the entry's
    /// recorded (base, static shift) with ONE combined `shifted` per segment,
    /// exactly as the walk computes it.
    pub(crate) fn render(
        &self,
        fold: &SessionKv,
        where_: &LaunchWhere,
    ) -> Result<(Vec<BatchedCompute>, bool)> {
        let Some(per) = UniformPages::sweeping(where_.n_fold_pages, fold) else {
            bail!(
                "cb_plan render: the page maps do not admit a uniform {}-page sweep",
                where_.n_fold_pages
            );
        };
        let mut items = Vec::with_capacity(self.entries.len());
        let mut barrier = false;
        for e in &self.entries {
            // ⭐ THE GEOMETRY IS MINTED PER ENTRY — each entry carries its own op's baked
            // modulus, and the render refuses (naming the entry) exactly as the walk's
            // `launch_site` would refuse naming the op: the twin's refusals track the
            // walk's at every step, not only its addresses.
            let geo = fold_plan::PageGeometry::agree(&e.kv, fold).map_err(|why| {
                anyhow!(
                    "cb_plan render: entry {} (slot {}): {why}",
                    e.label,
                    where_.slot_pos.0
                )
            })?;
            let aim = fold_plan::seg_deltas(
                &e.kv,
                fold,
                geo,
                e.rep,
                per,
                SlotPos::of_launch(where_.slot_pos.0),
                where_.kv_page_base,
            );
            // The walk skips an entry that does not launch at this `where_` (a page part
            // with no page); the render skips the SAME entry, so the twin's enumeration
            // tracks the walk's at every step the gate probes.
            let Some(delta) = aim.deltas() else {
                continue;
            };
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
            barrier |= e.barrier;
            items.push(BatchedCompute {
                prog: e.prog.clone(),
                tensor_allocs,
                bootstrap_offset: e.bootstrap_offset,
            });
        }
        Ok((items, barrier))
    }
}

/// Assemble the plan for one launch call: the (op, fold pass) enumeration and
/// every per-CB input that does not vary with the step's scalars. Built from
/// the same `op_reps` the walk calls; the bases come from the same
/// `segment_base` resolution the walk's `launch_site` performs.
pub(crate) fn build_cb_plan(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
) -> Result<CbPlan> {
    // The weight base is per-launch, not per-entry — resolve it once, the way
    // the walk resolves `wbase` before its segment loop.
    let wbase = segment_base(seg, banks, where_.weight_bank, SEG_WEIGHT.get())?;
    // The sweep witness, once, for the barrier probe below — the same
    // `UniformPages::sweeping` the walk's `launch_site` mints per pass; a fold
    // that cannot sweep refuses here exactly as the walk would refuse there.
    let Some(per) = UniformPages::sweeping(where_.n_fold_pages, fold) else {
        bail!(
            "cb_plan build: the page maps do not admit a uniform {}-page sweep",
            where_.n_fold_pages
        );
    };
    let mut entries = Vec::new();
    for oi in 0..ops.len() {
        let op = &ops[oi];
        let reps = op_reps(&op.kv, fold, where_.n_fold_pages, oi)?;
        // ⛔ THE NEXT SLOT-WRITE MUST ALSO LAUNCH (#620) for this one to drop its
        // barrier — a page part skipped for want of a page leaves THIS write as the
        // last before the readers. The probe is the walk's own predicate, verbatim,
        // so the twin's host-fence OR is the walk's at any `where_` the gate probes.
        let next_is_slot_write = op.kv.slot_write
            && ops.get(oi + 1).is_some_and(|n| {
                n.kv.slot_write
                    && fold_plan::PageGeometry::agree(&n.kv, fold).is_ok_and(|geo_n| {
                        fold_plan::seg_deltas(
                            &n.kv,
                            fold,
                            geo_n,
                            0,
                            per,
                            SlotPos::of_launch(where_.slot_pos.0),
                            where_.kv_page_base,
                        )
                        .deltas()
                        .is_some()
                    })
            });
        for rep in 0..reps {
            let mut bases = Vec::with_capacity(NUM_SEGMENTS);
            for i in 0..NUM_SEGMENTS {
                let base = if i == SEG_WEIGHT.get() {
                    wbase
                } else {
                    segment_base(seg, banks, where_.weight_bank, i)?
                };
                bases.push((base.clone(), where_.byte_off[i]));
            }
            entries.push(CbPlanEntry {
                label: format!("{}/rep{rep}", op.label()),
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
                rep,
                barrier: !next_is_slot_write,
                bases,
            });
        }
    }
    Ok(CbPlan { entries })
}

/// ⭐ THE 2a GATE. Render one launch both ways — the walk's staged batch and
/// the CbPlan's — through one render door, and REFUSE on the first differing
/// byte. Called only from `Executor::prepare` (and the split ladder's lazy
/// allocation): a mismatch means the plan's data twin of the walk is wrong,
/// and the player that will replay these images must not load. Returns the
/// number of CBs proven byte-identical.
pub(crate) fn parity_check(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    label: &str,
    stream: &Stream,
) -> Result<usize> {
    let (walked, walked_barrier) = walk_items(ops, fold, seg, banks, where_)?;
    let plan = build_cb_plan(ops, fold, seg, banks, where_)?;
    let (rendered, rendered_barrier) = plan.render(fold, where_)?;
    if walked.len() != rendered.len() {
        bail!(
            "[cbplan] PARITY REFUSED at {label}: the walk stages {} launch(es) but the plan \
             renders {} — the (op, fold-pass) ENUMERATION itself disagrees. The plan is the \
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
    let old = stream.render_compute_batch(&walked, 1)?;
    let new = stream.render_compute_batch(&rendered, 1)?;
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
                plan.entries[i].label
            );
        }
    }
    Ok(old.len())
}

// ─────────────────────────────────────────────────────────────────────────────
// 2b — the patch table: what moves between two steps of one image
// ─────────────────────────────────────────────────────────────────────────────

/// The per-step scalars — the WHOLE of what varies between two steps that
/// share an image key. Every address term the step moves is affine in these
/// two and nothing else:
///
/// * `slot_in_page` — `slot_write_delta`'s term: the write cursor's slot
///   within its page (`pos % page_slots`; the ABSOLUTE position for an
///   unpaged bundle, whose ops carry `page_slots == 0` and never wrap);
/// * `slab` — `slab_delta`'s index, `SlotPos::slab()` (`pos / SLAB_SLOTS`,
///   minted once in `fold_plan`, consumed here and by the delta).
///
/// The write PAGE is deliberately not one of them. Every step an image
/// replays writes the page its canvas was rendered on; the one step that
/// would move it — `pos == n·per`, slot 0 of the next page — is a page
/// BOUNDARY, and the player routes it to the walk, one step per page,
/// because a moved write base is not affine in any per-byte scalar: a
/// shifted write target's extent is END-PINNED (`DevAddr::shifted` returns
/// the bytes remaining past the shift), so its XLAT word's paddr and length
/// fields move together and no single per-byte addend expresses the pair —
/// measured on the first real bundle, where the page differential's render
/// moved paddr `+stride/2` flits and length `-stride/2` flits for a
/// one-stride base move and the solve refused.
///
/// The fold terms — per-pass page bases from the block tables, the mask and
/// intermediate rep strides — are per-rep CONSTANTS of the image: baked into
/// the canvas, never patched. The block tables are append-only within a row's
/// life, so a page base baked into an image stays valid for that row.
#[derive(Clone, Copy)]
pub(crate) struct StepScalars {
    pub(crate) slot_in_page: u64,
    pub(crate) slab: fold_plan::SlabIdx,
}

/// The scalar rule the player runs per step — ONE implementation shared by
/// the gate and 2c's player, so the coefficients derived here and the scalars
/// applied there cannot disagree about what "slot" and "slab" mean.
///
/// The slot term divides by the SESSION's page count here and by the OP's in
/// `slot_write_delta` — and the two can no longer disagree silently:
/// [`fold_plan::PageGeometry::agree`] refuses a slot-writing op whose baked
/// modulus differs from the session's before any delta or scalar is derived,
/// so the agreement this function always relied on is a checked fact, never
/// a comment.
pub(crate) fn step_scalars(paged: bool, per: i64, pos: i64) -> StepScalars {
    StepScalars {
        slot_in_page: (if paged {
            pos.rem_euclid(per.max(1))
        } else {
            pos
        }) as u64,
        slab: fold_plan::SlotPos::of_launch(pos).slab(),
    }
}

/// The canvas position's slot-within-page — ONE recipe shared by the
/// prepare gate's synthetic session and 2c's player, so the image the gate
/// proved and the image the player replays are cut the same way. It must
/// leave room for the +1 slot differential (≤ `per − 2` when paged, so the
/// step never crosses a page wrap) and the +1 slab differential (≤ 62, so
/// it never crosses a slab boundary), and it must not be slot 0 of a page
/// (a page boundary is the one step of the span the player does NOT replay —
/// it walks; see [`StepScalars`]). 5 — the gate's historical pick — whenever
/// it fits; `None` when no slot of the page can host a canvas (a page of
/// ≤ 2 slots), which the caller treats
/// as "no image for this shape", never as a wrong-coefficient derivation.
pub(crate) fn canvas_slot(per: i64, paged: bool) -> Option<i64> {
    let hi = if paged { per - 2 } else { 62 }.min(62);
    if hi >= 5 {
        Some(5)
    } else if hi >= 1 {
        Some(1)
    } else {
        None
    }
}

/// Whether a canvas at slot `r` leaves the OFF-SAMPLE ROOM the patch gate
/// needs: the `slot+65` sample — the one that crosses a slab boundary, and
/// so the only off-sample proof of the slab coefficient — must land strictly
/// inside the image's page span (the open interval `(n−1)·per < p < n·per`
/// the player replays), so `r + 66 ≤ per`. An unpaged session has no page
/// bound (its one image covers every position). ONE law, shared by the
/// prepare gate (without it, its patch leg skips loudly) and the player
/// (without it, every step walks): a page too narrow to host the proof
/// degrades to the walk, never to an unverified coefficient. Every shipped
/// bundle's page is 2048 slots, so this never fires in practice.
pub(crate) fn canvas_room(per: i64, paged: bool, r: i64) -> bool {
    !paged || r + 66 <= per
}

/// One 8-byte word of one CB image that moves with the step: where it sits,
/// and the addend per unit of each scalar — MEASURED by finite differences
/// through the render door, never read off the wire layout by hand. The XLAT
/// word packs paddr, mode and length in disjoint bit ranges, so a step's
/// movement is an additive delta within the word; these are that delta per
/// scalar unit, and [`PatchPlan::apply`] adds them with two's-complement
/// wrap (an image's early steps sit at slots BELOW the canvas's, so slot
/// deltas go NEGATIVE).
pub(crate) struct Patch {
    /// Which CB of the image — an index into the rendered batch.
    cb: usize,
    /// Byte offset of the 8-byte word within the CB. Every section is
    /// 8-aligned, so a moving field never straddles words.
    dst: usize,
    a_slot: i64,
    b_slab: i64,
}

/// The patches that turn one image into any step's CBs: the scalars the
/// canvas was rendered at, every moving word's affine addend, and the
/// launch's host-fence OR — the one `pipeline_barrier` the player submits
/// the whole image with (the walk's per-op ORs collapse to it; see
/// `Stream::compute_batch`'s flush). Built by [`build_patch_plan`] — at
/// prepare for the gate, at an image's first use for 2c's player — and
/// applied by copying the canvas and calling [`PatchPlan::apply`].
pub(crate) struct PatchPlan {
    reference: StepScalars,
    patches: Vec<Patch>,
    /// The OR of every entry's host-fence flag — a constant of the plan
    /// (entry-keyed, position-independent), so one value serves every step
    /// this image replays.
    pub(crate) barrier: bool,
}

impl PatchPlan {
    /// How many moving words the plan patches — the player's per-step patch
    /// workload, reported by its timers.
    pub(crate) fn sites(&self) -> usize {
        self.patches.len()
    }

    /// Patch a copied canvas in place for one step's scalars. The addend is
    /// computed in i128 and wrapped to u64: an image's early steps sit below
    /// the canvas slot, so slot deltas go negative, and the wire's arithmetic
    /// is two's-complement addition.
    pub(crate) fn apply(&self, image: &mut [ControlBlockWire], s: &StepScalars) {
        let d_slot = s.slot_in_page as i128 - self.reference.slot_in_page as i128;
        let d_slab = s.slab.get() as i128 - self.reference.slab.get() as i128;
        for p in &self.patches {
            let addend = (p.a_slot as i128 * d_slot + p.b_slab as i128 * d_slab) as u64;
            let b = image[p.cb].bytes_mut();
            let mut w = [0u8; 8];
            w.copy_from_slice(&b[p.dst..p.dst + 8]);
            let word = u64::from_le_bytes(w).wrapping_add(addend);
            b[p.dst..p.dst + 8].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// One little-endian word of a rendered image — the unit the patch model
/// works in (fields never straddle words; every section is 8-aligned).
fn word_at(image: &[ControlBlockWire], cb: usize, dst: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&image[cb].bytes()[dst..dst + 8]);
    u64::from_le_bytes(w)
}

/// Derive one image's patch table: render the canvas (the plan's own render —
/// the caller proves it byte-identical to the walk at this `where_`, which
/// [`patch_check`] does), render two differentials, and read every moving
/// word's coefficients off the differences. Finite differences through the
/// door: there is no second analysis of the wire layout to drift, and
/// anything that moves but is not affine in the two scalars refuses here or
/// mismatches in [`patch_check`]'s off-sample verification.
///
/// The canvas position must leave room for the differentials: the +1 step may
/// not cross a slab or page boundary, because it isolates the slot term. That
/// is the caller's choice — the gate picks it, and a violated precondition
/// is a REFUSAL, never a silently wrong coefficient.
///
/// Returns the canvas image and its patch plan: what 2c's player copies and
/// patches per step.
pub(crate) fn build_patch_plan(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    stream: &Stream,
) -> Result<(Vec<ControlBlockWire>, PatchPlan)> {
    let per = fold.page_slots.get().max(1);
    let pos = where_.slot_pos.0;
    if fold_plan::SlotPos::of_launch(pos).is_last_slot_of_its_slab() {
        bail!(
            "cb_plan patch: canvas position {pos} sits on a slab boundary — the +1 differential \
             cannot isolate the slot term from the slab term"
        );
    }
    if fold.paged && pos.rem_euclid(per) > per - 2 {
        bail!(
            "cb_plan patch: canvas position {pos} sits on a page boundary — the +1 differential \
             cannot isolate the slot term from the page wrap"
        );
    }
    // ⭐ THE OP-VS-SESSION PAGE-MODULUS LAW, AS A MINT — this loop was the canary the
    // boundary sample used to carry; `PageGeometry::agree` is now the ONE spelling of
    // the law, and every launch (`launch_site`), every render (`CbPlan::render`) and
    // this gate mint it, so a bundle whose ops disagreed with the session about the
    // page refuses at its FIRST launch, by name — not only when a CB image is cut.
    // An op that wrapped at a different modulus would put a page wrap INSIDE an
    // image's replayed span, which the patch model would replay linearly across.
    for o in ops {
        fold_plan::PageGeometry::agree(&o.kv, fold).map_err(|why| {
            anyhow!(
                "cb_plan patch: {why}. Refusing to patch a bundle whose ops disagree with \
                 the session about the page."
            )
        })?;
    }

    let plan = build_cb_plan(ops, fold, seg, banks, where_)?;
    let render = |w: &LaunchWhere| -> Result<Vec<ControlBlockWire>> {
        let (items, _) = plan.render(fold, w)?;
        Ok(stream.render_compute_batch(&items, 1)?)
    };
    // The canvas, rendered once — and the host-fence OR with it: a constant
    // of the plan (entry-keyed, the same for every render of this plan), so
    // the canvas render's value is the image's, what the player submits
    // every step of this image with.
    let (items, barrier) = plan.render(fold, where_)?;
    let canvas = stream.render_compute_batch(&items, 1)?;
    // The two differentials. Each moves exactly one scalar's worth of
    // arithmetic — the other held — so each moving word's measured difference
    // is one equation in that word's coefficients. There is NO page
    // differential: the write page never moves inside an image's replayed
    // span (the boundary step walks; see [`StepScalars`]), so no coefficient
    // for it exists to measure.
    let img_slot = render(&LaunchWhere {
        slot_pos: SeqPos(pos + 1),
        ..*where_
    })?;
    let img_slab = render(&LaunchWhere {
        slot_pos: SeqPos(pos + fold_plan::SLAB_SLOTS),
        ..*where_
    })?;
    for (name, img) in [("slot", &img_slot), ("slab", &img_slab)] {
        if img.len() != canvas.len() {
            bail!(
                "cb_plan patch: the {name} differential renders {} CB(s) but the canvas holds {} \
                 — the (op, fold-pass) enumeration moved with a step scalar, so no image can key \
                 it. Refusing to build a patch table for a moving sequence.",
                img.len(),
                canvas.len()
            );
        }
    }
    // How much the slot term moves in the slab differential: +SLAB_SLOTS
    // unwrapped, or that minus one page when it crosses a page boundary — the
    // wrap is arithmetic the model already covers (the slot scalar itself
    // wraps), so the coefficient solve just needs the true delta.
    let dslot_slab = if fold.paged {
        (pos + fold_plan::SLAB_SLOTS).rem_euclid(per) - pos.rem_euclid(per)
    } else {
        fold_plan::SLAB_SLOTS
    };
    let mut patches = Vec::new();
    for cb in 0..canvas.len() {
        for dst in (0..CB_NUM_BYTES).step_by(8) {
            let c = word_at(&canvas, cb, dst);
            let d_slot = word_at(&img_slot, cb, dst).wrapping_sub(c) as i64;
            let d_slab = word_at(&img_slab, cb, dst).wrapping_sub(c) as i64;
            if d_slot == 0 && d_slab == 0 {
                continue;
            }
            // Δslot = +1 with the slab held ⇒ the slot coefficient is the raw difference.
            let a_slot = d_slot;
            // Δslab = +1 with the slot moving `dslot_slab` ⇒ the slab coefficient is what
            // remains after the slot's contribution is subtracted.
            let b_slab = d_slab.wrapping_sub(a_slot.wrapping_mul(dslot_slab));
            patches.push(Patch {
                cb,
                dst,
                a_slot,
                b_slab,
            });
        }
    }
    let reference = step_scalars(fold.paged, per, pos);
    Ok((
        canvas,
        PatchPlan {
            reference,
            patches,
            barrier,
        },
    ))
}

/// One launch row's replay image — what 2c's player caches per row and
/// concatenates into a step image: the canvas it copies, and the patch plan
/// that turns the copy into any step of the same image key. The plan's
/// `barrier` is the row's host-fence OR; the player ORs the rows it merges.
pub(crate) struct RowImage {
    pub(crate) canvas: Vec<ControlBlockWire>,
    pub(crate) plan: PatchPlan,
}

/// One launch row's canvas with NO patch table — what a step-CONSTANT row
/// (prefix/suffix, the player's `follows_step == false` rows) replays from,
/// in full. A constant `where_` makes every step-differential zero, so such
/// a row's patch table is empty BY CONSTRUCTION — deriving one anyway would
/// probe a counterfactual no replay executes (measured on the first real
/// bundle: the gate's page probe walked the row's write base a full stride
/// past where any launch of that row ever puts it, the render returned a
/// structurally different allocation, and the solve refused). Parity
/// at the constant `where_` is the row's whole correctness statement; the
/// player copies this canvas verbatim and never consults the empty plan.
pub(crate) fn render_row(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    stream: &Stream,
) -> Result<RowImage> {
    let plan = build_cb_plan(ops, fold, seg, banks, where_)?;
    let (items, barrier) = plan.render(fold, where_)?;
    let canvas = stream.render_compute_batch(&items, 1)?;
    let per = fold.page_slots.get().max(1);
    Ok(RowImage {
        canvas,
        plan: PatchPlan {
            reference: step_scalars(fold.paged, per, where_.slot_pos.0),
            patches: Vec::new(),
            barrier,
        },
    })
}

/// ⭐ THE 2b GATE — the patch table proven off-sample, at prepare. Builds the
/// patch plan for one launch's image and then VERIFIES it against fresh walk
/// renders at steps neither the canvas nor the differentials visited: the
/// patched canvas must equal the OLD way's bytes, byte for byte, or the load
/// refuses. This is the whole correctness statement of 2c's player — replay
/// (canvas + patches) ≡ re-derive (the walk) — proven at prepare under a
/// synthetic session of the player's own shape, so no step ever pays for it.
/// The player runs this SAME check per image at the image's first use, under
/// the live session — and keeps the verified image it returns.
///
/// The sampled steps are the image's own OPEN span `(n−1)·per < p < n·per` —
/// exactly the steps the player replays this image over. The page BOUNDARY
/// (`p == n·per`, slot 0 of the next page) is deliberately not sampled: the
/// player routes it to the walk (see [`StepScalars`]), so there is no patched
/// render of it to prove — and an op that disagreed with the session about
/// the page modulus is refused outright by [`build_patch_plan`], by name,
/// rather than caught at a boundary probe.
/// Returns the number of patch sites verified and the
/// verified image itself.
pub(crate) fn patch_check(
    ops: &Ops,
    fold: &SessionKv,
    seg: &[Option<DevAddr>; NUM_SEGMENTS],
    banks: &[Option<DevAddr>],
    where_: &LaunchWhere,
    label: &str,
    stream: &Stream,
) -> Result<(usize, RowImage)> {
    // The canvas the patches are measured against is itself proven against
    // the walk at this exact `where_` — patching a canvas that already
    // disagreed with the walk would verify nothing.
    parity_check(ops, fold, seg, banks, where_, label, stream)?;
    let (canvas, patch_plan) = build_patch_plan(ops, fold, seg, banks, where_, stream)
        .map_err(|e| anyhow!("[cbpatch] {label}: patch-table derivation REFUSED — {e:#}"))?;
    let per = fold.page_slots.get().max(1);
    let pos = where_.slot_pos.0;
    let n = where_.n_fold_pages;
    // A paged step belongs to the image keyed by ceil(pos/per) — sample only
    // within the canvas's own key AND off its page boundary, which is the
    // OPEN span the player replays this image over (the boundary step walks).
    // An unpaged bundle has one image for every position.
    let in_image = |p: i64| !fold.paged || (p > (n - 1) * per && p < n * per);
    let mut samples: Vec<(i64, &str)> = vec![
        (pos + 5, "slot+5"),
        (pos + 33, "slot+33"),
        (pos + 65, "slot+65/slab+1"),
        (pos + 130, "slot+130/slab+2"),
    ];
    if pos >= 3 {
        samples.push((pos - 3, "slot-3 (negative delta)"));
    }
    let samples: Vec<_> = samples.into_iter().filter(|(p, _)| in_image(*p)).collect();
    if samples.is_empty() {
        bail!(
            "[cbpatch] {label} REFUSED: no off-sample step fits this image's position range — \
             the gate would pass vacuously"
        );
    }
    // Every sample renders the walk at the canvas's OWN page base — the base
    // every replayed step of this image writes at.
    let walk_render = |spos: i64| -> Result<Vec<ControlBlockWire>> {
        let (items, _) = walk_items(
            ops,
            fold,
            seg,
            banks,
            &LaunchWhere {
                slot_pos: SeqPos(spos),
                ..*where_
            },
        )?;
        Ok(stream.render_compute_batch(&items, 1)?)
    };
    for (spos, what) in samples {
        let expected = walk_render(spos)?;
        if expected.len() != canvas.len() {
            bail!(
                "[cbpatch] {label} REFUSED at sample `{what}`: the walk renders {} CB(s) but the \
                 image holds {} — the (op, fold-pass) enumeration moved with the step, so no \
                 image can key it",
                expected.len(),
                canvas.len()
            );
        }
        let mut got = canvas.clone();
        patch_plan.apply(&mut got, &step_scalars(fold.paged, per, spos));
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            if a != b {
                let (mut at, mut wb, mut pb) = (0, 0u8, 0u8);
                for (j, (x, y)) in a.bytes().iter().zip(b.bytes().iter()).enumerate() {
                    if x != y {
                        (at, wb, pb) = (j, *x, *y);
                        break;
                    }
                }
                bail!(
                    "[cbpatch] {label} REFUSED at sample `{what}`: CB {i} of {} differs after \
                     patching — first differing wire byte at +{at}: patched {wb:#04x} vs the \
                     walk's {pb:#04x}. The image-plus-patches model of this launch is not \
                     byte-identical to re-deriving the step; refusing to load rather than \
                     replay a wrong image.",
                    canvas.len()
                );
            }
        }
    }
    Ok((
        patch_plan.patches.len(),
        RowImage {
            canvas,
            plan: patch_plan,
        },
    ))
}
