// SPDX-License-Identifier: Apache-2.0
//! The resident SuperDSC executor — the live on-silicon run path, in Rust.
//!
//! The SuperDSC emitter produces a dxp device bundle (one device program per launch group + the
//! placement manifest). This module runs it DIRECTLY on the AIU through [`crate::runtime`]: the
//! program, the packed weights and the paged KV pool are allocated ONCE in [`Executor::prepare`]
//! and live on-card; each [`Executor::predict`] DMAs only the (small) activations in and the
//! logits out — weights and KV are NEVER re-uploaded. That residency is the whole perf point.
//!
//! fp16 is the device-native SEN169 (`senfp169`), not IEEE fp16: every 2-byte element converts
//! IEEE→sen on H2D and sen→IEEE on D2H ([`crate::sen_convert`]).
//!
//! ⛔ WHAT LIVES WHERE. Nothing here computes a KV address: which request an op touches, how many
//! passes its group takes and what that shifts each segment base by are all [`crate::fold_plan`]'s,
//! which owns the page maps and the write cursors. The bundle's own shape is
//! [`crate::bundle_code`]'s, byte conversion is [`crate::sen_convert`]/[`crate::stage_weight`]'s, and the card
//! surface is [`crate::runtime`]'s. This module is the sequencing: what to allocate, what to launch, in what
//! order, with which base.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use flex_rs::memory_region::MemoryType;
use flex_rs::stream::RuntimeStreamError;
use tracing::{debug, info};

use crate::bundle_code::{self as bundle, KernelWeight, NUM_SEGMENTS, Placement, SweptCols};
use crate::cb_plan;
use crate::fold_plan::{
    self, Bytes, OpKv, PageOrd, PerClass, RowCursors, RowIdx, RowPage, SessionKv, SlabStride,
    SlotPos, SlotStride, fused_rungs,
};
use crate::runtime::{
    BatchedCompute, ByteOffset, ByteSize, ComputeWireBatch, ControlBlockWire, DevAddr, HwError,
    Stream,
};
use crate::sen_convert;
use crate::stage_weight::{Element, RetileWalk, stage_weight_tiled};

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Typed quantities. Every number below names WHICH number it is; the executor never takes a bare
//  integer for a thing that has a unit, because the whole bug class this path has produced is one
//  quantity standing in for another (a slot for a position, a page for a row, elements for bytes).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A TENSOR-SEGMENT index, proven in range by construction. The segment COUNT is a property of the
/// executor model (the hardware caps a launch at 7 tensor segments; seg7 is the program), so it is a
/// CONST GENERIC — and a `Seg` therefore cannot name a segment the session has no region for.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Seg<const N: usize>(usize);

impl<const N: usize> Seg<N> {
    /// A segment named at COMPILE TIME. `I >= N` is a compile error at the const-eval of the call,
    /// not a runtime check — which is what makes the named roles below unfalsifiable.
    pub const fn known<const I: usize>() -> Self {
        assert!(I < N, "segment index out of range for this executor model");
        Seg(I)
    }

    /// A segment named at RUNTIME (a manifest placement, an FFI argument). `None` is a refusal:
    /// the caller has an index this bundle has no region for.
    pub fn checked(i: i64) -> Option<Self> {
        (i >= 0 && (i as usize) < N).then_some(Seg(i as usize))
    }

    /// The index, for array subscripting. Named rather than `.0` so the sites are countable.
    pub const fn get(self) -> usize {
        self.0
    }

    /// Every segment of the model, in order — the positional `tensor_allocs` walk.
    pub fn all() -> impl Iterator<Item = Seg<N>> {
        (0..N).map(Seg)
    }
}

/// This executor's segment type: `SegRole` order, 7 packed tensor segments.
pub type SegIdx = Seg<NUM_SEGMENTS>;

/// INTERMEDIATE (seg0) — activations and the online-softmax state. A row-batched fold pass shifts
/// here to reach its own request's row block.
pub const SEG_INTERMEDIATE: SegIdx = SegIdx::known::<0>();
/// WEIGHT (seg1) — the per-layer resident weights; the rolled body advances by `weight_stride`.
pub const SEG_WEIGHT: SegIdx = SegIdx::known::<1>();
/// KV (seg2) — the resident cache/pool; carries the write slot, the page base and the slab shift.
pub const SEG_KV: SegIdx = SegIdx::known::<2>();
/// MASK (seg3) — the prefix-validity mask, in its own segment precisely so a fold can shift it
/// without disturbing the running softmax state or any other activation.
pub const SEG_MASK: SegIdx = SegIdx::known::<3>();

/// ⭐ THE SEGMENTS A BORROWER ADOPTS INSTEAD OF FILLING — the resident ones. Three sites read this
/// (host-shadow sizing, device sizing, H2D) and they MUST agree: a segment sized as a placeholder but
/// then H2D'd from that placeholder writes 128 B of zeros over the owner's live data. Spelling the
/// set once is what keeps the three in step, and it is the list to extend if the weights ever occupy
/// more than one segment (see `scratchy_spyre_bundle::MAX_SEGMENT_BYTES`).
pub const SEG_RESIDENT: [SegIdx; 2] = [SEG_WEIGHT, SEG_KV];

/// Whether `i` names a resident segment (see [`SEG_RESIDENT`]).
fn is_resident_seg(i: usize) -> bool {
    SEG_RESIDENT.iter().any(|s| s.get() == i)
}

/// Logits column count.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Vocab(pub usize);
/// One resident K/V row width (= kv_heads·head_dim), in fp16 elements.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct KvDim(pub usize);
/// Physical pages in the resident pool.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PoolPages(pub i64);
/// Positions per page.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PageSlots(pub i64);
/// An ABSOLUTE sequence position — how many positions are already resident for this forward.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub struct SeqPos(pub i64);
/// How many NEW positions one forward appends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NewTokens(pub i64);
/// A layer index within the rolled body loop.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayerIdx(pub i64);

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Diagnostics — every env-gated instrument the C++ carried, read ONCE (a per-launch getenv is a
//  linear environ scan on the critical path between compute submissions).
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The measurement switches, sampled once per process.
/// WHERE a launch lands: the per-segment byte offsets it shifts by, which KV slot and page it writes,
/// how many fold passes it makes, and which weight region fills the positional weight slot.
///
/// One value rather than five parameters because they are ONE fact — "this launch, at this position" —
/// and they are always chosen together at the call site that knows the layer index. Passing them
/// separately also pushed `launch_ops_inner` past the argument limit.
#[derive(Clone, Copy)]
pub(crate) struct LaunchWhere {
    pub(crate) byte_off: [u64; NUM_SEGMENTS],
    pub(crate) slot_pos: SeqPos,
    pub(crate) kv_page_base: Bytes,
    pub(crate) n_fold_pages: i64,
    /// Which weight bank backs the positional weight slot — see `weight_bank_for`.
    pub(crate) weight_bank: usize,
}

/// One rung launch staged for the rung's single doorbell. `launch_ops_inner` collects these across
/// its whole (op, rep) loop and flushes once at the end ([`crate::runtime::Stream::compute_batch`]
/// — the vendor's multi-CB branch). The site context rides alongside the item because the per-op
/// `fault` closure dumps from loop locals (`ta`/`off`/the fold rep), which are gone by flush time:
/// a refused flush rebuilds the offending entry's dump from these, and the flex-rs error names the
/// entry index it refused.
struct PendingLaunch {
    item: BatchedCompute,
    /// Which op of the rung (`oi`) and which fold rep produced this entry.
    op_index: usize,
    rep: i64,
    /// The per-segment shifts this entry launched with — what `dump_launch_site` prints.
    off: [u64; NUM_SEGMENTS],
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  #603 PR2 2c — the replay player's cached step images
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One launch row inside a step image: where the row's canvas sits in the
/// concatenated step canvas, whether the row's render FOLLOWS the step
/// scalars, and its patch plan.
///
/// `follows_step` is a law of [`Executor::launch_forward`], read here as
/// data: the body rows launch at the STEP's position/page/count (their
/// bytes move with the step — patched), while the prefix and suffix launch
/// at slot 0 / page 0 / one page WHATEVER the step is (their bytes are the
/// same every step — copied, never patched). A row that is fixed replayed
/// through its plan would be wrong twice over: the plan was never verified
/// for the step's slots, and the walk never renders it there.
struct RowReplay {
    base: usize,
    len: usize,
    follows_step: bool,
    plan: cb_plan::PatchPlan,
}

/// The immutable half of a cached image — `Arc`'d because TWO handles must
/// share it: the cache keeps one, and every step's `ComputeWireBatch` carries
/// the canvas (one more `Arc`, to the same allocation) plus a stamp closure
/// holding this whole struct for the row table. All plain data, so the
/// closure is `Send` for free.
struct StepImageData {
    canvas: std::sync::Arc<Vec<ControlBlockWire>>,
    rows: Vec<RowReplay>,
}

/// One cached step image — the whole forward (prefix + every body row +
/// suffix) as ONE canvas, the row table that patches it, and the host-fence
/// OR the step's single submission carries (the walk's per-row ORs collapse
/// into it; see the flush at the end of `launch_ops_inner`).
#[derive(Clone)]
struct StepImage {
    data: std::sync::Arc<StepImageData>,
    barrier: bool,
}

/// What an image is keyed by — everything the render reads that is not
/// per-step: WHICH body the ladder picks for this context, how many pages
/// the resident prefix spans, and the fold shape (below).
#[derive(Clone, PartialEq, Eq, Hash)]
struct StepImageKey {
    sel: BodySel,
    n_fold_pages: i64,
    /// The fold state the canvas was rendered under: the geometry and
    /// request count, plus row 0's first `n_fold_pages` page ids. Those ids
    /// are the ONE fold mutation that can invalidate an image: append-only
    /// within a request's life (the prefix the fold sweeps is stable), but
    /// a NEW request reuses the row with different pages.
    ///
    /// ⛔ NO ROW CURSORS IN THE KEY, ON PURPOSE. They advance every step —
    /// keying on them would derive a fresh image per token. A single-request
    /// launch is `LaunchWidth::Single`, and `RequestSlot::resolve`'s Single
    /// arm answers the launch's own position without reading any cursor, so
    /// the render genuinely does not depend on them. `fold_requests != 1`
    /// never reaches the player (it takes the walk), which is what makes
    /// that true for every image in this cache.
    signature: Vec<i64>,
}

/// The image cache WITH its accounting. The counters live on the two arms
/// that fire them — not inlined at the executor's lookup site — because the
/// executor cannot be built without a card, and the accounting is exactly
/// what a cardless test must reach: the cadence line's
/// `% from cached` is `replay_hits` over `replay_steps`, and it read
/// structurally zero from the day the player landed — the hit arm timed the
/// lookup without counting the hit, so a card replaying 256 steps off 4
/// derivations still printed `0.0% from cached`. The cache worked; the
/// counter was the defect. Two methods rather than one lookup-or-derive
/// closure because the derive path needs `&mut self`, which a closure over
/// the cache's own borrow cannot capture.
#[derive(Default)]
struct StepImageCache {
    map: std::collections::HashMap<StepImageKey, StepImage>,
}

impl StepImageCache {
    /// ⭐ THE HIT ARM — lookup, timing, AND THE COUNT: `replay_hits` is the
    /// numerator of the cadence line's `% from cached`, and the hit arm never
    /// counted it — no line of code ever incremented it. Timed as the player
    /// always timed a hit — around the get+clone, guard acquisition included.
    fn lookup_cached(&mut self, key: &StepImageKey) -> Option<StepImage> {
        let t0 = Instant::now();
        let hit = self.map.get(key).cloned();
        if hit.is_some() {
            let mut t = timers();
            t.replay_hits += 1;
            t.replay_lookup_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        hit
    }

    /// The MISS arm — count the derivation and what it cost, then keep the
    /// image for every later step on this key. The cost window is the derive
    /// alone (`started` predates it, and the insert sits outside both the
    /// window and the guard), matching the player's original accounting.
    fn note_derived(&mut self, key: StepImageKey, image: StepImage, started: Instant) {
        {
            let mut t = timers();
            t.replay_derives += 1;
            t.replay_derive_ms += started.elapsed().as_secs_f64() * 1e3;
        }
        self.map.insert(key, image);
    }
}

struct Diag {
    /// `SCRATCHY_SDSC_PEROP_SYNC`: re-enable the per-group host drain. DEFAULT-OFF —
    /// pipeline_barrier + the stream's STRICT_ORDERING guarantee the dataflow, and the drain cost
    /// ~11% of decode (pod-confirmed coherent without it: 24.7 → 27.4 tok/s).
    perop_sync: bool,
    /// `SCRATCHY_SDSC_GROUP_TIME`: per-launch-group device time. Needs a drain to attribute anything,
    /// so it costs what the per-op sync costs — a MEASUREMENT MODE, not something to leave on.
    group_time: bool,
    /// `SCRATCHY_SDSC_SUBMIT_TIME`: host enqueue cost per launch, with NO drain, so the pipeline is
    /// left alone. The honest answer to "is a batch's launch count the cost".
    submit_time: bool,
    /// `SCRATCHY_SDSC_PREFIX_TIME`: one drain per call at a ROTATING group, giving a cumulative
    /// ladder whose successive differences are the per-group device times.
    prefix_time: bool,
    /// `SCRATCHY_SDSC_RB_TIME`: per-op device CYCLES read out of the response blocks' own timestamp
    /// section. The harvest itself is always on — the completion thread records every completed
    /// CB's five cycle counts (`flex_rs::rb_telemetry`) — so this knob costs nothing but the
    /// printing, and needs NO drain beyond the one `synchronize` a step already performs. It is
    /// the drain-free answer to what `group_time`/`prefix_time` pay ~11% of decode to estimate.
    /// Combine it with `repeat_probe` and per-op attribution is skipped loudly (the probe's extra
    /// launches are in the ring but not in the sequence).
    rb_time: bool,
    /// `SCRATCHY_SDSC_REPEAT_PROBE`: launch the same program twice back to back — separates "a
    /// launch is expensive" from "switching programs is expensive".
    repeat_probe: bool,
    /// `SCRATCHY_SDSC_BODY_TRACE`: which body a step actually selected.
    body_trace: bool,
    /// `SCRATCHY_SDSC_SEGADDR_DIAG`: one-shot dump of every segment's resolved runtime base.
    segaddr: bool,
    /// `SCRATCHY_SDSC_PROG_VERIFY`: before every launch, D2H the program allocation and compare it
    /// against the bytes the bake put there — printing ONLY when they differ.
    ///
    /// ⭐ IT ANSWERS A QUESTION THE FAULT PATH CANNOT. The unconditional readback on a failed launch
    /// happens AFTER the hardware already faulted, on a stream the abend may have poisoned, so a
    /// `rc != 0` there is ambiguous between "the image is fine" and "we could not look". Reading
    /// before each launch is the same comparison on a healthy stream, and it dates the divergence:
    /// a program that reads back wrong BEFORE it is ever launched was clobbered by the upload or by
    /// an earlier launch, not by its own.
    ///
    /// A whole-image D2H plus a drain per launch is far too expensive to leave on, hence the knob.
    prog_verify: bool,
    /// `SCRATCHY_SDSC_PHASE_TIME`: preamble-H2D / compute / logits-D2H split of a predict.
    phase_time: bool,
    /// `SCRATCHY_SDSC_PREP_TIME`: sub-phase timing of prepare.
    prep_time: bool,
    /// `SCRATCHY_SDSC_OPTRACE`: after each on-card op, D2H every segment and print nonzero-count +
    /// max|.| — the op index where the signal collapses IS the divergence.
    optrace: bool,
    /// `SCRATCHY_SDSC_TENSORDUMP=<dir>`: after each on-card op, D2H the readable segments and write
    /// every placement whose bytes MOVED since the last dump as LE-f32 files named to match the
    /// emulator's `SCRATCHY_EMU_TENSORDUMP` (`prog{idx}_seg{seg}_t{tid}.bin` + `.n`, plus a
    /// `op{op}_…` timeline copy and an `ops.txt` mapping op numbers to programs) — the bisection
    /// instrument for a card-vs-emulator divergence. The prog index is a per-FINGERPRINT global in
    /// first-dumped order, and `progs.txt` records the assignment so a diff never has to guess it.
    tensordump: Option<std::path::PathBuf>,
    /// `SCRATCHY_SDSC_LAUNCH_MARK`: one line BEFORE each submit naming the op and rep.
    ///
    /// ⭐ IT EXISTS FOR THE ONE FAULT NO OTHER PATH CAN ATTRIBUTE. `RAS::PCI::BusFence` (0xa35e) is
    /// raised on senlib's OWN `PfMSIMonitor::PollingThread`, which `throw`s into a thread with no
    /// handler — so the process `abort`s and EVERY host-side fault path (including `fault`'s
    /// submit-refused dump above) never runs. The last line this printed is therefore the only
    /// statement of which op the card was executing, and `aiuras.json` gives `BusFence` no `vars`,
    /// so the fence itself names nothing at all.
    launch_mark: bool,
    /// `SCRATCHY_SDSC_RESET_PROBE=N`: after the N-th forward, tear the runtime down and measure the
    /// RSS drop. BREAKS the session; a diagnostic to pick a fix, not a fix.
    reset_probe: Option<i64>,
}

impl Diag {
    fn get() -> &'static Diag {
        static D: std::sync::OnceLock<Diag> = std::sync::OnceLock::new();
        D.get_or_init(|| {
            let on = |k: &str| std::env::var_os(k).is_some();
            Diag {
                perop_sync: on("SCRATCHY_SDSC_PEROP_SYNC"),
                group_time: on("SCRATCHY_SDSC_GROUP_TIME"),
                submit_time: on("SCRATCHY_SDSC_SUBMIT_TIME"),
                prefix_time: on("SCRATCHY_SDSC_PREFIX_TIME"),
                rb_time: on("SCRATCHY_SDSC_RB_TIME"),
                repeat_probe: on("SCRATCHY_SDSC_REPEAT_PROBE"),
                body_trace: on("SCRATCHY_SDSC_BODY_TRACE"),
                segaddr: on("SCRATCHY_SDSC_SEGADDR_DIAG"),
                prog_verify: on("SCRATCHY_SDSC_PROG_VERIFY"),
                phase_time: on("SCRATCHY_SDSC_PHASE_TIME"),
                prep_time: on("SCRATCHY_SDSC_PREP_TIME"),
                optrace: on("SCRATCHY_SDSC_OPTRACE"),
                tensordump: std::env::var_os("SCRATCHY_SDSC_TENSORDUMP")
                    .map(std::path::PathBuf::from),
                launch_mark: on("SCRATCHY_SDSC_LAUNCH_MARK"),
                reset_probe: std::env::var("SCRATCHY_SDSC_RESET_PROBE")
                    .ok()
                    .and_then(|v| v.parse().ok()),
            }
        })
    }

    /// Whether EVERY instrument is off — the production configuration. The replay player
    /// (#603 PR2 2c) requires this: each knob above reads state the walk produces (per-op
    /// marks, per-op readbacks, per-op rb_seq entries, per-op dump files, mid-rung drains),
    /// so rather than reason per knob about which are batch-safe one level up, ANY instrument
    /// falls back to the walk bit for bit — the same discipline the collect path already
    /// applies to its five mid-rung knobs ("a drain between ops of an unflushed rung observes
    /// nothing"), extended to the player's whole surface. A knob run therefore measures the
    /// walk, never a half-player; the player's own phase timers (`Timers::replay_*`) carry
    /// its decomposition instead.
    fn is_quiet(&self) -> bool {
        let Diag {
            perop_sync,
            group_time,
            submit_time,
            prefix_time,
            rb_time,
            repeat_probe,
            body_trace,
            segaddr,
            prog_verify,
            phase_time,
            prep_time,
            optrace,
            tensordump,
            launch_mark,
            reset_probe,
        } = self;
        !(*perop_sync
            || *group_time
            || *submit_time
            || *prefix_time
            || *rb_time
            || *repeat_probe
            || *body_trace
            || *segaddr
            || *prog_verify
            || *phase_time
            || *prep_time
            || *optrace
            || *launch_mark)
            && tensordump.is_none()
            && reset_probe.is_none()
    }
}

/// One pipeline's harvested TS-phase durations plus a CB count, NAMED per the card-measured slot
/// semantics ([`flex_rs::control_block_wire::ResponseBlockWire::dmai_cycles`] and siblings: the
/// five TS words are event snapshots of one wrapping cycle counter, so a phase duration is the
/// wrapping difference of its pair — never a raw sum, which is why this holds durations and not
/// slot totals). `exec` is the headline — the compute engine's own time, which sums to a
/// forward's device span; `prep` is the queue+prep wait behind earlier CBs on a pipelined
/// stream; `tail` is the writeback tail on compute CBs and the D2H duration on D2H CBs.
#[derive(Default)]
struct RbCycles {
    dmai: u64,
    prep: u64,
    exec: u64,
    tail: u64,
    cbs: u64,
}

impl RbCycles {
    /// A compute CB: every phase is its own (`dmai` is exactly 0 — the card writes the same
    /// snapshot to words 0 and 1 when there is no DMA-in).
    fn add_compute(&mut self, rb: flex_rs::control_block_wire::ResponseBlockWire) {
        self.dmai += rb.dmai_cycles() as u64;
        self.prep += rb.prep_cmpt_cycles() as u64;
        self.exec += rb.exec_cmpt_cycles() as u64;
        self.tail += rb.dmao_cycles() as u64;
        self.cbs += 1;
    }

    /// An H2D CB: only its DMA-in pair is meaningful (words 3-4 read zero).
    fn add_dmai(&mut self, rb: flex_rs::control_block_wire::ResponseBlockWire) {
        self.dmai += rb.dmai_cycles() as u64;
        self.cbs += 1;
    }

    /// A D2H CB: only its DMA-out pair is meaningful (words 0-2 read zero).
    fn add_dmao(&mut self, rb: flex_rs::control_block_wire::ResponseBlockWire) {
        self.tail += rb.dmao_cycles() as u64;
        self.cbs += 1;
    }
}

/// One op's harvested cycles plus the label the launch loop knew it by.
#[derive(Default)]
struct RbOpCycles {
    cycles: RbCycles,
    label: String,
}

/// Accumulators the timing modes write into. Process-wide, like the C++ statics.
#[derive(Default)]
struct Timers {
    /// Per launch group, keyed by its LABEL (`bundle fp/group_i`, the launch mark's own name): (total
    /// ms, launches). ⛔ NOT by group index — groups are cut at a fixed descriptor count, so index `i`
    /// holds different ops in every chunk-width bundle (granite-8b's 36-page prompt bundle runs its
    /// cache write in `group_9` and its MLP in `group_10`; a narrower rung's MLP lands lower), and a
    /// per-index total summed across those bundles attributed the MLP's time to the cache write.
    group_ms: BTreeMap<String, (f64, u64)>,
    group_calls: u64,
    submit_ms: f64,
    prep_ms: f64,
    submit_calls: u64,
    /// Launches whose submission carried `pipeline_barrier` — a HOST fence call
    /// (`issue_barrier` waits the OTHER pipelines' fences; a no-op for all-compute runs), NOT a
    /// device barrier. Compute CBs are always built with the CTRL barrier bit clear
    /// (`CtrlSection::new(tag, false)`); what orders the device is the ring FIFO
    /// (STRICT_ORDERING) plus the always-set `cmpt.sync`. Pod-flex @ed96487 audit, 2026-10-17.
    host_fence_launches: u64,
    /// 🛑 KEYED BY (RUNG LENGTH, INDEX), NOT INDEX ALONE. A forward calls the launch loop once per
    /// rung — the body, but also the prefix and suffix lists — and each restarts its numbering at 0,
    /// so a map keyed on the index alone averages a 20-op body's group 2 with a 3-op prefix's.
    prefix_ms: BTreeMap<(usize, usize), (f64, u64, String)>,
    prefix_call: BTreeMap<usize, u64>,
    repeat_ms: BTreeMap<usize, (f64, f64, u64, String)>,
    /// Per-op harvested RB cycles — keyed like `prefix_ms` (same rung-numbering hazard).
    rb_cycles: BTreeMap<(usize, usize), RbOpCycles>,
    /// The DMA crossings' harvested cycles — the preamble-H2D and logits-D2H, which no op owns.
    /// Since #544 both ride the COMPUTE pipeline; `rb_attribute` routes them here by the
    /// direction `RbTiming::dma` carries, so these totals keep their pre-#544 meaning.
    rb_dmai: RbCycles,
    rb_dmao: RbCycles,
    rb_steps: u64,
    rb_cbs: u64,
    /// Cumulative `rb_cbs` at the last periodic report — the count jumps a whole forward at a
    /// time (~162 CBs), so a plain `is_multiple_of` would skip its own thresholds and never fire.
    rb_reported: u64,
    /// `SCRATCHY_SDSC_RB_TIME`'s launch sequence: one entry per compute launch this forward, in
    /// launch order — ((rung length, op index), label). Lives here, not on the executor, because
    /// the launch loop holds `self` immutably across the stream reference. Cleared at the start of
    /// every `predict` and zipped against the telemetry ring's id-sorted op entries (compute CBs
    /// minus the DMA crossings) after the step's synchronize; see [`rb_attribute`].
    rb_seq: Vec<((usize, usize), String)>,
    forwards: i64,
    // ── #603 PR2 2c — the replay player's own decomposition. Recorded
    // UNCONDITIONALLY (a handful of Instants and counters per STEP, not per
    // op) because every Diag knob forces the walk: these are the only
    // numbers that describe the player, reported on the cadence in
    // `report_replay` — they are also the player's liveness marker (a
    // benchmark whose stderr never shows [sdsc-replay] did not replay).
    /// Steps replayed from an image, and how many found their image already
    /// derived (the rest paid a derivation, counted below).
    replay_steps: u64,
    replay_hits: u64,
    /// Images derived (first use of a (body, page-count, fold-shape) key)
    /// and what those derivations cost — amortized over every step that
    /// reuses the image.
    replay_derives: u64,
    replay_derive_ms: f64,
    /// CBs replayed per step and moving-word patch sites carried by the
    /// derived images — the player's per-step payload.
    replay_cbs: u64,
    replay_sites: u64,
    /// The per-step host work: image lookup (key + fold-signature build +
    /// map hit) and the submit call (canvas memcpy + patches + tag stamps +
    /// ONE doorbell, all inside it).
    replay_lookup_ms: f64,
    replay_submit_ms: f64,
}

fn timers() -> std::sync::MutexGuard<'static, Timers> {
    static T: std::sync::OnceLock<std::sync::Mutex<Timers>> = std::sync::OnceLock::new();
    T.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Programs
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ONE LAUNCH GROUP: a contiguous run of up to `group_size()` trips that dxp compiled into a single
/// device program. A group size of `1` makes that one trip per program, which is the fault-isolation end
/// of the same range — not a different mode. See `GroupSize`: a compile-time constant, not an env read.
///
/// Each is launched against the SHARED resident segments, so there is no ModuleStitcher (which
/// orphaned the matmul in the fused bundle). A group that carries an address shift — a KV slot write,
/// a Kᵀ slab restickify, a page fold — is a singleton at ANY group size, because the shift is applied
/// once to the whole group.
pub(crate) struct OpProg {
    /// The launch: its address shifts and the compiled program they apply to.
    pub(crate) code: &'static bundle::LaunchGroup<'static>,
    /// The bundle this launch belongs to, and its position in that bundle's launch order — together,
    /// the whole of "which program is this".
    fp: &'static str,
    index: usize,
    /// Its resident Program-segment allocation. `None` until [`Executor::alloc_ops`] — the split
    /// ladder is allocated lazily, so an unallocated op is a real state, not an invariant break.
    pub(crate) addr: Option<DevAddr>,
    /// The KV facts `fold_plan` reads: whose KV, which pass shape, which strides.
    pub(crate) kv: OpKv,
}

/// ⛔⛔⛔ A SESSION-LIFETIME DEVICE HANDLE IS FORGOTTEN AT DROP ON PURPOSE. Per-step windows
/// (`shifted`, built per layer per launch) are NOT: they drop as owned values while the
/// runtime is alive.
///
/// The forget is kept by choice, and it costs nothing: the device region a session-lifetime
/// handle describes lives until process exit regardless (see the `DevAddr` doc), so a freeing
/// drop could reclaim only a descriptor value, once, at shutdown. Nothing crashes without it
/// today — the context in `runtime::runtime_cell` is a `static` that is never dropped, and
/// `DevAddr`'s drop cannot reach the allocator — but the law stays load-bearing anyway:
/// `runtime_reset` (the leak probe) tears the context down MID-PROCESS, and a future
/// `DevAddr` drop that learns to free must not find session handles betting on drop order
/// versus that.
///
/// The law predates this Rust surface: in the C++ runtime, the allocator was torn down before
/// these handles, and the throw out of a `noexcept` destructor turned perfect runs into
/// `RUN_RC=255` (measured 2026-08-12 — every run ended 255 **after** `Batch complete:
/// N succeeded, 0 failed`).
///
/// ⛔ DO NOT "SIMPLIFY" THIS BY FREEING THE SESSION-LIFETIME HANDLES AT DROP. The per-step
/// windows (`shifted`, built per layer per launch) must keep dying as owned values — at ~120
/// per forward a leaked descriptor each grows without bound.
impl Drop for OpProg {
    fn drop(&mut self) {
        std::mem::forget(self.addr.take());
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        for a in &mut self.seg_addr {
            std::mem::forget(a.take());
        }
    }
}

impl OpProg {
    /// Pair one launch-index entry with the program it names.
    ///
    /// `slot_write`/`slab_write` stay DERIVED from their strides ("> 0", as the C++ had them) so a
    /// stride and its flag cannot disagree.
    fn new(code: &'static bundle::LaunchGroup<'static>, fp: &'static str, index: usize) -> OpProg {
        let kv = &code.kv;
        OpProg {
            code,
            fp,
            index,
            addr: None,
            kv: OpKv {
                request: kv.request as u64,
                // The group's own part, off the launch index's single derivation — the typed value
                // the fusion walk compared, not a re-read number.
                part: kv.part,
                page_fold: kv.page_fold,
                slot_write: kv.slot_write(),
                page_slots: kv.page_slots as u64,
                // ⭐ THE PORT MINTS — the manifest's raw `kv_*_stride_bytes` become typed
                // strides exactly once, here where the launch index is read; the values
                // leave the types only through `PlaneSlot::shifted_by` /
                // `SlabIdx::shifted_by`, so a slot stride multiplying a slab index (the
                // mixed-stride group's wrong base shift) has no spelling past this point.
                slot_stride: SlotStride::of_bytes(Bytes(kv.slot_stride_bytes as u64)),
                slab_write: kv.slab_write(),
                slab_stride: SlabStride::of_bytes(Bytes(kv.slab_stride_bytes as u64)),
                batched_requests: kv.batched_requests,
                // ⭐ THE OTHER HALF OF `fold_plan::collapsed`, and what sends the fold's KV shift to
                // zero — a bake fact, carried on the group exactly like the axis itself.
                gathered: kv.gathered,
                // The staging channel's own mirror — the ruled custody chain's first link
                // (`OpKv → KvShifts → StepBody`), minted off the same group value as
                // `gathered` so the two bits cannot come from different derivations.
                region_form: kv.region_form,
            },
        }
    }

    /// How the timing modes name this program: its bundle's fingerprint and its launch order.
    pub(crate) fn label(&self) -> String {
        format!("{}/group_{}", self.fp, self.index)
    }
}

/// A launch sequence — a body rung, the prefix, the suffix, or a fold-fused twin.
pub(crate) type Ops = Vec<OpProg>;

/// The host-shadow slice covering the WHOLE of `addr` — the H2D source for a segment.
///
/// ⭐ RETURNING A SLICE IS THE POINT. [`Stream::h2d`] takes the transfer length FROM the slice, so
/// the DMA cannot read past the shadow the way an independent `size: u64` argument could — which is
/// exactly how `alloc_ops_at` came to pin past the end of a program binary. A shadow shorter than
/// its region is a STAGING bug (`prepare` grows every shadow it sends to the region's extent), so it
/// is reported here rather than becoming an out-of-bounds transfer.
fn whole_shadow<'a>(shadow: &'a [u8], addr: &DevAddr, seg: usize) -> Result<&'a [u8]> {
    let need = addr.total_size().as_u64() as usize;
    shadow.get(..need).ok_or_else(|| {
        anyhow!(
            "seg{seg}: the host shadow is {} B but its device region is {need} B — this H2D would \
             read past the shadow. `prepare` grows every shadow it sends to the region's extent; \
             this one was not grown, so the segment was never meant to be sent from here.",
            shadow.len()
        )
    })
}

/// ⭐ THE H2D WINDOW AN ACTIVATION AT `[offset, offset + len)` GOES OUT IN — `offset` down and the end
/// up to [`H2D_ALIGN`]. Slicing to a bind's EXACT range was tried and tripped a DMA
/// sub-address/alignment error; any alignment coarser than the hardware's is also legal, so this needs
/// no knowledge of the exact requirement. The one law both refill sources take — a `bound` activation
/// converted at the refill and one `bind_input_f32` converted at bind — so they go out alike.
fn h2d_run(offset: u64, len: u64) -> (u64, u64) {
    (
        (offset / H2D_ALIGN) * H2D_ALIGN,
        (offset + len).div_ceil(H2D_ALIGN) * H2D_ALIGN,
    )
}

/// The H2D window alignment [`h2d_run`] rounds to.
const H2D_ALIGN: u64 = 4096;

/// The smallest page a host maps. One write per this many bytes faults in every page of a buffer,
/// whatever page size the host actually uses.
const HOST_PAGE: usize = 4096;

/// ⭐ FAULT EVERY PAGE OF `buf` IN, LEAVING ITS BYTES AS THEY ARE: each page is written with the byte
/// it already holds. A zeroed allocation is mapped on first WRITE (a read maps the shared zero page
/// and commits nothing), so this is what moves those faults to the caller's moment. The value goes
/// through `black_box`, so the store cannot be proven redundant and dropped.
fn fault_in(buf: &mut [u8]) {
    for b in buf.iter_mut().step_by(HOST_PAGE) {
        *b = std::hint::black_box(*b);
    }
}

/// One keyed activation's state: the [`CausalKey`] its staged bytes are a function of, and whether
/// the current step re-bound it (so `refill_activations` converts and uploads it this step, and only
/// this step). See [`Executor::keyed`].
///
/// [`CausalKey`]: scratchy_subtile::sdsc_abstract::CausalKey
#[derive(Clone, Copy, Debug)]
struct Keyed {
    key: scratchy_subtile::sdsc_abstract::CausalKey,
    fresh: bool,
}

/// Whether `refill_activations` sends bound activation `id` this step: every unkeyed activation, every
/// step (today's re-upload, unchanged), and a keyed one only in the step that bound it. The one rule
/// both refill loops ask, so they cannot disagree about which binds go out.
fn refill_sends(keyed: &BTreeMap<bundle::PlaceId, Keyed>, id: &bundle::PlaceId) -> bool {
    keyed.get(id).is_none_or(|k| k.fresh)
}

/// ⛔⛔⛔ EVERY WAY A SOURCE TENSOR'S BYTES CAN GET THERE — THE COMPLETE LIST.
///
/// The device cannot report an unfilled source: it reads whatever the segment happens to hold, or it
/// waits. So `predict` refuses to launch when one is [`Self::Nobody`], and this enum is what makes
/// that verdict auditable instead of accidental. Being a closed set is the point — a new fill
/// mechanism cannot be introduced without adding a variant here, and adding a variant breaks every
/// exhaustive match until each has said what it means.
///
/// See [`Executor::filler_of`] for why this exists as a type rather than as a chain of `&& !`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceFiller {
    /// This bundle has no placement for the id, so it is not a source of THIS session.
    NotPlaced,
    /// The generated weight loader bound it before `prepare`, and `stage_weights` wrote it into a
    /// resident segment.
    WeightLoader,
    /// A forward-tape step bound it this step (`bind_input`) — activations, masks, block tables.
    TapeStep,
    /// Declared device-resident by the LOADER: a source BY ID that nothing binds per step, because
    /// its bytes were put there once and persist. Two populations, and they arrive differently — the
    /// paged KV cache, which the DEVICE wrote on a previous step, and the spilled weight tail, which
    /// the host copied into this session's own region at load (`adopt_from_owner`) because its slot
    /// cannot be aliased: the slot also carries an intermediate color, so no two bundles agree on that
    /// segment's extent and [`Self::SegmentOwner`] is unavailable to it.
    DeclaredResident,
    /// Its segment is ALIASED onto another session's region, so the OWNER filled it. The prefill
    /// ladder rungs and the batched-prefill session are entirely this case for seg1 (weights) and
    /// seg2 (KV): they bind NOTHING, precisely so the 2.6 GB stage+H2D happens once.
    /// `alias_seg_from` has already proven both bundles place the tensor at the same
    /// `(segment, offset, size)`, so the owner's write landed where this session's read will look.
    SegmentOwner,
    /// ⛔ NOBODY. Not a condition to carry on from — see [`Executor::require_sources_filled`].
    Nobody,
}

/// The re-rolled layer loop: ONE BODY PER ATTENTION CLASS, each re-launched for its own layers
/// with the weight and KV segment bases shifted per layer, between a prefix (embed) and a suffix
/// (lm_head). ⭐ THE LAUNCH TABLE IS DECLARED DATA — one row per layer, in true layer order, each
/// naming its body and its segment shifts. There is no per-layer arithmetic left here: no
/// `v·weight_stride`, no `v / layers_per_bank`, no `v·kv_stride`. A class-split model's classes
/// have DIFFERENT per-layer strides and different layer counts, which is exactly why the shifts
/// are a table the bake proved against the placements rather than formulae the runtime re-derives.
struct Rolled {
    /// How many attention classes there are (one body each). 1 for every single-class model.
    n_bodies: usize,
    /// ⭐ ONE ROW PER LAYER, IN TRUE LAYER ORDER — which body runs it and the segment shifts the
    /// launch applies. `(body, w_bank, w_off, kv_off)`: the launch binds bank `w_bank` into
    /// positional slot 1, shifts seg1's base by `w_off` and seg2's by `kv_off`.
    launches: Vec<(usize, u8, u64, u64)>,
    /// Bytes of ONE pool page — every layer of every class (`Σ_c n_c · stride_c`). Differs from
    /// any single class's stride exactly when the classes have different KV geometry.
    kv_page_stride: u64,
    /// The bank the PREFIX program's weight operands were baked in, and the SUFFIX's — the emitter
    /// proved each group's weights live in ONE bank, because a launch has one base per segment.
    prefix_weight_bank: usize,
    suffix_weight_bank: usize,
    // ⛔ NO HIDDEN-STREAM FIELDS. reroll_meta names `hidden_in`/`hidden_out`/`suffix_in` tids, but
    // the loop-carried residual and the body→suffix seam thread IN-PLACE via emitter placement
    // aliasing (hidden_out and suffix_in are aliased onto hidden_in's resident buffer), so the host
    // neither copies nor addresses them. They were carried here only for a tid-archaeology dump of
    // a bug that is fixed.
}

/// The paged-KV geometry the BUNDLE declares (an op carrying `kv_page_slots` was emitted against
/// the paged pool), so a paged bundle can never be driven by the unpaged address path.
#[derive(Clone, Copy, Default)]
struct KvGeometry {
    paged: bool,
    page_slots: PageSlots,
    /// Bytes per page = all layers of one page.
    page_stride_bytes: u64,
    pool_pages: PoolPages,
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  The session
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One resident SuperDSC session: program + weights + KV resident on-card.
pub struct Executor {
    /// THE BAKED BUNDLE — this session's program, memory plan and launch index, straight out of the
    /// binary.
    code: &'static bundle::BundleCode<'static>,

    pub vocab: Vocab,
    pub kv_dim: KvDim,
    /// The pool page count the caller asked for at create.
    num_blocks: PoolPages,

    // ── Parsed layout (bundle_layout.json) ──
    /// Placement per tensor IDENTITY.
    ///
    /// ⛔ WAS KEYED BY `t{id}` — a string the emitter and the host each built independently and
    /// joined on by convention. A miss was `continue` + an eprintln (a silently unstaged weight),
    /// so a drift in that convention could not surface as a build break. The key is now the id.
    places: BTreeMap<bundle::PlaceId, &'static Placement>,
    segment_bytes: [u64; NUM_SEGMENTS],
    /// Matmul KERNEL weights → their device-tile descriptor. The weight is staged into the device
    /// TILE layout (not row-major flat) by [`stage_weight_tiled`].
    kernel_weights: BTreeMap<bundle::PlaceId, &'static KernelWeight<'static>>,
    /// The role-`Logits` tensor — read back each step.
    logits_id: bundle::PlaceId,
    /// Ids `0..num_sources` are CALLER-FILLED every forward, EXCEPT the device-resident ones in
    /// [`Self::resident_sources`]. Set by [`Executor::declare_sources`]; `None` means nothing has
    /// told this executor how many, and the pre-launch completeness check cannot run.
    num_sources: Option<u32>,
    /// ⛔ SOURCES THE HOST DOES NOT WRITE PER STEP. The prefix-KV cache lives in seg2 and is
    /// filled by the pool and by the device's own cachewr — it is a caller-filled source by id,
    /// but no `bind_input` ever names it, so a completeness check that does not know this refuses
    /// EVERY launch.
    resident_sources: std::collections::BTreeSet<u32>,

    // ── Programs ──
    rolled: Option<Rolled>,
    prefix_ops: Ops,
    /// ONE LIST PER ATTENTION CLASS — `body_ops[c]` is class `c`'s body. A single-class model
    /// (every model but the hybrid-attention ones) has exactly one.
    body_ops: Vec<Ops>,
    suffix_ops: Ops,
    /// Each class's body with the per-page fold fused back in — one fewer launch per layer. Used
    /// whenever the context fits ONE page, which is the case that must cost what unpaged cost.
    /// `None` unless EVERY class has its twin ([`PerClass`]).
    body_fused: Option<PerClass<Ops>>,
    /// sk_bucket LADDER (decode): `(active_cap, one ops list per class)` ascending; the top rung
    /// is the full cap.
    body_rungs: Vec<(SweptCols, Vec<Ops>)>,
    /// The rungs whose EVERY class has a fold-fused twin, ascending, each with its twins in class
    /// order ([`fused_rungs`]). A rung with any class sitting out is absent, so it runs split.
    body_rungs_fused: Vec<(SweptCols, PerClass<Ops>)>,
    /// PAGED: the SPLIT bodies are only reachable once a context outgrows one page, so their
    /// Program memory is allocated on first selection.
    split_bodies_ready: bool,

    // ── Device state ──
    // ⛔ The `DevAddr`s below are leaked at shutdown by `impl Drop for Executor` — see the note on
    // `impl Drop for OpProg` for the abort that forced it.
    stream: Option<Stream>,
    seg_addr: [Option<DevAddr>; NUM_SEGMENTS],
    /// ⭐ THE WEIGHT SEGMENT'S EXTRA BANKS — regions, host shadows and extents for banks `1..N`,
    /// index `b - 1`. Bank 0 IS `seg_addr[SEG_WEIGHT]` / `seg_host[SEG_WEIGHT]`, so these three are
    /// EMPTY for every bundle whose weights fit one device region and every path below then behaves
    /// exactly as it did before banking existed.
    ///
    /// A bank is not another segment (there is no free slot, and the SDSC addresses all of them as
    /// segment 1); it is a different REGION bound into the same positional slot by the launch that
    /// needs it. See [`bundle::Placement::bank`] and [`Executor::slot_of`].
    weight_bank_addr: Vec<Option<DevAddr>>,
    weight_bank_host: Vec<Vec<u8>>,
    weight_bank_bytes: Vec<u64>,
    /// Per extra bank: its region is BORROWED from an owner session, so never H2D this session's
    /// shadow over it. Mirrors [`Self::seg_aliased`], which cannot cover banks because it is indexed
    /// by segment.
    weight_bank_aliased: Vec<bool>,
    /// Host staging (sen bytes) per segment.
    seg_host: [Vec<u8>; NUM_SEGMENTS],
    /// This segment's device region is BORROWED from another session, so `seg_host` is NOT its
    /// authoritative shadow and must never be H2D'd over it.
    seg_aliased: [bool; NUM_SEGMENTS],
    /// Declared BEFORE prepare: this session's own copy is dead on arrival (an alias will repoint
    /// it), so prepare gives it a 128 B placeholder and skips its staging and its H2D.
    seg_borrowed: [bool; NUM_SEGMENTS],

    // ── Host binds ──
    /// Bound host inputs. Ids bound BEFORE prepare are resident WEIGHTS (uploaded once); ids
    /// bound per predict are ACTIVATIONS (uploaded each step).
    bound: BTreeMap<bundle::PlaceId, Vec<u8>>,
    weight_ids: BTreeSet<bundle::PlaceId>,
    /// ⭐⭐⭐⭐⭐ IDS WHOSE BYTES ARE **NOT** IEEE-fp16 — copied into the shadow verbatim instead of
    /// going through [`sen_convert::ieee_to_sen_bytes`].
    ///
    /// ⛔ WITHOUT THIS, A GATHER'S INDEX TABLE IS DESTROYED BY THE BIND LOOP. Every staging path in
    /// this file ends in `ieee_to_sen_bytes(src, dst)` — unconditionally, for weights at prepare and
    /// for activations at every predict — because until now every bound tensor genuinely WAS a
    /// magnitude the card holds in SEN169 fp16. The KV block index is int32 addresses-in-waiting
    /// (`Staged::I32`): running them through an fp16 re-encoding reads each 4-byte entry as two fp16
    /// values and rewrites both mantissas. Block 37 does not become "roughly 37"; it becomes a
    /// different, valid, arbitrary block — fluent output from another request's keys, with no fault.
    ///
    /// ⛔ NOT A FLAG ON THE BIND CALL, A SET KEYED BY ID, because there are TWO convert loops
    /// (`stage_weights`, `predict`) and a per-call flag would have to be remembered by both anyway.
    /// One set, consulted by both, is the only arrangement in which they cannot disagree.
    raw_ids: BTreeSet<bundle::PlaceId>,
    /// ⭐ ACTIVATIONS STAGED BY KEY — the key their bytes are a function of, and whether this
    /// step re-bound them. Today exactly one: the new-block causal mask
    /// ([`scratchy_subtile::sdsc_abstract::CausalKey`]).
    ///
    /// A keyed bind is written to the shadow and uploaded in the step that binds it, and NOT
    /// re-uploaded by a step that does not: segment 3, where every activation lives, is written by
    /// no device op (measured over every compute op of the 252 granite-2b bundles), so the device
    /// keeps the bytes of the last bind. Every UNKEYED activation is re-uploaded every step exactly
    /// as before. That arrangement is deliberate: an input someone forgets to re-bind must still be
    /// caught by today's path, not hidden by this one.
    keyed: BTreeMap<bundle::PlaceId, Keyed>,
    /// ⭐ ACTIVATIONS ALREADY IN THE SHADOW — the byte length of each f32 activation that
    /// [`Self::bind_input_f32`] converted straight into its placement's slot of `seg_host`. The
    /// refill sends that range exactly as it sends a [`Self::bound`] one, without converting it a
    /// second time: the bytes are already the device's.
    ///
    /// Disjoint from `bound` by construction: every bind door removes its id from the other map, so
    /// one id has one source of bytes.
    staged: BTreeMap<bundle::PlaceId, u64>,
    /// ⭐⭐⭐⭐⭐ IDS BOUND THROUGH [`Executor::bind_const_fp32`] — fp32 constants whose LE host
    /// bytes ARE the device encoding (`Df::Fp32` is IEEE_FP32), so staging copies them VERBATIM:
    /// no IEEE→SEN re-encode, no tile walk. The composite argmax's descending ramp and scalar-V
    /// stick (#603 PR3a), and nothing else today.
    ///
    /// A set for the same reason [`Self::raw_ids`] is one: the staging loop consults it, and a
    /// flag remembered by the bind call alone is a flag the loop can forget.
    const_fp32_ids: BTreeSet<bundle::PlaceId>,

    // ── KV / fold ──
    kv: KvGeometry,
    /// The page maps, write cursors and declared strides — [`fold_plan`]'s state, owned here for
    /// the session's lifetime and read by every launch.
    fold: SessionKv,
    /// Has a caller installed a REAL page map (vs the create-time identity default)?
    block_table_set: bool,
    /// LEG-1(c) under `segaddr`: the last `[cachewr]` line this session printed, as its dedupe
    /// key (write slot + distinct target pages). The rolled body launches once per LAYER, so an
    /// undeduped line would print 40× per forward; the page targeting is a per-STEP fact.
    cachewr_seen: Option<(i64, Vec<i64>)>,

    prepared: bool,
    staged_host: bool,
    n_weights_staged: usize,
    /// ⭐ TENSORDUMP STATE (`SCRATCHY_SDSC_TENSORDUMP`): this session's last-seen bytes per placed
    /// tensor, its program index (per-fingerprint, first-dumped order — see [`Diag::tensordump`]),
    /// and its own op counter. `None` unless the env gate is set.
    dump: Option<TensorDump>,

    // ── #603 PR2 2c — the replay player ──
    /// Cached step images, keyed by (body, resident page count, fold shape) —
    /// [`StepImageKey`]. EMPTY until the first eligible decode step derives
    /// one; every batched or instrumented step takes the walk instead, so
    /// nothing here is consulted (or populated) off the player's own path.
    step_images: StepImageCache,
}

/// Per-session state for the card-side tensordump — see [`Diag::tensordump`].
struct TensorDump {
    /// This fingerprint's program index, assigned from the process-global first-dumped registry.
    prog: usize,
    /// This session's op counter (one per launch group, prefix/body/suffix alike).
    op: usize,
    /// Last-seen device bytes per placement (sen format, exactly what D2H brought back).
    snap: std::collections::HashMap<bundle::PlaceId, Vec<u8>>,
    /// Last-seen bytes per POOL WINDOW (see the LEG-2 watch in [`Executor::tensordump`]), in the
    /// fixed order the watch builds them: pg1, pg2. `None` until first read.
    wins: Vec<Option<Vec<u8>>>,
}

/// The process-global fingerprint → program-index registry for the tensordump, in first-dumped
/// order. On the card the sessions are built prefill-first-then-decode, which is the emulator's
/// program order (`new_multi`: prefill first, then the decode cap buckets), so the indices line up
/// with an `SCRATCHY_EMU_TENSORDUMP` of the same run for the two sessions that actually run — the
/// ladder rungs that never launch never claim an index.
fn dump_prog_index(fp: &str) -> usize {
    static REG: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, usize>>> =
        std::sync::OnceLock::new();
    let reg = REG.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut reg = reg.lock().unwrap_or_else(|e| e.into_inner());
    let n = reg.len();
    *reg.entry(fp.to_string()).or_insert(n)
}

// Driven single-threaded by the worker thread (the session moves between forwards, as the C++
// session pointer did).
unsafe impl Send for Executor {}

/// ⭐⭐⭐⭐⭐ A RAW (device-format) PAYLOAD **AND THE PLACEMENT IT FITS**, as one value — the only thing
/// [`Executor::bind_input_raw`] accepts, and the reason a raw bind can no longer overrun its tensor.
///
/// 🛑 **IT WAS TWO ARGUMENTS SIDE BY SIDE.** `bind_input_raw(id, bytes)` took a byte vector with no
/// relation to the tensor it would be written into; the relation was re-established far away and at
/// RUNTIME, by `refill_activations`'s over-bind `bail!` and by its `let Some(p) = places.get(nm) else
/// { continue }`. Those two arms are different failures wearing one shape:
///
/// * **Too long.** The shadow write is `seg_host[seg][offset..][..src.len()]`, so a payload past its
///   own placement overwrites the tensors the layout put after it. For the gather's index table the
///   tensor after it is read as BLOCK NUMBERS, and a block number is a valid address — a clean bake,
///   no fault, another page's keys.
/// * **Not placed at all.** `continue` — the bind was SKIPPED, in silence. An index table nobody
///   staged reads as entry 0 on every row, which is page 0's first block: a real address, so the whole
///   batch answers fluently from row 0's history. `assemble_attn` states this exact hazard ("a name no
///   placement matches is a bind the launcher skips in silence") and had nothing to enforce it with.
///
/// ⛔ THERE IS NO PUBLIC CONSTRUCTOR AND NO PUBLIC FIELD. [`Executor::place_raw`] is the only way to
/// make one, it consults the session's own placement table, and it returns `None` for either case
/// above — so neither is expressible at the bind, which is what makes the runtime checks redundant
/// rather than merely duplicated. The placement is CONSUMED by the constructor rather than carried:
/// a value that exists has already been measured against it, and keeping a second copy here would be
/// the same "one quantity, two declarations" the type removes.
pub struct RawBind {
    id: bundle::PlaceId,
    bytes: Vec<u8>,
}

/// The timers and H2D accounting of a step's FORWARD HALF — everything upstream of the
/// readback that [`Executor::predict`] (logits D2H) shares, held in one struct so the
/// step's tails print from the
/// SAME numbers and cannot drift on anything the step did before its readback.
struct ForwardHalf {
    t_stream0: Instant,
    t0: Instant,
    t0b: Instant,
    t1: Instant,
    t2: Instant,
    h2d_note: String,
    h2d_bytes: u64,
}

impl Executor {
    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Create
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Open a session on the baked bundle `code`: index its memory plan, bind every launch group to its
    /// compiled program, and latch the paged geometry the bundle declares. Allocates nothing on-device —
    /// that is [`Executor::prepare`].
    ///
    /// Nothing is read from disk: `code` is `&'static` data in this binary.
    pub fn load(
        code: &'static bundle::BundleCode<'static>,
        vocab: Vocab,
        kv_dim: KvDim,
        num_blocks: PoolPages,
    ) -> Result<Executor> {
        let places: BTreeMap<bundle::PlaceId, &'static Placement> =
            code.layout.places.iter().map(|p| (p.id, p)).collect();
        // ⛔⛔⛔ THE MAP MUST NOT HAVE LOST A PLACEMENT. This was keyed by `p.name` (a String) and
        // is now keyed by `p.id`; a `BTreeMap` silently keeps the LAST entry for a duplicate key,
        // so two placements sharing an id would leave one tensor with NO placement at all. The
        // host then never stages it and the device reads whatever that HBM held — which does not
        // fault, it hangs or answers with confident garbage.
        if places.len() != code.layout.places.len() {
            let mut seen = std::collections::BTreeSet::new();
            let dupes: Vec<String> = code
                .layout
                .places
                .iter()
                .filter(|p| !seen.insert(p.id))
                .map(|p| p.id.to_string())
                .collect();
            bail!(
                "bundle {}: {} placements collapse to {} distinct ids — {:?} appear more than \
                 once. Two tensors share an identity, so one of them has no placement.",
                code.fp,
                code.layout.places.len(),
                places.len(),
                dupes
            );
        }
        let logits_id = code
            .layout
            .logits()
            .ok_or_else(|| anyhow!("bundle {} places no Logits tensor", code.fp))?
            .id;
        let kernel_weights = code
            .layout
            .kernel_weights
            .iter()
            .map(|k| (k.id, k))
            .collect();

        let mut ex = Executor {
            code,
            vocab,
            kv_dim,
            num_blocks,
            places,
            segment_bytes: code.layout.segment_bytes,
            kernel_weights,
            logits_id,
            num_sources: None,
            resident_sources: std::collections::BTreeSet::new(),
            rolled: None,
            prefix_ops: Vec::new(),
            body_ops: Vec::new(),
            suffix_ops: Vec::new(),
            body_fused: None,
            body_rungs: Vec::new(),
            body_rungs_fused: Vec::new(),
            split_bodies_ready: false,
            stream: None,
            seg_addr: [const { None }; NUM_SEGMENTS],
            // ⭐ THE WEIGHT BANKS, STRAIGHT FROM THE BAKED LAYOUT — the bundle decided how many
            // regions its weights need at cargo-build; nothing here chooses. Empty (one bank) for
            // every model whose weights fit one region.
            weight_bank_addr: code.layout.weight_bank_bytes.iter().map(|_| None).collect(),
            weight_bank_host: code
                .layout
                .weight_bank_bytes
                .iter()
                .map(|_| Vec::new())
                .collect(),
            weight_bank_bytes: code.layout.weight_bank_bytes.to_vec(),
            weight_bank_aliased: vec![false; code.layout.weight_bank_bytes.len()],
            seg_host: [const { Vec::new() }; NUM_SEGMENTS],
            seg_aliased: [false; NUM_SEGMENTS],
            seg_borrowed: [false; NUM_SEGMENTS],
            bound: BTreeMap::new(),
            weight_ids: BTreeSet::new(),
            raw_ids: BTreeSet::new(),
            keyed: BTreeMap::new(),
            staged: BTreeMap::new(),
            const_fp32_ids: BTreeSet::new(),
            kv: KvGeometry::default(),
            fold: SessionKv::default(),
            block_table_set: false,
            cachewr_seen: None,
            prepared: false,
            staged_host: false,
            n_weights_staged: 0,
            dump: None,
            step_images: StepImageCache::default(),
        };

        // ── RE-ROLLED mode: a re-roll meta ⇒ this bundle is the BODY (one layer); the prefix (embed)
        //    and suffix (lm_head) are SIBLING bundles it names by fingerprint. The layout above is the
        //    FULL plan (every layer's weights/KV resident), shared by all three. ──
        if let Some(m) = &code.reroll {
            ex.load_rolled(m)?;
        }
        // The BODY's own launch groups (or the whole model's, unrolled).
        ex.load_bodies()?;
        // AFTER every program is bound: let the BUNDLE declare whether the KV is paged, and size the
        // pool from the create-time page count.
        ex.latch_paged_geometry();

        debug!(
            "[sdsc-superdsc] load OK — bundle={}, {} placements, seg_bytes={:?}, logits={}, \
             vocab={} kv_dim={} num_blocks={}",
            code.fp,
            ex.places.len(),
            ex.segment_bytes,
            ex.logits_id,
            ex.vocab.0,
            ex.kv_dim.0,
            ex.num_blocks.0,
        );
        Ok(ex)
    }

    /// Bind one sibling bundle's launch groups.
    ///
    /// ⛔ A SIBLING IS A FINGERPRINT THE REGISTRY RESOLVES. A rung this binary does not carry is a hard
    /// load error naming it and listing what IS carried — the alternative, treating it as absent, runs
    /// the batch on a bundle the ladder said was there.
    fn sibling_ops(what: &str, fp: &bundle::SiblingFp<'static>) -> Result<Ops> {
        let code = bundle::sibling(fp).ok_or_else(|| {
            anyhow!(
                "{what} {} is not compiled into this binary — the emit either produced no such \
                 bundle or ran for a different model. Registered: {:?}",
                fp.as_str(),
                bundle::registered_fps(),
            )
        })?;
        Ok(bind_ops(code))
    }

    fn load_rolled(&mut self, m: &'static bundle::RerollMeta<'static>) -> Result<()> {
        // A bank index the session has no region for would silently fall back to another bank's
        // base, which is a whole program reading the wrong weights. Check every launch's bank and
        // both seam groups at load.
        let banks = self.weight_bank_bytes.len() + 1;
        for (what, b) in [
            ("prefix", m.prefix_weight_bank),
            ("suffix", m.suffix_weight_bank),
        ] {
            if b as usize >= banks {
                bail!(
                    "load: reroll meta puts the {what}'s weights in bank {b}, but this bundle has \
                     only {banks} weight bank(s)"
                );
            }
        }
        // ⛔ EVERY LAUNCH'S BANK AND BODY, CHECKED AT LOAD. A launch naming a body the meta does
        // not carry, or a bank the bundle does not hold, is a bake/runtime disagreement that would
        // otherwise surface as a wrong-base launch — fluent garbage or an unaddressable region.
        // The launches' bodies also name the CLASS COUNT, which the rungs must agree with.
        for l in m.launches.iter() {
            if l.body as usize >= m.bodies.len() {
                bail!(
                    "load: reroll meta's launch table names body {} but the meta carries {} \
                     body(ies)",
                    l.body,
                    m.bodies.len()
                );
            }
            if l.w_bank as usize >= banks {
                bail!(
                    "load: reroll meta's launch table puts layer (body {}, iter {}) in bank {}, \
                     but this bundle has only {banks} weight bank(s)",
                    l.body,
                    l.iter,
                    l.w_bank
                );
            }
        }
        if m.launches.is_empty() {
            bail!(
                "load: reroll meta declares an EMPTY launch table — the layer sequence is the one \
                 thing the executor cannot re-derive"
            );
        }
        self.rolled = Some(Rolled {
            n_bodies: m.bodies.len(),
            launches: m
                .launches
                .iter()
                .map(|l| (l.body as usize, l.w_bank, l.w_off, l.kv_off))
                .collect(),
            kv_page_stride: m.kv_page_stride,
            prefix_weight_bank: m.prefix_weight_bank as usize,
            suffix_weight_bank: m.suffix_weight_bank as usize,
        });
        self.prefix_ops = Self::sibling_ops("rolled prefix", &m.prefix)?;
        self.suffix_ops = Self::sibling_ops("rolled suffix", &m.suffix)?;
        debug!(
            "[sdsc-superdsc] load: RE-ROLLED — {} body class(es) + prefix({}) + suffix({}); {} \
             launch(es), page_stride={} B",
            m.bodies.len(),
            m.prefix.as_str(),
            m.suffix.as_str(),
            m.launches.len(),
            m.kv_page_stride
        );
        // The launch table verbatim — body/iter/w_bank/w_off/kv_off per layer, in true order. This
        // is the arithmetic the card run's launch loop applies, so a card fault at a known device
        // address is traced to its producing launch by reading this against the region bases the
        // prepare-time seg-alloc debug prints.
        debug!("[sdsc-superdsc] launch table: {:?}", m.launches);
        Ok(())
    }

    /// Bind the body's launch groups: either the sk_bucket LADDER (one body per attention sweep
    /// extent, ONE LIST PER CLASS per rung) or the single bodies (one per class), each with its
    /// fold-fused twin where one was baked.
    ///
    /// ⭐ THE ANCHOR RULE: `bodies[0]` is THIS bundle, and every other class is a sibling the meta
    /// names — the same rule the prefix/suffix follow. A single-class model's `bodies` is exactly
    /// `[self.code]`.
    fn load_bodies(&mut self) -> Result<()> {
        let rungs: &[bundle::LadderRung<'static>] = match &self.code.reroll {
            Some(m) if m.rungs.len() > 1 => &m.rungs,
            _ => &[],
        };
        // ONE LIST PER CLASS, this bundle as class 0's anchor and every other class a sibling.
        let class_ops = |m: &'static bundle::RerollMeta<'static>| -> Result<Vec<Ops>> {
            let mut v = Vec::with_capacity(m.bodies.len());
            for (ci, fp) in m.bodies.iter().enumerate() {
                v.push(if ci == 0 {
                    bind_ops(self.code)
                } else {
                    Self::sibling_ops("rolled body class", fp)?
                });
            }
            Ok(v)
        };
        if rungs.is_empty() {
            if let Some(m) = &self.code.reroll {
                self.body_ops = class_ops(m)?;
                self.body_fused = PerClass::of_every_class(
                    self.body_ops.len(),
                    m.bodies_fused
                        .iter()
                        .map(|fp| bundle::sibling(fp).map(bind_ops)),
                );
            } else {
                self.body_ops = vec![bind_ops(self.code)];
            }
            // ⛔⛔⛔ THE PROGRAM COUNT IS NOT A DEBUG DETAIL — IT IS WHETHER THIS MODEL COMPUTES.
            //
            // A bundle carries a memory PLAN and a set of device PROGRAMS. The plan is what every session
            // line reports ("session ready", placements, `decode caps`), so a bundle whose programs are
            // MISSING loads clean and serves nothing: no launch, no error, a logits buffer that stays
            // zero, a constant argmax, and `"text": ""` with `finish_reason: "length"`. It took a whole
            // session to find because the only symptoms were an absurd 0.6 ms/token and a `cap` that came
            // out as `num_blocks * 64` — the unpaged formula, because `latch_paged_geometry` scans THESE
            // OP LISTS for `kv_page_slots` and finds nothing in an empty one.
            //
            // So the counts go at INFO, always, next to the fingerprint: one line that says whether the
            // thing that was loaded can run. And zero says so in its own words rather than by arithmetic.
            let n_body: usize = self.body_ops.iter().map(|o| o.len()).sum();
            if n_body == 0 && self.prefix_ops.is_empty() && self.suffix_ops.is_empty() {
                bail!(
                    "bundle {} has ZERO device programs (0 body / 0 prefix / 0 suffix launch groups). \
                     Its memory plan is present, which is why every session line would otherwise report \
                     itself ready — but NOTHING CAN BE LAUNCHED: forwards return in microseconds, the \
                     logits buffer stays zero, argmax lands on a constant token id, and every completion \
                     comes back EMPTY (`finish_reason: \"length\"`, ~0.6 ms/token) with no error anywhere. \
                     REFUSING TO LOAD: serving this is strictly worse than not starting. The cause is a \
                     BAKE that emitted placements without programs — set `DEEPTOOLS_PATH` (and the \
                     deeptools `LD_LIBRARY_PATH`) in the stage that runs `cargo build`, not only at \
                     runtime.",
                    self.code.fp,
                );
            } else {
                info!(
                    "[sdsc-superdsc] bundle {}: {} body ({} class(es)) + {} prefix + {} suffix \
                     launch group(s)",
                    self.code.fp,
                    n_body,
                    self.body_ops.len(),
                    self.prefix_ops.len(),
                    self.suffix_ops.len()
                );
            }
            return Ok(());
        }
        // The ladder came ASCENDING by `active_cap` from the emit, so it stays so. Each rung's
        // class 0 body IS this bundle's rung sibling — the registry resolves like any other.
        for r in rungs {
            self.body_rungs.push((
                r.active_cap,
                r.bodies
                    .iter()
                    .map(|fp| Self::sibling_ops("decode rung", fp))
                    .collect::<Result<Vec<_>>>()?,
            ));
        }
        // A rung's fold-fused twins are optional: a rung where any class sits out runs its split
        // bodies — correct, one launch per layer dearer. See [`fused_rungs`] for why it is per rung.
        self.body_rungs_fused = fused_rungs(rungs.iter().map(|r| {
            (
                r.active_cap,
                r.bodies.len(),
                r.bodies_fused
                    .iter()
                    .map(|fp| bundle::sibling(fp).map(bind_ops))
                    .collect::<Vec<_>>(),
            )
        }));
        debug!(
            "[sdsc-superdsc] PAGED KV: {} of {} rungs have a fold-fused twin (a context inside one \
             page runs it — the split body costs two extra launches per layer, ~1 ms/token)",
            self.body_rungs_fused.len(),
            self.body_rungs.len()
        );
        let caps: Vec<u32> = self.body_rungs.iter().map(|(c, _)| c.get()).collect();
        debug!(
            "[sdsc-superdsc] load: LADDER — {} decode rungs (active_cap {caps:?}), {} prefix + {} \
             suffix kernel(s)",
            self.body_rungs.len(),
            self.prefix_ops.len(),
            self.suffix_ops.len()
        );
        Ok(())
    }

    /// Let the BUNDLE declare whether the KV is paged — any op carrying `kv_page_slots` was emitted
    /// against the paged pool. Read from the OPS rather than a flag so a paged bundle can never be
    /// driven by the unpaged path (which would apply an absolute-position write shift and land past
    /// the page). seg2 is then sized `pool_pages * page_stride` at RUNTIME rather than from the
    /// manifest, which is what makes the servable context an allocation instead of a baked constant.
    fn latch_paged_geometry(&mut self) {
        let mut page_slots = 0i64;
        // A rung holds ONE `Ops` PER ATTENTION CLASS (the roll split by class), so the ladder's
        // contribution to the sweep is the rungs' class vectors FLATTENED — the same `&Ops` items
        // `lists()` yields.
        for ops in self
            .lists()
            .into_iter()
            .chain(self.body_rungs.iter().flat_map(|(_, o)| o))
        {
            for o in ops {
                if o.kv.page_slots > 0 {
                    page_slots = o.kv.page_slots as i64;
                }
            }
        }
        if page_slots == 0 {
            return;
        }
        // ⭐ THE PAGE STRIDE IS BAKED, NOT DERIVED. It used to be `iters × kv_stride`, which is
        // only right while every layer has the same KV geometry — a hybrid model's classes do not,
        // and the emitter's `kv_page_stride` already sums every class's extent (`Σ n_c·stride_c`),
        // proven against the placements. 0 on an UNROLLED bundle (no reroll meta) is the same value
        // the old `iters × kv_stride` formula produced there (0 × 0), and the geometry latch below
        // runs identically — an unrolled paged bundle keeps its pre-split behavior whole.
        let page_stride_bytes = self.rolled.as_ref().map_or(0, |r| r.kv_page_stride);
        self.kv = KvGeometry {
            paged: true,
            page_slots: PageSlots(page_slots),
            page_stride_bytes,
            pool_pages: PoolPages(if self.num_blocks.0 > 0 {
                self.num_blocks.0
            } else {
                1
            }),
        };
        self.fold.paged = true;
        self.fold.page_slots = fold_plan::PageSlots::per_page(page_slots);
        self.fold.page_stride_bytes = Bytes(self.kv.page_stride_bytes);
        // CREATE-TIME IDENTITY MAP. A session forwarded before any `set_block_table` addresses
        // logical page n as physical page n, which is what the unpaged path did.
        let ident: Vec<i64> = (0..self.kv.pool_pages.0).collect();
        self.install_page_map(RowIdx::from_launch_row(0), SeqPos(0), &ident);
        self.block_table_set = false;
        // LEG-1(b) under `segaddr`: this session's POOL placements — every seg2 offset the bundle
        // baked, one line, one write (the tensordump only watches the ones its one-page shadow
        // covers, so which page the rest sit on is exactly what this line settles). Once per
        // session by construction: the identity install above is the create-time path.
        if Diag::get().segaddr {
            use std::fmt::Write as _;
            let mut ps = String::new();
            let mut n = 0u32;
            for (id, p) in &self.places {
                if p.segment as usize == SEG_KV.get() && p.bank == 0 && p.size > 0 {
                    let _ = write!(ps, " {id}@{}+{}", p.offset, p.size);
                    n += 1;
                }
            }
            eprintln!("[poolplaces] fp={} n={n} placements:{ps}", self.code.fp);
        }
        debug!(
            "[sdsc-superdsc] PAGED KV: page={} slots, page_stride={} B, pool={} pages ({:.1} MB) \
             => {} positions of context available to one request",
            self.kv.page_slots.0,
            self.kv.page_stride_bytes,
            self.kv.pool_pages.0,
            (self.kv.pool_pages.0 as u64 * self.kv.page_stride_bytes) as f64 / 1048576.0,
            self.kv.pool_pages.0 * self.kv.page_slots.0,
        );
    }

    fn lists(&self) -> Vec<&Ops> {
        vec![&self.prefix_ops]
            .into_iter()
            .chain(self.body_ops.iter())
            .chain(std::iter::once(&self.suffix_ops))
            .collect()
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Binding
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// How many leading tensor ids are sources, and which of them the host does NOT write each
    /// step (the prefix-KV cache). Both come from the generated wiring. Without them
    /// [`Executor::require_sources_filled`] cannot run and `predict` says so.
    pub fn declare_sources(&mut self, n: u32, resident: impl IntoIterator<Item = u32>) {
        self.num_sources = Some(n);
        self.resident_sources = resident.into_iter().collect();
    }

    /// ⛔⛔⛔ REFUSE TO LAUNCH WITH A SOURCE NOBODY FILLED.
    ///
    /// Ids `0..num_sources` are written by the host every forward — by the generated weight loader
    /// before prepare, or by a forward-tape step each step. One that NEITHER wrote is not an error
    /// the device reports: it reads whatever the segment holds, or it waits. The failure surfaces
    /// as a HANG, or as fluent-but-wrong output, with nothing in any log.
    ///
    /// This is the runtime half of the bake-time completeness lock in the `#[forward]` macro. The
    /// bake proves the WIRING names a filler for every source; this proves the filler actually
    /// ran. Both exist because the gap between them is invisible: `superdsc_bake` refuses a plan
    /// with no device programs for the same reason — a binary that links, serves, and does
    /// nothing is worse than one that fails.
    fn require_sources_filled(&self) -> Result<()> {
        let Some(n) = self.num_sources else {
            bail!(
                "predict: no source count was declared, so the executor cannot check that every \
                 caller-filled tensor was written. Call `declare_sources` from the generated \
                 wiring at load — refusing to launch unchecked."
            );
        };
        let missing: Vec<bundle::PlaceId> = (0..n)
            .map(bundle::PlaceId::Act)
            // ⛔ SPELLED OUT, NOT `== Nobody`. A sixth fill mechanism added to `SourceFiller` fails
            // to compile HERE until someone says whether it counts as filled — which is the only
            // reason the classification is a type instead of a boolean.
            .filter(|id| match self.filler_of(id) {
                SourceFiller::Nobody => true,
                SourceFiller::NotPlaced
                | SourceFiller::WeightLoader
                | SourceFiller::TapeStep
                | SourceFiller::DeclaredResident
                | SourceFiller::SegmentOwner => false,
            })
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        bail!(
            "predict: {} caller-filled source(s) were never written this step — {}. The device \
             does not report this: it reads memory the host never filled, which surfaces as a \
             HANG or as confident garbage with no error anywhere. Refusing to launch.",
            missing.len(),
            missing
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    /// ⛔⛔⛔ WHO FILLED THIS SOURCE — ONE ANSWER, FROM ONE PLACE.
    ///
    /// This used to be an open-coded chain of negations inside `require_sources_filled`:
    /// `places.contains && !weight_ids.contains && !bound.contains && !resident_sources.contains`.
    /// The bug that shape produced: adding a FIFTH way to fill a source needs a fifth `&& !`, and
    /// forgetting it does not fail to compile — it silently reads as "nobody filled this", so the
    /// executor refuses a launch that was perfectly well fed. That is exactly what happened to the
    /// prefill ladder, which fills its weights by ALIAS and was refused for 642 of them.
    ///
    /// A total function into [`SourceFiller`] fixes the shape, not just the instance: the mechanisms
    /// are enumerated once, `Nobody` is one named case among them rather than the absence of all the
    /// others, and every consumer matches EXHAUSTIVELY — so a new variant is a compile error at
    /// every site that has to account for it.
    fn filler_of(&self, id: &bundle::PlaceId) -> SourceFiller {
        // Not in this bundle at all ⇒ not this session's source and not its problem.
        if !self.places.contains_key(id) {
            return SourceFiller::NotPlaced;
        }
        // ⭐ ORDER IS DELIBERATE: an id can be reachable by more than one mechanism (a weight is in
        // both `weight_ids` and `bound`), and the FIRST match is the one that actually wrote the
        // bytes. Any of them ⇒ filled, so the order changes only the label, never the verdict.
        if self.weight_ids.contains(id) {
            SourceFiller::WeightLoader
        } else if self.bound.contains_key(id) || self.staged.contains_key(id) {
            SourceFiller::TapeStep
        } else if self.resident_sources.contains(&id.tid()) {
            SourceFiller::DeclaredResident
        } else if self.source_segment_is_aliased(id) {
            SourceFiller::SegmentOwner
        } else {
            SourceFiller::Nobody
        }
    }

    /// Does this source live in a segment this session has ALIASED onto an owner's region?
    ///
    /// Keyed on `seg_aliased`, not `seg_borrowed`: `mark_borrowed` only PROMISES an alias and
    /// shrinks the segment to a 128 B placeholder. A session that marked and then failed to alias
    /// reads that placeholder, which is precisely the fill-nobody-did this guard exists to refuse.
    fn source_segment_is_aliased(&self, id: &bundle::PlaceId) -> bool {
        self.places
            .get(id)
            .and_then(|p| SegIdx::checked(p.segment as i64))
            .is_some_and(|seg| self.seg_aliased[seg.get()])
    }

    /// Does this session still owe an alias for a segment it declared borrowed? While
    /// true, that segment holds its 128 B placeholder, so no CB may be rendered through
    /// it — the parity gate waits (see [`Executor::prepare`]) and runs at the alias that
    /// completes the set ([`Executor::alias_seg_from`]).
    fn borrow_set_pending(&self) -> bool {
        self.seg_borrowed
            .iter()
            .zip(self.seg_aliased.iter())
            .any(|(borrowed, aliased)| *borrowed && !*aliased)
    }

    /// Bind a WEIGHT by name, taking ownership of its bytes. Weights are bound before
    /// [`Executor::prepare`], staged once into the resident segments, and their host bytes are
    /// reclaimed there.
    ///
    /// ⭐ MOVES, never copies. The C++ took a pointer to caller memory precisely to avoid
    /// duplicating an 8.9 GB model across the FFI; owning the `Vec` gets the same zero copies with
    /// no lifetime contract to violate.
    pub fn bind_weight(&mut self, id: bundle::PlaceId, bytes: Vec<u8>) {
        self.weight_ids.insert(id);
        self.bound.insert(id, bytes);
    }

    /// Bind an ACTIVATION by name (IEEE-fp16 bytes) — re-bound every step. A name bound before
    /// prepare is recorded as a weight instead, matching the C++ contract.
    ///
    /// ⛔ THE BIND COVERS ITS PLACEMENT'S WRITE-COVER OR REFUSES. A `Full` placement demands its
    /// whole extent; a `FoldBlocks` one admits a whole number of its blocks and nothing else
    /// ([`bundle::BindCover`], minted where the placement was sized). A short bind against a
    /// `Full` placement leaves a tail of whatever the device had there — for an ADDITIVE mask
    /// that tail reads as ZERO, i.e. as "no mask" — and the door is where that stops. A name
    /// this bundle never placed still binds (the refill skips it, as it always has); the cover
    /// law speaks only for tensors the bake placed.
    pub fn bind_input(&mut self, id: bundle::PlaceId, bytes: Vec<u8>) -> Result<()> {
        if let Some(p) = self.places.get(&id) {
            let len = bytes.len() as u64;
            if !p.cover.admits(len, p.size) {
                bail!(
                    "bind_input('{id}'): {len} B against a {} B placement (seg{} off {}) whose \
                     write-cover is {:?} — REFUSING, a bind that misses its cover leaves the \
                     untaken bytes holding whatever the device had there",
                    p.size,
                    p.segment,
                    p.offset,
                    p.cover
                );
            }
        }
        if !self.prepared {
            self.weight_ids.insert(id);
        }
        // A previously-raw id re-bound as fp16 must stop being raw, or the shadow would take an
        // unconverted fp16 buffer. Cleared here rather than left to the caller.
        self.raw_ids.remove(&id);
        // An id re-bound UNKEYED goes back to the per-step re-upload, so a keyed bind can never
        // shadow a later plain one.
        self.keyed.remove(&id);
        self.staged.remove(&id);
        self.bound.insert(id, bytes);
        Ok(())
    }

    /// ⭐ Bind an f32 ACTIVATION by converting it STRAIGHT INTO ITS SLOT OF THE SHADOW — f32 →
    /// SEN169 in one pass ([`sen_convert::f32_to_sen_bytes`]), the same bytes
    /// `bind_input(f32_to_f16_le(vals))` and the refill's `ieee_to_sen_bytes` produce between them,
    /// without the IEEE-f16 buffer in between or the refill's second convert pass. Re-bound every
    /// step, and the refill sends its range every step, exactly like [`Self::bind_input`]: only
    /// WHERE the conversion happens moves.
    ///
    /// The placement checks the refill makes for a `bound` activation are made here instead, where
    /// the write happens: a name this bundle never placed is skipped (as the refill skips it), and a
    /// bind that misses its placement's write-cover, or lands past its segment's shadow, is refused.
    pub fn bind_input_f32(&mut self, id: bundle::PlaceId, vals: &[f32]) -> Result<()> {
        if !self.prepared {
            bail!("bind_input_f32('{id}') before prepare — the shadow is sized at prepare");
        }
        // A RESIDENT WEIGHT is never re-sent by a step (the refill skips it), so a per-step bind of
        // one writes nothing — as before, where the refill skipped its `bound` entry.
        if self.weight_ids.contains(&id) {
            return Ok(());
        }
        let Some(p) = self.places.get(&id) else {
            return Ok(()); // no placement → not a program input, as the refill treats it
        };
        let Some(seg) = SegIdx::checked(p.segment as i64) else {
            return Ok(());
        };
        let len = vals.len() as u64 * 2;
        if !p.cover.admits(len, p.size) {
            bail!(
                "bind_input_f32('{id}'): {len} B against a {} B placement (seg{} off {}) whose \
                 write-cover is {:?} — REFUSING, a bind that misses its cover leaves the untaken \
                 bytes holding whatever the device had there",
                p.size,
                p.segment,
                p.offset,
                p.cover
            );
        }
        let dst = self.seg_host[seg.get()]
            .get_mut(p.offset as usize..(p.offset + len) as usize)
            .ok_or_else(|| anyhow!("bind: activation '{id}' overflows seg{}", p.segment))?;
        sen_convert::f32_to_sen_bytes(vals, dst);
        self.raw_ids.remove(&id);
        self.keyed.remove(&id);
        self.bound.remove(&id);
        self.staged.insert(id, len);
        Ok(())
    }

    /// The key this session's KEYED activation `id` holds — the [`CausalKey`] the bytes on the
    /// device are a function of — or `None` when `id` was never bound keyed. A launch whose key
    /// equals this binds nothing for `id`.
    ///
    /// [`CausalKey`]: scratchy_subtile::sdsc_abstract::CausalKey
    pub fn keyed_holds(
        &self,
        id: bundle::PlaceId,
    ) -> Option<scratchy_subtile::sdsc_abstract::CausalKey> {
        self.keyed.get(&id).map(|k| k.key)
    }

    /// ⭐ Bind an ACTIVATION whose bytes are a function of `key` alone (IEEE-fp16 bytes). It is
    /// converted and uploaded by the NEXT predict and not again: until a launch re-binds it under a
    /// different key, the device keeps these bytes (segment 3 is never written on card). Every
    /// other activation stays on [`Self::bind_input`]'s per-step re-upload.
    pub fn bind_input_keyed(
        &mut self,
        id: bundle::PlaceId,
        key: scratchy_subtile::sdsc_abstract::CausalKey,
        bytes: Vec<u8>,
    ) -> Result<()> {
        if !self.prepared {
            bail!(
                "bind_input_keyed('{id}') before prepare — a keyed activation is a per-step bind"
            );
        }
        if let Some(p) = self.places.get(&id) {
            let len = bytes.len() as u64;
            if !p.cover.admits(len, p.size) {
                bail!(
                    "bind_input_keyed('{id}'): {len} B against a {} B placement (seg{} off {}) \
                     whose write-cover is {:?} — REFUSING, a bind that misses its cover leaves \
                     the untaken bytes holding whatever the device had there",
                    p.size,
                    p.segment,
                    p.offset,
                    p.cover
                );
            }
        }
        self.raw_ids.remove(&id);
        self.staged.remove(&id);
        self.keyed.insert(id, Keyed { key, fresh: true });
        self.bound.insert(id, bytes);
        Ok(())
    }

    /// ⭐ [`Self::bind_input_keyed`], UPLOADED NOW — a session's load-time staging of the keyed
    /// activation its first launch would otherwise bind. The upload is the refill a predict runs
    /// first, so a launch under `key` then finds it held ([`Self::keyed_holds`]) and sends nothing for
    /// `id`, and a launch under any other key re-binds over it exactly as it would over a key a
    /// previous launch left.
    ///
    /// ⛔ A STAGED CONSTANT FILLS ITS PLACEMENT EXACTLY. The refill guard refuses only OVER-binds,
    /// because several per-step activations bind less than they reserve; an under-bind of an ADDITIVE
    /// mask leaves a tail the ops read as ZERO, i.e. as "no mask". Staging is done once, at load, for
    /// a buffer whose size is a law of the rung, so a size that is not the placement's is a
    /// wiring/layout desync, refused here rather than served.
    ///
    /// Returns the bytes the upload sent.
    pub fn stage_keyed(
        &mut self,
        id: bundle::PlaceId,
        key: scratchy_subtile::sdsc_abstract::CausalKey,
        bytes: Vec<u8>,
    ) -> Result<u64> {
        let placed = self
            .places
            .get(&id)
            .ok_or_else(|| anyhow!("stage_keyed('{id}'): this bundle places no such tensor"))?
            .size;
        if bytes.len() as u64 != placed {
            bail!(
                "stage_keyed('{id}', {key:?}): {} B staged into a {placed} B placement — a staged \
                 constant must fill its placement exactly",
                bytes.len()
            );
        }
        self.bind_input_keyed(id, key, bytes)?;
        let (_, sent) = self.refill_activations(Diag::get())?;
        self.stream_ref()?
            .synchronize()
            .map_err(|e| anyhow!("stage_keyed('{id}'): sync: {e}"))?;
        Ok(sent)
    }

    /// ⭐ MATERIALIZE THE ACTIVATION SEGMENT'S HOST SHADOW — fault every page of it in now, at load,
    /// so a request's first bind into it does not.
    ///
    /// The ACTIVATION segment is the one segment the host writes on a step: the placement pass puts
    /// every per-step input there (`SegRole::Activation` — embeddings, rotary tables, both masks, the
    /// gather index, the constants), and the same law names it here. Prepare allocates its shadow
    /// zeroed, which the allocator satisfies with pages mapped on first write, so before this the
    /// first request faulted in every page its binds touched, inside its TTFT. The other segments'
    /// shadows are left alone: the intermediates are device-written and the host never touches them,
    /// so committing them would be RSS for nothing.
    ///
    /// Contents unchanged ([`fault_in`]): bytes already staged there, such as a causal mask
    /// `stage_keyed` put in, stay exactly as they are, and the device is not touched. Only WHERE the
    /// faults land moves. Returns the shadow's length — the RSS this commits.
    pub fn materialize_activation_shadow(&mut self) -> Result<u64> {
        if !self.prepared {
            bail!("materialize_activation_shadow before prepare — the shadow is sized at prepare");
        }
        let seg = crate::lower_subtile_tape_to_superdsc::SegRole::Activation.segment();
        let shadow = self
            .seg_host
            .get_mut(seg)
            .ok_or_else(|| anyhow!("the activation segment {seg} is not a tensor segment"))?;
        fault_in(shadow);
        Ok(shadow.len() as u64)
    }

    /// ⭐⭐⭐⭐⭐ MINT THE ONLY VALUE [`Self::bind_input_raw`] ACCEPTS — the payload measured against
    /// the placement it will be written into, in one step, by the session that owns both.
    ///
    /// `None` on every shape a raw bind cannot have: a name this bundle never PLACED (a bind the
    /// refill loop used to skip in silence — and a skipped index table reads as entry 0, which is a
    /// REAL address: page 0's first block, i.e. the whole batch on row 0's keys), a payload LONGER
    /// than its own placement (past it is the next tensor's bytes, and for the index table those
    /// bytes are read as block numbers), or a byte count that is not a whole number of int32
    /// entries (a partial trailing entry is a truncated address).
    ///
    /// ⛔ SHORT IS LEGAL EXACTLY ON THE CLASSES THE BAKE DECLARED — the write-cover law
    /// ([`bundle::BindCover`]): a `Full` placement demands its whole extent, and a `FoldBlocks`
    /// placement admits a whole number of its blocks and nothing else. The short binds the old
    /// `<=` served are the fold's per-pass family — the index table that rides the mask's shift,
    /// whose block stride is the one number its mint shares with the mask's own — and those
    /// mints state `FoldBlocks` where they are sized. A short bind against any other placement
    /// now has no value to arrive in.
    pub fn place_raw(&self, id: bundle::PlaceId, bytes: Vec<u8>) -> Option<RawBind> {
        let p = self.places.get(&id)?;
        (bytes.len().is_multiple_of(4) && p.cover.admits(bytes.len() as u64, p.size))
            .then_some(RawBind { id, bytes })
    }

    /// ⭐⭐⭐⭐⭐ Bind an ACTIVATION whose bytes are **ALREADY THE DEVICE'S** — copied into the shadow
    /// verbatim, with no IEEE→SEN conversion. The gather's int32 index table, and nothing else today.
    ///
    /// ⛔ THIS IS NOT AN OPTIMISATION OF [`Self::bind_input`]. The f16 fast path
    /// (`run_step_no_logits_f16`) also hands over device-format bytes, and it still routes them
    /// through `bind_input` — which is sound only because `ieee_to_sen_bytes` is applied to them
    /// too, so "already narrowed" means "already f16", not "already staged". An int32 tensor has no
    /// fp16 encoding to survive that pass: see [`Self::raw_ids`] for what it does to a block number.
    ///
    /// ⛔⛔⛔ IT TAKES A [`RawBind`], NOT `(id, bytes)`, AND THAT IS THE WHOLE GUARD. The signature
    /// used to be `(PlaceId, Vec<u8>)` — bytes with NO relation to the placement they land in — and
    /// the relation was re-checked, far away and at runtime, by `refill_activations`'s over-bind
    /// `bail!`. A length that exceeds its placement now has no value to arrive in: see
    /// [`Self::place_raw`], the only constructor.
    pub fn bind_input_raw(&mut self, b: RawBind) {
        if !self.prepared {
            self.weight_ids.insert(b.id);
        }
        self.raw_ids.insert(b.id);
        self.staged.remove(&b.id);
        self.bound.insert(b.id, b.bytes);
    }

    /// ⭐ THE PLACEMENT EXTENT of step-input `slot`, in bytes — the QUERY half of the fill
    /// contract. The caller sizes its bytes against the artifact's own answer (the discovery
    /// table's tid, resolved through the placement the placement pass sized from the program's
    /// own view), and [`Self::fill_step_input`]'s `==` law re-measures the same number at the
    /// door — so a derivation that disagrees with the bake refuses twice, at both ends.
    ///
    /// `None` when this bundle DECLARES no such slot — the caller's query-then-fill gate, so an
    /// undeclared slot is a quiet skip and not a refusal the caller re-derives around — or when
    /// the table names a tid this bundle never placed (a bake desync, which the fill would
    /// refuse anyway; seeing it here names it earlier).
    pub fn step_input_extent(&self, slot: u32) -> Option<u64> {
        let si = self
            .code
            .layout
            .step_inputs
            .iter()
            .find(|s| s.slot == slot)?;
        Some(self.code.layout.place_of_tid(si.tid)?.size)
    }

    /// ⭐⭐⭐⭐⭐ FILL ONE HOST-STAGED STEP INPUT — the runtime's staging half of the per-step
    /// inputs a spliced program binds past the graph.
    ///
    /// The SLOT is the caller's whole handle. This resolves it through the bundle's own baked
    /// discovery table (data — the runtime never recomputes the producer's `tensors.len()+k`
    /// numbering), measures the bytes against the placement's extent (the program's own
    /// `construct_memory_view` statement, sized by the placement pass from that same view), and
    /// routes them through the channel the bake's staging law states — the f16 narrowing or the
    /// raw copy, both of which are the guarded channels above. Every disagreement refuses naming
    /// the slot and the tid (see [`bundle::BundleLayout::step_input_fill`], where the law lives).
    ///
    /// ⛔ ZERO KERNEL VOCABULARY, AND THAT IS THE WHOLE DESIGN: this asks WHICH slot, HOW MANY
    /// bytes, and WHICH channel — nothing about what the input IS. The caller that knows the
    /// content (a page table, a validity mask) computes it and hands over bytes; a second kernel
    /// family stating per-step inputs flows through here unchanged.
    ///
    /// ⛔ REFUSED BEFORE `prepare`, unlike the channels it routes to: a bind that arrives before
    /// prepare is latched as a RESIDENT weight and never re-sent, and a per-step input that stops
    /// being re-sent is every later launch reading the first launch's bytes — silently, fluently
    /// wrong. The refusal is the cheapest place that cannot happen.
    pub fn fill_step_input(&mut self, slot: u32, bytes: Vec<u8>) -> Result<()> {
        if !self.prepared {
            bail!(
                "fill_step_input(slot {slot}): session NOT YET PREPARED — a bind before prepare is \
                 latched as a resident weight and never re-sent, so a per-step input staged now \
                 would freeze the first launch's bytes into every later one"
            );
        }
        let fill = self
            .code
            .layout
            .step_input_fill(slot, bytes.len())
            .map_err(|e| anyhow!("{e} (bundle {})", self.code.fp))?;
        match fill.stage {
            bundle::StepInputStage::IeeeF16 => self.bind_input(fill.place, bytes)?,
            bundle::StepInputStage::RawU32 => {
                // `place_raw` is the ONLY constructor of the value `bind_input_raw` accepts; its
                // own guards (whole u32 entries, no over-bind) are subsumed by the fill's `==`
                // law but re-deriving nothing — it re-measures against the same placement.
                let rb = self.place_raw(fill.place, bytes).ok_or_else(|| {
                    anyhow!(
                        "fill_step_input(slot {slot}): the bytes passed the fill's extent law but \
                         not the raw channel's own entry law — a u32 staging whose length is not a \
                         whole number of entries"
                    )
                })?;
                self.bind_input_raw(rb);
            }
        }
        Ok(())
    }

    /// ⭐⭐⭐⭐⭐ Bind a RESIDENT FP32 CONSTANT at load — staged VERBATIM, once, never re-bound.
    /// The composite argmax's descending ramp and scalar-V stick (#603 PR3a), and nothing else
    /// today.
    ///
    /// ⛔ THIS IS NOT [`Self::bind_input_raw`], AND NOT A LAZIER [`Self::bind_weight`]. The raw
    /// channel is PER-STEP (a raw id bound before prepare is exactly the raw+weight combination
    /// the staging loop refuses); the weight channel re-encodes every byte through
    /// `ieee_to_sen_bytes`, which reads a 4-byte fp32 as two fp16s and rewrites both mantissas.
    /// A const bound here is a WEIGHT — resident, skipped by every per-step refill — whose bytes
    /// need no conversion at all, because `Df::Fp32` IS IEEE fp32 and the host's LE bytes are
    /// the device's own.
    ///
    /// Validated the way [`Self::place_raw`] validates its one product: a whole number of f32
    /// entries, and no longer than the const's own placement (`>` not `!=` — the over-bind law;
    /// a short bind leaves the placement's tail zero, which is a valid constant image).
    pub fn bind_const_fp32(&mut self, id: bundle::PlaceId, bytes: Vec<u8>) -> Result<()> {
        if self.prepared {
            bail!(
                "bind_const_fp32({id}): session ALREADY PREPARED — a resident constant must be \
                 bound before prepare, where staging copies it into the shadow the H2D ships \
                 once at load"
            );
        }
        let Some(p) = self.places.get(&id) else {
            bail!(
                "bind_const_fp32({id}): NO placement in this bundle — the layout did not place \
                 this const, so the caller's bind and the bundle's layout disagree about the \
                 composite argmax. REFUSING"
            );
        };
        if !bytes.len().is_multiple_of(4) {
            bail!(
                "bind_const_fp32({id}): {} B is not a whole number of fp32 entries",
                bytes.len()
            );
        }
        if bytes.len() as u64 > p.size {
            bail!(
                "bind_const_fp32({id}): {} B exceeds its own {} B placement (seg{} off {}) — \
                 REFUSING, this would corrupt the following tensors",
                bytes.len(),
                p.size,
                p.segment,
                p.offset
            );
        }
        self.weight_ids.insert(id);
        self.const_fp32_ids.insert(id);
        self.bound.insert(id, bytes);
        Ok(())
    }

    /// Declare, BEFORE prepare, that `seg` will be BORROWED from another session via
    /// [`Executor::alias_seg_from`]: prepare gives it a 128 B placeholder, skips its host staging
    /// and skips its H2D, so the session never builds the copy the alias immediately discards.
    ///
    /// The caller MUST then alias it — a borrowed segment left un-aliased holds only the
    /// placeholder, so a failed alias is fatal for that session.
    ///
    /// This is what makes the prefill WIDTH LADDER cheap. Measured per session (granite-3.1-2b,
    /// 2.6 GB of weights): stage 1.70s + bind 0.40s + seg1 H2D 0.45s ≈ 2.6s, all of it producing a
    /// duplicate the alias then throws away.
    pub fn mark_borrowed(&mut self, seg: SegIdx) -> Result<()> {
        // Must precede prepare: afterwards the full region is already allocated and H2D'd, so the
        // flag would silently do nothing. Refuse instead of pretending it took effect.
        if self.prepared {
            bail!(
                "mark_borrowed(seg{}): session ALREADY PREPARED — the segment is allocated and \
                 H2D'd; mark it before prepare",
                seg.get()
            );
        }
        self.seg_borrowed[seg.get()] = true;
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Host staging
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Convert + stage the bound weights into host staging (IEEE→SEN / tile-retile), in PARALLEL.
    /// Split out of prepare so the caller can run it WHILE the background device prewarm brings the
    /// card up — the ~1s convert then costs nothing on the critical path. Idempotent; prepare falls
    /// back to staging inline if this was never called.
    /// ⭐ THE SHADOW/REGION SLOT a `(segment, bank)` pair names: the segments first, then the weight
    /// segment's EXTRA banks, which is why staging can treat both as one flat list.
    ///
    /// ⛔ THE SEGMENT INDEX ALONE STOPPED BEING AN ADDRESS the moment one segment could be served by
    /// several regions. Everything that reaches for a shadow or a region by segment goes through
    /// here, so a banked weight cannot silently resolve to bank 0's bytes — which would be one whole
    /// program (the suffix, i.e. the lm_head) reading another bank's weights: fluent, wrong output.
    ///
    /// `None` for a bank this session has no region for, which callers turn into a refusal rather
    /// than a fallback.
    fn slot_of(&self, seg: usize, bank: u32) -> Option<usize> {
        match bank {
            0 => (seg < NUM_SEGMENTS).then_some(seg),
            b if seg == SEG_WEIGHT.get() && (b as usize) <= self.weight_bank_bytes.len() => {
                Some(NUM_SEGMENTS + b as usize - 1)
            }
            _ => None,
        }
    }

    /// Slot `s`'s host shadow (read-only). Slots past the segments are weight banks.
    fn shadow(&self, s: usize) -> &[u8] {
        match s.checked_sub(NUM_SEGMENTS) {
            None => &self.seg_host[s],
            Some(b) => &self.weight_bank_host[b],
        }
    }

    pub fn stage_weights(&mut self) -> Result<()> {
        if self.staged_host {
            return Ok(());
        }
        // NOTHING BOUND ⇒ this session holds allocations only; it is aliased onto an owner's
        // populated regions before use, so every host shadow here is dead. Sizing them at the
        // placeholder skips a 2.6 GB memset for the weight segment — measured as 2.134s on the
        // batched-prefill session, which binds nothing precisely because its ALLOCATION is the
        // point and its bytes are not.
        let reserve_only = self.bound.is_empty();
        for i in 0..NUM_SEGMENTS {
            // A BORROWED segment is never staged or H2D'd by this session, so it gets the
            // placeholder rather than a zero-fill the width of the whole segment. ...but ONLY for
            // the segments an alias replaces ([`SEG_RESIDENT`]). The ACTIVATION segments are bound
            // per-forward by this very session and must keep their real size.
            let aliased_later = reserve_only && is_resident_seg(i);
            let bytes = if self.seg_borrowed[i] || aliased_later {
                128
            } else if self.segment_bytes[i] > 0 {
                self.segment_bytes[i] as usize
            } else {
                128
            };
            self.seg_host[i] = vec![0u8; bytes];
        }
        // ⭐ AND ONE SHADOW PER EXTRA WEIGHT BANK, sized the same way. A bank holds only weights, so
        // it follows the RESIDENT rule exactly: a session that binds nothing gets a placeholder
        // because an owner's region will replace it.
        for b in 0..self.weight_bank_bytes.len() {
            let bytes = if reserve_only || self.seg_borrowed[SEG_WEIGHT.get()] {
                128
            } else {
                self.weight_bank_bytes[b].max(128) as usize
            };
            self.weight_bank_host[b] = vec![0u8; bytes];
        }

        // ⛔ EVERY PLACED COPY MUST BE BOUND — the owner-scoped mirror of the bound-without-place
        // refusal in the loop below. The head-major o_proj copies are PLACED by the layout but
        // STAGED only through the wiring's `weight_copies` list, which the worker turns into bound
        // rows at load. #299's registry rewrite of `graph_wiring` emptied that list while the
        // placement pass kept placing, so every copy window sat never-written while the m=1 fp8 o
        // matmul read it: fluent garbage, card-only, granite-3.1-8b fp8 m=1 alone (every other
        // geometry never reads a copy, and the emulator executes the KTIR program, not the staging,
        // so no bench caught it either). Scoped to sessions that stage weights — a borrower binds
        // none by design and adopts the owner's populated regions.
        let stages_weights = self.bound.keys().any(|nm| self.weight_ids.contains(nm));
        if stages_weights {
            let unbound: Vec<u32> = self
                .code
                .layout
                .weight_copies
                .iter()
                .map(|&(_, to)| to)
                .filter(|&to| !self.bound.contains_key(&bundle::PlaceId::Act(to)))
                .collect();
            if !unbound.is_empty() {
                bail!(
                    "stage: {} o_proj head-major cop{} placed but never bound (first: t{}) — the \
                     wiring's weight_copies list and the layout's placements disagree, so the m=1 \
                     fp8 o matmul would read never-written windows. REFUSING: this is the #299 \
                     defect class (graph_wiring's registry rewrite dropped the copy mint), and a \
                     session that stages weights must stage every copy the layout places.",
                    unbound.len(),
                    if unbound.len() == 1 { "y" } else { "ies" },
                    unbound[0],
                );
            }
        }

        // ── Collect the weight placements, validating each against its own extent. ──
        // One bound tensor's staging transform — what the parallel stage loop does to its bytes.
        // ⭐ VERBATIM IS A THIRD THING, NOT THE ABSENCE OF THE SECOND: this used to be
        // `Option<(&KernelWeight, Element)>`, where `None` meant "convert", so an fp32 const
        // had no honest spelling — its bytes are neither retiled nor IEEE-fp16-to-SEN
        // re-encoded, they are already the device's own (`Df::Fp32` = IEEE_FP32).
        #[derive(Clone, Copy)]
        enum StageMode<'a> {
            /// IEEE f16 → SEN169 f16 — the default for every magnitude the card holds in fp16.
            Convert,
            /// Retile into the device tile layout this matmul kernel reads.
            Retile(&'a KernelWeight<'a>, Element),
            /// Copy VERBATIM — a resident fp32 constant ([`Executor::bind_const_fp32`]).
            Verbatim,
        }
        struct Item<'a> {
            /// The SHADOW SLOT this weight stages into — a segment, or a weight bank beyond bank 0.
            /// See [`Executor::slot_of`]: banking gives one segment several regions, so the segment
            /// index alone stopped being an address.
            slot: usize,
            offset: usize,
            src: &'a [u8],
            mode: StageMode<'a>,
        }
        let mut items: Vec<Item<'_>> = Vec::new();
        for (nm, src) in &self.bound {
            if !self.weight_ids.contains(nm) {
                continue; // activation
            }
            // ⛔ A RAW BIND CANNOT BE A WEIGHT. This loop's every destination goes through
            // `ieee_to_sen_bytes` or the retile walk, neither of which may touch int32 index entries
            // (see [`Self::raw_ids`]) — and a raw id lands here only by having been bound BEFORE
            // prepare, which `bind_input_raw` records as a weight like any other. Refuse rather than
            // stage it through a converter that would rewrite every entry.
            if self.raw_ids.contains(nm) {
                bail!(
                    "stage: '{nm}' was bound RAW but is a WEIGHT (bound before prepare) — raw bytes \
                     are only honoured on the per-step activation path, and this loop would re-encode \
                     them as fp16. Bind it after prepare."
                );
            }
            let Some(p) = self.places.get(nm) else {
                // ⛔ WAS `eprintln!` + `continue` — a weight the bake never placed was SKIPPED, so
                // the device read whatever the segment happened to hold and the model produced
                // fluent garbage. With the id as the key, a miss means the host bound a tensor this
                // bundle does not have: a wiring bug, not a condition to carry on from.
                bail!("stage: weight {nm} has NO placement in this bundle — REFUSING");
            };
            let Some(seg) = SegIdx::checked(p.segment as i64) else {
                bail!(
                    "stage: weight '{nm}' names segment {} — out of range",
                    p.segment
                );
            };
            // OVER-BIND GUARD: a weight whose staged bytes exceed its own placement would
            // overwrite whatever the layout put after it. `>` not `!=`: a short bind is legitimate.
            if src.len() as u64 > p.size {
                bail!(
                    "stage: weight '{nm}' stages {} B into a {} B placement (seg{} off {}) — \
                     REFUSING, this would corrupt the following tensors",
                    src.len(),
                    p.size,
                    p.segment,
                    p.offset
                );
            }
            let Some(slot) = self.slot_of(seg.get(), p.bank) else {
                bail!(
                    "stage: weight '{nm}' names seg{} bank {}, which this session has no shadow for \
                     ({} weight bank(s))",
                    p.segment,
                    p.bank,
                    self.weight_bank_bytes.len() + 1
                );
            };
            if p.offset as usize + src.len() > self.shadow(slot).len() {
                bail!(
                    "stage: weight '{nm}' off {} + {} B exceeds seg{} bank {} ({} B)",
                    p.offset,
                    src.len(),
                    p.segment,
                    p.bank,
                    self.shadow(slot).len()
                );
            }
            let mode = if self.const_fp32_ids.contains(nm) {
                // A resident fp32 const — its LE bytes are the device encoding, so nothing
                // here may touch them. Checked FIRST so the kernel-weight lookup below can
                // never hand a const a tile walk (it has no KernelWeight, but the ordering
                // makes that a structural fact rather than a happy one).
                StageMode::Verbatim
            } else {
                match self.kernel_weights.get(nm) {
                    Some(&k) => {
                        let dev_elems: u64 = k.device_size.iter().product();
                        if dev_elems as usize * k.word_length as usize != src.len() {
                            bail!(
                                "stage: kernel weight '{nm}' host size {} B != \
                                 prod(device_size)*word_length ({} B) — manifest/shape mismatch",
                                src.len(),
                                dev_elems as usize * k.word_length as usize
                            );
                        }
                        let Some(elem) = Element::from_word_length(k.word_length) else {
                            bail!(
                                "stage: kernel weight '{nm}' word_length {} has no tile walk",
                                k.word_length
                            );
                        };
                        StageMode::Retile(k, elem)
                    }
                    None => StageMode::Convert,
                }
            };
            items.push(Item {
                slot,
                offset: p.offset as usize,
                src,
                mode,
            });
        }

        // ── Hand each item its OWN destination sub-slice. Splitting the segment buffers this way
        //    is what makes the parallel stage safe with no unsafe at all — and it PROVES the
        //    placements are disjoint (an overlap shows up as a backwards cursor, which is a
        //    refusal, where the C++ would simply have raced).
        /// One weight's staging work: its source bytes, its OWN destination sub-slice, and its
        /// staging transform ([`StageMode`]).
        type StageTask<'a> = (&'a [u8], &'a mut [u8], StageMode<'a>);
        let n_items = items.len();
        items.sort_by_key(|i| (i.slot, i.offset));
        let mut tasks: Vec<StageTask<'_>> = Vec::with_capacity(n_items);
        {
            // ⭐ THE SLOT SPACE IS THE SEGMENTS **THEN** THE EXTRA WEIGHT BANKS, in that order, which
            // is exactly what `slot_of` returns — one flat list of destinations, so the split-and-
            // advance below (and its disjointness proof) is unchanged by banking. `chain` here rather
            // than a second staging loop: a bank holds ~half a model's weights, and a serial pass
            // over those would cost more than the whole parallel stage it bypassed.
            let mut rests: Vec<&mut [u8]> = self
                .seg_host
                .iter_mut()
                .chain(self.weight_bank_host.iter_mut())
                .map(|v| &mut v[..])
                .collect();
            let mut cursor = vec![0usize; rests.len()];
            for it in &items {
                let Some(skip) = it.offset.checked_sub(cursor[it.slot]) else {
                    bail!(
                        "stage: placements overlap in slot {} at byte {} — refusing to stage \
                         (a weight would overwrite the previous one)",
                        it.slot,
                        it.offset
                    );
                };
                let rest = std::mem::take(&mut rests[it.slot]);
                let (_, r) = rest.split_at_mut(skip);
                let (dst, r2) = r.split_at_mut(it.src.len());
                rests[it.slot] = r2;
                cursor[it.slot] = it.offset + it.src.len();
                tasks.push((it.src, dst, it.mode));
            }
        }

        // ── LPT deal across the cores: sort by size descending and deal round-robin, which is
        //    better balanced than equal chunking (weights differ by orders of magnitude) and needs
        //    no shared counter — the C++ used an atomic work-stealing index for the same reason.
        tasks.sort_by_key(|(src, _, _)| std::cmp::Reverse(src.len()));
        let nthreads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(tasks.len().max(1));
        let mut buckets: Vec<Vec<_>> = (0..nthreads).map(|_| Vec::new()).collect();
        for (i, t) in tasks.into_iter().enumerate() {
            buckets[i % nthreads].push(t);
        }
        std::thread::scope(|scope| {
            for bucket in buckets {
                scope.spawn(move || {
                    for (src, dst, mode) in bucket {
                        match mode {
                            StageMode::Retile(k, elem) => stage_weight_tiled(
                                src,
                                dst,
                                RetileWalk {
                                    device_size: &k.device_size,
                                    stride_map: &k.stride_map,
                                },
                                elem,
                            ),
                            StageMode::Convert => sen_convert::ieee_to_sen_bytes(src, dst),
                            StageMode::Verbatim => dst.copy_from_slice(src),
                        }
                    }
                });
            }
        });

        self.n_weights_staged = n_items;
        self.staged_host = true;
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Prepare
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Init the device runtime + stream; allocate + H2D every program ONCE; allocate the ≤7 packed
    /// tensor-segment regions; H2D the staged weights; then reclaim the host bytes that are dead.
    pub fn prepare(&mut self) -> Result<()> {
        if self.prepared {
            return Ok(());
        }
        let d = Diag::get();
        let start = Instant::now();
        let mut t0 = Instant::now();
        let lap = |what: &str, from: &mut Instant| {
            if d.prep_time {
                eprintln!(
                    "[sdsc-prep-time] {what:<16} {:.3}s",
                    from.elapsed().as_secs_f64()
                );
            }
            *from = Instant::now();
        };

        crate::runtime::init_runtime().map_err(|e| anyhow!("flex runtime init failed: {e}"))?;
        lap("runtime-init", &mut t0);
        // A DEDICATED created stream (not the default), and PERSISTENT for the session: the
        // scheduler retires each op's fence from its per-stream deque at completion
        // (SchedulerState::retire_fence), so the deques hold only in-flight fences instead of
        // one entry per launch forever. Unbounded, not large per step — ~200 launches per
        // decode step (202 measured on 2b fp8) is tens of KB/forward, but a session stream
        // never dies, so it accumulates for its whole life (the old per-forward
        // recreate-and-destroy workaround papered over exactly this; the PR1 gate run retired
        // 331,220 fences over 1,570 forwards with RSS flat to the kB). The resident
        // program/weight/KV allocations live on the runtime, not the stream.
        self.stream =
            Some(Stream::create().map_err(|e| anyhow!("prepare: could not create a stream: {e}"))?);
        lap("stream-create", &mut t0);

        // ── Programs. Each op's device binary gets its OWN Program-segment alloc + H2D ONCE; they
        //    all SHARE the resident tensor segments below. ──
        {
            self.alloc_ops_at(ListSel::Prefix)?;
            if self.body_rungs.is_empty() {
                for c in 0..self.body_ops.len() {
                    self.alloc_ops_at(ListSel::Body(c))?;
                }
                self.split_bodies_ready = true;
            } else {
                // When every rung has a fold-fused twin, the SPLIT bodies are dead weight until
                // some context outgrows one page. Allocating both up front is what made the paged
                // decode session carry 104 program allocations where the unpaged one carried 39,
                // and the per-token cost of that lands in the DMA ENQUEUE (measured 1.21 ms to
                // enqueue a 59 KB transfer against 0.58 ms unpaged). Deferring keeps the
                // steady-state session the same size as the unpaged one.
                let twins_cover_every_rung = self.body_rungs_fused.len() == self.body_rungs.len();
                if !twins_cover_every_rung {
                    for i in 0..self.body_rungs.len() {
                        for c in 0..self.body_rungs[i].1.len() {
                            self.alloc_ops_at(ListSel::Rung(i, c))?;
                        }
                    }
                    self.split_bodies_ready = true;
                }
            }
            self.alloc_ops_at(ListSel::Suffix)?;
        }
        lap("program-h2d", &mut t0);

        // ── ONE device region per tensor segment, sized from the manifest. Even an empty segment
        //    gets a 128 B placeholder so `tensor_allocs` stays POSITIONAL (entry i backs the
        //    program's logical segment i). Packing many tensors into ≤7 regions is what defeats the
        //    7-tensor-segment hardware cap. ──
        // ⛔⛔⛔⭐⭐⭐ THE ALLOCATION ORDER IS NATURAL (seg 0..N), AND WEIGHT-FIRST IS **MEASURABLY
        // WORSE** — tried and rejected 2026-08-16, so nobody spends another bake on the inference.
        //
        // The note below this loop says "an allocation ahead of the weights therefore taxes decode and
        // leaves prefill alone, which is the exact shape of the gap", and it is applied to the fused
        // twins by allocating them LAST. It is TEMPTING to read that as "so allocate the weights
        // first", especially since 8b puts 110 MB of SEG_INTERMEDIATE ahead of an 8.9 GB weight region
        // while 2b puts only 16 MB ahead of 5.07 GB, and 8b streams at 103 GB/s against 2b's 185.
        //
        // MEASURED, granite-3.1-8b fp8, 10 runs each:
        //   natural order  modal 12.1-12.2 tok/s, ITL 82.4 ms
        //   weight-first   modal 11.8-11.9 tok/s, ITL 84.4 ms   <-- WORSE
        //
        // So the region effect is REAL but NOT MONOTONIC IN ORDER: allocating the weights first does
        // not hand them the fast region, it lands them somewhere slower. Whatever selects the fast
        // region is not "earliest large allocation", so an ordering heuristic here cannot reach it —
        // consistent with the recorded finding that flex exposes no placement input and only the region
        // table is readable.
        for i in 0..NUM_SEGMENTS {
            let mut bytes = if self.seg_borrowed[i] {
                // A BORROWED segment gets the same 128 B placeholder an empty one does: `alias_seg`
                // replaces this address with the owner's region, so allocating the full extent
                // would reserve device memory only to drop it immediately.
                128
            } else if self.segment_bytes[i] > 0 {
                self.segment_bytes[i]
            } else {
                128
            };
            // PAGED: the manifest sizes seg2 for ONE page — that is the point, no baked pool — so
            // the POOL extent is decided here, at run time.
            if i == SEG_KV.get() && self.kv.paged && !self.seg_borrowed[i] {
                let pool = self.kv.pool_pages.0 as u64 * self.kv.page_stride_bytes;
                bytes = bytes.max(pool);
            }
            self.seg_addr[i] = Some(
                DevAddr::alloc(ByteSize(bytes), MemoryType::Tensor)
                    .map_err(|e| anyhow!("prepare: seg{i} alloc of {bytes} B failed: {e}"))?,
            );
        }
        // ⭐ AND ONE REGION PER EXTRA WEIGHT BANK, immediately after the weight segment's own —
        // banks ARE the weight segment, split because a region is capped at 16 GiB while the
        // addresses are not. Same borrowed-placeholder rule: a borrowing session aliases the owner's
        // banks (a bank is aliasable BECAUSE it holds only weights, placed identically in every
        // bundle), so allocating the full extent here would reserve tens of GB only to drop it.
        for b in 0..self.weight_bank_bytes.len() {
            let bytes = if self.seg_borrowed[SEG_WEIGHT.get()] {
                128
            } else {
                self.weight_bank_bytes[b].max(128)
            };
            self.weight_bank_addr[b] = Some(
                DevAddr::alloc(ByteSize(bytes), MemoryType::Tensor).map_err(|e| {
                    anyhow!(
                        "prepare: weight bank {} alloc of {bytes} B failed: {e}",
                        b + 1
                    )
                })?,
            );
        }
        lap("seg-alloc", &mut t0);
        // ⭐ THE REGION BASES, ONCE PER SESSION. A card HMI fault names a device address; these
        // lines are what turns that address into "which segment/bank, which offset" — chunk0's
        // (region, offset) plus the alloc's padded total, for every segment and every weight bank.
        // Read against the launch-table debug line: together they are the full addressing state the
        // launch loop computes from.
        for i in 0..NUM_SEGMENTS {
            let Some(a) = self.seg_addr[i].as_ref() else {
                continue;
            };
            debug!(
                "[sdsc-superdsc] seg-alloc seg{i}: bytes={} chunk0={:?} (padded total {})",
                self.segment_bytes[i],
                a.chunk0(),
                a.total_size().as_u64()
            );
        }
        for b in 0..self.weight_bank_bytes.len() {
            let Some(a) = self.weight_bank_addr[b].as_ref() else {
                continue;
            };
            debug!(
                "[sdsc-superdsc] seg-alloc bank{}: bytes={} chunk0={:?} (padded total {})",
                b + 1,
                self.weight_bank_bytes[b],
                a.chunk0(),
                a.total_size().as_u64()
            );
        }

        // ── FUSED TWINS ALLOCATE LAST ── They are the only allocation PAGING adds to this session,
        //    and everything above now happens in the order, and at the sizes, the unpaged path used
        //    — so the weight region lands exactly where it did before. Decode re-streams those
        //    weights every token and is bandwidth-bound on them; prefill is compute-bound at the
        //    roofline. An allocation ahead of the weights therefore taxes decode and leaves prefill
        //    alone, which is the exact shape of the gap. They are KB each, so being last is free.
        for i in 0..self.body_rungs_fused.len() {
            for c in 0..self.body_rungs_fused[i].1.len() {
                self.alloc_ops_at(ListSel::RungFused(i, c))?;
            }
        }
        for c in 0..self.body_fused.as_ref().map_or(0, PerClass::len) {
            self.alloc_ops_at(ListSel::BodyFused(c))?;
        }
        lap("fused-twin-program-h2d", &mut t0);

        // ── Convert + stage weights into seg_host — a NO-OP if the caller already ran it off the
        //    critical path. Then grow each seg_host to the device alloc's size (alignment pad,
        //    zero) so the H2D copies a full region. ──
        self.stage_weights()?;
        let reserve_only = self.bound.is_empty();
        for i in 0..NUM_SEGMENTS {
            // Same rule as the staging shadow: skip only what an alias replaces. Activation
            // segments still grow to the device extent, because this session fills them itself.
            if reserve_only && is_resident_seg(i) {
                continue;
            }
            // NO HOST MIRROR OF THE KV POOL. seg2 is written by the DEVICE (the on-card `cachewr`
            // ops) and read by the device; nothing on the host needs a copy, and sizing one to the
            // pool cost 960 MB of RSS (3.84 GB at a 1024-slot page) to mirror bytes the host never
            // looks at. Two owners of the KV is also the shape that produced the block-table bug.
            if i == SEG_KV.get() && self.kv.paged {
                continue;
            }
            let need = self.seg_addr[i]
                .as_ref()
                .map_or(0, |a| a.total_size().as_u64()) as usize;
            if self.seg_host[i].len() < need {
                self.seg_host[i].resize(need, 0);
            }
        }
        // The weight banks follow the weight segment's own rule exactly: grow to the device extent
        // unless an alias is about to replace the region.
        if !(reserve_only || self.seg_borrowed[SEG_WEIGHT.get()]) {
            for b in 0..self.weight_bank_bytes.len() {
                let need = self.weight_bank_addr[b]
                    .as_ref()
                    .map_or(0, |a| a.total_size().as_u64()) as usize;
                if self.weight_bank_host[b].len() < need {
                    self.weight_bank_host[b].resize(need, 0);
                }
            }
        }
        let n_weights = self.n_weights_staged;
        lap("weight-convert", &mut t0);

        // ── H2D every tensor-segment region ONCE (weights staged in seg1; the rest zero-init).
        //    Activations refill per predict. ──
        for i in 0..NUM_SEGMENTS {
            // BORROWED: the owner already H2D'd this region and `alias_seg` is about to point us at
            // it. DMA'ing our placeholder would be waste — and once aliased, writing our stale
            // shadow over the owner's live data would be WRONG.
            if self.seg_borrowed[i] {
                continue;
            }
            // Nothing bound ⇒ the resident segments hold no content worth sending and are about to
            // be aliased.
            if reserve_only && is_resident_seg(i) {
                continue;
            }
            // The KV pool has NO host mirror to send (see the sizing loop), so it is zeroed
            // separately and in chunks by `zero_seg`. DMA'ing from the empty vector here would be a
            // null source for a pool-sized transfer.
            if i == SEG_KV.get() && self.kv.paged {
                continue;
            }
            let (stream, addr) = (self.stream_ref()?, self.seg_addr[i].as_ref().unwrap());
            // The shadow was grown to the device extent by the loop above, so this slice is the
            // whole region — and taking it AS A SLICE is what proves the H2D cannot read past the
            // shadow. `get` rather than `[..]`: a short shadow is a wiring bug to report, not a
            // panic, and never a DMA over-read.
            let src = whole_shadow(&self.seg_host[i], addr, i)?;
            stream
                .h2d(src, addr)
                .map_err(|e| anyhow!("prepare: seg{i} H2D: {e}"))?;
        }
        // ⭐ AND EVERY EXTRA WEIGHT BANK, under the weight segment's own conditions — it is the same
        // segment, so "borrowed" and "nothing bound" mean the same thing for it.
        if !(reserve_only || self.seg_borrowed[SEG_WEIGHT.get()]) {
            for b in 0..self.weight_bank_bytes.len() {
                let (stream, addr) = (
                    self.stream_ref()?,
                    self.weight_bank_addr[b]
                        .as_ref()
                        .ok_or_else(|| anyhow!("prepare: weight bank {} has no region", b + 1))?,
                );
                let src = whole_shadow(&self.weight_bank_host[b], addr, NUM_SEGMENTS + b)?;
                stream
                    .h2d(src, addr)
                    .map_err(|e| anyhow!("prepare: weight bank {} H2D: {e}", b + 1))?;
            }
        }
        lap("tensor-h2d", &mut t0);
        self.stream_ref()?
            .synchronize()
            .map_err(|e| anyhow!("prepare: sync: {e}"))?;
        lap("sync", &mut t0);
        // ⭐ #603 PR2 2a — THE CB PARITY GATE, before the session becomes loadable: a plan twin
        // that disagrees with the walk refuses the LOAD here, never a step.
        //
        // A BORROWING session cannot be gated here: its borrowed segments still hold the
        // 128 B placeholders `mark_borrowed` left them, and a placeholder has no renderable
        // address BY DESIGN (`alias_seg_from` replaces it with the owner's region only after
        // prepare returns — rendering through it is exactly the un-aliased-read the
        // placeholder exists to poison). Such a session is gated at the alias that
        // completes its borrowed set instead; one that never completes its aliases can
        // never launch (the un-aliased placeholder guard refuses its fold), so no session
        // escapes the gate.
        if self.borrow_set_pending() {
            debug!(
                "[sdsc-cbplan] parity gate DEFERRED: borrowed segment(s) still on placeholders — \
                 it runs at the alias that completes the borrowed set"
            );
        } else {
            self.check_cb_plan_parity()?;
        }
        lap("cb-parity", &mut t0);
        self.prepared = true;

        // ── RECLAIM host memory that is DEAD after prepare (~2× the model in host RSS otherwise).
        //    The weights are resident on-device; nothing reads their host bytes again, and a
        //    segment holding ONLY weights has a dead staging buffer too (predict re-H2Ds only
        //    activation segments). Guarded by !SCRATCHY_SUPERDSC_DBG, whose readback D2Hs arbitrary
        //    segments back into seg_host. ──
        for nm in &self.weight_ids {
            self.bound.remove(nm);
        }
        if std::env::var_os("SCRATCHY_SUPERDSC_DBG").is_none() {
            let mut seg_has_activation = [false; NUM_SEGMENTS];
            for (nm, p) in &self.places {
                if !self.weight_ids.contains(nm)
                    && let Some(s) = SegIdx::checked(p.segment as i64)
                {
                    seg_has_activation[s.get()] = true;
                }
            }
            for (i, (host, has_act)) in self.seg_host.iter_mut().zip(seg_has_activation).enumerate()
            {
                // The KV shadow backs read_seg/write_seg — keep it.
                if i == SEG_KV.get() || has_act {
                    continue;
                }
                *host = Vec::new();
            }
            // ⭐ AND EVERY BANK SHADOW, unconditionally: a bank holds ONLY weights (that is what
            // makes it aliasable), so there is no activation to keep one alive and nothing reads its
            // host bytes after the H2D. Not reclaiming these is the largest host allocation in the
            // process — a bank is up to 16 GiB.
            for host in self.weight_bank_host.iter_mut() {
                *host = Vec::new();
            }
        }
        debug!(
            "[sdsc-superdsc] prepare OK — {n_weights} weight(s) placed into {NUM_SEGMENTS} packed \
             segments"
        );
        lap("reclaim", &mut t0);
        if d.prep_time {
            eprintln!(
                "[sdsc-prep-time] {:<16} {:.3}s  (laps above should sum to this; a gap is work no \
                 lap covers)",
                "TOTAL",
                start.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    /// ⭐ #603 PR2 STAGE 2a — THE PREPARE-TIME CB PARITY GATE. Every allocated launch list is
    /// rendered TWO ways — the submit walk's own enumeration (`cb_plan::walk_items`) and the data
    /// twin (`cb_plan::build_cb_plan` + `CbPlan::render`) — through ONE render door
    /// (`Stream::render_compute_batch`, the same `build_compute_cb` the submit path runs), and a
    /// differing byte REFUSES THE LOAD. This is the gate that will let 2c's player replay
    /// pre-rendered images: it proves, at prepare, that the plan's data twin of the walk is
    /// byte-identical, so no step ever needs to re-derive (or re-check) what it replays.
    ///
    /// ⭐ STAGE 2b rides along: per context, `cb_plan::patch_check` proves the IMAGE MODEL itself
    /// — canvas + affine patches ≡ a fresh walk render, at off-sample steps — under the decode
    /// player's own single-request shape, and ONLY over the body-class lists: those are the
    /// lists the player launches at a step-following `where_` (prefix/suffix launch at a
    /// constant one, so their patch table is empty by construction and their proof is the
    /// parity leg alone). A geometry too narrow to host the off-sample proof around a mid-page
    /// canvas (`canvas_room`) skips that leg loudly; every shipped bundle's page
    /// is 2048 slots.
    ///
    /// ⛔ NOT A PER-STEP CHECK — a step never pays for this. It runs at [`Self::prepare`] (every
    /// list that exists by then), at [`Self::alias_seg_from`] (a borrowing session's segments are
    /// placeholder-poisoned at its prepare, so ITS gate waits for the alias that completes the
    /// borrowed set — the rungs and the prefix-capable prefill session), and once more at
    /// [`Self::ensure_split_bodies`] (the deferred rungs, at the one event that makes them exist).
    ///
    /// Both step shapes are SYNTHETIC, not the live session's. The parity gate's: 2 requests × 3
    /// resident pages, a mid-page write slot, request 0's page 2 as the write base — the shape
    /// that exercises the batched fold (reps > 1), the slot write, the non-fold page resolution
    /// and the per-request intermediate shift, more than any single real step would, while both
    /// parity sides read the SAME fold, so the gate's meaning (twin agreement) is independent of
    /// which shape it runs under. The patch gate's: ONE request × 3 pages — under a single row
    /// the launch slot IS the position every op's KV term reads (`op_slot_pos` resolves row
    /// cursors only when the batch is wide), which is the single-request decode regime the
    /// player serves; the batched shapes are PR4's. The geometry the synthetic folds must not
    /// contradict (paged-ness, page slots, page stride) is grounded in THIS session's, because a
    /// bundle that was never baked paged has no fold ops for a paged shape to exercise.
    fn check_cb_plan_parity(&self) -> Result<()> {
        let stream = self.stream_ref()?;
        let per = self.kv.page_slots.0.max(1);
        let fold = SessionKv {
            paged: self.kv.paged,
            page_slots: fold_plan::PageSlots::per_page(per),
            page_stride_bytes: Bytes(self.kv.page_stride_bytes.max(1)),
            // Two rows of three pages each — the rows AGREE about how many they hold (a uniform
            // sweep exists) and neither row holds the other's pages (a per-request shift is real,
            // not zero).
            block_tables: vec![vec![0, 1, 2], vec![3, 4, 5]],
            request_positions: RowCursors::bound(vec![2 * per + 5; 2]),
            fold_requests: 2,
            mask_rep_stride_bytes: 0,
            mask_blocks: 64,
            fold_pages: 3,
            int_rep_stride_bytes: 65536,
        };
        // The write slot lands MID-PAGE (slot 5 of `per`), so the page a non-fold op works in and
        // the slot a fold writes are different addresses; the write base is request 0's page 2 —
        // a real block-table walk, not a zero.
        let slot = SeqPos(2 * per + 5);
        let kv_page_base = RowPage::of(&fold, RowIdx::from_launch_row(0), PageOrd::of_index(2))
            .map(|rp| fold_plan::page_base_bytes(&fold, rp))
            .unwrap_or(Bytes(0));
        let mk = |bank: usize, off: [u64; NUM_SEGMENTS]| LaunchWhere {
            byte_off: off,
            slot_pos: slot,
            kv_page_base,
            n_fold_pages: 3,
            weight_bank: bank,
        };
        // One `LaunchWhere` per distinct weight bank the launch table names (a representative
        // row's shifts — parity is about the twin agreeing, not about covering every shift
        // value), plus the prefix's and suffix's own banks. A bank with no region is SKIPPED, not
        // refused: the submit walk would refuse that launch itself, and this gate is about the
        // plan twin, not the session.
        let bank_ok = |bank: usize| {
            if bank == 0 {
                self.seg_addr[SEG_WEIGHT.get()].is_some()
            } else {
                self.weight_bank_addr
                    .get(bank - 1)
                    .is_some_and(|a| a.is_some())
            }
        };
        // The launch ROWS both gates' sessions share: one (tag, bank, shifts) per distinct
        // weight bank the session could launch, enumerated once and rendered at each session's
        // own write slot and page.
        let mut rows: Vec<(String, usize, [u64; NUM_SEGMENTS])> = Vec::new();
        match self.rolled.as_ref() {
            // Unrolled: no layer boundary to bank on — bank 0 and zero shifts are the only shape
            // the walk ever launches this bundle with.
            None => rows.push(("unrolled".into(), 0, [0; NUM_SEGMENTS])),
            Some(r) => {
                if bank_ok(r.prefix_weight_bank) {
                    rows.push((
                        format!("prefix-bank{}", r.prefix_weight_bank),
                        r.prefix_weight_bank,
                        [0; NUM_SEGMENTS],
                    ));
                }
                if bank_ok(r.suffix_weight_bank) {
                    rows.push((
                        format!("suffix-bank{}", r.suffix_weight_bank),
                        r.suffix_weight_bank,
                        [0; NUM_SEGMENTS],
                    ));
                }
                let mut seen: Vec<usize> = Vec::new();
                for &(_class, wbank, woff, kvoff) in &r.launches {
                    let bank = wbank as usize;
                    if seen.contains(&bank) || !bank_ok(bank) {
                        continue;
                    }
                    seen.push(bank);
                    let mut off = [0u64; NUM_SEGMENTS];
                    off[SEG_WEIGHT.get()] = woff;
                    off[SEG_KV.get()] = kvoff;
                    rows.push((format!("launch-row-bank{bank}"), bank, off));
                }
            }
        }
        let wheres: Vec<(String, LaunchWhere)> = rows
            .iter()
            .map(|(tag, bank, off)| (tag.clone(), mk(*bank, *off)))
            .collect();

        // ⭐ 2b — THE PATCH GATE'S SESSION: the PLAYER's shape, not the batched fold's. ONE
        // request, three resident pages. The canvas slot is `cb_plan::canvas_slot` — the ONE
        // recipe, shared with the player, that picks a slot with room on both sides for the
        // differentials and the off-sample steps: mid-page, off every slab boundary, never
        // slot 0 of a page.
        let r_patch = cb_plan::canvas_slot(per, self.kv.paged);
        let fold1 = SessionKv {
            paged: self.kv.paged,
            page_slots: fold_plan::PageSlots::per_page(per),
            page_stride_bytes: Bytes(self.kv.page_stride_bytes.max(1)),
            block_tables: vec![vec![0, 1, 2]],
            request_positions: RowCursors::bound(vec![2 * per + r_patch.unwrap_or(1)]),
            fold_requests: 1,
            mask_rep_stride_bytes: 0,
            mask_blocks: 64,
            fold_pages: 3,
            // A single request has no per-request intermediate shift to exercise — zero is
            // the decode player's own value, not a skipped dimension.
            int_rep_stride_bytes: 0,
        };
        // The canvas write base is a SYNTHETIC mid-pool page (2·stride): a
        // stand-in for the table-sourced base the player's own canvas renders
        // at, far enough from either end of the address range that no render
        // clips. Nothing probes past it anymore — the patch model has no page
        // term (the boundary step walks; see `StepScalars`), so this base is
        // only where the canvas and every off-sample render sit.
        let page_c = Bytes(2 * self.kv.page_stride_bytes.max(1));
        let mk1 = |bank: usize, off: [u64; NUM_SEGMENTS]| LaunchWhere {
            byte_off: off,
            slot_pos: SeqPos(2 * per + r_patch.unwrap_or(1)),
            kv_page_base: page_c,
            n_fold_pages: 3,
            weight_bank: bank,
        };
        // The patch gate needs one image's worth of sampling room around its canvas: the
        // `slot+65` off-sample (the slab-coefficient proof) must land strictly inside the
        // image's page span. `canvas_room` is the ONE law, shared with the player — a page
        // too narrow for it (or one no canvas slot fits at all) skips the leg loudly; every
        // shipped bundle's page is 2048 slots.
        let patch_room = r_patch.is_some_and(|r| cb_plan::canvas_room(per, self.kv.paged, r));
        if !patch_room {
            debug!(
                "[sdsc-cbplan] patch gate SKIPPED: no canvas slot with sampling room fits a \
                 {per}-slot page. The parity gate above still ran."
            );
        }
        // Every list the session could launch — the class dimension is enumerated whole here, so
        // a where need not be matched to the classes its rows name (parity reads the same ops
        // either way). A list whose programs are not allocated yet (the deferred rungs) is
        // skipped here and gated at its allocation instead.
        let mut sels: Vec<ListSel> = vec![ListSel::Prefix, ListSel::Suffix];
        for c in 0..self.body_ops.len() {
            sels.push(ListSel::Body(c));
        }
        for c in 0..self.body_fused.as_ref().map_or(0, PerClass::len) {
            sels.push(ListSel::BodyFused(c));
        }
        for i in 0..self.body_rungs.len() {
            for c in 0..self.body_rungs[i].1.len() {
                sels.push(ListSel::Rung(i, c));
            }
        }
        for i in 0..self.body_rungs_fused.len() {
            for c in 0..self.body_rungs_fused[i].1.len() {
                sels.push(ListSel::RungFused(i, c));
            }
        }
        let mut cbs = 0usize;
        let mut contexts = 0usize;
        let mut sites = 0usize;
        let mut patch_contexts = 0usize;
        for sel in sels {
            let ops = self.list(sel);
            if ops.is_empty() || !ops.iter().all(|o| o.addr.is_some()) {
                continue;
            }
            for (tag, where_) in &wheres {
                let label = format!("{sel:?}/{tag}");
                cbs += cb_plan::parity_check(
                    ops,
                    &fold,
                    &self.seg_addr,
                    &self.weight_bank_addr,
                    where_,
                    &label,
                    stream,
                )?;
                contexts += 1;
            }
            // ⭐ THE PLAYER'S ROW LAW, read here as the gate's scope: only the
            // four body-class lists are ever launched at a step-following
            // `where_` — prefix/suffix launch at a constant one (see
            // `derive_step_image`), so their patch table is empty by
            // construction and there is nothing for this leg to prove.
            // Probing them at this leg's mid-pool `where_` measures a
            // counterfactual no replay executes (measured on the first real
            // bundle: the page differential's render returns a structurally
            // different allocation and the affine solve refuses).
            let step_following = !matches!(sel, ListSel::Prefix | ListSel::Suffix);
            if patch_room && step_following {
                for (tag, bank, off) in &rows {
                    let label = format!("{sel:?}/{tag}");
                    let (row_sites, _) = cb_plan::patch_check(
                        ops,
                        &fold1,
                        &self.seg_addr,
                        &self.weight_bank_addr,
                        &mk1(*bank, *off),
                        &label,
                        stream,
                    )?;
                    sites += row_sites;
                    patch_contexts += 1;
                }
            }
        }
        if contexts == 0 {
            // The gate may not pass vacuously: prepare allocates the prefix and suffix lists
            // unconditionally and bank 0's region exists by the time this runs, so zero
            // checkable contexts means the enumeration itself is broken — refuse rather than
            // "prove" nothing. Both legs share this enumeration, so it covers the patch gate
            // too (they cannot disagree about which (list × row) pairs exist).
            bail!(
                "[cbplan] parity gate REFUSED: no (list × launch-row) context was checkable — \
                 every bank was skipped or no list is allocated"
            );
        }
        if patch_room && patch_contexts == 0 {
            // Not a refusal: the rungs allocate lazily, so a session whose every body-class
            // list is still deferred legitimately has nothing to patch-prove here — each
            // rung runs this same gate at its own allocation. Said loudly because a silent
            // zero is indistinguishable from a broken enumeration.
            debug!(
                "[sdsc-cbplan] patch gate verified ZERO images: no body-class list is allocated \
                 yet (deferred rungs gate at their own allocation)"
            );
        }
        debug!(
            "[sdsc-cbplan] parity gate: {cbs} CB(s) proven byte-identical across {contexts} \
             (list × launch-row) context(s); patch gate: {sites} moving word(s) verified \
             off-sample across {patch_contexts} image(s)"
        );
        Ok(())
    }

    fn stream_ref(&self) -> Result<&Stream> {
        self.stream
            .as_ref()
            .ok_or_else(|| anyhow!("no stream — prepare was not called"))
    }

    /// Give every op in one list its own Program-segment allocation and upload its device binary.
    fn alloc_ops_at(&mut self, sel: ListSel) -> Result<()> {
        // The list is taken out so the stream (also `&self`) can be borrowed alongside it.
        let mut ops = std::mem::take(self.list_mut(sel));
        // Heap copies of any `.rodata`-backed program bytes, kept alive until the async H2Ds have
        // been synchronized at the end of the loop. See the `Cow` match below for why.
        let mut pinnable: Vec<Vec<u8>> = Vec::new();
        let r = (|| -> Result<()> {
            for op in &mut ops {
                // ⛔⛔⛔ THE IMAGE IS NOT ALWAYS AT THE HEAD OF ITS ALLOCATION. dxp's
                // `InitTransfer.dev_ptr` — which `parse_spyrecode` has REFUSED unless it equals
                // `job_bin_ptr` — can name an address above `PROG_OFFSET_BASE`, reserving a prologue
                // it does not fill. So `head` is that prologue, and it is the SAME number the launch
                // passes as the bootstrap offset below: the device starts where the image starts
                // because both come from `job_bin_ptr`, not because both happen to be zero.
                //
                // ⛔ IT WAS ZERO FOR EVERY PROGRAM THAT HAS EVER SHIPPED, which is why uploading at
                // offset 0 worked until the batched-decode collapse produced the first group with a
                // non-zero one — and then failed as `syndrome=0xc00 [PrepZeroFlitCnt, PrepSwVer]`,
                // `job_count=0`, at exactly `PROG_OFFSET_BASE + head`: the device read the image's own
                // flit `head/128` as its first job header. The image content was byte-identical to the
                // bake (D2H'd and diffed, all 8 and all 20 programs of both faulting bundles); only
                // its placement was wrong.
                //
                // ⭐ AND IT IS ONE VALUE, NOT A SUBTRACTION REPEATED AT THE LAUNCH. `prog_image` is the
                // only place `job_bin_ptr` becomes an offset; the bootstrap the launch passes is
                // `bootstrap()` of this same value, so the upload destination and the start address
                // cannot be computed from different arithmetic (this site used `checked_sub` + a refusal
                // while the launch used a bare `-`).
                let img = op
                    .code
                    .prog_image(crate::runtime::prog_offset_base())
                    .ok_or_else(|| {
                        anyhow!(
                            "program job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x}",
                            op.code.job_bin_ptr,
                            crate::runtime::prog_offset_base()
                        )
                    })?;
                let head = img.bootstrap();
                let addr = DevAddr::alloc(ByteSize(img.alloc_bytes()), MemoryType::Program)
                    .map_err(|e| {
                        anyhow!(
                            "program alloc of {} B ({head} B prologue + {} B image) failed: {e}",
                            img.alloc_bytes(),
                            img.image_bytes()
                        )
                    })?;
                // ⛔⛔⛔ DMA A PROGRAM OUT OF ANONYMOUS MEMORY, NEVER OUT OF THE EXECUTABLE.
                //
                // A BAKED program is `Cow::Borrowed(&'static [u8])`, which lives in the executable's
                // read-only FILE-BACKED mapping — the failing pin's `hmva` sat inside
                // `r--p 00000000 … /target/release/scr`. Pinning THAT for DMA returned `EFAULT`,
                // which senlib reports as RAS `0xad15` "Unable to map DMA to the device address" — a
                // message about the DEVICE address for a fault on the HOST side. The stream was
                // poisoned, the rest of that rung's programs were rejected, `destroy_stream`
                // swallowed the residue while draining, and granite-3.1-8b lost prefill rung m=11
                // out of its ladder.
                //
                // ⚠️ WHAT IS *NOT* ESTABLISHED: that pinning `.rodata` always fails. It plainly does
                // not — the `debug!` below counts EVERY program as `Cow::Borrowed` (3 of 3, 66 of
                // 66, every list), so this path DMA'd hundreds of programs out of the executable and
                // exactly one pin failed, deterministically, on the 8b and never on the 2b. Why that
                // one is open. What is certain is that the whole class is avoidable for the price of
                // a load-time copy, so the fix does not depend on the answer.
                //
                // The `Cow` variant IS the provenance: `Owned` is a heap `Vec` (anonymous, private,
                // always pinnable); `Borrowed` is the executable's pages. So copy exactly the
                // borrowed case. The staging buffers must outlive the ASYNC transfer, hence the sync
                // below.
                let staged: &[u8] = match &op.code.init_binary {
                    std::borrow::Cow::Owned(v) => v,
                    std::borrow::Cow::Borrowed(b) => {
                        pinnable.push(b.to_vec());
                        pinnable.last().expect("just pushed")
                    }
                };
                // ⛔ THE PROGRAM'S OWN BYTES, NOT THE REGION'S PADDED EXTENT. `DevAddr::alloc`
                // rounds up, so `addr.total_size()` can exceed the binary — sending that many bytes
                // reads past its end. The pad needs no content: nothing executes past the program.
                let dst = addr
                    .window(ByteOffset(head), ByteSize(op.code.init_binary.len() as u64))
                    .ok_or_else(|| anyhow!("program window at {head} B unaddressable"))?;
                self.stream_ref()?
                    .h2d(staged, &dst)
                    .map_err(|e| anyhow!("program H2D: {e}"))?;
                op.addr = Some(addr);
            }
            // ⛔ BEFORE `pinnable` DROPS. The H2Ds above are asynchronous, so the staging buffers
            // must still exist when the DMA reads them. Programs are uploaded once per list at
            // load, so this sync costs nothing per step.
            if !pinnable.is_empty() {
                self.stream_ref()?
                    .synchronize()
                    .map_err(|e| anyhow!("program H2D sync: {e}"))?;
            }
            // How much of a list is `.rodata`-backed says how much of the program set the copy above
            // is load-bearing for — 1-of-many and all-of-many are different stories about the bake,
            // and the difference is invisible without counting.
            debug!(
                "[sdsc-superdsc] program H2D {sel:?}: {} of {} op(s) staged through a heap copy \
                 (Cow::Borrowed ⇒ possibly .rodata ⇒ unpinnable for DMA)",
                pinnable.len(),
                ops.len(),
            );
            Ok(())
        })();
        *self.list_mut(sel) = ops;
        r
    }

    fn list_mut(&mut self, sel: ListSel) -> &mut Ops {
        match sel {
            ListSel::Prefix => &mut self.prefix_ops,
            ListSel::Body(c) => &mut self.body_ops[c],
            ListSel::Suffix => &mut self.suffix_ops,
            ListSel::BodyFused(c) => self
                .body_fused
                .as_mut()
                .expect("BodyFused is selected only when every class has its twin")
                .class_mut(c),
            ListSel::Rung(i, c) => &mut self.body_rungs[i].1[c],
            ListSel::RungFused(i, c) => self.body_rungs_fused[i].1.class_mut(c),
        }
    }

    /// The same list, read-only — for the per-body facts [`StepBody`] carries, which are read off the
    /// body's own OPS.
    fn list(&self, sel: ListSel) -> &Ops {
        match sel {
            ListSel::Prefix => &self.prefix_ops,
            ListSel::Body(c) => &self.body_ops[c],
            ListSel::Suffix => &self.suffix_ops,
            ListSel::BodyFused(c) => self
                .body_fused
                .as_ref()
                .expect("BodyFused is selected only when every class has its twin")
                .class(c),
            ListSel::Rung(i, c) => &self.body_rungs[i].1[c],
            ListSel::RungFused(i, c) => self.body_rungs_fused[i].1.class(c),
        }
    }

    /// ⭐⭐⭐⭐⭐ **THE BODY THIS STEP WILL RUN**, and the only way anyone gets one.
    ///
    /// ⛔⛔⛔ IT TAKES THE WRITE SLOT AND NOTHING ELSE, WHICH IS THE WHOLE POINT. The selector's two
    /// inputs are `valid_len` and `n_fold_pages`, and BOTH are functions of the slot this step writes —
    /// so if a caller could pass them it could pass a pair the launch will not use, and then the host's
    /// index table, the host's mask FORM and the launched descriptor would be about different bodies.
    /// Derived here, from the one value the host also hands to `run_step` as `start`, there is no second
    /// derivation to drift.
    ///
    /// ⛔ AND THIS IS THE THIRD TIME THE HOST ASKED A PER-BODY QUESTION OF THE WRONG BUNDLE.
    /// `bake_gathers_kv` asked the m=1 decode bucket (`8f03f9974`), the gather scratch's window count
    /// came off the CEILING body while an interior rung was launched (`49a99fd45`), and the swept extent
    /// the unswept-slot guard used was the ceiling's too. All three had the same shape: a fact that
    /// differs per BODY, read from something that is per BUNDLE. The fix is not a third careful call
    /// site — it is that `DecodeRung` no longer HAS a `swept` or a `gathers_kv` field to read, and the
    /// only value that answers either question is minted here, by the selector, from the selector's own
    /// answer.
    pub fn step_body(&mut self, seq_pos: SeqPos) -> Result<StepBody> {
        let n_fold_pages = self.n_fold_pages(seq_pos);
        // `valid_len = seq_pos + 1` — positions [0..seq_pos] are filled, including the token written to
        // KV before attention. Spelled ONCE, here, for both readers of the selection.
        let sel = self.select_body_paged(seq_pos.0 + 1, n_fold_pages)?;
        // ⭐ EVERY CLASS'S OWN OPS ANSWER IT, folded to the strongest claim: the gather scratch must
        // satisfy every class that reads through one, and the unswept-slot guard must use the
        // NARROWEST sweep any class runs (a rung's extent is shared by every class, so this is the
        // same number for each — but reading it per class keeps the fact with the ops that carry
        // it). Same discipline as `latch_paged_geometry` reading `page_slots` off the ops: a bake
        // fact rides on the launch group that carries it (`bundle::KvShifts::gathered`), so a body
        // that gathers cannot be driven as one that does not, or the reverse.
        let mut gathers_any = false;
        // Same authority, same reason: whether this body re-launches a page fold is the
        // `page_fold` its own launch groups carry (what `fold_plan::reps` multiplies by) — any
        // class's fold makes the step a fold, not a chunk.
        let mut folds_any = false;
        // ⭐ THE STAGING CHANNEL, folded from the same launch groups: whether this body's
        // gathered table is the FILL-STAGED REGION FORM. A region-form body answers
        // `gathers` YES (absolute entries, zero shift) but its table is already staged by
        // the host fill — the composer must not drive it as a chunk body, and no
        // KV-block-index step is minted for it. Folded strongest-claim like `gathers_any`:
        // any class's region-form group makes the step's page-list answer false.
        let mut region_form_any = false;
        for c in 0..self.n_classes() {
            let ls = self.list_sel_for(sel, c);
            gathers_any |= crate::wiring::GathersKv::of_launch_groups(
                self.list(ls).iter().map(|o| o.kv.gathered),
            )
            .get();
            folds_any |= self.list(ls).iter().any(|o| o.kv.page_fold);
            region_form_any |= self.list(ls).iter().any(|o| o.kv.region_form);
        }
        Ok(StepBody {
            sel,
            swept: self.body_swept(sel),
            gathers: crate::wiring::GathersKv::of_launch_groups([gathers_any]),
            folds: folds_any,
            region_form: gathers_any && region_form_any,
            // The rung's op-local pin slots, the same layout's baked sum — every bundle of the
            // rung carries the same number, so the body's own artifact is as good a seat as any.
            lx_op_local_bytes: self.code.layout.lx_op_local_bytes,
            // The chunk form's probability-part pins, the same layout's recorded term — the
            // admission summed it beside the gathers, so the host's read must carry it too.
            lx_prob_pin_bytes: self.code.layout.lx_prob_pin_bytes,
        })
    }

    /// How many attention classes this session carries (one body each). 1 for every single-class
    /// model — the case every existing rolled bundle was.
    fn n_classes(&self) -> usize {
        self.rolled.as_ref().map_or(1, |r| r.n_bodies).max(1)
    }

    /// THIS selection's swept extent — its own ladder rung's `active_cap`, which every class of a
    /// rung shares (a rung IS one swept extent; the classes are the programs baked for it).
    ///
    /// The ladderless arms (`Body`/`BodyFused`: an UNROLLED bundle, which has no `rungs` and therefore
    /// no ladder) sweep their whole baked capacity, and on the paged path that is one page — the number
    /// the ops themselves carry. It is the value the host's guard used for every bundle before the
    /// ladder's own extent was plumbed through, so a ladderless bundle behaves exactly as it did.
    fn body_swept(&self, sel: BodySel) -> SweptCols {
        match sel {
            BodySel::Rung(i) => self.body_rungs[i].0,
            BodySel::RungFused(i) => self.body_rungs_fused[i].0,
            _ => SweptCols::new(self.kv.page_slots.0.max(0) as u32),
        }
    }

    /// One `ListSel` for every class of a body selection — the class dimension is the launch
    /// table's to supply, not the selector's: a rung is ONE swept extent shared by every class.
    fn list_sel_for(&self, sel: BodySel, class: usize) -> ListSel {
        match sel {
            BodySel::Body => ListSel::Body(class),
            BodySel::BodyFused => ListSel::BodyFused(class),
            BodySel::Rung(i) => ListSel::Rung(i, class),
            BodySel::RungFused(i) => ListSel::RungFused(i, class),
        }
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Body selection
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Program memory for the split ladder, allocated on first need. Idempotent, and a no-op for a
    /// session whose bodies were allocated at prepare.
    ///
    /// ⛔ ALLOCATING HERE MAKES SELECTION AND ALLOCATION THE SAME EVENT. The gate used to live at
    /// ONE of the two call sites and re-spell the selector's rule, which is how a split body came to
    /// be selected and never allocated: the launch took a null device address
    /// (RAS::RUNTIMEOPERATION::NullArgument) and the whole batch returned no output. Twice, for two
    /// different reasons. A caller cannot obtain a split body that is not backed.
    fn ensure_split_bodies(&mut self) -> Result<()> {
        if self.split_bodies_ready || self.body_rungs.is_empty() {
            return Ok(());
        }
        for i in 0..self.body_rungs.len() {
            for c in 0..self.body_rungs[i].1.len() {
                self.alloc_ops_at(ListSel::Rung(i, c))?;
            }
        }
        // Their binaries must land before the first launch.
        self.stream_ref()?
            .synchronize()
            .map_err(|e| anyhow!("split-body sync: {e}"))?;
        self.split_bodies_ready = true;
        // ⭐ #603 PR2 2a — the DEFERRED rungs exist now, so the CB parity gate runs at this, the
        // one event that made them (never per step; the lists prepare already saw were proven
        // there and are merely re-proven here, once).
        self.check_cb_plan_parity()?;
        debug!(
            "[sdsc-superdsc] allocated {} split decode bodies (a fold that re-launches needs them)",
            self.body_rungs.len()
        );
        Ok(())
    }

    /// Select the SPLIT decode body for a valid KV length of `valid_len` positions: the smallest
    /// rung whose `active_cap` ≥ it, which bounds the O(cap) attention sweep. Every return is a
    /// split body, so this is also where they are guaranteed to EXIST.
    fn select_body(&mut self, valid_len: i64) -> Result<BodySel> {
        self.ensure_split_bodies()?;
        let reqs = self.fold_requests_or_1();
        if self.body_rungs.is_empty() {
            // The ladderless single body — traced for the same reason as the fused one: an untraced
            // return is indistinguishable from a step that never reached the selector at all.
            self.trace_body("split-noladder", valid_len, -1, reqs, 0, false);
            return Ok(BodySel::Body);
        }
        for (i, (swept, _)) in self.body_rungs.iter().enumerate() {
            // ⛔ `SweptCols::covers` AND NOT `>=` ON BARE INTEGERS. The sibling ladder is keyed by the
            // batch WIDTH, so an operand-swapped comparison — or one against the wrong ladder — used to
            // type-check and return a body baked for a 64-column sweep because four requests are live.
            if swept.covers(valid_len.max(0) as u64) {
                self.trace_body("split", valid_len, -1, reqs, swept.get() as i64, false);
                return Ok(BodySel::Rung(i));
            }
        }
        // ⚠️ THE FALLBACK, VISIBLE — AND BAKED. A `valid_len` past every rung's sweep runs the rung the
        // bake PROVED serves any length (`RerollMeta::fallback_rung`, never a split-KV chunk — see
        // `fold_plan::the_fallback_serves_any_length`). Chosen here, it was the top, which #438 made the
        // widest chunk, and every granite-2b step past 16,384 tokens was refused.
        let top = self
            .code
            .reroll
            .as_ref()
            .map(|m| m.fallback_rung as usize)
            .filter(|&at| at < self.body_rungs.len())
            .ok_or_else(|| {
                anyhow!(
                    "bundle {}: no baked fallback rung inside its {}-rung ladder — the bake proves one \
                     for every ladder it emits, so this artifact is not one it emitted",
                    self.code.fp,
                    self.body_rungs.len()
                )
            })?;
        let cap = self.body_rungs[top].0.get() as i64;
        self.trace_body("split-CEILING", valid_len, -1, reqs, cap, false);
        Ok(BodySel::Rung(top))
    }

    /// Pick the body for this step: the sk_bucket rung by live length, and — when the fold would
    /// run exactly ONE pass — the FOLD-FUSED variant, which carries the baseline's launch count.
    ///
    /// ⛔ THE FUSED BODY HAS NO FOLD (`trip_kinds_for` rewrites `PageFold` to `Pure` for the fused
    /// variant), so `reps` collapses to 1. That is correct for ONE request and WRONG for a batch:
    /// `reps = n_fold_pages * fold_requests`, and asking only about PAGES picked the fused body for
    /// a short context with N requests, ran the prefix fold once — the (request 0, page 0) pass —
    /// and left every other request attending NO resident prefix. Request 0 looked fine, everyone
    /// else produced garbage from their second token.
    fn select_body_paged(&mut self, valid_len: i64, n_fold_pages: i64) -> Result<BodySel> {
        let reqs = self.fold_requests_or_1();
        let single_page = !self.kv.paged || (n_fold_pages <= 1 && reqs <= 1);
        if single_page {
            if !self.body_rungs_fused.is_empty() {
                for (i, (swept, _)) in self.body_rungs_fused.iter().enumerate() {
                    if swept.covers(valid_len.max(0) as u64) {
                        self.trace_body(
                            "fused",
                            valid_len,
                            n_fold_pages,
                            reqs,
                            swept.get() as i64,
                            true,
                        );
                        return Ok(BodySel::RungFused(i));
                    }
                }
                // ⛔ NO FUSED RUNG COVERS THIS CONTEXT — fall through to the SPLIT ladder, do NOT
                // return the widest fused rung: that hands back a body whose sweep is NARROWER than
                // the live context and silently DROPS every key past it, answering from a prefix
                // that stops early, fluently and with no fault. The split ladder is the right
                // destination because it can FOLD, so a context past the widest single-launch sweep
                // is served by re-launching rather than by pretending it is shorter.
                self.trace_body("fused-none-fits", valid_len, n_fold_pages, reqs, 0, false);
                return self.select_body(valid_len);
            }
            // The ladderless fused bodies — present only when every class has its twin.
            if self.body_fused.is_some() {
                self.trace_body("fused-noladder", valid_len, n_fold_pages, reqs, 0, true);
                return Ok(BodySel::BodyFused);
            }
        }
        self.select_body(valid_len)
    }

    /// How many requests the fold sweeps, per `fold_plan` (which owns the page maps). 1 unbatched.
    fn fold_requests_or_1(&self) -> i64 {
        fold_plan::fold_requests(&self.fold).max(1)
    }

    /// WHICH BODY A STEP ACTUALLY RUNS — printed, because it was unobservable and that cost a day.
    /// `fold_reqs` is printed because it is the value the fused/split decision turns on: a ragged
    /// bs=4 batch came back with row 0 byte-identical to a solo run and rows 1..3 wrong — the exact
    /// fused-body signature — and the trace could not say whether the batch STEP had selected fused.
    fn trace_body(
        &self,
        variant: &str,
        valid_len: i64,
        pages: i64,
        reqs: i64,
        rung: i64,
        fused: bool,
    ) {
        if !Diag::get().body_trace {
            return;
        }
        eprintln!(
            "[body-choice] {variant} state={:p} valid_len={valid_len} n_fold_pages={pages} \
             fold_reqs={reqs} -> rung={rung}{}",
            &self.fold as *const SessionKv,
            if fused { " (fold-fused)" } else { " (split)" }
        );
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  The launch loop
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Launch each group of `sel` IN ORDER against the shared resident segments
    /// (positional `tensor_allocs[i]` = segment i's region); `byte_off` is the per-segment offset
    /// applied to every op (zero for prefix/suffix; `v·stride` for body layer v). Tensors stay
    /// resident — a producer op's output and the consumer's input resolve to the SAME
    /// segment+offset (one global placement per tensor), so threading is automatic.
    fn launch_ops(
        &mut self,
        sel: ListSel,
        byte_off: [u64; NUM_SEGMENTS],
        slot_pos: SeqPos,
        kv_page_base: Bytes,
        n_fold_pages: i64,
        // ⭐ WHICH WEIGHT BANK backs positional slot 1 for these launches. 0 for every unbanked
        // bundle, which is what makes banking invisible to every model whose weights fit one region.
        weight_bank: usize,
    ) -> Result<()> {
        let d = Diag::get();
        // The list is taken out so `&self` stays free for the stream and the fold state; every op
        // is read-only here, which is the point of the C++ `const std::vector&` — this function
        // runs once per forward per REQUEST, so anything written back through it is state one
        // request leaves for the next.
        let ops = std::mem::take(self.list_mut(sel));
        let r = self.launch_ops_inner(
            &ops,
            LaunchWhere {
                byte_off,
                slot_pos,
                kv_page_base,
                n_fold_pages,
                weight_bank,
            },
            d,
        );
        *self.list_mut(sel) = ops;
        r
    }

    /// LEG-1(c): the `[cachewr]` line — the first slot-write op's aim ingredients plus the
    /// DISTINCT target pages of every slot-write in the rung, with the write-request's own
    /// block-table row. `None` when the rung has no cache write or its page sweep is refused
    /// (the walk's own refusal follows). The aim is the SAME `fold_plan::seg_deltas` the walk
    /// calls, never a re-derivation.
    fn cachewr_aim_line(
        &self,
        ops: &Ops,
        where_: &LaunchWhere,
        byte_off: [u64; NUM_SEGMENTS],
        n_fold_pages: i64,
    ) -> Option<((i64, Vec<i64>), String)> {
        let op0 = ops.iter().find(|o| o.kv.slot_write)?;
        let per = fold_plan::UniformPages::sweeping(n_fold_pages, &self.fold)?;
        let launch = SlotPos::of_launch(where_.slot_pos.0);
        let stride = self.fold.page_stride_bytes.0;
        let mut pages: Vec<i64> = Vec::new();
        let mut first = String::new();
        for op in ops.iter().filter(|o| o.kv.slot_write) {
            // The geometry mint, per op — each op carries its own baked modulus. A
            // disagreement yields no line (the walk's own refusal follows, naming the op).
            let Ok(geo) = fold_plan::PageGeometry::agree(&op.kv, &self.fold) else {
                return None;
            };
            let Some(delta) =
                fold_plan::seg_deltas(&op.kv, &self.fold, geo, 0, per, launch, where_.kv_page_base)
                    .deltas()
            else {
                continue;
            };
            let abs = byte_off[SEG_KV.get()] + delta.kv.0;
            // ⛔ NOT `+ where_.kv_page_base`: `nonfold_page_delta` already RETURNS the caller's
            // base as the delta (fold_plan.rs `Some(caller_base)` — the solo arm), so the walk
            // composes the base exactly ONCE (`off = byte_off; off[seg] += delta.kv`). Adding
            // `kv_page_base` on top double-counted it and printed page 2 for a page-1 write —
            // chatG's pg1/pg2 windows caught the instrument, not the launch (pg2 never written).
            let pg = if stride > 0 {
                (abs / stride) as i64
            } else {
                -1
            };
            if !pages.contains(&pg) {
                pages.push(pg);
            }
            if first.is_empty() {
                use std::fmt::Write as _;
                let _ = write!(
                    first,
                    "op {} ({}): delta_kv={} byte_off_kv={} abs_in_seg={} page={pg}",
                    op.index,
                    op.label(),
                    delta.kv.0,
                    byte_off[SEG_KV.get()],
                    abs
                );
            }
        }
        if first.is_empty() {
            return None;
        }
        let row = self
            .fold
            .block_tables
            .get(op0.kv.request as usize)
            .cloned()
            .unwrap_or_default();
        let text = format!(
            "[cachewr] slot_pos={} kv_page_base={} page_stride={} row_pages={row:?} \
             distinct_target_pages={pages:?} first: {first}",
            where_.slot_pos.0, where_.kv_page_base.0, stride
        );
        Some(((where_.slot_pos.0, pages), text))
    }

    fn launch_ops_inner(&mut self, ops: &Ops, where_: LaunchWhere, d: &Diag) -> Result<()> {
        // `slot_pos`/`kv_page_base`/`weight_bank` are read only inside
        // `cb_plan::launch_site`, through `where_` — nothing else in the walk uses them.
        let LaunchWhere {
            byte_off,
            n_fold_pages,
            ..
        } = where_;
        // LEG-1(c) under `segaddr`: where this rung's cache writes AIM — one line per step (the
        // rolled body re-launches per layer, so the line is keyed on the write slot + the DISTINCT
        // target pages and only prints when that key moves). The aim is the SAME `seg_deltas` the
        // walk calls, never a re-derivation: the page each slot-write lands in is
        // `(byte_off_kv + delta.kv) / page_stride` — the caller's page base arrives ONCE, inside
        // the delta (`nonfold_page_delta` returns it), never as a separate addend.
        if d.segaddr {
            if let Some((key, text)) = self.cachewr_aim_line(ops, &where_, byte_off, n_fold_pages) {
                if self.cachewr_seen.as_ref() != Some(&key) {
                    self.cachewr_seen = Some(key);
                    eprintln!("{text}");
                }
            }
        }
        // 🛑 THE DRAIN AT THE START IS WHAT MAKES THE LADDER MEAN ANYTHING. A launch is async and
        // the stream is in-order, so at any moment the device is a whole layer behind the host.
        // Timing from an un-drained start measures `backlog + ops[0..j]`, and the backlog shrinks by
        // exactly the ops the rung adds — so every rung after the first read the SAME number, and
        // one came out NEGATIVE. Draining first costs one settle per layer instead of one per group.
        if d.prefix_time {
            let _ = self.stream_ref()?.synchronize();
        }
        let px0 = Instant::now();
        let px_target: Option<usize> = (d.prefix_time && !ops.is_empty()).then(|| {
            let mut t = timers();
            let c = t.prefix_call.entry(ops.len()).or_insert(0);
            let target = (*c as usize) % ops.len();
            *c += 1;
            target
        });

        // ⛔ NO KNOB OF ITS OWN — THE BATCH IS THE PATH. Every launch this rung would make is
        // collected below and paid as ONE doorbell at the end (the vendor's multi-CB branch;
        // `Stream::compute_batch`). The five knobs that sync or submit MID-RUNG cannot see a
        // pending batch — a drain between ops of an unflushed rung observes nothing — so each of
        // them falls back to this function's per-op submission, bit for bit: `repeat_probe`
        // (double-submits per op), `perop_sync`/`group_time` (drain per op), `prefix_time`
        // (drains at a rotating MID-RUNG group), `optrace` (drains + D2H after each op).
        // `prog_verify`/`segaddr`/`launch_mark`/`submit_time` are batch-safe: they read program
        // memory or host state this rung's compute ops never write, and `submit_time` measures
        // the flush itself as the enqueue.
        let collect =
            !(d.repeat_probe || d.perop_sync || d.group_time || d.prefix_time || d.optrace);
        let mut pending: Vec<PendingLaunch> = Vec::new();
        // The OR of every entry's per-op barrier — one head barrier on the batch is equivalent
        // to per-op barriers for an all-compute list (flex-rs drains the OTHER pipelines'
        // fences on the first true barrier and waits on nothing after; the ring stays FIFO).
        let mut batch_barrier = false;

        for (oi, op) in ops.iter().enumerate() {
            // ── PAGE FOLD ── Run this group once per page of resident prefix, rebased each time to
            //    that page's KV and that page's validity row. The fold ACCUMULATES into the running
            //    softmax state, so repeating it IS the loop. Zero passes when nothing is resident
            //    yet — the common case for a first prefill chunk, which attends only itself and
            //    would otherwise sweep an empty page. This is what makes context a RUNTIME page
            //    count instead of a baked extent. Whose KV an op touches and how many passes it
            //    takes are both `fold_plan`'s to answer — and the refusal for a count the pool
            //    cannot honour is `cb_plan::op_reps`'s, shared with the plan build so the two
            //    cannot disagree about how many passes an op takes.
            let reps = cb_plan::op_reps(&op.kv, &self.fold, n_fold_pages, oi)?;

            for rep in 0..reps {
                let stream = self.stream_ref()?;
                // ⭐ ONE (op, PASS)'S ARITHMETIC, SINGLE-SOURCED. Everything from the sweep
                //    witness through the fold deltas, the shifted tensor bases, the weight bank,
                //    the program, the host-fence flag and the bootstrap is `cb_plan::launch_site`
                //    — the submit walk below, the diag printers it feeds, and the CB-plan parity
                //    gate (which proves the plan's data twin of exactly this computation
                //    byte-identical at prepare) all read the ONE function, so no second
                //    derivation of the addressing can drift. The law comments that lived on the
                //    arithmetic moved with it.
                // ⛔ A PAGE PART WITH NO PAGE DOES NOT LAUNCH — a short last chunk's padding past
                //    the request's pages (#620). Its rows are never read, and the only base left
                //    to aim it at is the cursor's own page, which is part 0's keys; the skip
                //    lives in `launch_site`, shared by the walk and the plan twin.
                let Some(site) = cb_plan::launch_site(
                    ops,
                    oi,
                    rep,
                    &self.fold,
                    &self.seg_addr,
                    &self.weight_bank_addr,
                    &where_,
                )?
                else {
                    continue;
                };
                // The walk's own locals, re-bound from the site: `ta` as the borrows the submit
                // and dump paths take, everything else by copy or borrow.
                let ta: Vec<&DevAddr> = site.ta.iter().collect();
                let (off, prog, bootstrap, barrier) =
                    (site.off, &site.prog, site.bootstrap, site.barrier);
                let gt0 = Instant::now();
                if d.launch_mark {
                    // ⛔ ONE WRITE, and it must reach the fd BEFORE the submit: an abort from senlib's
                    // monitor thread takes the process down with no unwinding, so a buffered line is a
                    // line that never existed. Same single-`format!` discipline as `[progverify]`.
                    // ⭐ THE SEGMENT SHIFTS RIDE THE MARK. A gathered fold's index lives in seg3 and
                    // moves ONLY by `delta.mask`, so "did the host shift pass p's index" is answered
                    // by this line and nothing else — `[gather]` prints the TABLE, not the base.
                    let one = format!(
                        "[mark] op[{}] rep={rep} seg0+{} seg2+{} seg3+{} gathered={} pf={} {}\n",
                        op.index,
                        off[SEG_INTERMEDIATE.get()],
                        off[SEG_KV.get()],
                        off[SEG_MASK.get()],
                        op.kv.gathered,
                        op.kv.page_fold,
                        op.label()
                    );
                    eprint!("{one}");
                }

                if d.segaddr
                    && oi == 0
                    && byte_off[SEG_WEIGHT.get()] == 0
                    && byte_off[SEG_KV.get()] == 0
                {
                    self.dump_seg_addrs(&ta, op, prog, bootstrap, &off);
                }
                // ⛔ EVERY FAULTING LAUNCH NAMES ITS OWN AIM. The one-shot dump above describes op[0]
                // of whichever list launched FIRST, which is a different bundle from the one that
                // faults — and that mis-attribution cost a session: `job_bin_ptr=0x1c00000000` was
                // read off a prefill op and carried onto a decode fault whose bootstrap nobody had
                // measured. A fault path is not a mode, so this has no knob.
                let fault = |e: HwError| -> anyhow::Error {
                    self.dump_launch_site("FAULT(submit-refused)", &ta, op, prog, bootstrap, &off);
                    let sweep = self.prog_images_sweep("FAULT", ops);
                    eprint!("{sweep}");
                    launch_err(e)
                };
                // ⭐ DATES THE DIVERGENCE. The fault path's readback runs after the abend, on a stream
                // that may no longer answer; this one runs on a healthy stream immediately before the
                // launch, so a program that already reads back wrong here was clobbered by its upload
                // or by an EARLIER launch rather than by its own. Silent when the image matches —
                // every launch printing "OK" would bury the one that does not.
                if d.prog_verify {
                    // ⭐ THE WHOLE LIST BEFORE ITS FIRST LAUNCH — on a HEALTHY stream, which the fault
                    // path's sweep is not. Both of the load-time ways a program image can be wrong
                    // (the H2D staged the wrong bytes; two program allocations overlap, so the second
                    // upload lands in the first's flits) are already true here, before any op of this
                    // list has run, so this dates them without depending on the abend leaving the
                    // stream able to answer a D2H at all.
                    if oi == 0 {
                        let sweep = self.prog_images_sweep("PRE", ops);
                        eprint!("{sweep}");
                    }
                    let r = self.prog_image_report(op, prog);
                    if !r.starts_with("progimg: IDENTICAL") {
                        // ⛔ ONE WRITE. A multi-argument `eprintln!` reaches the fd in pieces and the
                        // CB-error lines are written from flex's completion THREAD, so a report built
                        // by the formatter came back shredded mid-number.
                        let one = format!("[progverify] op[{}] {} {r}\n", op.index, op.label());
                        eprint!("{one}");
                    }
                }

                if d.repeat_probe
                    && !op.kv.page_fold
                    && (op.kv.slot_write || op.kv.slab_write || op.kv.request > 0)
                {
                    // Both halves pay one drain, so the drain's host round trip cancels in the
                    // comparison and only the cold-vs-warm program difference is left.
                    let _ = stream.synchronize();
                    let r0 = Instant::now();
                    stream
                        .compute(prog, &ta, "", bootstrap, &[], barrier)
                        .map_err(launch_err)?;
                    let _ = stream.synchronize();
                    let r1 = Instant::now();
                    stream
                        .compute(prog, &ta, "", bootstrap, &[], barrier)
                        .map_err(launch_err)?;
                    let _ = stream.synchronize();
                    let r2 = Instant::now();
                    let mut t = timers();
                    let e = t
                        .repeat_ms
                        .entry(oi)
                        .or_insert((0.0, 0.0, 0, String::new()));
                    e.0 += (r1 - r0).as_secs_f64() * 1e3;
                    e.1 += (r2 - r1).as_secs_f64() * 1e3;
                    e.2 += 1;
                    if e.3.is_empty() {
                        e.3 = op.label();
                    }
                    continue;
                }

                if d.rb_time {
                    // The launch sequence the post-synchronize drain zips against: entry i is the
                    // i-th compute CB of this forward in submission order, keyed by (rung length,
                    // op index) so prefix/body/suffix op numbers cannot average into each other.
                    // Recorded before the collect branch so a batched rung's ops land here in the
                    // same submission order its single doorbell emits them in.
                    timers().rb_seq.push(((ops.len(), oi), op.label()));
                }

                if collect {
                    // STAGE, DON'T SUBMIT — the rung pays one doorbell at its end. The item OWNS
                    // its handles because the shifted bases and `ta` borrows above are rep-local.
                    // ⭐ And it is built by `LaunchSite::batched_compute` — the ONE constructor
                    // `cb_plan::walk_items` stages through too — so the item the real walk
                    // submits and the item the parity gate renders cannot drift apart in shape.
                    let st0 = Instant::now();
                    pending.push(PendingLaunch {
                        item: site.batched_compute(),
                        op_index: oi,
                        rep,
                        off,
                    });
                    batch_barrier |= barrier;
                    if d.submit_time {
                        // Prep is still per-op; the enqueue it feeds is now the rung's single
                        // flush, timed there — so only the prep half accumulates here.
                        let mut t = timers();
                        t.prep_ms += (st0 - gt0).as_secs_f64() * 1e3;
                    }
                } else if d.submit_time {
                    // `submit` is the enqueue alone: no synchronize on purpose — a drain would make
                    // this measure the very pipeline it is trying to leave undisturbed.
                    let st0 = Instant::now();
                    stream
                        .compute(prog, &ta, "", bootstrap, &[], barrier)
                        .map_err(fault)?;
                    let st1 = Instant::now();
                    let mut t = timers();
                    t.submit_ms += (st1 - st0).as_secs_f64() * 1e3;
                    t.prep_ms += (st0 - gt0).as_secs_f64() * 1e3;
                    t.submit_calls += 1;
                    if barrier {
                        t.host_fence_launches += 1;
                    }
                } else {
                    // Match torch-spyre's JobPlanStepCompute EXACTLY: kernel_name="" (NOT the mlir
                    // path) — it runs the resident dxp binary via its binary address, and passing
                    // the mlir path may route flex down a different (seed-only) path for the SFP
                    // transcendental while the matmul tolerates it. The same "" at the diag probes
                    // above and inside the rung flush's `compute_batch` / the CB render door.
                    stream
                        .compute(prog, &ta, "", bootstrap, &[], barrier)
                        .map_err(fault)?;
                }

                // DEFAULT-OFF: skipping the per-group drain removes the ~868-ops/layer
                // serialization — STRICT_ORDERING (ring FIFO) plus the always-set cmpt.sync
                // preserve the on-device dataflow (pod-confirmed coherent and ~11% faster);
                // `pipeline_barrier` itself is a host fence call, a no-op for all-compute runs.
                if d.perop_sync {
                    let _ = stream.synchronize();
                }
                if d.group_time {
                    // Needs a drain to attribute anything, so it costs what the per-op sync costs.
                    let _ = stream.synchronize();
                    let ms = gt0.elapsed().as_secs_f64() * 1e3;
                    let mut t = timers();
                    let e = t.group_ms.entry(op.label()).or_insert((0.0, 0));
                    e.0 += ms;
                    e.1 += 1;
                    t.group_calls += 1;
                    if t.group_calls.is_multiple_of(20_000) {
                        report_group_time(&t);
                    }
                }
            } // end per-page fold repeat

            if Some(oi) == px_target {
                // AFTER every rep of this group, so a fold's whole pass count lands in one rung.
                let _ = self.stream_ref()?.synchronize();
                let mut t = timers();
                let e = t
                    .prefix_ms
                    .entry((ops.len(), oi))
                    .or_insert((0.0, 0, String::new()));
                e.0 += px0.elapsed().as_secs_f64() * 1e3;
                e.1 += 1;
                if e.2.is_empty() {
                    e.2 = op.label();
                }
            }

            if d.optrace {
                // ⭐ THE OP'S OWN NAME, on the line that reports it — the label the group-time and
                // launch-mark paths already print (:3659/:3511), so a trace names its ops without a
                // harvest pull. Passed in (not re-derived) because the op is the loop's own local.
                self.optrace(oi, &op.label())?;
            }
            if d.tensordump.is_some() {
                self.tensordump(oi, op)?;
            }
        }

        // ── THE RUNG'S ONE DOORBELL ──
        // Everything the loop staged goes to the card as ONE submission: one capacity gate, one
        // grouped completion, one `QueueControlBlocksSBF`. The CB ring is FIFO, so the device
        // sees exactly the sequence the per-op path would have submitted — fewer doorbells,
        // same order, same dataflow (the head barrier carries the OR of the per-op barriers;
        // see `Stream::compute_batch`).
        if !pending.is_empty() {
            // Split the site context from the items once, at the boundary: the items are what
            // the doorbell takes, the context is what a refusal dumps.
            let mut sites: Vec<(usize, i64, [u64; NUM_SEGMENTS])> =
                Vec::with_capacity(pending.len());
            let items: Vec<BatchedCompute> = pending
                .drain(..)
                .map(|p| {
                    sites.push((p.op_index, p.rep, p.off));
                    p.item
                })
                .collect();
            let stream = self.stream_ref()?;
            let st0 = Instant::now();
            let r = stream.compute_batch(&items, batch_barrier);
            let st1 = Instant::now();
            if let Err(e) = r {
                // ⛔ EVERY FAULTING LAUNCH NAMES ITS OWN AIM — the flush's refusal must too.
                // The loop locals the per-op `fault` closure dumps from are gone here, so the
                // offending entry's site is rebuilt from what the batch carries: flex-rs wraps
                // the failing element's index when it has one; a refusal that names no element
                // (shutdown, capacity) still gets the sweep and the rung's shape.
                if let HwError::Stream(RuntimeStreamError::BatchElement { index, .. }) = &e
                    && let Some(&(oi, rep, off)) = sites.get(*index)
                {
                    let ta: Vec<&DevAddr> = items[*index].tensor_allocs.iter().collect();
                    // ⛔ ONE WRITE — same single-`format!` discipline as `[progverify]`.
                    let one = format!(
                        "[flush] refused at batch entry {index} (op {oi}, rep {rep}) of {} \
                         staged launch(es)\n",
                        items.len()
                    );
                    eprint!("{one}");
                    self.dump_launch_site(
                        "FAULT(batch-flush)",
                        &ta,
                        &ops[oi],
                        &items[*index].prog,
                        items[*index].bootstrap_offset,
                        &off,
                    );
                }
                let sweep = self.prog_images_sweep("FAULT", ops);
                eprint!("{sweep}");
                return Err(launch_err(e).context(format!(
                    "rung flush refused: {} launch(es), one doorbell, barrier={batch_barrier}",
                    items.len()
                )));
            }
            if d.submit_time {
                // The flush IS the enqueue the loop's per-op prep times fed: one submit call
                // for the whole rung, so `submit_calls` counts doorbells, not ops, on this
                // path — the honest answer to "what does one doorbell cost" is the flush.
                let mut t = timers();
                t.submit_ms += (st1 - st0).as_secs_f64() * 1e3;
                t.submit_calls += 1;
                if batch_barrier {
                    t.host_fence_launches += 1;
                }
            }
        }
        Ok(())
    }

    /// ONE-SHOT SEGMENT-BASE DUMP: every segment's resolved runtime base (region, byte offset,
    /// chunk size) for the FIRST op of the FIRST layer — checkable by hand against
    /// bundle_layout.json's declared (segment, offset), instead of guessing from static JSON.
    fn dump_seg_addrs(
        &self,
        ta: &[&DevAddr],
        op: &OpProg,
        prog: &DevAddr,
        bootstrap: u64,
        off: &[u64; NUM_SEGMENTS],
    ) {
        static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        self.dump_launch_site("op[0]", ta, op, prog, bootstrap, off);
    }

    /// ⭐⭐⭐ WHERE ONE LAUNCH WAS AIMED — every resolved segment base and, decisively, the PROGRAM
    /// segment's own two numbers: the bootstrap offset the device starts executing at and the extent
    /// of the allocation holding the binary.
    ///
    /// ⛔ THIS IS WHAT SEPARATES A PROGRAM FAULT FROM A DATA FAULT. A QGI/HMI address is a DMVA, and a
    /// DMVA's top bits ARE the segment (`SEGMENT_SIZE_BITS = 34`, `PROG_SEGMENT = 7`), so an address in
    /// `0x1c00000000..` is in the program segment by construction — but that is equally the shape of a
    /// data operand whose base got resolved against segment 7, so the segment bits alone settle
    /// nothing. The number that settles it is THIS LAUNCH'S BOOTSTRAP: if the faulting DMVA is
    /// `PROG_OFFSET_BASE + bootstrap`, the device faulted on the very address the host told it to start
    /// at, and no data operand is involved at all.
    fn dump_launch_site(
        &self,
        who: &str,
        ta: &[&DevAddr],
        op: &OpProg,
        prog: &DevAddr,
        bootstrap: u64,
        off: &[u64; NUM_SEGMENTS],
    ) {
        // ⛔ ONE WRITE, NOT TWELVE. `eprintln!` reaches the fd in several unlocked pieces and the
        // tracing subscriber writes the CB-error lines from the completion THREAD, so a per-line dump
        // came back shredded mid-number — unreadable exactly when it matters.
        use std::fmt::Write as _;
        let mut out = String::new();
        for (i, a) in ta.iter().enumerate() {
            match a.chunk0() {
                Some(c) if a.is_single_chunk() => {
                    let _ = writeln!(
                        out,
                        "[segaddr] {who} seg{i} region={} offset={} size={} shift={}",
                        c.region.0, c.offset.0, c.size.0, off[i]
                    );
                }
                _ => {
                    let _ = writeln!(out, "[segaddr] {who} seg{i} MULTI-CHUNK shift={}", off[i]);
                }
            }
        }
        let base = crate::runtime::prog_offset_base();
        match prog.chunk0() {
            Some(c) => {
                let _ = writeln!(
                    out,
                    "[segaddr] {who} seg7/PROG region={} offset={} size={} (binary {} B)",
                    c.region.0,
                    c.offset.0,
                    c.size.0,
                    op.code.init_binary.len()
                );
            }
            None => {
                let _ = writeln!(out, "[segaddr] {who} seg7/PROG no chunk");
            }
        }
        let _ = writeln!(
            out,
            "[segaddr] {who} op[{}] {} job_bin_ptr={:#x} PROG_OFFSET_BASE={base:#x} \
             bootstrap={bootstrap:#x} ({bootstrap} B = flit {}) bootstrap_dmva={:#x} \
             prog_extent={} B = {} flit(s)",
            op.index,
            op.label(),
            op.code.job_bin_ptr,
            // A flit IS a stick: 128 B (`sys-arch-spec/sysdef.cpp bytesPerStick`), the unit QGI and
            // HMI report their addresses in.
            bootstrap / u64::from(crate::lower_subtile_tape_to_superdsc::STICK_BYTES),
            base + bootstrap,
            prog.total_size().as_u64(),
            prog.total_size().as_u64()
                / u64::from(crate::lower_subtile_tape_to_superdsc::STICK_BYTES),
        );
        let _ = writeln!(out, "[segaddr] {who} {}", self.prog_image_report(op, prog));
        eprint!("{out}");
    }

    /// ⭐⭐⭐ EVERY PROGRAM IN ONE LIST, READ BACK AND DIFFED — the reading that needs no attribution.
    ///
    /// ⛔ THE FAULTING OP IS NOT THE OP THAT REPORTS THE FAULT, and assuming otherwise is a live
    /// hazard here. A compute launch returns `rc = 0` at submit and the hardware's verdict arrives
    /// later on its own control block, so what a failed `compute` call actually means is "the stream
    /// is in an error state from a PRIOR operation" — the log reads
    /// `scheduler rejected submission: deferred error from a prior operation`. The op named by the
    /// launch-site dump is therefore the first op to be REFUSED, which is some op after the one that
    /// faulted. Sweeping the whole list sidesteps the question entirely: if any program's device copy
    /// differs from its baked bytes, this finds it without having to know which one the hardware was
    /// running.
    fn prog_images_sweep(&self, who: &str, ops: &Ops) -> String {
        use std::fmt::Write as _;
        let mut same = 0usize;
        let mut unread = 0usize;
        let mut bad: Vec<usize> = Vec::new();
        let mut detail = String::new();
        for op in ops.iter() {
            let Some(prog) = op.addr.as_ref() else {
                continue;
            };
            let r = self.prog_image_report(op, prog);
            if r.starts_with("progimg: IDENTICAL") {
                same += 1;
            } else if r.starts_with("progimg: DIFFERS") {
                bad.push(op.index);
                if bad.len() <= 4 {
                    let i = op.index;
                    let _ = writeln!(detail, "[progsweep] {who} op[{i}] {} {r}", op.label());
                }
            } else {
                unread += 1;
                if unread == 1 {
                    let i = op.index;
                    let _ = writeln!(detail, "[progsweep] {who} op[{i}] {} {r}", op.label());
                }
            }
        }
        format!(
            "[progsweep] {who} {} op(s): {same} IDENTICAL, {} DIFFER {:?}, {unread} \
             UNREADABLE\n{detail}",
            ops.len(),
            bad.len(),
            bad,
        )
    }

    /// ⭐⭐⭐ THE DEVICE'S OWN COPY OF ONE PROGRAM, READ BACK AND DIFFED AGAINST THE BAKED BYTES.
    ///
    /// ⛔ THIS IS THE DIRECT READ OF THE ONLY SURVIVING CLASS behind a `PrepZeroFlitCnt`+`PrepSwVer`
    /// fault. Those two cases are the job-header parser refusing a header, and the baked header chain
    /// has been walked in full — every program is well-formed under dip's own writer arithmetic, and
    /// the flit the hardware names is ordinary payload, never a declared job boundary. Two statements
    /// that cannot both be true, so one of them is not about the same bytes: the device is executing
    /// an image that differs from the one the bake produced. Everything else about the launch (the
    /// allocation's size, the H2D's length, `bootstrap`, the segment-7 translation) has been measured
    /// and agrees, and none of those readings can see the CONTENT.
    ///
    /// The comparison is byte-for-byte over the binary's own length — never the allocation's
    /// alignment-padded extent, which holds nothing and is not uploaded.
    ///
    /// ⚠️ WHAT AN "IDENTICAL" READING DOES *NOT* PROVE: that the device fetched these bytes. A D2H
    /// resolves the host's `DevAddr` — region, offset, length — so it re-reads through the host's own
    /// notion of where the program is. If the device fetched instructions through a DIFFERENT
    /// translation for segment 7 than the one this address describes, the bytes here are the correct
    /// ones and the fault is in the mapping, not the image. That is the shape "identical" leaves
    /// standing, and it is worth naming because it is the only one left after this.
    fn prog_image_report(&self, op: &OpProg, prog: &DevAddr) -> String {
        use std::fmt::Write as _;
        let baked: &[u8] = &op.code.init_binary;
        let stick = usize::try_from(crate::lower_subtile_tape_to_superdsc::STICK_BYTES)
            .expect("STICK_BYTES fits usize");
        let Ok(stream) = self.stream_ref() else {
            return "progimg: no stream — nothing to read back".to_string();
        };
        // ⛔ READ WHERE THE IMAGE WAS PUT, not where the allocation starts — `alloc_ops_at` places it
        // at dxp's declared prologue, so a base-relative readback would diff the prologue against the
        // image and call every such program corrupt.
        let head = op
            .code
            .job_bin_ptr
            .saturating_sub(crate::runtime::prog_offset_base());
        let Some(win) = prog.window(ByteOffset(head), ByteSize(baked.len() as u64)) else {
            return "progimg: program window unaddressable".to_string();
        };
        let mut host = vec![0u8; baked.len()];
        if let Err(e) = stream.d2h(&mut host, &win) {
            return format!("progimg: D2H: {e} — THE READBACK ITSELF FAILED, no verdict");
        }
        if let Err(e) = stream.synchronize() {
            return format!("progimg: D2H sync: {e} — THE READBACK ITSELF FAILED, no verdict");
        }
        // The header as dip writes it and `QGHeader` reads it (`dip/dip.cpp:77-91`, `senulator/qg.h`):
        // `u8[0]` is `SW_VER`, bits 21:8 are the flit count EXCLUDING the header's own flit
        // (`myflits = totalFlits_ - 1`), and bit 22 marks the last job of a program.
        let hdr = |b: &[u8], f: usize| -> String {
            match b.get(f * stick..f * stick + 4) {
                None => "(short)".to_string(),
                Some(w) => {
                    let v = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                    format!(
                        "{v:#010x} swver={:#04x} count={} term={}",
                        v & 0xff,
                        (v >> 8) & 0x3fff,
                        (v >> 22) & 1
                    )
                }
            }
        };
        let hex = |b: &[u8], f: usize| -> String {
            let Some(s) = b.get(f * stick..f * stick + 32) else {
                return "(short)".to_string();
            };
            let mut h = String::new();
            for x in s {
                let _ = write!(h, "{x:02x}");
            }
            h
        };
        let Some(first) = (0..baked.len()).find(|&i| host[i] != baked[i]) else {
            return format!(
                "progimg: IDENTICAL {} B — the device's copy of this program IS the baked image \
                 (flit0 {})",
                baked.len(),
                hdr(baked, 0)
            );
        };
        let ndiff = (0..baked.len()).filter(|&i| host[i] != baked[i]).count();
        let nflits = baked.len() / stick;
        let bad: Vec<usize> = (0..nflits)
            .filter(|&f| {
                (f * stick..(f + 1) * stick).any(|i| i < baked.len() && host[i] != baked[i])
            })
            .collect();
        let ff = first / stick;
        let mut out = format!(
            "progimg: DIFFERS {} B of {} B, FIRST at byte {first} = flit {ff} + {} — \
             {} of {nflits} flit(s) differ, first differing flits {:?}{}",
            ndiff,
            baked.len(),
            first % stick,
            bad.len(),
            &bad[..bad.len().min(24)],
            if bad.len() > 24 { " …" } else { "" },
        );
        let _ = write!(
            out,
            "\n[segaddr]   flit0  hdr baked={} | device={}\
             \n[segaddr]   flit{ff} hdr baked={} | device={}\
             \n[segaddr]   flit{ff} baked [0..32]={}\
             \n[segaddr]   flit{ff} devic [0..32]={}",
            hdr(baked, 0),
            hdr(&host, 0),
            hdr(baked, ff),
            hdr(&host, ff),
            hex(baked, ff),
            hex(&host, ff),
        );
        out
    }

    /// ONE-PASS OPTRACE: after each on-card op, D2H the activation/intermediate/KV segments and
    /// print nonzero-count + max|.|. One prefill run then shows the signal flow op-by-op; the op
    /// index where nz collapses to 0 (or max → inf) IS the divergence. seg1 (weights, ~2.6 GB) is
    /// deliberately SKIPPED: static across the pass and already ruled out.
    fn optrace(&mut self, oi: usize, label: &str) -> Result<()> {
        let _ = self.stream_ref()?.synchronize();
        let mut line = format!("[optrace] op[{oi}] {label}");
        for i in [0usize, 2, 3, 4, 5, 6] {
            let seg = SegIdx::checked(i as i64).expect("literal < NUM_SEGMENTS");
            let got = self.d2h_seg_clamped(seg);
            let stat = match got {
                Ok(got) => {
                    let host = &self.seg_host[i][..got.min(self.seg_host[i].len())];
                    let (mut nz, mut mx) = (0usize, 0.0f32);
                    let mut n_inf = 0usize;
                    // First NON-FINITE element offsets (element index in the segment), so a divergence
                    // probe can say WHICH tensor went inf rather than only that the segment holds one.
                    // `SCRATCHY_SDSC_OPTRACE_INF=1` opts in: printing offsets costs a scan this line
                    // already pays, but the extra string only matters when it fires.
                    let mut inf_at: Vec<usize> = Vec::new();
                    // `as_chunks` over `chunks_exact`: a CONSTANT chunk width yields `&[u8; 2]`, so the
                    // pair cannot be short (clippy::chunks_exact_to_as_chunks).
                    for (ei, c) in host.as_chunks::<2>().0.iter().enumerate() {
                        let bits = u16::from_le_bytes(*c);
                        if bits != 0 {
                            nz += 1;
                            let v = sen_convert::sen_to_f32(sen_convert::SenF16(bits)).abs();
                            if v > mx {
                                mx = v;
                            }
                            if !v.is_finite() {
                                n_inf += 1;
                                if inf_at.len() < 4 {
                                    inf_at.push(ei);
                                }
                            }
                        }
                    }
                    if n_inf > 0 && std::env::var_os("SCRATCHY_SDSC_OPTRACE_INF").is_some() {
                        format!("nz={nz} max={mx:.3} INF={n_inf} at {inf_at:?}")
                    } else {
                        format!("nz={nz} max={mx:.3}")
                    }
                }
                Err(_) => "unreadable".to_string(),
            };
            line.push_str(&format!(" | seg{i} {stat}"));
        }
        eprintln!("{line}");
        Ok(())
    }

    /// ⭐ CARD-SIDE TENSORDUMP (`SCRATCHY_SDSC_TENSORDUMP=<dir>`) — the emulator-diff half of the
    /// bisection instrument. After the op at `oi` settles, D2H every readable segment and write each
    /// placement whose bytes MOVED since the last dump:
    ///
    /// * `prog{P}_seg{S}_t{T}.bin` + `.n` — the LATEST value, named to match the emulator's
    ///   `SCRATCHY_EMU_TENSORDUMP` (`ktir-emulator/src/resident.rs`) so a diff joins on
    ///   (program, tensor id). The emulator's `seg` is ITS planned-segment index (a run of fused
    ///   nodes) while ours is the memory segment — the join key is (prog, tid), never the seg number.
    /// * `op{O}_prog{P}_seg{S}_t{T}.bin` + `.n` — the timeline copy, one file per write, so the
    ///   FIRST divergent op for a tensor is the first `op…` file that disagrees with the emulator's
    ///   trajectory for that tid.
    /// * `ops.txt` — one line per op: `op{O} prog{P} list-oi {label}`. `progs.txt` — the
    ///   fingerprint → program-index assignment.
    ///
    /// First-writer semantics match the emulator's dump law (only tensors the segment actually
    /// wrote), by CHANGE DETECTION on the host shadow rather than by an output list: a card launch
    /// group carries its program's arg→tensor mapping but no direction, and a byte-move is the same
    /// evidence the emulator's `PROBE6` change detection uses. A recolored slot whose bytes moved
    /// under an ALIASED tensor fires here too — the diff script must tolerate a card-only dump file
    /// (the emulator has no entry for that tid at that point), which is why every file is
    /// self-describing via `.n`.
    ///
    /// Elements are LE f32 converted from the device's SEN169 fp16 (2 B/elem) — the same
    /// `sen_to_f32` the optrace's stats use. RAW int32 tensors (the gather index table, in
    /// `raw_ids`) are dumped as i32→f32 with `.n = size/4` so the file stays comparable; the
    /// emulator never dumps those (host-bound sources), so they simply have no reference file.
    /// Bank ≥ 1 weights and placements past a clamped shadow window are skipped (unreadable), each
    /// with one `skip.txt` line so the gap is visible rather than silent.
    fn tensordump(&mut self, oi: usize, op: &OpProg) -> Result<()> {
        let Some(dir) = Diag::get().tensordump.clone() else {
            return Ok(());
        };
        if self.dump.is_none() {
            let prog = dump_prog_index(&self.code.fp);
            let _ = std::fs::create_dir_all(&dir);
            // Append (never truncate): a run may build several sessions, and the registry line for
            // each is one row.
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("progs.txt"))
            {
                let _ = writeln!(f, "prog{prog} fp={}", self.code.fp);
            }
            self.dump = Some(TensorDump {
                prog,
                op: 0,
                snap: std::collections::HashMap::new(),
                wins: vec![None; 2],
            });
        }
        let _ = self.stream_ref()?.synchronize();
        // D2H every segment a placement can live in. seg2's shadow is one page (clamped read), and
        // an aliased segment's device region belongs to its owner — reading it back is still SOUND
        // (a read never writes), it just reflects the owner's bytes, which is exactly what this
        // session's launches compute against.
        let mut got = [0usize; NUM_SEGMENTS];
        for (i, g) in got.iter_mut().enumerate() {
            if self.seg_addr[i].is_none() {
                continue;
            }
            *g = self
                .d2h_seg_clamped(SegIdx::checked(i as i64).expect("< NUM_SEGMENTS"))
                .unwrap_or(0);
        }
        let dump = self.dump.as_mut().expect("just initialized");
        dump.op += 1;
        let (prog, opno) = (dump.prog, dump.op);
        {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("ops.txt"))
            {
                let _ = writeln!(f, "op{opno} prog{prog} list-oi={oi} {}", op.label());
            }
        }
        let mut skipped = String::new();
        for (id, p) in &self.places {
            if p.bank != 0 || p.size == 0 {
                continue; // a banked weight is not in its segment's region
            }
            let i = p.segment as usize;
            if i >= NUM_SEGMENTS || self.seg_addr[i].is_none() {
                continue;
            }
            // Only whole-tensor reads that the clamped D2H actually brought back: a placement past
            // the shadow's window would read stale host bytes and masquerade as a write.
            let avail = got[i] as u64;
            if avail == 0 || p.offset + p.size > avail {
                if p.segment as usize == SEG_WEIGHT.get() {
                    // The weight segment's shadow is full-size; a miss there is a layout shock.
                    skipped.push_str(&format!(
                        "op{opno} prog{prog} {id}: seg{} off {} size {} past the {} B read back\n",
                        p.segment, p.offset, p.size, avail
                    ));
                }
                continue;
            }
            let cur = &self.seg_host[i][p.offset as usize..][..p.size as usize];
            let moved = match dump.snap.get(id) {
                Some(prev) => prev.as_slice() != cur,
                None => cur.iter().any(|b| *b != 0),
            };
            if !moved {
                continue;
            }
            dump.snap.insert(*id, cur.to_vec());
            let raw = self.raw_ids.contains(id);
            let elems = if raw {
                p.size as usize / 4
            } else {
                p.size as usize / 2
            };
            let mut vals = Vec::with_capacity(elems);
            if raw {
                for c in cur.as_chunks::<4>().0 {
                    vals.push(i32::from_le_bytes(*c) as f32);
                }
            } else {
                for c in cur.as_chunks::<2>().0 {
                    vals.push(sen_convert::sen_to_f32(sen_convert::SenF16(
                        u16::from_le_bytes(*c),
                    )));
                }
            }
            let raw_bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            let stem = format!("prog{prog}_seg{}_t{id}", p.segment);
            let tl = format!("op{opno}_prog{prog}_seg{}_t{id}", p.segment);
            let _ = std::fs::write(dir.join(format!("{stem}.bin")), &raw_bytes);
            let _ = std::fs::write(dir.join(format!("{stem}.n")), format!("{elems}\n"));
            let _ = std::fs::write(dir.join(format!("{tl}.bin")), &raw_bytes);
            let _ = std::fs::write(dir.join(format!("{tl}.n")), format!("{elems}\n"));
        }
        // ── LEG-2: the pool windows NO PLACEMENT NAMES. The placement watch above covers page 0 of
        // seg2 only (the one-page host shadow), but the paged KV path lives at page 1 (the
        // request's block — the gather's read target) and page 2 (the doubled base the LEG-1 aim
        // line pinned as the cachewr target). Two fixed 1 MiB windows (layer 0's K/V/Kt planes =
        // 3 × 256 KiB, plus margin), same first-writer/change detection and file naming as the
        // placements: a window whose content appears, moves or stays empty is a FACT about where
        // the chunk KV landed, not an inference from page 0's leftovers.
        //
        // ⛔ NO BAKED-ABSOLUTE WINDOW, BY MEASUREMENT (chatG, 10-23): the emitter's own plan
        // addresses seg2 at SEGMENT_OFFSETS[2] = 0x8_0000_0000, a 16 GiB region no runtime
        // allocation holds — and the runtime CANNOT even address it: a D2H minted over that
        // region is rejected by the DeviceAddressResolver ("could not resolve region"), and the
        // rejection leaves a DEFERRED ERROR on the stream that refuses the next real launch. So
        // baked absolutes are unrepresentable at runtime — the symbolic bundle's segbase + off
        // decomposition is the only mechanism any op's addresses run under — and attempting the
        // read kills the run. The window set therefore stops at the pool's own pages.
        let stride = self.fold.page_stride_bytes.0;
        if stride > 0 {
            if let Some(pool) = self.seg_addr[SEG_KV.get()].clone() {
                const WIN: u64 = 1 << 20;
                let wants: [(&str, Option<DevAddr>); 2] = [
                    ("pg1", pool.window(ByteOffset(stride), ByteSize(WIN))),
                    ("pg2", pool.window(ByteOffset(2 * stride), ByteSize(WIN))),
                ];
                // Read every window BEFORE touching `dump` (the stream borrow and the dump borrow
                // are disjoint phases, the same split `d2h_seg_clamped` makes).
                let mut bufs: [Option<Vec<u8>>; 2] = [None, None];
                for (bi, (_, waddr)) in wants.iter().enumerate() {
                    let Some(waddr) = waddr else {
                        continue; // a window past the pool's end — not addressable, nothing to read
                    };
                    let mut buf = vec![0u8; WIN as usize];
                    let stream = self.stream_ref()?;
                    match stream
                        .d2h(&mut buf, waddr)
                        .and_then(|_| stream.synchronize())
                    {
                        Ok(()) => bufs[bi] = Some(buf),
                        Err(e) => {
                            use std::io::Write as _;
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(dir.join("windows.txt"))
                            {
                                let _ = writeln!(
                                    f,
                                    "op{opno} prog{prog} {}: window read FAILED: {e}",
                                    wants[bi].0
                                );
                            }
                        }
                    }
                }
                for (bi, (name, _)) in wants.iter().enumerate() {
                    let Some(cur) = bufs[bi].as_ref() else {
                        continue;
                    };
                    let dump = self.dump.as_mut().expect("initialized above");
                    let moved = match dump.wins[bi].as_ref() {
                        Some(prev) => prev != cur,
                        None => cur.iter().any(|b| *b != 0),
                    };
                    if !moved {
                        continue;
                    }
                    dump.wins[bi] = Some(cur.clone());
                    let elems = WIN as usize / 2; // the KV planes are SEN169_FP16
                    let mut vals = Vec::with_capacity(elems);
                    for c in cur.as_chunks::<2>().0 {
                        vals.push(sen_convert::sen_to_f32(sen_convert::SenF16(
                            u16::from_le_bytes(*c),
                        )));
                    }
                    let raw_bytes: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let stem = format!("prog{prog}_seg2win_{name}");
                    let tl = format!("op{opno}_prog{prog}_seg2win_{name}");
                    let _ = std::fs::write(dir.join(format!("{stem}.bin")), &raw_bytes);
                    let _ = std::fs::write(dir.join(format!("{stem}.n")), format!("{elems}\n"));
                    let _ = std::fs::write(dir.join(format!("{tl}.bin")), &raw_bytes);
                    let _ = std::fs::write(dir.join(format!("{tl}.n")), format!("{elems}\n"));
                }
            }
        }
        if !skipped.is_empty() {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("skip.txt"))
            {
                let _ = write!(f, "{skipped}");
            }
        }
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Predict
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Attribute this forward's harvested RB timings to the ops that launched them, and report on
    /// the same cadence `group_time` does. The completion thread records every CB's five device
    /// cycle counts as it completes (`flex_rs::rb_telemetry`), so after the step's `synchronize`
    /// — the one the step already performs, no extra drain — the ring holds exactly this
    /// forward's CBs. Op compute entries (DMA crossings excluded by
    /// [`flex_rs::rb_telemetry::RbTiming::dma`]) sorted by
    /// submission id zip 1:1 against [`Timers::rb_seq`], which the launch loop filled in the same
    /// order. This is the attribution `group_time`/`prefix_time` pay ~11% of decode to estimate,
    /// read for free instead.
    fn rb_attribute(&mut self) {
        let seq = std::mem::take(&mut timers().rb_seq);
        let mut entries = flex_rs::rb_telemetry::drain();
        let mut t = timers();
        t.rb_steps += 1;
        // The DMA crossings belong to no op — the preamble-H2D and logits-D2H — so they
        // accumulate wherever they arrive, attribution-free. Since #544 those two ride the
        // COMPUTE pipeline (they used to take the async ones), so `pipeline` alone no longer
        // routes them: a Compute entry carrying a direction is a crossing and goes to its
        // direction's total, exactly where it lived before the move.
        for e in &entries {
            match (e.pipeline, e.dma) {
                (flex_rs::scheduler::PipelineId::AsyncDmaI, _)
                | (
                    flex_rs::scheduler::PipelineId::Compute,
                    Some(flex_rs::dma::DmaDirection::HostToDevice),
                ) => t.rb_dmai.add_dmai(e.rb),
                (flex_rs::scheduler::PipelineId::AsyncDmaO, _)
                | (
                    flex_rs::scheduler::PipelineId::Compute,
                    Some(flex_rs::dma::DmaDirection::DeviceToHost),
                ) => t.rb_dmao.add_dmao(e.rb),
                (flex_rs::scheduler::PipelineId::Compute, None) => {}
            }
        }
        // PIN THE MAPPING ON-CARD: raw per-CB records for the first few forwards, verbatim. The
        // aggregates below blend forwards (prefill and decode CBs differ) and cannot decide the
        // slot-to-name mapping; these can — an H2D CB lights the dmai slots, a D2H the dmao
        // slots, a compute CB the rest, and only same-domain slot pairs difference to durations.
        // The compute-pipeline DMA crossings print here too (a Compute line in this print IS one
        // — no op entry reaches it), so their own slot mapping gets pinned the same way.
        if t.rb_steps <= 3 {
            for e in entries.iter().filter(|e| {
                !matches!(e.pipeline, flex_rs::scheduler::PipelineId::Compute) || e.dma.is_some()
            }) {
                eprintln!(
                    "[sdsc-rbtime] fwd {} {:?} id {} slots {:?} power {}",
                    t.rb_steps,
                    e.pipeline,
                    e.id,
                    e.rb.timestamps(),
                    e.rb.power()
                );
            }
        }
        // The inter-step device-idle gap's raw material, one line EVERY forward (the raw
        // per-CB probe above stops at three): each class's device-clock window. The
        // post-processor differences fwd N's `last` into fwd N+1's `first` — wrap-exact at
        // any real step gap, the counter is free-running u32 — and decomposes the window by
        // class: last compute end → D2H span → host tail → H2D span → first compute start.
        // Classes mirror the routing above: crossings by direction on whichever pipeline
        // carried them, non-crossing async/fill CBs as `other`, op CBs as `compute`.
        {
            use flex_rs::dma::DmaDirection;
            use flex_rs::scheduler::PipelineId;
            let compute = rb_class_window(&entries, |e| {
                matches!(e.pipeline, PipelineId::Compute) && e.dma.is_none()
            });
            let h2d = rb_class_window(&entries, |e| e.dma == Some(DmaDirection::HostToDevice));
            let d2h = rb_class_window(&entries, |e| e.dma == Some(DmaDirection::DeviceToHost));
            let other = rb_class_window(&entries, |e| {
                e.dma.is_none() && !matches!(e.pipeline, PipelineId::Compute)
            });
            let all = rb_class_window(&entries, |_| true);
            let fmt = |w: &Option<RbClassWindow>| match w {
                Some(w) => format!("[{},{},{}]", w.first, w.last, w.n),
                None => "-".to_string(),
            };
            eprintln!(
                "[sdsc-rbgap] fwd {} all{} compute{} h2d{} d2h{} other{}",
                t.rb_steps,
                fmt(&all),
                fmt(&compute),
                fmt(&h2d),
                fmt(&d2h),
                fmt(&other)
            );
        }
        // The op entries: compute CBs that are not DMA crossings. A crossing kept here would
        // both break the 1:1 zip below and bill the preamble/readback to whichever op it
        // happened to sort next to.
        entries.retain(|e| {
            matches!(e.pipeline, flex_rs::scheduler::PipelineId::Compute) && e.dma.is_none()
        });
        entries.sort_unstable_by_key(|e| e.id);
        if entries.len() != seq.len() {
            // LOUD, NOT SILENT: a mismatch means per-op attribution would be WRONG, not merely
            // missing — `repeat_probe` launches computes that are in the ring but not in the
            // sequence, and a rotated ring loses launches from its front. (The DMA crossings
            // are NOT a mismatch source anymore: `dma` excludes them above.) Skip the
            // attribution entirely; the DMA totals above stay honest either way.
            eprintln!(
                "[sdsc-rbtime] sequence mismatch: {} completed compute CBs vs {} recorded \
                 launches — per-op attribution SKIPPED for this forward",
                entries.len(),
                seq.len()
            );
            return;
        }
        let zipped: Vec<_> = entries.iter().zip(seq).collect();
        if t.rb_steps <= 3 {
            let n = zipped.len();
            for (i, (e, ((rung, oi), label))) in zipped
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < 20 || *i + 8 >= n)
            {
                eprintln!(
                    "[sdsc-rbtime] fwd {} op ({rung},{oi}) rec {i} id {} slots {:?} power {}  {}",
                    t.rb_steps,
                    e.id,
                    e.rb.timestamps(),
                    e.rb.power(),
                    label
                );
            }
        }
        let mut new_cbs = 0u64;
        for (e, ((rung, oi), label)) in zipped {
            let acc = t.rb_cycles.entry((rung, oi)).or_default();
            acc.cycles.add_compute(e.rb);
            new_cbs += 1;
            if acc.label.is_empty() {
                acc.label = label;
            }
        }
        t.rb_cbs += new_cbs;
        // Same cadence as `group_time`'s report: rare enough not to distort, often enough that a
        // long run does not have to end first. Watermarked rather than `is_multiple_of` because
        // the cumulative count jumps a whole forward at a time.
        if t.rb_cbs - t.rb_reported >= 20_000 {
            t.rb_reported = t.rb_cbs;
            report_rb_time(&t);
        }
    }

    /// The shared FORWARD HALF of a step: the prepared/source gates, the RB window reset, the
    /// activation refill, and the replay-or-launch decision — extracted from `predict`, so
    /// the refill law, the replay gate and
    /// the launch are ONE behavior the readback rides, never a copy. `suffix` is
    /// [`SuffixLaunch::Skipped`] only for a forward whose logits are discarded ([`Logits`]).
    fn forward_half(
        &mut self,
        n_new: NewTokens,
        seq_pos: SeqPos,
        suffix: SuffixLaunch,
        d: &Diag,
    ) -> Result<ForwardHalf> {
        if !self.prepared {
            bail!("predict before prepare");
        }
        self.require_sources_filled()?;

        // RB timing: start the forward's window empty. The previous forward drained its own
        // entries at its synchronize; what is left in the ring now is pre-forward history
        // (prepare's weight uploads belong to no op), and the launch sequence starts at zero.
        if d.rb_time {
            let _ = flex_rs::rb_telemetry::drain();
        }
        timers().rb_seq.clear();

        // The stream is PERSISTENT: the scheduler retires each op's fence from its per-stream
        // deque the moment that fence resolves (SchedulerState::retire_fence), so nothing
        // per-forward accumulates on it and the old recreate-to-drain-every-forward workaround
        // is gone. The previous forward already synchronized before returning (the trailing
        // `synchronize` below waits out every op, H2Ds included), so dropping the recreate's
        // implicit destroy-sync changes no ordering; the resident weight/KV/program
        // allocations live on the runtime either way.
        let t_stream0 = Instant::now();
        let t0 = t_stream0;
        let (h2d_note, h2d_bytes) = self.refill_activations(d)?;
        let t0b = Instant::now();
        if d.phase_time {
            let _ = self.stream_ref()?.synchronize();
        }
        let t1 = Instant::now();

        // ⭐ #603 PR2 2c — a ONE-TOKEN step on a quiet diag replays from a cached image
        // (canvas copy + patches + ONE doorbell); everything else takes the walk.
        // `n_new == 1` is the step's own shape, not a toggle: a chunked-prefill step
        // (n_new > 1) sweeps a fresh page count per chunk and would pay a derivation
        // per chunk — PR4's rungs, not the player's. `try_replay_forward` states its
        // own other refusals (batched fold, instruments) and hands back `None` for
        // the walk.
        // The player's images all carry the suffix, so a forward that skips it takes the walk.
        let replayed = n_new.0 == 1
            && suffix == SuffixLaunch::Launched
            && self.try_replay_forward(seq_pos)?.is_some();
        if !replayed {
            self.launch_forward(seq_pos, suffix)?;
        }

        if d.phase_time {
            let _ = self.stream_ref()?.synchronize();
        }
        let t2 = Instant::now();
        Ok(ForwardHalf {
            t_stream0,
            t0,
            t0b,
            t1,
            t2,
            h2d_note,
            h2d_bytes,
        })
    }

    /// The `[sdsc-phase]` block both readback tails print — one renderer, parameterized only by
    /// WHICH readback ran (`"logits"` / `"idx"`), so the token step's line set cannot drift from
    /// the logits step's. The rbgap and phase harnesses grep these lines.
    fn print_step_phase(
        half: &ForwardHalf,
        d2h_name: &str,
        d2h_seg: i64,
        d2h_bytes: u64,
        t2b: Instant,
        t3: Instant,
        n_new: i64,
    ) {
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        let ForwardHalf {
            t_stream0,
            t0,
            t0b,
            t1,
            t2,
            h2d_note,
            h2d_bytes,
        } = half;
        eprintln!(
            "[sdsc-phase] stream_recreate={:.2}  preamble_h2d={:.2}  compute={:.2}  \
             {d2h_name}_d2h={:.2}  predict_total={:.2} ms  (n_new={})\n\
             [sdsc-phase]   h2d: {}KB over{} | enqueue={:.2} sync={:.2}\n\
             [sdsc-phase]   d2h: seg{} {}KB | enqueue={:.2} sync={:.2}",
            ms(*t_stream0, *t0),
            ms(*t0, *t1),
            ms(*t1, *t2),
            ms(*t2, t3),
            ms(*t_stream0, t3),
            n_new,
            h2d_bytes >> 10,
            if h2d_note.is_empty() {
                " (nothing)".to_string()
            } else {
                h2d_note.clone()
            },
            ms(*t0, *t0b),
            ms(*t0b, *t1),
            d2h_seg,
            d2h_bytes >> 10,
            ms(*t2, t2b),
            ms(t2b, t3),
        );
    }

    /// The shared post-readback tail: RB attribution and the one-shot flex-leak probe — the
    /// two things every step does after its synchronize, extracted with `forward_half` so the
    /// token step keeps them by construction rather than by copy.
    fn step_epilogue(&mut self, d: &Diag) {
        if d.rb_time {
            self.rb_attribute();
        }

        // FLEX-LEAK PROBE (one-shot): after the N-th forward, tear the stream + the whole runtime
        // down. This BREAKS the session (no re-prepare) — measure, then stop the serve.
        if let Some(n) = d.reset_probe {
            let mut t = timers();
            t.forwards += 1;
            if t.forwards == n {
                let before = vmrss_kb();
                self.stream = None;
                let _ = crate::runtime::runtime_reset();
                let after = vmrss_kb();
                eprintln!(
                    "[sdsc-leak-probe] fwd={n} VmRSS {before} -> {after} kB (freed {} kB)",
                    before - after
                );
            }
        }
    }

    /// Run ONE forward over `n_new` NEW positions appended at `[seq_pos..seq_pos+n_new)`: refill
    /// the bound activations, launch prefix → body×iters → suffix, and copy the logits row back.
    /// `logits` says what the caller does with this forward's logits ([`Logits`]): a prompt chunk
    /// whose logits nobody reads skips the suffix and the read-back both.
    pub fn predict(&mut self, n_new: NewTokens, seq_pos: SeqPos, logits: Logits<'_>) -> Result<()> {
        let (suffix, out_logits) = match logits {
            Logits::Discarded => (SuffixLaunch::Skipped, None),
            Logits::ReadBack(out) => (SuffixLaunch::Launched, Some(out)),
            Logits::LeftOnDevice => (SuffixLaunch::Launched, None),
        };
        let d = Diag::get();
        let n_new = NewTokens(n_new.0.max(1));
        let half = self.forward_half(n_new, seq_pos, suffix, d)?;

        // ── D2H the logits' segment, sen→IEEE the logits slice at its baked offset. ──
        // Skipped entirely when the caller wants no logits: only the LAST prefill chunk asks for
        // them (the prefill suffix runs its lm-head tail at m=1 over the chunk's last row), so every
        // earlier chunk would transfer ~3 MB of untouched memory (~1.7 ms at the measured DMA rate).
        //
        // NOTE the read is at the logits placement's offset with NO row indexing — exactly right for
        // the folded tail, which writes ONE [1,vocab] row at offset 0 whatever `n_new` is, and
        // unchanged for decode (n_new=1).
        let lp = self
            .places
            .get(&self.logits_id)
            .cloned()
            .ok_or_else(|| anyhow!("logits tensor {} has no placement", self.logits_id))?;
        let (mut d2h_seg, mut d2h_bytes) = (-1i64, 0u64);
        if out_logits.is_some() {
            let seg = SegIdx::checked(lp.segment as i64)
                .ok_or_else(|| anyhow!("logits segment {} out of range", lp.segment))?;
            d2h_seg = lp.segment as i64;
            let addr = self.seg_addr[seg.get()].as_ref().unwrap();
            let tot = addr.total_size().as_u64();
            // ⭐ THE LOGITS **PLACEMENT**, NOT ITS SEGMENT'S ALLOCATION — `d2h_bytes =
            // addr.total_size()` copied the WHOLE region every token to obtain one [1,vocab] row.
            //
            // ⛔ THAT COST WAS INVISIBLE BECAUSE IT IS NOT WRITTEN DOWN ANYWHERE: it is whatever
            // else the layout happens to pack into the logits segment. Today that is nothing (~102
            // KB, so the whole-region read was ~the row), which is exactly why a whole-region read
            // survived review. But it is what disqualified seg4 as the weight-tail spill slot in
            // `spill_weight_tail`: a 419 MB tenant there would have cost ~240 ms PER TOKEN, with
            // nothing about the placement looking wrong. Reading the PLACEMENT makes the transfer a
            // property of the logits row (vocab × 2 B) instead of a property of the segment's other
            // tenants, so no future packing decision can put bytes on this critical path.
            //
            // Two facts make the window safe, and both are guarded at cargo-build in
            // `audit_layout_addresses` rather than trusted here:
            //   * `lp.offset % 128 == 0` — a chunk offset is written in flits, and flex's
            //     `each_chunk_128_byte_aligned` rejects an unaligned one.
            //   * `lp.offset + lp.size <= segment_bytes[seg]` — and `segment_bytes` is an align128
            //     accumulation, so rounding the LENGTH up to a flit stays inside the region. The
            //     round-up is not optional: `Flits::from_bytes` TRUNCATES (`>> 7`), so a length that
            //     is not a multiple of 128 would transfer a short row and lose the tail logits.
            // The destination is the SAME `[lp.offset, …)` window of the shadow the sen→IEEE convert
            // below reads from, so the row lands where the caller already looks for it.
            //
            // ⭐ AND ONLY THE ROW THE CONVERT BELOW READS. The convert takes `vocab` elements at
            // `lp.offset` — the folded tail's one `[1, vocab]` row — but a prefill rung's logits
            // PLACEMENT is `[rows, vocab]` wide: 25.6 MB at granite-2b's 256-row rung, read back to
            // convert 100 KB of it (15.1 ms on a fresh session, 3.6 ms warm, measured). The window
            // is the row, so the bytes converted are the same bytes; decode's placement already is
            // one row, so it is unchanged.
            let lo = lp.offset;
            let row = (self.vocab.0 as u64 * 2).min(lp.size);
            let len = crate::lower_subtile_tape_to_superdsc::align128(row).min(tot - lo);
            // A multi-chunk region has no addressable sub-window (`chunk0` fails); fall back to the
            // whole-region read, which is what this always did.
            let win = (len < tot)
                .then(|| addr.window(ByteOffset(lo), ByteSize(len)))
                .flatten();
            let (src, a, b) = match win.as_ref() {
                Some(w) => (w, lo, lo + len),
                None => (addr, 0, tot),
            };
            d2h_bytes = b - a;
            let host = &mut self.seg_host[seg.get()][a as usize..b as usize];
            // ON THE COMPUTE PIPELINE (#544): the logits readback is the LAST op of every
            // step, right after the step's COMPUTE ops. Riding ASYNC_DMAO, its switch
            // barrier drained the COMPUTE deque — the readback submission HOST-BLOCKED until
            // the whole step's compute work completed. On COMPUTE the barrier drains the
            // (empty) DMA deques instead, the readback submits immediately, and the queue's
            // FIFO — readback CB behind the step's compute CBs on the same queue — carries
            // the ordering the host-side wait used to.
            self.stream
                .as_ref()
                .unwrap()
                .d2h_on_compute(host, src)
                .map_err(|e| anyhow!("logits D2H: {e}"))?;
        }
        let t2b = Instant::now();
        self.stream_ref()?
            .synchronize()
            .map_err(|e| anyhow!("predict: sync: {e}"))?;

        if d.phase_time {
            let t3 = Instant::now();
            Self::print_step_phase(&half, "logits", d2h_seg, d2h_bytes, t2b, t3, n_new.0);
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
            report_phase_timers(ms(half.t1, half.t2));
        }

        if let Some(out) = out_logits {
            let seg = SegIdx::checked(lp.segment as i64).unwrap();
            let elems = self.vocab.0.min(out.len() / 2);
            let src = &self.seg_host[seg.get()][lp.offset as usize..];
            sen_convert::sen_to_ieee_bytes(&src[..elems * 2], &mut out[..elems * 2]);
        }

        self.step_epilogue(d);
        Ok(())
    }

    /// Refill per-step ACTIVATIONS into their segment's host staging at the baked offset
    /// (IEEE→sen), and re-H2D the dirtied segments. Weights (seg1) and KV (seg2) stay RESIDENT —
    /// never re-uploaded, which is the whole perf point.
    fn refill_activations(&mut self, d: &Diag) -> Result<(String, u64)> {
        // ⛔ A WHOLE-SEGMENT H2D IS THE O(cap) DECODE CLIFF. seg3 is 8425 KB and a bs=8 decode step
        // writes a small fraction of it, so nearly every byte uploaded is unchanged — a flat
        // ~1.5 ms no amount of shrinking the mask could reach, because the transfer is sized by the
        // SEGMENT and not by the content.
        //
        // Each bind goes out in its own ALIGNED COVERING WINDOW ([`h2d_run`]). It degrades to the
        // whole segment whenever a window cannot be formed, so the worst case is exactly the old
        // behaviour.
        let mut seg_dirty = [false; NUM_SEGMENTS];
        // ⛔ ONE COVERING WINDOW IS NOT ENOUGH — MEASURED. The bound tensors are SCATTERED across
        // seg3 (model constants low, masks high), so min-to-max spanned the whole 8425 KB and
        // narrowed nothing. Keep each tensor's own range and merge only what actually overlaps.
        let mut seg_runs: [Vec<(u64, u64)>; NUM_SEGMENTS] = [const { Vec::new() }; NUM_SEGMENTS];

        // Collect first (the borrow of `bound` and the mutation of `seg_host` cannot overlap).
        let mut writes: Vec<(usize, u64, &[u8])> = Vec::new();
        for (nm, src) in &self.bound {
            if self.weight_ids.contains(nm) {
                continue; // weight, resident
            }
            if !refill_sends(&self.keyed, nm) {
                continue; // keyed, and the device already holds this key's bytes
            }
            let Some(p) = self.places.get(nm) else {
                continue; // no placement → not a program input, skip
            };
            let Some(seg) = SegIdx::checked(p.segment as i64) else {
                continue;
            };
            // OVER-BIND GUARD: the host must never write more bytes into a tensor than its OWN
            // placement reserves. Checking the SEGMENT bound alone passes happily while silently
            // overwriting the NEXT tensors — which is exactly how a one-stick ATTN_ZERO reservation
            // vs an [mq_pad,hd] bind zeroed the fp8 clamp constants at mq_pad=128 and made every
            // fp8 projection emit 0 (all-zero prefill KV, fluent but off-topic output). MUST be `>`
            // and not `!=`: several activations legitimately bind LESS than their placement (pmask
            // reserves [nqh*mq, cap] but binds a single broadcast row).
            if src.len() as u64 > p.size {
                bail!(
                    "predict: activation '{nm}' binds {} B into a {} B placement (seg{} off {}) — \
                     REFUSING, this would corrupt the following tensors",
                    src.len(),
                    p.size,
                    p.segment,
                    p.offset
                );
            }
            if p.offset as usize + src.len() > self.seg_host[seg.get()].len() {
                bail!("predict: activation '{nm}' overflows seg{}", p.segment);
            }
            writes.push((seg.get(), p.offset, src.as_slice()));
            seg_dirty[seg.get()] = true;
            seg_runs[seg.get()].push(h2d_run(p.offset, src.len() as u64));
            if d.phase_time {
                eprintln!(
                    "[sdsc-bind] '{nm}' -> seg{} off={} inputsize={:.3} MB",
                    p.segment,
                    p.offset,
                    src.len() as f64 / 1048576.0
                );
            }
        }
        // ⭐ AND EVERY ACTIVATION ALREADY CONVERTED INTO THE SHADOW (`bind_input_f32`): its range
        // goes out this step exactly as a `bound` one's does — same window law, every step — and
        // nothing converts it again. Its placement was checked where it was written.
        for (nm, &len) in &self.staged {
            let Some(p) = self.places.get(nm) else {
                continue;
            };
            let Some(seg) = SegIdx::checked(p.segment as i64) else {
                continue;
            };
            seg_dirty[seg.get()] = true;
            seg_runs[seg.get()].push(h2d_run(p.offset, len));
            if d.phase_time {
                eprintln!(
                    "[sdsc-bind] '{nm}' -> seg{} off={} inputsize={:.3} MB (converted at bind)",
                    p.segment,
                    p.offset,
                    len as f64 / 1048576.0
                );
            }
        }
        // The convert cannot borrow `self.bound` and `self.seg_host` at once, so the sources move
        // out for the duration — a `Vec` swap, not a copy of the bytes.
        let bound = std::mem::take(&mut self.bound);
        for (nm, src) in &bound {
            if self.weight_ids.contains(nm) {
                continue;
            }
            if !refill_sends(&self.keyed, nm) {
                continue;
            }
            let Some(p) = self.places.get(nm) else {
                continue;
            };
            let Some(seg) = SegIdx::checked(p.segment as i64) else {
                continue;
            };
            let dst = &mut self.seg_host[seg.get()][p.offset as usize..][..src.len()];
            // ⛔ A RAW BIND IS COPIED, NOT RE-ENCODED. `ieee_to_sen_bytes` treats its source as fp16
            // pairs; for the gather's int32 index table that rewrites every entry into a different
            // valid block number. See [`Self::raw_ids`].
            if self.raw_ids.contains(nm) {
                dst.copy_from_slice(src);
            } else {
                sen_convert::ieee_to_sen_bytes(src, dst);
            }
        }
        self.bound = bound;
        // This step's keyed binds are in the shadow and in the runs below; the next step that does
        // not re-bind them sends nothing for them.
        for k in self.keyed.values_mut() {
            k.fresh = false;
        }

        let mut note = String::new();
        let mut total = 0u64;
        for i in 0..NUM_SEGMENTS {
            if !seg_dirty[i] {
                continue; // only re-upload activation segments
            }
            // NEVER H2D a BORROWED segment: its device region belongs to another session and this
            // session's shadow is a stale zero-fill, so uploading it would wipe the owner's live
            // data (for seg2 that is the whole KV cache).
            if self.seg_aliased[i] {
                continue;
            }
            let addr = self.seg_addr[i].as_ref().unwrap();
            let tot = addr.total_size().as_u64();
            // THE MERGED ALIGNED RUNS, or the whole segment when they cannot be used.
            let mut runs: Vec<(u64, u64)> = Vec::new();
            if addr.is_single_chunk() {
                let mut r = seg_runs[i].clone();
                r.sort_unstable();
                for (a, b) in r {
                    let b = b.min(tot);
                    if b <= a {
                        continue;
                    }
                    match runs.last_mut() {
                        Some(last) if a <= last.1 => last.1 = last.1.max(b),
                        _ => runs.push((a, b)),
                    }
                }
                // If the runs cover essentially everything, one whole-segment DMA is cheaper.
                let covered: u64 = runs.iter().map(|(a, b)| b - a).sum();
                if runs.is_empty() || covered * 8 >= tot * 7 {
                    runs.clear();
                }
            }
            let sent = if runs.is_empty() {
                tot
            } else {
                runs.iter().map(|(a, b)| b - a).sum()
            };
            total += sent;
            if d.phase_time {
                note.push_str(&format!(" seg{i}={}KB", sent >> 10));
            }
            let stream = self.stream.as_ref().unwrap();
            // ON THE COMPUTE PIPELINE (#544): the preamble is the FIRST op of every step with
            // the step's COMPUTE ops right behind it. Riding ASYNC_DMAI made the compute
            // submission HOST-BLOCK on the preamble's fence (its switch barrier drained the
            // DMAI deque), and made the preamble itself block on the PREVIOUS step's readback
            // (its own barrier drained the DMAO deque). On COMPUTE the junction isn't a
            // switch at all, the remaining cross-step barrier drains deques that hold
            // nothing, and the queue's own FIFO orders the preamble CB ahead of the step's
            // compute CBs — the whole step submits with no host-side waits.
            if runs.is_empty() {
                stream
                    .h2d_on_compute(whole_shadow(&self.seg_host[i], addr, i)?, addr)
                    .map_err(|e| anyhow!("preamble H2D seg{i}: {e}"))?;
            } else {
                for (a, b) in runs {
                    let win = addr
                        .window(ByteOffset(a), ByteSize(b - a))
                        .ok_or_else(|| anyhow!("seg{i} window {a}..{b} unaddressable"))?;
                    stream
                        .h2d_on_compute(&self.seg_host[i][a as usize..b as usize], &win)
                        .map_err(|e| anyhow!("preamble H2D seg{i} window: {e}"))?;
                }
            }
        }
        Ok((note, total))
    }

    /// The forward itself: prefix → body×iters → suffix (rolled), or the single body. `suffix` is
    /// [`SuffixLaunch::Skipped`] only for a forward whose logits are discarded ([`Logits`]).
    fn launch_forward(&mut self, seq_pos: SeqPos, suffix: SuffixLaunch) -> Result<()> {
        let zero = [0u64; NUM_SEGMENTS];
        let Some(rolled) = self.rolled.as_ref() else {
            // Not rolled: one body over the whole model. Pages the RESIDENT PREFIX spans; the
            // selector also picks the FOLD-FUSED twin when the context fits ONE page, which is the
            // baseline's group count (3 vs the split body's 5) — selecting the split body
            // regardless cost two extra launches per layer, ~1 ms/token, for a fold launched once.
            // ⭐ THE SAME DOOR THE HOST ASKS THROUGH — [`Executor::step_body`], from the same write slot.
            // Two call sites here and one on the host, all three taking `seq_pos` and nothing else, so
            // the body the host staged for IS the body launched.
            let n_fold_pages = self.n_fold_pages(seq_pos);
            let sel = self.step_body(seq_pos)?.sel;
            let base = self.page_base_for_write(seq_pos);
            // UNROLLED: one body over the whole model (class 0 — an unrolled bundle has no class
            // split), and an unrolled bundle has no layer boundary to bank on — its weights are one
            // region by construction (`bank_weight_segment` refuses to bank without layer classes),
            // so bank 0 is the only bank there is.
            let ls = self.list_sel_for(sel, 0);
            return self.launch_ops(ls, zero, seq_pos, base, n_fold_pages, 0);
        };
        let pre_bank = rolled.prefix_weight_bank;
        let suf_bank = rolled.suffix_weight_bank;

        self.launch_ops(ListSel::Prefix, zero, SeqPos(0), Bytes(0), 1, pre_bank)?;

        // sk_bucket ladder: pick this token's decode body rung by its KV length (valid_len =
        // seq_pos+1 — positions [0..seq_pos] are filled, including the token just written to KV
        // before attention). The same rung serves every layer this step; only the swept extents
        // differ, and the resident KV is shared.
        let n_fold_pages = self.n_fold_pages(seq_pos);
        let sel = self.step_body(seq_pos)?.sel;
        let base = self.page_base_for_write(seq_pos);
        // ⭐⭐ THE LAUNCH TABLE IS THE WHOLE PER-LAYER ARITHMETIC. One row per layer, in true layer
        // order, each naming its body's CLASS and the segment shifts the launch applies — no
        // `v·stride`, no `v / layers_per_bank`, no class base to add. The bake PROVED every row
        // against the placements that were baked (see the launch-table guard in
        // `lower_subtile_tape_to_ktir`), so this loop cannot disagree with the layout: it reads
        // the table the layout was checked against.
        //
        // A class-split model runs its bodies OUT of tape order here — the table is TRUE ORDER
        // (sorted by absolute layer id), which a single-class roll degenerates to
        // `[(0,0), (0,1), …]`: the sequence the `for v in 0..iters` loop used to run.
        // COPIED OUT before the loop: each launch takes `&mut self`, so the table cannot be
        // borrowed out of `self.rolled` across them. It is one small tuple per layer — the copy
        // is the table's bytes, not a graph.
        let launches: Vec<_> = self
            .rolled
            .as_ref()
            .expect("checked above")
            .launches
            .clone();
        for (body, wbank, woff, kvoff) in launches {
            let mut off = [0u64; NUM_SEGMENTS];
            off[SEG_WEIGHT.get()] = woff;
            off[SEG_KV.get()] = kvoff;
            let ls = self.list_sel_for(sel, body);
            // The WRITE lands in the page holding seq_pos; the FOLD covers every page the RESIDENT
            // PREFIX [0, seq_pos) spans — zero of them when there is no prefix.
            self.launch_ops(ls, off, seq_pos, base, n_fold_pages, wbank as usize)?;
        }
        // NO HOST ROUTING: the loop-carried residual + the body→suffix seam thread IN-PLACE via
        // emitter placement aliasing (hidden_out/suffix_in aliased onto hidden_in's resident
        // buffer). No D2H/memcpy/H2D — the residual lives on-device across iterations.
        //
        // ⭐ A DISCARDED FORWARD STOPS HERE. The suffix (final norm → lm-head → logits scaling) only
        // READS the residual the last layer left in that shared buffer and writes the suffix's own
        // tensors, which only the logits read-back consumes: the KV every later chunk attends was
        // written by the bodies above. MEASURED (granite-2b fp16, RB_TIME): 2.73 ms per prompt
        // chunk at 512 rows, 2.11 at 256, run on every chunk though only the last chunk's logits
        // are ever sampled — 53 ms of a ~10k prompt's 20 chunks.
        match suffix {
            SuffixLaunch::Launched => {
                self.launch_ops(ListSel::Suffix, zero, SeqPos(0), Bytes(0), 1, suf_bank)
            }
            SuffixLaunch::Skipped => Ok(()),
        }
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  #603 PR2 2c — the replay player: one decode step as canvas-copy + patches + ONE doorbell
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// ⭐ THE PLAYER — the decode step's whole forward from a cached image:
    /// select the image by (body, page count, fold shape), let the submit
    /// path copy the canvas and stamp it (the step's patches + the
    /// pool-leased tags), and ring ONE doorbell for the step. The per-step
    /// host work is a memcpy, K word patches, N tag stamps and one
    /// submission — no per-op `launch_site`, no `BatchedCompute` build, no
    /// per-rung flush. Everything the image is keyed by, and everything the
    /// stamp writes, was proven at the image's derivation:
    /// `cb_plan::patch_check` per body row (canvas ≡ walk, patched canvas ≡
    /// fresh walk at steps the derivation never visited) and
    /// `cb_plan::parity_check` per fixed row.
    ///
    /// Returns `Ok(None)` — the caller takes the walk — for every step that
    /// is not the player's regime, with the reason stated where it fires (a
    /// per-step debug line would be noise at decode rate; the boundary reason
    /// fires once per page): a BATCHED step (two or more requests installed —
    /// PR4's images), any Diag instrument on ([`Diag::is_quiet`] — each one
    /// reads walk state), a page boundary (slot 0 of a new page — a
    /// write-base move, not a patchable one), or a page no canvas slot or
    /// off-sample room fits. There is no third path and no toggle: the
    /// production configuration (quiet diag, one request) is the player,
    /// minus one step per page.
    fn try_replay_forward(&mut self, seq_pos: SeqPos) -> Result<Option<()>> {
        if !Diag::get().is_quiet() || fold_plan::fold_requests(&self.fold) != 1 {
            return Ok(None);
        }
        let per = self.kv.page_slots.0.max(1);
        // ⭐ THE PAGE-BOUNDARY STEP WALKS. `pos % per == 0` writes slot 0 of a
        // page no image of this key ever rendered — a WRITE-BASE move, and a
        // moved write base is not affine in the patch model's scalars: a
        // shifted write target's extent is END-PINNED (`DevAddr::shifted`
        // returns the bytes remaining past the shift), so its XLAT word's
        // paddr and length fields move together and no per-byte addend
        // expresses the pair — measured on the first real bundle, where the
        // page differential's render moved paddr `+stride/2` flits and
        // length `-stride/2` flits for a one-stride base move and the solve
        // refused. That includes pos 0 — slot 0 of page 0, whose image key
        // (n = 0) has no page behind it. One step per page takes the walk;
        // every step between boundaries writes the image's own page and
        // replays.
        if self.kv.paged && seq_pos.0 % per == 0 {
            debug!(
                "[sdsc-replay] page-boundary step {} walks: slot 0 of a new page is a \
                 write-base move, not a patchable one",
                seq_pos.0
            );
            return Ok(None);
        }
        let n = self.n_fold_pages(seq_pos);
        let sel = self.step_body(seq_pos)?.sel;
        let Some(r) = cb_plan::canvas_slot(per, self.kv.paged) else {
            debug!(
                "[sdsc-replay] walking every step: no canvas slot fits a {per}-slot page, so no \
                 image can be derived for this session"
            );
            return Ok(None);
        };
        if !cb_plan::canvas_room(per, self.kv.paged, r) {
            debug!(
                "[sdsc-replay] walking every step: a {per}-slot page leaves no off-sample room \
                 around canvas slot {r}, so no slab-crossing proof fits — no image is derived"
            );
            return Ok(None);
        }
        // The canvas position: inside THIS image's page span (`(n-1)·per` is
        // the image's first page; `r ≥ 1` keeps off the boundary), or any
        // position at all for an unpaged session, whose one image covers
        // them all. The write base is the page holding it — a RESIDENT page,
        // unlike the boundary step's brand-new one, so the reference always
        // reads a page the block table already names.
        let pos_c = if self.kv.paged { (n - 1) * per + r } else { r };
        let page_c = self.page_base_for_write(SeqPos(pos_c));
        let key = StepImageKey {
            sel,
            n_fold_pages: n,
            signature: fold_signature(&self.fold, n),
        };
        let image = match self.step_images.lookup_cached(&key) {
            Some(hit) => hit,
            None => {
                let t1 = Instant::now();
                let image = self.derive_step_image(sel, n, pos_c, page_c)?;
                self.step_images.note_derived(key, image.clone(), t1);
                image
            }
        };
        // The step's scalars — the two values every moving word of the body
        // rows is affine in, measured at the position this step writes. The
        // write page is not among them and cannot move between two replayed
        // steps of one image: both write the image's own page (the boundary
        // step that would move it walked, above).
        let scalars = cb_plan::step_scalars(self.kv.paged, per, seq_pos.0);
        // The stamp closure: patches the body rows onto the backend's canvas
        // copy, then stamps the leased tags over the render-time ones. The
        // fixed rows are skipped by `follows_step` — see `RowReplay`.
        let data = std::sync::Arc::clone(&image.data);
        let batch = ComputeWireBatch {
            canvas: std::sync::Arc::clone(&data.canvas),
            patch: Box::new(move |tags, cbs| {
                for row in &data.rows {
                    if row.follows_step {
                        row.plan
                            .apply(&mut cbs[row.base..row.base + row.len], &scalars);
                    }
                }
                for (cb, tag) in cbs.iter_mut().zip(tags.iter()) {
                    cb.set_tag(*tag);
                }
            }),
        };
        let stream = self.stream_ref()?;
        let t1 = Instant::now();
        let cbs = image.data.canvas.len();
        let r = stream.compute_wires(batch, image.barrier);
        let t2 = Instant::now();
        if let Err(e) = r {
            // The image itself was proven at derivation (per-row gates above);
            // a refused submit here is capacity or shutdown, not addressing —
            // so the refusal names the image and the step, not an op site.
            return Err(anyhow::Error::new(e).context(format!(
                "replay submit refused: {cbs} CB(s) over {} launch row(s), barrier={}, \
                 sel={sel:?}, n_fold_pages={n}, step at {}",
                image.data.rows.len(),
                image.barrier,
                seq_pos.0
            )));
        }
        let mut t = timers();
        t.replay_steps += 1;
        t.replay_cbs += cbs as u64;
        t.replay_sites += image
            .data
            .rows
            .iter()
            .map(|row| row.plan.sites())
            .sum::<usize>() as u64;
        t.replay_submit_ms += (t2 - t1).as_secs_f64() * 1e3;
        // First step AND the 256 cadence: the canonical gate cell is 64 tokens,
        // so a cadence-only marker would stay silent in exactly the run that
        // must prove replay happened — the line IS the liveness proof.
        if t.replay_steps == 1 || t.replay_steps.is_multiple_of(256) {
            report_replay(&t);
        }
        Ok(Some(()))
    }

    /// Derive one step image at its first use — the row sequence is
    /// EXACTLY `launch_forward`'s (that is the merge's one law: the step's
    /// CBs, in the order the walk would submit them), each row rendered
    /// through the gates that prove it, concatenated into one canvas.
    ///
    /// The body rows render at the CANVAS position (`pos_c`/`page_c`, inside
    /// the image's own page span), not at the calling step's — the patch
    /// model's reference can be any step of the image, and a fixed reference
    /// keeps the derivation off the slab and page boundaries the calling
    /// step might sit on. The fixed rows render at their own constant
    /// `where_` (slot 0, page 0, one page), which is where the walk launches
    /// them every step whatever the body is doing.
    fn derive_step_image(
        &self,
        sel: BodySel,
        n: i64,
        pos_c: i64,
        page_c: Bytes,
    ) -> Result<StepImage> {
        let stream = self.stream_ref()?;
        // (list, where, follows-step) per launch row, mirroring
        // `launch_forward` arm by arm. An empty list launches nothing in the
        // walk (its flush stages zero CBs), so it contributes no row here.
        let mut specs: Vec<(ListSel, LaunchWhere, bool)> = Vec::new();
        let zero = [0u64; NUM_SEGMENTS];
        match self.rolled.as_ref() {
            Some(r) => {
                specs.push((
                    ListSel::Prefix,
                    LaunchWhere {
                        byte_off: zero,
                        slot_pos: SeqPos(0),
                        kv_page_base: Bytes(0),
                        n_fold_pages: 1,
                        weight_bank: r.prefix_weight_bank,
                    },
                    false,
                ));
                for (class, wbank, woff, kvoff) in r.launches.clone() {
                    let mut off = [0u64; NUM_SEGMENTS];
                    off[SEG_WEIGHT.get()] = woff;
                    off[SEG_KV.get()] = kvoff;
                    specs.push((
                        self.list_sel_for(sel, class),
                        LaunchWhere {
                            byte_off: off,
                            slot_pos: SeqPos(pos_c),
                            kv_page_base: page_c,
                            n_fold_pages: n,
                            weight_bank: wbank as usize,
                        },
                        true,
                    ));
                }
                specs.push((
                    ListSel::Suffix,
                    LaunchWhere {
                        byte_off: zero,
                        slot_pos: SeqPos(0),
                        kv_page_base: Bytes(0),
                        n_fold_pages: 1,
                        weight_bank: r.suffix_weight_bank,
                    },
                    false,
                ));
            }
            None => {
                // Unrolled: one body over the whole model, bank 0, no seams.
                specs.push((
                    self.list_sel_for(sel, 0),
                    LaunchWhere {
                        byte_off: zero,
                        slot_pos: SeqPos(pos_c),
                        kv_page_base: page_c,
                        n_fold_pages: n,
                        weight_bank: 0,
                    },
                    true,
                ));
            }
        }
        let mut canvas: Vec<ControlBlockWire> = Vec::new();
        let mut rows: Vec<RowReplay> = Vec::new();
        let mut barrier = false;
        let mut sites = 0usize;
        for (ls, where_, follows_step) in specs {
            let ops = self.list(ls);
            if ops.is_empty() {
                continue;
            }
            let label = format!("{ls:?}/replay-canvas");
            let (row_canvas, plan) = if follows_step {
                // A body row: the FULL gate — parity at the canvas position,
                // the patch table, and the off-sample proof that patched
                // canvas ≡ fresh walk at steps the derivation never visited.
                // This is the exact check the prepare gate ran under a
                // synthetic session, now under the live one.
                let (_, img) = cb_plan::patch_check(
                    ops,
                    &self.fold,
                    &self.seg_addr,
                    &self.weight_bank_addr,
                    &where_,
                    &label,
                    stream,
                )?;
                (img.canvas, img.plan)
            } else {
                // A fixed row: the walk launches it at this SAME `where_`
                // every step, so its replay target is its own canvas,
                // unchanged — parity at that `where_` is the whole
                // correctness statement. Its patch table is empty BY
                // CONSTRUCTION (a constant `where_` makes every
                // step-differential zero), so none is derived: probing one
                // would measure a counterfactual no replay executes — the
                // measured failure mode the prepare gate's scope law also
                // encodes (its patch leg skips the prefix/suffix lists).
                cb_plan::parity_check(
                    ops,
                    &self.fold,
                    &self.seg_addr,
                    &self.weight_bank_addr,
                    &where_,
                    &label,
                    stream,
                )?;
                let img = cb_plan::render_row(
                    ops,
                    &self.fold,
                    &self.seg_addr,
                    &self.weight_bank_addr,
                    &where_,
                    stream,
                )?;
                (img.canvas, img.plan)
            };
            barrier |= plan.barrier;
            sites += plan.sites();
            rows.push(RowReplay {
                base: canvas.len(),
                len: row_canvas.len(),
                follows_step,
                plan,
            });
            canvas.extend(row_canvas);
        }
        if canvas.is_empty() {
            bail!(
                "replay image derivation REFUSED: sel={sel:?} n_fold_pages={n} produced zero \
                 CBs — the walk it mirrors launches nothing, which no body selection can do"
            );
        }
        // The image's own first CB carries the same EDEP1 the submit path
        // stamps (`ResponseWorker::prepare_cbs`, the vendor `Finalize` port at
        // control_block_stream.cpp:423): the replay patch rewrites tags in
        // place and never touches the CTRL mode bits, so what derivation
        // stores is what every replay of this image submits — a cached image
        // must not hold the EDEP0 its rows were rendered with. The
        // submit-side stamp makes the two agree rather than this one being
        // load-bearing (a double-set of the same value is fine; a miss is the
        // defect).
        canvas[0].set_edep(1);
        debug!(
            "[sdsc-replay] derived image sel={sel:?} n_fold_pages={n}: {} CB(s) over {} \
             launch row(s), {sites} patch site(s), barrier={barrier}",
            canvas.len(),
            rows.len()
        );
        Ok(StepImage {
            data: std::sync::Arc::new(StepImageData {
                canvas: std::sync::Arc::new(canvas),
                rows,
            }),
            barrier,
        })
    }

    /// How many pages the RESIDENT PREFIX `[0, seq_pos)` spans.
    /// THE PAGES OF **RESIDENT PREFIX** THIS STEP FOLDS OVER — `ceil(seq_pos / per)`, EXCLUSIVE of the
    /// slot being written, because the page that slot lands in is the BODY's job and not a fold pass.
    ///
    /// ⛔ DO NOT "FIX" THIS TO `ceil((seq_pos + 1) / per)`. It looks one short — a row writing slot
    /// 2048 does hold 2049 slots — and it is not: MEASURED 2026-08-12, that change REGRESSED the
    /// ragged gate from 3/6 to 1/6 (`Rome` and `Cario` became `'The provided text does not contain…'`,
    /// `Nairobi` became `Nariobi`), in both batch orders. The extra pass folds a page the body already
    /// accounts for.
    ///
    /// The host's [`FoldPages`] staging must therefore match THIS count, which is what
    /// `FoldPages::resident_before` is for. It used `covering` (= `ceil((slot + 1) / per)`), so at every
    /// exact page multiple the host staged one block more than the launch walks and the fold refused:
    ///
    ///   `fold: 8 pass(es) REFUSED — the launch walks 8 page(s) but the host STAGED the mask for 9`
    ///
    /// That fired on the BATCHED path only (a lone request takes the solo path and stages no batched
    /// mask, which is why single-request sweeps across 768 all passed and the 76..1840 gate probes
    /// never hit it), and it read as a "2048 wall" only because `max_num_batched_tokens`' 2048 default
    /// truncated long prompts to exactly 2048 — a page multiple.
    fn n_fold_pages(&self, seq_pos: SeqPos) -> i64 {
        if self.kv.paged {
            let per = self.kv.page_slots.0.max(1);
            (seq_pos.0 + per - 1) / per
        } else {
            1
        }
    }

    /// The byte base of the page this step's WRITE lands in (request 0's — a batched step installs
    /// per-request maps and the per-op request tag selects among them inside `fold_plan`).
    fn page_base_for_write(&self, seq_pos: SeqPos) -> Bytes {
        if !self.kv.paged {
            return Bytes(0);
        }
        let lp = seq_pos.0 / self.kv.page_slots.0.max(1);
        RowPage::of(
            &self.fold,
            RowIdx::from_launch_row(0),
            PageOrd::of_index(lp),
        )
        .map_or(Bytes(0), |rp| fold_plan::page_base_bytes(&self.fold, rp))
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Selftest entry points (opt-in bisection; NOT a run path)
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Bind synthetic activations, then run ONLY the prefix program, leaving every tid resident for
    /// read-back so the caller can compare each to the `eval_dag` golden.
    pub fn run_prefix_only(&mut self) -> Result<()> {
        if !self.prepared {
            bail!("run_prefix_only before prepare");
        }
        self.refill_activations(Diag::get())?;
        let zero = [0u64; NUM_SEGMENTS];
        let (sel, bank) = match self.rolled.as_ref() {
            Some(r) => (ListSel::Prefix, r.prefix_weight_bank),
            None => (ListSel::Body(0), 0),
        };
        self.launch_ops(sel, zero, SeqPos(0), Bytes(0), 1, bank)?;
        self.stream_ref()?
            .synchronize()
            .map_err(|e| anyhow!("run_prefix_only sync: {e}"))
    }

    /// The FULL forward for the selftest: prefix → body×iters (per-layer advance + in-place hidden
    /// threading) → suffix, leaving every tid resident. Exercises what prefix-only cannot: the
    /// per-layer weight/KV advance, the body→suffix seam, and the suffix ops (final norm + tiled
    /// lm_head). Only a ROLLED bundle has those seams, so an unrolled one reports `false`.
    pub fn run_full(&mut self) -> Result<bool> {
        if !self.prepared {
            bail!("run_full before prepare");
        }
        if self.rolled.is_none() {
            return Ok(false);
        }
        self.refill_activations(Diag::get())?;
        self.launch_forward(SeqPos(0), SuffixLaunch::Launched)?;
        self.stream_ref()?
            .synchronize()
            .map_err(|e| anyhow!("run_full sync: {e}"))?;
        Ok(true)
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Read-back / segment surgery
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// D2H a whole tensor segment into its host shadow, CLAMPED to the shadow's allocated size.
    ///
    /// The two sizes genuinely differ on the paged path: prepare sizes seg2's DEVICE region for the
    /// whole pool but leaves its host shadow at ONE page ("NO HOST MIRROR OF THE KV POOL"), so a
    /// whole-region D2H would write pool-many pages into a one-page buffer. When the shadow is
    /// smaller, only its leading window transfers, and the clamped-vs-requested sizes are printed —
    /// a truncated readback is visible, never silent. Returns the bytes transferred.
    fn d2h_seg_clamped(&mut self, seg: SegIdx) -> Result<usize> {
        let i = seg.get();
        let dev = self.seg_addr[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))?
            .total_size()
            .as_u64();
        let host = self.seg_host[i].len() as u64;
        // The shadow, the region and the stream are three disjoint fields of `self`; splitting the
        // borrow here is what lets the DMA write into a real `&mut [u8]` instead of a pointer
        // conjured from a shared reference.
        let Executor {
            seg_addr,
            seg_host,
            stream,
            ..
        } = self;
        let addr = seg_addr[i].as_ref().expect("checked above");
        let stream = stream
            .as_ref()
            .ok_or_else(|| anyhow!("no stream — prepare was not called"))?;
        if host >= dev {
            stream
                .d2h(&mut seg_host[i][..dev as usize], addr)
                .map_err(|e| anyhow!("seg{i} D2H: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| anyhow!("seg{i} D2H sync: {e}"))?;
            return Ok(dev as usize);
        }
        if !addr.is_single_chunk() {
            bail!(
                "read seg{i}: host shadow {host} B < device region {dev} B and the region is \
                 multi-chunk — readback REFUSED (no addressable leading window)"
            );
        }
        if host > 0 {
            let win = addr
                .window(ByteOffset(0), ByteSize(host))
                .ok_or_else(|| anyhow!("seg{i} window unaddressable"))?;
            stream
                .d2h(&mut seg_host[i][..host as usize], &win)
                .map_err(|e| anyhow!("seg{i} D2H: {e}"))?;
            stream
                .synchronize()
                .map_err(|e| anyhow!("seg{i} D2H sync: {e}"))?;
        }
        debug!(
            "[sdsc-superdsc] read seg{i} CLAMPED to its host shadow: {host} of {dev} B D2H'd \
             (readable window = the leading {host} B; the rest has no host mirror)"
        );
        Ok(host as usize)
    }

    /// D2H tensor `id`'s segment and convert its `[offset, size)` slice to IEEE-f16 elements.
    /// Clamped to `cap_elems` and to what the transfer actually brought back.
    pub fn read_tensor(&mut self, id: bundle::PlaceId, cap_elems: usize) -> Result<Vec<u8>> {
        let name = id;
        let p = self
            .places
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!("unknown tensor {id}"))?;
        let seg = SegIdx::checked(p.segment as i64)
            .ok_or_else(|| anyhow!("tensor '{name}' segment {} out of range", p.segment))?;
        // ⛔ A BANKED TENSOR IS NOT IN ITS SEGMENT'S REGION. `d2h_seg_clamped` reads `seg_addr[seg]`,
        // which for a weight in bank ≥ 1 is a DIFFERENT region — the read would return bank 0's bytes
        // at this offset and the caller would compare a golden against another tensor entirely. This
        // is a selftest/debug reader, so refusing is free; making it bank-aware would mean giving
        // every segment reader the bank coordinate for a path no forward uses.
        if p.bank != 0 {
            bail!(
                "read_tensor('{name}'): it is in weight bank {}, and a segment read addresses only \
                 bank 0's region — REFUSING rather than returning another bank's bytes",
                p.bank
            );
        }
        let got = self.d2h_seg_clamped(seg)?;
        let mut elems = (p.size / 2) as usize;
        elems = elems.min(cap_elems);
        // The convert may only read the bytes the transfer actually brought back: a placement
        // reaching past the shadow's window yields what exists, and says how much.
        let avail = (got as u64).saturating_sub(p.offset);
        if elems as u64 * 2 > avail {
            debug!(
                "[sdsc-superdsc] read_tensor({name}): {elems} elems requested at seg{} off {} but \
                 only {got} B of the segment are host-readable — CLAMPED to {} elems",
                p.segment,
                p.offset,
                avail / 2
            );
            elems = (avail / 2) as usize;
        }
        if elems == 0 {
            return Ok(Vec::new()); // a wholly-unreadable placement also has no valid base
        }
        let src = &self.seg_host[seg.get()][p.offset as usize..][..elems * 2];
        let mut out = vec![0u8; elems * 2];
        sen_convert::sen_to_ieee_bytes(src, &mut out);
        Ok(out)
    }

    /// The host shadow's byte size for `seg` — what to size a [`Executor::read_seg`] buffer at.
    pub fn seg_bytes(&self, seg: SegIdx) -> usize {
        self.seg_host[seg.get()].len()
    }

    /// ZERO a whole tensor segment ON THE DEVICE, in chunks, from a small host buffer.
    ///
    /// Exists so zeroing the KV pool does not require a host mirror of it. The pool has to start
    /// zeroed — a page handed to a request otherwise holds whatever was in device memory, and decode
    /// computes q·k over the WHOLE page before the validity mask applies, so uninitialised bytes go
    /// through the arithmetic as denormals and NaN. The old route (read_seg → write_seg) addressed a
    /// host allocation the size of the entire pool: 960 MB, and 3.84 GB at a 1024-slot page. This
    /// walks the segment with an 8 MB buffer, and the device sees the same zeros.
    ///
    /// The walk is [`DevAddr::windows`], so a multi-chunk (interleaved) region is zeroed chunk by
    /// chunk with the same 8 MB windows — every byte the region owns, none it does not.
    pub fn zero_seg(&mut self, seg: SegIdx) -> Result<()> {
        let i = seg.get();
        let addr = self.seg_addr[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))?;
        let total = addr.total_size().as_u64();
        if total == 0 {
            return Ok(());
        }
        const CHUNK: NonZeroU64 = NonZeroU64::new(8 << 20).unwrap();
        let zeros = vec![0u8; CHUNK.get().min(total) as usize];
        let stream = self.stream_ref()?;
        for win in addr.windows(CHUNK) {
            let sz = win.total_size().as_u64() as usize;
            stream
                .h2d(&zeros[..sz], &win)
                .map_err(|e| anyhow!("zero H2D (seg{i}): {e}"))?;
            // The window must outlive the transfer.
            stream
                .synchronize()
                .map_err(|e| anyhow!("zero sync (seg{i}): {e}"))?;
        }
        Ok(())
    }

    /// D2H the WHOLE tensor segment (raw sen-fp16 bytes, NO conversion).
    pub fn read_seg(&mut self, seg: SegIdx) -> Result<Vec<u8>> {
        let got = self.d2h_seg_clamped(seg)?;
        Ok(self.seg_host[seg.get()][..got.min(self.seg_host[seg.get()].len())].to_vec())
    }

    /// Overwrite the FIRST `data.len()` bytes of a tensor segment and H2D exactly that window.
    ///
    /// ⛔ WHY NOT [`Executor::write_seg`]: that one is a WHOLE-segment copy, and the thing this
    /// exists for — handing a BORROWING session its own copy of the spilled weight tail — crosses
    /// two bundles whose extents for that segment DIFFER. The spill slot is also an intermediate
    /// COLOR segment, and a prefill rung's intermediates are wider than decode's, so a whole-segment
    /// copy is not merely wasteful but unrepresentable: MEASURED, 421,257,216 source bytes against a
    /// 432,218,112 B segment, which `write_seg`'s equality guard correctly refuses.
    ///
    /// The tail occupies `[0, len)` by construction (`spill_weight_tail` places it at offset 0 for
    /// exactly this reason), so writing that prefix and leaving the rest of the shadow as `prepare`
    /// zeroed it is both the correct result and the smaller transfer.
    pub fn write_seg_prefix(&mut self, seg: SegIdx, data: &[u8]) -> Result<()> {
        let i = seg.get();
        if data.len() > self.seg_host[i].len() {
            bail!(
                "write_seg_prefix(seg{i}): {} B does not fit this session's {} B segment",
                data.len(),
                self.seg_host[i].len()
            );
        }
        self.seg_host[i][..data.len()].copy_from_slice(data);
        let addr = self.seg_addr[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))?;
        let win = addr
            .window(ByteOffset(0), ByteSize(data.len() as u64))
            .ok_or_else(|| anyhow!("seg{i} window 0..{} unaddressable", data.len()))?;
        let stream = self.stream_ref()?;
        stream
            .h2d(&self.seg_host[i][..data.len()], &win)
            .map_err(|e| anyhow!("write_seg_prefix H2D seg{i}: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow!("write_seg_prefix sync seg{i}: {e}"))
    }

    /// Overwrite the WHOLE tensor segment from raw sen-fp16 bytes + H2D it. The counterpart of
    /// [`Executor::read_seg`].
    pub fn write_seg(&mut self, seg: SegIdx, data: &[u8]) -> Result<()> {
        let i = seg.get();
        let n = data.len().min(self.seg_host[i].len());
        self.seg_host[i][..n].copy_from_slice(&data[..n]);
        let addr = self.seg_addr[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))?;
        let src = whole_shadow(&self.seg_host[i], addr, i)?;
        let stream = self.stream_ref()?;
        stream
            .h2d(src, addr)
            .map_err(|e| anyhow!("write_seg H2D: {e}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow!("write_seg sync: {e}"))
    }

    /// Point THIS session's segment `seg` at `owner`'s DEVICE region so the two SHARE it outright
    /// and no copy is ever needed.
    ///
    /// Used to alias the batched-PREFILL session's seg2 (resident KV) onto the DECODE session's.
    /// Both bundles place seg2 identically BY CONSTRUCTION — `compute_bundle_layout` sizes kc/vc at
    /// `nqh*hd*cap*2` and the resident kct at `nkvh*hd*cap*2` per AttnDecode node in node order, and
    /// not one of those terms depends on the query-row count. So the prefill forward can write the
    /// KV directly where decode will read it, deleting a 188,743,680 B per-prompt D2H+H2D round trip
    /// (~125-133 ms including the two 90 MiB host memcpys).
    ///
    /// Only THIS session's descriptor is replaced; `owner` is untouched, so aliasing prefill onto
    /// decode leaves the decode session bit-for-bit unchanged. MUST be called AFTER both sessions'
    /// prepare (which zero-inits and H2Ds every segment).
    pub fn alias_seg_from(&mut self, owner: &Executor, seg: SegIdx) -> Result<()> {
        let i = seg.get();
        let (Some(dst), Some(src)) = (self.seg_addr[i].as_ref(), owner.seg_addr[i].as_ref()) else {
            bail!("alias_seg(seg{i}): a session has no region for it");
        };
        // A session that DECLARED this segment borrowed holds a 128 B placeholder on purpose, so
        // its region size is expected to differ — skip only that comparison for it. The two checks
        // that actually prove the layouts match are untouched.
        if self.segment_bytes[i] != owner.segment_bytes[i]
            || (!self.seg_borrowed[i] && dst.total_size() != src.total_size())
        {
            bail!(
                "alias_seg(seg{i}): SIZE MISMATCH dst(packed={},region={}) src(packed={},region={}) \
                 — REFUSING",
                self.segment_bytes[i],
                dst.total_size().as_u64(),
                owner.segment_bytes[i],
                src.total_size().as_u64()
            );
        }
        // Every tensor EITHER session places in this segment must agree on (offset, size). This is
        // the real layout-equality check; a byte total alone would happily accept two different
        // packings that sum the same.
        for (nm, p) in &self.places {
            if p.segment as usize != i {
                continue;
            }
            match owner.places.get(nm) {
                // ⛔ THE BANK IS PART OF THE ADDRESS. Two bundles could agree on every offset and
                // size and still put a weight in DIFFERENT regions, and sharing the regions then
                // gives the borrower one program's worth of the wrong weights. Comparing the bank is
                // what makes "the layouts match" mean the same thing it meant before banking.
                Some(q)
                    if q.segment == p.segment
                        && q.bank == p.bank
                        && q.offset == p.offset
                        && q.size == p.size => {}
                _ => bail!("alias_seg(seg{i}): PLACEMENT MISMATCH for '{nm}' — REFUSING"),
            }
        }
        for (nm, p) in &owner.places {
            if p.segment as usize == i && !self.places.contains_key(nm) {
                bail!("alias_seg(seg{i}): '{nm}' placed in src but ABSENT in dst — REFUSING");
            }
        }
        // The rebuild can only express a single-chunk region (the same restriction every windowed
        // transfer operates under). Refuse loudly rather than aliasing only the first chunk.
        let c0 = src
            .chunk0()
            .filter(|_| src.is_single_chunk())
            .ok_or_else(|| anyhow!("alias_seg(seg{i}): src region is MULTI-CHUNK — REFUSING"))?;
        // Infallible by construction: `DevAddr::from_chunk` builds a descriptor value.
        self.seg_addr[i] = Some(DevAddr::from_chunk(c0));
        // The borrowed segment is marked so this session's preamble can never H2D its own stale
        // host shadow over the owner's live data.
        self.seg_aliased[i] = true;
        debug!(
            "[sdsc-superdsc] alias_seg(seg{i}): dst now SHARES src's region ({} B)",
            c0.size.0
        );
        // ⭐⭐ AND THE WEIGHT SEGMENT'S EXTRA BANKS — because they ARE the weight segment. Aliasing
        // seg1 while leaving a bank pointing at this session's own (zero-filled) region would give
        // the borrower a working body and a suffix that reads zeros: an lm_head of zeros makes every
        // logit zero, so argmax returns token 0. One wrong token, no crash.
        //
        // ⭐ THIS IS THE PAYOFF OVER THE SEG6 TAIL SPILL. A bank holds ONLY weights, placed
        // identically in every bundle of the ladder, so both equality checks above pass and the
        // region can be SHARED. The spill slot was also an intermediate COLOUR, whose extent depends
        // on the rung's query width, so it could never be aliased and every borrowing session needed
        // its own COPY of the tail — 27–28 × 419,430,400 B = 10.5 GiB of device memory and ~7.6 s of
        // load for granite-3.1-8b-fp16, all of which banking removes.
        if i == SEG_WEIGHT.get() {
            if self.weight_bank_bytes.len() != owner.weight_bank_bytes.len() {
                bail!(
                    "alias_seg(seg{i}): BANK COUNT MISMATCH dst has {} extra weight bank(s), src \
                     has {} — REFUSING",
                    self.weight_bank_bytes.len(),
                    owner.weight_bank_bytes.len()
                );
            }
            for b in 0..self.weight_bank_bytes.len() {
                if self.weight_bank_bytes[b] != owner.weight_bank_bytes[b] {
                    bail!(
                        "alias_seg(seg{i}): weight bank {} SIZE MISMATCH dst={} src={} — REFUSING",
                        b + 1,
                        self.weight_bank_bytes[b],
                        owner.weight_bank_bytes[b]
                    );
                }
                let Some(sb) = owner.weight_bank_addr[b].as_ref() else {
                    bail!(
                        "alias_seg(seg{i}): src has no region for weight bank {}",
                        b + 1
                    );
                };
                let bc0 = sb
                    .chunk0()
                    .filter(|_| sb.is_single_chunk())
                    .ok_or_else(|| {
                        anyhow!(
                            "alias_seg(seg{i}): src weight bank {} is MULTI-CHUNK — REFUSING",
                            b + 1
                        )
                    })?;
                // Infallible by construction: `DevAddr::from_chunk` builds a descriptor value.
                self.weight_bank_addr[b] = Some(DevAddr::from_chunk(bc0));
                self.weight_bank_aliased[b] = true;
            }
        }
        // ⭐ #603 PR2 2a — A BORROWING SESSION'S PARITY GATE runs HERE, at the alias that
        // completes its borrowed set: the first moment every segment it declared borrowed
        // holds a real, renderable address (prepare had to defer it — the placeholders it
        // rendered through then are poison by contract). A disagreement refuses the
        // ADOPTION, which the caller already treats as fatal-for-this-session, never a
        // step. A session with NO borrowed segments gated at prepare and is not re-gated
        // here (the batched-prefill top rung aliases seg2 post-prepare without borrowing).
        if self.seg_borrowed.iter().any(|b| *b) && !self.borrow_set_pending() {
            self.check_cb_plan_parity()?;
        }
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Accessors + the declared session facts
    // ──────────────────────────────────────────────────────────────────────────────────────────

    pub fn is_paged(&self) -> bool {
        self.kv.paged
    }
    pub fn pool_pages(&self) -> PoolPages {
        self.kv.pool_pages
    }
    pub fn page_slots(&self) -> PageSlots {
        self.kv.page_slots
    }

    /// ⛔⛔⛔ THE COLUMNS THE BROADCAST PREFIX MASK RESERVES — **READ OFF THE PLACEMENT THE BIND GUARD
    /// CHECKS AGAINST**, so the extent the host stages and the size it is validated against are ONE
    /// FIELD and cannot disagree.
    ///
    /// This exists because they did disagree. The mask extent used to come from [`Self::cap`], and for an
    /// UNPAGED bundle that is `num_blocks * 64` — a fact about the POOL with no request in it. Sizing the
    /// pool from the card took it to 513 * 64 = 32832 slots against placements reserving 8192, and every
    /// prompt died in `refill_activations`: `binds 65664 B into a 16384 B placement`. Fixing the prefill
    /// staging alone moved the SAME over-bind to the solo decode staging one step later
    /// (`run_step (pos 6)`), because two readers shared one wrong number.
    ///
    /// ⛔ ONLY FOR THE BROADCAST (`pm_rows == 1`) MASK a prompt chunk or a SOLO decode step stages. A
    /// decode-BATCH bundle places `[nqh*mq, ...]` instead, and its staging is blocked by the DECLARED fold
    /// grid (`PrefixMaskShape` / `MaskBlocks`), not by this. `None` when the bundle placed no mask, or when
    /// the placement is not a whole number of pages — either way the caller must refuse rather than guess.
    /// ⭐ BYTES PER PHYSICAL PAGE — `iters * kv_stride`, all layers of one page. THE number
    /// `fold_plan::page_base_bytes` multiplies a block-table entry by, so it is also the number the
    /// gather's index unit must divide. `0` for an unpaged bundle.
    pub fn kv_page_stride_bytes(&self) -> u64 {
        self.kv.page_stride_bytes
    }

    pub fn pmask_slots(&self) -> Option<usize> {
        let nm = bundle::PlaceId::Act(crate::lower_subtile_tape_to_superdsc::ATTN_MASK_TID);
        let slots = (self.places.get(&nm)?.size / 2) as usize;
        let per_page = scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS;
        (slots != 0 && slots.is_multiple_of(per_page)).then_some(slots)
    }

    /// Resident capacity in slots. PAGED: what ONE REQUEST can hold — the pages it has been given.
    /// The pool is SHARED, so reporting it here would let a request run off its own pages into
    /// another's.
    pub fn cap(&self) -> i64 {
        if self.kv.paged {
            let held = self.fold.block_tables.first().map_or(0, |t| t.len() as i64);
            held * self.kv.page_slots.0
        } else {
            self.num_blocks.0 * 64
        }
    }

    /// Install request `req`'s logical→physical page map, its POOL ROW, and the position its next
    /// token writes at. EVERY session that runs a forward for a request needs it — prefill rungs are
    /// separate sessions, and one left on the create-time identity map writes the prompt into page 0
    /// while decode reads the request's real pages.
    ///
    /// A page outside the pool is REFUSED, never clamped: a clamp points this request at another's
    /// KV. The `row` is likewise the bundle's to bound — row `ROWS` of one kv head is row 0 of the
    /// next — so a refusal from `fold_plan` is forwarded, never dropped.
    pub fn set_block_table(&mut self, req: RowIdx, pos: SeqPos, pages: &[i64]) -> Result<()> {
        if !self.kv.paged {
            bail!("set_block_table on a NON-PAGED bundle — refused");
        }
        if pages.is_empty() {
            bail!("set_block_table: empty page list");
        }
        for (i, &p) in pages.iter().enumerate() {
            if p < 0 || p >= self.kv.pool_pages.0 {
                bail!(
                    "set_block_table: page {p} (entry {i}) outside the {}-page pool — refused",
                    self.kv.pool_pages.0
                );
            }
        }
        self.block_table_set = true;
        self.install_page_map(req, pos, pages);
        Ok(())
    }

    /// The map install itself — shared by the create-time identity default and `set_block_table`.
    /// The RULE (slot 0 retires the last forward's rows, so the inferred fold width is this
    /// forward's and not a high-water mark) lives on [`SessionKv`], which owns the state.
    fn install_page_map(&mut self, req: RowIdx, pos: SeqPos, pages: &[i64]) {
        // LEG-1(a) under `segaddr`: the page assignment each request actually runs with — one
        // line, one write. `real` separates a `set_block_table` install from the create-time
        // identity default (which is every page, so long lists print as a range, not 545 numbers).
        if Diag::get().segaddr {
            let shown = if pages.len() > 16 {
                format!(
                    "n={} [{:?}..{}]",
                    pages.len(),
                    &pages[..4],
                    pages[pages.len() - 1]
                )
            } else {
                format!("{pages:?}")
            };
            eprintln!(
                "[pageassign] req={} pos={} real={} pages={shown}",
                req.get(),
                pos.0,
                self.block_table_set
            );
        }
        self.fold.install_row(req, SlotPos::of_launch(pos.0), pages);
    }

    /// The prefix mask's per-fold-pass stride in bytes. A batched decode's mask is one
    /// `[nqh*mq, page]` block per (request, page) pass, so the runtime steps it by a whole block;
    /// only the worker, which stages it, knows how deep a block is. 0 restores the one-row-per-page
    /// step a prompt uses.
    pub fn set_mask_stride(&mut self, stride_bytes: u64) {
        self.fold.mask_rep_stride_bytes = stride_bytes;
    }

    /// HOW MANY BLOCKS the staged mask has. Only the worker knows — it allocated the buffer — and
    /// the fold needs it to refuse a pass past the last block, where it would read unstaged zeros
    /// that an additive mask treats as VALID.
    pub fn set_mask_blocks(&mut self, blocks: u64) {
        self.fold.mask_blocks = blocks;
    }

    /// HOW MANY PAGES THE FOLD WALKS, decided by the HOST from the batch's shared write slot. This
    /// executor derives the same number itself (`n_fold_pages`); declaring the host's value lets
    /// `fold_plan::reps` REFUSE on a disagreement instead of folding a mask blocked by one number
    /// with passes indexed by another — a mismatch makes every row but row 0 read another row's
    /// page, and an unstaged additive-mask byte reads as ZERO, i.e. VALID.
    pub fn set_fold_pages(&mut self, pages: u64) {
        self.fold.fold_pages = pages;
    }

    /// The per-request row-block stride in the INTERMEDIATE segment. A row-batched fold's ops are
    /// baked for ONE request's `nqh` rows, so each pass has to be rebased onto its own request's
    /// block; only the emitter knows it baked them that way. 0 = the fold spans the whole batch.
    pub fn set_int_stride(&mut self, stride_bytes: u64) {
        self.fold.int_rep_stride_bytes = stride_bytes;
    }
}

/// ⭐ WHICH BODY — NOT WHICH LIST. The selector picks the BODY (a swept extent, possibly
/// fold-fused); the launch table then names the CLASS whose list runs it. Splitting the two is
/// what keeps a hybrid model's selector identical to a single-class one: a rung is one extent
/// shared by every class, so "which body serves this context" never needs a class argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum BodySel {
    Body,
    BodyFused,
    Rung(usize),
    RungFused(usize),
}

/// Which op list a selection names. An index rather than a reference so the selector can hand one
/// back while `&mut self` is still needed to allocate it.
///
/// ⭐ EVERY BODY ARM CARRIES A CLASS INDEX. A hybrid-attention model has one body per attention
/// class; a single-class model has exactly one, so `Body(0)`/`BodyFused(0)`/`Rung(i, 0)` is the
/// case every existing rolled bundle was. The RUNG index names the ladder entry (shared by every
/// class — a rung is one swept extent); the class index names which class's list inside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ListSel {
    Prefix,
    Body(usize),
    Suffix,
    BodyFused(usize),
    Rung(usize, usize),
    RungFused(usize, usize),
}

/// ⭐ WHAT A FORWARD'S CALLER DOES WITH ITS LOGITS — the one fact that decides both whether the suffix
/// runs and whether its row comes back, so the two cannot be asked separately and disagree.
///
/// ⛔ THREE STATES, NOT A `bool`. A decode batch passes no host buffer and still READS the logits: it
/// pulls the whole `[rows, vocab]` tensor itself afterwards (`read_tensor`), because the read-back here
/// converts one row. "No buffer" therefore cannot mean "skip the suffix" — only [`Logits::Discarded`]
/// does, and only a caller that never looks at the tensor can name it.
pub enum Logits<'a> {
    /// The suffix runs and its `[1, vocab]` row is converted into this buffer — decode, and a prompt's
    /// last chunk.
    ReadBack(&'a mut [u8]),
    /// The suffix runs and its output stays on the device for the caller to read itself — a decode
    /// batch's `[rows, vocab]` logits.
    LeftOnDevice,
    /// Nobody reads this forward's logits — every prompt chunk but the last — so the suffix (final
    /// norm → lm-head → logits scaling) is not launched at all.
    Discarded,
}

/// Whether [`Executor::launch_forward`] launches the suffix — read off [`Logits`] in one place.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SuffixLaunch {
    Launched,
    Skipped,
}

/// ⭐⭐⭐⭐⭐ THE BODY ONE STEP RUNS, AND THE PER-BODY FACTS THE HOST MUST DECIDE FROM — as ONE value,
/// minted only by [`Executor::step_body`].
///
/// ⛔ WHY THE FACTS TRAVEL WITH THE SELECTION AND NOT BESIDE IT. The host has to stage the gather's
/// index table and choose the prefix mask's block FORM **before** the launch, and both answers belong
/// to the body the launch will pick — not to the bundle, which holds several. Staging the wrong form is
/// silent in both directions (`PerRowPage` against a collapsed fold leaves `mq-1` of every `mq` rows at
/// −∞; `PerPage` against an uncollapsed one gives every pass row 0's validity), and staging no index for
/// a body that gathers is `0xa35e RAS::PCI::BusFence` with no operand named. So the selection and the
/// facts are one value: holding the facts means having asked the selector.
///
/// ⛔ `sel` IS PRIVATE. The host may not name a body — only ask which one, and read what it declares.
#[derive(Clone, Copy, Debug)]
pub struct StepBody {
    sel: BodySel,
    swept: SweptCols,
    gathers: crate::wiring::GathersKv,
    folds: bool,
    region_form: bool,
    lx_op_local_bytes: u64,
    lx_prob_pin_bytes: u64,
}

impl StepBody {
    /// ⭐ THE COLUMNS ONE FOLD PASS OF **THIS** BODY SWEEPS — its own `active_cap`.
    ///
    /// Two host decisions derive from it and both used to take the CEILING rung's: the gather scratch's
    /// window count (`nb`), and how far the fold's sweep reaches (`first_unswept_slot`). The ceiling is
    /// the LOOSE bound for the guard — it passed at 1536 covered slots while a 128-column body really
    /// covered 768 — and for the scratch it is a different NUMBER from the one the body reads.
    pub const fn swept(self) -> SweptCols {
        self.swept
    }

    /// ⭐ WHETHER **THIS** BODY READS ITS KV THROUGH A GATHERED SCRATCH — off the body's own launch
    /// groups (`bundle::KvShifts::gathered`).
    pub const fn gathers(self) -> crate::wiring::GathersKv {
        self.gathers
    }

    /// ⭐ WHETHER **THIS** BODY'S GATHERED TABLE IS THE FILL-STAGED REGION FORM — the staging
    /// channel, off the same launch groups (`bundle::KvShifts::region_form`). The body still
    /// `gathers` (absolute entries, zero shift); its table is staged by the host fill
    /// (`fill_attn_step_inputs`), so the composer's page-list answer for it is FALSE —
    /// see [`Self::page_list`].
    pub const fn region_form(self) -> bool {
        self.region_form
    }

    /// ⭐ THE PAGE-LIST ANSWER — whether the HOST must stage this body's index table through
    /// the composer and mint a KV-block-index step for it: it gathers AND its table is not
    /// the fill-staged region form. The ONE expression every staging-side consumer keys on
    /// (the composer's gate and the forward's `gathers` parameter), so the two cannot drift.
    pub const fn page_list(self) -> bool {
        self.gathers.get() && !self.region_form
    }

    /// ⭐ WHETHER **THIS** BODY IS A SPLIT-KV CHUNK — it gathers and carries no page fold, so it attends
    /// exactly its own chunk in one launch and the host stages the CHUNK's index and mask, not the
    /// fold's. On a decode batch this is what separates a chunk rung from the gathered fold below and
    /// past it — both gather, and staging one's table for the other reads real block numbers from the
    /// wrong layout. Keyed on [`Self::page_list`]: a region-form body's staging is the fill's, not
    /// the chunk table's.
    pub const fn chunk(self) -> bool {
        self.page_list() && !self.folds
    }

    /// ⭐ THE RUNG'S OP-LOCAL LX PIN SLOTS, in bytes — read off the selected body's own baked
    /// layout (`bundle::BundleLayout::lx_op_local_bytes`), the host's term of the one
    /// `lx_window_fits` budget. The host's chunk-table arm threads this into
    /// `chunk_gather_index_table`, so the host refuses exactly the chunks the bake's `lx_form`
    /// witness refused — neither arm budgets against another rung's pins.
    pub const fn lx_op_local_bytes(self) -> u64 {
        self.lx_op_local_bytes
    }

    /// ⭐ THE CHUNK FORM'S PROBABILITY-PART LX PINS, in bytes — read off the selected body's own
    /// baked layout (`bundle::BundleLayout::lx_prob_pin_bytes`), the host's SECOND core-local
    /// term of the same budget: the bake's `lx_form` admission summed it beside the gather pins,
    /// and the host's chunk-table arm threads it in beside [`Self::lx_op_local_bytes`], so the
    /// host refuses exactly what the bake refused.
    pub const fn lx_prob_pin_bytes(self) -> u64 {
        self.lx_prob_pin_bytes
    }
}

/// This bundle's launches, in launch order.
///
/// Infallible: a launch IS its program ([`bundle::LaunchGroup`]), so there is no index that could name
/// a program the bundle does not carry.
fn bind_ops(code: &'static bundle::BundleCode<'static>) -> Ops {
    code.groups
        .iter()
        .enumerate()
        .map(|(i, g)| OpProg::new(g, &code.fp, i))
        .collect()
}

fn launch_err(e: HwError) -> anyhow::Error {
    anyhow!("compute launch: {e}")
}

/// Process RSS in KB (for the flex-leak reset probe).
fn vmrss_kb() -> i64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return -1;
    };
    s.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(-1)
}

fn report_group_time(t: &Timers) {
    let total: f64 = t.group_ms.values().map(|(ms, _)| ms).sum();
    let mut top: Vec<_> = t
        .group_ms
        .iter()
        .map(|(label, (ms, n))| (*ms, label.as_str(), *n))
        .collect();
    top.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "[sdsc-grouptime] top groups by total ms (of {total:.1} ms over {} launches)",
        t.group_calls
    );
    // Every group: a prompt spans one bundle per chunk width, each with its own groups.
    for (ms, label, n) in &top {
        eprintln!(
            "[sdsc-grouptime]   {label:<40}  {ms:8.1} ms total  {:6.3} ms/launch  {n:6} launches  \
             {:5.1}%",
            ms / *n as f64,
            100.0 * ms / total.max(1e-9)
        );
    }
}

/// One CB class's device-clock window inside one forward: the extreme WRITTEN timestamp snapshots
/// over that class's CBs. The five TS words are snapshots of one free-running u32 counter
/// ([`flex_rs::control_block_wire::ResponseBlockWire::timestamps`]'s measured layout: ~1012
/// cyc/µs, ~4.2 s wrap; a slot a CB does not write reads zero), so `first`/`last` are raw ticks —
/// consecutive forwards' windows difference wrap-exactly into the inter-step device-idle gap
/// (fwd N `last` → fwd N+1 `first`), the number #603's loop closure exists to move.
struct RbClassWindow {
    first: u32,
    last: u32,
    n: usize,
}

/// The window of every entry matching `class`, over written (nonzero) TS slots only: a CB
/// hardware left unwritten cannot bound the window, and a class with no CBs is `None` — absent,
/// never zero-bounded, so a missing class cannot swallow the gap being measured.
fn rb_class_window(
    entries: &[flex_rs::rb_telemetry::RbTiming],
    class: impl Fn(&flex_rs::rb_telemetry::RbTiming) -> bool,
) -> Option<RbClassWindow> {
    let mut window: Option<RbClassWindow> = None;
    for e in entries.iter().filter(|e| class(e)) {
        let mut cb_first: Option<u32> = None;
        let mut cb_last = 0u32;
        for &ts in &e.rb.timestamps() {
            if ts != 0 {
                cb_first = Some(cb_first.map_or(ts, |f: u32| f.min(ts)));
                cb_last = cb_last.max(ts);
            }
        }
        let Some(cb_first) = cb_first else {
            continue;
        };
        match &mut window {
            Some(w) => {
                w.first = w.first.min(cb_first);
                w.last = w.last.max(cb_last);
                w.n += 1;
            }
            None => {
                window = Some(RbClassWindow {
                    first: cb_first,
                    last: cb_last,
                    n: 1,
                })
            }
        }
    }
    window
}

/// The RB-timestamp report: per-op device cycles, harvested from the response blocks at
/// completion (no drain), cumulative across the run. Phases are named per the card-measured slot
/// semantics ([`flex_rs::control_block_wire::ResponseBlockWire::exec_cmpt_cycles`] and siblings):
/// `exec` is the compute engine's own time, `prep` the queue+prep wait behind earlier CBs — on a
/// pipelined stream consecutive CBs overlap (the next CB's prep runs during the previous CB's
/// exec), so Σ exec, not Σ total, is the stream's device span.
fn report_rb_time(t: &Timers) {
    let exec_total: u64 = t.rb_cycles.values().map(|a| a.cycles.exec).sum();
    let prep_total: u64 = t.rb_cycles.values().map(|a| a.cycles.prep).sum();
    let dma_total = t.rb_dmai.dmai + t.rb_dmao.tail;
    eprintln!(
        "[sdsc-rbtime] device exec cycles by op (of {exec_total} exec + {prep_total} prep + \
         {dma_total} dma cycles over {} compute CBs, {} forwards)",
        t.rb_cbs, t.rb_steps
    );
    let mut top: Vec<_> = t
        .rb_cycles
        .iter()
        .map(|(k, a)| (a.cycles.exec, *k, a))
        .collect();
    top.sort_by_key(|a| std::cmp::Reverse(a.0));
    for (cyc, (rung, oi), a) in top.iter().take(20) {
        eprintln!(
            "[sdsc-rbtime]   op ({rung},{oi:<3}) {:>12} cyc exec  {:>9} cyc/cb  {:6} cbs  \
             prep {:>12}  {}",
            cyc,
            cyc / a.cycles.cbs.max(1),
            a.cycles.cbs,
            a.cycles.prep,
            a.label
        );
    }
    eprintln!(
        "[sdsc-rbtime]   dmai: {} cb, {} cyc ({} cyc/cb)  |  dmao: {} cb, {} cyc ({} cyc/cb)",
        t.rb_dmai.cbs,
        t.rb_dmai.dmai,
        t.rb_dmai.dmai / t.rb_dmai.cbs.max(1),
        t.rb_dmao.cbs,
        t.rb_dmao.tail,
        t.rb_dmao.tail / t.rb_dmao.cbs.max(1),
    );
}

/// The fold shape an image was derived under, as one comparable value: every
/// `SessionKv` field the render reads that is not per-step, plus row 0's
/// first `n_fold_pages` page ids (see [`StepImageKey::signature`] for why
/// those, and why no row cursors). Cheap to build and compare — a dozen
/// i64s against a cached key — against a step whose device time is
/// milliseconds.
fn fold_signature(fold: &SessionKv, n_fold_pages: i64) -> Vec<i64> {
    let mut s = vec![
        fold.paged as i64,
        fold.page_slots.get(),
        fold.page_stride_bytes.0 as i64,
        fold_plan::fold_requests(fold),
        fold.mask_rep_stride_bytes as i64,
        fold.mask_blocks as i64,
        fold.fold_pages as i64,
        fold.int_rep_stride_bytes as i64,
    ];
    if let Some(row0) = fold.block_tables.first() {
        s.extend(row0.iter().take(n_fold_pages.max(0) as usize).copied());
    }
    s
}

/// The replay player's decomposition — cumulative since process start, on a
/// 256-step cadence. ⛔ ONE WRITE: the same single-`format!` discipline as
/// `[progverify]` — an abort from senlib's monitor thread takes the process
/// down with no unwinding, so a buffered line is a line that never existed.
/// This line is also the player's LIVENESS MARKER: a run whose stderr never
/// shows it replayed nothing.
fn report_replay(t: &Timers) {
    let steps = t.replay_steps.max(1);
    let derives = t.replay_derives.max(1);
    let one = format!(
        "[sdsc-replay] {} step(s) replayed: {:.1}% from cached image(s) | {:.1} CB(s)/step, {} \
         patch site(s)/image | lookup {:.1} us/step, submit {:.3} ms/step | {} derivation(s) at \
         {:.1} ms each\n",
        t.replay_steps,
        100.0 * t.replay_hits as f64 / steps as f64,
        t.replay_cbs as f64 / steps as f64,
        t.replay_sites / derives,
        1000.0 * t.replay_lookup_ms / steps as f64,
        t.replay_submit_ms / steps as f64,
        t.replay_derives,
        t.replay_derive_ms / derives as f64,
    );
    eprint!("{one}");
}

/// The submit / prefix-ladder / repeat-probe reports, emitted with the phase line.
fn report_phase_timers(compute_ms: f64) {
    let d = Diag::get();
    let mut t = timers();
    // WHERE THE COMPUTE PHASE'S TIME ISN'T: `compute` is wall time from the last H2D to the drain,
    // so it holds device work AND whatever the host spent feeding the stream. These say how much of
    // it the host owns, per launch — the number that decides whether a batch's launch count is the
    // cost.
    if d.submit_time && t.submit_calls > 0 {
        eprintln!(
            "[sdsc-phase]   host: {} launches ({} with a host fence call) | submit={:.2} ms \
             ({:.1} us/ea) prep={:.2} ms ({:.1} us/ea) | = {:.0}% of compute",
            t.submit_calls,
            t.host_fence_launches,
            t.submit_ms,
            1000.0 * t.submit_ms / t.submit_calls as f64,
            t.prep_ms,
            1000.0 * t.prep_ms / t.submit_calls as f64,
            100.0 * (t.submit_ms + t.prep_ms) / compute_ms.max(1e-9),
        );
        t.submit_ms = 0.0;
        t.prep_ms = 0.0;
        t.submit_calls = 0;
        t.host_fence_launches = 0;
    }
    // THE HARVEST, DRAIN-FREE: per-op device cycles the response blocks carried all along. This
    // section rides the phase line (turn `SCRATCHY_SDSC_RB_TIME` on WITH `phase_time` for the
    // per-forward view; alone, the report fires on the same 20k-CB cadence as `group_time`).
    if d.rb_time && t.rb_cbs > 0 {
        report_rb_time(&t);
    }
    // THE LADDER: T(j) is the mean time from a layer's first launch to the drain after group j, so
    // the RISE from one rung to the next is group j's own device time. A rung with a small rise is a
    // group the card finishes while the host is still submitting. Ragged rungs mean the backlog
    // assumption broke down for that sample and the rise is not readable — take more layers.
    if d.prefix_time && !t.prefix_ms.is_empty() {
        let mut prev = 0.0;
        let mut cur_len = 0usize;
        for ((len, oi), (ms, n, prog)) in &t.prefix_ms {
            if *len != cur_len {
                cur_len = *len;
                prev = 0.0;
                eprintln!(
                    "[sdsc-prefix] {cur_len}-group rung: cumulative ladder per call (rise = that \
                     group's device time)"
                );
            }
            let v = ms / *n as f64;
            eprintln!(
                "[sdsc-prefix]   op {oi:<3}  T={v:7.3} ms  dev={:+7.3} ms  n={n:<4}  {prog}",
                v - prev
            );
            prev = v;
        }
        t.prefix_ms.clear();
    }
    // THE REPEAT PROBE: `1st` and `again` both contain one drain, so `switch` — their difference —
    // is what the card pays to bring in a program it was not already holding. If `switch` is most of
    // `1st`, the per-request groups are paying for being distinct baked programs rather than for
    // being launches, and one program relaunched per request removes it.
    if d.repeat_probe && !t.repeat_ms.is_empty() {
        eprintln!("[sdsc-repeat] same program launched twice back to back");
        for (oi, (first, again, n, prog)) in &t.repeat_ms {
            let (a, b) = (first / *n as f64, again / *n as f64);
            eprintln!(
                "[sdsc-repeat]   op {oi:<3}  1st={a:7.3} ms  again={b:7.3} ms  switch={:+7.3} ms  \
                 n={n:<4}  {prog}",
                a - b
            );
        }
        t.repeat_ms.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compile-time-named segment cannot be out of range, and a runtime one is checked.
    #[test]
    fn segment_indices_are_bounded() {
        assert_eq!(SEG_INTERMEDIATE.get(), 0);
        assert_eq!(SEG_WEIGHT.get(), 1);
        assert_eq!(SEG_KV.get(), 2);
        assert_eq!(SEG_MASK.get(), 3);
        assert!(SegIdx::checked(6).is_some());
        assert!(
            SegIdx::checked(7).is_none(),
            "seg7 is the PROGRAM segment, not a tensor segment"
        );
        assert!(SegIdx::checked(-1).is_none());
        assert_eq!(SegIdx::all().count(), NUM_SEGMENTS);
    }

    /// `Seg::known` is const-evaluated, so this is a compile-time proof rather than a test — the
    /// value exists only if the assert held.
    const _SEG6: SegIdx = SegIdx::known::<6>();

    /// ⭐ ONLY A KEYED BIND IS SPARED THE PER-STEP RE-UPLOAD, AND ONLY AFTER ITS FIRST STEP. An
    /// unkeyed activation is sent every step (an input someone forgets to re-bind must keep being
    /// caught by today's path); a keyed one is sent in the step that bound it and not again until a
    /// new key re-binds it.
    #[test]
    fn a_keyed_bind_goes_out_once_and_an_unkeyed_one_every_step() {
        use scratchy_subtile::sdsc_abstract::{CausalKey, RungWidth};
        let key =
            CausalKey::chunk(RungWidth::of_baked_rows(512).expect("a width"), 512).expect("a key");
        let (mask, rope) = (bundle::PlaceId::Act(7), bundle::PlaceId::Act(8));
        let mut keyed = BTreeMap::new();
        assert!(refill_sends(&keyed, &rope), "unkeyed: sent");
        keyed.insert(mask, Keyed { key, fresh: true });
        assert!(refill_sends(&keyed, &mask), "keyed, bound this step: sent");
        assert!(
            refill_sends(&keyed, &rope),
            "a keyed neighbour changes nothing for the rest"
        );
        // What `refill_activations` does after the convert: every keyed bind is now on the device.
        for k in keyed.values_mut() {
            k.fresh = false;
        }
        assert!(
            !refill_sends(&keyed, &mask),
            "keyed, already uploaded: not sent"
        );
        assert!(
            refill_sends(&keyed, &rope),
            "unkeyed: still sent every step"
        );
    }

    /// ⭐ A BIND CONVERTED AT BIND GOES OUT EXACTLY AS ONE CONVERTED AT THE REFILL: the same window
    /// (one [`h2d_run`] law for both sources), and the same bytes in it — the shadow slot
    /// `bind_input_f32` writes in one pass is the slot the refill's `ieee_to_sen_bytes` wrote from
    /// `f32_to_f16_le`'s buffer. Over offsets on and off the 4 KiB grid and lengths on both sides
    /// of it, so the window's round-down and round-up both move.
    #[test]
    fn a_bind_converted_at_bind_goes_out_as_one_converted_at_refill() {
        let vals: Vec<f32> = (0..3000).map(|i| (i as f32 - 1500.0) * 0.37).collect();
        for (offset, n) in [
            (0u64, 1usize),
            (128, 2048),
            (4096, 2048),
            (4224, 3000),
            (8064, 64),
        ] {
            let len = n as u64 * 2;
            let (a, b) = h2d_run(offset, len);
            assert!(
                a <= offset && offset + len <= b,
                "the window covers the bind"
            );
            assert!(a % H2D_ALIGN == 0 && b % H2D_ALIGN == 0, "on the grid");
            assert!(
                b - a < len + 2 * H2D_ALIGN,
                "and no wider than the grid forces"
            );
            // The two shadows: the refill's two-pass write, and the one-pass write at bind.
            let mut two = vec![0u8; b as usize];
            let ieee = sen_convert::f32_to_f16_le(&vals[..n]);
            sen_convert::ieee_to_sen_bytes(&ieee, &mut two[offset as usize..][..ieee.len()]);
            let mut one = vec![0u8; b as usize];
            sen_convert::f32_to_sen_bytes(&vals[..n], &mut one[offset as usize..][..len as usize]);
            assert!(
                one[a as usize..b as usize] == two[a as usize..b as usize],
                "offset {offset} n {n}: the window's bytes differ"
            );
        }
    }

    /// ⭐ FAULTING A SHADOW IN CHANGES NONE OF ITS BYTES — the property that lets the load-time
    /// materialize run after `stage_keyed` (or in any order with it): the keyed bytes already in
    /// the shadow are the ones a later aligned H2D window may re-send. Over a length that is not a
    /// whole number of pages, so the last partial page is covered too.
    #[test]
    fn faulting_a_shadow_in_changes_none_of_its_bytes() {
        let n = 5 * HOST_PAGE + 123;
        let want: Vec<u8> = (0..n).map(|i| (i * 31 % 251) as u8).collect();
        let mut got = want.clone();
        fault_in(&mut got);
        assert!(got == want, "a byte moved");
        let mut zeros = vec![0u8; n];
        fault_in(&mut zeros);
        assert!(zeros.iter().all(|&b| b == 0), "a zeroed shadow stays zero");
    }

    /// Two steps on ONE image key: the first derives, the second hits the cache —
    /// and the hit arm must COUNT it. The cadence line's "% from cached" read 0.0
    /// forever because it didn't: 256 replayed steps off 4 derivations
    /// printed "0.0% from cached" on a live card.
    #[test]
    fn replay_image_cache_counts_its_hits() {
        let (hits, derives) = {
            let t = timers();
            (t.replay_hits, t.replay_derives)
        };
        let mut cache = StepImageCache::default();
        let key = StepImageKey {
            sel: BodySel::RungFused(0),
            n_fold_pages: 1,
            signature: vec![7],
        };
        let dummy = StepImage {
            data: std::sync::Arc::new(StepImageData {
                canvas: std::sync::Arc::new(Vec::new()),
                rows: Vec::new(),
            }),
            barrier: false,
        };
        assert!(
            cache.lookup_cached(&key).is_none(),
            "an empty cache has no image: the first step derives"
        );
        cache.note_derived(key.clone(), dummy, Instant::now());
        assert!(
            cache.lookup_cached(&key).is_some(),
            "the second step on the same key must hit the cache"
        );
        let t = timers();
        assert_eq!(t.replay_hits, hits + 1, "the hit arm counted it");
        assert_eq!(t.replay_derives, derives + 1);
    }

    /// The rbgap window takes the extreme WRITTEN TS slots per class: a CB whose slots hardware
    /// left zero never bounds it, and a class with no CBs is absent rather than zero-bounded —
    /// the inter-step gap differences consecutive forwards' windows, so a bogus zero would
    /// swallow the very window being measured.
    #[test]
    fn rbgap_window_bounds_only_written_slots() {
        use flex_rs::control_block_wire::{RB_NUM_BYTES, ResponseBlockWire, TS_SECTION};
        use flex_rs::dma::DmaDirection;
        use flex_rs::rb_telemetry::RbTiming;
        use flex_rs::scheduler::PipelineId;

        let rb = |ts: [u32; 5]| {
            let mut b = [0u8; RB_NUM_BYTES];
            for (i, w) in ts.iter().enumerate() {
                b[TS_SECTION + i * 4..TS_SECTION + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            ResponseBlockWire::from_bytes(&b)
        };
        let entry =
            |id: u64, pipeline: PipelineId, dma: Option<DmaDirection>, ts: [u32; 5]| RbTiming {
                id,
                pipeline,
                dma,
                rb: rb(ts),
            };
        // A compute CB writes all five words (word1 == word0 per the measured layout); a D2H
        // crossing writes only words 3-4; one pathological CB writes nothing at all.
        let entries = vec![
            entry(1, PipelineId::Compute, None, [100, 100, 120, 200, 210]),
            entry(
                2,
                PipelineId::Compute,
                Some(DmaDirection::DeviceToHost),
                [0, 0, 0, 300, 340],
            ),
            entry(3, PipelineId::AsyncDmaI, None, [0; 5]),
        ];
        let all = rb_class_window(&entries, |_| true).unwrap();
        assert_eq!((all.first, all.last, all.n), (100, 340, 2));
        let compute = rb_class_window(&entries, |e| {
            matches!(e.pipeline, PipelineId::Compute) && e.dma.is_none()
        })
        .unwrap();
        assert_eq!((compute.first, compute.last, compute.n), (100, 210, 1));
        let d2h = rb_class_window(&entries, |e| e.dma == Some(DmaDirection::DeviceToHost)).unwrap();
        assert_eq!((d2h.first, d2h.last, d2h.n), (300, 340, 1));
        assert!(
            rb_class_window(&entries, |e| e.dma == Some(DmaDirection::HostToDevice)).is_none(),
            "an absent class is absent, not zero-bounded"
        );
    }
}
