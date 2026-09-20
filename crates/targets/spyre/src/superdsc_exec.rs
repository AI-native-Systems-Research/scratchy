// SPDX-License-Identifier: Apache-2.0
//! The resident SuperDSC executor — the live on-silicon run path, in Rust.
//!
//! The SuperDSC emitter produces a dxp device bundle (one device program per launch group + the
//! placement manifest). This module runs it DIRECTLY on the AIU through [`crate::sdk_abi`]: the
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
//! [`crate::bundle_code`]'s, byte conversion is [`crate::sen_convert`]/[`crate::stage_weight`]'s, and the SDK calls are
//! [`crate::sdk_abi`]'s. This module is the sequencing: what to allocate, what to launch, in what
//! order, with which base.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use tracing::{debug, info};

use crate::bundle_code::{self as bundle, KernelWeight, NUM_SEGMENTS, Placement, SweptCols};
use crate::fold_plan::{self, Bytes, OpKv, RowIdx, RowPage, SessionKv, SlotPos, UniformPages};
use crate::sdk_abi::{DevAddr, MemKind, Stream};
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
struct LaunchWhere {
    byte_off: [u64; NUM_SEGMENTS],
    slot_pos: SeqPos,
    kv_page_base: Bytes,
    n_fold_pages: i64,
    /// Which weight bank backs the positional weight slot — see `weight_bank_for`.
    weight_bank: usize,
}

struct Diag {
    /// `SCRATCHY_SDSC_PEROP_SYNC`: re-enable the per-group host drain. DEFAULT-OFF —
    /// pipeline_barrier + the stream's STRICT_ORDERING guarantee the dataflow, and the drain cost
    /// ~11% of decode (pod-confirmed coherent without it: 24.7 → 27.4 tok/s).
    perop_sync: bool,
    /// `SCRATCHY_SDSC_GROUP_TIME`: per-op-index device time. Needs a drain to attribute anything,
    /// so it costs what the per-op sync costs — a MEASUREMENT MODE, not something to leave on.
    group_time: bool,
    /// `SCRATCHY_SDSC_SUBMIT_TIME`: host enqueue cost per launch, with NO drain, so the pipeline is
    /// left alone. The honest answer to "is a batch's launch count the cost".
    submit_time: bool,
    /// `SCRATCHY_SDSC_PREFIX_TIME`: one drain per call at a ROTATING group, giving a cumulative
    /// ladder whose successive differences are the per-group device times.
    prefix_time: bool,
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
                repeat_probe: on("SCRATCHY_SDSC_REPEAT_PROBE"),
                body_trace: on("SCRATCHY_SDSC_BODY_TRACE"),
                segaddr: on("SCRATCHY_SDSC_SEGADDR_DIAG"),
                prog_verify: on("SCRATCHY_SDSC_PROG_VERIFY"),
                phase_time: on("SCRATCHY_SDSC_PHASE_TIME"),
                prep_time: on("SCRATCHY_SDSC_PREP_TIME"),
                optrace: on("SCRATCHY_SDSC_OPTRACE"),
                launch_mark: on("SCRATCHY_SDSC_LAUNCH_MARK"),
                reset_probe: std::env::var("SCRATCHY_SDSC_RESET_PROBE")
                    .ok()
                    .and_then(|v| v.parse().ok()),
            }
        })
    }
}

/// Accumulators the timing modes write into. Process-wide, like the C++ statics — the run is one
/// model, so op indices do not collide across bundles in a way that misleads.
#[derive(Default)]
struct Timers {
    /// Per op index: (total ms, launches).
    group_ms: BTreeMap<usize, (f64, u64)>,
    group_calls: u64,
    submit_ms: f64,
    prep_ms: f64,
    submit_calls: u64,
    barrier_launches: u64,
    /// 🛑 KEYED BY (RUNG LENGTH, INDEX), NOT INDEX ALONE. A forward calls the launch loop once per
    /// rung — the body, but also the prefix and suffix lists — and each restarts its numbering at 0,
    /// so a map keyed on the index alone averages a 20-op body's group 2 with a 3-op prefix's.
    prefix_ms: BTreeMap<(usize, usize), (f64, u64, String)>,
    prefix_call: BTreeMap<usize, u64>,
    repeat_ms: BTreeMap<usize, (f64, f64, u64, String)>,
    forwards: i64,
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
/// device program. `SCRATCHY_SUPERDSC_GROUP_SIZE=1` makes that one trip per program, which is the
/// fault-isolation end of the same knob — not a different mode.
///
/// Each is launched against the SHARED resident segments, so there is no ModuleStitcher (which
/// orphaned the matmul in the fused bundle). A group that carries an address shift — a KV slot write,
/// a Kᵀ slab restickify, a page fold — is a singleton at ANY group size, because the shift is applied
/// once to the whole group.
struct OpProg {
    /// The launch: its address shifts and the compiled program they apply to.
    code: &'static bundle::LaunchGroup<'static>,
    /// The bundle this launch belongs to, and its position in that bundle's launch order — together,
    /// the whole of "which program is this".
    fp: &'static str,
    index: usize,
    /// Its resident Program-segment allocation. `None` until [`Executor::alloc_ops`] — the split
    /// ladder is allocated lazily, so an unallocated op is a real state, not an invariant break.
    addr: Option<DevAddr>,
    /// The KV facts `fold_plan` reads: whose KV, which pass shape, which strides.
    kv: OpKv,
}

/// ⛔⛔⛔ A SESSION-LIFETIME DEVICE HANDLE IS LEAKED ON PURPOSE, BECAUSE IT OUTLIVES THE RUNTIME THAT
/// COULD FREE IT.
///
/// MEASURED 2026-08-12: every run ended `RUN_RC=255` **after** `Batch complete: N succeeded, 0 failed`
/// — `RAS::FLEXALLOCATOR::AddressNotFound — deallocate: CompositeAddress not found in allocation_map`,
/// thrown out of `flex::CompositeAddress::~CompositeAddress` into `__cxa_call_terminate`. Exit codes
/// became lies: any gate that reads `$?` calls a perfect run a failure.
///
/// It is a LIFETIME ORDER problem, not an ownership-kind one. The deleted `sdsc_shim.cpp` deleted
/// chunk-built sub-addresses ~120x per forward (`sdsc_dma_slice`) and never aborted, because those died
/// inside a session object while the flex runtime was still up. These handles are dropped during
/// PROCESS SHUTDOWN instead, by which point the runtime context and its allocator are gone, so the
/// destructor looks the address up in an empty map and throws. `fxa_addr_free`'s `catch (...)` cannot
/// save it: a destructor is implicitly `noexcept`, so the throw calls `std::terminate` directly.
///
/// Leaking is CORRECT here, not a workaround: the device region lives until process exit either way
/// (the C++ executor never freed regions), so all that leaks is a descriptor, once, at shutdown.
///
/// ⛔ DO NOT "SIMPLIFY" THIS BY DELETING `impl Drop for DevAddr` in `sdk_abi.rs`. The per-step windows
/// (`shifted`, built per layer per launch) must keep freeing — they drop while the runtime is alive,
/// exactly as the old shim's did, and at ~120 per forward a leaked descriptor each grows without bound.
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
                page_fold: kv.page_fold,
                slot_write: kv.slot_write(),
                page_slots: kv.page_slots as u64,
                slot_stride_bytes: kv.slot_stride_bytes as u64,
                slab_write: kv.slab_write(),
                slab_stride_bytes: kv.slab_stride_bytes as u64,
                batched_requests: kv.batched_requests,
                // ⭐ THE OTHER HALF OF `fold_plan::collapsed`, and what sends the fold's KV shift to
                // zero — a bake fact, carried on the group exactly like the axis itself.
                gathered: kv.gathered,
            },
        }
    }

    /// How the timing modes name this program: its bundle's fingerprint and its launch order.
    fn label(&self) -> String {
        format!("{}/group_{}", self.fp, self.index)
    }
}

/// A launch sequence — a body rung, the prefix, the suffix, or a fold-fused twin.
type Ops = Vec<OpProg>;

/// The host-shadow slice covering the WHOLE of `addr` — the H2D source for a segment.
///
/// ⭐ RETURNING A SLICE IS THE POINT. [`Stream::h2d`] takes the transfer length FROM the slice, so
/// the DMA cannot read past the shadow the way an independent `size: u64` argument could — which is
/// exactly how `alloc_ops_at` came to pin past the end of a program binary. A shadow shorter than
/// its region is a STAGING bug (`prepare` grows every shadow it sends to the region's extent), so it
/// is reported here rather than becoming an out-of-bounds transfer.
fn whole_shadow<'a>(shadow: &'a [u8], addr: &DevAddr, seg: usize) -> Result<&'a [u8]> {
    let need = addr.total_size() as usize;
    shadow.get(..need).ok_or_else(|| {
        anyhow!(
            "seg{seg}: the host shadow is {} B but its device region is {need} B — this H2D would \
             read past the shadow. `prepare` grows every shadow it sends to the region's extent; \
             this one was not grown, so the segment was never meant to be sent from here.",
            shadow.len()
        )
    })
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

/// The re-rolled layer loop: the body is ONE layer, re-launched `iters` times with the weight and
/// KV segment bases advanced per layer, between a prefix (embed) and a suffix (lm_head).
struct Rolled {
    iters: i64,
    /// seg1 per-layer byte stride.
    weight_stride: u64,
    /// ⭐ LAYERS PER WEIGHT BANK. Layer `v` is `(v / layers_per_bank)`'s bank at
    /// `(v % layers_per_bank) · weight_stride` — the whole of banking, at launch time. Equal to
    /// `iters` when the weights fit one region, which makes the division a no-op and this path
    /// byte-identical to the single-region one. NEVER 0 (`load_rolled` refuses that: it would divide
    /// by zero on the first launch).
    layers_per_bank: i64,
    /// The bank the PREFIX program's weight operands were baked in, and the SUFFIX's — the emitter
    /// proved each group's weights live in ONE bank, because a launch has one base per segment.
    prefix_weight_bank: usize,
    suffix_weight_bank: usize,
    /// seg2 per-layer byte stride.
    kv_stride: u64,
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
    body_ops: Ops,
    suffix_ops: Ops,
    /// The SAME body with the per-page fold fused back in — one fewer launch per layer. Used
    /// whenever the context fits ONE page, which is the case that must cost what unpaged cost.
    body_fused: Ops,
    /// sk_bucket LADDER (decode): `(active_cap, ops)` ascending; the top rung is the full cap.
    body_rungs: Vec<(SweptCols, Ops)>,
    body_rungs_fused: Vec<(SweptCols, Ops)>,
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

    // ── KV / fold ──
    kv: KvGeometry,
    /// The page maps, write cursors and declared strides — [`fold_plan`]'s state, owned here for
    /// the session's lifetime and read by every launch.
    fold: SessionKv,
    /// Has a caller installed a REAL page map (vs the create-time identity default)?
    block_table_set: bool,

    prepared: bool,
    staged_host: bool,
    n_weights_staged: usize,
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
            body_fused: Vec::new(),
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
            kv: KvGeometry::default(),
            fold: SessionKv::default(),
            block_table_set: false,
            prepared: false,
            staged_host: false,
            n_weights_staged: 0,
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
        // ⛔ `layers_per_bank` DIVIDES a layer index on every launch, so a 0 here is a division by
        // zero on the first body launch — and it can only be 0 if a bundle was baked before the
        // field existed or by an emitter that did not set it. Refuse at LOAD, naming the bundle,
        // rather than faulting mid-forward.
        if m.layers_per_bank == 0 {
            bail!(
                "load: reroll meta declares layers_per_bank=0 (iters={}, {} weight bank(s)) — the \
                 per-layer advance divides by it",
                m.iters,
                self.weight_bank_bytes.len() + 1
            );
        }
        // A bank index the session has no region for would silently fall back to another bank's
        // base, which is a whole program reading the wrong weights. Check both groups at load.
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
        self.rolled = Some(Rolled {
            iters: m.iters as i64,
            weight_stride: m.weight_stride,
            layers_per_bank: m.layers_per_bank as i64,
            prefix_weight_bank: m.prefix_weight_bank as usize,
            suffix_weight_bank: m.suffix_weight_bank as usize,
            kv_stride: m.kv_stride,
        });
        self.prefix_ops = Self::sibling_ops("rolled prefix", &m.prefix)?;
        self.suffix_ops = Self::sibling_ops("rolled suffix", &m.suffix)?;
        debug!(
            "[sdsc-superdsc] load: RE-ROLLED — body + prefix({}) + suffix({}); iters={} \
             wstride={} kvstride={}",
            m.prefix.as_str(),
            m.suffix.as_str(),
            m.iters,
            m.weight_stride,
            m.kv_stride
        );
        Ok(())
    }

    /// Bind the body's launch groups: either the sk_bucket LADDER (one body per attention sweep
    /// extent) or the single body, each with its fold-fused twin where one was baked.
    fn load_bodies(&mut self) -> Result<()> {
        let rungs: &[bundle::LadderRung<'static>] = match &self.code.reroll {
            Some(m) if m.rungs.len() > 1 => &m.rungs,
            _ => &[],
        };
        if rungs.is_empty() {
            self.body_ops = bind_ops(self.code);
            if let Some(m) = &self.code.reroll {
                self.body_fused = match bundle::sibling(&m.body_fused) {
                    Some(f) => bind_ops(f),
                    None => Vec::new(),
                };
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
            if self.body_ops.is_empty() && self.prefix_ops.is_empty() && self.suffix_ops.is_empty()
            {
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
                    "[sdsc-superdsc] bundle {}: {} body + {} prefix + {} suffix launch group(s)",
                    self.code.fp,
                    self.body_ops.len(),
                    self.prefix_ops.len(),
                    self.suffix_ops.len()
                );
            }
            return Ok(());
        }
        // The ladder came ASCENDING by `active_cap` from the emit, so it stays so. The ceiling rung's
        // body IS this bundle, which the registry resolves like any other sibling.
        for r in rungs {
            self.body_rungs
                .push((r.active_cap, Self::sibling_ops("decode rung", &r.body)?));
            // A rung's fold-fused twin is optional: absent simply means this rung always uses the
            // split body — correct, one launch per layer dearer.
            if let Some(f) = bundle::sibling(&r.body_fused) {
                self.body_rungs_fused.push((r.active_cap, bind_ops(f)));
            }
        }
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
        for ops in self
            .lists()
            .into_iter()
            .chain(self.body_rungs.iter().map(|(_, o)| o))
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
        let iters = self.rolled.as_ref().map_or(0, |r| r.iters);
        let kv_stride = self.rolled.as_ref().map_or(0, |r| r.kv_stride);
        self.kv = KvGeometry {
            paged: true,
            page_slots: PageSlots(page_slots),
            page_stride_bytes: iters as u64 * kv_stride, // all layers of one page
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
        vec![&self.prefix_ops, &self.body_ops, &self.suffix_ops]
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
        } else if self.bound.contains_key(id) {
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
    pub fn bind_input(&mut self, id: bundle::PlaceId, bytes: Vec<u8>) {
        if !self.prepared {
            self.weight_ids.insert(id);
        }
        // A previously-raw id re-bound as fp16 must stop being raw, or the shadow would take an
        // unconverted fp16 buffer. Cleared here rather than left to the caller.
        self.raw_ids.remove(&id);
        self.bound.insert(id, bytes);
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
    /// ⛔ SHORT IS LEGITIMATE AND LONG IS NOT, which is why this is `>` and not `!=` — several
    /// activations bind less than their placement on purpose (pmask reserves `[nqh*mq, cap]` and
    /// binds one broadcast row; the index table's own pass block is the MASK's stride, of which one
    /// pass fills the head).
    pub fn place_raw(&self, id: bundle::PlaceId, bytes: Vec<u8>) -> Option<RawBind> {
        let p = self.places.get(&id)?;
        (bytes.len().is_multiple_of(4) && bytes.len() as u64 <= p.size)
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
        self.bound.insert(b.id, b.bytes);
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

        // ── Collect the weight placements, validating each against its own extent. ──
        struct Item<'a> {
            /// The SHADOW SLOT this weight stages into — a segment, or a weight bank beyond bank 0.
            /// See [`Executor::slot_of`]: banking gives one segment several regions, so the segment
            /// index alone stopped being an address.
            slot: usize,
            offset: usize,
            src: &'a [u8],
            walk: Option<(&'a KernelWeight<'a>, Element)>,
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
            let walk = match self.kernel_weights.get(nm) {
                Some(k) => {
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
                    Some((*k, elem))
                }
                None => None,
            };
            items.push(Item {
                slot,
                offset: p.offset as usize,
                src,
                walk,
            });
        }

        // ── Hand each item its OWN destination sub-slice. Splitting the segment buffers this way
        //    is what makes the parallel stage safe with no unsafe at all — and it PROVES the
        //    placements are disjoint (an overlap shows up as a backwards cursor, which is a
        //    refusal, where the C++ would simply have raced).
        /// One weight's staging work: its source bytes, its OWN destination sub-slice, and the
        /// tile walk (None = a plain IEEE→SEN convert).
        type StageTask<'a> = (
            &'a [u8],
            &'a mut [u8],
            Option<(&'a KernelWeight<'a>, Element)>,
        );
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
                tasks.push((it.src, dst, it.walk));
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
                    for (src, dst, walk) in bucket {
                        match walk {
                            Some((k, elem)) => stage_weight_tiled(
                                src,
                                dst,
                                RetileWalk {
                                    device_size: &k.device_size,
                                    stride_map: &k.stride_map,
                                },
                                elem,
                            ),
                            None => sen_convert::ieee_to_sen_bytes(src, dst),
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

        crate::sdk_abi::init_runtime()
            .map_err(|rc| anyhow!("flex runtime init failed (rc={rc})"))?;
        lap("runtime-init", &mut t0);
        // A DEDICATED created stream (not the default): predict destroys and recreates it each
        // forward to DRAIN the RuntimeScheduler's per-stream fence deques, which are otherwise
        // cleared only on destroy. A resident session's single long-lived stream accumulates a
        // promise per launch (~34,680 compute launches/forward) → ~2.57 GB/forward host leak.
        // The resident program/weight/KV allocations live on the runtime, not the stream, so
        // recreating it leaves them intact.
        self.stream =
            Some(Stream::create().ok_or_else(|| anyhow!("prepare: could not create a stream"))?);
        lap("stream-create", &mut t0);

        // ── Programs. Each op's device binary gets its OWN Program-segment alloc + H2D ONCE; they
        //    all SHARE the resident tensor segments below. ──
        {
            self.alloc_ops_at(ListSel::Prefix)?;
            if self.body_rungs.is_empty() {
                self.alloc_ops_at(ListSel::Body)?;
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
                        self.alloc_ops_at(ListSel::Rung(i))?;
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
                DevAddr::alloc(bytes, MemKind::Tensor)
                    .ok_or_else(|| anyhow!("prepare: seg{i} alloc of {bytes} B failed"))?,
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
            self.weight_bank_addr[b] =
                Some(DevAddr::alloc(bytes, MemKind::Tensor).ok_or_else(|| {
                    anyhow!("prepare: weight bank {} alloc of {bytes} B failed", b + 1)
                })?);
        }
        lap("seg-alloc", &mut t0);

        // ── FUSED TWINS ALLOCATE LAST ── They are the only allocation PAGING adds to this session,
        //    and everything above now happens in the order, and at the sizes, the unpaged path used
        //    — so the weight region lands exactly where it did before. Decode re-streams those
        //    weights every token and is bandwidth-bound on them; prefill is compute-bound at the
        //    roofline. An allocation ahead of the weights therefore taxes decode and leaves prefill
        //    alone, which is the exact shape of the gap. They are KB each, so being last is free.
        for i in 0..self.body_rungs_fused.len() {
            self.alloc_ops_at(ListSel::RungFused(i))?;
        }
        self.alloc_ops_at(ListSel::BodyFused)?;
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
            let need = self.seg_addr[i].as_ref().map_or(0, |a| a.total_size()) as usize;
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
                    .map_or(0, |a| a.total_size()) as usize;
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
                .map_err(|rc| anyhow!("prepare: seg{i} H2D rc={rc}"))?;
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
                    .map_err(|rc| anyhow!("prepare: weight bank {} H2D rc={rc}", b + 1))?;
            }
        }
        lap("tensor-h2d", &mut t0);
        self.stream_ref()?
            .synchronize()
            .map_err(|rc| anyhow!("prepare: sync rc={rc}"))?;
        lap("sync", &mut t0);
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
                    .prog_image(crate::sdk_abi::prog_offset_base())
                    .ok_or_else(|| {
                        anyhow!(
                            "program job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x}",
                            op.code.job_bin_ptr,
                            crate::sdk_abi::prog_offset_base()
                        )
                    })?;
                let head = img.bootstrap();
                let addr =
                    DevAddr::alloc(img.alloc_bytes(), MemKind::Program).ok_or_else(|| {
                        anyhow!(
                            "program alloc of {} B ({head} B prologue + {} B image) failed",
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
                    .window(head, op.code.init_binary.len() as u64)
                    .ok_or_else(|| anyhow!("program window at {head} B unaddressable"))?;
                self.stream_ref()?
                    .h2d(staged, &dst)
                    .map_err(|rc| anyhow!("program H2D rc={rc}"))?;
                op.addr = Some(addr);
            }
            // ⛔ BEFORE `pinnable` DROPS. The H2Ds above are asynchronous, so the staging buffers
            // must still exist when the DMA reads them. Programs are uploaded once per list at
            // load, so this sync costs nothing per step.
            if !pinnable.is_empty() {
                self.stream_ref()?
                    .synchronize()
                    .map_err(|rc| anyhow!("program H2D sync rc={rc}"))?;
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
            ListSel::Body => &mut self.body_ops,
            ListSel::Suffix => &mut self.suffix_ops,
            ListSel::BodyFused => &mut self.body_fused,
            ListSel::Rung(i) => &mut self.body_rungs[i].1,
            ListSel::RungFused(i) => &mut self.body_rungs_fused[i].1,
        }
    }

    /// The same list, read-only — for the per-body facts [`StepBody`] carries, which are read off the
    /// body's own OPS.
    fn list(&self, sel: ListSel) -> &Ops {
        match sel {
            ListSel::Prefix => &self.prefix_ops,
            ListSel::Body => &self.body_ops,
            ListSel::Suffix => &self.suffix_ops,
            ListSel::BodyFused => &self.body_fused,
            ListSel::Rung(i) => &self.body_rungs[i].1,
            ListSel::RungFused(i) => &self.body_rungs_fused[i].1,
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
        Ok(StepBody {
            sel,
            swept: self.body_swept(sel),
            // ⭐ THE BODY'S OWN OPS ANSWER IT, not a flag beside them and not the bundle's placement.
            // Same discipline as `latch_paged_geometry` reading `page_slots` off the ops: a bake fact
            // rides on the launch group that carries it (`bundle::KvShifts::gathered`), so a body that
            // gathers cannot be driven as one that does not, or the reverse.
            gathers: crate::wiring::GathersKv::of_launch_groups(
                self.list(sel).iter().map(|o| o.kv.gathered),
            ),
        })
    }

    /// THIS body's swept extent — its own ladder rung's `active_cap`.
    ///
    /// The ladderless arms (`Body`/`BodyFused`: an UNROLLED bundle, which has no `rungs` and therefore
    /// no ladder) sweep their whole baked capacity, and on the paged path that is one page — the number
    /// the ops themselves carry. It is the value the host's guard used for every bundle before the
    /// ladder's own extent was plumbed through, so a ladderless bundle behaves exactly as it did.
    fn body_swept(&self, sel: ListSel) -> SweptCols {
        match sel {
            ListSel::Rung(i) => self.body_rungs[i].0,
            ListSel::RungFused(i) => self.body_rungs_fused[i].0,
            _ => SweptCols::new(self.kv.page_slots.0.max(0) as u32),
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
            self.alloc_ops_at(ListSel::Rung(i))?;
        }
        // Their binaries must land before the first launch.
        self.stream_ref()?
            .synchronize()
            .map_err(|rc| anyhow!("split-body sync rc={rc}"))?;
        self.split_bodies_ready = true;
        debug!(
            "[sdsc-superdsc] allocated {} split decode bodies (a fold that re-launches needs them)",
            self.body_rungs.len()
        );
        Ok(())
    }

    /// Select the SPLIT decode body for a valid KV length of `valid_len` positions: the smallest
    /// rung whose `active_cap` ≥ it, which bounds the O(cap) attention sweep. Every return is a
    /// split body, so this is also where they are guaranteed to EXIST.
    fn select_body(&mut self, valid_len: i64) -> Result<ListSel> {
        self.ensure_split_bodies()?;
        let reqs = self.fold_requests_or_1();
        if self.body_rungs.is_empty() {
            // The ladderless single body — traced for the same reason as the fused one: an untraced
            // return is indistinguishable from a step that never reached the selector at all.
            self.trace_body("split-noladder", valid_len, -1, reqs, 0, false);
            return Ok(ListSel::Body);
        }
        for (i, (swept, _)) in self.body_rungs.iter().enumerate() {
            // ⛔ `SweptCols::covers` AND NOT `>=` ON BARE INTEGERS. The sibling ladder is keyed by the
            // batch WIDTH, so an operand-swapped comparison — or one against the wrong ladder — used to
            // type-check and return a body baked for a 64-column sweep because four requests are live.
            if swept.covers(valid_len.max(0) as u64) {
                self.trace_body("split", valid_len, -1, reqs, swept.get() as i64, false);
                return Ok(ListSel::Rung(i));
            }
        }
        // ⚠️ THE FALLBACK, VISIBLE. A `valid_len` past the top rung takes the ceiling; silently,
        // this looked identical to a correct choice.
        let top = self.body_rungs.len() - 1;
        let cap = self.body_rungs[top].0.get() as i64;
        self.trace_body("split-CEILING", valid_len, -1, reqs, cap, false);
        Ok(ListSel::Rung(top))
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
    fn select_body_paged(&mut self, valid_len: i64, n_fold_pages: i64) -> Result<ListSel> {
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
                        return Ok(ListSel::RungFused(i));
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
            if !self.body_fused.is_empty() {
                self.trace_body("fused-noladder", valid_len, n_fold_pages, reqs, 0, true);
                return Ok(ListSel::BodyFused);
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

    fn launch_ops_inner(&mut self, ops: &Ops, where_: LaunchWhere, d: &Diag) -> Result<()> {
        let LaunchWhere {
            byte_off,
            slot_pos,
            kv_page_base,
            n_fold_pages,
            weight_bank,
        } = where_;
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

        for (oi, op) in ops.iter().enumerate() {
            // ── PAGE FOLD ── Run this group once per page of resident prefix, rebased each time to
            //    that page's KV and that page's validity row. The fold ACCUMULATES into the running
            //    softmax state, so repeating it IS the loop. Zero passes when nothing is resident
            //    yet — the common case for a first prefill chunk, which attends only itself and
            //    would otherwise sweep an empty page. This is what makes context a RUNTIME page
            //    count instead of a baked extent. Whose KV an op touches and how many passes it
            //    takes are both `fold_plan`'s to answer.
            let reps = fold_plan::reps(&op.kv, &self.fold, n_fold_pages);
            if reps < 0 {
                // A NEGATIVE COUNT IS A REFUSAL, not a loop bound: the fold would take more passes
                // than the mask has blocks, and the bytes past the last block are unstaged zeros
                // that an additive mask reads as VALID. Abandon rather than fold in whatever the
                // pool holds — loud and incomplete beats quiet and wrong.
                // ⛔ NAME WHICH REFUSAL FIRED. `reps` returns -1 for TWO unrelated reasons and this
                // message used to describe only the second, so a page-count DISAGREEMENT read as a
                // capacity shortfall and sent the reader off to enlarge a mask that was big enough.
                let (declared_pages, blocks) = (self.fold.fold_pages, self.fold.mask_blocks);
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

            for rep in 0..reps {
                let stream = self.stream_ref()?;
                let mut off = byte_off;
                // WHERE THIS OP READS AND WRITES KV — resolved in `fold_plan`. It composes, in this
                // order: the fold pass's page and validity row, the write slot within its page, the
                // page a non-fold op works in, and the current slab.
                //
                // ⭐ THE WITNESS IS MINTED HERE. `UniformPages::sweeping` checks the sweep against
                // the pool that is actually installed; a disagreement is a REFUSAL, never an
                // address — the alternative is a pass naming a page its row does not own.
                let Some(per) = UniformPages::sweeping(n_fold_pages, &self.fold) else {
                    bail!(
                        "fold_plan refused op {oi} (rep {rep}): the installed page maps do not \
                         admit a uniform {n_fold_pages}-page sweep"
                    );
                };
                let delta = fold_plan::seg_deltas(
                    &op.kv,
                    &self.fold,
                    rep,
                    per,
                    SlotPos::of_launch(slot_pos.0),
                    kv_page_base,
                );
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
                let wbase = match weight_bank.checked_sub(1) {
                    None => self.seg_addr[SEG_WEIGHT.get()].as_ref(),
                    Some(b) => self
                        .weight_bank_addr
                        .get(b)
                        .ok_or_else(|| {
                            anyhow!(
                                "launch: weight bank {weight_bank} has no region ({} bank(s))",
                                self.weight_bank_bytes.len() + 1
                            )
                        })?
                        .as_ref(),
                };
                let mut shifted: [Option<DevAddr>; NUM_SEGMENTS] = [const { None }; NUM_SEGMENTS];
                for i in 0..NUM_SEGMENTS {
                    let sh = off[i];
                    let base = if i == SEG_WEIGHT.get() {
                        wbase
                    } else {
                        self.seg_addr[i].as_ref()
                    };
                    if sh > 0
                        && let Some(base) = base
                        && base.is_single_chunk()
                    {
                        shifted[i] = base.shifted(sh);
                    }
                }
                let ta: Vec<&DevAddr> = (0..NUM_SEGMENTS)
                    .map(|i| {
                        let base = if i == SEG_WEIGHT.get() {
                            wbase
                        } else {
                            self.seg_addr[i].as_ref()
                        };
                        shifted[i]
                            .as_ref()
                            .or(base)
                            .ok_or_else(|| anyhow!("seg{i} has no region"))
                    })
                    .collect::<Result<_>>()?;

                let prog = op
                    .addr
                    .as_ref()
                    .ok_or_else(|| anyhow!("op {oi} has no program allocation"))?;
                // A BARRIER AFTER EVERY GROUP IS WHAT MAKES A 2 KB COPY COST A LAUNCH. It is a
                // device-wide settle, so consecutive groups cannot overlap however small they are —
                // and a batched decode layer is B per-request cache writes in a row, each moving
                // 2 KB into a DIFFERENT request's page. They are disjoint by construction, so
                // ordering them against EACH OTHER buys nothing; only the last has to settle before
                // the Kᵀ restickify and the next layer read what they wrote. Drop the barrier ONLY
                // between consecutive slot-writes; STRICT_ORDERING still holds the stream in order.
                let next_is_slot_write =
                    op.kv.slot_write && ops.get(oi + 1).is_some_and(|n| n.kv.slot_write);
                let barrier = !next_is_slot_write;

                // Match torch-spyre's JobPlanStepCompute EXACTLY: kernel_name="" (NOT the mlir
                // path) — it runs the resident dxp binary via its binary address, and passing the
                // mlir path may route flex down a different (seed-only) path for the SFP
                // transcendental while the matmul tolerates it. #2814: the 4th argument is the
                // bootstrap OFFSET, not the full VA (flex bounds the seg7 translation to the
                // program allocation's size).
                // ⛔ THE SAME VALUE `alloc_ops_at` UPLOADED AT, not a second subtraction. This line was
                // `op.code.job_bin_ptr - prog_offset_base()` — bare, so an underflow wrapped here while
                // the allocation path refused it, and the two could only agree by both being written
                // correctly. `ProgImage` is now the one door: see `LaunchGroup::prog_image`.
                let bootstrap = op
                    .code
                    .prog_image(crate::sdk_abi::prog_offset_base())
                    .ok_or_else(|| {
                        anyhow!(
                            "op {oi}: job_bin_ptr {:#x} is below PROG_OFFSET_BASE {:#x} — it names no \
                             bootstrap offset at all",
                            op.code.job_bin_ptr,
                            crate::sdk_abi::prog_offset_base()
                        )
                    })?
                    .bootstrap();
                let gt0 = Instant::now();
                if d.launch_mark {
                    // ⛔ ONE WRITE, and it must reach the fd BEFORE the submit: an abort from senlib's
                    // monitor thread takes the process down with no unwinding, so a buffered line is a
                    // line that never existed. Same single-`format!` discipline as `[progverify]`.
                    let one = format!("[mark] op[{}] rep={rep} {}\n", op.index, op.label());
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
                let fault = |rc: i32| -> anyhow::Error {
                    self.dump_launch_site("FAULT(submit-refused)", &ta, op, prog, bootstrap, &off);
                    let sweep = self.prog_images_sweep("FAULT", ops);
                    eprint!("{sweep}");
                    launch_err(rc)
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

                if d.submit_time {
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
                        t.barrier_launches += 1;
                    }
                } else {
                    stream
                        .compute(prog, &ta, "", bootstrap, &[], barrier)
                        .map_err(fault)?;
                }

                // DEFAULT-OFF: skipping the per-group drain removes the ~868-ops/layer
                // serialization — pipeline_barrier + STRICT_ORDERING preserve the on-device
                // dataflow (pod-confirmed coherent and ~11% faster).
                if d.perop_sync {
                    let _ = stream.synchronize();
                }
                if d.group_time {
                    // Needs a drain to attribute anything, so it costs what the per-op sync costs.
                    let _ = stream.synchronize();
                    let ms = gt0.elapsed().as_secs_f64() * 1e3;
                    let mut t = timers();
                    let e = t.group_ms.entry(oi).or_insert((0.0, 0));
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
                self.optrace(oi)?;
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
                        c.region, c.offset, c.size, off[i]
                    );
                }
                _ => {
                    let _ = writeln!(out, "[segaddr] {who} seg{i} MULTI-CHUNK shift={}", off[i]);
                }
            }
        }
        let base = crate::sdk_abi::prog_offset_base();
        match prog.chunk0() {
            Some(c) => {
                let _ = writeln!(
                    out,
                    "[segaddr] {who} seg7/PROG region={} offset={} size={} (binary {} B)",
                    c.region,
                    c.offset,
                    c.size,
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
            prog.total_size(),
            prog.total_size() / u64::from(crate::lower_subtile_tape_to_superdsc::STICK_BYTES),
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
            .saturating_sub(crate::sdk_abi::prog_offset_base());
        let Some(win) = prog.window(head, baked.len() as u64) else {
            return "progimg: program window unaddressable".to_string();
        };
        let mut host = vec![0u8; baked.len()];
        if let Err(rc) = stream.d2h(&mut host, &win) {
            return format!("progimg: D2H rc={rc} — THE READBACK ITSELF FAILED, no verdict");
        }
        if let Err(rc) = stream.synchronize() {
            return format!("progimg: D2H sync rc={rc} — THE READBACK ITSELF FAILED, no verdict");
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
    fn optrace(&mut self, oi: usize) -> Result<()> {
        let _ = self.stream_ref()?.synchronize();
        let mut line = format!("[optrace] op[{oi}]");
        for i in [0usize, 2, 3, 4, 5, 6] {
            let seg = SegIdx::checked(i as i64).expect("literal < NUM_SEGMENTS");
            let got = self.d2h_seg_clamped(seg);
            let stat = match got {
                Ok(got) => {
                    let host = &self.seg_host[i][..got.min(self.seg_host[i].len())];
                    let (mut nz, mut mx) = (0usize, 0.0f32);
                    // `as_chunks` over `chunks_exact`: a CONSTANT chunk width yields `&[u8; 2]`, so the
                    // pair cannot be short (clippy::chunks_exact_to_as_chunks).
                    for c in host.as_chunks::<2>().0 {
                        let bits = u16::from_le_bytes(*c);
                        if bits != 0 {
                            nz += 1;
                            let v = sen_convert::sen_to_f32(sen_convert::SenF16(bits)).abs();
                            if v > mx {
                                mx = v;
                            }
                        }
                    }
                    format!("nz={nz} max={mx:.3}")
                }
                Err(_) => "unreadable".to_string(),
            };
            line.push_str(&format!(" | seg{i} {stat}"));
        }
        eprintln!("{line}");
        Ok(())
    }

    // ──────────────────────────────────────────────────────────────────────────────────────────
    //  Predict
    // ──────────────────────────────────────────────────────────────────────────────────────────

    /// Run ONE forward over `n_new` NEW positions appended at `[seq_pos..seq_pos+n_new)`: refill
    /// the bound activations, launch prefix → body×iters → suffix, and copy the logits row back.
    /// `out_logits` is `None` for a chunk whose logits nobody reads (every prefill chunk but the
    /// last), which skips the read-back entirely.
    pub fn predict(
        &mut self,
        n_new: NewTokens,
        seq_pos: SeqPos,
        out_logits: Option<&mut [u8]>,
    ) -> Result<()> {
        if !self.prepared {
            bail!("predict before prepare");
        }
        self.require_sources_filled()?;
        let d = Diag::get();
        let n_new = NewTokens(n_new.0.max(1));

        // LEAK FIX: recreate the stream each forward so the destroy DRAINS the scheduler's
        // per-stream fence deques (~2.57 GB/forward otherwise → 200 GB OOM). The previous forward
        // already synchronized (destroy also implicitly syncs); the resident weight/KV/program
        // allocations live on the runtime, so device state survives.
        let t_stream0 = Instant::now();
        self.stream = None; // drop = destroyStream
        self.stream =
            Some(Stream::create().ok_or_else(|| anyhow!("predict: stream recreate failed"))?);

        let t0 = Instant::now();
        let (h2d_note, h2d_bytes) = self.refill_activations(d)?;
        let t0b = Instant::now();
        if d.phase_time {
            let _ = self.stream_ref()?.synchronize();
        }
        let t1 = Instant::now();

        self.launch_forward(seq_pos)?;

        if d.phase_time {
            let _ = self.stream_ref()?.synchronize();
        }
        let t2 = Instant::now();

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
            let tot = addr.total_size();
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
            let lo = lp.offset;
            let len = crate::lower_subtile_tape_to_superdsc::align128(lp.size).min(tot - lo);
            // A multi-chunk region has no addressable sub-window (`chunk0` fails); fall back to the
            // whole-region read, which is what this always did.
            let win = (len < tot).then(|| addr.window(lo, len)).flatten();
            let (src, a, b) = match win.as_ref() {
                Some(w) => (w, lo, lo + len),
                None => (addr, 0, tot),
            };
            d2h_bytes = b - a;
            let host = &mut self.seg_host[seg.get()][a as usize..b as usize];
            self.stream
                .as_ref()
                .unwrap()
                .d2h(host, src)
                .map_err(|rc| anyhow!("logits D2H rc={rc}"))?;
        }
        let t2b = Instant::now();
        self.stream_ref()?
            .synchronize()
            .map_err(|rc| anyhow!("predict: sync rc={rc}"))?;

        if d.phase_time {
            let t3 = Instant::now();
            let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
            eprintln!(
                "[sdsc-phase] stream_recreate={:.2}  preamble_h2d={:.2}  compute={:.2}  \
                 logits_d2h={:.2}  predict_total={:.2} ms  (n_new={})\n\
                 [sdsc-phase]   h2d: {}KB over{} | enqueue={:.2} sync={:.2}\n\
                 [sdsc-phase]   d2h: seg{} {}KB | enqueue={:.2} sync={:.2}",
                ms(t_stream0, t0),
                ms(t0, t1),
                ms(t1, t2),
                ms(t2, t3),
                ms(t_stream0, t3),
                n_new.0,
                h2d_bytes >> 10,
                if h2d_note.is_empty() {
                    " (nothing)".to_string()
                } else {
                    h2d_note
                },
                ms(t0, t0b),
                ms(t0b, t1),
                d2h_seg,
                d2h_bytes >> 10,
                ms(t2, t2b),
                ms(t2b, t3),
            );
            report_phase_timers(ms(t1, t2));
        }

        if let Some(out) = out_logits {
            let seg = SegIdx::checked(lp.segment as i64).unwrap();
            let elems = self.vocab.0.min(out.len() / 2);
            let src = &self.seg_host[seg.get()][lp.offset as usize..];
            sen_convert::sen_to_ieee_bytes(&src[..elems * 2], &mut out[..elems * 2]);
        }

        // FLEX-LEAK PROBE (one-shot): after the N-th forward, tear the stream + the whole runtime
        // down. This BREAKS the session (no re-prepare) — measure, then stop the serve.
        if let Some(n) = d.reset_probe {
            let mut t = timers();
            t.forwards += 1;
            if t.forwards == n {
                let before = vmrss_kb();
                self.stream = None;
                let _ = crate::sdk_abi::runtime_reset();
                let after = vmrss_kb();
                eprintln!(
                    "[sdsc-leak-probe] fwd={n} VmRSS {before} -> {after} kB (freed {} kB)",
                    before - after
                );
            }
        }
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
        // Slicing to the bound input's EXACT [offset, size) was tried and tripped a DMA
        // sub-address/alignment error. This sends an ALIGNED COVERING WINDOW instead — `lo` down and
        // `hi` up to `H2D_ALIGN` — which needs no knowledge of the hardware's exact requirement,
        // since any alignment coarser than it is also legal. It degrades to the whole segment
        // whenever a window cannot be formed, so the worst case is exactly the old behaviour.
        const H2D_ALIGN: u64 = 4096;
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
            let a = (p.offset / H2D_ALIGN) * H2D_ALIGN;
            let e = p.offset + src.len() as u64;
            let b = e.div_ceil(H2D_ALIGN) * H2D_ALIGN;
            seg_runs[seg.get()].push((a, b));
            if d.phase_time {
                eprintln!(
                    "[sdsc-bind] '{nm}' -> seg{} off={} inputsize={:.3} MB",
                    p.segment,
                    p.offset,
                    src.len() as f64 / 1048576.0
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
            let tot = addr.total_size();
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
            if runs.is_empty() {
                stream
                    .h2d(whole_shadow(&self.seg_host[i], addr, i)?, addr)
                    .map_err(|rc| anyhow!("preamble H2D seg{i} rc={rc}"))?;
            } else {
                for (a, b) in runs {
                    let win = addr
                        .window(a, b - a)
                        .ok_or_else(|| anyhow!("seg{i} window {a}..{b} unaddressable"))?;
                    stream
                        .h2d(&self.seg_host[i][a as usize..b as usize], &win)
                        .map_err(|rc| anyhow!("preamble H2D seg{i} window rc={rc}"))?;
                }
            }
        }
        Ok((note, total))
    }

    /// The forward itself: prefix → body×iters → suffix (rolled), or the single body.
    fn launch_forward(&mut self, seq_pos: SeqPos) -> Result<()> {
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
            // UNROLLED: one body over the whole model, and an unrolled bundle has no layer boundary
            // to bank on — its weights are one region by construction (`bank_weight_segment` refuses
            // to bank without layer classes), so bank 0 is the only bank there is.
            return self.launch_ops(sel, zero, seq_pos, base, n_fold_pages, 0);
        };
        let (iters, wstride, kvstride) = (rolled.iters, rolled.weight_stride, rolled.kv_stride);
        let (lpb, pre_bank, suf_bank) = (
            rolled.layers_per_bank.max(1),
            rolled.prefix_weight_bank,
            rolled.suffix_weight_bank,
        );

        self.launch_ops(ListSel::Prefix, zero, SeqPos(0), Bytes(0), 1, pre_bank)?;

        // sk_bucket ladder: pick this token's decode body rung by its KV length (valid_len =
        // seq_pos+1 — positions [0..seq_pos] are filled, including the token just written to KV
        // before attention). The same rung serves every layer this step; only the swept extents
        // differ, and the resident KV is shared.
        let n_fold_pages = self.n_fold_pages(seq_pos);
        let sel = self.step_body(seq_pos)?.sel;
        let base = self.page_base_for_write(seq_pos);
        for v in 0..iters {
            let mut off = [0u64; NUM_SEGMENTS];
            // ⭐ THE PER-LAYER WEIGHT ADDRESS, IN TWO COORDINATES. `v·wstride` alone was a byte
            // offset into ONE region, which capped a model's weights at that region's 16 GiB. Layer
            // `v` is now `(v / lpb)`'s BANK at `(v % lpb)·wstride` — the same descriptors, a
            // different base — so the ceiling is banks × 16 GiB. `lpb == iters` for an unbanked
            // bundle, which makes this `(0, v·wstride)`: byte-identical to what it replaced.
            //
            // ⛔ THE DIVISION AND THE REMAINDER MUST AGREE WITH THE PLACEMENTS, and they are not
            // checked here — they are PROVEN at cargo-build, per layer, against the addresses that
            // were actually baked (see the per-layer formula guard in `lower_subtile_tape_to_superdsc`).
            let wbank = (v / lpb) as usize;
            off[SEG_WEIGHT.get()] = (v % lpb) as u64 * wstride;
            off[SEG_KV.get()] = v as u64 * kvstride;
            // The WRITE lands in the page holding seq_pos; the FOLD covers every page the RESIDENT
            // PREFIX [0, seq_pos) spans — zero of them when there is no prefix.
            self.launch_ops(sel, off, seq_pos, base, n_fold_pages, wbank)?;
        }
        // NO HOST ROUTING: the loop-carried residual + the body→suffix seam thread IN-PLACE via
        // emitter placement aliasing (hidden_out/suffix_in aliased onto hidden_in's resident
        // buffer). No D2H/memcpy/H2D — the residual lives on-device across iterations.
        self.launch_ops(ListSel::Suffix, zero, SeqPos(0), Bytes(0), 1, suf_bank)
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
        RowPage::of(&self.fold, RowIdx::from_launch_row(0), lp)
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
            None => (ListSel::Body, 0),
        };
        self.launch_ops(sel, zero, SeqPos(0), Bytes(0), 1, bank)?;
        self.stream_ref()?
            .synchronize()
            .map_err(|rc| anyhow!("run_prefix_only sync rc={rc}"))
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
        self.launch_forward(SeqPos(0))?;
        self.stream_ref()?
            .synchronize()
            .map_err(|rc| anyhow!("run_full sync rc={rc}"))?;
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
            .total_size();
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
                .map_err(|rc| anyhow!("seg{i} D2H rc={rc}"))?;
            stream
                .synchronize()
                .map_err(|rc| anyhow!("seg{i} D2H sync rc={rc}"))?;
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
                .window(0, host)
                .ok_or_else(|| anyhow!("seg{i} window unaddressable"))?;
            stream
                .d2h(&mut seg_host[i][..host as usize], &win)
                .map_err(|rc| anyhow!("seg{i} D2H rc={rc}"))?;
            stream
                .synchronize()
                .map_err(|rc| anyhow!("seg{i} D2H sync rc={rc}"))?;
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
    pub fn zero_seg(&mut self, seg: SegIdx) -> Result<()> {
        let i = seg.get();
        let addr = self.seg_addr[i]
            .as_ref()
            .ok_or_else(|| anyhow!("seg{i} has no region"))?;
        if !addr.is_single_chunk() {
            bail!("zero_seg(seg{i}): region is multi-chunk");
        }
        let total = addr.total_size();
        if total == 0 {
            return Ok(());
        }
        const CHUNK: u64 = 8 << 20;
        let zeros = vec![0u8; CHUNK.min(total) as usize];
        let stream = self.stream_ref()?;
        let mut off = 0u64;
        while off < total {
            let sz = CHUNK.min(total - off);
            let win = addr
                .window(off, sz)
                .ok_or_else(|| anyhow!("seg{i} window unaddressable"))?;
            stream
                .h2d(&zeros[..sz as usize], &win)
                .map_err(|rc| anyhow!("zero H2D rc={rc}"))?;
            // The window must outlive the transfer.
            stream
                .synchronize()
                .map_err(|rc| anyhow!("zero sync rc={rc}"))?;
            off += sz;
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
            .window(0, data.len() as u64)
            .ok_or_else(|| anyhow!("seg{i} window 0..{} unaddressable", data.len()))?;
        let stream = self.stream_ref()?;
        stream
            .h2d(&self.seg_host[i][..data.len()], &win)
            .map_err(|rc| anyhow!("write_seg_prefix H2D seg{i} rc={rc}"))?;
        stream
            .synchronize()
            .map_err(|rc| anyhow!("write_seg_prefix sync seg{i} rc={rc}"))
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
            .map_err(|rc| anyhow!("write_seg H2D rc={rc}"))?;
        stream
            .synchronize()
            .map_err(|rc| anyhow!("write_seg sync rc={rc}"))
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
                dst.total_size(),
                owner.segment_bytes[i],
                src.total_size()
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
        self.seg_addr[i] = Some(
            DevAddr::from_chunk(c0).ok_or_else(|| anyhow!("alias_seg(seg{i}): rebuild failed"))?,
        );
        // The borrowed segment is marked so this session's preamble can never H2D its own stale
        // host shadow over the owner's live data.
        self.seg_aliased[i] = true;
        debug!(
            "[sdsc-superdsc] alias_seg(seg{i}): dst now SHARES src's region ({} B)",
            c0.size
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
                self.weight_bank_addr[b] = Some(DevAddr::from_chunk(bc0).ok_or_else(|| {
                    anyhow!("alias_seg(seg{i}): weight bank {} rebuild failed", b + 1)
                })?);
                self.weight_bank_aliased[b] = true;
            }
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

/// Which op list a selection names. An index rather than a reference so the selector can hand one
/// back while `&mut self` is still needed to allocate it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ListSel {
    Prefix,
    Body,
    Suffix,
    BodyFused,
    Rung(usize),
    RungFused(usize),
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
    sel: ListSel,
    swept: SweptCols,
    gathers: crate::wiring::GathersKv,
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

fn launch_err(rc: i32) -> anyhow::Error {
    anyhow!("compute launch rc={rc}")
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
        .map(|(i, (ms, n))| (*ms, *i, *n))
        .collect();
    top.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!(
        "[sdsc-grouptime] top ops by total ms (of {total:.1} ms over {} launches)",
        t.group_calls
    );
    for (ms, i, n) in top.iter().take(20) {
        eprintln!(
            "[sdsc-grouptime]   op {i:<5}  {ms:8.1} ms total  {:6.3} ms/launch  {n:6} launches  \
             {:5.1}%",
            ms / *n as f64,
            100.0 * ms / total.max(1e-9)
        );
    }
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
            "[sdsc-phase]   host: {} launches ({} with a device-wide barrier) | submit={:.2} ms \
             ({:.1} us/ea) prep={:.2} ms ({:.1} us/ea) | = {:.0}% of compute",
            t.submit_calls,
            t.barrier_launches,
            t.submit_ms,
            1000.0 * t.submit_ms / t.submit_calls as f64,
            t.prep_ms,
            1000.0 * t.prep_ms / t.submit_calls as f64,
            100.0 * (t.submit_ms + t.prep_ms) / compute_ms.max(1e-9),
        );
        t.submit_ms = 0.0;
        t.prep_ms = 0.0;
        t.submit_calls = 0;
        t.barrier_launches = 0;
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
}
