// SPDX-License-Identifier: Apache-2.0
//! Gated-DeltaNet (GDN) recurrent-state pool — persistent GPU memory for the
//! linear-attention layers of hybrid models (Qwen3.5 / Qwen3-Next).
//!
//! This is the non-paged sibling of `KvCachePool`. A GDN
//! layer keeps **two** state buffers, sized **one slot per concurrently
//! resident sequence** (`num_slots = max_num_seqs`), NOT by token blocks:
//!
//! ```text
//!   conv_state : [num_slots, conv_dim, conv_kernel-1]            (causal-conv1d ring)
//!   ssm_state  : [num_slots, num_v_heads, head_v_dim, head_k_dim] (recurrent delta-rule state)
//! ```
//!
//! Both buffers are **f32** (the `gdn_*` kernels operate in f32, and the
//! model's `mamba_ssm_dtype` is `float32`). The slot a sequence owns is
//! decided host-side by the forward compiler's `GdnSlotAllocator`;
//! the per-step `state_indices` tensor (one slot id per batched sequence) is
//! built by the worker and consumed by the kernels.
//!
//! Unlike `KvCachePool`, which allocates every layer, only **linear-attention
//! (GDN) layers** get buffers: full-attention layers store `None` at their
//! global layer index. This is the single intentional divergence from the
//! `KvCachePool` template — it keeps the index space global (so `conv_state(l)`
//! takes the model's global layer index directly) while not wasting the large
//! `ssm_state` allocation on full-attention layers.
//!
//! Like `KvCachePool` the buffer-allocation is a caller-supplied closure (metal
//! wraps `device.new_buffer`); the layout /
//! sizing / accessor logic is backend-neutral.
//!
//! **Checkpoints (speculative decoding).** A verify step runs a sequence's last token and its
//! drafts as rows of one step, and a rejected draft must not stay in the recurrent state.
//!
//! The causal conv's state is small: with `checkpoint_rows = k` a verify step keeps it after each
//! of its `k + 1` rows, and the next step starts from the one at the last kept row.
//!
//! The recurrent (ssm) state is not — a full state per row would be most of the pool, and the
//! pool is sized for every resident sequence. A slot keeps ONE state entry, and a verify step
//! records its rows' inputs instead: it leaves the state it started from in the entry and records
//! each of its rows' conv row, decay and `β` (KBs); the next step starts from the entry and
//! replays the kept rows from their records through the scan's own loop — the same arithmetic, so
//! the same state ([`GdnStart::Checkpoint`]). The records alternate between two areas of `k + 1`
//! rows each, so a step replays one while it writes the other; which is current ([`RecordArea`])
//! is the host's to track ([`GdnStep::after`]).
//!
//! **Draft lanes.** Only a sequence that drafts needs those checkpoints and records, from the step
//! it drafts in through the step that replays its verify step's kept rows — and a step drafts for
//! only as many sequences as its head's budget drafts for. So they live apart from the slots, in
//! [`DraftLanes`]:
//! per layer a conv buffer of `k + 1` entries a lane and an ssm buffer of two record areas a lane
//! (`GdnStateDims::ssm_lane_len`). A slot is the same entry with drafts as without, and a step that
//! replays or records names its sequence's lane in its code ([`GdnStep::lane`]). The host hands
//! lanes out ([`DraftLane`]): a sequence takes one when it drafts and gives it back after the step
//! that no longer needs it.

use std::num::NonZeroU16;

use anyhow::Result;
use scratchy_tensors::{DType, GpuTensor, PoolMemory, TensorView};

/// GDN recurrent-state buffers are always f32 (kernels + `mamba_ssm_dtype`).
pub const GDN_STATE_DTYPE: DType = DType::F32;

/// The drafts a verify step carries: it checkpoints the conv state after, and records the ssm
/// inputs of, each of its `drafts + 1` rows (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointRows(pub u8);

impl CheckpointRows {
    /// No checkpoints: a deployment without speculative decoding.
    pub const NONE: Self = Self(0);

    /// The rows a verify step of these drafts checkpoints: its sequence's token and its drafts.
    pub fn rows(self) -> usize {
        usize::from(self.0) + 1
    }
}

/// The draft lanes of a pool whose model drafts (module docs): how many sequences hold draft
/// state at once, and the drafts each lane's verify step carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftLanes {
    rows: CheckpointRows,
    count: NonZeroU16,
}

impl DraftLanes {
    /// The most lanes a step code can name ([`GdnStep::encode`]: bits 17–31).
    pub const MAX: u16 = (1 << 15) - 1;

    /// `count` lanes of `rows` drafts; `None` when the model drafts nothing or keeps no lane.
    ///
    /// # Panics
    /// When `count` is past [`Self::MAX`].
    pub fn new(rows: CheckpointRows, count: u16) -> Option<Self> {
        assert!(
            count <= Self::MAX,
            "{count} draft lanes: a step code names at most {}",
            Self::MAX
        );
        let count = NonZeroU16::new(count)?;
        (rows != CheckpointRows::NONE).then_some(Self { rows, count })
    }

    pub fn rows(self) -> CheckpointRows {
        self.rows
    }

    pub fn count(self) -> u16 {
        self.count.get()
    }

    /// Every lane.
    pub fn lanes(self) -> impl DoubleEndedIterator<Item = DraftLane> {
        (0..self.count.get()).map(DraftLane)
    }
}

/// A lane of a pool's [`DraftLanes`]: one sequence's conv checkpoints and ssm records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DraftLane(u16);

impl DraftLane {
    pub fn index(self) -> u16 {
        self.0
    }
}

/// Which of a lane's two record areas holds the previous verify step's draft rows (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecordArea {
    #[default]
    First,
    Second,
}

impl RecordArea {
    pub fn other(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }
}

/// The state a sequence's GDN layers start a step from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnStart {
    /// The state its slot holds — the previous step's last row.
    Slot,
    /// Zero: the sequence's first step.
    Fresh,
    /// The state after row `r` of the previous step, a verify step whose rows up to `r` were kept:
    /// its start state replayed through rows `0..=r`.
    Checkpoint(u8),
}

/// What one step does with one sequence's GDN state, as the per-sequence `u32` the GDN kernels
/// read (the buffer the forward calls `is_fresh`: `0` and `1` keep their meaning).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnStep {
    pub start: GdnStart,
    /// The step's drafts — a verify step's: it checkpoints the conv state after, and records the
    /// ssm inputs of, its `drafts + 1` rows. [`CheckpointRows::NONE`] outside speculative decoding.
    /// A lane's layout is the model's drafts, which its kernels bake (`SPEC_DRAFTS`).
    pub checkpoint_rows: CheckpointRows,
    /// The lane's record area before the step: the previous verify step's rows.
    pub records: RecordArea,
    /// The sequence's draft lane: the previous verify step's checkpoints and records a checkpoint
    /// start replays, and the ones a verify step writes. `None` exactly on a plain step
    /// ([`Self::is_plain`]).
    pub lane: Option<DraftLane>,
}

impl GdnStep {
    /// The kernels' encoding: bits 0–7 the start (`0` slot, `1` fresh, `2 + r` checkpoint `r`),
    /// bits 8–15 the step's drafts, bit 16 the record area, bits 17–31 the lane.
    pub fn encode(self) -> u32 {
        debug_assert_eq!(
            self.lane.is_none(),
            self.is_plain(),
            "{self:?}: a lane exactly when not plain"
        );
        let start = match self.start {
            GdnStart::Slot => 0,
            GdnStart::Fresh => 1,
            GdnStart::Checkpoint(r) => 2 + u32::from(r),
        };
        let records = u32::from(self.records == RecordArea::Second);
        let lane = self.lane.map_or(0, |l| u32::from(l.0));
        start | (u32::from(self.checkpoint_rows.0) << 8) | (records << 16) | (lane << 17)
    }

    /// The step an [`Self::encode`] code describes.
    pub fn decode(code: u32) -> Self {
        let [start, drafts, records, _] = code.to_le_bytes();
        let mut step = Self {
            start: match start {
                0 => GdnStart::Slot,
                1 => GdnStart::Fresh,
                s => GdnStart::Checkpoint(s - 2),
            },
            checkpoint_rows: CheckpointRows(drafts),
            records: match records & 1 {
                0 => RecordArea::First,
                _ => RecordArea::Second,
            },
            lane: None,
        };
        let lane = u16::try_from(code >> 17).expect("a lane is the code's top 15 bits");
        step.lane = (!step.is_plain()).then_some(DraftLane(lane));
        step
    }

    /// The step starts from the slot's state or from zero and carries no drafts: it neither
    /// replays a lane's records nor records, so a scan that knows only the slot's state entry
    /// runs it.
    pub fn is_plain(self) -> bool {
        matches!(self.start, GdnStart::Slot | GdnStart::Fresh)
            && self.checkpoint_rows == CheckpointRows::NONE
    }

    /// The lane's record area once the step ran: a verify step records its rows to the other
    /// area, any other step leaves it.
    pub fn after(self) -> RecordArea {
        match self.checkpoint_rows {
            CheckpointRows::NONE => self.records,
            _ => self.records.other(),
        }
    }
}

/// Where one sequence's recurrent state comes from and goes in a step — the per-sequence element
/// of the `state_indices` buffer every GDN and Mamba-2 kernel reads (the shaders' `GdnSeqSlots`,
/// `shaders/gdn_state.h`, the same 24 bytes).
///
/// The prefix cache keeps snapshots of the state in slots past the live ones, in the same
/// buffers, so a kernel reaches them by index alone:
/// * a restore — a request resuming at a cached prefix — reads its start state from the
///   snapshot (`source`) and writes the step's state to its own `slot`, with no copy;
/// * a save stores the state after a given row into a snapshot slot as the scan passes that row,
///   inside the forward that runs it.
///
/// The record is built only through [`Self::plain`], [`Self::restoring`] and [`Self::saving`];
/// [`Self::words`] alone writes the kernels' encoding (`source` as slot + 1, a save of `0` rows as
/// none), so no caller writes a sentinel by hand. All zeros is a plain step on slot 0: a zeroed
/// buffer tail is a no-op (`runtime.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnSeqSlots {
    /// The sequence's own slot: where the step's state lands.
    slot: GdnSlot,
    /// The snapshot a [`GdnStart::Slot`] start reads instead of `slot` — a restore. Ignored by
    /// every other start.
    source: Option<GdnSlot>,
    /// Up to [`GDN_SAVES`] snapshot stores, in any order.
    saves: [Option<GdnSave>; GDN_SAVES],
}

/// Snapshot stores one sequence's step can carry: the kernels' `GDN_SAVES` (`gdn_state.h`).
pub const GDN_SAVES: usize = 2;

/// A slot of a GDN state pool — a live sequence's or, past them, a snapshot's — as the kernels
/// index it (an `i32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GdnSlot(u32);

impl GdnSlot {
    /// Slot `index`. Panics from `i32::MAX` on (a `source` word is the slot + 1), which no pool
    /// reaches: a slot is megabytes of state.
    pub fn new(index: usize) -> Self {
        let slot = u32::try_from(index)
            .ok()
            .filter(|&s| s < i32::MAX as u32)
            .expect("a GDN state slot index fits the kernels' i32");
        Self(slot)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// The slot as the kernels index it.
    pub fn word(self) -> i32 {
        self.0 as i32
    }
}

/// One snapshot store: once a step has run the sequence's first `rows_through` rows, the state
/// after them is ALSO stored into `snapshot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnSave {
    pub rows_through: std::num::NonZeroU32,
    pub snapshot: GdnSlot,
}

impl GdnSeqSlots {
    /// `i32`s per sequence in the flattened `state_indices` buffer.
    pub const WORDS: usize = 6;

    /// A step on `slot` with no snapshot work — what every sequence was before snapshots.
    pub fn plain(slot: GdnSlot) -> Self {
        Self {
            slot,
            source: None,
            saves: [None; GDN_SAVES],
        }
    }

    /// This step's slot start reads `source` — a snapshot it resumes from — not its own slot.
    pub fn restoring(self, source: GdnSlot) -> Self {
        Self {
            source: Some(source),
            ..self
        }
    }

    /// This step stores `saves` too.
    pub fn saving(self, saves: [Option<GdnSave>; GDN_SAVES]) -> Self {
        Self { saves, ..self }
    }

    pub fn slot(self) -> GdnSlot {
        self.slot
    }

    pub fn source(self) -> Option<GdnSlot> {
        self.source
    }

    pub fn saves(self) -> [Option<GdnSave>; GDN_SAVES] {
        self.saves
    }

    /// The record as the kernels read it: `[slot, source, rows_0, snapshot_0, rows_1,
    /// snapshot_1]`, `source` `0` for none and `s + 1` for slot `s`, a save of `0` rows none.
    pub fn words(self) -> [i32; Self::WORDS] {
        let save = |s: Option<GdnSave>| {
            s.map_or([0, 0], |s| {
                let rows = i32::try_from(s.rows_through.get())
                    .expect("a step's rows fit the kernels' i32");
                [rows, s.snapshot.word()]
            })
        };
        let [[r0, s0], [r1, s1]] = self.saves.map(save);
        let source = self.source.map_or(0, |s| s.word() + 1);
        [self.slot.word(), source, r0, s0, r1, s1]
    }
}

/// One GDN layer's state geometry: its causal conv's channels and width, and its delta-rule
/// heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnStateDims {
    pub conv_dim: usize,
    /// Causal conv kernel width (e.g. 4).
    pub conv_kernel: usize,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
}

impl GdnStateDims {
    /// One ssm state entry, in f32s: a slot's.
    pub fn ssm_entry_len(self) -> usize {
        self.num_v_heads * self.head_v_dim * self.head_k_dim
    }

    /// One recorded verify row, in f32s: its conv row, then its decay and beta (one each a value
    /// head).
    pub fn ssm_record_len(self) -> usize {
        self.conv_dim + 2 * self.num_v_heads
    }

    /// A lane's ssm memory, in f32s: two record areas of `rows.rows()` records each. The scan
    /// kernels lay a lane out the same way.
    pub fn ssm_lane_len(self, rows: CheckpointRows) -> usize {
        2 * rows.rows() * self.ssm_record_len()
    }

    /// One slot's conv and ssm bytes: the stride a slot index steps through each of a linear
    /// layer's two slot buffers.
    pub fn slot_bytes(self) -> (usize, usize) {
        let ([conv, ssm], _) = self.bytes(1, None);
        (conv, ssm)
    }

    /// Conv-state ring length: `conv_kernel - 1` past tokens retained per channel.
    fn conv_state_len(self) -> usize {
        self.conv_kernel - 1
    }

    /// One conv entry, in f32s: a slot's ring, or a lane's checkpoint after one verify row.
    pub fn conv_entry_len(self) -> usize {
        self.conv_dim * self.conv_state_len()
    }

    /// One layer's slot shapes: a conv entry, `[slots, conv_dim, conv_kernel - 1]`, and an ssm
    /// entry, `[slots, heads, head_v, head_k]`, a slot.
    fn slot_shapes(self, slots: usize) -> ([usize; 3], [usize; 4]) {
        (
            [slots, self.conv_dim, self.conv_state_len()],
            [slots, self.num_v_heads, self.head_v_dim, self.head_k_dim],
        )
    }

    /// One layer's lane shapes: a conv checkpoint a verify row, `[lanes · (k + 1), conv_dim,
    /// conv_kernel - 1]`, and a lane's records, `[lanes, ssm_lane_len]`.
    fn lane_shapes(self, lanes: DraftLanes) -> ([usize; 3], [usize; 2]) {
        let (count, rows) = (usize::from(lanes.count()), lanes.rows());
        (
            [count * rows.rows(), self.conv_dim, self.conv_state_len()],
            [count, self.ssm_lane_len(rows)],
        )
    }

    /// One layer's conv and ssm bytes: its slots', then its lanes' (zero without lanes).
    fn bytes(self, slots: usize, lanes: Option<DraftLanes>) -> ([usize; 2], [usize; 2]) {
        let sz = GDN_STATE_DTYPE.size_bytes();
        let elems = |dims: &[usize]| dims.iter().product::<usize>() * sz;
        let (conv, ssm) = self.slot_shapes(slots);
        let lane = lanes.map_or([0; 2], |lanes| {
            let (conv, ssm) = self.lane_shapes(lanes);
            [elems(&conv), elems(&ssm)]
        });
        ([elems(&conv), elems(&ssm)], lane)
    }
}

/// Recurrent-state pool for the GDN (linear-attention) layers of a hybrid model.
///
/// Allocates persistent f32 GPU memory at init time for every linear-attention
/// layer; full-attention layers hold `None`. Slot assignment/recycling is the
/// host allocator's job — this struct just owns the storage and hands out
/// lifetime-checked views.
pub struct GdnStatePool<M: PoolMemory> {
    /// Causal-conv1d ring per layer: `[num_slots, conv_dim, conv_kernel-1]`.
    /// `None` for non-linear (full-attention) layers.
    conv_states: Vec<Option<GpuTensor>>,
    /// Recurrent delta-rule state per layer: `[num_slots, num_v_heads, head_v_dim, head_k_dim]`.
    /// `None` for non-linear layers.
    ssm_states: Vec<Option<GpuTensor>>,
    /// RAII wrappers for the conv-state GPU allocations — auto-freed on drop.
    _conv_ptrs: Vec<Option<M>>,
    _ssm_ptrs: Vec<Option<M>>,
    /// The draft lanes' conv checkpoints and ssm records per layer (module docs): `None` for
    /// non-linear layers, and every layer of a pool without lanes.
    conv_lanes: Vec<Option<M>>,
    ssm_lanes: Vec<Option<M>>,
    pub num_layers: usize,
    pub num_slots: usize,
    /// The sequences that hold draft state at once (module docs); `None` when the model drafts
    /// nothing.
    pub lanes: Option<DraftLanes>,
    pub dims: GdnStateDims,
}

// Safety: GdnStatePool holds GPU device pointers (GpuTensor views + PoolMem
// allocations). These are allocated via the backend's device memory and are
// accessible from any host thread after backend setup. The pool is created once
// and moved to the worker thread; no concurrent mutation occurs.
unsafe impl<M: PoolMemory> Send for GdnStatePool<M> {}
unsafe impl<M: PoolMemory> Sync for GdnStatePool<M> {}

impl<M: PoolMemory> GdnStatePool<M> {
    /// Allocate the GDN state pool.
    ///
    /// `is_linear_layer[l]` selects which of the `num_layers` global layers are
    /// GDN (linear-attention) layers — only those get `conv_state`/`ssm_state`
    /// buffers; the rest store `None`. `num_slots = max_num_seqs`; `lanes` the draft state of the
    /// sequences that draft (module docs).
    ///
    /// # Safety
    /// Caller must ensure the backend context is current (the `alloc_buffer`
    /// closure performs GPU allocations).
    pub unsafe fn new(
        num_layers: usize,
        is_linear_layer: &[bool],
        num_slots: usize,
        lanes: Option<DraftLanes>,
        dims: GdnStateDims,
        mut alloc_buffer: impl FnMut(usize) -> Result<M>,
    ) -> Result<Self> {
        assert_eq!(
            is_linear_layer.len(),
            num_layers,
            "GdnStatePool: is_linear_layer mask length must equal num_layers"
        );
        assert!(
            dims.conv_kernel >= 1,
            "GdnStatePool: conv_kernel must be >= 1"
        );
        let dtype = GDN_STATE_DTYPE;
        let (conv_shape, ssm_shape) = dims.slot_shapes(num_slots);
        let ([conv_bytes, ssm_bytes], [conv_lane_bytes, ssm_lane_bytes]) =
            dims.bytes(num_slots, lanes);

        let mut conv_states = Vec::with_capacity(num_layers);
        let mut ssm_states = Vec::with_capacity(num_layers);
        let mut conv_ptrs = Vec::with_capacity(num_layers);
        let mut ssm_ptrs = Vec::with_capacity(num_layers);
        let mut conv_lanes = Vec::with_capacity(num_layers);
        let mut ssm_lanes = Vec::with_capacity(num_layers);

        let mut num_linear = 0usize;
        for &is_linear in is_linear_layer.iter() {
            if is_linear {
                num_linear += 1;
                let conv_mem = alloc_buffer(conv_bytes)?;
                let ssm_mem = alloc_buffer(ssm_bytes)?;
                // SAFETY: `conv_mem`/`ssm_mem` are freshly-allocated device
                // buffers of the matching byte size, owned by this pool.
                conv_states.push(Some(unsafe {
                    GpuTensor::new(conv_mem.ptr(), &conv_shape, dtype)
                }));
                ssm_states.push(Some(unsafe {
                    GpuTensor::new(ssm_mem.ptr(), &ssm_shape, dtype)
                }));
                conv_ptrs.push(Some(conv_mem));
                ssm_ptrs.push(Some(ssm_mem));
            } else {
                conv_states.push(None);
                ssm_states.push(None);
                conv_ptrs.push(None);
                ssm_ptrs.push(None);
            }
            let (conv_lane, ssm_lane) = match (is_linear, lanes) {
                (true, Some(_)) => (
                    Some(alloc_buffer(conv_lane_bytes)?),
                    Some(alloc_buffer(ssm_lane_bytes)?),
                ),
                _ => (None, None),
            };
            conv_lanes.push(conv_lane);
            ssm_lanes.push(ssm_lane);
        }

        let per_layer = conv_bytes + ssm_bytes + conv_lane_bytes + ssm_lane_bytes;
        let total_mb = (num_linear * per_layer) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "GdnStatePool: {num_linear}/{num_layers} linear layers × {num_slots} slots, {} draft \
             lanes of {} draft rows ({dims:?}) = {total_mb:.0} MB f32",
            lanes.map_or(0, DraftLanes::count),
            lanes.map_or(0, |l| l.rows().0),
        );

        Ok(Self {
            conv_states,
            ssm_states,
            _conv_ptrs: conv_ptrs,
            _ssm_ptrs: ssm_ptrs,
            conv_lanes,
            ssm_lanes,
            num_layers,
            num_slots,
            lanes,
            dims,
        })
    }

    /// Placeholder pool with zero layers and no GPU allocations. Used to
    /// satisfy an `Option<&GdnStatePool>`-free placeholder need or as a borrow
    /// source for arches with no GDN layers. Reading any layer index would
    /// panic — by contract, non-GDN codegen never emits a `GatedDeltaNet` read.
    pub fn empty() -> Self {
        Self {
            conv_states: Vec::new(),
            ssm_states: Vec::new(),
            _conv_ptrs: Vec::new(),
            _ssm_ptrs: Vec::new(),
            conv_lanes: Vec::new(),
            ssm_lanes: Vec::new(),
            num_layers: 0,
            num_slots: 0,
            lanes: None,
            dims: GdnStateDims {
                conv_dim: 0,
                conv_kernel: 0,
                num_v_heads: 0,
                head_v_dim: 0,
                head_k_dim: 0,
            },
        }
    }

    /// Number of bytes the pool would consume for `num_linear` linear layers
    /// at the given dims — used by the worker to reserve budget *before*
    /// allocation (mirrors how the KV byte budget is computed up front).
    pub fn reserve_bytes(
        num_linear: usize,
        num_slots: usize,
        lanes: Option<DraftLanes>,
        dims: GdnStateDims,
    ) -> usize {
        let (slot, lane) = dims.bytes(num_slots, lanes);
        num_linear * (slot.iter().sum::<usize>() + lane.iter().sum::<usize>())
    }

    /// Whether global layer `layer` is a GDN (linear-attention) layer.
    pub fn is_linear(&self, layer: usize) -> bool {
        self.conv_states
            .get(layer)
            .map(|s| s.is_some())
            .unwrap_or(false)
    }

    /// Conv-state buffer for a GDN layer, as a lifetime-checked view.
    /// Panics if `layer` is not a linear-attention layer.
    pub fn conv_state(&self, layer: usize) -> TensorView<'_> {
        // Safety: GdnStatePool owns the memory via _conv_ptrs; view borrows &self.
        let t = self.conv_states[layer]
            .expect("GdnStatePool::conv_state: layer is not a GDN (linear-attention) layer");
        unsafe { TensorView::from_raw(t) }
    }

    /// Recurrent (ssm) state buffer for a GDN layer, as a lifetime-checked view.
    /// Panics if `layer` is not a linear-attention layer.
    pub fn ssm_state(&self, layer: usize) -> TensorView<'_> {
        // Safety: GdnStatePool owns the memory via _ssm_ptrs; view borrows &self.
        let t = self.ssm_states[layer]
            .expect("GdnStatePool::ssm_state: layer is not a GDN (linear-attention) layer");
        unsafe { TensorView::from_raw(t) }
    }

    /// Per-layer conv-state backing memory (metal ICB binding). Panics if the
    /// layer is not a GDN layer.
    pub fn conv_layer_mem(&self, layer: usize) -> &M {
        self._conv_ptrs[layer]
            .as_ref()
            .expect("GdnStatePool::conv_layer_mem: layer is not a GDN layer")
    }

    /// Per-layer ssm-state backing memory (metal ICB binding). See
    /// [`Self::conv_layer_mem`].
    pub fn ssm_layer_mem(&self, layer: usize) -> &M {
        self._ssm_ptrs[layer]
            .as_ref()
            .expect("GdnStatePool::ssm_layer_mem: layer is not a GDN layer")
    }

    /// Per-layer backing memory of the draft lanes' conv checkpoints (metal ICB binding); `None`
    /// on a non-GDN layer or a pool without lanes.
    pub fn conv_lanes_mem(&self, layer: usize) -> Option<&M> {
        self.conv_lanes.get(layer)?.as_ref()
    }

    /// Per-layer backing memory of the draft lanes' ssm records. See [`Self::conv_lanes_mem`].
    pub fn ssm_lanes_mem(&self, layer: usize) -> Option<&M> {
        self.ssm_lanes.get(layer)?.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The record's words are the kernels' encoding (`gdn_state.h`): a plain step is its slot and
    /// zeros, a source its slot + 1, a save `[rows, snapshot]` in the place it was given.
    #[test]
    fn a_seq_record_encodes_as_the_kernels_read_it() {
        let rows = |n| std::num::NonZeroU32::new(n).unwrap();
        assert_eq!(GdnSeqSlots::plain(GdnSlot::new(0)).words(), [0; 6]);
        assert_eq!(
            GdnSeqSlots::plain(GdnSlot::new(4)).words(),
            [4, 0, 0, 0, 0, 0]
        );
        let save = |n, s| {
            Some(GdnSave {
                rows_through: rows(n),
                snapshot: GdnSlot::new(s),
            })
        };
        let seq = GdnSeqSlots::plain(GdnSlot::new(3))
            .restoring(GdnSlot::new(0))
            .saving([None, save(96, 9)]);
        assert_eq!(seq.words(), [3, 1, 0, 0, 96, 9]);
        assert_eq!(seq.source(), Some(GdnSlot::new(0)));
    }

    /// Every step the pool can describe decodes back from its code, and only a slot or fresh
    /// start without drafts is plain.
    #[test]
    fn a_step_code_decodes_to_its_step() {
        let starts = [
            GdnStart::Slot,
            GdnStart::Fresh,
            GdnStart::Checkpoint(0),
            GdnStart::Checkpoint(2),
        ];
        for start in starts {
            for rows in [CheckpointRows::NONE, CheckpointRows(2)] {
                for records in [RecordArea::First, RecordArea::Second] {
                    let plain = matches!(start, GdnStart::Slot | GdnStart::Fresh)
                        && rows == CheckpointRows::NONE;
                    let lanes = match plain {
                        true => vec![None],
                        false => vec![Some(DraftLane(0)), Some(DraftLane(DraftLanes::MAX - 1))],
                    };
                    for lane in lanes {
                        let step = GdnStep {
                            start,
                            checkpoint_rows: rows,
                            records,
                            lane,
                        };
                        assert_eq!(GdnStep::decode(step.encode()), step);
                        assert_eq!(step.is_plain(), plain, "{step:?}");
                    }
                }
            }
        }
    }

    /// A slot costs what it costs without drafts; the drafts' checkpoints and records cost their
    /// lanes alone. Qwen3.6-35B-A3B's geometry: 128 slots and 4 lanes of 2 drafts keep the
    /// draft state of 4 sequences, where a draft area in every slot kept it for 128 (14.1 MiB a
    /// slot over its 30 GDN layers).
    #[test]
    fn draft_state_costs_its_lanes_not_its_slots() {
        let dims = GdnStateDims {
            conv_dim: 8192,
            conv_kernel: 4,
            num_v_heads: 32,
            head_v_dim: 128,
            head_k_dim: 128,
        };
        let rows = CheckpointRows(2);
        let pool = |slots, lanes| GdnStatePool::<NoMem>::reserve_bytes(30, slots, lanes, dims);
        let plain = pool(128, None);
        let lanes = DraftLanes::new(rows, 4);
        let entry = 4 * (dims.conv_entry_len() + dims.ssm_entry_len());
        assert_eq!(plain, 30 * 128 * entry);
        // A lane is the draft area every slot used to carry: (k + 1) conv checkpoints and two
        // record areas, 14.1 MiB over the 30 layers.
        let lane = 4 * (rows.rows() * dims.conv_entry_len() + dims.ssm_lane_len(rows));
        assert_eq!(30 * lane, 14_791_680);
        assert_eq!(pool(128, lanes), plain + 30 * 4 * lane);
        assert_eq!(DraftLanes::new(CheckpointRows::NONE, 4), None);
        assert_eq!(DraftLanes::new(rows, 0), None);
    }

    /// Memory that is never allocated: [`GdnStatePool::reserve_bytes`] only sizes.
    struct NoMem;

    impl PoolMemory for NoMem {
        fn ptr(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
    }
}
