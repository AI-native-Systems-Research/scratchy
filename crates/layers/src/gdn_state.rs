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
//! Both buffers are **f32** (the surviving `gdn_*` CUDA kernels operate in f32,
//! and the model's `mamba_ssm_dtype` is `float32`). The slot a sequence owns is
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
//! Like `KvCachePool` the buffer-allocation is a caller-supplied closure (cuda
//! wraps `driver::mem_alloc`, metal wraps `device.new_buffer`); the layout /
//! sizing / accessor logic is backend-neutral.
//!
//! **Checkpoints (speculative decoding).** A verify step runs a sequence's last token and its
//! drafts as rows of one step, and a rejected draft must not stay in the recurrent state. With
//! `checkpoint_rows = k`, each slot owns `1 + k` state entries: the state itself, then the
//! state after each of the step's first `k` rows ([`GdnStep::checkpoint_rows`]). The next step
//! starts from the checkpoint at the accepted row ([`GdnStart::Checkpoint`]) — the accepted
//! count is data the step reads, so nothing is copied or replayed.

use anyhow::Result;
use scratchy_tensors::{DType, GpuTensor, PoolMemory, TensorView};

/// GDN recurrent-state buffers are always f32 (kernels + `mamba_ssm_dtype`).
pub const GDN_STATE_DTYPE: DType = DType::F32;

/// State entries a GDN slot keeps after its own: the state after each of a verify step's first
/// rows (module docs) — the most drafts one verify step may carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointRows(pub u8);

impl CheckpointRows {
    /// No checkpoints: a deployment without speculative decoding.
    pub const NONE: Self = Self(0);

    /// State entries per slot: its own and its checkpoints.
    pub fn entries_per_slot(self) -> usize {
        1 + usize::from(self.0)
    }
}

/// The state a sequence's GDN layers start a step from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnStart {
    /// The state its slot holds — the previous step's last row.
    Slot,
    /// Zero: the sequence's first step.
    Fresh,
    /// The state after row `r` of the previous step: a verify step accepted the drafts up to it.
    Checkpoint(u8),
}

/// What one step does with one sequence's GDN state, as the per-sequence `u32` the GDN kernels
/// read (the buffer the forward calls `is_fresh`: `0` and `1` keep their meaning).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnStep {
    pub start: GdnStart,
    /// Rows whose resulting state the step also writes to the slot's checkpoints, `0..rows`:
    /// a verify step's drafts. [`CheckpointRows::NONE`] outside speculative decoding.
    pub checkpoint_rows: CheckpointRows,
}

impl GdnStep {
    /// The kernels' encoding: bits 0–7 the start (`0` slot, `1` fresh, `2 + r` checkpoint `r`),
    /// bits 8–15 the checkpoint rows.
    pub fn encode(self) -> u32 {
        let start = match self.start {
            GdnStart::Slot => 0,
            GdnStart::Fresh => 1,
            GdnStart::Checkpoint(r) => 2 + u32::from(r),
        };
        start | (u32::from(self.checkpoint_rows.0) << 8)
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
    /// Conv-state ring length: `conv_kernel - 1` past tokens retained per channel.
    fn conv_state_len(self) -> usize {
        self.conv_kernel - 1
    }

    /// One state entry's shapes and bytes, `(conv, ssm)`, `entries` of them.
    fn shapes(self, entries: usize) -> ([usize; 3], [usize; 4]) {
        let conv = [entries, self.conv_dim, self.conv_state_len()];
        let ssm = [entries, self.num_v_heads, self.head_v_dim, self.head_k_dim];
        (conv, ssm)
    }

    /// Bytes of `entries` conv and ssm state entries.
    fn bytes(self, entries: usize) -> (usize, usize) {
        let (conv, ssm) = self.shapes(entries);
        let sz = GDN_STATE_DTYPE.size_bytes();
        (
            conv.iter().product::<usize>() * sz,
            ssm.iter().product::<usize>() * sz,
        )
    }
}

/// Recurrent-state pool for the GDN (linear-attention) layers of a hybrid model.
///
/// Allocates persistent f32 GPU memory at init time for every linear-attention
/// layer; full-attention layers hold `None`. Slot assignment/recycling is the
/// host allocator's job — this struct just owns the storage and hands out
/// lifetime-checked views.
pub struct GdnStatePool<M: PoolMemory> {
    /// Causal-conv1d ring per layer: `[num_slots · entries_per_slot, conv_dim, conv_kernel-1]`.
    /// `None` for non-linear (full-attention) layers.
    conv_states: Vec<Option<GpuTensor>>,
    /// Recurrent delta-rule state per layer:
    /// `[num_slots · entries_per_slot, num_v_heads, head_v_dim, head_k_dim]`. `None` for
    /// non-linear layers.
    ssm_states: Vec<Option<GpuTensor>>,
    /// RAII wrappers for the conv-state GPU allocations — auto-freed on drop.
    _conv_ptrs: Vec<Option<M>>,
    _ssm_ptrs: Vec<Option<M>>,
    pub num_layers: usize,
    pub num_slots: usize,
    /// Checkpoint entries each slot owns after its state (module docs).
    pub checkpoint_rows: CheckpointRows,
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
    /// buffers; the rest store `None`. `num_slots = max_num_seqs`; `checkpoint_rows` the most
    /// drafts a verify step carries.
    ///
    /// # Safety
    /// Caller must ensure the backend context is current (the `alloc_buffer`
    /// closure performs GPU allocations).
    pub unsafe fn new(
        num_layers: usize,
        is_linear_layer: &[bool],
        num_slots: usize,
        checkpoint_rows: CheckpointRows,
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
        let entries = num_slots * checkpoint_rows.entries_per_slot();
        let (conv_shape, ssm_shape) = dims.shapes(entries);
        let (conv_bytes, ssm_bytes) = dims.bytes(entries);

        let mut conv_states = Vec::with_capacity(num_layers);
        let mut ssm_states = Vec::with_capacity(num_layers);
        let mut conv_ptrs = Vec::with_capacity(num_layers);
        let mut ssm_ptrs = Vec::with_capacity(num_layers);

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
        }

        let total_mb = (num_linear * (conv_bytes + ssm_bytes)) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "GdnStatePool: {num_linear}/{num_layers} linear layers × {num_slots} slots × \
             {} entries ({dims:?}) = {total_mb:.0} MB f32",
            checkpoint_rows.entries_per_slot()
        );

        Ok(Self {
            conv_states,
            ssm_states,
            _conv_ptrs: conv_ptrs,
            _ssm_ptrs: ssm_ptrs,
            num_layers,
            num_slots,
            checkpoint_rows,
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
            num_layers: 0,
            num_slots: 0,
            checkpoint_rows: CheckpointRows::NONE,
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
        checkpoint_rows: CheckpointRows,
        dims: GdnStateDims,
    ) -> usize {
        let (conv_bytes, ssm_bytes) = dims.bytes(num_slots * checkpoint_rows.entries_per_slot());
        num_linear * (conv_bytes + ssm_bytes)
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
}
