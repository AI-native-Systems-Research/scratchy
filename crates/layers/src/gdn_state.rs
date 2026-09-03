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

use anyhow::Result;
use scratchy_tensors::{DType, GpuTensor, PoolMemory, TensorView};

/// GDN recurrent-state buffers are always f32 (kernels + `mamba_ssm_dtype`).
pub const GDN_STATE_DTYPE: DType = DType::F32;

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
    /// Recurrent delta-rule state per layer:
    /// `[num_slots, num_v_heads, head_v_dim, head_k_dim]`. `None` for
    /// non-linear layers.
    ssm_states: Vec<Option<GpuTensor>>,
    /// RAII wrappers for the conv-state GPU allocations — auto-freed on drop.
    _conv_ptrs: Vec<Option<M>>,
    _ssm_ptrs: Vec<Option<M>>,
    pub num_layers: usize,
    pub num_slots: usize,
    pub conv_dim: usize,
    /// Causal conv kernel width (e.g. 4).
    pub conv_kernel: usize,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
}

// Safety: GdnStatePool holds GPU device pointers (GpuTensor views + PoolMem
// allocations). These are allocated via the backend's device memory and are
// accessible from any host thread after backend setup. The pool is created once
// and moved to the worker thread; no concurrent mutation occurs.
unsafe impl<M: PoolMemory> Send for GdnStatePool<M> {}
unsafe impl<M: PoolMemory> Sync for GdnStatePool<M> {}

impl<M: PoolMemory> GdnStatePool<M> {
    /// Conv-state ring length: `conv_kernel - 1` past tokens retained per channel.
    #[inline]
    fn conv_state_len(conv_kernel: usize) -> usize {
        conv_kernel - 1
    }

    /// Allocate the GDN state pool.
    ///
    /// `is_linear_layer[l]` selects which of the `num_layers` global layers are
    /// GDN (linear-attention) layers — only those get `conv_state`/`ssm_state`
    /// buffers; the rest store `None`. `num_slots = max_num_seqs`.
    ///
    /// # Safety
    /// Caller must ensure the backend context is current (the `alloc_buffer`
    /// closure performs GPU allocations).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        num_layers: usize,
        is_linear_layer: &[bool],
        num_slots: usize,
        conv_dim: usize,
        conv_kernel: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
        mut alloc_buffer: impl FnMut(usize) -> Result<M>,
    ) -> Result<Self> {
        assert_eq!(
            is_linear_layer.len(),
            num_layers,
            "GdnStatePool: is_linear_layer mask length must equal num_layers"
        );
        assert!(conv_kernel >= 1, "GdnStatePool: conv_kernel must be >= 1");

        let dtype = GDN_STATE_DTYPE;
        let conv_state_len = Self::conv_state_len(conv_kernel);
        let conv_shape = [num_slots, conv_dim, conv_state_len];
        let ssm_shape = [num_slots, num_v_heads, head_v_dim, head_k_dim];
        let conv_bytes = num_slots * conv_dim * conv_state_len * dtype.size_bytes();
        let ssm_bytes = num_slots * num_v_heads * head_v_dim * head_k_dim * dtype.size_bytes();

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
            "GdnStatePool: {num_linear}/{num_layers} linear layers × {num_slots} slots \
             (conv_dim {conv_dim}, ssm {num_v_heads}×{head_v_dim}×{head_k_dim}) = {total_mb:.0} MB f32"
        );

        Ok(Self {
            conv_states,
            ssm_states,
            _conv_ptrs: conv_ptrs,
            _ssm_ptrs: ssm_ptrs,
            num_layers,
            num_slots,
            conv_dim,
            conv_kernel,
            num_v_heads,
            head_v_dim,
            head_k_dim,
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
            conv_dim: 0,
            conv_kernel: 0,
            num_v_heads: 0,
            head_v_dim: 0,
            head_k_dim: 0,
        }
    }

    /// Number of bytes the pool would consume for `num_linear` linear layers
    /// at the given dims — used by the worker to reserve budget *before*
    /// allocation (mirrors how the KV byte budget is computed up front).
    pub fn reserve_bytes(
        num_linear: usize,
        num_slots: usize,
        conv_dim: usize,
        conv_kernel: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
    ) -> usize {
        let sz = GDN_STATE_DTYPE.size_bytes();
        let conv_bytes = num_slots * conv_dim * Self::conv_state_len(conv_kernel) * sz;
        let ssm_bytes = num_slots * num_v_heads * head_v_dim * head_k_dim * sz;
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
