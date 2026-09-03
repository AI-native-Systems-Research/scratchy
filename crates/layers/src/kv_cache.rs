// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache pool using `GpuTensor` — persistent GPU memory.
//!
//! Layout per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
//! Backend-neutral: storage, sizing, slot decomposition, and span
//! bookkeeping live here for both cuda and metal. The actual buffer
//! allocation is a caller-supplied `alloc_buffer` closure — cuda
//! passes one that wraps `driver::mem_alloc`, metal passes one that
//! wraps `device.new_buffer`. The cuda-only surface (FP8 scale H2D,
//! the `gather_kv_contiguous` D2D copy path, and the GPU mirrors of
//! the span flags) lives in an extension trait in `targets/cuda`,
//! reaching the storage through the public accessors below; this type
//! itself names no backend.

use anyhow::Result;
use scratchy_tensors::{DType, GpuTensor, PoolMemory, TensorView};

/// Paged KV cache pool for all transformer layers.
///
/// Allocates persistent GPU memory for K and V caches at init time.
/// Block allocation/freeing is managed by the scheduler — this struct
/// just provides the raw tensor views.
///
/// When `cache_dtype` is `Fp8E4m3`, the cache stores 1 byte/element and
/// per-layer scale factors are maintained for quantization/dequantization.
/// FP8 is cuda-only: the scale machinery uses cudarc primitives.
pub struct KvCachePool<M: PoolMemory> {
    /// K cache per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`
    k_caches: Vec<GpuTensor>,
    /// V cache per layer: same layout
    v_caches: Vec<GpuTensor>,
    /// RAII wrappers for KV cache GPU allocations — auto-freed on drop.
    _k_ptrs: Vec<M>,
    _v_ptrs: Vec<M>,
    pub num_blocks: usize,
    pub block_size: usize,
    /// Per-sequence block-table row capacity: the number of paged blocks one
    /// sequence can span. Runtime-derived by the worker as
    /// `min(ceil(max_model_len / block_size), num_blocks).max(1)` — this
    /// replaces the compile-time `MAX_BLOCKS_PER_SEQ` (default 128 ≈ 2k tokens)
    /// so long contexts are not silently truncated. The same value drives the
    /// kernel's `MaxBlocksPerSeq` function constant (block-table row stride +
    /// scratch bound), the host block-table row stride, and the rope-once
    /// `roped_k_scratch` size — all three MUST read this so they agree.
    pub max_blocks_per_seq: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    /// Number of distinct PHYSICAL chunk tensors. Equals `num_layers` for every
    /// uniform model (one tensor per layer). For vLLM's group-shared hybrid
    /// layout (gemma4) it is the `group_size` (= layers-per-group, e.g. 5):
    /// layers at the same POSITION across groups share one physical tensor, so
    /// the chunk vecs (`k_chunks`/`k_chunk_tables`/…) are indexed by tensor, not
    /// layer. This is the ~6x VA cut that lets 32k fit the working set.
    pub num_tensors: usize,
    /// Maps a logical layer → its physical tensor index (`< num_tensors`).
    /// Identity (`layer == tensor`) on the uniform path. Block IDs are disjoint
    /// across the groups that share a tensor (one shared free pool hands each ID
    /// to one group), so co-resident layers never write the same physical block.
    layer_to_tensor: Vec<usize>,
    /// vLLM KV-cache-GROUP layout (distinct from `layer_to_tensor`): number of
    /// per-group block tables and each layer's group index. The metal worker
    /// binds `block_tables[layer_to_group[layer]]` / `slot_mappings[...]`.
    /// `None` = uniform single group (every layer → group 0); `Some((n, map))`
    /// for gemma4 SWA (full + N sliding groups). Set from `HybridKvLayout`.
    kv_group_layout: Option<(usize, Vec<u32>)>,
    /// Per-layer `num_kv_heads * block_size * head_dim` overriding the
    /// uniform `num_kv_heads * block_size * head_dim` when the model's
    /// attention geometry differs per layer class (Gemma4: sliding
    /// layers 8×16×256 = 32768 elems/block, global layers 1×16×512 =
    /// 8192). `None` = uniform (every existing arch).
    // Read only by the metal `grow_to_cover` path; populated under cuda
    // (uniform geometry → `None`) but unused there.
    per_layer_block_elems: Option<Vec<usize>>,
    /// The dtype stored in cache (may differ from model dtype when FP8).
    cache_dtype: DType,
    /// Per-layer K scale: GPU f32 scalar RAII wrappers. Only used when FP8
    /// (cuda); empty on every other path. The FP8 H2D init lives in the
    /// cuda extension trait.
    k_scale_ptrs: Vec<M>,
    /// Per-layer V scale: GPU f32 scalar RAII wrappers. See `k_scale_ptrs`.
    v_scale_ptrs: Vec<M>,

    /// Per-physical-block flag: `true` = K is **currently** stored unrotated.
    /// Used by the pre-attention rotation kernel (rotate these blocks).
    /// Indexed by physical block ID.
    pub block_is_unrotated: Vec<bool>,

    /// Per-physical-block flag: `true` = this is a span block (should be
    /// stored unrotated in the resting state between steps).
    /// Used by the post-attention un-rotation kernel (un-rotate these blocks).
    pub block_is_span: Vec<bool>,

    /// GPU mirror of `block_is_unrotated` — for pre-attention forward rotation.
    /// Cuda-only span machinery (lazily allocated via the cuda extension);
    /// `None` on every other path.
    block_unrotated_gpu_ptr: Option<M>,
    /// GPU mirror of `block_is_span` — for post-attention inverse rotation.
    block_span_gpu_ptr: Option<M>,

    // ── Reactive (chunked) KV storage — metal only ──────────────────
    //
    // On metal the per-layer KV cache is NOT one contiguous buffer; it
    // is a series of fixed-size *chunk* buffers (`blocks_per_chunk`
    // blocks each) so the `MTLResidencySet` can hold only the chunks
    // that back live KV. The kernels bind a per-layer chunk-address
    // *table* (device uint64 gpuAddresses) at the `KvCacheK/V` slot and
    // deref `table[block_id / blocks_per_chunk]` then address
    // `block_id % blocks_per_chunk` within that chunk. CUDA keeps the
    // single-buffer `k_caches` / `_k_ptrs` layout above untouched.
    //
    // `k_chunks[layer][chunk]` / `v_chunks[layer][chunk]` own the chunk
    // data buffers (StorageModePrivate); `k_chunk_tables[layer]` /
    // `v_chunk_tables[layer]` own the per-layer address tables
    // (StorageModeShared, host-written via `fill_chunk_tables`).
    // (metal-only in practice; empty on the cuda single-buffer path.)
    k_chunks: Vec<Vec<M>>,
    v_chunks: Vec<Vec<M>>,
    k_chunk_tables: Vec<M>,
    v_chunk_tables: Vec<M>,
    blocks_per_chunk: usize,
}

// Safety: KvCachePool holds GPU device pointers (GpuTensor arrays and raw
// *mut f32 scale pointers). These are allocated via the backend's device
// memory and are accessible from any host thread after backend setup. The
// pool is created once and moved to the worker thread; no concurrent
// mutation occurs.
unsafe impl<M: PoolMemory> Send for KvCachePool<M> {}
unsafe impl<M: PoolMemory> Sync for KvCachePool<M> {}

impl<M: PoolMemory> KvCachePool<M> {
    /// Allocate KV cache for all layers.
    ///
    /// `alloc_buffer(bytes)` is the backend-neutral way to grab a
    /// raw `bytes`-sized GPU allocation wrapped in a [`PoolMem`].
    /// Cuda passes a closure that calls `driver::mem_alloc` and
    /// wraps the result; metal passes one that calls
    /// `device.new_buffer(StorageModeShared)`. Splitting this out
    /// keeps every backend-specific call out of the constructor body
    /// and lets the storage layout / sizing / span bookkeeping live
    /// in one place.
    ///
    /// FP8 scale buffers are also allocated via the same closure, but
    /// the scale-init H2D copy is cuda-only (FP8 cache is not
    /// supported under metal); under metal an FP8 dtype here returns
    /// an error.
    ///
    /// # Safety
    /// Caller must ensure the backend context is current on the
    /// invoking thread (cuda: `ctx_set_current`; metal: any thread
    /// after device init).
    ///
    /// FP8 scale buffers (cuda only) are allocated via `alloc_buffer` and
    /// initialized to `1.0` via the caller-supplied `init_fp8_scale` closure
    /// (cuda does the H2D copy; the closure is invoked only when `dtype` is
    /// FP8, so non-FP8 / non-cuda callers may pass a no-op). This keeps the
    /// driver H2D out of this backend-neutral type.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        // Per-sequence block-table row capacity (see the struct field). Runtime-
        // derived `min(ceil(max_model_len/block_size), num_blocks).max(1)`.
        max_blocks_per_seq: usize,
        // Per-layer per-block elem override for hybrid-attention-geometry
        // arches (Gemma-4: sliding 8×256 vs global 1×512/2×512). `None` =
        // uniform pool (the common case). When `Some`, len == num_layers and
        // each entry is that layer's `block_size * kv_heads * head_dim`.
        per_layer_block_elems: Option<Vec<usize>>,
        dtype: DType,
        mut alloc_buffer: impl FnMut(usize) -> Result<M>,
        mut init_fp8_scale: impl FnMut(&M) -> Result<()>,
    ) -> Result<Self> {
        if let Some(v) = per_layer_block_elems.as_ref() {
            assert_eq!(
                v.len(),
                num_layers,
                "per_layer_block_elems len must equal num_layers"
            );
        }
        let uniform_block_elems = block_size * num_kv_heads * head_dim;
        let layer_block_elems = |l: usize| -> usize {
            per_layer_block_elems
                .as_ref()
                .map_or(uniform_block_elems, |v| v[l])
        };

        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        let mut k_ptrs = Vec::with_capacity(num_layers);
        let mut v_ptrs = Vec::with_capacity(num_layers);

        let mut total_bytes = 0usize;
        for l in 0..num_layers {
            let pbe = layer_block_elems(l);
            let bytes_per_layer = num_blocks * pbe * dtype.size_bytes();
            total_bytes += 2 * bytes_per_layer;
            // Keep the true (kv_heads, head_dim) split for the uniform/sliding
            // class (flash paged decode reads dim(2)=kv_heads); collapse to
            // [.,.,1,token_elems] for a hybrid-geometry layer (global class is
            // gathered with explicit dims, so its shape split is unused).
            let tok = pbe / block_size;
            let (kvh, hd) = if tok == num_kv_heads * head_dim {
                (num_kv_heads, head_dim)
            } else {
                (1, tok)
            };
            let shape = [num_blocks, block_size, kvh, hd];
            let k_mem = alloc_buffer(bytes_per_layer)?;
            let v_mem = alloc_buffer(bytes_per_layer)?;

            // SAFETY: freshly-allocated device buffers, owned by this pool
            // via `_k_ptrs` / `_v_ptrs`.
            k_caches.push(unsafe { GpuTensor::new(k_mem.ptr(), &shape, dtype) });
            v_caches.push(unsafe { GpuTensor::new(v_mem.ptr(), &shape, dtype) });
            k_ptrs.push(k_mem);
            v_ptrs.push(v_mem);
        }

        // FP8 scales (cuda only): allocate per-layer scalar buffers and let the
        // caller H2D-init them to 1.0. Metal/non-cuda never builds an FP8 pool
        // through `new` (it uses `new_metal_chunked`, which rejects FP8).
        let mut k_scale_ptrs = Vec::new();
        let mut v_scale_ptrs = Vec::new();
        if dtype.is_fp8() {
            for _ in 0..num_layers {
                let k_scale = alloc_buffer(4)?;
                let v_scale = alloc_buffer(4)?;
                init_fp8_scale(&k_scale)?;
                init_fp8_scale(&v_scale)?;
                k_scale_ptrs.push(k_scale);
                v_scale_ptrs.push(v_scale);
            }
        }

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        let dtype_label = if dtype.is_fp8() {
            "FP8 E4M3"
        } else {
            &format!("{}", dtype)
        };
        tracing::info!(
            "KvCachePool: {num_layers} layers × {num_blocks} blocks × {block_size} slots = {total_mb:.0} MB ({dtype_label})"
        );

        Ok(Self {
            k_caches,
            v_caches,
            _k_ptrs: k_ptrs,
            _v_ptrs: v_ptrs,
            num_blocks,
            block_size,
            max_blocks_per_seq,
            num_kv_heads,
            head_dim,
            num_layers,
            num_tensors: num_layers,
            layer_to_tensor: (0..num_layers).collect(),
            kv_group_layout: None,
            per_layer_block_elems,
            cache_dtype: dtype,
            k_scale_ptrs,
            v_scale_ptrs,
            block_is_unrotated: vec![false; num_blocks],
            block_is_span: vec![false; num_blocks],
            block_unrotated_gpu_ptr: None,
            block_span_gpu_ptr: None,
            // CUDA / single-buffer path: no chunked storage.
            k_chunks: Vec::new(),
            v_chunks: Vec::new(),
            k_chunk_tables: Vec::new(),
            v_chunk_tables: Vec::new(),
            blocks_per_chunk: 0,
        })
    }

    /// Allocate a **group-shared single-buffer** KV pool (cuda hybrid SWA,
    /// gemma4): `num_tensors = max(layer_to_tensor)+1` contiguous physical
    /// buffers instead of one-per-layer, so the ~6× VA cut that lets 32k fit
    /// applies on cuda too. Same eager single-buffer storage as [`Self::new`]
    /// (the cuda kernels read the contiguous `k_cache(layer)` view) — this is
    /// the single-buffer sibling of [`Self::new_metal_chunked`]'s
    /// `layer_to_tensor` path.
    ///
    /// Layers that map to the SAME physical tensor (same position across
    /// vLLM groups) are handed DISJOINT global block ids by the scheduler's
    /// hybrid allocator, so they never collide in the shared buffer. Each
    /// layer's `k_cache(layer)` view carries that layer's own page-unified
    /// shape (`layer_block_size[l]`, kv geometry from `layer_kv_token_elems`)
    /// over the shared bytes; the page (`block_size × kv_elems`) is uniform
    /// across the layers sharing a tensor by construction (asserted).
    ///
    /// `layer_kv_token_elems[l]` = layer l's `num_kv_heads*head_dim`.
    /// `layer_block_size[l]` = layer l's page-unified block size (gemma4:
    /// full=64, sliding=16). `layer_to_tensor[l] < num_tensors`. All three
    /// have len `num_layers`. FP8 is rejected (gemma4 hybrid is bf16; fp8 KV
    /// keeps the uniform pool).
    ///
    /// # Safety
    /// Same contract as [`Self::new`] — backend context current on this thread.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_grouped(
        num_layers: usize,
        num_blocks: usize,
        base_block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_blocks_per_seq: usize,
        layer_kv_token_elems: Vec<usize>,
        layer_block_size: Vec<usize>,
        layer_to_tensor: Vec<usize>,
        dtype: DType,
        mut alloc_buffer: impl FnMut(usize) -> Result<M>,
    ) -> Result<Self> {
        if dtype.is_fp8() {
            anyhow::bail!("new_grouped: FP8 cache dtype keeps the uniform pool (cuda)");
        }
        assert_eq!(
            layer_kv_token_elems.len(),
            num_layers,
            "layer_kv_token_elems len must equal num_layers"
        );
        assert_eq!(
            layer_block_size.len(),
            num_layers,
            "layer_block_size len must equal num_layers"
        );
        assert_eq!(
            layer_to_tensor.len(),
            num_layers,
            "layer_to_tensor len must equal num_layers"
        );
        let num_tensors = layer_to_tensor.iter().copied().max().map_or(0, |m| m + 1);

        // Per-tensor page (elements per block) — uniform across the layers that
        // share a tensor after page-unification. Derive from each layer and
        // assert agreement so a mis-computed layout is a build/init error, not
        // silent cross-group corruption.
        let mut tensor_page: Vec<Option<usize>> = vec![None; num_tensors];
        for l in 0..num_layers {
            let page = layer_block_size[l] * layer_kv_token_elems[l];
            let t = layer_to_tensor[l];
            match tensor_page[t] {
                None => tensor_page[t] = Some(page),
                Some(p) => assert_eq!(
                    p, page,
                    "layers sharing physical tensor {t} have unequal page \
                     ({p} vs {page}) — page-unification broken"
                ),
            }
        }
        let tensor_page: Vec<usize> = tensor_page
            .into_iter()
            .map(|p| p.expect("every physical tensor must back at least one layer"))
            .collect();

        // Allocate `num_tensors` contiguous K and V buffers.
        let mut k_ptrs = Vec::with_capacity(num_tensors);
        let mut v_ptrs = Vec::with_capacity(num_tensors);
        let mut total_bytes = 0usize;
        for &page in &tensor_page {
            let bytes = num_blocks * page * dtype.size_bytes();
            total_bytes += 2 * bytes;
            k_ptrs.push(alloc_buffer(bytes)?);
            v_ptrs.push(alloc_buffer(bytes)?);
        }

        // Build per-layer views over the shared buffers. Same shape-collapse as
        // `new`: keep (kv_heads, head_dim) for the class whose token stride
        // matches (sliding), collapse to [.,.,1,tok] for the global class
        // (gathered with explicit dims).
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let t = layer_to_tensor[l];
            let bs = layer_block_size[l];
            let tok = layer_kv_token_elems[l];
            let (kvh, hd) = if tok == num_kv_heads * head_dim {
                (num_kv_heads, head_dim)
            } else {
                (1, tok)
            };
            let shape = [num_blocks, bs, kvh, hd];
            // SAFETY: buffers owned via k_ptrs/v_ptrs; multiple layer views over
            // one buffer are disjoint at runtime (disjoint block ids per group).
            k_caches.push(unsafe { GpuTensor::new(k_ptrs[t].ptr(), &shape, dtype) });
            v_caches.push(unsafe { GpuTensor::new(v_ptrs[t].ptr(), &shape, dtype) });
        }

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "KvCachePool(grouped): {num_layers} layers / {num_tensors} physical tensors × \
             {num_blocks} blocks = {total_mb:.0} MB ({dtype})"
        );

        Ok(Self {
            k_caches,
            v_caches,
            _k_ptrs: k_ptrs,
            _v_ptrs: v_ptrs,
            num_blocks,
            block_size: base_block_size,
            max_blocks_per_seq,
            num_kv_heads,
            head_dim,
            num_layers,
            num_tensors,
            layer_to_tensor,
            kv_group_layout: None,
            per_layer_block_elems: None,
            cache_dtype: dtype,
            k_scale_ptrs: Vec::new(),
            v_scale_ptrs: Vec::new(),
            block_is_unrotated: vec![false; num_blocks],
            block_is_span: vec![false; num_blocks],
            block_unrotated_gpu_ptr: None,
            block_span_gpu_ptr: None,
            k_chunks: Vec::new(),
            v_chunks: Vec::new(),
            k_chunk_tables: Vec::new(),
            v_chunk_tables: Vec::new(),
            blocks_per_chunk: 0,
        })
    }

    /// Placeholder pool with zero layers and no GPU allocations. Used
    /// to satisfy the `&KvCachePool` field on `ForwardCtx` for vision
    /// encoder forwards, which never reach a kv_cache-touching
    /// instruction (vision body uses `VarlenAttention`, not the paged
    /// `Attention` op). Reading any layer index from this pool would
    /// panic — by contract, vision-mode codegen never emits such reads.
    pub fn empty_for_vision() -> Self {
        Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            _k_ptrs: Vec::new(),
            _v_ptrs: Vec::new(),
            num_blocks: 0,
            block_size: 0,
            // Vision body never reaches paged attention (uses VarlenAttention),
            // so no block-table is ever indexed against this; 0 is inert.
            max_blocks_per_seq: 0,
            num_kv_heads: 0,
            head_dim: 0,
            num_layers: 0,
            num_tensors: 0,
            layer_to_tensor: Vec::new(),
            kv_group_layout: None,
            per_layer_block_elems: None,
            cache_dtype: DType::BF16,
            k_scale_ptrs: Vec::new(),
            v_scale_ptrs: Vec::new(),
            block_is_unrotated: Vec::new(),
            block_is_span: Vec::new(),
            block_unrotated_gpu_ptr: None,
            block_span_gpu_ptr: None,
            k_chunks: Vec::new(),
            v_chunks: Vec::new(),
            k_chunk_tables: Vec::new(),
            v_chunk_tables: Vec::new(),
            blocks_per_chunk: 0,
        }
    }

    /// Get K cache tensor for a layer as a lifetime-checked view.
    pub fn k_cache(&self, layer: usize) -> TensorView<'_> {
        // Safety: KvCachePool owns the memory via _k_ptrs; view borrows &self.
        unsafe { TensorView::from_raw(self.k_caches[layer]) }
    }

    /// Get V cache tensor for a layer as a lifetime-checked view.
    pub fn v_cache(&self, layer: usize) -> TensorView<'_> {
        // Safety: KvCachePool owns the memory via _v_ptrs; view borrows &self.
        unsafe { TensorView::from_raw(self.v_caches[layer]) }
    }

    /// Per-layer paged block size (tokens/block). Uniform (`block_size`) on
    /// every uniform model; page-unified per class on the grouped hybrid path
    /// (gemma4: full layers 64, sliding 16) — read from the layer's own view
    /// shape so the slot-mapping encoding, `reshape_and_cache`, and paged
    /// attention all agree per layer. Callers that touch the paged cache for a
    /// specific layer MUST use this, not the scalar `block_size`.
    pub fn block_size_for(&self, layer: usize) -> usize {
        self.k_caches
            .get(layer)
            .map_or(self.block_size, |t| t.dim(1))
    }

    /// The largest per-layer block size — the page-unified full-group block
    /// size on the hybrid path (gemma4: 64), or the uniform `block_size`
    /// otherwise. The worker encodes the full/global group's slot_mapping with
    /// this (sliding groups keep the base `block_size`).
    pub fn max_block_size(&self) -> usize {
        (0..self.num_layers)
            .map(|l| self.block_size_for(l))
            .max()
            .unwrap_or(self.block_size)
    }

    /// The dtype used for cache storage.
    pub fn cache_dtype(&self) -> DType {
        self.cache_dtype
    }

    /// Whether this pool stores FP8 data.
    pub fn is_fp8(&self) -> bool {
        self.cache_dtype.is_fp8()
    }

    /// Per-layer K-cache backing memory. Metal callers use this to
    /// reach into the underlying `metal::Buffer` (via
    /// `PoolMem::buffer()`) for ICB binding without re-allocating.
    /// One `PoolMem` per layer, shape `[num_layers]`.
    pub fn k_layer_mem(&self, layer: usize) -> &M {
        &self._k_ptrs[self.layer_to_tensor[layer]]
    }

    /// Per-layer V-cache backing memory. See [`Self::k_layer_mem`].
    pub fn v_layer_mem(&self, layer: usize) -> &M {
        &self._v_ptrs[self.layer_to_tensor[layer]]
    }

    /// Chunk DATA buffers for `layer`'s physical tensor (chunked pool). Metal
    /// callers `useResource` these when a kernel reaches the data via the
    /// chunk table's gpuAddress (the chunks are otherwise bound only as the
    /// table, so a separate command buffer wouldn't have them resident). Empty
    /// on the non-chunked pool. Indexed by tensor via `layer_to_tensor`.
    pub fn k_chunk_bufs(&self, layer: usize) -> &[M] {
        self.k_chunks
            .get(self.layer_to_tensor[layer])
            .map_or(&[], |v| v.as_slice())
    }
    /// See [`Self::k_chunk_bufs`].
    pub fn v_chunk_bufs(&self, layer: usize) -> &[M] {
        self.v_chunks
            .get(self.layer_to_tensor[layer])
            .map_or(&[], |v| v.as_slice())
    }

    /// Allocate a **reactive (chunked)** metal KV pool: per layer per
    /// K/V, a series of `blocks_per_chunk`-block chunk buffers plus one
    /// chunk-address table buffer. RAM-neutral vs the single-buffer
    /// `new` (the chunk bytes sum to the same total, last chunk
    /// partial), but it lets the `MTLResidencySet` track only the
    /// chunks that back live KV. The single-buffer fields
    /// (`k_caches` / `_k_ptrs`) stay empty — metal binds chunk *tables*
    /// at the `KvCacheK/V` slots (see [`Self::k_chunk_table_mem`]), not
    /// the cache buffers, and never reads the `k_cache()` TensorView.
    ///
    /// `alloc_chunk(bytes)` allocates a StorageModePrivate chunk data
    /// buffer (and should insert it into the device residency set);
    /// `alloc_table(bytes)` allocates a StorageModeShared table buffer
    /// (host-writable via [`Self::fill_chunk_tables`], also residency-
    /// inserted). The chunk gpuAddresses are written into the tables by
    /// a later `fill_chunk_tables` call (the caller supplies the
    /// metal-specific `gpuAddress` accessor so this crate stays
    /// objc2-free).
    ///
    /// # Safety
    /// Caller must ensure the metal device is initialized on this
    /// thread.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_metal_chunked(
        num_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        // Per-sequence block-table row capacity (see the struct field). Runtime-
        // derived `min(ceil(max_model_len/block_size), num_blocks).max(1)`.
        max_blocks_per_seq: usize,
        // Per-TENSOR block-elem override for hybrid-geometry models
        // (Gemma4). `None` = uniform (the common case). When `Some`, len must
        // equal the physical tensor count (`num_layers`, or `group_size` when
        // `layer_to_tensor` is `Some`) and each entry is that tensor's
        // `kv_heads * block_size * head_dim` (page-unified → uniform per tensor).
        per_layer_block_elems: Option<Vec<usize>>,
        // vLLM group-shared layout: maps each logical layer → physical tensor.
        // `None` = one tensor per layer (identity; every uniform model). `Some`
        // collapses the `num_layers` chunk vecs to `max(map)+1` shared tensors.
        layer_to_tensor: Option<Vec<usize>>,
        dtype: DType,
        blocks_per_chunk: usize,
        // Chunks to allocate up front. `1` = reactive/lazy (grow on
        // demand via `grow_to_cover`); `usize::MAX` = eager (allocate
        // the whole pool now — used for the draft pool, whose spec-decode
        // forward path has no growth hook).
        initial_chunks: usize,
        mut alloc_chunk: impl FnMut(usize) -> Result<M>,
        mut alloc_table: impl FnMut(usize) -> Result<M>,
    ) -> Result<Self> {
        assert!(blocks_per_chunk > 0, "blocks_per_chunk must be non-zero");
        if dtype.is_fp8() {
            anyhow::bail!("new_metal_chunked: FP8 cache dtype is cuda-only");
        }
        let num_chunks = num_blocks.div_ceil(blocks_per_chunk);
        let elem = dtype.size_bytes();
        // Elements per paged block = kv_blk_stride (num_kv_heads *
        // block_size * head_dim). A chunk holds `blocks_in_chunk` of these.
        let per_block_elems = num_kv_heads * block_size * head_dim;
        // Physical tensors: one per layer (uniform), or `group_size` shared
        // tensors (hybrid). `layer_to_tensor` resolves layer → tensor.
        let layer_to_tensor: Vec<usize> =
            layer_to_tensor.unwrap_or_else(|| (0..num_layers).collect());
        assert_eq!(
            layer_to_tensor.len(),
            num_layers,
            "layer_to_tensor len must equal num_layers"
        );
        let num_tensors = layer_to_tensor.iter().copied().max().map_or(0, |m| m + 1);
        if let Some(v) = per_layer_block_elems.as_ref() {
            assert_eq!(
                v.len(),
                num_tensors,
                "per_layer_block_elems len must equal the physical tensor count"
            );
        }
        let tensor_block_elems = |tensor: usize| {
            per_layer_block_elems
                .as_ref()
                .map_or(per_block_elems, |v| v[tensor])
        };

        let mut k_chunks: Vec<Vec<M>> = Vec::with_capacity(num_tensors);
        let mut v_chunks: Vec<Vec<M>> = Vec::with_capacity(num_tensors);
        let mut k_chunk_tables: Vec<M> = Vec::with_capacity(num_tensors);
        let mut v_chunk_tables: Vec<M> = Vec::with_capacity(num_tensors);

        // REACTIVE (2b): allocate only the FIRST chunk up front; the
        // rest grow on demand via `grow_one_chunk` as sequences fill
        // blocks (driven by the worker's per-step high-water-mark). The
        // chunk-address tables are allocated at FULL size (one u64 slot
        // per potential chunk — tiny, ~8 B/chunk) and bound once; only
        // slot 0 is filled here, the rest filled lazily on growth. This
        // is what drops startup RAM from ~the whole pool to weights +
        // one chunk.
        // At least one chunk (if the pool is non-empty), at most all.
        let initial_chunks = initial_chunks.clamp(num_chunks.min(1), num_chunks);
        for tensor in 0..num_tensors {
            let mut kc = Vec::with_capacity(num_chunks);
            let mut vc = Vec::with_capacity(num_chunks);
            for chunk in 0..initial_chunks {
                // Chunk may be partial so chunk bytes sum to exactly
                // num_blocks (no over-allocation).
                let blocks_here = blocks_per_chunk.min(num_blocks - chunk * blocks_per_chunk);
                let bytes = blocks_here * tensor_block_elems(tensor) * elem;
                kc.push(alloc_chunk(bytes)?);
                vc.push(alloc_chunk(bytes)?);
            }
            k_chunks.push(kc);
            v_chunks.push(vc);
            // Full-size address tables (one u64 slot per potential chunk).
            k_chunk_tables.push(alloc_table(num_chunks * 8)?);
            v_chunk_tables.push(alloc_table(num_chunks * 8)?);
        }

        let initial_mb =
            (2 * num_tensors * initial_chunks * blocks_per_chunk * per_block_elems * elem) as f64
                / (1024.0 * 1024.0);
        let ceiling_mb =
            (2 * num_tensors * num_blocks * per_block_elems * elem) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "KvCachePool(metal chunked, reactive): {num_layers} layers / {num_tensors} physical \
             tensors, {initial_chunks}/{num_chunks} chunks resident at init ({initial_mb:.0} MB) — \
             grows on demand up to {num_blocks} blocks ({ceiling_mb:.0} MB ceiling) at \
             {blocks_per_chunk} blocks/chunk ({dtype})"
        );

        Ok(Self {
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            _k_ptrs: Vec::new(),
            _v_ptrs: Vec::new(),
            num_blocks,
            block_size,
            max_blocks_per_seq,
            num_kv_heads,
            head_dim,
            num_layers,
            num_tensors,
            layer_to_tensor,
            kv_group_layout: None,
            per_layer_block_elems,
            cache_dtype: dtype,
            k_scale_ptrs: Vec::new(),
            v_scale_ptrs: Vec::new(),
            block_is_unrotated: vec![false; num_blocks],
            block_is_span: vec![false; num_blocks],
            block_unrotated_gpu_ptr: None,
            block_span_gpu_ptr: None,
            k_chunks,
            v_chunks,
            k_chunk_tables,
            v_chunk_tables,
            blocks_per_chunk,
        })
    }

    /// Per-layer K chunk-address *table* backing memory (the buffer the
    /// kernels bind at the `KvCacheK` slot — a device array of chunk
    /// gpuAddresses, not the cache data). Empty unless built via
    /// [`Self::new_metal_chunked`].
    pub fn k_chunk_table_mem(&self, layer: usize) -> &M {
        &self.k_chunk_tables[self.layer_to_tensor[layer]]
    }

    /// Number of vLLM KV-cache GROUPS (per-group block tables). `1` for
    /// uniform models; `1 + N` sliding for gemma4 SWA. The metal worker
    /// allocates this many `slot_mappings` / `block_tables` buffers.
    pub fn num_kv_groups(&self) -> usize {
        self.kv_group_layout.as_ref().map_or(1, |(n, _)| *n)
    }

    /// Per-layer KV-cache group index (`[num_layers]` u32), indexing the
    /// worker's `block_tables` / `slot_mappings`. All-zero on uniform models
    /// (every layer → group 0). Set via [`Self::set_kv_group_layout`].
    pub fn layer_to_group_u32(&self) -> Vec<u32> {
        self.kv_group_layout
            .as_ref()
            .map(|(_, m)| m.clone())
            .unwrap_or_else(|| vec![0u32; self.num_layers])
    }

    /// Layer → KV-cache group index (`0` on uniform models, no allocation).
    /// The per-forward hot path (every attention op) uses this to pick its
    /// group's block table / slot mapping; prefer it over `layer_to_group_u32`
    /// (which clones a Vec).
    pub fn group_of_layer(&self, layer: usize) -> usize {
        self.kv_group_layout
            .as_ref()
            .map_or(0, |(_, m)| m.get(layer).copied().unwrap_or(0) as usize)
    }

    /// Attach the vLLM KV-group layout (gemma4 SWA): `num_groups` per-group
    /// block tables and each layer's group index. Asserted consistent with
    /// `num_layers`. No-op contract for uniform models (never called → stays
    /// the single-group default).
    pub fn set_kv_group_layout(&mut self, num_groups: usize, layer_to_group: Vec<u32>) {
        assert_eq!(
            layer_to_group.len(),
            self.num_layers,
            "layer_to_group len must equal num_layers"
        );
        assert!(
            layer_to_group.iter().all(|&g| (g as usize) < num_groups),
            "layer_to_group has a group index >= num_groups"
        );
        self.kv_group_layout = Some((num_groups, layer_to_group));
    }

    /// Per-layer V chunk-address table. See [`Self::k_chunk_table_mem`].
    pub fn v_chunk_table_mem(&self, layer: usize) -> &M {
        &self.v_chunk_tables[self.layer_to_tensor[layer]]
    }

    /// Blocks backed by one physical chunk buffer (0 unless chunked).
    pub fn blocks_per_chunk(&self) -> usize {
        self.blocks_per_chunk
    }

    /// Write each chunk's `gpuAddress` into its per-layer chunk-address
    /// table (StorageModeShared, host-writable). `gpu_addr` extracts the
    /// 64-bit GPU virtual address of a chunk buffer — the caller
    /// supplies it (`|m| m.buffer().gpuAddress()`) so this crate needs
    /// no objc2-metal dependency. Call once after
    /// [`Self::new_metal_chunked`] and (in later phases) again whenever
    /// the chunk set changes.
    pub fn fill_chunk_tables(&self, gpu_addr: impl Fn(&M) -> u64) {
        for tensor in 0..self.num_tensors {
            let kt = self.k_chunk_tables[tensor].ptr() as *mut u64;
            let vt = self.v_chunk_tables[tensor].ptr() as *mut u64;
            for (c, buf) in self.k_chunks[tensor].iter().enumerate() {
                // Safety: table is a StorageModeShared buffer of
                // `num_chunks` u64 slots; `c` is in range by construction.
                unsafe {
                    kt.add(c).write(gpu_addr(buf));
                }
            }
            for (c, buf) in self.v_chunks[tensor].iter().enumerate() {
                unsafe {
                    vt.add(c).write(gpu_addr(buf));
                }
            }
        }
    }

    /// Total chunks this pool can grow to (`ceil(num_blocks / BPC)`).
    pub fn num_chunks_total(&self) -> usize {
        if self.blocks_per_chunk == 0 {
            0
        } else {
            self.num_blocks.div_ceil(self.blocks_per_chunk)
        }
    }

    /// Number of paged blocks currently backed by allocated chunks
    /// (`allocated_chunks * BPC`, capped at `num_blocks`). The scheduler
    /// may hand out any block id `< num_blocks`; the worker must
    /// `grow_to_cover` before a forward references a block ≥ this.
    pub fn allocated_blocks(&self) -> usize {
        let chunks = self.k_chunks.first().map_or(0, |v| v.len());
        (chunks * self.blocks_per_chunk).min(self.num_blocks)
    }

    /// Reactive shrink (2c): drop all chunks past the first `keep`,
    /// returning the freed K+V chunk buffers (all layers) so the caller
    /// can `residency.remove` + `commit` them BEFORE they're dropped
    /// (freed). The chunk-address table entries for the dropped chunks
    /// become stale but are never read — a later `grow_to_cover`
    /// re-allocates from the shrunk length and overwrites them. Only
    /// safe to call when no live block falls in a dropped chunk (e.g.
    /// the batch is fully idle); the caller owns that invariant.
    pub fn shrink_to_chunks(&mut self, keep: usize) -> Vec<M> {
        let keep = keep.max(1); // always retain chunk 0
        let cur = self.k_chunks.first().map_or(0, |v| v.len());
        if keep >= cur {
            return Vec::new();
        }
        let mut freed = Vec::with_capacity((cur - keep) * 2 * self.num_tensors);
        for tensor in 0..self.num_tensors {
            while self.k_chunks[tensor].len() > keep {
                freed.push(self.k_chunks[tensor].pop().expect("k chunk"));
                freed.push(self.v_chunks[tensor].pop().expect("v chunk"));
            }
        }
        freed
    }

    /// Grow the pool until `block_id` is backed by an allocated chunk.
    /// Allocates each missing chunk's K+V buffers for every layer (via
    /// `alloc_chunk`, which must residency-insert) and writes their
    /// `gpuAddress` (via `gpu_addr`) into the per-layer chunk tables.
    /// Returns the number of chunks newly allocated (0 = already
    /// covered); the caller must `residency.commit()` once if > 0 before
    /// dispatching, so the new chunk pages are wired for the bindless
    /// deref. No-op for non-chunked (cuda / single-buffer) pools.
    pub fn grow_to_cover(
        &mut self,
        block_id: usize,
        mut alloc_chunk: impl FnMut(usize) -> Result<M>,
        gpu_addr: impl Fn(&M) -> u64,
    ) -> Result<usize> {
        if self.blocks_per_chunk == 0 || self.k_chunks.is_empty() {
            return Ok(0); // not a chunked pool
        }
        let target_block = block_id.min(self.num_blocks.saturating_sub(1));
        let num_chunks = self.num_chunks_total();
        let elem = self.cache_dtype.size_bytes();
        let per_block_elems = self.num_kv_heads * self.block_size * self.head_dim;
        let mut grew = 0usize;
        // All layers grow in lockstep, so chunk count == k_chunks[0].len().
        while self.allocated_blocks() <= target_block {
            let next = self.k_chunks[0].len();
            if next >= num_chunks {
                break;
            }
            let blocks_here = self
                .blocks_per_chunk
                .min(self.num_blocks - next * self.blocks_per_chunk);
            for tensor in 0..self.num_tensors {
                // Per-tensor block size for hybrid-geometry models
                // (Gemma4, page-unified → uniform); uniform fallback otherwise.
                let lbe = self
                    .per_layer_block_elems
                    .as_ref()
                    .map_or(per_block_elems, |v| v[tensor]);
                let bytes = blocks_here * lbe * elem;
                let kc = alloc_chunk(bytes)?;
                let vc = alloc_chunk(bytes)?;
                let ka = gpu_addr(&kc);
                let va = gpu_addr(&vc);
                // Write the chunk gpuAddresses into the (already-bound)
                // per-tensor address tables at slot `next`.
                unsafe {
                    (self.k_chunk_tables[tensor].ptr() as *mut u64)
                        .add(next)
                        .write(ka);
                    (self.v_chunk_tables[tensor].ptr() as *mut u64)
                        .add(next)
                        .write(va);
                }
                self.k_chunks[tensor].push(kc);
                self.v_chunks[tensor].push(vc);
            }
            grew += 1;
        }
        Ok(grew)
    }

    /// GPU pointer to K scale for a layer (only valid when FP8).
    pub fn k_scale_ptr(&self, layer: usize) -> *const f32 {
        self.k_scale_ptrs[layer].ptr() as *const f32
    }

    /// GPU pointer to V scale for a layer (only valid when FP8).
    pub fn v_scale_ptr(&self, layer: usize) -> *const f32 {
        self.v_scale_ptrs[layer].ptr() as *const f32
    }

    /// Mutable GPU pointer to K scale for a layer (for writing computed scales).
    pub fn k_scale_ptr_mut(&self, layer: usize) -> *mut f32 {
        self.k_scale_ptrs[layer].ptr() as *mut f32
    }

    /// Mutable GPU pointer to V scale for a layer (for writing computed scales).
    pub fn v_scale_ptr_mut(&self, layer: usize) -> *mut f32 {
        self.v_scale_ptrs[layer].ptr() as *mut f32
    }

    /// Raw K-cache `GpuTensor` for a layer (the single-buffer cuda path).
    /// For the cuda extension's block-gather; `k_cache` returns the
    /// lifetime-checked view used everywhere else.
    pub fn k_cache_raw(&self, layer: usize) -> GpuTensor {
        self.k_caches[layer]
    }

    /// Raw V-cache `GpuTensor` for a layer. See [`Self::k_cache_raw`].
    pub fn v_cache_raw(&self, layer: usize) -> GpuTensor {
        self.v_caches[layer]
    }

    /// Lazily allocate the one-byte-per-block GPU mirror buffers for the
    /// rotation/span flag arrays via `alloc`, if not already present. The
    /// cuda extension's `sync_block_flags_to_gpu` calls this and then uploads
    /// the flag bytes to [`Self::block_unrotated_gpu`] / [`Self::block_span_gpu`];
    /// metal has no span machinery and never calls it.
    pub fn ensure_block_flag_mirrors(
        &mut self,
        mut alloc: impl FnMut(usize) -> Result<M>,
    ) -> Result<()> {
        let n = self.block_is_unrotated.len();
        if self.block_unrotated_gpu_ptr.is_none() {
            self.block_unrotated_gpu_ptr = Some(alloc(n)?);
        }
        if self.block_span_gpu_ptr.is_none() {
            self.block_span_gpu_ptr = Some(alloc(n)?);
        }
        Ok(())
    }

    /// Mark a physical block's current rotation state and span identity.
    pub fn mark_block(&mut self, physical_block_id: usize, is_span: bool, is_unrotated: bool) {
        if physical_block_id < self.block_is_unrotated.len() {
            self.block_is_span[physical_block_id] = is_span;
            self.block_is_unrotated[physical_block_id] = is_unrotated;
        }
    }

    /// GPU pointer to `block_is_unrotated` flags (for pre-attention rotation).
    pub fn block_unrotated_gpu(&self) -> *const u8 {
        self.block_unrotated_gpu_ptr
            .as_ref()
            .map(|m| m.ptr() as *const u8)
            .unwrap_or(std::ptr::null())
    }

    /// GPU pointer to `block_is_span` flags (for post-attention un-rotation).
    pub fn block_span_gpu(&self) -> *const u8 {
        self.block_span_gpu_ptr
            .as_ref()
            .map(|m| m.ptr() as *const u8)
            .unwrap_or(std::ptr::null())
    }
}

// All GPU allocations (_k_ptrs, _v_ptrs, k_scale_ptrs, v_scale_ptrs,
// block_unrotated_gpu_ptr, block_span_gpu_ptr) are PoolMem and freed
// automatically via Drop — no manual impl needed.

#[cfg(test)]
mod group_shared_tests {
    use super::*;

    /// Host mock of a `PoolMemory` allocation: owns a heap buffer so the
    /// chunk-address-table writes (`grow_to_cover`) land in valid memory.
    struct MockMem {
        ptr: *mut u8,
        len: usize,
    }
    impl MockMem {
        fn alloc(bytes: usize) -> Self {
            let boxed = vec![0u8; bytes.max(1)].into_boxed_slice();
            let len = boxed.len();
            MockMem {
                ptr: Box::into_raw(boxed) as *mut u8,
                len,
            }
        }
    }
    impl PoolMemory for MockMem {
        fn ptr(&self) -> *mut u8 {
            self.ptr
        }
    }
    // Test-only: the mock is used single-threaded.
    unsafe impl Send for MockMem {}
    unsafe impl Sync for MockMem {}
    impl Drop for MockMem {
        fn drop(&mut self) {
            unsafe {
                drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    self.ptr, self.len,
                )));
            }
        }
    }

    fn build(num_layers: usize, map: Option<Vec<usize>>) -> KvCachePool<MockMem> {
        unsafe {
            KvCachePool::<MockMem>::new_metal_chunked(
                num_layers,
                64,
                16,
                8,
                256,
                // max_blocks_per_seq: 64 blocks / 16 block_size = 4.
                4,
                None,
                map,
                DType::BF16,
                16,
                1,
                |b| Ok(MockMem::alloc(b)),
                |b| Ok(MockMem::alloc(b)),
            )
            .unwrap()
        }
    }

    #[test]
    fn group_shared_tensors_are_position_indexed() {
        // 5 layers → 2 physical tensors (positions 0,1,0,1,0). Same-position
        // layers MUST share one chunk-table buffer (block IDs are disjoint
        // across the groups sharing a tensor, so co-resident is safe).
        let pool = build(5, Some(vec![0, 1, 0, 1, 0]));
        assert_eq!(pool.num_tensors, 2, "5 layers collapse to 2 shared tensors");
        assert_eq!(
            pool.k_chunks.len(),
            2,
            "chunk vecs indexed by tensor, not layer"
        );
        let p0 = pool.k_chunk_table_mem(0).ptr();
        for l in [2usize, 4] {
            assert_eq!(
                pool.k_chunk_table_mem(l).ptr(),
                p0,
                "layer {l} (position 0) must share tensor 0's buffer"
            );
            assert_eq!(
                pool.v_chunk_table_mem(l).ptr(),
                pool.v_chunk_table_mem(0).ptr()
            );
        }
        assert_ne!(
            pool.k_chunk_table_mem(1).ptr(),
            p0,
            "position 1 is a different physical tensor"
        );
    }

    #[test]
    fn uniform_path_is_one_tensor_per_layer() {
        // None → identity mapping → byte-identical to the pre-hybrid pool.
        let pool = build(3, None);
        assert_eq!(pool.num_tensors, 3, "uniform: one tensor per layer");
        assert_ne!(
            pool.k_chunk_table_mem(0).ptr(),
            pool.k_chunk_table_mem(1).ptr()
        );
        assert_ne!(
            pool.k_chunk_table_mem(1).ptr(),
            pool.k_chunk_table_mem(2).ptr()
        );
    }

    #[test]
    fn grow_to_cover_grows_shared_tensors_in_lockstep() {
        let mut pool = build(5, Some(vec![0, 1, 0, 1, 0]));
        assert_eq!(
            pool.allocated_blocks(),
            16,
            "1 chunk × 16 blocks/chunk at init"
        );
        let grew = pool
            .grow_to_cover(40, |b| Ok(MockMem::alloc(b)), |m| m.ptr() as u64)
            .unwrap();
        assert!(
            grew >= 2,
            "covering block 40 needs >=3 chunks (grew {grew})"
        );
        assert!(pool.allocated_blocks() > 40);
        assert_eq!(
            pool.k_chunks[0].len(),
            pool.k_chunks[1].len(),
            "both shared tensors grow in lockstep"
        );
    }
}
