// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache pool — persistent device memory.
//!
//! Layout per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`,
//! stored as a series of fixed-size chunk buffers. Backend-neutral: storage,
//! sizing and chunk bookkeeping live here; the actual buffer allocation is a
//! caller-supplied closure (metal wraps `device.new_buffer`), so this type
//! names no backend.

use anyhow::Result;
use scratchy_tensors::{DType, PoolMemory};

/// Paged KV cache pool for all transformer layers.
///
/// Allocates persistent device memory for K and V caches. Block
/// allocation/freeing is managed by the scheduler — this struct just owns
/// the storage.
pub struct KvCachePool<M: PoolMemory> {
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
    per_layer_block_elems: Option<Vec<usize>>,
    /// The dtype stored in cache.
    cache_dtype: DType,

    // ── Reactive (chunked) KV storage ───────────────────────────────
    //
    // The per-layer KV cache is NOT one contiguous buffer; it is a
    // series of fixed-size *chunk* buffers (`blocks_per_chunk` blocks
    // each) so the `MTLResidencySet` can hold only the chunks that back
    // live KV. The kernels bind a per-layer chunk-address *table*
    // (device uint64 gpuAddresses) at the `KvCacheK/V` slot and deref
    // `table[block_id / blocks_per_chunk]` then address
    // `block_id % blocks_per_chunk` within that chunk.
    //
    // `k_chunks[layer][chunk]` / `v_chunks[layer][chunk]` own the chunk
    // data buffers (StorageModePrivate); `k_chunk_tables[layer]` /
    // `v_chunk_tables[layer]` own the per-layer address tables
    // (StorageModeShared, host-written via `fill_chunk_tables`).
    k_chunks: Vec<Vec<M>>,
    v_chunks: Vec<Vec<M>>,
    k_chunk_tables: Vec<M>,
    v_chunk_tables: Vec<M>,
    blocks_per_chunk: usize,
}

// Safety: KvCachePool holds device memory handles. These are allocated via the backend's device
// memory and are accessible from any host thread after backend setup. The
// pool is created once and moved to the worker thread; no concurrent
// mutation occurs.
unsafe impl<M: PoolMemory> Send for KvCachePool<M> {}
unsafe impl<M: PoolMemory> Sync for KvCachePool<M> {}

impl<M: PoolMemory> KvCachePool<M> {
    /// Placeholder pool with zero layers and no GPU allocations. Used
    /// to satisfy the `&KvCachePool` field on `ForwardCtx` for vision
    /// encoder forwards, which never reach a kv_cache-touching
    /// instruction (vision body uses `VarlenAttention`, not the paged
    /// `Attention` op). Reading any layer index from this pool would
    /// panic — by contract, vision-mode codegen never emits such reads.
    pub fn empty_for_vision() -> Self {
        Self {
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
            k_chunks: Vec::new(),
            v_chunks: Vec::new(),
            k_chunk_tables: Vec::new(),
            v_chunk_tables: Vec::new(),
            blocks_per_chunk: 0,
        }
    }

    /// Allocate a **reactive (chunked)** metal KV pool: per layer per
    /// K/V, a series of `blocks_per_chunk`-block chunk buffers plus one
    /// chunk-address table buffer (the chunk bytes sum to the whole
    /// pool, last chunk partial), so the `MTLResidencySet` tracks only
    /// the chunks that back live KV. Metal binds chunk *tables* at the
    /// `KvCacheK/V` slots (see [`Self::k_chunk_table_mem`]).
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
            anyhow::bail!("new_metal_chunked: FP8 cache dtype is not supported");
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
    /// decides when they're dropped (freed). The chunk-address table
    /// entries for the dropped chunks become stale but are never read — a
    /// later `grow_to_cover`
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

    /// The block a grow should cover to serve `block_id`: at least double the
    /// allocated chunks, so a filling pool grows in a handful of steps — each
    /// allocates and wires memory inside a serving step — while staying within
    /// 2× of the blocks in use.
    pub fn growth_target(&self, block_id: usize) -> usize {
        let allocated = self.allocated_blocks();
        if block_id < allocated {
            block_id
        } else {
            block_id.max((2 * allocated).saturating_sub(1))
        }
    }

    /// Grow the pool until `block_id` is backed by an allocated chunk.
    /// Allocates each missing chunk's K+V buffers for every layer (via
    /// `alloc_chunk`, which must return resident memory) and writes their
    /// `gpuAddress` (via `gpu_addr`) into the per-layer chunk tables.
    /// Returns the number of chunks newly allocated (0 = already
    /// covered); the caller must `residency.commit()` once if > 0 before
    /// dispatching, so the new chunk pages are wired for the bindless
    /// deref. On Err the pool holds exactly the chunks it held whole before
    /// the failing one. No-op for a pool with no chunks (the vision placeholder).
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
            // Allocate chunk `next` for EVERY tensor before the pool records any of
            // it: a failure partway leaves the pool as it was (no tensor holds the
            // chunk, no table entry changed), so the next grow retries it whole.
            let mut fresh = Vec::with_capacity(self.num_tensors);
            for tensor in 0..self.num_tensors {
                // Per-tensor block size for hybrid-geometry models
                // (Gemma4, page-unified → uniform); uniform fallback otherwise.
                let lbe = self
                    .per_layer_block_elems
                    .as_ref()
                    .map_or(per_block_elems, |v| v[tensor]);
                let bytes = blocks_here * lbe * elem;
                fresh.push((alloc_chunk(bytes)?, alloc_chunk(bytes)?));
            }
            for (tensor, (kc, vc)) in fresh.into_iter().enumerate() {
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
}

// All device allocations (the chunk buffers and tables) are `M` handles freed
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

    #[test]
    fn growth_target_at_least_doubles_the_pool() {
        let pool = build(3, None); // 1 chunk of 16 blocks, 64 blocks in all
        assert_eq!(pool.growth_target(10), 10, "already covered: no growth");
        assert_eq!(
            pool.growth_target(16),
            31,
            "first block past the pool: double it"
        );
        assert_eq!(
            pool.growth_target(40),
            40,
            "past double: cover the block itself"
        );
    }

    #[test]
    fn a_grow_that_fails_partway_leaves_the_pool_whole() {
        // 2 shared tensors → 4 allocations per chunk; the 3rd (tensor 1's K) fails.
        let mut pool = build(5, Some(vec![0, 1, 0, 1, 0]));
        let table = pool.k_chunk_table_mem(0).ptr() as *const u64;
        let calls = std::cell::Cell::new(0);
        let failing = |b| {
            calls.set(calls.get() + 1);
            if calls.get() == 3 {
                Err(anyhow::anyhow!("allocation failed"))
            } else {
                Ok(MockMem::alloc(b))
            }
        };
        assert!(pool.grow_to_cover(20, failing, |m| m.ptr() as u64).is_err());
        assert_eq!(
            pool.allocated_blocks(),
            16,
            "the half-built chunk is not counted"
        );
        assert_eq!((pool.k_chunks[0].len(), pool.k_chunks[1].len()), (1, 1));
        assert_eq!(unsafe { *table.add(1) }, 0, "no table entry written for it");
        // The next grow retries the chunk whole.
        let grew = pool.grow_to_cover(20, |b| Ok(MockMem::alloc(b)), |m| m.ptr() as u64);
        assert_eq!(grew.unwrap(), 1);
        assert_eq!((pool.k_chunks[0].len(), pool.k_chunks[1].len()), (2, 2));
    }
}
