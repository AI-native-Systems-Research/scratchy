// SPDX-License-Identifier: Apache-2.0
//! Caching GPU memory allocator — matches PyTorch's CUDACachingAllocator design.
//!
//! Key features (all matching PyTorch):
//! - **Segments**: Large `cudaMalloc` calls (2 MB for small, ≥10 MB for large)
//! - **Block splitting**: A segment is divided into blocks; when a request
//!   is smaller than a free block, the block is split and the remainder stays free
//! - **Block coalescing**: When a block is freed, it merges with adjacent free blocks
//! - **Two pools**: small (≤1 MB) and large (>1 MB), each with a sorted free set
//! - **Private pools**: For CUDA graph capture (like PyTorch's beginAllocateToPool)

use crate::driver;
use crate::dtype::DType;
use crate::tensor::{GpuTensor, TensorView};
use cudarc::driver::sys as cuda_sys;
use std::collections::BTreeSet;
use std::ptr;

// `RawGpuMem` and `OwnedTensor` are now the backend-neutral
// `scratchy_tensors` types. Re-export both at the historic `crate::alloc::*`
// paths so existing call sites compile unchanged.
pub use scratchy_tensors::{OwnedTensor, RawGpuMem};

// ---------------------------------------------------------------------------
// Capture mode guard (matches PyTorch's CUDAStreamCaptureModeGuard)
// ---------------------------------------------------------------------------

/// RAII guard that switches the thread-local stream capture mode to RELAXED,
/// restoring the previous mode on drop.
///
/// During CUDA graph capture, cudaMalloc is only allowed in relaxed mode.
/// PyTorch uses the same pattern (CUDAStreamCaptureModeGuard) in its caching
/// allocator to allow new segment allocations during capture.
pub struct RelaxedCaptureModeGuard {
    prev_mode: cuda_sys::CUstreamCaptureMode,
}

impl RelaxedCaptureModeGuard {
    pub unsafe fn new() -> Self {
        let mut mode = cuda_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED;
        // Exchange: sets relaxed, returns previous mode.
        let _ = cuda_sys::cuThreadExchangeStreamCaptureMode(&mut mode);
        Self { prev_mode: mode }
    }
}

impl Drop for RelaxedCaptureModeGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_sys::cuThreadExchangeStreamCaptureMode(&mut self.prev_mode);
        }
    }
}

// ---------------------------------------------------------------------------
// Constants (matching PyTorch exactly)
// ---------------------------------------------------------------------------

/// Minimum block size (all allocations rounded up to this).
const K_MIN_BLOCK_SIZE: usize = 512;
/// Largest "small" allocation.
const K_SMALL_SIZE: usize = 1_048_576; // 1 MiB
/// Segment size for small allocations.
const K_SMALL_BUFFER: usize = 2_097_152; // 2 MiB
/// Minimum size for a direct large allocation.
const K_MIN_LARGE_ALLOC: usize = 10_485_760; // 10 MiB
/// Round up large allocations to this.
const K_ROUND_LARGE: usize = 2_097_152; // 2 MiB

// ---------------------------------------------------------------------------
// Block — sub-allocation within a segment
// ---------------------------------------------------------------------------

struct Block {
    ptr: *mut u8,
    size: usize,
    allocated: bool,
    prev: *mut Block,
    next: *mut Block,
    pool_is_small: bool,
}

impl Block {
    #[allow(dead_code)]
    fn is_split(&self) -> bool {
        !self.prev.is_null() || !self.next.is_null()
    }
}

// ---------------------------------------------------------------------------
// BlockPool — sorted set of free blocks (by size then address)
// ---------------------------------------------------------------------------

/// Comparison key for the free block set: (size, ptr address).
/// Sorted by size first (best-fit), then by address for determinism.
#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
struct BlockKey {
    size: usize,
    ptr: usize,
}

impl BlockKey {
    fn from_block(b: &Block) -> Self {
        Self {
            size: b.size,
            ptr: b.ptr as usize,
        }
    }
}

struct BlockPool {
    /// Free blocks sorted by (size, address) for best-fit search.
    free_blocks: BTreeSet<(BlockKey, *mut Block)>,
    is_small: bool,
}

// Safety: BlockPool contains raw pointers to Block structs and GPU memory,
// but these are only accessed through &mut self methods. The allocator is
// single-threaded; Send is needed so CachingAllocator (which owns pools) can
// be moved to the worker thread.
unsafe impl Send for BlockPool {}

impl BlockPool {
    fn new(is_small: bool) -> Self {
        Self {
            free_blocks: BTreeSet::new(),
            is_small,
        }
    }

    fn insert(&mut self, block: *mut Block) {
        let b = unsafe { &*block };
        self.free_blocks.insert((BlockKey::from_block(b), block));
    }

    fn remove(&mut self, block: *mut Block) {
        let b = unsafe { &*block };
        self.free_blocks.remove(&(BlockKey::from_block(b), block));
    }

    /// Find the smallest free block >= `size`.
    fn find_best_fit(&mut self, size: usize) -> Option<*mut Block> {
        let search = BlockKey { size, ptr: 0 };
        if let Some(&(_, block)) = self.free_blocks.range((search, ptr::null_mut())..).next() {
            Some(block)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// CachingAllocator
// ---------------------------------------------------------------------------

/// A caching GPU memory allocator matching PyTorch's CUDACachingAllocator.
pub struct CachingAllocator {
    small_pool: BlockPool,
    large_pool: BlockPool,
    /// All allocated segments (for cleanup on drop).
    segments: Vec<(*mut u8, usize)>,
    /// All Block structs (for cleanup on drop).
    all_blocks: Vec<*mut Block>,
    /// Active blocks keyed by ptr (for fast lookup on free).
    active_blocks: std::collections::HashMap<usize, *mut Block>,
    /// Private pool redirect (for CUDA graph capture).
    private_small_pool: Option<BlockPool>,
    private_large_pool: Option<BlockPool>,
    /// Bytes currently live (allocated but not freed). Mirrors PyTorch's
    /// `allocated_bytes.all.current`.
    active_bytes: usize,
    /// Peak value of `active_bytes` since last `reset_peak_stats()`. Mirrors
    /// PyTorch's `allocated_bytes.all.peak`.
    peak_active_bytes: usize,
}

// Safety: CachingAllocator contains raw pointers to GPU memory and Block
// structs. All mutation goes through &mut self, so there is no interior
// mutability. The allocator is created on the main thread and moved to the
// GPU worker thread (Send). Sync is needed because &CachingAllocator may be
// read from diagnostic/stats paths while the worker holds &mut.
unsafe impl Send for CachingAllocator {}
unsafe impl Sync for CachingAllocator {}

impl Default for CachingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CachingAllocator {
    pub fn new() -> Self {
        Self {
            small_pool: BlockPool::new(true),
            large_pool: BlockPool::new(false),
            segments: Vec::new(),
            all_blocks: Vec::new(),
            active_blocks: std::collections::HashMap::new(),
            private_small_pool: None,
            private_large_pool: None,
            active_bytes: 0,
            peak_active_bytes: 0,
        }
    }

    /// Reset peak-active counter (call before a section you want to profile).
    pub fn reset_peak_stats(&mut self) {
        self.peak_active_bytes = self.active_bytes;
    }

    /// Bytes currently allocated and not yet freed. Mirrors PyTorch's
    /// `allocated_bytes.all.current` stat.
    pub fn active_bytes(&self) -> usize {
        self.active_bytes
    }

    /// Peak active bytes since last `reset_peak_stats()`. Mirrors PyTorch's
    /// `allocated_bytes.all.peak` stat.
    pub fn peak_active_bytes(&self) -> usize {
        self.peak_active_bytes
    }

    /// Begin allocating to a private pool (for CUDA graph capture).
    pub fn begin_allocate_to_pool(&mut self) {
        assert!(
            self.private_small_pool.is_none(),
            "already allocating to a pool"
        );
        self.private_small_pool = Some(BlockPool::new(true));
        self.private_large_pool = Some(BlockPool::new(false));
    }

    /// Stop allocating to the private pool. Future `alloc()` calls go to the
    /// regular pools, leaving captured-kernel addresses (which live in the
    /// private pool's segments) frozen for the lifetime of the captured graph.
    ///
    /// Why: at piecewise CUDA graph replay, host-side helpers (e.g.,
    /// `copy_to_fresh_owned`, `all_gather_last_dim`) allocate logits buffers
    /// from `device.caching`. If those allocs come back from the private pool,
    /// they alias scratch addresses the captured kernels still write to —
    /// producing replay-time data races that look like garbage logits past the
    /// first decode token at TP>1.
    ///
    /// The dropped `BlockPool::free_blocks` set holds Block pointers that
    /// were freed during capture. Those Block structs still live in
    /// `all_blocks` (so cuMemFree fires on `Drop`), but their entries are no
    /// longer in any free list — they become permanently held until allocator
    /// teardown. Acceptable: the private-pool free-list is small (per-step
    /// scratch tensors) and the trade is correctness vs a few-MB hold.
    pub fn end_allocate_to_pool(&mut self) {
        self.private_small_pool = None;
        self.private_large_pool = None;
    }

    pub fn is_pool_active(&self) -> bool {
        self.private_small_pool.is_some()
    }

    fn get_pool(&mut self, size: usize) -> &mut BlockPool {
        self.get_pool_by_flag(size <= K_SMALL_SIZE)
    }

    fn get_pool_by_flag(&mut self, is_small: bool) -> &mut BlockPool {
        if is_small {
            if let Some(ref mut pp) = self.private_small_pool {
                pp
            } else {
                &mut self.small_pool
            }
        } else if let Some(ref mut pp) = self.private_large_pool {
            pp
        } else {
            &mut self.large_pool
        }
    }

    /// Round size to minimum block size.
    fn round_size(size: usize) -> usize {
        if size < K_MIN_BLOCK_SIZE {
            K_MIN_BLOCK_SIZE
        } else {
            K_MIN_BLOCK_SIZE * size.div_ceil(K_MIN_BLOCK_SIZE)
        }
    }

    /// Determine cudaMalloc size for a given request.
    fn get_allocation_size(size: usize) -> usize {
        if size <= K_SMALL_SIZE {
            K_SMALL_BUFFER
        } else if size < K_MIN_LARGE_ALLOC {
            // Default large segment: 20 MB (matching PyTorch's default)
            20 * 1024 * 1024
        } else {
            K_ROUND_LARGE * size.div_ceil(K_ROUND_LARGE)
        }
    }

    fn should_split(block: &Block, size: usize) -> bool {
        let remaining = block.size - size;
        if block.pool_is_small {
            remaining >= K_MIN_BLOCK_SIZE
        } else {
            remaining > K_SMALL_SIZE
        }
    }

    /// Allocate GPU memory.
    pub fn alloc(&mut self, orig_size: usize) -> *mut u8 {
        let size = Self::round_size(orig_size);

        // 1. Try to find a free block in the pool.
        let pool = self.get_pool(size);
        let is_small = pool.is_small;
        if let Some(block_ptr) = pool.find_best_fit(size) {
            let block = unsafe { &mut *block_ptr };
            pool.remove(block_ptr);

            // Split if remainder is large enough.
            if Self::should_split(block, size) {
                let remaining_size = block.size - size;
                let remaining_ptr = unsafe { block.ptr.add(size) };

                let remaining = Box::into_raw(Box::new(Block {
                    ptr: remaining_ptr,
                    size: remaining_size,
                    allocated: false,
                    prev: block_ptr,
                    next: block.next,
                    pool_is_small: is_small,
                }));
                self.all_blocks.push(remaining);

                if !block.next.is_null() {
                    unsafe { (*block.next).prev = remaining };
                }
                block.next = remaining;
                block.size = size;

                // Remainder stays in the same pool as the parent block.
                self.get_pool_by_flag(is_small).insert(remaining);
            }

            block.allocated = true;
            self.active_blocks.insert(block.ptr as usize, block_ptr);
            self.active_bytes += size;
            if self.active_bytes > self.peak_active_bytes {
                self.peak_active_bytes = self.active_bytes;
            }
            return block.ptr;
        }

        // 2. Also try the main pools if using private pool.
        if self.private_small_pool.is_some() {
            let main_pool = if size <= K_SMALL_SIZE {
                &mut self.small_pool
            } else {
                &mut self.large_pool
            };
            if let Some(block_ptr) = main_pool.find_best_fit(size) {
                let block = unsafe { &mut *block_ptr };
                main_pool.remove(block_ptr);

                if Self::should_split(block, size) {
                    let remaining_size = block.size - size;
                    let remaining_ptr = unsafe { block.ptr.add(size) };

                    let remaining = Box::into_raw(Box::new(Block {
                        ptr: remaining_ptr,
                        size: remaining_size,
                        allocated: false,
                        prev: block_ptr,
                        next: block.next,
                        pool_is_small: is_small,
                    }));
                    self.all_blocks.push(remaining);

                    if !block.next.is_null() {
                        unsafe { (*block.next).prev = remaining };
                    }
                    block.next = remaining;
                    block.size = size;

                    // Remainder stays in the same pool as the parent block.
                    self.get_pool_by_flag(is_small).insert(remaining);
                }

                block.allocated = true;
                self.active_blocks.insert(block.ptr as usize, block_ptr);
                self.active_bytes += size;
                if self.active_bytes > self.peak_active_bytes {
                    self.peak_active_bytes = self.active_bytes;
                }
                return block.ptr;
            }
        }

        // 3. Allocate a new segment from the CUDA driver.
        //
        // During CUDA graph capture, cudaMalloc is only allowed in relaxed
        // capture mode. PyTorch does the same: CUDAStreamCaptureModeGuard
        // switches to relaxed before allocating (CUDACachingAllocator.cpp:1177).
        // The allocated VA is captured into the graph but the allocation itself
        // is replayed as a no-op — safe as long as we never cudaFree before replay.
        let alloc_size = Self::get_allocation_size(size);
        tracing::debug!(
            "CachingAllocator: cudaMalloc {alloc_size} bytes for request of {size} bytes \
             (small_free={}, large_free={}, private={}, segments={})",
            self.small_pool.free_blocks.len(),
            self.large_pool.free_blocks.len(),
            self.private_small_pool.is_some(),
            self.segments.len(),
        );
        let segment_ptr = unsafe {
            let _guard = RelaxedCaptureModeGuard::new();
            driver::mem_alloc(alloc_size)
        }
        .expect("CachingAllocator: GPU OOM");
        self.segments.push((segment_ptr, alloc_size));

        let block = Box::into_raw(Box::new(Block {
            ptr: segment_ptr,
            size: alloc_size,
            allocated: false,
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
            pool_is_small: is_small,
        }));
        self.all_blocks.push(block);

        // Split the segment block if needed.
        let b = unsafe { &mut *block };
        if Self::should_split(b, size) {
            let remaining_size = b.size - size;
            let remaining_ptr = unsafe { b.ptr.add(size) };

            let remaining = Box::into_raw(Box::new(Block {
                ptr: remaining_ptr,
                size: remaining_size,
                allocated: false,
                prev: block,
                next: ptr::null_mut(),
                pool_is_small: is_small,
            }));
            self.all_blocks.push(remaining);

            b.next = remaining;
            b.size = size;

            self.get_pool_by_flag(is_small).insert(remaining);
        }

        b.allocated = true;
        self.active_blocks.insert(b.ptr as usize, block);
        self.active_bytes += size;
        if self.active_bytes > self.peak_active_bytes {
            self.peak_active_bytes = self.active_bytes;
        }
        b.ptr
    }

    /// Free GPU memory (returns block to free pool, coalesces with neighbors).
    pub unsafe fn free(&mut self, ptr: *mut u8, _size_bytes: usize) {
        let Some(block_ptr) = self.active_blocks.remove(&(ptr as usize)) else {
            return; // not tracked (e.g., persistent allocation)
        };

        let block = &mut *block_ptr;
        block.allocated = false;
        self.active_bytes = self.active_bytes.saturating_sub(block.size);

        // Try merge with prev.
        if !block.prev.is_null() {
            let prev = &mut *block.prev;
            if !prev.allocated {
                // Remove prev from its free pool.
                self.get_pool_by_flag(prev.pool_is_small).remove(block.prev);

                // Merge: prev absorbs block.
                prev.size += block.size;
                prev.next = block.next;
                if !block.next.is_null() {
                    (*block.next).prev = block.prev;
                }

                // block is now dead; continue with prev as the merged block.
                // (We don't delete block — it stays in all_blocks for cleanup.)
                let merged = block.prev;
                // Try merge merged with next.
                let merged_block = &mut *merged;
                if !merged_block.next.is_null() {
                    let next = &mut *merged_block.next;
                    if !next.allocated {
                        self.get_pool_by_flag(next.pool_is_small)
                            .remove(merged_block.next);

                        merged_block.size += next.size;
                        merged_block.next = next.next;
                        if !next.next.is_null() {
                            (*next.next).prev = merged;
                        }
                    }
                }

                self.get_pool_by_flag(merged_block.pool_is_small)
                    .insert(merged);
                return;
            }
        }

        // Try merge with next.
        if !block.next.is_null() {
            let next = &mut *block.next;
            if !next.allocated {
                self.get_pool_by_flag(next.pool_is_small).remove(block.next);

                block.size += next.size;
                block.next = next.next;
                if !next.next.is_null() {
                    (*next.next).prev = block_ptr;
                }
            }
        }

        // Insert into free pool.
        self.get_pool_by_flag(block.pool_is_small).insert(block_ptr);
    }

    /// Allocate a GPU tensor (leaked — not auto-freed on drop).
    pub fn alloc_gpu_tensor(&mut self, shape: &[usize], dtype: DType) -> GpuTensor {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * dtype.size_bytes();
        let ptr = self.alloc(size_bytes);
        unsafe { GpuTensor::new(ptr, shape, dtype) }
    }

    /// Allocate an owned tensor (freed on drop).
    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> OwnedTensor {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * dtype.size_bytes();
        let ptr = self.alloc(size_bytes);
        let inner = unsafe { GpuTensor::new(ptr, shape, dtype) };
        // Capture the allocator pointer in the neutral OwnedTensor's lifecycle
        // thunks: `free` returns the block on Drop; `detach` unregisters it
        // (without freeing) for `into_gpu_tensor`.
        let alloc_ptr = self as *mut CachingAllocator;
        unsafe {
            OwnedTensor::from_parts(
                inner,
                size_bytes,
                Some(Box::new(move |ptr, sz| (*alloc_ptr).free(ptr, sz))),
                Some(Box::new(move |ptr| {
                    (*alloc_ptr).unregister_active_block(ptr as usize)
                })),
            )
        }
    }

    /// Remove a pointer from the active-blocks tracker without
    /// freeing it. Used by [`OwnedTensor::into_gpu_tensor`] when the
    /// caller wants to leak the GPU memory and detach it from the
    /// caching allocator's bookkeeping. Pub-crate-scoped because
    /// `OwnedTensor` lives in a sibling module.
    pub(crate) fn unregister_active_block(&mut self, ptr: usize) {
        self.active_blocks.remove(&ptr);
    }

    pub fn free_block_count(&self) -> usize {
        self.small_pool.free_blocks.len() + self.large_pool.free_blocks.len()
    }

    /// Total bytes currently held from the CUDA driver (allocated + free pool).
    /// Mirrors PyTorch's `reserved_bytes.all.current`.
    pub fn memory_reserved(&self) -> usize {
        self.segments.iter().map(|&(_, sz)| sz).sum()
    }

    pub fn total_block_count(&self) -> usize {
        self.all_blocks.len()
    }

    /// Release all GPU memory and reset the allocator to its initial state.
    ///
    /// Frees every segment back to the CUDA driver, deallocates all Block
    /// structs, and clears private pools. After this call the allocator is
    /// empty — equivalent to a freshly constructed `CachingAllocator::new()`.
    ///
    /// Used during sleep/wake cycles where all GPU resources are torn down
    /// and will be re-created from scratch on wake.
    ///
    /// # Safety
    /// All pointers previously returned by `alloc()` become invalid.
    /// Callers must ensure no live references to allocated memory remain.
    pub unsafe fn release_all(&mut self) {
        // Free all segments back to the CUDA driver.
        for &(ptr, _) in &self.segments {
            let _ = driver::mem_free(ptr);
        }
        // Free all Block structs.
        for &bp in &self.all_blocks {
            let _ = Box::from_raw(bp);
        }
        // Reset to initial state.
        self.small_pool = BlockPool::new(true);
        self.large_pool = BlockPool::new(false);
        self.segments.clear();
        self.all_blocks.clear();
        self.active_blocks.clear();
        self.private_small_pool = None;
        self.private_large_pool = None;
        self.active_bytes = 0;
        self.peak_active_bytes = 0;
    }

    /// Release all free, unsplit segments back to the CUDA driver.
    /// Matches PyTorch's `torch.cuda.empty_cache()` / `release_cached_blocks()`.
    pub fn trim(&mut self) {
        let mut freed_bytes: usize = 0;
        let mut freed_count: usize = 0;

        // First pass: identify releasable segments (collect to avoid borrow conflict).
        // Each entry: (segment_index, head_block_ptr, seg_ptr, seg_size, is_small).
        let mut releasable: Vec<(usize, *mut Block, *mut u8, usize, bool)> = Vec::new();

        for (seg_idx, &(seg_ptr, seg_size)) in self.segments.iter().enumerate() {
            let head = self.all_blocks.iter().find(|&&bp| {
                let b = unsafe { &*bp };
                b.ptr == seg_ptr && b.prev.is_null()
            });

            let Some(&head_ptr) = head else { continue };
            let head_block = unsafe { &*head_ptr };

            if head_block.next.is_null() && !head_block.allocated && head_block.size == seg_size {
                releasable.push((
                    seg_idx,
                    head_ptr,
                    seg_ptr,
                    seg_size,
                    head_block.pool_is_small,
                ));
            }
        }

        // Second pass: release them.
        let mut to_remove: Vec<usize> = Vec::with_capacity(releasable.len());
        for (seg_idx, head_ptr, seg_ptr, seg_size, is_small) in releasable {
            self.get_pool_by_flag(is_small).remove(head_ptr);
            unsafe {
                let _ = driver::mem_free(seg_ptr);
            }
            freed_bytes += seg_size;
            freed_count += 1;
            to_remove.push(seg_idx);
        }

        // Remove segments in reverse order to keep indices valid.
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            self.segments.swap_remove(idx);
        }

        if freed_count > 0 {
            tracing::debug!(
                "CachingAllocator::trim: released {freed_count} segments, \
                 {:.1} MiB back to CUDA driver",
                freed_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
}

impl Drop for CachingAllocator {
    fn drop(&mut self) {
        // Free all segments.
        for &(ptr, _) in &self.segments {
            unsafe {
                let _ = driver::mem_free(ptr);
            }
        }
        // Free all Block structs.
        for &bp in &self.all_blocks {
            unsafe {
                let _ = Box::from_raw(bp);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RawGpuAlloc — RAII wrapper for raw driver::mem_alloc() GPU memory
// ---------------------------------------------------------------------------

/// RAII wrapper for GPU memory allocated directly via `driver::mem_alloc()`.
///
/// Unlike `OwnedTensor` (which returns memory to the caching allocator),
/// `RawGpuAlloc` calls `driver::mem_free()` on drop, matching the raw
/// allocation. Used for persistent buffers (e.g. PP recv buffers) that
/// are not managed by the caching allocator.
pub struct RawGpuAlloc {
    inner: GpuTensor,
}

// Safety: GPU device pointers are accessible from any host thread after the
// CUDA context is established. The GpuTensor inside is a plain pointer+metadata
// with no thread-local state.
unsafe impl Send for RawGpuAlloc {}
unsafe impl Sync for RawGpuAlloc {}

impl RawGpuAlloc {
    /// Allocate raw GPU memory and wrap it in a `GpuTensor` with the given shape/dtype.
    ///
    /// # Safety
    /// Requires an active CUDA context on the current thread.
    pub unsafe fn new(shape: &[usize], dtype: DType) -> anyhow::Result<Self> {
        let numel: usize = shape.iter().product();
        let ptr = driver::mem_alloc(numel * dtype.size_bytes())?;
        let inner = GpuTensor::new(ptr, shape, dtype);
        Ok(Self { inner })
    }

    /// Access the underlying `GpuTensor` (non-owning, Copy).
    pub fn as_gpu_tensor(&self) -> GpuTensor {
        self.inner
    }

    /// Borrow as a lifetime-checked `TensorView`.
    pub fn view(&self) -> TensorView<'_> {
        // Safety: self owns the GPU memory; the view borrows &self.
        unsafe { TensorView::from_raw(self.inner) }
    }
}

impl Drop for RawGpuAlloc {
    fn drop(&mut self) {
        unsafe {
            let _ = driver::mem_free(self.inner.raw_ptr());
        }
    }
}

impl std::ops::Deref for RawGpuAlloc {
    type Target = GpuTensor;
    fn deref(&self) -> &GpuTensor {
        &self.inner
    }
}

// `OwnedTensor` and `RawGpuMem` are backend-neutral types from
// `scratchy-tensors`, re-exported from the crate root.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_size() {
        assert_eq!(CachingAllocator::round_size(0), K_MIN_BLOCK_SIZE);
        assert_eq!(CachingAllocator::round_size(1), K_MIN_BLOCK_SIZE);
        assert_eq!(CachingAllocator::round_size(512), 512);
        assert_eq!(CachingAllocator::round_size(513), 1024);
        assert_eq!(CachingAllocator::round_size(1024), 1024);
    }

    #[test]
    fn test_get_allocation_size() {
        // Small: always 2 MB segment.
        assert_eq!(CachingAllocator::get_allocation_size(256), K_SMALL_BUFFER);
        assert_eq!(
            CachingAllocator::get_allocation_size(K_SMALL_SIZE),
            K_SMALL_BUFFER
        );
        // Medium: 20 MB segment.
        assert_eq!(
            CachingAllocator::get_allocation_size(K_SMALL_SIZE + 1),
            20 * 1024 * 1024
        );
        // Large: rounded to 2 MB.
        assert_eq!(
            CachingAllocator::get_allocation_size(K_MIN_LARGE_ALLOC),
            K_MIN_LARGE_ALLOC
        );
        assert_eq!(
            CachingAllocator::get_allocation_size(K_MIN_LARGE_ALLOC + 1),
            12 * 1024 * 1024
        );
    }

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;

        fn init_cuda() {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
            }
        }

        #[test]
        fn test_alloc_and_free() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            let ptr1 = alloc.alloc(1024);
            assert!(!ptr1.is_null());

            unsafe { alloc.free(ptr1, 1024) };

            // Re-alloc same size should reuse.
            let ptr2 = alloc.alloc(1024);
            assert_eq!(ptr1, ptr2);

            unsafe { alloc.free(ptr2, 1024) };
        }

        #[test]
        fn test_block_splitting() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            // Alloc 1 KB from a 2 MB segment — should split.
            let ptr1 = alloc.alloc(1024);
            // Free it — block returns to pool.
            unsafe { alloc.free(ptr1, 1024) };

            // Alloc a different size (512 bytes) — should reuse the same block
            // (split from the same segment).
            let ptr2 = alloc.alloc(512);
            assert_eq!(ptr1, ptr2); // same start address (best fit = first block)

            unsafe { alloc.free(ptr2, 512) };

            // Only 1 segment should have been allocated.
            assert_eq!(alloc.segments.len(), 1);
        }

        #[test]
        fn test_block_coalescing() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            // Alloc two blocks from the same segment.
            let ptr1 = alloc.alloc(1024);
            let ptr2 = alloc.alloc(1024);
            assert_ne!(ptr1, ptr2);

            // Free both — they should coalesce.
            unsafe {
                alloc.free(ptr1, 1024);
                alloc.free(ptr2, 1024);
            }

            // One segment, and the coalesced free block should be large enough
            // for a bigger allocation without a new segment.
            let ptr3 = alloc.alloc(2048);
            assert_eq!(ptr3, ptr1); // reuses the coalesced block
            assert_eq!(alloc.segments.len(), 1);

            unsafe { alloc.free(ptr3, 2048) };
        }

        #[test]
        fn test_owned_tensor() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            let ptr;
            {
                let t = alloc.alloc_tensor(&[32, 4096], DType::BF16);
                ptr = t.as_gpu_tensor().raw_ptr();
                assert_eq!(t.dim(0), 32);
                assert_eq!(t.dim(1), 4096);
            }
            // Block freed on drop — should be reusable.
            let t2 = alloc.alloc_tensor(&[32, 4096], DType::BF16);
            assert_eq!(t2.as_gpu_tensor().raw_ptr(), ptr);
        }

        #[test]
        fn test_into_gpu_tensor_no_free() {
            init_cuda();
            let mut alloc = CachingAllocator::new();
            let t = alloc.alloc_tensor(&[16, 128], DType::F32);
            let free_before = alloc.free_block_count(); // remainder from segment split
            let gpu = t.into_gpu_tensor();
            // Block is leaked: not in active_blocks, not in free pool.
            // free_block_count unchanged (remainder is still there, leaked block is not freed).
            assert_eq!(alloc.free_block_count(), free_before);
            assert_eq!(alloc.active_blocks.len(), 0); // removed by into_gpu_tensor
            let _ = gpu;
        }
        #[test]
        fn test_private_pool_blocks_not_leaked() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            // Simulate CUDA graph capture: allocate to private pool.
            alloc.begin_allocate_to_pool();

            // Allocate some blocks during "graph capture".
            let p1 = alloc.alloc(1024);
            let p2 = alloc.alloc(2048);
            assert!(!p1.is_null());
            assert!(!p2.is_null());

            // Track how many segments were allocated.
            let segments_after_capture = alloc.segments.len();
            assert!(segments_after_capture > 0);

            // End pool allocation (simulating end of graph capture).
            alloc.end_allocate_to_pool();

            // BUG: Without the fix, the private pool blocks are lost (set to None).
            // They're no longer tracked in any pool, causing a memory leak.
            // The segments remain allocated but blocks can't be reused.

            // Try to allocate again - this should reuse blocks from the private pool
            // if they're still tracked, or allocate new segments if they were leaked.
            let segments_before_realloc = alloc.segments.len();

            // Allocate same sizes again - should reuse if private pool is kept.
            let p3 = alloc.alloc(1024);
            let p4 = alloc.alloc(2048);

            // With the fix: no new segments needed (reuses private pool blocks).
            // Without the fix: new segments allocated (private pool was discarded).
            assert_eq!(
                alloc.segments.len(),
                segments_before_realloc,
                "Private pool blocks should be reusable after end_allocate_to_pool()"
            );

            // Clean up.
            unsafe {
                alloc.free(p1, 1024);
                alloc.free(p2, 2048);
                alloc.free(p3, 1024);
                alloc.free(p4, 2048);
            }
        }

        #[test]
        fn test_different_sizes_share_segment() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            // Multiple small allocations should share one 2 MB segment.
            let p1 = alloc.alloc(512);
            let p2 = alloc.alloc(1024);
            let p3 = alloc.alloc(2048);

            assert_eq!(alloc.segments.len(), 1); // all from one 2 MB segment

            unsafe {
                alloc.free(p1, 512);
                alloc.free(p2, 1024);
                alloc.free(p3, 2048);
            }
        }

        #[test]
        fn test_trim_releases_free_segments() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            // Allocate from two different segments (small + large).
            let p1 = alloc.alloc(512); // small pool → 2 MB segment
            let p2 = alloc.alloc(2 * 1024 * 1024); // large pool → 20 MB segment
            assert_eq!(alloc.segments.len(), 2);

            // Free both — blocks coalesce back to full segments.
            unsafe {
                alloc.free(p1, 512);
                alloc.free(p2, 2 * 1024 * 1024);
            }

            // trim() should release both segments.
            alloc.trim();
            assert_eq!(alloc.segments.len(), 0);
            assert_eq!(alloc.free_block_count(), 0);
        }

        #[test]
        fn test_trim_keeps_split_segments() {
            init_cuda();
            let mut alloc = CachingAllocator::new();

            // Allocate two blocks from one segment.
            let p1 = alloc.alloc(512);
            let _p2 = alloc.alloc(1024);
            assert_eq!(alloc.segments.len(), 1);

            // Free only the first — segment is still split (p2 allocated).
            unsafe { alloc.free(p1, 512) };

            // trim() should NOT release (segment is split).
            alloc.trim();
            assert_eq!(alloc.segments.len(), 1);
        }
    }
}
