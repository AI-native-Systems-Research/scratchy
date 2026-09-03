// SPDX-License-Identifier: Apache-2.0
//! Scratch arena: bump-allocated GPU memory, reset per engine step.
//!
//! All intermediate activations live here. Allocation is O(1) — a pointer
//! bump with 256-byte alignment. Zero driver API calls on the hot path.
//!
//! During warmup, the arena grows as needed. After warmup, `lock()` freezes
//! the capacity — any allocation exceeding it is a fatal error (same as
//! Python vLLM's WorkspaceManager).

use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::Result;

/// Alignment for all arena allocations (256 bytes covers all CUDA requirements).
const ALIGNMENT: usize = 256;

/// A single contiguous GPU memory segment.
struct Segment {
    base: *mut u8,
    capacity: usize,
}

impl Segment {
    unsafe fn new(capacity: usize) -> Result<Self> {
        let base = driver::mem_alloc(capacity)?;
        Ok(Self { base, capacity })
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        if !self.base.is_null() {
            unsafe {
                let _ = driver::mem_free(self.base);
            }
        }
    }
}

/// Bump-allocated GPU scratch memory.
///
/// Uses a primary segment for fast-path allocation. When the primary overflows
/// during warmup, a new larger segment is allocated and becomes primary.
/// The old segment is kept alive in `retired` so existing `GpuTensor` pointers
/// remain valid until `reset()`. On `reset()`, retired segments are freed and
/// the primary is kept at its grown capacity.
pub struct ScratchArena {
    primary: Segment,
    offset: usize,
    high_water: usize,
    locked: bool,
    /// Old segments kept alive until reset() so pointers stay valid.
    retired: Vec<Segment>,
    /// Debug: generation counter, incremented on each reset.
    #[cfg(debug_assertions)]
    generation: u64,
}

// Safety: GPU device pointers are valid from any host thread.
unsafe impl Send for ScratchArena {}
unsafe impl Sync for ScratchArena {}

impl ScratchArena {
    /// Create a new arena with the given initial capacity (bytes).
    ///
    /// # Safety
    /// Must be called while a CUDA context is active on this thread.
    pub unsafe fn new(initial_capacity: usize) -> Result<Self> {
        let capacity = align_up(initial_capacity);
        let primary = if capacity > 0 {
            Segment::new(capacity)?
        } else {
            Segment {
                base: std::ptr::null_mut(),
                capacity: 0,
            }
        };
        Ok(Self {
            primary,
            offset: 0,
            high_water: 0,
            locked: false,
            retired: Vec::new(),
            #[cfg(debug_assertions)]
            generation: 0,
        })
    }

    /// Allocate a tensor from the arena. O(1), zero driver calls.
    ///
    /// # Panics
    /// Panics if the arena is locked and the allocation exceeds capacity.
    pub fn alloc(&mut self, shape: &[usize], dtype: DType) -> GpuTensor {
        let elem_bytes: usize = shape.iter().product::<usize>() * dtype.size_bytes();
        let aligned = align_up(elem_bytes);

        let new_offset = self.offset + aligned;
        if new_offset > self.primary.capacity {
            if self.locked {
                panic!(
                    "ScratchArena overflow after lock! Need {} bytes at offset {}, capacity {}",
                    aligned, self.offset, self.primary.capacity
                );
            }
            // During warmup: grow. Old segment is retired (pointers stay valid).
            unsafe {
                self.grow(new_offset);
            }
        }

        let ptr = unsafe { self.primary.base.add(self.offset) };
        self.offset += aligned;
        if self.offset > self.high_water {
            self.high_water = self.offset;
        }

        unsafe { GpuTensor::new(ptr, shape, dtype) }
    }

    /// Reset the arena — all prior tensors are invalidated.
    /// Called once per engine step, after sampling completes.
    pub fn reset(&mut self) {
        self.offset = 0;
        // Free retired segments — their tensors are no longer referenced.
        self.retired.clear();
        #[cfg(debug_assertions)]
        {
            self.generation += 1;
        }
    }

    /// Lock the arena: any future alloc exceeding capacity panics.
    /// Call after warmup to guarantee zero-allocation inference.
    pub fn lock(&mut self) {
        self.locked = true;
        // Free any retired segments before locking.
        self.retired.clear();
        tracing::info!(
            "ScratchArena locked: capacity={} bytes ({:.1} MB), high_water={} bytes ({:.1} MB)",
            self.primary.capacity,
            self.primary.capacity as f64 / (1024.0 * 1024.0),
            self.high_water,
            self.high_water as f64 / (1024.0 * 1024.0),
        );
    }

    /// Whether the arena is locked.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Current byte offset (amount used).
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.primary.capacity
    }

    /// High-water mark: maximum bytes used in any step.
    pub fn high_water_mark(&self) -> usize {
        self.high_water
    }

    /// Advance the bump offset to `offset` bytes.
    ///
    /// Used after CUDA graph replay to reserve the arena region that the
    /// graph's captured kernels wrote to, preventing subsequent allocations
    /// (e.g. sampling) from overlapping with graph outputs.
    ///
    /// # Panics
    /// Panics if `offset` exceeds capacity.
    pub fn set_offset(&mut self, offset: usize) {
        assert!(
            offset <= self.primary.capacity,
            "set_offset({offset}) exceeds arena capacity ({})",
            self.primary.capacity
        );
        self.offset = offset;
        if offset > self.high_water {
            self.high_water = offset;
        }
    }

    /// Grow the arena to at least `min_capacity` bytes.
    ///
    /// The old segment is retired (kept alive so existing GpuTensor pointers
    /// remain valid). A new, larger segment becomes primary and the bump
    /// offset resets to 0. The caller's allocation will be the first in the
    /// new segment.
    ///
    /// # Safety
    /// Must be called while a CUDA context is active.
    unsafe fn grow(&mut self, min_capacity: usize) {
        let new_capacity = align_up(min_capacity.max(self.primary.capacity * 2).max(1024 * 1024));
        tracing::warn!(
            "ScratchArena growing: {} -> {} bytes ({:.1} MB)",
            self.primary.capacity,
            new_capacity,
            new_capacity as f64 / (1024.0 * 1024.0),
        );

        let new_seg = Segment::new(new_capacity).expect("ScratchArena: GPU OOM during grow");

        // Retire old segment — its pointers stay valid until reset().
        let old = std::mem::replace(&mut self.primary, new_seg);
        if !old.base.is_null() {
            self.retired.push(old);
        }

        // Reset offset — new allocations start from the new segment's base.
        self.offset = 0;
    }
}

/// Round up to 256-byte alignment.
fn align_up(size: usize) -> usize {
    (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Alignment tests (no GPU needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_align_up_zero() {
        assert_eq!(align_up(0), 0);
    }

    #[test]
    fn test_align_up_one() {
        assert_eq!(align_up(1), 256);
    }

    #[test]
    fn test_align_up_exact() {
        assert_eq!(align_up(256), 256);
        assert_eq!(align_up(512), 512);
        assert_eq!(align_up(1024), 1024);
    }

    #[test]
    fn test_align_up_not_exact() {
        assert_eq!(align_up(257), 512);
        assert_eq!(align_up(300), 512);
        assert_eq!(align_up(513), 768);
    }

    #[test]
    fn test_align_up_large() {
        assert_eq!(align_up(1_000_000), 1_000_192); // next multiple of 256
        assert_eq!(align_up(1024 * 1024), 1024 * 1024); // already aligned
    }

    #[test]
    fn test_align_up_power_of_two() {
        for p in 0..20 {
            let size = 1usize << p;
            let aligned = align_up(size);
            assert!(aligned >= size);
            assert_eq!(aligned % ALIGNMENT, 0);
        }
    }

    // -----------------------------------------------------------------------
    // GPU arena tests (require CUDA)
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;

        /// Helper: initialize CUDA context for test.
        fn init_cuda() {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
            }
        }

        #[test]
        fn test_arena_create_and_drop() {
            init_cuda();
            let arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };
            assert_eq!(arena.capacity(), 1024 * 1024);
            assert_eq!(arena.used(), 0);
            assert!(!arena.is_locked());
            drop(arena); // should not panic
        }

        #[test]
        fn test_arena_alloc_basic() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let t = arena.alloc(&[4, 128], DType::F16);
            assert_eq!(t.numel(), 512);
            assert_eq!(t.dtype(), DType::F16);
            assert!(!t.is_null());
            assert!(arena.used() > 0);
        }

        #[test]
        fn test_arena_alloc_sequential_non_overlapping() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let t1 = arena.alloc(&[4, 128], DType::F16); // 1024 bytes → aligned 1024
            let t2 = arena.alloc(&[4, 128], DType::F16); // another 1024

            // Tensors should NOT overlap.
            let p1 = t1.raw_ptr() as usize;
            let p2 = t2.raw_ptr() as usize;
            assert!(p2 >= p1 + t1.size_bytes(), "allocations overlap");
        }

        #[test]
        fn test_arena_alloc_alignment() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            // Allocate something small (e.g. 10 bytes) — should still be 256-aligned.
            let t = arena.alloc(&[5], DType::F16); // 10 bytes
            assert_eq!(t.raw_ptr() as usize % ALIGNMENT, 0);

            let t2 = arena.alloc(&[1], DType::F32); // 4 bytes
            assert_eq!(t2.raw_ptr() as usize % ALIGNMENT, 0);
        }

        #[test]
        fn test_arena_reset() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let t1 = arena.alloc(&[4, 128], DType::F16);
            let used_before = arena.used();
            assert!(used_before > 0);

            arena.reset();
            assert_eq!(arena.used(), 0);
            assert_eq!(arena.high_water_mark(), used_before);

            // New alloc should start from the same base address.
            let t2 = arena.alloc(&[4, 128], DType::F16);
            assert_eq!(t1.raw_ptr(), t2.raw_ptr());
        }

        #[test]
        fn test_arena_high_water_mark() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            // Step 1: allocate 1 KB
            arena.alloc(&[256], DType::F32); // 1024 bytes
            let hw1 = arena.high_water_mark();
            arena.reset();

            // Step 2: allocate 2 KB
            arena.alloc(&[512], DType::F32); // 2048 bytes
            let hw2 = arena.high_water_mark();
            assert!(hw2 > hw1, "high water should increase");
            arena.reset();

            // Step 3: allocate 512 bytes (less than previous)
            arena.alloc(&[128], DType::F32); // 512 bytes
            let hw3 = arena.high_water_mark();
            assert_eq!(hw3, hw2, "high water should not decrease");
        }

        #[test]
        fn test_arena_lock_within_capacity() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            // Warmup: allocate to establish pattern.
            arena.alloc(&[128, 128], DType::F16); // 32 KB
            arena.reset();

            arena.lock();
            assert!(arena.is_locked());

            // Alloc within capacity should succeed.
            let t = arena.alloc(&[128, 128], DType::F16);
            assert!(!t.is_null());
        }

        #[test]
        #[should_panic(expected = "ScratchArena overflow after lock")]
        fn test_arena_lock_overflow_panics() {
            init_cuda();
            // Create a very small arena.
            let mut arena = unsafe { ScratchArena::new(512).unwrap() };
            arena.lock();

            // Try to allocate more than capacity — should panic.
            arena.alloc(&[1024], DType::F32); // 4096 bytes > 512
        }

        #[test]
        fn test_arena_grow_during_warmup() {
            init_cuda();
            // Start tiny — force growth.
            let mut arena = unsafe { ScratchArena::new(256).unwrap() };
            let initial_cap = arena.capacity();

            // Allocate more than initial capacity.
            arena.alloc(&[1024], DType::F32); // 4096 bytes > 256

            assert!(
                arena.capacity() > initial_cap,
                "arena should have grown: {} -> {}",
                initial_cap,
                arena.capacity()
            );
        }

        #[test]
        fn test_arena_multiple_allocs_in_step() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(4 * 1024 * 1024).unwrap() };

            // Simulate a model forward: multiple allocs per step.
            let tensors: Vec<GpuTensor> = (0..10)
                .map(|_| arena.alloc(&[32, 4096], DType::BF16)) // 256 KB each
                .collect();

            // All should be non-null and non-overlapping.
            for (i, t) in tensors.iter().enumerate() {
                assert!(!t.is_null(), "tensor {} is null", i);
                for (j, tj) in tensors.iter().enumerate().skip(i + 1) {
                    let pi = t.raw_ptr() as usize;
                    let pj = tj.raw_ptr() as usize;
                    assert_ne!(pi, pj, "tensors {} and {} overlap", i, j);
                }
            }
        }

        #[test]
        fn test_arena_reset_reuses_memory() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let t1_ptr = arena.alloc(&[64], DType::F32).raw_ptr() as usize;
            arena.reset();
            let t2_ptr = arena.alloc(&[64], DType::F32).raw_ptr() as usize;

            assert_eq!(t1_ptr, t2_ptr, "reset should reuse the same memory");
        }

        #[test]
        fn test_arena_zero_capacity() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(0).unwrap() };
            assert_eq!(arena.capacity(), 0);

            // Should grow on first alloc.
            let t = arena.alloc(&[4], DType::F32);
            assert!(!t.is_null());
            assert!(arena.capacity() > 0);
        }

        // -------------------------------------------------------------------
        // Per-layer scoping tests
        // -------------------------------------------------------------------

        #[test]
        fn test_set_offset_reclaims_memory() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            // Allocate a persistent tensor.
            let persistent = arena.alloc(&[64], DType::F32); // 256 bytes
            let saved = arena.used();

            // Allocate scratch (simulating layer intermediates).
            arena.alloc(&[1024], DType::F32); // 4096 bytes
            arena.alloc(&[1024], DType::F32); // 4096 bytes
            let peak = arena.used();
            assert!(peak > saved, "scratch should grow offset");

            // Restore offset — reclaim scratch.
            arena.set_offset(saved);
            assert_eq!(arena.used(), saved);

            // Persistent tensor is still at the same address.
            let next = arena.alloc(&[64], DType::F32);
            // The new alloc should start right after the persistent tensor,
            // reusing the reclaimed scratch space.
            assert_eq!(
                next.raw_ptr() as usize,
                persistent.raw_ptr() as usize + align_up(64 * 4)
            );
        }

        #[test]
        fn test_set_offset_high_water_preserved() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let saved = arena.used();

            // Allocate large scratch.
            arena.alloc(&[2048], DType::F32); // 8192 bytes
            let hw = arena.high_water_mark();

            // Restore offset.
            arena.set_offset(saved);
            assert_eq!(arena.used(), saved);
            // High water mark should NOT decrease.
            assert_eq!(arena.high_water_mark(), hw);
        }

        #[test]
        fn test_scoped_layer_pattern() {
            // Simulates the per-layer arena scoping pattern used in model forward.
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(4 * 1024 * 1024).unwrap() };

            // Pre-allocate persistent buffers (like hs_buf + res_buf).
            let hs_buf = arena.alloc(&[32, 128], DType::BF16); // 8 KB
            let res_buf = arena.alloc(&[32, 128], DType::BF16); // 8 KB
            let layer_scratch_base = arena.used();

            let hs_ptr = hs_buf.raw_ptr() as usize;
            let res_ptr = res_buf.raw_ptr() as usize;

            // Simulate 10 layers, each allocating ~200 KB of scratch.
            for _ in 0..10 {
                // Simulate layer intermediates.
                arena.alloc(&[32, 128], DType::BF16); // normed
                arena.alloc(&[32, 384], DType::BF16); // qkv
                arena.alloc(&[32, 128], DType::BF16); // attn_output
                arena.alloc(&[32, 256], DType::BF16); // gate_up
                let _mlp_out = arena.alloc(&[32, 128], DType::BF16); // mlp_output

                // Reclaim scratch.
                arena.set_offset(layer_scratch_base);
            }

            // After all layers, offset should be right after persistent buffers.
            assert_eq!(arena.used(), layer_scratch_base);

            // Persistent buffers should still be at the same addresses.
            // (Re-alloc at the same offset to verify.)
            let new_hs = arena.alloc(&[32, 128], DType::BF16);
            // This new alloc starts at layer_scratch_base, not at hs_buf's address.
            // The persistent buffers are before layer_scratch_base.
            assert!(new_hs.raw_ptr() as usize >= layer_scratch_base);
            // Original buffers are untouched.
            assert_eq!(hs_buf.raw_ptr() as usize, hs_ptr);
            assert_eq!(res_buf.raw_ptr() as usize, res_ptr);
        }

        #[test]
        fn test_scoped_layer_memory_bounded() {
            // Verify that per-layer scoping keeps peak memory proportional to
            // 1 layer's scratch, not N layers combined.
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(4 * 1024 * 1024).unwrap() };

            let persistent = arena.alloc(&[32, 128], DType::BF16);
            let _ = arena.alloc(&[32, 128], DType::BF16);
            let layer_scratch_base = arena.used();

            // Run 50 layers, each using 100 KB of scratch.
            let scratch_per_layer = align_up(100 * 1024);
            for _ in 0..50 {
                arena.alloc(&[100 * 1024 / 2], DType::BF16); // 100 KB
                arena.set_offset(layer_scratch_base);
            }

            // Without scoping, 50 layers × 100 KB = 5 MB > 4 MB capacity → OOM.
            // With scoping, peak scratch = persistent + 1 layer ≈ 116 KB.
            assert!(
                arena.high_water_mark() < layer_scratch_base + scratch_per_layer + 256,
                "peak memory too high: {} (expected < {})",
                arena.high_water_mark(),
                layer_scratch_base + scratch_per_layer + 256,
            );
            let _ = persistent; // keep alive
        }

        #[test]
        fn test_set_offset_then_alloc_reuses_space() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            // Alloc A, then B.
            let a = arena.alloc(&[64], DType::F32);
            let saved = arena.used();
            let b = arena.alloc(&[64], DType::F32);
            let b_ptr = b.raw_ptr() as usize;

            // Restore to after A.
            arena.set_offset(saved);

            // Next alloc should reuse B's address.
            let c = arena.alloc(&[64], DType::F32);
            assert_eq!(c.raw_ptr() as usize, b_ptr);
            let _ = a; // keep alive
        }

        #[test]
        fn test_set_offset_to_zero_acts_like_reset() {
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let a = arena.alloc(&[128], DType::F32);
            let a_ptr = a.raw_ptr() as usize;
            assert!(arena.used() > 0);

            arena.set_offset(0);
            assert_eq!(arena.used(), 0);

            // Next alloc should start from the base (same address as a).
            let b = arena.alloc(&[128], DType::F32);
            assert_eq!(b.raw_ptr() as usize, a_ptr);
        }

        #[test]
        #[should_panic(expected = "set_offset")]
        fn test_set_offset_beyond_capacity_panics() {
            init_cuda();
            let arena = unsafe { ScratchArena::new(1024).unwrap() };
            // This is a mutable operation but set_offset takes &mut self.
            let mut arena = arena;
            arena.set_offset(2048); // > capacity
        }

        #[test]
        fn test_scoped_dtod_copy_correctness() {
            // Verify that D2D copies within the scoping pattern produce
            // correct data (write to scratch, copy to persistent, reclaim).
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            // Persistent buffer.
            let persistent = arena.alloc(&[4], DType::F32); // 16 bytes
            let layer_scratch_base = arena.used();

            // Write known values to scratch, then copy to persistent.
            let scratch = arena.alloc(&[4], DType::F32);

            // H2D: write [1.0, 2.0, 3.0, 4.0] to scratch.
            let src_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            unsafe {
                driver::memcpy_htod_async(
                    scratch.raw_ptr(),
                    src_data.as_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(), // default stream
                )
                .expect("h2d");
            }

            // D2D copy scratch → persistent.
            unsafe {
                driver::memcpy_dtod_async(
                    persistent.raw_ptr(),
                    scratch.raw_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("dtod");
            }

            // Reclaim scratch.
            arena.set_offset(layer_scratch_base);

            // Verify persistent has correct data.
            let mut result = [0.0f32; 4];
            unsafe {
                driver::memcpy_dtoh_async(
                    result.as_mut_ptr() as *mut u8,
                    persistent.raw_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("d2h");
                // Synchronize to ensure copies complete.
                driver::stream_synchronize(std::ptr::null_mut()).expect("sync");
            }
            assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_scoped_dtod_no_overlap() {
            // Verify that after reclaiming scratch and re-allocating,
            // the new allocation doesn't corrupt the persistent buffer.
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let persistent = arena.alloc(&[4], DType::F32);
            let layer_scratch_base = arena.used();

            // Write [1, 2, 3, 4] to persistent.
            let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            unsafe {
                driver::memcpy_htod_async(
                    persistent.raw_ptr(),
                    vals.as_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("h2d");
            }

            // Simulate 5 scoped layers: alloc scratch, zero it, reclaim.
            for _ in 0..5 {
                let scratch = arena.alloc(&[256], DType::F32); // 1 KB
                // Zero the scratch (simulating kernel writes).
                unsafe {
                    driver::memset_d8(scratch.raw_ptr(), 0, 1024, std::ptr::null_mut()).ok();
                }
                arena.set_offset(layer_scratch_base);
            }

            // Persistent should still have [1, 2, 3, 4].
            let mut result = [0.0f32; 4];
            unsafe {
                driver::memcpy_dtoh_async(
                    result.as_mut_ptr() as *mut u8,
                    persistent.raw_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("d2h");
                driver::stream_synchronize(std::ptr::null_mut()).expect("sync");
            }
            assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_scoped_multiple_persistent_buffers() {
            // Mirrors the model pattern: two persistent buffers (hs_buf, res_buf),
            // scratch allocated after them, verify both survive scoping.
            init_cuda();
            let mut arena = unsafe { ScratchArena::new(1024 * 1024).unwrap() };

            let hs_buf = arena.alloc(&[4], DType::F32);
            let res_buf = arena.alloc(&[4], DType::F32);
            let layer_scratch_base = arena.used();

            // Write distinct values to each persistent buffer.
            let hs_vals: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
            let res_vals: [f32; 4] = [100.0, 200.0, 300.0, 400.0];
            unsafe {
                driver::memcpy_htod_async(
                    hs_buf.raw_ptr(),
                    hs_vals.as_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("h2d hs");
                driver::memcpy_htod_async(
                    res_buf.raw_ptr(),
                    res_vals.as_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("h2d res");
            }

            // 20 scoped "layers".
            for i in 0..20 {
                // Allocate varying amounts of scratch.
                let n = 64 * (1 + i % 4); // 64, 128, 192, 256 floats
                let scratch = arena.alloc(&[n], DType::F32);

                // Write to scratch (simulating kernel output).
                let new_hs: [f32; 4] = [i as f32; 4];
                unsafe {
                    driver::memcpy_htod_async(
                        scratch.raw_ptr(),
                        new_hs.as_ptr() as *const u8,
                        16,
                        std::ptr::null_mut(),
                    )
                    .expect("h2d scratch");
                    // "Copy" scratch output to persistent hs_buf.
                    driver::memcpy_dtod_async(
                        hs_buf.raw_ptr(),
                        scratch.raw_ptr() as *const u8,
                        16,
                        std::ptr::null_mut(),
                    )
                    .expect("dtod hs");
                }

                arena.set_offset(layer_scratch_base);
            }

            // hs_buf should have the last layer's values [19, 19, 19, 19].
            // res_buf should be unchanged [100, 200, 300, 400].
            unsafe {
                driver::stream_synchronize(std::ptr::null_mut()).expect("sync");
            }

            let mut hs_result = [0.0f32; 4];
            let mut res_result = [0.0f32; 4];
            unsafe {
                driver::memcpy_dtoh_async(
                    hs_result.as_mut_ptr() as *mut u8,
                    hs_buf.raw_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("d2h hs");
                driver::memcpy_dtoh_async(
                    res_result.as_mut_ptr() as *mut u8,
                    res_buf.raw_ptr() as *const u8,
                    16,
                    std::ptr::null_mut(),
                )
                .expect("d2h res");
                driver::stream_synchronize(std::ptr::null_mut()).expect("sync");
            }
            assert_eq!(hs_result, [19.0, 19.0, 19.0, 19.0]);
            assert_eq!(res_result, [100.0, 200.0, 300.0, 400.0]);
        }
    }
}
