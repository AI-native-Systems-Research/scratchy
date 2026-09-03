// SPDX-License-Identifier: Apache-2.0
//! Metal device-memory handle: an `MTLBuffer` plus the GPU-address /
//! sub-range accessors the paged-KV machinery needs (which the neutral
//! [`scratchy_tensors::RawGpuMem`] descriptor cannot carry).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

pub struct MetalMem {
    buffer: Buffer,
    base: *mut u8,
    size: usize,
    /// Byte offset into `buffer`'s gpuAddress range. 0 for ordinary per-chunk
    /// buffers; non-zero for a tile sub-range of a shared placement-sparse
    /// buffer (one `MTLBuffer` per layer, many chunks at distinct offsets).
    /// Dropping a clone-shared `MetalMem` is harmless: the underlying buffer is
    /// ref-counted (`Retained`) and stays alive as long as any clone does.
    metal_offset: usize,
}

// SAFETY: the buffer is `Retained` (ref-counted); the raw base pointer is a
// device-visible address governed by the metal runtime — same invariant the
// previous metal arm of `RawGpuMem` upheld.
unsafe impl Send for MetalMem {}
unsafe impl Sync for MetalMem {}

impl MetalMem {
    pub fn from_buffer(buffer: Buffer) -> Self {
        // GUARD: this wrapper derives CPU pointers from `contents()` — only
        // valid for CPU-accessible storage. A StorageModePrivate buffer here
        // silently yields a garbage base on macOS 26.5.1+ (large allocations
        // return an unmapped pointer). Fail at construction instead.
        use objc2_metal::MTLResource as _;
        let mode = buffer.storageMode();
        assert!(
            mode == objc2_metal::MTLStorageMode::Shared,
            "MetalMem::from_buffer requires StorageModeShared (CPU-accessible) \
             storage; got {mode:?}. Private/Memoryless buffers have no valid \
             contents() pointer — use a GPU-only wrapper or allocate Shared.",
        );
        let base = buffer.contents().as_ptr() as *mut u8;
        let size = buffer.length();
        assert!(
            !base.is_null() || size == 0,
            "MetalMem::from_buffer: contents() returned NULL for a {size}-byte \
             Shared buffer",
        );
        Self {
            buffer,
            base,
            size,
            metal_offset: 0,
        }
    }

    /// Wrap a sub-range `[offset, offset+size)` of a SHARED buffer. Used by the
    /// placement-sparse KV path where one `MTLBuffer` per layer backs many
    /// chunk indices at distinct offsets. `buffer` is a clone of a `Retained`,
    /// so dropping this `MetalMem` does NOT free the underlying buffer.
    pub fn from_buffer_with_offset(buffer: Buffer, offset: usize, size: usize) -> Self {
        // For StorageModePrivate sparse buffers, `contents()` returns null;
        // `base` is non-load-bearing on the metal path, kept for parity.
        let base = std::ptr::null_mut();
        Self {
            buffer,
            base,
            size,
            metal_offset: offset,
        }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.base
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// GPU virtual address of this allocation's first byte. For ordinary
    /// per-chunk buffers this equals `buffer.gpuAddress()`; for sparse
    /// sub-ranges it adds the byte offset within the shared buffer.
    pub fn gpu_address(&self) -> u64 {
        use objc2_metal::MTLBuffer;
        self.buffer.gpuAddress() + self.metal_offset as u64
    }

    /// Byte offset within `buffer()` where this allocation begins. Zero for
    /// ordinary per-chunk buffers, non-zero for sparse sub-ranges.
    pub fn metal_offset(&self) -> usize {
        self.metal_offset
    }
}

impl scratchy_tensors::PoolMemory for MetalMem {
    fn ptr(&self) -> *mut u8 {
        self.base
    }
}
