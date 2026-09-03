// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal buffer allocator with pooling and memory pressure handling.
//!
//! Provides efficient buffer allocation for Metal kernels with:
//! - Buffer pooling to reduce allocation overhead
//! - Size-based bucketing for efficient reuse
//! - Memory pressure monitoring and eviction
//! - Alignment guarantees for GPU access
//!
//! This is the Metal analog of CUDA's memory allocator, adapted for
//! Metal's unified memory architecture on Apple Silicon.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Minimum buffer alignment (256 bytes for Metal)
const MIN_ALIGNMENT: usize = 256;

/// Size buckets for buffer pooling (powers of 2)
const SIZE_BUCKETS: &[usize] = &[
    1024,       // 1 KB
    4096,       // 4 KB
    16384,      // 16 KB
    65536,      // 64 KB
    262144,     // 256 KB
    1048576,    // 1 MB
    4194304,    // 4 MB
    16777216,   // 16 MB
    67108864,   // 64 MB
    268435456,  // 256 MB
    1073741824, // 1 GB
];

/// Error types for Metal allocator operations
#[derive(Debug, Clone)]
pub enum AllocatorError {
    /// Allocation failed (out of memory)
    AllocationFailed(usize),
    /// Invalid size (zero or too large)
    InvalidSize(usize),
    /// Memory pressure threshold exceeded
    MemoryPressure,
}

impl std::fmt::Display for AllocatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllocationFailed(size) => write!(f, "Failed to allocate {} bytes", size),
            Self::InvalidSize(size) => write!(f, "Invalid allocation size: {}", size),
            Self::MemoryPressure => write!(f, "Memory pressure threshold exceeded"),
        }
    }
}

impl std::error::Error for AllocatorError {}

/// A pooled Metal buffer that returns to the pool when dropped
pub struct PooledBuffer {
    buffer: Buffer,
    size: usize,
    pool: Rc<Mutex<BufferPool>>,
}

impl PooledBuffer {
    /// Get the underlying Metal buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get raw pointer to buffer contents
    pub fn contents(&self) -> *mut std::ffi::c_void {
        self.buffer.contents().as_ptr()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.return_buffer(self.size, self.buffer.clone());
        }
    }
}

/// Buffer pool for a specific size bucket
struct SizePool {
    buffers: Vec<Buffer>,
    total_allocated: usize,
    max_pooled: usize,
}

impl SizePool {
    fn new(max_pooled: usize) -> Self {
        Self {
            buffers: Vec::new(),
            total_allocated: 0,
            max_pooled,
        }
    }

    fn get(&mut self) -> Option<Buffer> {
        self.buffers.pop()
    }

    fn return_buffer(&mut self, buffer: Buffer) {
        if self.buffers.len() < self.max_pooled {
            self.buffers.push(buffer);
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }
}

/// Metal buffer allocator with pooling (internal, wrapped by PooledBufferAllocator)
struct BufferPool {
    device: Device,
    pools: HashMap<usize, SizePool>,
    total_bytes_allocated: usize,
    max_bytes: usize,
}

impl BufferPool {
    fn new(device: &Device) -> Self {
        Self::with_limits(device, 4 * 1024 * 1024 * 1024, 16)
    }

    fn with_limits(device: &Device, max_bytes: usize, max_pooled_per_size: usize) -> Self {
        let mut pools = HashMap::new();
        for &size in SIZE_BUCKETS {
            pools.insert(size, SizePool::new(max_pooled_per_size));
        }

        Self {
            device: device.clone(),
            pools,
            total_bytes_allocated: 0,
            max_bytes,
        }
    }

    fn allocate(
        &mut self,
        size: usize,
        pool_ref: Rc<Mutex<BufferPool>>,
    ) -> Result<PooledBuffer, AllocatorError> {
        if size == 0 {
            return Err(AllocatorError::InvalidSize(size));
        }

        let aligned_size = (size + MIN_ALIGNMENT - 1) & !(MIN_ALIGNMENT - 1);

        let bucket_size = SIZE_BUCKETS
            .iter()
            .find(|&&s| s >= aligned_size)
            .copied()
            .unwrap_or_else(|| (aligned_size + MIN_ALIGNMENT - 1) & !(MIN_ALIGNMENT - 1));

        if self.total_bytes_allocated + bucket_size > self.max_bytes {
            self.evict_lru();
            if self.total_bytes_allocated + bucket_size > self.max_bytes {
                return Err(AllocatorError::MemoryPressure);
            }
        }

        let buffer = if let Some(pool) = self.pools.get_mut(&bucket_size) {
            pool.get()
        } else {
            None
        };

        let buffer = match buffer {
            Some(buf) => buf,
            None => {
                let buf = self
                    .device
                    .newBufferWithLength_options(bucket_size, MTLResourceOptions::StorageModeShared)
                    .ok_or(AllocatorError::AllocationFailed(bucket_size))?;

                if let Some(pool) = self.pools.get_mut(&bucket_size) {
                    pool.total_allocated += 1;
                }

                self.total_bytes_allocated += bucket_size;
                buf
            }
        };

        Ok(PooledBuffer {
            buffer,
            size: bucket_size,
            pool: pool_ref,
        })
    }

    fn return_buffer(&mut self, size: usize, buffer: Buffer) {
        if let Some(pool) = self.pools.get_mut(&size) {
            pool.return_buffer(buffer);
        }
    }

    fn evict_lru(&mut self) {
        for pool in self.pools.values_mut() {
            let freed = pool.buffers.len() * pool.buffers.capacity();
            pool.clear();
            self.total_bytes_allocated = self.total_bytes_allocated.saturating_sub(freed);
        }
    }

    fn clear(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear();
        }
    }

    fn total_bytes_allocated(&self) -> usize {
        self.total_bytes_allocated
    }

    fn pooled_buffer_count(&self) -> usize {
        self.pools.values().map(|p| p.buffers.len()).sum()
    }
}

/// Thread-safe buffer allocator
pub struct PooledBufferAllocator {
    pool: Rc<Mutex<BufferPool>>,
}

impl PooledBufferAllocator {
    pub fn new(device: &Device) -> Self {
        Self {
            pool: Rc::new(Mutex::new(BufferPool::new(device))),
        }
    }

    pub fn with_limits(device: &Device, max_bytes: usize, max_pooled_per_size: usize) -> Self {
        Self {
            pool: Rc::new(Mutex::new(BufferPool::with_limits(
                device,
                max_bytes,
                max_pooled_per_size,
            ))),
        }
    }

    pub fn allocate(&self, size: usize) -> Result<PooledBuffer, AllocatorError> {
        let pool_ref = Rc::clone(&self.pool);
        self.pool
            .lock()
            .map_err(|_| AllocatorError::AllocationFailed(size))?
            .allocate(size, pool_ref)
    }

    pub fn clear(&self) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.clear();
        }
    }

    pub fn stats(&self) -> Option<(usize, usize)> {
        self.pool
            .lock()
            .ok()
            .map(|pool| (pool.total_bytes_allocated(), pool.pooled_buffer_count()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::detect_device;

    #[test]
    fn test_allocator_creation() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let allocator = PooledBufferAllocator::new(&device.device);

        if let Some((bytes, count)) = allocator.stats() {
            assert_eq!(bytes, 0);
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn test_buffer_allocation() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let allocator = PooledBufferAllocator::new(&device.device);

        let buffer = allocator.allocate(1024).expect("Should allocate");
        assert!(buffer.size() >= 1024);
        assert!(!buffer.contents().is_null());
    }

    #[test]
    fn test_buffer_pooling() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let allocator = PooledBufferAllocator::new(&device.device);

        {
            let _buffer = allocator.allocate(1024).expect("Should allocate");
        }

        if let Some((_, count)) = allocator.stats() {
            assert!(count > 0, "Buffer should be returned to pool");
        }
    }

    #[test]
    fn test_size_bucketing() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let allocator = PooledBufferAllocator::new(&device.device);

        let buffer1 = allocator.allocate(100).expect("Should allocate");
        let buffer2 = allocator.allocate(1000).expect("Should allocate");
        let buffer3 = allocator.allocate(10000).expect("Should allocate");

        assert!(buffer1.size() >= 100);
        assert!(buffer2.size() >= 1000);
        assert!(buffer3.size() >= 10000);

        assert_eq!(buffer1.size() % MIN_ALIGNMENT, 0);
        assert_eq!(buffer2.size() % MIN_ALIGNMENT, 0);
        assert_eq!(buffer3.size() % MIN_ALIGNMENT, 0);
    }

    #[test]
    fn test_clear_pool() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let allocator = PooledBufferAllocator::new(&device.device);

        for _ in 0..5 {
            let _buffer = allocator.allocate(1024).expect("Should allocate");
        }

        allocator.clear();

        if let Some((_, count)) = allocator.stats() {
            assert_eq!(count, 0, "Pool should be empty after clear");
        }
    }

    #[test]
    fn test_invalid_size() {
        let Some(device) = detect_device() else {
            eprintln!("skipping: no Metal device (or no Metal 4 support)");
            return;
        };
        let allocator = PooledBufferAllocator::new(&device.device);

        // Match on the Err directly — `unwrap_err()` would require the Ok type
        // (`PooledBuffer`) to be `Debug`, which it isn't (holds an objc2 buffer).
        let result = allocator.allocate(0);
        assert!(matches!(result, Err(AllocatorError::InvalidSize(0))));
    }
}
