// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Metal kernel execution.
//!
//! These tests verify the complete pipeline:
//! - Device detection and initialization
//! - Buffer allocation and management
//! - Result verification

use scratchy_target_metal::{PooledBufferAllocator, detect_device};

/// Test basic Metal device initialization and buffer allocation
#[test]
fn test_device_and_buffer_allocation() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let allocator = PooledBufferAllocator::new(&device.device);

    // Allocate a buffer
    let buffer = allocator
        .allocate(1024 * 1024)
        .expect("Should allocate 1MB buffer");
    assert!(buffer.size() >= 1024 * 1024);
    assert!(!buffer.contents().is_null());

    // Verify we can write to the buffer
    unsafe {
        let ptr = buffer.contents() as *mut f32;
        for i in 0..256 {
            *ptr.add(i) = i as f32;
        }

        // Verify the data
        for i in 0..256 {
            assert_eq!(*ptr.add(i), i as f32);
        }
    }
}

/// Test buffer pooling and reuse
#[test]
fn test_buffer_pooling_integration() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let allocator = PooledBufferAllocator::new(&device.device);

    // Allocate and drop multiple buffers
    for _ in 0..10 {
        let buffer = allocator.allocate(4096).expect("Should allocate");
        // Buffer is automatically returned to pool on drop
        drop(buffer);
    }

    // Check that buffers are being pooled
    if let Some((total_bytes, pooled_count)) = allocator.stats() {
        assert!(pooled_count > 0, "Should have buffers in pool");
        assert!(total_bytes > 0, "Should have allocated memory");
    }

    // Allocate again - should reuse from pool
    let buffer = allocator.allocate(4096).expect("Should allocate from pool");
    assert!(buffer.size() >= 4096);
}

/// Test error handling for invalid operations
#[test]
fn test_error_handling() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let allocator = PooledBufferAllocator::new(&device.device);

    // Test invalid allocation size
    let result = allocator.allocate(0);
    assert!(result.is_err(), "Should fail to allocate zero bytes");
}
