// SPDX-License-Identifier: Apache-2.0
//! Probes whether Apple's `Shared` MTLBuffer correctly exposes bytes
//! at offsets > 4 GiB to GPU kernels. Empirically the production code
//! sees a `Qwen3-MoE-30B-A3B-4bit` `switch_mlp.down_proj.weight` bound
//! at offset 0x12ad13db0 (4.67 GiB) inside a 5.16 GiB shard buffer
//! return zeros from `MTLBuffer.contents()`; the kernel that consumes
//! it produces outputs consistent with `W == 0`.
//!
//! This standalone test removes the loader/dispatch complexity:
//!   1. Allocate a `Shared` MTLBuffer ≥ 5 GiB on the same device.
//!   2. Write a distinctive 32-byte pattern at three offsets:
//!      1 GiB, 4 GiB, 4.7 GiB.
//!   3. Use a GPU blit to copy 32 bytes from each of those offsets
//!      into a small (4 KiB) Shared destination buffer.
//!   4. CPU-read the destination buffer and check whether the bytes
//!      survived. If the 4.7 GiB sample reads back as zeros, Apple's
//!      Shared buffer addressing capped at 4 GiB — confirms the
//!      production hypothesis and motivates splitting register_mmap
//!      shards.
//!
//! `#[ignore]`d so it doesn't run in the default test sweep — the
//! 5 GiB allocation is too aggressive for shared CI. Run explicitly
//! when reproducing:
//!   cargo test --release -p scratchy-target-metal \
//!       --test large_buffer_offset_probe_test \
//!       -- --ignored --nocapture

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLBuffer,
    MTLDevice, MTLLibrary, MTLResourceOptions, MTLSharedEvent,
};
use scratchy_target_metal::device::detect_device;
use scratchy_target_metal::residency::MetalResidencySet;

const GIB: usize = 1024 * 1024 * 1024;

/// MTL4 equivalent of a classic blit copy. MTL4 exposes no dedicated blit
/// command encoder; it folds buffer-to-buffer copies into the compute
/// command encoder
/// (`copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`). This
/// runs the production MTL4 command lifecycle — `newMTL4CommandQueue` →
/// `beginCommandBufferWithAllocator` → attach a committed residency set →
/// encode the copies → `endCommandBuffer` → `commit_count` → signal + host
/// event-wait — around whatever copies the `encode` closure emits on the
/// compute encoder.
///
/// Returns `false` if the host has no MTL4 queue (caller should skip).
fn mtl4_blit_copies<F>(
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    residency: &[&Retained<ProtocolObject<dyn MTLBuffer>>],
    encode: F,
) -> bool
where
    F: FnOnce(&ProtocolObject<dyn MTL4ComputeCommandEncoder>),
{
    let Some(queue4) = device.newMTL4CommandQueue() else {
        eprintln!("skipping: no MTL4 queue");
        return false;
    };
    let res = MetalResidencySet::new(device);
    for &b in residency {
        res.insert(b);
    }
    res.commit();

    let alloc4 = device
        .newCommandAllocator()
        .expect("MTL4 command allocator");
    let event = device.newSharedEvent().expect("shared event");
    let cb = device.newCommandBuffer().expect("mtl4 command buffer");
    cb.beginCommandBufferWithAllocator(&alloc4);
    let cb_ptr: *mut AnyObject = Retained::as_ptr(&cb) as *const AnyObject as *mut AnyObject;
    unsafe {
        res.attach_to_mtl4_command_buffer(cb_ptr);
    }
    let enc = cb.computeCommandEncoder().expect("mtl4 compute encoder");
    encode(&enc);
    enc.endEncoding();
    cb.endCommandBuffer();

    let cb_protocol: &ProtocolObject<dyn MTL4CommandBuffer> = &cb;
    let mut cb_array = [std::ptr::NonNull::from(cb_protocol)];
    unsafe {
        queue4.commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1);
    }
    queue4.signalEvent_value(ProtocolObject::from_ref(&*event), 1);
    assert!(
        event.waitUntilSignaledValue_timeoutMS(1, 120_000),
        "MTL4 blit-copy command buffer timed out"
    );
    true
}

#[test]
#[ignore]
fn shared_buffer_above_4_gib_offset_round_trip() {
    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = &mdev.device;

    let max_len = device.maxBufferLength() as usize;
    eprintln!(
        "device.maxBufferLength = {} bytes ({:.2} GiB)",
        max_len,
        max_len as f64 / GIB as f64
    );

    let big_len: usize = 5 * GIB + 256 * 1024 * 1024; // 5.25 GiB
    if max_len < big_len {
        panic!(
            "device.maxBufferLength = {} < required {} — cannot run >4 GiB probe on this device",
            max_len, big_len
        );
    }

    // 5.25 GiB Shared buffer.
    let big = device
        .newBufferWithLength_options(big_len, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength(big) returned nil");
    eprintln!(
        "allocated big buffer: length={} contents()={:p}",
        big.length(),
        big.contents().as_ptr()
    );

    // 4 KiB Shared destination for GPU blit-back-to-CPU.
    let small = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength(small) returned nil");

    // CPU-write a 32-byte pattern at three offsets.
    let offsets = [GIB, 4 * GIB, 4 * GIB + 700 * 1024 * 1024];
    let patterns: Vec<[u8; 32]> = offsets
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut p = [0u8; 32];
            for (j, b) in p.iter_mut().enumerate() {
                *b = ((i + 1) as u8).wrapping_mul((j as u8).wrapping_add(1));
            }
            p
        })
        .collect();

    // Write via CPU contents() to test CPU-mapping at those offsets.
    let cpu_base = big.contents().as_ptr() as *mut u8;
    for (i, &off) in offsets.iter().enumerate() {
        unsafe {
            std::ptr::copy_nonoverlapping(patterns[i].as_ptr(), cpu_base.add(off), 32);
        }
    }

    // Read back via CPU to see if the writes stuck.
    eprintln!("--- CPU readback at each offset ---");
    for (i, &off) in offsets.iter().enumerate() {
        let mut buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(cpu_base.add(off), buf.as_mut_ptr(), 32);
        }
        let matches = buf == patterns[i];
        eprintln!(
            "  offset=0x{:x} ({:.2} GiB) cpu_readback_matches_pattern={} first8={:02x?}",
            off,
            off as f64 / GIB as f64,
            matches,
            &buf[..8]
        );
    }

    // GPU-copy each offset to a separate slot in `small`, on the MTL4
    // compute encoder's buffer-copy primitive (production's only command
    // path — there is no MTL4 blit encoder).
    if !mtl4_blit_copies(device, &[&big, &small], |enc| {
        for (i, &off) in offsets.iter().enumerate() {
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &big,
                    off,
                    &small,
                    i * 64,
                    32,
                );
            }
        }
    }) {
        eprintln!("skipping: no MTL4 queue");
        return;
    }

    // CPU-read the small buffer (which is < 4 KiB, definitely within
    // any mapping limit) to see what the GPU saw.
    let small_base = small.contents().as_ptr() as *const u8;
    eprintln!("--- GPU-blit readback (via small buffer) at each offset ---");
    let mut any_mismatch = false;
    for (i, &off) in offsets.iter().enumerate() {
        let mut buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(small_base.add(i * 64), buf.as_mut_ptr(), 32);
        }
        let matches = buf == patterns[i];
        if !matches {
            any_mismatch = true;
        }
        eprintln!(
            "  offset=0x{:x} ({:.2} GiB) gpu_blit_matches_pattern={} first8={:02x?}",
            off,
            off as f64 / GIB as f64,
            matches,
            &buf[..8]
        );
    }
    assert!(
        !any_mismatch,
        "GPU blit failed to read back the CPU-written pattern at one or more offsets — \
         this confirms Apple Silicon `Shared` MTLBuffer addressing breaks above some limit \
         (likely 4 GiB)."
    );

    // Suppress unused-import warnings.
    let _ = NonNull::<c_void>::dangling();
}

/// Matches production `register_mmap` shape: source buffer is a
/// `newBufferWithBytesNoCopy` wrapper around a real mmap, destination
/// is a freshly-allocated `Shared` buffer, blit copies the entire
/// mmap from src offset 0 to dst offset `shift`. Then probes the
/// destination at several offsets (including > 4 GiB).
///
/// Uses a 6 GiB file from `/tmp`; if the file doesn't exist the test
/// creates one filled with a deterministic pattern.
#[test]
#[ignore]
fn nocopy_source_blit_above_4_gib_round_trip() {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = &mdev.device;

    let path = "/tmp/scratchy_large_buffer_probe.bin";
    let len: usize = 5 * GIB + 256 * 1024 * 1024; // 5.25 GiB

    // Create the file once with a deterministic pattern if it doesn't
    // exist (or is the wrong size).
    let needs_create = match std::fs::metadata(path) {
        Ok(m) => m.len() as usize != len,
        Err(_) => true,
    };
    if needs_create {
        eprintln!("creating {path} ({} bytes, this may take a moment)", len);
        let mut f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .unwrap();
        let chunk: Vec<u8> = (0..(1024 * 1024))
            .map(|i| (i as u8) ^ ((i >> 8) as u8))
            .collect();
        for off in (0..len).step_by(chunk.len()) {
            let remain = (len - off).min(chunk.len());
            f.write_all(&chunk[..remain]).unwrap();
        }
        f.flush().unwrap();
    }

    // mmap the file via raw mmap (avoid pulling in memmap2 / libc as
    // dev-deps for this one probe).
    let file = std::fs::File::open(path).unwrap();
    use std::os::fd::AsRawFd;
    let fd = file.as_raw_fd();
    let mmap_len = std::fs::metadata(path).unwrap().len() as usize;
    let prot_read: i32 = 0x01;
    let map_private: i32 = 0x0002;
    let mmap_addr = unsafe {
        let addr = mmap_libc(
            std::ptr::null_mut(),
            mmap_len,
            prot_read,
            map_private,
            fd,
            0,
        );
        assert!(addr as isize != -1, "mmap failed");
        addr as *const u8
    };
    eprintln!("mmap base={:p} len={}", mmap_addr, mmap_len);

    // Build a NoCopy source MTLBuffer wrapping the mmap (exactly like
    // `register_mmap`).
    let page_size: usize = 16384;
    let src_buffer_len = (mmap_len + page_size - 1) & !(page_size - 1);
    let src_buffer = unsafe {
        let bytes = NonNull::new(mmap_addr as *mut c_void).expect("non-null");
        device
            .newBufferWithBytesNoCopy_length_options_deallocator(
                bytes,
                src_buffer_len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
            .expect("newBufferWithBytesNoCopy returned nil")
    };
    eprintln!("src_buffer length={}", src_buffer.length());

    // Destination: fresh Shared MTLBuffer same length.
    let shift: usize = 10; // simulate non-zero shift like register_mmap
    let dst_capacity = mmap_len + shift;
    let dst = device
        .newBufferWithLength_options(dst_capacity, MTLResourceOptions::StorageModeShared)
        .expect("dst alloc");
    eprintln!(
        "dst length={} contents()={:p}",
        dst.length(),
        dst.contents().as_ptr()
    );

    // Copy src[0..len] → dst[shift..shift+len], CHUNKED to ≤2 GiB per
    // call to dodge Apple's silent ~4 GiB copy-size cap on NoCopy-backed
    // sources. Runs on the MTL4 compute encoder's buffer-copy primitive.
    const CHUNK: usize = 2 * GIB;
    if !mtl4_blit_copies(device, &[&src_buffer, &dst], |enc| {
        let mut copied = 0usize;
        while copied < mmap_len {
            let n = (mmap_len - copied).min(CHUNK);
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src_buffer,
                    copied,
                    &dst,
                    shift + copied,
                    n,
                );
            }
            copied += n;
        }
    }) {
        eprintln!("skipping: no MTL4 queue");
        return;
    }

    // Sample several offsets and compare CPU readbacks of dst to the
    // mmap. Includes 4.68 GiB which is where production's down_proj
    // binding lands.
    let mut file2 = std::fs::File::open(path).unwrap();
    let mmap_offsets = [
        GIB,
        4 * GIB,
        4 * GIB + 700 * 1024 * 1024, // ~4.68 GiB
        5 * GIB,
    ];
    let dst_base = dst.contents().as_ptr() as *const u8;
    eprintln!("--- CPU readback of dst after blit ---");
    let mut any_mismatch = false;
    for off in mmap_offsets {
        let mut file_buf = [0u8; 32];
        file2.seek(SeekFrom::Start(off as u64)).unwrap();
        file2.read_exact(&mut file_buf).unwrap();

        let mut dst_buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(dst_base.add(off + shift), dst_buf.as_mut_ptr(), 32);
        }
        let matches = file_buf == dst_buf;
        if !matches {
            any_mismatch = true;
        }
        eprintln!(
            "  mmap_off=0x{:x} ({:.2} GiB) cpu_dst_matches_file={} file_first8={:02x?} dst_first8={:02x?}",
            off,
            off as f64 / GIB as f64,
            matches,
            &file_buf[..8],
            &dst_buf[..8]
        );
    }

    // Also GPU-copy dst → small probe buffer to see what GPU sees.
    let small = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("small");
    if !mtl4_blit_copies(device, &[&dst, &small], |enc| {
        for (i, &off) in mmap_offsets.iter().enumerate() {
            unsafe {
                enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &dst,
                    off + shift,
                    &small,
                    i * 64,
                    32,
                );
            }
        }
    }) {
        eprintln!("skipping: no MTL4 queue");
        return;
    }
    let small_base = small.contents().as_ptr() as *const u8;
    eprintln!("--- GPU-blit readback of dst ---");
    for (i, &off) in mmap_offsets.iter().enumerate() {
        let mut file_buf = [0u8; 32];
        file2.seek(SeekFrom::Start(off as u64)).unwrap();
        file2.read_exact(&mut file_buf).unwrap();

        let mut small_buf = [0u8; 32];
        unsafe {
            std::ptr::copy_nonoverlapping(small_base.add(i * 64), small_buf.as_mut_ptr(), 32);
        }
        let matches = file_buf == small_buf;
        if !matches {
            any_mismatch = true;
        }
        eprintln!(
            "  mmap_off=0x{:x} ({:.2} GiB) gpu_dst_matches_file={} file_first8={:02x?} gpu_first8={:02x?}",
            off,
            off as f64 / GIB as f64,
            matches,
            &file_buf[..8],
            &small_buf[..8]
        );
    }

    assert!(
        !any_mismatch,
        "blit-from-NoCopy round trip failed at some offset"
    );
}

// Raw mmap binding (avoids pulling in libc/memmap2 as dev-deps).
unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;
}
unsafe fn mmap_libc(
    addr: *mut std::ffi::c_void,
    length: usize,
    prot: i32,
    flags: i32,
    fd: i32,
    offset: i64,
) -> *mut std::ffi::c_void {
    unsafe { mmap(addr, length, prot, flags, fd, offset) }
}

/// MTL4 argument-table sibling: bind the big buffer's huge offset via
/// `gpuAddress() + off` + `setAddress:atIndex:` — the EXACT production
/// binding mechanism (`interpreter/metal/mtl4.rs`). The MTL3
/// `setBuffer:offset:` compute probe above passes on macOS 26.5.1; if
/// THIS one corrupts, the regression is the MTL4 bindless path.
#[test]
#[ignore = "isolation-only: allocates ~5 GiB; under concurrent full-suite \
            residency pressure the large-offset read can corrupt (the very \
            macOS bug this probes), so it is flaky in the default suite. Run: \
            cargo test --test large_buffer_offset_probe_test -- --ignored --test-threads=1"]
fn shared_buffer_mtl4_gpuaddress_read_at_large_offsets() {
    use objc2_metal::{
        MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandEncoder,
        MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLSharedEvent,
    };
    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = &mdev.device;
    let Some(queue4) = device.newMTL4CommandQueue() else {
        eprintln!("skipping: no MTL4");
        return;
    };

    let big_len: usize = 5 * GIB + 256 * 1024 * 1024;
    if (device.maxBufferLength() as usize) < big_len {
        eprintln!("skipping: maxBufferLength too small");
        return;
    }
    let big = device
        .newBufferWithLength_options(big_len, MTLResourceOptions::StorageModeShared)
        .expect("big buffer");
    let dst = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("dst buffer");

    const MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void copy32(device const uchar* src [[buffer(0)]],
                           device uchar* dst        [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
            if (i < 32) { dst[i] = src[i]; }
        }
    "#;
    let opts = objc2_metal::MTLCompileOptions::new();
    let lib = device
        .newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(MSL), Some(&opts))
        .expect("compile probe lib");
    let func = lib
        .newFunctionWithName(&objc2_foundation::NSString::from_str("copy32"))
        .expect("copy32 fn");
    let pso = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pso");

    let offsets: [usize; 3] = [GIB, 2 * GIB + GIB / 2, 4 * GIB + 700 * 1024 * 1024];
    let base = big.contents().as_ptr() as *mut u8;
    for (k, &off) in offsets.iter().enumerate() {
        let pat: Vec<u8> = (0..32).map(|i| (0xC0 + k as u8) ^ (i as u8)).collect();
        unsafe { std::ptr::copy_nonoverlapping(pat.as_ptr(), base.add(off), 32) };
    }

    // Residency: MTL4 requires explicit residency for address-bound
    // buffers — mirror production (residency set attached to the CB).
    let res = scratchy_target_metal::residency::MetalResidencySet::new(device);
    res.insert(&big);
    res.insert(&dst);
    res.commit();

    let mut failures = Vec::new();
    for (k, &off) in offsets.iter().enumerate() {
        unsafe { std::ptr::write_bytes(dst.contents().as_ptr() as *mut u8, 0, 64) };

        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(2);
        let table = device
            .newArgumentTableWithDescriptor_error(&desc)
            .expect("argument table");
        unsafe {
            table.setAddress_atIndex(big.gpuAddress() + off as u64, 0);
            table.setAddress_atIndex(dst.gpuAddress(), 1);
        }

        // Mirror the production CB lifecycle (pool.rs run path):
        // begin(allocator) → attach residency → encode → end → commit
        // → signalEvent → host wait.
        let alloc4 = device
            .newCommandAllocator()
            .expect("MTL4 command allocator");
        let event = device.newSharedEvent().expect("shared event");
        let cb = device.newCommandBuffer().expect("mtl4 command buffer");
        cb.beginCommandBufferWithAllocator(&alloc4);
        let cb_ptr: *mut objc2::runtime::AnyObject = objc2::rc::Retained::as_ptr(&cb)
            as *const objc2::runtime::AnyObject
            as *mut objc2::runtime::AnyObject;
        unsafe { res.attach_to_mtl4_command_buffer(cb_ptr) };
        let enc = cb.computeCommandEncoder().expect("mtl4 encoder");
        enc.setComputePipelineState(&pso);
        enc.setArgumentTable(Some(&table));
        enc.dispatchThreads_threadsPerThreadgroup(
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.endCommandBuffer();
        let cb_protocol: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandBuffer> = &cb;
        let cb_nn = std::ptr::NonNull::from(cb_protocol);
        let mut cb_array = [cb_nn];
        unsafe { queue4.commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1) };
        queue4.signalEvent_value(objc2::runtime::ProtocolObject::from_ref(&*event), 1);
        assert!(
            event.waitUntilSignaledValue_timeoutMS(1, 30_000),
            "MTL4 probe CB timed out"
        );

        let got = unsafe { std::slice::from_raw_parts(dst.contents().as_ptr() as *const u8, 32) };
        let want: Vec<u8> = (0..32).map(|i| (0xC0 + k as u8) ^ (i as u8)).collect();
        let ok = got == want.as_slice();
        eprintln!(
            "MTL4 gpuAddress read at offset {:.2} GiB: {} (got[0..8]={:02x?})",
            off as f64 / GIB as f64,
            if ok { "OK" } else { "CORRUPT" },
            &got[..8]
        );
        if !ok {
            failures.push(off);
        }
    }
    assert!(
        failures.is_empty(),
        "MTL4 gpuAddress reads corrupted at offsets: {failures:?}"
    );
}

/// Pressure sibling of the MTL4 probe: same gpuAddress+offset compute
/// read, but with ~20 GiB of residency-committed ballast resident —
/// the actual Qwen3.5-MoE-35B condition (4x ~4.9 GiB shards + arenas;
/// its embed reads at offset 2.87 GiB and returns zeros on macOS
/// 26.5.1 while every host-side check passes).
#[test]
#[ignore = "isolation-only: allocates ~20 GiB of residency ballast; flaky \
            under concurrent full-suite memory pressure. Run: cargo test \
            --test large_buffer_offset_probe_test -- --ignored --test-threads=1"]
fn shared_buffer_mtl4_read_under_residency_pressure() {
    use objc2_metal::{
        MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandEncoder,
        MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLSharedEvent,
    };
    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = &mdev.device;
    let Some(queue4) = device.newMTL4CommandQueue() else {
        eprintln!("skipping: no MTL4");
        return;
    };

    let shard_len: usize = 4 * GIB + 940 * 1024 * 1024; // ~4.92 GiB, shard-like
    let res = scratchy_target_metal::residency::MetalResidencySet::new(device);
    let mut shards = Vec::new();
    for i in 0..4 {
        let Some(b) =
            device.newBufferWithLength_options(shard_len, MTLResourceOptions::StorageModeShared)
        else {
            eprintln!("skipping: shard {i} alloc failed");
            return;
        };
        res.insert(&b);
        shards.push(b);
    }
    let dst = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("dst");
    res.insert(&dst);
    res.commit();

    const MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void copy32(device const uchar* src [[buffer(0)]],
                           device uchar* dst        [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
            if (i < 32) { dst[i] = src[i]; }
        }
    "#;
    let opts = objc2_metal::MTLCompileOptions::new();
    let lib = device
        .newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(MSL), Some(&opts))
        .expect("lib");
    let func = lib
        .newFunctionWithName(&objc2_foundation::NSString::from_str("copy32"))
        .expect("fn");
    let pso = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pso");

    // The production failure point: 2.87 GiB into shard 0; also touch
    // every shard to fault broad residency like a real load does.
    let embed_off: usize = 3_081_934_880 & !63;
    for (i, b) in shards.iter().enumerate() {
        let base = b.contents().as_ptr() as *mut u8;
        let pat: Vec<u8> = (0..32).map(|j| (0xD0 + i as u8) ^ (j as u8)).collect();
        unsafe { std::ptr::copy_nonoverlapping(pat.as_ptr(), base.add(embed_off), 32) };
    }

    let alloc4 = device.newCommandAllocator().expect("alloc4");
    let event = device.newSharedEvent().expect("event");
    let mut failures = Vec::new();
    for (i, b) in shards.iter().enumerate() {
        unsafe { std::ptr::write_bytes(dst.contents().as_ptr() as *mut u8, 0, 64) };
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(2);
        let table = device
            .newArgumentTableWithDescriptor_error(&desc)
            .expect("table");
        unsafe {
            table.setAddress_atIndex(b.gpuAddress() + embed_off as u64, 0);
            table.setAddress_atIndex(dst.gpuAddress(), 1);
        }
        let cb = device.newCommandBuffer().expect("cb");
        cb.beginCommandBufferWithAllocator(&alloc4);
        let cb_ptr: *mut objc2::runtime::AnyObject = objc2::rc::Retained::as_ptr(&cb)
            as *const objc2::runtime::AnyObject
            as *mut objc2::runtime::AnyObject;
        unsafe { res.attach_to_mtl4_command_buffer(cb_ptr) };
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pso);
        enc.setArgumentTable(Some(&table));
        enc.dispatchThreads_threadsPerThreadgroup(
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.endCommandBuffer();
        let cb_protocol: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandBuffer> = &cb;
        let mut cb_array = [std::ptr::NonNull::from(cb_protocol)];
        unsafe { queue4.commit_count(std::ptr::NonNull::from(&mut cb_array[0]), 1) };
        queue4.signalEvent_value(
            objc2::runtime::ProtocolObject::from_ref(&*event),
            (i + 1) as u64,
        );
        assert!(event.waitUntilSignaledValue_timeoutMS((i + 1) as u64, 30_000));

        let got = unsafe { std::slice::from_raw_parts(dst.contents().as_ptr() as *const u8, 32) };
        let want: Vec<u8> = (0..32).map(|j| (0xD0 + i as u8) ^ (j as u8)).collect();
        let ok = got == want.as_slice();
        eprintln!(
            "shard{i} @2.87GiB under ~20GiB residency: {} (got[0..8]={:02x?})",
            if ok { "OK" } else { "CORRUPT" },
            &got[..8]
        );
        if !ok {
            failures.push(i);
        }
    }
    assert!(failures.is_empty(), "corrupt shards: {failures:?}");
}

/// Throughput A/B for the loader's realign-copy: GPU blit from a
/// bytesNoCopy file-backed staging buffer vs CPU memcpy with
/// madvise(WILLNEED) — interleaved across the shards of a real
/// checkpoint so page-cache state can't favor one arm.
///
///   SCRATCHY_PROBE_DIR=<snapshot dir> cargo test --release \
///     -p scratchy-target-metal --test large_buffer_offset_probe_test \
///     blit_vs_memcpy -- --nocapture
#[test]
fn blit_vs_memcpy_shard_throughput() {
    let Some(dir) = std::env::var_os("SCRATCHY_PROBE_DIR") else {
        eprintln!("skipping: set SCRATCHY_PROBE_DIR to a snapshot dir");
        return;
    };
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    assert!(!shards.is_empty());

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = &mdev.device;
    let page: usize = 16384;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    for (i, path) in shards.iter().enumerate() {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path).expect("open shard");
        let len = file.metadata().unwrap().len() as usize;
        let base = unsafe { mmap(std::ptr::null_mut(), len, 0x01, 0x0002, file.as_raw_fd(), 0) };
        assert!(base as isize != -1, "mmap failed");

        let dst = device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .expect("dst alloc");
        let t0 = std::time::Instant::now();
        let mode = if i % 2 == 0 { "gpu-blit" } else { "cpu-memcpy" };
        if i % 2 == 0 {
            let rounded = (len + page - 1) & !(page - 1);
            let src = unsafe {
                device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        NonNull::new(base).unwrap(),
                        rounded,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                    .expect("noCopy wrap")
            };
            const CHUNK: usize = 2 * GIB;
            if !mtl4_blit_copies(device, &[&src, &dst], |enc| {
                let mut off = 0usize;
                while off < len {
                    let n = (len - off).min(CHUNK);
                    unsafe {
                        enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &src, off, &dst, off, n,
                        );
                    }
                    off += n;
                }
            }) {
                eprintln!("skipping gpu-blit arm: no MTL4 queue");
                unsafe { munmap(base, len) };
                continue;
            }
        } else {
            unsafe {
                madvise(base, len, 3 /* MADV_WILLNEED */)
            };
            let d = dst.contents().as_ptr() as *mut u8;
            unsafe { std::ptr::copy_nonoverlapping(base as *const u8, d, len) };
        }
        let dt = t0.elapsed().as_secs_f64();
        // integrity spot-check: 32 bytes at an odd interior offset
        let probe_off = (len / 3) | 7;
        let s = unsafe { std::slice::from_raw_parts((base as *const u8).add(probe_off), 32) };
        let g = unsafe {
            std::slice::from_raw_parts((dst.contents().as_ptr() as *const u8).add(probe_off), 32)
        };
        assert_eq!(s, g, "copy mismatch in {mode}");
        eprintln!(
            "{mode:<10} {:>6.2} GiB in {dt:6.2}s = {:5.2} GiB/s  ({})",
            len as f64 / GIB as f64,
            len as f64 / GIB as f64 / dt,
            path.file_name().unwrap().to_string_lossy()
        );
        unsafe { munmap(base, len) };
    }
}

/// Phase-1 risk probe for the aligned-sidecar zero-copy loader: wrap
/// ALL shards of a real checkpoint (~19 GiB) as file-backed
/// bytesNoCopy buffers, insert into an MTL4 residency set, and time
/// wrap / commit / first GPU dispatch / steady-state dispatch. The
/// open question is whether macOS 26.5.1 wires that much file-backed
/// memory (a) correctly and (b) at what cost.
///
///   SCRATCHY_PROBE_DIR=<snapshot dir> cargo test --release ... nocopy_residency_wiring -- --nocapture
#[test]
fn nocopy_residency_wiring_at_scale() {
    use objc2_metal::{
        MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandBuffer, MTL4CommandEncoder,
        MTL4CommandQueue, MTL4ComputeCommandEncoder, MTLSharedEvent,
    };
    let Some(dir) = std::env::var_os("SCRATCHY_PROBE_DIR") else {
        eprintln!("skipping: set SCRATCHY_PROBE_DIR");
        return;
    };
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    assert!(!shards.is_empty());

    let Some(mdev) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let device = &mdev.device;
    let Some(queue4) = device.newMTL4CommandQueue() else {
        eprintln!("skipping: no MTL4");
        return;
    };
    let page: usize = 16384;
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
    }

    // Wrap every shard.
    let t_wrap = std::time::Instant::now();
    let mut bufs = Vec::new();
    let mut total = 0usize;
    for path in &shards {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path).expect("open");
        let len = file.metadata().unwrap().len() as usize;
        let base = unsafe { mmap(std::ptr::null_mut(), len, 0x01, 0x0002, file.as_raw_fd(), 0) };
        assert!(base as isize != -1);
        let rounded = (len + page - 1) & !(page - 1);
        let buf = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(base).unwrap(),
                    rounded,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .expect("noCopy wrap")
        };
        total += len;
        bufs.push((buf, len));
        std::mem::forget(file); // keep fd+mapping alive for the probe
    }
    eprintln!(
        "wrap: {} shards / {:.2} GiB in {:?}",
        bufs.len(),
        total as f64 / GIB as f64,
        t_wrap.elapsed()
    );

    // Residency set insert + commit.
    let res = scratchy_target_metal::residency::MetalResidencySet::new(device);
    let t_ins = std::time::Instant::now();
    for (b, _) in &bufs {
        res.insert(b);
    }
    eprintln!("insert: {:?}", t_ins.elapsed());
    let t_commit = std::time::Instant::now();
    res.commit();
    eprintln!("residency commit: {:?}", t_commit.elapsed());

    // Compute read probe: copy32 from a deep offset of every shard.
    const MSL: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void copy32(device const uchar* src [[buffer(0)]],
                           device uchar* dst        [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
            if (i < 32) { dst[i] = src[i]; }
        }
    "#;
    let opts = objc2_metal::MTLCompileOptions::new();
    let lib = device
        .newLibraryWithSource_options_error(&objc2_foundation::NSString::from_str(MSL), Some(&opts))
        .expect("lib");
    let func = lib
        .newFunctionWithName(&objc2_foundation::NSString::from_str("copy32"))
        .expect("fn");
    let pso = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pso");
    let dst = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .expect("dst");
    res.insert(&dst);
    res.commit();

    let alloc4 = device.newCommandAllocator().expect("alloc4");
    let event = device.newSharedEvent().expect("event");
    let mut sig = 0u64;
    for round in 0..2 {
        let t_round = std::time::Instant::now();
        for (k, (b, len)) in bufs.iter().enumerate() {
            let off = ((len * 2 / 3) & !63) as u64;
            let desc = MTL4ArgumentTableDescriptor::new();
            desc.setMaxBufferBindCount(2);
            let table = device
                .newArgumentTableWithDescriptor_error(&desc)
                .expect("table");
            unsafe {
                table.setAddress_atIndex(b.gpuAddress() + off, 0);
                table.setAddress_atIndex(dst.gpuAddress(), 1);
            }
            let cb = device.newCommandBuffer().expect("cb");
            cb.beginCommandBufferWithAllocator(&alloc4);
            let cb_ptr: *mut objc2::runtime::AnyObject = objc2::rc::Retained::as_ptr(&cb)
                as *const objc2::runtime::AnyObject
                as *mut objc2::runtime::AnyObject;
            unsafe { res.attach_to_mtl4_command_buffer(cb_ptr) };
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pso);
            enc.setArgumentTable(Some(&table));
            enc.dispatchThreads_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cb.endCommandBuffer();
            let cbp: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandBuffer> = &cb;
            let mut arr = [std::ptr::NonNull::from(cbp)];
            unsafe { queue4.commit_count(std::ptr::NonNull::from(&mut arr[0]), 1) };
            sig += 1;
            queue4.signalEvent_value(objc2::runtime::ProtocolObject::from_ref(&*event), sig);
            assert!(
                event.waitUntilSignaledValue_timeoutMS(sig, 120_000),
                "timeout"
            );
            // integrity: GPU bytes == CPU mmap bytes at same offset
            let g = unsafe { std::slice::from_raw_parts(dst.contents().as_ptr() as *const u8, 32) };
            let c = unsafe {
                std::slice::from_raw_parts(
                    (b.contents().as_ptr() as *const u8).add(off as usize),
                    32,
                )
            };
            assert_eq!(g, c, "shard {k} GPU/CPU mismatch");
        }
        eprintln!(
            "round {round} ({}): all-shard dispatch+wait {:?}",
            if round == 0 { "first touch" } else { "steady" },
            t_round.elapsed()
        );
    }
}
