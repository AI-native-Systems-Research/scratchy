//! Correctness test for the kv-head-major dense gather kernel
//! (`attention_dense_gather.metal`, the gemma4 hd512 unfused-attention path).
//!
//! Validates: paged block_table indirection, single-chunk pointer deref, and
//! the kv-head-major dense output layout [kv_head, pos, head_dim] with the row
//! stride derived from the live kv_len. Uses the f16 COPY variant (V-gather);
//! the rope (K) variant reuses the same gather + the production rope math.

mod common;

use std::ffi::c_void;
use std::ptr::NonNull;

use half::f16;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary, MTLResourceOptions,
    MTLSize,
};
use scratchy_target_metal::detect_device;

// Small, all-f16-exact shapes (values <= 191 are exact in f16).
const HEAD_DIM: u32 = 8;
const NUM_KV: u32 = 2;
const BLOCK_SIZE: u32 = 4;
const NUM_PHYS_BLOCKS: u32 = 3;
const KV_LEN: u32 = 10; // logical blocks 0,1,2 -> positions 0-3,4-7,8-9

// Source value at physical (block, kv_head, tib, dim). <= 191, exact in f16.
fn enc(block: u32, kv_head: u32, tib: u32, dim: u32) -> f32 {
    (((block * NUM_KV + kv_head) * BLOCK_SIZE + tib) * HEAD_DIM + dim) as f32
}

fn shared_buffer_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> objc2::rc::Retained<ProtocolObject<dyn MTLBuffer>> {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes")
    }
}

#[test]
fn dense_gather_kvmajor_copy_matches_reference() {
    let Some(device) = detect_device() else {
        eprintln!("skipping: no Metal 4 GPU");
        return;
    };
    let dev = &device.device;

    // Compile the kernel from source (bfloat in-file needs default opts).
    let src = include_str!("../shaders/attention_dense_gather.metal");
    let opts = objc2_metal::MTLCompileOptions::new();
    let library = dev
        .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
        .expect("compile attention_dense_gather.metal");

    // Function constants used by the COPY kernel: 0=head_dim,1=num_kv,2=block_size,3=blocks_per_chunk.
    let constants = MTLFunctionConstantValues::new();
    let blocks_per_chunk: u32 = 0; // single chunk
    unsafe {
        for (val, idx) in [
            (HEAD_DIM, 0u32),
            (NUM_KV, 1),
            (BLOCK_SIZE, 2),
            (blocks_per_chunk, 3),
            (KV_LEN, 6), // GDK_MAX_KV stride == kv_len for this test's packed dst
        ] {
            constants.setConstantValue_type_atIndex(
                NonNull::new(&val as *const u32 as *mut c_void).unwrap(),
                MTLDataType::UInt,
                idx as usize,
            );
        }
    }
    let func = library
        .newFunctionWithName_constantValues_error(
            &NSString::from_str("gather_dense_kvmajor_copy_f16"),
            &constants,
        )
        .expect("specialize gather_dense_kvmajor_copy_f16");
    let pipeline = dev
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline");

    // Paged K buffer: [phys_block, kv_head, tib, dim], block-major.
    let kv_blk_stride = (NUM_KV * BLOCK_SIZE * HEAD_DIM) as usize;
    let mut k_data = vec![0u16; NUM_PHYS_BLOCKS as usize * kv_blk_stride];
    for p in 0..NUM_PHYS_BLOCKS {
        for h in 0..NUM_KV {
            for t in 0..BLOCK_SIZE {
                for d in 0..HEAD_DIM {
                    let off = p as usize * kv_blk_stride
                        + (h * BLOCK_SIZE * HEAD_DIM + t * HEAD_DIM + d) as usize;
                    k_data[off] = f16::from_f32(enc(p, h, t, d)).to_bits();
                }
            }
        }
    }
    let k_bytes: &[u8] = bytemuck_cast(&k_data);
    let k_buf = shared_buffer_bytes(dev, k_bytes);

    // cache[0] = gpuAddress of the K buffer (single chunk).
    let cache_addr: u64 = k_buf.gpuAddress();
    let cache_buf = shared_buffer_bytes(dev, &cache_addr.to_ne_bytes());

    // block_table: logical -> physical with a non-identity permutation.
    let block_table: [u32; NUM_PHYS_BLOCKS as usize] = [2, 0, 1];
    let bt_bytes: &[u8] = bytemuck_cast(&block_table);
    let bt_buf = shared_buffer_bytes(dev, bt_bytes);

    // seq_used = [kv_len]
    let seq_used = [KV_LEN];
    let seq_buf = shared_buffer_bytes(dev, bytemuck_cast(&seq_used));

    // Dest: [num_kv, kv_len, head_dim], init to sentinel to catch unwritten.
    let dst_len = (NUM_KV * KV_LEN * HEAD_DIM) as usize;
    let dst_init = vec![f16::from_f32(-1.0).to_bits(); dst_len];
    let dst_buf = shared_buffer_bytes(dev, bytemuck_cast(&dst_init));

    // Dispatch on the production MTL4 path: grid (kv_len, num_kv, 1), one
    // thread per (pos, kv_head). Bindings: buffer(0)=dst, buffer(1)=block_table,
    // buffer(2)=cache (chunk pointers), buffer(3)=seq_used.
    if !common::dispatch_threadgroups(
        dev,
        &pipeline,
        &[&dst_buf, &bt_buf, &cache_buf, &seq_buf],
        MTLSize {
            width: KV_LEN as usize,
            height: NUM_KV as usize,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    ) {
        return; // no MTL4 queue on this host
    }

    // Verify against the reference.
    let out: &[u16] =
        unsafe { std::slice::from_raw_parts(dst_buf.contents().as_ptr() as *const u16, dst_len) };
    for pos in 0..KV_LEN {
        let logical = pos / BLOCK_SIZE;
        let tib = pos % BLOCK_SIZE;
        let physical = block_table[logical as usize];
        for h in 0..NUM_KV {
            for d in 0..HEAD_DIM {
                let idx = (h * KV_LEN * HEAD_DIM + pos * HEAD_DIM + d) as usize;
                let got = f16::from_bits(out[idx]).to_f32();
                let want = enc(physical, h, tib, d);
                assert_eq!(
                    got, want,
                    "mismatch at kv_head={h} pos={pos} dim={d}: got {got} want {want}"
                );
            }
        }
    }
}

// Minimal byte-cast helper (avoid a bytemuck dep assumption).
fn bytemuck_cast<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}
