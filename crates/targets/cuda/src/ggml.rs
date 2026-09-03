// SPDX-License-Identifier: Apache-2.0
//! GGML quantized types and CUDA kernel wrappers for GGUF inference.
//!
//! This module provides:
//! - `GgmlDType`: enum of GGML quantization types with `type_size()` / `block_size()`
//! - `GgmlStorage`: raw quantized bytes on GPU + metadata
//! - FFI wrappers calling the llama.cpp-derived kernels in `csrc/quantized.cu`
//!
//! Launch configs match llama.cpp's quantized kernel configs.

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use std::sync::atomic::{AtomicBool, Ordering};

type CUstream = cudarc::driver::sys::CUstream;

// One flag per IQ type — set on first dispatch, so the log line prints exactly once.
static IQ1M_SEEN: AtomicBool = AtomicBool::new(false);
static IQ1S_SEEN: AtomicBool = AtomicBool::new(false);
static IQ2XXS_SEEN: AtomicBool = AtomicBool::new(false);
static IQ2XS_SEEN: AtomicBool = AtomicBool::new(false);
static IQ2S_SEEN: AtomicBool = AtomicBool::new(false);
static IQ3S_SEEN: AtomicBool = AtomicBool::new(false);

fn note_iq(flag: &AtomicBool, name: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        eprintln!("[ggml dispatch] first matmul via {name}");
    }
}

// ---------------------------------------------------------------------------
// GgmlDType + GgmlStorage live in scratchy-target-cuda::ggml_quant so
// `GpuWeights` can hold a `quantized: HashMap<String, GgmlStorage>`
// field without a circular crate dependency. Re-exported here for
// backward compat with all existing call sites.
// ---------------------------------------------------------------------------
pub use crate::ggml_quant::{GgmlDType, GgmlStorage};

// ---------------------------------------------------------------------------
// Constants (match llama.cpp quantized kernels)
// ---------------------------------------------------------------------------

pub const MATRIX_ROW_PADDING: usize = 512;

pub fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

// ---------------------------------------------------------------------------
// FFI declarations — host-side launch wrappers from quantized.cu
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // --- dequantize_mul_mat_vec wrappers ---
    fn launch_dequantize_mul_mat_vec_q4_0(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q4_1(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q5_0(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q5_1(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q8_0(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q2_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q3_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q4_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q5_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );
    fn launch_dequantize_mul_mat_vec_q6_k(
        vx: *const u8,
        y: *const f32,
        dst: *mut f32,
        ncols: i32,
        nrows: i32,
        stream: CUstream,
    );

    // --- mul_mat_vec Q*×Q8_1 wrappers (BS=1) ---
    fn launch_mul_mat_vec_q4_0_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q4_1_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q5_0_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q5_1_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q8_0_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q2_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q3_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q4_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q5_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_q6_K_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );

    // --- quantize activations to Q8_1 ---
    fn launch_quantize_q8_1(
        src: *const f32,
        dst: *mut u8,
        k: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: CUstream,
    );

    // --- dequantize to f32 ---
    fn launch_dequantize_block_q4_0_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_1_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_0_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_1_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_0_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q2_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q3_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q6_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_K_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );

    // --- dequantize to f16 ---
    fn launch_dequantize_block_q4_0_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_1_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_0_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_1_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_0_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q2_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q3_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q4_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q5_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q6_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_q8_K_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ4 mul_mat_vec Q*×Q8_1 wrappers ---
    fn launch_mul_mat_vec_iq4_nl_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_mul_mat_vec_iq4_xs_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );

    // --- IQ4 dequantize to f32 ---
    fn launch_dequantize_block_iq4_nl_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq4_xs_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ4 dequantize to f16 ---
    fn launch_dequantize_block_iq4_nl_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq4_xs_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ1_M mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq1_m_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_m_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_m_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ1_S mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq1_s_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_s_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq1_s_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ2_XXS mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq2_xxs_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_xxs_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_xxs_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ2_XS mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq2_xs_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_xs_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_xs_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ2_S mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq2_s_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_s_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq2_s_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- IQ3_S mul_mat_vec + dequantize wrappers ---
    fn launch_mul_mat_vec_iq3_s_q8_1(
        vx: *const u8,
        vy: *const u8,
        dst: *mut f32,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        ncols_y: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq3_s_f32(
        vx: *const u8,
        dst: *mut f32,
        elem_count: i32,
        stream: CUstream,
    );
    fn launch_dequantize_block_iq3_s_f16(
        vx: *const u8,
        dst: *mut u16,
        elem_count: i32,
        stream: CUstream,
    );

    // --- indexed_moe_forward wrappers ---
    fn launch_indexed_moe_forward_q2k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q3k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q4k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q5k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q6k_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q8_0_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q4_0_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q4_1_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q5_0_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
    fn launch_indexed_moe_forward_q5_1_q8_1(
        all_weights: *const u8,
        all_inputs: *const u8,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: CUstream,
    );
}

// ---------------------------------------------------------------------------
// Kernel launch wrappers
// ---------------------------------------------------------------------------

/// Fused dequantize + matrix-vector multiply for a single input row (BS=1).
///
/// Computes: `dst[nrows] = weight[nrows, ncols] @ x[ncols]`
/// where `weight` is in GGML quantized format.
///
/// # Safety
/// All pointers must be valid GPU memory.
pub unsafe fn ggml_dequant_mul_mat_vec(
    storage: &GgmlStorage,
    x: *const f32,
    dst: *mut f32,
    stream: CUstream,
) {
    let ncols = storage.ncols as i32;
    let nrows = storage.nrows as i32;
    let vx = storage.ptr as *const u8;

    match storage.dtype {
        GgmlDType::Q4_0 => launch_dequantize_mul_mat_vec_q4_0(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q4_1 => launch_dequantize_mul_mat_vec_q4_1(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q5_0 => launch_dequantize_mul_mat_vec_q5_0(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q5_1 => launch_dequantize_mul_mat_vec_q5_1(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q8_0 => launch_dequantize_mul_mat_vec_q8_0(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q2K => launch_dequantize_mul_mat_vec_q2_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q3K => launch_dequantize_mul_mat_vec_q3_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q4K => launch_dequantize_mul_mat_vec_q4_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q5K => launch_dequantize_mul_mat_vec_q5_k(vx, x, dst, ncols, nrows, stream),
        GgmlDType::Q6K => launch_dequantize_mul_mat_vec_q6_k(vx, x, dst, ncols, nrows, stream),
        dt if dt.is_iq_quant() => panic!(
            "IQ types must use Q8_1 path, not dequant_mul_mat_vec: {}",
            storage.dtype
        ),
        _ => panic!(
            "unsupported dtype for dequant_mul_mat_vec: {}",
            storage.dtype
        ),
    }
}

/// Dequantize GGML data to f32 on GPU.
///
/// # Safety
/// `src` must point to valid quantized GPU data, `dst` must have room for `elem_count` f32s.
pub unsafe fn ggml_dequantize_f32(
    src: *const u8,
    dst: *mut f32,
    dtype: GgmlDType,
    elem_count: usize,
    stream: CUstream,
) {
    let n = elem_count as i32;
    match dtype {
        GgmlDType::Q4_0 => launch_dequantize_block_q4_0_f32(src, dst, n, stream),
        GgmlDType::Q4_1 => launch_dequantize_block_q4_1_f32(src, dst, n, stream),
        GgmlDType::Q5_0 => launch_dequantize_block_q5_0_f32(src, dst, n, stream),
        GgmlDType::Q5_1 => launch_dequantize_block_q5_1_f32(src, dst, n, stream),
        GgmlDType::Q8_0 => launch_dequantize_block_q8_0_f32(src, dst, n, stream),
        GgmlDType::Q2K => launch_dequantize_block_q2_K_f32(src, dst, n, stream),
        GgmlDType::Q3K => launch_dequantize_block_q3_K_f32(src, dst, n, stream),
        GgmlDType::Q4K => launch_dequantize_block_q4_K_f32(src, dst, n, stream),
        GgmlDType::Q5K => launch_dequantize_block_q5_K_f32(src, dst, n, stream),
        GgmlDType::Q6K => launch_dequantize_block_q6_K_f32(src, dst, n, stream),
        GgmlDType::Q8K => launch_dequantize_block_q8_K_f32(src, dst, n, stream),
        GgmlDType::IQ4NL => launch_dequantize_block_iq4_nl_f32(src, dst, n, stream),
        GgmlDType::IQ4XS => launch_dequantize_block_iq4_xs_f32(src, dst, n, stream),
        GgmlDType::IQ1M => launch_dequantize_block_iq1_m_f32(src, dst, n, stream),
        GgmlDType::IQ1S => launch_dequantize_block_iq1_s_f32(src, dst, n, stream),
        GgmlDType::IQ2XXS => launch_dequantize_block_iq2_xxs_f32(src, dst, n, stream),
        GgmlDType::IQ2XS => launch_dequantize_block_iq2_xs_f32(src, dst, n, stream),
        GgmlDType::IQ2S => launch_dequantize_block_iq2_s_f32(src, dst, n, stream),
        GgmlDType::IQ3S => launch_dequantize_block_iq3_s_f32(src, dst, n, stream),
        _ => panic!("unsupported dtype for dequantize_f32: {}", dtype),
    }
}

/// Dequantize GGML data to f16 on GPU.
///
/// # Safety
/// Same requirements as `ggml_dequantize_f32`.
pub unsafe fn ggml_dequantize_f16(
    src: *const u8,
    dst: *mut u16,
    dtype: GgmlDType,
    elem_count: usize,
    stream: CUstream,
) {
    let n = elem_count as i32;
    match dtype {
        GgmlDType::Q4_0 => launch_dequantize_block_q4_0_f16(src, dst, n, stream),
        GgmlDType::Q4_1 => launch_dequantize_block_q4_1_f16(src, dst, n, stream),
        GgmlDType::Q5_0 => launch_dequantize_block_q5_0_f16(src, dst, n, stream),
        GgmlDType::Q5_1 => launch_dequantize_block_q5_1_f16(src, dst, n, stream),
        GgmlDType::Q8_0 => launch_dequantize_block_q8_0_f16(src, dst, n, stream),
        GgmlDType::Q2K => launch_dequantize_block_q2_K_f16(src, dst, n, stream),
        GgmlDType::Q3K => launch_dequantize_block_q3_K_f16(src, dst, n, stream),
        GgmlDType::Q4K => launch_dequantize_block_q4_K_f16(src, dst, n, stream),
        GgmlDType::Q5K => launch_dequantize_block_q5_K_f16(src, dst, n, stream),
        GgmlDType::Q6K => launch_dequantize_block_q6_K_f16(src, dst, n, stream),
        GgmlDType::Q8K => launch_dequantize_block_q8_K_f16(src, dst, n, stream),
        GgmlDType::IQ4NL => launch_dequantize_block_iq4_nl_f16(src, dst, n, stream),
        GgmlDType::IQ4XS => launch_dequantize_block_iq4_xs_f16(src, dst, n, stream),
        GgmlDType::IQ1M => launch_dequantize_block_iq1_m_f16(src, dst, n, stream),
        GgmlDType::IQ1S => launch_dequantize_block_iq1_s_f16(src, dst, n, stream),
        GgmlDType::IQ2XXS => launch_dequantize_block_iq2_xxs_f16(src, dst, n, stream),
        GgmlDType::IQ2XS => launch_dequantize_block_iq2_xs_f16(src, dst, n, stream),
        GgmlDType::IQ2S => launch_dequantize_block_iq2_s_f16(src, dst, n, stream),
        GgmlDType::IQ3S => launch_dequantize_block_iq3_s_f16(src, dst, n, stream),
        _ => panic!("unsupported dtype for dequantize_f16: {}", dtype),
    }
}

/// Dequantize GGML data to a `GpuTensor` in the target dtype.
///
/// Allocates output via `CachingAllocator`. Used at load time for norms and embeddings.
///
/// # Safety
/// Requires valid CUDA context. `storage` must reference valid GPU memory.
pub unsafe fn ggml_dequantize_to_tensor(
    storage: &GgmlStorage,
    target_dtype: DType,
    shape: &[usize],
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let elem_count = shape.iter().product::<usize>();
    let out = alloc.alloc_tensor(shape, target_dtype);
    let dst_ptr = out.as_gpu_tensor().raw_ptr();

    match target_dtype {
        DType::F32 => {
            ggml_dequantize_f32(
                storage.ptr,
                dst_ptr as *mut f32,
                storage.dtype,
                elem_count,
                stream,
            );
        }
        DType::F16 => {
            ggml_dequantize_f16(
                storage.ptr,
                dst_ptr as *mut u16,
                storage.dtype,
                elem_count,
                stream,
            );
        }
        DType::BF16 => {
            // No direct quant→BF16 kernel. Dequant to F32, then cast F32→BF16.
            let f32_tmp = alloc.alloc_tensor(shape, DType::F32);
            ggml_dequantize_f32(
                storage.ptr,
                f32_tmp.as_gpu_tensor().raw_ptr() as *mut f32,
                storage.dtype,
                elem_count,
                stream,
            );
            crate::kernels::cast_from_f32_into(
                f32_tmp.as_gpu_tensor().raw_ptr() as *const f32,
                out.as_gpu_tensor().raw_ptr(),
                DType::BF16,
                elem_count,
                stream,
            );
            drop(f32_tmp);
        }
        _ => panic!(
            "unsupported target dtype for dequantize: {:?}",
            target_dtype
        ),
    }

    out
}

/// Quantize f32 activations to Q8_1 format on GPU.
///
/// Returns: `(gpu_ptr, total_bytes)` for the Q8_1 buffer.
///
/// # Safety
/// Valid CUDA context and GPU pointers required.
pub unsafe fn ggml_quantize_q8_1_alloc(
    src: *const f32,
    ncols: usize,
    num_rows: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> (*mut u8, usize) {
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let num_blocks_per_row = ncols_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let total_bytes = num_rows * dst_row_size_bytes;

    let dst_ptr = alloc.alloc(total_bytes);

    launch_quantize_q8_1(
        src,
        dst_ptr,
        ncols as i32,
        ncols_padded as i32,
        num_rows as i32,
        stream,
    );

    (dst_ptr, total_bytes)
}

/// Batched quantized matvec via Q8_1 intermediate quantization (BS=1 per call).
///
/// # Safety
/// All pointers must be valid GPU memory.
pub unsafe fn ggml_mul_mat_vec_q8_1(
    storage: &GgmlStorage,
    y_q8_1: *const u8,
    ncols_padded: usize,
    batch_size: usize,
    dst: *mut f32,
    stream: CUstream,
) {
    let ncols_x = storage.ncols as i32;
    let nrows_x = storage.nrows as i32;
    let nrows_y = ncols_padded as i32;
    let nrows_dst = nrows_x;
    let vx = storage.ptr as *const u8;

    // Batched: call the kernel once with ncols_y = min(batch_size, MAX), then loop for the rest.
    // All wired Q and IQ types now have cuda1..cuda8 variants.
    let q8_1_row_bytes =
        (ncols_padded / GgmlDType::Q8_1.block_size()) * GgmlDType::Q8_1.type_size();
    let dst_row_elems = storage.nrows;
    let max_ncols_y: usize = 8;
    let mut b = 0;
    while b < batch_size {
        let chunk = (batch_size - b).min(max_ncols_y);
        let y_offset = y_q8_1.add(b * q8_1_row_bytes);
        let dst_offset = dst.add(b * dst_row_elems);
        let n = chunk as i32;
        match storage.dtype {
            GgmlDType::Q4_0 => launch_mul_mat_vec_q4_0_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q4_1 => launch_mul_mat_vec_q4_1_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q5_0 => launch_mul_mat_vec_q5_0_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q5_1 => launch_mul_mat_vec_q5_1_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q8_0 => launch_mul_mat_vec_q8_0_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q2K => launch_mul_mat_vec_q2_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q3K => launch_mul_mat_vec_q3_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q4K => launch_mul_mat_vec_q4_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q5K => launch_mul_mat_vec_q5_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::Q6K => launch_mul_mat_vec_q6_K_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::IQ4NL => launch_mul_mat_vec_iq4_nl_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::IQ4XS => launch_mul_mat_vec_iq4_xs_q8_1(
                vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
            ),
            GgmlDType::IQ1M => {
                note_iq(&IQ1M_SEEN, "IQ1_M");
                launch_mul_mat_vec_iq1_m_q8_1(
                    vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
                );
            }
            GgmlDType::IQ1S => {
                note_iq(&IQ1S_SEEN, "IQ1_S");
                launch_mul_mat_vec_iq1_s_q8_1(
                    vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
                );
            }
            GgmlDType::IQ2XXS => {
                note_iq(&IQ2XXS_SEEN, "IQ2_XXS");
                launch_mul_mat_vec_iq2_xxs_q8_1(
                    vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
                );
            }
            GgmlDType::IQ2XS => {
                note_iq(&IQ2XS_SEEN, "IQ2_XS");
                launch_mul_mat_vec_iq2_xs_q8_1(
                    vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
                );
            }
            GgmlDType::IQ2S => {
                note_iq(&IQ2S_SEEN, "IQ2_S");
                launch_mul_mat_vec_iq2_s_q8_1(
                    vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
                );
            }
            GgmlDType::IQ3S => {
                note_iq(&IQ3S_SEEN, "IQ3_S");
                launch_mul_mat_vec_iq3_s_q8_1(
                    vx, y_offset, dst_offset, ncols_x, nrows_x, nrows_y, nrows_dst, n, stream,
                );
            }
            _ => panic!("unsupported dtype for mul_mat_vec_q8_1: {}", storage.dtype),
        }
        b += chunk;
    }
}

/// High-level quantized matrix-vector multiply.
///
/// `x`: f32 `[num_tokens, ncols]` (GPU)
/// Returns: f32 `[num_tokens, nrows]` (GPU, allocated from `alloc`)
///
/// For num_tokens==1: uses fused dequant+dot.
/// For num_tokens>1: quantizes activations to Q8_1, then int-dot path.
///
/// # Safety
/// Valid CUDA context, valid GPU pointers.
pub unsafe fn ggml_matmul(
    storage: &GgmlStorage,
    x: GpuTensor,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    let num_tokens = x.dim(0);
    debug_assert_eq!(x.dim(1), storage.ncols);
    assert_eq!(
        x.dtype(),
        DType::F32,
        "GGML matmul requires f32 activations, got {:?}",
        x.dtype()
    );

    let out = alloc.alloc_tensor(&[num_tokens, storage.nrows], DType::F32);
    let dst = out.as_gpu_tensor().raw_ptr() as *mut f32;

    if num_tokens == 1 && !storage.dtype.is_iq_quant() {
        // Fast path: fused dequant+dot for standard quant types at BS=1.
        ggml_dequant_mul_mat_vec(storage, x.as_ptr::<f32>(), dst, stream);
    } else {
        // IQ types always use Q8_1 path (no fused dequant+dot kernel).
        // Standard types use Q8_1 path for BS>1.
        let ncols_padded = pad(storage.ncols, MATRIX_ROW_PADDING);
        let (q8_ptr, _q8_bytes) =
            ggml_quantize_q8_1_alloc(x.as_ptr::<f32>(), storage.ncols, num_tokens, alloc, stream);
        ggml_mul_mat_vec_q8_1(storage, q8_ptr, ncols_padded, num_tokens, dst, stream);
    }

    out
}

/// Indexed MoE forward: quantized expert weights × Q8_1 inputs → f32 outputs.
///
/// - `storage`: 3D quantized expert weights `[num_experts, n, k]` flattened into GgmlStorage
///   where nrows = num_experts * n, ncols = k.
/// - `q8_input`: Q8_1-quantized input, layout depends on `input_dim1`:
///   - `input_dim1 == 1`: `[batch, k_padded]` (shared across topk per batch item)
///   - `input_dim1 != 1`: `[batch * topk, k_padded]` (unique per task)
/// - `indices`: `[batch * topk]` u32 expert indices.
/// - `output`: `[batch * topk, n]` f32 output buffer.
/// - `n`: output features per expert (nrows per expert).
/// - `k`: input features per expert (ncols).
/// - `batch`: batch size.
/// - `topk`: number of experts per token.
/// - `k_padded`: padded input dimension (for Q8_1 alignment).
/// - `input_dim1`: controls input sharing. 1 = all topk experts for a batch item share
///   the same input row. Otherwise each task_id indexes a unique input row.
///
/// # Safety
/// All pointers must be valid GPU memory. `storage.dtype` must be a supported MoE quant type.
pub unsafe fn ggml_moe_forward(
    storage: &GgmlStorage,
    q8_input: *const u8,
    indices: *const u32,
    output: *mut f32,
    n: usize,
    k: usize,
    batch: usize,
    topk: usize,
    k_padded: usize,
    input_dim1: usize,
    stream: CUstream,
) {
    let n_i = n as i32;
    let k_i = k as i32;
    let batch_i = batch as i32;
    let topk_i = topk as i32;
    let k_padded_i = k_padded as i32;
    let input_dim1_i = input_dim1 as i32;
    let vx = storage.ptr as *const u8;

    match storage.dtype {
        GgmlDType::Q2K => launch_indexed_moe_forward_q2k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q3K => launch_indexed_moe_forward_q3k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q4K => launch_indexed_moe_forward_q4k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q5K => launch_indexed_moe_forward_q5k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q6K => launch_indexed_moe_forward_q6k_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q8_0 => launch_indexed_moe_forward_q8_0_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q4_0 => launch_indexed_moe_forward_q4_0_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q4_1 => launch_indexed_moe_forward_q4_1_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q5_0 => launch_indexed_moe_forward_q5_0_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        GgmlDType::Q5_1 => launch_indexed_moe_forward_q5_1_q8_1(
            vx,
            q8_input,
            indices,
            output,
            n_i,
            k_i,
            batch_i,
            topk_i,
            k_padded_i,
            input_dim1_i,
            stream,
        ),
        _ => panic!("unsupported dtype for ggml_moe_forward: {}", storage.dtype),
    }
}

// ---------------------------------------------------------------------------
// GGUF weight loading — raw quantized bytes from GGUF → GPU
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// GGUF tensor descriptor (parsed from header, before loading to GPU).
pub struct GgufTensorInfo {
    pub hf_name: String,
    pub ggml_dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size_bytes: usize,
}

/// Loaded GGUF weights on GPU — either quantized (`GgmlStorage`) or dequantized (`GpuTensor`).
pub enum GgufWeight {
    /// Quantized weight (linear layers).
    Quantized(GgmlStorage),
    /// Dequantized weight (norms, embeddings).
    Dense(GpuTensor),
}

/// GGUF weight store: loads all tensors from a GGUF file onto GPU.
///
/// Linear weights stay quantized (raw bytes → `GgmlStorage`).
/// Norms are dequantized to f32. Embeddings are dequantized to the model dtype.
pub struct GgufGpuWeights {
    weights: HashMap<String, GgufWeight>,
}

impl GgufGpuWeights {
    /// Load all tensors from a GGUF file onto GPU.
    ///
    /// At `tp_world_size > 1` per-tensor block-aligned slicing kicks
    /// in based on `gguf_shard_kind_for_hf_name` (mirrors the
    /// safetensors codegen's `tp_lowering` rule table):
    ///
    /// - `ShardDim0` (q/k/v/gate/up/embed/lm_head): rows split. Each
    ///   row is a whole number of GGML blocks, so the per-rank slice
    ///   is a contiguous byte range — a single seek+read.
    /// - `ShardDim1` (o/down): per-row column slice. Refuse-at-load
    ///   if `(in_features / tp) % block_size != 0`.
    /// - `Replicate` (norms): full tensor on every rank.
    ///
    /// 3D MoE experts and 1D weights are always replicated.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn load(
        path: &Path,
        model_dtype: DType,
        alloc: &mut CachingAllocator,
        stream: CUstream,
        tp_rank: usize,
        tp_world_size: usize,
    ) -> anyhow::Result<Self> {
        use scratchy_quantizations::gguf::Content;

        if tp_world_size == 0 {
            anyhow::bail!("tp_world_size must be >= 1");
        }
        if tp_rank >= tp_world_size {
            anyhow::bail!("tp_rank ({tp_rank}) >= tp_world_size ({tp_world_size})");
        }

        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let content =
            Content::read(&mut reader).map_err(|e| anyhow::anyhow!("GGUF parse error: {e}"))?;

        let tensor_data_offset = content.tensor_data_offset;
        let mut weights = HashMap::new();

        // Allocate pinned host buffer for H2D transfers (reuse for all tensors).
        let max_tensor_bytes = content
            .tensor_infos
            .values()
            .map(|info| {
                let elems = info.shape.elem_count();
                let bs = info.ggml_dtype.block_size();
                (elems / bs) * info.ggml_dtype.type_size()
            })
            .max()
            .unwrap_or(0);
        let host_buf = crate::driver::mem_alloc_host(max_tensor_bytes)?;

        // GGML's RoPE convention pre-permutes q_proj / k_proj rows
        // (interleaved pairs) at file-write time. Per-arch decision —
        // each arch declares
        // `qk_permute: true|false` in its `configs/quantizations.json`
        // ggml entry, and that flag flows through the
        // `scratchy_quantizations::gguf::register!` block the macro emits.
        let qk_arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned());
        let qk_permuted = qk_arch
            .as_deref()
            .and_then(scratchy_quantizations::gguf::find_spec)
            .map(|s| s.qk_permute)
            .unwrap_or(false);
        let qk_meta: Option<(usize, usize, usize)> = qk_arch.as_deref().and_then(|arch| {
            let head_count = content
                .metadata
                .get(&format!("{arch}.attention.head_count"))?
                .to_u32()
                .ok()? as usize;
            let head_count_kv = content
                .metadata
                .get(&format!("{arch}.attention.head_count_kv"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(head_count);
            let key_length = content
                .metadata
                .get(&format!("{arch}.attention.key_length"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .or_else(|| {
                    let hidden = content
                        .metadata
                        .get(&format!("{arch}.embedding_length"))?
                        .to_u32()
                        .ok()? as usize;
                    Some(hidden / head_count)
                })?;
            Some((head_count, head_count_kv, key_length))
        });

        let gguf_arch_str = qk_arch.as_deref().unwrap_or("");
        // Per-arch baked offset on rmsnorm weights (Gemma2/3 store `1+w`
        // so a vanilla `rmsnorm(x, w)` matches HF's `rmsnorm(x, 1+w)`).
        // Subtracted from each F32 norm element below so the loaded
        // weight is canonical raw `w`; scratchy's runtime kernel then
        // applies its own `(w + offset)` exactly once.
        let norm_baked_offset: f32 = scratchy_quantizations::gguf::find_spec(gguf_arch_str)
            .map(|s| s.norm_weight_offset)
            .unwrap_or(0.0);
        for (gguf_name, info) in &content.tensor_infos {
            let hf_name = scratchy_quantizations::gguf::gguf_to_hf_name(gguf_name, gguf_arch_str);
            // `gguf_format` already reverses the on-disk ggml dim
            // order to HF's [rows, cols] = [out, in] convention, so
            // `info.shape.dims()` is already row-major-friendly here.
            let dims_full = info.shape.dims();
            let gguf_dtype = info.ggml_dtype;

            // Compute per-rank shard. Returns
            //   (sliced_dims, sliced_elem_count, sliced_size_bytes,
            //    Vec<(byte_offset_in_file, byte_count)>)
            // where the offsets are RELATIVE to the tensor's data start
            // (i.e. tensor_data_offset + info.offset).
            let shard_kind = if dims_full.len() == 2 && tp_world_size > 1 {
                gguf_shard_kind_for_hf_name(&hf_name)
            } else {
                GgufShardKind::Replicate
            };
            let bs = gguf_dtype.block_size();
            let ts = gguf_dtype.type_size();
            let (dims, slice_reads, sliced_elem_count, sliced_size_bytes) = match shard_kind {
                GgufShardKind::Replicate => {
                    let elems = info.shape.elem_count();
                    let sb = (elems / bs) * ts;
                    (dims_full.to_vec(), vec![(0u64, sb)], elems, sb)
                }
                GgufShardKind::ShardDim0 => {
                    // 2D row-major: dims = [out, in]. Per-rank rows.
                    let total_rows = dims_full[0];
                    let cols = dims_full[1];
                    if total_rows % tp_world_size != 0 {
                        anyhow::bail!(
                            "tp shard: tensor `{}` ShardDim0 rows={} not divisible by \
                             tp_world_size={}",
                            hf_name,
                            total_rows,
                            tp_world_size
                        );
                    }
                    let rows_per_rank = total_rows / tp_world_size;
                    if cols % bs != 0 {
                        anyhow::bail!(
                            "tp shard: tensor `{}` cols={} not a multiple of block_size={} \
                             (dtype {:?})",
                            hf_name,
                            cols,
                            bs,
                            gguf_dtype
                        );
                    }
                    let row_bytes = (cols / bs) * ts;
                    let row_offset = tp_rank * rows_per_rank;
                    let byte_offset = (row_offset * row_bytes) as u64;
                    let slice_bytes = rows_per_rank * row_bytes;
                    (
                        vec![rows_per_rank, cols],
                        vec![(byte_offset, slice_bytes)],
                        rows_per_rank * cols,
                        slice_bytes,
                    )
                }
                GgufShardKind::ShardDim1 => {
                    // 2D row-major: dims = [out, in]. Per-rank cols.
                    let rows = dims_full[0];
                    let total_cols = dims_full[1];
                    if total_cols % tp_world_size != 0 {
                        anyhow::bail!(
                            "tp shard: tensor `{}` ShardDim1 cols={} not divisible by \
                             tp_world_size={}",
                            hf_name,
                            total_cols,
                            tp_world_size
                        );
                    }
                    let cols_per_rank = total_cols / tp_world_size;
                    if !cols_per_rank.is_multiple_of(bs) {
                        anyhow::bail!(
                            "tp shard: tensor `{}` per-rank cols={} not a multiple of \
                             block_size={} (dtype {:?}) — refuse-at-load",
                            hf_name,
                            cols_per_rank,
                            bs,
                            gguf_dtype
                        );
                    }
                    let row_bytes_full = (total_cols / bs) * ts;
                    let row_bytes_local = (cols_per_rank / bs) * ts;
                    let col_byte_offset = tp_rank * row_bytes_local;
                    // Per-row strided reads: one (offset, size) per row.
                    let reads: Vec<(u64, usize)> = (0..rows)
                        .map(|r| {
                            (
                                (r * row_bytes_full + col_byte_offset) as u64,
                                row_bytes_local,
                            )
                        })
                        .collect();
                    (
                        vec![rows, cols_per_rank],
                        reads,
                        rows * cols_per_rank,
                        rows * row_bytes_local,
                    )
                }
            };
            let dims = dims.as_slice();
            let elem_count = sliced_elem_count;
            let size_bytes = sliced_size_bytes;

            // Map GGUF dtype tag to our quantized GgmlDType (None for float types).
            let our_dtype = GgmlDType::from_gguf(gguf_dtype);

            // Determine if this is a norm or embedding (dequantize) vs linear (keep quantized).
            // Layer norms (RmsNorm) → f32; QK norms → model dtype; embeddings → model dtype.
            let is_layer_norm = hf_name.contains("layernorm")
                || hf_name.ends_with("model.norm.weight")
                || (hf_name.contains("norm.weight")
                    && !hf_name.contains("q_norm")
                    && !hf_name.contains("k_norm"));
            let is_qk_norm = hf_name.contains("q_norm.weight") || hf_name.contains("k_norm.weight");
            let is_norm = is_layer_norm || is_qk_norm;
            let is_embedding = hf_name == "model.embed_tokens.weight";
            let is_lm_head = hf_name == "lm_head.weight";
            let is_f32_or_f16 = gguf_dtype.is_float();

            // Read per-rank bytes from disk into the pinned host
            // buffer. Replicate / ShardDim0 = single read; ShardDim1
            // = per-row strided reads concatenated.
            let mut written = 0usize;
            for (rel_off, n) in &slice_reads {
                reader.seek(SeekFrom::Start(tensor_data_offset + info.offset + *rel_off))?;
                let host_slice = std::slice::from_raw_parts_mut(host_buf.add(written), *n);
                reader.read_exact(host_slice)?;
                written += *n;
            }
            debug_assert_eq!(written, size_bytes);
            let _ = elem_count; // silence unused if no downstream use

            // GGML→HF row un-permute for q_proj / k_proj. llama.cpp's
            // `convert_hf_to_gguf` reshapes [n_heads, head_dim, in]
            // as [n_heads, 2, head_dim/2, in] and swaps the middle
            // axes — i.e. interleaves "first half" and "second half"
            // of head_dim into pairs. scratchy's RoPE follows the HF
            // split-halves convention, so we need to reverse that
            // permutation by re-interleaving rows. The permutation
            // operates strictly on rows (dim 0), so block-quantized
            // rows can be moved bytewise without touching the
            // intra-row block layout.
            //
            // ShardDim0 keeps each rank's row range head-aligned
            // (verified upstream by `rows_per_rank % head_dim == 0`)
            // so the permutation is well-defined per-rank.
            let is_q_proj = gguf_name.ends_with(".attn_q.weight");
            let is_k_proj = gguf_name.ends_with(".attn_k.weight");
            if qk_permuted
                && dims_full.len() == 2
                && (is_q_proj || is_k_proj)
                && let Some((n_heads, n_kv_heads, head_dim)) = qk_meta
            {
                let rows_local = dims[0];
                let row_bytes_local = if dims[1] % bs == 0 {
                    (dims[1] / bs) * ts
                } else {
                    dims[1] * ts
                };
                debug_assert_eq!(rows_local * row_bytes_local, sliced_size_bytes);
                let head_count_for_this = if is_q_proj { n_heads } else { n_kv_heads };
                if rows_local.is_multiple_of(head_dim) && head_dim.is_multiple_of(2) {
                    let heads_local = rows_local / head_dim;
                    let half = head_dim / 2;
                    // tp ShardDim0 may give us only a contiguous
                    // slice of heads; in that case heads_local <
                    // total head_count_for_this and head boundaries
                    // still align. The permutation only mixes rows
                    // within a head, so works on any whole-head set.
                    let _ = head_count_for_this;
                    let mut tmp = vec![0u8; sliced_size_bytes];
                    let src = std::slice::from_raw_parts(host_buf, sliced_size_bytes);
                    for head in 0..heads_local {
                        for pos_hf in 0..head_dim {
                            let r_hf = head * head_dim + pos_hf;
                            let r_ggml = if pos_hf < half {
                                head * head_dim + 2 * pos_hf
                            } else {
                                head * head_dim + 2 * (pos_hf - half) + 1
                            };
                            let dst_off = r_hf * row_bytes_local;
                            let src_off = r_ggml * row_bytes_local;
                            tmp[dst_off..dst_off + row_bytes_local]
                                .copy_from_slice(&src[src_off..src_off + row_bytes_local]);
                        }
                    }
                    let dst = std::slice::from_raw_parts_mut(host_buf, sliced_size_bytes);
                    dst.copy_from_slice(&tmp);
                }
            }

            if is_f32_or_f16 || is_norm || is_embedding || is_lm_head {
                // Dequantize path: for unquantized types, just H2D copy.
                // For quantized norms/embeddings, upload then dequant on GPU.
                if is_f32_or_f16 {
                    // Direct H2D copy of f32/f16/bf16 data.
                    let dtype_size = ts; // 4 for f32, 2 for f16/bf16
                    let source_dtype = if dtype_size == 4 {
                        DType::F32
                    } else if gguf_dtype == scratchy_quantizations::gguf::GgufDType::BF16 {
                        DType::BF16
                    } else {
                        DType::F16
                    };
                    // All norm weights (layer + QK) and linear weights must match the model
                    // dtype so the downstream kernels read the right bits (rms_norm_bf16
                    // expects __nv_bfloat16* weight; reinterpreting f32 as bf16 bytes → NaN).
                    let target_dtype = model_dtype;

                    if source_dtype == target_dtype
                        || (source_dtype == DType::F32 && target_dtype == DType::F32)
                    {
                        let gpu_ptr = crate::driver::mem_alloc(size_bytes)?;
                        crate::driver::memcpy_htod_async(gpu_ptr, host_buf, size_bytes, stream)?;
                        let tensor = GpuTensor::new(gpu_ptr, dims, source_dtype);
                        weights.insert(hf_name, GgufWeight::Dense(tensor));
                    } else {
                        // Need dtype conversion: upload as source, dequant/convert on GPU.
                        // For f32 → bf16/f16: upload f32, then use dequantize (identity for f32 blocks).
                        // Simplest approach: upload f32 to GPU, then convert via a kernel.
                        // Since QK norms are tiny, do CPU conversion.
                        if source_dtype == DType::F32 {
                            let f32_slice =
                                std::slice::from_raw_parts(host_buf as *const f32, elem_count);
                            let out_size = elem_count * target_dtype.size_bytes();
                            let conv_buf = crate::driver::mem_alloc_host(out_size)?;
                            // Subtract per-arch baked offset only on rmsnorm
                            // weights (the QK-norm pair included). Embeddings
                            // / lm_head pass through unchanged.
                            let bake_off = if is_norm { norm_baked_offset } else { 0.0 };
                            match target_dtype {
                                DType::BF16 => {
                                    let out = std::slice::from_raw_parts_mut(
                                        conv_buf as *mut u16,
                                        elem_count,
                                    );
                                    for (i, &v) in f32_slice.iter().enumerate() {
                                        out[i] = half::bf16::from_f32(v - bake_off).to_bits();
                                    }
                                }
                                DType::F16 => {
                                    let out = std::slice::from_raw_parts_mut(
                                        conv_buf as *mut u16,
                                        elem_count,
                                    );
                                    for (i, &v) in f32_slice.iter().enumerate() {
                                        out[i] = half::f16::from_f32(v - bake_off).to_bits();
                                    }
                                }
                                _ => panic!(
                                    "unsupported conversion: {source_dtype:?} -> {target_dtype:?}"
                                ),
                            }
                            let gpu_ptr = crate::driver::mem_alloc(out_size)?;
                            crate::driver::memcpy_htod_async(gpu_ptr, conv_buf, out_size, stream)?;
                            crate::driver::stream_synchronize(stream)?;
                            crate::driver::mem_free_host(conv_buf)?;
                            let tensor = GpuTensor::new(gpu_ptr, dims, target_dtype);
                            weights.insert(hf_name, GgufWeight::Dense(tensor));
                        } else {
                            // Non-f32 source needing conversion (f16↔bf16): upload raw,
                            // cast src → f32 → target via existing kernels.
                            let src_ptr = crate::driver::mem_alloc(size_bytes)?;
                            crate::driver::memcpy_htod_async(
                                src_ptr, host_buf, size_bytes, stream,
                            )?;
                            let src_tensor = GpuTensor::new(src_ptr, dims, source_dtype);
                            let f32_tmp =
                                crate::kernels::cast_logits_to_f32(src_tensor, alloc, stream);
                            let out_bytes = elem_count * target_dtype.size_bytes();
                            let out_ptr = crate::driver::mem_alloc(out_bytes)?;
                            crate::kernels::cast_from_f32_into(
                                f32_tmp.as_gpu_tensor().raw_ptr() as *const f32,
                                out_ptr,
                                target_dtype,
                                elem_count,
                                stream,
                            );
                            crate::driver::stream_synchronize(stream)?;
                            crate::driver::mem_free(src_ptr)?;
                            drop(f32_tmp);
                            let gpu_tensor = GpuTensor::new(out_ptr, dims, target_dtype);
                            weights.insert(hf_name, GgufWeight::Dense(gpu_tensor));
                        }
                    }
                } else if let Some(our_dt) = our_dtype {
                    // Quantized norm/embedding: upload raw bytes, then dequant on GPU.
                    let gpu_raw = crate::driver::mem_alloc(size_bytes)?;
                    crate::driver::memcpy_htod_async(gpu_raw, host_buf, size_bytes, stream)?;
                    crate::driver::stream_synchronize(stream)?;

                    let storage = GgmlStorage {
                        ptr: gpu_raw,
                        len: size_bytes,
                        dtype: our_dt,
                        nrows: if dims.len() >= 2 { dims[0] } else { 1 },
                        ncols: if dims.len() >= 2 { dims[1] } else { dims[0] },
                    };

                    let target = model_dtype;
                    let tensor = ggml_dequantize_to_tensor(&storage, target, dims, alloc, stream);
                    crate::driver::stream_synchronize(stream)?;
                    // Free the raw quantized buffer since we dequantized.
                    crate::driver::mem_free(gpu_raw)?;
                    // Weight tensors are permanent — leak from allocator tracking.
                    let gpu_tensor = tensor.into_gpu_tensor();
                    weights.insert(hf_name, GgufWeight::Dense(gpu_tensor));
                } else {
                    anyhow::bail!(
                        "unsupported GGUF dtype {:?} for tensor {}",
                        gguf_dtype,
                        gguf_name
                    );
                }
            } else if let Some(our_dt) = our_dtype {
                // Quantized linear: raw H2D copy, keep compressed.
                let gpu_ptr = crate::driver::mem_alloc(size_bytes)?;
                crate::driver::memcpy_htod_async(gpu_ptr, host_buf, size_bytes, stream)?;

                // For 3D tensors (fused MoE experts), flatten first dims:
                // [num_experts, output_dim, input_dim] → nrows = num_experts * output_dim.
                let (nrows, ncols) = match dims.len() {
                    3 => (dims[0] * dims[1], dims[2]),
                    2 => (dims[0], dims[1]),
                    1 => (1, dims[0]),
                    _ => anyhow::bail!("unexpected shape {:?} for weight {}", dims, gguf_name),
                };

                let storage = GgmlStorage {
                    ptr: gpu_ptr,
                    len: size_bytes,
                    dtype: our_dt,
                    nrows,
                    ncols,
                };
                // Keep GGUF weights quantized on GPU (saves ~2× weight memory vs
                // dequant-to-BF16).
                //
                // Force the async H2D to finish before we loop and reuse host_buf
                // for the next tensor — otherwise the next reader.read_exact
                // overwrites pinned memory that DMA is still reading from, and
                // every subsequent quantized weight lands on GPU corrupted.
                crate::driver::stream_synchronize(stream)?;
                weights.insert(hf_name, GgufWeight::Quantized(storage));
            } else {
                anyhow::bail!(
                    "unsupported GGUF dtype {:?} for tensor {}",
                    gguf_dtype,
                    gguf_name
                );
            }
        }

        crate::driver::stream_synchronize(stream)?;
        crate::driver::mem_free_host(host_buf)?;

        tracing::info!(
            "GgufGpuWeights: loaded {} tensors from {}",
            weights.len(),
            path.display()
        );
        Ok(Self { weights })
    }

    /// Take a weight by HF name. Returns None if not found.
    pub fn take(&mut self, name: &str) -> Option<GgufWeight> {
        self.weights.remove(name)
    }

    /// Consume self, return the underlying weights map. Used by the
    /// `load_gguf_into_weights` adapter to transfer storages into a
    /// `GpuWeights` without re-uploading anything.
    pub fn into_weights(self) -> HashMap<String, GgufWeight> {
        self.weights
    }

    /// Take a quantized weight, returning the GgmlStorage.
    pub fn take_quantized(&mut self, name: &str) -> anyhow::Result<GgmlStorage> {
        match self.take(name) {
            Some(GgufWeight::Quantized(s)) => Ok(s),
            Some(GgufWeight::Dense(_)) => {
                anyhow::bail!("expected quantized weight for {name}, got dense")
            }
            None => anyhow::bail!("weight not found: {name}"),
        }
    }

    /// Take a dense (dequantized) weight, returning the GpuTensor.
    pub fn take_dense(&mut self, name: &str) -> anyhow::Result<GpuTensor> {
        match self.take(name) {
            Some(GgufWeight::Dense(t)) => Ok(t),
            Some(GgufWeight::Quantized(_)) => {
                anyhow::bail!("expected dense weight for {name}, got quantized")
            }
            None => anyhow::bail!("weight not found: {name}"),
        }
    }

    /// Check if a weight exists.
    pub fn contains(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    /// Number of remaining weights.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

// ---------------------------------------------------------------------------
// GpuWeights adapter — load a GGUF file into a `GpuWeights` so the
// scratchy codegen path (which holds `&mut GpuWeights`) can consume
// quantized-linear and dequantized-norm/embed/lm_head tensors via
// the standard `take_*` accessors. Today's eager GGUF loader stays
// the source of truth — this is a thin wrapper that transfers the
// storages without re-uploading.
// ---------------------------------------------------------------------------

/// How a tensor is split across `tp_world_size` ranks. Mirrors
/// `scratchy_forward_compiler_macro::tp_lowering::ShardKind` — kept here as
/// pure data so the GGUF loader (which runs at runtime) can decide
/// per-tensor slicing without depending on the proc-macro crate.
///
/// Future: when the `Ggml` `StorageFormat` lands in `quantization.rs`,
/// the proc-macro and this enum should both ultimately read the
/// same rule table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufShardKind {
    ShardDim0,
    ShardDim1,
    Replicate,
}

/// Apply the standard HF-name shard rule. Mirrors
/// `tp_lowering::shard_kind_for_last_segment` exactly — must stay in
/// sync if either side changes. This is the rule table the
/// safetensors codegen path emits today.
pub fn gguf_shard_kind_for_hf_name(hf_name: &str) -> GgufShardKind {
    // Strip the trailing `.weight` / `.bias` suffix that GGUF loader
    // emits, then look at the final segment.
    let stem = hf_name
        .strip_suffix(".weight")
        .or_else(|| hf_name.strip_suffix(".bias"))
        .unwrap_or(hf_name);
    let last = stem.rsplit('.').next().unwrap_or("");
    match last {
        "q_proj" | "k_proj" | "v_proj" | "gate_proj" | "up_proj" => GgufShardKind::ShardDim0,
        "o_proj" | "down_proj" => GgufShardKind::ShardDim1,
        "embed_tokens" | "lm_head" => GgufShardKind::ShardDim0,
        _ => GgufShardKind::Replicate,
    }
}

// Register `load_gguf_into_weights` as the workspace's GGUF loader via
// `inventory`, so `GpuWeights::from_gguf_file` can route through it
// without a direct dependency on this module.
inventory::submit! {
    crate::gguf_loader::GgufLoaderRegistration {
        name: "scratchy-target-cuda",
        load: load_gguf_into_weights,
    }
}

/// Load a GGUF file into a freshly-constructed `GpuWeights`.
///
/// Quantized linears land in `gw.quantized_map_mut()`; dequantized
/// norms / embeddings / lm_head land in `gw.gguf_dense_map_mut()`.
/// Both maps are keyed by HF tensor name.
///
/// **TP status (2026-04-29):** `tp_world_size = 1` is implemented;
/// `tp_world_size > 1` returns an error. The shard-rule lookup
/// (`gguf_shard_kind_for_hf_name`) is wired so a future TP slicing
/// pass can be added inside `GgufGpuWeights::load` without changing
/// this adapter's signature. The slicing semantics are:
///   - `ShardDim0` (q/k/v/gate/up/embed/lm_head): per-rank rows of
///     the row-major [out, in] tensor — contiguous byte-range slice.
///     Always block-aligned because each row is a whole number of
///     blocks.
///   - `ShardDim1` (o/down): per-rank columns within each row —
///     strided byte slice. Requires `(in_features / tp) %
///     block_size == 0` for the underlying GgmlDType. Refuse-at-load
///     on misalignment.
///   - `Replicate` (norms / qk_norm): full tensor on every rank.
///
/// # Safety
/// Requires a valid CUDA context and stream. The returned
/// `GpuWeights` retains the GGUF tensor pointers for the lifetime
/// of the model — same lifetime contract as the existing
/// `GgufGpuWeights::load`.
pub unsafe fn load_gguf_into_weights(
    path: &std::path::Path,
    model_dtype: DType,
    alloc: &mut CachingAllocator,
    stream: CUstream,
    tp_rank: usize,
    tp_world_size: usize,
) -> anyhow::Result<crate::weights::GpuWeights> {
    let gguf =
        unsafe { GgufGpuWeights::load(path, model_dtype, alloc, stream, tp_rank, tp_world_size)? };
    // Stage A's `GpuWeights::empty` takes a `BackendAllocator` (= `CudaAllocator` under
    // cuda) instead of the raw stream — wrap to match. The fresh allocator is
    // empty; the GGUF loader makes its allocations through the caller's
    // `CachingAllocator`, not through this `GpuWeights::allocator`. Stays
    // identical to the pre-Stage-A behavior since the allocator field was
    // unused on this path.
    let mut gw = crate::weights::GpuWeights::empty(crate::CudaAllocator::new(stream));

    for (name, weight) in gguf.into_weights() {
        match weight {
            GgufWeight::Quantized(storage) => {
                gw.quantized_map_mut().insert(name, storage);
            }
            GgufWeight::Dense(tensor) => {
                gw.gguf_dense_map_mut().insert(name, tensor);
            }
        }
    }

    Ok(gw)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_dtype_from_u32() {
        assert_eq!(GgmlDType::from_u32(2), Some(GgmlDType::Q4_0));
        assert_eq!(GgmlDType::from_u32(12), Some(GgmlDType::Q4K));
        assert_eq!(GgmlDType::from_u32(99), None);
    }

    #[test]
    fn test_ggml_dtype_type_size() {
        assert_eq!(GgmlDType::Q4_0.type_size(), 18);
        assert_eq!(GgmlDType::Q8_0.type_size(), 34);
        assert_eq!(GgmlDType::Q8_1.type_size(), 40);
    }

    #[test]
    fn test_ggml_dtype_block_size() {
        assert_eq!(GgmlDType::Q4_0.block_size(), 32);
        assert_eq!(GgmlDType::Q4K.block_size(), 256);
    }

    #[test]
    fn test_ggml_dtype_is_k_quant() {
        assert!(!GgmlDType::Q4_0.is_k_quant());
        assert!(GgmlDType::Q4K.is_k_quant());
        assert!(GgmlDType::Q6K.is_k_quant());
    }

    #[test]
    fn test_ggml_dtype_iq4_from_u32() {
        assert_eq!(GgmlDType::from_u32(20), Some(GgmlDType::IQ4NL));
        assert_eq!(GgmlDType::from_u32(23), Some(GgmlDType::IQ4XS));
    }

    #[test]
    fn test_ggml_dtype_iq4_type_size() {
        assert_eq!(GgmlDType::IQ4NL.type_size(), 18);
        assert_eq!(GgmlDType::IQ4XS.type_size(), 136);
    }

    #[test]
    fn test_ggml_dtype_iq4_block_size() {
        assert_eq!(GgmlDType::IQ4NL.block_size(), 32);
        assert_eq!(GgmlDType::IQ4XS.block_size(), 256);
    }

    #[test]
    fn test_gguf_shard_kind_for_hf_name() {
        // Column-parallel: q/k/v/gate/up + embed + lm_head.
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.7.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
        ] {
            assert_eq!(
                gguf_shard_kind_for_hf_name(name),
                GgufShardKind::ShardDim0,
                "expected ShardDim0 for {name}"
            );
        }

        // Row-parallel: o_proj / down_proj.
        for name in [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
            "model.layers.3.mlp.down_proj.bias",
        ] {
            assert_eq!(
                gguf_shard_kind_for_hf_name(name),
                GgufShardKind::ShardDim1,
                "expected ShardDim1 for {name}"
            );
        }

        // Replicate: norms (incl. q_norm / k_norm / final norm).
        for name in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.5.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.norm.weight",
        ] {
            assert_eq!(
                gguf_shard_kind_for_hf_name(name),
                GgufShardKind::Replicate,
                "expected Replicate for {name}"
            );
        }
    }

    #[test]
    fn test_ggml_dtype_is_iq_quant() {
        assert!(GgmlDType::IQ4NL.is_iq_quant());
        assert!(GgmlDType::IQ4XS.is_iq_quant());
        assert!(GgmlDType::IQ1M.is_iq_quant());
        assert!(!GgmlDType::Q4_0.is_iq_quant());
        assert!(!GgmlDType::Q4K.is_iq_quant());
    }

    #[test]
    fn test_ggml_dtype_iq4_display() {
        assert_eq!(format!("{}", GgmlDType::IQ4NL), "IQ4_NL");
        assert_eq!(format!("{}", GgmlDType::IQ4XS), "IQ4_XS");
    }

    #[test]
    fn test_ggml_dtype_iq1m_from_u32() {
        assert_eq!(GgmlDType::from_u32(29), Some(GgmlDType::IQ1M));
    }

    #[test]
    fn test_ggml_dtype_iq1m_type_size() {
        assert_eq!(GgmlDType::IQ1M.type_size(), 56);
    }

    #[test]
    fn test_ggml_dtype_iq1m_block_size() {
        assert_eq!(GgmlDType::IQ1M.block_size(), 256);
    }

    #[test]
    fn test_ggml_dtype_iq1m_display() {
        assert_eq!(format!("{}", GgmlDType::IQ1M), "IQ1_M");
    }

    #[test]
    fn test_ggml_storage_verify_size() {
        let s = GgmlStorage {
            ptr: 0x1000 as *mut u8,
            len: 18 * (4096 * 4096 / 32),
            dtype: GgmlDType::Q4_0,
            nrows: 4096,
            ncols: 4096,
        };
        assert!(s.verify_size());
    }

    #[test]
    fn test_ggml_storage_numel() {
        let s = GgmlStorage {
            ptr: std::ptr::null_mut(),
            len: 0,
            dtype: GgmlDType::Q4_0,
            nrows: 128,
            ncols: 256,
        };
        assert_eq!(s.numel(), 128 * 256);
    }

    #[test]
    fn test_padding_helpers() {
        assert_eq!(pad(100, 512), 512);
        assert_eq!(pad(512, 512), 512);
        assert_eq!(pad(513, 512), 1024);
    }
}
