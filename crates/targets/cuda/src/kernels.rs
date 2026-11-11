// SPDX-License-Identifier: Apache-2.0
//! The CUDA kernels' launchers, linked from `libscratchy_cuda_kernels.a` (`build.rs`).
//!
//! Each kernel in `kernels/*.cu` exposes one `extern "C"` launcher taking device pointers,
//! dimensions and a stream. This module declares them and wraps each in a typed function. It does
//! not choose how memory or streams are managed: the pointers and the stream are the caller's.
//!
//! Every kernel also exposes a `_prepare` function, listed in [`PREPARE`]: it sets the kernel's
//! function attributes (its dynamic shared memory size) or, for a kernel with none, only loads it,
//! so a worker's graph capture never loads a kernel. [`prepare_all`] runs them once, when the device
//! is opened (`runtime::CudaDevice::open`); a launcher never repeats that setup.

use std::ffi::c_void;
use std::fmt;

use crate::step::RopePairing;

/// `SCRATCHY_CUDA_ERR_SHAPE` in the `.cu` launchers: the shape was refused, nothing launched.
const ERR_SHAPE: i32 = -1;
/// `SCRATCHY_CUDA_ERR_HOST` in the `.cu` launchers: a C++ exception was caught on the host side
/// (e.g. a failed tensor-map encode).
const ERR_HOST: i32 = -2;

unsafe extern "C" {
    // The kernels with no attribute to set: each `_prepare` only loads its kernels (#1048), so a
    // worker's graph capture never loads one mid-capture.
    fn scratchy_cuda_embed_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_split_bf16_prepare() -> i32;
    fn scratchy_cuda_gdn_conv1d_prepare() -> i32;
    fn scratchy_cuda_gdn_gating_prepare() -> i32;
    fn scratchy_cuda_gdn_rms_norm_gated_prepare() -> i32;
    fn scratchy_cuda_gdn_scan_prepare() -> i32;
    fn scratchy_cuda_kv_append_bf16_prepare() -> i32;
    fn scratchy_cuda_kv_append_fp8_prepare() -> i32;
    fn scratchy_cuda_moe_combine_bf16_prepare() -> i32;
    fn scratchy_cuda_moe_route_bf16_prepare() -> i32;
    fn scratchy_cuda_moe_sort_prepare() -> i32;
    fn scratchy_cuda_rope_bf16_prepare() -> i32;
    fn scratchy_cuda_rope_append_fp8_prepare() -> i32;
    fn scratchy_cuda_embed_bf16(
        tokens: *const c_void,
        table: *const c_void,
        out: *mut c_void,
        rows: u32,
        hidden: u32,
        vocab: u32,
        stream: *mut c_void,
    ) -> i32;
    // FP8 S3: the per-token quantization and the prefill fp8 GEMM family.
    fn scratchy_cuda_quant_rows_fp8_prepare() -> i32;
    fn scratchy_cuda_quant_row_fp8_one_prepare() -> i32;
    fn scratchy_cuda_quant_row_fp8_one(
        x: *const c_void,
        q: *mut c_void,
        scale: *mut c_void,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_quant_row_fp8_one_variant(
        x: *const c_void,
        q: *mut c_void,
        scale: *mut c_void,
        cols: u32,
        variant: u32,
        trace: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_quant_rows_fp8(
        x: *const c_void,
        q: *mut c_void,
        scale: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_fp8_prepare() -> i32;
    fn scratchy_cuda_gemm_fp8_abt(
        a: *const c_void,
        a_scale: *const c_void,
        b: *const c_void,
        b_scale: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        col64: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_group_fp8_abt(
        a: *const c_void,
        a_scale: *const c_void,
        b: *const *const c_void,
        b_scale: *const *const c_void,
        c: *const *mut c_void,
        n: *const u32,
        m: u32,
        k: u32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_up_gemm_fp8(
        a: *const c_void,
        a_scale: *const c_void,
        gate: *const c_void,
        gate_scale: *const c_void,
        up: *const c_void,
        up_scale: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        act: u32,
        col64: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemm_fp8(
        a: *const c_void,
        a_scale: *const c_void,
        b: *const c_void,
        b_scale: *const c_void,
        offsets: *const c_void,
        tile_expert: *const c_void,
        c: *mut c_void,
        max_tiles: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        n_tile: u32,
        stream: *mut c_void,
    ) -> i32;
    // FP8 S8 W4c: the one-row rope append (`kernels/rope_append_bf16.cu`).
    fn scratchy_cuda_rope_append_fp8(
        q: *const c_void,
        q_out: *mut c_void,
        k: *const c_void,
        k_out: *mut c_void,
        v: *const c_void,
        table: *const c_void,
        slot_mapping: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rot_dim: u32,
        pairing: u32,
        block_size: u32,
        num_blocks: u32,
        q_gain: *const c_void,
        k_gain: *const c_void,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    // FP8 S8 W4a: the one-row GEMVs with the residual epilogue (`kernels/gemv_fp8.cu`,
    // `kernels/gemv_bf16.cu`).
    fn scratchy_cuda_gemv_fp8_residual(
        a: *const c_void,
        b: *const c_void,
        b_scale: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        n: u32,
        k: u32,
        splits: u32,
        residual: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_bf16_residual(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        n: u32,
        k: u32,
        splits: u32,
        residual: *const c_void,
        stream: *mut c_void,
    ) -> i32;
    // FP8 S4: the decode fp8 GEMV family, one prepare for the file.
    fn scratchy_cuda_gemv_fp8_prepare() -> i32;
    // FP8 S8: the tensor-core decode GEMV (`kernels/gemv_fp8_tc.cu`), no tape launches it yet.
    fn scratchy_cuda_gemv_fp8_tc_prepare() -> i32;
    fn scratchy_cuda_gemv_fp8_b8(
        q: *const c_void,
        a_scale: *const c_void,
        a_rows: u32,
        b: *const *const c_void,
        b_scale: *const *const c_void,
        c: *const *mut c_void,
        residual: *const *const c_void,
        n: *const u32,
        count: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        topk_idx: *const c_void,
        pairs: u32,
        row_div: u32,
        out_f32: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_fp8_b8_box_probe(
        x: *const c_void,
        rows: u32,
        w: *const c_void,
        n: u32,
        k: u32,
        row: u32,
        tile: u32,
        j: u32,
        poison_x: *const c_void,
        poison_w: *const c_void,
        dump_x: *mut c_void,
        dump_w: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_act_quant_fp8_prepare() -> i32;
    fn scratchy_cuda_act_quant_fp8(
        gate: *const c_void,
        up: *const c_void,
        q: *mut c_void,
        scale: *mut c_void,
        act: *mut c_void,
        rows: u32,
        n: u32,
        op: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_fp8_tc_partial_bytes(n: *const u32, count: u32, k: u32) -> u64;
    fn scratchy_cuda_gemv_fp8_tc(
        q: *const c_void,
        a_scale: *const c_void,
        b: *const *const c_void,
        b_scale: *const *const c_void,
        c: *const *mut c_void,
        n: *const u32,
        count: u32,
        k: u32,
        partials: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_fp8_tc_variant(
        q: *const c_void,
        a_scale: *const c_void,
        b: *const *const c_void,
        b_scale: *const *const c_void,
        c: *const *mut c_void,
        n: *const u32,
        count: u32,
        k: u32,
        partials: *mut c_void,
        variant: u32,
        main_only: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_fp8_abt(
        a: *const c_void,
        b: *const c_void,
        b_scale: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        n: u32,
        k: u32,
        splits: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_group_fp8_abt(
        a: *const c_void,
        b: *const *const c_void,
        b_scale: *const *const c_void,
        c: *const *mut c_void,
        partials: *const *mut c_void,
        n: *const u32,
        k: u32,
        splits: *const u32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_up_gemv_fp8(
        a: *const c_void,
        gate: *const c_void,
        gate_scale: *const c_void,
        up: *const c_void,
        up_scale: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        n: u32,
        k: u32,
        splits: u32,
        act: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemv_fp8(
        a: *const c_void,
        b: *const c_void,
        b_scale: *const c_void,
        topk_idx: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        m: u32,
        top_k: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        pair_rows: u32,
        splits: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gate_up_gemv_fp8(
        a: *const c_void,
        gate: *const c_void,
        gate_scale: *const c_void,
        up: *const c_void,
        up_scale: *const c_void,
        topk_idx: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        m: u32,
        top_k: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        splits: u32,
        act: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_fp8_quant_probe(
        a: *const c_void,
        out: *mut c_void,
        scale: *mut c_void,
        k: u32,
        splits: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_narrow_fp8_prepare() -> i32;
    fn scratchy_cuda_gemm_narrow_fp8_abt(
        a: *const c_void,
        a_scale: *const c_void,
        b: *const c_void,
        b_scale: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_narrow_group_fp8_abt(
        a: *const c_void,
        a_scale: *const c_void,
        b: *const *const c_void,
        b_scale: *const *const c_void,
        c: *const *mut c_void,
        n: *const u32,
        m: u32,
        k: u32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_col64_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_col64_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_rms_norm_bf16_prepare() -> i32;
    fn scratchy_cuda_rms_norm_bf16(
        x: *const c_void,
        gain: *const c_void,
        y: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        gain_offset: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_add_rms_norm_bf16(
        a: *const c_void,
        b: *const c_void,
        gain: *const c_void,
        sum: *mut c_void,
        y: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        gain_offset: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_norm_add_bf16(
        x: *const c_void,
        gain: *const c_void,
        residual: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        gain_offset: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_rms_norm_unit_bf16(
        x: *const c_void,
        y: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_pointwise_bf16_prepare() -> i32;
    fn scratchy_cuda_scale_bf16(
        x: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_layer_scale_bf16(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_soft_cap_bf16(
        x: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        cap: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_add_bf16_prepare() -> i32;
    fn scratchy_cuda_add_bf16(
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_silu_mul_bf16_prepare() -> i32;
    fn scratchy_cuda_silu_mul_bf16(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gelu_mul_bf16(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_rope_bf16(
        x: *const c_void,
        table: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_rope_paired_bf16(
        x: *const c_void,
        table: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
        pairing: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_split_bf16(
        qg: *const c_void,
        q: *mut c_void,
        gate: *mut c_void,
        rows: u32,
        half_cols: u32,
        head_dim: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_apply_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_apply_bf16(
        attn: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_scale_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_scale_bf16(
        routed: *const c_void,
        shared: *const c_void,
        g: *const c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_kv_append_bf16(
        k: *const c_void,
        v: *const c_void,
        slot_mapping: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        rows: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_kv_append_fp8(
        k: *const c_void,
        v: *const c_void,
        slot_mapping: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        rows: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_prefill_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_prefill_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_prefill_fp8kv_prepare() -> i32;
    fn scratchy_cuda_attn_prefill_fp8kv(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_prefill_d512_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_prefill_d512_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_prefill_d512_fp8kv_prepare() -> i32;
    fn scratchy_cuda_attn_prefill_d512_fp8kv(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_narrow_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_narrow_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_group_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_group_bf16_abt(
        a: *const c_void,
        b: *const *const c_void,
        c: *const *mut c_void,
        n: *const u32,
        m: u32,
        k: u32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemm_narrow_group_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemm_narrow_group_bf16_abt(
        a: *const c_void,
        b: *const *const c_void,
        c: *const *mut c_void,
        n: *const u32,
        m: u32,
        k: u32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_group_bf16_abt_prepare() -> i32;
    fn scratchy_cuda_gemv_group_bf16_abt(
        a: *const c_void,
        b: *const *const c_void,
        c: *const *mut c_void,
        partials: *const *mut c_void,
        n: *const u32,
        k: u32,
        splits: *const u32,
        count: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_decode_prepare() -> i32;
    fn scratchy_cuda_gdn_decode(
        qkv: *const c_void,
        a: *const c_void,
        b: *const c_void,
        z: *const c_void,
        w: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        norm: *const c_void,
        conv_state: *mut c_void,
        conv_slot_elems: u64,
        ssm_state: *mut c_void,
        ssm_slot_elems: u64,
        sequence: *const c_void,
        out: *mut c_void,
        nk: u32,
        nv: u32,
        head_k: u32,
        head_v: u32,
        kernel: u32,
        scale: f32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_up_gemv_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_up_gemv_bf16(
        a: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        n: u32,
        k: u32,
        splits: u32,
        act: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gate_up_gemv_bf16(
        a: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        topk_idx: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        m: u32,
        top_k: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        splits: u32,
        act: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gate_up_gemm_bf16_prepare() -> i32;
    fn scratchy_cuda_gate_up_gemm_bf16(
        a: *const c_void,
        gate: *const c_void,
        up: *const c_void,
        c: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        act: u32,
        col64: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_split_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        n: u32,
        k: u32,
        splits: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gemv_bf16_abt(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        n: u32,
        k: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_box_probe_load(
        src: *const c_void,
        poison: *const c_void,
        dump: *mut c_void,
        rows: u32,
        cols: u32,
        tile_row: u32,
        tile_col: u32,
        swizzled: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_box_probe_store(
        pattern: *const c_void,
        dst: *mut c_void,
        rows: u32,
        cols: u32,
        tile_row: u32,
        tile_col: u32,
        swizzled: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_ring_probe(
        b: *const c_void,
        out: *mut c_void,
        n: u32,
        k: u32,
        variant: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_consume_probe(
        b: *const c_void,
        a: *const c_void,
        out: *mut c_void,
        n: u32,
        k: u32,
        consume: u32,
        rows: u32,
        ctas: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_tma_probe_occupancy(
        variant: u32,
        consume: u32,
        rows: u32,
        ctas: u32,
        blocks: *mut i32,
    ) -> i32;
    fn scratchy_cuda_tc_stream_probe(
        b: *const c_void,
        a: *const c_void,
        out: *mut c_void,
        n: u32,
        k: u32,
        variant: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_decode_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_decode_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_decode_fp8kv_prepare() -> i32;
    fn scratchy_cuda_attn_decode_fp8kv(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_decode_d512_bf16_prepare() -> i32;
    fn scratchy_cuda_attn_decode_d512_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_attn_decode_d512_fp8kv_prepare() -> i32;
    fn scratchy_cuda_attn_decode_d512_fp8kv(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        block_table: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
        scale: f32,
        window: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_conv1d(
        x: *const c_void,
        w: *const c_void,
        conv_state: *mut c_void,
        slot_elems: u64,
        sequence: *const c_void,
        conv_out: *mut c_void,
        rows: u32,
        conv_dim: u32,
        kernel: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_gating(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        sequence: *const c_void,
        g: *mut c_void,
        beta: *mut c_void,
        rows: u32,
        nv: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_scan(
        conv_out: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        ssm_state: *mut c_void,
        slot_elems: u64,
        sequence: *const c_void,
        o: *mut c_void,
        rows: u32,
        nk: u32,
        nv: u32,
        head_k: u32,
        head_v: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_rank_one_probe(
        s: *const c_void,
        t: *const c_void,
        u: *const c_void,
        out: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_moe_route_bf16(
        logits: *const c_void,
        expert_scale: *const c_void,
        topk_idx: *mut c_void,
        scores: *mut c_void,
        m: u32,
        experts: u32,
        top_k: u32,
        renorm: u32,
        program: u32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_moe_combine_bf16(
        down: *const c_void,
        pos: *const c_void,
        scores: *const c_void,
        out: *mut c_void,
        m: u32,
        top_k: u32,
        hidden: u32,
        down_rows: u32,
        sorted: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_moe_sort(
        x: *const c_void,
        topk_idx: *const c_void,
        chunk_hist: *mut c_void,
        counts: *mut c_void,
        offsets: *mut c_void,
        tile_expert: *mut c_void,
        pos: *mut c_void,
        rows: *mut c_void,
        m: u32,
        experts: u32,
        top_k: u32,
        hidden: u32,
        max_tiles: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemm_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemm_e128_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemm_e128_col64_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemm_bf16(
        a: *const c_void,
        b: *const c_void,
        offsets: *const c_void,
        tile_expert: *const c_void,
        c: *mut c_void,
        max_tiles: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        n_tile: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemv_bf16_prepare() -> i32;
    fn scratchy_cuda_expert_gemv_split_bf16(
        a: *const c_void,
        b: *const c_void,
        topk_idx: *const c_void,
        c: *mut c_void,
        partials: *mut c_void,
        m: u32,
        top_k: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        pair_rows: u32,
        splits: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_expert_gemv_bf16(
        a: *const c_void,
        b: *const c_void,
        topk_idx: *const c_void,
        c: *mut c_void,
        m: u32,
        top_k: u32,
        n: u32,
        k: u32,
        experts: u32,
        expert_rows: u32,
        pair_rows: u32,
        stream: *mut c_void,
    ) -> i32;
    fn scratchy_cuda_gdn_rms_norm_gated(
        o: *const c_void,
        z: *const c_void,
        w: *const c_void,
        sequence: *const c_void,
        out: *mut c_void,
        rows: u32,
        nv: u32,
        head_v: u32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;
}

/// Why a launcher did not launch, or the launch failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// The embed does not take this shape; nothing was launched.
    EmbedShape { rows: u32, hidden: u32, vocab: u32 },
    /// The GEMM does not take this shape; nothing was launched.
    Shape { m: u32, n: u32, k: u32 },
    /// A row kernel (RMSNorm, add, SiluMul, GateApply, and Gemma's unit RMSNorm, GeluMul, scale,
    /// layer scale and soft-cap) does not take this shape; nothing was launched.
    RowShape { rows: u32, cols: u32 },
    /// The rope does not take this shape; nothing was launched.
    RopeShape {
        rows: u32,
        cols: u32,
        head_dim: u32,
        rot_dim: u32,
    },
    /// The gate split does not take this shape; nothing was launched.
    GateShape {
        rows: u32,
        half_cols: u32,
        head_dim: u32,
    },
    /// The KV append does not take this shape; nothing was launched.
    KvAppendShape {
        rows: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
    },
    /// The prefill attention does not take this shape, scale or window; nothing was launched.
    AttnShape {
        rows: u32,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
    },
    /// The narrow GEMM does not take this shape; nothing was launched.
    NarrowShape { m: u32, n: u32, k: u32 },
    /// The GEMV does not take this shape; nothing was launched.
    GemvShape { n: u32, k: u32 },
    /// The split-K GEMV does not take this shape, split count or scratch (a null scratch above
    /// one split); nothing was launched.
    GemvSplit { n: u32, k: u32, splits: u32 },
    /// A grouped GEMM (#1049) does not take this member count or one of its members' shapes;
    /// nothing was launched. `n` is each member's (0 past `count`).
    GroupShape {
        kernel: &'static str,
        m: u32,
        k: u32,
        n: [u32; 3],
    },
    /// A gate/up launch (#1049) does not take this shape, split or activation; nothing was
    /// launched. `dims` is `[m, n, k, splits]` (`m` the pairs for the expert form).
    GateUpShape {
        kernel: &'static str,
        dims: [u32; 4],
    },
    /// The one-launch GDN decode (#1049) does not take this geometry, its state strides, its
    /// scale or its eps; nothing was launched.
    GdnDecodeShape {
        nk: u32,
        nv: u32,
        head_v: u32,
        kernel: u32,
    },
    /// The decode attention does not take this shape, scale or window; nothing was launched.
    AttnDecodeShape {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        block_size: u32,
        num_blocks: u32,
    },
    /// The GDN conv1d does not take this shape or state stride; nothing was launched.
    GdnConvShape {
        rows: u32,
        conv_dim: u32,
        kernel: u32,
    },
    /// The GDN gating does not take this shape; nothing was launched.
    GdnGatingShape { rows: u32, nv: u32 },
    /// The GDN scan does not take this geometry, state stride or scale; nothing was launched.
    GdnScanShape {
        rows: u32,
        nk: u32,
        nv: u32,
        head_k: u32,
        head_v: u32,
    },
    /// The GDN gated norm does not take this shape or eps; nothing was launched.
    GdnNormShape { rows: u32, nv: u32, head_v: u32 },
    /// A MoE kernel (`kernel`, its file's stem) does not take this shape; `dims` are its launch
    /// dimensions in its launcher's order. Nothing was launched.
    MoeShape {
        kernel: &'static str,
        dims: [u32; 4],
    },
    /// The launcher's C++ host code threw (caught before it could reach Rust).
    HostException,
    /// A `cudaError_t` from the kernel's one-time setup or from its launch.
    Cuda(i32),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::EmbedShape {
                rows,
                hidden,
                vocab,
            } => write!(
                f,
                "embed refused shape rows={rows} hidden={hidden} vocab={vocab}"
            ),
            KernelError::Shape { m, n, k } => write!(f, "kernel refused shape m={m} n={n} k={k}"),
            KernelError::RowShape { rows, cols } => {
                write!(f, "kernel refused shape rows={rows} cols={cols}")
            }
            KernelError::RopeShape {
                rows,
                cols,
                head_dim,
                rot_dim,
            } => write!(
                f,
                "rope refused shape rows={rows} cols={cols} head_dim={head_dim} rot_dim={rot_dim}"
            ),
            KernelError::GateShape {
                rows,
                half_cols,
                head_dim,
            } => write!(
                f,
                "gate split refused shape rows={rows} half_cols={half_cols} head_dim={head_dim}"
            ),
            KernelError::KvAppendShape {
                rows,
                kv_heads,
                head_dim,
                block_size,
            } => write!(
                f,
                "kv append refused shape rows={rows} kv_heads={kv_heads} head_dim={head_dim} \
                 block_size={block_size}"
            ),
            KernelError::AttnShape {
                rows,
                q_heads,
                kv_heads,
                head_dim,
                block_size,
                num_blocks,
            } => write!(
                f,
                "prefill attention refused shape rows={rows} q_heads={q_heads} \
                 kv_heads={kv_heads} head_dim={head_dim} block_size={block_size} \
                 num_blocks={num_blocks}, or its scale or window"
            ),
            KernelError::NarrowShape { m, n, k } => {
                write!(f, "narrow GEMM refused shape m={m} n={n} k={k}")
            }
            KernelError::GemvShape { n, k } => write!(f, "GEMV refused shape n={n} k={k}"),
            KernelError::AttnDecodeShape {
                q_heads,
                kv_heads,
                head_dim,
                block_size,
                num_blocks,
            } => write!(
                f,
                "decode attention refused shape q_heads={q_heads} kv_heads={kv_heads} \
                 head_dim={head_dim} block_size={block_size} num_blocks={num_blocks}, or its scale \
                 or window"
            ),
            KernelError::GdnConvShape {
                rows,
                conv_dim,
                kernel,
            } => write!(
                f,
                "GDN conv1d refused shape rows={rows} conv_dim={conv_dim} kernel={kernel}, or its \
                 state stride"
            ),
            KernelError::GdnGatingShape { rows, nv } => {
                write!(f, "GDN gating refused shape rows={rows} nv={nv}")
            }
            KernelError::GdnScanShape {
                rows,
                nk,
                nv,
                head_k,
                head_v,
            } => write!(
                f,
                "GDN scan refused shape rows={rows} nk={nk} nv={nv} head_k={head_k} \
                 head_v={head_v}, or its state stride or scale"
            ),
            KernelError::GdnNormShape { rows, nv, head_v } => write!(
                f,
                "GDN gated norm refused shape rows={rows} nv={nv} head_v={head_v}, or its eps"
            ),
            KernelError::GemvSplit { n, k, splits } => write!(
                f,
                "GEMV refused split shape n={n} k={k} splits={splits}, or its scratch"
            ),
            KernelError::MoeShape { kernel, dims } => {
                write!(f, "{kernel} refused shape {dims:?}")
            }
            KernelError::GroupShape { kernel, m, k, n } => {
                write!(f, "{kernel} refused shape m={m} k={k} n={n:?}")
            }
            KernelError::GdnDecodeShape {
                nk,
                nv,
                head_v,
                kernel,
            } => write!(
                f,
                "GDN decode refused nk={nk} nv={nv} head_v={head_v} conv kernel={kernel}, or its \
                 state strides, scale or eps"
            ),
            KernelError::GateUpShape { kernel, dims } => {
                write!(f, "{kernel} refused [m, n, k, splits] {dims:?}")
            }
            KernelError::HostException => write!(f, "kernel launcher's host code threw"),
            KernelError::Cuda(code) => write!(f, "CUDA error {code}"),
        }
    }
}

impl std::error::Error for KernelError {}

/// Every kernel's one-time setup, by kernel name: a later kernel adds one row. A kernel with an
/// attribute to set (its dynamic shared memory) sets it; every other kernel's setup only loads it,
/// so every kernel any tape launches is loaded before a worker captures its graphs.
const PREPARE: &[(&str, unsafe extern "C" fn() -> i32)] = &[
    ("gemm_bf16_abt", scratchy_cuda_gemm_bf16_abt_prepare),
    (
        "gemm_col64_bf16_abt",
        scratchy_cuda_gemm_col64_bf16_abt_prepare,
    ),
    ("attn_prefill_bf16", scratchy_cuda_attn_prefill_bf16_prepare),
    (
        "gemm_narrow_bf16_abt",
        scratchy_cuda_gemm_narrow_bf16_abt_prepare,
    ),
    ("gemv_bf16_abt", scratchy_cuda_gemv_bf16_abt_prepare),
    // Every form of `rms_norm_bf16.cu`: the norm, the unit norm, and #1049's add-norm folds.
    ("rms_norm_bf16", scratchy_cuda_rms_norm_bf16_prepare),
    ("add_bf16", scratchy_cuda_add_bf16_prepare),
    ("silu_mul_bf16", scratchy_cuda_silu_mul_bf16_prepare),
    ("pointwise_bf16", scratchy_cuda_pointwise_bf16_prepare),
    ("gate_apply_bf16", scratchy_cuda_gate_apply_bf16_prepare),
    ("gate_scale_bf16", scratchy_cuda_gate_scale_bf16_prepare),
    ("attn_decode_bf16", scratchy_cuda_attn_decode_bf16_prepare),
    (
        "attn_prefill_d512_bf16",
        scratchy_cuda_attn_prefill_d512_bf16_prepare,
    ),
    (
        "attn_decode_d512_bf16",
        scratchy_cuda_attn_decode_d512_bf16_prepare,
    ),
    // FP8 S5: the attention kernels over the e4m3 cache, the attribute on both instances of each.
    (
        "attn_prefill_fp8kv",
        scratchy_cuda_attn_prefill_fp8kv_prepare,
    ),
    ("attn_decode_fp8kv", scratchy_cuda_attn_decode_fp8kv_prepare),
    (
        "attn_prefill_d512_fp8kv",
        scratchy_cuda_attn_prefill_d512_fp8kv_prepare,
    ),
    (
        "attn_decode_d512_fp8kv",
        scratchy_cuda_attn_decode_d512_fp8kv_prepare,
    ),
    ("expert_gemm_bf16", scratchy_cuda_expert_gemm_bf16_prepare),
    (
        "expert_gemm_e128_bf16",
        scratchy_cuda_expert_gemm_e128_bf16_prepare,
    ),
    (
        "expert_gemm_e128_col64_bf16",
        scratchy_cuda_expert_gemm_e128_col64_bf16_prepare,
    ),
    ("expert_gemv_bf16", scratchy_cuda_expert_gemv_bf16_prepare),
    // #1049's grouped GEMMs: the attribute, which loads each before any graph capture.
    (
        "gemm_group_bf16_abt",
        scratchy_cuda_gemm_group_bf16_abt_prepare,
    ),
    (
        "gemm_narrow_group_bf16_abt",
        scratchy_cuda_gemm_narrow_group_bf16_abt_prepare,
    ),
    (
        "gemv_group_bf16_abt",
        scratchy_cuda_gemv_group_bf16_abt_prepare,
    ),
    // #1049's gate/up GEMVs (dense and expert) and GEMMs: the attribute on each instance, and a
    // load of the GEMV's reduces.
    ("gate_up_gemv_bf16", scratchy_cuda_gate_up_gemv_bf16_prepare),
    // #1049's one-launch GDN decode: static shared memory only, so a load of every instance.
    ("gdn_decode", scratchy_cuda_gdn_decode_prepare),
    ("gate_up_gemm_bf16", scratchy_cuda_gate_up_gemm_bf16_prepare),
    // FP8 S3: the attribute on every instance of each file, which loads each.
    ("quant_rows_fp8", scratchy_cuda_quant_rows_fp8_prepare),
    ("quant_row_fp8_one", scratchy_cuda_quant_row_fp8_one_prepare),
    ("gemm_fp8", scratchy_cuda_gemm_fp8_prepare),
    ("gemm_narrow_fp8", scratchy_cuda_gemm_narrow_fp8_prepare),
    // FP8 S4: the attribute on every main instance of `gemv_fp8.cu`, and a load of its reduces.
    ("gemv_fp8", scratchy_cuda_gemv_fp8_prepare),
    ("gemv_fp8_tc", scratchy_cuda_gemv_fp8_tc_prepare),
    ("act_quant_fp8", scratchy_cuda_act_quant_fp8_prepare),
    // Load-only (#1048): these set no attribute, but a worker captures its tapes into CUDA Graphs
    // at construction, and a kernel the runtime has not loaded must not first load during a
    // capture. Every kernel a tape can launch is loaded here, when the device opens.
    ("embed_bf16", scratchy_cuda_embed_bf16_prepare),
    ("gate_split_bf16", scratchy_cuda_gate_split_bf16_prepare),
    ("gdn_conv1d", scratchy_cuda_gdn_conv1d_prepare),
    ("gdn_gating", scratchy_cuda_gdn_gating_prepare),
    (
        "gdn_rms_norm_gated",
        scratchy_cuda_gdn_rms_norm_gated_prepare,
    ),
    ("gdn_scan", scratchy_cuda_gdn_scan_prepare),
    ("kv_append_bf16", scratchy_cuda_kv_append_bf16_prepare),
    ("kv_append_fp8", scratchy_cuda_kv_append_fp8_prepare),
    ("moe_combine_bf16", scratchy_cuda_moe_combine_bf16_prepare),
    ("moe_route_bf16", scratchy_cuda_moe_route_bf16_prepare),
    ("moe_sort", scratchy_cuda_moe_sort_prepare),
    ("rope_bf16", scratchy_cuda_rope_bf16_prepare),
    ("rope_append_fp8", scratchy_cuda_rope_append_fp8_prepare),
];

/// Run every kernel's one-time setup on the current device, stopping at the first failure, which
/// names its kernel. Every launch on that device needs it to have succeeded first.
pub fn prepare_all() -> Result<(), (&'static str, KernelError)> {
    for &(kernel, prepare) in PREPARE {
        // SAFETY: a prepare function takes no argument, only sets or queries attributes of its own
        // kernels on the current device, and returns a `cudaError_t`.
        match unsafe { prepare() } {
            0 => {}
            code => return Err((kernel, KernelError::Cuda(code))),
        }
    }
    Ok(())
}

/// Shared tail of every launcher wrapper: the launcher's return code as a `Result`, `shape` the
/// refusal its `SCRATCHY_CUDA_ERR_SHAPE` means.
fn check(code: i32, shape: KernelError) -> Result<(), KernelError> {
    match code {
        0 => Ok(()),
        ERR_SHAPE => Err(shape),
        ERR_HOST => Err(KernelError::HostException),
        code => Err(KernelError::Cuda(code)),
    }
}

/// Launch the bf16 token-embedding gather: `out[r, :] = table[tokens[r], :]` for every one of the
/// `rows` rows, padding rows included, the table's bits copied as they are; an id at or above
/// `vocab` writes a `+0.0` row and reads nothing. Asynchronous.
///
/// Refused with [`KernelError::EmbedShape`], before any pointer is touched, unless `rows`,
/// `hidden` and `vocab` are non-zero and at most `i32::MAX`, `hidden` a multiple of 8
/// (`step::EMBED_BF16_COLS`), and `rows · hidden` at most `i32::MAX`. A failed launch returns the
/// launch's own `cudaError_t`. Its one-time setup in [`PREPARE`] only loads the kernel.
///
/// # Safety
///
/// `tokens` must be a 4-byte-aligned device pointer to `rows` u32; `table` and `out`
/// 16-byte-aligned device pointers to at least `vocab·hidden` and `rows·hidden` bf16, valid for
/// the whole kernel, with `out` overlapping neither input. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn embed_bf16(
    tokens: *const c_void,
    table: *const c_void,
    out: *mut c_void,
    (rows, hidden, vocab): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer.
    let code = unsafe { scratchy_cuda_embed_bf16(tokens, table, out, rows, hidden, vocab, stream) };
    check(
        code,
        KernelError::EmbedShape {
            rows,
            hidden,
            vocab,
        },
    )
}

/// Launch the bf16 GEMM `C = A · Bᵀ` on the H100: `A` is `[m, k]`, `B` is `[n, k]` and `C` is
/// `[m, n]`, all row-major bf16, accumulated in f32. The launch is asynchronous; nothing is
/// synchronized.
///
/// `m`, `n` and `k` must be non-zero multiples of 128, 256 and 64 (`step::GEMM_BF16_TILE`), and
/// none above `i32::MAX`, since ThunderKittens' `gl` stores each dimension as an `int`; any other
/// shape returns [`KernelError::Shape`] before any pointer is touched, and launches nothing. A
/// failed launch returns the launch's own `cudaError_t`, never an error an earlier, unrelated
/// runtime call left behind.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a`, `b` and `c` must be
/// 16-byte-aligned device pointers to at least `m*k`, `n*k` and `m*n` bf16 elements, valid for
/// the whole kernel, with `c` not overlapping `a` or `b`. `stream` must be a live `cudaStream_t`
/// of the current device, or null for the default stream.
pub unsafe fn gemm_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and catches every C++ exception before returning.
    let code = unsafe { scratchy_cuda_gemm_bf16_abt(a, b, c, m, n, k, stream) };
    check(code, KernelError::Shape { m, n, k })
}

/// Launch the bf16 GEMM `C = A · Bᵀ` on 64-column tiles (`kernels/gemm_bf16.cu`'s template at one
/// 64-column B block): the operands and the contract of [`gemm_bf16_abt`], for an `n` that is a
/// multiple of 64 (Gemma 4's 128, 704 and 2112). On an `n` both take, each output element is the
/// bits [`gemm_bf16_abt`] gives: the same WGMMA K order. Asynchronous.
///
/// `m`, `n` and `k` must be non-zero multiples of 128, 64 and 64 (`step::GEMM_COL64_TILE`), none
/// above `i32::MAX`; any other shape returns [`KernelError::Shape`] before any pointer is touched,
/// and launches nothing.
///
/// # Safety
///
/// As [`gemm_bf16_abt`]'s.
pub unsafe fn gemm_col64_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gemm_bf16_abt`'s.
    let code = unsafe { scratchy_cuda_gemm_col64_bf16_abt(a, b, c, m, n, k, stream) };
    check(code, KernelError::Shape { m, n, k })
}

// THE ROW KERNELS (RMSNorm, add, SiluMul). Each launcher takes `(rows, cols)` and applies one
// shape rule: `rows` and `cols` non-zero, `cols` a multiple of 8 (one 16-byte vector of bf16:
// `step::RMS_NORM_BF16_COLS`, `ADD_BF16_COLS`, `SILU_MUL_BF16_COLS`), and `rows · cols` at most
// `i32::MAX`. Any other shape returns `KernelError::RowShape` before any pointer is touched, and
// launches nothing. A failed launch returns the launch's own `cudaError_t`; a C++ exception from
// ThunderKittens' host code (a tensor-map encode), `KernelError::HostException`. Each is a
// ThunderKittens 16-row tile kernel whose dynamic shared memory is over the default 48 KiB, so
// each is in `PREPARE`, which must have run on the current device first.
//
// At `rows == 1` the four RMSNorm launchers (the norm, the unit norm, and the add-norm folds) take
// `kernels/rms_norm_bf16.cu`'s one-row form (FP8 S8 W1): one CTA per 256-column tile, each
// computing the row's sum of squares in the 16-row kernel's order, so the bits are its bits.

/// Launch the bf16 RMSNorm: `y[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps) · (gain_offset +
/// gain[c])`, `x` and `y` `[rows, cols]`, `gain` `[cols]`, in f32 with one bf16 rounding per
/// output. `gain_offset` is the convention's (`GainConvention::offset`). Asynchronous; the shape
/// rule is the row kernels'.
///
/// # Safety
///
/// `x`, `gain` and `y` must be 16-byte-aligned device pointers to at least `rows·cols`, `cols`
/// and `rows·cols` bf16 elements, valid for the whole kernel, with `y` overlapping neither input.
/// `stream` must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn rms_norm_bf16(
    x: *const c_void,
    gain: *const c_void,
    y: *mut c_void,
    (rows, cols): (u32, u32),
    eps: f32,
    gain_offset: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer.
    let code =
        unsafe { scratchy_cuda_rms_norm_bf16(x, gain, y, rows, cols, eps, gain_offset, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 add and RMSNorm in one (#1049, `kernels/rms_norm_bf16.cu`'s `ADD_NORM` form):
/// `sum = a + b`, then `y` the norm of `sum` as [`rms_norm_bf16`] computes it, both `[rows,
/// cols]`. The bits of [`add_bf16`] then [`rms_norm_bf16`] of its output. Asynchronous; the shape
/// rule is the row kernels'.
///
/// # Safety
///
/// `a`, `b`, `sum` and `y` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each and `gain` to `cols`, valid for the whole kernel, with `sum` and `y` overlapping
/// neither each other nor an input. `stream` must be a live `cudaStream_t` of the current device,
/// or null for the default stream.
pub unsafe fn add_rms_norm_bf16(
    (a, b): (*const c_void, *const c_void),
    gain: *const c_void,
    (sum, y): (*mut c_void, *mut c_void),
    (rows, cols): (u32, u32),
    (eps, gain_offset): (f32, f32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_add_rms_norm_bf16(a, b, gain, sum, y, rows, cols, eps, gain_offset, stream)
    };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 RMSNorm and residual add in one (#1049, `kernels/rms_norm_bf16.cu`'s
/// `NORM_ADD` form): `out = residual + bf16(norm(x))`, the norm as [`rms_norm_bf16`] computes it.
/// The bits of [`rms_norm_bf16`] then [`add_bf16`] of its output and `residual`. Asynchronous; the
/// shape rule is the row kernels'.
///
/// # Safety
///
/// `x`, `residual` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each and `gain` to `cols`, valid for the whole kernel, with `out` overlapping neither
/// input. `stream` must be a live `cudaStream_t` of the current device, or null for the default
/// stream.
pub unsafe fn norm_add_bf16(
    (x, residual): (*const c_void, *const c_void),
    gain: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    (eps, gain_offset): (f32, f32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_norm_add_bf16(x, gain, residual, out, rows, cols, eps, gain_offset, stream)
    };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 add `out = a + b` over `[rows, cols]`: one f32 add and one bf16 rounding per
/// element. Asynchronous; the shape rule is the row kernels'.
///
/// # Safety
///
/// `a`, `b` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each, valid for the whole kernel, with `out` overlapping neither input. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn add_bf16(
    a: *const c_void,
    b: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_add_bf16(a, b, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = gate / (1 + e^(−gate)) · up` over `[rows, cols]`, in f32 with one bf16
/// rounding per element. Asynchronous; the shape rule is the row kernels'.
///
/// # Safety
///
/// `gate`, `up` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each, valid for the whole kernel, with `out` overlapping neither input. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn silu_mul_bf16(
    gate: *const c_void,
    up: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_silu_mul_bf16(gate, up, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 RMSNorm with no gain (Gemma's `v_norm`, `kernels/rms_norm_bf16.cu`'s unit
/// instantiation): `y[r, c] = x[r, c] / sqrt(mean(x[r, :]²) + eps)`, `x` and `y` `[rows, cols]`,
/// the bits of [`rms_norm_bf16`] with a gain of all ones and offset 0. Asynchronous; the shape rule
/// is the row kernels'.
///
/// # Safety
///
/// `x` and `y` must be 16-byte-aligned device pointers to at least `rows·cols` bf16 elements,
/// valid for the whole kernel, with `y` not overlapping `x`. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn rms_norm_unit_bf16(
    x: *const c_void,
    y: *mut c_void,
    (rows, cols): (u32, u32),
    eps: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_rms_norm_unit_bf16(x, y, rows, cols, eps, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = 0.5 · gate · (1 + tanh(√(2/π) · (gate + 0.044715 · gate³))) · up` (GeGLU,
/// `kernels/silu_mul_bf16.cu`'s GELU instantiation) over `[rows, cols]`, in f32 with one bf16
/// rounding per element. Asynchronous; the shape rule is the row kernels'.
///
/// # Safety
///
/// As [`silu_mul_bf16`]'s.
pub unsafe fn gelu_mul_bf16(
    gate: *const c_void,
    up: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_gelu_mul_bf16(gate, up, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

// GEMMA'S ONE-OPERAND ROW MAPS (`kernels/pointwise_bf16.cu`): the row kernels' shape rule
// (`step::SCALE_BF16_COLS`, `SOFT_CAP_BF16_COLS`), one setup in `PREPARE`. `out` may equal `x`
// exactly (the same bits as out of place), and may overlap it in no other way.

/// Launch the bf16 `out = x · scale` over `[rows, cols]` (Gemma's embedding scale): one f32
/// multiply of the exact input, one bf16 rounding. Asynchronous.
///
/// # Safety
///
/// `x` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16 elements,
/// valid for the whole kernel, `out` equal to `x` or overlapping it nowhere. `stream` must be a
/// live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn scale_bf16(
    x: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_scale_bf16(x, out, rows, cols, scale, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = x · w[0]` over `[rows, cols]`, `w` a `[1]` bf16 weight (Gemma's
/// `layer_scalar`) widened exactly: one f32 multiply, one bf16 rounding. Asynchronous.
///
/// # Safety
///
/// As [`scale_bf16`]'s, and `w` a device pointer to one bf16, overlapping neither `x` nor `out`.
pub unsafe fn layer_scale_bf16(
    x: *const c_void,
    w: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_layer_scale_bf16(x, w, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 `out = cap · tanh(x / cap)` over `[rows, cols]` (Gemma's final-logit
/// soft-cap), in f32 with one bf16 rounding. Asynchronous.
///
/// # Safety
///
/// As [`scale_bf16`]'s.
pub unsafe fn soft_cap_bf16(
    x: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    cap: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_soft_cap_bf16(x, out, rows, cols, cap, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 partial rotary embedding over `x` and `out` `[rows, cols]`, `cols = heads ·
/// head_dim`: per head, the first `rot_dim` lanes rotated in NeoX pairing within that width
/// (`(d, d + rot_dim/2)`) by `table`'s row for the token, `[cos(0..R/2) | sin(0..R/2)]`, in f32
/// with one bf16 rounding per output; the lanes `rot_dim..head_dim` copied bit for bit.
/// Asynchronous.
///
/// Refused with [`KernelError::RopeShape`], before any pointer is touched, unless `rows`, `cols`
/// and `head_dim` are non-zero, `rot_dim` is 64 or 256 (the half-rotated entries of
/// `step::ROPE_BF16_KERNEL_SHAPES`, the widths the kernel is instantiated for), `head_dim` a multiple of 64
/// (`step::ROPE_BF16_KERNEL_HEAD_MULTIPLE`) and of `rot_dim`, `cols` a multiple of `head_dim`, and
/// `rows · cols` at most `i32::MAX`. A failed launch returns the launch's own `cudaError_t`.
///
/// # Safety
///
/// `x`, `table` and `out` must be 16-byte-aligned device pointers to at least `rows·cols`,
/// `rows·rot_dim` and `rows·cols` bf16 elements, valid for the whole kernel, with `out`
/// overlapping neither input. `stream` must be a live `cudaStream_t` of the current device, or
/// null for the default stream.
pub unsafe fn rope_bf16(
    x: *const c_void,
    table: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    (head_dim, rot_dim): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code =
        unsafe { scratchy_cuda_rope_bf16(x, table, out, rows, cols, head_dim, rot_dim, stream) };
    check(
        code,
        KernelError::RopeShape {
            rows,
            cols,
            head_dim,
            rot_dim,
        },
    )
}

/// [`rope_bf16`] in either pairing: `HalfRotated` is [`rope_bf16`] itself, the same launch; in
/// `HalfHead` (Gemma 4's proportional rope) lane `d < rot_dim/2` pairs with lane `d + head_dim/2`,
/// and every other lane is copied bit for bit. Asynchronous.
///
/// Refused with [`KernelError::RopeShape`] as [`rope_bf16`], except that `(rot_dim, pairing)` must
/// be an entry of `step::ROPE_BF16_KERNEL_SHAPES`, and a `HalfHead` `head_dim` must be 512
/// (`step::ROPE_BF16_HALF_HEAD_DIM`).
///
/// # Safety
///
/// As [`rope_bf16`]'s.
pub unsafe fn rope_paired_bf16(
    x: *const c_void,
    table: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    (head_dim, rot_dim, pairing): (u32, u32, RopePairing),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // The launcher's pairing codes, `kernels/rope_bf16.cu`'s `PAIRING_*`.
    let code = match pairing {
        RopePairing::HalfRotated => 0,
        RopePairing::HalfHead => 1,
    };
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_rope_paired_bf16(x, table, out, rows, cols, head_dim, rot_dim, code, stream)
    };
    check(
        code,
        KernelError::RopeShape {
            rows,
            cols,
            head_dim,
            rot_dim,
        },
    )
}

/// Launch the bf16 per-head gate split: for head `h` and lane `d`, `q[r, h·hd + d] =
/// qg[r, 2h·hd + d]` and `gate[r, h·hd + d] = qg[r, 2h·hd + hd + d]`, `qg` `[rows, 2·half_cols]`
/// and `q`, `gate` `[rows, half_cols]`. Copies, so bit-exact. Asynchronous.
///
/// Refused with [`KernelError::GateShape`], before any pointer is touched, unless `rows` and
/// `half_cols` are non-zero, `head_dim` is 256 (`step::GATE_SPLIT_BF16_HEAD_DIM`, the one head dim
/// the kernel is instantiated for), `half_cols` a multiple of `head_dim`, and
/// `rows · 2·half_cols` at most `i32::MAX`. A failed launch returns the launch's own
/// `cudaError_t`.
///
/// # Safety
///
/// `qg`, `q` and `gate` must be 16-byte-aligned device pointers to at least `rows·2·half_cols`,
/// `rows·half_cols` and `rows·half_cols` bf16 elements, valid for the whole kernel, no two
/// overlapping. `stream` must be a live `cudaStream_t` of the current device, or null for the
/// default stream.
pub unsafe fn gate_split_bf16(
    qg: *const c_void,
    q: *mut c_void,
    gate: *mut c_void,
    (rows, half_cols): (u32, u32),
    head_dim: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code =
        unsafe { scratchy_cuda_gate_split_bf16(qg, q, gate, rows, half_cols, head_dim, stream) };
    check(
        code,
        KernelError::GateShape {
            rows,
            half_cols,
            head_dim,
        },
    )
}

/// Launch the bf16 `out = attn / (1 + e^(−gate))` over `[rows, cols]`, in f32 with one bf16
/// rounding per element. Asynchronous; the shape rule is the row kernels'
/// (`step::GATE_APPLY_BF16_COLS`).
///
/// # Safety
///
/// `attn`, `gate` and `out` must be 16-byte-aligned device pointers to at least `rows·cols` bf16
/// elements each, valid for the whole kernel, with `out` overlapping neither input. `stream` must
/// be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gate_apply_bf16(
    attn: *const c_void,
    gate: *const c_void,
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_gate_apply_bf16(attn, gate, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the bf16 shared-expert gate scale `out = routed + shared · (1 / (1 + e^(−g)))` over
/// `[rows, cols]`, `g` `[rows, 1]` broadcast across its row, in f32 with one bf16 rounding per
/// element (`kernels/gate_scale_bf16.cu`). Asynchronous; the shape rule is the row kernels'
/// (`step::GATE_SCALE_BF16_COLS`).
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `routed`, `shared` and `out` must
/// be 16-byte-aligned device pointers to at least `rows·cols` bf16 elements each and `g` to `rows`,
/// valid for the whole kernel, with `out` overlapping no input. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gate_scale_bf16(
    (routed, shared, g): (*const c_void, *const c_void, *const c_void),
    out: *mut c_void,
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe { scratchy_cuda_gate_scale_bf16(routed, shared, g, out, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Launch the paged KV cache's write: for each token row `t` whose `slot_mapping[t]` is not
/// `u32::MAX`, row `t` of `k` and of `v` (`[rows, kv_heads · head_dim]`) copied bit for bit into
/// `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, head_dim]`) at that slot. A slot whose
/// block is at or past `num_blocks` writes nothing. Asynchronous.
///
/// Refused with [`KernelError::KvAppendShape`], before any pointer is touched, unless `rows` and
/// `kv_heads` are non-zero, `head_dim` is 256 or 512 (`step::KV_APPEND_BF16_HEAD_DIMS`, the head
/// dims the kernel is instantiated for), `block_size` 64 (`step::CUDA_KV_BLOCK_SIZE`), and
/// `rows · kv_heads · head_dim` at most `i32::MAX`. A `num_blocks` of 0 writes nothing and
/// launches nothing. A failed
/// launch returns the launch's own `cudaError_t`.
///
/// # Safety
///
/// `k` and `v` must be 16-byte-aligned device pointers to at least `rows · kv_heads · head_dim`
/// bf16, `slot_mapping` to `rows` u32, `k_cache` and `v_cache` to `num_blocks · kv_heads · 64 ·
/// head_dim` bf16 each, valid for the whole kernel, the caches overlapping no input. `stream`
/// must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn kv_append_bf16(
    (k, v): (*const c_void, *const c_void),
    slot_mapping: *const c_void,
    (k_cache, v_cache): (*mut c_void, *mut c_void),
    (rows, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_kv_append_bf16(
            k,
            v,
            slot_mapping,
            k_cache,
            v_cache,
            rows,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            stream,
        )
    };
    check(
        code,
        KernelError::KvAppendShape {
            rows,
            kv_heads,
            head_dim,
            block_size,
        },
    )
}

/// Launch the causal GQA prefill attention of `q` (`[rows, q_heads · 256]`) over one layer's
/// paged `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, 256]`) into `out`, the keys read
/// through `block_table`, `P` and `L` read from `sequence`: query row `i` attends to keys `0 ..=
/// P + i` below `P + L`, query head `g` to kv head `g / (q_heads / kv_heads)`, the softmax scaled
/// by `scale`. Asynchronous. The full causal attention: [`attn_prefill_windowed_bf16`] at a
/// window of 0, with its contract.
///
/// Refused with [`KernelError::AttnShape`], before any pointer is touched, unless `head_dim` is
/// 256 (`step::ATTN_PREFILL_BF16_HEAD_DIM`), `rows` a non-zero multiple of 128
/// (`step::ATTN_PREFILL_BF16_ROWS`), `q_heads` and `kv_heads` non-zero with `q_heads` a multiple
/// of `kv_heads`, `block_size` 64, `num_blocks` non-zero, and `scale` finite and above 0. Returns
/// [`KernelError::HostException`] if building a tensor map throws, and the launch's own
/// `cudaError_t` if it fails.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `q` and `out` must be
/// 16-byte-aligned device pointers to `rows · q_heads · 256` bf16, `out` overlapping no input;
/// `k_cache`, `v_cache` to `num_blocks · kv_heads · 64 · 256` bf16 each; `sequence` to `[P, L]`
/// u32 with `1 ≤ L ≤ rows`; `block_table` to at least `ceil((P + L) / 64)` u32 block ids, each
/// below `num_blocks`. All valid for the whole kernel. `stream` must be a live `cudaStream_t` of
/// the current device, or null for the default stream.
pub unsafe fn attn_prefill_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the contract above, which is the windowed launch's.
    unsafe {
        attn_prefill_windowed_bf16(
            (q, out),
            (k_cache, v_cache),
            (block_table, sequence),
            (rows, q_heads, kv_heads, head_dim),
            (block_size, num_blocks),
            (scale, 0),
            stream,
        )
    }
}

/// [`attn_prefill_bf16`] under a sliding window: query row `i` also drops the keys `j ≤ P + i −
/// window`, so it sees its own key and the `window − 1` before it (Gemma 4's sliding layers,
/// `window` 1024); `window` 0 is the full causal attention, the same launch. The KV tiles wholly
/// below every row's window in a CTA are never read, so the blocks there may hold anything.
/// Asynchronous.
///
/// Refused with [`KernelError::AttnShape`] as [`attn_prefill_bf16`], and also unless `window` is
/// a multiple of 64 (`step::ATTN_WINDOW_MULTIPLE`).
///
/// # Safety
///
/// As [`attn_prefill_bf16`]'s, but `block_table` need only name valid blocks from the window's
/// first tile on: an entry below it is never read.
pub unsafe fn attn_prefill_windowed_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape, scale and window before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_attn_prefill_bf16(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnShape {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// [`attn_prefill_windowed_bf16`] at head dim 512 (Gemma 4's global layers,
/// `kernels/attn_prefill_d512_bf16.cu`): `q` and `out` are `[rows, q_heads · 512]`, the caches
/// `[num_blocks, kv_heads, 64, 512]`. Each of a CTA's two consumer warpgroups keeps one
/// 256-column half of the output over the same 64 query rows; the masks and the online softmax are
/// the head-dim-256 kernel's. Asynchronous.
///
/// Refused with [`KernelError::AttnShape`] as [`attn_prefill_windowed_bf16`], except that
/// `head_dim` must be 512 (`step::ATTN_D512_BF16_HEAD_DIM`).
///
/// # Safety
///
/// As [`attn_prefill_windowed_bf16`]'s, with 512 for 256 in every size.
pub unsafe fn attn_prefill_d512_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_prefill_windowed_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_prefill_d512_bf16(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnShape {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// Launch the narrow bf16 GEMM `C = A · Bᵀ` for at most 64 output columns: `A` is `[m, k]`, `B`
/// `[n, k]` and `C` `[m, n]`, row-major bf16, accumulated in f32 and rounded once. ThunderKittens'
/// WGMMA over 64-row tiles of `A` against all of `B`'s rows, in a fixed K order, so an output
/// row's bits depend only on its own row of `A` and on `B`, not on the launch or on `m`.
/// Asynchronous.
///
/// `n = 1` (`step::GEMM_NARROW_BF16_ONE`, a shared expert's gate) runs a ThunderKittens
/// warp-vector kernel instead, one warp per row in a fixed order, its one bf16 per row rounded
/// once.
///
/// Refused with [`KernelError::NarrowShape`], before any pointer is touched, unless `m` is
/// non-zero, `n` 1 or a non-zero multiple of 8 and at most 64 (`step::GEMM_NARROW_BF16_MULTIPLE`,
/// `GEMM_NARROW_BF16_MAX_N`), `k` a non-zero multiple of 8 (of 256 when `n` is 1,
/// `step::GEMM_NARROW_BF16_ONE_K_MULTIPLE`), and `m` and `k` at most `i32::MAX`.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a`, `b` and `c` must be
/// 16-byte-aligned device pointers to at least `m*k`, `n*k` and `m*n` bf16, valid for the whole
/// kernel, with `c` overlapping neither input. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gemm_narrow_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_gemm_narrow_bf16_abt(a, b, c, m, n, k, stream) };
    check(code, KernelError::NarrowShape { m, n, k })
}

/// The members of a grouped GEMM launch (#1049): 2 or 3 `(B, C, n)`, sharing `A`, its `m` and
/// `k`.
pub type GroupOperands<'g> = &'g [(*const c_void, *mut c_void, u32)];

/// The member columns of `members` as the launcher reads them, and its refusal.
fn group_parts(
    kernel: &'static str,
    members: GroupOperands<'_>,
    (m, k): (u32, u32),
) -> ([*const c_void; 3], [*mut c_void; 3], [u32; 3], KernelError) {
    let (mut b, mut c, mut n) = ([std::ptr::null(); 3], [std::ptr::null_mut(); 3], [0u32; 3]);
    for (i, &(bi, ci, ni)) in members.iter().take(3).enumerate() {
        (b[i], c[i], n[i]) = (bi, ci, ni);
    }
    (b, c, n, KernelError::GroupShape { kernel, m, k, n })
}

/// [`gemm_bf16_abt`] of every member of `members` in one launch (`kernels/gemm_bf16.cu`'s
/// grouped template): member `i`'s `C_i [m, n_i] = A · B_iᵀ`, each the bits of its own
/// [`gemm_bf16_abt`]. Asynchronous.
///
/// Refused with [`KernelError::GroupShape`], before any pointer is touched, unless there are 2 or
/// 3 members and each meets [`gemm_bf16_abt`]'s rule with the shared `m` and `k`.
///
/// # Safety
///
/// As [`gemm_bf16_abt`]'s, for `a` and every member's `b` and `c`; and no member's `c` overlaps
/// another's.
pub unsafe fn gemm_group_bf16_abt(
    a: *const c_void,
    members: GroupOperands<'_>,
    (m, k): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let (b, c, n, refused) = group_parts("gemm_group_bf16_abt", members, (m, k));
    let count = members.len() as u32;
    // SAFETY: the caller upholds the pointer and stream contract above; the three arrays are
    // three long; the launcher checks the count and every member's shape before it reads a
    // pointer, and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_gemm_group_bf16_abt(
            a,
            b.as_ptr(),
            c.as_ptr(),
            n.as_ptr(),
            m,
            k,
            count,
            stream,
        )
    };
    check(code, refused)
}

/// [`gemm_narrow_bf16_abt`] of every member of `members` in one launch: member `i`'s
/// `C_i [m, n_i] = A · B_iᵀ`, each the bits of its own [`gemm_narrow_bf16_abt`]. Asynchronous.
///
/// Refused with [`KernelError::GroupShape`], before any pointer is touched, unless there are 2 or
/// 3 members and each meets [`gemm_narrow_bf16_abt`]'s rule with the shared `m` and `k`, its `n`
/// a multiple of 8 (not the `n = 1` warp-vector form).
///
/// # Safety
///
/// As [`gemm_narrow_bf16_abt`]'s, for `a` and every member's `b` and `c`; and no member's `c`
/// overlaps another's.
pub unsafe fn gemm_narrow_group_bf16_abt(
    a: *const c_void,
    members: GroupOperands<'_>,
    (m, k): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let (b, c, n, refused) = group_parts("gemm_narrow_group_bf16_abt", members, (m, k));
    let count = members.len() as u32;
    // SAFETY: as `gemm_group_bf16_abt`'s.
    let code = unsafe {
        scratchy_cuda_gemm_narrow_group_bf16_abt(
            a,
            b.as_ptr(),
            c.as_ptr(),
            n.as_ptr(),
            m,
            k,
            count,
            stream,
        )
    };
    check(code, refused)
}

/// [`gemv_split_bf16_abt`] of every member of `members` in two launches at most (the members'
/// CTAs, then one reduce over the split ones): member `i`'s `c_i [n_i] = a · B_iᵀ` at
/// `splits[i]`, its partials at `partials[i]`, each the bits of its own [`gemv_split_bf16_abt`]
/// at that split. Asynchronous.
///
/// Refused with [`KernelError::GroupShape`], before any pointer is touched, unless there are 2 or
/// 3 members, `splits` has one entry per member, and each member meets
/// [`gemv_split_bf16_abt`]'s rule at its split.
///
/// # Safety
///
/// As [`gemv_split_bf16_abt`]'s, for `a` and every member's `b`, `c` and partials; no member's
/// `c` or partials overlaps another's.
pub unsafe fn gemv_group_bf16_abt(
    a: *const c_void,
    members: GroupOperands<'_>,
    partials: &[*mut c_void],
    (k, splits): (u32, &[u32]),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let (b, c, n, refused) = group_parts("gemv_group_bf16_abt", members, (1, k));
    if splits.len() != members.len() || partials.len() != members.len() {
        return Err(refused);
    }
    let (mut s, mut p) = ([1u32; 3], [std::ptr::null_mut(); 3]);
    for (i, (&si, &pi)) in splits.iter().zip(partials).take(3).enumerate() {
        (s[i], p[i]) = (si, pi);
    }
    let count = members.len() as u32;
    // SAFETY: as `gemm_group_bf16_abt`'s, and the splits and partials arrays are three long.
    let code = unsafe {
        scratchy_cuda_gemv_group_bf16_abt(
            a,
            b.as_ptr(),
            c.as_ptr(),
            p.as_ptr(),
            n.as_ptr(),
            k,
            s.as_ptr(),
            count,
            stream,
        )
    };
    check(code, refused)
}

/// The gate/up GEMV (#1049, `kernels/gemv_bf16.cu`): `c [n] = act(bf16(a · gateᵀ), bf16(a ·
/// upᵀ))`, rounded once, `act` a [`crate::step::GatedOp`] code. The bits of
/// [`gemv_split_bf16_abt`] of the gate and the up at `splits`, then [`silu_mul_bf16`] (or
/// [`gelu_mul_bf16`]). Asynchronous.
///
/// Refused with [`KernelError::GateUpShape`], before any pointer is touched, unless each of the
/// gate and the up meets [`gemv_split_bf16_abt`]'s rule at `splits` and `act` is a code.
///
/// # Safety
///
/// As [`gemv_split_bf16_abt`]'s for `a`, `gate`, `up` and `c`; at `splits > 1`, `partials` holds
/// `2 · splits · n_pad` f32 (`n_pad` `n` rounded up to 32), overlapping no operand.
pub unsafe fn gate_up_gemv_bf16(
    (a, gate, up): (*const c_void, *const c_void, *const c_void),
    (c, partials): (*mut c_void, *mut c_void),
    (n, k, splits): (u32, u32, u32),
    act: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape, split and code before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_gate_up_gemv_bf16(a, gate, up, c, partials, n, k, splits, act, stream)
    };
    check(
        code,
        KernelError::GateUpShape {
            kernel: "gate_up_gemv_bf16",
            dims: [1, n, k, splits],
        },
    )
}

/// The expert gate/up GEMV (#1049): per pair `p`, `c[p] = act(bf16(a[p / top_k] · gate_eᵀ),
/// bf16(a[p / top_k] · up_eᵀ))`, `e = topk_idx[p]`, `gate` and `up` expert 0's `[n, k]` of two
/// slabs `expert_rows` rows apart per expert. The bits of [`expert_gemv_split_bf16`] of each, then
/// the activation. Asynchronous; refused with [`KernelError::GateUpShape`] unless each meets that
/// launcher's rule with token rows and `act` is a code.
///
/// # Safety
///
/// As [`expert_gemv_split_bf16`]'s for `a`, each slab, `topk_idx` and `c`; at `splits > 1`,
/// `partials` holds `2 · splits · m · top_k · n` f32, overlapping no operand.
pub unsafe fn expert_gate_up_gemv_bf16(
    (a, gate, up, topk_idx): (*const c_void, *const c_void, *const c_void, *const c_void),
    (c, partials): (*mut c_void, *mut c_void),
    (m, top_k, n, k): (u32, u32, u32, u32),
    (experts, expert_rows, splits): (u32, u32, u32),
    act: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gate_up_gemv_bf16`'s.
    let code = unsafe {
        scratchy_cuda_expert_gate_up_gemv_bf16(
            a,
            gate,
            up,
            topk_idx,
            c,
            partials,
            m,
            top_k,
            n,
            k,
            experts,
            expert_rows,
            splits,
            act,
            stream,
        )
    };
    check(
        code,
        KernelError::GateUpShape {
            kernel: "expert_gate_up_gemv_bf16",
            dims: [m.saturating_mul(top_k), n, k, splits],
        },
    )
}

/// The gate/up GEMM (#1049, `kernels/gemm_bf16.cu`): `c [m, n] = act(bf16(a · gateᵀ), bf16(a ·
/// upᵀ))`, rounded once. At `col64` false the bits of [`gemm_bf16_abt`] of each, then the
/// activation, under that GEMM's rule; at `col64` true [`gemm_col64_bf16_abt`]'s bits and rule.
/// Asynchronous; refused with [`KernelError::GateUpShape`] before any pointer is touched.
///
/// # Safety
///
/// As [`gemm_bf16_abt`]'s for `a`, `gate`, `up` (each `[n, k]`) and `c`.
pub unsafe fn gate_up_gemm_bf16(
    (a, gate, up): (*const c_void, *const c_void, *const c_void),
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    (act, col64): (u32, bool),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and codes before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_gate_up_gemm_bf16(a, gate, up, c, m, n, k, act, u32::from(col64), stream)
    };
    check(
        code,
        KernelError::GateUpShape {
            kernel: "gate_up_gemm_bf16",
            dims: [m, n, k, 1],
        },
    )
}

/// Launch the bf16 GEMV `c = a · Bᵀ`, the one-row GEMM: `a` is `[1, k]`, `B` `[n, k]` and `c`
/// `[1, n]`, row-major bf16, accumulated in f32 and rounded once: [`gemv_split_bf16_abt`] at one
/// split, each row's products added in a fixed K order, so the bits do not depend on the launch;
/// only the `n` elements of `c` are written. Asynchronous.
///
/// Refused with [`KernelError::GemvShape`], before any pointer is touched, unless `n` is
/// non-zero, `k` a non-zero multiple of 8 (`step::GEMV_BF16_MULTIPLE`), and both at most
/// `i32::MAX`.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a`, `b` and `c` must be
/// 16-byte-aligned device pointers to at least `k`, `n*k` and `n` bf16, valid for the whole
/// kernel, with `c` overlapping neither input. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gemv_bf16_abt(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    (n, k): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_gemv_bf16_abt(a, b, c, n, k, stream) };
    check(code, KernelError::GemvShape { n, k })
}

/// [`gemv_bf16_abt`] with its `k` cut into `splits` runs of whole 256-element chunks
/// (`step::gemv_splits` chooses it at expansion): each split's f32 partials go to `partials`, and
/// a second launch adds them in ascending split order and rounds once. Within the GEMV bound of
/// f64, and bit-identical on a repeat; at `splits` 1 it is [`gemv_bf16_abt`]'s bits.
/// Asynchronous.
///
/// Refused with [`KernelError::GemvSplit`], before any pointer is touched, unless the shape is
/// [`gemv_bf16_abt`]'s, `splits` is in `1..=16` (`step::GEMV_MAX_SPLITS`) and cuts `⌈k / 256⌉`
/// chunks into non-empty runs (`step::gemv_split_chunks`), `partials` is not null above one split,
/// and `splits · n_pad` is at most `i32::MAX`.
///
/// # Safety
///
/// As [`gemv_bf16_abt`]'s; and at `splits > 1`, `partials` must be a 16-byte-aligned device
/// pointer to `step::gemv_scratch_bytes(n, 1, splits)` bytes, valid for both launches and
/// overlapping no operand.
pub unsafe fn gemv_split_bf16_abt(
    (a, b, c): (*const c_void, *const c_void, *mut c_void),
    partials: *mut c_void,
    (n, k, splits): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and split before it touches a pointer and catches every C++ exception.
    let code =
        unsafe { scratchy_cuda_gemv_split_bf16_abt(a, b, c, partials, n, k, splits, stream) };
    check(code, KernelError::GemvSplit { n, k, splits })
}

// THE TMA BOX PROBE (`kernels/tma_box_probe.cu`): test support, for `tests/tma_bounds.rs` only.
// No `CudaStep` and no lowering use it. It records what a ThunderKittens TMA tile load and store
// do with a 64×64 bf16 box that runs past its array's edge, swizzled or plain: the facts the GEMV's
// and the narrow GEMM's choice of tiles rests on. Its shared memory is under 48 KiB, so it is not
// in `PREPARE`.

/// Test support for `tests/tma_bounds.rs` only. TMA-load the 64×64 box at tile `(tile_row,
/// tile_col)` of `src` (`[rows, cols]` bf16) into a swizzled (`swizzled`) or plain shared tile,
/// over the tile first loaded from `poison` (`[64, 64]` bf16), then TMA-store the tile into `dump`
/// (`[64, 64]` bf16). Refused with [`KernelError::RowShape`] when `rows` or `cols` is zero, `cols`
/// is not a multiple of 8, or a dimension or tile coordinate is at or above `i32::MAX / 64`.
/// Asynchronous.
///
/// # Safety
///
/// `src`, `poison` and `dump` must be 16-byte-aligned device pointers to `rows · cols`, 4096 and
/// 4096 bf16, valid for the whole kernel; `dump` overlaps neither input. A swizzled load may read
/// past `src`'s `rows · cols` elements, up to the next 64-element boundary of its last row (that
/// is what it records), so `src` must be readable that far. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn tma_box_probe_load(
    (src, poison, dump): (*const c_void, *const c_void, *mut c_void),
    (rows, cols): (u32, u32),
    (tile_row, tile_col): (u32, u32),
    swizzled: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_tma_box_probe_load(
            src,
            poison,
            dump,
            rows,
            cols,
            tile_row,
            tile_col,
            u32::from(swizzled),
            stream,
        )
    };
    check(code, KernelError::RowShape { rows, cols })
}

/// Test support for `tests/tma_bounds.rs` only. TMA-load `pattern` (`[64, 64]` bf16) into a
/// swizzled (`swizzled`) or plain shared tile, then TMA-store it at tile `(tile_row, tile_col)` of
/// `dst` (`[rows, cols]` bf16). Refused as [`tma_box_probe_load`]. Asynchronous.
///
/// # Safety
///
/// `pattern` and `dst` must be 16-byte-aligned device pointers to 4096 and `rows · cols` bf16,
/// valid for the whole kernel, not overlapping. A swizzled store may write past `dst`'s
/// `rows · cols` elements, up to the next 64-element boundary of its last row (that is what it
/// records), so `dst` must be writable that far. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn tma_box_probe_store(
    (pattern, dst): (*const c_void, *mut c_void),
    (rows, cols): (u32, u32),
    (tile_row, tile_col): (u32, u32),
    swizzled: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_tma_box_probe_store(
            pattern,
            dst,
            rows,
            cols,
            tile_row,
            tile_col,
            u32::from(swizzled),
            stream,
        )
    };
    check(code, KernelError::RowShape { rows, cols })
}

// THE TMA RING PROBE (`kernels/tma_ring_probe.cu`): test support, for
// `tests/gemv_fp8_ring_probe.rs` only (FP8 S8 W2c). No `CudaStep` and no lowering use it. It
// streams an e4m3 weight through the fp8 GEMV's grid and rings with no dot product, at five sizes
// of TMA operation, to tell whether that GEMV is bound by its TMA operation rate. It sets its own
// shared-memory attribute per call, so it is not in `PREPARE`.

/// Test support for `tests/gemv_fp8_ring_probe.rs` only. Stream `b` (`[n, k]` e4m3) through the
/// ring probe's variant `variant` (0: 256-byte vectors at 3 rows a warp; 1: 256-byte vectors;
/// 2: 512-byte 2-row tiles; 3: 4 KB 16-row tiles per 4 warps; 4: 16 KB 16 × 1,024 tiles per 4
/// warps) and store each thread's fold of the bytes it read into `out`. Refused with
/// [`KernelError::RowShape`] (`rows` `n`, `cols` `k`) unless `n` and `k` are non-zero and at most
/// `i32::MAX / 2`, `variant` at most 4, `k` a multiple of 256 (1,024 for variant 4) and, for
/// variants 2 to 4, `n` a multiple of 32. Asynchronous.
///
/// # Safety
///
/// `b` must be a 16-byte-aligned device pointer to `n · k` bytes and `out` to `16 · n + 256` f32,
/// valid for the whole kernel, not overlapping. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn tma_ring_probe(
    (b, out): (*const c_void, *mut c_void),
    (n, k): (u32, u32),
    variant: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_tma_ring_probe(b, out, n, k, variant, stream) };
    check(code, KernelError::RowShape { rows: n, cols: k })
}

/// Test support for `tests/gemv_fp8_ring_probe.rs` only (the ceiling probe, /tmp/d-cap.md §3
/// P-A). Stream `b` (`[n, k]` e4m3) in the tensor-core GEMV's geometry (132 CTAs, stream-K runs,
/// one producer warp, one consumer warpgroup reading one byte per lane) as variant `variant`: 0
/// A1 `b64` (`st<128, 64>` boxes), 1 A2 `b64+a` (plus a `[1, k]` A box from `a`), 2 A3 `b128`, 3
/// A4 `b128+a`, 4 A5 `b256` (`st<64, 256>`), 5 A6 `b64x2cta` (A1 at 264 CTAs); each thread's fold
/// into `out`. Refused with [`KernelError::RowShape`] (`rows` `n`, `cols` `k`) unless `n` is a
/// non-zero multiple of the variant's box rows, `k` of its K step, both at most `i32::MAX / 2`,
/// and `variant` at most 5. Asynchronous.
///
/// # Safety
///
/// `b` must be a 16-byte-aligned device pointer to `n · k` bytes, `a` to `k` bytes and `out` to
/// `264 · 256` f32, valid for the whole kernel, `out` overlapping neither. `stream` as
/// [`tma_ring_probe`]'s.
#[doc(hidden)]
pub unsafe fn tc_stream_probe(
    (b, a, out): (*const c_void, *const c_void, *mut c_void),
    (n, k): (u32, u32),
    variant: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above; the launcher checks the shape before it touches a pointer.
    let code = unsafe { scratchy_cuda_tc_stream_probe(b, a, out, n, k, variant, stream) };
    check(code, KernelError::RowShape { rows: n, cols: k })
}

/// Test support for `tests/gemv_fp8_ring_probe.rs` only (FP8 S8 W2d). Stream `b` (`[n, k]` e4m3)
/// through the consume probe's variant `consume` (0 wait, 1 every byte, 2 the widen, 3 the dot
/// with the activation row `a` (`[1, k]` bf16), 4 the dot with the GEMV's quantizing prologue) at
/// `rows` (3 or 4) rows a warp and exactly `ctas` (2 or 3) CTAs per SM, and store each thread's
/// fold into `out` (int). Refused with [`KernelError::RowShape`] (`rows` `n`, `cols` `k`) unless
/// `n` is non-zero and at most `i32::MAX / 2`, `k` a non-zero multiple of 256 at most 17,408 (at
/// most 8,192 for variants 3 and 4 at 3 CTAs), and `consume`, `rows` and `ctas` in range.
/// Asynchronous.
///
/// # Safety
///
/// `b` must be a 16-byte-aligned device pointer to `n · k` bytes, `a` to `k` bf16 and `out` to
/// `11 · n + 256` i32, valid for the whole kernel, `out` overlapping neither input. `stream` as
/// [`tma_ring_probe`]'s.
#[doc(hidden)]
pub unsafe fn tma_consume_probe(
    (b, a, out): (*const c_void, *const c_void, *mut c_void),
    (n, k): (u32, u32),
    (consume, rows, ctas): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code =
        unsafe { scratchy_cuda_tma_consume_probe(b, a, out, n, k, consume, rows, ctas, stream) };
    check(code, KernelError::RowShape { rows: n, cols: k })
}

/// Test support for `tests/gemv_fp8_ring_probe.rs` only: the CTAs per SM the CUDA runtime gives a
/// probe instance on the current device: ring variant `variant` when `consume` is `None`, else
/// consume variant `(consume, rows, ctas)` as [`tma_consume_probe`] takes it. It sets the
/// instance's shared-memory attribute, so call it outside a capture.
#[doc(hidden)]
pub fn tma_probe_occupancy(
    variant: u32,
    consume: Option<(u32, u32, u32)>,
) -> Result<i32, KernelError> {
    let (c, rows, ctas) = consume.unwrap_or((5, 4, 2));
    let mut blocks = 0i32;
    // SAFETY: `blocks` is a live local the call writes once; no device pointer is passed.
    let code = unsafe { scratchy_cuda_tma_probe_occupancy(variant, c, rows, ctas, &mut blocks) };
    check(code, KernelError::RowShape { rows, cols: ctas }).map(|()| blocks)
}

/// Launch the GQA decode attention of one query row `q` (`[1, q_heads · 256]`) over one layer's
/// paged `k_cache` and `v_cache` (`[num_blocks, kv_heads, 64, 256]`) into `out`, the keys
/// `0 .. P + L` read through `block_table`, `P` and `L` read from `sequence`; query head `g` reads
/// kv head `g / (q_heads / kv_heads)`, the softmax scaled by `scale`. Whole 64-slot blocks are
/// loaded; the slots at or past `P + L` are neutralised before use (their scores set to −inf,
/// their V rows zeroed), so no unwritten slot reaches the output. Asynchronous. Every key:
/// [`attn_decode_windowed_bf16`] at a window of 0, with its contract.
///
/// Refused with [`KernelError::AttnDecodeShape`], before any pointer is touched, unless
/// `head_dim` is 256 (`step::ATTN_DECODE_BF16_HEAD_DIM`), `q_heads` and `kv_heads` non-zero with
/// `q_heads` a multiple of `kv_heads` by at most 16 (`step::ATTN_DECODE_BF16_MAX_GROUP`),
/// `block_size` 64, `num_blocks` non-zero, and `scale` finite and above 0. Returns the launch's
/// own `cudaError_t` if it fails.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `q` and `out` must be 16-byte-aligned device pointers to `q_heads · 256` bf16, `out`
/// overlapping no input; `k_cache`, `v_cache` to `num_blocks · kv_heads · 64 · 256` bf16 each;
/// `sequence` to `[P, L]` u32 with `L ≥ 1`; `block_table` to at least `ceil((P + L) / 64)` u32
/// block ids, each below `num_blocks`. All valid for the whole kernel. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn attn_decode_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the contract above, which is the windowed launch's.
    unsafe {
        attn_decode_windowed_bf16(
            (q, out),
            (k_cache, v_cache),
            (block_table, sequence),
            (q_heads, kv_heads, head_dim),
            (block_size, num_blocks),
            (scale, 0),
            stream,
        )
    }
}

/// [`attn_decode_bf16`] under a sliding window: the one query, at `P + L − 1`, sees only the keys
/// from `max(0, P + L − window)` on (Gemma 4's sliding layers, `window` 1024); `window` 0 is every
/// key, the same launch. The KV tiles wholly below the window are never read, so the blocks there
/// may hold anything. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`] as [`attn_decode_bf16`], and also unless
/// `window` is a multiple of 64 (`step::ATTN_WINDOW_MULTIPLE`).
///
/// # Safety
///
/// As [`attn_decode_bf16`]'s, but `block_table` need only name valid blocks from the window's
/// first tile on: an entry below it is never read.
pub unsafe fn attn_decode_windowed_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape, scale and window before it touches a pointer.
    let code = unsafe {
        scratchy_cuda_attn_decode_bf16(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnDecodeShape {
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// [`attn_decode_windowed_bf16`] at head dim 512 (Gemma 4's global layers,
/// `kernels/attn_decode_d512_bf16.cu`): `q` and `out` are `[1, q_heads · 512]`, the caches
/// `[num_blocks, kv_heads, 64, 512]`. Both consumer warpgroups walk every KV tile in ascending
/// order, each keeping one 256-column half of the output, so there is no merge. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`] as [`attn_decode_windowed_bf16`], except that
/// `head_dim` must be 512 (`step::ATTN_D512_BF16_HEAD_DIM`).
///
/// # Safety
///
/// As [`attn_decode_windowed_bf16`]'s, with 512 for 256 in every size.
pub unsafe fn attn_decode_d512_bf16(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_decode_windowed_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_decode_d512_bf16(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnDecodeShape {
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// [`kv_append_bf16`] into an e4m3 cache (FP8 S5, `kernels/kv_append_fp8.cu`): each bf16 element
/// of row `t` is written as `e4m3_satfinite(x)`, round to nearest even and clamped to ±448, at a
/// fixed per-tensor scale of 1.0 (vLLM's `--kv-cache-dtype fp8` for a checkpoint without KV
/// scales). A padding slot and a block at or past `num_blocks` write nothing. Asynchronous.
///
/// Refused with [`KernelError::KvAppendShape`] as [`kv_append_bf16`].
///
/// # Safety
///
/// As [`kv_append_bf16`]'s, but `k_cache` and `v_cache` hold `num_blocks · kv_heads · 64 ·
/// head_dim` e4m3 bytes each.
pub unsafe fn kv_append_fp8(
    (k, v): (*const c_void, *const c_void),
    slot_mapping: *const c_void,
    (k_cache, v_cache): (*mut c_void, *mut c_void),
    (rows, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `kv_append_bf16`'s.
    let code = unsafe {
        scratchy_cuda_kv_append_fp8(
            k,
            v,
            slot_mapping,
            k_cache,
            v_cache,
            rows,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            stream,
        )
    };
    check(
        code,
        KernelError::KvAppendShape {
            rows,
            kv_heads,
            head_dim,
            block_size,
        },
    )
}

/// [`attn_prefill_windowed_bf16`] over an e4m3 cache (FP8 S5, `kernels/attn_prefill_fp8kv.cu`):
/// each K and V tile is widened to bf16 in shared memory (exact) and the bf16 kernel's math runs on
/// it, so the output is the bf16 kernel's on the widened cache, bit for bit. Asynchronous.
///
/// Refused with [`KernelError::AttnShape`] as [`attn_prefill_windowed_bf16`].
///
/// # Safety
///
/// As [`attn_prefill_windowed_bf16`]'s, but `k_cache` and `v_cache` hold `num_blocks · kv_heads ·
/// 64 · 256` e4m3 bytes each.
pub unsafe fn attn_prefill_windowed_fp8kv(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_prefill_windowed_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_prefill_fp8kv(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnShape {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// [`attn_prefill_d512_bf16`] over an e4m3 cache (FP8 S5, `kernels/attn_prefill_d512_fp8kv.cu`),
/// widened on read as [`attn_prefill_windowed_fp8kv`]. Asynchronous.
///
/// Refused with [`KernelError::AttnShape`] as [`attn_prefill_d512_bf16`].
///
/// # Safety
///
/// As [`attn_prefill_d512_bf16`]'s, but `k_cache` and `v_cache` hold `num_blocks · kv_heads · 64 ·
/// 512` e4m3 bytes each.
pub unsafe fn attn_prefill_d512_fp8kv(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (rows, q_heads, kv_heads, head_dim): (u32, u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_prefill_d512_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_prefill_d512_fp8kv(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnShape {
            rows,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// [`attn_decode_windowed_bf16`] over an e4m3 cache (FP8 S5, `kernels/attn_decode_fp8kv.cu`),
/// widened on read as [`attn_prefill_windowed_fp8kv`]. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`] as [`attn_decode_windowed_bf16`].
///
/// # Safety
///
/// As [`attn_decode_windowed_bf16`]'s, but `k_cache` and `v_cache` hold `num_blocks · kv_heads ·
/// 64 · 256` e4m3 bytes each.
pub unsafe fn attn_decode_windowed_fp8kv(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_decode_windowed_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_decode_fp8kv(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnDecodeShape {
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

/// [`attn_decode_d512_bf16`] over an e4m3 cache (FP8 S5, `kernels/attn_decode_d512_fp8kv.cu`),
/// widened on read as [`attn_prefill_windowed_fp8kv`]. Asynchronous.
///
/// Refused with [`KernelError::AttnDecodeShape`] as [`attn_decode_d512_bf16`].
///
/// # Safety
///
/// As [`attn_decode_d512_bf16`]'s, but `k_cache` and `v_cache` hold `num_blocks · kv_heads · 64 ·
/// 512` e4m3 bytes each.
pub unsafe fn attn_decode_d512_fp8kv(
    (q, out): (*const c_void, *mut c_void),
    (k_cache, v_cache): (*const c_void, *const c_void),
    (block_table, sequence): (*const c_void, *const c_void),
    (q_heads, kv_heads, head_dim): (u32, u32, u32),
    (block_size, num_blocks): (u32, u32),
    (scale, window): (f32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `attn_decode_d512_bf16`'s.
    let code = unsafe {
        scratchy_cuda_attn_decode_d512_fp8kv(
            q,
            k_cache,
            v_cache,
            block_table,
            sequence,
            out,
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
            scale,
            window,
            stream,
        )
    };
    check(
        code,
        KernelError::AttnDecodeShape {
            q_heads,
            kv_heads,
            head_dim,
            block_size,
            num_blocks,
        },
    )
}

// THE GDN KERNELS. Four launches make one `GatedDeltaNet` step, in this order on one stream, each
// reading `[slot, q_len, start, 0]` u32 from `sequence` (`start` 1 on a sequence's first chunk, 0
// to continue from its slot) and touching only the `q_len` real rows. Each is a ThunderKittens
// kernel with under 48 KiB of static shared memory; each one's setup in `PREPARE` only loads it.

/// Launch the GDN causal depthwise conv1d, then SiLU: `conv_out[t, c]` (f32) for the real rows of
/// `x [rows, conv_dim]` (bf16) under the taps `w [conv_dim, kernel]` (f32), the rows before 0 the
/// slot's conv ring (zero, never read, on a fresh start); then the ring updated to the last
/// `kernel − 1` inputs. One warp per 128 channels and 64 tokens, the channels a ThunderKittens
/// warp vector, each tap one `fmaf` in ascending order (`kernels/gdn_conv1d.cu`). Asynchronous.
///
/// Refused with [`KernelError::GdnConvShape`], before any pointer is touched, unless `rows` and
/// `conv_dim` are non-zero, `conv_dim` a multiple of 128 (`step::GDN_CONV_CHANNEL_GROUP`),
/// `kernel` in `step::GDN_CONV_KERNEL_RANGE`, `rows · conv_dim` at most `i32::MAX`, and
/// `slot_elems` exactly `conv_dim · (kernel − 1)`.
///
/// # Safety
///
/// `x`, `w`, `conv_out` must be 16-byte-aligned device pointers to `rows·conv_dim` bf16,
/// `conv_dim·kernel` f32 and `rows·conv_dim` f32; `sequence` to 4 u32 with `1 ≤ q_len ≤ rows` and
/// `slot` a slot of `conv_state`, which must hold `(slot + 1) · slot_elems` f32. All valid for
/// the whole kernel; `conv_out` and `conv_state` overlap nothing else. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gdn_conv1d(
    (x, w): (*const c_void, *const c_void),
    (conv_state, slot_elems): (*mut c_void, u64),
    sequence: *const c_void,
    conv_out: *mut c_void,
    (rows, conv_dim, kernel): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_conv1d(
            x, w, conv_state, slot_elems, sequence, conv_out, rows, conv_dim, kernel, stream,
        )
    };
    check(
        code,
        KernelError::GdnConvShape {
            rows,
            conv_dim,
            kernel,
        },
    )
}

/// Launch the GDN gating: for the real rows of `a` and `b` (`[rows, nv]` bf16), `g = −exp(a_log)
/// · softplus(a + dt_bias)` and `beta = sigmoid(b)`, f32 `[rows, nv]` each. One warp per real
/// row, the row a ThunderKittens warp vector (`kernels/gdn_gating.cu`). Asynchronous.
///
/// Refused with [`KernelError::GdnGatingShape`], before any pointer is touched, unless `rows` and
/// `nv` are non-zero, `nv` a multiple of `step::GDN_GATING_NV_MULTIPLE` (the kernel's chunk width;
/// 48 at qwen3.6-27b, 64 at qwen3.5-122b-a10b), and `rows · nv` at most `i32::MAX`.
///
/// # Safety
///
/// `a`, `b` must be device pointers to `rows·nv` bf16, `a_log`, `dt_bias` to `nv` f32, `g`,
/// `beta` to `rows·nv` f32, `sequence` to 4 u32 with `1 ≤ q_len ≤ rows`; all 4-byte aligned and
/// valid for the whole kernel, `g` and `beta` overlapping nothing else. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gdn_gating(
    (a, b): (*const c_void, *const c_void),
    (a_log, dt_bias): (*const c_void, *const c_void),
    sequence: *const c_void,
    (g, beta): (*mut c_void, *mut c_void),
    (rows, nv): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_gating(a, b, a_log, dt_bias, sequence, g, beta, rows, nv, stream)
    };
    check(code, KernelError::GdnGatingShape { rows, nv })
}

/// Launch the GDN recurrent gated delta-rule scan: per value head, the slot's f32 state (zero,
/// never read, on a fresh start) through the real tokens of `conv_out` (`q | k | v`), `g` and
/// `beta`, each token's output into `o [rows, nv·head_v]` (f32); the state written back once.
/// Value head `h` reads key head `h / (nv / nk)`. Each warp holds 16 state rows as one
/// ThunderKittens `rt_fl<16, 128>`: the matrix-vector products are `mul_col` then `row_sum`, the
/// rank-1 update `row_map<fma_AxCtB>` (`kernels/gdn_scan.cu`). Asynchronous.
///
/// Refused with [`KernelError::GdnScanShape`], before any pointer is touched, unless `head_k` is
/// 128 (`step::GDN_HEAD_K`), `head_v` a non-zero multiple of 32 and at most 256, `nk` and `nv`
/// non-zero with `nv` a multiple of `nk`, `rows` non-zero, `rows · conv_dim` at most `i32::MAX`
/// (`conv_dim = 2·nk·128 + nv·head_v`), `scale` finite and above 0, and `slot_elems` exactly
/// `nv · head_v · 128`.
///
/// # Safety
///
/// `conv_out` must be a device pointer to `rows·conv_dim` f32, `g` and `beta` to `rows·nv` f32,
/// `o` to `rows·nv·head_v` f32, `sequence` to 4 u32 with `1 ≤ q_len ≤ rows` and `slot` a slot of
/// `ssm_state`, which must hold `(slot + 1) · slot_elems` f32. All 16-byte aligned (`conv_out`'s
/// q, k and v rows are staged by bulk copy) and valid for the whole kernel; `o` and `ssm_state` overlap nothing else. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn gdn_scan(
    (conv_out, g, beta): (*const c_void, *const c_void, *const c_void),
    (ssm_state, slot_elems): (*mut c_void, u64),
    sequence: *const c_void,
    o: *mut c_void,
    (rows, nk, nv): (u32, u32, u32),
    (head_k, head_v, scale): (u32, u32, f32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_scan(
            conv_out, g, beta, ssm_state, slot_elems, sequence, o, rows, nk, nv, head_k, head_v,
            scale, stream,
        )
    };
    check(
        code,
        KernelError::GdnScanShape {
            rows,
            nk,
            nv,
            head_k,
            head_v,
        },
    )
}

/// The operands of [`gdn_decode`]: the token's `qkv`, `a`, `b` and `z`, the bundle's conv taps,
/// `a_log`, `dt_bias` and norm gain, and the output.
pub struct GdnDecodeOperands {
    pub qkv: *const c_void,
    pub a: *const c_void,
    pub b: *const c_void,
    pub z: *const c_void,
    pub w: *const c_void,
    pub a_log: *const c_void,
    pub dt_bias: *const c_void,
    pub norm: *const c_void,
    pub out: *mut c_void,
}

/// The one-launch GDN decode (#1049, `kernels/gdn_decode.cu`): [`gdn_conv1d`], [`gdn_gating`],
/// [`gdn_scan`] and [`gdn_rms_norm_gated`] at one row, in one launch, their output, ring and state
/// bits. `dims` is `(nk, nv, head_k, head_v, kernel)`. Asynchronous.
///
/// Refused with [`KernelError::GdnDecodeShape`], before any pointer is touched, unless the four
/// launchers take the geometry at one row, `head_v` is 128 or 256, `nv / nk` at most
/// [`crate::step::GDN_DECODE_MAX_HEADS`] dividing 16, and `scale` and `eps` finite and positive.
///
/// # Safety
///
/// As the four launchers' at `rows = 1`, for the operands, `sequence` (`q_len` 1) and the two
/// state buffers (`slot_elems` apart per slot).
pub unsafe fn gdn_decode(
    o: &GdnDecodeOperands,
    (conv_state, conv_slot_elems): (*mut c_void, u64),
    (ssm_state, ssm_slot_elems): (*mut c_void, u64),
    sequence: *const c_void,
    (nk, nv, head_k, head_v, kernel): (u32, u32, u32, u32, u32),
    (scale, eps): (f32, f32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // geometry before it touches a pointer and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_gdn_decode(
            o.qkv,
            o.a,
            o.b,
            o.z,
            o.w,
            o.a_log,
            o.dt_bias,
            o.norm,
            conv_state,
            conv_slot_elems,
            ssm_state,
            ssm_slot_elems,
            sequence,
            o.out,
            nk,
            nv,
            head_k,
            head_v,
            kernel,
            scale,
            eps,
            stream,
        )
    };
    check(
        code,
        KernelError::GdnDecodeShape {
            nk,
            nv,
            head_v,
            kernel,
        },
    )
}

/// Test support for `tests/gdn.rs` only (R8): `out = T · u + S` by the GDN scan's own rank-1
/// update, `warp::row_map<base_ops::fma_AxCtB>(S, T, S, u)` on its `rt_fl<16, 128>` state tile
/// (`kernels/gdn_scan.cu`), `u` one value per row. No `CudaStep` uses it. Never refused: a failed
/// launch returns its `cudaError_t`. Asynchronous.
///
/// # Safety
///
/// `s`, `t` and `out` must be 16-byte-aligned device pointers to 16 · 128 f32 and `u` to 16 f32,
/// valid for the whole kernel, `out` overlapping no input. `stream` must be a live `cudaStream_t`
/// of the current device, or null for the default stream.
#[doc(hidden)]
pub unsafe fn gdn_rank_one_probe(
    (s, t, u): (*const c_void, *const c_void, *const c_void),
    out: *mut c_void,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher catches
    // every C++ exception.
    let code = unsafe { scratchy_cuda_gdn_rank_one_probe(s, t, u, out, stream) };
    check(
        code,
        KernelError::RowShape {
            rows: 16,
            cols: 128,
        },
    )
}

/// Launch the GDN gated RMSNorm: per real row and value head, `o / sqrt(mean(o²) + eps) · w ·
/// SiLU(z)` over the head's `head_v` lanes, rounded once to bf16 into `out`; every padding row of
/// `out` written `+0.0` without reading `o` or `z`. One warp per row, the row a ThunderKittens
/// warp vector, its sum of squares by `warp::mul` then `warp::sum` (`kernels/gdn_rms_norm_gated.cu`).
/// Asynchronous.
///
/// Refused with [`KernelError::GdnNormShape`], before any pointer is touched, unless `rows` and
/// `nv` are non-zero, `head_v` a non-zero multiple of 32 and at most 256, `rows · nv · head_v` at
/// most `i32::MAX`, and `eps` finite and above 0.
///
/// # Safety
///
/// `o` must be a device pointer to `rows·nv·head_v` f32, `z` and `out` to that many bf16, `w` to
/// `head_v` f32, `sequence` to 4 u32 with `q_len ≤ rows`; all 4-byte aligned and valid for the
/// whole kernel, `out` overlapping no input. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gdn_rms_norm_gated(
    (o, z, w): (*const c_void, *const c_void, *const c_void),
    sequence: *const c_void,
    out: *mut c_void,
    (rows, nv, head_v): (u32, u32, u32),
    eps: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_gdn_rms_norm_gated(o, z, w, sequence, out, rows, nv, head_v, eps, stream)
    };
    check(code, KernelError::GdnNormShape { rows, nv, head_v })
}

// THE MOE KERNELS (#857): the qwen3-5-moe block's steps. Each is a ThunderKittens kernel; a shape
// it does not take is refused with [`KernelError::MoeShape`] naming it.

/// Launch the MoE routing (`kernels/moe_route_bf16.cu`): per token row of `logits` (`[m, 256]`
/// bf16), `softmax` in f32, its `top_k` largest picked (equal probabilities by the higher expert
/// index, a NaN lowest; always `top_k` distinct indices in `0..256`), stored ascending (slot
/// `top_k − 1` the best) as i32 into `topk_idx` and their probabilities, divided by their sum when
/// `renorm`, as f32 into `scores` (`[m, top_k]` each). Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_route_bf16", dims: [m, experts, top_k,
/// 0] }`, before any pointer is touched, unless `m` is non-zero and at most `i32::MAX`, `experts`
/// is 256 (`step::MOE_ROUTE_EXPERTS`) and `top_k` in `1..=32` (`step::MOE_ROUTE_MAX_TOP_K`).
///
/// # Safety
///
/// `logits` must be a device pointer to `m · 256` bf16, `topk_idx` to `m · top_k` i32 and `scores`
/// to `m · top_k` f32, 16-byte aligned and valid for the whole kernel, the outputs overlapping
/// nothing. `stream` must be a live `cudaStream_t` of the current device, or null for the default
/// stream.
pub unsafe fn moe_route_bf16(
    logits: *const c_void,
    (topk_idx, scores): (*mut c_void, *mut c_void),
    (m, experts, top_k): (u32, u32, u32),
    renorm: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s; the Qwen program reads no per-expert scale.
    let code = unsafe {
        scratchy_cuda_moe_route_bf16(
            logits,
            std::ptr::null(),
            topk_idx,
            scores,
            m,
            experts,
            top_k,
            u32::from(renorm),
            ROUTE_QWEN,
            0.0,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_route_bf16",
            dims: [m, experts, top_k, ROUTE_QWEN],
        },
    )
}

/// `kernels/moe_route_bf16.cu`'s `program`: the Qwen routing.
const ROUTE_QWEN: u32 = 0;
/// `kernels/moe_route_bf16.cu`'s `program`: the Gemma routing.
const ROUTE_GEMMA: u32 = 1;

/// Launch Gemma's MoE routing (`kernels/moe_route_bf16.cu`'s Gemma program): per token row of
/// `logits` (`[m, experts]` bf16), the `top_k` largest RAW logits picked (equal logits by the
/// higher expert index, a NaN lowest; always `top_k` distinct indices in `0..experts`), each
/// scaled by `scale` (`hidden^-0.5`), a softmax over the picks in f32, then each times its
/// expert's `expert_scale` entry; stored ascending (slot `top_k − 1` the best) as i32 into
/// `topk_idx` and f32 into `scores` (`[m, top_k]` each). Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_route_bf16", dims: [m, experts, top_k,
/// 1] }`, before any pointer is touched, unless `m` is non-zero and at most `i32::MAX`, `experts`
/// one of `op_abi::MOE_EXPERT_COUNTS` (128, 256), `top_k` in `1..=32`
/// (`step::MOE_ROUTE_MAX_TOP_K`), and `expert_scale` non-null.
///
/// # Safety
///
/// `logits` must be a device pointer to `m · experts` bf16, `expert_scale` to `experts` bf16,
/// `topk_idx` to `m · top_k` i32 and `scores` to `m · top_k` f32, 16-byte aligned and valid for
/// the whole kernel, the outputs overlapping nothing. `stream` must be a live `cudaStream_t` of the
/// current device, or null for the default stream.
pub unsafe fn gemma_route_bf16(
    (logits, expert_scale): (*const c_void, *const c_void),
    (topk_idx, scores): (*mut c_void, *mut c_void),
    (m, experts, top_k): (u32, u32, u32),
    scale: f32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_moe_route_bf16(
            logits,
            expert_scale,
            topk_idx,
            scores,
            m,
            experts,
            top_k,
            0,
            ROUTE_GEMMA,
            scale,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_route_bf16",
            dims: [m, experts, top_k, ROUTE_GEMMA],
        },
    )
}

/// Launch the MoE combine (`kernels/moe_combine_bf16.cu`): `out[t, :] = Σ_j scores[t, j] ·
/// down[row(t, j), :]` over the `top_k` slots in order, in f32, rounded once to bf16, every token
/// row written; `row(t, j)` is `pos[t · top_k + j]` when `pos` is given (sorted pairs), else
/// `t · top_k + j` (gathered). Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_combine_bf16", dims: [m, top_k, hidden,
/// down_rows] }`, before any pointer is touched, unless `m` is non-zero, `top_k` in `1..=32`,
/// `hidden` a non-zero multiple of 256 (`step::MOE_COMBINE_BF16_CHUNK`), `down_rows` at least
/// `m · top_k`, and `m · top_k` and `down_rows · hidden` at most `i32::MAX`.
///
/// # Safety
///
/// `down` must be a device pointer to `down_rows · hidden` bf16, `pos` (when given) to
/// `m · top_k` u32 each below `down_rows`, `scores` to `m · top_k` f32 and `out` to `m · hidden`
/// bf16, 16-byte aligned and valid for the whole kernel, `out` overlapping no input. `stream`
/// must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn moe_combine_bf16(
    (down, pos, scores): (*const c_void, Option<*const c_void>, *const c_void),
    out: *mut c_void,
    (m, top_k, hidden, down_rows): (u32, u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let (pos, sorted) = match pos {
        Some(p) => (p, 1),
        None => (std::ptr::null(), 0),
    };
    // SAFETY: as `rms_norm_bf16`'s; a null `pos` is never read (`sorted` 0).
    let code = unsafe {
        scratchy_cuda_moe_combine_bf16(
            down, pos, scores, out, m, top_k, hidden, down_rows, sorted, stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_combine_bf16",
            dims: [m, top_k, hidden, down_rows],
        },
    )
}

/// The sort's index regions, each a device pointer into the MoE scratch ([`crate::step::SortLayout`]):
/// `chunk_hist`, `counts`, `offsets`, `tile_expert` and `pos`.
#[derive(Clone, Copy, Debug)]
pub struct SortRegions {
    pub chunk_hist: *mut c_void,
    pub counts: *mut c_void,
    pub offsets: *mut c_void,
    pub tile_expert: *mut c_void,
    pub pos: *mut c_void,
}

/// Launch the MoE sort (`kernels/moe_sort.cu`, three stream-ordered launches): the `m · top_k`
/// pairs, pair `p` naming expert `topk_idx[p]`, sorted by expert stably, each expert's run padded
/// to whole 128-row tiles. Writes `rows[pos[p], :] = x[p / top_k, :]` (bit-exact) and `+0.0` on
/// every padding row of every live tile, and the index regions: `counts`, `offsets` (the padded
/// exclusive scan, `offsets[experts]` the total), `tile_expert` for each live tile, `pos`, and
/// `chunk_hist` as scratch. No row outside a live tile is written. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "moe_sort", dims: [m, top_k, hidden,
/// max_tiles] }`, before any pointer is touched, unless `m` is non-zero, `experts` one of
/// `op_abi::MOE_EXPERT_COUNTS` (128, 256), `top_k` in `1..=32`, `hidden` a non-zero multiple of 8, `m ·
/// top_k` and `m · hidden` at most `i32::MAX`, and `max_tiles` at least `tape::moe_tiles`' bound
/// with `max_tiles · 128 · hidden` at most `i32::MAX`. A failed launch returns the first failing
/// launch's own `cudaError_t`, the ones after it not launched. Its one-time setup in [`PREPARE`]
/// only loads its three kernels.
///
/// # Safety
///
/// `x` must be a device pointer to `m · hidden` bf16, `topk_idx` to `m · top_k` i32 each in
/// `0..experts`, `rows` to `max_tiles · 128 · hidden` bf16, and `regions` to the sort's regions of
/// [`crate::step::SortLayout`] for this geometry, all 16-byte aligned and valid for the whole
/// kernel, no output overlapping another or an input. `stream` must be a live `cudaStream_t` of
/// the current device, or null for the default stream.
pub unsafe fn moe_sort(
    (x, topk_idx): (*const c_void, *const c_void),
    regions: SortRegions,
    rows: *mut c_void,
    (m, experts, top_k, hidden, max_tiles): (u32, u32, u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `rms_norm_bf16`'s.
    let code = unsafe {
        scratchy_cuda_moe_sort(
            x,
            topk_idx,
            regions.chunk_hist,
            regions.counts,
            regions.offsets,
            regions.tile_expert,
            regions.pos,
            rows,
            m,
            experts,
            top_k,
            hidden,
            max_tiles,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "moe_sort",
            dims: [m, top_k, hidden, max_tiles],
        },
    )
}

/// Launch the grouped expert GEMM (`kernels/expert_gemm_bf16.cu`) on 256-column blocks:
/// [`expert_gemm_tiled_bf16`] at `n_tile` 256, the qwen3-5-moe tapes' one tile.
///
/// # Safety
///
/// As [`expert_gemm_tiled_bf16`]'s.
pub unsafe fn expert_gemm_bf16(
    (a, b): (*const c_void, *const c_void),
    (offsets, tile_expert): (*const c_void, *const c_void),
    c: *mut c_void,
    (max_tiles, n, k, experts, expert_rows): (u32, u32, u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds `expert_gemm_tiled_bf16`'s contract.
    unsafe {
        expert_gemm_tiled_bf16(
            (a, b),
            (offsets, tile_expert),
            c,
            (max_tiles, n, k, experts, expert_rows),
            EXPERT_GEMM_WIDE_TILE,
            stream,
        )
    }
}

/// [`crate::step::EXPERT_GEMM_N_TILES`]' wide tile, every Qwen tape's.
const EXPERT_GEMM_WIDE_TILE: u32 = crate::step::EXPERT_GEMM_N_TILES[1];

/// Launch the grouped expert GEMM (`kernels/expert_gemm_bf16.cu`): `c[128t .. 128t + 128, :] =
/// a[128t .. 128t + 128, :] · B_eᵀ` for every live tile `t < offsets[experts] / 128` of the
/// sorted rows, `e = tile_expert[t]`, `B_e` the first `n` rows of expert `e`'s `[expert_rows, k]`
/// block at `b`, in column blocks of `n_tile`. A tile past the live ones is never read or written.
/// Every live tile's rows are bit-identical to [`gemm_bf16_abt`] (`n_tile` 256) or
/// [`gemm_col64_bf16_abt`] (`n_tile` 64) on the same rows and expert slice. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "expert_gemm_bf16", dims: [max_tiles, n, k,
/// expert_rows] }`, before any pointer is touched, unless `max_tiles` is non-zero, `(experts,
/// n_tile)` one of (256, 256), (128, 256) and (128, 64) (`op_abi::MOE_EXPERT_COUNTS`,
/// `step::EXPERT_GEMM_N_TILES`), `n` a non-zero multiple of `n_tile` and `k` of 64, `expert_rows`
/// at least `n`, and `max_tiles · 128 · max(n, k)` and `experts · expert_rows · k` at most
/// `i32::MAX`. A failed launch returns the launch's own `cudaError_t`.
///
/// # Safety
///
/// `a` must be a device pointer to `max_tiles · 128 · k` bf16, `b` to `experts · expert_rows · k`
/// bf16, `offsets` and `tile_expert` to the sort's regions (`experts + 1` and `max_tiles` u32,
/// the live tiles each naming an expert below `experts`), and `c` to `max_tiles · 128 · n` bf16,
/// all 16-byte aligned and valid for the whole kernel, `c` overlapping no input. The kernel's
/// setup in [`PREPARE`] must have run on the current device. `stream` must be a live
/// `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn expert_gemm_tiled_bf16(
    (a, b): (*const c_void, *const c_void),
    (offsets, tile_expert): (*const c_void, *const c_void),
    c: *mut c_void,
    (max_tiles, n, k, experts, expert_rows): (u32, u32, u32, u32, u32),
    n_tile: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gemm_bf16_abt`'s.
    let code = unsafe {
        scratchy_cuda_expert_gemm_bf16(
            a,
            b,
            offsets,
            tile_expert,
            c,
            max_tiles,
            n,
            k,
            experts,
            expert_rows,
            n_tile,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemm_bf16",
            dims: [max_tiles, n, k, expert_rows],
        },
    )
}

/// Launch the gathered expert GEMV (`kernels/expert_gemv_bf16.cu`): `c[p, :] = a[row(p), :] ·
/// B_eᵀ` for each of the `m · top_k` pairs, `e = topk_idx[p]`, `B_e` the first `n` rows of expert
/// `e`'s `[expert_rows, k]` block at `b`; `row(p)` is `p` when `pair_rows` (the down), else the
/// pair's token row `p / top_k` (the gate and up). Each pair's row is bit-identical to
/// [`gemv_bf16_abt`] on the same row and expert slice. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "expert_gemv_bf16", dims: [m, top_k, n, k]
/// }`, before any pointer is touched, unless `m` is non-zero, `top_k` in `1..=32`, `m · top_k` at
/// most 65,535, `n` and `k` non-zero multiples of 64, `experts` one of
/// `op_abi::MOE_EXPERT_COUNTS` (128, 256), `expert_rows` at least `n`,
/// and the operands' element counts at most `i32::MAX`. A failed launch returns the launch's own
/// `cudaError_t`.
///
/// # Safety
///
/// `a` must be a device pointer to `a_rows · k` bf16 (`a_rows` `m · top_k` when `pair_rows`, else
/// `m`), `b` to `experts · expert_rows · k` bf16, `topk_idx` to `m · top_k` i32 each in
/// `0..experts`
/// and `c` to `m · top_k · n` bf16, all 16-byte aligned and valid for the whole kernel, `c`
/// overlapping no input. The kernel's setup in [`PREPARE`] must have run on the current device.
/// `stream` must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn expert_gemv_bf16(
    (a, b, topk_idx): (*const c_void, *const c_void, *const c_void),
    c: *mut c_void,
    (m, top_k, n, k): (u32, u32, u32, u32),
    (experts, expert_rows): (u32, u32),
    pair_rows: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gemv_bf16_abt`'s.
    let code = unsafe {
        scratchy_cuda_expert_gemv_bf16(
            a,
            b,
            topk_idx,
            c,
            m,
            top_k,
            n,
            k,
            experts,
            expert_rows,
            u32::from(pair_rows),
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemv_bf16",
            dims: [m, top_k, n, k],
        },
    )
}

/// [`expert_gemv_bf16`] at `splits` K runs (`step::gemv_splits` of `(n, k)` and the `m · top_k`
/// pairs), its partials in `partials`, added in ascending split order by a second launch: each
/// pair's row is [`gemv_split_bf16_abt`]'s bits at the same split on the same row and expert
/// slice. Asynchronous.
///
/// Refused with [`KernelError::MoeShape`] `{ kernel: "expert_gemv_bf16", .. }` for a shape
/// [`expert_gemv_bf16`] refuses, or a split [`gemv_split_bf16_abt`] refuses, before any pointer
/// is touched.
///
/// # Safety
///
/// As [`expert_gemv_bf16`]'s; and at `splits > 1`, `partials` must be a 16-byte-aligned device
/// pointer to `step::gemv_scratch_bytes(n, m · top_k, splits)` bytes, valid for both launches and
/// overlapping no operand.
pub unsafe fn expert_gemv_split_bf16(
    (a, b, topk_idx): (*const c_void, *const c_void, *const c_void),
    (c, partials): (*mut c_void, *mut c_void),
    (m, top_k, n, k): (u32, u32, u32, u32),
    (experts, expert_rows): (u32, u32),
    (pair_rows, splits): (bool, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `expert_gemv_bf16`'s, and the scratch contract above.
    let code = unsafe {
        scratchy_cuda_expert_gemv_split_bf16(
            a,
            b,
            topk_idx,
            c,
            partials,
            m,
            top_k,
            n,
            k,
            experts,
            expert_rows,
            u32::from(pair_rows),
            splits,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemv_bf16",
            dims: [m, top_k, n, k],
        },
    )
}

// FP8 S3: THE PER-TOKEN QUANTIZATION AND THE PREFILL FP8 GEMMS. An fp8 operand is a pair, its e4m3
// bytes and its f32 scales: the activation's `[m, k]` rows and `[m]` per-token scales
// (`quant_rows_fp8`'s output, in the quant scratch), and the weight's `[n, k]` bytes and `[n]`
// per-channel scales (`step::fp8_weight`'s one buffer). Every output is bf16, rounded once from
// `(A · Bᵀ) · w_scale[n] · a_scale[m]` in f32.

/// One fp8 operand: its e4m3 bytes and its f32 scales.
pub type Fp8Operand = (*const c_void, *const c_void);

/// Launch the per-token dynamic e4m3 quantization (`kernels/quant_rows_fp8.cu`, FP8 S3): for every
/// row `r` of `x [rows, cols]` bf16, `scale[r] = max(amax_r / 448, 1 / (448·512))` and `q[r, c] =
/// e4m3_satfinite(clamp(x[r, c] / scale[r], ±448))`, vLLM v0.31.0's per-token formula with its
/// division. Asynchronous.
///
/// Refused with [`KernelError::RowShape`], before any pointer is touched, unless `rows` is a
/// non-zero multiple of 16, `cols` of 64 (`step::GEMM_FP8_K_MULTIPLE`) and `rows · cols` at most
/// `i32::MAX`.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `x` must be a 16-byte-aligned device
/// pointer to `rows·cols` bf16, `q` to `rows·cols` bytes and `scale` to `rows` f32 (4-byte
/// aligned), valid for the whole kernel, the outputs overlapping neither `x` nor each other.
/// `stream` must be a live `cudaStream_t` of the current device, or null for the default stream.
pub unsafe fn quant_rows_fp8(
    x: *const c_void,
    (q, scale): (*mut c_void, *mut c_void),
    (rows, cols): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_quant_rows_fp8(x, q, scale, rows, cols, stream) };
    check(code, KernelError::RowShape { rows, cols })
}

/// Test support for `tests/gemv_fp8_tc.rs` only (FP8 S8's Q1, no tape launches it yet):
/// [`quant_rows_fp8`] of one row, `x [1, cols]` bf16 into `q [1, cols]` e4m3 and `scale [1]` f32,
/// the bits of [`quant_rows_fp8`]'s row 0 for the same row, one CTA per 256 columns. Refused with
/// [`KernelError::RowShape`] (`rows` 1) unless `cols` is a non-zero multiple of 64 at most 17,408.
/// Asynchronous.
///
/// # Safety
///
/// As [`quant_rows_fp8`]'s at one row.
#[doc(hidden)]
pub unsafe fn quant_row_fp8_one(
    x: *const c_void,
    (q, scale): (*mut c_void, *mut c_void),
    cols: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above; the launcher checks the shape before it touches a pointer.
    let code = unsafe { scratchy_cuda_quant_row_fp8_one(x, q, scale, cols, stream) };
    check(code, KernelError::RowShape { rows: 1, cols })
}

/// Test support for `tests/gemv_fp8_tc.rs` only: [`quant_row_fp8_one`] at `variant` (0 its first
/// form, which divides the tile's 15 zero-fill rows too; 1 the tile dividing row 0 only; 2
/// [`quant_row_fp8_one`]'s own, one element a thread), and, when
/// `trace` is not null, each CTA's phase cycles into `trace` (`[⌈cols / 256⌉, 8]` f32: entry,
/// chunks arrived, amax shared, tile arrived, tile divided, stored). Refused as
/// [`quant_row_fp8_one`], or for `variant` above 2. Asynchronous.
///
/// # Safety
///
/// As [`quant_row_fp8_one`]'s; `trace`, when not null, `⌈cols / 256⌉ · 8` f32 of device memory.
#[doc(hidden)]
pub unsafe fn quant_row_fp8_one_variant(
    x: *const c_void,
    (q, scale): (*mut c_void, *mut c_void),
    cols: u32,
    (variant, trace): (u32, *mut c_void),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above; the launcher checks the shape before it touches a pointer.
    let code = unsafe {
        scratchy_cuda_quant_row_fp8_one_variant(x, q, scale, cols, variant, trace, stream)
    };
    check(code, KernelError::RowShape { rows: 1, cols })
}

/// Launch the fp8 GEMM (`kernels/gemm_fp8.cu`): `c [m, n] = bf16((a · bᵀ) · b_scale[n] ·
/// a_scale[m])`, accumulated in f32. At `col64` false on [`gemm_bf16_abt`]'s tile grid (`m`, `n`,
/// `k` non-zero multiples of 128, 256 and 64), at `col64` true on [`gemm_col64_bf16_abt`]'s (128,
/// 64, 64); none above `i32::MAX`. On an `n` both take, each element is the same bits.
/// Asynchronous; refused with [`KernelError::Shape`] before any pointer is touched.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a` (`m·k` bytes), `b` (`n·k`) and
/// `c` (`m·n` bf16) must be 16-byte-aligned device pointers and the scales (`m` and `n` f32)
/// 8-byte-aligned, valid for the whole kernel, `c` overlapping no input. `stream` as
/// [`gemm_bf16_abt`]'s.
pub unsafe fn gemm_fp8_abt(
    a: Fp8Operand,
    b: Fp8Operand,
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    col64: bool,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above; the launcher checks the shape and catches every C++ exception.
    let code = unsafe {
        scratchy_cuda_gemm_fp8_abt(a.0, a.1, b.0, b.1, c, m, n, k, u32::from(col64), stream)
    };
    check(code, KernelError::Shape { m, n, k })
}

/// One grouped fp8 member: its weight (bytes and scales), its output and its `n`.
pub type Fp8GroupOperands<'g> = &'g [(Fp8Operand, *mut c_void, u32)];

/// The member arrays of `members` as the grouped launchers read them, and their refusal.
fn fp8_group_parts(
    kernel: &'static str,
    members: Fp8GroupOperands<'_>,
    (m, k): (u32, u32),
) -> (
    [[*const c_void; 3]; 2],
    [*mut c_void; 3],
    [u32; 3],
    KernelError,
) {
    let mut w = [[std::ptr::null(); 3]; 2];
    let (mut c, mut n) = ([std::ptr::null_mut(); 3], [0u32; 3]);
    for (i, &((b, s), ci, ni)) in members.iter().take(3).enumerate() {
        (w[0][i], w[1][i], c[i], n[i]) = (b, s, ci, ni);
    }
    (w, c, n, KernelError::GroupShape { kernel, m, k, n })
}

/// [`gemm_fp8_abt`] at `col64` false of every member of `members` off one `a` in one launch: each
/// member's output is the bits of its own [`gemm_fp8_abt`]. Asynchronous; refused with
/// [`KernelError::GroupShape`] before any pointer is touched unless there are 2 or 3 members, each
/// under that GEMM's rule with the shared `m` and `k`.
///
/// # Safety
///
/// As [`gemm_fp8_abt`]'s, for `a` and every member; no member's output overlaps another's.
pub unsafe fn gemm_group_fp8_abt(
    a: Fp8Operand,
    members: Fp8GroupOperands<'_>,
    (m, k): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let ([b, s], c, n, refused) = fp8_group_parts("gemm_group_fp8_abt", members, (m, k));
    let count = members.len() as u32;
    // SAFETY: as above; the arrays are three long and the launcher checks the count and every
    // member's shape before it reads a pointer.
    let code = unsafe {
        scratchy_cuda_gemm_group_fp8_abt(
            a.0,
            a.1,
            b.as_ptr(),
            s.as_ptr(),
            c.as_ptr(),
            n.as_ptr(),
            m,
            k,
            count,
            stream,
        )
    };
    check(code, refused)
}

/// The fp8 gate/up GEMM: `c [m, n] = bf16(act(g, u))`, `g` and `u` each [`gemm_fp8_abt`]'s value
/// before its rounding (scaled by its own weight's scales and by `a`'s), `act` 0 for SwiGLU's
/// `silu_mul` and 1 for GeGLU's `gelu_tanh_mul`, in f32, rounded once. Under [`gemm_fp8_abt`]'s
/// rule at the same `col64`. Asynchronous; refused with [`KernelError::GateUpShape`] before any
/// pointer is touched.
///
/// # Safety
///
/// As [`gemm_fp8_abt`]'s for `a`, `gate`, `up` (each `[n, k]`) and `c`.
pub unsafe fn gate_up_gemm_fp8(
    (a, gate, up): (Fp8Operand, Fp8Operand, Fp8Operand),
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    (act, col64): (u32, bool),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above; the launcher checks the shape and codes before it touches a pointer.
    let code = unsafe {
        scratchy_cuda_gate_up_gemm_fp8(
            a.0,
            a.1,
            gate.0,
            gate.1,
            up.0,
            up.1,
            c,
            m,
            n,
            k,
            act,
            u32::from(col64),
            stream,
        )
    };
    check(
        code,
        KernelError::GateUpShape {
            kernel: "gate_up_gemm_fp8",
            dims: [m, n, k, 1],
        },
    )
}

/// The grouped expert GEMM over an fp8 bank (`kernels/gemm_fp8.cu`): [`expert_gemm_tiled_bf16`]'s
/// task walk, index regions and shape rule, with `a` the pair rows' e4m3 bytes and per-row scales
/// and `b` expert 0's `[expert_rows, k]` e4m3 block and its `[expert_rows]` f32 scale row (of
/// `[experts, expert_rows]`). Every live tile's rows are the bits [`gemm_fp8_abt`] gives on the
/// same rows and expert slice. Asynchronous; refused with [`KernelError::MoeShape`] `{ kernel:
/// "expert_gemm_fp8", .. }` before any pointer is touched.
///
/// # Safety
///
/// As [`expert_gemm_tiled_bf16`]'s, with `a` `max_tiles · 128 · k` bytes and its scales as many
/// f32, `b` `experts · expert_rows · k` bytes and its scales `experts · expert_rows` f32.
pub unsafe fn expert_gemm_fp8(
    (a, b): (Fp8Operand, Fp8Operand),
    (offsets, tile_expert): (*const c_void, *const c_void),
    c: *mut c_void,
    (max_tiles, n, k, experts, expert_rows): (u32, u32, u32, u32, u32),
    n_tile: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above.
    let code = unsafe {
        scratchy_cuda_expert_gemm_fp8(
            a.0,
            a.1,
            b.0,
            b.1,
            offsets,
            tile_expert,
            c,
            max_tiles,
            n,
            k,
            experts,
            expert_rows,
            n_tile,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemm_fp8",
            dims: [max_tiles, n, k, expert_rows],
        },
    )
}

/// The narrow fp8 GEMM (`kernels/gemm_narrow_fp8.cu`), for `n` of at most 64: [`gemm_fp8_abt`]'s
/// operands and value, `m` a non-zero multiple of 16, `n` of 8 at most 64 and `k` of 64, `m` and
/// `k` at most `i32::MAX`. Asynchronous; refused with [`KernelError::NarrowShape`] before any
/// pointer is touched.
///
/// # Safety
///
/// As [`gemm_fp8_abt`]'s, with every pointer 16-byte aligned.
pub unsafe fn gemm_narrow_fp8_abt(
    a: Fp8Operand,
    b: Fp8Operand,
    c: *mut c_void,
    (m, n, k): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above.
    let code = unsafe { scratchy_cuda_gemm_narrow_fp8_abt(a.0, a.1, b.0, b.1, c, m, n, k, stream) };
    check(code, KernelError::NarrowShape { m, n, k })
}

/// [`gemm_narrow_fp8_abt`] of every member of `members` off one `a` in one launch, each the bits
/// of its own [`gemm_narrow_fp8_abt`]. Asynchronous; refused with [`KernelError::GroupShape`]
/// before any pointer is touched unless there are 2 or 3 members, each under the narrow rule.
///
/// # Safety
///
/// As [`gemm_narrow_fp8_abt`]'s, for `a` and every member; no member's output overlaps another's.
pub unsafe fn gemm_narrow_group_fp8_abt(
    a: Fp8Operand,
    members: Fp8GroupOperands<'_>,
    (m, k): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let ([b, s], c, n, refused) = fp8_group_parts("gemm_narrow_group_fp8_abt", members, (m, k));
    let count = members.len() as u32;
    // SAFETY: as `gemm_group_fp8_abt`'s.
    let code = unsafe {
        scratchy_cuda_gemm_narrow_group_fp8_abt(
            a.0,
            a.1,
            b.as_ptr(),
            s.as_ptr(),
            c.as_ptr(),
            n.as_ptr(),
            m,
            k,
            count,
            stream,
        )
    };
    check(code, refused)
}

// FP8 S4: THE DECODE FP8 GEMVS (`kernels/gemv_fp8.cu`). The activation row is bf16, quantized in
// each kernel's prologue exactly as [`quant_rows_fp8`] quantizes it; the weight is an
// [`Fp8Operand`], its e4m3 bytes and per-channel f32 scales. Every output is bf16, rounded once
// from `(q · Bᵀ) · w_scale[n] · a_scale` in f32 (split partials are scaled before the reduce adds
// them). The split rule is the bf16 GEMV's ([`gemv_split_bf16_abt`]), with `k` a non-zero
// multiple of 64 (`step::GEMM_FP8_K_MULTIPLE`) at most 17,408.

/// Launch the fp8 GEMV `c [n] = bf16((q · bᵀ) · b_scale[n] · a_scale)`, `q` and `a_scale` the
/// per-token quantization of `a [1, k]` bf16, at `splits` K runs (`step::gemv_splits`): at
/// `splits > 1` each split's scaled f32 partials go to `partials` and a second launch adds them in
/// ascending split order and rounds once. Asynchronous.
///
/// Refused with [`KernelError::GemvSplit`], before any pointer is touched, unless `n` is non-zero
/// and at most `i32::MAX`, `k` a non-zero multiple of 64 at most 17,408, `splits` in `1..=16`
/// cutting `⌈k / 256⌉` chunks into non-empty runs (`step::gemv_split_chunks`), `partials` not null
/// above one split, and `splits · n_pad` at most `i32::MAX`.
///
/// # Safety
///
/// [`prepare_all`] must have succeeded on the current device. `a` (`k` bf16), `b` (`n·k` bytes)
/// and `c` (`n` bf16) must be 16-byte-aligned device pointers and `b`'s scales (`n` f32) 4-byte
/// aligned, valid for the whole kernel, `c` overlapping no input. At `splits > 1`, `partials` must
/// be a 16-byte-aligned device pointer to `step::gemv_scratch_bytes(n, 1, splits)` bytes,
/// overlapping no operand. `stream` must be a live `cudaStream_t` of the current device, or null
/// for the default stream.
pub unsafe fn gemv_fp8_abt(
    (a, b, c): (*const c_void, Fp8Operand, *mut c_void),
    partials: *mut c_void,
    (n, k, splits): (u32, u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape and split before it touches a pointer and catches every C++ exception.
    let code =
        unsafe { scratchy_cuda_gemv_fp8_abt(a, b.0, b.1, c, partials, n, k, splits, stream) };
    check(code, KernelError::GemvSplit { n, k, splits })
}

/// [`gemv_fp8_abt`] of every member of `members` off one `a` in two launches at most: member `i`
/// at `splits[i]`, its partials at `partials[i]`, each the bits of its own [`gemv_fp8_abt`] at that
/// split. Refused with [`KernelError::GroupShape`] before any pointer is touched unless there are
/// 2 or 3 members, `splits` and `partials` have one entry per member, and each member meets
/// [`gemv_fp8_abt`]'s rule at its split. Asynchronous.
///
/// # Safety
///
/// As [`gemv_fp8_abt`]'s, for `a` and every member's weight, `c` and partials; no member's `c` or
/// partials overlaps another's.
pub unsafe fn gemv_group_fp8_abt(
    a: *const c_void,
    members: Fp8GroupOperands<'_>,
    partials: &[*mut c_void],
    (k, splits): (u32, &[u32]),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let ([b, s], c, n, refused) = fp8_group_parts("gemv_group_fp8_abt", members, (1, k));
    if splits.len() != members.len() || partials.len() != members.len() {
        return Err(refused);
    }
    let (mut sp, mut p) = ([1u32; 3], [std::ptr::null_mut(); 3]);
    for (i, (&si, &pi)) in splits.iter().zip(partials).take(3).enumerate() {
        (sp[i], p[i]) = (si, pi);
    }
    let count = members.len() as u32;
    // SAFETY: as above; the arrays are three long and the launcher checks the count and every
    // member's shape before it reads a pointer.
    let code = unsafe {
        scratchy_cuda_gemv_group_fp8_abt(
            a,
            b.as_ptr(),
            s.as_ptr(),
            c.as_ptr(),
            p.as_ptr(),
            n.as_ptr(),
            k,
            sp.as_ptr(),
            count,
            stream,
        )
    };
    check(code, refused)
}

/// The `partials` bytes [`gemv_fp8_tc`] needs for members of `n` rows each over `k`, or `None`
/// when that launcher refuses the shape.
pub fn gemv_fp8_tc_partial_bytes(n: &[u32], k: u32) -> Option<u64> {
    let count = u32::try_from(n.len()).ok()?;
    // SAFETY: `n` is `count` u32s; the function only reads them, after checking `count`.
    let bytes = unsafe { scratchy_cuda_gemv_fp8_tc_partial_bytes(n.as_ptr(), count, k) };
    (bytes != 0).then_some(bytes)
}

/// The tensor-core decode GEMV (`kernels/gemv_fp8_tc.cu`, FP8 S8, not yet in a tape): member `i`
/// of `members` is `c [n] = bf16(((q · bᵀ) · b_scale) · a_scale)` off one quantized row `q [1, k]`
/// e4m3 and its scale `a_scale` (one f32, `quant_rows_fp8`'s), 1 to 3 members, each `n` a non-zero
/// multiple of 64 and `k` of 64. `partials` holds [`gemv_fp8_tc_partial_bytes`] bytes. Two launches,
/// asynchronous; refused with [`KernelError::GroupShape`] before any pointer is touched.
///
/// # Safety
///
/// Every pointer is 16-byte-aligned device memory of its size; no output overlaps an input or
/// another output; `stream` is a live stream or null.
pub unsafe fn gemv_fp8_tc(
    (q, a_scale): Fp8Operand,
    members: Fp8GroupOperands<'_>,
    partials: *mut c_void,
    k: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let ([b, s], c, n, refused) = fp8_group_parts("gemv_fp8_tc", members, (1, k));
    if members.is_empty() || members.len() > 3 {
        return Err(refused);
    }
    // SAFETY: as above; the arrays are three long and the launcher checks the count and every
    // member's shape before it reads a pointer.
    let code = unsafe {
        scratchy_cuda_gemv_fp8_tc(
            q,
            a_scale,
            b.as_ptr(),
            s.as_ptr(),
            c.as_ptr(),
            n.as_ptr(),
            members.len() as u32,
            k,
            partials,
            stream,
        )
    };
    check(code, refused)
}

/// B8's expert form (FP8 S8 T1): every member reads its first `n` rows of expert `topk_idx[p]`'s
/// block of the bank `[experts, expert_rows, k]` e4m3 (its scales `[experts, expert_rows]` f32),
/// `topk_idx [pairs]` i32 (`kernels/gemv_fp8.cu`'s `EXPERT` contract).
#[derive(Clone, Copy, Debug)]
pub struct B8Experts {
    pub topk_idx: *const c_void,
    pub experts: u32,
    pub expert_rows: u32,
}

/// What B8 stores (FP8 S8 T1): each member's rows rounded once to bf16, with `residual[i] [pairs,
/// n]` bf16 added as `scratchy::ops::with_residual` does when it is given (W4a's epilogue); or the
/// scaled f32 sums, unrounded (`F32`, the gate/up members, for [`act_quant_fp8`]).
#[derive(Clone, Copy, Debug)]
pub enum B8Store<'r> {
    Bf16 {
        residual: Option<&'r [*const c_void]>,
    },
    F32,
}

/// B8, the tensor-core decode GEMV in its production form (FP8 S8 T1, `kernels/gemv_fp8_tc.cu`;
/// no tape launches it yet): up to 3 members off the e4m3 activation `q [a_rows, k]` and its
/// per-row `a_scale [a_rows]` f32 (`quant_rows_fp8`'s bytes). Pair `p` of member `i` is
/// `((q[p / row_div] · b_e[j, :]) · b_scale_e[j]) · a_scale[p / row_div]` over its `n` rows, `e`
/// 0 dense or `topk_idx[p]` ([`B8Experts`]), stored as [`B8Store`] says into `c [pairs, n]`. Each
/// member, each pair, is its own launch's bits: one CTA per 64-row tile over the whole K. One
/// launch, asynchronous.
///
/// Refused with [`KernelError::GroupShape`] (`m` the pairs) before any pointer is touched: 1 to 3
/// members; `k` and every `n` non-zero multiples of 64; `pairs` 1 to 65,535 and `row_div` non-zero,
/// every pair's row below `a_rows`; an expert bank's `expert_rows` at least every `n`; a residual
/// for every member, none with `F32`.
///
/// # Safety
///
/// Every pointer 16-byte-aligned device memory holding its elements, no output overlapping an
/// input; [`prepare_all`] has succeeded on the current device; `stream` a live `cudaStream_t` of
/// it, or null.
pub unsafe fn gemv_fp8_b8(
    (q, a_scale, a_rows): (*const c_void, *const c_void, u32),
    members: Fp8GroupOperands<'_>,
    k: u32,
    (pairs, row_div, experts): (u32, u32, Option<B8Experts>),
    store: B8Store<'_>,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let ([b, s], c, n, _) = fp8_group_parts("gemv_fp8_b8", members, (pairs, k));
    let refused = KernelError::GroupShape {
        kernel: "gemv_fp8_b8",
        m: pairs,
        k,
        n,
    };
    let (residual, out_f32) = match store {
        B8Store::Bf16 { residual } => (residual, false),
        B8Store::F32 => (None, true),
    };
    if members.is_empty() || members.len() > 3 || residual.is_some_and(|r| r.len() != members.len())
    {
        return Err(refused);
    }
    let mut r = [std::ptr::null(); 3];
    for (slot, &p) in r.iter_mut().zip(residual.unwrap_or(&[])) {
        *slot = p;
    }
    let e = experts.unwrap_or(B8Experts {
        topk_idx: std::ptr::null(),
        experts: 1,
        expert_rows: 0,
    });
    // SAFETY: as above; the arrays are three long and the launcher checks the count, every shape
    // and every null before it reads a pointer.
    let code = unsafe {
        scratchy_cuda_gemv_fp8_b8(
            q,
            a_scale,
            a_rows,
            b.as_ptr(),
            s.as_ptr(),
            c.as_ptr(),
            if residual.is_some() {
                r.as_ptr()
            } else {
                std::ptr::null()
            },
            n.as_ptr(),
            members.len() as u32,
            k,
            e.experts,
            e.expert_rows,
            e.topk_idx,
            pairs,
            row_div,
            u32::from(out_f32),
            stream,
        )
    };
    check(code, refused)
}

/// Test support for `tests/gemv_fp8_b8.rs` only (TK-1's R1 on B8's 128-byte-swizzled fp8 boxes):
/// B8's activation box (`x [rows, 1, k]` e4m3, row `row`) and weight box (`w [n, k]` e4m3, tile
/// `tile`) at K step `j`, each first TMA-loaded with its all-NaN poison (`poison_x [32, 256]`,
/// `poison_w [64, 256]`), then TMA-stored to the exact-fit `dump_x`, `dump_w`. Refused with
/// [`KernelError::RowShape`] (`rows`, `k`) for a zero or misaligned size, a coordinate past its
/// array, or a null pointer. Asynchronous.
///
/// # Safety
///
/// As [`gemv_fp8_b8`]'s, for those buffers.
#[doc(hidden)]
pub unsafe fn gemv_fp8_b8_box_probe(
    (x, rows, row): (*const c_void, u32, u32),
    (w, n, tile): (*const c_void, u32, u32),
    (k, j): (u32, u32),
    (poison_x, poison_w): (*const c_void, *const c_void),
    (dump_x, dump_w): (*mut c_void, *mut c_void),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller's contract; the launcher checks every size and null first.
    let code = unsafe {
        scratchy_cuda_gemv_fp8_b8_box_probe(
            x, rows, w, n, k, row, tile, j, poison_x, poison_w, dump_x, dump_w, stream,
        )
    };
    check(code, KernelError::RowShape { rows, cols: k })
}

/// The one-row activation and its quantization (FP8 S8 T1, `kernels/act_quant_fp8.cu`): for each
/// of `rows` rows, `act = bf16(OP(gate, up))` from the gate/up GEMV's f32 `gate`, `up [rows, n]`
/// (`op` a [`crate::step::GatedOp`] code), then that row's `quant_rows_fp8` bytes into `q [rows, n]`
/// e4m3 and its scale into `scale [rows]` f32, bit for bit; `act [rows, n]` bf16 too when it is
/// not null. One CTA a row; asynchronous.
///
/// Refused with [`KernelError::RowShape`] when `rows` is 0 or above 65,535, `n` 0, off 64 or above
/// 17,408, `op` neither SwiGLU's nor GeGLU's, or a pointer but `act` null.
///
/// # Safety
///
/// Every pointer 16-byte-aligned device memory holding its elements, no output overlapping an
/// input; [`prepare_all`] has succeeded on the current device; `stream` a live `cudaStream_t` of
/// it, or null.
pub unsafe fn act_quant_fp8(
    (gate, up): (*const c_void, *const c_void),
    (q, scale, act): (*mut c_void, *mut c_void, *mut c_void),
    (rows, n): (u32, u32),
    op: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller's contract; the launcher checks the shape and the nulls first.
    let code = unsafe { scratchy_cuda_act_quant_fp8(gate, up, q, scale, act, rows, n, op, stream) };
    check(code, KernelError::RowShape { rows, cols: n })
}

/// Test support for `tests/gemv_fp8_tc.rs` only (the ceiling probe, /tmp/d-cap.md §3 P-B):
/// [`gemv_fp8_tc`] at instance `variant` (0 B1, [`gemv_fp8_tc`]'s own; 1 B2 `wait1`, the same
/// bits; 2 B3 `k128`, `k` a multiple of 128; 3 B4 `2cta`; 4 B6 `k256`, 5 B8 `swapab32` and 6 B9
/// `swapab32sk`, `k` a multiple of 256), and with `main_only` (B5 `noreduce`) the
/// stream-K kernel alone, `c` left unwritten. Refused with [`KernelError::GroupShape`] as
/// [`gemv_fp8_tc`], or for `variant` above 6. Asynchronous.
///
/// # Safety
///
/// As [`gemv_fp8_tc`]'s.
#[doc(hidden)]
pub unsafe fn gemv_fp8_tc_variant(
    (q, a_scale): Fp8Operand,
    members: Fp8GroupOperands<'_>,
    partials: *mut c_void,
    k: u32,
    (variant, main_only): (u32, bool),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let ([b, s], c, n, refused) = fp8_group_parts("gemv_fp8_tc_variant", members, (1, k));
    if members.is_empty() || members.len() > 3 {
        return Err(refused);
    }
    // SAFETY: as above; the arrays are three long and the launcher checks the count, the variant
    // and every member's shape before it reads a pointer.
    let code = unsafe {
        scratchy_cuda_gemv_fp8_tc_variant(
            q,
            a_scale,
            b.as_ptr(),
            s.as_ptr(),
            c.as_ptr(),
            n.as_ptr(),
            members.len() as u32,
            k,
            partials,
            variant,
            u32::from(main_only),
            stream,
        )
    };
    check(code, refused)
}

/// The fp8 gate/up GEMV: `c [n] = bf16(act(g, u))`, `g` and `u` each [`gemv_fp8_abt`]'s value
/// before its rounding, `act` a [`crate::step::GatedOp`] code, in f32, rounded once. Under
/// [`gemv_fp8_abt`]'s rule at `splits` for each; at `splits > 1` `partials` holds
/// `2 · splits · n_pad` f32. Asynchronous; refused with [`KernelError::GateUpShape`] before any
/// pointer is touched.
///
/// # Safety
///
/// As [`gemv_fp8_abt`]'s for `a`, `gate`, `up` and `c`.
pub unsafe fn gate_up_gemv_fp8(
    (a, gate, up): (*const c_void, Fp8Operand, Fp8Operand),
    (c, partials): (*mut c_void, *mut c_void),
    (n, k, splits): (u32, u32, u32),
    act: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above; the launcher checks the shape, split and code before it touches a pointer.
    let code = unsafe {
        scratchy_cuda_gate_up_gemv_fp8(
            a, gate.0, gate.1, up.0, up.1, c, partials, n, k, splits, act, stream,
        )
    };
    check(
        code,
        KernelError::GateUpShape {
            kernel: "gate_up_gemv_fp8",
            dims: [1, n, k, splits],
        },
    )
}

/// The gathered expert GEMV over an fp8 bank: [`expert_gemv_split_bf16`]'s pairs, rows and rule
/// (with `k` a multiple of 64 at most 17,408), `b` expert 0's `[expert_rows, k]` e4m3 block and
/// its `[expert_rows]` f32 scale row (of `[experts, expert_rows]`). Each pair's row is
/// [`gemv_fp8_abt`]'s bits at the same split on its row and expert slice. Asynchronous; refused
/// with [`KernelError::MoeShape`] `{ kernel: "expert_gemv_fp8", .. }` before any pointer is
/// touched.
///
/// # Safety
///
/// As [`expert_gemv_split_bf16`]'s, with `b` `experts · expert_rows · k` bytes and its scales
/// `experts · expert_rows` f32.
pub unsafe fn expert_gemv_fp8(
    (a, b, topk_idx): (*const c_void, Fp8Operand, *const c_void),
    (c, partials): (*mut c_void, *mut c_void),
    (m, top_k, n, k): (u32, u32, u32, u32),
    (experts, expert_rows): (u32, u32),
    (pair_rows, splits): (bool, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as above.
    let code = unsafe {
        scratchy_cuda_expert_gemv_fp8(
            a,
            b.0,
            b.1,
            topk_idx,
            c,
            partials,
            m,
            top_k,
            n,
            k,
            experts,
            expert_rows,
            u32::from(pair_rows),
            splits,
            stream,
        )
    };
    check(
        code,
        KernelError::MoeShape {
            kernel: "expert_gemv_fp8",
            dims: [m, top_k, n, k],
        },
    )
}

/// The expert gate/up GEMV over an fp8 bank: per pair, [`gate_up_gemv_fp8`]'s fold of the gate's
/// and the up's [`expert_gemv_fp8`] values on the token rows. Asynchronous; refused with
/// [`KernelError::GateUpShape`] unless each meets that launcher's rule and `act` is a code.
///
/// # Safety
///
/// As [`expert_gemv_fp8`]'s for `a`, each slab, `topk_idx` and `c`; at `splits > 1`, `partials`
/// holds `2 · splits · m · top_k · n` f32, overlapping no operand.
pub unsafe fn expert_gate_up_gemv_fp8(
    (a, gate, up, topk_idx): (*const c_void, Fp8Operand, Fp8Operand, *const c_void),
    (c, partials): (*mut c_void, *mut c_void),
    (m, top_k, n, k): (u32, u32, u32, u32),
    (experts, expert_rows, splits): (u32, u32, u32),
    act: u32,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as `gate_up_gemv_fp8`'s.
    let code = unsafe {
        scratchy_cuda_expert_gate_up_gemv_fp8(
            a,
            gate.0,
            gate.1,
            up.0,
            up.1,
            topk_idx,
            c,
            partials,
            m,
            top_k,
            n,
            k,
            experts,
            expert_rows,
            splits,
            act,
            stream,
        )
    };
    check(
        code,
        KernelError::GateUpShape {
            kernel: "expert_gate_up_gemv_fp8",
            dims: [m.saturating_mul(top_k), n, k, splits],
        },
    )
}

/// Test support for `tests/gemv_fp8.rs` only: the fp8 GEMV's prologue on `a [1, k]` bf16 at
/// `splits`, one CTA per split; split `s` stores the f32 values of its chunks' quantized elements
/// into `out` (`⌈k / 256⌉ · 256` f32, 0 past `k`) and its scale into `scale[s]`. No `CudaStep`
/// uses it. Refused with [`KernelError::GemvSplit`] (`n` 1) unless `k` and `splits` meet
/// [`gemv_fp8_abt`]'s rule. Asynchronous.
///
/// # Safety
///
/// `a` must be a 16-byte-aligned device pointer to `k` bf16, `out` to `⌈k / 256⌉ · 256` f32 and
/// `scale` to `splits` f32, valid for the whole kernel, the outputs overlapping no input. `stream`
/// as [`gemv_fp8_abt`]'s.
#[doc(hidden)]
pub unsafe fn gemv_fp8_quant_probe(
    a: *const c_void,
    (out, scale): (*mut c_void, *mut c_void),
    (k, splits): (u32, u32),
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the pointer and stream contract above; the launcher checks the
    // shape before it touches a pointer and catches every C++ exception.
    let code = unsafe { scratchy_cuda_gemv_fp8_quant_probe(a, out, scale, k, splits, stream) };
    check(code, KernelError::GemvSplit { n: 1, k, splits })
}

/// [`gemv_fp8_abt`] with the residual epilogue (FP8 S8 W4a): `c = bf16(bf16(dot) + residual)`,
/// [`add_bf16`] on [`gemv_fp8_abt`]'s output, the bits of the two launches, at any split.
/// Asynchronous; refused as [`gemv_fp8_abt`] is.
///
/// # Safety
///
/// As [`gemv_fp8_abt`]'s, and `residual` a 16-byte-aligned device pointer to `n` bf16 overlapping
/// no output.
pub unsafe fn gemv_fp8_residual(
    (a, b, c): (*const c_void, Fp8Operand, *mut c_void),
    partials: *mut c_void,
    (n, k, splits): (u32, u32, u32),
    residual: *const c_void,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: the caller upholds the contract above; the launcher checks the shape first.
    let code = unsafe {
        scratchy_cuda_gemv_fp8_residual(a, b.0, b.1, c, partials, n, k, splits, residual, stream)
    };
    check(code, KernelError::GemvSplit { n, k, splits })
}

/// [`gemv_split_bf16_abt`] with the residual epilogue (FP8 S8 W4a), as [`gemv_fp8_residual`].
///
/// # Safety
///
/// As [`gemv_split_bf16_abt`]'s, and `residual` as [`gemv_fp8_residual`]'s.
pub unsafe fn gemv_bf16_residual(
    (a, b, c): (*const c_void, *const c_void, *mut c_void),
    partials: *mut c_void,
    (n, k, splits): (u32, u32, u32),
    residual: *const c_void,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    // SAFETY: as [`gemv_fp8_residual`]'s.
    let code = unsafe {
        scratchy_cuda_gemv_bf16_residual(a, b, c, partials, n, k, splits, residual, stream)
    };
    check(code, KernelError::GemvSplit { n, k, splits })
}

/// The one-row rope append (FP8 S8 W4c, `kernels/rope_append_bf16.cu`): at one row, `q_out =
/// rope(q)`, `k_out = rope(k)`, and k_out's and v's head rows appended to the e4m3 caches at
/// `slot_mapping[0]`, the bits of [`rope_paired_bf16`] on q, the same on k and [`kv_append_fp8`]; with
/// `norms` (`(q_gain, k_gain, eps)`), first q's and k's heads by their RMSNorm and v's by the unit
/// norm, the bits of [`rms_norm_bf16`] (stored gain) and [`rms_norm_unit_bf16`] on the head views.
/// Asynchronous; refused with [`KernelError::KvAppendShape`] before any pointer is touched unless
/// `(rot_dim, pairing, head_dim)` is (64, half-rotated, 256), (256, half-rotated, 256) or (128,
/// half-head, 512), `block_size` 64, and, normed, (256, half-rotated, 256) or (128, half-head, 512)
/// with at most 16 heads of each and a finite positive `eps`.
///
/// # Safety
///
/// `q`, `q_out` must be 16-byte-aligned device pointers to `q_heads · head_dim` bf16; `k`, `k_out`,
/// `v` to `kv_heads · head_dim`; `table` to `rot_dim`; `slot_mapping` to one u32; the caches as
/// [`kv_append_fp8`]'s; the gains to `head_dim` bf16; the outputs overlap no input. [`prepare_all`]
/// must have succeeded on the current device; `stream` as [`rope_paired_bf16`]'s.
pub unsafe fn rope_append_fp8(
    (q, q_out, k, k_out, v): (
        *const c_void,
        *mut c_void,
        *const c_void,
        *mut c_void,
        *const c_void,
    ),
    (table, slot_mapping): (*const c_void, *const c_void),
    (k_cache, v_cache): (*mut c_void, *mut c_void),
    (q_heads, kv_heads, head_dim, rot_dim, pairing): (u32, u32, u32, u32, RopePairing),
    (block_size, num_blocks): (u32, u32),
    norms: Option<(*const c_void, *const c_void, f32)>,
    stream: *mut c_void,
) -> Result<(), KernelError> {
    let pairing_code = match pairing {
        RopePairing::HalfRotated => 0,
        RopePairing::HalfHead => 1,
    };
    let (q_gain, k_gain, eps) = norms.unwrap_or((std::ptr::null(), std::ptr::null(), 0.0));
    // SAFETY: the caller upholds the contract above; the launcher checks every rule first.
    let code = unsafe {
        scratchy_cuda_rope_append_fp8(
            q,
            q_out,
            k,
            k_out,
            v,
            table,
            slot_mapping,
            k_cache,
            v_cache,
            q_heads,
            kv_heads,
            head_dim,
            rot_dim,
            pairing_code,
            block_size,
            num_blocks,
            q_gain,
            k_gain,
            eps,
            stream,
        )
    };
    check(
        code,
        KernelError::KvAppendShape {
            rows: 1,
            kv_heads,
            head_dim,
            block_size,
        },
    )
}
