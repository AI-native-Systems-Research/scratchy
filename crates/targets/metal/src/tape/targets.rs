// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal target profiles for Apple Silicon devices.
//!
//! Provides device-specific parameters for M1-M5 chips. Kernel
//! selection is closed-form (see `tape::quantized`); costs, where a
//! solver still wants them, come from the analytical roofline
//! (bandwidth/TFLOPS below) — there is no empirical cost table.

use serde::{Deserialize, Serialize};

/// Apple Silicon architecture generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AppleSiliconGen {
    M1,
    M2,
    M3,
    M4,
    /// Apple9 gen 17+ — first generation with the NAX (Neural Accelerator
    /// eXtension) hardware MMA. See [`is_nax_capable`].
    M5,
}

/// Metal device profile containing hardware specs and cost models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalTargetProfile {
    /// Architecture generation (M1/M2/M3/M4)
    pub generation: AppleSiliconGen,

    /// Number of GPU cores
    pub gpu_cores: u32,

    /// Peak TFLOPS for FP16 operations
    pub peak_tflops_fp16: f64,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,

    /// Unified memory size in GB
    pub unified_memory_gb: u32,

    /// Maximum threadgroup memory in bytes (32KB for all Apple Silicon)
    pub threadgroup_memory_bytes: u32,

    /// Maximum threads per threadgroup
    pub max_threads_per_threadgroup: u32,
}

/// M1 device profile (base model, 8 GPU cores)
pub const M1_8CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M1,
    gpu_cores: 8,
    peak_tflops_fp16: 2.6,
    memory_bandwidth_gbps: 68.25,
    unified_memory_gb: 16,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
};

/// M1 Max device profile (32 GPU cores)
pub const M1_MAX: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M1,
    gpu_cores: 32,
    peak_tflops_fp16: 10.4,
    memory_bandwidth_gbps: 400.0,
    unified_memory_gb: 64,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
};

/// M2 device profile (10 GPU cores)
pub const M2_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M2,
    gpu_cores: 10,
    peak_tflops_fp16: 3.6,
    memory_bandwidth_gbps: 100.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
};

/// M3 device profile (base model, 10 GPU cores)
pub const M3_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M3,
    gpu_cores: 10,
    peak_tflops_fp16: 4.0,
    memory_bandwidth_gbps: 100.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
};

/// M4 device profile (10 GPU cores)
pub const M4_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M4,
    gpu_cores: 10,
    peak_tflops_fp16: 4.5,
    memory_bandwidth_gbps: 120.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
};

/// M5 device profile (base model, 10 GPU cores).
///
/// First generation with NAX hardware MMA (gen 17 ≥ 17 — see
/// [`is_nax_capable`]). Perf figures are estimates pending a cost sweep
/// on this chip; the empty `cost_table` forces the solver onto the
/// analytical roofline, so these only affect cost-model scoring, not
/// correctness.
pub const M5_10CORE: MetalTargetProfile = MetalTargetProfile {
    generation: AppleSiliconGen::M5,
    gpu_cores: 10,
    peak_tflops_fp16: 5.0,
    memory_bandwidth_gbps: 150.0,
    unified_memory_gb: 24,
    threadgroup_memory_bytes: 32768,
    max_threads_per_threadgroup: 1024,
};

/// Returns `true` if the given generation has the NAX (Neural Accelerator
/// eXtension) hardware MMA that MLX's `BaseNAXFrag` cooperative-tensor
/// layout assumes.
///
/// NAX is **M5+ / A19+ only** — not M4. MLX's own gate is `arch_gen >= 17`
/// (M5 is gen 17, M4 is gen 16; see `mlx/backend/metal/device.cpp:828`
/// `is_nax_available`). `MetalPerformancePrimitives matmul2d` is callable
/// on M4 but emulates via the standard simdgroup matmul with a
/// cooperative-tensor per-thread layout that does NOT match
/// `BaseNAXFrag`'s 2-row × 4-col assumption — see the diagnostic
/// `nax_probe_dump_layout` reproducer in
/// `crates/targets/metal/tests/quantized_qmm_test.rs`.
///
/// Returns `false` for M1–M4 (gen ≤ 16, no NAX hardware — M4 emulates
/// `matmul2d` via the standard simdgroup matmul, yielding a
/// cooperative-tensor layout that does NOT match `BaseNAXFrag`). Returns
/// `true` for M5+ (gen ≥ 17), validated against the layout probe on an
/// Apple M5 (MacBook Pro, macOS 26.5): the `ct_c` per-thread coords come
/// back in the contiguous 2×4 `BaseNAXFrag` pattern, distinct from the
/// M4 emulation layout. See `nax_probe_dump_layout`.
pub fn is_nax_capable(g: AppleSiliconGen) -> bool {
    matches!(g, AppleSiliconGen::M5)
}

/// Returns `true` when the GPU's bf16 simdgroup MMA path is slow
/// enough that loading bf16 from memory and running the MMA in fp16
/// is a perf win — the M1 generation only.
///
/// On M1 (Apple7), `simdgroup_multiply_accumulate` of
/// `simdgroup_matrix<bfloat>` runs through a software emulation path
/// and clocks ~1.7× slower than `simdgroup_matrix<half>` on the same
/// shapes (validated empirically: 4.58 TF/s bf16 vs 7.74 TF/s f16 on
/// `affine_qmm_t_*_gs_64_b_4_alN_true_batch_0` at M=1024 N=3072 K=3072).
/// M2 added partial hardware bf16 support; M3+ has fully accelerated
/// bf16 plus the NAX matrix unit.
///
/// The qmm_t lowering reads this to pick a `T_compute=half`
/// instantiation when the model's activation dtype is bf16, casting
/// bf16↔half inside the kernel only — the residual stream stays bf16
/// so dynamic-range correctness is preserved (full-f16 streams break
/// Llama-3.x exponent range — see `scratchy-forward-compiler/src/instr.rs:199-202`).
pub fn bf16_simdgroup_is_slow_path(g: AppleSiliconGen) -> bool {
    matches!(g, AppleSiliconGen::M1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)] // deliberate const sanity checks on the profile tables
    fn test_profile_constants() {
        assert_eq!(M1_8CORE.generation, AppleSiliconGen::M1);
        assert_eq!(M1_8CORE.gpu_cores, 8);
        assert_eq!(M2_10CORE.gpu_cores, 10);
        assert!(M3_10CORE.peak_tflops_fp16 > M2_10CORE.peak_tflops_fp16);
    }
}
