// SPDX-License-Identifier: Apache-2.0
//! CUTLASS standalone GEMM kernel FFI.
//!
//! Each `cutlass_gemm_<TB_M>x<TB_N>_s<STAGES>_launch` is a specialised
//! CUTLASS GEMM kernel compiled from
//! `scratchy-serving-cuda/csrc/cutlass_standalone_gemm.cu`. They all share the
//! same signature:
//!
//! ```text
//! int cutlass_gemm_..._launch(
//!     uint16* C, const uint16* A, const uint16* B,
//!     int M, int N, int K,
//!     float alpha, float beta, uint64_t stream,
//! );
//! ```
//!
//! where `A` is `[M, K]` row-major, `B` is `[N, K]` row-major
//! (cuBLAS-style weight layout — B^T is the mathematical operand),
//! `C` is `[M, N]`, and the computation is `C = alpha * A @ B^T +
//! beta * C` in BF16 / FP16.
//!
//! `cutlass_gemv_launch` is the M=1 specialisation (SIMT GEMV) with
//! the same prototype.
//!
//! These symbols are defined by `scratchy-serving-cuda`'s C++ build; the final
//! binary (`scratchy-cli`) links both crates so the scratchy-forward-compiler-
//! emitted code in `scratchy-models` can reach them through here.
//!
//! scratchy-forward-compiler's solver picks a tile variant per Gemm tile ×
//! workload bucket from the calibrated `target_profiles/cost_*.csv`
//! tables; the macro-emitted forward calls the corresponding launch
//! fn directly — no runtime tile-variant dispatch.

/// The cutlass tile size + pipeline stages — the `(tile_m, tile_n,
/// stages)` triple uniquely identifies one kernel variant in the
/// library. Use [`CutlassTile::launch`] to invoke.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutlassTile {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
}

impl CutlassTile {
    pub const fn new(tile_m: u32, tile_n: u32, stages: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            stages,
        }
    }

    /// The kernel name used in `target_profiles/cost_*.csv`.
    /// Matches the CSV column value byte-for-byte.
    pub fn csv_name(self) -> String {
        format!("cutlass_{}x{}_s{}", self.tile_m, self.tile_n, self.stages)
    }
}

// Every tile variant compiled by `cutlass_standalone_gemm.cu` that
// is covered by the calibrated CSV tables. When the .cu file gains
// a variant, regenerate the CSV via `gpu_cost_sweep` and add the
// entry here.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn cutlass_gemm_16x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_32x256_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x256_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s3_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_256x64_s4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    // ── SplitK parallel variants ──
    //
    // `cutlass_gemm_<WxH>_s<STAGES>_sk<SLICES>_launch`: same signature
    // as the standard GEMM launchers plus a caller-owned `workspace`
    // pointer. The kernel splits the K dim across `SLICES` CTAs and
    // reduces via a separate reduction kernel.
    //
    // Workspace contract (GemmSplitKParallel): f32 scratch sized
    // `SLICES × M × N × 4` bytes. The Rust safe wrapper allocates this
    // via the scratchy CachingAllocator.
    pub fn cutlass_gemm_64x64_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x64_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_64x128_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x64_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_128x128_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;

    // tile_m=16 splitK — small-M long-K regime (M=8 + K≥8192).
    pub fn cutlass_gemm_16x64_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x64_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_sk2_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_sk4_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_16x128_s4_sk8_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        workspace: *mut u8,
        stream: u64,
    ) -> i32;

    pub fn cutlass_gemv_launch(
        c: *mut u16,
        a: *const u16,
        b: *const u16,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        stream: u64,
    ) -> i32;

    /// Fused Gate GEMM + SiLU + Mul via CUTLASS 2.x EVT.
    /// Computes `D[M,N] = silu(A[M,K] @ B_gate[N,K]^T) * C_up[M,N]`,
    /// where `C_up` is the separately-computed up-projection output
    /// (aux-loaded from GMEM). Internal 3-way tile dispatch on `m`.
    pub fn cutlass_gemm_silu_mul_launch(
        d: *mut u16,
        a: *const u16,
        b_gate: *const u16,
        c_up: *mut u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;

    // ── Bias-add GEMM zoo ──
    //
    // `cutlass_gemm_bias_<TB_M>x<TB_N>_s<STAGES>_launch`. Computes
    // `D[M,N] = A[M,K] @ B[N,K]^T + bias[N]` in one launch, where
    // `bias` is `[N]` bf16 broadcast across rows.
    //
    // Backed by `cutlass::gemm::device::Gemm` (same template family
    // as the standalone tile zoo) with `LinearCombination` epilogue +
    // ldc=0 broadcast — the bias rides as the C operand at stride 0.
    // Tile zoo mirrors `CUTLASS_TILE_ZOO` in
    // scratchy-forward-compiler-macro/src/impl_lib.rs.
    pub fn cutlass_gemm_bias_16x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_16x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_32x256_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_64x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x128_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x128_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_128x256_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_256x64_s3_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
    pub fn cutlass_gemm_bias_256x64_s4_launch(
        d: *mut u16,
        a: *const u16,
        b: *const u16,
        bias: *const u16,
        m: i32,
        n: i32,
        k: i32,
        stream: u64,
    ) -> i32;
}

type CutlassLaunchFn =
    unsafe extern "C" fn(*mut u16, *const u16, *const u16, i32, i32, i32, f32, f32, u64) -> i32;

/// Resolve the `(tile_m, tile_n, stages)` triple to its specialized
/// launch fn. Panics if the triple isn't in the zoo.
#[cfg(feature = "cuda")]
fn launch_fn_for(tile: CutlassTile) -> CutlassLaunchFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_16x64_s3_launch,
        (16, 64, 4) => cutlass_gemm_16x64_s4_launch,
        (16, 128, 3) => cutlass_gemm_16x128_s3_launch,
        (16, 128, 4) => cutlass_gemm_16x128_s4_launch,
        (32, 64, 3) => cutlass_gemm_32x64_s3_launch,
        (32, 64, 4) => cutlass_gemm_32x64_s4_launch,
        (32, 128, 3) => cutlass_gemm_32x128_s3_launch,
        (32, 128, 4) => cutlass_gemm_32x128_s4_launch,
        (32, 256, 3) => cutlass_gemm_32x256_s3_launch,
        (64, 64, 3) => cutlass_gemm_64x64_s3_launch,
        (64, 64, 4) => cutlass_gemm_64x64_s4_launch,
        (64, 128, 3) => cutlass_gemm_64x128_s3_launch,
        (64, 128, 4) => cutlass_gemm_64x128_s4_launch,
        (128, 64, 3) => cutlass_gemm_128x64_s3_launch,
        (128, 64, 4) => cutlass_gemm_128x64_s4_launch,
        (128, 128, 3) => cutlass_gemm_128x128_s3_launch,
        (128, 128, 4) => cutlass_gemm_128x128_s4_launch,
        (128, 256, 3) => cutlass_gemm_128x256_s3_launch,
        (256, 64, 3) => cutlass_gemm_256x64_s3_launch,
        (256, 64, 4) => cutlass_gemm_256x64_s4_launch,
        other => panic!(
            "cutlass: unsupported tile {:?} — add its extern declaration + csv entry",
            other,
        ),
    }
}

/// Safe wrapper: allocate output `[M, N]` and invoke the tile
/// variant's launch fn on the given cuBLAS-convention pointers.
///
/// # Safety
/// All inputs must be valid GPU BF16/FP16 memory with the shapes
/// claimed by their `GpuTensor`. `stream` must be the live compute
/// stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm(
    a: crate::tensor::GpuTensor,
    b: crate::tensor::GpuTensor,
    tile: CutlassTile,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let launch = launch_fn_for(tile);
    let rc = unsafe {
        launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            stream as u64,
        )
    };
    assert_eq!(
        rc, 0,
        "cutlass_gemm {:?} M={m} N={n} K={k} returned {rc}",
        tile
    );
    out
}

/// SplitK tile variant — `(tile_m, tile_n, stages, split_k_slices)`.
/// Backed by `cutlass::gemm::device::GemmSplitKParallel` in the .cu;
/// the kernel launches a partial-GEMM grid followed by a reduction
/// kernel. Workspace is managed statically inside the .cu.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutlassSplitKTile {
    pub tile_m: u32,
    pub tile_n: u32,
    pub stages: u32,
    pub split_k: u32,
}

impl CutlassSplitKTile {
    pub const fn new(tile_m: u32, tile_n: u32, stages: u32, split_k: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            stages,
            split_k,
        }
    }

    /// CSV column name — `cutlass_WxH_sS_splitN`. Must match the
    /// sweep's emitted row name and `CutlassGemmSplitKImpl::csv_name`.
    pub fn csv_name(self) -> String {
        format!(
            "cutlass_{}x{}_s{}_split{}",
            self.tile_m, self.tile_n, self.stages, self.split_k
        )
    }
}

/// SplitK launch fn type — adds a workspace pointer before stream.
#[cfg(feature = "cuda")]
type CutlassSplitKLaunchFn = unsafe extern "C" fn(
    *mut u16,
    *const u16,
    *const u16,
    i32,
    i32,
    i32,
    f32,
    f32,
    *mut u8,
    u64,
) -> i32;

#[cfg(feature = "cuda")]
fn launch_fn_for_splitk(tile: CutlassSplitKTile) -> CutlassSplitKLaunchFn {
    match (tile.tile_m, tile.tile_n, tile.stages, tile.split_k) {
        (64, 64, 4, 2) => cutlass_gemm_64x64_s4_sk2_launch,
        (64, 64, 4, 4) => cutlass_gemm_64x64_s4_sk4_launch,
        (64, 64, 4, 8) => cutlass_gemm_64x64_s4_sk8_launch,
        (64, 128, 4, 2) => cutlass_gemm_64x128_s4_sk2_launch,
        (64, 128, 4, 4) => cutlass_gemm_64x128_s4_sk4_launch,
        (64, 128, 4, 8) => cutlass_gemm_64x128_s4_sk8_launch,
        (128, 64, 4, 2) => cutlass_gemm_128x64_s4_sk2_launch,
        (128, 64, 4, 4) => cutlass_gemm_128x64_s4_sk4_launch,
        (128, 64, 4, 8) => cutlass_gemm_128x64_s4_sk8_launch,
        (128, 128, 4, 2) => cutlass_gemm_128x128_s4_sk2_launch,
        (128, 128, 4, 4) => cutlass_gemm_128x128_s4_sk4_launch,
        (128, 128, 4, 8) => cutlass_gemm_128x128_s4_sk8_launch,
        (16, 64, 4, 2) => cutlass_gemm_16x64_s4_sk2_launch,
        (16, 64, 4, 4) => cutlass_gemm_16x64_s4_sk4_launch,
        (16, 64, 4, 8) => cutlass_gemm_16x64_s4_sk8_launch,
        (16, 128, 4, 2) => cutlass_gemm_16x128_s4_sk2_launch,
        (16, 128, 4, 4) => cutlass_gemm_16x128_s4_sk4_launch,
        (16, 128, 4, 8) => cutlass_gemm_16x128_s4_sk8_launch,
        other => panic!(
            "cutlass splitk: unsupported tile {:?} — add its extern + csv entry",
            other,
        ),
    }
}

/// SplitK GEMM: `C[M, N] = A[M, K] @ B[N, K]^T` with the K dim split
/// across `split_k` CTAs, reduced in a second kernel. Same calling
/// convention as [`cutlass_gemm`].
///
/// # Safety
/// All inputs must be valid GPU BF16/FP16 memory with the shapes
/// claimed by their `GpuTensor`. `stream` must be the live compute
/// stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_splitk(
    a: crate::tensor::GpuTensor,
    b: crate::tensor::GpuTensor,
    tile: CutlassSplitKTile,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "splitK GEMM K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());

    // GemmSplitKParallel workspace: partial-sum buffer of shape
    // [split_k, M, N] in f32 (accumulator type). Allocated through
    // the same caching allocator that owns every other scratch
    // tensor in the forward — never cudaMalloc/Free per call.
    let ws_elems = (tile.split_k as usize) * m * n;
    let workspace = alloc.alloc_tensor(&[ws_elems], crate::DType::F32);

    let launch = launch_fn_for_splitk(tile);
    let rc = unsafe {
        launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            workspace.as_mut_ptr::<u8>(),
            stream as u64,
        )
    };
    assert_eq!(
        rc, 0,
        "cutlass_gemm_splitk {:?} M={m} N={n} K={k} returned {rc}",
        tile
    );
    // workspace dropped here → returned to caching pool. Safe because
    // the forward runs on a single compute stream: any later alloc
    // that recycles these bytes enqueues its kernel behind the splitK
    // GEMM+reduce on the same stream (FIFO), so the bytes are read to
    // completion before they're rewritten. Same discipline as
    // `cutlass_gemm_silu_mul` dropping `up_out` post-launch.
    drop(workspace);
    out
}

/// Fused GEMM + residual-add via CUTLASS `LinearCombination`
/// epilogue: mutates `residual` in-place with `residual += A @ B^T`
/// (alpha=1.0, beta=1.0). Same buffer-aliasing convention as
/// `add_inplace` — caller binds the downstream tile to a TensorView
/// of `residual` via `as_view()`.
///
/// # Safety
/// Same as [`cutlass_gemm`]. `residual` must point to writable GPU
/// memory of shape `[M, N]` (the output shape of `A @ B^T`).
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_add(
    a: crate::tensor::GpuTensor,
    b: crate::tensor::GpuTensor,
    residual: crate::tensor::GpuTensor,
    tile: CutlassTile,
    stream: cudarc::driver::sys::CUstream,
) {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(residual.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "cutlass_gemm_add K mismatch");
    debug_assert_eq!(a.dim(0), residual.dim(0), "cutlass_gemm_add M mismatch");
    debug_assert_eq!(b.dim(0), residual.dim(1), "cutlass_gemm_add N mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let launch = launch_fn_for(tile);
    let rc = unsafe {
        launch(
            residual.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            1.0,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemm_add {:?} returned {}", tile, rc);
}

/// M=1 specialisation — SIMT GEMV.
///
/// # Safety
/// Same as [`cutlass_gemm`].
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemv(
    a: crate::tensor::GpuTensor,
    b: crate::tensor::GpuTensor,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    debug_assert_eq!(a.dim(1), b.dim(1), "GEMV K mismatch");
    let m = a.dim(0);
    let n = b.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let rc = unsafe {
        cutlass_gemv_launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            b.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            1.0,
            0.0,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemv returned {}", rc);
    out
}

/// Bias-add launch fn type — same prototype as the per-tile externs
/// in `cutlass_gemm_bias.cu`.
#[cfg(feature = "cuda")]
type CutlassBiasLaunchFn =
    unsafe extern "C" fn(*mut u16, *const u16, *const u16, *const u16, i32, i32, i32, u64) -> i32;

#[cfg(feature = "cuda")]
fn launch_fn_for_bias(tile: CutlassTile) -> CutlassBiasLaunchFn {
    match (tile.tile_m, tile.tile_n, tile.stages) {
        (16, 64, 3) => cutlass_gemm_bias_16x64_s3_launch,
        (16, 64, 4) => cutlass_gemm_bias_16x64_s4_launch,
        (16, 128, 3) => cutlass_gemm_bias_16x128_s3_launch,
        (16, 128, 4) => cutlass_gemm_bias_16x128_s4_launch,
        (32, 64, 3) => cutlass_gemm_bias_32x64_s3_launch,
        (32, 64, 4) => cutlass_gemm_bias_32x64_s4_launch,
        (32, 128, 3) => cutlass_gemm_bias_32x128_s3_launch,
        (32, 128, 4) => cutlass_gemm_bias_32x128_s4_launch,
        (32, 256, 3) => cutlass_gemm_bias_32x256_s3_launch,
        (64, 64, 3) => cutlass_gemm_bias_64x64_s3_launch,
        (64, 64, 4) => cutlass_gemm_bias_64x64_s4_launch,
        (64, 128, 3) => cutlass_gemm_bias_64x128_s3_launch,
        (64, 128, 4) => cutlass_gemm_bias_64x128_s4_launch,
        (128, 64, 3) => cutlass_gemm_bias_128x64_s3_launch,
        (128, 64, 4) => cutlass_gemm_bias_128x64_s4_launch,
        (128, 128, 3) => cutlass_gemm_bias_128x128_s3_launch,
        (128, 128, 4) => cutlass_gemm_bias_128x128_s4_launch,
        (128, 256, 3) => cutlass_gemm_bias_128x256_s3_launch,
        (256, 64, 3) => cutlass_gemm_bias_256x64_s3_launch,
        (256, 64, 4) => cutlass_gemm_bias_256x64_s4_launch,
        other => panic!(
            "cutlass_gemm_bias: unsupported tile {:?} — add its extern + csv entry",
            other,
        ),
    }
}

/// Fused GEMM + bias broadcast.
///
/// Computes `D[M, N] = A @ W^T + bias` where:
/// - `a` is `[M, K]` bf16 activation,
/// - `weight` is `[N, K]` bf16 weight (row-major, cuBLAS-convention),
/// - `bias` is `[N]` bf16 per-column bias,
/// - output `D` is `[M, N]` bf16 allocated fresh.
///
/// `tile` selects the threadblock variant; the DP solver picks per
/// (M, N, K) workload from calibrated CSV rows.
///
/// # Safety
/// All inputs must be valid GPU bf16 memory with the shapes claimed
/// by their `GpuTensor`. `stream` must be the live compute stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_bias(
    a: crate::tensor::GpuTensor,
    weight: crate::tensor::GpuTensor,
    bias: crate::tensor::GpuTensor,
    tile: CutlassTile,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(weight.ndim(), 2);
    debug_assert_eq!(bias.ndim(), 1);
    debug_assert_eq!(a.dim(1), weight.dim(1), "cutlass_gemm_bias K mismatch");
    debug_assert_eq!(weight.dim(0), bias.dim(0), "cutlass_gemm_bias N mismatch");
    let m = a.dim(0);
    let n = weight.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let launch = launch_fn_for_bias(tile);
    let rc = unsafe {
        launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            weight.as_ptr::<u16>(),
            bias.as_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemm_bias {:?} returned {}", tile, rc);
    out
}

/// Fused Gate GEMM + SiLU + Mul in a single CUTLASS EVT kernel.
///
/// Computes `D[M, I] = silu(A @ W_gate^T) * C_up` where:
/// - `a` is `[M, K]` bf16 activation (row-major),
/// - `gate_weight` is `[I, K]` bf16 gate-projection weight (row-major,
///   cuBLAS-convention — the kernel treats it as `W^T`),
/// - `up_out` is a pre-computed `[M, I]` bf16 up-projection output
///   that the epilogue aux-loads from GMEM,
/// - output `D` is `[M, I]` bf16 allocated fresh.
///
/// # Safety
/// All inputs must be valid GPU bf16 memory with the shapes claimed
/// by their `GpuTensor`. `stream` must be the live compute stream.
#[cfg(feature = "cuda")]
pub unsafe fn cutlass_gemm_silu_mul(
    a: crate::tensor::GpuTensor,
    gate_weight: crate::tensor::GpuTensor,
    up_out: crate::alloc::OwnedTensor,
    alloc: &mut crate::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> crate::alloc::OwnedTensor {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(gate_weight.ndim(), 2);
    debug_assert_eq!(up_out.ndim(), 2);
    debug_assert_eq!(
        a.dim(1),
        gate_weight.dim(1),
        "cutlass_gemm_silu_mul K mismatch"
    );
    debug_assert_eq!(a.dim(0), up_out.dim(0), "cutlass_gemm_silu_mul M mismatch");
    debug_assert_eq!(
        gate_weight.dim(0),
        up_out.dim(1),
        "cutlass_gemm_silu_mul I mismatch"
    );
    let m = a.dim(0);
    let n = gate_weight.dim(0);
    let k = a.dim(1);
    let out = alloc.alloc_tensor(&[m, n], a.dtype());
    let rc = unsafe {
        cutlass_gemm_silu_mul_launch(
            out.as_mut_ptr::<u16>(),
            a.as_ptr::<u16>(),
            gate_weight.as_ptr::<u16>(),
            up_out.as_mut_ptr::<u16>(),
            m as i32,
            n as i32,
            k as i32,
            stream as u64,
        )
    };
    debug_assert_eq!(rc, 0, "cutlass_gemm_silu_mul returned {}", rc);
    // up_out is consumed — its buffer is no longer needed after the
    // epilogue aux-load completes.
    drop(up_out);
    out
}
