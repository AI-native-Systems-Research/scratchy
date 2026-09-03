// SPDX-License-Identifier: Apache-2.0
//! cuBLAS handle with cached GEMM plans for zero-overhead repeated calls.
//!
//! All GEMMs go through **cublasLt** with heuristic algorithm selection.
//! Plans (descriptors + algorithm) are cached by `(M, K, N, dtype, has_bias)`
//! so that the first call creates the plan and subsequent calls with the same
//! shapes reuse it — eliminating ~20 API calls per GEMM.
//!
//! The workspace is bound to the handle via `cublasSetWorkspace()` at init,
//! so cuBLAS never lazily allocates on the default stream — this is what
//! makes CUDA graph capture work.

use std::collections::HashMap;

use crate::alloc::{CachingAllocator, OwnedTensor};
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::{Result, bail};
use cudarc::cublas::sys::{self, cublasComputeType_t, cublasHandle_t, cublasOperation_t};
use cudarc::cublaslt::sys as lt;
use cudarc::driver::sys::CUstream;

/// cuBLAS-Lt workspace size — matches PyTorch's `getCUDABlasLtWorkspaceSize`
/// default of 1024 KiB on every arch (per `aten/src/ATen/cuda/CublasHandlePool.cpp`,
/// upstream PR #73328). The previous 32 MiB matched PyTorch's separate non-Lt
/// cuBLAS workspace on Hopper (`parseChosenWorkspaceSize`) but cuBLAS-Lt itself
/// stays at 1 MiB, and that's the path scratchy uses for all GEMMs.
const CUBLAS_WORKSPACE_SIZE: usize = 1024 * 1024;

/// Cache key for a GEMM plan.
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct PlanKey {
    m: usize,
    k: usize,
    n: usize,
    dtype: DType,
    has_bias: bool,
    /// If true, weight (A in cuBLAS) is transposed (default). If false, no transpose on weight.
    weight_trans: bool,
}

/// Maximum number of candidate algorithms to request from the cuBLAS heuristic.
/// If the top-ranked algorithm fails at runtime (e.g. during CUDA graph capture),
/// we fall back to the next candidate.
const MAX_ALGO_CANDIDATES: usize = 4;

/// A cached cublasLt GEMM plan — holds pre-created descriptors and candidate algorithms.
/// The first algorithm in `algos` is the heuristic's top pick; subsequent entries are
/// fallbacks tried in order if the primary fails.
struct GemmPlan {
    matmul_desc: lt::cublasLtMatmulDesc_t,
    layout_a: lt::cublasLtMatrixLayout_t,
    layout_b: lt::cublasLtMatrixLayout_t,
    layout_c: lt::cublasLtMatrixLayout_t,
    /// Candidate algorithms in heuristic-ranked order.
    algos: Vec<lt::cublasLtMatmulAlgo_t>,
    /// GEMM dimensions (for diagnostic messages on failure).
    key: PlanKey,
}

impl Drop for GemmPlan {
    fn drop(&mut self) {
        unsafe {
            lt::cublasLtMatrixLayoutDestroy(self.layout_a);
            lt::cublasLtMatrixLayoutDestroy(self.layout_b);
            lt::cublasLtMatrixLayoutDestroy(self.layout_c);
            lt::cublasLtMatmulDescDestroy(self.matmul_desc);
        }
    }
}

/// Cache key for an FP8 GEMM plan (input FP8, output BF16/F16).
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct Fp8PlanKey {
    m: usize,
    k: usize,
    n: usize,
    output_dtype: DType, // BF16 or F16
}

/// A cached FP8 GEMM plan. Scale pointers are set per-call.
struct Fp8GemmPlan {
    matmul_desc: lt::cublasLtMatmulDesc_t,
    layout_a: lt::cublasLtMatrixLayout_t,
    layout_b: lt::cublasLtMatrixLayout_t,
    layout_c: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
}

impl Drop for Fp8GemmPlan {
    fn drop(&mut self) {
        unsafe {
            lt::cublasLtMatrixLayoutDestroy(self.layout_a);
            lt::cublasLtMatrixLayoutDestroy(self.layout_b);
            lt::cublasLtMatrixLayoutDestroy(self.layout_c);
            lt::cublasLtMatmulDescDestroy(self.matmul_desc);
        }
    }
}

/// cuBLAS + cublasLt handles bound to a non-default stream with pre-allocated workspace.
pub struct CublasHandle {
    handle: cublasHandle_t,
    lt_handle: lt::cublasLtHandle_t,
    stream: CUstream,
    workspace: *mut u8,
    /// Cached GEMM plans keyed by (M, K, N, dtype, has_bias).
    plans: HashMap<PlanKey, GemmPlan>,
    /// Cached FP8 GEMM plans keyed by (M, K, N, output_dtype).
    fp8_plans: HashMap<Fp8PlanKey, Fp8GemmPlan>,
}

// Safety: cuBLAS handle is thread-safe when each thread uses its own handle
// or synchronizes externally. We use one handle per GpuDevice.
unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    /// Create a new cuBLAS handle bound to `stream` with a pre-allocated workspace.
    ///
    /// Workspace is allocated from the caching allocator (not raw cudaMalloc)
    /// so it participates in the graph-aware memory pool during CUDA graph
    /// capture — matching PyTorch's CublasHandlePool.cpp:getNewWorkspace()
    /// which allocates via CUDACachingAllocator::get()->allocate().
    ///
    /// # Safety
    /// `stream` must be a valid non-default CUDA stream.
    pub unsafe fn new(stream: CUstream, alloc: &mut CachingAllocator) -> Result<Self> {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        check(sys::cublasCreate_v2(&mut handle))?;

        // Bind to our non-default stream.
        check(sys::cublasSetStream_v2(handle, stream as _))?;

        // Enable TF32 tensor cores (2x throughput for F32 on Ampere+).
        check(sys::cublasSetMathMode(
            handle,
            sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        ))?;

        // Allocate workspace from the caching allocator so it's part of the
        // graph-aware pool during CUDA graph capture.
        let workspace = alloc.alloc(CUBLAS_WORKSPACE_SIZE);
        check(sys::cublasSetWorkspace_v2(
            handle,
            workspace as *mut _,
            CUBLAS_WORKSPACE_SIZE,
        ))?;

        // Create cublasLt handle.
        let mut lt_handle: lt::cublasLtHandle_t = std::ptr::null_mut();
        check_lt(lt::cublasLtCreate(&mut lt_handle))?;

        Ok(Self {
            handle,
            lt_handle,
            stream,
            workspace,
            plans: HashMap::new(),
            fp8_plans: HashMap::new(),
        })
    }

    /// Re-allocate the cuBLAS workspace from `alloc` and rebind it.
    ///
    /// Must be called after `CachingAllocator::release_all()` freed the old
    /// workspace (e.g. during sleep → wake cycle). Also clears cached plans
    /// since they may reference the old workspace.
    ///
    /// # Safety
    /// Caller must ensure no in-flight cuBLAS operations reference the old workspace.
    pub unsafe fn rebind_workspace(&mut self, alloc: &mut CachingAllocator) {
        self.plans.clear();
        self.fp8_plans.clear();
        let workspace = alloc.alloc(CUBLAS_WORKSPACE_SIZE);
        check(sys::cublasSetWorkspace_v2(
            self.handle,
            workspace as *mut _,
            CUBLAS_WORKSPACE_SIZE,
        ))
        .expect("rebind cuBLAS workspace");
        self.workspace = workspace;
    }

    /// Ensure a cached GEMM plan exists for the given shapes, creating it if needed.
    ///
    /// `weight_trans`: if true, weight is transposed (TRANSA=T, default for `[N,K]` weights).
    /// If false, weight is NOT transposed (TRANSA=N, for BNB `[K,N]` dequanted weights).
    unsafe fn ensure_plan(
        &mut self,
        m: usize,
        k: usize,
        n: usize,
        dtype: DType,
        has_bias: bool,
        weight_trans: bool,
    ) {
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias,
            weight_trans,
        };
        if self.plans.contains_key(&key) {
            return;
        }

        let (_compute_type, data_type) = gemm_types(dtype);
        let lt_data_type = cublas_to_lt_dtype(data_type);
        let lt_compute_type = cublas_to_lt_compute(_compute_type);

        // Create matmul descriptor.
        let mut matmul_desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatmulDescCreate(
            &mut matmul_desc,
            lt_compute_type,
            lt::cudaDataType_t::CUDA_R_32F, // scale type always F32
        ))
        .expect("cublasLtMatmulDescCreate failed");

        // Set transpose operations.
        // weight_trans=true:  C^T = B @ A^T in col-major → C = A @ B^T in row-major (B is [N,K])
        // weight_trans=false: C^T = B @ A in col-major → C = A @ B in row-major (B is [K,N], e.g. BNB dequant)
        let transa = if weight_trans {
            cublasOperation_t::CUBLAS_OP_T as i32
        } else {
            cublasOperation_t::CUBLAS_OP_N as i32
        };
        let transb = cublasOperation_t::CUBLAS_OP_N as i32;
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ))
        .expect("set TRANSA failed");
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ))
        .expect("set TRANSB failed");

        // Set bias epilogue if needed.
        if has_bias {
            let epilogue = lt::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
            check_lt(lt::cublasLtMatmulDescSetAttribute(
                matmul_desc,
                lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue as *const _ as *const _,
                std::mem::size_of_val(&epilogue),
            ))
            .expect("set EPILOGUE failed");
        }

        // Create matrix layouts (column-major convention).
        // A in cuBLAS = B (weight).
        //   weight_trans=true:  [N, K] row-major → col-major rows=K, cols=N, ld=K
        //   weight_trans=false: [K, N] row-major → col-major rows=N, cols=K, ld=N
        let mut layout_a: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        let (a_rows, a_cols, a_ld) = if weight_trans { (k, n, k) } else { (n, k, n) };
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_a,
            lt_data_type,
            a_rows as u64,
            a_cols as u64,
            a_ld as i64,
        ))
        .expect("layout A");

        // B in cuBLAS = A (activation) [M, K] col-major → rows=K, cols=M, ld=K
        let mut layout_b: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_b,
            lt_data_type,
            k as u64,
            m as u64,
            k as i64,
        ))
        .expect("layout B");

        // C/D = output [M, N] col-major → rows=N, cols=M, ld=N
        let mut layout_c: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_c,
            lt_data_type,
            n as u64,
            m as u64,
            n as i64,
        ))
        .expect("layout C");

        // Get heuristic for candidate algorithms (request multiple for fallback).
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatmulPreferenceCreate(&mut pref)).expect("pref create");
        let ws_size = CUBLAS_WORKSPACE_SIZE;
        check_lt(lt::cublasLtMatmulPreferenceSetAttribute(
            pref,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_size as *const _ as *const _,
            std::mem::size_of::<usize>(),
        ))
        .expect("set pref workspace");

        let mut heuristics =
            vec![std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>(); MAX_ALGO_CANDIDATES];
        let mut algo_count: i32 = 0;
        check_lt(lt::cublasLtMatmulAlgoGetHeuristic(
            self.lt_handle,
            matmul_desc,
            layout_a,
            layout_b,
            layout_c,
            layout_c, // D layout = C layout
            pref,
            MAX_ALGO_CANDIDATES as i32,
            heuristics.as_mut_ptr(),
            &mut algo_count,
        ))
        .expect("cublasLtMatmulAlgoGetHeuristic failed");
        assert!(algo_count > 0, "no cublasLt algorithm found");

        lt::cublasLtMatmulPreferenceDestroy(pref);

        let algos: Vec<lt::cublasLtMatmulAlgo_t> = heuristics[..algo_count as usize]
            .iter()
            .map(|h| h.algo)
            .collect();

        let plan = GemmPlan {
            matmul_desc,
            layout_a,
            layout_b,
            layout_c,
            algos,
            key,
        };

        self.plans.insert(key, plan);
    }

    /// Execute a GEMM, handling both capture and non-capture paths.
    ///
    /// During CUDA graph capture, goes straight to cublasGemmEx without
    /// creating cublasLt plans (plan creation via cublasLtMatmulAlgoGetHeuristic
    /// can poison the capture). Outside capture, creates/caches cublasLt plans
    /// for optimal algorithm selection.
    unsafe fn run_gemm(
        &mut self,
        key: PlanKey,
        a_ptr: *const std::ffi::c_void,
        b_ptr: *const std::ffi::c_void,
        out_ptr: *mut std::ffi::c_void,
    ) {
        let _capture_guard = crate::alloc::RelaxedCaptureModeGuard::new();

        let is_capturing = {
            let mut status =
                cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            let ret = cudarc::driver::sys::cuStreamIsCapturing(self.stream, &mut status);
            if ret == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                match status {
                    cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE => true,
                    cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED => {
                        panic!(
                            "CUDA graph capture already INVALIDATED before GEMM [M={}, K={}, N={}] {:?}",
                            key.m, key.k, key.n, key.dtype,
                        );
                    }
                    _ => false,
                }
            } else {
                false
            }
        };

        if is_capturing {
            // Skip ensure_plan — cublasLtMatmulAlgoGetHeuristic can poison capture.
            self.gemm_ex(&key, a_ptr, b_ptr, out_ptr);
            return;
        }

        // Outside capture: use cublasLt with cached plans.
        self.ensure_plan(
            key.m,
            key.k,
            key.n,
            key.dtype,
            key.has_bias,
            key.weight_trans,
        );
        let plan = &self.plans[&key];
        self.run_matmul_with_fallback(plan, a_ptr, b_ptr, out_ptr);
    }

    /// Run a GEMM using cublasLt with fallback to cublasGemmEx.
    /// Only called outside CUDA graph capture (capture path uses run_gemm directly).
    unsafe fn run_matmul_with_fallback(
        &self,
        plan: &GemmPlan,
        a_ptr: *const std::ffi::c_void,
        b_ptr: *const std::ffi::c_void,
        out_ptr: *mut std::ffi::c_void,
    ) {
        let k = &plan.key;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        for (i, algo) in plan.algos.iter().enumerate() {
            let status = lt::cublasLtMatmul(
                self.lt_handle,
                plan.matmul_desc,
                &alpha as *const f32 as *const _,
                a_ptr,
                plan.layout_a,
                b_ptr,
                plan.layout_b,
                &beta as *const f32 as *const _,
                out_ptr,
                plan.layout_c,
                out_ptr,
                plan.layout_c,
                algo,
                self.workspace as *mut _,
                CUBLAS_WORKSPACE_SIZE,
                self.stream as _,
            );
            if status == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                if i > 0 {
                    tracing::warn!(
                        "cublasLtMatmul: primary algo failed for GEMM [M={}, K={}, N={}] {:?}, \
                         fell back to candidate #{}",
                        k.m,
                        k.k,
                        k.n,
                        k.dtype,
                        i + 1
                    );
                }
                return;
            }
            tracing::debug!(
                "cublasLtMatmul algo #{} failed ({:?}) for GEMM [M={}, K={}, N={}] {:?}",
                i + 1,
                status,
                k.m,
                k.k,
                k.n,
                k.dtype,
            );
        }

        // All cublasLt algorithms failed outside capture — fall back to cublasGemmEx.
        tracing::warn!(
            "cublasLtMatmul: all {} algos failed for GEMM [M={}, K={}, N={}] {:?}, \
             falling back to cublasGemmEx",
            plan.algos.len(),
            k.m,
            k.k,
            k.n,
            k.dtype,
        );
        self.gemm_ex(k, a_ptr, b_ptr, out_ptr);
    }

    /// cublasGemmEx — always capture-safe.
    ///
    /// Uses CUBLAS_GEMM_DEFAULT which lets cuBLAS pick the algorithm
    /// internally, matching PyTorch's default GEMM path.
    /// Primary path during CUDA graph capture; fallback outside capture.
    unsafe fn gemm_ex(
        &self,
        k: &PlanKey,
        a_ptr: *const std::ffi::c_void,
        b_ptr: *const std::ffi::c_void,
        out_ptr: *mut std::ffi::c_void,
    ) {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let (_compute_type, data_type) = gemm_types(k.dtype);

        // cublasGemmEx uses column-major convention.
        // Our row-major C = A @ B^T becomes col-major C^T = B @ A^T.
        //   cuBLAS op: C_col(N,M) = B_col(N,K) * A_col^T(K,M)
        //     transa = T (transpose A_col to get K×M → M×K → matches our A row-major)
        //     transb = N (B_col is K×N which is B row-major [N,K] reinterpreted)
        //
        // Wait — this is the same mapping as cublasLt:
        //   cuBLAS A = our weight b [N,K] row = [K,N] col, transa = T → [N,K]
        //   cuBLAS B = our activation a [M,K] row = [K,M] col, transb = N → [K,M]
        //   result: [N,K] × [K,M] = [N,M] col = [M,N] row ✓
        let (transa, transb) = if k.weight_trans {
            (
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
            )
        } else {
            (
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
            )
        };

        let (lda, ldb) = if k.weight_trans {
            (k.k as i32, k.k as i32) // weight [N,K] col has ld=K; activation [M,K] col has ld=K
        } else {
            (k.n as i32, k.k as i32) // weight [K,N] col has ld=N; activation [M,K] col has ld=K
        };

        let status = sys::cublasGemmEx(
            self.handle,
            transa,
            transb,
            k.n as i32, // M in col-major = N (output rows)
            k.m as i32, // N in col-major = M (output cols)
            k.k as i32, // K
            &alpha as *const f32 as *const _,
            a_ptr, // cuBLAS A = our weight (b)
            data_type,
            lda,
            b_ptr, // cuBLAS B = our activation (a)
            data_type,
            ldb,
            &beta as *const f32 as *const _,
            out_ptr,
            data_type,
            k.n as i32, // ldc = N (output leading dim in col-major)
            _compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );
        check(status).unwrap_or_else(|_| {
            panic!(
                "cublasGemmEx failed for GEMM [M={}, K={}, N={}] {:?} status={} trans={}",
                k.m, k.k, k.n, k.dtype, status as u32, k.weight_trans,
            )
        });
    }

    /// GEMM: out = A @ B^T
    ///
    /// - `a`: `[M, K]` row-major
    /// - `b`: `[N, K]` row-major (weight stored as `[out_features, in_features]`)
    /// - Returns: `[M, N]` allocated from `arena`
    ///
    /// Uses cublasLt with cached plans for optimal algorithm selection,
    /// especially split-K for decode-regime skinny GEMMs (M=1-8).
    ///
    /// # Safety
    /// `a` and `b` must be valid GPU tensors with compatible dtypes and shapes.
    pub unsafe fn gemm(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(a.ndim(), 2);
        debug_assert_eq!(b.ndim(), 2);
        debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");

        let m = a.dim(0);
        let k = a.dim(1);
        let n = b.dim(0);
        let dtype = a.dtype();

        let out = alloc.alloc_tensor(&[m, n], dtype);
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias: false,
            weight_trans: true,
        };

        self.run_gemm(
            key,
            b.as_ptr::<u8>() as *const _,
            a.as_ptr::<u8>() as *const _,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
        );

        out
    }

    /// GEMM with fused bias add: out = A @ B^T + bias
    ///
    /// Uses cublasLt with `CUBLASLT_EPILOGUE_BIAS` to fuse the bias add into
    /// the GEMM kernel — zero extra kernel launches, zero extra memory traffic.
    pub unsafe fn gemm_bias(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        bias: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(a.ndim(), 2);
        debug_assert_eq!(b.ndim(), 2);
        debug_assert_eq!(bias.ndim(), 1);
        debug_assert_eq!(a.dim(1), b.dim(1), "GEMM K mismatch");
        debug_assert_eq!(bias.dim(0), b.dim(0), "bias size must match N");

        let m = a.dim(0);
        let k = a.dim(1);
        let n = b.dim(0);
        let dtype = a.dtype();

        let out = alloc.alloc_tensor(&[m, n], dtype);
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias: true,
            weight_trans: true,
        };

        // Bias requires cublasLt (not supported by gemm_ex fallback).
        // ensure_plan + cublasLt path; shapes should be cached from warmup.
        self.ensure_plan(m, k, n, dtype, true, true);
        let plan = &self.plans[&key];

        let bias_ptr = bias.raw_ptr() as *const std::ffi::c_void;
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            plan.matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias_ptr as *const _ as *const _,
            std::mem::size_of::<*const std::ffi::c_void>(),
        ))
        .expect("set BIAS_POINTER failed");

        self.run_matmul_with_fallback(
            plan,
            b.as_ptr::<u8>() as *const _,
            a.as_ptr::<u8>() as *const _,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
        );

        out
    }

    /// GEMM without weight transpose: out = A @ B (no transpose on B).
    ///
    /// - `a`: `[M, K]` row-major (activations)
    /// - `b`: `[K, N]` row-major (weight, e.g. BNB dequanted W^T)
    /// - Returns: `[M, N]` allocated from `alloc`
    ///
    /// Used for BNB 4-bit: dequant produces W^T in `[K, N]` row-major layout,
    /// so we compute `x @ W^T` directly without an extra transpose.
    ///
    /// # Safety
    /// `a` and `b` must be valid GPU tensors with compatible dtypes and shapes.
    pub unsafe fn gemm_nt(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(a.ndim(), 2);
        debug_assert_eq!(b.ndim(), 2);
        debug_assert_eq!(a.dim(1), b.dim(0), "GEMM K mismatch (nt)");

        let m = a.dim(0);
        let k = a.dim(1); // = b.dim(0)
        let n = b.dim(1);
        let dtype = a.dtype();

        let out = alloc.alloc_tensor(&[m, n], dtype);
        let key = PlanKey {
            m,
            k,
            n,
            dtype,
            has_bias: false,
            weight_trans: false,
        };

        self.run_gemm(
            key,
            b.as_ptr::<u8>() as *const _,
            a.as_ptr::<u8>() as *const _,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
        );

        out
    }

    // -----------------------------------------------------------------------
    // FP8 GEMM: FP8 activations × FP8 weights → BF16 output with per-tensor scales
    // -----------------------------------------------------------------------

    /// Ensure a cached FP8 GEMM plan exists, creating it if needed.
    ///
    /// FP8 GEMM: A_fp8 [M,K] × B_fp8 [N,K]^T → C_bf16 [M,N]
    /// with per-tensor scale pointers set per-call.
    unsafe fn ensure_fp8_plan(&mut self, m: usize, k: usize, n: usize, output_dtype: DType) {
        let key = Fp8PlanKey {
            m,
            k,
            n,
            output_dtype,
        };
        if self.fp8_plans.contains_key(&key) {
            return;
        }

        let lt_compute = lt::cublasComputeType_t::CUBLAS_COMPUTE_32F;

        // Create matmul descriptor with F32 compute and F32 scale type.
        let mut matmul_desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatmulDescCreate(
            &mut matmul_desc,
            lt_compute,
            lt::cudaDataType_t::CUDA_R_32F, // scale type
        ))
        .expect("cublasLtMatmulDescCreate (FP8) failed");

        // TRANSA = T (weight [N,K] → col-major), TRANSB = N (activation [M,K]).
        let transa = cublasOperation_t::CUBLAS_OP_T as i32;
        let transb = cublasOperation_t::CUBLAS_OP_N as i32;
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &transa as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ))
        .expect("FP8: set TRANSA");
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &transb as *const _ as *const _,
            std::mem::size_of::<i32>(),
        ))
        .expect("FP8: set TRANSB");

        let lt_fp8 = lt::cudaDataType_t::CUDA_R_8F_E4M3;
        let lt_out = match output_dtype {
            DType::BF16 => lt::cudaDataType_t::CUDA_R_16BF,
            DType::F16 => lt::cudaDataType_t::CUDA_R_16F,
            _ => panic!("FP8 GEMM output must be BF16 or F16, got {output_dtype}"),
        };

        // A (cuBLAS) = weight [N,K] FP8, transposed → rows=K, cols=N, ld=K
        let mut layout_a: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_a,
            lt_fp8,
            k as u64,
            n as u64,
            k as i64,
        ))
        .expect("FP8: layout A");

        // B (cuBLAS) = activation [M,K] FP8, not transposed → rows=K, cols=M, ld=K
        let mut layout_b: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_b,
            lt_fp8,
            k as u64,
            m as u64,
            k as i64,
        ))
        .expect("FP8: layout B");

        // C/D = output [M,N] in output_dtype → rows=N, cols=M, ld=N
        let mut layout_c: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatrixLayoutCreate(
            &mut layout_c,
            lt_out,
            n as u64,
            m as u64,
            n as i64,
        ))
        .expect("FP8: layout C");

        // Get heuristic.
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        check_lt(lt::cublasLtMatmulPreferenceCreate(&mut pref)).expect("FP8: pref create");
        let ws_size = CUBLAS_WORKSPACE_SIZE;
        check_lt(lt::cublasLtMatmulPreferenceSetAttribute(
            pref,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws_size as *const _ as *const _,
            std::mem::size_of::<usize>(),
        ))
        .expect("FP8: set pref workspace");

        let mut heuristic = std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>();
        let mut algo_count: i32 = 0;
        check_lt(lt::cublasLtMatmulAlgoGetHeuristic(
            self.lt_handle,
            matmul_desc,
            layout_a,
            layout_b,
            layout_c,
            layout_c,
            pref,
            1,
            &mut heuristic,
            &mut algo_count,
        ))
        .expect("FP8: cublasLtMatmulAlgoGetHeuristic failed");
        assert!(algo_count > 0, "FP8: no cublasLt algorithm found");

        lt::cublasLtMatmulPreferenceDestroy(pref);

        self.fp8_plans.insert(
            key,
            Fp8GemmPlan {
                matmul_desc,
                layout_a,
                layout_b,
                layout_c,
                algo: heuristic.algo,
            },
        );
    }

    /// FP8 GEMM: out = A_fp8 @ B_fp8^T * (a_scale * b_scale)
    ///
    /// - `a`: `[M, K]` FP8 E4M3 (activations, dynamically quantized)
    /// - `b`: `[N, K]` FP8 E4M3 (weights)
    /// - `a_scale`: GPU pointer to f32 scalar (per-tensor activation scale)
    /// - `b_scale`: GPU pointer to f32 scalar (per-tensor weight scale)
    /// - `output_dtype`: BF16 or F16
    /// - Returns: `[M, N]` in output_dtype
    ///
    /// Uses cublasLt with `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` and
    /// `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER` for fused FP8→output dequantization.
    ///
    /// # Safety
    /// All pointers must be valid GPU memory. Scale pointers must point to valid f32 scalars.
    pub unsafe fn gemm_fp8(
        &mut self,
        a: GpuTensor,
        b: GpuTensor,
        a_scale: *const f32,
        b_scale: *const f32,
        output_dtype: DType,
        alloc: &mut CachingAllocator,
    ) -> OwnedTensor {
        debug_assert_eq!(a.ndim(), 2);
        debug_assert_eq!(b.ndim(), 2);
        debug_assert_eq!(a.dtype(), DType::Fp8E4m3);
        debug_assert_eq!(b.dtype(), DType::Fp8E4m3);
        debug_assert_eq!(a.dim(1), b.dim(1), "FP8 GEMM K mismatch");

        let m = a.dim(0);
        let k = a.dim(1);
        let n = b.dim(0);

        let out = alloc.alloc_tensor(&[m, n], output_dtype);

        self.ensure_fp8_plan(m, k, n, output_dtype);
        let key = Fp8PlanKey {
            m,
            k,
            n,
            output_dtype,
        };
        let plan = &self.fp8_plans[&key];

        // Set per-call scale pointers on the matmul descriptor.
        // These are GPU pointers to f32 scalars.
        let a_scale_ptr = a_scale as *const std::ffi::c_void;
        let b_scale_ptr = b_scale as *const std::ffi::c_void;
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            plan.matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &b_scale_ptr as *const _ as *const _, // cuBLAS A = our weight (b)
            std::mem::size_of::<*const std::ffi::c_void>(),
        ))
        .expect("FP8: set A_SCALE_POINTER");
        check_lt(lt::cublasLtMatmulDescSetAttribute(
            plan.matmul_desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &a_scale_ptr as *const _ as *const _, // cuBLAS B = our activation (a)
            std::mem::size_of::<*const std::ffi::c_void>(),
        ))
        .expect("FP8: set B_SCALE_POINTER");

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        check_lt(lt::cublasLtMatmul(
            self.lt_handle,
            plan.matmul_desc,
            &alpha as *const f32 as *const _,
            b.as_ptr::<u8>() as *const _, // cuBLAS A = weight
            plan.layout_a,
            a.as_ptr::<u8>() as *const _, // cuBLAS B = activation
            plan.layout_b,
            &beta as *const f32 as *const _,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
            plan.layout_c,
            out.as_gpu_tensor().as_mut_ptr::<u8>() as *mut _,
            plan.layout_c,
            &plan.algo,
            self.workspace as *mut _,
            CUBLAS_WORKSPACE_SIZE,
            self.stream as _,
        ))
        .expect("FP8 cublasLtMatmul failed");

        out
    }

    /// Benchmark all cached GEMM plans: for each shape, try the top N
    /// algorithms from the heuristic and keep the fastest.
    ///
    /// Call this during warmup after the first forward pass has populated
    /// the plan cache with all model GEMM shapes.
    ///
    /// # Safety
    /// Requires active CUDA context. Arena must have enough space for the
    /// largest GEMM output.
    pub unsafe fn benchmark_plans(&mut self) {
        use crate::driver;

        let keys: Vec<PlanKey> = self.plans.keys().copied().collect();
        if keys.is_empty() {
            return;
        }

        tracing::info!(
            "Benchmarking {} cublasLt GEMM plans ({} warmup + {} timed iters)...",
            keys.len(),
            3,
            10
        );

        let (event_start, event_end) = match (driver::event_create(), driver::event_create()) {
            (Ok(s), Ok(e)) => (s, e),
            _ => {
                tracing::warn!("Failed to create CUDA events for benchmarking, skipping");
                return;
            }
        };

        for key in &keys {
            // Allocate scratch buffers for benchmarking.
            let a_bytes = key.m * key.k * key.dtype.size_bytes();
            let b_bytes = key.n * key.k * key.dtype.size_bytes();
            let c_bytes = key.m * key.n * key.dtype.size_bytes();
            let total = a_bytes + b_bytes + c_bytes;

            // Use temp allocations (not arena, which may be sized for the model).
            let Ok(buf) = driver::mem_alloc(total) else {
                continue;
            };
            let a_ptr = buf;
            let b_ptr = unsafe { buf.add(a_bytes) };
            let c_ptr = unsafe { buf.add(a_bytes + b_bytes) };
            let _ = driver::memset_d8(buf, 0, total, self.stream);

            // Get the plan's descriptors.
            let plan = &self.plans[key];
            let matmul_desc = plan.matmul_desc;
            let layout_a = plan.layout_a;
            let layout_b = plan.layout_b;
            let layout_c = plan.layout_c;

            // Get top N algorithms.
            let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
            if check_lt(lt::cublasLtMatmulPreferenceCreate(&mut pref)).is_err() {
                let _ = driver::mem_free(buf);
                continue;
            }
            let ws_size = CUBLAS_WORKSPACE_SIZE;
            let _ = check_lt(lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_size as *const _ as *const _,
                std::mem::size_of::<usize>(),
            ));

            const MAX_ALGOS: usize = 8;
            let mut heuristics =
                [std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>(); MAX_ALGOS];
            let mut algo_count: i32 = 0;
            let _ = lt::cublasLtMatmulAlgoGetHeuristic(
                self.lt_handle,
                matmul_desc,
                layout_a,
                layout_b,
                layout_c,
                layout_c,
                pref,
                MAX_ALGOS as i32,
                heuristics.as_mut_ptr(),
                &mut algo_count,
            );
            lt::cublasLtMatmulPreferenceDestroy(pref);

            if algo_count <= 1 {
                // Only one algorithm available, nothing to benchmark.
                let _ = driver::mem_free(buf);
                continue;
            }

            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            let mut best_time = f32::MAX;
            let mut best_algo_idx = 0usize;
            let mut heuristic_time = f32::MAX; // time for algo 0 (heuristic pick)

            for (ai, heuristic) in heuristics.iter().enumerate().take(algo_count as usize) {
                let algo = &heuristic.algo;

                // Warmup.
                for _ in 0..3 {
                    let _ = lt::cublasLtMatmul(
                        self.lt_handle,
                        matmul_desc,
                        &alpha as *const f32 as *const _,
                        b_ptr as *const _,
                        layout_a,
                        a_ptr as *const _,
                        layout_b,
                        &beta as *const f32 as *const _,
                        c_ptr as *mut _,
                        layout_c,
                        c_ptr as *mut _,
                        layout_c,
                        algo,
                        self.workspace as *mut _,
                        CUBLAS_WORKSPACE_SIZE,
                        self.stream as _,
                    );
                }

                // Timed iterations.
                let _ = driver::event_record(event_start, self.stream);
                for _ in 0..10 {
                    let _ = lt::cublasLtMatmul(
                        self.lt_handle,
                        matmul_desc,
                        &alpha as *const f32 as *const _,
                        b_ptr as *const _,
                        layout_a,
                        a_ptr as *const _,
                        layout_b,
                        &beta as *const f32 as *const _,
                        c_ptr as *mut _,
                        layout_c,
                        c_ptr as *mut _,
                        layout_c,
                        algo,
                        self.workspace as *mut _,
                        CUBLAS_WORKSPACE_SIZE,
                        self.stream as _,
                    );
                }
                let _ = driver::event_record(event_end, self.stream);
                let _ = driver::stream_synchronize(self.stream);

                if let Ok(elapsed) = driver::event_elapsed(event_start, event_end) {
                    let avg = elapsed / 10.0;
                    if ai == 0 {
                        heuristic_time = avg;
                    }
                    if avg < best_time {
                        best_time = avg;
                        best_algo_idx = ai;
                    }
                }
            }

            // Update the plan: move the best algorithm to the front.
            if best_algo_idx != 0 {
                let plan = self.plans.get_mut(key).unwrap();
                plan.algos.swap(0, best_algo_idx);
                let speedup = (1.0 - best_time / heuristic_time) * 100.0;
                tracing::info!(
                    "  GEMM M={} K={} N={}: algo #{} is {:.1}% faster ({:.3}ms vs {:.3}ms)",
                    key.m,
                    key.k,
                    key.n,
                    best_algo_idx,
                    speedup,
                    best_time,
                    heuristic_time,
                );
            }

            let _ = driver::mem_free(buf);
        }

        let _ = driver::event_destroy(event_start);
        let _ = driver::event_destroy(event_end);

        tracing::info!("cublasLt plan benchmarking complete");
    }

    /// Raw cuBLAS handle (for advanced usage).
    pub fn raw_handle(&self) -> cublasHandle_t {
        self.handle
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        // Drop all cached plans first (they reference the lt_handle indirectly).
        self.plans.clear();
        self.fp8_plans.clear();
        unsafe {
            let _ = lt::cublasLtDestroy(self.lt_handle);
            let _ = sys::cublasDestroy_v2(self.handle);
            // workspace is owned by the CachingAllocator — freed when
            // the allocator is dropped or release_all() is called.
        }
    }
}

/// Map our DType to cuBLAS compute and data types.
fn gemm_types(dtype: DType) -> (cublasComputeType_t, sys::cudaDataType_t) {
    match dtype {
        DType::F16 => (
            // F16 uses F32 accumulation (matches Python vLLM / PyTorch default).
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_16F,
        ),
        DType::BF16 => (
            // BF16 uses F32 accumulation for numerical stability.
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_16BF,
        ),
        DType::F32 => (
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            sys::cudaDataType_t::CUDA_R_32F,
        ),
        _ => panic!("unsupported dtype for GEMM: {dtype}"),
    }
}

fn check(status: sys::cublasStatus_t) -> Result<()> {
    if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("cuBLAS error: {:?}", status);
    }
    Ok(())
}

fn check_lt(status: lt::cublasStatus_t) -> Result<()> {
    if status != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        bail!("cublasLt error: {:?}", status);
    }
    Ok(())
}

/// Convert cuBLAS data type to cublasLt data type (same enum values, different Rust types).
fn cublas_to_lt_dtype(dt: sys::cudaDataType_t) -> lt::cudaDataType_t {
    // The enum values match between cuBLAS and cublasLt bindings but are
    // separate Rust types in cudarc. Transmute is correct here.
    match dt {
        sys::cudaDataType_t::CUDA_R_16F => lt::cudaDataType_t::CUDA_R_16F,
        sys::cudaDataType_t::CUDA_R_16BF => lt::cudaDataType_t::CUDA_R_16BF,
        sys::cudaDataType_t::CUDA_R_32F => lt::cudaDataType_t::CUDA_R_32F,
        _ => panic!("unsupported dtype for cublasLt: {:?}", dt),
    }
}

/// Convert cuBLAS compute type to cublasLt compute type (same enum values, different Rust types).
fn cublas_to_lt_compute(ct: cublasComputeType_t) -> lt::cublasComputeType_t {
    match ct {
        cublasComputeType_t::CUBLAS_COMPUTE_16F => lt::cublasComputeType_t::CUBLAS_COMPUTE_16F,
        cublasComputeType_t::CUBLAS_COMPUTE_32F => lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 => {
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        }
        _ => panic!("unsupported compute type for cublasLt: {:?}", ct),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver;

    fn init_cuda() -> CUstream {
        unsafe {
            driver::init().expect("CUDA init");
            let dev = driver::device_get(0).expect("device");
            let _ctx = driver::ctx_create(dev).expect("context");
            driver::stream_create().expect("stream")
        }
    }

    #[test]
    fn test_gemm_types_f16() {
        let (compute, data) = gemm_types(DType::F16);
        assert_eq!(compute, cublasComputeType_t::CUBLAS_COMPUTE_32F);
        assert_eq!(data, sys::cudaDataType_t::CUDA_R_16F);
    }

    #[test]
    fn test_gemm_types_bf16() {
        let (compute, data) = gemm_types(DType::BF16);
        assert_eq!(compute, cublasComputeType_t::CUBLAS_COMPUTE_32F);
        assert_eq!(data, sys::cudaDataType_t::CUDA_R_16BF);
    }

    #[test]
    fn test_gemm_types_f32() {
        let (compute, data) = gemm_types(DType::F32);
        assert_eq!(compute, cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32);
        assert_eq!(data, sys::cudaDataType_t::CUDA_R_32F);
    }

    #[test]
    #[should_panic(expected = "unsupported dtype for GEMM")]
    fn test_gemm_types_u32_panics() {
        gemm_types(DType::U32);
    }

    #[test]
    #[should_panic(expected = "unsupported dtype for GEMM")]
    fn test_gemm_types_i64_panics() {
        gemm_types(DType::I64);
    }

    #[test]
    fn test_cublas_create_and_drop() {
        let stream = init_cuda();
        let handle = unsafe { CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap() };
        assert!(!handle.raw_handle().is_null());
        drop(handle);
        unsafe { driver::stream_destroy(stream).unwrap() };
    }

    #[test]
    fn test_gemm_f32_identity() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [2, 3], B = identity-like [3, 3]
            // A @ B^T should give A when B = I (but B is [N,K] so B^T = I^T = I).
            let m = 2usize;
            let k = 3usize;
            let n = 3usize;

            // Prepare A on host: [[1, 2, 3], [4, 5, 6]]
            let host_a = driver::mem_alloc_host(m * k * 4).unwrap();
            let a_slice = std::slice::from_raw_parts_mut(host_a as *mut f32, m * k);
            a_slice.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

            // Prepare B (identity) on host: [[1,0,0],[0,1,0],[0,0,1]]
            let host_b = driver::mem_alloc_host(n * k * 4).unwrap();
            let b_slice = std::slice::from_raw_parts_mut(host_b as *mut f32, n * k);
            b_slice.copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

            // Upload to GPU.
            let gpu_a = driver::mem_alloc(m * k * 4).unwrap();
            let gpu_b = driver::mem_alloc(n * k * 4).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, m * k * 4, stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, n * k * 4, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[m, k], DType::F32);
            let b = GpuTensor::new(gpu_b, &[n, k], DType::F32);

            let c = handle.gemm(a, b, &mut arena);
            assert_eq!(c.dim(0), m);
            assert_eq!(c.dim(1), n);

            // Read back.
            let host_c = driver::mem_alloc_host(m * n * 4).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), m * n * 4, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let c_slice = std::slice::from_raw_parts(host_c as *const f32, m * n);
            // A @ I = A → [[1,2,3],[4,5,6]]
            let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            for (i, (got, exp)) in c_slice.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "mismatch at {i}: got {got}, expected {exp}"
                );
            }

            // Cleanup.
            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_f32_known_values() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [[1, 2], [3, 4]] (2x2)
            // B = [[5, 6], [7, 8]] (2x2)
            // C = A @ B^T = [[1*5+2*6, 1*7+2*8], [3*5+4*6, 3*7+4*8]]
            //             = [[17, 23], [39, 53]]
            let host_a = driver::mem_alloc_host(16).unwrap();
            let host_b = driver::mem_alloc_host(16).unwrap();
            std::slice::from_raw_parts_mut(host_a as *mut f32, 4)
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            std::slice::from_raw_parts_mut(host_b as *mut f32, 4)
                .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

            let gpu_a = driver::mem_alloc(16).unwrap();
            let gpu_b = driver::mem_alloc(16).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, 16, stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, 16, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 2], DType::F32);
            let b = GpuTensor::new(gpu_b, &[2, 2], DType::F32);
            let c = handle.gemm(a, b, &mut arena);

            let host_c = driver::mem_alloc_host(16).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), 16, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, 4);
            let expected = [17.0, 23.0, 39.0, 53.0];
            for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "mismatch at {i}: got {got}, expected {exp}"
                );
            }

            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_non_square() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [4, 8], B = [16, 8] → C = [4, 16]
            let m = 4usize;
            let k = 8usize;
            let n = 16usize;

            let gpu_a = driver::mem_alloc(m * k * 4).unwrap();
            let gpu_b = driver::mem_alloc(n * k * 4).unwrap();

            // Fill with zeros (result should be all zeros).
            driver::memset_d8(gpu_a, 0, m * k * 4, stream).unwrap();
            driver::memset_d8(gpu_b, 0, n * k * 4, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[m, k], DType::F32);
            let b = GpuTensor::new(gpu_b, &[n, k], DType::F32);
            let c = handle.gemm(a, b, &mut arena);

            assert_eq!(c.dim(0), m);
            assert_eq!(c.dim(1), n);

            // Verify output shape and that it's all zeros.
            let host_c = driver::mem_alloc_host(m * n * 4).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), m * n * 4, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, m * n);
            for (i, v) in result.iter().enumerate() {
                assert!(v.abs() < 1e-6, "expected 0 at {i}, got {v}");
            }

            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_output_from_arena() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            let mut arena = CachingAllocator::new();

            let gpu_a = driver::mem_alloc(64).unwrap();
            let gpu_b = driver::mem_alloc(64).unwrap();
            driver::memset_d8(gpu_a, 0, 64, stream).unwrap();
            driver::memset_d8(gpu_b, 0, 64, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 8], DType::F32);
            let b = GpuTensor::new(gpu_b, &[4, 8], DType::F32);

            let blocks_before = arena.total_block_count();
            let c = handle.gemm(a, b, &mut arena);
            let blocks_after = arena.total_block_count();

            // Should have allocated a new block for the output.
            assert!(blocks_after > blocks_before);
            assert_eq!(c.numel(), 8); // 2 * 4

            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_bias_f32() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            let mut arena = CachingAllocator::new();

            // A = [[1, 2], [3, 4]] (2x2)
            // B = [[5, 6], [7, 8]] (2x2)  (weight)
            // bias = [10, 20]
            // C = A @ B^T + bias = [[17+10, 23+20], [39+10, 53+20]] = [[27, 43], [49, 73]]
            let host_a = driver::mem_alloc_host(16).unwrap();
            let host_b = driver::mem_alloc_host(16).unwrap();
            let host_bias = driver::mem_alloc_host(8).unwrap();
            std::slice::from_raw_parts_mut(host_a as *mut f32, 4)
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            std::slice::from_raw_parts_mut(host_b as *mut f32, 4)
                .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
            std::slice::from_raw_parts_mut(host_bias as *mut f32, 2).copy_from_slice(&[10.0, 20.0]);

            let gpu_a = driver::mem_alloc(16).unwrap();
            let gpu_b = driver::mem_alloc(16).unwrap();
            let gpu_bias = driver::mem_alloc(8).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, 16, stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, 16, stream).unwrap();
            driver::memcpy_htod_async(gpu_bias, host_bias, 8, stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 2], DType::F32);
            let b = GpuTensor::new(gpu_b, &[2, 2], DType::F32);
            let bias = GpuTensor::new(gpu_bias, &[2], DType::F32);
            let c = handle.gemm_bias(a, b, bias, &mut arena);

            let host_c = driver::mem_alloc_host(16).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), 16, stream).unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, 4);
            let expected = [27.0, 43.0, 49.0, 73.0];
            for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "gemm_bias mismatch at {i}: got {got}, expected {exp}"
                );
            }

            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_bias).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
            driver::mem_free(gpu_bias).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_plan_caching() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            assert_eq!(handle.plans.len(), 0);

            // First call creates a plan.
            handle.ensure_plan(8, 896, 896, DType::BF16, false, true);
            assert_eq!(handle.plans.len(), 1);

            // Second call with same shapes reuses it.
            handle.ensure_plan(8, 896, 896, DType::BF16, false, true);
            assert_eq!(handle.plans.len(), 1);

            // Different shapes create a new plan.
            handle.ensure_plan(8, 896, 4864, DType::BF16, false, true);
            assert_eq!(handle.plans.len(), 2);

            // Same shapes but with bias create a separate plan.
            handle.ensure_plan(8, 896, 896, DType::BF16, true, true);
            assert_eq!(handle.plans.len(), 3);

            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_fp8_plan_caching() {
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            assert_eq!(handle.fp8_plans.len(), 0);

            // First call creates an FP8 plan.
            handle.ensure_fp8_plan(8, 896, 896, DType::BF16);
            assert_eq!(handle.fp8_plans.len(), 1);

            // Second call with same shapes reuses it.
            handle.ensure_fp8_plan(8, 896, 896, DType::BF16);
            assert_eq!(handle.fp8_plans.len(), 1);

            // Different shapes create a new plan.
            handle.ensure_fp8_plan(8, 896, 4864, DType::BF16);
            assert_eq!(handle.fp8_plans.len(), 2);

            // Different output dtype creates a separate plan.
            handle.ensure_fp8_plan(8, 896, 896, DType::F16);
            assert_eq!(handle.fp8_plans.len(), 3);

            driver::stream_destroy(stream).unwrap();
        }
    }

    #[test]
    fn test_gemm_fp8_identity() {
        // FP8 GEMM with identity-like matrices.
        // This test verifies the FP8 GEMM path works end-to-end.
        // A = FP8 [M, K], B = FP8 [N, K], output = BF16 [M, N]
        let stream = init_cuda();
        unsafe {
            let mut handle = CublasHandle::new(stream, &mut CachingAllocator::new()).unwrap();
            let mut alloc = crate::alloc::CachingAllocator::new();

            // FP8 cublasLt requires 16-aligned dimensions on SM89.
            let m = 16;
            let k = 32;
            let n = 16;

            // Create FP8 E4M3 data (all 1.0 = 0x38 in FP8 E4M3)
            let a_data = vec![0x38u8; m * k]; // all 1.0
            let b_data = vec![0x38u8; n * k]; // all 1.0

            // Upload to GPU
            let a_ptr = driver::mem_alloc(m * k).unwrap();
            let b_ptr = driver::mem_alloc(n * k).unwrap();
            driver::memcpy_htod_async(a_ptr, a_data.as_ptr(), m * k, stream).unwrap();
            driver::memcpy_htod_async(b_ptr, b_data.as_ptr(), n * k, stream).unwrap();

            let a = GpuTensor::new(a_ptr, &[m, k], DType::Fp8E4m3);
            let b = GpuTensor::new(b_ptr, &[n, k], DType::Fp8E4m3);

            // Scales = 1.0 (identity scaling)
            let scale_val = 1.0f32;
            let scale_ptr = driver::mem_alloc(4).unwrap();
            driver::memcpy_htod_async(scale_ptr, &scale_val as *const f32 as *const u8, 4, stream)
                .unwrap();

            let output = handle.gemm_fp8(
                a,
                b,
                scale_ptr as *const f32,
                scale_ptr as *const f32,
                DType::BF16,
                &mut alloc,
            );

            // Read output: each element should be K (=8) since dot(ones, ones) = K
            let nbytes = m * n * DType::BF16.size_bytes();
            let host = driver::mem_alloc_host(nbytes).unwrap();
            driver::memcpy_dtoh_async(host, output.as_gpu_tensor().raw_ptr(), nbytes, stream)
                .unwrap();
            driver::stream_synchronize(stream).unwrap();

            let result = std::slice::from_raw_parts(host as *const half::bf16, m * n);
            for val in result {
                let f = val.to_f32();
                assert!((f - k as f32).abs() < 0.5, "expected ~{k}, got {f}");
            }

            driver::mem_free_host(host).unwrap();
            driver::mem_free(a_ptr).unwrap();
            driver::mem_free(b_ptr).unwrap();
            driver::mem_free(scale_ptr as *mut u8).unwrap();
            driver::stream_destroy(stream).unwrap();
        }
    }
}
