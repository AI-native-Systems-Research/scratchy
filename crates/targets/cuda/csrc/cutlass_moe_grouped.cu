/*
 * CUTLASS grouped MoE GEMM — arch-optimal expert GEMM for gemma4-moe.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Replaces the legacy `nvcuda::wmma` `fused_moe_gemm` (identical on every arch)
 * with a CUTLASS grouped GEMM that picks the native tensor-core datapath:
 *   - Sm90 (Hopper): wgmma + TMA, ptr-array warp-specialized cooperative
 *     (CUTLASS 3.x CollectiveBuilder — the proven /build/.cutlass_probe config).
 *   - Sm80 (used on sm89/Ada): mma.sync multistage grouped GEMM
 *     (CUTLASS 2.x cutlass::gemm::device::GemmGrouped, arch::Sm80).
 *
 * The experts are dequant'd int4 -> bf16 at load, so this is a BF16 grouped GEMM:
 *   C[m, N] = A[m, K] @ B_e[N, K]^T  for the expert e that owns row m.
 *
 * Layouts (match the legacy fused_moe_gemm + moe_align_block_size contract):
 *   A_permuted: RowMajor [total_padded_rows, K]   (gathered, expert-grouped)
 *   weights B:  [E, N, K] row-major  -> per-expert [N, K] row-major
 *               = ColumnMajor [K, N]  (CUTLASS LayoutB = ColumnMajor)
 *   C_permuted: RowMajor [total_padded_rows, N]
 *
 * Both arch launchers are always defined so the Rust FFI links on any build
 * host; the lib is single-arch-built (CUDA_COMPUTE_CAP), so only the matching
 * arch's device code is emitted. Arch dispatch happens in Rust via
 * device_sm_version(). The Sm90 3.x body is gated on CUDA >= 12.0.
 *
 * Numerics caveat: ptxas/SASS proves the tensor-core core compiles, NOT that
 * the permute/marshal/scatter indexing is numerically correct — that needs a
 * GPU. CUTLASS's GEMM core is NVIDIA-validated; residual risk is the (CPU-
 * checkable) glue kernels below.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────
// Glue kernels: permute (gather A), arg-marshal (per-group ptr/problem
// arrays), and scatter (write grouped C back to token-indexed output).
//
// These are arch-independent and read the moe_align_block_size output exactly
// like the legacy fused_moe_gemm_bf16_wmma kernel does:
//   - sorted_token_ids[p]  : token slot for padded row p (>= num_valid means pad)
//   - expert_ids[blk]      : expert owning M-block `blk` (-1 => empty block)
//   - num_tokens_post_padded[0] : total padded rows in use
// ─────────────────────────────────────────────────────────────────────────

namespace scratchy_moe_grouped {

// Gather the contiguous, expert-grouped A_permuted from `input`.
//   A_permuted[p, :] = input[sorted_token_ids[p] / top_k, :]   (zero if pad)
// `sorted_token_ids` is already expert-sorted by moe_align_block_size, so
// A_permuted is contiguous-per-expert with no extra reordering.
__global__ void permute_gather_a_kernel(
    __nv_bfloat16* __restrict__ a_permuted,        // [total_padded_rows, K]
    const __nv_bfloat16* __restrict__ input,        // [num_tokens, K]
    const int32_t* __restrict__ sorted_token_ids,   // [total_padded_rows]
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,                       // num_tokens * top_k
    int32_t top_k,
    int32_t K)
{
    const int32_t total_padded = *num_tokens_post_padded;
    const int64_t row = blockIdx.x;
    if (row >= total_padded) return;

    const int32_t token_id = sorted_token_ids[row];
    __nv_bfloat16* dst = a_permuted + row * (int64_t)K;

    if (token_id >= num_valid_tokens) {
        // Padding row — zero so it never pollutes a real expert's output.
        for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
            dst[k] = __float2bfloat16(0.0f);
        }
        return;
    }
    const __nv_bfloat16* src = input + (int64_t)(token_id / top_k) * K;
    for (int32_t k = threadIdx.x; k < K; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// Build the per-group (per-expert) ptr/problem/stride arrays.
//
// Walks `expert_ids` (one entry per M-block). Each expert e owns a contiguous
// run of `block_m`-sized M-blocks (moe_align groups them). For each expert we
// emit one problem {M_e, N, K} where M_e = (#blocks for e) * block_m, plus the
// A/B/C base pointers and the row strides.
//
// Single-thread kernel (E is small — gemma4-moe has 128 experts max; the scan
// is O(num_m_blocks) and runs once per GEMM). Writes the int strides for the
// Sm80 GemmGrouped path; the Sm90 path packs cute strides from these in Rust-
// free device code via a second tiny fill below.
//
// problem_sizes layout: int32[3*E] = {M_0,N,K, M_1,N,K, ...}
// lda/ldb/ldc:          int64[E]
// ptr_A/ptr_B/ptr_C:    void*[E]
__global__ void marshal_groups_kernel(
    int32_t* __restrict__ problem_sizes,   // [3*E]  {M_e, N, K}
    void** __restrict__ ptr_a,             // [E]
    void** __restrict__ ptr_b,             // [E]
    void** __restrict__ ptr_c,             // [E]
    int64_t* __restrict__ lda,             // [E]  = K
    int64_t* __restrict__ ldb,             // [E]  = K  (col-major [K,N] => ldb=K)
    int64_t* __restrict__ ldc,             // [E]  = N
    const int32_t* __restrict__ expert_ids,        // [num_m_blocks]
    const int32_t* __restrict__ num_tokens_post_padded,
    __nv_bfloat16* __restrict__ a_permuted,        // [total_padded_rows, K]
    const __nv_bfloat16* __restrict__ weights,      // [E, N, K]
    __nv_bfloat16* __restrict__ c_permuted,         // [total_padded_rows, N]
    int32_t num_experts,
    int32_t block_m,
    int32_t N,
    int32_t K)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_m_blocks = total_padded / block_m;

    // Count blocks per expert (expert_ids is non-decreasing over valid blocks).
    // Row offset for expert e = (#padded rows before its first block).
    int64_t row_off = 0;  // running A/C row offset (in rows)
    int32_t blk = 0;
    for (int32_t e = 0; e < num_experts; ++e) {
        int32_t nblk = 0;
        while (blk < num_m_blocks && expert_ids[blk] == e) {
            ++nblk;
            ++blk;
        }
        const int32_t M_e = nblk * block_m;
        problem_sizes[3 * e + 0] = M_e;
        problem_sizes[3 * e + 1] = N;
        problem_sizes[3 * e + 2] = K;
        ptr_a[e] = a_permuted + row_off * (int64_t)K;
        ptr_b[e] = const_cast<__nv_bfloat16*>(weights) + (int64_t)e * N * K;
        ptr_c[e] = c_permuted + row_off * (int64_t)N;
        lda[e] = K;          // A RowMajor [M_e, K]
        ldb[e] = K;          // B ColumnMajor [K, N] => leading dim = K
        ldc[e] = N;          // C RowMajor [M_e, N]
        row_off += M_e;
    }
    // Any expert_ids == -1 (empty padding blocks) past the valid run are left
    // out of the per-expert M counts naturally: the while-loop only advances on
    // an exact expert-id match, and moe_align emits experts in 0..E order.
}

// Scatter the grouped C_permuted back into the token-indexed output, exactly
// like the legacy fused_moe_gemm epilogue:
//   output[token_id, n] = (apply_weights ? weight*val : val)
// Padding rows (token_id >= num_valid) are skipped.
__global__ void scatter_c_kernel(
    __nv_bfloat16* __restrict__ output,             // [num_valid_tokens, N]
    const __nv_bfloat16* __restrict__ c_permuted,   // [total_padded_rows, N]
    const float* __restrict__ topk_weights,         // [num_valid_tokens] or null
    const int32_t* __restrict__ sorted_token_ids,   // [total_padded_rows]
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t apply_weights,
    int32_t N)
{
    const int32_t total_padded = *num_tokens_post_padded;
    const int64_t row = blockIdx.x;
    if (row >= total_padded) return;

    const int32_t token_id = sorted_token_ids[row];
    if (token_id >= num_valid_tokens) return;

    const __nv_bfloat16* src = c_permuted + row * (int64_t)N;
    __nv_bfloat16* dst = output + (int64_t)token_id * N;
    const float w = apply_weights ? topk_weights[token_id] : 1.0f;

    for (int32_t n = threadIdx.x; n < N; n += blockDim.x) {
        float v = __bfloat162float(src[n]);
        if (apply_weights) v *= w;
        dst[n] = __float2bfloat16(v);
    }
}

} // namespace scratchy_moe_grouped

// ─────────────────────────────────────────────────────────────────────────
// extern "C" glue launchers (arch-independent).
// ─────────────────────────────────────────────────────────────────────────

extern "C" void scratchy_moe_permute_gather_a(
    void* a_permuted, const void* input,
    const int32_t* sorted_token_ids, const int32_t* num_tokens_post_padded,
    int32_t total_padded_rows_cap, int32_t num_valid_tokens, int32_t top_k, int32_t K,
    uint64_t stream)
{
    const int threads = 256;
    // grid sized to the max padded rows; the kernel early-outs past the real
    // total_tokens_post_padded read from device.
    scratchy_moe_grouped::permute_gather_a_kernel<<<total_padded_rows_cap, threads, 0,
        (cudaStream_t)stream>>>(
        (__nv_bfloat16*)a_permuted, (const __nv_bfloat16*)input,
        sorted_token_ids, num_tokens_post_padded,
        num_valid_tokens, top_k, K);
}

extern "C" void scratchy_moe_marshal_groups(
    int32_t* problem_sizes, void** ptr_a, void** ptr_b, void** ptr_c,
    int64_t* lda, int64_t* ldb, int64_t* ldc,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    void* a_permuted, const void* weights, void* c_permuted,
    int32_t num_experts, int32_t block_m, int32_t N, int32_t K,
    uint64_t stream)
{
    scratchy_moe_grouped::marshal_groups_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        problem_sizes, ptr_a, ptr_b, ptr_c, lda, ldb, ldc,
        expert_ids, num_tokens_post_padded,
        (__nv_bfloat16*)a_permuted, (const __nv_bfloat16*)weights,
        (__nv_bfloat16*)c_permuted,
        num_experts, block_m, N, K);
}

extern "C" void scratchy_moe_scatter_c(
    void* output, const void* c_permuted, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* num_tokens_post_padded,
    int32_t total_padded_rows_cap, int32_t num_valid_tokens, int32_t apply_weights, int32_t N,
    uint64_t stream)
{
    const int threads = 256;
    scratchy_moe_grouped::scatter_c_kernel<<<total_padded_rows_cap, threads, 0,
        (cudaStream_t)stream>>>(
        (__nv_bfloat16*)output, (const __nv_bfloat16*)c_permuted, topk_weights,
        sorted_token_ids, num_tokens_post_padded,
        num_valid_tokens, apply_weights, N);
}

// ═════════════════════════════════════════════════════════════════════════
// Sm80 grouped GEMM (CUTLASS 2.x) — runs on sm89/Ada via arch::Sm80.
// ═════════════════════════════════════════════════════════════════════════

#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace sm80_moe {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

// A RowMajor [M,K], B ColumnMajor [K,N] (= RowMajor [N,K]), C RowMajor [M,N].
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

static constexpr int kAlignmentA = 8;   // 8 bf16 = 16B
static constexpr int kAlignmentB = 8;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, kAlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, kAlignmentB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,   // threadblock
    cutlass::gemm::GemmShape<64, 64, 32>,     // warp
    cutlass::gemm::GemmShape<16, 8, 16>,      // mma instruction (bf16 tensor core)
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,  // stages
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

} // namespace sm80_moe

// Sm80 grouped launcher. Takes the device per-group arrays built by the marshal
// kernel. problem_sizes is GemmCoord[E] ({M,N,K}); the int32[3*E] from the
// marshal kernel is bit-compatible with cutlass::gemm::GemmCoord (three ints).
extern "C" int cutlass_moe_grouped_bf16_sm80(
    const void* problem_sizes,   // GemmCoord[E] device ptr (== int32[3*E])
    void** ptr_a, void** ptr_b, void** ptr_c,
    int64_t* lda, int64_t* ldb, int64_t* ldc,
    int32_t num_experts,
    int32_t threadblock_count,
    void* workspace,
    uint64_t stream)
{
    using Gemm = sm80_moe::GemmGrouped;
    using ElementA = sm80_moe::ElementA;
    using ElementB = sm80_moe::ElementB;
    using ElementC = sm80_moe::ElementC;
    using ElementAccumulator = sm80_moe::ElementAccumulator;

    typename Gemm::EpilogueOutputOp::Params epilogue_op(
        ElementAccumulator(1.0f), ElementAccumulator(0.0f));

    typename Gemm::Arguments args(
        const_cast<cutlass::gemm::GemmCoord*>(
            reinterpret_cast<const cutlass::gemm::GemmCoord*>(problem_sizes)),
        num_experts,
        threadblock_count,
        epilogue_op,
        reinterpret_cast<ElementA**>(ptr_a),
        reinterpret_cast<ElementB**>(ptr_b),
        reinterpret_cast<ElementC**>(ptr_c),
        reinterpret_cast<ElementC**>(ptr_c),   // D = C in-place (beta=0)
        lda, ldb, ldc, ldc,
        nullptr   // host problem_sizes unavailable — device-only schedule
    );

    Gemm op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    status = op.initialize(args, workspace, (cudaStream_t)stream);
    if (status != cutlass::Status::kSuccess) return -2;
    status = op.run((cudaStream_t)stream);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

// Query the grid size CUTLASS wants for the Sm80 grouped GEMM (occupancy-based).
// Called once from Rust; problem_sizes_host may be null (device-only schedule
// still returns a valid count from the kernel's max active blocks).
extern "C" int cutlass_moe_grouped_bf16_sm80_threadblock_count() {
    // Mirror example 24: sufficient() with null problem list returns the
    // occupancy-limited block count (independent of the specific problems).
    return sm80_moe::GemmGrouped::sufficient(nullptr, 0);
}

// ═════════════════════════════════════════════════════════════════════════
// Sm90 grouped GEMM (CUTLASS 3.x) — wgmma + TMA, ptr-array warp-specialized.
// Verbatim from /build/.cutlass_probe/grouped_probe.cu (proven to emit HGMMA).
// Gated on CUDA >= 12 (CUTLASS 3.x requirement); the lib is sm_90a-built so the
// device code is only emitted on Hopper hosts.
// ═════════════════════════════════════════════════════════════════════════

#if (__CUDACC_VER_MAJOR__ >= 12)

#include <cute/tensor.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/util/packed_stride.hpp>

namespace sm90_moe {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC*, AlignmentC,
    ElementC, LayoutC*, AlignmentC,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA*, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

} // namespace sm90_moe

// Fill the Sm90 cute packed-stride arrays from the per-group problem sizes.
// One thread per group: stride_A[e] = packed_stride({M_e, K, 1}), etc.
// (The Sm90 ptr-array kernel wants cute strides, not raw ints.)
__global__ void sm90_fill_strides_kernel(
    sm90_moe::StrideA* stride_a,
    sm90_moe::StrideB* stride_b,
    sm90_moe::StrideC* stride_c,
    sm90_moe::StrideD* stride_d,
    const int32_t* problem_sizes,   // [3*E] {M,N,K}
    int32_t num_experts)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_experts) return;
    const int M = problem_sizes[3 * e + 0];
    const int N = problem_sizes[3 * e + 1];
    const int K = problem_sizes[3 * e + 2];
    stride_a[e] = cutlass::make_cute_packed_stride(sm90_moe::StrideA{}, cute::make_shape(M, K, 1));
    stride_b[e] = cutlass::make_cute_packed_stride(sm90_moe::StrideB{}, cute::make_shape(N, K, 1));
    stride_c[e] = cutlass::make_cute_packed_stride(sm90_moe::StrideC{}, cute::make_shape(M, N, 1));
    stride_d[e] = cutlass::make_cute_packed_stride(sm90_moe::StrideD{}, cute::make_shape(M, N, 1));
}

// Sm90 grouped launcher. problem_sizes is the device int32[3*E] ({M,N,K}) array
// (bit-compatible with ProblemShape::UnderlyingProblemShape = Shape<int,int,int>).
// The ptr arrays + cute strides come from the marshal + fill kernels.
extern "C" int cutlass_moe_grouped_bf16_sm90(
    const void* problem_sizes,        // int32[3*E] device, == UnderlyingProblemShape[E]
    void** ptr_a, void** ptr_b, void** ptr_c,
    void* stride_a, void* stride_b, void* stride_c, void* stride_d,
    int32_t num_experts,
    int32_t sm_count,
    void* workspace,
    uint64_t stream)
{
    using namespace sm90_moe;

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = sm_count;

    // Derive the epilogue fusion-args type from a real lvalue (ex57 pattern):
    // decltype on a member of a default-constructed Arguments lvalue yields the
    // declared (non-reference) member type.
    typename Gemm::Arguments args;
    decltype(args.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    args = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts,
         reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
             const_cast<void*>(problem_sizes)),
         nullptr},  // host problem shapes unavailable (device-computed M_e)
        {(const ElementA**)ptr_a, reinterpret_cast<StrideA*>(stride_a),
         (const ElementB**)ptr_b, reinterpret_cast<StrideB*>(stride_b)},
        {fusion_args,
         (const ElementC**)ptr_c, reinterpret_cast<StrideC*>(stride_c),
         (ElementC**)ptr_c,       reinterpret_cast<StrideD*>(stride_d)},
        hw_info
    };

    Gemm op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;
    size_t ws = Gemm::get_workspace_size(args);
    void* ws_ptr = workspace;
    bool owned_ws = false;
    if (ws > 0 && ws_ptr == nullptr) {
        cudaMalloc(&ws_ptr, ws);
        owned_ws = true;
    }
    status = op.initialize(args, ws_ptr, (cudaStream_t)stream);
    if (status != cutlass::Status::kSuccess) {
        if (owned_ws) cudaFree(ws_ptr);
        return -2;
    }
    status = op.run((cudaStream_t)stream);
    if (owned_ws) cudaFree(ws_ptr);
    return (status == cutlass::Status::kSuccess) ? 0 : -3;
}

// Fill-strides launcher (Sm90 only).
extern "C" void cutlass_moe_grouped_sm90_fill_strides(
    void* stride_a, void* stride_b, void* stride_c, void* stride_d,
    const int32_t* problem_sizes, int32_t num_experts, uint64_t stream)
{
    const int threads = 128;
    const int blocks = (num_experts + threads - 1) / threads;
    sm90_fill_strides_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<sm90_moe::StrideA*>(stride_a),
        reinterpret_cast<sm90_moe::StrideB*>(stride_b),
        reinterpret_cast<sm90_moe::StrideC*>(stride_c),
        reinterpret_cast<sm90_moe::StrideD*>(stride_d),
        problem_sizes, num_experts);
}

// Sizes (bytes) of the Sm90 cute stride structs — so Rust allocates the right
// per-group stride arrays without hardcoding cute internals.
extern "C" int cutlass_moe_grouped_sm90_stride_bytes() {
    return (int)sizeof(sm90_moe::StrideA);
}

#else  // CUDA < 12: Sm90 path unavailable; define stubs so Rust FFI links.

extern "C" int cutlass_moe_grouped_bf16_sm90(
    const void*, void**, void**, void**, void*, void*, void*, void*,
    int32_t, int32_t, void*, uint64_t) { return -100; }
extern "C" void cutlass_moe_grouped_sm90_fill_strides(
    void*, void*, void*, void*, const int32_t*, int32_t, uint64_t) {}
extern "C" int cutlass_moe_grouped_sm90_stride_bytes() { return 0; }

#endif // __CUDACC_VER_MAJOR__ >= 12
