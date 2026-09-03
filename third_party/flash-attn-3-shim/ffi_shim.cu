// FA3 shim — bf16, hdim128, sm90. AOT-scheduling variant.
//
// Replaces the PyTorch-driven mha_fwd_get_scheduler_metadata + mha_fwd entry
// points in vllm-flash-attention-3/hopper/flash_api.cpp with raw-pointer C
// ABIs suitable for calling from Rust.
//
// Two functions are exported:
//
//   fa3_get_scheduler_metadata_bf16_hdim128_sm90:
//     Runs `prepare_varlen_num_blocks` once into a caller-provided workspace.
//     Mirrors flash_api.cpp::mha_fwd_get_scheduler_metadata. Call this once
//     per scheduler step (or once per forward in the layer-0 AOT pattern).
//
//   fa3_paged_decode_bf16_hdim128_sm90:
//     Runs the main attention kernel (Split=true with combine when
//     num_splits>1, else Split=false). With `skip_scheduler_metadata=1` the
//     kernel reads the pre-populated workspace and skips its own prelude;
//     with `skip_scheduler_metadata=0` the kernel runs its own prelude
//     (eager prefill / standalone microbench path).
//
// Workspace layout — `1 + b_rounded * num_prepare_batch_vectors` ints for
// the Hopper paged-decode varlen+head_swizzle config. This shim only
// supports use_prepare_varlen=true (varlen always on).
//   Slots in order: prepare_seqlen_q, num_splits_dynamic?, varlen_batch_idx?,
//                   num_nheads_in_l2 (head_swizzle), tile_count_semaphore.
//
// num_prepare_batch_vectors:
//   1                                   // use_prepare_varlen always
//   + (use_dynamic_split ? 1 : 0)       // splits>1 + b<=992
//   + (varlen_sort_batches ? 1 : 0)     // 0 in our config
//   + (head_swizzle ? 1 : 0)            // 1 when is_causal (always for us)

#include "flash.h"
#include "heuristics.h"   // use_one_mma_wg
#include "tile_size.h"    // tile_size_fwd_sm90
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

template<int Arch, typename T, int kHeadDimQK, int kHeadDimVO, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template <typename T, typename Tpartial, int kBlockK>
void run_mha_fwd_combine_(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl);

void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, bool packgqa, int blockM, int blockN, bool enable_pdl);

static inline int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

// ---------------------------------------------------------------------------
// Shared param setup. Both entry points use identical Flash_fwd_params layout
// for the paged-bf16-hdim128-causal-varlen config — factored out here so the
// scheduler-prelude (`get_scheduler_metadata`) and the consumer (`paged_decode`)
// can't drift.
// ---------------------------------------------------------------------------
static void fa3_fill_params(
    Flash_fwd_params& params,
    int batch_size, int total_q_tokens, int num_q_heads, int num_kv_heads,
    int max_pages_per_seq, int num_pages_total, int page_size,
    int max_seqlen_q, int max_seqlen_k,
    int num_splits, float softmax_scale, int sm_count,
    const int* page_table, const int* cu_seqlens_q, const int* seqused_k
) {
    constexpr int head_dim = 128;
    params.is_bf16 = true;
    params.is_e4m3 = false;
    params.q_row_stride = (int64_t)num_q_heads * head_dim;
    params.q_head_stride = head_dim;
    params.k_row_stride = (int64_t)num_kv_heads * head_dim;
    params.k_head_stride = head_dim;
    params.k_batch_stride = (int64_t)page_size * num_kv_heads * head_dim;
    params.v_row_stride = params.k_row_stride;
    params.v_head_stride = params.k_head_stride;
    params.v_batch_stride = params.k_batch_stride;
    params.v_dim_stride = 1;
    params.o_row_stride = (int64_t)num_q_heads * head_dim;
    params.o_head_stride = head_dim;

    params.cu_seqlens_q = const_cast<int*>(cu_seqlens_q);
    params.cu_seqlens_k = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = const_cast<int*>(seqused_k);
    params.leftpad_k = nullptr;
    params.knew_ptr = nullptr;
    params.cu_seqlens_knew = nullptr;
    params.seqlen_knew = 0;

    params.b = batch_size;
    params.b_k = batch_size;
    params.h = num_q_heads;
    params.h_k = num_kv_heads;
    params.seqlen_q = max_seqlen_q;
    params.seqlen_k = max_seqlen_k;
    params.seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    params.d = head_dim;
    params.dv = head_dim;
    params.d_rounded = head_dim;
    params.dv_rounded = head_dim;
    params.total_q = total_q_tokens;
    params.total_k = batch_size * page_size;

    params.scale_softmax = softmax_scale;
    params.softcap = 0.0f;
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;

    params.is_causal = true;
    params.is_local = false;
    params.window_size_left = max_seqlen_k - 1;
    params.window_size_right = 0;
    params.head_swizzle = true;
    params.varlen_sort_batches = false;

    params.arch = 90;
    params.num_sm = sm_count;

    params.page_size = page_size;
    params.page_table = const_cast<int*>(page_table);
    params.page_table_batch_stride = max_pages_per_seq;
    params.num_pages = num_pages_total;
    params.pagedkv_tma = false;

    params.num_splits = num_splits;
    params.pack_gqa = true;
    params.prepare_varlen_pdl = (batch_size <= 992);
}

// Bind the workspace pointers (prepare_seqlen_q, num_splits_dynamic?,
// num_nheads_in_l2, tile_count_semaphore) for our config. Both entry points
// share this so the layout matches.
//
// num_prepare_batch_vectors = 1 (prepare_varlen) + use_dynamic_split + 1 (head_swizzle)
//                           = 2 when num_splits == 1
//                           = 3 when num_splits  > 1
// Workspace size: b_rounded * num_prepare_batch_vectors + 1 ints.
static void fa3_bind_workspace(Flash_fwd_params& params, int* workspace) {
    int b_rounded = round_multiple(params.b, 4);
    bool use_dynamic_split = (params.num_splits > 1) && (params.b <= 992);
    int num_prepare_batch_vectors = 1 + (use_dynamic_split ? 1 : 0) + 1;
    int head_swizzle_offset = b_rounded * (num_prepare_batch_vectors - 1);
    int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;

    params.prepare_seqlen_q_ptr = workspace;
    params.num_splits_dynamic_ptr = use_dynamic_split ? workspace + b_rounded : nullptr;
    params.varlen_batch_idx_ptr = nullptr;
    params.num_nheads_in_l2_ptr = workspace + head_swizzle_offset;
    params.tile_count_semaphore = workspace + tile_count_semaphore_offset;
    params.tile_count_semaphore_offset = tile_count_semaphore_offset;
}

// ---------------------------------------------------------------------------
// Scheduler-metadata prelude. Mirrors mha_fwd_get_scheduler_metadata from
// flash_api.cpp (paged-bf16-hdim128 path). Runs `prepare_varlen_num_blocks`
// into the caller-provided workspace.
//
// `num_splits` must match the value the consumer will pass to `fa3_paged_
// decode_..._sm90` so the workspace layout matches.
//
// Workspace must be at least
//     1 + round_up(batch_size, 4) * (num_splits > 1 ? 3 : 2)  ints,
// pre-allocated. The kernel writes the prelude vectors and zeroes the
// semaphore as part of its execution; the caller does NOT need to memset.
// ---------------------------------------------------------------------------
extern "C" int fa3_get_scheduler_metadata_bf16_hdim128_sm90(
    int* workspace,
    const int* page_table,
    const int* cu_seqlens_q,
    const int* seqused_k,
    int batch_size,
    int total_q_tokens,
    int num_q_heads,
    int num_kv_heads,
    int max_pages_per_seq,
    int num_pages_total,
    int page_size,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_splits,
    float softmax_scale,
    int sm_count,
    cudaStream_t stream
) {
    Flash_fwd_params params{};
    fa3_fill_params(params, batch_size, total_q_tokens, num_q_heads, num_kv_heads,
                    max_pages_per_seq, num_pages_total, page_size,
                    max_seqlen_q, max_seqlen_k, num_splits, softmax_scale, sm_count,
                    page_table, cu_seqlens_q, seqused_k);
    fa3_bind_workspace(params, workspace);

    // Match flash_api.cpp::mha_fwd_get_scheduler_metadata: kBlockM/kBlockN
    // come from tile_size_fwd_sm90 with use_one_mma_wg(params).
    auto kBlockMN = tile_size_fwd_sm90(
        params.d_rounded, params.dv_rounded, params.is_causal, params.is_local,
        /*element_size*/ 2, /*v_colmajor*/ false,
        params.page_table && !params.pagedkv_tma,
        params.softcap > 0.f, use_one_mma_wg(params));
    int kBlockM = std::get<0>(kBlockMN);
    int kBlockN = std::get<1>(kBlockMN);
    prepare_varlen_num_blocks(params, stream, params.pack_gqa, kBlockM, kBlockN, /*enable_pdl=*/false);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fa3_get_scheduler_metadata launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// FA3 paged decode consumer.
//
// `skip_scheduler_metadata`:
//   0 = run the prelude here (standalone path; eager + microbench).
//   1 = the workspace was already populated by `fa3_get_scheduler_metadata_*`
//       earlier this step; skip the prelude. Saves one captured kernel per
//       attention call (matches Python vLLM AOT-scheduling behavior).
//
// num_splits == 1: no oaccum/lseaccum needed (oaccum / softmax_lseaccum may
//                  be null). Only `Split=false` kernel runs.
// num_splits  > 1: oaccum [num_splits, h, total_q, dv] and lseaccum
//                  [num_splits, h, total_q] must be allocated. After the
//                  split kernel runs, run_mha_fwd_combine_ writes the final
//                  out + softmax_lse from these partials.
// ---------------------------------------------------------------------------
extern "C" int fa3_paged_decode_bf16_hdim128_sm90(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* out,
    void* softmax_lse,
    void* oaccum,
    void* softmax_lseaccum,
    int* scheduler_workspace,
    const int* page_table,
    const int* cu_seqlens_q,
    const int* seqused_k,
    int batch_size,
    int total_q_tokens,
    int num_q_heads,
    int num_kv_heads,
    int max_pages_per_seq,
    int num_pages_total,
    int page_size,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_splits,
    int skip_scheduler_metadata,
    float softmax_scale,
    int sm_count,
    cudaStream_t stream
) {
    Flash_fwd_params params{};
    fa3_fill_params(params, batch_size, total_q_tokens, num_q_heads, num_kv_heads,
                    max_pages_per_seq, num_pages_total, page_size,
                    max_seqlen_q, max_seqlen_k, num_splits, softmax_scale, sm_count,
                    page_table, cu_seqlens_q, seqused_k);

    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k_cache);
    params.v_ptr = const_cast<void*>(v_cache);
    params.o_ptr = out;
    params.softmax_lse_ptr = softmax_lse;

    fa3_bind_workspace(params, scheduler_workspace);
    params.skip_scheduler_metadata_computation = (skip_scheduler_metadata != 0);

    if (num_splits > 1) {
        constexpr int head_dim = 128;
        params.is_fp32 = false;
        params.oaccum_ptr = oaccum;
        params.softmax_lseaccum_ptr = softmax_lseaccum;
        params.oaccum_split_stride = (int64_t)num_q_heads * total_q_tokens * head_dim;
        params.oaccum_row_stride = head_dim;
        params.oaccum_head_stride = (int64_t)total_q_tokens * head_dim;
        params.lseaccum_split_stride = (int64_t)num_q_heads * total_q_tokens;
        params.lseaccum_head_stride = total_q_tokens;

        run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, /*Split=*/true,  /*PagedKVNonTMA=*/true, /*Has_softcap=*/false, /*PackGQA=*/true>(params, stream);
        cudaError_t err1 = cudaGetLastError();
        if (err1 != cudaSuccess) {
            fprintf(stderr, "fa3 split kernel launch error: %s\n", cudaGetErrorString(err1));
            return -1;
        }
        run_mha_fwd_combine_<cutlass::bfloat16_t, float, /*kBlockK=*/128>(params, stream, /*enable_pdl=*/false);
    } else {
        run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, /*Split=*/false, /*PagedKVNonTMA=*/true, /*Has_softcap=*/false, /*PackGQA=*/true>(params, stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fa3_paged_decode launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
