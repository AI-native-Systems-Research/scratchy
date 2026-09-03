// SPDX-License-Identifier: Apache-2.0
//
// Faithful port of MLX `affine_dequantize` kernel
// (mlx/backend/metal/kernels/quantized.h:2536). Bits = 4 only, scales /
// biases stored in F16 per `INT4_PARITY_PROBES.md` §1, dispatched in
// dtype × group_size combinations the int4 parity mandate exercises.
//
// For each output pair (one byte of packed `w` covers two nibbles, i.e.
// `pack_factor = 8 / bits = 2`), the kernel reads
//   - one byte of packed `w`
//   - one scale + one bias per `group_size` output elements
// and writes
//   out[oindex + 0] = scale * (byte        & 0x0f) + bias
//   out[oindex + 1] = scale * ((byte >> 4) & 0x0f) + bias
//
// The 2D grid matches MLX (`offset = x + grid_dim.x * y`) so callers
// can use the same `get_2d_grid_dims` pattern for large tensors;
// 1D dispatches still work — set `grid_dim.y == 1`, `index.y == 0`.

#include <metal_stdlib>
using namespace metal;

// `T_act` is the activation / output dtype (f16 or bf16). `T_scale` is the
// scales/biases storage dtype on disk (always f16 in every sampled
// mlx-community 4bit checkpoint — see `INT4_PARITY_PROBES.md` §7
// `Decision: in-register cast`). The cast `T_scale → T_act` happens on
// first load below, mirroring MLX's storage-vs-arithmetic split.
template <typename T_act, typename T_scale, const int group_size>
inline void affine_dequantize_b4_kernel(
    const device uint8_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    device T_act* out,
    uint2 index,
    uint2 grid_dim) {
    // pack_factor = 8 / bits = 2 for bits=4.
    constexpr int pack_factor = 2;

    size_t offset = index.x + grid_dim.x * size_t(index.y);
    size_t oindex = offset * pack_factor;
    size_t gindex = oindex / group_size;

    // In-register T_scale → T_act cast (`INT4_PARITY_PROBES.md` §7).
    T_act scale = static_cast<T_act>(scales[gindex]);
    T_act bias  = static_cast<T_act>(biases[gindex]);

    uint val = w[offset];
    out[oindex + 0] = scale * T_act(val & 0x0f)        + bias;
    out[oindex + 1] = scale * T_act((val >> 4) & 0x0f) + bias;
}

#define DEFINE_AFFINE_DEQUANTIZE_B4(act_tag, act_type, scale_tag, scale_type, gs)        \
    kernel void affine_dequantize_##act_tag##_s_##scale_tag##_gs_##gs##_b_4(             \
        const device uint8_t* w           [[buffer(0)]],                                 \
        const device scale_type* scales   [[buffer(1)]],                                 \
        const device scale_type* biases   [[buffer(2)]],                                 \
        device act_type* out              [[buffer(3)]],                                 \
        uint2 index    [[thread_position_in_grid]],                                      \
        uint2 grid_dim [[threads_per_grid]]) {                                           \
        affine_dequantize_b4_kernel<act_type, scale_type, gs>(                           \
            w, scales, biases, out, index, grid_dim);                                    \
    }

// Coverage: T_scale=half always (every sampled mlx-community 4bit ships
// F16 scales/biases — verified `INT4_PARITY_PROBES.md:73,287`). T_act in
// {half, bfloat} per `torch_dtype`. Pre-P10b also had `bfloat×bfloat`
// (loader cast F16→BF16 at load); that path is the regression site and
// is removed here.
DEFINE_AFFINE_DEQUANTIZE_B4(f16,  half,   f16, half,  32)
DEFINE_AFFINE_DEQUANTIZE_B4(f16,  half,   f16, half,  64)
DEFINE_AFFINE_DEQUANTIZE_B4(f16,  half,   f16, half, 128)
DEFINE_AFFINE_DEQUANTIZE_B4(bf16, bfloat, f16, half,  32)
DEFINE_AFFINE_DEQUANTIZE_B4(bf16, bfloat, f16, half,  64)
DEFINE_AFFINE_DEQUANTIZE_B4(bf16, bfloat, f16, half, 128)

// ============================================================================
// affine_embed_b4_kernel — gather + dequantize in one pass.
//
// Faithful port of MLX's `nn.QuantizedEmbedding.__call__`
// (`python/mlx/nn/layers/quantized.py:144`):
//   x  = self.weight[indices]      # gather packed u32 rows by token id
//   s  = self.scales[indices]      # gather scales rows
//   b  = self.biases[indices]      # gather biases rows
//   y  = mx.dequantize(x, s, b, group_size, bits, "affine")
// MLX dispatches the four ops as four kernels; scratchy-target-metal fuses them
// here because the gather-row indirection is a constant per output row,
// not a cross-op fusion of independent kernels.
//
// 2D grid:
//   index.x = byte-offset within a token row  ∈ [0, hidden_size / 2)
//   index.y = output token row                ∈ [0, num_tokens)
// Each thread reads one packed byte (= 2 nibbles for bits=4), fetches
// the corresponding scale + bias for that group, and writes 2 output
// elements. Total threads = num_tokens × (hidden_size / 2).
//
// Bindings:
//   buffer(0) w        : [vocab_size, hidden_size / 2] u8 (packed nibbles)
//   buffer(1) scales   : [vocab_size, hidden_size / group_size] T
//   buffer(2) biases   : [vocab_size, hidden_size / group_size] T
//   buffer(3) indices  : [num_tokens] uint32_t
//   buffer(4) out      : [num_tokens, hidden_size] T
//   function_constant(0) AFFINE_EMBED_HIDDEN_SIZE : uint = hidden_size
//
// `hidden_size` rides as a function constant rather than a kernel arg
// so the bucket-specialized pipeline bakes it in (mirroring
// `embed_bf16_specialized` in `embed.metal`).
//
// Note: bits=4 only — the int4 parity mandate doesn't exercise other
// widths from a quantized embedding in any sampled mlx-community
// checkpoint (`INT4_PARITY_PROBES.md` §2). Add other widths if a model
// surfaces.

constant uint AFFINE_EMBED_HIDDEN_SIZE [[function_constant(0)]];

template <typename T_act, typename T_scale, const int group_size>
inline void affine_embed_b4_kernel(
    const device uint8_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    const device uint* indices,
    device T_act* out,
    uint hidden_size,
    uint2 index) {
    constexpr int pack_factor = 2;
    // The dispatch rounds threadgroups.x up by threads_per_threadgroup.x;
    // hidden_size/2 may not align (3B's hidden=3072 → K/2=1536 > 1024
    // tpg cap forces 2 threadgroups, last one with partial). Without
    // this check, threads past the row boundary chase past-end bytes
    // of `w` and corrupt out[].
    if (index.x * pack_factor >= hidden_size) return;
    uint vocab_idx = indices[index.y];
    size_t bytes_per_row  = size_t(hidden_size) / pack_factor;
    size_t groups_per_row = size_t(hidden_size) / group_size;

    size_t w_offset    = size_t(vocab_idx) * bytes_per_row  + size_t(index.x);
    size_t out_col     = size_t(index.x) * pack_factor;
    size_t gindex      = size_t(vocab_idx) * groups_per_row + (out_col / group_size);
    size_t out_offset  = size_t(index.y) * size_t(hidden_size) + out_col;

    // In-register T_scale → T_act cast (`INT4_PARITY_PROBES.md` §7).
    T_act scale = static_cast<T_act>(scales[gindex]);
    T_act bias  = static_cast<T_act>(biases[gindex]);
    uint val = w[w_offset];

    out[out_offset + 0] = scale * T_act(val & 0x0f)        + bias;
    out[out_offset + 1] = scale * T_act((val >> 4) & 0x0f) + bias;
}

#define DEFINE_AFFINE_EMBED_B4(act_tag, act_type, scale_tag, scale_type, gs)             \
    kernel void affine_embed_##act_tag##_s_##scale_tag##_gs_##gs##_b_4(                  \
        const device uint8_t* w           [[buffer(0)]],                                 \
        const device scale_type* scales   [[buffer(1)]],                                 \
        const device scale_type* biases   [[buffer(2)]],                                 \
        const device uint*       indices  [[buffer(3)]],                                 \
        device act_type* out              [[buffer(4)]],                                 \
        uint2 index    [[thread_position_in_grid]]) {                                    \
        affine_embed_b4_kernel<act_type, scale_type, gs>(                                \
            w, scales, biases, indices, out, AFFINE_EMBED_HIDDEN_SIZE, index);           \
    }

DEFINE_AFFINE_EMBED_B4(f16,  half,   f16, half,    32)
DEFINE_AFFINE_EMBED_B4(f16,  half,   f16, half,    64)
DEFINE_AFFINE_EMBED_B4(f16,  half,   f16, half,   128)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, f16, half,    32)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, f16, half,    64)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, f16, half,   128)

// affine_embed_b8_kernel — 8-bit sibling of the b4 gather+dequant, for
// MLX-native mixed/dynamic quant (OptiQ) whose `embed_tokens` ships at
// 8-bit while the transformer default is 4-bit. One byte = one 8-bit
// code (pack_factor=1), so each thread reads one byte and writes ONE
// output element (vs the b4 kernel's 2 nibbles → 2 elements). Same MLX
// affine dequant `y = scale * code + bias`.
template <typename T_act, typename T_scale, const int group_size>
inline void affine_embed_b8_kernel(
    const device uint8_t* w,
    const device T_scale* scales,
    const device T_scale* biases,
    const device uint* indices,
    device T_act* out,
    uint hidden_size,
    uint2 index) {
    if (index.x >= hidden_size) return;
    uint vocab_idx = indices[index.y];
    size_t bytes_per_row  = size_t(hidden_size);
    size_t groups_per_row = size_t(hidden_size) / group_size;

    size_t w_offset    = size_t(vocab_idx) * bytes_per_row + size_t(index.x);
    size_t out_col     = size_t(index.x);
    size_t gindex      = size_t(vocab_idx) * groups_per_row + (out_col / group_size);
    size_t out_offset  = size_t(index.y) * size_t(hidden_size) + out_col;

    T_act scale = static_cast<T_act>(scales[gindex]);
    T_act bias  = static_cast<T_act>(biases[gindex]);
    uint val = w[w_offset];
    out[out_offset] = scale * T_act(val) + bias;
}

#define DEFINE_AFFINE_EMBED_B8(act_tag, act_type, scale_tag, scale_type, gs)             \
    kernel void affine_embed_##act_tag##_s_##scale_tag##_gs_##gs##_b_8(                  \
        const device uint8_t* w           [[buffer(0)]],                                 \
        const device scale_type* scales   [[buffer(1)]],                                 \
        const device scale_type* biases   [[buffer(2)]],                                 \
        const device uint*       indices  [[buffer(3)]],                                 \
        device act_type* out              [[buffer(4)]],                                 \
        uint2 index    [[thread_position_in_grid]]) {                                    \
        affine_embed_b8_kernel<act_type, scale_type, gs>(                                \
            w, scales, biases, indices, out, AFFINE_EMBED_HIDDEN_SIZE, index);           \
    }

DEFINE_AFFINE_EMBED_B8(f16,  half,   f16, half,    32)
DEFINE_AFFINE_EMBED_B8(f16,  half,   f16, half,    64)
DEFINE_AFFINE_EMBED_B8(f16,  half,   f16, half,   128)
DEFINE_AFFINE_EMBED_B8(bf16, bfloat, f16, half,    32)
DEFINE_AFFINE_EMBED_B8(bf16, bfloat, f16, half,    64)
DEFINE_AFFINE_EMBED_B8(bf16, bfloat, f16, half,   128)
// bf16-scale variants — Qwen3-MoE / `torch_dtype: bfloat16` ships BF16
// scales/biases.
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, bf16, bfloat, 32)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, bf16, bfloat, 64)
DEFINE_AFFINE_EMBED_B4(bf16, bfloat, bf16, bfloat, 128)
DEFINE_AFFINE_EMBED_B4(f16,  half,   bf16, bfloat, 32)
DEFINE_AFFINE_EMBED_B4(f16,  half,   bf16, bfloat, 64)
DEFINE_AFFINE_EMBED_B4(f16,  half,   bf16, bfloat, 128)

// bf16-scale 8-bit embed (Gemma4 / OptiQ ships bf16 scales).
DEFINE_AFFINE_EMBED_B8(bf16, bfloat, bf16, bfloat, 32)
DEFINE_AFFINE_EMBED_B8(bf16, bfloat, bf16, bfloat, 64)
DEFINE_AFFINE_EMBED_B8(bf16, bfloat, bf16, bfloat, 128)
DEFINE_AFFINE_EMBED_B8(f16,  half,   bf16, bfloat, 32)
DEFINE_AFFINE_EMBED_B8(f16,  half,   bf16, bfloat, 64)
DEFINE_AFFINE_EMBED_B8(f16,  half,   bf16, bfloat, 128)
