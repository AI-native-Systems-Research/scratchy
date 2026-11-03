// SPDX-License-Identifier: Apache-2.0
//
// Splash's register decode tile (~/git/splash @ 1482f6bb runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal:
// `gguf_sg::Shape`, `operand`, `coefficient_source`, `coefficient`, `gguf_sg::decode`, `decode_linear_gguf_prepare`;
// kernels/common/gguf_sgmatrix.h, the Table16 activation table; kernels/common/sgmatrix.h, the plain-register 8x8
// MMA; kernels/common/split_reduce.h), over Splash's af4g64 planes (`FmtAffine<4, 2>`, metal/abi/QuantFormat.h): MLX
// affine 4-bit codes in groups of 64, each 64-column tile's groups of 32 side by side in lane-owned chunk order, a
// (scale, bias) bf16 pair a 64-group. With these changes, and no other:
//
//   - The format is af4g64's alone, so its `Shape` is constants: codes with a min (value = s * code + z) enter the
//     MMA as the exact bf16 128 + code, one coefficient group per 32 inputs, four 64-input spans a coefficient unit.
//   - A step's rows are the bucket's (`QMM_M`): the prepare writes zero rows past them into the table, and the decode
//     stores only them. Splash's lanes always hold eight.
//
// The decode runs four simdgroups of 16 columns (`GGUF_REGISTER_COLUMNS`) a 64-column threadgroup and every 8-row
// lane of the step in each, one K partition (grid.y, Splash's `decodeSplits` with its register tiers) of whole
// 256-input units. Each unit's coefficients are decoded once per simdgroup into threadgroup memory; each 32-input
// group closes one MMA chain with s * chain + (z - 128 s) * sum, the sum of the group's inputs from the table.
//
// Baked constants: QMM_K (slot 0, the inputs), QMM_N (1, the columns), QMM_M (2, the step's rows), and the decode's
// REGISTER_RESIDUAL (3): it adds its result into the residual its output holds (Splash's EpResidual, aux = out).
// Prepare: input[0] (`[M][K]`), table[1], sums[2]; grid (K / 32, lanes), 128 threads.
// Decode: table[0], sums[1], w0[2] (plane0), meta[3], y[4], partials[5] (`[lanes][splits][8][N]` fp32), counters[6]
// (one u32 a 64-column tile, zero between dispatches); grid (N / 64, splits), 128 threads.

#pragma clang fp reassociate(off)
#include <metal_stdlib>
#include "baked.h"

using namespace metal;

SCRATCHY_CONSTANT_OPTIONAL(int, QMM_K, 0);
SCRATCHY_CONSTANT_OPTIONAL(int, QMM_N, 1);
SCRATCHY_CONSTANT_OPTIONAL(int, QMM_M, 2);
SCRATCHY_CONSTANT_OPTIONAL(bool, REGISTER_RESIDUAL, 3);

// metal/abi/Gguf.h and QuantFormat.h.
constant constexpr uint kTileColumns = 64;      // GGUF_TILE_COLUMNS
constant constexpr uint kRegisterColumns = 16;  // GGUF_REGISTER_COLUMNS
constant constexpr uint kRegisterThreads = 128; // GGUF_REGISTER_THREADS
constant constexpr uint kQuantTileRows = 64;    // QUANT_TILE_ROWS
constant constexpr uint kSpanInputs = 64;       // GGUF_TABLE16_SPAN_INPUTS
constant constexpr uint kSpanValues = 512;      // GGUF_TABLE16_SPAN_VALUES
constant constexpr uint kSpanSeeds = 32;        // GGUF_TABLE16_SPAN_SEEDS
constant constexpr uint kSpanSums = 16;         // GGUF_TABLE16_SPAN_SUMS
inline ulong table16_sums32_offset(uint width) { return ulong(width) / 16 * 8; }
inline ulong table16_sums_per_tile(uint width) { return table16_sums32_offset(width) + ulong(width) / 32 * 8; }

// af4g64: plane0 16 bytes a group of 32 (no plane1), meta 4 bytes a meta unit of two groups.
constant constexpr uint kP0 = 16;
constant constexpr uint kMetaBytes = 4;
constant constexpr uint kMetaGroups = 2;
// Its register `Shape`: HasMin, not Seeded, not Wide; the bf16 operand of code 0 is 128 (0x4300); one coefficient
// group per 32 inputs (CG); four spans a coefficient unit (UnitSpans); J coefficients a column and unit.
constant constexpr uint kOperand = 0x4300;
constant constexpr uint kCG = 1;
constant constexpr uint kUnitSpans = 4;
constant constexpr uint kJ = 2 * kUnitSpans * kCG;
constant constexpr uint kCoefs = kTileColumns * kJ;
// gguf_sgmatrix.h: the rows of a table, one lane's verify rows.
constant constexpr uint kRows = 8;
static_assert(kSpanValues == kSpanInputs * kRows, "a span holds every row's inputs");

// ---- kernels/common/sgmatrix.h: the 8x8 simdgroup_matrix MMA on plain-register operands.
struct Lane {
  ushort fm;
  ushort fn;
};
// A lane's thread_elements() are M[fm][fn] and M[fm][fn + 1].
inline Lane lane_map(uint lane) {
  const uint qid = lane >> 2;
  Lane l;
  l.fm = ushort((qid & 4) | ((lane >> 1) & 3));
  l.fn = ushort(((qid & 2) << 1) | ((lane & 1) << 1));
  return l;
}
template <typename T>
__attribute__((always_inline)) inline thread vec<T, 2> &te(thread simdgroup_matrix<T, 8, 8> &m) {
  return reinterpret_cast<thread vec<T, 2> &>(m.thread_elements());
}
// c += a x b, the simdgroup_matrix objects living only inside the call, so the accumulator stays a float2 in
// registers.
template <typename T>
__attribute__((always_inline)) inline void mma_acc(thread float2 &c, vec<T, 2> a, vec<T, 2> b) {
  simdgroup_matrix<T, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, D;
  te(A) = a;
  te(B) = b;
  te(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = te(D);
}

// ---- kernels/common/gguf_sgmatrix.h: Table16, the eight-row X^T table. A 64-element span occupies 512 bfloat in
// the xt_offset order; fragment 4q + f holds pair f of every chunk of the span's 32-element group q. Per row the sum
// of every 32 inputs (formats with a min) and the chain seed of every 16 (formats with a zero point) are stored
// row-minor.
constant constexpr uint kZeroPointOffset = 160;
//   offset(j, k', m) = (((j >> 2) 8 + k') 4 + (m >> 1)) 8 + (j & 3) 2 + (m & 1)
inline uint xt_offset(uint j, uint kp, uint m) { return (((j >> 2) * 8 + kp) * 4 + (m >> 1)) * 8 + (j & 3) * 2 + (m & 1); }
// Physical k inside a span -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint q = k >> 5, h = (k >> 4) & 1, a = (k >> 2) & 3, b = (k >> 1) & 1, e = k & 1;
  return uint2(4 * q + 2 * h + b, 2 * a + e);
}
// One simdgroup writes one span of one row; the lane holds elements 2 lane, 2 lane + 1.
inline void write_input(device bfloat *table, device float *sums, uint width, uint span, uint row, uint lane,
                        bfloat a, bfloat b) {
  const uint2 l = klogical(2 * lane);
  table[span * kSpanValues + xt_offset(l.x, l.y, row)] = a;
  table[span * kSpanValues + xt_offset(l.x, l.y + 1, row)] = b;
  float s = float(a) + float(b);
  s += simd_shuffle_xor(s, 1u);
  s += simd_shuffle_xor(s, 2u);
  s += simd_shuffle_xor(s, 4u);
  if ((lane & 7) == 0) sums[(span * (kSpanSeeds / kRows) + lane / 8) * kRows + row] = -float(kZeroPointOffset) * s;
  const float s32 = s + simd_shuffle_xor(s, 8u);
  if ((lane & 15) == 0) sums[table16_sums32_offset(width) + (span * (kSpanSums / kRows) + lane / 16) * kRows + row] = s32;
}

// ---- kernels/common/split_reduce.h.
inline bool split_arrive_last(device atomic_uint *counter, uint splits, uint thread_index,
                              threadgroup uint *arrival) {
  threadgroup_barrier(mem_flags::mem_device);
  if (thread_index == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  return *arrival == splits - 1;
}
template <typename T, typename Partial> inline T split_sum(T own, uint self, uint splits, Partial partial) {
  T total = T(0);
  for (uint s = 0; s < splits; ++s) total += s == self ? own : partial(s);
  return total;
}
inline void split_release(device atomic_uint *counter, uint thread_index) {
  if (thread_index == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

// ---- linear_gguf_sgmatrix.metal, af4g64.
// The exact bf16 operand of pair f of a chunk: 128 + code in both halves (quant_nibble_pairs).
inline bfloat2 operand(uint chunk, uint f) {
  const uint pair = (chunk >> (4 * f)) & 0x000F000Fu;
  return as_type<bfloat2>(pair + kOperand * 0x00010001u);
}
// Coefficient j of a column in coefficient unit u: (s, z - 128 s) of its group's meta unit.
inline float2 coefficient(uint meta, uint) {
  const float s = float(as_type<bfloat>(ushort(meta & 0xFFFFu))), m = float(as_type<bfloat>(ushort(meta >> 16)));
  return float2(s, fma(-128.0f, s, m));
}
inline uint coefficient_source(device const uchar *meta, uint plane_tile, uint groups, uint column, uint u, uint j) {
  const uint g = u * 2 * kUnitSpans + j / kCG;
  const uint units = groups / kMetaGroups;
  return *((device const uint *)(meta + ((ulong(plane_tile) * units + g / kMetaGroups) * kQuantTileRows + column) * kMetaBytes));
}

// One threadgroup: 4 simdgroups x 16 = 64 columns, L request lanes of eight rows, one K partition (tg.y) of `splits`.
template <typename T, uint L>
inline void register_decode(device const bfloat *table, device const float *sums, device const uchar *w0,
                            device const uchar *meta, device T *out, device coherent(device) float *partials,
                            device atomic_uint *counters, uint splits, uint2 tg, uint tid, uint sg, uint lane,
                            threadgroup float2 *coefs, threadgroup uint *arrival) {
  constexpr uint FG = 4 / kCG;  // fragments per coefficient group
  const uint K = QMM_K, spans = K / kSpanInputs, groups = K / 32;
  const ulong tileSums = table16_sums_per_tile(K);
  const uint units = spans / kUnitSpans;
  const uint u0 = tg.y * units / splits, u1 = (tg.y + 1) * units / splits;
  const Lane l = lane_map(lane);
  const uint fm = l.fm, fn = l.fn, c = fn / 2;
  const uint base = tg.x * kTileColumns + sg * kRegisterColumns, plane_tile = base / kQuantTileRows,
             plane_row = base % kQuantTileRows;
  constexpr uint NC = kRegisterColumns;
  threadgroup float2 *cu = coefs + sg * NC * kJ;

  // Weight streams: lane c reads chunk c of each group; a span's two groups of one column are a plane tile of
  // payloads apart.
  const uint first = u0 * kUnitSpans;
  device const uchar *p0[2];
#pragma unroll
  for (uint nf = 0; nf < 2; ++nf) {
    const ulong at = (ulong(plane_tile) * groups + 2 * first) * kQuantTileRows + plane_row + nf * 8 + fm;
    p0[nf] = w0 + at * kP0;
  }
  uint cur[2][2], nxt[2][2];
  const auto load = [&](thread uint (&ch)[2][2]) __attribute__((always_inline)) {
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      ch[nf][0] = *((device const uint *)(p0[nf] + 4 * c));
      ch[nf][1] = *((device const uint *)(p0[nf] + kQuantTileRows * kP0 + 4 * c));
    }
  };
  device const vec<bfloat, 8> *xt = reinterpret_cast<device const vec<bfloat, 8> *>(table + ulong(first) * kSpanValues);
  device const float *s32 = sums + table16_sums32_offset(K) + ulong(first) * kSpanSums;

  float2 acc[L][2];
#pragma unroll
  for (uint r = 0; r < L; ++r) acc[r][0] = acc[r][1] = float2(0);
  load(cur);
  for (uint u = u0; u < u1; ++u) {
    // This unit's coefficients for the simdgroup's NC columns, each decoded once; every source load is issued
    // before the first decode waits for one.
    constexpr uint I = (NC * kJ + 31) / 32;
    uint src[I];
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * kJ) src[i] = coefficient_source(meta, plane_tile, groups, plane_row + e % NC, u, e / NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint i = 0; i < I; ++i) {
      const uint e = lane + 32 * i;
      if (e < NC * kJ) cu[e] = coefficient(src[i], e / NC);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (uint us = 0; us < kUnitSpans; ++us) {
      const bool more = us + 1 < kUnitSpans || u + 1 < u1;
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) p0[nf] += 2 * kQuantTileRows * kP0;
      if (more) load(nxt);
#pragma unroll
      for (uint q = 0; q < 2; ++q) {
        bfloat2 a[2][4];
        float2 cs[kCG][2];
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) {
#pragma unroll
          for (uint f = 0; f < 4; ++f) a[nf][f] = operand(cur[nf][q], f);
#pragma unroll
          for (uint h = 0; h < kCG; ++h) cs[h][nf] = cu[((us * 2 + q) * kCG + h) * NC + nf * 8 + fm];
        }
#pragma unroll
        for (uint r = 0; r < L; ++r) {
          const vec<bfloat, 8> bq = xt[ulong(r) * K + (8 * q + fm) * 4 + c];
          const float2 sum = *(device const float2 *)(s32 + ulong(r) * tileSums + q * 8 + fn);
#pragma unroll
          for (uint h = 0; h < kCG; ++h) {
#pragma unroll
            for (uint nf = 0; nf < 2; ++nf) {
              float2 dot = float2(0);
#pragma unroll
              for (uint f = h * FG; f < (h + 1) * FG; ++f)
                mma_acc<bfloat>(dot, a[nf][f], reinterpret_cast<thread const bfloat2 *>(&bq)[f]);
              acc[r][nf] = fma(dot, cs[h][nf].x, acc[r][nf]);
              acc[r][nf] = fma(sum, cs[h][nf].y, acc[r][nf]);
            }
          }
        }
      }
      xt += kSpanValues / 8;
      s32 += kSpanSums;
      if (more) {
#pragma unroll
        for (uint nf = 0; nf < 2; ++nf) cur[nf][0] = nxt[nf][0], cur[nf][1] = nxt[nf][1];
      }
    }
  }
  const uint stride = QMM_N;
  if (splits > 1) {
    // Partials [lane][split][8 rows][columns], one counter per 64-column tile.
    device atomic_uint *counter = counters + base / kTileColumns;
#pragma unroll
    for (uint r = 0; r < L; ++r)
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf) {
        device coherent(device) float *slot =
            partials + ((ulong(r) * splits + tg.y) * 8 + fn) * stride + base + nf * 8 + fm;
        slot[0] = acc[r][nf].x;
        slot[stride] = acc[r][nf].y;
      }
    if (!split_arrive_last(counter, splits, tid, arrival)) return;
#pragma unroll
    for (uint r = 0; r < L; ++r)
#pragma unroll
      for (uint nf = 0; nf < 2; ++nf)
        acc[r][nf] = split_sum(acc[r][nf], tg.y, splits, [&](uint s) {
          device coherent(device) float *slot =
              partials + ((ulong(r) * splits + s) * 8 + fn) * stride + base + nf * 8 + fm;
          return float2(slot[0], slot[stride]);
        });
    split_release(counter, tid);
  }
#pragma unroll
  for (uint r = 0; r < L; ++r)
#pragma unroll
    for (uint nf = 0; nf < 2; ++nf) {
      const uint column = base + nf * 8 + fm;
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        const uint row = r * 8 + fn + i;
        if (int(row) >= QMM_M) continue;
        const ulong at = ulong(row) * stride + column;
        float v = acc[r][nf][i];
        if (REGISTER_RESIDUAL) v += float(out[at]);
        out[at] = T(v);
      }
    }
}

// Table16 for the step's lanes of eight rows of `QMM_K` inputs, one simdgroup per (span, row); the rows past the
// step's are zero.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(128)]] void affine_register_prepare(
    device const T *input [[buffer(0)]],
    device bfloat *table [[buffer(1)]],
    device float *sums [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint width = QMM_K;
  const uint span = (tg.x * 4 + sg) / 8, row = (tg.x * 4 + sg) % 8;
  const uint step_row = tg.y * 8 + row;
  const uint k = span * kSpanInputs + 2 * lane;
  const bool live = int(step_row) < QMM_M;
  const bfloat a = live ? bfloat(float(input[ulong(step_row) * width + k])) : bfloat(0.0f);
  const bfloat b = live ? bfloat(float(input[ulong(step_row) * width + k + 1])) : bfloat(0.0f);
  write_input(table + ulong(tg.y) * width * 8, sums + ulong(tg.y) * table16_sums_per_tile(width), width, span, row,
              lane, a, b);
}

template <typename T, uint L>
[[kernel, max_total_threads_per_threadgroup(128)]] void affine_register_decode(
    device const bfloat *table [[buffer(0)]],
    device const float *sums [[buffer(1)]],
    device const uchar *w0 [[buffer(2)]],
    device const uchar *meta [[buffer(3)]],
    device T *y [[buffer(4)]],
    device coherent(device) float *partials [[buffer(5)]],
    device atomic_uint *counters [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 coefs[kCoefs];
  threadgroup uint arrival;
  register_decode<T, L>(table, sums, w0, meta, y, partials, counters, grid.y, tg, tid, sg, lane, coefs, &arrival);
}

#define INST_REGISTER(act_tag, act_type)                                                                  \
  SCRATCHY_KERNEL(affine_register_prepare_##act_tag, affine_register_prepare<act_type>)                    \
  SCRATCHY_KERNEL(affine_register_decode_##act_tag##_l_1, affine_register_decode<act_type, 1>)            \
  SCRATCHY_KERNEL(affine_register_decode_##act_tag##_l_2, affine_register_decode<act_type, 2>)

INST_REGISTER(bf16, bfloat)
INST_REGISTER(f16, half)
