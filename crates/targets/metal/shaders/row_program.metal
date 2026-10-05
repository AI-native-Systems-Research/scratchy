// SPDX-License-Identifier: Apache-2.0
//
// row_program — a group of row-wise steps over one width (`MetalFusion::RowProgram`) as one
// command: loads, adds, RMSNorms and scales, the row held in registers between them, one
// threadgroup per row. Every value is held in the activation type, as its own kernel stores it,
// and each norm sums its input's squares over `row_sum`'s tree as `rmsnorm_*` does — though the
// fast-math compile may keep a value unrounded into the step reading it, so a row can differ from
// its kernels' in the last place (`tests/row_program_test.rs` bounds it).
//
// Constants: 0 = the width, 1 = the threads (`NORM_THREADS`), 2 = the instructions. Instruction k
// is slot 3 + 3k — its word: op (bits 0-3), dst (4-7), a (8-11), b (12-15), weight (16-19) — and
// slots 4 + 3k (an epsilon or a scale) and 5 + 3k (a gain offset):
//   op 0 LOAD   reg[dst] = input a          op 3 SCALE_W  reg[dst] = reg[a] * scalar weight w
//   op 1 ADD    reg[dst] = reg[a] + reg[b]  op 4 SCALE    reg[dst] = reg[a] * F
//   op 2 NORM   reg[dst] = rmsnorm(reg[a]) * (gain w + G), epsilon F
//   op 5 STORE  output dst = reg[a]
// Bindings: inputs 0-3, outputs 4-6, gains 7-14 (the norms' dtype), scalar weights 15-16.
// Dispatch: (rows, 1, 1) threadgroups of the threads.

#include <metal_stdlib>
#include "baked.h"
#include "row_sum.h"
using namespace metal;

SCRATCHY_CONSTANT(uint, RP_WIDTH, 0);
SCRATCHY_CONSTANT(uint, RP_THREADS, 1);
SCRATCHY_CONSTANT(uint, RP_LEN, 2);
constant constexpr uint RP_PER = (RP_WIDTH + RP_THREADS - 1) / RP_THREADS;
// Registers: one per loaded input and per step (4 + 8).
constant constexpr uint RP_REGS = 12;

#define RP_INSTR(k, w, f, g)                                                                   \
  SCRATCHY_CONSTANT_OPTIONAL(uint, RP_WORD_##k, w);                                           \
  SCRATCHY_CONSTANT_OPTIONAL(float, RP_F_##k, f);                                             \
  SCRATCHY_CONSTANT_OPTIONAL(float, RP_G_##k, g);
RP_INSTR(0, 3, 4, 5)
RP_INSTR(1, 6, 7, 8)
RP_INSTR(2, 9, 10, 11)
RP_INSTR(3, 12, 13, 14)
RP_INSTR(4, 15, 16, 17)
RP_INSTR(5, 18, 19, 20)
RP_INSTR(6, 21, 22, 23)
RP_INSTR(7, 24, 25, 26)
RP_INSTR(8, 27, 28, 29)
RP_INSTR(9, 30, 31, 32)
RP_INSTR(10, 33, 34, 35)
RP_INSTR(11, 36, 37, 38)
RP_INSTR(12, 39, 40, 41)
RP_INSTR(13, 42, 43, 44)
RP_INSTR(14, 45, 46, 47)
RP_INSTR(15, 48, 49, 50)

// The program's buffers.
template <typename T, typename G>
struct RowBuffers {
  device const T* in[4];
  device T* out[3];
  device const G* gain[8];
  device const T* scalar[2];
};

// Run instruction `word` (F, Gw its floats) over this thread's elements of row `row`.
template <typename T, typename G>
inline void rp_run(uint word, float F, float Gw, thread T (&reg)[RP_REGS][RP_PER],
                   thread const RowBuffers<T, G>& b, uint row, uint tid,
                   threadgroup float* scratch) {
  const uint op = word & 15u, dst = (word >> 4) & 15u, a = (word >> 8) & 15u;
  const uint rb = (word >> 12) & 15u, w = (word >> 16) & 15u;
  const uint base = row * RP_WIDTH;
  if (op == 2u) {
    float local = 0.0f;
    for (uint j = 0; j < RP_PER; ++j) {
      const uint i = tid + j * RP_THREADS;
      if (i < RP_WIDTH) {
        local += float(reg[a][j]) * float(reg[a][j]);
      }
    }
    const float rms = sqrt(row_sum(local, tid, RP_THREADS, scratch) / float(RP_WIDTH) + F);
    // Every thread has the sum before the next norm's partials overwrite it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < RP_PER; ++j) {
      const uint i = tid + j * RP_THREADS;
      if (i < RP_WIDTH) {
        reg[dst][j] = T((float(reg[a][j]) / rms) * (float(b.gain[w][i]) + Gw));
      }
    }
    return;
  }
  for (uint j = 0; j < RP_PER; ++j) {
    const uint i = tid + j * RP_THREADS;
    if (i >= RP_WIDTH) {
      continue;
    }
    switch (op) {
      case 0u: reg[dst][j] = b.in[a][base + i]; break;
      case 1u: reg[dst][j] = T(float(reg[a][j]) + float(reg[rb][j])); break;
      case 3u: reg[dst][j] = T(float(reg[a][j]) * float(b.scalar[w][0])); break;
      case 4u: reg[dst][j] = T(float(reg[a][j]) * F); break;
      default: b.out[dst][base + i] = reg[a][j]; break;
    }
  }
}

#define RP_STEP(k) \
  if (k < RP_LEN) rp_run<T, G>(RP_WORD_##k, RP_F_##k, RP_G_##k, reg, b, gid, tid, scratch);

template <typename T, typename G>
[[kernel]] void row_program_impl(
    device const T* in0 [[buffer(0)]],
    device const T* in1 [[buffer(1)]],
    device const T* in2 [[buffer(2)]],
    device const T* in3 [[buffer(3)]],
    device T* out0 [[buffer(4)]],
    device T* out1 [[buffer(5)]],
    device T* out2 [[buffer(6)]],
    device const G* g0 [[buffer(7)]],
    device const G* g1 [[buffer(8)]],
    device const G* g2 [[buffer(9)]],
    device const G* g3 [[buffer(10)]],
    device const G* g4 [[buffer(11)]],
    device const G* g5 [[buffer(12)]],
    device const G* g6 [[buffer(13)]],
    device const G* g7 [[buffer(14)]],
    device const T* s0 [[buffer(15)]],
    device const T* s1 [[buffer(16)]],
    uint gid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[RP_THREADS];
  const RowBuffers<T, G> b = {
      {in0, in1, in2, in3}, {out0, out1, out2}, {g0, g1, g2, g3, g4, g5, g6, g7}, {s0, s1}};
  T reg[RP_REGS][RP_PER];
  RP_STEP(0) RP_STEP(1) RP_STEP(2) RP_STEP(3) RP_STEP(4) RP_STEP(5) RP_STEP(6) RP_STEP(7)
  RP_STEP(8) RP_STEP(9) RP_STEP(10) RP_STEP(11) RP_STEP(12) RP_STEP(13) RP_STEP(14) RP_STEP(15)
}

#define INST_ROW_PROGRAM(act_tag, act_type, gain_tag, gain_type) \
  SCRATCHY_KERNEL(row_program_##act_tag##_s_##gain_tag, row_program_impl<act_type, gain_type>)

INST_ROW_PROGRAM(f16, half, f16, half)
INST_ROW_PROGRAM(bf16, bfloat, f16, half)
INST_ROW_PROGRAM(bf16, bfloat, bf16, bfloat)
INST_ROW_PROGRAM(f16, half, bf16, bfloat)
