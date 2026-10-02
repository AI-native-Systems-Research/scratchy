// SPDX-License-Identifier: Apache-2.0
// DEVICE FACTS — what the decode's launch split is solved from (`split.rs`), measured on the GPU
// at load with synthetic work: no model, no weights. (The streaming rates are measured with each
// tape's own calibration kernels, generated beside its runs.)
#include <metal_stdlib>
using namespace metal;

// One step of a chain of dependent launches: the first thread bumps what the last launch wrote.
[[kernel, max_total_threads_per_threadgroup(1024)]] void df_launch(
    device uint* flag [[buffer(0)]],
    uint t [[thread_index_in_threadgroup]],
    uint g [[threadgroup_position_in_grid]]) {
  if (t == 0u && g == 0u) {
    flag[0] = flag[0] + 1u;
  }
}

// `steps` dependent steps in ONE threadgroup, each as small as a norm: every thread reads a word
// the last step wrote, the threadgroup sums them, one thread writes the sum back for the next.
[[kernel, max_total_threads_per_threadgroup(1024)]] void df_chain(
    device uint* words [[buffer(0)]],
    constant uint& steps [[buffer(1)]],
    uint t [[thread_index_in_threadgroup]],
    uint n [[threads_per_threadgroup]],
    uint s [[simdgroup_index_in_threadgroup]],
    uint l [[thread_index_in_simdgroup]]) {
  threadgroup uint partial[32];
  for (uint i = 0u; i < steps; ++i) {
    const uint v = simd_sum(words[(t + i) % n]);
    if (l == 0u) {
      partial[s] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
      uint sum = 0u;
      for (uint k = 0u; k < (n + 31u) / 32u; ++k) {
        sum += partial[k];
      }
      words[i % n] = sum;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}
