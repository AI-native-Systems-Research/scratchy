# SPDX-License-Identifier: Apache-2.0
"""spyre-inference's `reshape_and_cache_head_major_kernel`
(`spyre_inference/v1/attention/ops/reshape_and_cache_head_major.py`, spyre-inference 297fc48), in
Triton: the KV store into the head-major paged cache, one kv head at a time.

Their kernel, statement for statement. The operands are theirs, in their order:

* `key` / `value` `[T, num_kv_heads, head_size]`, this step's keys and values;
* `k_rows` / `v_rows` the cache as rows, `[num_blocks * num_kv_heads * block_size, head_size]`
  (`kv_slot_views`);
* `row_index`: one `[T]` int32 index per kv head (`kv_write_index`: head h of the token at
  `block * block_size + offset` lives at row `(block * num_kv_heads + h) * block_size + offset`),
  the `num_kv_heads` indices one after another.

`k_rows.index_copy_(0, idx, key[:, h])` is the device's scatter: `desc.scatter` lowers to an
indirect access tile over the cache's rows, indexed through `row_index`, and a store through it.

Where Triton spells a torch op differently, the spelling is Triton's and the op is theirs: an index
of fewer than 8 rows is below Triton's scatter floor, so under 8 tokens each row is stored on its
own, at the row the index names. Their `key = key * 1.0` (materializing a strided view of the fused
QKV projection, whose per-head slice torch-spyre stores wrong) has no counterpart: `key` here is its
own tensor, not a view.
"""

import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_head_major_kernel(key, value, k_rows, v_rows, row_index,  #
                                        T: tl.constexpr, NUM_KV_HEADS: tl.constexpr,  #
                                        HEAD_SIZE: tl.constexpr, NUM_ROWS: tl.constexpr):
    k_dst = tl.make_tensor_descriptor(k_rows, shape=[NUM_ROWS, HEAD_SIZE], strides=[HEAD_SIZE, 1],
                                      block_shape=[1, HEAD_SIZE])
    v_dst = tl.make_tensor_descriptor(v_rows, shape=[NUM_ROWS, HEAD_SIZE], strides=[HEAD_SIZE, 1],
                                      block_shape=[1, HEAD_SIZE])
    if T >= 8:
        # key[:, h]: the [T, head_size] column block of head h.
        k_src = tl.make_tensor_descriptor(key, shape=[T, NUM_KV_HEADS * HEAD_SIZE],
                                          strides=[NUM_KV_HEADS * HEAD_SIZE, 1],
                                          block_shape=[T, HEAD_SIZE])
        v_src = tl.make_tensor_descriptor(value, shape=[T, NUM_KV_HEADS * HEAD_SIZE],
                                          strides=[NUM_KV_HEADS * HEAD_SIZE, 1],
                                          block_shape=[T, HEAD_SIZE])
        idx_desc = tl.make_tensor_descriptor(row_index, shape=[NUM_KV_HEADS * T], strides=[1],
                                             block_shape=[T])
        # for h, idx in enumerate(row_index):
        for h in tl.static_range(NUM_KV_HEADS):
            idx = idx_desc.load([h * T])
            # k_rows.index_copy_(0, idx, key[:, h])
            k_dst.scatter(k_src.load([0, h * HEAD_SIZE]), idx, 0)
            # v_rows.index_copy_(0, idx, value[:, h])
            v_dst.scatter(v_src.load([0, h * HEAD_SIZE]), idx, 0)
    else:
        k_src = tl.make_tensor_descriptor(key, shape=[T, NUM_KV_HEADS * HEAD_SIZE],
                                          strides=[NUM_KV_HEADS * HEAD_SIZE, 1],
                                          block_shape=[1, HEAD_SIZE])
        v_src = tl.make_tensor_descriptor(value, shape=[T, NUM_KV_HEADS * HEAD_SIZE],
                                          strides=[NUM_KV_HEADS * HEAD_SIZE, 1],
                                          block_shape=[1, HEAD_SIZE])
        for h in tl.static_range(NUM_KV_HEADS):
            for t in tl.static_range(T):
                row = tl.load(row_index + h * T + t)
                k_dst.store([row, 0], k_src.load([t, h * HEAD_SIZE]))
                v_dst.store([row, 0], v_src.load([t, h * HEAD_SIZE]))
