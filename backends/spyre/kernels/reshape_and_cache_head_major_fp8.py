# SPDX-License-Identifier: Apache-2.0
"""The fp8 KV pool's twin of `reshape_and_cache_head_major.py`'s `reshape_and_cache_head_major_kernel`.

An fp8 page is TOKEN-major, `[block_size, num_kv_heads, head_size]`: one token's keys are
`num_kv_heads * head_size` bytes, whole sticks at every head size, so the write lands whole
sticks. Each written token's row is narrowed to fp8 and lands as whole sticks.
"""

import triton
import triton.language as tl


@triton.jit
def _to_fp8(x):
    # The fp8 pool's write: the rows clamped to fp8 E4M3's finite range, then narrowed.
    return tl.minimum(tl.maximum(x, -448.0), 448.0).to(tl.float8e4nv)


@triton.jit
def reshape_and_cache_head_major_fp8_kernel(key, value, k_rows, v_rows, row_index,  #
                                        T: tl.constexpr, NUM_KV_HEADS: tl.constexpr,  #
                                        HEAD_SIZE: tl.constexpr, NUM_ROWS: tl.constexpr,  #
                                        STICK: tl.constexpr):
    # The fp8 pool's write. Its page is TOKEN-major, `[block_size, num_kv_heads, head_size]`, so the
    # pool is `NUM_ROWS // NUM_KV_HEADS` token rows of `W` bytes, `S` sticks each, and `row_index`
    # names each written token's row.
    W: tl.constexpr = NUM_KV_HEADS * HEAD_SIZE
    SLOTS: tl.constexpr = NUM_ROWS // NUM_KV_HEADS
    S: tl.constexpr = W // STICK
    if T >= 8:
        # k_rows.index_copy_(0, row_index, key): every token's row narrowed and scattered at once.
        k_dst = tl.make_tensor_descriptor(k_rows, shape=[SLOTS, W], strides=[W, 1],
                                          block_shape=[1, W])
        v_dst = tl.make_tensor_descriptor(v_rows, shape=[SLOTS, W], strides=[W, 1],
                                          block_shape=[1, W])
        k_src = tl.make_tensor_descriptor(key, shape=[T, W], strides=[W, 1], block_shape=[T, W])
        v_src = tl.make_tensor_descriptor(value, shape=[T, W], strides=[W, 1], block_shape=[T, W])
        idx = tl.make_tensor_descriptor(row_index, shape=[T], strides=[1], block_shape=[T]).load([0])
        k_dst.scatter(_to_fp8(k_src.load([0, 0])), idx, 0)
        v_dst.scatter(_to_fp8(v_src.load([0, 0])), idx, 0)
    else:
        # Under the scatter's 8-row floor: each token's row as its `S` sticks, narrowed together
        # and stored as one block at the token's first stick.
        k_dst = tl.make_tensor_descriptor(k_rows, shape=[SLOTS * S, STICK], strides=[STICK, 1],
                                          block_shape=[S, STICK])
        v_dst = tl.make_tensor_descriptor(v_rows, shape=[SLOTS * S, STICK], strides=[STICK, 1],
                                          block_shape=[S, STICK])
        k_src = tl.make_tensor_descriptor(key, shape=[T * S, STICK], strides=[STICK, 1],
                                          block_shape=[S, STICK])
        v_src = tl.make_tensor_descriptor(value, shape=[T * S, STICK], strides=[STICK, 1],
                                          block_shape=[S, STICK])
        for t in tl.static_range(T):
            row = tl.load(row_index + t)
            k_dst.store([row * S, 0], _to_fp8(k_src.load([t * S, 0])))
            v_dst.store([row * S, 0], _to_fp8(v_src.load([t * S, 0])))
