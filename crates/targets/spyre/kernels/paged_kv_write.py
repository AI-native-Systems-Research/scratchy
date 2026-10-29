# SPDX-License-Identifier: Apache-2.0
"""THE PAGED KV CACHE WRITE — spyre-inference's `reshape_and_cache_head_major_kernel`
(`spyre_inference/v1/attention/ops/reshape_and_cache_head_major.py`, spyre-inference 297fc48), in
Triton, as a HARDWARE SCATTER.

THE CACHE (`SpyreHeadMajorAttentionBackend.get_kv_cache_shape`): one K pool and one V pool per
layer, each `[num_pages, NKVH, BLOCK, HD]` — a page holds `BLOCK` token slots of every KV head,
head-major. The write views a pool as ROWS, one per `(page, kv head, slot)`
(`kv_slot_views`: `[num_pages * NKVH * BLOCK, HD]`), so token `t` of kv head `h` lands at row

    (page(t) * NKVH + h) * BLOCK + slot(t)          page(t), slot(t) = divmod(slot_mapping[t], BLOCK)

which is `SpyreHeadMajorAttentionImpl.kv_write_index`. spyre-inference states it as one `[T]` index
per head and stores each head with its own `index_copy_`; this kernel states the same rows in one
`[T * NKVH]` table, `(token, head)`-major, so ONE scatter per pool moves every head: the step's keys
`[T, NKVH, HD]` are already `[T * NKVH, HD]` rows in that order.

THE SCATTER IS THE DEVICE'S: `desc.scatter(rows_of_values, row_ids, 0)` lowers to a
`ktdp.construct_indirect_access_tile` over the pool's rows, indexed through the row-id table's view,
and a `ktdp.store` through it — the address is `row_ids[i] * HD`, read from memory, never computed
in the kernel.

FEWER THAN 8 ROWS (a decode step's single token over fewer than 8 kv heads) is below Triton's scatter
floor (`descriptor scatter must have at least 8 rows`), so those rows are stored one at a time, each
at the row the table names: `desc.store([tl.load(row_ids + r), 0], value)` — a store offset by a
value read from memory, which lowers to the same indirect access tile with a one-entry index. Every
row still lands where `kv_write_index` puts it; only the number of index entries per store differs.
"""

import triton
import triton.language as tl


@triton.jit
def paged_kv_write_fwd(desc_k, desc_v, desc_kp, desc_vp, desc_rows,  #
                       T: tl.constexpr, NKVH: tl.constexpr, HD: tl.constexpr,  #
                       POOL_ROWS: tl.constexpr):
    # The pools as rows, one per (page, kv head, slot): the write moves whole rows.
    kp_desc = tl.make_tensor_descriptor(desc_kp, shape=[POOL_ROWS, HD], strides=[HD, 1],
                                        block_shape=[1, HD])
    vp_desc = tl.make_tensor_descriptor(desc_vp, shape=[POOL_ROWS, HD], strides=[HD, 1],
                                        block_shape=[1, HD])
    if T * NKVH >= 8:
        # This step's keys and values, one row per (token, kv head), and the destination row of
        # each — `kv_write_index`'s rows.
        k_desc = tl.make_tensor_descriptor(desc_k, shape=[T * NKVH, HD], strides=[HD, 1],
                                           block_shape=[T * NKVH, HD])
        v_desc = tl.make_tensor_descriptor(desc_v, shape=[T * NKVH, HD], strides=[HD, 1],
                                           block_shape=[T * NKVH, HD])
        rows_desc = tl.make_tensor_descriptor(desc_rows, shape=[T * NKVH], strides=[1],
                                              block_shape=[T * NKVH])
        rows = rows_desc.load([0])
        kp_desc.scatter(k_desc.load([0, 0]), rows, 0)
        vp_desc.scatter(v_desc.load([0, 0]), rows, 0)
    else:
        k_desc = tl.make_tensor_descriptor(desc_k, shape=[T * NKVH, HD], strides=[HD, 1],
                                           block_shape=[1, HD])
        v_desc = tl.make_tensor_descriptor(desc_v, shape=[T * NKVH, HD], strides=[HD, 1],
                                           block_shape=[1, HD])
        for r in tl.static_range(T * NKVH):
            row = tl.load(desc_rows + r)
            kp_desc.store([row, 0], k_desc.load([r, 0]))
            vp_desc.store([row, 0], v_desc.load([r, 0]))
