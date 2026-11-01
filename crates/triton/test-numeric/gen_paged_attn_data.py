# SPDX-License-Identifier: Apache-2.0
"""THE PAGED SERVING STEP'S REFERENCE, produced by SPYRE-INFERENCE'S OWN KERNELS.

    python3 gen_paged_attn_data.py --spyre-inference <checkout> <out-root> [<config> ...]

One step of `SpyreHeadMajorAttentionImpl` (spyre-inference's default decoder backend): the KV cache
write `reshape_and_cache_head_major_kernel`, then the page walk -- `page_attn_head_major_decode_kernel`
at one query row, `page_attn_head_major_prefill_kernel` past it -- run AS WRITTEN, imported from the
checkout, on CPU. Nothing here restates their arithmetic: this script builds the backend's metadata
(the page table, the mask stack, the write rows) the way `SpyreAttentionMetadataBuilder.build` and
`kv_write_index` do, calls their kernels, and dumps what comes back. Their `walk_tiles` runs its
Python walk (`SPYRE_ATTN_FOR_EACH_TILE=0`), the same block body `for_each_tile` traces.

Per config, into `<out-root>/<config>/`:

  * the step's inputs (the numeric test binds them as the Triton kernels' operands):
    `q.bin` `[MQ, NQH*HD]` f16, `k_new.bin`/`v_new.bin` `[T*NKVH, HD]` f16, `kp_pre.bin`/`vp_pre.bin`
    `[NUM_PAGES, NKVH, BLOCK, HD]` f16 (the pools before this step, holding the earlier context),
    `write_rows.bin` `[T*NKVH]` i32, `page_table.bin` `[NUM_BLOCKS, 32]` i32, `mask.bin`
    `[NUM_BLOCKS*MQ, BLOCK]` f16;
  * the references: `kp_post.bin`/`vp_post.bin` (the pools after their write), `out_f32.bin`
    (their kernels on f32 copies of the inputs -- the algorithm's answer) and `out_f16.bin` (their
    kernels in the model dtype, as served), both `[MQ, NQH*HD]`;
  * `meta.json` (shapes, dtypes, the step's metadata, spyre-inference's commit) and `sha256.txt`.

The BATCHED configs (`BATCHED_CONFIGS`) are one decode step of several sequences through
`batched_decode_head_major_kernel` (`SPYRE_BATCHED_DECODE=1`, their default): the metadata
`SpyreAttentionMetadataBuilder.build` builds for a decode batch (`spyre_attn.py:892-967`) at the
sequence and block buckets, their kernel on it, and the same metadata in the tiled walk's layout
(`_mirror_batched_decode_indices`) for the Triton kernel, which is what is recorded:
`q.bin` `[B, NQH*HD]`, `kp.bin`/`vp.bin` the pools after the step's writes, `rep_row_ids.bin`
`[E*P]` i32 and `chunk_page_ids.bin` `[C, E*P, 1]` i32 (`P` the Triton kernel's `ROWS_PER_PAGE`: 1, or
NKVH for a chunk under Triton's gather floor in pages, its tables folded by kv head), `mask.bin`
`[C*E*NKVH, BLOCK]` (the `[C, J, B, NKVH, 1, BLOCK]` stack), `out_f16.bin`/`out_f32.bin`
`[B, NQH*HD]`.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import torch

SEED = 0
STICK_I32 = 32  # `layout.INT32_ELEMS_PER_STICK`: a page-table row is one int32 stick wide

# Granite-3.1-2b's attention geometry (32 query heads, 8 KV heads, head size 64) at the backend's
# 64-slot page; a 4-page pool, so the scattered page tables below are a permutation of it.
GEOM = dict(NQH=32, NKVH=8, HD=64, BLOCK=64, NUM_PAGES=4)

# (context already cached, this step's query length, the sequence's block table; a config's own
# `geom` overrides `GEOM`).
CONFIGS = {
    # One generated token after 150 cached ones: 151 keys, three pages, the last one partly valid.
    "paged_attn_decode": dict(context=150, query=1, block_table=[3, 0, 2]),
    # The same step at TWO KV heads (qwen2.5-0.5b's attention: 14 query heads, 2 KV heads, head
    # size 64): its write moves 2 rows, under Triton's scatter floor of 8.
    "paged_attn_decode_kv2": dict(context=150, query=1, block_table=[3, 0, 2],
                                  geom=dict(NQH=14, NKVH=2)),
    # A 32-token prompt chunk continuing a 100-token prefix: causal inside the chunk, three pages.
    "paged_attn_prefill": dict(context=100, query=32, block_table=[1, 3, 0]),
    # A first 32-token prompt chunk: one page, causal, past-length masked.
    "paged_attn_prefill_first": dict(context=0, query=32, block_table=[2]),
}


# Several sequences' decode steps in one launch (the contexts already cached, one generated token
# each; the sequence bucket the batch pads to; the pool's pages, more than `B * J` so the batch is
# eligible -- `_batched_decode_preconditions_met`).
#
# At TWO KV heads (qwen2.5-0.5b's attention, as `paged_attn_decode_kv2`): the numeric test runs the
# kernel as one grid-[1] program on one emulated core, whose LX must hold a chunk's gathered pages
# whole -- 16 KB a page here. At granite's 8 KV heads a chunk of 16 pages is 1 MB per pool, which the
# card spreads over its cores (spyre-inference sizes a chunk's entries to the 32 cores); granite's
# geometry runs through the production executor in the spyre target's executed paged step.
KV2 = dict(NQH=14, NKVH=2)
BATCHED_CONFIGS = {
    # Three sequences in the 4-sequence bucket (the fourth row padded), one chunk: blocks bucket 4,
    # J = 4, C = 1, E = 16.
    "batched_decode": dict(contexts=[150, 40, 100], seqs_bucket=4, num_pages=17, geom=KV2),
    # Four sequences over two chunks, so each block slot keeps its own softmax and the slots merge
    # after the walk: blocks bucket 16, J = 8, C = 2, E = 32.
    "batched_decode_chunks": dict(contexts=[600, 70, 300, 1], seqs_bucket=4, num_pages=33,
                                  geom=KV2),
    # Two sequences in one block each: J = 1, E = 2, four kv rows at two KV heads, under Triton's
    # gather floor of 8, so the Triton kernel's chunk takes 2 block slots (the extra one masked and
    # gathering page 0) and E = 4.
    "batched_decode_floor": dict(contexts=[50, 30], seqs_bucket=2, num_pages=9, geom=KV2),
}

SPYRE_CORE_COUNT = 32  # `spyre_attn_bucketer._SPYRE_CORE_COUNT`
TRITON_GATHER_FLOOR = 8  # `semantic.py`'s `descriptor_gather`: at least 8 rows


def load_spyre_inference(root):
    """Import spyre-inference's three head-major kernels from the checkout at `root`.

    Their modules import `walk_tiles` from `tile_loop`, whose only dependency is
    `spyre_inference.envs.SPYRE_ATTN_FOR_EACH_TILE`; that one attribute is stated here (0, the
    Python walk) so the package's vLLM-importing `__init__` never runs. Every file executed is
    theirs, unmodified.
    """
    ops = os.path.join(root, "spyre_inference", "v1", "attention", "ops")
    for pkg in ("spyre_inference", "spyre_inference.v1", "spyre_inference.v1.attention",
                "spyre_inference.v1.attention.ops"):
        m = types.ModuleType(pkg)
        m.__path__ = []
        sys.modules[pkg] = m
    envs = types.ModuleType("spyre_inference.envs")
    envs.SPYRE_ATTN_FOR_EACH_TILE = False
    sys.modules["spyre_inference.envs"] = envs
    sys.modules["spyre_inference"].envs = envs

    def load(name):
        full = f"spyre_inference.v1.attention.ops.{name}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(ops, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    load("tile_loop")
    return (load("reshape_and_cache_head_major"), load("page_attn_head_major_prefill"),
            load("page_attn_head_major_decode"), load("batched_decode_head_major"))


def write_index(slots, nkvh, block):
    """`SpyreHeadMajorAttentionImpl.kv_write_index`: per KV head, the row of each token's slot."""
    blk = torch.div(slots, block, rounding_mode="floor")
    base = blk * nkvh * block + slots % block
    return [base + h * block for h in range(nkvh)]


def mask_stack(kv_len, query, context, num_blocks, block, causal):
    """`SpyreAttentionMetadataBuilder._build_attention_mask`, then its per-block tiles."""
    dtype = torch.float16
    q_pos = torch.arange(query)
    kv_pos = torch.arange(num_blocks * block)
    attend = (kv_pos.unsqueeze(0) < kv_len).expand(query, -1)
    if causal:
        attend = attend & (kv_pos.unsqueeze(0) <= (context + q_pos).unsqueeze(1))
    mask = torch.where(~attend, torch.tensor(torch.finfo(dtype).min, dtype=dtype),
                       torch.tensor(0.0, dtype=dtype))
    # `.reshape(aligned, -1, block).transpose(0, 1)`: one [query, block] tile per block.
    return mask.reshape(query, num_blocks, block).transpose(0, 1).contiguous()


def geometry(cfg):
    """The step's attention geometry: `GEOM`, under the config's own overrides."""
    return {**GEOM, **cfg.get("geom", {})}


def step(kernels, cfg, dtype):
    """One serving step through spyre-inference's kernels at `dtype`; returns every buffer."""
    reshape_and_cache, prefill, decode, _ = kernels
    g = geometry(cfg)
    nqh, nkvh, hd, block, npages = g["NQH"], g["NKVH"], g["HD"], g["BLOCK"], g["NUM_PAGES"]
    context, query, table = cfg["context"], cfg["query"], cfg["block_table"]
    kv_len = context + query
    num_blocks = (kv_len + block - 1) // block
    assert num_blocks == len(table), (num_blocks, table)

    gen = torch.Generator().manual_seed(SEED)
    # The earlier context's keys/values and this step's: one draw each, so every dtype sees the
    # same numbers.
    k_all = torch.randn(kv_len, nkvh, hd, generator=gen).to(torch.float16)
    v_all = torch.randn(kv_len, nkvh, hd, generator=gen).to(torch.float16)
    q = (torch.randn(query, nqh, hd, generator=gen)).to(torch.float16)
    k_all, v_all, q = k_all.to(dtype), v_all.to(dtype), q.to(dtype)

    def slots_of(positions):
        pos = torch.as_tensor(positions, dtype=torch.int64)
        return torch.as_tensor(table, dtype=torch.int64)[pos // block] * block + pos % block

    # The pools as the backend allocates them (zeroed), holding the context the earlier steps wrote.
    kp = torch.zeros(npages, nkvh, block, hd, dtype=dtype)
    vp = torch.zeros(npages, nkvh, block, hd, dtype=dtype)
    k_rows, v_rows = kp.view(-1, hd), vp.view(-1, hd)
    if context:
        reshape_and_cache.reshape_and_cache_head_major_kernel(
            k_all[:context], v_all[:context], k_rows, v_rows,
            write_index(slots_of(range(context)), nkvh, block))
    kp_pre, vp_pre = kp.clone(), vp.clone()

    # THIS STEP: the write, then the attention.
    new_slots = slots_of(range(context, kv_len))
    rows = write_index(new_slots, nkvh, block)
    reshape_and_cache.reshape_and_cache_head_major_kernel(
        k_all[context:], v_all[context:], k_rows, v_rows, rows)

    page_table = torch.zeros(num_blocks, STICK_I32, dtype=torch.int32)
    page_table[:, 0] = torch.as_tensor(table, dtype=torch.int32)
    masks = mask_stack(kv_len, query, context, num_blocks, block, causal=query > 1).to(dtype)
    row_index = torch.arange(query, dtype=torch.int32)
    scale = 1.0 / hd ** 0.5
    if query == 1:
        folded = (kp.view(-1, block, hd), vp.view(-1, block, hd))
        pool = torch.arange(npages * nkvh, dtype=torch.int32).reshape(npages, nkvh, 1)
        out = decode.page_attn_head_major_decode_kernel(
            q, row_index, folded[0], folded[1], page_table, pool, masks, scale, num_blocks, 1,
            nqh, nkvh, hd, block)
    else:
        out = prefill.page_attn_head_major_prefill_kernel(
            q, row_index, kp, vp, page_table, masks, scale, num_blocks, query, nqh, nkvh, hd)

    # The Triton kernels' layouts: the write's rows `(token, head)`-major, the mask stack flat.
    write_rows = torch.stack(rows, dim=1).reshape(-1).to(torch.int32)
    return dict(
        q=q.reshape(query, nqh * hd), k_new=k_all[context:].reshape(-1, hd),
        v_new=v_all[context:].reshape(-1, hd), kp_pre=kp_pre, vp_pre=vp_pre,
        write_rows=write_rows, page_table=page_table, mask=masks.reshape(-1, block),
        kp_post=kp, vp_post=vp, out=out.reshape(query, nqh * hd),
        meta=dict(kv_len=kv_len, num_blocks=num_blocks, scale=scale, new_slots=new_slots.tolist()),
    )


def batched_decode_chunking(b_seqs, b_blocks):
    """`spyre_attn_bucketer.batched_decode_chunking`: `(blocks_per_chunk, num_chunks)`."""
    blocks_per_chunk = max(1, min(SPYRE_CORE_COUNT // b_seqs, b_blocks))
    return blocks_per_chunk, (b_blocks + blocks_per_chunk - 1) // blocks_per_chunk


def blocks_bucket(num_blocks):
    """`SpyreAttnBucketer.find_blocks_bucket` over the default kv buckets: the powers of two."""
    b = 1
    while b < num_blocks:
        b *= 2
    return b


def batched_metadata(tables, blocks, stacks, b_seqs, b_blocks, j, nkvh, block, dtype):
    """`SpyreAttentionMetadataBuilder.build`'s decode-batch precomputes at `j` block slots per
    chunk: `(rep_row_ids, chunk_page_ids [C*J, B], mask [B, C*J, block])`."""
    n = len(tables)
    c = (b_blocks + j - 1) // j
    padded_batch_blocks = c * j
    query_row_ids = torch.zeros(b_seqs, dtype=torch.int32)
    query_row_ids[:n] = torch.arange(n, dtype=torch.int32)
    rep_row_ids = query_row_ids.repeat(j)
    block_ids_padded = torch.zeros(padded_batch_blocks, b_seqs, dtype=torch.int32)
    mask_bs_bb = torch.full((b_seqs, padded_batch_blocks, block), float("-inf"), dtype=dtype)
    for s, (table, nb, stack) in enumerate(zip(tables, blocks, stacks)):
        n_use = min(nb, b_blocks)
        block_ids_padded[:n_use, s] = torch.as_tensor(table[:n_use], dtype=torch.int32)
        mask_bs_bb[s, :n_use] = stack[:n_use, 0]
    mask_bs_bb[n:, 0] = torch.finfo(dtype).min
    return rep_row_ids, block_ids_padded, mask_bs_bb, c


def batched_step(kernels, cfg, dtype):
    """One decode step of several sequences through spyre-inference's batched kernel at `dtype`."""
    reshape_and_cache, _, _, batched = kernels
    g = {**geometry(cfg), "NUM_PAGES": cfg["num_pages"]}
    nqh, nkvh, hd, block, npages = g["NQH"], g["NKVH"], g["HD"], g["BLOCK"], g["NUM_PAGES"]
    contexts, b_seqs = cfg["contexts"], cfg["seqs_bucket"]
    n = len(contexts)
    kv_lens = [ctx + 1 for ctx in contexts]
    blocks = [(kv + block - 1) // block for kv in kv_lens]
    b_blocks = blocks_bucket(max(blocks))
    j, c = batched_decode_chunking(b_seqs, b_blocks)
    assert b_seqs * j < npages, "`_batched_decode_preconditions_met`: B * J < num_pages"

    gen = torch.Generator().manual_seed(SEED)
    # Every sequence's blocks on its own pages, scattered over the pool.
    perm = torch.randperm(npages, generator=gen).tolist()
    tables, at = [], 0
    for nb in blocks:
        tables.append(perm[at:at + nb])
        at += nb
    kp = torch.zeros(npages, nkvh, block, hd, dtype=dtype)
    vp = torch.zeros(npages, nkvh, block, hd, dtype=dtype)
    k_rows, v_rows = kp.view(-1, hd), vp.view(-1, hd)
    q = torch.zeros(b_seqs, nqh, hd, dtype=dtype)
    stacks = []
    for s, (kv_len, table) in enumerate(zip(kv_lens, tables)):
        k_all = torch.randn(kv_len, nkvh, hd, generator=gen).to(torch.float16).to(dtype)
        v_all = torch.randn(kv_len, nkvh, hd, generator=gen).to(torch.float16).to(dtype)
        q[s] = torch.randn(nqh, hd, generator=gen).to(torch.float16).to(dtype)
        pos = torch.arange(kv_len)
        slots = torch.as_tensor(table, dtype=torch.int64)[pos // block] * block + pos % block
        # The earlier steps' writes and this step's, through their write kernel.
        reshape_and_cache.reshape_and_cache_head_major_kernel(
            k_all, v_all, k_rows, v_rows, write_index(slots, nkvh, block))
        stacks.append(mask_stack(kv_len, 1, kv_len - 1, blocks[s], block, causal=False).to(dtype))

    # THEIR KERNEL, at their chunking: the Python walk's metadata, the mask's KV axis broadcast.
    rep, ids, mask_bs_bb, _ = batched_metadata(tables, blocks, stacks, b_seqs, b_blocks, j, nkvh,
                                               block, dtype)
    mask_by_chunk = (mask_bs_bb.transpose(0, 1).unsqueeze(2).unsqueeze(3)
                     .expand(c * j, b_seqs, 1, 1, block).contiguous())
    scale = 1.0 / hd ** 0.5
    out = batched.batched_decode_head_major_kernel(
        q.reshape(b_seqs, nqh * hd), rep, kp, vp, ids, mask_by_chunk, scale, b_seqs, j, nkvh,
        nqh // nkvh, block, hd)

    # THE TRITON KERNEL'S METADATA: the same builder in the tiled walk's layout
    # (`_mirror_batched_decode_indices`), the mask `[C, J, B, KV, 1, block]` with its KV axis
    # materialized. A chunk of fewer than 8 entries is under Triton's gather floor in pages, so the
    # kernel gathers its pages' kv heads (`ROWS_PER_PAGE = KV`) and both index tables are FOLDED by
    # kv head (entry e's kv head a at e * KV + a holding x * KV + a, the per-sequence decode
    # kernel's `kv_row_pool` form -- ours: their batched path never folds): the query rows
    # `[E * KV]`, the page ids `[C, E * KV]`. The floor then counts those rows, so the block slots
    # are theirs unless `B * J * KV < 8`.
    per = nkvh if b_seqs * j < TRITON_GATHER_FLOOR else 1
    jt = max(j, -(-TRITON_GATHER_FLOOR // (b_seqs * per)))
    rep_t, ids_t, mask_t, ct = batched_metadata(tables, blocks, stacks, b_seqs, b_blocks, jt, nkvh,
                                                block, dtype)
    e = jt * b_seqs
    mask_tiled = (mask_t.transpose(0, 1).unsqueeze(2).unsqueeze(3)
                  .expand(ct * jt, b_seqs, nkvh, 1, block).contiguous()
                  .reshape(ct * e * nkvh, block))
    rows = torch.arange(per, dtype=torch.int32)

    def fold(x):
        return (x.reshape(-1, 1) * per + rows).reshape(-1)

    return dict(
        q=q.reshape(b_seqs, nqh * hd), kp=kp, vp=vp, rep_row_ids=fold(rep_t),
        chunk_page_ids=fold(ids_t).reshape(ct, e * per, 1), mask=mask_tiled,
        out=out.reshape(b_seqs, nqh * hd),
        meta=dict(kv_lens=kv_lens, blocks=blocks, tables=tables, blocks_bucket=b_blocks,
                  theirs=dict(blocks_per_chunk=j, num_chunks=c),
                  triton=dict(blocks_per_chunk=jt, num_chunks=ct, entries=e,
                              rows_per_page=per), scale=scale,
                  geometry=g),
    )


def dump(t, path, dtype):
    t = t.to({"f16": torch.float16, "f32": torch.float32, "i32": torch.int32}[dtype]).contiguous()
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())
    return list(t.shape)


def generate_batched(kernels, name, cfg, out_root, spyre_sha):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    served = batched_step(kernels, cfg, torch.float16)
    exact = batched_step(kernels, cfg, torch.float32)
    files = {}
    for key, dtype in [("q", "f16"), ("kp", "f16"), ("vp", "f16"), ("rep_row_ids", "i32"),
                       ("chunk_page_ids", "i32"), ("mask", "f16")]:
        files[key] = dict(file=f"{key}.bin", dtype=dtype,
                          shape=dump(served[key], os.path.join(out, f"{key}.bin"), dtype))
    files["out_f16"] = dict(file="out_f16.bin", dtype="f16",
                            shape=dump(served["out"], os.path.join(out, "out_f16.bin"), "f16"))
    files["out_f32"] = dict(file="out_f32.bin", dtype="f32",
                            shape=dump(exact["out"], os.path.join(out, "out_f32.bin"), "f32"))
    shas = {}
    for f in sorted(v["file"] for v in files.values()):
        with open(os.path.join(out, f), "rb") as fh:
            shas[f] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(out, "sha256.txt"), "w") as fh:
        for f, h in shas.items():
            fh.write(f"{h}  {f}\n")
    meta = dict(config=name, geometry=served["meta"]["geometry"], step=cfg,
                derived=served["meta"], seqs=cfg["seqs_bucket"], files=files, seed=SEED,
                torch=torch.__version__, spyre_inference=spyre_sha,
                generated_by="crates/triton/test-numeric/gen_paged_attn_data.py",
                reference=("spyre-inference reshape_and_cache_head_major_kernel, then "
                           "batched_decode_head_major_kernel, unmodified, "
                           "SPYRE_ATTN_FOR_EACH_TILE=0"))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    gap = (served["out"].float() - exact["out"]).abs().max().item()
    d = served["meta"]
    print(f"{name}: blocks={d['blocks']} theirs J,C={d['theirs']['blocks_per_chunk']},"
          f"{d['theirs']['num_chunks']} triton J,C={d['triton']['blocks_per_chunk']},"
          f"{d['triton']['num_chunks']} |f16 - f32| max {gap:.3e}")


def generate(kernels, name, cfg, out_root, spyre_sha):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    served = step(kernels, cfg, torch.float16)
    exact = step(kernels, cfg, torch.float32)
    files = {}
    for key, dtype in [("q", "f16"), ("k_new", "f16"), ("v_new", "f16"), ("kp_pre", "f16"),
                       ("vp_pre", "f16"), ("write_rows", "i32"), ("page_table", "i32"),
                       ("mask", "f16"), ("kp_post", "f16"), ("vp_post", "f16")]:
        files[key] = dict(file=f"{key}.bin", dtype=dtype,
                          shape=dump(served[key], os.path.join(out, f"{key}.bin"), dtype))
    files["out_f16"] = dict(file="out_f16.bin", dtype="f16",
                            shape=dump(served["out"], os.path.join(out, "out_f16.bin"), "f16"))
    files["out_f32"] = dict(file="out_f32.bin", dtype="f32",
                            shape=dump(exact["out"], os.path.join(out, "out_f32.bin"), "f32"))
    shas = {}
    for f in sorted(v["file"] for v in files.values()):
        with open(os.path.join(out, f), "rb") as fh:
            shas[f] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(out, "sha256.txt"), "w") as fh:
        for f, h in shas.items():
            fh.write(f"{h}  {f}\n")
    meta = dict(config=name, geometry=geometry(cfg), step=cfg, derived=served["meta"],
                mq=cfg["query"], files=files, seed=SEED, torch=torch.__version__,
                spyre_inference=spyre_sha,
                generated_by="crates/triton/test-numeric/gen_paged_attn_data.py",
                reference=("spyre-inference reshape_and_cache_head_major_kernel, then "
                           + ("page_attn_head_major_decode_kernel" if cfg["query"] == 1
                              else "page_attn_head_major_prefill_kernel")
                           + ", unmodified, SPYRE_ATTN_FOR_EACH_TILE=0"))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    gap = (served["out"].float() - exact["out"]).abs().max().item()
    print(f"{name}: kv_len={served['meta']['kv_len']} blocks={served['meta']['num_blocks']} "
          f"|f16 - f32| max {gap:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spyre-inference", required=True)
    ap.add_argument("out_root")
    ap.add_argument("configs", nargs="*")
    a = ap.parse_args()
    root = os.path.abspath(a.spyre_inference)
    sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    kernels = load_spyre_inference(root)
    for name in a.configs or [*CONFIGS, *BATCHED_CONFIGS]:
        if name in BATCHED_CONFIGS:
            generate_batched(kernels, name, BATCHED_CONFIGS[name], a.out_root, sha)
        else:
            generate(kernels, name, CONFIGS[name], a.out_root, sha)


if __name__ == "__main__":
    main()
