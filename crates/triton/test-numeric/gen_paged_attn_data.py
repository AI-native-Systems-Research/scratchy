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

  * the step's inputs, in the Triton kernels' own layouts (`paged_kv_write.py`, `paged_attn.py`):
    `q.bin` `[MQ, NQH*HD]` f16, `k_new.bin`/`v_new.bin` `[T*NKVH, HD]` f16, `kp_pre.bin`/`vp_pre.bin`
    `[NUM_PAGES, NKVH, BLOCK, HD]` f16 (the pools before this step, holding the earlier context),
    `write_rows.bin` `[T*NKVH]` i32, `page_table.bin` `[NUM_BLOCKS, 32]` i32, `mask.bin`
    `[NUM_BLOCKS*MQ, BLOCK]` f16;
  * the references: `kp_post.bin`/`vp_post.bin` (the pools after their write), `out_f32.bin`
    (their kernels on f32 copies of the inputs -- the algorithm's answer) and `out_f16.bin` (their
    kernels in the model dtype, as served), both `[MQ, NQH*HD]`;
  * `meta.json` (shapes, dtypes, the step's metadata, spyre-inference's commit) and `sha256.txt`.
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
            load("page_attn_head_major_decode"))


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
    reshape_and_cache, prefill, decode = kernels
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


def dump(t, path, dtype):
    t = t.to({"f16": torch.float16, "f32": torch.float32, "i32": torch.int32}[dtype]).contiguous()
    with open(path, "wb") as f:
        f.write(t.numpy().tobytes())
    return list(t.shape)


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
    for name in a.configs or list(CONFIGS):
        generate(kernels, name, CONFIGS[name], a.out_root, sha)


if __name__ == "__main__":
    main()
