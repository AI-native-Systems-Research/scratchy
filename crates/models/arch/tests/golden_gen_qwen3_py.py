#!/usr/bin/env python3
"""E2E parity oracle for the Python-as-DSL flip: runs `dsl/qwen3.py` AS
PYTHON under torch and dumps the logits scratchy must reproduce.

The carrier (`crates/models/arch/dsl/qwen3.py`) is the reference
implementation and the spec at once: the compiler PARSES it (never
executes it); this script EXECUTES it. The two sides agree iff the
compiled forward reproduces the torch logits computed from the same
text.

Because the DSL op names (`embed`, `rmsnorm`, `gemm`, `rope_append`,
`attention`, `silu`, `add`) are free functions in the carrier body,
executing it needs exactly those names bound in the caller's globals —
this script provides them as torch implementations reading weights it
loaded from the checkpoint safetensors. The carrier's ambient names
(`input_ids`, `positions`, `kv_cache`, `block_table`, `num_hidden_layers`,
and the weight trees `embed_tokens` / `self_attn` / `mlp` / ...) are the
same way: bound here, consumed by the carrier.

Semantics mirrored from the metal side (crates/targets/metal/src/
cpu_golden.rs — the repo's own CPU goldens for the same kernels):
  - rope_append: GPT-NeoX half-split rotation (x0,x1 = d, d+half), V
    written unrotated; cos/sin rows laid out [cos(half) | sin(half)].
  - attention: causal, scale = 1/sqrt(head_dim), GQA broadcast
    (kv_head = q_head // group_ratio).
  - gemm: y = x @ W^T (weights are [out, in] on disk, untransposed).

The paged-KV addresses (block_table, slot_mapping) mirror
crates/targets/metal/src/paged_kv_layout.rs:
  cache[physical_block, kv_head, slot_in_block, dim]
  slot = block_id * block_size + offset  (global slot id)

Run (torch venv with numpy + safetensors):
    python golden_gen_qwen3_py.py --checkpoint <snapshot-dir> --out goldens/

Outputs (under --out, each .bin raw little-endian + an entry in
goldens.json with shape/dtype):
    input_ids.bin     u32 [n]       token ids
    positions.bin     u32 [n]       RoPE position per token
    logits.bin        f32 [n, vocab] torch oracle logits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

# Model geometry (configs/qwen3/qwen3-0.6b.json — the same values the
# macro bakes as CanonicalParams consts).
HIDDEN = 1024
LAYERS = 28
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
INTER = 3072
VOCAB = 151936
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
ATTN_SCALE = 1.0 / np.sqrt(HEAD_DIM)
GROUP_RATIO = Q_HEADS // KV_HEADS

# KV pool geometry the Rust-side green gate builds (block_size 16,
# BLOCKS_PER_CHUNK 128 — the metal worker's defaults).
BLOCK_SIZE = 16


# ───────────────────────── op shims (torch) ─────────────────────────
# These are the names the carrier body calls. Each mirrors the matching
# metal kernel's semantics exactly (see cpu_golden.rs for the reference
# arithmetic the GPU kernels implement).


def embed(ids, table):
    """`embed(input_ids, embed_tokens)` — row gather, [n, hidden]."""
    return table[ids]


def gemm(x, w):
    """`gemm(x, layer)` — y = x @ w^T; w is [out, in] (F.linear layout)."""
    return torch.nn.functional.linear(x, w)


def rmsnorm(x, w):
    """`rmsnorm(x, norm)` — torch RMSNorm.

    For the QK-norms the DSL writes `rmsnorm(q, q_norm)` with q of
    [..., heads*head_dim] and q_norm of [head_dim]: scratchy's shape
    inference recovers that with a per-head reshape (shape.rs —
    "per-head operations like Qwen3's QK-norm"), so the norm is applied
    per head-slice. Detect it by 1-D gain length < last activation dim
    and divisible into it.
    """
    if w.dim() == 1 and x.shape[-1] % w.shape[0] == 0 and w.shape[0] != x.shape[-1]:
        heads = x.shape[-1] // w.shape[0]
        xv = x.view(*x.shape[:-1], heads, w.shape[0])
        var = xv.pow(2).mean(-1, keepdim=True)
        xv = xv * torch.rsqrt(var + RMS_EPS)
        return (xv * w).view(*x.shape)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + RMS_EPS) * w


def rope_append(q, k, v, positions, rotary, layer_cache):
    """`rope_append(q, k, v, positions, rotary, kv_cache[layer])`.

    GPT-NeoX half-split rotation (cpu_golden.rs `rope_append`): pair
    (d, d+half); Q and K rotate in place, V passes through unrotated.
    K and V are ALSO appended to the layer's paged cache at this step's
    slot_mapping slots — that side effect happens here so the attention
    shim below reads a cache in exactly the state the metal forward
    leaves it in.
    """
    n = q.shape[0]
    half = HEAD_DIM // 2
    pos = torch.as_tensor(positions, dtype=torch.long)
    cos_t, sin_t = rotary  # [max_pos, half] each
    cos = cos_t[pos]  # [n, half]
    sin = sin_t[pos]  # [n, half]
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    for name, t in (("q", q), ("k", k)):
        heads = Q_HEADS if name == "q" else KV_HEADS
        tv = t.view(n, heads, 2, half)  # (x0-half, x1-half) pairing
        x0, x1 = tv[:, :, 0, :], tv[:, :, 1, :]
        r0 = x0 * cos.unsqueeze(1) - x1 * sin.unsqueeze(1)
        r1 = x1 * cos.unsqueeze(1) + x0 * sin.unsqueeze(1)
        out = torch.stack((r0, r1), dim=2).reshape(n, heads * HEAD_DIM)
        if name == "q":
            out_q = out
        else:
            out_k = out
    # Paged append (paged_kv_layout.rs): slot = block*block_size+offset,
    # cache[physical_block, kv_head, slot_in_block, dim].
    kcache, vcache = layer_cache
    for t in range(n):
        slot = SLOT_MAPPING[t]
        blk, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
        kcache[blk, :, off, :] = out_k[t].view(KV_HEADS, HEAD_DIM)
        vcache[blk, :, off, :] = v[t].view(KV_HEADS, HEAD_DIM)
    return out_q, out_k, v


def attention(q, k, v, layer_cache, block_table):
    """`attention(q, k, v, kv_cache[layer], block_table)`.

    Prefill semantics (cpu_golden.rs `attention_prefill_paged`): causal
    over the full paged K/V — token i attends to cache positions [0, i]
    resolved through block_table. GQA: kv_head = q_head // group_ratio.
    Scale 1/sqrt(head_dim). The k/v arguments this shim receives are
    the (already rotated / passthrough) per-step tensors; the read
    source of truth is the paged cache (rope_append wrote it above).
    """
    kcache, vcache = layer_cache
    n = q.shape[0]
    qv = q.view(n, Q_HEADS, HEAD_DIM)
    out = torch.empty_like(qv)
    bt = block_table
    for t in range(n):
        pos = POSITIONS[t]
        attend = pos + 1  # causal: [0, pos]
        # Gather the [attend, KV_HEADS, HEAD_DIM] K/V window for this
        # token's sequence through the block table.
        ks = torch.empty(attend, KV_HEADS, HEAD_DIM, dtype=kcache.dtype)
        vs = torch.empty_like(ks)
        for j in range(attend):
            blk, off = j // BLOCK_SIZE, j % BLOCK_SIZE
            pb = bt[blk]
            ks[j] = kcache[pb, :, off, :]
            vs[j] = vcache[pb, :, off, :]
        for h in range(Q_HEADS):
            kv_h = h // GROUP_RATIO
            scores = (ks[:, kv_h, :] @ qv[t, h]) * ATTN_SCALE
            probs = torch.softmax(scores, dim=0)
            out[t, h] = probs @ vs[:, kv_h, :]
    return out.reshape(n, Q_HEADS * HEAD_DIM)


def silu(x):
    return torch.nn.functional.silu(x)


def add(a, b):
    return a + b


def forward(fn):
    """`@forward` under torch is the identity — it carries metadata the
    COMPILER reads (the `#[forward]` attribute's replacement); executing
    the carrier cares only about the body."""
    return fn


def vision_forward(*, workloads=None, sk_buckets=None, processor=None, pixel_pack=None):
    """`@vision_forward(...)` likewise — kwargs are compile-time metadata."""
    def deco(fn):
        return fn
    return deco


class Tree:
    """Attribute-style weight tree so the carrier's `self_attn.q_proj[layer]`
    reads work under plain Python (the compiler lowers the same chains)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ───────────────────────── harness ─────────────────────────


def build_rotary(max_pos):
    """[max_pos, half] cos/sin tables — mirrors RotaryCache (layers/
    rotary.rs): inv_freq[i] = theta^(-2i/head_dim), row = pos*inv_freq,
    cos then sin stacked [cos | sin] is NOT this layout; the kernel
    reads row[d] = cos, row[half+d] = sin, so build [max_pos, head_dim]
    and split halves here."""
    half = HEAD_DIM // 2
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, half, dtype=torch.float64) * 2.0 / HEAD_DIM))
    pos = torch.arange(max_pos, dtype=torch.float64).unsqueeze(1)
    ang = pos * inv.unsqueeze(0)
    table = torch.cat([ang.cos(), ang.sin()], dim=1)  # [max_pos, head_dim]
    # The rope shim indexes cos/sin separately per half.
    return table[:, :half].float(), table[:, half:].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="snapshot dir with model.safetensors")
    ap.add_argument("--out", default=Path(__file__).parent / "goldens", type=Path)
    ap.add_argument("--num-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    args = ap.parse_args()

    n = args.num_tokens
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    sd = load_file(str(Path(args.checkpoint) / "model.safetensors"))
    # tie_word_embeddings: lm_head shares model.embed_tokens.weight.
    w_embed = sd["model.embed_tokens.weight"].float()

    # Weight trees keyed the way the carrier names them.
    def L(key):
        return sd[f"model.layers.{key}"].float()

    # Ambient names the carrier body reads.
    global input_ids, positions, num_hidden_layers, kv_cache, block_table
    global embed_tokens, self_attn, mlp, rotary, POSITIONS, SLOT_MAPPING
    global input_layernorm, post_attention_layernorm, norm, lm_head

    num_hidden_layers = LAYERS

    # Deterministic, low-range token ids (vocab is large; keep the ids
    # small so the embed gather can't be all-zeros-by-luck).
    input_ids = torch.from_numpy(
        rng.integers(low=1, high=1_000, size=n).astype(np.int64)
    )
    positions = list(range(n))  # plain continuation from 0
    POSITIONS = positions

    # Paged KV: one page-chain for the single sequence. Physical blocks
    # 0.. ceil(n/block_size)-1 in order; slot i maps to block i//bs, off
    # i%bs — i.e. the identity layout the engine produces for one
    # fresh sequence. Rust side mirrors this exactly.
    n_blocks_needed = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = 128  # pool size the Rust gate builds
    block_table = [i for i in range(n_blocks_needed)]  # row 0 of table
    SLOT_MAPPING = list(range(n))

    # Per-layer paged caches in the pool's layout
    # [num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM].
    kv_cache = []
    for _ in range(LAYERS):
        k = torch.zeros(num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM)
        v = torch.zeros(num_blocks, KV_HEADS, BLOCK_SIZE, HEAD_DIM)
        kv_cache.append((k, v))

    cos_t, sin_t = build_rotary(max_pos=n)
    rotary = (cos_t, sin_t)

    # Carrier weight trees: the body references `self_attn.q_proj[layer]`
    # etc. — plain Python lists of torch tensors.
    self_attn = Tree(
        q_proj=[L(f"{i}.self_attn.q_proj.weight") for i in range(LAYERS)],
        k_proj=[L(f"{i}.self_attn.k_proj.weight") for i in range(LAYERS)],
        v_proj=[L(f"{i}.self_attn.v_proj.weight") for i in range(LAYERS)],
        o_proj=[L(f"{i}.self_attn.o_proj.weight") for i in range(LAYERS)],
        q_norm=[L(f"{i}.self_attn.q_norm.weight") for i in range(LAYERS)],
        k_norm=[L(f"{i}.self_attn.k_norm.weight") for i in range(LAYERS)],
    )
    mlp = Tree(
        gate_proj=[L(f"{i}.mlp.gate_proj.weight") for i in range(LAYERS)],
        up_proj=[L(f"{i}.mlp.up_proj.weight") for i in range(LAYERS)],
        down_proj=[L(f"{i}.mlp.down_proj.weight") for i in range(LAYERS)],
    )
    input_layernorm = [L(f"{i}.input_layernorm.weight") for i in range(LAYERS)]
    post_attention_layernorm = [
        L(f"{i}.post_attention_layernorm.weight") for i in range(LAYERS)
    ]
    norm = sd["model.norm.weight"].float()
    embed_tokens = w_embed
    lm_head = w_embed  # tied

    # ── EXECUTE THE CARRIER AS PYTHON ──
    carrier = (Path(__file__).parent.parent / "dsl" / "qwen3.py").read_text()
    env = dict(globals())
    exec(compile(carrier, "qwen3.py", "exec"), env)
    logits = env["qwen3"]()

    assert logits.shape == (n, VOCAB), f"logits shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def dump(name, arr):
        a = arr.detach().contiguous().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
        a.astype(a.dtype).tofile(args.out / f"{name}.bin")
        manifest[name] = {"shape": list(a.shape), "dtype": str(a.dtype)}

    dump("input_ids", input_ids.to(torch.int64))
    dump("positions", np.asarray(positions, dtype=np.int64))
    dump("logits", logits.float())
    # block table row 0 + slot mapping, so the Rust side can rebuild the
    # same addresses (they are inputs, not just fixtures).
    dump("block_table", np.asarray(block_table + [0] * (128 - len(block_table)), dtype=np.int64))
    dump("slot_mapping", np.asarray(SLOT_MAPPING, dtype=np.int64))

    (args.out / "goldens.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} goldens to {args.out}:")
    for k, v in manifest.items():
        print(f"  {k}: {v['shape']} {v['dtype']}")


if __name__ == "__main__":
    main()
