#!/usr/bin/env python
"""Validate the Qwen3.5-VL ViT block-0 ARRANGEMENT against the mlx golden,
reconstructing it in numpy from the real block-0 weights + my validated kernel
formulas (layernorm, qkv split, NeoX rope, per-segment SDPA, gelu_tanh). If this
matches `post_block_0`, the #[vision_forward] DSL arrangement is correct.

Run: ~/.venv/bin/python tools/vision_parity/validate_block.py
(reuses golden/*.npy for inputs/outputs; loads the model only for the weights)
"""
import os, numpy as np, mlx.core as mx
from mlx_vlm import load

G = os.path.join(os.path.dirname(__file__), "golden")
ld = lambda n: np.load(os.path.join(G, n + ".npy"))


def np32(a):
    return np.array(a.astype(mx.float32)) if isinstance(a, mx.array) else np.asarray(a, np.float32)


def layernorm(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def rope(x, freqs):  # x [L,H,D], freqs [L, D/2]  (NeoX, validated)
    L, H, D = x.shape
    half = D // 2
    cf = np.concatenate([np.cos(freqs), np.cos(freqs)], -1)  # [L,D]
    sf = np.concatenate([np.sin(freqs), np.sin(freqs)], -1)
    rh = np.concatenate([-x[..., half:], x[..., :half]], -1)
    return x * cf[:, None, :] + rh * sf[:, None, :]


def sdpa_seg(q, k, v, cu, scale):  # q/k/v [L,H,D]; segments from cu
    L, H, D = q.shape
    out = np.empty_like(q)
    for s in range(len(cu) - 1):
        a, b = cu[s], cu[s + 1]
        for h in range(H):
            sc = (q[a:b, h] @ k[a:b, h].T) * scale  # [seg,seg]
            sc = sc - sc.max(-1, keepdims=True)
            p = np.exp(sc); p /= p.sum(-1, keepdims=True)
            out[a:b, h] = p @ v[a:b, h]
    return out


print("loading model for block-0 weights...")
model, _ = load("mlx-community/Qwen3.5-9B-MLX-4bit")
vt = model.vision_tower
blk = vt.blocks[0]
w = lambda m: np32(m.weight)
bias = lambda m: np32(m.bias)

# inputs/outputs from the golden dump
hs = ld("post_patch_embed") + ld("pos_embeds")    # block-0 input [256,1152]
freqs = ld("rot_pos_emb_table")                    # [256,36]
want = ld("post_block_0")                          # [256,1152]
L, dim = hs.shape
H = 16; D = dim // H; scale = D ** -0.5
cu = np.array([0, L])                              # single image → one segment

# ---- block-0 forward (mirrors mlx-vlm qwen3_vl/vision.py) ----
n1 = layernorm(hs, np32(blk.norm1.weight), np32(blk.norm1.bias))
qkv = n1 @ w(blk.attn.qkv).T + bias(blk.attn.qkv)  # [L, 3*dim]
q = qkv[:, 0:dim].reshape(L, H, D)
k = qkv[:, dim:2 * dim].reshape(L, H, D)
v = qkv[:, 2 * dim:3 * dim].reshape(L, H, D)
q = rope(q, freqs); k = rope(k, freqs)
attn = sdpa_seg(q, k, v, cu, scale).reshape(L, dim)
oproj = attn @ w(blk.attn.proj).T + bias(blk.attn.proj)
hs = hs + oproj
n2 = layernorm(hs, np32(blk.norm2.weight), np32(blk.norm2.bias))
fc1 = gelu_tanh(n2 @ w(blk.mlp.linear_fc1).T + bias(blk.mlp.linear_fc1))
fc2 = fc1 @ w(blk.mlp.linear_fc2).T + bias(blk.mlp.linear_fc2)
got = hs + fc2

err = np.abs(got - want).max()
cos = (got.flatten() @ want.flatten()) / (np.linalg.norm(got) * np.linalg.norm(want))
print(f"block-0 reconstruction vs mlx golden: max_abs_err={err:.4e} cosine={cos:.8f}")
print("->", "MATCH (arrangement correct)" if cos > 0.999 else "MISMATCH")
