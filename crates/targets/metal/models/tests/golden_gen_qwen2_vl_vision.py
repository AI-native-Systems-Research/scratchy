#!/usr/bin/env python3
"""Generate Qwen2-VL-2B vision-encoder goldens for scratchy to diff against.

Run with the project venv:
    ~/vllm/.venv/bin/python golden_gen_qwen2_vl_vision.py [--out DIR]

Inputs are deterministic — a synthetic 224x224 RGB pattern saved next to the
goldens so the scratchy-side path can decode the same PNG bytes through its
own image processor for a clean A/B.

Outputs (raw little-endian, paired with goldens.json describing shapes/dtypes):
    input.png                  — synthetic RGB image, 224x224.
    patches.bin                — bf16, [L, 1176]; HF Qwen2VLImageProcessor output.
    grid_thw.bin               — i64, [N, 3]; per-image (T, H, W) in patch units.
    rotary_pos_emb_half.bin    — f32, [L, 40]; rope before cat(rope,rope) — what
                                 scratchy's vision_rope_apply expects as cos/sin
                                 *argument*, indexed via cos()/sin().
    cos.bin, sin.bin           — bf16, [L, 80]; full cos/sin Python feeds blocks.
    cu_seqlens.bin             — i32, [N+1].
    patch_embed_out.bin        — bf16, [L, 1280]; output of patch_embed.
    block_{i}_out.bin          — bf16, [L, 1280]; for i in {0, 1, 15, 31}.
    merger_out.bin             — bf16, [L_out, 1536]; final encoder output.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "goldens"


def make_input_image(h: int = 224, w: int = 224) -> Image.Image:
    # Deterministic gradient + noise so resampling, normalization, and the
    # 9D-transpose patch flatten all see structured input — not an even
    # gray field that hides bugs.
    rng = np.random.default_rng(0xC0FFEE)
    yy, xx = np.indices((h, w), dtype=np.float32)
    r = (xx / (w - 1)) * 255.0
    g = (yy / (h - 1)) * 255.0
    b = ((xx + yy) / (h + w - 2)) * 255.0
    noise = rng.integers(low=-12, high=13, size=(h, w, 3), dtype=np.int16)
    arr = np.stack([r, g, b], axis=-1).astype(np.int16) + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def dump(arr: torch.Tensor | np.ndarray, name: str, out_dir: Path, manifest: dict):
    if isinstance(arr, torch.Tensor):
        a = arr.detach().contiguous().cpu().numpy()
    else:
        a = np.ascontiguousarray(arr)
    path = out_dir / f"{name}.bin"
    a.tofile(path)
    manifest[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes}
    print(f"  wrote {name}: shape={a.shape} dtype={a.dtype} bytes={a.nbytes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build deterministic input ────────────────────────────────────
    img = make_input_image(224, 224)
    img.save(out_dir / "input.png")

    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel,
    )

    proc = AutoImageProcessor.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    vcfg = cfg.vision_config

    # ── HF image processor → patches + grid_thw ──────────────────────
    feat = proc(images=img, return_tensors="pt")
    patches = feat["pixel_values"]  # [L, 1176] (already bf16-ready f32)
    grid_thw = feat["image_grid_thw"]  # [N, 3]
    print(f"patches: {tuple(patches.shape)} {patches.dtype}")
    print(f"grid_thw: {grid_thw.tolist()}")

    # ── Materialize the vision tower from the full HF checkpoint ─────
    # Loading just the vision backbone needs the full state_dict; do it
    # at fp32 on CPU for clean numerics, then cast activations to bf16
    # on the way out so the goldens match scratchy's storage dtype.
    print("loading vision tower (this takes ~30s) …")
    from transformers import AutoModelForVision2Seq

    full = AutoModelForVision2Seq.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    vit = full.visual.eval()  # Qwen2VisionTransformerPretrainedModel

    # ── Trace per-block intermediates via forward hooks ──────────────
    intermediates: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            intermediates[name] = t.detach().clone()
        return fn

    handles = []
    handles.append(vit.patch_embed.register_forward_hook(hook("patch_embed_out")))
    for i in (0, 1, 15, 31):
        handles.append(vit.blocks[i].register_forward_hook(hook(f"block_{i}_out")))
    handles.append(vit.merger.register_forward_hook(hook("merger_out")))

    # Re-derive rope + cu_seqlens the same way the model does, separately
    # from the forward, so we can dump them.
    with torch.no_grad():
        rotary_half = vit.rot_pos_emb(grid_thw)  # [L, head_dim/2]
        emb = torch.cat((rotary_half, rotary_half), dim=-1)
        cos_full = emb.cos()
        sin_full = emb.sin()

        lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).to(torch.int32)])

    # Run the encoder.
    with torch.no_grad():
        out = vit(patches, grid_thw)
    print(f"encoder output: {tuple(out.shape)} {out.dtype}")

    for h in handles:
        h.remove()

    # ── Dump ─────────────────────────────────────────────────────────
    manifest: dict = {
        "model": args.model,
        "input_png": "input.png",
        "vision_config": {
            "embed_dim": int(vcfg.embed_dim),
            "depth": int(vcfg.depth),
            "num_heads": int(vcfg.num_heads),
            "patch_size": int(vcfg.patch_size),
            "temporal_patch_size": int(vcfg.temporal_patch_size),
            "spatial_merge_size": int(vcfg.spatial_merge_size),
            "in_chans": int(vcfg.in_chans),
            "hidden_size": int(cfg.hidden_size),
        },
    }

    dump(patches.to(torch.bfloat16).view(torch.uint16), "patches", out_dir, manifest)
    dump(grid_thw.to(torch.int64), "grid_thw", out_dir, manifest)
    dump(rotary_half.to(torch.float32), "rotary_pos_emb_half", out_dir, manifest)
    dump(cos_full.to(torch.bfloat16).view(torch.uint16), "cos", out_dir, manifest)
    dump(sin_full.to(torch.bfloat16).view(torch.uint16), "sin", out_dir, manifest)
    dump(cu.to(torch.int32), "cu_seqlens", out_dir, manifest)
    dump(intermediates["patch_embed_out"].to(torch.bfloat16).view(torch.uint16),
         "patch_embed_out", out_dir, manifest)
    for i in (0, 1, 15, 31):
        dump(intermediates[f"block_{i}_out"].to(torch.bfloat16).view(torch.uint16),
             f"block_{i}_out", out_dir, manifest)
    dump(intermediates["merger_out"].to(torch.bfloat16).view(torch.uint16),
         "merger_out", out_dir, manifest)
    dump(out.to(torch.bfloat16).view(torch.uint16), "encoder_out", out_dir, manifest)

    with open(out_dir / "goldens.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out_dir / 'goldens.json'}")


if __name__ == "__main__":
    main()
