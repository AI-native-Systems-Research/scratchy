#!/usr/bin/env python
"""P-1 oracle: dump per-stage Qwen3.5-VL-9B vision-tower tensors from mlx-vlm
(runs natively on the Mac, no torch) as golden fixtures for the scratchy-metal
vision port. One model load; instruments the vision tower via monkeypatch.

Usage: ~/.venv/bin/python tools/vision_parity/dump_golden.py
Outputs: tools/vision_parity/golden/*.npy  +  manifest.txt
"""
import os, sys, json
import numpy as np
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.utils import load_image

MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"
OUT = os.path.join(os.path.dirname(__file__), "golden")
os.makedirs(OUT, exist_ok=True)

def to_np(arr):
    if isinstance(arr, mx.array):
        return np.array(arr.astype(mx.float32))  # numpy has no bf16
    return np.asarray(arr)

def save(name, arr):
    a = to_np(arr)
    np.save(os.path.join(OUT, name + ".npy"), a)
    print(f"  saved {name}: shape={a.shape} dtype={a.dtype}")
    return a

# ---- deterministic test image: red circle on white, 224x224 -------------
from PIL import Image, ImageDraw
img_path = os.path.join(os.path.dirname(__file__), "red_circle_224.png")
im = Image.new("RGB", (224, 224), (255, 255, 255))
ImageDraw.Draw(im).ellipse([56, 56, 168, 168], fill=(220, 30, 30))
im.save(img_path)
print("test image:", img_path)

print("loading", MODEL, "...")
model, processor = load(MODEL)
cfg = model.config
print("model:", type(model).__name__)
vt = getattr(model, "vision_tower", None) or getattr(model, "visual", None)
print("vision tower:", type(vt).__name__, "| blocks:", len(vt.blocks))

# ---- instrument: capture per-stage outputs by patching CLASS __call__ ----
# (Python resolves obj() via type(obj).__call__, so instance-attr patching
#  does NOT intercept — patch the class methods.)
caps = {}
block_outs = []  # ordered: one entry per block call

PEcls = type(vt.patch_embed)
_pe = PEcls.__call__
PEcls.__call__ = lambda self, x, *a, **k: caps.__setitem__("post_patch_embed", _pe(self, x, *a, **k)) or caps["post_patch_embed"]

MGcls = type(vt.merger)
_mg = MGcls.__call__
MGcls.__call__ = lambda self, x, *a, **k: caps.__setitem__("merger_out", _mg(self, x, *a, **k)) or caps["merger_out"]

BLKcls = type(vt.blocks[0])
_blk = BLKcls.__call__
def blk_call(self, *a, **k):
    out = _blk(self, *a, **k); block_outs.append(out); return out
BLKcls.__call__ = blk_call

# fast_pos_embed_interpolate + rot_pos_emb are regular methods → instance patch OK
if hasattr(vt, "fast_pos_embed_interpolate"):
    _fp = vt.fast_pos_embed_interpolate
    def patched_fp(grid_thw):
        out = _fp(grid_thw); caps["pos_embeds"] = out; return out
    vt.fast_pos_embed_interpolate = patched_fp

_rpe = vt.rot_pos_emb
def patched_rpe(grid_thw):
    out = _rpe(grid_thw); caps["rot_pos_emb_table"] = out; return out
vt.rot_pos_emb = patched_rpe

# apply_rotary_pos_emb_vision is a module global the block looks up at call
# time → patch it; record block-0's q then k (first two calls) for the P1
# vision_rope_2d golden (input tensor, freqs table, output).
import mlx_vlm.models.qwen3_vl.vision as _qv
rope_calls = []
_apply = _qv.apply_rotary_pos_emb_vision
def patched_apply(tensor, freqs):
    out = _apply(tensor, freqs)
    if len(rope_calls) < 2:
        rope_calls.append((tensor, freqs, out))
    return out
_qv.apply_rotary_pos_emb_vision = patched_apply

# ---- build inputs through the processor + run the vision tower -----------
image = load_image(img_path)
# mlx-vlm processors expose image preprocessing; get pixel_values + grid_thw.
proc_out = processor(text=["<image>"], images=[image], return_tensors="np") \
    if callable(processor) else None
print("processor output keys:", list(proc_out.keys()) if proc_out else "N/A")

pv = mx.array(proc_out["pixel_values"])          # mlx coerces numpy/list directly
gt = mx.array(np.asarray(proc_out["image_grid_thw"], dtype=np.int64))
print("running vision tower: pixel_values", pv.shape, "grid_thw", gt.shape)
vis_out = vt(pv, gt)
if isinstance(vis_out, tuple):
    vis_out = vis_out[0]
caps["vision_output"] = vis_out

# pick the dumped block indices out of the ordered block_outs
DUMP_BLOCKS = sorted({0, len(vt.blocks) // 2, len(vt.blocks) - 1})
for i in DUMP_BLOCKS:
    if i < len(block_outs):
        caps[f"post_block_{i}"] = block_outs[i]

# block-0 rope golden (q then k): input tensor, freqs table, rotated output
for tag, (t, fr, o) in zip(("q", "k"), rope_calls):
    caps[f"rope_block0_{tag}_in"] = t
    caps[f"rope_block0_{tag}_freqs"] = fr
    caps[f"rope_block0_{tag}_out"] = o

manifest = {}
manifest["pixel_values"] = save("pixel_values", pv).shape
manifest["grid_thw"] = save("grid_thw", gt).tolist()
for k, v in caps.items():
    manifest[k] = list(save(k, v).shape)

with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write(f"model={MODEL}\nimage={img_path}\n")
    f.write(f"vision_blocks={len(vt.blocks)} dumped_blocks={DUMP_BLOCKS}\n")
    for k, shp in manifest.items():
        f.write(f"{k}: {shp}\n")
print("DONE — golden fixtures in", OUT)
