#!/usr/bin/env python
"""P-1 oracle: dump per-stage LocateAnything-3B (MoonViT) vision tensors from
mlx-vlm as golden fixtures for the scratchy-metal port. One model load;
instruments the vision tower + projector via class-level monkeypatch.

Requires the mlx-vlm checkout that has models/locateanything (origin/main
worktree at ~/git/mlx-vlm-la). Run with that on sys.path ahead of the pinned
editable install:

  PYTHONPATH=$HOME/git/mlx-vlm-la ~/.venv/bin/python \
      tools/vision_parity/dump_golden_locateanything.py

Outputs: tools/vision_parity/golden_locateanything/*.npy + manifest.txt
"""
import os, sys
import numpy as np
import mlx.core as mx
import mlx_vlm

assert "mlx-vlm-la" in mlx_vlm.__file__, (
    f"mlx_vlm resolved to {mlx_vlm.__file__}; need the locateanything-capable "
    "worktree — run with PYTHONPATH=$HOME/git/mlx-vlm-la"
)

from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL = "mlx-community/LocateAnything-3B-4bit"
OUT = os.path.join(os.path.dirname(__file__), "golden_locateanything")
os.makedirs(OUT, exist_ok=True)


def save(name, arr):
    """complex64 (rope freqs_cis) is split into _real/_imag f32 so the Rust
    side's minimal npy parser only ever sees float32."""
    if isinstance(arr, mx.array) and arr.dtype == mx.complex64:
        save(name + "_real", mx.real(arr))
        save(name + "_imag", mx.imag(arr))
        return None
    a = np.array(arr.astype(mx.float32)) if isinstance(arr, mx.array) else np.asarray(arr)
    np.save(os.path.join(OUT, name + ".npy"), a)
    print(f"  saved {name}: shape={a.shape} dtype={a.dtype}")
    return a


# ---- deterministic test image: red circle on white, 224x224 -------------
from PIL import Image, ImageDraw
img_path = os.path.join(os.path.dirname(__file__), "red_circle_224.png")
if not os.path.exists(img_path):
    im = Image.new("RGB", (224, 224), (255, 255, 255))
    ImageDraw.Draw(im).ellipse([56, 56, 168, 168], fill=(220, 30, 30))
    im.save(img_path)
print("test image:", img_path)

print("loading", MODEL, "...")
model, processor = load(MODEL)
vt = model.vision_tower
proj = model.multi_modal_projector
print("model:", type(model).__name__, "| vision blocks:", len(vt.blocks))

# ---- instrument (patch CLASS __call__; instance-attr patching does NOT
#      intercept obj() — Python resolves via type(obj).__call__) ----------
caps = {}
block_outs = []

import mlx_vlm.models.locateanything.vision as _lav

# patch_embed: capture out (NOTE: includes the Learnable2DInterpPosEmb add)
PEcls = type(vt.patch_embed)
_pe = PEcls.__call__
def pe_call(self, x, *a, **k):
    out = _pe(self, x, *a, **k)
    caps["post_patch_embed"] = out
    return out
PEcls.__call__ = pe_call

# Learnable2DInterpPosEmb: capture in/out → pos contribution = out - in
POScls = type(vt.patch_embed.pos_emb)
_pos = POScls.__call__
def pos_call(self, x, *a, **k):
    out = _pos(self, x, *a, **k)
    caps["pre_pos_emb"] = x
    caps["pos_embeds"] = out - x
    return out
POScls.__call__ = pos_call

# blocks: ordered capture
BLKcls = type(vt.blocks[0])
_blk = BLKcls.__call__
def blk_call(self, *a, **k):
    out = _blk(self, *a, **k)
    block_outs.append(out)
    return out
BLKcls.__call__ = blk_call

# apply_rope is a module global looked up at call time → patch it; record
# block-0's (q, k, freqs_cis, q_out, k_out) for the interleaved-rope golden.
rope_calls = []
_apply = _lav.apply_rope
def patched_apply(q, k, freqs_cis):
    q_out, k_out = _apply(q, k, freqs_cis)
    if not rope_calls:
        rope_calls.append((q, k, freqs_cis, q_out, k_out))
    return q_out, k_out
_lav.apply_rope = patched_apply

# patch_merger is a module-level FUNCTION: input == post final_layernorm,
# output == list of per-image [n_merged, 4, 1152]
_pm = _lav.patch_merger
def patched_pm(x, grid_thw, merge_kernel_size, grid_shapes=None):
    caps["post_final_layernorm"] = x
    outs = _pm(x, grid_thw, merge_kernel_size, grid_shapes=grid_shapes)
    caps["merger_out"] = mx.concatenate(outs, axis=0)
    return outs
_lav.patch_merger = patched_pm

# projector: capture output [n_merged, 2048]
PRcls = type(proj)
_pr = PRcls.__call__
def pr_call(self, x, *a, **k):
    out = _pr(self, x, *a, **k)
    caps["projector_out"] = out
    return out
PRcls.__call__ = pr_call

# ---- build inputs exactly like generate() does ---------------------------
prompt = apply_chat_template(processor, model.config, "Locate the red circle.", num_images=1)
print("prompt:", repr(prompt))
inputs = prepare_inputs(processor, images=[img_path], prompts=prompt)
input_ids = inputs["input_ids"]
pv = inputs["pixel_values"]
grid = inputs.get("image_grid_hws")
kwargs = {k: v for k, v in inputs.items()
          if k not in ("input_ids", "pixel_values", "attention_mask")}
print("input_ids", input_ids.shape, "| pixel_values", pv.shape, pv.dtype,
      "| image_grid_hws", None if grid is None else np.asarray(grid).tolist())

# full multimodal embedding path: vision tower + projector + splice
feats = model.get_input_embeddings(input_ids, pv, **kwargs)
caps["inputs_embeds"] = feats.inputs_embeds[0]  # [seq, 2048]

for i in sorted({0, len(vt.blocks) // 2, len(vt.blocks) - 1}):
    if i < len(block_outs):
        caps[f"post_block_{i}"] = block_outs[i]

assert rope_calls, "apply_rope was never called — instrumentation broke"
q, k, fr, q_out, k_out = rope_calls[0]
caps["rope_block0_q_in"] = q
caps["rope_block0_k_in"] = k
caps["rope_block0_freqs_cis"] = fr           # complex64 → saved as _real/_imag
caps["rope_block0_q_out"] = q_out
caps["rope_block0_k_out"] = k_out

manifest = {}
manifest["pixel_values"] = save("pixel_values", mx.array(np.asarray(pv))).shape
if grid is not None:
    manifest["image_grid_hws"] = save("image_grid_hws",
                                      np.asarray(grid, dtype=np.int64)).tolist()
manifest["input_ids"] = save("input_ids",
                             np.asarray(input_ids, dtype=np.int64)).shape
for kk, vv in caps.items():
    a = save(kk, vv)
    if a is not None:
        manifest[kk] = list(a.shape)

with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write(f"model={MODEL}\nimage={img_path}\nprompt={prompt!r}\n")
    f.write(f"vision_blocks={len(vt.blocks)}\n")
    for kk, shp in manifest.items():
        f.write(f"{kk}: {shp}\n")
print("DONE — golden fixtures in", OUT)
