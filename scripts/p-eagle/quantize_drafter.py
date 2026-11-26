# AWS's P-EAGLE drafter at MLX affine group 64 (mlx-lm's layout: `<name>.weight` packed uint32,
# `<name>.scales` / `<name>.biases` bf16), mixed: the MLPs and `fc` at 4 bits, the attention
# projections and the embedding at 8 (oracle.py's sweep: the mask rows' drafts lose most of their
# acceptance at a 4-bit embedding or attention). The norms and `mask_hidden` are kept, the lm_head
# (its target's, lent) and the unused vocabulary maps (`d2t`, `t2d`) dropped. The config's
# per-module map names every quantized module, the lent lm_head's at its target's 4 bits too.
# Usage: python quantize_drafter.py <drafter dir> <out dir>
import json, os, shutil, sys
import mlx.core as mx

src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
W = mx.load(os.path.join(src, "model.safetensors"))
GROUP = 64


RECIPE = dict(kv.split("=") for kv in os.environ.get("RECIPE", "mlp=3,attn=4,fc=4,embed=8").split(","))


def bits_of(base):
    if base == "embed_tokens":
        return int(RECIPE["embed"])
    if ".self_attn." in base:
        return int(RECIPE["attn"])
    if ".mlp." in base:
        return int(RECIPE["mlp"])
    return int(RECIPE["fc"])


q, modules = {}, {}
for name, w in W.items():
    if name in ("lm_head.weight", "d2t", "t2d"):
        continue
    if name.endswith(".weight") and w.ndim == 2 and "norm" not in name:
        base = name[: -len(".weight")]
        bits = bits_of(base)
        wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=bits)
        q[base + ".weight"] = wq
        q[base + ".scales"] = scales.astype(mx.bfloat16)
        q[base + ".biases"] = biases.astype(mx.bfloat16)
        modules[base] = {"group_size": GROUP, "bits": bits}
    else:
        q[name] = w
modules["lm_head"] = {"group_size": GROUP, "bits": 4}
mx.save_safetensors(os.path.join(out, "model.safetensors"), q, metadata={"format": "mlx"})
cfg = json.load(open(os.path.join(src, "config.json")))
recipe = {"group_size": GROUP, "bits": 4, **dict(sorted(modules.items()))}
cfg["quantization"] = recipe
cfg["quantization_config"] = recipe
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
if os.path.exists(os.path.join(src, "README.md")):
    shutil.copy(os.path.join(src, "README.md"), out)
total = sum(v.nbytes for v in q.values())
print(f"{len(q)} tensors, {len(modules)} quantized modules, {total / 1e9:.2f} GB -> {out}")
