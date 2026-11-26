# The weight bytes a gpt-oss decode step reads, from a checkpoint's safetensors headers alone (no
# download): what every step reads (attention, router, norms, the lm_head), one routed expert's
# bytes across the layers, and the expected bytes of a verify step of R rows under uniform routing,
# whose experts are the union of its rows': E * (1 - (1 - k/E)^R) a layer, not R * k.
# Usage: python bytes.py <target repo> [drafter repo or dir]
import json
import os
import sys

from huggingface_hub import get_safetensors_metadata, hf_hub_download


def tensors(repo):
    if os.path.isdir(repo):
        from safetensors import safe_open
        out = {}
        for f in os.listdir(repo):
            if f.endswith(".safetensors"):
                with safe_open(os.path.join(repo, f), "np") as st:
                    for k in st.keys():
                        sl = st.get_slice(k)
                        n = 1
                        for d in sl.get_shape():
                            n *= d
                        out[k] = n * {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}[sl.get_dtype()]
        return out
    meta = get_safetensors_metadata(repo)
    out = {}
    for fm in meta.files_metadata.values():
        for k, t in fm.tensors.items():
            n = 1
            for d in t.shape:
                n *= d
            out[k] = n * {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1}[t.dtype]
    return out


target = sys.argv[1]
cfg = json.load(open(hf_hub_download(target, "config.json")))
E, k, L = cfg["num_local_experts"], cfg["num_experts_per_tok"], cfg["num_hidden_layers"]
T = tensors(target)
experts = sum(v for n, v in T.items() if ".mlp.experts." in n)
embed = sum(v for n, v in T.items() if "embed_tokens" in n)
fixed = sum(T.values()) - experts - embed  # an embedding row a token is negligible
per_expert = experts / E  # one expert's bytes summed over the layers
plain = fixed + k * per_expert
G = 1 << 30
print(f"{target}: {L} layers, {E} experts top {k}; total {sum(T.values()) / G:.2f} GiB")
print(f"  plain step: fixed {fixed / G:.3f} GiB + {k} experts {k * per_expert / G:.3f} GiB = {plain / G:.3f} GiB "
      f"(experts {100 * k * per_expert / plain:.0f}%)")
for R in (1, 2, 3, 4, 5, 8):
    union = E * (1 - (1 - k / E) ** R)
    b = fixed + union * per_expert
    lin = fixed + R * k * per_expert
    print(f"  R={R}: expected experts a layer {union:5.2f}; bytes {b / G:.3f} GiB = {b / plain:.2f}x plain "
          f"(linear per-row charge: {lin / plain:.2f}x)")
if len(sys.argv) > 2:
    D = tensors(sys.argv[2])
    d = sum(v for n, v in D.items() if "embed_tokens" not in n and "lm_head" not in n)
    vocab = 65536
    head = [v for n, v in T.items() if n.startswith("lm_head")]
    head_rows = sum(head) * vocab / cfg["vocab_size"]
    print(f"  drafter pass: {d / G:.3f} GiB of its own + the lent lm_head's {vocab} rows {head_rows / G:.3f} GiB "
          f"= {(d + head_rows) / plain:.2f}x a plain step's bytes")
