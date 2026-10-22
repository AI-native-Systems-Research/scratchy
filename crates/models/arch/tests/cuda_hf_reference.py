#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace transformers' qwen3.5-0.8b logits: the reference `cuda_hf_parity.rs` checks the
CUDA forward against.

Runs `Qwen3_5ForConditionalGeneration` from a local checkpoint on the CPU, twice: in float32 (the
reference) and in bfloat16 (the noise yardstick: how far a bf16 forward of the same weights lands
from the float32 one). Text only (`input_ids`), `use_cache=False`, `torch.no_grad()`, `eval()`.

Tokenization happens here, once: raw text, no chat template, no special tokens. The ids are written
to `meta.json`, and both this script and the Rust test read those ids, so a tokenizer difference
cannot make a false mismatch.

Prompts: the first 5 lines of `--prompts`, and prompt 6, all its lines joined with a newline.

Writes to `--out`:
  meta.json          checkpoint dir, revision, safetensors sha256, versions, and per prompt its
                     text, ids, and the bf16-vs-f32 statistics (minimum row cosine, RMS, max)
  ref_f32_<p>.bin    logits of every row, [L, vocab] float32 little-endian
  ref_bf16_<p>.bin   the same, from the bfloat16 run
  greedy_<p>.json    prompts 1-5: 8 greedy tokens, each from a fresh full forward (no cache) of the
                     prompt plus the tokens so far, in float32, with each step's float32 top-1 minus
                     top-2 margin and the bfloat16 run's max |error| on that step's last row

Needs only torch and transformers. The plain-torch path must be the one that runs, so this refuses
to start when `fla`, `causal_conv1d` or `kernels` is importable.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

PINNED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
GREEDY_PROMPTS = 5
GREEDY_STEPS = 8


def refuse_kernel_packages():
    present = [m for m in ("fla", "causal_conv1d", "kernels") if importlib.util.find_spec(m) is not None]
    if present:
        sys.exit(
            f"cuda_hf_reference: {present} importable; the reference must run HF's plain-torch path, "
            "so run it in an environment without them"
        )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def safetensors_files(checkpoint):
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    return ["model.safetensors"]


def prompts(path):
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    return lines[:5] + ["\n".join(lines)]


def logits(model, torch, ids):
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    return out.logits[0].float()


def stats(torch, bf16, f32):
    """Per prompt: the minimum row cosine of bf16 against f32, the RMS and max |bf16 - f32|."""
    d = (bf16 - f32).double()
    cos = torch.nn.functional.cosine_similarity(bf16.double(), f32.double(), dim=1)
    return {
        "min_cos": cos.min().item(),
        "rms": d.pow(2).mean().sqrt().item(),
        "max": d.abs().max().item(),
    }


def write_bin(path, t):
    t.contiguous().numpy().astype("<f4").tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", default=PINNED_REVISION)
    args = ap.parse_args()
    refuse_kernel_packages()

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    models = {}
    for name, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        m = Qwen3_5ForConditionalGeneration.from_pretrained(args.checkpoint, dtype=dtype)
        m.eval()
        models[name] = m
    files = safetensors_files(args.checkpoint)
    meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "revision": args.revision,
        "safetensors_sha256": {f: sha256(os.path.join(args.checkpoint, f)) for f in files},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vocab": models["f32"].config.text_config.vocab_size,
        "greedy_steps": GREEDY_STEPS,
        "prompts": [],
    }
    for p, text in enumerate(prompts(args.prompts), start=1):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        f32 = logits(models["f32"], torch, ids)
        bf16 = logits(models["bf16"], torch, ids)
        write_bin(os.path.join(args.out, f"ref_f32_{p}.bin"), f32)
        write_bin(os.path.join(args.out, f"ref_bf16_{p}.bin"), bf16)
        s = stats(torch, bf16, f32)
        print(
            f"prompt {p}: {len(ids)} tokens, HF-bf16 vs f32: min cosine {s['min_cos']:.6f}, "
            f"RMS {s['rms']:.6e}, max {s['max']:.6e}",
            flush=True,
        )
        entry = {"prompt": p, "text": text, "ids": ids, "bf16_vs_f32": s}
        if p <= GREEDY_PROMPTS:
            seq, steps = list(ids), []
            for _ in range(GREEDY_STEPS):
                last32 = logits(models["f32"], torch, seq)[-1]
                last16 = logits(models["bf16"], torch, seq)[-1]
                top = torch.topk(last32, 2)
                token = int(top.indices[0])
                steps.append(
                    {
                        "token": token,
                        "margin": (top.values[0] - top.values[1]).item(),
                        "bf16_max_err": (last16 - last32).abs().max().item(),
                    }
                )
                seq.append(token)
            with open(os.path.join(args.out, f"greedy_{p}.json"), "w") as f:
                json.dump({"prompt": p, "steps": steps}, f, indent=1)
            print(f"prompt {p}: greedy f32 {[s['token'] for s in steps]}", flush=True)
        meta["prompts"].append(entry)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
