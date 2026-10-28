#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace transformers' greedy qwen3.5-0.8b replies to chat-templated prompts: the reference
`scr chat --device cuda` and `scr serve` are compared against (`cuda_chat_compare.py`).

For prompts 1-5 of `--prompts`, one user message each, rendered with the checkpoint's own chat
template (`apply_chat_template(..., add_generation_prompt=True)`, `enable_thinking` unset, as scr
renders it), it runs `--steps` greedy steps of `Qwen3_5ForConditionalGeneration` on the CPU in
float32, each a fresh full forward (no cache) of the ids so far. It never stops early, so every
prompt has `--steps` tokens; the compare decides where a stop would have ended each side. At each
step it also records the float32 top-1 minus top-2 margin, and how far a bfloat16 forward of the
same ids lands from the float32 one on that row (`bf16_max_err`): the near-tie yardstick. It
also runs HF-bf16's own `--steps` greedy tokens on the same ids (`bf16_tokens`), so the compare can
show where a bf16 forward alone already departs from float32.

The stop ids each side honours are recorded with the reason, because they differ for this
checkpoint:
  * scr's (`resolve_eos_token_ids`): config.json's top-level `eos_token_id` (absent here),
    generation_config.json's (the file is absent at the pinned revision), tokenizer_config's
    `eos_token`, and the chat template's turn terminator: `<|im_end|>`.
  * HF's (`GenerationConfig.from_model_config`, used when there is no generation_config.json):
    the top-level `eos_token_id`, else the text config's: `<|endoftext|>`.

Writes `--out` (JSON): the revision, the checkpoint's sha256, versions, both stop sets, whether
generation_config.json exists, and per prompt its text, ids, the steps and the decoded text of
all `--steps` tokens (`skip_special_tokens=True`).

Needs only torch and transformers, and the plain-torch path: it refuses to start when `fla`,
`causal_conv1d` or `kernels` is importable, or when the checkpoint is not the pinned revision's.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

PINNED_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
PINNED_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
PROMPTS = 5
# scr's template turn terminators (`init.rs` `TEMPLATE_TURN_TERMINATORS`).
SCR_TEMPLATE_TERMINATORS = ["<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]


def refuse_kernel_packages():
    present = [m for m in ("fla", "causal_conv1d", "kernels") if importlib.util.find_spec(m) is not None]
    if present:
        sys.exit(f"cuda_hf_chat_reference: {present} importable; run it without them")


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
    return sorted(f for f in os.listdir(checkpoint) if f.endswith(".safetensors"))


def ids_of(v):
    if v is None:
        return []
    return [v] if isinstance(v, int) else list(v)


def scr_stop_ids(checkpoint, tok):
    """The stop ids scr resolves (`init.rs` `resolve_eos_token_ids`), from the same files."""
    out = []

    def push(i):
        if i is not None and i not in out:
            out.append(i)

    with open(os.path.join(checkpoint, "config.json")) as f:
        cfg = json.load(f)
    for i in ids_of(cfg.get("eos_token_id")):
        push(i)
    gen = os.path.join(checkpoint, "generation_config.json")
    if os.path.exists(gen):
        with open(gen) as f:
            for i in ids_of(json.load(f).get("eos_token_id")):
                push(i)
    with open(os.path.join(checkpoint, "tokenizer_config.json")) as f:
        tcfg = json.load(f)
    eos = tcfg.get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    if isinstance(eos, str):
        push(tok.convert_tokens_to_ids(eos))
    tpl = tcfg.get("chat_template") or ""
    for cand in SCR_TEMPLATE_TERMINATORS:
        if cand in tpl:
            i = tok.convert_tokens_to_ids(cand)
            if i is not None and i != tok.unk_token_id:
                push(i)
    return out


def hf_stop_ids(checkpoint):
    """The stop ids HF's `generate` honours with no generation_config.json."""
    from transformers import GenerationConfig

    gen = os.path.join(checkpoint, "generation_config.json")
    if os.path.exists(gen):
        return ids_of(GenerationConfig.from_pretrained(checkpoint).eos_token_id)
    from transformers import AutoConfig

    return ids_of(GenerationConfig.from_model_config(AutoConfig.from_pretrained(checkpoint)).eos_token_id)


def prompts(path):
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    return lines[:PROMPTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    refuse_kernel_packages()
    files = safetensors_files(args.checkpoint)
    shas = {f: sha256(os.path.join(args.checkpoint, f)) for f in files}
    if sorted(shas.values()) != [PINNED_SHA256]:
        sys.exit(f"cuda_hf_chat_reference: safetensors sha256 {shas}; the pinned file is {PINNED_SHA256}")

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    models = {}
    for name, dtype in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        m = Qwen3_5ForConditionalGeneration.from_pretrained(args.checkpoint, dtype=dtype)
        m.eval()
        models[name] = m

    def last_row(model, seq):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([seq]), use_cache=False)
        return out.logits[0, -1].float()

    stops = {"scr": scr_stop_ids(args.checkpoint, tok), "hf": hf_stop_ids(args.checkpoint)}
    out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "revision": PINNED_REVISION,
        "safetensors_sha256": shas,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "generation_config_json": os.path.exists(os.path.join(args.checkpoint, "generation_config.json")),
        "stop_ids": stops,
        "stop_tokens": {k: [tok.convert_ids_to_tokens(i) for i in v] for k, v in stops.items()},
        "steps": args.steps,
        "prompts": [],
    }
    print(f"stop ids: scr {stops['scr']} {out['stop_tokens']['scr']}, hf {stops['hf']} {out['stop_tokens']['hf']}")
    for p, text in enumerate(prompts(args.prompts), start=1):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
        )
        if isinstance(ids, dict) or hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        ids = [int(i) for i in ids]
        seq, steps = list(ids), []
        for _ in range(args.steps):
            r32 = last_row(models["f32"], seq)
            r16 = last_row(models["bf16"], seq)
            top = torch.topk(r32, 2)
            token = int(top.indices[0])
            steps.append(
                {
                    "token": token,
                    "margin": (top.values[0] - top.values[1]).item(),
                    "bf16_max_err": (r16 - r32).abs().max().item(),
                }
            )
            seq.append(token)
        tokens = [s["token"] for s in steps]
        reply = tok.decode(tokens, skip_special_tokens=True)
        # HF-bf16's own greedy run on the same ids, the same fresh full forwards: the yardstick for
        # how far a bf16 forward of these weights strays from the float32 one, token by token.
        seq16, bf16_tokens = list(ids), []
        for _ in range(args.steps):
            token = int(torch.argmax(last_row(models["bf16"], seq16)))
            bf16_tokens.append(token)
            seq16.append(token)
        out["prompts"].append(
            {
                "prompt": p,
                "text": text,
                "ids": ids,
                "steps": steps,
                "reply": reply,
                "bf16_tokens": bf16_tokens,
                "bf16_reply": tok.decode(bf16_tokens, skip_special_tokens=True),
            }
        )
        print(f"prompt {p}: {len(ids)} ids, tokens {tokens}, reply {reply!r}", flush=True)
        print(f"prompt {p}: HF-bf16 tokens {bf16_tokens}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
