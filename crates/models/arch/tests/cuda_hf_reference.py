#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace transformers' qwen3.5 logits: the reference `cuda_hf_parity.rs` (qwen3.5-0.8b),
`cuda_hf_parity_9b.rs` (qwen3.5-9b) and `cuda_hf_parity_35b.rs` (qwen3.6-35b-a3b) check the CUDA
forward against.

Runs the model's class (`MODEL_CLASS`: `Qwen3_5ForConditionalGeneration`, or
`Qwen3_5MoeForConditionalGeneration` for the MoE model) from a local checkpoint on the CPU, twice:
in float32 (the reference) and in bfloat16 (the noise yardstick: how far a bf16 forward of the same
weights lands from the float32 one). Text only (`input_ids`), `use_cache=False`, `torch.no_grad()`,
`eval()`. One model is resident at a time: every float32 forward (the prompts and their greedy
steps) runs first, that model is freed, then the bfloat16 one runs the same inputs. The outputs are
those of loading both at once; the peak is one model's (the 35B's f32 alone is about 140 GB).

The MoE model's experts run HF's per-expert loop, `experts_implementation="eager"`, passed
explicitly (the default is `grouped_mm`, whose CPU summation order differs) and recorded in
`meta.json`; and every layer's router (`mlp.gate`) is hooked for the routing dump below.

Tokenization happens here, once: raw text, no chat template, no special tokens. The ids are written
to `meta.json`, and both this script and the Rust test read those ids, so a tokenizer difference
cannot make a false mismatch.

Prompts: the first 5 lines of `--prompts`, and prompt 6, all its lines joined with a newline.

`--model` names the checkpoint (`qwen3.5-0.8b`, `qwen3.5-9b` or `qwen3.6-35b-a3b`) and picks its
row of `PINNED` and `MODEL_CLASS`.

Writes to `--out`:
  meta.json          model, checkpoint dir, revision, safetensors sha256, versions, and per prompt its
                     text, ids, and the bf16-vs-f32 statistics (minimum row cosine, RMS, max)
  ref_f32_<p>.bin    logits of every row, [L, vocab] float32 little-endian
  ref_bf16_<p>.bin   the same, from the bfloat16 run
  greedy_<p>.json    prompts 1-5: 8 greedy tokens, each from a fresh full forward (no cache) of the
                     prompt plus the tokens so far, in float32, with each step's float32 top-1 minus
                     top-2 margin and the bfloat16 run's max |error| on that step's last row
  route_<p>.bin      the MoE model only, `cuda_route_parity_35b.rs`'s P9: per layer and row, the
                     router's top-8 expert indices and the rank-8 and rank-9 router probabilities
                     (softmax of the router logits in f32): the f32 run's [layers, L, 8] int32 then
                     [layers, L, 2] float32, then the bf16 run's, little-endian
  route_greedy_<p>.bin  prompts 1-5, the same for each greedy step's last row ([layers, 8, ...])

Needs only torch and transformers. The plain-torch path must be the one that runs, so this refuses
to start when `fla`, `causal_conv1d` or `kernels` is importable (so no hub kernel, such as the MoE
router's `SoftmaxTopKRouter`, can replace HF's code). It also refuses a `--revision` other than the
model's pinned one, a checkpoint with no text decoder (an MTP head or vision tower alone), and a
checkpoint whose safetensors files' sha256s, sorted, are not the model's pinned list, before it
loads anything: the goldens are of that checkpoint only.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import struct
import sys
import time

# Per `--model`: the pinned revision, and the sorted sha256s of its safetensors files.
PINNED = {
    # Qwen/Qwen3.5-0.8B: one file.
    "qwen3.5-0.8b": (
        "2fc06364715b967f1860aea9cf38778875588b17",
        ["04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"],
    ),
    # Qwen/Qwen3.5-9B: four shards, model.safetensors-0000{1,2,3,4}-of-00004.safetensors.
    "qwen3.5-9b": (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        sorted(
            [
                "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
                "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
                "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
                "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
            ]
        ),
    ),
    # Qwen/Qwen3.6-35B-A3B: 26 shards, model-000{01..26}-of-00026.safetensors (AI-native-Systems-
    # Research/scratchy#859's table, the LFS oids at the revision).
    "qwen3.6-35b-a3b": (
        "995ad96eacd98c81ed38be0c5b274b04031597b0",
        sorted(
            [
                "adee7bcb930aed22e0677e58d4873b48dadb1ed8001cb5c6a0487286eadb3478",
                "88f2dfd2b9e73e4b70be533dbf61bcfa3c9a0003758900fcbc9d9b96f5751d4b",
                "8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb",
                "12d7db38689ba3c8af74b23ef8523eca41e0cd95db870583d0663a3ee8a6bd60",
                "a836047305d0f7a7b50f0815d09d5c03ec03d59ec2c763fcdc4bf7e9936bf902",
                "c9080d718e9c5f9e337443225aa417d4c24d00ae7995d76ee3f1cc296b557d15",
                "e8c05e23131b1dd45a455ec38cfac7db14667358268623c3938d00cf3e959a68",
                "4b6a6d495053089f4a80e7cbc82e848fba44e2c0c60122233d8fdff79fa7b296",
                "a31a954bb72d1c714e751bf0aabf2ff533f5a509693ebf7dd22ad6e90be46f67",
                "246560e66570fe746653b8443e245dc334c9b8b831ea43d2d9f1b7d98623994e",
                "7180392817fe3ecb3a27a1da43b7ff22c1a94806bac49975f9f122c3126df675",
                "043fb525f6625c2f2acb75e65a9959ee3fa7b6e3fdd2034b5cfe1859b01d3cfb",
                "33a20fb20a21379bf43c84a43105f9c0cc35bd50d740b1c302dcbe4b700f5425",
                "be823e33c5cb6120ad3769d081f34a2449dc2358041fca7c29d636c1ba19130d",
                "a89d547c6f9d0b535ee5ea2f2478f163089539f3f0dd330cb23d278a19d76123",
                "69fc3ae0316482288afdcdd0b9eb7d626703ae26f7567e89aa3fc8d1ffd4ff5b",
                "e356e3943cf3852b76bb8992e674f3256013e27d54b78e8250514151cdc29637",
                "9e5e63fd1cc7d6848330c1fa363dfcb661bbc2ac87e672d0e28b71c9cb7f3c7f",
                "708644ad34f1de727bf484f396944d8ec628645d52c183e9a992e65671685e21",
                "ca083a1d1aa64f8e8a785998f543a43374f13436dc85d396eee4e72c7a84e1ae",
                "ada4ae48f3d48fe01b4c53f2f82bce25e798a9631fd33959c881156fef2ccbce",
                "def207fb42d7db31efb512755557763c23233c6e4d4c433027cb5102a7bce2f7",
                "864d52ca7768a36f514069222e8de8626264ae124097ba8fcce5b5da2c6e2ed7",
                "391acd27420cdce5935ff18152423c70620d19dac3c39a5ef1a81d369f82d737",
                "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154",
                "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613",
            ]
        ),
    ),
}
# Per `--model`: the transformers class, the `from_pretrained` keywords beyond `dtype`, and
# whether its layers' routers are dumped (`route_<p>.bin`). The dense models load exactly as before.
MODEL_CLASS = {
    "qwen3.5-0.8b": ("Qwen3_5ForConditionalGeneration", {}, False),
    "qwen3.5-9b": ("Qwen3_5ForConditionalGeneration", {}, False),
    "qwen3.6-35b-a3b": (
        "Qwen3_5MoeForConditionalGeneration",
        {"experts_implementation": "eager"},
        True,
    ),
}
# The text decoder's tensors: a checkpoint with none of them (an MTP head or a vision tower
# alone) is refused.
TEXT_EMBED = "model.language_model.embed_tokens.weight"
TEXT_LAYERS = "model.language_model.layers."
# The router's picks per token, and the two ranks whose probabilities the dump records.
ROUTE_TOP_K = 8
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


def tensor_names(checkpoint, files):
    """Every tensor name of the checkpoint: the index's weight map, or the one file's header."""
    index = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return set(json.load(f)["weight_map"])
    names = set()
    for name in files:
        with open(os.path.join(checkpoint, name), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            names |= set(json.loads(f.read(n))) - {"__metadata__"}
    return names


def refuse_no_text_decoder(checkpoint, names):
    """Exit unless the checkpoint holds the text decoder: its embedding table and its layers."""
    if TEXT_EMBED not in names or not any(n.startswith(TEXT_LAYERS) for n in names):
        heads = sorted({".".join(n.split(".")[:2]) for n in names})
        sys.exit(
            f"cuda_hf_reference: {checkpoint} has no text decoder ({TEXT_EMBED} and "
            f"{TEXT_LAYERS}*); its tensors are under {heads}, an MTP head or vision tower alone"
        )


def refuse_unpinned(model, revision, shas):
    """Exit unless `revision` is `model`'s pinned one and `shas` (file -> sha256), sorted, is its
    pinned list."""
    pinned_revision, pinned_shas = PINNED[model]
    if revision != pinned_revision:
        sys.exit(
            f"cuda_hf_reference: --revision {revision}; the {model} goldens are of {pinned_revision}"
        )
    if sorted(shas.values()) != pinned_shas:
        sys.exit(
            f"cuda_hf_reference: the checkpoint's safetensors sha256 are {shas}; the pinned "
            f"{model} revision's are {pinned_shas}"
        )


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


class RouteDump:
    """Forward hooks on every layer's router (`Qwen3_5MoeTopKRouter`, which returns
    `(router_logits, router_scores, router_indices)`): after a forward, `take()` is the run's
    top-8 indices `[layers, L, 8]` int32 and its rank-8 and rank-9 probabilities `[layers, L, 2]`
    float32, from `softmax(router_logits)` in f32."""

    def __init__(self, torch, model):
        self.torch = torch
        self.layers = []
        self.handles = [
            layer.mlp.gate.register_forward_hook(self.hook)
            for layer in model.model.language_model.layers
        ]
        self.n = len(self.handles)

    def hook(self, _module, _inputs, out):
        torch = self.torch
        logits, _, indices = out
        probs = torch.softmax(logits.float(), dim=-1)
        p89 = torch.topk(probs, ROUTE_TOP_K + 1, dim=-1).values[:, ROUTE_TOP_K - 1 :]
        self.layers.append((indices.to(torch.int32), p89.float()))

    def take(self):
        assert len(self.layers) == self.n, f"{len(self.layers)} router calls, {self.n} layers"
        idx = self.torch.stack([i for i, _ in self.layers])
        p89 = self.torch.stack([p for _, p in self.layers])
        self.layers = []
        return idx, p89

    def remove(self):
        for h in self.handles:
            h.remove()


def write_route(path, runs):
    """The f32 run's indices and probabilities, then the bf16 run's, little-endian."""
    with open(path, "wb") as f:
        for idx, p89 in runs:
            f.write(idx.contiguous().numpy().astype("<i4").tobytes())
            f.write(p89.contiguous().numpy().astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PINNED))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision", help="default: the model's pinned revision")
    args = ap.parse_args()
    revision = args.revision or PINNED[args.model][0]
    refuse_kernel_packages()
    files = safetensors_files(args.checkpoint)
    refuse_no_text_decoder(args.checkpoint, tensor_names(args.checkpoint, files))
    shas = {f: sha256(os.path.join(args.checkpoint, f)) for f in files}
    refuse_unpinned(args.model, revision, shas)

    import torch
    import transformers
    from transformers import AutoTokenizer

    class_name, load_kwargs, route = MODEL_CLASS[args.model]
    model_class = getattr(transformers, class_name)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    texts = list(enumerate(prompts(args.prompts), start=1))
    ids = {p: tok(text, add_special_tokens=False)["input_ids"] for p, text in texts}

    def load(dtype):
        m = model_class.from_pretrained(args.checkpoint, dtype=dtype, **load_kwargs)
        m.eval()
        want = load_kwargs.get("experts_implementation")
        got = m.config.text_config._experts_implementation if want else None
        if got != want:
            sys.exit(f"cuda_hf_reference: experts_implementation {got!r}, asked for {want!r}")
        return m

    # Float32 first: every prompt's logits, then the greedy steps (which make the sequences the
    # bfloat16 run replays), then the model is freed before the bfloat16 one loads.
    model = load(torch.float32)
    meta = {
        "model": args.model,
        "checkpoint": os.path.abspath(args.checkpoint),
        "revision": revision,
        "safetensors_sha256": shas,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vocab": model.config.text_config.vocab_size,
        "greedy_steps": GREEDY_STEPS,
        "prompts": [],
    }
    if "experts_implementation" in load_kwargs:
        meta["experts_implementation"] = load_kwargs["experts_implementation"]
    dump = RouteDump(torch, model) if route else None
    f32, greedy, f32_route = {}, {}, {}
    for p, _ in texts:
        t = time.time()
        f32[p] = logits(model, torch, ids[p])
        write_bin(os.path.join(args.out, f"ref_f32_{p}.bin"), f32[p])
        f32_route[p] = [dump.take()] if dump else []
        if p <= GREEDY_PROMPTS:
            seq, steps = list(ids[p]), []
            for _ in range(GREEDY_STEPS):
                last32 = logits(model, torch, seq)[-1]
                if dump:
                    idx, p89 = dump.take()
                    f32_route[p].append((idx[:, -1:], p89[:, -1:]))
                top = torch.topk(last32, 2)
                token = int(top.indices[0])
                steps.append((list(seq), token, (top.values[0] - top.values[1]).item(), last32))
                seq.append(token)
            greedy[p] = steps
            print(f"prompt {p}: greedy f32 {[s[1] for s in steps]}", flush=True)
        print(f"prompt {p}: float32 run {time.time() - t:.1f}s", flush=True)
    if dump:
        dump.remove()
    del model, dump
    gc.collect()

    # Bfloat16: the same inputs, each greedy step's sequence the float32 run's.
    model = load(torch.bfloat16)
    dump = RouteDump(torch, model) if route else None
    for p, text in texts:
        t = time.time()
        bf16 = logits(model, torch, ids[p])
        write_bin(os.path.join(args.out, f"ref_bf16_{p}.bin"), bf16)
        s = stats(torch, bf16, f32[p])
        print(
            f"prompt {p}: {len(ids[p])} tokens, HF-bf16 vs f32: min cosine {s['min_cos']:.6f}, "
            f"RMS {s['rms']:.6e}, max {s['max']:.6e}",
            flush=True,
        )
        entry = {"prompt": p, "text": text, "ids": ids[p], "bf16_vs_f32": s}
        bf16_route = [dump.take()] if dump else []
        if p <= GREEDY_PROMPTS:
            steps = []
            for seq, token, margin, last32 in greedy[p]:
                last16 = logits(model, torch, seq)[-1]
                if dump:
                    idx, p89 = dump.take()
                    bf16_route.append((idx[:, -1:], p89[:, -1:]))
                steps.append(
                    {
                        "token": token,
                        "margin": margin,
                        "bf16_max_err": (last16 - last32).abs().max().item(),
                    }
                )
            with open(os.path.join(args.out, f"greedy_{p}.json"), "w") as f:
                json.dump({"prompt": p, "steps": steps}, f, indent=1)
        if dump:
            write_route(os.path.join(args.out, f"route_{p}.bin"), [f32_route[p][0], bf16_route[0]])
            if p <= GREEDY_PROMPTS:
                runs = [
                    tuple(torch.cat([r[i] for r in rs[1:]], dim=1) for i in (0, 1))
                    for rs in (f32_route[p], bf16_route)
                ]
                write_route(os.path.join(args.out, f"route_greedy_{p}.bin"), runs)
        print(f"prompt {p}: bfloat16 run {time.time() - t:.1f}s", flush=True)
        meta["prompts"].append(entry)
    if route:
        meta["route"] = {
            "layers": dump.n,
            "top_k": ROUTE_TOP_K,
            "route_<p>.bin": "f32 run: indices [layers, L, 8] int32, then the rank-8 and rank-9 "
            "router probabilities [layers, L, 2] float32 (softmax of the router logits in f32); "
            "then the bf16 run's, the same; little-endian",
            "route_greedy_<p>.bin": "prompts 1-5: the same, of each greedy step's last row, "
            f"[layers, {GREEDY_STEPS}, ...]",
        }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
