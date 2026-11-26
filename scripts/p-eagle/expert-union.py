# The experts a verify step reads, against a plain step's: over the target's own greedy replies (the
# model card's benchmarks), each layer's routed experts per token, and for every window of W
# consecutive reply tokens (a verify step's rows: the newest token and W - 1 drafts) the distinct
# experts per layer over top_k — 1.0 is a plain step's expert reads, W the rows reading apart.
# Usage: expert-union.py <target> <bench dir> [prompts a benchmark] [max tokens]
import importlib
import json
import os
import statistics
import sys

import mlx.core as mx
from mlx_lm import load

generate = importlib.import_module("mlx_lm.generate")
gpt_oss = importlib.import_module("mlx_lm.models.gpt_oss")
target_dir, bench = sys.argv[1:3]
per_set = int(sys.argv[3]) if len(sys.argv) > 3 else 3
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256
model, tok = load(target_dir)
top_k = model.model.layers[0].mlp.num_experts_per_tok
experts = model.model.layers[0].mlp.num_local_experts
print(f"{len(model.model.layers)} layers, {experts} experts, top {top_k}", flush=True)

picks = []  # one [tokens, top_k] array per layer, in layer order, per recorded forward
recording = [False]
call = gpt_oss.MLPBlock.__call__


def recorded(self, x):
    if recording[0]:
        g = self.router(x)
        _, indices = gpt_oss.mlx_topk(g, k=self.num_experts_per_tok, axis=-1)
        picks.append(indices.reshape(-1, self.num_experts_per_tok))
    return call(self, x)


gpt_oss.MLPBlock.__call__ = recorded


def prompts():
    if os.path.exists(f"{bench}/mtbench.jsonl"):
        rows = lambda f: [r["row"] for r in json.load(open(f"{bench}/{f}"))["rows"]]
        return {
            "mtbench": [json.loads(line)["prompt"][0] for line in open(f"{bench}/mtbench.jsonl")],
            "humaneval": [r["prompt"] for r in rows("humaneval.json")],
            "gsm8k": [r["question"] for r in rows("gsm8k.json")],
        }
    return {name: json.load(open(f"{bench}/{name}.json")) for name in ("mtbench", "humaneval", "gsm8k")}


greedy = lambda logits: mx.argmax(logits, axis=-1)
eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
windows = (2, 3, 5)
for name, texts in prompts().items():
    ratios = {w: [] for w in windows}
    for text in texts[:per_set]:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        reply = []
        for token, _ in generate.generate_step(mx.array(ids), model, max_tokens=max_tokens, sampler=greedy):
            reply.append(int(token))
            if reply[-1] in eos:
                break
        picks.clear()
        recording[0] = True
        mx.eval(model(mx.array([list(ids) + reply])))
        recording[0] = False
        layers = [p[len(ids):].tolist() for p in picks]
        for w in windows:
            for start in range(0, len(reply) - w + 1):
                per_layer = [len({e for row in layer[start:start + w] for e in row}) / top_k for layer in layers]
                ratios[w].append(statistics.mean(per_layer))
    print(f"{name}: " + " | ".join(
        f"{w} rows read {statistics.mean(r):.2f}x a plain step's experts (apart: {w})"
        for w, r in ratios.items()), flush=True)
