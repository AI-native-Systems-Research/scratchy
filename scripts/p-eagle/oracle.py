# P-EAGLE oracle in MLX: the target (mlx-lm's gpt-oss) and AWS's drafter, transcribed from vLLM's
# llama_eagle3.py forward and its parallel-drafting row layout (llm_base_proposer / utils.py
# copy_and_expand_eagle_inputs). Prints the drafts after the prompt's first sampled token and, over a
# greedy continuation, the drafts each step would make with their per-position acceptance.
# Usage: python oracle.py <target dir> <drafter dir> <prompt> [steps]
import json, os, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope

target_dir, drafter_dir, prompt = sys.argv[1:4]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 0
AUX = (2, 12, 21)
VOCAB = int(os.environ.get("VOCAB", 65536))  # the head's drafting vocabulary (0 = all)

model, tok = load(target_dir)
ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
print("prompt tokens", len(ids))


def target(tokens):
    m = model.model
    x = m.embed_tokens(mx.array([tokens]))
    full = create_attention_mask(x, None)
    swa = create_attention_mask(x, None, window_size=m.window_size)
    aux = []
    for i, (layer, lt) in enumerate(zip(m.layers, m.layer_types)):
        if i in AUX:
            aux.append(x)
        x = layer(x, full if lt == "full_attention" else swa, None)
    logits = model.lm_head(m.norm(x))
    return logits[0], mx.concatenate(aux, axis=-1)[0]


cfg = json.load(open(f"{drafter_dir}/config.json"))
W = mx.load(f"{drafter_dir}/model.safetensors")
# A quantized drafter (quantize_drafter.py's layout): dequantize each packed weight, and borrow the
# lm_head from the target's (bf16) checkpoint dir given as DRAFTER_LM_HEAD.
for base in [k[: -len(".scales")] for k in list(W) if k.endswith(".scales")]:
    q = cfg["quantization"]
    W[base + ".weight"] = mx.dequantize(W[base + ".weight"], W.pop(base + ".scales"), W.pop(base + ".biases"),
                                        group_size=q["group_size"], bits=q["bits"]).astype(mx.bfloat16)
if "lm_head.weight" not in W:
    W["lm_head.weight"] = mx.load(os.environ["DRAFTER_LM_HEAD"])["lm_head.weight"]
# Simulated quantization of a bf16 drafter: QUANT="<substr>:<bits>,..." rounds each matching 2-D
# weight (not the lm_head, its target's) through MLX affine g64 at that width.
for spec_ in filter(None, os.environ.get("QUANT", "").split(",")):
    pat, bits = spec_.split(":")
    for k in list(W):
        if pat in k and k.endswith(".weight") and W[k].ndim == 2 and "norm" not in k and k != "lm_head.weight":
            wq, s, b = mx.quantize(W[k], group_size=64, bits=int(bits))
            W[k] = mx.dequantize(wq, s, b, group_size=64, bits=int(bits)).astype(mx.bfloat16)
print("quant", os.environ.get("QUANT", "none"))
H, NH, KVH, HD, EPS = 2880, cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
rope = initialize_rope(HD, cfg["rope_theta"], False, cfg["rope_scaling"], cfg["max_position_embeddings"])
MASK = cfg["ptd_token_id"]
mask_hidden = W["mask_hidden"].reshape(-1)


def rms(x, w):
    return mx.fast.rms_norm(x, w, EPS)


def lin(x, name):
    return x @ W[name].T


def drafter(tokens, hidden, start):
    """Rows `tokens` / `hidden` ([n, 8640]) at positions start.. causal over themselves. `start`
    may be a list of `(first row, position)` segments: rows from each sit at positions from its."""
    n = len(tokens)
    h = lin(hidden, "fc.weight")
    emb = W["embed_tokens.weight"][mx.array(tokens)]
    segments = start if isinstance(start, list) else [(0, start)]
    bounds = [r for r, _ in segments[1:]] + [n]

    def rope_rows(x):
        parts = [rope(x[:, :, r:e], offset=p) for (r, p), e in zip(segments, bounds)]
        return mx.concatenate(parts, axis=2)

    mask = mx.triu(mx.full((n, n), -mx.inf, dtype=mx.float32), k=1)
    for l in range(cfg["num_hidden_layers"]):
        p = "midlayer." if l == 0 else f"layers.{l}."
        if l == 0:
            x = mx.concatenate([rms(emb, W[p + "input_layernorm.weight"]), rms(h, W[p + "hidden_norm.weight"])], axis=-1)
        else:
            x = rms(h, W[p + "input_layernorm.weight"])
        q = lin(x, p + "self_attn.q_proj.weight").reshape(1, n, NH, HD).transpose(0, 2, 1, 3)
        k = lin(x, p + "self_attn.k_proj.weight").reshape(1, n, KVH, HD).transpose(0, 2, 1, 3)
        v = lin(x, p + "self_attn.v_proj.weight").reshape(1, n, KVH, HD).transpose(0, 2, 1, 3)
        q, k = rope_rows(q), rope_rows(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HD ** -0.5, mask=mask.astype(q.dtype))
        h = h + lin(o.transpose(0, 2, 1, 3).reshape(n, NH * HD), p + "self_attn.o_proj.weight")
        x2 = rms(h, W[p + "post_attention_layernorm.weight"])
        g = lin(x2, p + "mlp.gate_proj.weight")
        h = h + lin((g * mx.sigmoid(g)) * lin(x2, p + "mlp.up_proj.weight"), p + "mlp.down_proj.weight")
    logits = lin(rms(h, W["norm.weight"]), "lm_head.weight")
    return logits[:, :VOCAB] if VOCAB else logits


PRIME_WINDOW = int(os.environ.get("PRIME_WINDOW", 0))  # prime only the prompt's last W rows (0: all)
PRIME_SINK = int(os.environ.get("PRIME_SINK", 0))  # and its first S rows


def drafts(seq, aux, k=3):
    """The K drafts after `seq` (prompt + sampled so far): rows 0..len-2 pair token p+1 with the
    target hidden at p; then K-1 mask rows. With PRIME_WINDOW the head has KV only from the
    prompt's last W rows on (and its first PRIME_SINK rows): its rows are those."""
    L = len(seq) - 1
    start = max(0, len(ids) - PRIME_WINDOW) if PRIME_WINDOW else 0
    sink = min(PRIME_SINK, start)
    rows = list(range(sink)) + list(range(start, L))
    toks = [seq[r + 1] for r in rows] + [MASK] * (k - 1)
    hid = mx.concatenate([aux[mx.array(rows)], mx.broadcast_to(mask_hidden, (k - 1, mask_hidden.shape[0]))], axis=0)
    segments = [(0, 0), (sink, start)] if sink else start
    logits = drafter(toks, hid, segments)
    return [int(t) for t in mx.argmax(logits[len(rows) - 1:], axis=-1).tolist()]


seq = list(ids)
logits, aux = target(seq)
for r in (0, len(ids) - 1):
    a = aux[r].astype(mx.float32)
    print(f"oracle aux row {r}: l2 {[round(float(x), 4) for x in a[:3]]} l12 {[round(float(x), 4) for x in a[2880:2882]]} l21 {[round(float(x), 4) for x in a[5760:5762]]}")
seq.append(int(mx.argmax(logits[-1])))
d = drafts(seq, aux)
print("first sampled", seq[-1], repr(tok.decode([seq[-1]])), "drafts", d, [tok.decode([t]) for t in d])
if steps == 0:
    # What the first drafts would be had the exports been the residual stream at later layers
    # (an in-place residual update clobbering an export before the concatenation reads it).
    def at_layers(layers):
        m = model.model
        x = m.embed_tokens(mx.array([ids]))
        full = create_attention_mask(x, None)
        swa = create_attention_mask(x, None, window_size=m.window_size)
        h = {}
        for i, (layer, lt) in enumerate(zip(m.layers, m.layer_types)):
            h[i] = x
            x = layer(x, full if lt == "full_attention" else swa, None)
        h[len(m.layers)] = x
        return mx.concatenate([h[l] for l in layers], axis=-1)[0]
    def show(what, toks, hid, start=0, rows=None):
        logits = drafter(toks, hid, start)
        L = len(seq) - 1
        d = [int(t) for t in mx.argmax(logits[L - 1:], axis=-1).tolist()]
        print(what, d, [tok.decode([t]) for t in d])
    L = len(seq) - 1
    base_t = seq[1:] + [MASK] * 2
    mh = mx.broadcast_to(mask_hidden, (2, mask_hidden.shape[0]))
    base_h = mx.concatenate([aux[:L], mh], axis=0)
    show("reference", base_t, base_h)
    show("newest token 0", base_t[:L - 1] + [0] + base_t[L:], base_h)
    for r in (0, 1, 2, 5, 40):
        h = mx.concatenate([aux[:L - 1], aux[r : r + 1], mh], axis=0)
        show(f"newest hidden of row {r}", base_t, h)
    show("prompt hidden zero", base_t, mx.concatenate([mx.zeros_like(aux[:L - 1]), aux[L - 1 : L], mh], axis=0))
    show("positions from 1", base_t, base_h, start=1)
    # Tokens not shifted (row p reads token p, not p+1).
    show("tokens unshifted", seq[:-1] + [MASK] * 2, base_h)

    def alone(what, tok_in, hid_in, start):
        logits = drafter([tok_in], hid_in[None, :], start)
        t = int(mx.argmax(logits[0]))
        print(what, t, repr(tok.decode([t])))
    import os
    import numpy as np
    dump = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"export-row-{L - 1}.bin")
    if os.path.exists(dump):
        raw = np.frombuffer(open(dump, "rb").read(), dtype=np.uint16).astype(np.uint32) << 16
        mine = mx.array(raw.view(np.float32)).astype(aux.dtype)
        alone("scratchy's export as the newest, alone", seq[-1], mine, L - 1)
        show("scratchy's export as the newest, in context", base_t,
             mx.concatenate([aux[:L - 1], mine[None, :], mh], axis=0))
    alone("newest alone at its position", seq[-1], aux[L - 1], L - 1)
    alone("newest alone at position 0", seq[-1], aux[L - 1], 0)
    alone("row-0 hidden alone at position 0", seq[-1], aux[0], 0)
    alone("row-1 hidden alone at position 0", seq[-1], aux[1], 0)
    alone("row-1 hidden alone at its position", seq[-1], aux[1], L - 1)
if steps:
    # Greedy speculative decoding, simulated: each step drafts after the newest token (its
    # hidden state the target's at the token before it), the target's greedy continuation is the
    # truth, and the step keeps the drafts that match it plus the target's next token.
    # FORCED=<reply.txt>: the truth is a streamed reply's tokens (`<token_N>`, scratchy's), not
    # the target's own greedy continuation; KMAX caps the drafts a step keeps (1: one draft).
    forced = None
    if os.environ.get("FORCED"):
        import re
        forced = [int(t) for t in re.findall(r"<token_(\d+)>", open(os.environ["FORCED"]).read())]
        if forced[0] != 200005:
            forced.insert(0, 200005)
        if forced[1] == 35644 and forced[2] != 200008:
            forced.insert(2, 200008)
        seq = list(ids) + forced[:1]
    kmax = int(os.environ.get("KMAX", 3))
    hits, made = [0, 0, 0], 0
    # Forced, the target's hidden states are the whole sequence's, once (causal: a prefix's rows
    # are its own).
    aux_all = target(list(ids) + forced)[1] if forced is not None else None
    for _ in range(steps):
        pos = len(seq) - len(ids)
        if forced is not None and pos + 4 > len(forced):
            break
        aux = aux_all if aux_all is not None else target(seq[:-1])[1]
        d = drafts(seq, aux)
        truth, s2 = [], list(seq)
        if forced is not None:
            truth = forced[pos:pos + 4]
        for _ in range(4 if forced is None else 0):
            lg, _ = target(s2)
            truth.append(int(mx.argmax(lg[-1])))
            s2.append(truth[-1])
        run = next((j for j in range(kmax) if d[j] != truth[j]), kmax)
        made += 1
        for j in range(run):
            hits[j] += 1
        seq += truth[: run + 1]
    print("steps", made, "P(>=1,2,3)", [round(h / made, 2) for h in hits],
          "tokens/step", round(1 + sum(hits) / made, 2))
