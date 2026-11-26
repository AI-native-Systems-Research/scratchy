# P-EAGLE's acceptance in MLX, independent of scratchy, at every draft count 1..KMAX at once: the
# ceiling a scratchy build can reach with this drafter on this target.
#
# oracle.py's model (the target: mlx-lm's gpt-oss; the drafter: AWS's, transcribed from vLLM's
# llama_eagle3.py and its parallel-drafting rows) without its per-step recomputation, so replies
# can be long and prompts many. Greedy decoding emits the target's own greedy reply whatever the
# drafts, and a step's drafts depend only on the sequence before them (the drafter is causal, its
# real rows each a verified token and the target's hidden state before it). So, per prompt:
#   1. the target's greedy reply (KV-cached), then ONE target forward over prompt + reply for the
#      exported hidden states (the input of layers AUX, side by side);
#   2. ONE causal drafter forward over every real row (row r: token r + 1 with hidden r, at
#      position r), keeping each layer's K/V; its logits at row b are draft 1 at the step whose
#      newest token is b + 1;
#   3. per step boundary b, the KMAX - 1 mask rows (positions b + 1 ..), attending to real rows
#      0..b and to the mask rows before them, all boundaries batched;
#   4. walk each draft count K: from the first sampled token, keep the drafts that match the reply
#      plus the target's token, jump, repeat. Tokens a step = 1 + accepted.
# The drafts come from the drafter's vocabulary prefix VOCAB (scratchy's: 65536) through the
# target's lm_head (the drafter borrows it), as scratchy serves it.
#
# Usage: python acceptance.py <target> <drafter dir> <bench dir> [prompts a set] [max tokens]
#   <bench dir> holds mtbench.json / humaneval.json / gsm8k.json (bench-prompts.py's lists).
#   Env: KMAX (default 7), VOCAB (65536; 0 = all), SETS (comma list), JSON=<path> per-step dump.
import json
import os
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope

target_dir, drafter_dir, bench = sys.argv[1:4]
per_set = int(sys.argv[4]) if len(sys.argv) > 4 else 20
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 512
KMAX = int(os.environ.get("KMAX", 7))
VOCAB = int(os.environ.get("VOCAB", 65536))
SETS = os.environ.get("SETS", "mtbench,humaneval,gsm8k").split(",")

model, tok = load(target_dir)
layers = model.model.layers
# EAGLE-3's exported layers: 2, the middle, and three from the end (vLLM's default for a target
# whose drafter config names none) — (2, 12, 21) on gpt-oss-20b's 24 layers.
cfg = json.load(open(os.path.join(drafter_dir, "config.json")))
AUX = tuple(cfg.get("eagle_aux_hidden_state_layer_ids") or (2, len(layers) // 2, len(layers) - 3))
print(f"target {target_dir}: {len(layers)} layers, exports {AUX}; drafter {drafter_dir}", flush=True)


def target_hidden(tokens):
    m = model.model
    x = m.embed_tokens(mx.array([tokens]))
    full = create_attention_mask(x, None)
    swa = create_attention_mask(x, None, window_size=m.window_size)
    aux = []
    for i, (layer, lt) in enumerate(zip(m.layers, m.layer_types)):
        if i in AUX:
            aux.append(x)
        x = layer(x, full if lt == "full_attention" else swa, None)
    return mx.concatenate(aux, axis=-1)[0]


W = mx.load(os.path.join(drafter_dir, "model.safetensors"))
quant = cfg.get("quantization", {})
for base in [k[: -len(".scales")] for k in list(W) if k.endswith(".scales")]:
    q = quant.get(base, quant)  # a mixed layout's per-module width, else the global one
    W[base + ".weight"] = mx.dequantize(W[base + ".weight"], W.pop(base + ".scales"), W.pop(base + ".biases"),
                                        group_size=q["group_size"], bits=q["bits"]).astype(mx.bfloat16)
NH, KVH, HD, EPS = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"]
NL = cfg["num_hidden_layers"]
rope = initialize_rope(HD, cfg["rope_theta"], False, cfg["rope_scaling"], cfg["max_position_embeddings"])
MASK = cfg["ptd_token_id"]
mask_hidden = W["mask_hidden"].reshape(-1)
print(f"drafter: {NL} layers, {NH}x{HD} heads over {KVH}, quant {'mixed' if quant else 'none'}, "
      f"KMAX {KMAX}, vocab {VOCAB or 'all'}", flush=True)


def rms(x, w):
    return mx.fast.rms_norm(x, w, EPS)


def lin(x, name):
    y = x @ W[name + ".weight"].T
    return y + W[name + ".bias"] if name + ".bias" in W else y


def head(h):
    logits = model.lm_head(rms(h, W["norm.weight"]))  # the target's lm_head, lent
    return logits[..., :VOCAB] if VOCAB else logits


def layer_in(l, emb, h):
    p = "midlayer." if l == 0 else f"layers.{l}."
    if l == 0:
        return mx.concatenate([rms(emb, W[p + "input_layernorm.weight"]), rms(h, W[p + "hidden_norm.weight"])], axis=-1)
    return rms(h, W[p + "input_layernorm.weight"])


def layer_out(l, h, o):
    p = "midlayer." if l == 0 else f"layers.{l}."
    h = h + lin(o, p + "self_attn.o_proj")
    x2 = rms(h, W[p + "post_attention_layernorm.weight"])
    g = lin(x2, p + "mlp.gate_proj")
    return h + lin((g * mx.sigmoid(g)) * lin(x2, p + "mlp.up_proj"), p + "mlp.down_proj")


def qkv(l, x, positions):
    """x [..., n, d] at `positions` [..., n] -> q [..., NH, n, HD], k/v [..., KVH, n, HD]."""
    p = "midlayer." if l == 0 else f"layers.{l}."
    lead, n = x.shape[:-2], x.shape[-2]
    q = lin(x, p + "self_attn.q_proj").reshape(*lead, n, NH, HD)
    k = lin(x, p + "self_attn.k_proj").reshape(*lead, n, KVH, HD)
    v = lin(x, p + "self_attn.v_proj").reshape(*lead, n, KVH, HD)
    q, k, v = (mx.swapaxes(t, -2, -3) for t in (q, k, v))
    if positions is None:  # real rows: positions 0..n-1
        return rope(q), rope(k), v
    # mask rows: each batch row b's rows at its own offset — rope at offset 0, then rotate per row
    # by position, done as rope over a [b, heads, n, HD] with offset per b via a loop-free gather.
    return rope_at(q, positions), rope_at(k, positions), v


_rope_cache = {}


def rope_at(x, positions):
    """Rope of x [B, H, n, HD] at positions [B, n]: the real-row rope's angles gathered per row."""
    top = int(positions.max().item()) + 1
    if top not in _rope_cache:
        # cos/sin per position from rope applied to unit vectors (traditional=False: halves).
        half = HD // 2
        e0 = mx.concatenate([mx.ones((1, 1, top, half)), mx.zeros((1, 1, top, half))], axis=-1)
        e1 = mx.concatenate([mx.zeros((1, 1, top, half)), mx.ones((1, 1, top, half))], axis=-1)
        r0, r1 = rope(e0), rope(e1)  # r0 = [cos, sin], r1 = [-sin, cos] per frequency
        assert mx.allclose(r1[..., :half], -r0[..., half:], atol=1e-5).item(), "rope is not half-split"
        _rope_cache[top] = (r0[0, 0, :, :half], r0[0, 0, :, half:])
    cos, sin = _rope_cache[top]
    c, s = cos[positions][:, None], sin[positions][:, None]  # [B, 1, n, half]
    half = HD // 2
    x1, x2 = x[..., :half].astype(mx.float32), x[..., half:].astype(mx.float32)
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1).astype(x.dtype)


def all_drafts(reply_ids, n_prompt):
    """drafts[b] = the KMAX drafts at the step whose last real drafter row is b."""
    seq = reply_ids
    aux = target_hidden(seq[:-1])  # hidden before AUX layers at positions 0..N-2
    n = len(seq) - 1
    toks = mx.array(seq[1:])
    emb = W["embed_tokens.weight"][toks]
    h = lin(aux, "fc")
    causal = mx.triu(mx.full((n, n), -mx.inf, dtype=mx.float32), k=1).astype(mx.bfloat16)
    ks, vs = [], []
    for l in range(NL):
        x = layer_in(l, emb, h)
        q, k, v = qkv(l, x[None], None)
        ks.append(k[0])
        vs.append(v[0])
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HD ** -0.5, mask=causal)
        h = layer_out(l, h, o[0].swapaxes(0, 1).reshape(n, NH * HD))
    first = mx.argmax(head(h), axis=-1)  # [n]: draft 1 at boundary b
    mx.eval(first, *ks, *vs)
    out = [[int(t)] for t in first.tolist()]
    if KMAX == 1:
        return out
    # Mask rows: boundaries from the prompt's last row on (earlier ones are never a step's).
    J = KMAX - 1
    bs_all = list(range(n_prompt - 1, n))
    memb = W["embed_tokens.weight"][MASK]
    mfc = lin(mask_hidden[None], "fc")[0]
    chunk = 64
    for c0 in range(0, len(bs_all), chunk):
        bs = mx.array(bs_all[c0:c0 + chunk])
        B = bs.shape[0]
        positions = bs[:, None] + mx.arange(1, J + 1)[None]  # [B, J]
        hm = mx.broadcast_to(mfc, (B, J, mfc.shape[0]))
        em = mx.broadcast_to(memb, (B, J, memb.shape[0]))
        # real key j visible to boundary b iff j <= b; mask key i visible to mask query j iff i <= j
        real_ok = mx.arange(n)[None, :] <= bs[:, None]  # [B, n]
        own_ok = mx.tril(mx.ones((J, J), dtype=mx.bool_))
        bias = mx.concatenate([
            mx.where(mx.broadcast_to(real_ok[:, None, :], (B, J, n)), 0.0, -mx.inf),
            mx.where(mx.broadcast_to(own_ok[None], (B, J, J)), 0.0, -mx.inf)], axis=-1)
        bias = bias[:, None].astype(mx.bfloat16)  # [B, 1, J, n + J]
        for l in range(NL):
            x = layer_in(l, em, hm)
            q, k, v = qkv(l, x, positions)  # [B, heads, J, HD]
            kr = mx.broadcast_to(ks[l][None], (B, *ks[l].shape))
            vr = mx.broadcast_to(vs[l][None], (B, *vs[l].shape))
            kk = mx.concatenate([kr, k], axis=2)
            vv = mx.concatenate([vr, v], axis=2)
            o = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=HD ** -0.5, mask=bias)
            hm = layer_out(l, hm, o.swapaxes(1, 2).reshape(B, J, NH * HD))
        d = mx.argmax(head(hm), axis=-1)  # [B, J]
        mx.eval(d)
        for b, row in zip(bs_all[c0:c0 + chunk], d.tolist()):
            out[b] += [int(t) for t in row]
    return out


def reference(seq, b):
    """oracle.py's drafts at boundary b: one causal forward over real rows 0..b and the mask rows."""
    aux = target_hidden(seq[:b + 1])
    toks = mx.array(seq[1:b + 2] + [MASK] * (KMAX - 1))
    hid = mx.concatenate([aux, mx.broadcast_to(mask_hidden, (KMAX - 1, mask_hidden.shape[0]))], axis=0)
    n = toks.shape[0]
    emb, h = W["embed_tokens.weight"][toks], lin(hid, "fc")
    causal = mx.triu(mx.full((n, n), -mx.inf, dtype=mx.float32), k=1).astype(mx.bfloat16)
    for l in range(NL):
        q, k, v = qkv(l, layer_in(l, emb, h)[None], None)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HD ** -0.5, mask=causal)
        h = layer_out(l, h, o[0].swapaxes(0, 1).reshape(n, NH * HD))
    return [int(t) for t in mx.argmax(head(h[b:]), axis=-1).tolist()]


def walk(seq, drafts, n_prompt, k):
    """Greedy speculative decoding at k drafts over the reply: tokens a step, per step."""
    steps = []
    b = n_prompt - 1  # the first sampled token (seq[n_prompt]) is the prompt's; its row is b
    end = len(seq) - 1
    while b + 1 < end:
        truth = seq[b + 2:b + 2 + k]
        d = drafts[b][:k]
        run = next((j for j in range(len(truth)) if d[j] != truth[j]), len(truth))
        steps.append(run + 1)
        b += run + 1
    return steps


sets = {name: json.load(open(os.path.join(bench, f"{name}.json"))) for name in SETS}
eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
greedy = lambda logits: mx.argmax(logits, axis=-1)
dump = {}
for name, prompts in sets.items():
    per_k = {k: [] for k in range(1, KMAX + 1)}
    lens = []
    t0 = time.time()
    for text in prompts[:per_set]:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
        reply = []
        for token, _ in generate_step(mx.array(ids), model, max_tokens=max_tokens, sampler=greedy):
            reply.append(int(token))
            if reply[-1] in eos:
                break
        seq = list(ids) + reply
        lens.append(len(reply))
        drafts = all_drafts(seq, len(ids))
        if os.environ.get("CHECK"):
            for b in (len(ids) - 1, len(ids) + len(reply) // 2, len(seq) - 3):
                print(f"check b={b}: batched {drafts[b]} reference {reference(seq, b)}", flush=True)
        for k in per_k:
            per_k[k] += walk(seq, drafts, len(ids), k)
        if os.environ.get("JSON"):
            dump.setdefault(name, []).append({"prompt": len(ids), "seq": seq, "drafts": drafts[len(ids) - 1:]})
    tps = {k: sum(s) / len(s) for k, s in per_k.items()}
    print(f"{name}: {len(lens)} prompts, reply tokens median {statistics.median(lens):.0f} "
          f"(max {max(lens)}), {time.time() - t0:.0f} s; tokens/step "
          + " ".join(f"k{k} {v:.2f}" for k, v in tps.items()), flush=True)
if os.environ.get("JSON"):
    json.dump(dump, open(os.environ["JSON"], "w"))
