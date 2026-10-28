#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Score scratchy's greedy answer under mlx-lm, against mlx-lm's own greedy answer.

    parity_ties.py --model ID --prompt TEXT --tokens IDS.json [--chat-template-config JSON]

The parity gate (`scr bench startup --exec --parity-cmd`) compares the two answers' text and
calls this only when they differ. It renders PROMPT with the checkpoint's chat template, as
mlx_lm.generate does, and runs one mlx-lm forward over the prompt and scratchy's answer token
ids (IDS.json, a JSON list): mlx-lm teacher-forced on scratchy's tokens, so the comparison goes
on past a split instead of ending at it. At each answer token it takes mlx-lm's logit gap from
its top token to scratchy's: 0 where mlx-lm would pick the same token, small where the two are
a near-tie that two correct stacks break differently, large where scratchy's token is one
mlx-lm would not pick.

It also scores the whole answer: the mean log-probability per token mlx-lm gives scratchy's
answer, teacher-forced, and the same for mlx-lm's own greedy answer of the same length. A
correct stack that broke a near-tie differently writes text the reference finds about as likely
as its own; a wrong computation writes text it finds far less likely, at nearly every token.

Prints one JSON object: the gap at each token where mlx-lm's top differs, each with both
tokens' text, the largest gap, and both answers' mean log-probability. The gate decides; this
only measures. Needs mlx and mlx-lm.
"""

import argparse
import json
import sys


def gaps(logits, start, answer):
    """(index, top id, gap to scratchy's token) where mlx-lm's top token is not scratchy's.
    `logits` is (positions, vocab); position `start - 1 + i` predicts answer token i."""
    import mlx.core as mx

    out = []
    for i, t in enumerate(answer):
        row = logits[start - 1 + i]
        top = int(mx.argmax(row).item())
        if top != t:
            out.append((i, top, float((row[top] - row[t]).item())))
    return out


def mean_logprob(model, head, answer):
    """Mean log-probability per token mlx-lm gives `answer` after `head`, teacher-forced."""
    import mlx.core as mx

    logits = model(mx.array([head + answer]))[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return sum(lp[len(head) - 1 + i, t].item() for i, t in enumerate(answer)) / len(answer)


def greedy(model, head, n, eos):
    """mlx-lm's own greedy answer: up to `n` tokens after `head`, ending at an EOS."""
    import mlx.core as mx

    out = []
    while len(out) < n:
        t = int(mx.argmax(model(mx.array([head + out]))[0, -1]).item())
        out.append(t)
        if t in eos:
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", required=True, help="scratchy's answer token ids, a JSON list")
    ap.add_argument("--chat-template-config", default=None,
                    help="extra chat template arguments, as mlx_lm.generate takes them")
    a = ap.parse_args(argv)

    import mlx.core as mx
    from mlx_lm import load

    with open(a.tokens) as f:
        answer = json.load(f)
    if not (isinstance(answer, list) and answer and all(isinstance(t, int) for t in answer)):
        sys.exit(f"{a.tokens}: not a non-empty list of token ids")
    model, tok = load(a.model)
    extra = json.loads(a.chat_template_config) if a.chat_template_config else {}
    head = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                   add_generation_prompt=True, **extra)
    logits = model(mx.array([head + answer]))[0].astype(mx.float32)
    found = gaps(logits, len(head), answer)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    reference = greedy(model, head, len(answer), eos)
    json.dump({
        "tokens": len(answer),
        "differ": [{"at": i, "scratchy": tok.decode([answer[i]]), "mlx_lm": tok.decode([top]),
                    "gap": round(g, 4)} for i, top, g in found],
        "max_gap": round(max((g for *_, g in found), default=0.0), 4),
        "mean_logprob": round(mean_logprob(model, head, answer), 4),
        "reference_mean_logprob": round(mean_logprob(model, head, reference), 4),
    }, sys.stdout)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
