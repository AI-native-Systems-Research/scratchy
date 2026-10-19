#!/usr/bin/env python3
"""The rung-1 scratchy harness client, length-calibrated (#677).

PROMPT_TOKENS is the length the SERVER must record -- the sweep file's
contexts value for the bucket, not a client-side word count. With
PROMPTS_DIR the client sends the 10-18 sweep's own prompt texts: per rep
"Run {r}.\\n{base}" over the staged base file whose rendered prompt
tokenizes closest to the target (sweep_prompts.py's exact formula -- the
like-for-like text with the 2026-10-18 cells), refusing any base off the
target by more than 2%. Without PROMPTS_DIR it binary-searches a
deterministic filler's parameter against the model's own tokenizer and
chat template until the rendered prompt tokenizes to that target (within
2%) -- the fallback generator. Either way it sends REPS requests on
distinct prompts and records the serve-reported length from each
stream's usage chunk (stream_options.include_usage). A rep whose
serve-recorded length is present but off the target by more than 2%
fails the whole cell -- never emit a cell over a prompt measured at the
wrong length. The serve's request-finished log lines are the
cross-check and the authoritative gate (the leg wrapper checks them).

History: the 2026-10-19 sweep ran this client's precursor, which sized
prompts by a 1.35-words-per-token heuristic; its prompts recorded 1.5x
their parameters (207/1614/15209 for 128/1061/10091), so that sweep's
rung-1 cells are pending calibrated re-measurement (#677).

Prints one JSON object on stdout:
  {"ttft_ms": ..., "itl_ms": ..., "prompt_tokens": <target>,
   "prompt_param": <filler parameter or null>, "prompt_base": <base
   filename or null>, "reps": [{"ttft_ms": ..., "itl_p50_ms": ...,
   "chunks": n, "recorded_prompt_tokens": k}]}

Exits non-zero if the target cannot be hit within 2%, transformers or
the tokenizer cannot be loaded, any rep produced no content chunks, a
request failed, or a serve-reported length is present but off by more
than 2%. This serve build sends no usage chunk (measured 2026-10-19:
stream_options.include_usage set, no usage in the stream), so
recorded_prompt_tokens is normally None; the authoritative length gate
is the leg's check of the serve's request-finished log lines.

Usage: stream_client.py URL MODEL TARGET_PROMPT_TOKENS REPS OUT_TOKENS [PROMPTS_DIR]

Runs under the reference venv (it imports transformers for the
tokenizer); the leg wrapper is:
  cd /work/spyre-inference && TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  HF_HUB_OFFLINE=1 uv run --no-sync python stream_client.py ...
"""

import glob
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

TOLERANCE = 0.02  # serve-recorded length must land within 2% of target

SENTENCE = (
    "The granite model streams one token at a time while the card holds "
    "every weight in place and the host submits one command buffer per step. "
)


def prompt(n_tokens, rep):
    """Deterministic filler sized in prompt-parameter units, distinct per
    rep. The server records the true length; calibrate() picks the
    parameter that hits the target."""
    want_words = max(16, int(n_tokens * 1.35))
    words = SENTENCE.split()
    out = []
    while len(out) < want_words:
        out.extend(words)
    out = out[:want_words]
    out[0] = f"Rep{rep}:" + out[0]
    return " ".join(out)


def template_tokens(tok, user_text):
    """The count that matches what the serve records: render the chat
    template (empty system message, as the client sends it) and tokenize
    without special tokens. Verified against the serve's recorded
    lengths (#677: parameter 6692 -> serve-recorded 10092)."""
    msgs = [{"role": "system", "content": ""},
            {"role": "user", "content": user_text}]
    render = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     tokenize=False)
    return len(tok(render, add_special_tokens=False)["input_ids"])


def texts_from_dir(tok, prompts_dir, target, reps):
    """The 10-18 sweep's own prompt texts (the like-for-like text for the
    rung-1 cells): per rep "Run {r}.\\n{base}" over the staged base whose
    rendered prompt tokenizes closest to the target -- sweep_prompts.py's
    exact formula. Refuses unless every rep lands within 2%: a staged
    base that misses the bucket never produces a cell."""
    paths = sorted(glob.glob(os.path.join(prompts_dir, "*.txt")))
    if not paths:
        raise SystemExit(f"stream_client: REFUSED: no *.txt prompt bases "
                         f"in {prompts_dir}")
    best = None
    for path in paths:
        base = open(path, encoding="utf-8").read().strip()
        texts = [f"Run {r}.\n{base}" for r in range(1, reps + 1)]
        counts = [template_tokens(tok, t) for t in texts]
        dist = min(abs(c - target) for c in counts)
        if best is None or dist < best[0]:
            best = (dist, os.path.basename(path), counts, texts)
    _, base_name, counts, texts = best
    worst = max(abs(c - target) / target for c in counts)
    if worst > TOLERANCE:
        raise SystemExit(
            f"stream_client: REFUSED: base {base_name} tokenizes to "
            f"{counts} (target {target}, tolerance {TOLERANCE:.0%}) -- the "
            f"staged 10-18 base does not hit this bucket; never emit a "
            f"cell over a prompt measured at the wrong length")
    return base_name, counts, texts


def calibrate(tok, target, reps):
    """The smallest prompt parameter whose rendered prompt tokenizes to at
    least the target; refuses if the best fit is off by more than 2% for
    any rep's prefix. Monotone in the parameter, so binary search."""
    def count(n, rep):
        return template_tokens(tok, prompt(n, rep))

    lo, hi = 1, target * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid, 1) >= target:
            hi = mid
        else:
            lo = mid + 1
    n = lo
    counts = [count(n, r) for r in range(1, reps + 1)]
    worst = max(abs(c - target) / target for c in counts)
    if worst > TOLERANCE:
        raise SystemExit(
            f"stream_client: REFUSED: cannot hit target {target} within "
            f"{TOLERANCE:.0%}: parameter {n} tokenizes to {counts}")
    return n, counts


def one_rep(url, model, user_text, out_tokens, rep):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": out_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    stamps = []
    recorded = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                # the serve reports the prompt length in the usage chunk
                # (stream_options.include_usage); the usage chunk carries
                # an EMPTY choices array, so the delta read below must
                # tolerate it
                recorded = chunk["usage"].get("prompt_tokens")
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                stamps.append(time.monotonic())
    if not stamps:
        raise RuntimeError(f"rep {rep}: no content chunks in the stream")
    ttft = stamps[0] - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    itl_p50 = statistics.median(gaps) if gaps else 0.0
    return {
        "ttft_ms": ttft * 1000.0,
        "itl_p50_ms": itl_p50 * 1000.0,
        "chunks": len(stamps),
        "recorded_prompt_tokens": recorded,
    }


def main():
    if len(sys.argv) not in (6, 7):
        print(__doc__, file=sys.stderr)
        return 64
    url, model, target, reps, out_tokens = sys.argv[1:6]
    prompts_dir = sys.argv[6] if len(sys.argv) == 7 else None
    target, reps, out_tokens = int(target), int(reps), int(out_tokens)

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"stream_client: REFUSED: transformers is not importable "
              f"({e}) -- run under the reference venv", file=sys.stderr)
        return 69
    try:
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        print(f"stream_client: REFUSED: cannot load the tokenizer for "
              f"{model!r} ({e}) -- the client calibrates against the "
              f"model's own tokenizer", file=sys.stderr)
        return 69

    if prompts_dir is not None:
        base_name, counts, texts = texts_from_dir(tok, prompts_dir, target,
                                                  reps)
        param = None
        print(f"prompt base: {base_name} -> {counts} tokens "
              f"(target {target})", file=sys.stderr)
    else:
        base_name = None
        param, counts = calibrate(tok, target, reps)
        texts = [prompt(param, r) for r in range(1, reps + 1)]
        print(f"calibrated: parameter {param} -> {counts} tokens "
              f"(target {target})", file=sys.stderr)

    results = []
    for rep in range(1, reps + 1):
        results.append(one_rep(url, model, texts[rep - 1],
                               out_tokens, rep))
        print(f"rep {rep}: {results[-1]}", file=sys.stderr)

    for r in results:
        rec = r["recorded_prompt_tokens"]
        if rec is not None and abs(rec - target) / target > TOLERANCE:
            # this serve build sends no usage chunk (measured 10-19:
            # stream_options.include_usage set, no usage in the stream),
            # so rec is normally None and the serve's request-finished
            # log lines are the authoritative gate (leg.sh checks them);
            # a PRESENT-but-off value is still a hard refusal here
            print(f"stream_client: REFUSED: serve recorded {rec} "
                  f"(target {target}, tolerance {TOLERANCE:.0%}) -- never "
                  f"emit a cell over a prompt measured at the wrong "
                  f"length; the serve log is the cross-check",
                  file=sys.stderr)
            return 1

    cell = {
        "ttft_ms": statistics.median(r["ttft_ms"] for r in results),
        "itl_ms": statistics.median(r["itl_p50_ms"] for r in results),
        "prompt_tokens": target,
        "prompt_param": param,
        "prompt_base": base_name,
        "reps": results,
    }
    print(json.dumps(cell))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, RuntimeError, OSError) as e:
        print(f"stream_client: FAILED: {e}", file=sys.stderr)
        sys.exit(1)
