#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare `scr chat` / `scr serve` on CUDA with HuggingFace's greedy qwen3.5-0.8b: #838's H100
check. Prints PASS or FAIL per item, both sides' text, and exits 1 on any FAIL.

`--mode chat` reads `<logs>/chat<p>.log` (prompts 1-5, `--device cuda`) and `<logs>/chatauto.log`
(prompt 1, no `--device`), each a `VLLM_PRINT_TOKEN_IDS=1 scr chat --temperature 0 --max-tokens 16
-q` run's output, and compares the reply against `--ref` (`cuda_hf_chat_reference.py`'s JSON). The
reply is what follows the `Using model:` line, tracing lines dropped; the `[diag] token_ids=` line
gives its token ids.

THE INPUT FIRST. `scr chat` renders its prompt through another call than the server's
`/tokenize`, so each chat prompt's ids as `scr chat` fed them (the `[diag] prompt_token_ids=` line)
are compared with HF's chat-templated ids first: MATCH, or MISMATCH at the first differing index. A
mismatch, or no such line, is a FAIL reported before the reply is compared, since the reply of
another input says nothing about the kernels.

THE EDGES. A 1-token prompt (a fresh first step at token 0 on the one-row decode tape) and a 4097-
token prompt (a 4096 chunk and a remainder of one): `--mode serve` sends both as `/v1/completions`
ids. A chat turn cannot be one token (the template alone is longer), so `scr chat` covers the 4097
case only: `--edge-prompts DIR` writes `edge4097.txt`, a `-q` text whose rendered prompt is 4097
tokens, and `--mode edges` reads that run's `<logs>/edge4097.log`. Each must give a non-empty reply
and no error line; the text is not compared.

`--mode serve` talks to a running `scr serve` at `--url`, in one process, in this order:
  1. precondition: `/tokenize` of each prompt's chat messages is HF's chat-templated ids (if not,
     the template differs, not the kernels, and nothing else is run);
  2. `/v1/completions` on the `--goldens` ids (`meta.json`, no template) for prompts 1-5, 8
     tokens, against `greedy_<p>.json`;
  3. `/v1/chat/completions` on prompts 1-5, 16 tokens, against the chat reference, with the
     yardstick columns (this path's input ids are the ones step 1 checked);
  4. two `/v1/completions` at once (prompts 1 and 2): both 200 with step 2's texts;
  5. 8193 ids: a 4xx naming 8192;
  6. temperature 0.7: an error naming greedy-only sampling;
  7. 5000 ids, 4 tokens: 200 (smoke only; its text is not compared);
  8. the edges below: 1 and 4097 ids, 8 tokens each, 200 with a non-empty reply.

THE COMPARISON. The two stacks honour different stop ids for this checkpoint (the reference
records both): scr stops at `<|im_end|>`, HF's `generate` at `<|endoftext|>`. The reference never
stops, so with U the union of both sets, k is the index of HF's first token in U (or all of them).
scr must give HF's first k tokens exactly; past k it may stop (a stop it honours) or go on (one it
does not), and only the first k are compared. Per prompt it prints k, which stop id ended each
side, and whether the comparison was truncated.

THE NEAR-TIE RULE (#816 P5): at most one of the five prompts may diverge before k, only at a step
whose float32 top-1 minus top-2 margin is below twice that step's bfloat16 max error, and the text
before the divergence must match.

THE YARDSTICK (evidence for that rule, no verdict): when the reference carries HF-bf16's own greedy
tokens, each chat prompt also prints where HF-bf16 departs from HF-f32, and where CUDA departs from
HF-f32 and from HF-bf16 (CUDA's tokens need `VLLM_PRINT_TOKEN_IDS=1` on the `scr chat` run).

Needs `transformers` (the tokenizer only) and the standard library.
"""

import argparse
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request

LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\dT\S+\s+(TRACE|DEBUG|INFO|WARN|ERROR)\b")
DIAG = re.compile(r"\[diag\] token_ids=\[([0-9, ]*)\]")
PROMPT_DIAG = re.compile(r"\[diag\] prompt_token_ids=\[([0-9, ]*)\]")
ERROR_LINE = re.compile(r"^Error:|\bERROR\b|panicked")

results = []


def report(item, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {item}{': ' + detail if detail else ''}", flush=True)


def first_stop(tokens, stops):
    """k: the index of the first token in `stops`, else len(tokens)."""
    return next((i for i, t in enumerate(tokens) if t in stops), len(tokens))


class Ref:
    """One reference reply: its tokens, per-step margins and noise, and the stop sets."""

    def __init__(self, tok, steps, stop_ids):
        self.tok = tok
        self.tokens = [s["token"] for s in steps]
        self.steps = steps
        self.scr_stops = set(stop_ids["scr"])
        self.hf_stops = set(stop_ids["hf"])
        self.union = self.scr_stops | self.hf_stops
        self.k = first_stop(self.tokens, self.union)

    def text(self, n):
        return self.tok.decode(self.tokens[:n], skip_special_tokens=True)

    def near_tie(self, step):
        s = self.steps[step]
        return s["margin"] < 2 * s["bf16_max_err"], s["margin"], s["bf16_max_err"]


def compare(name, ref, got_text, got_tokens=None):
    """One prompt: `None` when it matches through k, else the divergence step (an int), or
    `False` when it differs in a way the near-tie rule cannot excuse."""
    k = ref.k
    hf_end = ref.tokens[k] if k < len(ref.tokens) else None
    truncated = k < len(ref.tokens)
    print(f"--- {name}: k={k}, HF's stop {'id ' + str(hf_end) if hf_end is not None else 'none'} in "
          f"its {len(ref.tokens)} tokens, comparison {'truncated at k' if truncated else 'whole'}")
    print(f"    HF  : {ref.text(k)!r}")
    print(f"    scr : {got_text!r}")
    if got_tokens is not None:
        scr_end = next((t for t in got_tokens if t in ref.scr_stops), None)
        print(f"    scr tokens {got_tokens}; scr's stop {scr_end if scr_end is not None else 'none'}")
        print(f"    HF  tokens {ref.tokens[:k]}")
        upto = min(k, len(got_tokens))
        div = next((i for i in range(upto) if got_tokens[i] != ref.tokens[i]), None)
        if div is None and len(got_tokens) < k:
            # scr stopped before HF's first stop: at a token HF does not stop at.
            div = len(got_tokens)
        if div is None and hf_end in ref.scr_stops and len(got_tokens) > k:
            # A stop scr honours: scr emits it at k (the id may or may not be listed) and ends.
            if got_tokens[k] != hf_end:
                div = k
            elif len(got_tokens) > k + 1:
                print("    scr went past a stop id it honours")
                return False
        return div
    got = got_text.strip()
    want = ref.text(k).strip()
    if got == want or (truncated and hf_end not in ref.scr_stops and got.startswith(want)):
        return None
    # The divergence step: the first j whose reference text no longer prefixes scr's.
    j = 0
    while j < k and got.startswith(ref.text(j + 1).strip()):
        j += 1
    if j == k:
        # Every reference token prefixes scr's text and scr's runs on past it: no step of the
        # reference is where they part, so it is not a near-tie divergence.
        print(f"    scr's text runs past the reference's {k} tokens")
        return False
    return j


def first_diff(a, b):
    """The first index where `a` and `b` differ, or None when the shorter is a prefix of the longer."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def text_diff(tok, got_text, tokens):
    """From text alone: the first step j whose decoded tokens[:j+1] no longer prefix `got_text`,
    or None when all of `tokens` do."""
    got = got_text.strip()
    j = 0
    while j < len(tokens) and got.startswith(tok.decode(tokens[: j + 1], skip_special_tokens=True).strip()):
        j += 1
    return None if j == len(tokens) else j


def yardstick(name, entry, got_tokens, ref, got_text=None):
    """Evidence only, no verdict: where HF-bf16 departs from HF-f32 on this prompt, and where CUDA
    departs from each. Printed per prompt when the reference has `bf16_tokens`."""
    bf16 = entry.get("bf16_tokens")
    if bf16 is None:
        return
    f32 = ref.tokens
    show = lambda d: "none" if d is None else f"step {d}"  # noqa: E731
    line = f"    yardstick {name}: HF-bf16 vs HF-f32 {show(first_diff(bf16, f32))}"
    if got_tokens is not None:
        line += (f"; CUDA vs HF-f32 {show(first_diff(got_tokens, f32))}"
                 f"; CUDA vs HF-bf16 {show(first_diff(got_tokens, bf16))}")
    elif got_text is not None:
        line += (f"; CUDA vs HF-f32 {show(text_diff(ref.tok, got_text, f32))}"
                 f"; CUDA vs HF-bf16 {show(text_diff(ref.tok, got_text, bf16))} (from the text)")
    else:
        line += "; CUDA's tokens not logged (run with VLLM_PRINT_TOKEN_IDS=1)"
    print(line)
    print(f"      HF-bf16 tokens {bf16}")


def judge(cases):
    """`cases`: (name, ref, divergence) per prompt. Applies the near-tie rule."""
    diverged = [(n, r, d) for n, r, d in cases if d is not None]
    for name, ref, d in cases:
        if d is None:
            report(name, True, f"matches HF's first {ref.k} tokens")
    if any(d is False for _, _, d in diverged):
        for name, _, d in diverged:
            if d is False:
                report(name, False, "differs from HF beyond the near-tie rule")
        return
    for name, ref, d in diverged:
        if d >= len(ref.steps):
            report(name, False, f"differs at step {d}, past the reference's {len(ref.steps)} steps")
            continue
        tie, margin, noise = ref.near_tie(d)
        ok = tie and len(diverged) <= 1
        report(name, ok, f"diverges at step {d}: f32 margin {margin:.4g}, bf16 max err {noise:.4g}, "
               f"{'a near-tie' if tie else 'NOT a near-tie'} ({len(diverged)} of {len(cases)} diverge)")


def ids_of_line(pattern, lines):
    """The ids of the last line `pattern` matches, or None."""
    ids = None
    for line in lines:
        m = pattern.search(line)
        if m:
            ids = [int(t) for t in m.group(1).split(",") if t.strip()]
    return ids


def check_input(name, want, got):
    """The ids scr fed the model against HF's: True on MATCH. A FAIL is reported on a MISMATCH
    or when the log has no prompt ids."""
    if got is None:
        report(f"{name} input ids", False, "no `[diag] prompt_token_ids=` line: run with VLLM_PRINT_TOKEN_IDS=1")
        return False
    if got == want:
        print(f"    {name} input ids: MATCH ({len(got)} ids)")
        report(f"{name} input ids", True, f"MATCH, {len(got)} ids")
        return True
    i = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    report(f"{name} input ids", False,
           f"MISMATCH at index {i}: scr {got[i:i + 8]}..., HF {want[i:i + 8]}... "
           f"(scr {len(got)} ids, HF {len(want)}); the reply is not compared")
    return False


def read_log(log):
    """scr chat's output as (reply, token ids, prompt ids, error lines). The reply is the lines
    after `Using model:`, tracing and diagnostic lines dropped (None without that line)."""
    with open(log, errors="replace") as f:
        lines = f.read().splitlines()
    tokens = ids_of_line(DIAG, lines)
    prompt_ids = ids_of_line(PROMPT_DIAG, lines)
    errors = [line for line in lines if ERROR_LINE.search(line)]
    start = next((i + 1 for i, line in enumerate(lines) if line.startswith("Using model:")), None)
    if start is None:
        return None, tokens, prompt_ids, errors
    body = [
        line
        for line in lines[start:]
        if not LOG_LINE.match(line) and not DIAG.search(line) and not PROMPT_DIAG.search(line)
    ]
    return "\n".join(body).strip(), tokens, prompt_ids, errors


def reply_of(log):
    """scr chat's reply and its token ids (see `read_log`)."""
    text, tokens, _, _ = read_log(log)
    return text, tokens


def mode_chat(args, chat, tok):
    cases = []
    for p, entry in enumerate(chat["prompts"], start=1):
        ref = Ref(tok, entry["steps"], chat["stop_ids"])
        text, tokens, prompt_ids, _ = read_log(os.path.join(args.logs, f"chat{p}.log"))
        if text is None:
            report(f"chat prompt {p}", False, "no `Using model:` line in the log")
            continue
        if not check_input(f"chat prompt {p}", entry["ids"], prompt_ids):
            continue
        cases.append((f"chat prompt {p} (--device cuda)", ref, compare(f"chat {p}", ref, text, tokens)))
        yardstick(f"chat {p}", entry, tokens, ref, text)
    judge(cases)
    ref = Ref(tok, chat["prompts"][0]["steps"], chat["stop_ids"])
    text, tokens, prompt_ids, _ = read_log(os.path.join(args.logs, "chatauto.log"))
    if text is None:
        report("chat prompt 1 (no --device)", False, "no `Using model:` line in the log")
    elif check_input("chat prompt 1 (no --device)", chat["prompts"][0]["ids"], prompt_ids):
        d = compare("chat 1, auto", ref, text, tokens)
        report("chat prompt 1 (no --device)", d is None, "matches" if d is None else f"diverges at {d}")


def mode_edges(args):
    """The edge prompts (see the module docs): a reply and no error, text not compared."""
    for name, log in (("scr chat, 4097-token prompt", "edge4097.log"),):
        path = os.path.join(args.logs, log)
        if not os.path.exists(path):
            report(name, False, f"no {path}")
            continue
        text, tokens, prompt_ids, errors = read_log(path)
        n = len(prompt_ids) if prompt_ids is not None else None
        detail = f"prompt ids {n}, reply {text!r}, tokens {tokens}"
        ok = text is not None and text != "" and not errors and n == 4097
        report(name, ok, detail if ok else f"{detail}, errors {errors[:3]}")


def chat_ids(tok, text):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True
    )
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def edge_prompts(tok, out_dir):
    """Write `edge4097.txt`: a `-q` text whose chat-templated prompt is exactly 4097 tokens (one
    word at a time, from below; exits if no text lands on 4097)."""
    words = 1
    while len(chat_ids(tok, " hello" * words)) < 4097:
        words = words * 2
    lo, hi = words // 2, words
    while lo < hi:
        mid = (lo + hi) // 2
        if len(chat_ids(tok, " hello" * mid)) < 4097:
            lo = mid + 1
        else:
            hi = mid
    text = " hello" * lo
    n = len(chat_ids(tok, text))
    if n != 4097:
        sys.exit(f"--edge-prompts: no text of repeated words renders to 4097 ids (closest {n})")
    with open(os.path.join(out_dir, "edge4097.txt"), "w") as f:
        f.write(text)
    print(f"edge prompt: {lo} words render to {n} chat-templated ids")


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}


def completion(url, ids, max_tokens, temperature=0):
    return post(url, "/v1/completions", {"prompt": ids, "max_tokens": max_tokens, "temperature": temperature})


def mode_serve(args, chat, tok):
    with open(os.path.join(args.goldens, "meta.json")) as f:
        meta = json.load(f)
    # 1. The precondition.
    same = True
    for p, entry in enumerate(chat["prompts"], start=1):
        status, body = post(args.url, "/tokenize", {
            "messages": [{"role": "user", "content": entry["text"]}],
            "add_generation_prompt": True,
        })
        got = body.get("tokens")
        ok = status == 200 and got == entry["ids"]
        same &= ok
        report(f"/tokenize prompt {p} is HF's chat-templated ids", ok,
               "" if ok else f"status {status}, scr {got}, HF {entry['ids']}")
    if not same:
        print("STOP: scr's chat-templated ids differ from HF's; the template differs, not the kernels")
        sys.exit(1)
    # 2. /v1/completions on the goldens' ids.
    texts, cases = {}, []
    for p in range(1, 6):
        entry = meta["prompts"][p - 1]
        with open(os.path.join(args.goldens, f"greedy_{p}.json")) as f:
            steps = json.load(f)["steps"]
        ref = Ref(tok, steps, chat["stop_ids"])
        status, body = completion(args.url, entry["ids"], len(steps))
        if status != 200:
            report(f"/v1/completions prompt {p}", False, f"status {status}: {body}")
            continue
        texts[p] = body["choices"][0]["text"]
        cases.append((f"/v1/completions prompt {p}", ref, compare(f"completions {p}", ref, texts[p])))
    judge(cases)
    # 3. /v1/chat/completions on prompts 1-5: their ids are the ones `/tokenize` checked.
    cases = []
    for p, entry in enumerate(chat["prompts"], start=1):
        ref = Ref(tok, entry["steps"], chat["stop_ids"])
        status, body = post(args.url, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": entry["text"]}],
            "max_tokens": len(entry["steps"]),
            "temperature": 0,
        })
        if status != 200:
            report(f"/v1/chat/completions prompt {p}", False, f"status {status}: {body}")
            continue
        msg = body["choices"][0]["message"]
        content = msg.get("content") or ""
        print(f"    reasoning field: {msg.get('reasoning_content') or msg.get('reasoning')!r}")
        d = compare(f"chat/completions {p}", ref, content)
        yardstick(f"chat/completions {p}", entry, None, ref, content)
        cases.append((f"/v1/chat/completions prompt {p}", ref, d))
    judge(cases)
    # 4. Two at once.
    got = {}

    def run(p):
        got[p] = completion(args.url, meta["prompts"][p - 1]["ids"], 8)

    threads = [threading.Thread(target=run, args=(p,)) for p in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for p in (1, 2):
        status, body = got[p]
        text = body.get("choices", [{}])[0].get("text") if status == 200 else None
        report(f"concurrent /v1/completions prompt {p}", status == 200 and text == texts.get(p),
               f"status {status}, text {text!r}, alone {texts.get(p)!r}")
    # 5. Too long.
    ids = (meta["prompts"][0]["ids"] * 8193)[:8193]
    status, body = completion(args.url, ids, 1)
    report("8193 ids are refused naming 8192", 400 <= status < 500 and "8192" in json.dumps(body),
           f"status {status}: {json.dumps(body)[:300]}")
    # 6. Not greedy.
    status, body = completion(args.url, meta["prompts"][0]["ids"], 4, temperature=0.7)
    report("temperature 0.7 is refused naming greedy-only", status != 200 and "greedily only" in json.dumps(body),
           f"status {status}: {json.dumps(body)[:300]}")
    # 7. Chunked prefill.
    ids = (meta["prompts"][0]["ids"] * 5000)[:5000]
    status, body = completion(args.url, ids, 4)
    report("5000 ids, 4 tokens, chunked", status == 200,
           f"status {status}: {json.dumps(body)[:200]} (the step sizes are in the server log)")
    # 8. The edges: a 1-token prompt (a fresh first step on the one-row decode tape) and 4097
    #    tokens (a 4096 chunk and a remainder of one), each a 200 with a non-empty reply.
    for n in (1, 4097):
        ids = (meta["prompts"][0]["ids"] * 4097)[:n]
        status, body = completion(args.url, ids, 8)
        text = body.get("choices", [{}])[0].get("text") if status == 200 else None
        report(f"{n}-token prompt, 8 tokens", status == 200 and bool(text),
               f"status {status}, text {text!r}" if status == 200 else f"status {status}: {json.dumps(body)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="cuda_hf_chat_reference.py's JSON")
    ap.add_argument("--mode", choices=("chat", "serve", "edges"), required=True)
    ap.add_argument("--edge-prompts", metavar="DIR",
                    help="write the edge prompts' texts to DIR and exit (needs only --ref or --checkpoint)")
    ap.add_argument("--logs", default="/logs")
    ap.add_argument("--goldens", help="cuda_hf_reference.py's --out (serve mode)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--checkpoint", help="the tokenizer's checkpoint (default: the reference's)")
    args = ap.parse_args()
    with open(args.ref) as f:
        chat = json.load(f)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint or chat["checkpoint"])
    print(f"stop ids: scr {chat['stop_ids']['scr']}, HF {chat['stop_ids']['hf']}; "
          f"generation_config.json present: {chat['generation_config_json']}")
    if args.edge_prompts:
        edge_prompts(tok, args.edge_prompts)
        return
    if args.mode == "chat":
        mode_chat(args, chat, tok)
    elif args.mode == "edges":
        mode_edges(args)
    else:
        if not args.goldens:
            sys.exit("--mode serve needs --goldens")
        mode_serve(args, chat, tok)
    failed = results.count(False)
    print(f"{len(results) - failed} PASS, {failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
