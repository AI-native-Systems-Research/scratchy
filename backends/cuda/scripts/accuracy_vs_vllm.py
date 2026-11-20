#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Accuracy evidence for a scratchy model against vLLM on the same GPU: a greedy agreement set and
teacher-forced scoring of a fixed text. Run one engine's server at a time, then compare.

    accuracy_vs_vllm.py collect --engine scr  --url http://127.0.0.1:8000 --out scr.json  [options]
    accuracy_vs_vllm.py collect --engine vllm --url http://127.0.0.1:8000 --out vllm.json --answers scr.json [options]
    accuracy_vs_vllm.py compare scr.json vllm.json > report.md

Options: --prompts FILE (testdata/accuracy_prompts.txt, beside this script), --text FILE
(testdata/pride_and_prejudice_ch1-3.txt), --max-tokens 64, --text-tokens 4000,
--positions 512, --raw-text [--bos-id ID].

1. **The agreement set.** Each prompt is one user message to /v1/chat/completions, temperature 0,
   `max_tokens` tokens, `chat_template_kwargs {"enable_thinking": false}`. vLLM also returns
   `logprobs` with its top 2 per token. Both servers' /tokenize of the rendered messages is
   recorded, so the report shows both rendered the same prompt. Answers are compared as UTF-8
   bytes: scratchy returns no token ids. vLLM's per-token bytes say which of its tokens holds the
   first differing byte. A prompt matches its first N tokens when scratchy's answer starts with
   the bytes of vLLM's first N tokens (or, when vLLM stopped sooner, equals vLLM's answer). At the
   first difference the report gives vLLM's top-2 gap; the D2 rule calls a gap under 1 nat a
   near-tie.
2. **Teacher-forced scoring of a fixed text** (Pride and Prejudice ch. 1-3, Project Gutenberg #1342,
   public domain; the first `--text-tokens` ids under the server's tokenizer), by default as the
   model's own turn after a fixed chat request (`TEXT_REQUEST`, the chat template rendered by the
   server); `--raw-text` scores it bare (`--bos-id` prepended), which an instruct model may not
   take as in-distribution input (Gemma 4: perplexity 19,000 on this text):
   a. the next-token agreement: at `--positions` evenly spaced positions, each engine's greedy
      next token after the text's prefix (/v1/completions over the prefix ids, `max_tokens` 1),
      compared as text; vLLM's top-2 gap where they differ;
   b. vLLM's perplexity of the text (`prompt_logprobs`, one request);
   c. each engine's greedy answers scored teacher-forced under vLLM: the mean log-probability per
      token vLLM gives scratchy's answer and its own (`--answers`, re-tokenized by vLLM's
      tokenizer, the same for both). A correct stack that broke near-ties differently writes text
      vLLM finds about as likely as its own; a wrong computation, text it finds far less likely
      (`scripts/parity_ties.py`'s rule against mlx-lm);
   d. scratchy's own log-probabilities, when its server returns `prompt_logprobs`: the mean
      |delta log-prob| against vLLM's at every text token, and scratchy's perplexity. A server that
      refuses them (the CUDA worker at #978's head: `crates/targets/cuda/worker/src/plan.rs`,
      `PlanError::PromptLogprobs`) is recorded as such, and (d) is reported as not available.
Standard library only.
"""

import argparse
import json
import math
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NEAR_TIE_NAT = 1.0
KWARGS = {"enable_thinking": False}
TEXT_REQUEST = "Recite the opening chapters of Pride and Prejudice by Jane Austen, word for word."


def call(url, path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data, {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout)), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"


def must(r):
    out, err = r
    if err:
        sys.exit(err)
    return out


def tokenize(url, model, **kw):
    return must(call(url, "/tokenize", {"model": model, **kw}))["tokens"]


def positions(n, count, start=16):
    if n - 1 <= start:
        return []
    step = max(1, (n - 1 - start) // count)
    return list(range(start, n - 1, step))[:count]


def collect(a):
    url = a.url.rstrip("/")
    model = must(call(url, "/v1/models"))["data"][0]["id"]
    prompts = [l.rstrip("\n") for l in open(a.prompts) if l.strip() and not l.startswith("#")]
    out = {"engine": a.engine, "model": model, "max_tokens": a.max_tokens, "agree": [], "text": {}}
    # 1. the agreement set
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        rec = {"idx": i, "prompt": p}
        rec["prompt_ids"] = tokenize(url, model, messages=msgs, add_generation_prompt=True,
                                     chat_template_kwargs=KWARGS)
        body = {"model": model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0,
                "chat_template_kwargs": KWARGS}
        if a.engine == "vllm":
            body.update({"logprobs": True, "top_logprobs": 2})
        r, err = call(url, "/v1/chat/completions", body)
        if err:
            rec["error"] = err
        else:
            c = r["choices"][0]
            m = c["message"]
            rec["text"] = (m.get("reasoning_content") or m.get("reasoning") or "") + (m.get("content") or "")
            rec["finish_reason"] = c.get("finish_reason")
            if a.engine == "vllm":
                rec["tokens"] = [{"bytes": e.get("bytes") or list(e["token"].encode()),
                                  "top2": [x["logprob"] for x in e.get("top_logprobs") or []]}
                                 for e in ((c.get("logprobs") or {}).get("content") or [])]
        out["agree"].append(rec)
        print(f"{a.engine} prompt {i}: {'ERROR ' + rec['error'] if 'error' in rec else repr(rec['text'][:60])}",
              file=sys.stderr)
    # 2. the text: by default the model's own turn after a fixed chat request, so an instruct model
    # scores it in distribution (Gemma 4 on raw text after <bos> is not); --raw-text
    # scores it bare, with --bos-id prepended
    text_ids = tokenize(url, model, prompt=open(a.text).read(), add_special_tokens=False)
    if a.raw_text:
        head = [a.bos_id] if a.bos_id is not None else []
    else:
        head = tokenize(url, model, messages=[{"role": "user", "content": TEXT_REQUEST}],
                        add_generation_prompt=True, chat_template_kwargs=KWARGS)
    ids = (head + text_ids)[:len(head) + a.text_tokens]
    t = out["text"]
    t["ids"], t["start"], t["mode"] = ids, len(head), "raw" if a.raw_text else "chat"
    t["positions"] = [len(head) + q for q in positions(len(ids) - len(head), a.positions)]
    t["next"] = []
    for pos in t["positions"]:
        body = {"model": model, "prompt": ids[:pos], "max_tokens": 1, "temperature": 0}
        if a.engine == "vllm":
            body["logprobs"] = 2
        r, err = call(url, "/v1/completions", body)
        if err:
            t["next"].append({"pos": pos, "error": err})
            continue
        c = r["choices"][0]
        rec = {"pos": pos, "text": c["text"]}
        lp = c.get("logprobs")
        if lp and lp.get("top_logprobs"):
            rec["top"] = lp["top_logprobs"][0]  # {token text: logprob}, vLLM's top 2
        t["next"].append(rec)
    # (b) and (d): each engine's log-probability of every text id after its prefix, one request; a
    # server that refuses prompt_logprobs (scratchy's CUDA worker today) is recorded, not fatal
    t["prompt_logprobs"], t["prompt_logprobs_refused"] = prompt_logprobs(url, model, ids)
    if t["prompt_logprobs_refused"] and a.engine == "vllm":
        sys.exit(f"vLLM refused prompt_logprobs: {t['prompt_logprobs_refused']}")
    if a.engine == "vllm":
        # (c) both engines' answers scored under vLLM
        if a.answers:
            scr = json.load(open(a.answers))
            t["answer_scores"] = {"scr": score(url, model, scr["agree"]), "vllm": score(url, model, out["agree"])}
    json.dump(out, open(a.out, "w"))
    print(f"wrote {a.out}", file=sys.stderr)


def prompt_logprobs(url, model, ids):
    """(the server's log-probability of each id after its prefix, index 0 None; or None and why not)."""
    r, err = call(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1,
                                           "temperature": 0, "prompt_logprobs": 1})
    if err:
        return None, err
    plp = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
    if not plp:
        return None, "the response has no prompt_logprobs"
    return [None if e is None else e[str(tok)]["logprob"] for e, tok in zip(plp, ids)], None


def score(url, model, recs):
    """Mean log-probability per answer token vLLM gives each answer, teacher-forced after its
    rendered prompt; the answer re-tokenized by vLLM's tokenizer."""
    out = []
    for r in recs:
        if "error" in r or not r.get("text"):
            out.append(None)
            continue
        ans = tokenize(url, model, prompt=r["text"], add_special_tokens=False)
        lps, err = prompt_logprobs(url, model, r["prompt_ids"] + ans)
        if err:
            sys.exit(f"scoring prompt {r['idx']}: {err}")
        lps = lps[len(r["prompt_ids"]):]
        out.append(sum(lps) / len(lps) if lps else None)
    return out


def first_diff(sb, vb):
    for i, (x, y) in enumerate(zip(sb, vb)):
        if x != y:
            return i
    return None if len(sb) == len(vb) else min(len(sb), len(vb))


def agree_rows(s, v):
    rows = []
    for rs, rv in zip(s["agree"], v["agree"]):
        if "error" in rs or "error" in rv:
            rows.append({"idx": rs["idx"], "error": rs.get("error") or rv.get("error")})
            continue
        sb, vb = rs["text"].encode(), rv["text"].encode()
        toks = rv.get("tokens") or []
        # vLLM's logprobs include the stop token (`<|im_end|>`, `<turn|>`) its text leaves out: keep the
        # tokens that make up the text it returned
        ends, acc, kept = [], 0, []
        for t in toks:
            if acc >= len(vb) and acc > 0:
                break
            acc += len(t["bytes"]); ends.append(acc); kept.append(t)
        toks = kept
        match = {}
        for n in (8, 32, 64):
            k = min(n, len(toks))
            pref = vb[:ends[k - 1]] if k else b""
            match[n] = sb.startswith(pref) and (len(toks) >= n or sb == vb)
        d = first_diff(sb, vb)
        tok = None if d is None else next((j for j, e in enumerate(ends) if e > d), len(toks) - 1)
        gap = None
        if tok is not None and 0 <= tok < len(toks) and len(toks[tok]["top2"]) == 2:
            gap = toks[tok]["top2"][0] - toks[tok]["top2"][1]
        # scratchy's reply is vLLM's minus exactly its last token's text, both cut at max_tokens: the
        # last token's text did not come back, not a different token we can see
        last = len(toks) - 1
        missing_last = (sb != vb and toks and tok == last and rs.get("finish_reason") == "length"
                        and rv.get("finish_reason") == "length" and vb[:ends[last - 1] if last else 0] == sb)
        rows.append({"idx": rs["idx"], "same_prompt": rs["prompt_ids"] == rv["prompt_ids"], "match": match,
                     "first_diff_token": tok, "gap": gap, "missing_last": missing_last,
                     "scr": rs["text"], "vllm": rv["text"]})
    return rows


def compare(a):
    s, v = json.load(open(a.scr)), json.load(open(a.vllm))
    p = print
    p(f"# Accuracy against vLLM: {v['model']}\n")
    p(f"scratchy `{s['model']}` and vLLM `{v['model']}`, greedy, {s['max_tokens']} tokens; "
      f"the D2 rule: a first difference where vLLM's top-2 gap is under {NEAR_TIE_NAT} nat is a near-tie.\n")
    rows = agree_rows(s, v)
    ok = [r for r in rows if "error" not in r]
    p(f"## 1. Greedy agreement, {len(rows)} chat prompts\n")
    p(f"- rendered prompt ids identical on both servers: {sum(r['same_prompt'] for r in ok)}/{len(ok)}")
    for n in (8, 32, 64):
        p(f"- first {n} tokens match: **{sum(r['match'][n] for r in ok)}/{len(ok)}**")
    lasts = [r for r in ok if r["missing_last"]]
    diffs = [r for r in ok if r["first_diff_token"] is not None and not r["match"][64] and not r["missing_last"]]
    ties = [r for r in diffs if r["gap"] is not None and r["gap"] < NEAR_TIE_NAT]
    p(f"- prompts that differ within {s['max_tokens']} tokens: {len(diffs)}; at a near-tie: **{len(ties)}**; "
      f"not at a near-tie: **{len(diffs) - len(ties)}**")
    if lasts:
        p(f"- prompts whose scratchy reply is vLLM's minus only the last token's text "
          f"(both cut at {s['max_tokens']} tokens): {len(lasts)} ({[r['idx'] for r in lasts]}); not counted as a difference")
    if len(rows) != len(ok):
        p(f"- errors: {len(rows) - len(ok)} ({[r['idx'] for r in rows if 'error' in r]})")
    p("\n| prompt | same prompt | 8 | 32 | 64 | first differing vLLM token | vLLM top-2 gap (nat) | near-tie |")
    p("|---|---|---|---|---|---|---|---|")
    for r in ok:
        g = "-" if r["gap"] is None else f"{r['gap']:.3f}"
        tie = "-" if r["match"][64] else ("last token's text missing" if r["missing_last"] else
                                           ("yes" if r["gap"] is not None and r["gap"] < NEAR_TIE_NAT else "**no**"))
        m = lambda n: "yes" if r["match"][n] else "no"
        p(f"| {r['idx']} | {'yes' if r['same_prompt'] else '**no**'} | {m(8)} | {m(32)} | {m(64)} | "
          f"{'-' if r['match'][64] else r['first_diff_token']} | {g} | {tie} |")
    for r in diffs:
        if r["gap"] is None or r["gap"] >= NEAR_TIE_NAT:
            p(f"\n- prompt {r['idx']} (not a near-tie): scratchy `{r['scr'][:160]!r}`; vLLM `{r['vllm'][:160]!r}`")
    ts, tv = s["text"], v["text"]
    p(f"\n## 2. Teacher-forced scoring of a fixed text ({len(tv['ids']) - tv.get('start', 0)} tokens, "
      f"{'after the chat request' if tv.get('mode') == 'chat' else 'raw'})\n")
    same_ids = ts["ids"] == tv["ids"]
    p(f"- text ids identical under both servers' tokenizers: {'yes' if same_ids else '**no**'}")
    nv = {r["pos"]: r for r in tv["next"] if "error" not in r}
    agree = tie = hard = skip = 0
    for r in ts["next"]:
        q = nv.get(r["pos"])
        if "error" in r or q is None:
            skip += 1; continue
        if not r["text"] or not q["text"] or "�" in r["text"] + q["text"]:
            skip += 1; continue  # a byte piece of a character: no text to compare
        if r["text"] == q["text"]:
            agree += 1; continue
        top = sorted((q.get("top") or {}).values(), reverse=True)
        if len(top) >= 2 and top[0] - top[1] < NEAR_TIE_NAT:
            tie += 1
        else:
            hard += 1
    n = agree + tie + hard
    p(f"- (a) next-token agreement at {n} positions: **{agree}/{n}** ({100 * agree / max(n, 1):.1f}%); "
      f"differences at a near-tie {tie}, not at a near-tie **{hard}**; skipped {skip} (errors or byte pieces)")
    st = max(tv.get("start", 1), 1)
    lps = [x for x in (tv.get("prompt_logprobs") or [])[st:] if x is not None]
    if lps:
        p(f"- (b) vLLM's perplexity of the text ({tv.get('mode', 'raw')}): **{math.exp(-sum(lps) / len(lps)):.3f}** "
          f"over {len(lps)} tokens")
    sc = tv.get("answer_scores")
    if sc:
        pairs = [(x, y) for x, y in zip(sc["scr"], sc["vllm"]) if x is not None and y is not None]
        if pairs:
            ms, mv = sum(x for x, _ in pairs) / len(pairs), sum(y for _, y in pairs) / len(pairs)
            worst = min(pairs, key=lambda xy: xy[0] - xy[1])
            p(f"- (c) mean log-prob per answer token under vLLM: scratchy's answers **{ms:.4f}**, vLLM's own "
              f"**{mv:.4f}** (difference {ms - mv:+.4f} nat; worst prompt {worst[0] - worst[1]:+.4f}) over "
              f"{len(pairs)} prompts")
    sl, vl = ts.get("prompt_logprobs"), tv.get("prompt_logprobs")
    if sl and vl and same_ids:
        d = [abs(x - y) for x, y in zip(sl[st:], vl[st:]) if x is not None and y is not None]
        slp = [x for x in sl[st:] if x is not None]
        p(f"- (d) scratchy's own log-probability of every text token: mean |delta log-prob| against vLLM's "
          f"**{sum(d) / len(d):.4f}** nat (max {max(d):.4f}) over {len(d)} tokens; scratchy's perplexity "
          f"**{math.exp(-sum(slp) / len(slp)):.3f}** against vLLM's above")
    else:
        why = ts.get("prompt_logprobs_refused") or ("the text ids differ" if not same_ids else "no data")
        p(f"- (d) scratchy's own log-probabilities: **not available**: {why[:200]!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--engine", choices=["scr", "vllm"], required=True)
    c.add_argument("--url", default="http://127.0.0.1:8000")
    c.add_argument("--out", required=True)
    c.add_argument("--answers", help="vllm only: scratchy's collect output, whose answers vLLM scores")
    c.add_argument("--prompts", default=os.path.join(HERE, "testdata", "accuracy_prompts.txt"))
    c.add_argument("--text", default=os.path.join(HERE, "testdata", "pride_and_prejudice_ch1-3.txt"))
    c.add_argument("--max-tokens", type=int, default=64)
    c.add_argument("--text-tokens", type=int, default=4000)
    c.add_argument("--positions", type=int, default=512)
    c.add_argument("--raw-text", action="store_true", help="score the text bare, not as the model's turn")
    c.add_argument("--bos-id", type=int, help="with --raw-text: an id prepended to the text (2 for Gemma)")
    m = sub.add_parser("compare")
    m.add_argument("scr")
    m.add_argument("vllm")
    a = ap.parse_args()
    (collect if a.cmd == "collect" else compare)(a)


if __name__ == "__main__":
    main()
