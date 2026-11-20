"""Gemma's prompt ids as each server encodes them (whether each prepends Gemma's <bos>).

usage: python3 -I pids.py ENGINE SIDE OUT
For prompts 3 and 4 of /work/prompts.txt: POST /tokenize {"prompt": p} (the default
add_special_tokens), then the plain coherence step's /v1/completions (32 tokens, temperature 0) and
its usage.prompt_tokens. Report-only: every error is recorded, and the exit is always 0.
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


engine, side, out = sys.argv[1:4]
prompts = [l.rstrip("\n") for l in open("/work/prompts.txt") if l.strip()][:5]
res = []
try:
    model = call("/v1/models")["data"][0]["id"]
except Exception as e:
    model = None
    res.append({"models_error": repr(e)})
for i in (3, 4):
    p = prompts[i]
    rec = {"idx": i, "prompt": p}
    try:
        rec["ids"] = call("/tokenize", {"prompt": p})["tokens"]
    except Exception as e:
        rec["ids_error"] = repr(e)
    try:
        c = call("/v1/completions", {"model": model, "prompt": p, "max_tokens": 32, "temperature": 0})
        rec["usage_prompt_tokens"] = (c.get("usage") or {}).get("prompt_tokens")
        rec["text"] = c["choices"][0]["text"]
    except Exception as e:
        rec["completion_error"] = repr(e)
    res.append(rec)
    ids = rec.get("ids") or []
    print(f"PROMPT_IDS {engine}-{side} prompt {i}: {len(ids)} ids, first {ids[:4]}, "
          f"bos(2) first={bool(ids) and ids[0] == 2}, usage.prompt_tokens={rec.get('usage_prompt_tokens')}")
json.dump({"engine": engine, "side": side, "model": model, "records": res}, open(out, "w"), indent=1)

