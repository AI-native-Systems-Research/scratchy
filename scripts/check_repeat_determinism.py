"""Greedy output reproducibility on a running server: one prompt cold, then repeated (prefix-cache hits), with a
different prompt in between; prints each reply's length and whether the repeats' texts are identical. Exits 1 if
they are not. For the DFlash2 draft PR's experiments (and the #1012 check: a verify step's output must not vary).

    python3 scripts/check_repeat_determinism.py <port> <model> [repeats]
"""
import json
import sys
import urllib.request

port, model = sys.argv[1], sys.argv[2]
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
base = f"http://127.0.0.1:{port}"
readme = open("README.md").read()[:6000]
build = open("docs/BUILD.md").read()[:6000]


def ask(document):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 200,
                       "messages": [{"role": "user", "content": document + "\n\nSummarize the document above in one paragraph."}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"]["content"], r["usage"]


texts = []
for name, document in [("cold", readme)] + [(f"hit {i + 1}", readme) for i in range(repeats)] + \
        [("other prompt", build), ("hit after other", readme)]:
    text, usage = ask(document)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"{name:16} cached {cached:5} tokens {usage['completion_tokens']:4}")
    if name.startswith("hit"):
        texts.append(text)
same = all(t == texts[0] for t in texts)
print("repeats identical:", same)
sys.exit(0 if same else 1)
