"""Coherence requests, one server at a time (pod-side).

usage: python3 -I coh.py MODE CK OUT        MODE = scr | vllm | scr-chat | vllm-chat

The first 5 prompts of /work/prompts.txt (crates/e2e/testdata/prompts.txt), each a greedy
/v1/completions of 32 tokens to the model /v1/models serves.
- vllm: asks for `logprobs: 2` with token ids, so each record carries vLLM's
  token ids and its top-2 logprobs per position, which cmp.py uses to excuse a divergence.
- scr: the CUDA worker refuses logprobs (plan.rs `PlanError::Logprobs`), so the same request goes
  without them, and the reply's token ids come from POST /tokenize of the returned text with
  `add_special_tokens: false`.
- scr-chat / vllm-chat: the same 5 prompts as one user message each to /v1/chat/completions,
  `max_tokens 32`, `temperature 0`, `chat_template_kwargs {"enable_thinking": false}` on both (so
  Gemma 4's thinking channel does not take the 32 tokens). vllm-chat asks `logprobs: true,
  top_logprobs: 2`; its token ids are the logprobs entries' tokens, through POST /tokenize when
  they are text. scr-chat sends no logprobs; its ids are /tokenize of the reply text with
  `add_special_tokens: false`. Both also record the rendered prompt's ids (/tokenize of the
  messages, `add_generation_prompt: true`), so cmp.py can show both servers rendered the same
  prompt before it blames the forward.
Each record: {idx, prompt, text, token_ids, top2 (vllm only: [{id: logprob, ...}] per position)}.
A failed request is recorded with its error, and the script exits 1 after writing every record.
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def token_id(tok):
    # vLLM's `return_tokens_as_token_ids`: "token_id:198".
    return int(str(tok).split(":", 1)[1])


def main():
    mode, ck, out = sys.argv[1:4]
    assert mode in ("scr", "vllm", "scr-chat", "vllm-chat"), mode
    if mode.endswith("-chat"):
        return chat(mode, ck, out)
    model = call("/v1/models")["data"][0]["id"]
    prompts = [l.rstrip("\n") for l in open("/work/prompts.txt") if l.strip()][:5]
    res, failed = [], 0
    for i, p in enumerate(prompts):
        body = {"model": model, "prompt": p, "max_tokens": 32, "temperature": 0}
        if mode == "vllm":
            body.update({"logprobs": 2, "return_tokens_as_token_ids": True})
        rec = {"idx": i, "prompt": p, "model": model}
        try:
            c = call("/v1/completions", body)["choices"][0]
            rec["text"] = c["text"]
            rec["finish_reason"] = c.get("finish_reason")
            if mode == "vllm":
                lp = c["logprobs"]
                rec["token_ids"] = [token_id(t) for t in lp["tokens"]]
                rec["top2"] = [
                    {str(token_id(k)): v for k, v in (pos or {}).items()} for pos in lp["top_logprobs"]
                ]
            else:
                t = call("/tokenize", {"prompt": c["text"], "add_special_tokens": False})
                rec["token_ids"] = t["tokens"]
        except Exception as e:  # recorded, so one bad prompt does not hide the others
            rec["error"] = repr(e)
            failed += 1
        res.append(rec)
        print(f"{mode} prompt {i}: {'ERROR ' + rec['error'] if 'error' in rec else repr(rec['text'][:80])}")
    json.dump({"mode": mode, "ck": ck, "model": model, "records": res}, open(out, "w"), indent=1)
    sys.exit(1 if failed else 0)


def ids_of(text):
    return call("/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]


def chat(mode, ck, out):
    model = call("/v1/models")["data"][0]["id"]
    prompts = [l.rstrip("\n") for l in open("/work/prompts.txt") if l.strip()][:5]
    kwargs = {"enable_thinking": False}
    res, failed = [], 0
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        body = {"model": model, "messages": msgs, "max_tokens": 32, "temperature": 0,
                "chat_template_kwargs": kwargs}
        if mode == "vllm-chat":
            body.update({"logprobs": True, "top_logprobs": 2, "return_tokens_as_token_ids": True})
        rec = {"idx": i, "prompt": p, "model": model}
        try:
            rec["prompt_ids"] = call("/tokenize", {"model": model, "messages": msgs,
                                                    "add_generation_prompt": True,
                                                    "chat_template_kwargs": kwargs})["tokens"]
        except Exception as e:
            rec["prompt_ids_error"] = repr(e)
        try:
            c = call("/v1/chat/completions", body)["choices"][0]
            m = c["message"]
            rec["text"] = m.get("content") or ""
            rec["reasoning"] = m.get("reasoning_content") or m.get("reasoning")
            rec["finish_reason"] = c.get("finish_reason")
            rec["token_ids"] = ids_of(rec["text"]) if rec["text"] else []
            if mode == "vllm-chat":
                content = (c.get("logprobs") or {}).get("content") or []
                # Each position's token as vLLM generated it ("token_id:N", from
                # `return_tokens_as_token_ids`), and its top 2 by id: exact, no re-tokenizing.
                # If a vLLM build ignores the flag, the strings go through /tokenize (a single
                # generated token is one id), and the reply text's ids are kept when one is not.
                def tid(tok):
                    if str(tok).startswith("token_id:"):
                        return [token_id(tok)]
                    return ids_of(tok)
                rec["gen_tokens"] = [e["token"] for e in content]
                rec["top2"] = [
                    {str(t[0]) if len(t) == 1 else "text:" + x["token"]: x["logprob"]
                     for x in (e.get("top_logprobs") or []) for t in [tid(x["token"])]}
                    for e in content
                ]
                gen = [tid(t) for t in rec["gen_tokens"]]
                if gen and all(len(g) == 1 for g in gen):
                    rec["token_ids"] = [g[0] for g in gen]
                else:
                    rec["gen_ids_note"] = "a generated token is not one id; token_ids are the reply text's"
        except Exception as e:
            rec["error"] = repr(e)
            failed += 1
        res.append(rec)
        print(f"{mode} prompt {i}: {'ERROR ' + rec['error'] if 'error' in rec else repr(rec['text'][:80])}")
    json.dump({"mode": mode, "ck": ck, "model": model, "records": res}, open(out, "w"), indent=1)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
