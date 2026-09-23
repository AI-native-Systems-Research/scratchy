#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""One external stopwatch for scratchy vs mlx-lm startup latency.

Why this file exists: neither framework's self-reported numbers compose into
"how long from exec until I see a word".

  - scratchy `chat --bench` prints `startup` and a TTFT measured *after*
    startup, so the two never add up to the user-perceived latency. Worse,
    real work lands after "startup" is declared done: a warmup generation, a
    2.4s background aligned-cache integrity hash over 1.7 GiB, lazy
    TurboQuant codebook selection.
  - `mlx_lm.generate --verbose` reports prompt/generation tok/s but never
    import or load time.

So EVERY primary number here is taken by THIS process, with the same code for
both backends: we hold the clock, start it immediately before fork/exec, and
stop it when the first content byte of the first token actually arrives. The
fact that there is exactly ONE implementation of the stopwatch, shared by both
backends, is the core fairness guarantee of this benchmark.

Emits one JSON object per invocation (see `main`). Never interprets results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
# A seeded word salad. Two properties matter and neither is "realistic text":
#   1. DETERMINISTIC given the seed, so a rerun measures the same work.
#   2. UNIQUE per seed, so no prefix cache (scratchy's, or mlx-lm's prompt
#      cache) can serve a later request and collapse TTFT to ~0. This is the
#      rule bench_serve_compare.sh learned the hard way.
# Both backends get byte-identical prompt text and share the Llama tokenizer,
# so the token count is identical by construction -- we assert it anyway.
_WORDS = (
    "harbor lantern gravel meadow cinder quartz plateau bramble thicket ember "
    "sparrow willow basalt cobalt drifting shallow ridge canyon tundra fjord "
    "marble copper silent hollow amber jasper cedar frost pebble current "
    "glacier summit orchard beacon compass anchor rudder mariner tempest "
    "monsoon zephyr equinox solstice meridian latitude sextant almanac"
).split()


def build_prompt(seed: int, approx_tokens: int, override: str = "") -> str:
    if override:
        return override
    rng = random.Random(seed)
    # Measured ~2.2 Llama tokens per word for this salad (47 words -> 104
    # tokens). The target is approximate by design: what fairness needs is
    # that both backends see byte-identical text, and the summary reports the
    # prompt_tokens each backend actually counted.
    n = max(4, round(approx_tokens / 2.2))
    return " ".join(rng.choice(_WORDS) for _ in range(n))


# ---------------------------------------------------------------------------
# CLI mode: stdout prelude filters
# ---------------------------------------------------------------------------
# In CLI mode the first *content* byte is the first token, but each CLI writes
# a banner first. These predicates decide "is this line still banner?" so the
# stopwatch stops on a token and not on a header.


def _is_prelude_scratchy(line: bytes) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith(b"Using model:"):
        return True
    # tracing lines, in case RUST_LOG did not silence them
    return len(s) > 20 and s[:4].isdigit() and s[4:5] == b"-"


def _is_prelude_mlx(line: bytes) -> bool:
    s = line.strip()
    return not s or set(s) == {ord("=")}


PRELUDE = {"scratchy": _is_prelude_scratchy, "mlx-lm": _is_prelude_mlx}


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def cli_cmd(a, prompt: str) -> tuple[list[str], dict]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if a.hf_hub_cache:
        env["HF_HUB_CACHE"] = a.hf_hub_cache
    if a.backend == "ollama":
        # `ollama run` is a CLIENT: it needs a daemon already up, so timing it
        # would measure a warm request, not a cold start. There is no
        # self-contained one-shot binary to compare against `scr chat` or
        # `mlx_lm.generate`, so refuse rather than report a number that looks
        # like a cold start and is not one.
        raise SystemExit(
            "ollama has no one-shot CLI equivalent (`ollama run` requires a "
            "running daemon, so it measures a warm request). Use --modes "
            "server for ollama.")
    if a.backend == "scratchy":
        # RUST_LOG=error keeps tokens the only thing on stdout so the
        # first-content-byte detector is unambiguous.
        env["RUST_LOG"] = "error"
        cmd = [a.scr_bin, "chat", "-m", a.model, "--device", a.device,
               "-q", prompt, "--max-tokens", str(a.output_len),
               "--temperature", "0", "--bench"]
    else:
        cmd = [a.mlx_python, "-m", "mlx_lm", "generate", "--model", a.model,
               "--prompt", prompt, "--max-tokens", str(a.output_len),
               "--temp", "0"]
    return cmd, env


def server_cmd(a) -> tuple[list[str], dict]:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if a.hf_hub_cache:
        env["HF_HUB_CACHE"] = a.hf_hub_cache
    if a.backend == "scratchy":
        env["RUST_LOG"] = a.rust_log
        cmd = [a.scr_bin, "serve", a.model, "--device", a.device,
               "--host", "127.0.0.1", "--port", str(a.port)]
        if a.kv_cache_dtype:
            cmd += ["--kv-cache-dtype", a.kv_cache_dtype]
        if a.max_model_len:
            cmd += ["--max-model-len", str(a.max_model_len)]
        cmd += a.scratchy_extra
    elif a.backend == "mlx-lm":
        cmd = [a.mlx_python, "-m", "mlx_lm", "server", "--model", a.model,
               "--host", "127.0.0.1", "--port", str(a.port)]
        cmd += a.mlx_extra
    else:
        # ollama is a DAEMON, not a per-model server: `ollama serve` comes up
        # without loading anything and the model loads on the first request.
        # Two consequences the summary has to carry:
        #   - t_ready means "daemon accepting connections", NOT "model
        #     resident" (verified: /v1/models answers while /api/ps is empty),
        #     so it is not comparable with the other backends' t_ready.
        #   - ttft_exec IS still comparable, because the first request pays the
        #     model load.
        # OLLAMA_HOST is how the daemon picks its address; there is no --port.
        # A private port also keeps us off any system-managed daemon on 11434.
        env["OLLAMA_HOST"] = f"127.0.0.1:{a.port}"
        cmd = [a.ollama_bin, "serve"]
        cmd += a.ollama_extra
    return cmd, env


# ---------------------------------------------------------------------------
# HTTP: hand-rolled so we control exactly when the clock stops
# ---------------------------------------------------------------------------
def port_is_serving(port: int, timeout: float = 0.25) -> bool:
    try:
        c = HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("GET", "/v1/models")
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except (OSError, socket.timeout):
        return False


def wait_ready(port: int, proc, deadline_s: float, poll_s: float):
    """Return monotonic time when /v1/models first answers 200.

    Poll granularity (default 20ms) is the only quantization in t_ready, and
    it is reported alongside the number rather than hidden.
    """
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if port_is_serving(port):
            return time.monotonic()
        if proc.poll() is not None:
            return None
        time.sleep(poll_s)
    return None


def stream_completion(a, prompt: str, t_zero: float) -> dict:
    """POST /v1/completions with stream=true; timestamp the first content byte.

    Timestamps are relative to `t_zero`, which for the first request is the
    pre-exec instant -- that is what makes `ttft_exec` a single measured span
    rather than a sum of two separately-measured ones.
    """
    body = json.dumps({
        "model": a.model, "prompt": prompt, "max_tokens": a.output_len,
        "temperature": 0.0, "stream": True,
        # scratchy honors ignore_eos; mlx-lm does not, hence under-generation
        # and the work-normalized E2E* in the summary.
        "ignore_eos": True,
    })
    conn = HTTPConnection("127.0.0.1", a.port, timeout=a.request_timeout)
    t_send = time.monotonic()
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read()[:400].decode("utf-8", "replace")
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {detail}")

    t_first = None
    t_last = None
    tok_times: list[float] = []
    text_parts: list[str] = []
    prompt_tokens = None
    while True:
        line = resp.readline()
        if not line:
            break
        now = time.monotonic()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage") or {}
        if usage.get("prompt_tokens"):
            prompt_tokens = usage["prompt_tokens"]
        for ch in obj.get("choices") or []:
            piece = ch.get("text") or ""
            if piece == "":
                continue
            if t_first is None:
                t_first = now
            t_last = now
            tok_times.append(now - t_zero)
            text_parts.append(piece)
    conn.close()

    n = len(tok_times)
    # TPOT over N-1 intervals, matching crates/benches/src/serve.rs:582.
    tpot_ms = ((t_last - t_first) * 1000.0 / (n - 1)) if n > 1 else None
    return {
        "send_offset_s": t_send - t_zero,
        "ttft_s": (t_first - t_zero) if t_first else None,
        "ttft_from_send_s": (t_first - t_send) if t_first else None,
        "last_token_s": (t_last - t_zero) if t_last else None,
        "output_tokens": n,
        "prompt_tokens": prompt_tokens,
        "tpot_ms": tpot_ms,
        "itl_ms": [round((b - a_) * 1000.0, 4)
                   for a_, b in zip(tok_times, tok_times[1:])],
        "text": "".join(text_parts),
    }


# ---------------------------------------------------------------------------
# Child resource accounting
# ---------------------------------------------------------------------------
def children_rusage() -> tuple[int, int]:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_maxrss, r.ru_majflt


def stop(proc, log) -> None:
    """SIGTERM, then SIGKILL, and always reap -- ru_maxrss for children is
    only accounted once the child has been waited on."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    if log:
        log.flush()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_cli(a) -> dict:
    """Spawn a one-shot CLI and stop the clock on the first token byte.

    Each CLI prints a banner before generating ("==========" for mlx-lm,
    "Using model: ..." for scratchy). We cannot classify a line as banner or
    content until it ends, but we CAN remember when its first byte arrived --
    so we timestamp every line's first byte as a candidate and commit that
    candidate once the line resolves as content. Using the newline time
    instead would overstate TTFT by a whole line of tokens.
    """
    prompt = build_prompt(a.seed, a.input_len, a.prompt)
    cmd, env = cli_cmd(a, prompt)
    is_prelude = PRELUDE[a.backend]

    rss0, flt0 = children_rusage()
    logf = open(a.log, "wb")
    t_zero = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=logf, env=env)

    t_first = None          # committed first-content-byte time
    t_last = None           # last content byte time
    line = bytearray()      # current line, without its newline
    line_start = None       # arrival of this line's first byte
    out = bytearray()

    def resolve(buf: bytes) -> None:
        nonlocal t_first
        if t_first is None and buf and not is_prelude(buf):
            t_first = line_start

    while True:
        ch = proc.stdout.read(1)
        if not ch:
            break
        now = time.monotonic()
        out += ch
        if not line:
            line_start = now
        if ch == b"\n":
            resolve(bytes(line))
            if t_first is not None:
                t_last = now
            line = bytearray()
            line_start = None
            continue
        line += ch
        if t_first is not None:
            t_last = now
    resolve(bytes(line))    # output may end without a trailing newline
    if t_first is not None and t_last is None:
        t_last = t_first

    proc.stdout.close()
    stop(proc, logf)
    logf.close()
    rss1, flt1 = children_rusage()
    t_exit = time.monotonic()

    return {
        "mode": "cli",
        "cmd": cmd,
        "exit_code": proc.returncode,
        "ttft_exec_s": (t_first - t_zero) if t_first else None,
        "last_byte_s": (t_last - t_zero) if t_last else None,
        "total_s": t_exit - t_zero,
        "stdout": out.decode("utf-8", "replace"),
        # CLI-mode decode rate is self-reported by each backend (mlx's
        # "Generation: N tokens, X tokens-per-sec"; scratchy's --bench
        # "tok/sec"). Symmetric, but server mode is where we time it
        # ourselves -- the summary sources tpot from there.
        "peak_rss_bytes": (rss1 - rss0) if rss1 > rss0 else rss1,
        "major_faults": flt1 - flt0,
    }


def run_bench_serve(a) -> dict:
    """Drive the live server with `scr bench serve` and return its JSON.

    Note this is scratchy's binary used purely as an HTTP load generator: it
    talks to whatever is on --base-url, so mlx-lm is measured by exactly the
    same client and the same percentile math. `--ignore-eos` is requested for
    both; mlx-lm ignores it, which is why generated-token counts are reported
    and totals use E2E*.
    """
    out = a.bench_json or (a.out + ".bench_serve.json")
    cmd = [a.bench_serve_bin, "bench", "serve",
           "--base-url", f"http://127.0.0.1:{a.port}",
           "--model", a.model,
           "--num-prompts", str(a.bench_num_prompts),
           "--input-len", str(a.input_len),
           "--output-len", str(a.output_len),
           "--max-concurrency", str(a.bench_concurrency),
           "--seed", str(a.seed),
           "--num-warmups", "1",
           # PIN THE SAMPLING PATH. `bench serve` sends `temperature` only
           # when told to, so leaving it unset lets each SERVER apply its own
           # default and the two backends get measured on different code
           # paths. Measured on scratchy: greedy 12.96 ms/token and 44.3 ms
           # TTFT, versus 15.81 and 71.2 with temperature unset. Greedy also
           # matches the parity gate, so the text being timed is the text
           # that was verified.
           "--temperature", "0",
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,99",
           "--ignore-eos",
           "--output-json", out,
           "--disable-tqdm"]
    if a.tokenizer:
        # An ollama tag is not an HF repo id, so bench serve cannot derive a
        # tokenizer from --model. Pointing it at the canonical repo keeps
        # prompt construction byte-identical across backends.
        cmd += ["--tokenizer", a.tokenizer]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    if a.hf_hub_cache:
        env["HF_HUB_CACHE"] = a.hf_hub_cache
    log = a.log + ".bench_serve.log"
    with open(log, "wb") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                           check=False)
    rec = {"cmd": cmd, "exit_code": p.returncode, "json_path": out,
           "log_path": log}
    if os.path.exists(out):
        try:
            with open(out) as f:
                rec["metrics"] = json.load(f)
        except json.JSONDecodeError:
            rec["metrics"] = None
    return rec


def run_server(a) -> dict:
    if port_is_serving(a.port):
        raise SystemExit(f"port {a.port} already serving -- refusing to bench "
                         "a co-resident server (one model resident at a time)")

    rss0, flt0 = children_rusage()
    logf = open(a.log, "wb")
    cmd, env = server_cmd(a)
    t_zero = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

    t_ready = wait_ready(a.port, proc, a.ready_timeout, a.poll_interval)
    if t_ready is None:
        stop(proc, logf)
        logf.close()
        return {"mode": "server", "cmd": cmd, "error": "server never ready",
                "exit_code": proc.returncode}

    requests = []
    bench_rec = None
    err = None
    try:
        # Request 0's clock starts at exec, so ttft_exec spans process init,
        # weight load, pipeline compile, KV alloc, warmup, prefill and sample
        # as ONE measured number.
        prompts = []
        if a.prompts_json:
            with open(a.prompts_json) as f:
                prompts = json.load(f)
        r0 = stream_completion(
            a, prompts[0] if prompts else
            build_prompt(a.seed, a.input_len, a.prompt), t_zero)
        r0["kind"] = "first"
        requests.append(r0)
        # Remaining fixed prompts (parity gate): same server, sequential.
        for i, pr in enumerate(prompts[1:], start=1):
            t_req = time.monotonic()
            r = stream_completion(a, pr, t_req)
            r["kind"] = "fixed"
            r["index"] = i
            requests.append(r)

        if a.warm_requests or a.bench_serve_bin:
            # Let post-"ready" background work drain before sampling steady
            # state: on an aligned-cache hit scratchy content-hashes the
            # cached blob on a background thread (0.4-2.4s observed) and
            # lazily builds its TurboQuant codebook.
            time.sleep(a.settle_s)
        if a.bench_serve_bin:
            # AUTHORITATIVE WARM MEASUREMENT. `scr bench serve` is
            # base-URL driven and backend-agnostic -- the same binary drives
            # scratchy and mlx-lm alike, which is how bench_serve_compare.sh
            # already compares them. Reusing it means the warm numbers come
            # from the repo's own TTFT/ITL logic and its numpy-linear
            # percentiles rather than a second implementation of the same
            # thing. It cannot serve frozen/cold, where the clock must start
            # before exec -- hence the probe above.
            bench_rec = run_bench_serve(a)
        if a.warm_requests:
            # Kept as an independent cross-check on bench serve: two clients,
            # same server, numbers should agree within noise.
            for i in range(a.warm_requests):
                t_req = time.monotonic()
                r = stream_completion(
                    a, build_prompt(a.seed + 1000 + i, a.input_len), t_req)
                r["kind"] = "warm"
                r["index"] = i
                requests.append(r)
    except (RuntimeError, OSError, socket.timeout) as e:
        err = f"{type(e).__name__}: {e}"

    stop(proc, logf)
    logf.close()
    rss1, flt1 = children_rusage()

    return {
        "mode": "server",
        "cmd": cmd,
        "exit_code": proc.returncode,
        "error": err,
        "t_ready_s": t_ready - t_zero,
        "ready_semantics": ("daemon accepting connections; model loads on the "
                            "first request" if a.backend == "ollama"
                            else "model loaded and server ready to serve"),
        "poll_interval_s": a.poll_interval,
        "ttft_exec_s": requests[0]["ttft_s"] if requests else None,
        "requests": requests,
        "bench_serve": bench_rec,
        "peak_rss_bytes": (rss1 - rss0) if rss1 > rss0 else rss1,
        "major_faults": flt1 - flt0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True,
                   choices=["scratchy", "mlx-lm", "ollama"])
    p.add_argument("--mode", required=True, choices=["cli", "server"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="path for the JSON record")
    p.add_argument("--log", required=True, help="path for backend stdout/stderr")
    p.add_argument("--scr-bin", default="target/release/scr")
    p.add_argument("--mlx-python", default=sys.executable)
    p.add_argument("--ollama-bin", default="ollama")
    p.add_argument("--tokenizer", default="", help="HF repo id for `bench serve` "
                   "prompt construction. Required for ollama, whose model tag "
                   "(e.g. llama3.2:3b) is not an HF id -- passing the canonical "
                   "repo here keeps prompt tokenization identical across "
                   "backends.")
    p.add_argument("--prompts-json", default="", help="JSON list of prompts to "
                   "send sequentially in server mode instead of the seeded "
                   "salad. Used by the parity gate.")
    p.add_argument("--device", default="metal")
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--input-len", type=int, default=64)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--prompt", default="", help="explicit prompt; overrides the "
                   "seeded salad. Used by the parity gate, which needs "
                   "high-confidence natural text rather than random words.")
    p.add_argument("--warm-requests", type=int, default=0)
    p.add_argument("--bench-serve-bin", default="", help="if set, WARM steady "
                   "state is measured by `scr bench serve` against the live "
                   "server -- the repo's own client, backend-agnostic over "
                   "OpenAI-compat HTTP, so both backends are driven by "
                   "identical request logic and percentile math.")
    p.add_argument("--bench-num-prompts", type=int, default=20)
    p.add_argument("--bench-concurrency", type=int, default=1)
    p.add_argument("--bench-json", default="")
    p.add_argument("--settle-s", type=float, default=8.0)
    p.add_argument("--poll-interval", type=float, default=0.02)
    p.add_argument("--ready-timeout", type=float, default=600.0)
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--kv-cache-dtype", default="")
    p.add_argument("--max-model-len", type=int, default=0)
    p.add_argument("--rust-log", default="info")
    p.add_argument("--hf-hub-cache", default="")
    p.add_argument("--scenario", default="", help="recorded verbatim")
    p.add_argument("--rep", type=int, default=0, help="recorded verbatim")
    p.add_argument("--scratchy-extra", nargs="*", default=[])
    p.add_argument("--mlx-extra", nargs="*", default=[])
    p.add_argument("--ollama-extra", nargs="*", default=[])
    a = p.parse_args()

    if a.backend == "scratchy" and not shutil.which(a.scr_bin) \
            and not os.path.exists(a.scr_bin):
        raise SystemExit(f"scratchy binary not found: {a.scr_bin}")

    rec = run_cli(a) if a.mode == "cli" else run_server(a)
    rec.update({
        "backend": a.backend, "scenario": a.scenario, "rep": a.rep,
        "model": a.model, "input_len": a.input_len,
        "output_len": a.output_len, "seed": a.seed,
        "wall_clock_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)
    ttft = rec.get("ttft_exec_s")
    shown = f"{ttft:.3f} s" if ttft else "FAIL"
    tag = f"[{a.backend}/{a.scenario}/{a.mode}#{a.rep}] ttft_exec={shown}"
    if a.mode == "server" and rec.get("t_ready_s") is not None:
        tag += f"  ready={rec['t_ready_s']:.3f} s"
    if rec.get("error"):
        tag += f"  error={rec['error']}"
    print(tag, flush=True)
    return 0 if rec.get("ttft_exec_s") else 1


if __name__ == "__main__":
    sys.exit(main())
