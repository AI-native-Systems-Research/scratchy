#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Record coding-agent traffic for the Lane A replay benchmark (epic #158).

`record` is an HTTP relay between an agent (Claude Code, or any other client of
an Anthropic- or OpenAI-style API) and whichever engine is serving. It keeps a
copy of every request and of every response body, and passes both through
untouched.

    scripts/cc_trace.py record --listen 127.0.0.1:8787 \\
        --upstream http://127.0.0.1:8000 \\
        --out bench_results/claude-code/corpus/gemma12b-01/
    scr launch claude -m gemma-4-12b-it --server-url http://127.0.0.1:8787

This is the raw first stage. The corpus holds the session's real prompts, file
contents and paths, so it stays on the measuring machine (`bench_results/` is
gitignored); the redacted profile that #163 checks in is derived from it later.
Only credentials are redacted here. Layout:

    manifest.json           agent, engine, versions, start/stop, counts
    index.jsonl             one row per request, appended when its response ends
    requests/NNNNNN.json    method, path, query, headers, body
    responses/NNNNNN.body   the response body exactly as the engine sent it

The relay rewrites nothing and parses no response: the `usage` block, the
output length and which requests are agent turns are all read later, from the
stored files, by whatever reads the corpus.

`replay` is that reader. It re-sends a corpus's `POST /v1/messages` turns to
one engine, one at a time, times each from the SSE stream, and writes one run
in the leaderboard's Lane A shape (`site/build_leaderboard.py` validates it):

    scripts/cc_trace.py replay --corpus bench_results/claude-code/corpus/gemma12b-01/ \\
        --endpoint http://127.0.0.1:8000 --engine scratchy --model gemma-4-12b-it \\
        --machine m5max-64 --rung R1 --repeat 3 --scratchy-sha "$(git rev-parse HEAD)" \\
        --ollama-version 0.40.0 --kv-cache-dtype scratchy=bf16,ollama=f16 \\
        --quantization scratchy=mlx-affine-b4-g64,ollama=Q4_K_M \\
        --checkpoint-revision scratchy=<repo>@<sha>,ollama=<tag>@<digest> \\
        --out run-laneA.json

One engine per invocation: alternating engines across repeats is the runner's
job, not this tool's. One warm-up pass runs before the measured repeats and is
reported beside them, never in them (`--no-warmup` skips it). Every pass starts
with the engine's prefix cache cleared: scratchy through its reset endpoint,
ollama by unloading and reloading the model.

A run that cannot be trusted is written with `status: "void"` and the reason,
and exits non-zero. That covers a session too long for the engine's context
window, a request counter that disagrees with what was sent, a cache reset that
failed or a cache that survived it, and a turn that returned no tokens or no
content, timed out, answered 4xx, was cut short at max_tokens, or counted a
prompt shorter than the recording's by more than the tolerance.

`live` is Lane B: it runs headless Claude Code on a graded task several times,
records each session through the relay above, grades it, and writes one run in
the leaderboard's Lane B shape. Replay owns every latency claim and live every
outcome claim, and neither writes the other's metrics.

    scripts/cc_trace.py live --task tasks/01-bugfix-python-phone-number \\
        --endpoint http://127.0.0.1:8000 --engine scratchy --model qwen3.6-35b-a3b \\
        --served-model mlx-community/Qwen3.6-35B-A3B-4bit --machine m5max-64 --rung R1 \\
        --scr target/release/scr --scratchy-sha "$(git rev-parse HEAD)" ... --out run-laneB.json

A tool call counts as failed when the engine broke it (arguments that are not
JSON, a call Claude Code rejected, call markup its parser left in plain text),
never when the agent's own command failed. A session that never reached the
engine, or that the engine's request counter does not account for, voids the
run.
Standard library only.
"""

import argparse
import gzip
import hashlib
import http.client
import http.server
import json
import os
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

FORMAT = 1

# RFC 9110 §7.6.1 connection-scoped headers, plus the ones the relay sets itself.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
# Header and query-parameter names whose values never reach the saved copy:
# `authorization`, `x-api-key`, `proxy-authorization`, `cookie`, and the
# per-gateway variants (`api-key`, `x-goog-api-key`, `x-litellm-api-key`, ...).
SECRET_NAME = re.compile(r"key|token|secret|auth|cookie", re.IGNORECASE)
REDACTED = "<redacted>"
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
# `User-Agent` product token -> agent name, for the agents we know by name.
AGENTS = {"claude-cli": "claude-code"}
DRAIN_SECONDS = 30


def _is_secret(name, extra):
    return bool(SECRET_NAME.search(name)) or name.lower() in extra


def redact_headers(items, extra=frozenset()):
    """Header pairs with secret values replaced; repeated headers are kept."""
    return [[k, REDACTED if _is_secret(k, extra) else v] for k, v in items]


def redact_query(query, extra=frozenset()):
    pairs = parse_qsl(query, keep_blank_values=True)
    if not any(_is_secret(k, extra) for k, _ in pairs):
        return query
    return urlencode([(k, REDACTED if _is_secret(k, extra) else v) for k, v in pairs])


def parse_agent(user_agent):
    """(agent, version) from a `User-Agent` such as `claude-cli/2.1.295 (external, sdk-cli)`."""
    product = (user_agent or "").split(" ", 1)[0]
    name, _, version = product.partition("/")
    if not name:
        return None, None
    return AGENTS.get(name.lower(), name.lower()), version or None


def _json_or_none(data):
    try:
        return json.loads(data)
    except ValueError:
        return None


def _connect(scheme, host, port, timeout):
    cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    return cls(host, port, timeout=timeout)


def script_sha():
    """The checkout this script runs from, with `-dirty` when it has local edits."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "describe", "--always",
             "--dirty", "--abbrev=40"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def probe_engine(upstream, timeout=3):
    """Ask the engine behind `upstream` what it is: {"name", "version"} or None."""
    scheme, host, port, base = upstream
    # ollama answers /api/version; scratchy answers /server_info.
    for path, read in (("/api/version", lambda j: ("ollama", j.get("version"))),
                       ("/server_info", lambda j: ("scratchy", j["system_env"]["vllm_version"]))):
        conn = _connect(scheme, host, port, timeout)
        try:
            conn.request("GET", base + path)
            resp = conn.getresponse()
            data = resp.read()
            if resp.status == 200:
                name, version = read(json.loads(data))
                return {"name": name, "version": version}
        except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError):
            pass
        finally:
            conn.close()
    return None


class Corpus:
    """One recording session's output directory. Thread-safe."""

    def __init__(self, out, upstream, listen, *, agent=None, engine=None, engine_sha=None):
        self.dir = Path(out)
        if (self.dir / "manifest.json").exists() or (self.dir / "index.jsonl").exists():
            raise SystemExit(f"{self.dir} already holds a corpus; pick a new --out")
        (self.dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.dir / "responses").mkdir(exist_ok=True)
        self._cond = threading.Condition()
        self._seq = 0
        self._open = set()
        self._models = set()
        self.manifest = {
            "format": FORMAT,
            "upstream": upstream,
            "listen": listen,
            "started": time.time(),
            "stopped": None,
            # From the first request's User-Agent unless --agent names it.
            "agent": agent,
            "agent_version": None,
            "user_agent": None,
            # Asked of the engine itself; `engine_sha` is whatever --engine-sha said.
            "engine": engine,
            "engine_sha": engine_sha,
            "script_sha": script_sha(),
            "models": [],
            "relayed": 0,
            "saved": 0,
            "unfinished": [],
        }
        self._write_manifest()

    def _write_manifest(self):
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2) + "\n")
        tmp.replace(self.dir / "manifest.json")

    def begin(self):
        """Allocate a sequence number; returns (seq, requests already in flight)."""
        with self._cond:
            self._seq += 1
            in_flight = len(self._open)
            self._open.add(self._seq)
            return self._seq, in_flight

    def note_client(self, user_agent, model):
        with self._cond:
            if user_agent and self.manifest["user_agent"] is None:
                agent, version = parse_agent(user_agent)
                self.manifest["user_agent"] = user_agent
                self.manifest["agent"] = self.manifest["agent"] or agent
                self.manifest["agent_version"] = version
                self._write_manifest()
            if isinstance(model, str):
                self._models.add(model)

    def save_request(self, seq, method, path, query, headers, body, parsed, t_start):
        record = {"seq": seq, "method": method, "path": path, "query": query,
                  "headers": headers, "t_start": t_start}
        if parsed is not None:
            record["body"] = parsed
        else:
            record["body_text"] = body.decode("utf-8", "replace")
        (self.dir / "requests" / f"{seq:06d}.json").write_text(json.dumps(record) + "\n")
        with self._cond:
            self.manifest["saved"] += 1

    def response_file(self, seq):
        return open(self.dir / "responses" / f"{seq:06d}.body", "wb")

    def end(self, row):
        with self._cond:
            self._open.discard(row["seq"])
            self.manifest["relayed"] += 1
            with open(self.dir / "index.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            self._cond.notify_all()

    def drain(self, timeout):
        """Wait up to `timeout` s for open relays to finish; True when none are left."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._open, timeout)

    def close(self):
        with self._cond:
            self.manifest["stopped"] = time.time()
            self.manifest["models"] = sorted(self._models)
            # Saved in requests/ but never finished, so they have no index row.
            self.manifest["unfinished"] = sorted(self._open)
            self._write_manifest()


class RelayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Set per server by make_server.
    corpus: Corpus
    upstream: tuple
    upstream_timeout: float
    redact: frozenset

    def log_message(self, fmt, *args):
        pass

    def _read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";")[0], 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _relay(self):
        t_start = time.time()
        seq, in_flight = self.corpus.begin()
        path, _, query = self.path.partition("?")
        row = {"seq": seq, "method": self.command, "path": path, "status": None,
               "t_start": t_start, "t_end": None, "in_flight": in_flight,
               "req_bytes": 0, "resp_bytes": 0, "stream": None,
               "resp_content_type": None, "resp_content_encoding": None, "error": None}
        try:
            body = self._read_body()
            row["req_bytes"] = len(body)
            parsed = _json_or_none(body) if body else None
            fields = parsed if isinstance(parsed, dict) else {}
            row["stream"] = fields.get("stream")
            if body:
                # Body-less requests can come from someone else: `scr launch`
                # checks /server_info through the relay with its own client.
                self.corpus.note_client(self.headers.get("User-Agent"), fields.get("model"))
                self.corpus.save_request(seq, self.command, path, redact_query(query, self.redact),
                                         redact_headers(self.headers.items(), self.redact),
                                         body, parsed, t_start)
            self._forward(body, row)
        except (BrokenPipeError, ConnectionResetError) as e:
            # The client gave up while the relay was writing to it.
            row["error"] = f"client disconnected: {e}"
            self.close_connection = True
        except Exception as e:
            # Bad framing from the client, a disk error: record it, then let the
            # server print the traceback.
            row["error"] = f"relay: {e!r}"
            self.close_connection = True
            raise
        finally:
            row["t_end"] = time.time()
            self.corpus.end(row)
            print(f"[{seq:06d}] {self.command} {path} -> {row['status']} "
                  f"{row['resp_bytes']}B {row['t_end'] - t_start:.2f}s"
                  + (f" ERROR {row['error']}" if row["error"] else ""),
                  file=sys.stderr, flush=True)

    def _client_gone(self):
        """True when the client has closed its end (readable, and at EOF)."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _send_502(self, e):
        msg = f"cc_trace: upstream failed: {e}\n".encode()
        self.send_response(502)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def _forward(self, body, row):
        scheme, host, port, base = self.upstream
        conn = _connect(scheme, host, port, self.upstream_timeout)
        saved = None
        # Everything below closes the upstream socket on the way out, so an
        # engine is never left generating for a client that has gone.
        try:
            try:
                # putrequest/putheader rather than request(): keeps repeated
                # headers, and adds no Accept-Encoding the client did not send.
                conn.putrequest(self.command, base + self.path, skip_accept_encoding=True)
                for k, v in self.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        conn.putheader(k, v)
                if body or self.command in ("POST", "PUT", "PATCH"):
                    conn.putheader("Content-Length", str(len(body)))
                conn.endheaders(body or None)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                row["status"], row["error"] = 502, f"upstream: {type(e).__name__}: {e}"
                self._send_502(e)
                return
            row["status"] = resp.status
            row["resp_content_type"] = resp.getheader("Content-Type")
            row["resp_content_encoding"] = resp.getheader("Content-Encoding")
            if self._client_gone():
                # A write to a half-closed socket can still succeed, so ask instead.
                row["error"] = "client disconnected before the response"
                self.close_connection = True
                return
            no_body = self.command == "HEAD" or resp.status in (204, 304) or 100 <= resp.status < 200
            length = resp.getheader("Content-Length")
            # Keep the upstream's framing when it gave a length; otherwise
            # (chunked, or close-delimited) re-chunk what http.client de-chunks.
            chunked = not no_body and length is None
            self.send_response_only(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            if length is not None:
                self.send_header("Content-Length", length)
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while not no_body:
                try:
                    chunk = resp.read1(65536)
                except (OSError, http.client.HTTPException) as e:
                    # The status line is already sent; all that is left is to cut the stream.
                    row["error"] = f"upstream mid-stream: {type(e).__name__}: {e}"
                    self.close_connection = True
                    return
                if not chunk:
                    break
                if saved is None:
                    saved = self.corpus.response_file(row["seq"])
                saved.write(chunk)
                row["resp_bytes"] += len(chunk)
                if chunked:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        finally:
            conn.close()
            if saved is not None:
                saved.close()


for _m in METHODS:
    setattr(RelayHandler, f"do_{_m}", RelayHandler._relay)


def _parse_url(url, flag="--upstream"):
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"{flag} must be an http(s) URL, got {url!r}")
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/")


class _Server6(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_server(listen, upstream, out, *, agent=None, engine_sha=None,
                upstream_timeout=1200, redact=()):
    """Build (not start) a recording relay. `listen` is (host, port); port 0 picks one."""
    up = _parse_url(upstream)
    corpus = Corpus(out, upstream, f"{listen[0]}:{listen[1]}", agent=agent,
                    engine=probe_engine(up), engine_sha=engine_sha)
    handler = type("Handler", (RelayHandler,), {
        "corpus": corpus, "upstream": up, "upstream_timeout": upstream_timeout,
        "redact": frozenset(h.lower() for h in redact)})
    server_cls = _Server6 if ":" in listen[0] else http.server.ThreadingHTTPServer
    server = server_cls(listen, handler)
    server.daemon_threads = True
    server.corpus = corpus
    server.upstream = up
    return server


def parse_listen(listen):
    """`host:port` or `[v6]:port` -> (host, port)."""
    host, _, port = listen.rpartition(":")
    return host.strip("[]") or "127.0.0.1", int(port)


def cmd_record(args):
    server = make_server(parse_listen(args.listen), args.upstream, args.out,
                         agent=args.agent, engine_sha=args.engine_sha,
                         upstream_timeout=args.upstream_timeout, redact=args.redact_header)
    m = server.corpus.manifest
    engine = (f"{m['engine']['name']} {m['engine']['version']}" if m["engine"]
              else "not answering yet (asked again on stop)")
    print(f"recording {args.listen} -> {args.upstream} into {args.out}\n"
          f"  engine {engine}, engine sha {m['engine_sha']}, script {m['script_sha']}\n"
          f"  Ctrl-C to stop", file=sys.stderr, flush=True)
    # SIGTERM finalizes the manifest the same way Ctrl-C does. SIGINT is set
    # too: a script that starts this with `&` hands it SIGINT already ignored.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            if not server.corpus.drain(DRAIN_SECONDS):
                print(f"stopping with requests still open after {DRAIN_SECONDS}s; "
                      "they are listed as unfinished", file=sys.stderr)
        except KeyboardInterrupt:
            pass
        if server.corpus.manifest["engine"] is None:
            server.corpus.manifest["engine"] = probe_engine(server.upstream)
        server.corpus.close()
        m = server.corpus.manifest
        print(f"stopped: {m['relayed']} requests relayed, {m['saved']} saved"
              + (f", {len(m['unfinished'])} unfinished" if m["unfinished"] else ""),
              file=sys.stderr)


# ---------------------------------------------------------------------- replay

SCHEMA = 1
TURN_PATH = "/v1/messages"
# What each engine's server offers the replay's checks. Data, not a code path:
# the replay itself is the same for both.
#   counts_requests  reports `requests_served` on /server_info (#161 fairness rule 7)
#   reset            how its prefix cache is cleared before every pass: "post"
#                    POSTs `reset_path`; "reload" unloads the model and loads it
#                    again, the only way ollama has, since its cache lives in the
#                    process holding the model (#551). None: it cannot be cleared.
ENGINES = {
    "scratchy": {"counts_requests": True, "reset": "post", "reset_path": "/reset_prefix_cache"},
    "ollama": {"counts_requests": False, "reset": "reload", "reset_path": None},
}
# How often a reload checks ollama's /api/ps for the model to leave or return.
RELOAD_POLL_S = 0.25
# Provenance a run names for both engines. It comes from flags: the tool cannot
# discover it, and the validator requires both sides' values on every run.
PER_ENGINE_PROVENANCE = ("kv_cache_dtype", "quantization", "checkpoint_revision")
# `site/build_leaderboard.py`'s TURN_CLASSES; "all" is the headline cell itself.
TURN_CLASSES = ("all", "tool-call", "file-write")
# Claude Code writes files through these tools, so a turn that calls one is a
# file-write rather than a tool call.
FILE_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
# Delta fields that carry generated content. A delta with all of them empty
# (or a `signature_delta`) is framing, not a first token.
DELTA_CONTENT = ("text", "partial_json", "thinking")
# Request headers the replay does not resend: framing, and compression (the
# stream is timed as plain bytes). Redacted secrets are dropped by value.
NOT_REPLAYED = HOP_BY_HOP | {"accept-encoding"}
# Lane A metrics read straight off each turn, and why a run can have none.
TURN_METRICS = {
    "ttft_ms": "no turn produced content",
    "tpot_ms": "no turn streamed its reply in more than one delta",
    "turn_e2e_ms": "no turn finished",
    "reprefill_ratio": "no turn reported its prompt usage",
    "e2e_star_ms": "no turn both streamed its reply and has a recorded response to take the "
                   "output length from",
}
# How much shorter than the recorded prompt a replayed prompt may be before the
# run is void: the larger of the two. Room for two tokenizers or chat templates
# counting the same prompt differently, not for an engine that silently dropped
# part of it (#161 fairness rule 3). A longer prompt is only counted.
PROMPT_SHORTFALL_TOKENS = 64
PROMPT_SHORTFALL_FRACTION = 0.05
# Disclosure ids, declared in site/data/leaderboard/board.json (#176).
DISCLOSE_KV = "kv-dtype-unmatched"
DISCLOSE_QUANT = "quantizer-differs"
DISCLOSE_NO_COUNTER = "requests-served-unavailable"
DISCLOSE_NO_RESET = "prefix-cache-not-reset"
DISCLOSE_SERIAL = "replay-serialized"


class TurnFailed(Exception):
    """A turn the run cannot average in: it voids the run."""


class ResetFailed(Exception):
    """A prefix-cache reset that did not happen: it voids the run."""


def percentile(xs, q):
    """Linearly interpolated percentile, `q` in 0..100."""
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def reprefill_ratio(input_tokens, cache_read, cache_creation=0):
    """Prompt tokens prefilled / prompt tokens sent, from Anthropic usage.

    `input_tokens` excludes both cache reads and cache writes, so a token
    written to the cache this turn was still prefilled this turn.
    """
    prefilled = (input_tokens or 0) + (cache_creation or 0)
    total = prefilled + (cache_read or 0)
    return prefilled / total if total else None


def prompt_tokens(usage):
    """The whole prompt's length from Anthropic usage, or None when it has no count."""
    if usage.get("input_tokens") is None:
        return None
    return sum(usage.get(k) or 0 for k in
               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))


def sse_events(lines):
    """(data dict, arrival time) per SSE event from an iterable of raw lines."""
    event, data = None, []
    for raw in lines:
        line = raw.rstrip(b"\r\n")
        if line.startswith(b"event:"):
            event = line[6:].strip().decode()
        elif line.startswith(b"data:"):
            data.append(line[5:].strip())
        elif not line and data:
            try:
                payload = json.loads(b"\n".join(data))
            except ValueError:
                raise TurnFailed(f"unparseable SSE data: {b''.join(data)[:200]!r}") from None
            if isinstance(payload, dict):
                payload.setdefault("type", event)
                yield payload, time.monotonic()
            event, data = None, []


def time_stream(lines, t_sent):
    """One turn's timings from its Anthropic SSE stream. Raises TurnFailed."""
    usage, deltas, t_stop, stop_reason = {}, [], None, None
    for ev, t in sse_events(lines):
        kind = ev.get("type")
        if kind == "message_start":
            usage.update((ev.get("message") or {}).get("usage") or {})
        elif kind == "content_block_delta":
            d = ev.get("delta") or {}
            if any(d.get(k) for k in DELTA_CONTENT):
                deltas.append(t)
        elif kind == "message_delta":
            # The final usage; it overrides message_start's provisional one.
            usage.update(ev.get("usage") or {})
            stop_reason = (ev.get("delta") or {}).get("stop_reason") or stop_reason
        elif kind == "error":
            raise TurnFailed(f"error event: {ev.get('error')}")
        elif kind == "message_stop":
            t_stop = t
            break
    if t_stop is None:
        raise TurnFailed("stream ended without message_stop")
    out = usage.get("output_tokens") or 0
    if out <= 0:
        # A dead worker behind a live HTTP layer looks exactly like this (#413).
        raise TurnFailed(f"zero output tokens (output_tokens={out}, {len(deltas)} content deltas)")
    if not deltas:
        # Tokens were reported but nothing reached the client: an empty reply,
        # or a tool call streamed with no name and no arguments (#498).
        raise TurnFailed(f"no content: the engine reported {out} output tokens but streamed no "
                         "text, tool arguments or thinking")
    # A reply that arrives in one delta has no decode interval to measure. Its
    # TPOT would read as 0 and flatter an engine that buffers its output, so it
    # has none, the way bench_serve_compare.sh leaves unstreamed requests out.
    unstreamed = out > 1 and len(deltas) < 2
    gaps = [(b - a) * 1000 for a, b in zip(deltas, deltas[1:])]
    return {
        "ttft_ms": (deltas[0] - t_sent) * 1000,
        "turn_e2e_ms": (t_stop - t_sent) * 1000,
        "tpot_ms": None if unstreamed or out < 2 else (deltas[-1] - deltas[0]) * 1000 / (out - 1),
        # Per chunk: Anthropic deltas carry text, not token counts (#208).
        "itl_chunk_p99_ms": percentile(gaps, 99) if gaps else None,
        "unstreamed": unstreamed,
        "stop_reason": stop_reason,
        "output_tokens": out,
        "input_tokens": usage.get("input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "prompt_tokens": prompt_tokens(usage),
        "reprefill_ratio": reprefill_ratio(usage.get("input_tokens"),
                                           usage.get("cache_read_input_tokens"),
                                           usage.get("cache_creation_input_tokens")),
    }


def recorded_usage(corpus, row):
    """{output_tokens, prompt_tokens, stop_reason} of the response the recorder
    saved for `row`; empty when it saved none.

    The engine-independent reference: E2E* normalizes to its output length, and
    the context and cut-short checks compare against it.
    """
    path = Path(corpus) / "responses" / f"{row['seq']:06d}.body"
    if not path.exists():
        return {}
    data = path.read_bytes()
    if (row.get("resp_content_encoding") or "").lower() == "gzip":
        data = gzip.decompress(data)
    usage, stop_reason = {}, None
    try:
        for ev, _ in sse_events(data.splitlines(keepends=True) + [b"\n"]):
            usage.update((ev.get("message") or {}).get("usage") or ev.get("usage") or {})
            stop_reason = (ev.get("delta") or {}).get("stop_reason") or stop_reason
    except TurnFailed:
        return {}
    return {"output_tokens": usage.get("output_tokens") or None,
            "prompt_tokens": prompt_tokens(usage), "stop_reason": stop_reason}


def load_corpus(corpus):
    """(manifest, turns, skipped) for a corpus directory.

    `turns` are its `POST /v1/messages` requests in seq order. Everything else —
    GETs, token counting, requests that failed or never finished when recorded —
    is skipped and counted in `skipped` by reason (#161 fairness rule 9).
    """
    d = Path(corpus)
    manifest = json.loads((d / "manifest.json").read_text())
    rows = {}
    for line in (d / "index.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["seq"]] = row
    reqs = {}
    for p in (d / "requests").glob("*.json"):
        req = json.loads(p.read_text())
        reqs[req["seq"]] = req
    turns, skipped = [], {}
    for seq in sorted(set(rows) | set(reqs)):
        req, row = reqs.get(seq), rows.get(seq)
        rec = req or row
        if (rec["method"], rec["path"]) != ("POST", TURN_PATH):
            reason = "not a turn"
        elif row is None:
            reason = "unfinished when recorded"
        elif row.get("status") != 200 or row.get("error"):
            reason = "failed when recorded"
        elif req is None or not isinstance(req.get("body"), dict):
            reason = "no JSON body saved"
        else:
            turns.append({"seq": seq, "query": req.get("query") or "",
                          "headers": req.get("headers") or [], "body": req["body"],
                          "in_flight": row.get("in_flight", 0),
                          "recorded": recorded_usage(d, row)})
            continue
        skipped[reason] = skipped.get(reason, 0) + 1
    return manifest, turns, skipped


def _without_cache_control(x):
    # Claude Code moves its cache_control breakpoints every turn, so they are
    # not part of what was said.
    if isinstance(x, dict):
        return {k: _without_cache_control(v) for k, v in x.items() if k != "cache_control"}
    if isinstance(x, list):
        return [_without_cache_control(v) for v in x]
    return x


def _prefix_hashes(messages):
    """h[k] identifies messages[:k], ignoring cache_control."""
    h = [b""]
    for m in messages:
        blob = json.dumps(_without_cache_control(m), sort_keys=True).encode()
        h.append(hashlib.sha256(h[-1] + hashlib.sha256(blob).digest()).digest())
    return h


def reply_class(message):
    """The turn class of one assistant message, or None for "all" only."""
    content = message.get("content")
    blocks = content if isinstance(content, list) else []
    tools = {b.get("name") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"}
    if tools & FILE_WRITE_TOOLS:
        return "file-write"
    return "tool-call" if tools else None


def classify(turns):
    """Set each turn's `turn_class` from the corpus, never from a replayed reply.

    Claude Code re-sends the whole conversation, so turn N's reply is the
    assistant message that follows turn N's messages in the first later request
    that extends them. Reading it there keeps the class engine-independent.
    """
    hashes = [_prefix_hashes(t["body"].get("messages") or []) for t in turns]
    for i, t in enumerate(turns):
        n = len(hashes[i]) - 1
        t["turn_class"] = None
        for j in range(i + 1, len(turns)):
            later = turns[j]["body"].get("messages") or []
            if (len(later) > n and isinstance(later[n], dict)
                    and later[n].get("role") == "assistant" and hashes[j][n] == hashes[i][n]):
                t["turn_class"] = reply_class(later[n])
                break
    return turns


def _http(method, url, body=None, timeout=10):
    """(status, parsed JSON or None) for one small request."""
    scheme, host, port, base = _parse_url(url, "URL")
    path = base or "/"
    if urlsplit(url).query:
        path += "?" + urlsplit(url).query
    conn = _connect(scheme, host, port, timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, _json_or_none(resp.read())
    finally:
        conn.close()


def requests_served(url):
    """`requests_served` from a /server_info URL, or None when it cannot say."""
    try:
        status, info = _http("GET", url)
    except (OSError, http.client.HTTPException):
        return None
    if status != 200 or not isinstance(info, dict):
        return None
    n = info.get("requests_served")
    return n if isinstance(n, int) else None


def served_context_length(endpoint, model):
    """`max_model_len` from GET /v1/models: `model`'s card, else the only card's."""
    try:
        status, listing = _http("GET", endpoint.rstrip("/") + "/v1/models")
    except (OSError, http.client.HTTPException):
        return None
    if status != 200 or not isinstance(listing, dict):
        return None
    cards = [c for c in listing.get("data") or [] if isinstance(c, dict)]
    mine = [c for c in cards if c.get("id") == model] or (cards if len(cards) == 1 else [])
    n = mine[0].get("max_model_len") if mine else None
    return n if isinstance(n, int) else None


def _call(method, url, body=None, timeout=10):
    """_http, with a connection failure as a status string rather than an exception."""
    try:
        return _http(method, url, body=body, timeout=timeout)
    except (OSError, http.client.HTTPException) as e:
        return f"{type(e).__name__}: {e}", None


def ollama_loaded(endpoint, model):
    """Whether ollama's GET /api/ps lists `model` as loaded. Raises ResetFailed."""
    status, ps = _call("GET", endpoint.rstrip("/") + "/api/ps")
    if status != 200 or not isinstance(ps, dict):
        raise ResetFailed(f"GET /api/ps answered {status}")
    return any(m.get("name") == model or m.get("model") == model
               for m in ps.get("models") or [] if isinstance(m, dict))


def ollama_reload(endpoint, model, timeout_s):
    """Clear ollama's prefix cache by unloading `model` and loading it again.

    The cache lives in the process that holds the model, so a reload is the only
    way to clear it (#551). A request's own `keep_alive: 0` unloads even when the
    server runs with OLLAMA_KEEP_ALIVE=-1; `keep_alive: -1` then keeps the
    reloaded model resident. Both steps wait on /api/ps rather than a fixed
    sleep. Returns the seconds the load took. Raises ResetFailed.
    """
    url = endpoint.rstrip("/") + "/api/generate"
    for keep_alive, want_loaded, step in ((0, False, "unload"), (-1, True, "load")):
        t0 = time.monotonic()
        body = json.dumps({"model": model, "keep_alive": keep_alive}).encode()
        status, reply = _call("POST", url, body=body, timeout=timeout_s)
        if status != 200:
            error = reply.get("error") if isinstance(reply, dict) else None
            raise ResetFailed(f"{step} of {model!r} answered {status}"
                              + (f": {error}" if error else ""))
        while ollama_loaded(endpoint, model) != want_loaded:
            if time.monotonic() - t0 > timeout_s:
                raise ResetFailed(f"{model!r} still {'not ' if want_loaded else ''}loaded "
                                  f"{timeout_s:.0f}s after the {step}")
            time.sleep(RELOAD_POLL_S)
    return time.monotonic() - t0


def make_reset(engine, endpoint, *, reset_cache_url=None, model=None, timeout_s=600):
    """(how the prefix cache is reset, a function that resets it, the URL it
    POSTs to or None) for one run.

    The function returns the seconds the reset took and raises ResetFailed.
    (None, None, None) when this engine's cache cannot be cleared.
    """
    if reset_cache_url or ENGINES[engine]["reset"] == "post":
        url = reset_cache_url or endpoint.rstrip("/") + ENGINES[engine]["reset_path"]

        def post():
            t0 = time.monotonic()
            status, _ = _call("POST", url, body=b"{}")
            if status not in (200, 204):
                raise ResetFailed(f"POST {url} answered {status}")
            return time.monotonic() - t0

        return f"POST {url}", post, url
    if ENGINES[engine]["reset"] == "reload":
        return (f"unload and reload {model}",
                lambda: ollama_reload(endpoint, model, timeout_s), None)
    return None, None, None


def send_turn(endpoint, turn, *, temperature, served_model=None, timeout_s=600):
    """Replay one turn and time it. Raises TurnFailed."""
    scheme, host, port, base = endpoint
    body = dict(turn["body"])
    # The edits: pinned sampling, a stream to time, and (only when asked) the
    # name the engine serves the model under. max_tokens, the messages, the
    # tools and cache_control are the work; they stay as recorded.
    body["temperature"] = temperature
    body["stream"] = True
    if served_model:
        body["model"] = served_model
    data = json.dumps(body).encode()
    conn = _connect(scheme, host, port, timeout_s)
    try:
        path = base + TURN_PATH + (f"?{turn['query']}" if turn["query"] else "")
        conn.putrequest("POST", path, skip_accept_encoding=True)
        names = set()
        for k, v in turn["headers"]:
            if k.lower() not in NOT_REPLAYED and v != REDACTED:
                conn.putheader(k, v)
                names.add(k.lower())
        if "content-type" not in names:
            conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", str(len(data)))
        conn.endheaders(data)
        t_sent = time.monotonic()
        resp = conn.getresponse()
        if resp.status != 200:
            snippet = resp.read(500).decode("utf-8", "replace")
            hint = (" — a 4xx mid-corpus usually means it does not fit this engine's "
                    "context window (#322)" if 400 <= resp.status < 500 else "")
            raise TurnFailed(f"HTTP {resp.status}{hint}: {snippet}")
        ctype = resp.getheader("Content-Type") or ""
        if not ctype.startswith("text/event-stream"):
            raise TurnFailed(f"expected an event stream, got Content-Type {ctype!r}")
        deadline = t_sent + timeout_s

        def lines():
            for line in iter(resp.readline, b""):
                if time.monotonic() > deadline:
                    raise TurnFailed(f"timed out after {timeout_s}s")
                yield line

        timing = time_stream(lines(), t_sent)
        # Read past message_stop to the end of the body, so the engine sees a
        # clean close rather than a reset.
        resp.read()
        return timing
    except TimeoutError:
        raise TurnFailed(f"timed out: no byte for {timeout_s}s") from None
    except (OSError, http.client.HTTPException) as e:
        raise TurnFailed(f"{type(e).__name__}: {e}") from None
    finally:
        conn.close()


def parse_per_engine(flag, text):
    """`scratchy=a,ollama=b` -> {"scratchy": "a", "ollama": "b"}, every engine named."""
    if not text:
        raise SystemExit(f"{flag} is required: give it per engine, e.g. "
                         f"{','.join(f'{e}=<value>' for e in ENGINES)}")
    out = {}
    for part in text.split(","):
        k, sep, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if not sep or k not in ENGINES or not v:
            raise SystemExit(f"{flag}: expected engine=value with engine in "
                             f"{sorted(ENGINES)}, got {part!r}")
        out[k] = v
    missing = sorted(set(ENGINES) - set(out))
    if missing:
        raise SystemExit(f"{flag} has no value for {', '.join(missing)}; every run names "
                         "both engines' so the two sides' records agree")
    return out


def _stat(values):
    # `samples` is how many values the stat is over, beside `n`, the repeat count.
    return {"median": round(percentile(values, 50), 4), "p10": round(percentile(values, 10), 4),
            "p90": round(percentile(values, 90), 4), "p99": round(percentile(values, 99), 4),
            "samples": len(values)}


def _group(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return dict(sorted(out.items()))


def e2e_star(t):
    """ttft + (recorded output length - 1) * tpot for one turn, or None.

    The per-turn form of bench_serve_compare.sh's E2E*. That script can work
    from medians because each of its cells has one fixed output length; a
    coding session mixes short tool calls with long file writes, so a product
    of medians would describe no real turn. The length is the one the
    *recording* produced, so an engine that stops early gains nothing. An
    unstreamed turn has no TPOT, so it has no E2E* either.
    """
    out = t["recorded_output_tokens"]
    if not out or t["unstreamed"]:
        return None
    if out == 1:
        return t["ttft_ms"]
    return None if t["tpot_ms"] is None else t["ttft_ms"] + (out - 1) * t["tpot_ms"]


def prompt_shortfall(recorded, replayed):
    """Tokens the replayed prompt is short of the recorded one, when that is more
    than the tolerance; 0 otherwise, or when either side has no count."""
    if not recorded or replayed is None:
        return 0
    short = recorded - replayed
    allowed = max(PROMPT_SHORTFALL_TOKENS, PROMPT_SHORTFALL_FRACTION * recorded)
    return short if short > allowed else 0


def aggregate(samples, *, engine, model, repeat):
    """(measurements, gaps, by_turn_index) from every replayed turn of every pass.

    Each stat is over every (repeat, turn) value of the measured repeats, and
    `stat.samples` says how many; `n` is the repeat count, the page's "×reps".
    The board takes series points as [x, y] pairs, so the series carries each
    turn index's p50, and `by_turn_index` (written to the .turns.json artifact)
    carries its p50 and p99.

    The warm-up pass is never in any of that. It is reported beside it, as the
    same metrics under a `pass: warmup` facet with n 1: the cost of the first
    session after the engine starts, which is a different thing (#551).
    """
    warm = [t for t in samples if t["warmup"]]
    samples = [t for t in samples if not t["warmup"]]
    base = {"lane": "A", "engine": engine, "model": model}
    measurements, gaps, by_turn_index = [], [], {}
    classes = {tc: [t for t in samples if t["turn_class"] == tc] for tc in TURN_CLASSES[1:]}
    per_turn = _group(samples, "turn_index")

    def add(metric, values, facet=None, points=None):
        if not values:
            return False
        m = {**base, "metric": metric, "n": repeat, "stat": _stat(values)}
        if facet:
            m["facet"] = facet
        if points:
            m["series"] = {"x": "turn_index", "points": points}
        measurements.append(m)
        return True

    def per_turn_cells(metric, field, facet=None):
        def values(rows):
            return [t[field] for t in rows if t[field] is not None]

        turns = {i: v for i, rows in per_turn.items() if (v := values(rows))}
        by_turn_index[field] = [{"turn_index": i, "p50": round(percentile(v, 50), 4),
                                 "p99": round(percentile(v, 99), 4), "samples": len(v)}
                                for i, v in turns.items()]
        points = None if facet else [[i, round(percentile(v, 50), 4)] for i, v in turns.items()]
        found = add(metric, values(samples), facet=facet, points=points)
        for tc, rows in classes.items():
            add(metric, values(rows), facet={**(facet or {}), "turn_class": tc})
        return found

    for metric, none_reason in TURN_METRICS.items():
        if not per_turn_cells(metric, metric):
            gaps.append({**base, "metric": metric, "state": "not-run", "reason": none_reason})

    # ITL: only a per-chunk number exists on the wire, so the headline is a gap
    # and the chunk figure sits beside it under its own facet.
    gaps.append({**base, "metric": "itl_p99_ms", "state": "blocked", "issue": 208,
                 "reason": "Anthropic SSE deltas carry text, not token counts, so per-token "
                           "ITL is not on the wire; see the granularity=chunk facet"})
    per_turn_cells("itl_p99_ms", "itl_chunk_p99_ms", facet={"granularity": "chunk"})
    gaps.append({**base, "metric": "spec_accept_rate", "state": "not-run",
                 "reason": "the Anthropic stream reports no draft acceptance; the replay "
                           "cannot measure it"})

    for metric in TURN_METRICS:
        if values := [t[metric] for t in warm if t[metric] is not None]:
            measurements.append({**base, "metric": metric, "n": 1, "stat": _stat(values),
                                 "facet": {"pass": "warmup"}})
    return measurements, gaps, by_turn_index


def check_endpoint(up, endpoint, engine, *, ollama_version, context_length, model_name):
    """(ollama_version, context_length), once the endpoint is shown to be `engine`.

    Runs before the counter snapshot, because these requests may count.
    """
    found = probe_engine(up)
    if found is None or found["name"] != engine:
        raise SystemExit(f"--endpoint {endpoint} answers as {found and found['name']!r}, "
                         f"not {engine!r}")
    if engine == "ollama":
        if ollama_version and ollama_version != found["version"]:
            raise SystemExit(f"--ollama-version {ollama_version} but the endpoint reports "
                             f"{found['version']}")
        ollama_version = found["version"]
    if not ollama_version:
        raise SystemExit("--ollama-version is required: the ollama side's version")
    reported = served_context_length(endpoint, model_name)
    if context_length and reported and context_length != reported:
        raise SystemExit(f"--context-length {context_length} but /v1/models reports {reported}")
    if not (context_length or reported):
        raise SystemExit("/v1/models reports no max_model_len; pass --context-length")
    return ollama_version, context_length or reported


def replay_turns(up, turns, *, repeat, warmup, temperature, served_model, timeout_s,
                 reset, reset_name, resets_count, resets, log):
    """(samples, requests sent, void reason or None) for the passes over `turns`.

    With `warmup`, pass 0 runs first: a whole pass before the measured repeats
    1..`repeat`, so one-time work in a freshly started engine lands there and not
    in repeat 1 (#551). `reset`, when there is one, clears the prefix cache before
    every pass, the warm-up included, so each pass starts cold; how long each
    reset took is appended to `resets`. Stops at the first turn that voids the
    run: nothing after it is averaged in.
    """
    sent, samples = 0, []
    for r in range(0 if warmup else 1, repeat + 1):
        name = "warm-up" if r == 0 else f"repeat {r}"
        if reset:
            sent += resets_count
            try:
                resets.append({"repeat": r, "seconds": round(reset(), 3)})
            except ResetFailed as e:
                return samples, sent, f"{name}: cache reset ({reset_name}) failed: {e}"
            log(f"[r{r}] {reset_name}: {resets[-1]['seconds']}s")
        for i, t in enumerate(turns, 1):
            where = f"{name} turn {i} (seq {t['seq']})"
            sent += 1
            try:
                timing = send_turn(up, t, temperature=temperature, served_model=served_model,
                                   timeout_s=timeout_s)
            except TurnFailed as e:
                return samples, sent, f"{where}: {e}"
            if reset and i == 1 and timing["cache_read_input_tokens"]:
                return samples, sent, (
                    f"{where}: read {timing['cache_read_input_tokens']} prompt tokens from "
                    f"the prefix cache right after {reset_name} cleared it, so this pass "
                    "did not start cold")
            recorded_out = t["recorded"].get("output_tokens")
            cut = timing["stop_reason"] == "max_tokens"
            if cut and (recorded_out or 0) > timing["output_tokens"]:
                return samples, sent, (
                    f"{where}: stopped at max_tokens after {timing['output_tokens']} tokens "
                    f"where the recording produced {recorded_out}: the reply was cut short "
                    "(#161 fairness rule 3)")
            recorded_prompt = t["recorded"].get("prompt_tokens")
            short = prompt_shortfall(recorded_prompt, timing["prompt_tokens"])
            if short:
                return samples, sent, (
                    f"{where}: the engine counted a {timing['prompt_tokens']}-token prompt where "
                    f"the recording had {recorded_prompt}, {short} tokens short — more than the "
                    f"{PROMPT_SHORTFALL_TOKENS}-token / {PROMPT_SHORTFALL_FRACTION:.0%} "
                    "tolerance, so part of the prompt was dropped (#161 fairness rule 3)")
            # `repeat` 0 is the warm-up; `warmup` says so without knowing that.
            sample = {"repeat": r, "warmup": r == 0, "turn_index": i, "seq": t["seq"],
                      "turn_class": t["turn_class"], "in_flight": t["in_flight"],
                      "recorded_output_tokens": recorded_out,
                      "recorded_prompt_tokens": recorded_prompt, **timing}
            sample["e2e_star_ms"] = e2e_star(sample)
            samples.append(sample)
            log(f"[r{r} t{i:03d} seq {t['seq']}] ttft {timing['ttft_ms']:.0f}ms "
                f"e2e {timing['turn_e2e_ms']:.0f}ms out {timing['output_tokens']} "
                f"reprefill {timing['reprefill_ratio']}")
    return samples, sent, None


def build_provenance(manifest, corpus, turns, skipped, samples, *, machine, rung,
                     scratchy_sha, ollama_version, context_length, engine_provenance,
                     temperature, cache_reset, resets, warmup, served_model, disclosed):
    """The run's provenance block, every key the leaderboard requires.

    The turn counts below are over the measured repeats; the warm-up pass is
    reported on its own.
    """
    samples = [t for t in samples if not t["warmup"]]
    differs = [t for t in samples if None not in (t["prompt_tokens"], t["recorded_prompt_tokens"])
               and t["prompt_tokens"] != t["recorded_prompt_tokens"]]
    return {
        "machine": machine,
        "rung": rung,
        "scratchy_sha": scratchy_sha,
        "claude_version": manifest["agent_version"],
        "ollama_version": ollama_version,
        "context_length_tokens": context_length,
        **engine_provenance,
        "sampling": {"temperature": temperature},
        "trace_corpus": {
            "id": Path(corpus).resolve().name,
            "claude_version": manifest["agent_version"],
            "recorded_engine": manifest.get("engine"),
            "turns": len(turns),
            "skipped": sum(skipped.values()),
            "skipped_by_reason": skipped,
            # The only record of Claude Code's real concurrency; the replay
            # itself is serial (DISCLOSE_SERIAL).
            "turns_recorded_in_flight": sum(1 for t in turns if t["in_flight"]),
            "max_in_flight": max(t["in_flight"] for t in turns),
            "turn_classes": {tc: sum(1 for t in turns if t["turn_class"] == tc)
                             for tc in TURN_CLASSES[1:]},
            "max_recorded_prompt_tokens": max(t["recorded"].get("prompt_tokens") or 0
                                              for t in turns),
            # How this run replayed it. The board's provenance has no key of its
            # own for this, so it travels with the corpus it describes.
            "replay": {
                "concurrency": 1,
                # How the cache was cleared before every pass (None: it was not),
                # and how long each clear took: an ollama reload loads the
                # weights again, so a slow one shows up here, outside the turns.
                "cache_reset": cache_reset,
                "cache_reset_seconds": resets,
                # One pass before the measured repeats, kept out of every
                # headline cell; False when --no-warmup skipped it.
                "warmup": warmup,
                "served_model": served_model,
                "unstreamed_turns": sum(1 for t in samples if t["unstreamed"]),
                "max_tokens_turns": sum(1 for t in samples if t["stop_reason"] == "max_tokens"),
                "prompt_tokens_differ_turns": len(differs),
                "prompt_shortfall_tolerance": {"tokens": PROMPT_SHORTFALL_TOKENS,
                                               "fraction": PROMPT_SHORTFALL_FRACTION},
            },
        },
        "disclosed": disclosed,
    }


def replay(corpus, endpoint, *, engine, model, machine, rung, scratchy_sha, engine_provenance,
           repeat=3, temperature=0.0, ollama_version=None, context_length=None,
           served_model=None, server_info_url=None, reset_cache_url=None, timeout_s=600,
           warmup=True, run_id=None, log=lambda msg: print(msg, file=sys.stderr, flush=True)):
    """Replay `corpus` against `endpoint` and return (run, turns detail).

    One warm-up pass, unless `warmup` is False, then `repeat` measured passes,
    each after a prefix-cache reset where the engine allows one.
    `engine_provenance` maps each PER_ENGINE_PROVENANCE key to a per-engine dict.
    Configuration problems raise SystemExit before anything is sent. A run that
    cannot be trusted comes back with status "void" and a `void` reason.
    """
    if engine not in ENGINES:
        raise SystemExit(f"--engine must be one of {sorted(ENGINES)}, got {engine!r}")
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if not scratchy_sha:
        raise SystemExit("--scratchy-sha is required: the commit the scratchy side was built from")
    for key in PER_ENGINE_PROVENANCE:
        if sorted(engine_provenance.get(key) or {}) != sorted(ENGINES):
            raise SystemExit(f"{key} must name every engine in {sorted(ENGINES)}")
    up = _parse_url(endpoint, "--endpoint")
    manifest, turns, skipped = load_corpus(corpus)
    if not turns:
        raise SystemExit(f"{corpus} holds no replayable {TURN_PATH} turns ({skipped or 'empty'})")
    if manifest.get("agent") != "claude-code" or not manifest.get("agent_version"):
        raise SystemExit(f"{corpus}: manifest names agent {manifest.get('agent')!r} "
                         f"{manifest.get('agent_version')!r}, not a Claude Code version")
    classify(turns)
    model_name = served_model or turns[0]["body"].get("model")
    ollama_version, context_length = check_endpoint(
        up, endpoint, engine, ollama_version=ollama_version, context_length=context_length,
        model_name=model_name)

    reset_name, reset, reset_url = make_reset(engine, endpoint, reset_cache_url=reset_cache_url,
                                              model=model_name, timeout_s=timeout_s)
    info_url = server_info_url or endpoint.rstrip("/") + "/server_info"
    disclosed = [d for d, applies in (
        (DISCLOSE_KV, len(set(engine_provenance["kv_cache_dtype"].values())) > 1),
        (DISCLOSE_QUANT, len(set(engine_provenance["quantization"].values())) > 1),
        (DISCLOSE_SERIAL, any(t["in_flight"] for t in turns)),
        (DISCLOSE_NO_RESET, reset is None),
    ) if applies]

    void = None
    biggest = max(t["recorded"].get("prompt_tokens") or 0 for t in turns)
    if biggest > context_length:
        void = (f"the corpus's largest recorded prompt is {biggest} tokens and this engine's "
                f"context window is {context_length}: the session does not fit "
                "(#161 fairness rule 3, #322)")
    served_before = requests_served(info_url)
    if served_before is None:
        if ENGINES[engine]["counts_requests"]:
            void = void or (f"{info_url} reports no requests_served, so nothing proves this "
                            "endpoint served the replay (#161 fairness rule 7)")
        else:
            disclosed.append(DISCLOSE_NO_COUNTER)

    samples, sent, resets = [], 0, []
    if not void:
        # Requests one reset adds to `requests_served`: a POST to the counting
        # server is one. A reload goes to ollama, which has no counter to match.
        resets_count = int(bool(reset_url)
                           and urlsplit(reset_url).netloc == urlsplit(info_url).netloc)
        samples, sent, void = replay_turns(
            up, turns, repeat=repeat, warmup=warmup, temperature=temperature,
            served_model=served_model, timeout_s=timeout_s, reset=reset,
            reset_name=reset_name, resets_count=resets_count, resets=resets, log=log)
    served_after = requests_served(info_url) if served_before is not None else None
    if not void and served_before is not None:
        if served_after is None or served_after - served_before != sent:
            void = (f"requests_served went {served_before} -> {served_after} but this tool "
                    f"sent {sent}: the endpoint did not serve exactly this replay "
                    "(#161 fairness rule 7)")

    prov = build_provenance(
        manifest, corpus, turns, skipped, samples, machine=machine, rung=rung,
        scratchy_sha=scratchy_sha, ollama_version=ollama_version,
        context_length=context_length, engine_provenance=engine_provenance,
        temperature=temperature, cache_reset=reset_name, resets=resets, warmup=warmup,
        served_model=served_model, disclosed=disclosed)
    run = {"schema": SCHEMA, "run_id": run_id or f"laneA-{engine}-{model}-{machine}-{rung}",
           "lane": "A", "status": "published", "provenance": prov,
           "measurements": [], "gaps": [], "ablations": []}
    by_turn_index = {}
    if void:
        # Unpublishable on purpose: the validator rejects both the status and the key.
        run["status"] = "void"
        run["void"] = void
    else:
        run["measurements"], run["gaps"], by_turn_index = aggregate(
            samples, engine=engine, model=model, repeat=repeat)
    detail = {"requests_served_before": served_before, "requests_served_after": served_after,
              "sent": sent, "void": void, "by_turn_index": by_turn_index, "turns": samples}
    return run, detail


def cmd_replay(args):
    out = Path(args.out)
    run, detail = replay(
        args.corpus, args.endpoint, engine=args.engine, model=args.model,
        machine=args.machine, rung=args.rung, repeat=args.repeat,
        temperature=args.temperature, scratchy_sha=args.scratchy_sha,
        ollama_version=args.ollama_version, context_length=args.context_length,
        engine_provenance={k: parse_per_engine("--" + k.replace("_", "-"), getattr(args, k))
                           for k in PER_ENGINE_PROVENANCE},
        served_model=args.served_model, server_info_url=args.server_info_url,
        reset_cache_url=args.reset_cache_url, timeout_s=args.timeout_s,
        warmup=not args.no_warmup, run_id=args.run_id or out.stem)
    turns_path = out.with_name(out.stem + ".turns.json")
    run["provenance"]["artifacts"] = [turns_path.name]
    out.parent.mkdir(parents=True, exist_ok=True)
    turns_path.write_text(json.dumps(detail, indent=2) + "\n")
    out.write_text(json.dumps(run, indent=2) + "\n")
    tc = run["provenance"]["trace_corpus"]
    warm = "" if args.no_warmup else " after a warm-up pass"
    print(f"{tc['turns']} turns x {args.repeat} replayed{warm}, {tc['skipped']} requests "
          f"skipped, {tc['replay']['unstreamed_turns']} unstreamed; wrote {out} and {turns_path}",
          file=sys.stderr)
    if run["status"] == "void":
        raise SystemExit(f"VOID: {run['void']}")


# ------------------------------------------------------------------------ live

# Lane B (#164 T2.2). Where replay owns every latency claim, live owns every
# outcome claim: it runs the real agent on a graded task and reports whether the
# job got done. Its runs never carry a Lane A metric, and replay's never carry
# one of these.
#
# A task is a directory holding its own git repo (so it is copied fresh for every
# session), a TASK.md whose text is the prompt, and a grade.sh that prints PASS
# or FAIL. A grader the agent must not read lives outside the task and is passed
# with --grader; it is run with the session's working copy as its argument.
TASK_PROMPT = "TASK.md"
TASK_GRADER = "grade.sh"
# Lane B's minimum session count, from board.json's lanes.B.min_n.
LIVE_MIN_SESSIONS = 5
# Claude Code's own rejection of a tool call, as it phrases it in the tool
# result. Anything else marked is_error is the agent's command failing (a test
# that fails, a file that does not exist), which is the agent working.
TOOL_REJECTED = ("InputValidationError", "No such tool available")
# Tool-call markup left in a reply's text: the engine produced a call and its
# parser did not turn it into a tool_use block, so the agent never received it.
TOOL_MARKUP = ("<tool_call>", "<function=")
# Seconds between the end of one session and the start of the next (#161 rule 6).
LIVE_COOLDOWN_S = 30
# 80% interval on a pass rate, matching the p10-p90 band the page draws.
WILSON_Z = 1.2816
# Routes scratchy leaves out of requests_served (server.rs COUNTED_EXCLUDED_ROUTES).
UNCOUNTED_PATHS = frozenset({"/health", "/version", "/server_info", "/is_sleeping",
                             "/gpu_memory", "/metrics"})


def load_task(task):
    """(task dir, prompt) for a task directory. Raises SystemExit."""
    d = Path(task)
    if not (d / ".git").exists():
        raise SystemExit(f"{d}: not a task (no git repo, so it cannot be reset between sessions)")
    prompt = d / TASK_PROMPT
    if not prompt.is_file() or not prompt.read_text().strip():
        raise SystemExit(f"{d}: no {TASK_PROMPT} to use as the prompt")
    return d, prompt.read_text().strip()


def fresh_copy(task, dest):
    """A clean working copy of the task's committed state at `dest`: what the
    agent sees, with nothing from an earlier session."""
    dest = Path(dest)
    if dest.exists():
        raise SystemExit(f"{dest} already exists; every session needs a fresh copy")
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(Path(task).resolve()),
                    str(dest)], check=True)
    return dest


def grade(workdir, grader):
    """(passed, the grader's last line) for one session's working copy.

    The grader is the task's own grade.sh unless `grader` names another; it runs
    with the working copy as its argument and in it as its directory, and passes
    when it exits 0 and its last line is PASS.
    """
    cmd = [str(Path(grader).resolve())] if grader else [str(Path(workdir) / TASK_GRADER)]
    if not Path(cmd[0]).is_file():
        raise SystemExit(f"no grader at {cmd[0]}")
    p = subprocess.run(cmd + [str(workdir)], cwd=workdir, capture_output=True, text=True,
                       timeout=600)
    last = (p.stdout.strip().splitlines() or [""])[-1]
    return p.returncode == 0 and last == "PASS", last or f"exit {p.returncode}"


def session_stats(corpus):
    """Lane B facts read from one session's recording.

    The recorder keeps every request and response, so this needs nothing the
    session did not already write: turns, tokens, and tool calls, of which a
    failed one is a failure of the engine, never of the agent (see TOOL_REJECTED
    and TOOL_MARKUP).
    """
    d = Path(corpus)
    if not (d / "index.jsonl").exists():
        # Nothing reached the relay, so nothing was recorded.
        manifest = json.loads((d / "manifest.json").read_text())
        return {"claude_version": manifest.get("agent_version"), "turns": 0, "skipped": 0,
                "tool_calls": 0, "tool_calls_lost": 0, "tool_call_failures": 0,
                "tool_call_parse_failure_rate": 0.0, "tokens_in": 0, "tokens_out": 0,
                "wall_clock_s": 0.0}
    manifest, turns, skipped = load_corpus(corpus)
    rows = {}
    for line in (d / "index.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["seq"]] = row
    calls = broken = lost = tokens_in = tokens_out = 0
    for t in turns:
        raw = (d / "responses" / f"{t['seq']:06d}.body")
        data = raw.read_bytes() if raw.exists() else b""
        if (rows[t["seq"]].get("resp_content_encoding") or "").lower() == "gzip":
            data = gzip.decompress(data)
        args, text = {}, []
        try:
            for ev, _ in sse_events(data.splitlines(keepends=True) + [b"\n"]):
                kind = ev.get("type")
                if kind == "content_block_start":
                    block = ev.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        args[ev.get("index")] = ""
                elif kind == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "input_json_delta":
                        i = ev.get("index")
                        args[i] = args.get(i, "") + (delta.get("partial_json") or "")
                    elif delta.get("type") == "text_delta":
                        text.append(delta.get("text") or "")
        except TurnFailed:
            broken += 1  # an unreadable stream is a failure of whatever produced it
            continue
        calls += len(args)
        for a in args.values():
            if _json_or_none((a or "{}").encode()) is None:
                broken += 1
        reply = "".join(text)
        lost += reply.count(TOOL_MARKUP[0]) or int(TOOL_MARKUP[1] in reply)
        usage = t["recorded"]
        tokens_in += usage.get("prompt_tokens") or 0
        tokens_out += usage.get("output_tokens") or 0
    # Claude Code's rejections come back as tool results, so the last request's
    # history holds every one the session got.
    rejected = 0
    if turns:
        for m in turns[-1]["body"].get("messages") or []:
            for b in m.get("content") if isinstance(m.get("content"), list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                    c = b.get("content")
                    c = c if isinstance(c, str) else " ".join(
                        x.get("text", "") for x in c or [] if isinstance(x, dict))
                    rejected += any(k in c for k in TOOL_REJECTED)
    times = [r for r in rows.values() if r.get("t_end")]
    failures = broken + rejected + lost
    return {
        "claude_version": manifest.get("agent_version"),
        "turns": len(turns),
        "skipped": sum(skipped.values()),
        "tool_calls": calls,
        "tool_calls_lost": lost,
        "tool_call_failures": failures,
        "tool_call_parse_failure_rate": failures / (calls + lost) if calls + lost else 0.0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_clock_s": (max(r["t_end"] for r in times) - min(r["t_start"] for r in times))
                        if times else 0.0,
    }


def pass_rate_stat(passes):
    """Pass rate with an 80% Wilson interval as its p10-p90, clamped to contain it."""
    n, k = len(passes), sum(passes)
    p, z = k / n, WILSON_Z
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return {"median": round(p, 4), "p10": round(min(p, max(0.0, centre - half)), 4),
            "p90": round(max(p, min(1.0, centre + half)), 4), "samples": n}


def run_session(task, workdir, corpus, endpoint, *, scr, model, served_model, engine_sha,
                prompt, max_turns, timeout_s, env, log):
    """One live session: record it through an in-process relay, then grade it.

    Returns the session's record, with `void` set when it cannot be trusted.
    """
    server = make_server(("127.0.0.1", 0), endpoint, corpus, engine_sha=engine_sha)
    relay = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    info_url = endpoint.rstrip("/") + "/server_info"
    served_before = requests_served(info_url)
    cmd = [scr, "launch", "claude", "-m", served_model, "--server-url", relay, "--",
           "-p", prompt, "--max-turns", str(max_turns), "--permission-mode", "acceptEdits",
           "--allowedTools", "Read,Write,Edit,Bash"]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                           timeout=timeout_s)
        exit_code, timed_out, output = p.returncode, False, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        # text=True does not decode what a timeout leaves behind.
        exit_code, timed_out = None, True
        output = b"".join(x or b"" for x in (e.stdout, e.stderr)).decode("utf-8", "replace")
    finally:
        server.shutdown()
        server.server_close()
        server.corpus.drain(DRAIN_SECONDS)
        if server.corpus.manifest["engine"] is None:
            server.corpus.manifest["engine"] = probe_engine(server.upstream)
        server.corpus.close()
    seconds = time.monotonic() - t0
    served_after = requests_served(info_url)
    (Path(corpus) / "claude.log").write_text(output)
    passed, grader_said = grade(workdir, task["grader"])
    stats = session_stats(corpus)
    relayed = server.corpus.manifest["relayed"]
    void = None
    if stats["turns"] == 0:
        void = "no /v1/messages turn reached the engine: the agent never talked to it"
    elif served_before is not None and served_after is not None:
        # Every request the relay forwarded is one the engine should have counted,
        # less the introspection routes it never counts.
        index = Path(corpus) / "index.jsonl"
        expected = relayed - sum(
            1 for line in (index.read_text().splitlines() if index.exists() else [])
            if line.strip() and json.loads(line)["path"] in UNCOUNTED_PATHS)
        if served_after - served_before < expected:
            void = (f"the engine counted {served_after - served_before} requests but the relay "
                    f"forwarded {expected}: it did not serve this session (#161 fairness rule 7)")
    log(f"  pass={passed} ({grader_said}) turns={stats['turns']} "
        f"tool calls={stats['tool_calls']} failed={stats['tool_call_failures']} "
        f"{seconds:.0f}s exit={exit_code}{' TIMED OUT' if timed_out else ''}"
        + (f" VOID: {void}" if void else ""))
    return {"corpus": Path(corpus).name, "pass": passed, "grader": grader_said,
            "claude_exit": exit_code, "timed_out": timed_out, "void": void,
            "requests_served_before": served_before, "requests_served_after": served_after,
            **stats}


def live(task, endpoint, *, engine, model, machine, rung, scratchy_sha, engine_provenance,
         out_dir, sessions=LIVE_MIN_SESSIONS, scr="scr", served_model=None, grader=None,
         ollama_version=None, context_length=None, max_turns=30, timeout_s=1800,
         cooldown_s=LIVE_COOLDOWN_S, run_id=None, env=None,
         log=lambda msg: print(msg, file=sys.stderr, flush=True)):
    """Run `sessions` live Claude Code sessions of `task` against one engine.

    Each session gets a fresh copy of the task and the engine's prefix cache
    cleared, is recorded into `out_dir/session-N/`, and graded. Returns
    (run, detail): the run in the leaderboard's Lane B shape. Configuration
    problems raise SystemExit before any session starts.
    """
    if engine not in ENGINES:
        raise SystemExit(f"--engine must be one of {sorted(ENGINES)}, got {engine!r}")
    if sessions < 1:
        raise SystemExit("--sessions must be at least 1")
    if not scratchy_sha:
        raise SystemExit("--scratchy-sha is required: the commit the scratchy side was built from")
    for key in PER_ENGINE_PROVENANCE:
        if sorted(engine_provenance.get(key) or {}) != sorted(ENGINES):
            raise SystemExit(f"{key} must name every engine in {sorted(ENGINES)}")
    task_dir, prompt = load_task(task)
    if grader is None and not (task_dir / TASK_GRADER).is_file():
        raise SystemExit(f"{task_dir}: no {TASK_GRADER}, and no --grader given")
    scr_path = shutil.which(scr) or (scr if Path(scr).is_file() else None)
    if not scr_path:
        raise SystemExit(f"no scr binary at {scr!r}; build it with the `claude` feature")
    if not shutil.which("claude"):
        raise SystemExit("`claude` (Claude Code) is not on PATH")
    up = _parse_url(endpoint, "--endpoint")
    # `scr launch claude --server-url` starts Claude Code even when nothing
    # answers there, so find out here, before any session.
    served_model = served_model or model
    ollama_version, context_length = check_endpoint(
        up, endpoint, engine, ollama_version=ollama_version, context_length=context_length,
        model_name=served_model)
    reset_name, reset, _ = make_reset(engine, endpoint, model=served_model, timeout_s=timeout_s)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; every run needs its own --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Claude Code compacts at its own idea of the window unless told the
    # server's; and holds back 32000 tokens for its reply unless capped.
    session_env = dict(os.environ if env is None else env)
    session_env.setdefault("CLAUDE_CODE_MAX_CONTEXT_TOKENS", str(context_length))
    session_env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "8192")

    results, resets = [], []
    for i in range(1, sessions + 1):
        log(f"[session {i}/{sessions}]")
        if reset:
            try:
                resets.append({"session": i, "seconds": round(reset(), 3)})
            except ResetFailed as e:
                raise SystemExit(f"session {i}: cache reset ({reset_name}) failed: {e}") from None
        workdir = fresh_copy(task_dir, out_dir / f"session-{i}" / "workdir")
        results.append(run_session(
            {"grader": grader}, workdir, out_dir / f"session-{i}" / "corpus", endpoint,
            scr=scr_path, model=model, served_model=served_model, engine_sha=scratchy_sha,
            prompt=prompt, max_turns=max_turns, timeout_s=timeout_s, env=session_env, log=log))
        if i < sessions and cooldown_s:
            time.sleep(cooldown_s)

    versions = {r["claude_version"] for r in results if r["claude_version"]}
    disclosed = [d for d, applies in (
        (DISCLOSE_KV, len(set(engine_provenance["kv_cache_dtype"].values())) > 1),
        (DISCLOSE_QUANT, len(set(engine_provenance["quantization"].values())) > 1),
        (DISCLOSE_NO_COUNTER, not ENGINES[engine]["counts_requests"]),
    ) if applies]
    prov = {
        "machine": machine, "rung": rung, "scratchy_sha": scratchy_sha,
        "claude_version": sorted(versions)[0] if len(versions) == 1 else None,
        "ollama_version": ollama_version, "context_length_tokens": context_length,
        **engine_provenance,
        # Live runs cannot pin sampling: Claude Code sends its own.
        "sampling": {"temperature": "as sent by Claude Code"},
        "trace_corpus": {
            "id": task_dir.name,
            "claude_version": sorted(versions)[0] if len(versions) == 1 else None,
            "turns": sum(r["turns"] for r in results),
            "live": {"sessions": sessions, "max_turns": max_turns, "cooldown_s": cooldown_s,
                     "cache_reset": reset_name, "cache_reset_seconds": resets,
                     "served_model": served_model,
                     "grader": str(grader) if grader else f"{task_dir.name}/{TASK_GRADER}",
                     "task_commit": subprocess.run(
                         ["git", "-C", str(task_dir), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()},
        },
        "artifacts": [f"session-{i}" for i in range(1, sessions + 1)],
        "disclosed": disclosed,
    }
    run = {"schema": SCHEMA, "run_id": run_id or f"laneB-{engine}-{model}-{machine}-{rung}",
           "lane": "B", "status": "published", "provenance": prov,
           "measurements": [], "gaps": [], "ablations": []}
    void = next((f"session {i}: {r['void']}" for i, r in enumerate(results, 1) if r["void"]), None)
    if len(versions) > 1:
        void = void or (f"sessions ran under different Claude Code versions {sorted(versions)}: "
                        "a release changes the prompt and the tools, so they are not one run")
    if void:
        run["status"], run["void"] = "void", void
    else:
        base = {"lane": "B", "engine": engine, "model": model, "n": sessions}
        run["measurements"].append(
            {**base, "metric": "task_pass", "stat": pass_rate_stat([r["pass"] for r in results])})
        for metric, field in (("tool_call_parse_failure_rate", "tool_call_parse_failure_rate"),
                              ("turns_to_completion", "turns"), ("wall_clock_s", "wall_clock_s"),
                              ("tokens_in", "tokens_in"), ("tokens_out", "tokens_out")):
            run["measurements"].append(
                {**base, "metric": metric, "stat": _stat([r[field] for r in results])})
        gap = {"lane": "B", "engine": engine, "model": model, "state": "not-run",
               "reason": "energy needs powermetrics, which live does not sample yet",
               "issue": 164}
        run["gaps"] = [{**gap, "metric": m}
                       for m in ("joules_per_task", "joules_per_1k_output_tokens")]
    detail = {"sessions": results}
    return run, detail


def cmd_live(args):
    out = Path(args.out)
    run, detail = live(
        args.task, args.endpoint, engine=args.engine, model=args.model, machine=args.machine,
        rung=args.rung, scratchy_sha=args.scratchy_sha,
        engine_provenance={k: parse_per_engine("--" + k.replace("_", "-"), getattr(args, k))
                           for k in PER_ENGINE_PROVENANCE},
        out_dir=args.out_dir or out.with_name(out.stem + ".sessions"), sessions=args.sessions,
        scr=args.scr, served_model=args.served_model, grader=args.grader,
        ollama_version=args.ollama_version, context_length=args.context_length,
        max_turns=args.max_turns, timeout_s=args.timeout_s, cooldown_s=args.cooldown_s,
        run_id=args.run_id or out.stem)
    detail_path = out.with_name(out.stem + ".sessions.json")
    run["provenance"]["artifacts"] = [detail_path.name] + run["provenance"]["artifacts"]
    out.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(detail, indent=2) + "\n")
    out.write_text(json.dumps(run, indent=2) + "\n")
    s = detail["sessions"]
    print(f"{len(s)} sessions: {sum(r['pass'] for r in s)} passed, "
          f"{sum(r['tool_call_failures'] for r in s)} failed tool calls of "
          f"{sum(r['tool_calls'] for r in s)}; wrote {out} and {detail_path}", file=sys.stderr)
    if run["status"] == "void":
        raise SystemExit(f"VOID: {run['void']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="relay to an engine and keep every request and response")
    r.add_argument("--listen", default="127.0.0.1:8787", help="host:port to listen on")
    r.add_argument("--upstream", required=True, help="engine base URL, e.g. http://127.0.0.1:8000")
    r.add_argument("--out", required=True, help="new corpus directory")
    r.add_argument("--agent", help="agent name for the manifest (default: from the User-Agent)")
    r.add_argument("--engine-sha", help="commit the engine was built from (it cannot say itself)")
    r.add_argument("--upstream-timeout", type=float, default=1200,
                   help="seconds the engine may stay silent before the relay gives up")
    r.add_argument("--redact-header", action="append", default=[], metavar="NAME",
                   help="another header or query parameter to redact (repeatable)")
    r.set_defaults(func=cmd_record)

    a = sub.add_parser("replay", help="re-send a corpus's turns to one engine and time them")
    a.add_argument("--corpus", required=True, help="corpus directory written by `record`")
    a.add_argument("--endpoint", required=True, help="engine base URL, e.g. http://127.0.0.1:8000")
    a.add_argument("--engine", required=True, choices=sorted(ENGINES),
                   help="which engine the endpoint is; checked against what it answers as")
    a.add_argument("--model", required=True, help="leaderboard model id, e.g. gemma-4-12b-it")
    a.add_argument("--machine", required=True, help="leaderboard machine id, e.g. m5max-64")
    a.add_argument("--rung", required=True, help="leaderboard rung id, e.g. R1")
    a.add_argument("--repeat", type=int, default=3,
                   help="measured passes over the corpus (default 3)")
    a.add_argument("--no-warmup", action="store_true",
                   help="skip the warm-up pass before the measured ones (recorded in provenance)")
    a.add_argument("--out", required=True, help="Lane A run JSON to write")
    a.add_argument("--run-id", help="run_id (default: the --out file's stem)")
    a.add_argument("--temperature", type=float, default=0.0, help="pinned sampling (default 0.0)")
    a.add_argument("--served-model",
                   help="model name to send, and on ollama the model reloaded to clear its "
                        "cache (default: as recorded)")
    a.add_argument("--scratchy-sha", required=True, help="commit the scratchy side was built from")
    a.add_argument("--ollama-version",
                   help="the ollama side's version (default: asked of an ollama endpoint)")
    a.add_argument("--context-length", type=int,
                   help="context window in tokens, when GET /v1/models reports no max_model_len")
    for flag in ("--kv-cache-dtype", "--quantization", "--checkpoint-revision"):
        a.add_argument(flag, required=True, metavar="scratchy=X,ollama=Y",
                       help="per-engine provenance the tool cannot discover")
    a.add_argument("--server-info-url",
                   help="where requests_served is read (default: <endpoint>/server_info)")
    a.add_argument("--reset-cache-url",
                   help="POSTed before each pass to clear the prefix cache, in place of the "
                        "engine's own reset (scratchy: POST <endpoint>/reset_prefix_cache; "
                        "ollama: unload and reload the model). Neither can be turned off")
    a.add_argument("--timeout-s", type=float, default=600,
                   help="per-turn ceiling; a timeout voids the run (default 600)")
    a.set_defaults(func=cmd_replay)

    v = sub.add_parser("live", help="run Claude Code on a graded task, several times, and score it")
    v.add_argument("--task", required=True,
                   help=f"task directory: a git repo with {TASK_PROMPT} (the prompt) and, unless "
                        f"--grader is given, {TASK_GRADER}")
    v.add_argument("--endpoint", required=True, help="engine base URL, e.g. http://127.0.0.1:8000")
    v.add_argument("--engine", required=True, choices=sorted(ENGINES),
                   help="which engine the endpoint is; checked against what it answers as")
    v.add_argument("--model", required=True, help="leaderboard model id, e.g. qwen3.6-35b-a3b")
    v.add_argument("--served-model",
                   help="model name Claude Code asks for, e.g. mlx-community/Qwen3.6-35B-A3B-4bit "
                        "or qwen3.6:35b (default: --model)")
    v.add_argument("--machine", required=True, help="leaderboard machine id, e.g. m5max-64")
    v.add_argument("--rung", required=True, help="leaderboard rung id, e.g. R1")
    v.add_argument("--sessions", type=int, default=LIVE_MIN_SESSIONS,
                   help=f"live sessions to run (default {LIVE_MIN_SESSIONS}, the leaderboard's "
                        "minimum)")
    v.add_argument("--out", required=True, help="Lane B run JSON to write")
    v.add_argument("--out-dir",
                   help="where each session's working copy and recording go "
                        "(default: <out>.sessions/)")
    v.add_argument("--run-id", help="run_id (default: the --out file's stem)")
    v.add_argument("--grader",
                   help=f"grader to run instead of the task's own {TASK_GRADER}, for one the "
                        "agent must not see; called with the session's working copy")
    v.add_argument("--scr", default="scr", help="the scr binary that runs `launch claude`")
    v.add_argument("--max-turns", type=int, default=30, help="Claude Code's turn limit (default 30)")
    v.add_argument("--timeout-s", type=float, default=1800,
                   help="per-session ceiling; a timed-out session is graded as it stands "
                        "(default 1800)")
    v.add_argument("--cooldown-s", type=float, default=LIVE_COOLDOWN_S,
                   help=f"pause between sessions (default {LIVE_COOLDOWN_S})")
    v.add_argument("--scratchy-sha", required=True, help="commit the scratchy side was built from")
    v.add_argument("--ollama-version",
                   help="the ollama side's version (default: asked of an ollama endpoint)")
    v.add_argument("--context-length", type=int,
                   help="context window in tokens, when GET /v1/models reports no max_model_len")
    for flag in ("--kv-cache-dtype", "--quantization", "--checkpoint-revision"):
        v.add_argument(flag, required=True, metavar="scratchy=X,ollama=Y",
                       help="per-engine provenance the tool cannot discover")
    v.set_defaults(func=cmd_live)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
