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
job, not this tool's. A run that cannot be trusted — the engine's request
counter disagrees with what was sent, a turn returned no tokens, timed out or
answered 4xx — is written with `status: "void"` and the reason, and exits
non-zero. Standard library only.
"""

import argparse
import gzip
import hashlib
import http.client
import http.server
import json
import re
import select
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
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    # ollama answers /api/version; scratchy answers /server_info.
    for path, read in (("/api/version", lambda j: ("ollama", j.get("version"))),
                       ("/server_info", lambda j: ("scratchy", j["system_env"]["vllm_version"]))):
        conn = conn_cls(host, port, timeout=timeout)
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
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(host, port, timeout=self.upstream_timeout)
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


def _parse_upstream(url, flag="--upstream"):
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"{flag} must be an http(s) URL, got {url!r}")
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/")


class _Server6(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_server(listen, upstream, out, *, agent=None, engine_sha=None,
                upstream_timeout=1200, redact=()):
    """Build (not start) a recording relay. `listen` is (host, port); port 0 picks one."""
    up = _parse_upstream(upstream)
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
# The engines a leaderboard run carries per-engine provenance for, and whether
# each one's server reports `requests_served` (#161 fairness rule 7). Data, not
# a code path: the replay is the same for both.
ENGINES = {"scratchy": {"counts_requests": True}, "ollama": {"counts_requests": False}}
# `site/build_leaderboard.py`'s TURN_CLASSES; "all" is the headline cell itself.
TURN_CLASSES = ("all", "tool-call", "file-write")
# Claude Code writes files through these tools, so a turn that calls one is a
# file-write rather than a tool call. A text-only reply this long is a file
# written inline.
FILE_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
LONG_TEXT_CHARS = 2000
# Delta fields that carry generated content. A delta with all of them empty
# (or a `signature_delta`) is framing, not a first token.
DELTA_CONTENT = ("text", "partial_json", "thinking")
# Request headers the replay does not resend: framing, and compression (the
# stream is timed as plain bytes). Redacted secrets are dropped by value.
NOT_REPLAYED = HOP_BY_HOP | {"accept-encoding"}
# Disclosure ids, declared in site/data/leaderboard/board.json.
DISCLOSE_NO_COUNTER = "requests-served-unavailable"
DISCLOSE_KV = "kv-dtype-unmatched"
DISCLOSE_QUANT = "quantizer-differs"


class TurnFailed(Exception):
    """A turn the run cannot average in: it voids the run."""


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


def sse_events(lines, clock=time.monotonic):
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
                yield payload, clock()
            event, data = None, []


def time_stream(lines, t_sent, clock=time.monotonic):
    """One turn's timings from its Anthropic SSE stream. Raises TurnFailed."""
    usage, deltas, t_stop = {}, [], None
    for ev, t in sse_events(lines, clock):
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
        elif kind == "error":
            raise TurnFailed(f"error event: {ev.get('error')}")
        elif kind == "message_stop":
            t_stop = t
            break
    if t_stop is None:
        raise TurnFailed("stream ended without message_stop")
    out = usage.get("output_tokens") or 0
    if out <= 0 or not deltas:
        # A dead worker behind a live HTTP layer looks exactly like this (#413).
        raise TurnFailed(f"zero output tokens (output_tokens={out}, {len(deltas)} content deltas)")
    gaps = [(b - a) * 1000 for a, b in zip(deltas, deltas[1:])]
    return {
        "ttft_ms": (deltas[0] - t_sent) * 1000,
        "turn_e2e_ms": (t_stop - t_sent) * 1000,
        "tpot_ms": (deltas[-1] - deltas[0]) * 1000 / (out - 1) if out > 1 else None,
        # Per chunk: Anthropic deltas carry text, not token counts (#208).
        "itl_chunk_p99_ms": percentile(gaps, 99) if gaps else None,
        "output_tokens": out,
        "input_tokens": usage.get("input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "reprefill_ratio": reprefill_ratio(usage.get("input_tokens"),
                                           usage.get("cache_read_input_tokens"),
                                           usage.get("cache_creation_input_tokens")),
    }


def recorded_output_tokens(corpus, row):
    """Output tokens of the response the recorder saved for `row`, or None.

    The engine-independent output length E2E* normalizes against.
    """
    path = Path(corpus) / "responses" / f"{row['seq']:06d}.body"
    if not path.exists():
        return None
    data = path.read_bytes()
    if (row.get("resp_content_encoding") or "").lower() == "gzip":
        data = gzip.decompress(data)
    out = None
    try:
        for ev, _ in sse_events(data.splitlines(keepends=True) + [b"\n"]):
            out = (ev.get("usage") or {}).get("output_tokens", out)
    except TurnFailed:
        return None
    return out or None


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
                          "recorded_output_tokens": recorded_output_tokens(d, row)})
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
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
    tools = [b.get("name") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
    if any(t in FILE_WRITE_TOOLS for t in tools):
        return "file-write"
    if tools:
        return "tool-call"
    text = sum(len(b.get("text") or "") for b in blocks
               if isinstance(b, dict) and b.get("type") == "text")
    return "file-write" if text >= LONG_TEXT_CHARS else None


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
    scheme, host, port, base = _parse_upstream(url, "URL")
    path = base or "/"
    if urlsplit(url).query:
        path += "?" + urlsplit(url).query
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
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


def send_turn(endpoint, turn, *, temperature, served_model=None, timeout_s=600):
    """Replay one turn and time it. Raises TurnFailed."""
    scheme, host, port, base = endpoint
    body = dict(turn["body"])
    # The only edits: pinned sampling, and a stream to time. max_tokens, the
    # messages, the tools and cache_control are the work; they stay as recorded.
    body["temperature"] = temperature
    body["stream"] = True
    if served_model:
        body["model"] = served_model
    data = json.dumps(body).encode()
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout_s)
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
    return {"median": round(percentile(values, 50), 4), "p10": round(percentile(values, 10), 4),
            "p90": round(percentile(values, 90), 4), "p99": round(percentile(values, 99), 4)}


def _e2e_star(t):
    # median_ttft + (output_len - 1) * median_tpot, per turn, with the output
    # length the *recording* produced: an engine that stops early gains nothing.
    out = t["recorded_output_tokens"]
    if out is None:
        return None
    if out == 1:
        return t["ttft_ms"]
    return None if t["tpot_ms"] is None else t["ttft_ms"] + (out - 1) * t["tpot_ms"]


def aggregate(samples, *, engine, model, repeat):
    """(measurements, gaps) from every replayed turn of every repeat."""
    base = {"lane": "A", "engine": engine, "model": model}
    headline = {
        "ttft_ms": lambda t: t["ttft_ms"],
        "tpot_ms": lambda t: t["tpot_ms"],
        "turn_e2e_ms": lambda t: t["turn_e2e_ms"],
        "reprefill_ratio": lambda t: t["reprefill_ratio"],
        "e2e_star_ms": _e2e_star,
    }
    measurements, gaps = [], []

    def cell(metric, pick, facet=None):
        rows = [(t["turn_index"], v) for t in samples if (v := pick(t)) is not None]
        if not rows:
            return False
        m = {**base, "metric": metric, "n": repeat, "stat": _stat([v for _, v in rows])}
        if facet:
            m["facet"] = facet
        else:
            by_turn = {}
            for i, v in rows:
                by_turn.setdefault(i, []).append(v)
            m["series"] = {"x": "turn_index", "points": [
                [i, round(percentile(vs, 50), 4)] for i, vs in sorted(by_turn.items())]}
        measurements.append(m)
        return True

    for metric, pick in headline.items():
        if not cell(metric, pick):
            gaps.append({**base, "metric": metric, "state": "not-run",
                         "reason": f"no replayed turn yielded a {metric} value"
                                   + (" (the corpus saved no response to take the output "
                                      "length from)" if metric == "e2e_star_ms" else "")})
        for tc in TURN_CLASSES[1:]:
            cell(metric, lambda t, pick=pick, tc=tc: pick(t) if t["turn_class"] == tc else None,
                 facet={"turn_class": tc})
    # ITL: only a per-chunk number exists on the wire, so the headline is a gap
    # and the chunk figure sits beside it under its own facet.
    gaps.append({**base, "metric": "itl_p99_ms", "state": "blocked", "issue": 208,
                 "reason": "Anthropic SSE deltas carry text, not token counts, so per-token "
                           "ITL is not on the wire; see the granularity=chunk facet"})
    cell("itl_p99_ms", lambda t: t["itl_chunk_p99_ms"], facet={"granularity": "chunk"})
    gaps.append({**base, "metric": "spec_accept_rate", "state": "not-run",
                 "reason": "the Anthropic stream reports no draft acceptance; the replay "
                           "cannot measure it"})
    return measurements, gaps


def replay(corpus, endpoint, *, engine, model, machine, rung, scratchy_sha,
           kv_cache_dtype, quantization, checkpoint_revision, repeat=3, temperature=0.0,
           ollama_version=None, context_length=None, served_model=None,
           server_info_url=None, reset_cache_url=None, timeout_s=600, run_id=None,
           log=lambda msg: print(msg, file=sys.stderr, flush=True)):
    """Replay `corpus` against `endpoint` and return (run, turns detail).

    Configuration problems raise SystemExit before anything is sent. A run that
    cannot be trusted comes back with status "void" and a `void` reason.
    """
    if engine not in ENGINES:
        raise SystemExit(f"--engine must be one of {sorted(ENGINES)}, got {engine!r}")
    if repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    up = _parse_upstream(endpoint, "--endpoint")
    manifest, turns, skipped = load_corpus(corpus)
    if not turns:
        raise SystemExit(f"{corpus} holds no replayable {TURN_PATH} turns ({skipped or 'empty'})")
    classify(turns)
    if manifest.get("agent") != "claude-code" or not manifest.get("agent_version"):
        raise SystemExit(f"{corpus}: manifest names agent {manifest.get('agent')!r} "
                         f"{manifest.get('agent_version')!r}, not a Claude Code version")
    if not scratchy_sha:
        raise SystemExit("--scratchy-sha is required: the commit the scratchy side was built from")

    # Everything this tool asks before the counter snapshot may itself count.
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
    reported = served_context_length(endpoint, served_model or turns[0]["body"].get("model"))
    if context_length and reported and context_length != reported:
        raise SystemExit(f"--context-length {context_length} but /v1/models reports {reported}")
    context_length = context_length or reported
    if not context_length:
        raise SystemExit("/v1/models reports no max_model_len; pass --context-length")

    disclosed = []
    if len(set(kv_cache_dtype.values())) > 1:
        disclosed.append(DISCLOSE_KV)
    if len(set(quantization.values())) > 1:
        disclosed.append(DISCLOSE_QUANT)
    info_url = server_info_url or endpoint.rstrip("/") + "/server_info"
    counts = ENGINES[engine]["counts_requests"]
    void = None
    served_before = requests_served(info_url)
    if served_before is None:
        if counts:
            void = (f"{info_url} reports no requests_served, so nothing proves this "
                    "endpoint served the replay (#161 fairness rule 7)")
        else:
            disclosed.append(DISCLOSE_NO_COUNTER)
    resets_count = reset_cache_url and urlsplit(reset_cache_url).netloc == urlsplit(info_url).netloc

    sent, samples = 0, []
    for r in range(1, repeat + 1):
        if void:
            break
        if reset_cache_url:
            try:
                status, _ = _http("POST", reset_cache_url, body=b"{}")
            except (OSError, http.client.HTTPException) as e:
                status = f"{type(e).__name__}: {e}"
            sent += 1 if resets_count else 0
            if status not in (200, 204):
                void = f"--reset-cache-url {reset_cache_url} answered {status}"
                break
        for i, t in enumerate(turns, 1):
            sent += 1
            try:
                timing = send_turn(up, t, temperature=temperature, served_model=served_model,
                                   timeout_s=timeout_s)
            except TurnFailed as e:
                void = f"repeat {r} turn {i} (seq {t['seq']}): {e}"
                break
            samples.append({"repeat": r, "turn_index": i, "seq": t["seq"],
                            "turn_class": t["turn_class"], "in_flight": t["in_flight"],
                            "recorded_output_tokens": t["recorded_output_tokens"], **timing})
            log(f"[{engine} r{r} t{i:03d} seq {t['seq']}] ttft {timing['ttft_ms']:.0f}ms "
                f"e2e {timing['turn_e2e_ms']:.0f}ms out {timing['output_tokens']} "
                f"reprefill {timing['reprefill_ratio']}")

    served_after = requests_served(info_url) if served_before is not None else None
    if not void and served_before is not None:
        if served_after is None or served_after - served_before != sent:
            void = (f"requests_served went {served_before} -> {served_after} but this tool "
                    f"sent {sent}: the endpoint did not serve exactly this replay "
                    "(#161 fairness rule 7)")

    prov = {
        "machine": machine,
        "rung": rung,
        "scratchy_sha": scratchy_sha,
        "claude_version": manifest["agent_version"],
        "ollama_version": ollama_version,
        "context_length_tokens": context_length,
        "kv_cache_dtype": kv_cache_dtype,
        "quantization": quantization,
        "checkpoint_revision": checkpoint_revision,
        "sampling": {"temperature": temperature},
        "trace_corpus": {
            "id": Path(corpus).resolve().name,
            "claude_version": manifest["agent_version"],
            "recorded_engine": manifest.get("engine"),
            "turns": len(turns),
            "skipped": sum(skipped.values()),
            "skipped_by_reason": skipped,
            # The only record of Claude Code's real concurrency; the replay
            # itself is serial.
            "turns_recorded_in_flight": sum(1 for t in turns if t["in_flight"]),
            "max_in_flight": max(t["in_flight"] for t in turns),
            "turn_classes": {tc: sum(1 for t in turns if t["turn_class"] == tc)
                             for tc in TURN_CLASSES[1:]},
        },
        "disclosed": disclosed,
    }
    run = {"schema": SCHEMA, "run_id": run_id or f"laneA-{engine}-{model}-{machine}-{rung}",
           "lane": "A", "status": "published", "provenance": prov,
           "measurements": [], "gaps": [], "ablations": []}
    if void:
        # Unpublishable on purpose: the validator rejects both the status and the key.
        run["status"] = "void"
        run["void"] = void
    else:
        run["measurements"], run["gaps"] = aggregate(samples, engine=engine, model=model,
                                                     repeat=repeat)
    detail = {"requests_served_before": served_before, "requests_served_after": served_after,
              "sent": sent, "void": void, "turns": samples}
    return run, detail


def cmd_replay(args):
    out = Path(args.out)
    run, detail = replay(
        args.corpus, args.endpoint, engine=args.engine, model=args.model,
        machine=args.machine, rung=args.rung, repeat=args.repeat,
        temperature=args.temperature, scratchy_sha=args.scratchy_sha,
        ollama_version=args.ollama_version, context_length=args.context_length,
        kv_cache_dtype=parse_per_engine("--kv-cache-dtype", args.kv_cache_dtype),
        quantization=parse_per_engine("--quantization", args.quantization),
        checkpoint_revision=parse_per_engine("--checkpoint-revision", args.checkpoint_revision),
        served_model=args.served_model, server_info_url=args.server_info_url,
        reset_cache_url=args.reset_cache_url, timeout_s=args.timeout_s,
        run_id=args.run_id or out.stem)
    turns_path = out.with_name(out.stem + ".turns.json")
    run["provenance"]["artifacts"] = [turns_path.name]
    out.parent.mkdir(parents=True, exist_ok=True)
    turns_path.write_text(json.dumps(detail, indent=2) + "\n")
    out.write_text(json.dumps(run, indent=2) + "\n")
    tc = run["provenance"]["trace_corpus"]
    print(f"{tc['turns']} turns x {args.repeat} replayed, {tc['skipped']} requests skipped; "
          f"wrote {out} and {turns_path}", file=sys.stderr)
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
    a.add_argument("--repeat", type=int, default=3, help="passes over the corpus (default 3)")
    a.add_argument("--out", required=True, help="Lane A run JSON to write")
    a.add_argument("--run-id", help="run_id (default: the --out file's stem)")
    a.add_argument("--temperature", type=float, default=0.0, help="pinned sampling (default 0.0)")
    a.add_argument("--served-model", help="model name to send (default: as recorded)")
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
                   help="POSTed before each repeat, e.g. <endpoint>/reset_prefix_cache "
                        "(a sanity control)")
    a.add_argument("--timeout-s", type=float, default=600,
                   help="per-turn ceiling; a timeout voids the run (default 600)")
    a.set_defaults(func=cmd_replay)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
