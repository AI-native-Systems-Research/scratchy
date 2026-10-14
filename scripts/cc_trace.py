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
stored files, by whatever reads the corpus. Standard library only.
"""

import argparse
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


def _parse_upstream(url):
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"--upstream must be an http(s) URL, got {url!r}")
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
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
