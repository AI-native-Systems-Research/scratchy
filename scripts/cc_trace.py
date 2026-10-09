#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Record Claude Code traffic for the Lane A replay benchmark (epic #158).

`record` is an HTTP relay between `claude` and whichever engine is serving. It
keeps a copy of every request and passes every response back untouched. Only
requests are kept: Claude Code re-sends the whole conversation each turn, so
turn N's request already holds turn N-1's response.

    scripts/cc_trace.py record --listen 127.0.0.1:8787 \\
        --upstream http://127.0.0.1:8000 \\
        --out bench_results/claude-code/corpus/gemma12b-01/
    scr launch claude -m gemma-4-12b-it --server-url http://127.0.0.1:8787

The corpus is unredacted (prompts, file contents, paths) and stays on the
measuring machine; `bench_results/` is gitignored. Layout:

    manifest.json         versions, upstream, start/stop, request count
    index.jsonl           one row per request, appended when its response ends
    requests/NNNNNN.json  method, path, headers (auth redacted), body

The relay rewrites nothing and classifies nothing: which requests are agent
turns is decided later, from the stored bodies, by whatever reads the corpus.
Standard library only.
"""

import argparse
import http.client
import http.server
import json
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

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
SECRET_HEADERS = {"authorization", "x-api-key"}
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")


def _version(argv):
    """First line of `argv`'s output, or None when the tool is missing or fails."""
    if shutil.which(argv[0]) is None:
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or out.stderr).strip().splitlines()
    return line[0] if out.returncode == 0 and line else None


def _scratchy_sha():
    return _version(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"])


class Corpus:
    """One recording session's output directory. Thread-safe."""

    def __init__(self, out, upstream, listen):
        self.dir = Path(out)
        if (self.dir / "manifest.json").exists() or (self.dir / "index.jsonl").exists():
            raise SystemExit(f"{self.dir} already holds a corpus; pick a new --out")
        (self.dir / "requests").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._open = 0
        self.manifest = {
            "format": FORMAT,
            "upstream": upstream,
            "listen": listen,
            "started": time.time(),
            "stopped": None,
            "claude_version": _version(["claude", "--version"]),
            "ollama_version": _version(["ollama", "--version"]),
            "scratchy_sha": _scratchy_sha(),
            "requests": 0,
        }
        self._write_manifest()

    def _write_manifest(self):
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, indent=2) + "\n")
        tmp.replace(self.dir / "manifest.json")

    def begin(self):
        """Allocate a sequence number; returns (seq, requests already in flight)."""
        with self._lock:
            self._seq += 1
            in_flight = self._open
            self._open += 1
            return self._seq, in_flight

    def save_request(self, seq, method, path, query, headers, body, t_start):
        record = {"seq": seq, "method": method, "path": path, "query": query,
                  "headers": headers, "t_start": t_start}
        try:
            record["body"] = json.loads(body)
        except ValueError:
            record["body_text"] = body.decode("utf-8", "replace")
        (self.dir / "requests" / f"{seq:06d}.json").write_text(json.dumps(record) + "\n")

    def end(self, row):
        with self._lock:
            self._open -= 1
            self.manifest["requests"] += 1
            with open(self.dir / "index.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

    def close(self):
        with self._lock:
            self.manifest["stopped"] = time.time()
            self._write_manifest()


class RelayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    # Set per server by make_server.
    corpus: Corpus
    upstream: tuple

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
        body = self._read_body()
        path, _, query = self.path.partition("?")
        stream = None
        if body:
            saved = {k: ("<redacted>" if k.lower() in SECRET_HEADERS else v)
                     for k, v in self.headers.items()}
            self.corpus.save_request(seq, self.command, path, query, saved, body, t_start)
            try:
                parsed = json.loads(body)
                stream = parsed.get("stream") if isinstance(parsed, dict) else None
            except ValueError:
                pass
        row = {"seq": seq, "method": self.command, "path": path, "status": None,
               "t_start": t_start, "t_end": None, "in_flight": in_flight,
               "req_bytes": len(body), "resp_bytes": 0, "stream": stream, "error": None}
        try:
            self._forward(body, row)
        except (BrokenPipeError, ConnectionResetError) as e:
            # The client gave up before the headers went out: a non-streaming
            # request whose upstream only answers once generation is done.
            row["error"] = f"client disconnected: {e}"
            self.close_connection = True
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

    def _forward(self, body, row):
        scheme, host, port, base = self.upstream
        conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(host, port)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, base + self.path, body=body or None, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            row["status"], row["error"] = 502, f"upstream: {e}"
            msg = f"cc_trace: upstream unreachable: {e}\n".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            conn.close()
            return
        row["status"] = resp.status
        if self._client_gone():
            # A write to a half-closed socket can still succeed, so ask instead.
            row["error"] = "client disconnected before the response"
            self.close_connection = True
            conn.close()
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
        try:
            while not no_body:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                row["resp_bytes"] += len(chunk)
                if chunked:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else:
                    self.wfile.write(chunk)
                self.wfile.flush()
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as e:
            row["error"] = f"client disconnected: {e}"
            self.close_connection = True
        except (OSError, http.client.HTTPException) as e:
            # The status line is already sent; all that is left is to cut the stream.
            row["error"] = f"upstream mid-stream: {e}"
            self.close_connection = True
        finally:
            conn.close()


for _m in METHODS:
    setattr(RelayHandler, f"do_{_m}", RelayHandler._relay)


def _parse_upstream(url):
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"--upstream must be an http(s) URL, got {url!r}")
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/")


def make_server(listen, upstream, out):
    """Build (not start) a recording relay. `listen` is (host, port); port 0 picks one."""
    corpus = Corpus(out, upstream, f"{listen[0]}:{listen[1]}")
    handler = type("Handler", (RelayHandler,), {"corpus": corpus, "upstream": _parse_upstream(upstream)})
    server = http.server.ThreadingHTTPServer(listen, handler)
    server.daemon_threads = True
    server.corpus = corpus
    return server


def cmd_record(args):
    host, _, port = args.listen.rpartition(":")
    server = make_server((host or "127.0.0.1", int(port)), args.upstream, args.out)
    m = server.corpus.manifest
    print(f"recording {args.listen} -> {args.upstream} into {args.out}\n"
          f"  claude {m['claude_version']}, ollama {m['ollama_version']}, "
          f"scratchy {m['scratchy_sha']}\n  Ctrl-C to stop", file=sys.stderr, flush=True)
    # SIGTERM finalizes the manifest the same way Ctrl-C does.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.corpus.close()
        print(f"stopped: {server.corpus.manifest['requests']} requests recorded",
              file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="relay to an engine and keep every request")
    r.add_argument("--listen", default="127.0.0.1:8787", help="host:port to listen on")
    r.add_argument("--upstream", required=True, help="engine base URL, e.g. http://127.0.0.1:8000")
    r.add_argument("--out", required=True, help="new corpus directory")
    r.set_defaults(func=cmd_record)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
