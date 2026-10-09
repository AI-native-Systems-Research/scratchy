# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Tests for scripts/cc_trace.py. Standard library only:

    python3 -m unittest discover -s scripts/tests
"""

import http.client
import http.server
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cc_trace  # noqa: E402

SSE_EVENTS = [b"event: message_start\ndata: {}\n\n",
              b"event: content_block_delta\ndata: {\"x\":1}\n\n",
              b"event: message_stop\ndata: {}\n\n"]
EVENT_GAP = 0.3


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """Streams SSE slowly on POST /v1/messages; answers GET /server_info with JSON."""

    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeEngine.seen.append(("GET", self.path, dict(self.headers), b""))
        body = b'{"requests_served": 7}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeEngine.seen.append(("POST", self.path, dict(self.headers), body))
        if json.loads(body or b"{}").get("stream") is False:
            # Non-streaming: nothing goes out until generation is done.
            time.sleep(EVENT_GAP * 2)
            self.close_connection = True
            msg = b'{"type": "message"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        if self.path.startswith("/v1/messages/count_tokens"):
            msg = b'{"error": "not found"}'
            self.send_response(404)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for ev in SSE_EVENTS:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev))
            self.wfile.flush()
            time.sleep(EVENT_GAP)
        self.wfile.write(b"0\r\n\r\n")


def _serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


class RecordTest(unittest.TestCase):
    def setUp(self):
        FakeEngine.seen = []
        self.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        self.engine.daemon_threads = True
        engine_port = _serve(self.engine)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "corpus"
        self.proxy = cc_trace.make_server(("127.0.0.1", 0), f"http://127.0.0.1:{engine_port}", self.out)
        self.port = _serve(self.proxy)

    def tearDown(self):
        for s in (self.proxy, self.engine):
            s.shutdown()
            s.server_close()
        self.tmp.cleanup()

    def _post(self, body, path="/v1/messages"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        data = json.dumps(body).encode()
        conn.request("POST", path, body=data, headers={
            "Content-Type": "application/json", "x-api-key": "sk-secret",
            "anthropic-version": "2023-06-01", "anthropic-beta": "tools-x"})
        return conn, conn.getresponse(), data

    def _index(self):
        return [json.loads(line) for line in (self.out / "index.jsonl").read_text().splitlines()]

    def test_streams_incrementally_and_byte_identical(self):
        t0 = time.time()
        conn, resp, _ = self._post({"model": "m", "stream": True, "messages": []})
        first = resp.read1(65536)
        t_first = time.time() - t0
        rest = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        # The first event must arrive before the engine has finished streaming.
        self.assertLess(t_first, EVENT_GAP * (len(SSE_EVENTS) - 1))
        self.assertEqual(first + rest, b"".join(SSE_EVENTS))

    def test_request_saved_verbatim_with_auth_redacted(self):
        body = {"model": "m", "stream": True, "temperature": 1, "tools": [{"name": "Bash"}],
                "messages": [{"role": "user", "content": "hi"}]}
        conn, resp, sent = self._post(body)
        resp.read()
        conn.close()
        # Forwarded unchanged, auth included: the relay only redacts its own copy.
        _, path, fwd_headers, fwd_body = FakeEngine.seen[0]
        self.assertEqual((path, fwd_body), ("/v1/messages", sent))
        self.assertEqual(fwd_headers["x-api-key"], "sk-secret")
        self.assertEqual(fwd_headers["anthropic-beta"], "tools-x")
        rec = json.loads((self.out / "requests" / "000001.json").read_text())
        self.assertEqual(rec["body"], body)
        self.assertEqual(rec["headers"]["x-api-key"], "<redacted>")
        self.assertEqual(rec["headers"]["anthropic-version"], "2023-06-01")
        [row] = self._index()
        self.assertEqual((row["seq"], row["status"], row["stream"], row["in_flight"]),
                         (1, 200, True, 0))
        self.assertEqual(row["req_bytes"], len(sent))
        self.assertEqual(row["resp_bytes"], len(b"".join(SSE_EVENTS)))
        self.assertGreaterEqual(row["t_end"], row["t_start"])

    def test_concurrent_requests_get_distinct_seq_and_in_flight(self):
        results = []

        def one(i):
            conn, resp, _ = self._post({"model": "m", "stream": True, "messages": [i]})
            results.append(resp.read())
            conn.close()

        threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.05)
        for t in threads:
            t.join()
        rows = sorted(self._index(), key=lambda r: r["seq"])
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3])
        self.assertEqual([r["in_flight"] for r in rows], [0, 1, 2])
        self.assertEqual(len(list((self.out / "requests").iterdir())), 3)
        self.assertTrue(all(r == b"".join(SSE_EVENTS) for r in results))

    def test_get_and_error_status_pass_through(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("GET", "/server_info")
        resp = conn.getresponse()
        self.assertEqual((resp.status, json.loads(resp.read())), (200, {"requests_served": 7}))
        # Same keep-alive connection, an upstream 404 comes back as-is.
        conn.request("POST", "/v1/messages/count_tokens?beta=true", body=b'{"model":"m"}',
                     headers={"Content-Length": "13"})
        resp = conn.getresponse()
        self.assertEqual((resp.status, resp.read()), (404, b'{"error": "not found"}'))
        conn.close()
        rows = self._index()
        self.assertEqual([(r["method"], r["path"], r["status"]) for r in rows],
                         [("GET", "/server_info", 200),
                          ("POST", "/v1/messages/count_tokens", 404)])
        # A bodiless GET gets an index row but no request file.
        self.assertEqual(sorted(p.name for p in (self.out / "requests").iterdir()), ["000002.json"])
        rec = json.loads((self.out / "requests" / "000002.json").read_text())
        self.assertEqual(rec["query"], "beta=true")

    def test_manifest_and_refuses_a_used_out(self):
        m = json.loads((self.out / "manifest.json").read_text())
        for key in ("format", "upstream", "listen", "started", "stopped",
                    "claude_version", "ollama_version", "scratchy_sha", "requests"):
            self.assertIn(key, m)
        conn, resp, _ = self._post({"model": "m", "messages": []})
        resp.read()
        conn.close()
        self.proxy.corpus.close()
        m = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(m["requests"], 1)
        self.assertIsNotNone(m["stopped"])
        with self.assertRaises(SystemExit):
            cc_trace.make_server(("127.0.0.1", 0), "http://127.0.0.1:1", self.out)

    def test_client_gone_before_headers_is_recorded(self):
        # Claude Code times out on a slow non-streaming request and hangs up
        # before the upstream has sent a single header.
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        data = b'{"model": "m", "stream": false}'
        conn.request("POST", "/v1/messages", body=data, headers={"Content-Length": str(len(data))})
        conn.sock.shutdown(2)
        conn.close()
        deadline = time.time() + 5
        while not (self.out / "index.jsonl").exists() and time.time() < deadline:
            time.sleep(0.05)
        [row] = self._index()
        self.assertEqual((row["status"], row["stream"]), (200, False))
        self.assertTrue(row["error"].startswith("client disconnected"), row["error"])

    def test_unreachable_upstream_is_502(self):
        with tempfile.TemporaryDirectory() as d:
            proxy = cc_trace.make_server(("127.0.0.1", 0), "http://127.0.0.1:1", Path(d) / "c")
            port = _serve(proxy)
            conn = http.client.HTTPConnection("127.0.0.1", port)
            conn.request("POST", "/v1/messages", body=b"{}", headers={"Content-Length": "2"})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            proxy.shutdown()
            proxy.server_close()
            self.assertEqual(resp.status, 502)
            [row] = [json.loads(x) for x in (Path(d) / "c" / "index.jsonl").read_text().splitlines()]
            self.assertTrue(row["error"].startswith("upstream:"))


if __name__ == "__main__":
    unittest.main()
