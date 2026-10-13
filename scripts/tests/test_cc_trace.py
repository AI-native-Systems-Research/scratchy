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
UA = "claude-cli/2.1.295 (external, sdk-cli)"


class FakeEngine(http.server.BaseHTTPRequestHandler):
    """A scratchy-shaped engine: slow SSE on POST /v1/messages, JSON on GET /server_info.

    A body with `"hold": true` waits on `release` before answering, so a test can
    keep several requests open at once without relying on timing.
    """

    protocol_version = "HTTP/1.1"
    seen = []
    release = threading.Event()

    def log_message(self, *a):
        pass

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeEngine.seen.append(("GET", self.path, self.headers.items(), b""))
        if self.path == "/server_info":
            self._send(200, b'{"requests_served": 7, "system_env": {"vllm_version": "0.1.0"}}')
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeEngine.seen.append(("POST", self.path, self.headers.items(), body))
        fields = json.loads(body or b"{}")
        if self.path == "/bad-status":
            self.wfile.write(b"NOT HTTP AT ALL\r\n\r\n")
            self.close_connection = True
            return
        if fields.get("hold"):
            FakeEngine.release.wait(10)
        if fields.get("stream") is False:
            # Non-streaming: nothing goes out until generation is done.
            time.sleep(EVENT_GAP * 2)
            self.close_connection = True
            self._send(200, b'{"type": "message"}')
            return
        if self.path.startswith("/v1/messages/count_tokens"):
            self._send(404, b'{"error": "not found"}')
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
        FakeEngine.release = threading.Event()
        self.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
        self.engine.daemon_threads = True
        engine_port = _serve(self.engine)
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "corpus"
        self.proxy = cc_trace.make_server(("127.0.0.1", 0), f"http://127.0.0.1:{engine_port}",
                                          self.out, engine_sha="abc123", redact=["X-Custom-Pass"])
        self.port = _serve(self.proxy)
        FakeEngine.seen = []  # drop make_server's engine probe

    def tearDown(self):
        FakeEngine.release.set()
        for s in (self.proxy, self.engine):
            s.shutdown()
            s.server_close()
        self.tmp.cleanup()

    def _post(self, body, path="/v1/messages", extra=()):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        data = json.dumps(body).encode()
        conn.putrequest("POST", path, skip_accept_encoding=True)
        for k, v in [("Content-Type", "application/json"), ("Content-Length", str(len(data))),
                     ("User-Agent", UA), ("x-api-key", "sk-secret"),
                     ("anthropic-version", "2023-06-01"), *extra]:
            conn.putheader(k, v)
        conn.endheaders(data)
        return conn, conn.getresponse(), data

    def _index(self):
        return [json.loads(line) for line in (self.out / "index.jsonl").read_text().splitlines()]

    def _wait_rows(self, n):
        deadline = time.time() + 10
        while time.time() < deadline:
            if (self.out / "index.jsonl").exists() and len(self._index()) >= n:
                return self._index()
            time.sleep(0.05)
        self.fail(f"fewer than {n} index rows")

    def test_streams_incrementally_and_saves_the_response(self):
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
        [row] = self._wait_rows(1)
        self.assertEqual(row["resp_content_type"], "text/event-stream")
        self.assertEqual((self.out / "responses" / "000001.body").read_bytes(), b"".join(SSE_EVENTS))

    def test_request_saved_verbatim_with_secrets_redacted(self):
        body = {"model": "m", "stream": True, "temperature": 1, "tools": [{"name": "Bash"}],
                "messages": [{"role": "user", "content": "hi"}]}
        extra = [("Cookie", "session=1"), ("x-goog-api-key", "g-secret"),
                 ("X-Custom-Pass", "hunter2"), ("X-Repeat", "one"), ("X-Repeat", "two")]
        conn, resp, sent = self._post(body, path="/v1/messages?beta=true&key=q-secret", extra=extra)
        resp.read()
        conn.close()
        # Forwarded unchanged, secrets and repeats included: only the saved copy is redacted.
        _, path, fwd_headers, fwd_body = FakeEngine.seen[0]
        self.assertEqual((path, fwd_body), ("/v1/messages?beta=true&key=q-secret", sent))
        self.assertIn(("x-api-key", "sk-secret"), fwd_headers)
        self.assertEqual([v for k, v in fwd_headers if k == "X-Repeat"], ["one", "two"])
        self.assertNotIn("Accept-Encoding", dict(fwd_headers))
        rec = json.loads((self.out / "requests" / "000001.json").read_text())
        self.assertEqual(rec["body"], body)
        self.assertEqual(rec["query"], "beta=true&key=%3Credacted%3E")
        saved = rec["headers"]
        for name in ("x-api-key", "Cookie", "x-goog-api-key", "X-Custom-Pass"):
            self.assertIn([name, "<redacted>"], saved)
        self.assertIn(["anthropic-version", "2023-06-01"], saved)
        self.assertEqual([v for k, v in saved if k == "X-Repeat"], ["one", "two"])
        [row] = self._wait_rows(1)
        self.assertEqual((row["seq"], row["status"], row["stream"], row["in_flight"]),
                         (1, 200, True, 0))
        self.assertEqual(row["req_bytes"], len(sent))
        self.assertEqual(row["resp_bytes"], len(b"".join(SSE_EVENTS)))
        self.assertGreaterEqual(row["t_end"], row["t_start"])

    def test_concurrent_requests_get_distinct_seq_and_in_flight(self):
        results = []

        def one(i):
            conn, resp, _ = self._post({"model": "m", "stream": True, "hold": True, "messages": [i]})
            results.append(resp.read())
            conn.close()

        threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        # All three are open at the engine at once before any of them answers.
        deadline = time.time() + 10
        while len(FakeEngine.seen) < 3 and time.time() < deadline:
            time.sleep(0.02)
        FakeEngine.release.set()
        for t in threads:
            t.join()
        rows = self._wait_rows(3)
        self.assertEqual(sorted(r["seq"] for r in rows), [1, 2, 3])
        self.assertEqual(sorted(r["in_flight"] for r in rows), [0, 1, 2])
        self.assertEqual(len(list((self.out / "requests").iterdir())), 3)
        self.assertTrue(all(r == b"".join(SSE_EVENTS) for r in results))

    def test_get_and_error_status_pass_through(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("GET", "/server_info")
        resp = conn.getresponse()
        self.assertEqual((resp.status, json.loads(resp.read())["requests_served"]), (200, 7))
        # Same keep-alive connection, an upstream 404 comes back as-is.
        conn.request("POST", "/v1/messages/count_tokens?beta=true", body=b'{"model":"m"}',
                     headers={"Content-Length": "13"})
        resp = conn.getresponse()
        self.assertEqual((resp.status, resp.read()), (404, b'{"error": "not found"}'))
        conn.close()
        rows = self._wait_rows(2)
        self.assertEqual([(r["method"], r["path"], r["status"]) for r in rows],
                         [("GET", "/server_info", 200),
                          ("POST", "/v1/messages/count_tokens", 404)])
        # A bodiless GET gets an index row and a saved response, but no request file.
        self.assertEqual(sorted(p.name for p in (self.out / "requests").iterdir()), ["000002.json"])
        self.assertEqual(sorted(p.name for p in (self.out / "responses").iterdir()),
                         ["000001.body", "000002.body"])

    def test_manifest_names_agent_engine_and_counts(self):
        m = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(m["engine"], {"name": "scratchy", "version": "0.1.0"})
        self.assertEqual(m["engine_sha"], "abc123")
        self.assertIn("script_sha", m)
        self.assertIsNone(m["agent"])
        # `scr launch` checks /server_info through the relay with its own client
        # before Claude Code sends anything; that must not name the agent.
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request("GET", "/server_info", headers={"User-Agent": "ureq/3.4.2"})
        conn.getresponse().read()
        conn.close()
        conn, resp, _ = self._post({"model": "gemma", "messages": []})
        resp.read()
        conn.close()
        self._wait_rows(2)
        self.proxy.corpus.close()
        m = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual((m["agent"], m["agent_version"], m["user_agent"]),
                         ("claude-code", "2.1.295", UA))
        self.assertEqual(m["models"], ["gemma"])
        self.assertEqual((m["relayed"], m["saved"], m["unfinished"]), (2, 1, []))
        self.assertIsNotNone(m["stopped"])
        with self.assertRaises(SystemExit):
            cc_trace.make_server(("127.0.0.1", 0), "http://127.0.0.1:1", self.out)

    def test_close_waits_for_open_relays_and_lists_the_rest(self):
        def held():
            conn, resp, _ = self._post({"model": "m", "hold": True})
            resp.read()
            conn.close()

        t = threading.Thread(target=held)
        t.start()
        deadline = time.time() + 10
        while not FakeEngine.seen and time.time() < deadline:
            time.sleep(0.02)
        self.assertFalse(self.proxy.corpus.drain(0.1))
        self.proxy.corpus.close()
        m = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(m["unfinished"], [1])
        FakeEngine.release.set()
        self.assertTrue(self.proxy.corpus.drain(10))
        t.join()

    def test_client_gone_before_headers_is_recorded(self):
        # Claude Code times out on a slow non-streaming request and hangs up
        # before the upstream has sent a single header.
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        data = b'{"model": "m", "stream": false}'
        conn.request("POST", "/v1/messages", body=data, headers={"Content-Length": str(len(data))})
        conn.sock.shutdown(2)
        conn.close()
        [row] = self._wait_rows(1)
        self.assertEqual((row["status"], row["stream"]), (200, False))
        self.assertTrue(row["error"].startswith("client disconnected"), row["error"])

    def test_bad_request_does_not_leak_in_flight(self):
        self.proxy.handle_error = lambda *a: None  # the traceback is expected here
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.putrequest("POST", "/v1/messages")
        conn.putheader("Content-Length", "not-a-number")
        conn.endheaders()
        with self.assertRaises((http.client.HTTPException, OSError)):
            conn.getresponse()
        conn.close()
        [row] = self._wait_rows(1)
        self.assertTrue(row["error"].startswith("relay: ValueError"), row["error"])
        conn, resp, _ = self._post({"model": "m", "messages": []})
        resp.read()
        conn.close()
        self.assertEqual(self._wait_rows(2)[1]["in_flight"], 0)

    def test_malformed_upstream_reply_is_502(self):
        conn, resp, _ = self._post({"model": "m"}, path="/bad-status")
        self.assertEqual(resp.status, 502)
        resp.read()
        conn.close()
        [row] = self._wait_rows(1)
        self.assertTrue(row["error"].startswith("upstream: BadStatusLine"), row["error"])

    def test_unreachable_upstream_is_502(self):
        with tempfile.TemporaryDirectory() as d:
            proxy = cc_trace.make_server(("127.0.0.1", 0), "http://127.0.0.1:1", Path(d) / "c")
            self.assertIsNone(proxy.corpus.manifest["engine"])
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


class HelperTest(unittest.TestCase):
    def test_parse_agent(self):
        self.assertEqual(cc_trace.parse_agent(UA), ("claude-code", "2.1.295"))
        self.assertEqual(cc_trace.parse_agent("opencode/0.15.3 ai-sdk/5"), ("opencode", "0.15.3"))
        self.assertEqual(cc_trace.parse_agent(None), (None, None))

    def test_parse_listen(self):
        self.assertEqual(cc_trace.parse_listen("127.0.0.1:8787"), ("127.0.0.1", 8787))
        self.assertEqual(cc_trace.parse_listen("[::1]:8787"), ("::1", 8787))
        self.assertEqual(cc_trace.parse_listen(":8787"), ("127.0.0.1", 8787))

    def test_redact_query_leaves_clean_queries_alone(self):
        self.assertEqual(cc_trace.redact_query("beta=true"), "beta=true")
        self.assertEqual(cc_trace.redact_query("api_key=x&b=1"), "api_key=%3Credacted%3E&b=1")


if __name__ == "__main__":
    unittest.main()
