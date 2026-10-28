# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Tests for scripts/cc_trace.py. Standard library only:

    python3 -m unittest discover -s scripts/tests
"""

import http.client
import http.server
import importlib.util
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


REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "site" / "build_leaderboard.py"
BOARD = REPO / "site" / "data" / "leaderboard" / "board.json"
TTFT_GAP = 0.25


def _ev(kind, **fields):
    data = json.dumps({"type": kind, **fields}).encode()
    return b"event: %s\ndata: %s\n\n" % (kind.encode(), data)


def _stream(output_tokens=3, input_tokens=100, cache_read=0, stop_reason="end_turn"):
    """(delay before, event) pairs for one Anthropic SSE reply."""
    usage = {"input_tokens": input_tokens, "cache_read_input_tokens": cache_read}
    return [
        (0, _ev("message_start", message={"usage": {**usage, "output_tokens": 1}})),
        (0, _ev("content_block_start", index=0, content_block={"type": "text", "text": ""})),
        (0, _ev("ping")),
        # Framing with no content in it: not a first token.
        (TTFT_GAP, _ev("content_block_delta", index=0, delta={"type": "text_delta", "text": ""})),
        (TTFT_GAP, _ev("content_block_delta", index=0, delta={"type": "text_delta", "text": "Hi"})),
        (0.05, _ev("content_block_delta", index=0, delta={"type": "text_delta", "text": " more"})),
        (0, _ev("content_block_stop", index=0)),
        (0, _ev("message_delta", delta={"stop_reason": stop_reason},
                usage={**usage, "output_tokens": output_tokens})),
        (0, _ev("message_stop")),
    ]


def _one_delta(output_tokens):
    """A reply the engine buffered: every token arrives in a single delta."""
    return [(0, _ev("message_start", message={"usage": {"input_tokens": 100}})),
            (TTFT_GAP, _ev("content_block_delta", index=0,
                           delta={"type": "text_delta", "text": "all at once"})),
            (0, _ev("message_delta", delta={"stop_reason": "end_turn"},
                    usage={"input_tokens": 100, "output_tokens": output_tokens})),
            (0, _ev("message_stop"))]


class FakeAnthropic(http.server.BaseHTTPRequestHandler):
    """A scratchy-shaped /v1/messages engine that counts requests like scratchy does.

    Class attributes steer it: `reply(body)` returns the stream (or an HTTP
    status to fail with), and `counting = False` leaves /v1/messages out of
    `requests_served` — what a replay that reached some other server looks like
    from here.
    """

    protocol_version = "HTTP/1.1"
    served = 0
    resets = 0
    counting = True
    seen = []
    reply = staticmethod(lambda body: _stream())

    def log_message(self, *a):
        pass

    def _send(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # State is read and written through type(self), so a subclass keeps its own.

    def do_GET(self):
        cls = type(self)
        if self.path == "/server_info":
            self._send(200, {"requests_served": cls.served,
                             "system_env": {"vllm_version": "0.1.0"}})
            return
        cls.served += 1
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "gemma", "max_model_len": 32768}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if cls.counting or not self.path.startswith("/v1/messages"):
            cls.served += 1
        if self.path == "/reset_prefix_cache":
            cls.resets += 1
            self._send(200, {})
            return
        if not self.path.startswith("/v1/messages"):
            self._send(404, {"error": "not found"})
            return
        cls.seen.append((self.path, self.headers.items(), body))
        reply = cls.reply(body)
        if isinstance(reply, int):
            self._send(reply, {"type": "error", "error": {"message": "prompt is too long"}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for delay, ev in reply:
            time.sleep(delay)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(ev), ev))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


class FakeOllama(FakeAnthropic):
    """An ollama-shaped engine: no request counter, no reset endpoint, and a
    prefix cache that lives as long as the loaded model.

    It remembers every prompt it has served since the model was last loaded, and
    serves a prompt it has seen as a cache hit. Unloading the model
    (`keep_alive: 0` on /api/generate) forgets them, which is the only way to
    clear it. `reloads` counts loads; `loadable` lists the models it knows.
    """

    loaded = set()
    loadable = {"gemma"}
    cached = set()
    reloads = 0
    hits = []

    def do_GET(self):
        if self.path == "/api/version":
            self._send(200, {"version": "0.40.4"})
        elif self.path == "/api/ps":
            self._send(200, {"models": [{"name": m, "model": m} for m in sorted(self.loaded)]})
        elif self.path == "/v1/models":
            # ollama reports no window; the replay is told it by --context-length.
            self._send(200, {"data": [{"id": "gemma"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/generate":
            return super().do_POST()
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        model = body.get("model")
        if model not in self.loadable:
            self._send(404, {"error": f"model '{model}' not found"})
            return
        if body.get("keep_alive") == 0:
            FakeOllama.loaded.discard(model)
            FakeOllama.cached.clear()
        else:
            FakeOllama.loaded.add(model)
            FakeOllama.reloads += 1
        self._send(200, {"model": model, "done": True})


def _ollama_reply(body):
    """The stream FakeOllama sends: a cache hit when it has served this prompt before."""
    key = json.dumps(body["messages"], sort_keys=True)
    hit = key in FakeOllama.cached
    FakeOllama.cached.add(key)
    FakeOllama.hits.append(hit)
    return _stream(input_tokens=1 if hit else 100, cache_read=99 if hit else 0)


def _user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(*blocks):
    return {"role": "assistant", "content": list(blocks)}


def _tool(name, tid):
    return {"type": "tool_use", "id": tid, "name": name, "input": {}}


def _result(tid):
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}


def _cached(messages):
    """`messages` with a cache_control breakpoint on the last block, as Claude Code sends."""
    out = json.loads(json.dumps(messages))
    out[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return out


def conversation():
    """Request bodies for a 4-turn session with an unrelated side request second.

    After turn 1 the assistant called Bash, after turn 2 Write, after turn 3 it
    answered in short text; nothing records what turn 4 produced.
    """
    m1 = [_user("fix the bug")]
    m2 = m1 + [_assistant(_tool("Bash", "t1")), _result("t1")]
    m3 = m2 + [_assistant({"type": "text", "text": "writing"}, _tool("Write", "t2")), _result("t2")]
    m4 = m3 + [_assistant({"type": "text", "text": "done"}), _user("thanks")]
    side = [_user("summarize this conversation in 5 words")]
    return [{"model": "gemma", "max_tokens": 32000, "temperature": 1, "stream": True,
             "tools": [{"name": "Bash"}], "messages": _cached(m)} for m in (m1, side, m2, m3, m4)]


def write_corpus(root, recorded_output_tokens=5, recorded_input_tokens=100, concurrent=True,
                 unsaved=()):
    """A corpus as `record` lays it out, with a GET and a count_tokens call mixed in.

    With `concurrent`, the side request (seq 4) was recorded while another was open.
    The turns whose seq is in `unsaved` have no saved response, so nothing records
    their lengths.
    """
    root = Path(root)
    (root / "requests").mkdir(parents=True)
    (root / "responses").mkdir()
    (root / "manifest.json").write_text(json.dumps(
        {"format": 1, "agent": "claude-code", "agent_version": "2.1.295",
         "engine": {"name": "scratchy", "version": "0.1.0"}, "engine_sha": "abc123"}))
    bodies = conversation()
    plan = [("POST", "/v1/messages", bodies[0]), ("GET", "/v1/models", None),
            ("POST", "/v1/messages/count_tokens", {"messages": []})]
    plan += [("POST", "/v1/messages", b) for b in bodies[1:]]
    recorded = b"".join(ev for _, ev in _stream(output_tokens=recorded_output_tokens,
                                                input_tokens=recorded_input_tokens))
    rows = []
    for seq, (method, path, body) in enumerate(plan, 1):
        if body is not None:
            (root / "requests" / f"{seq:06d}.json").write_text(json.dumps(
                {"seq": seq, "method": method, "path": path, "query": "beta=true",
                 "headers": [["x-api-key", "<redacted>"], ["anthropic-version", "2023-06-01"],
                             ["Accept-Encoding", "gzip"], ["Content-Type", "application/json"]],
                 "t_start": 0.0, "body": body}))
        if path == "/v1/messages" and seq not in unsaved:
            (root / "responses" / f"{seq:06d}.body").write_bytes(recorded)
        rows.append({"seq": seq, "method": method, "path": path, "status": 200, "t_start": 0.0,
                     "t_end": 0.0, "in_flight": int(concurrent and seq == 4), "req_bytes": 0,
                     "resp_bytes": 0, "stream": True, "error": None})
    (root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def board_problems(run):
    """What site/build_leaderboard.py's validator (#176) says about one run.

    Coverage of the other cells is the whole board's concern, not one run's, so
    only problems naming this run count.
    """
    sys.path.insert(0, str(VALIDATOR.parent))  # it imports its sibling charts.py
    spec = importlib.util.spec_from_file_location("build_leaderboard", VALIDATOR)
    lb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lb)
    board = json.loads(BOARD.read_text())
    board["runs"] = [run]
    lb.index(board)
    return [p for p in lb.validate(board) if p.startswith(run["run_id"])]


PER_ENGINE = {
    "kv_cache_dtype": {"scratchy": "bf16", "ollama": "f16"},
    "quantization": {"scratchy": "mlx-affine-b4-g64", "ollama": "Q4_K_M"},
    "checkpoint_revision": {"scratchy": "mlx-community/gemma@0000000",
                            "ollama": "gemma@sha256:0000000"},
}


class ReplayTest(unittest.TestCase):
    def setUp(self):
        FakeAnthropic.served = 0
        FakeAnthropic.resets = 0
        FakeAnthropic.counting = True
        FakeAnthropic.seen = []
        FakeAnthropic.reply = staticmethod(lambda body: _stream())
        self.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeAnthropic)
        self.engine.daemon_threads = True
        self.url = f"http://127.0.0.1:{_serve(self.engine)}"
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = write_corpus(Path(self.tmp.name) / "corpus")

    def tearDown(self):
        self.engine.shutdown()
        self.engine.server_close()
        self.tmp.cleanup()

    def _replay(self, corpus=None, **kw):
        # No warm-up unless a test asks: most tests are about one measured pass,
        # and the warm-up has tests of its own.
        args = dict(engine="scratchy", model="gemma-4-12b-it", machine="m5max-64", rung="R1",
                    repeat=1, warmup=False, scratchy_sha="abc123", ollama_version="0.40.0",
                    engine_provenance=PER_ENGINE, run_id="test-laneA", log=lambda msg: None)
        args.update(kw)
        return cc_trace.replay(corpus or self.corpus, self.url, **args)

    def _corpus(self, name, **kw):
        return write_corpus(Path(self.tmp.name) / name, **kw)

    def _meas(self, run, metric, facet=None):
        return [m for m in run["measurements"] if m["metric"] == metric and m.get("facet") == facet]

    def test_replays_turns_in_seq_order_and_counts_the_rest(self):
        run, detail = self._replay(repeat=2)
        self.assertEqual(run["status"], "published", run.get("void"))
        tc = run["provenance"]["trace_corpus"]
        self.assertEqual((tc["turns"], tc["skipped"], tc["skipped_by_reason"]),
                         (5, 2, {"not a turn": 2}))
        self.assertEqual((tc["turns_recorded_in_flight"], tc["max_in_flight"]), (1, 1))
        self.assertEqual([t["seq"] for t in detail["turns"]], [1, 4, 5, 6, 7] * 2)
        # scratchy's prefix cache is cleared before every repeat, without being asked.
        self.assertEqual(FakeAnthropic.resets, 2)
        self.assertEqual((tc["replay"]["cache_reset"], tc["replay"]["concurrency"]),
                         (f"POST {self.url}/reset_prefix_cache", 1))
        self.assertEqual([r["repeat"] for r in tc["replay"]["cache_reset_seconds"]], [1, 2])
        self.assertIs(tc["replay"]["warmup"], False)
        # 10 turns plus 2 resets, every one counted by the engine.
        self.assertEqual((detail["sent"], len(FakeAnthropic.seen)), (12, 10))
        # Recorded with two requests open at once, replayed one at a time: disclosed.
        self.assertIn(cc_trace.DISCLOSE_SERIAL, run["provenance"]["disclosed"])
        self.assertNotIn(cc_trace.DISCLOSE_NO_RESET, run["provenance"]["disclosed"])
        path, headers, first = FakeAnthropic.seen[0]
        self.assertEqual(path, "/v1/messages?beta=true")
        names = {k.lower() for k, _ in headers}
        self.assertNotIn("x-api-key", names)
        self.assertNotIn("accept-encoding", names)
        self.assertIn("anthropic-version", names)
        # Only temperature and stream change; the work is sent as recorded.
        self.assertEqual(first, dict(conversation()[0], temperature=0.0, stream=True))
        # A stat is over turns x repeats; `n` is the repeats.
        [ttft] = self._meas(run, "ttft_ms")
        self.assertEqual((ttft["n"], ttft["stat"]["samples"]), (2, 10))
        p50s = {r["turn_index"]: r["p50"] for r in detail["by_turn_index"]["ttft_ms"]}
        self.assertEqual(ttft["series"]["points"], [[i, v] for i, v in p50s.items()])
        self.assertEqual(set(detail["by_turn_index"]["ttft_ms"][0]),
                         {"turn_index", "p50", "p99", "samples"})

    def test_ttft_is_the_first_delta_with_content(self):
        run, detail = self._replay()
        t = detail["turns"][0]
        # message_start and an empty delta came earlier; neither counts.
        self.assertGreaterEqual(t["ttft_ms"], 2 * TTFT_GAP * 1000)
        self.assertLess(t["ttft_ms"], t["turn_e2e_ms"])
        # Two content deltas 50 ms apart over 3 output tokens.
        self.assertAlmostEqual(t["tpot_ms"], 25, delta=20)
        [itl] = self._meas(run, "itl_p99_ms", {"granularity": "chunk"})
        self.assertEqual(itl["n"], 1)
        self.assertEqual(self._meas(run, "itl_p99_ms"), [])
        self.assertEqual({g["metric"]: g["state"] for g in run["gaps"]},
                         {"itl_p99_ms": "blocked", "spec_accept_rate": "not-run"})

    def test_e2e_star_is_computed_per_turn(self):
        run, detail = self._replay(repeat=2)
        turns = detail["turns"]
        # Each turn's own TTFT and TPOT, at the length the recording produced
        # (5 tokens), not the replay's (3).
        for t in turns:
            self.assertAlmostEqual(t["e2e_star_ms"], t["ttft_ms"] + 4 * t["tpot_ms"])
        # The stat is over turns x repeats, like every other per-turn metric.
        [e2e] = self._meas(run, "e2e_star_ms")
        values = [t["e2e_star_ms"] for t in turns]
        self.assertEqual((e2e["n"], e2e["stat"]["samples"]), (2, 10))
        self.assertAlmostEqual(e2e["stat"]["median"], cc_trace.percentile(values, 50), places=3)
        self.assertAlmostEqual(e2e["stat"]["p99"], cc_trace.percentile(values, 99), places=3)
        first = [t["e2e_star_ms"] for t in turns if t["turn_index"] == 1]
        self.assertAlmostEqual(e2e["series"]["points"][0][1], cc_trace.percentile(first, 50),
                               places=3)
        # A recorded length of 1 is just the TTFT; no recorded reply, no E2E*.
        t = dict(turns[0], recorded_output_tokens=1)
        self.assertEqual(cc_trace.e2e_star(t), t["ttft_ms"])
        self.assertIsNone(cc_trace.e2e_star(dict(turns[0], recorded_output_tokens=None)))

    def test_a_reply_in_one_delta_has_no_tpot(self):
        FakeAnthropic.reply = staticmethod(lambda body: _one_delta(output_tokens=40))
        run, detail = self._replay()
        self.assertEqual(run["status"], "published", run.get("void"))
        self.assertTrue(all(t["unstreamed"] and t["tpot_ms"] is None for t in detail["turns"]))
        self.assertEqual(run["provenance"]["trace_corpus"]["replay"]["unstreamed_turns"], 5)
        # Neither TPOT nor E2E* can be made up from it; TTFT is still what the user waited.
        self.assertEqual(self._meas(run, "tpot_ms") + self._meas(run, "e2e_star_ms"), [])
        gaps = {g["metric"]: g["reason"] for g in run["gaps"]}
        self.assertEqual(gaps["tpot_ms"], cc_trace.TURN_METRICS["tpot_ms"])
        self.assertEqual(gaps["e2e_star_ms"], cc_trace.TURN_METRICS["e2e_star_ms"])
        self.assertEqual(len(self._meas(run, "ttft_ms")), 1)

    def test_reprefill_ratio(self):
        self.assertEqual(cc_trace.reprefill_ratio(100, 0), 1.0)
        self.assertEqual(cc_trace.reprefill_ratio(1, 166), 1 / 167)
        self.assertEqual(cc_trace.reprefill_ratio(10, 80, 10), 0.2)
        self.assertIsNone(cc_trace.reprefill_ratio(0, 0))
        # Every prompt is the recording's 100 tokens. The two one-message requests
        # have nothing to reuse; the other three reuse 75 of them.
        FakeAnthropic.reply = staticmethod(lambda body: _stream(
            **({"input_tokens": 25, "cache_read": 75} if len(body["messages"]) > 1
               else {"input_tokens": 100})))
        run, detail = self._replay()
        self.assertEqual([t["reprefill_ratio"] for t in detail["turns"]],
                         [1.0, 1.0, 0.25, 0.25, 0.25])
        [m] = self._meas(run, "reprefill_ratio")
        self.assertEqual(m["stat"]["median"], 0.25)

    def test_turn_class_comes_from_the_next_turn_not_the_reply(self):
        def tool_reply(body):
            # Whatever this engine says, the class must not follow it.
            ev = _ev("content_block_delta", index=0,
                     delta={"type": "input_json_delta", "partial_json": "{}"})
            return [(0, _ev("message_start", message={"usage": {"input_tokens": 100}})),
                    (0, ev), (0, ev),
                    (0, _ev("message_delta", usage={"output_tokens": 2})), (0, _ev("message_stop"))]

        for reply in (lambda body: _stream(), tool_reply):
            FakeAnthropic.reply = staticmethod(reply)
            run, detail = self._replay()
            self.assertEqual([t["turn_class"] for t in detail["turns"]],
                             ["tool-call", None, "file-write", None, None])
            self.assertEqual(run["provenance"]["trace_corpus"]["turn_classes"],
                             {"tool-call": 1, "file-write": 1})
            self.assertEqual(len(self._meas(run, "ttft_ms", {"turn_class": "tool-call"})), 1)
        # Prose is not a file write, however long.
        self.assertIsNone(cc_trace.reply_class({"role": "assistant", "content": "x" * 10000}))

    def test_counter_disagreeing_with_what_was_sent_voids_the_run(self):
        FakeAnthropic.counting = False
        run, detail = self._replay()
        self.assertEqual(run["status"], "void")
        self.assertIn("requests_served", run["void"])
        self.assertEqual(run["measurements"], [])
        # 5 turns plus the reset, which the engine did count.
        self.assertEqual(detail["sent"], 6)

    def test_empty_turns_and_4xx_void_the_run(self):
        cases = {
            "zero output tokens": lambda body: [
                (0, _ev("message_start", message={"usage": {"input_tokens": 9}})),
                (0, _ev("message_delta", usage={"output_tokens": 0})), (0, _ev("message_stop"))],
            # Tokens reported, nothing delivered: the empty tool call of #498.
            "no content: the engine reported 1 output tokens": lambda body: [
                (0, _ev("message_start", message={"usage": {"input_tokens": 100}})),
                (0, _ev("content_block_start", index=0, content_block={
                    "type": "tool_use", "id": "call_1", "name": "", "input": {}})),
                (0, _ev("content_block_delta", index=0,
                        delta={"type": "input_json_delta", "partial_json": ""})),
                (0, _ev("content_block_stop", index=0)),
                (0, _ev("message_delta", delta={"stop_reason": "tool_use"},
                        usage={"output_tokens": 1})),
                (0, _ev("message_stop"))],
            "HTTP 400": lambda body: 400 if len(body["messages"]) > 3 else _stream(),
        }
        for want, reply in cases.items():
            with self.subTest(want):
                FakeAnthropic.reply = staticmethod(reply)
                run, detail = self._replay(repeat=3)
                self.assertEqual(run["status"], "void")
                self.assertIn(want, run["void"])
                self.assertEqual(run["measurements"], [])
        # The 4xx run stopped at its first bad turn (seq 6) rather than averaging past it.
        self.assertEqual([t["seq"] for t in detail["turns"]], [1, 4, 5])

    def test_a_reply_cut_short_at_max_tokens_voids_the_run(self):
        FakeAnthropic.reply = staticmethod(lambda body: _stream(output_tokens=3,
                                                                stop_reason="max_tokens"))
        # The recording's replies were 5 tokens long: these 3 are a truncation.
        run, detail = self._replay()
        self.assertEqual(run["status"], "void")
        self.assertIn("cut short", run["void"])
        self.assertEqual(detail["turns"], [])
        # The recording stopped at 2, so hitting max_tokens at 3 is not a cut.
        run, detail = self._replay(self._corpus("short", recorded_output_tokens=2))
        self.assertEqual(run["status"], "published", run.get("void"))
        self.assertEqual(run["provenance"]["trace_corpus"]["replay"]["max_tokens_turns"], 5)

    def test_a_max_tokens_turn_with_no_recorded_reply_is_counted_unverified(self):
        # Every reply stops at max_tokens after 3 tokens. The recording stopped at 2
        # where it saved a response, so those turns were not cut; seqs 5 and 6 have
        # none, so nothing can say whether theirs were.
        FakeAnthropic.reply = staticmethod(lambda body: _stream(output_tokens=3,
                                                                stop_reason="max_tokens"))
        corpus = self._corpus("unsaved", recorded_output_tokens=2, unsaved={5, 6})
        run, detail = self._replay(corpus, repeat=2)
        self.assertEqual(run["status"], "published", run.get("void"))
        replay = run["provenance"]["trace_corpus"]["replay"]
        self.assertEqual((replay["max_tokens_turns"], replay["max_tokens_turns_unverified"]),
                         (10, 4))
        # A full recording leaves nothing unverified.
        run, detail = self._replay(self._corpus("saved", recorded_output_tokens=2))
        self.assertEqual(run["provenance"]["trace_corpus"]["replay"]
                         ["max_tokens_turns_unverified"], 0)

    def test_a_prompt_that_comes_back_shorter_voids_the_run(self):
        # The recording's prompts were 100 tokens; this engine counts 10.
        FakeAnthropic.reply = staticmethod(lambda body: _stream(input_tokens=10))
        run, detail = self._replay()
        self.assertEqual(run["status"], "void")
        self.assertIn("90 tokens short", run["void"])
        self.assertEqual(detail["turns"], [])
        # 60 short is inside the 64-token floor: counted, not void.
        FakeAnthropic.reply = staticmethod(lambda body: _stream(input_tokens=40))
        run, detail = self._replay()
        self.assertEqual(run["status"], "published", run.get("void"))
        replay = run["provenance"]["trace_corpus"]["replay"]
        self.assertEqual(replay["prompt_tokens_differ_turns"], 5)
        self.assertEqual(replay["prompt_shortfall_tolerance"],
                         {"tokens": cc_trace.PROMPT_SHORTFALL_TOKENS,
                          "fraction": cc_trace.PROMPT_SHORTFALL_FRACTION})
        # On a long prompt the percentage sets the bar, not the floor.
        self.assertEqual(cc_trace.prompt_shortfall(10000, 9600), 0)
        self.assertEqual(cc_trace.prompt_shortfall(10000, 9400), 600)
        self.assertEqual(cc_trace.prompt_shortfall(None, 5), 0)

    def test_a_cache_that_survives_the_reset_voids_the_run(self):
        FakeAnthropic.reply = staticmethod(lambda body: _stream(input_tokens=25, cache_read=75))
        run, detail = self._replay()
        self.assertEqual(run["status"], "void")
        self.assertIn("did not start cold", run["void"])
        self.assertEqual(FakeAnthropic.resets, 1)

    def test_a_session_too_long_for_the_window_voids_before_sending(self):
        # The fake engine's window is 32768 tokens.
        run, detail = self._replay(self._corpus("long", recorded_input_tokens=40000))
        self.assertEqual(run["status"], "void")
        self.assertIn("does not fit", run["void"])
        self.assertEqual((FakeAnthropic.seen, FakeAnthropic.resets, detail["sent"]), ([], 0, 0))
        tc = run["provenance"]["trace_corpus"]
        self.assertEqual(tc["max_recorded_prompt_tokens"], 40000)
        # The window came from the endpoint's own /v1/models.
        self.assertEqual(tc["context_length_source"], "/v1/models")

    def test_a_session_with_no_room_for_its_reply_voids_before_sending(self):
        # Every prompt fits the 32768-token window, but not with its 100-token reply.
        run, detail = self._replay(self._corpus("no-room", recorded_input_tokens=32700,
                                                recorded_output_tokens=100))
        self.assertEqual(run["status"], "void")
        self.assertIn("takes 32800 tokens", run["void"])
        self.assertIn("does not fit", run["void"])
        self.assertEqual(detail["sent"], 0)
        # With room for the reply, it runs (the engine counting the recorded prompt).
        FakeAnthropic.reply = staticmethod(lambda body: _stream(input_tokens=32600))
        run, detail = self._replay(self._corpus("room", recorded_input_tokens=32600,
                                                recorded_output_tokens=100))
        self.assertEqual(run["status"], "published", run.get("void"))

    def test_turns_with_no_recorded_prompt_are_counted_and_a_corpus_of_them_voids(self):
        # Two of five turns saved no response: their prompts are counted as unknown.
        run, detail = self._replay(self._corpus("some", unsaved={5, 6}))
        self.assertEqual(run["status"], "published", run.get("void"))
        self.assertEqual(run["provenance"]["trace_corpus"]["turns_without_recorded_prompt"], 2)
        # No turn saved one: nothing shows the session fits, so nothing is sent.
        FakeAnthropic.seen = []
        run, detail = self._replay(self._corpus("none", unsaved={1, 4, 5, 6, 7}))
        self.assertEqual(run["status"], "void")
        self.assertIn("no prompt length for any turn", run["void"])
        self.assertEqual((detail["sent"], FakeAnthropic.seen), (0, []))
        self.assertEqual(run["provenance"]["trace_corpus"]["turns_without_recorded_prompt"], 5)

    def test_wrong_engine_and_missing_provenance_fail_before_sending(self):
        with self.assertRaises(SystemExit):
            self._replay(engine="ollama")
        with self.assertRaises(SystemExit):
            cc_trace.parse_per_engine("--quantization", "scratchy=mlx-affine-b4-g64")
        with self.assertRaises(SystemExit):
            self._replay(engine_provenance={**PER_ENGINE, "quantization": {"scratchy": "q"}})
        self.assertEqual(cc_trace.parse_per_engine("--x", "scratchy=a, ollama=b@c"),
                         {"scratchy": "a", "ollama": "b@c"})
        self.assertEqual(FakeAnthropic.seen, [])

    def test_warmup_is_reported_beside_the_headline_never_in_it(self):
        # The warm-up pass is slow, as a freshly started engine is; the measured
        # passes are not. Nothing slow may reach a headline cell.
        warm_ttft = 5.0
        calls = []

        def reply(body):
            calls.append(1)
            first_pass = len(calls) <= 5
            return [(warm_ttft if first_pass and i == 0 else 0, ev)
                    for i, (_, ev) in enumerate(_stream())]

        FakeAnthropic.reply = staticmethod(reply)
        run, detail = self._replay(repeat=2, warmup=True)
        self.assertEqual(run["status"], "published", run.get("void"))
        # One warm-up pass, then the two measured ones, a reset before each.
        self.assertEqual([t["repeat"] for t in detail["turns"]], [0] * 5 + [1] * 5 + [2] * 5)
        self.assertEqual([t["warmup"] for t in detail["turns"]], [True] * 5 + [False] * 10)
        self.assertEqual(FakeAnthropic.resets, 3)
        replay = run["provenance"]["trace_corpus"]["replay"]
        self.assertIs(replay["warmup"], True)
        self.assertEqual([r["repeat"] for r in replay["cache_reset_seconds"]], [0, 1, 2])
        # Turn counts in provenance are over the measured passes only.
        self.assertEqual(replay["prompt_tokens_differ_turns"], 0)
        # Headline cells: the measured passes, n = 2, and none of the 5 s turns.
        for m in run["measurements"]:
            if (m.get("facet") or {}).get("pass") == "warmup":
                continue
            self.assertEqual(m["n"], 2, m)
            if m["metric"] in ("ttft_ms", "turn_e2e_ms", "e2e_star_ms"):
                self.assertLess(m["stat"]["p99"], warm_ttft * 1000, m)
        [ttft] = self._meas(run, "ttft_ms")
        self.assertEqual(ttft["stat"]["samples"], 10)
        # The warm-up sits beside them under its own facet, n = 1, with the 5 s in it.
        warm = {m["metric"]: m for m in run["measurements"]
                if m.get("facet") == {"pass": "warmup"}}
        self.assertEqual(set(warm), set(cc_trace.TURN_METRICS))
        self.assertTrue(all(m["n"] == 1 and "series" not in m for m in warm.values()))
        self.assertEqual(warm["ttft_ms"]["stat"]["samples"], 5)
        self.assertGreaterEqual(warm["ttft_ms"]["stat"]["median"], warm_ttft * 1000)

    def test_a_failed_reset_voids_the_run(self):
        run, detail = self._replay(reset_cache_url=self.url + "/no-such-reset")
        self.assertEqual(run["status"], "void")
        self.assertIn("cache reset", run["void"])
        self.assertIn("404", run["void"])
        self.assertEqual(detail["turns"], [])


    @unittest.skipUnless(VALIDATOR.exists() and BOARD.exists(),
                         "site/build_leaderboard.py is not in this checkout (#176)")
    def test_output_passes_the_leaderboard_validator(self):
        # A recording with overlapping requests, as every real Claude Code
        # session has, so the run names `replay-serialized`; and a warm-up pass.
        run, _ = self._replay(repeat=3, warmup=True)
        self.assertEqual(run["status"], "published", run.get("void"))
        self.assertIn(cc_trace.DISCLOSE_SERIAL, run["provenance"]["disclosed"])
        self.assertEqual(board_problems(run), [])
        broken = json.loads(json.dumps(run))
        del broken["provenance"]["checkpoint_revision"]
        self.assertTrue(any("'checkpoint_revision'" in p for p in board_problems(broken)))
        void = dict(run, status="void", void="test")
        self.assertTrue(board_problems(void))


class OllamaReplayTest(unittest.TestCase):
    """The same replay against an ollama-shaped engine, whose cache only a reload clears."""

    def setUp(self):
        FakeOllama.served = 0
        FakeOllama.resets = 0
        FakeOllama.counting = True
        FakeOllama.seen = []
        FakeOllama.loaded = {"gemma"}
        FakeOllama.loadable = {"gemma"}
        FakeOllama.cached = set()
        FakeOllama.reloads = 0
        FakeOllama.hits = []
        FakeOllama.reply = staticmethod(_ollama_reply)
        self.engine = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        self.engine.daemon_threads = True
        self.url = f"http://127.0.0.1:{_serve(self.engine)}"
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = write_corpus(Path(self.tmp.name) / "corpus")

    def tearDown(self):
        self.engine.shutdown()
        self.engine.server_close()
        self.tmp.cleanup()

    def _replay(self, **kw):
        args = dict(engine="ollama", model="gemma-4-12b-it", machine="m5max-64", rung="R1",
                    repeat=3, scratchy_sha="abc123", context_length=32768,
                    engine_provenance=PER_ENGINE, run_id="test-laneA-ollama",
                    log=lambda msg: None)
        args.update(kw)
        return cc_trace.replay(self.corpus, self.url, **args)

    def test_every_pass_starts_cold_after_a_reload(self):
        run, detail = self._replay()
        self.assertEqual(run["status"], "published", run.get("void"))
        # ollama reports no window: the one checked against is the flag's, and says so.
        tc = run["provenance"]["trace_corpus"]
        self.assertEqual((run["provenance"]["context_length_tokens"],
                          tc["context_length_source"]), (32768, "--context-length"))
        # A reload before the warm-up and before each of the 3 measured passes.
        self.assertEqual(FakeOllama.reloads, 4)
        self.assertEqual(FakeOllama.loaded, {"gemma"})
        # Turn 1 of every pass re-reads its whole prompt, as scratchy's does.
        for r in range(4):
            [first] = [t for t in detail["turns"] if t["repeat"] == r and t["turn_index"] == 1]
            self.assertEqual(first["reprefill_ratio"], 1.0, r)
        # No pass is served from the previous one's cache.
        self.assertEqual(FakeOllama.hits, [False] * 20)
        prov = run["provenance"]
        self.assertEqual(prov["trace_corpus"]["replay"]["cache_reset"], "unload and reload gemma")
        self.assertEqual(len(prov["trace_corpus"]["replay"]["cache_reset_seconds"]), 4)
        self.assertNotIn(cc_trace.DISCLOSE_NO_RESET, prov["disclosed"])
        self.assertIn(cc_trace.DISCLOSE_NO_COUNTER, prov["disclosed"])
        self.assertEqual(prov["ollama_version"], "0.40.4")

    def test_without_the_reload_the_cache_would_carry_over(self):
        # The control: the same engine with nothing clearing its cache serves the
        # second pass from the first's, which is what #551 measured.
        for _ in range(2):
            for body in conversation():
                _ollama_reply(body)
        self.assertEqual(FakeOllama.hits, [False] * 5 + [True] * 5)

    def test_a_reload_of_the_wrong_model_voids_the_run(self):
        run, detail = self._replay(served_model="no-such-model")
        self.assertEqual(run["status"], "void")
        self.assertIn("unload of 'no-such-model' answered 404", run["void"])
        self.assertIn("not found", run["void"])
        self.assertEqual((detail["turns"], FakeOllama.seen), ([], []))

    def test_a_cache_that_survives_the_reload_voids_the_run(self):
        # A reload that leaves the cache behind: what the cold-start check exists for.
        FakeOllama.reply = staticmethod(lambda body: _stream(input_tokens=1, cache_read=99))
        run, detail = self._replay()
        self.assertEqual(run["status"], "void")
        self.assertIn("warm-up turn 1", run["void"])
        self.assertIn("did not start cold", run["void"])

    def test_a_model_that_never_comes_back_voids_the_run(self):
        # The load is accepted, but the model never shows in /api/ps.
        original = FakeOllama.do_POST

        def lost(handler):
            original(handler)
            FakeOllama.loaded.clear()

        FakeOllama.do_POST = lost
        try:
            run, _ = self._replay(timeout_s=0.5)
        finally:
            FakeOllama.do_POST = original
        self.assertEqual(run["status"], "void")
        self.assertIn("still not loaded", run["void"])

    @unittest.skipUnless(VALIDATOR.exists() and BOARD.exists(),
                         "site/build_leaderboard.py is not in this checkout (#176)")
    def test_output_passes_the_leaderboard_validator(self):
        # Every disclosure an ollama run can name, and a warm-up pass.
        run, _ = self._replay()
        self.assertEqual(run["status"], "published", run.get("void"))
        self.assertIn(cc_trace.DISCLOSE_NO_COUNTER, run["provenance"]["disclosed"])
        self.assertIn(cc_trace.DISCLOSE_SERIAL, run["provenance"]["disclosed"])
        self.assertEqual(board_problems(run), [])


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
