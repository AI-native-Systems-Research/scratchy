#!/usr/bin/env bash
# verify_prefill_real.sh — the SOLE authority on whether batched prefill is REAL.
#
# WHY THIS EXISTS: the assistant repeatedly claimed "batched prefill verified
# coherent on-card" and checked off task #33 while the running binary actually
# did sequential mq=1 prefill (its own log: "no distinct SUPERDSC prefill bundle
# — prompt prefill is sequential (mq=1 per token)"). This script removes the need
# to trust ANY prose claim about prefill: YOU run it, it reads a truth-signal the
# worker emits during the real forward, and it FAILS CLOSED.
#
# It can only print PREFILL_REAL=PASS if, on THIS request, the card actually ran a
# batched forward (path=batched, m_used>1) AND produced coherent output. There is
# no code path that prints PASS otherwise.
#
# Contract the worker MUST satisfy (see spyre_worker.rs): per request it prints
# exactly one line to its serve log:
#     PREFILL_PATH=batched m_used=<N>      # real: whole prompt prefix in one mq=N forward
#     PREFILL_PATH=sequential              # fake: prompt walked one token at a time
# The silent "prefill=none, serve sequential anyway" fallback is a FAIL here.
#
# Usage (run it YOURSELF — do not let anyone paste you its result):
#   bash scripts/verify_prefill_real.sh <serve_log_path> <model_path> [port]
# e.g. on the pod:
#   bash scripts/verify_prefill_real.sh /tmp/serve.log \
#        /tmp/hf/hub/models--ibm-granite--granite-3.3-2b-instruct/snapshots/707f574c... 8000
set -uo pipefail

SERVE_LOG=${1:?usage: verify_prefill_real.sh <serve_log> <model_path> [port]}
MODEL=${2:?usage: verify_prefill_real.sh <serve_log> <model_path> [port]}
PORT=${3:-8000}

PROMPT='The capital of France is'
# Coherence anchor: transformers fp32 / mlx-lm golden continuation begins " Paris".
GOLDEN=' Paris'

fail() { echo; echo "PREFILL_REAL=FAIL: $1"; exit 1; }

[ -f "$SERVE_LOG" ] || fail "serve log '$SERVE_LOG' not found (is the serve up and logging there?)"

before=$(wc -l < "$SERVE_LOG")

resp=$(curl -s "http://localhost:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":8,\"temperature\":0}") \
  || fail "curl to serve failed (serve not up on :${PORT}?)"

text=$(printf '%s' "$resp" | python3 -c 'import sys,json
try:
    print(json.load(sys.stdin)["choices"][0]["text"])
except Exception as e:
    import sys; sys.stderr.write(str(e)); sys.exit(3)') \
  || fail "serve response was not valid completion JSON: $resp"

# only the log lines produced by THIS request
sleep 1
newlog=$(tail -n +$((before + 1)) "$SERVE_LOG")

echo "===== RAW COMPLETION TEXT ====="
printf '%q\n' "$text"
echo "===== RAW SERVE-LOG LINES (this request only) ====="
echo "$newlog"
echo "==============================================="

# ---- fail-closed checks on RAW on-card evidence ----

# (1) explicit fake/sequential markers => FAIL
echo "$newlog" | grep -qiE 'prefill=none|sequential \(mq=1|PREFILL_PATH=sequential' \
  && fail "worker ran the sequential mq=1 path (fake prefill), not batched"

# (2) the batched truth line must be present with m_used>1
mline=$(echo "$newlog" | grep -oE 'PREFILL_PATH=batched m_used=[0-9]+' | head -1)
[ -n "$mline" ] || fail "no 'PREFILL_PATH=batched m_used=N' truth line in the log — \
worker did not run (or did not report) a real batched forward"
m=$(echo "$mline" | grep -oE '[0-9]+$')
[ "${m:-0}" -gt 1 ] || fail "batched forward reported m_used=${m:-?}, which is not > 1 (that is sequential)"

# (3) output must be coherent
printf '%s' "$text" | grep -qF "$GOLDEN" \
  || fail "output not coherent — expected to contain '${GOLDEN}', got: $(printf '%q' "$text")"

echo
echo "PREFILL_REAL=PASS m_used=${m} text=$(printf '%q' "$text")"
