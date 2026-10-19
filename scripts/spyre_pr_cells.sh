#!/usr/bin/env bash
# A perf PR's merge gate, as one script: the standard cell set, main against
# the PR, on this pod, back to back — writing the progress entry the spyre
# page charts (site/data/spyre/progress/<pr>.json) directly, so the gate's
# numbers and the site's data are the same numbers.
#
# The standard cell set (the sweep grid's own parameters, from the
# .sweep-legs cells.tsv the 10-19 sweep ran):
#   rung-1  2b fp16 ~1k and ~10k — TTFT and ITL, the length-calibrated
#           streaming client (scripts/bench_spyre_rung1.py, 3 streamed
#           requests a run, medians), the serve's request-finished lines the
#           authoritative length gate (#677);
#   conc-8  2b and 8b fp16 ~1k — tok/s and TPOT, `vllm bench serve` on the
#           shared conc-8 ~1k dataset, 16 prompts, 256 output tokens.
# Opt-in beyond the standard set (the legs' own parameters, cells a PR
# moving long-context batched decode gates on):
#   conc-4/8 2b fp16 ~10k — tok/s, TPOT and TTFT, `vllm bench serve` on the
#           shared ~10k dataset, 16 prompts, 256 output tokens, mns 4/8,
#           mml 12288. About an hour a group on the card.
# Each side of each cell measures RUNS back-to-back runs (default 2) on its
# own boot; each headline is its side's LAST run — the steady-state law the
# site enforces on the file, enforced here at write time too, so the script
# cannot emit a file the site would refuse.
#
# Honesty rules, from the sweep's legs: both sides on this one pod, cell by
# cell (main's boot then the PR's, back to back); torch-spyre is never
# measured (the page pairs the entry with the newest sweep's torch cell);
# the script refuses to run with SCRATCHY_SDSC_* knobs set, with a leftover
# serve or any vfio holder, with a live build, or over an empty extraction.
#
# Usage:
#   scripts/spyre_pr_cells.sh --pr 683 \
#       --title "perf(spyre): ..." --head <40-hex> --base <40-hex> \
#       --2b <main-scr> <pr-scr> --8b <main-scr> <pr-scr> \
#       [--cells rung1-2b,conc8-2b,conc8-8b] \
#       [--out site/data/spyre/progress] [--runs 2] [--port 8079] \
#       [--self-test]
#
# Environment:
#   DS_1K   required when a ~1k conc group runs: the shared conc-8 ~1k
#           dataset (16 prompts), the same file the sweep's batched cells ran
#   DS_10K  required when a ~10k group runs: the shared ~10k dataset
#           (16 prompts), the same file the sweep's batched ~10k cells ran —
#           pinned by md5 to the sweep's own conc-10300.jsonl, refused on any
#           mismatch
#   CORES   the cores serve and client are pinned to (default 48-95)
#
# --cells selects a subset of the cell groups to measure: rung1-2b (the two
#   rung-1 cells), conc8-2b, conc8-8b — comma-separated, default all three —
#   plus the opt-in long-context groups conc4-10k-2b and conc8-10k-2b. A
#   group's binary pair and dataset are only required when its group is
#   selected: a 2b-only PR runs --cells rung1-2b,conc8-2b with no --8b pair,
#   a rung-1-only gate needs no dataset, and a PR moving ~10k batched decode
#   adds conc4-10k-2b,conc8-10k-2b. The assembler enforces exactly the
#   selected cells, no more, no less.
# --self-test runs the argument parser and the binary-pair guard on dummy
#   paths and exits — no pod, no card; run it anywhere.
#
# Exits non-zero on any refusal; a failed cell fails the whole run — the
# gate is complete or it did not happen. Writes <out>/<pr>.json (atomically)
# and refuses to overwrite an existing entry.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"

# ---- arguments ---------------------------------------------------------------
PR="" TITLE="" HEAD="" BASE="" OUT="" RUNS=2 PORT=8079
B2_MAIN="" B2_PR="" B8_MAIN="" B8_PR=""
CELLS="rung1-2b,conc8-2b,conc8-8b" SELF_TEST=no
usage() { awk 'NR == 1 { next } /^#/ { print substr($0, 3); next } { exit }' "$0" >&2; exit 64; }
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --pr) PR=$2; shift 2 ;;
      --title) TITLE=$2; shift 2 ;;
      --head) HEAD=$2; shift 2 ;;
      --base) BASE=$2; shift 2 ;;
      --2b) B2_MAIN=$2; B2_PR=$3; shift 3 ;;
      --8b) B8_MAIN=$2; B8_PR=$3; shift 3 ;;
      --cells) CELLS=$2; shift 2 ;;
      --out) OUT=$2; shift 2 ;;
      --runs) RUNS=$2; shift 2 ;;
      --port) PORT=$2; shift 2 ;;
      --self-test) SELF_TEST=yes; shift ;;
      *) echo "REFUSING: unknown argument '$1'" >&2; usage ;;
    esac
  done
}
parse_args "$@"

# The cell groups --cells selected. A group's binary pair and dataset are
# only required when its group is selected.
WANT_R1_2B=no WANT_C8_2B=no WANT_C8_8B=no WANT_C4_10K_2B=no WANT_C8_10K_2B=no
select_cells() {
  WANT_R1_2B=no WANT_C8_2B=no WANT_C8_8B=no WANT_C4_10K_2B=no WANT_C8_10K_2B=no
  for G in ${CELLS//,/ }; do
    case "$G" in
      rung1-2b) WANT_R1_2B=yes ;;
      conc8-2b) WANT_C8_2B=yes ;;
      conc8-8b) WANT_C8_8B=yes ;;
      conc4-10k-2b) WANT_C4_10K_2B=yes ;;
      conc8-10k-2b) WANT_C8_10K_2B=yes ;;
      *) echo "REFUSING: unknown cell group '$G' in --cells '$CELLS' — known: rung1-2b, conc8-2b, conc8-8b, conc4-10k-2b, conc8-10k-2b" >&2; return 64 ;;
    esac
  done
  if [ "$WANT_R1_2B$WANT_C8_2B$WANT_C8_8B$WANT_C4_10K_2B$WANT_C8_10K_2B" = nonononono ]; then
    echo "REFUSING: --cells '$CELLS' selects nothing — the gate is complete or it did not happen" >&2
    return 64
  fi
}

# A main/PR pair that is the same file (same bytes, so the same md5) would
# run a same-binary ABAB and write a file that compares main with main —
# vacuous, and it looks valid. Refuse it up front, per pair actually used.
refuse_identical() {  # refuse_identical <label> <main-bin> <pr-bin>
  if cmp -s "$2" "$3"; then
    echo "REFUSING: the $1 main and PR binaries are identical (same bytes, so the same md5) — a same-binary ABAB is a vacuous gate" >&2
    exit 65
  fi
}

# md5_of <file>: md5sum on the pods, md5 -q on a mac.
md5_of() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  else
    md5 -q "$1"
  fi
}

# A pinned input must be THE file, not a same-named drift: refuse unless the
# file's md5 is the pinned one.
refuse_md5() {  # refuse_md5 <file> <want-md5> <label>
  local GOT
  GOT=$(md5_of "$1")
  [ "$GOT" = "$2" ] || {
    echo "REFUSING: $3 '$1' has md5 $GOT != $2 — a pinned input must be the exact file" >&2
    exit 65
  }
}

# The #677 length gate, on the serve's own record (never a heuristic): the
# serve's last <want> request-finished lines must all carry prompt_tokens
# within 2% of the target — never emit a cell over a prompt measured at the
# wrong length. Rung-1 runs gate its 3 streamed requests; a ~10k bench run
# gates its 18 (2 warmups + 16 prompts).
length_gate() {  # length_gate <cell> <want> <target>
  local CELL=$1 WANT_N=$2 TARGET=$3 RECS
  RECS=$(grep -oE 'prompt_tokens=[0-9]+' "$SRVLOG" | tail -"$WANT_N" | cut -d= -f2 | tr '\n' ' ')
  python3 - "$CELL" "$RECS" "$TARGET" "$WANT_N" <<'PY'
import sys
cell, recs_s, target_s, want_s = sys.argv[1:5]
recs = [int(x) for x in recs_s.split()]
target, want = int(target_s), int(want_s)
if len(recs) != want:
    print(f"pr_cells: REFUSED: {cell}: expected {want} serve request lines, got "
          f"{len(recs)} -- verdict over nothing must fail", file=sys.stderr)
    sys.exit(1)
bad = [r for r in recs if abs(r - target) / target > 0.02]
if bad:
    print(f"pr_cells: REFUSED: {cell}: serve recorded prompt_tokens {bad} off "
          f"target {target} by >2% -- never emit a cell over a prompt measured "
          "at the wrong length", file=sys.stderr)
    sys.exit(1)
PY
}

# ---- self-test: the parser and the guards, on dummy paths --------------------
# No pod, no card: catches argument bugs before any card time. The P0 in #690
# — both --2b paths landing in the main slot, so every file the script wrote
# would have compared main with main — shipped because nothing checked the
# parse.
self_test() {
  local fails=0 T1 T2
  check() {  # check <what> <got> <want>
    if [ "$2" = "$3" ]; then
      echo "ok: $1"
    else
      echo "FAIL: $1: got '$2', want '$3'" >&2
      fails=$((fails + 1))
    fi
  }
  # Two full 40-hex shas, the shape --head/--base refuse to miss.
  local H40=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  local B40=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
  parse_args --pr 42 --title t --head "$H40" --base "$B40" \
    --2b /dummy/main-2b /dummy/pr-2b --8b /dummy/main-8b /dummy/pr-8b
  check '--2b keeps the main binary' "$B2_MAIN" /dummy/main-2b
  check '--2b keeps the PR binary' "$B2_PR" /dummy/pr-2b
  check '--8b keeps the main binary' "$B8_MAIN" /dummy/main-8b
  check '--8b keeps the PR binary' "$B8_PR" /dummy/pr-8b
  CELLS=rung1-2b
  select_cells
  check '--cells rung1-2b keeps the rung-1 cells' "$WANT_R1_2B" yes
  check '--cells rung1-2b drops the 2b conc-8 cells' "$WANT_C8_2B" no
  check '--cells rung1-2b drops the 8b conc-8 cells' "$WANT_C8_8B" no
  CELLS=conc4-10k-2b,conc8-10k-2b
  select_cells
  check '--cells conc4-10k-2b,conc8-10k-2b keeps the conc-4 ~10k cells' "$WANT_C4_10K_2B" yes
  check '--cells conc4-10k-2b,conc8-10k-2b keeps the conc-8 ~10k cells' "$WANT_C8_10K_2B" yes
  check '--cells conc4-10k-2b,conc8-10k-2b drops the rung-1 cells' "$WANT_R1_2B" no
  check '--cells conc4-10k-2b,conc8-10k-2b drops the 2b conc-8 ~1k cells' "$WANT_C8_2B" no
  check '--cells conc4-10k-2b,conc8-10k-2b drops the 8b conc-8 ~1k cells' "$WANT_C8_8B" no
  CELLS=bogus
  if select_cells 2>/dev/null; then
    echo "FAIL: --cells bogus was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: --cells bogus is refused"
  fi
  CELLS=""
  if select_cells 2>/dev/null; then
    echo "FAIL: an empty --cells was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: an empty --cells is refused"
  fi
  T1=$(mktemp); T2=$(mktemp)
  printf 'same' > "$T1"; cp "$T1" "$T2"
  if ( refuse_identical selftest "$T1" "$T2" ) 2>/dev/null; then
    echo "FAIL: the identical-binary guard accepted identical binaries" >&2; fails=$((fails + 1))
  else
    echo "ok: the identical-binary guard refuses identical binaries"
  fi
  printf 'other' > "$T2"
  if ( refuse_identical selftest "$T1" "$T2" ) 2>/dev/null; then
    echo "ok: the identical-binary guard accepts differing binaries"
  else
    echo "FAIL: the identical-binary guard refused differing binaries" >&2; fails=$((fails + 1))
  fi
  # The md5 pin: a pinned input must be the exact file, not a same-named drift.
  local TM
  TM=$(md5_of "$T1")
  if ( refuse_md5 "$T1" "$TM" selftest ) 2>/dev/null; then
    echo "ok: the md5 pin accepts the file it pinned"
  else
    echo "FAIL: the md5 pin refused its own file" >&2; fails=$((fails + 1))
  fi
  if ( refuse_md5 "$T1" 00000000000000000000000000000000 selftest ) 2>/dev/null; then
    echo "FAIL: the md5 pin accepted a wrong file" >&2; fails=$((fails + 1))
  else
    echo "ok: the md5 pin refuses a wrong file"
  fi
  # The #677 length gate: the serve's own record, 2% either way, and never a
  # verdict over missing lines.
  local SRVLOG_OLD=${SRVLOG:-}
  SRVLOG=$(mktemp)
  printf 'request finished prompt_tokens=10300\n%.0s' 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 > "$SRVLOG"
  if ( length_gate selftest 18 10300 ) 2>/dev/null; then
    echo "ok: the length gate accepts 18 on-target requests"
  else
    echo "FAIL: the length gate refused on-target requests" >&2; fails=$((fails + 1))
  fi
  printf 'request finished prompt_tokens=10300\n%.0s' 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 > "$SRVLOG"
  printf 'request finished prompt_tokens=9000\n' >> "$SRVLOG"
  if ( length_gate selftest 18 10300 ) 2>/dev/null; then
    echo "FAIL: the length gate accepted a wrong-length request" >&2; fails=$((fails + 1))
  else
    echo "ok: the length gate refuses a wrong-length request"
  fi
  printf 'request finished prompt_tokens=10300\n%.0s' 1 2 3 4 5 6 7 8 9 10 > "$SRVLOG"
  if ( length_gate selftest 18 10300 ) 2>/dev/null; then
    echo "FAIL: the length gate accepted missing request lines" >&2; fails=$((fails + 1))
  else
    echo "ok: the length gate refuses missing request lines"
  fi
  rm -f "$SRVLOG"
  SRVLOG=$SRVLOG_OLD
  rm -f "$T1" "$T2"
  if [ "$fails" -gt 0 ]; then
    echo "self-test: $fails check(s) FAILED" >&2
    return 1
  fi
  echo "self-test: ok"
}
if [ "$SELF_TEST" = yes ]; then
  self_test && exit 0
  exit 1
fi

select_cells || exit $?
refuse_blank() { [ -n "$2" ] || { echo "REFUSING: $1 is required" >&2; exit 64; }; }
refuse_blank --pr "$PR"; refuse_blank --title "$TITLE"
refuse_blank --head "$HEAD"; refuse_blank --base "$BASE"
# The ~10k groups run the 2b pair too.
if [ "$WANT_R1_2B" = yes ] || [ "$WANT_C8_2B" = yes ] || [ "$WANT_C4_10K_2B" = yes ] || [ "$WANT_C8_10K_2B" = yes ]; then
  refuse_blank "--2b main binary" "$B2_MAIN"; refuse_blank "--2b pr binary" "$B2_PR"
fi
if [ "$WANT_C8_8B" = yes ]; then
  refuse_blank "--8b main binary" "$B8_MAIN"; refuse_blank "--8b pr binary" "$B8_PR"
fi
case "$PR" in ''|*[!0-9]*) echo "REFUSING: --pr must be a number" >&2; exit 64 ;; esac
sha40() { [ ${#1} -eq 40 ] && case "$1" in *[!0-9a-fA-F]*) false ;; *) true ;; esac; }
sha40 "$HEAD" || { echo "REFUSING: --head is not a full 40-hex commit sha" >&2; exit 64; }
sha40 "$BASE" || { echo "REFUSING: --base is not a full 40-hex commit sha" >&2; exit 64; }
case "$RUNS" in ''|*[!0-9]*|0|1) echo "REFUSING: --runs must be at least 2 (the file needs two runs a side)" >&2; exit 64 ;; esac
OUT="${OUT:-$REPO/site/data/spyre/progress}"

# ---- the standard cell set (cells.tsv's own parameters) -----------------------
M2B=ibm-granite/granite-3.1-2b-instruct
M8B=ibm-granite/granite-3.1-8b-instruct
CORES="${CORES:-48-95}"

# ---- tools live outside a plain exec shell's PATH (leg.sh's loop) -------------
for D in /work/.cargo/bin /root/.local/bin "$HOME/.cargo/bin" "$HOME/.local/bin"; do
  [ -d "$D" ] || continue
  case ":$PATH:" in *":$D:"*) ;; *) PATH="$D:$PATH" ;; esac
done
export PATH

# ---- guards -------------------------------------------------------------------
if env | grep -q '^SCRATCHY_SDSC_'; then
  echo "REFUSING: SCRATCHY_SDSC_* knobs are set; the gate measures shipping behavior" >&2
  exit 66
fi
command -v uv >/dev/null 2>&1 || {
  echo "REFUSING: uv not on PATH — the reference venv's runner (a login shell has it)" >&2
  exit 69
}
[ -d /work/spyre-inference ] || { echo "REFUSING: /work/spyre-inference (the reference venv) is missing on this pod" >&2; exit 69; }
[ -f "$HERE/bench_spyre_rung1.py" ] || { echo "REFUSING: $HERE/bench_spyre_rung1.py is missing" >&2; exit 69; }
for PB in p400.txt q3700-1.txt n10k-1.txt; do
  [ -f "$HERE/spyre_rung1_prompts/$PB" ] || {
    echo "REFUSING: $HERE/spyre_rung1_prompts/$PB (a 10-18 base prompt) is missing" >&2; exit 69; }
done
# A group's dataset is required only when one of its groups runs: the rung-1
# cells take their prompts from the harness's own directory.
if [ "$WANT_C8_2B" = yes ] || [ "$WANT_C8_8B" = yes ]; then
  refuse_blank 'DS_1K (the shared conc-8 ~1k dataset)' "${DS_1K:-}"
  [ -f "$DS_1K" ] || { echo "REFUSING: dataset '$DS_1K' not found on this pod" >&2; exit 65; }
fi
if [ "$WANT_C4_10K_2B" = yes ] || [ "$WANT_C8_10K_2B" = yes ]; then
  refuse_blank 'DS_10K (the shared ~10k batched dataset)' "${DS_10K:-}"
  [ -f "$DS_10K" ] || { echo "REFUSING: dataset '$DS_10K' not found on this pod" >&2; exit 65; }
  # Pinned by content to the sweep's own conc-10300.jsonl (16 rows, the file
  # #701's worker re-verified on the card): the ~10k cells pair with the
  # sweep's batched cells only on those exact prompts.
  refuse_md5 "$DS_10K" 68c030b68dbbf93c80a40e7e5aa28766 "DS_10K (the sweep's conc-10300.jsonl)"
fi
if [ "$WANT_R1_2B" = yes ] || [ "$WANT_C8_2B" = yes ] || [ "$WANT_C4_10K_2B" = yes ] || [ "$WANT_C8_10K_2B" = yes ]; then
  for B in "$B2_MAIN" "$B2_PR"; do
    [ -x "$B" ] || { echo "REFUSING: '$B' is not an executable scr binary" >&2; exit 69; }
  done
  refuse_identical --2b "$B2_MAIN" "$B2_PR"
fi
if [ "$WANT_C8_8B" = yes ]; then
  for B in "$B8_MAIN" "$B8_PR"; do
    [ -x "$B" ] || { echo "REFUSING: '$B' is not an executable scr binary" >&2; exit 69; }
  done
  refuse_identical --8b "$B8_MAIN" "$B8_PR"
fi

# A process counts only if it is NOT a zombie: the pods' PID 1 does not reap,
# so zombie cargo piles up and a naive pgrep sees it forever. Match the
# executable name (comm), never the full command line.
proc_alive() {
  ps -eo stat=,comm= | awk -v c="$1" 'index($1, "Z") != 1 && $2 == c { ok = 1 } END { exit !ok }'
}
for COMM in cargo rustc dxp_standalone; do
  if proc_alive "$COMM"; then
    echo "REFUSING: a non-zombie $COMM is still running; finish it (foreground) first" >&2
    exit 67
  fi
done

# An orphaned ' serve .*--device spyre' held a card once, so kill any leftover
# first, then refuse if one survives or anything still holds a vfio device.
LEFTOVER=$(pgrep -f ' serve .*--device spyre' || true)
if [ -n "$LEFTOVER" ]; then
  echo "killing leftover spyre serve: $LEFTOVER" >&2
  kill $LEFTOVER 2>/dev/null || true
  for _ in $(seq 1 30); do
    LEFTOVER=$(pgrep -f ' serve .*--device spyre' || true)
    [ -z "$LEFTOVER" ] && break
    sleep 1
  done
fi
if [ -n "$(pgrep -f ' serve .*--device spyre' || true)" ]; then
  echo "REFUSING: a ' serve .*--device spyre' process survived the kill; the card is not free" >&2
  exit 67
fi
if ls -l /proc/[0-9]*/fd 2>/dev/null | grep -q 'vfio-device'; then
  echo "REFUSING: a process still holds a vfio-device fd (the card is busy)" >&2
  exit 67
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "REFUSING: something already listens on $PORT — a stale server?" >&2
  exit 68
fi

mkdir -p "$OUT"
DEST="$OUT/$PR.json"
[ -e "$DEST" ] && { echo "REFUSING: $DEST already exists — never overwrite a landed entry" >&2; exit 65; }

# ---- the pod env (verbatim, from the legs) ------------------------------------
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HOME=/work/.cache/huggingface

WORK="$(mktemp -d "$OUT/.pr$PR.XXXXXX")"

# ---- serve up/down (the legs' lines) ------------------------------------------
SRV=""
SRVLOG=""
cleanup() {
  if [ -n "$SRV" ] && kill -0 "$SRV" 2>/dev/null; then
    kill "$SRV" 2>/dev/null || true
    pkill -P "$SRV" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "$SRV" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SRV" 2>/dev/null || true
  fi
}
trap 'cleanup; rm -rf "$WORK"' EXIT
RUNS_JSONL="$WORK/runs.jsonl"

# serve_up <binary> <model> <mns> <mml> <rung1:yes|no> <tag>
serve_up() {
  local BIN=$1 MODEL=$2 MNS=$3 MML=$4 R1=$5 TAG=$6
  SRVLOG="$WORK/$TAG.server.log"
  local LAUNCH=(taskset -c "$CORES" "$BIN" serve --model "$MODEL" --device spyre
                --host 0.0.0.0 --port "$PORT" --max-num-seqs "$MNS" --max-model-len "$MML")
  if [ "$R1" = yes ]; then
    LAUNCH+=(--no-prefix-caching)  # the rung-1 harness: one request at a time
  fi
  "${LAUNCH[@]}" > "$SRVLOG" 2>&1 &
  SRV=$!
  SERVED=""
  local DEADLINE=$((SECONDS + 900))
  until [ -n "$SERVED" ]; do
    if ! kill -0 "$SRV" 2>/dev/null; then
      echo "FAILED: server died during startup; log: $SRVLOG" >&2
      tail -5 "$SRVLOG" >&2 || true
      exit 70
    fi
    if [ "$SECONDS" -ge "$DEADLINE" ]; then
      echo "FAILED: server not ready within 900 s; log: $SRVLOG" >&2
      exit 71
    fi
    SERVED=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin)["data"][0]["id"])
except Exception:
    pass
' || true)
    [ -n "$SERVED" ] || sleep 1
  done
}

serve_down() {
  cleanup
  SRV=""
  # The listen socket can outlive the kill by a moment; poll for it to close
  # before the next boot tries to bind the same port.
  for _ in $(seq 1 30); do
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null || return 0
    sleep 1
  done
  echo "REFUSING: the port $PORT is still held 30 s after teardown" >&2
  exit 72
}

# ---- one rung-1 run: the streaming client + the serve-side length gate --------
# Emits one runs.jsonl line per metric: {cell, side, run, metric, value}.
rung1_run() {  # rung1_run <tag> <cell> <side> <run> <ptoks>
  local TAG=$1 CELL=$2 SIDE=$3 RUN=$4 PTOKS=$5
  local RESULT="$WORK/$TAG.r$RUN.json" RUNLOG="$WORK/$TAG.r$RUN.log"
  ( cd /work/spyre-inference && TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 \
      taskset -c "$CORES" uv run --no-sync python "$HERE/bench_spyre_rung1.py" \
      "http://127.0.0.1:$PORT" "$SERVED" "$PTOKS" 3 128 "$HERE/spyre_rung1_prompts" ) \
    > "$RESULT" 2> "$RUNLOG"
  # The authoritative length gate: the serve's last 3 request-finished lines
  # must carry prompt_tokens within 2% of the target — never emit a cell over
  # a prompt measured at the wrong length.
  length_gate "$CELL" 3 "$PTOKS"
  python3 - "$CELL" "$SIDE" "$RUN" "$RESULT" >> "$RUNS_JSONL" <<'PY'
import json, math, sys
cell, side, run, path = sys.argv[1:5]
try:
    with open(path) as f:
        got = json.load(f)
except (OSError, ValueError) as e:
    raise SystemExit(f"pr_cells: FAILED: {cell}: client result is not JSON ({e})")
for key in ("ttft_ms", "itl_ms"):
    v = got.get(key)
    if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise SystemExit(
            f"pr_cells: FAILED: {cell}: metric {key} is {v!r} -- refusing to write "
            "a cell over nothing")
    print(json.dumps({"cell": cell, "side": side, "run": int(run),
                      "metric": key, "value": float(v)}))
PY
}

# ---- one batched run: vllm bench serve on the group's shared dataset ----------
# Emits one runs.jsonl line per metric: {cell, side, run, metric, value}.
bench_run() {  # bench_run <tag> <cell> <side> <run> <model> <dataset> <conc> <ttft> <ptoks>
  local TAG=$1 CELL=$2 SIDE=$3 RUN=$4 MODEL=$5 DATASET=$6 CONC=$7 TTFT=$8 PTOKS=$9
  local RAW="$WORK/$TAG.r$RUN.json" RUNLOG="$WORK/$TAG.r$RUN.log"
  ( cd /work/spyre-inference && TORCH_DEVICE_BACKEND_AUTOLOAD=0 taskset -c "$CORES" \
      uv run --no-sync vllm bench serve \
      --backend openai --base-url "http://127.0.0.1:$PORT" \
      --model "$SERVED" --tokenizer "$MODEL" \
      --dataset-name custom --dataset-path "$DATASET" \
      --num-prompts 16 --num-warmups 2 --max-concurrency "$CONC" \
      --temperature 0 --ignore-eos --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --disable-shuffle --disable-tqdm \
      --save-result --result-dir "$(dirname "$RAW")" --result-filename "$(basename "$RAW")" ) \
    > "$RUNLOG" 2>&1
  # The same #677 length gate as rung-1's, on the serve's own record: every
  # request this run sent (2 warmups + 16 prompts) must have arrived within 2%
  # of the bucket's prompt tokens. "-" leaves it off — the ~1k groups
  # predate it, and their dataset's serve record sits ~3% off the site's
  # bucket number (conc-1024.jsonl is torch-scale 1083 vs ~1k's 1051), so
  # gating them means pinning their own target first.
  [ "$PTOKS" = - ] || length_gate "$CELL" 18 "$PTOKS"
  # vllm's key spellings move between versions: try the known ones, fail
  # loudly listing the file's own keys — never guess a key.
  python3 - "$CELL" "$SIDE" "$RUN" "$RAW" "$TTFT" >> "$RUNS_JSONL" <<'PY'
import json, math, sys
cell, side, run, path, ttft = sys.argv[1:6]
try:
    with open(path) as f:
        result = json.load(f)
except (OSError, ValueError) as e:
    raise SystemExit(f"pr_cells: FAILED: {cell}: bench result is not JSON ({e})")
def first(keys, what):
    for k in keys:
        v = result.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            return float(v)
    raise SystemExit(
        f"pr_cells: FAILED: {cell}: no {what} in keys {sorted(result)}; add the "
        "spelling this vllm version uses and re-run")
# TPOT's p50 where reported, else the mean: excludes the first token, same as
# the streaming harness's inter-chunk p50. tok/s is output throughput —
# request_throughput is req/s, ~256x off; fail rather than emit it as tok/s.
metrics = [
    ("itl_ms", ["median_tpot_ms", "p50_tpot_ms", "mean_tpot_ms",
                "median_itl_ms", "p50_itl_ms", "mean_itl_ms", "tpot_ms"],
     "TPOT/ITL (ms)"),
    ("tok_s", ["output_throughput", "mean_output_throughput",
               "total_output_throughput"], "output throughput (tok/s)"),
]
if ttft == "yes":
    # The ~10k groups carry TTFT too — the legs' own three metrics; half the
    # long-context starvation story sits in it.
    metrics.append(("ttft_ms", ["median_ttft_ms", "p50_ttft_ms", "mean_ttft_ms",
                                "ttft_ms"], "TTFT (ms)"))
for metric, keys, what in metrics:
    print(json.dumps({"cell": cell, "side": side, "run": int(run),
                      "metric": metric, "value": first(keys, what)}))
PY
}

# ---- measure: per cell, main's boot then the PR's, back to back ---------------
measure_cell() {  # measure_cell <cell> <kind> <main-bin> <pr-bin> <extra...>
  local CELL=$1 KIND=$2; shift 2
  local MAIN_BIN=$1 PR_BIN=$2; shift 2
  local BIN TAG
  for SIDE in main pr; do
    if [ "$SIDE" = main ]; then BIN=$MAIN_BIN; else BIN=$PR_BIN; fi
    TAG="$CELL.$SIDE"
    echo "== $CELL $SIDE ($(date -u +%H:%M:%SZ))" >&2
    if [ "$KIND" = rung1 ]; then
      local PTOKS=$1
      serve_up "$BIN" "$M2B" 2 34816 yes "$TAG"
      for RUN in $(seq 1 "$RUNS"); do rung1_run "$TAG" "$CELL" "$SIDE" "$RUN" "$PTOKS"; done
    else
      local MODEL=$1 MNS=$2 MML=$3 DATASET=$4 CONC=$5 TTFT=$6 PTOKS=$7
      serve_up "$BIN" "$MODEL" "$MNS" "$MML" no "$TAG"
      for RUN in $(seq 1 "$RUNS"); do bench_run "$TAG" "$CELL" "$SIDE" "$RUN" "$MODEL" "$DATASET" "$CONC" "$TTFT" "$PTOKS"; done
    fi
    serve_down
  done
}

# ---- run the selected cells, each main's boot then the PR's, back to back -----
SELECTED=""  # the cell ids that ran, in grid order — the assembler's expected set
if [ "$WANT_R1_2B" = yes ]; then
  measure_cell r1-2b-16-1k  rung1 "$B2_MAIN" "$B2_PR" 1061
  measure_cell r1-2b-16-10k rung1 "$B2_MAIN" "$B2_PR" 10091
  SELECTED="$SELECTED r1-2b-16-1k r1-2b-16-10k"
fi
if [ "$WANT_C8_2B" = yes ]; then
  measure_cell c8-2b-16-1k  bench "$B2_MAIN" "$B2_PR" "$M2B"  8 12288 "$DS_1K"  8 no -
  SELECTED="$SELECTED c8-2b-16-1k"
fi
if [ "$WANT_C8_8B" = yes ]; then
  measure_cell c8-8b-16-1k  bench "$B8_MAIN" "$B8_PR" "$M8B"  8 8704  "$DS_1K"  8 no -
  SELECTED="$SELECTED c8-8b-16-1k"
fi
# The long-context groups, from the 10-19 sweep's own G3/G4 legs: mns and
# offered concurrency 4 and 8, mml 12288, the shared ~10k dataset. The 10300
# is cells.tsv's own bucket target for these cells — the length gate holds
# every request the bench sends to it.
if [ "$WANT_C4_10K_2B" = yes ]; then
  measure_cell c4-2b-16-10k bench "$B2_MAIN" "$B2_PR" "$M2B" 4 12288 "$DS_10K" 4 yes 10300
  SELECTED="$SELECTED c4-2b-16-10k"
fi
if [ "$WANT_C8_10K_2B" = yes ]; then
  measure_cell c8-2b-16-10k bench "$B2_MAIN" "$B2_PR" "$M2B" 8 12288 "$DS_10K" 8 yes 10300
  SELECTED="$SELECTED c8-2b-16-10k"
fi

# ---- assemble the progress entry, enforcing the site's own laws ---------------
# The cell ids above carry the sweep grid's coordinates; this maps them to
# the file's cells (the metric comes from each run line). Each headline is
# its side's LAST run; every side needs RUNS runs; both sides the same
# count — the same laws the site's loader enforces, so a file this script
# writes cannot be one the page refuses.
python3 - "$PR" "$TITLE" "$HEAD" "$BASE" "$RUNS_JSONL" "$DEST" "$(hostname)" "$RUNS" "$CORES" "$SELECTED" <<'PY'
import collections
import datetime
import json
import os
import sys

pr, title, head, base, runs_path, dest, pod, runs, cores, selected = sys.argv[1:11]
pr = int(pr)
selected = selected.split()

# cell id -> (model, dtype, conc, context, note)
GRID = {
    "r1-2b-16-1k":  ("granite-3.1-2b", "fp16", 1, "~1k",
                     "rung-1: the length-calibrated streaming client, median of 3"),
    "r1-2b-16-10k": ("granite-3.1-2b", "fp16", 1, "~10k",
                     "rung-1: the length-calibrated streaming client, median of 3"),
    "c8-2b-16-1k":  ("granite-3.1-2b", "fp16", 8, "~1k",
                     "conc-8: vllm bench serve on the shared ~1k dataset, 16 prompts, out 256"),
    "c8-8b-16-1k":  ("granite-3.1-8b", "fp16", 8, "~1k",
                     "conc-8: vllm bench serve on the shared ~1k dataset, 16 prompts, out 256"),
    "c4-2b-16-10k": ("granite-3.1-2b", "fp16", 4, "~10k",
                     "conc-4: vllm bench serve on the shared ~10k dataset, 16 prompts, out 256"),
    "c8-2b-16-10k": ("granite-3.1-2b", "fp16", 8, "~10k",
                     "conc-8: vllm bench serve on the shared ~10k dataset, 16 prompts, out 256"),
}

# The metrics each cell id carries: the rung-1 client reports TTFT and ITL;
# the ~1k bench reports tok/s and TPOT; the ~10k bench carries the legs' own
# three, TTFT included — half the long-context starvation story sits in it.
CELL_METRICS = {
    "r1-2b-16-1k": ("ttft_ms", "itl_ms"),
    "r1-2b-16-10k": ("ttft_ms", "itl_ms"),
    "c8-2b-16-1k": ("tok_s", "itl_ms"),
    "c8-8b-16-1k": ("tok_s", "itl_ms"),
    "c4-2b-16-10k": ("tok_s", "itl_ms", "ttft_ms"),
    "c8-2b-16-10k": ("tok_s", "itl_ms", "ttft_ms"),
}
if not selected:
    print("pr_cells: REFUSED: no cells were selected -- verdict over nothing "
          "must fail", file=sys.stderr)
    sys.exit(1)
# Exactly the selected cells, no more, no less — a subset gate is a subset
# file, and a dropped cell still fails the run.
expected = {(cell, metric) for cell in selected for metric in CELL_METRICS[cell]}

runs_needed = int(runs)
by = collections.defaultdict(list)  # (cell, metric, side) -> [(run, value)]
with open(runs_path) as f:
    for line in f:
        r = json.loads(line)
        by[(r["cell"], r["metric"], r["side"])].append((r["run"], r["value"]))

cells = {}
problems = []
for (cell_id, metric, side), got in sorted(by.items()):
    key = (cell_id, metric)
    if cell_id not in GRID:
        problems.append(f"unknown cell id '{cell_id}' in the runs")
        continue
    model, dtype, conc, context, note = GRID[cell_id]
    got.sort()
    values = [v for _, v in got]
    if len(values) != runs_needed:
        problems.append(f"{cell_id} {metric} {side}: {len(values)} run(s), need {runs_needed}")
    cells.setdefault(key, {"model": model, "dtype": dtype, "metric": metric,
                           "conc": conc, "context": context, "note": note,
                           "main": None, "pr": None,
                           "reps": {"main": [], "pr": []}})
    cells[key]["reps"][side] = values
    cells[key][side] = values[-1]  # the headline is the last run

for key, c in sorted(cells.items()):
    for side in ("main", "pr"):
        if not c["reps"][side] or c[side] is None:
            problems.append(f"{key[0]} {key[1]}: no {side} run at all")
        elif c[side] != c["reps"][side][-1]:
            problems.append(f"{key[0]} {key[1]}: {side} headline is not its last run")
    if len(c["reps"]["main"]) != len(c["reps"]["pr"]):
        problems.append(f"{key[0]} {key[1]}: main and pr run counts differ")

if set(cells) != expected:
    missing = expected - set(cells)
    extra = set(cells) - expected
    if missing:
        problems.append(f"missing cells: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected cells: {sorted(extra)}")
if problems:
    for p in problems:
        print(f"pr_cells: REFUSED: {p}", file=sys.stderr)
    sys.exit(1)

entry = {
    "schema": 1,
    "pr": pr,
    "title": title,
    "head": head,
    "base": base,
    "merged": datetime.date.today().isoformat(),
    "pod": pod,
    "harness": (f"scripts/spyre_pr_cells.sh: rung-1 via the length-calibrated "
                f"streaming client (3 streamed requests, medians, the serve's "
                f"request-finished lines the length gate) and the batched cells "
                f"(conc 4 and 8, ~1k and ~10k) via vllm bench serve (16 "
                f"prompts, 256 output tokens) on the shared datasets; "
                f"{runs_needed} back-to-back runs a side on one boot "
                f"each, headline = the last run; serve and client pinned to "
                f"cores {cores}"),
    "cells": [cells[k] for k in sorted(cells)],
}
tmp = dest + ".tmp"
with open(tmp, "w") as f:
    json.dump(entry, f, indent=2)
    f.write("\n")
os.replace(tmp, dest)
print(dest)
PY
echo "OK: wrote $DEST ($(date -u +%H:%M:%SZ))"
