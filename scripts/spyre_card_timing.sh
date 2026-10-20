#!/usr/bin/env bash
# A fast card-side timing check for ONE fresh spyre binary — the flex-rs
# pure-Triton granite shas — in about two minutes (2b; 8b takes two boots,
# below), next to the lead's own chat check. It is a triage baseline, not a
# gate cell and not a sweep cell: it writes nothing (stdout is the report,
# progress on stderr) and never touches site/.
#
# What it measures, granite-3.1-2b or -8b fp16, on the candidate's boot(s):
#   ~1k  solo  TTFT and ITL — the rung-1 streaming client
#   ~10k solo  TTFT and ITL — the rung-1 streaming client
#   conc-8 ~1k ITL p50 (TPOT) and tok/s — vllm bench serve on the shared
#                             ~1k dataset, the first 8 of its 16 rows, one
#                             wave. TPOT is the ratio row: a per-token
#                             measure, comparable against torch's 256-out
#                             cell; tok/s is printed alongside with NO
#                             torch ratio, because at 64 output tokens it
#                             is prefill-diluted (a proof run read 51.3 vs
#                             torch's 86.4 tok/s — not comparable).
# An opt-in group, --cells long (2b only), adds the batched ~10k cells —
# conc-4 and conc-8 at ~10k, the sweep's 0.39x/0.31x TPOT cells — each as
# ONE wave of `conc` requests on the shared ~10k dataset (its first conc
# rows), 64 output tokens, one warm-up, on its own boot at the sweep's own
# shapes (conc-4: mns 4; conc-8: mns 8; both mml 12288). TPOT p50 is each
# cell's torch ratio row; TTFT p99 is reported alongside with NO torch
# ratio — the sweep's torch TTFT cells are medians, and the p99 tail is
# where #701's wave-admission fix shows. These cells are prefill-heavy (a
# wave is conc x ~7 s of prefill before any decode), so the group is not
# fast and the wall is stated, not trimmed: expect roughly 5-6 min beyond
# the base run (two more boots; conc-4 bench ~1.5 min, conc-8 ~2.5-3 min).
#
# Each measured run is preceded by exactly one warm-up request (a streamed
# curl with filler text for the solo cells, vllm bench's own --num-warmups
# for the batched cell) and is the only run — no reps, no medians across
# runs. Every measured request is length-gated on the serve's own record
# (#677): 1061 / 10091 / 1083 within 2%.
#
# Boots, per model — pinned, never guessed at runtime:
#   2b: ONE boot, mns 8 mml 12288, hosts every cell (validated on the
#       card: solo cells matched the sweep within ~3%).
#   8b: TWO boots. The KV pool (273 pages of 60 MB) cannot host one boot
#       for every cell: mml 12288 at mns 8 needs 384 pages, and the mml
#       8704 that mns 8 fits cannot host the ~10k prompt (10155 tokens).
#       So the solo cells run mns 2 mml 12288, and the conc-8 cell runs
#       the sweep's own 8b batched shape, mns 8 mml 8704.
# Prefix caching is OFF on every boot: the rung-1 base prompts nest
# (p400 ⊂ q3700 ⊂ n10k are slices of one document), so a caching boot
# would fake the ~10k solo TTFT with the ~1k cell's prefix. What that
# costs, and every other way this is not the sweep's cell, is printed in
# the report's footer — the ratios pair with the newest sweep's torch
# cells, so the pairing's diffs travel with the numbers.
#
# The candidate and reference binaries are both required, and the script
# refuses when they are the same bytes: a "new" sha that builds identical
# to what is already characterized is a stale handoff, not a measurement.
# The reference is never booted — only compared by bytes.
#
# Usage:
#   scripts/spyre_card_timing.sh <candidate-scr> <reference-scr> \
#       [--model 2b|8b] [--cells long] [--port N]
#   scripts/spyre_card_timing.sh --self-test
#
# Environment:
#   DS_1K   the shared conc-8 ~1k dataset (default
#           /work/fronts/ab/ds/conc-1024.jsonl, md5-pinned to the sweep's
#           own file — the batched cell runs its first 8 rows)
#   DS_10K  the shared ~10k dataset, needed only for --cells long (default
#           /work/fronts/ab/ds/conc-10300.jsonl, md5-pinned to the sweep's
#           own file — the long cells run its first conc rows)
#   CORES   the cores serve and client are pinned to (default 48-95)
#
# Exits non-zero on any refusal; a failed cell fails the whole run. On the
# pod, run it from a synced checkout that carries site/data/spyre/ (the
# torch ratios come from the newest sweep file there).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"

CAND="" REF="" PORT=8079 MODEL_KIND="" CELLS="" SELF_TEST=no
usage() { awk 'NR == 1 { next } /^#/ { print substr($0, 3); next } { exit }' "$0" >&2; exit 64; }
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) MODEL_KIND=$2; shift 2 ;;
      --cells) CELLS=$2; shift 2 ;;
      --port) PORT=$2; shift 2 ;;
      --self-test) SELF_TEST=yes; shift ;;
      -*) echo "REFUSING: unknown argument '$1'" >&2; usage ;;
      *)
        if [ -z "$CAND" ]; then CAND=$1
        elif [ -z "$REF" ]; then REF=$1
        else echo "REFUSING: unexpected argument '$1' — usage is <candidate-scr> <reference-scr>" >&2; usage
        fi
        shift ;;
    esac
  done
  case "$MODEL_KIND" in
    "") MODEL_KIND=2b ;;
    2b|8b) ;;
    *) echo "REFUSING: --model must be 2b or 8b, got '$MODEL_KIND'" >&2; exit 64 ;;
  esac
  case "$CELLS" in
    ""|long) ;;
    *) echo "REFUSING: --cells must be long, got '$CELLS'" >&2; exit 64 ;;
  esac
  if [ "$CELLS" = long ] && [ "$MODEL_KIND" = 8b ]; then
    echo "REFUSING: --cells long is 2b-only — the sweep has no 8b ~10k conc torch cells to pair (its KV budget at ~10k fits ≤2 concurrent seqs), and mml 12288 at mns 8 exceeds the 8b pool" >&2
    exit 64
  fi
}
parse_args "$@"

refuse_blank() { [ -n "$2" ] || { echo "REFUSING: $1 is required" >&2; exit 64; }; }

# A candidate that is the same bytes as the reference would re-measure a
# binary that is already characterized and print it as news. Refuse it.
refuse_identical() {  # refuse_identical <candidate-bin> <reference-bin>
  if cmp -s "$1" "$2"; then
    echo "REFUSING: the candidate and reference binaries are identical (same bytes, so the same md5) — a stale handoff, not a measurement" >&2
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
# within 2% of the target — never print a number over a prompt measured at
# the wrong length. The solo runs gate their one measured request; the
# batched run gates everything it sent (1 warmup + 8 prompts). Each boot
# gets its own serve log, so the tail never reaches a previous boot's lines.
length_gate() {  # length_gate <cell> <want> <target>
  local CELL=$1 WANT_N=$2 TARGET=$3 RECS
  RECS=$(grep -oE 'prompt_tokens=[0-9]+' "$SRVLOG" | tail -"$WANT_N" | cut -d= -f2 | tr '\n' ' ')
  python3 - "$CELL" "$RECS" "$TARGET" "$WANT_N" <<'PY'
import sys
cell, recs_s, target_s, want_s = sys.argv[1:5]
recs = [int(x) for x in recs_s.split()]
target, want = int(target_s), int(want_s)
if len(recs) != want:
    print(f"card_timing: REFUSED: {cell}: expected {want} serve request lines, got "
          f"{len(recs)} -- verdict over nothing must fail", file=sys.stderr)
    sys.exit(1)
bad = [r for r in recs if abs(r - target) / target > 0.02]
if bad:
    print(f"card_timing: REFUSED: {cell}: serve recorded prompt_tokens {bad} off "
          f"target {target} by >2% -- never print a number over a prompt measured "
          "at the wrong length", file=sys.stderr)
    sys.exit(1)
PY
}

# The report: the measured numbers against the newest sweep's torch cells,
# with the pairing's diffs in the footer. The conc-8 ratio row is ITL p50
# (TPOT); the tok/s line carries no torch ratio (prefill-diluted at 64 out).
# Refuses over a missing metric or a missing torch cell — never half a
# report. Stdout only.
emit_report() {  # emit_report <runs.jsonl> <sweep.json> <model-short> <serve-shapes> <cand> <cand-md5> <ref> <ref-md5> <host> <date> <boot-s> <boots> <w1k-s> <m1k-s> <w10k-s> <m10k-s> <bench-s> <c4-10k-s> <c8-10k-s> <total-s>
  python3 - "$@" <<'PY'
import json, math, sys

(runs_path, sweep_path, short, shapes, cand, cand_md5, ref, ref_md5, host,
 date_s, boot_s, boots, w1k_s, m1k_s, w10k_s, m10k_s, bench_s, c4_s, c8_s,
 total_s) = sys.argv[1:21]

DTYPE = "fp16"
# cell -> (conc, context); the metric comes from each run line.
GRID = {"r1-1k": (1, "~1k"), "r1-10k": (1, "~10k"), "c8-1k": (8, "~1k"),
        "c4-10k": (4, "~10k"), "c8-10k": (8, "~10k")}
ROWS = [  # the ratio rows, in print order — the report is complete or it refuses
    ("r1-1k", "ttft_ms", "TTFT  ~1k   solo", "ms"),
    ("r1-1k", "itl_ms", "ITL   ~1k   solo", "ms"),
    ("r1-10k", "ttft_ms", "TTFT  ~10k  solo", "ms"),
    ("r1-10k", "itl_ms", "ITL   ~10k  solo", "ms"),
    ("c8-1k", "itl_ms", "ITL   ~1k  conc-8", "ms"),
]
# The opt-in long group (--cells long): per cell, the TPOT ratio row and the
# unpaired TTFT p99 line. Present only in a long run; all four metrics or
# none — the report refuses a half-measured group.
LONG_ROWS = [
    ("c4-10k", "ITL   ~10k conc-4", "TTFT  ~10k conc-4 p99"),
    ("c8-10k", "ITL   ~10k conc-8", "TTFT  ~10k conc-8 p99"),
]

def refuse(msg):
    print(f"card_timing: REFUSED: {msg}", file=sys.stderr)
    sys.exit(1)

runs = {}
with open(runs_path) as f:
    for line in f:
        r = json.loads(line)
        key = (r["cell"], r["metric"])
        if key in runs:
            refuse(f"{key} measured twice")
        v = r["value"]
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            refuse(f"{key} is {v!r} -- never a number over nothing")
        runs[key] = float(v)
expected = {(c, m) for c, m, _, _ in ROWS} | {("c8-1k", "tok_s")}
long_present = any(c in ("c4-10k", "c8-10k") for c, _ in runs)
if long_present:
    expected |= {("c4-10k", "itl_ms"), ("c4-10k", "ttft_p99_ms"),
                 ("c8-10k", "itl_ms"), ("c8-10k", "ttft_p99_ms")}
if set(runs) != expected:
    refuse(f"the measured set is {sorted(runs)}, need exactly {sorted(expected)}")

try:
    with open(sweep_path) as f:
        sweep = json.load(f)
except (OSError, ValueError) as e:
    refuse(f"cannot read the sweep file {sweep_path} ({e})")
torch = {}
for c in sweep.get("cells", []):
    if c.get("model") == short and c.get("dtype") == DTYPE:
        torch[(c.get("conc", 1), c["context"], c["metric"])] = c["torch"]
ratio_rows = [(c, m) for c, m, _, _ in ROWS]
if long_present:
    ratio_rows += [(c, "itl_ms") for c, _, _ in LONG_ROWS]
for cell, metric in ratio_rows:
    if (GRID[cell][0], GRID[cell][1], metric) not in torch:
        refuse(f"the sweep file {sweep_path} has no torch cell for "
               f"{short} {DTYPE} {GRID[cell]} {metric} -- cannot print its ratio")

measured = sweep.get("torch_spyre", {}).get("measured", "?")
print(f"spyre card timing -- {short} {DTYPE} -- {date_s} -- {host}")
print(f"candidate: {cand} (md5 {cand_md5})")
print(f"reference: {ref} (md5 {ref_md5}) -- differs")
print(f"torch:     {sweep_path.rsplit('/', 1)[-1]} (torch side measured {measured})")
print()
for cell, metric, label, unit in ROWS:
    conc, context = GRID[cell]
    v = runs[(cell, metric)]
    t = torch[(conc, context, metric)]
    # ms is lower-better (torch/scratchy): >1 is scratchy ahead.
    ratio = t / v
    ahead = ("scratchy ahead" if ratio >= 1.02 else
             "torch ahead" if ratio <= 0.98 else "near parity")
    print(f"{label}: {v:>10.2f} {unit:<5} torch {t:>10.2f}   {ratio:.2f}x ({ahead})")
tok = runs[("c8-1k", "tok_s")]
print(f"tok/s ~1k  conc-8: {tok:>10.2f}       (no torch ratio: at 64 out it is "
      "prefill-diluted against torch's 256-out cell)")
if long_present:
    print()
    for cell, itl_label, p99_label in LONG_ROWS:
        conc, context = GRID[cell]
        v = runs[(cell, "itl_ms")]
        t = torch[(conc, context, "itl_ms")]
        ratio = t / v
        ahead = ("scratchy ahead" if ratio >= 1.02 else
                 "torch ahead" if ratio <= 0.98 else "near parity")
        print(f"{itl_label}: {v:>10.2f} ms    torch {t:>10.2f}   {ratio:.2f}x ({ahead})")
        p99 = runs[(cell, "ttft_p99_ms")]
        print(f"{p99_label}: {p99:>10.2f} ms    (no torch ratio: torch's cell is "
              "the median TTFT — the p99 tail is where #701's fix shows)")
print()
print(f"timings: {'boots' if boots != '1' else 'boot'} {boot_s}s ({boots}) | "
      f"warm ~1k {w1k_s}s | ~1k run {m1k_s}s | "
      f"warm ~10k {w10k_s}s | ~10k run {m10k_s}s | conc-8 bench {bench_s}s | "
      f"total {total_s}s")
if long_present:
    print(f"long cells: c4-10k bench {c4_s}s | c8-10k bench {c8_s}s")
print()
print(f"rules: {shapes}, prefix caching off, 64 output")
print("  tokens, one warm-up request + one measured run per cell, every")
print("  measured request's serve-recorded prompt_tokens gated to 2% (1061")
print("  / 10091 / 1083); the reference binary was compared by bytes only.")
if short.endswith("-8b"):
    print("  (8b's two boots: the KV pool is 273 pages of 60 MB — mml 12288 at")
    print("  mns 8 needs 384 pages, and the mml 8704 that mns 8 fits cannot host")
    print("  the ~10k prompt, so each cell takes the boot where it fits; the")
    print("  conc-8 boot is the sweep's own 8b batched shape)")
print("differs from the sweep cells these ratios pair with: out 64 (they ran")
print("  256), one run (they ran 2 reps, steady state = rep 2), prefix caching")
print("  off (their conc-8 cells ran caching on; the solo cells need it off —")
print("  the staged base prompts nest, a caching boot would fake ~10k TTFT),")
print("  the solo boot's mns/mml (the sweep's rung-1 cells ran mns 2, mml")
print("  34816), and conc-8 = the first 8 of the shared 16 rows. The conc-8")
print("  ratio row is ITL p50 (TPOT) — a per-token measure, comparable against")
print("  torch's 256-out cell; the tok/s line carries no torch ratio because at")
print("  64 out it is prefill-diluted. A triage baseline, not a gate cell.")
if long_present:
    print("the ~10k conc cells pair with the sweep's 16-prompt cells: one wave")
    print("  of conc (not 16 prompts), out 64 (they ran 256), one warm-up (they")
    print("  ran two), each on its own boot at the sweep's own shapes (mns 4 /")
    print("  mns 8, mml 12288); their TTFT p99 lines carry no torch ratio — the")
    print("  sweep's torch TTFT cells are medians, and the p99 tail is where")
    print("  #701's wave-admission fix shows")
PY
}

# ---- self-test: the parser, the guards, the gates and the report -------------
# No pod, no card: the same bar as scripts/spyre_pr_cells.sh's --self-test,
# plus the report assembler on fixtures — the numbers, ratios and refusals
# the pod run will print, checked before any card time.
self_test() {
  local fails=0
  check() {  # check <what> <got> <want>
    if [ "$2" = "$3" ]; then
      echo "ok: $1"
    else
      echo "FAIL: $1: got '$2', want '$3'" >&2
      fails=$((fails + 1))
    fi
  }
  parse_args /bin/cand /bin/ref --model 8b --port 8091
  check 'the first positional lands in the candidate slot' "$CAND" /bin/cand
  check 'the second positional lands in the reference slot' "$REF" /bin/ref
  check '--model is kept' "$MODEL_KIND" 8b
  check '--port is kept' "$PORT" 8091
  CAND=""; REF=""
  MODEL_KIND=""
  parse_args /bin/cand /bin/ref
  check '--model defaults to 2b' "$MODEL_KIND" 2b
  if ( parse_args --model 7b ) 2>/dev/null; then
    echo "FAIL: --model 7b was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: --model 7b is refused"
  fi
  CAND=""; REF=""
  parse_args /bin/cand /bin/ref --cells long
  check '--cells long is kept' "$CELLS" long
  # A call site that lost its arguments in a refactor dies only on the pod,
  # under set -u, at the moment the cell starts (the 8b conc-8 cell shipped
  # bare once). No call in this script may be bare.
  bare=$(grep -nE '^[[:space:]]*(bench_run|rung1_run)[[:space:]]*$' "$0" || true)
  check 'no bare bench_run/rung1_run call site' "$bare" ""
  if ( parse_args --cells bogus ) 2>/dev/null; then
    echo "FAIL: --cells bogus was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: --cells bogus is refused"
  fi
  if ( parse_args --model 8b --cells long ) 2>/dev/null; then
    echo "FAIL: --model 8b --cells long was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: --model 8b --cells long is refused (2b-only group)"
  fi
  CELLS=""
  if ( parse_args --bogus ) 2>/dev/null; then
    echo "FAIL: an unknown flag was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: an unknown flag is refused"
  fi
  if ( parse_args a b c ) 2>/dev/null; then
    echo "FAIL: a third positional was not refused" >&2; fails=$((fails + 1))
  else
    echo "ok: a third positional is refused"
  fi
  CAND=""; REF=""
  parse_args /bin/only
  check 'one positional leaves the reference blank' "$REF" ""
  # The identical-binary guard: a stale handoff is not a measurement.
  local T1 T2
  T1=$(mktemp); T2=$(mktemp)
  printf 'same' > "$T1"; cp "$T1" "$T2"
  if ( refuse_identical "$T1" "$T2" ) 2>/dev/null; then
    echo "FAIL: the identical-binary guard accepted identical binaries" >&2; fails=$((fails + 1))
  else
    echo "ok: the identical-binary guard refuses identical binaries"
  fi
  printf 'other' > "$T2"
  if ( refuse_identical "$T1" "$T2" ) 2>/dev/null; then
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
  # verdict over missing lines. want=1 is a solo run; want=9 is the bench run.
  local SRVLOG_OLD=${SRVLOG:-}
  SRVLOG=$(mktemp)
  printf 'request finished prompt_tokens=1061\n' > "$SRVLOG"
  if ( length_gate selftest 1 1061 ) 2>/dev/null; then
    echo "ok: the length gate accepts an on-target solo request"
  else
    echo "FAIL: the length gate refused an on-target solo request" >&2; fails=$((fails + 1))
  fi
  printf 'request finished prompt_tokens=900\n' >> "$SRVLOG"
  if ( length_gate selftest 1 1061 ) 2>/dev/null; then
    echo "FAIL: the length gate gated the warm-up's heuristic line into the run" >&2
    fails=$((fails + 1))
  else
    echo "ok: the length gate refuses a wrong-length request"
  fi
  printf 'request finished prompt_tokens=1083\n%.0s' 1 2 3 4 5 6 7 8 9 > "$SRVLOG"
  if ( length_gate selftest 9 1083 ) 2>/dev/null; then
    echo "ok: the length gate accepts 9 on-target bench requests"
  else
    echo "FAIL: the length gate refused on-target bench requests" >&2; fails=$((fails + 1))
  fi
  printf 'request finished prompt_tokens=1083\n%.0s' 1 2 3 4 5 6 7 8 > "$SRVLOG"
  if ( length_gate selftest 9 1083 ) 2>/dev/null; then
    echo "FAIL: the length gate accepted missing bench request lines" >&2; fails=$((fails + 1))
  else
    echo "ok: the length gate refuses missing bench request lines"
  fi
  rm -f "$SRVLOG"
  SRVLOG=$SRVLOG_OLD
  # The report, 2b: the ratio rows against the sweep's torch cells, the
  # conc-8 TPOT ratio row, the unpaired tok/s line, and the refusals.
  local FIX OUT
  FIX=$(mktemp -d)
  cat > "$FIX/sweep-2b.json" <<'JSON'
{"schema": 1, "torch_spyre": {"measured": "2026-10-19"},
 "cells": [
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "ttft_ms", "conc": 1, "context": "~1k", "torch": 458.0},
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "itl_ms", "conc": 1, "context": "~1k", "torch": 63.0},
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "ttft_ms", "conc": 1, "context": "~10k", "torch": 8227.0},
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "itl_ms", "conc": 1, "context": "~10k", "torch": 75.0},
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "itl_ms", "conc": 8, "context": "~1k", "torch": 88.88},
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "itl_ms", "conc": 4, "context": "~10k", "torch": 208.216346245186},
  {"model": "granite-3.1-2b", "dtype": "fp16", "metric": "itl_ms", "conc": 8, "context": "~10k", "torch": 370.1270589299098}
 ]}
JSON
  cat > "$FIX/runs-2b.jsonl" <<'JSONL'
{"cell": "r1-1k", "metric": "ttft_ms", "value": 561.31}
{"cell": "r1-1k", "metric": "itl_ms", "value": 43.56}
{"cell": "r1-10k", "metric": "ttft_ms", "value": 7755.62}
{"cell": "r1-10k", "metric": "itl_ms", "value": 57.57}
{"cell": "c8-1k", "metric": "itl_ms", "value": 91.14}
{"cell": "c8-1k", "metric": "tok_s", "value": 51.30}
JSONL
  OUT=$(emit_report "$FIX/runs-2b.jsonl" "$FIX/sweep-2b.json" granite-3.1-2b \
    "one boot (mns 8, mml 12288)" /bin/cand abc /bin/ref def host 2026-10-19 41 1 4 11 12 19 28 0 0 118)
  check 'the 2b report carries the ~1k TTFT ratio' \
    "$(printf '%s\n' "$OUT" | grep -c 'TTFT  ~1k   solo: .*458\.00.*0\.82x (torch ahead)')" 1
  check 'the 2b report carries the ~1k ITL ratio' \
    "$(printf '%s\n' "$OUT" | grep -c 'ITL   ~1k   solo: .*63\.00.*1\.45x (scratchy ahead)')" 1
  check 'the 2b report carries the ~10k TTFT ratio' \
    "$(printf '%s\n' "$OUT" | grep -c 'TTFT  ~10k  solo: .*8227\.00.*1\.06x (scratchy ahead)')" 1
  check 'the conc-8 ratio row is TPOT against torch' \
    "$(printf '%s\n' "$OUT" | grep -c 'ITL   ~1k  conc-8: .*88\.88.*0\.98x (torch ahead)')" 1
  check 'the conc-8 tok/s line is unpaired' \
    "$(printf '%s\n' "$OUT" | grep -c 'tok/s ~1k  conc-8: .*51\.30.*no torch ratio')" 1
  check 'the report carries the total' \
    "$(printf '%s\n' "$OUT" | grep -c 'total 118s')" 1
  check 'the base report carries no long rows' \
    "$(printf '%s\n' "$OUT" | grep -c 'conc-4')" 0
  # The report, 8b: the same sweep's 8b torch cells and the two-boot footer.
  cat > "$FIX/sweep-8b.json" <<'JSON'
{"schema": 1, "torch_spyre": {"measured": "2026-10-19"},
 "cells": [
  {"model": "granite-3.1-8b", "dtype": "fp16", "metric": "ttft_ms", "conc": 1, "context": "~1k", "torch": 946.27},
  {"model": "granite-3.1-8b", "dtype": "fp16", "metric": "itl_ms", "conc": 1, "context": "~1k", "torch": 148.92},
  {"model": "granite-3.1-8b", "dtype": "fp16", "metric": "ttft_ms", "conc": 1, "context": "~10k", "torch": 13035.0},
  {"model": "granite-3.1-8b", "dtype": "fp16", "metric": "itl_ms", "conc": 1, "context": "~10k", "torch": 164.0},
  {"model": "granite-3.1-8b", "dtype": "fp16", "metric": "itl_ms", "conc": 8, "context": "~1k", "torch": 173.90}
 ]}
JSON
  cat > "$FIX/runs-8b.jsonl" <<'JSONL'
{"cell": "r1-1k", "metric": "ttft_ms", "value": 1057.14}
{"cell": "r1-1k", "metric": "itl_ms", "value": 135.86}
{"cell": "r1-10k", "metric": "ttft_ms", "value": 12329.60}
{"cell": "r1-10k", "metric": "itl_ms", "value": 154.61}
{"cell": "c8-1k", "metric": "itl_ms", "value": 199.31}
{"cell": "c8-1k", "metric": "tok_s", "value": 38.43}
JSONL
  OUT=$(emit_report "$FIX/runs-8b.jsonl" "$FIX/sweep-8b.json" granite-3.1-8b \
    "two boots (solo: mns 2, mml 12288; conc-8: mns 8, mml 8704)" /bin/cand abc /bin/ref def host 2026-10-19 96 2 9 20 22 40 55 0 0 198)
  check 'the 8b report names the model' \
    "$(printf '%s\n' "$OUT" | grep -c 'spyre card timing -- granite-3.1-8b fp16')" 1
  check 'the 8b report carries the ~1k ITL ratio' \
    "$(printf '%s\n' "$OUT" | grep -c 'ITL   ~1k   solo: .*148\.92.*1\.10x (scratchy ahead)')" 1
  check 'the 8b conc-8 ratio row is TPOT against torch' \
    "$(printf '%s\n' "$OUT" | grep -c 'ITL   ~1k  conc-8: .*173\.90.*0\.87x (torch ahead)')" 1
  check 'the 8b report states the two boots' \
    "$(printf '%s\n' "$OUT" | grep -c 'boots 96s (2)')" 1
  check 'the 8b footer carries the pool arithmetic' \
    "$(printf '%s\n' "$OUT" | grep -c '8b.s two boots: the KV pool is 273 pages')" 1
  # The report, 2b + the long group: the conc-4/conc-8 ~10k TPOT ratio rows
  # (the sweep's 0.39x/0.31x cells), the unpaired TTFT p99 lines, the long
  # walls, and the refusals.
  cat > "$FIX/runs-2b-long.jsonl" <<'JSONL'
{"cell": "r1-1k", "metric": "ttft_ms", "value": 561.31}
{"cell": "r1-1k", "metric": "itl_ms", "value": 43.56}
{"cell": "r1-10k", "metric": "ttft_ms", "value": 7755.62}
{"cell": "r1-10k", "metric": "itl_ms", "value": 57.57}
{"cell": "c8-1k", "metric": "itl_ms", "value": 91.14}
{"cell": "c8-1k", "metric": "tok_s", "value": 51.30}
{"cell": "c4-10k", "metric": "itl_ms", "value": 531.20}
{"cell": "c4-10k", "metric": "ttft_p99_ms", "value": 29850.0}
{"cell": "c8-10k", "metric": "itl_ms", "value": 1160.40}
{"cell": "c8-10k", "metric": "ttft_p99_ms", "value": 63210.0}
JSONL
  OUT=$(emit_report "$FIX/runs-2b-long.jsonl" "$FIX/sweep-2b.json" granite-3.1-2b \
    "one boot (mns 8, mml 12288) + long boots (c4-10k: mns 4, mml 12288; c8-10k: mns 8, mml 12288)" \
    /bin/cand abc /bin/ref def host 2026-10-19 41 3 4 11 12 19 28 76 149 298)
  check 'the long report carries the conc-4 ~10k TPOT ratio' \
    "$(printf '%s\n' "$OUT" | grep -c 'ITL   ~10k conc-4: .*208\.22.*0\.39x (torch ahead)')" 1
  check 'the long report carries the conc-8 ~10k TPOT ratio' \
    "$(printf '%s\n' "$OUT" | grep -c 'ITL   ~10k conc-8: .*370\.13.*0\.32x (torch ahead)')" 1
  check 'the conc-4 TTFT p99 line is unpaired' \
    "$(printf '%s\n' "$OUT" | grep -c 'TTFT  ~10k conc-4 p99: .*29850\.00.*no torch ratio')" 1
  check 'the conc-8 TTFT p99 line is unpaired' \
    "$(printf '%s\n' "$OUT" | grep -c 'TTFT  ~10k conc-8 p99: .*63210\.00.*no torch ratio')" 1
  check 'the long cells print their own walls' \
    "$(printf '%s\n' "$OUT" | grep -c 'long cells: c4-10k bench 76s | c8-10k bench 149s')" 1
  check 'the long report states three boots' \
    "$(printf '%s\n' "$OUT" | grep -c 'boots 41s (3)')" 1
  check 'the long footer states the pairing diffs' \
    "$(printf '%s\n' "$OUT" | grep -c 'the ~10k conc cells pair with the sweep.s 16-prompt cells')" 1
  grep -v '"cell": "c8-10k"' "$FIX/runs-2b-long.jsonl" > "$FIX/runs-half-long.jsonl"
  if ( emit_report "$FIX/runs-half-long.jsonl" "$FIX/sweep-2b.json" granite-3.1-2b "one boot" /bin/cand abc /bin/ref def host 2026-10-19 1 1 1 1 1 1 1 0 0 1 ) 2>/dev/null; then
    echo "FAIL: the report accepted a half-measured long group" >&2; fails=$((fails + 1))
  else
    echo "ok: the report refuses a half-measured long group"
  fi
  python3 - "$FIX/sweep-2b.json" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    s = json.load(f)
s["cells"] = [c for c in s["cells"]
              if not (c["metric"] == "itl_ms" and c.get("conc") == 4
                      and c["context"] == "~10k")]
with open(p + ".nolongtorch", "w") as f:
    json.dump(s, f)
PY
  if ( emit_report "$FIX/runs-2b-long.jsonl" "$FIX/sweep-2b.json.nolongtorch" granite-3.1-2b "one boot" /bin/cand abc /bin/ref def host 2026-10-19 1 1 1 1 1 1 1 0 0 1 ) 2>/dev/null; then
    echo "FAIL: the report accepted a long row with no torch cell" >&2; fails=$((fails + 1))
  else
    echo "ok: the report refuses a long row with no torch cell"
  fi
  grep -v '"cell": "c8-1k", "metric": "itl_ms"' "$FIX/runs-2b.jsonl" > "$FIX/runs-missing.jsonl"
  if ( emit_report "$FIX/runs-missing.jsonl" "$FIX/sweep-2b.json" granite-3.1-2b "one boot" /bin/cand abc /bin/ref def host 2026-10-19 1 1 1 1 1 1 1 0 0 1 ) 2>/dev/null; then
    echo "FAIL: the report accepted a missing metric" >&2; fails=$((fails + 1))
  else
    echo "ok: the report refuses a missing metric"
  fi
  python3 - "$FIX/sweep-2b.json" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    s = json.load(f)
s["cells"] = [c for c in s["cells"]
              if not (c["metric"] == "itl_ms" and c.get("conc") == 8)]
with open(p + ".notorch", "w") as f:
    json.dump(s, f)
PY
  if ( emit_report "$FIX/runs-2b.jsonl" "$FIX/sweep-2b.json.notorch" granite-3.1-2b "one boot" /bin/cand abc /bin/ref def host 2026-10-19 1 1 1 1 1 1 1 0 0 1 ) 2>/dev/null; then
    echo "FAIL: the report accepted a missing torch cell" >&2; fails=$((fails + 1))
  else
    echo "ok: the report refuses a missing torch cell"
  fi
  if ( emit_report "$FIX/runs-2b.jsonl" "$FIX/no-such-sweep.json" granite-3.1-2b "one boot" /bin/cand abc /bin/ref def host 2026-10-19 1 1 1 1 1 1 1 0 0 1 ) 2>/dev/null; then
    echo "FAIL: the report accepted a missing sweep file" >&2; fails=$((fails + 1))
  else
    echo "ok: the report refuses a missing sweep file"
  fi
  rm -rf "$FIX" "$T1" "$T2"
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

# ---- guards -------------------------------------------------------------------
refuse_blank 'the candidate binary (<candidate-scr>)' "$CAND"
refuse_blank 'the reference binary (<reference-scr>)' "$REF"
for B in "$CAND" "$REF"; do
  [ -x "$B" ] || { echo "REFUSING: '$B' is not an executable scr binary" >&2; exit 69; }
done
refuse_identical "$CAND" "$REF"
if env | grep -q '^SCRATCHY_SDSC_'; then
  echo "REFUSING: SCRATCHY_SDSC_* knobs are set; this measures shipping behavior" >&2
  exit 66
fi

# The torch ratios come from the newest sweep file — refuse without one
# rather than print a report with holes.
SWEEP=$(ls -1 "$REPO"/site/data/spyre/*.json 2>/dev/null | sort | tail -1 || true)
[ -n "$SWEEP" ] || {
  echo "REFUSING: no sweep file in $REPO/site/data/spyre — the torch ratios come from the newest sweep; run from a synced checkout" >&2
  exit 69
}

# ---- tools live outside a plain exec shell's PATH (leg.sh's loop) -------------
for D in /work/.cargo/bin /root/.local/bin "$HOME/.cargo/bin" "$HOME/.local/bin"; do
  [ -d "$D" ] || continue
  case ":$PATH:" in *":$D:"*) ;; *) PATH="$D:$PATH" ;; esac
done
export PATH
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
DS_1K="${DS_1K:-/work/fronts/ab/ds/conc-1024.jsonl}"
[ -f "$DS_1K" ] || { echo "REFUSING: dataset '$DS_1K' not found on this pod (set DS_1K)" >&2; exit 65; }
# Pinned by content to the sweep's own conc-1024.jsonl (16 rows, the shared
# conc-8 ~1k dataset): the batched cell pairs with the sweep's cells only on
# that exact file.
refuse_md5 "$DS_1K" 1f1653aeac61b307ab876c6c4397818c "DS_1K (the sweep's conc-1024.jsonl)"
# The long group's ~10k dataset, pinned the same way — checked only when the
# group is opted into, so a base run never demands it.
DS_10K=""
if [ "$CELLS" = long ]; then
  DS_10K="${DS_10K:-/work/fronts/ab/ds/conc-10300.jsonl}"
  [ -f "$DS_10K" ] || { echo "REFUSING: dataset '$DS_10K' not found on this pod (set DS_10K)" >&2; exit 65; }
  refuse_md5 "$DS_10K" 68c030b68dbbf93c80a40e7e5aa28766 "DS_10K (the sweep's conc-10300.jsonl)"
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

# nickm3's card pin: 1c:00.0 = /dev/vfio/122; 1d:00.0 is dead node-side, and
# an unpinned serve can land on it (leg.sh's pin, applied by hostname).
case "$(hostname)" in
  nickm3*)
    SENLIB_CFG=/work/i407-n3stage/fp8/senlib_1c.json
    [ -f "$SENLIB_CFG" ] || { echo "REFUSING: $SENLIB_CFG missing (the 1c:00.0 pin)" >&2; exit 69; }
    export SENLIB_DEVEL_CONFIG_FILE="$SENLIB_CFG"
    export PCIDEVICE_IBM_COM_AIU_PF=0000:1c:00.0
    ;;
esac

# ---- the pod env (verbatim, from the legs) ------------------------------------
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HOME=/work/.cache/huggingface

case "$MODEL_KIND" in
  2b) M=ibm-granite/granite-3.1-2b-instruct; SHORT=granite-3.1-2b ;;
  8b) M=ibm-granite/granite-3.1-8b-instruct; SHORT=granite-3.1-8b ;;
esac
CORES="${CORES:-48-95}"
WORK="$(mktemp -d /tmp/spyre-card-timing.XXXXXX)"
RUNS_JSONL="$WORK/runs.jsonl"
T_START=$SECONDS

# ---- serve up/down (the legs' lines; one log a boot) ---------------------------
SRV=""
SRVLOG=""
BOOTS=0
BOOT_S_TOTAL=0
CUR_MML=0
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

# serve_up <mns> <mml> — the shapes are pinned per model in the case below,
# never chosen at runtime. Prefix caching off on every boot: the rung-1
# base prompts nest, so a caching boot would fake the ~10k solo TTFT.
serve_up() {
  local MNS=$1 MML=$2 T0 DEADLINE
  BOOTS=$((BOOTS + 1))
  SRVLOG="$WORK/serve.$BOOTS.log"
  CUR_MML=$MML
  T0=$SECONDS
  taskset -c "$CORES" "$CAND" serve --model "$M" --device spyre \
    --host 0.0.0.0 --port "$PORT" --max-num-seqs "$MNS" --max-model-len "$MML" \
    --no-prefix-caching > "$SRVLOG" 2>&1 &
  SRV=$!
  SERVED=""
  DEADLINE=$((SECONDS + 900))
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
  BOOT_S_TOTAL=$((BOOT_S_TOTAL + SECONDS - T0))
  echo "== boot $BOOTS (mns $MNS, mml $MML), ${BOOT_S_TOTAL}s of boots so far (served id: $SERVED)" >&2
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

# ---- one warm-up request (streamed curl, filler text — discarded, not gated) ---
# Prefix caching is off, so the warm-up's text cannot collide with the
# measured run's; its length is heuristic (a discarded request is never a
# number), only its shape matters: same endpoint, same body shape, 64 tokens.
# The filler is capped where a 1.35-tokens-a-word worst case stays under the
# boot's max-model-len: a warm-up the serve rejects is no warm-up at all,
# and curl -f makes that loud instead of silent (the first cut's ~10k
# warm-up was 13.6k WORDS — over mml in tokens alone, silently rejected).
warmup_curl() {  # warmup_curl <target-tokens>
  python3 - "$SERVED" "$1" "$CUR_MML" > "$WORK/warmup.json" <<'PY'
import json, sys
served, target, mml = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sentence = ("The granite model streams one token at a time while the card "
            "holds every weight in place and the host submits one command "
            "buffer per step. ")
words = sentence.split()
want = min(max(16, int(target * 1.35)), int((mml - 2048) / 1.35))
filler = " ".join([words[i % len(words)] for i in range(want)])
body = {"model": served,
        "messages": [{"role": "system", "content": ""},
                     {"role": "user", "content": filler}],
        "max_tokens": 64, "temperature": 0.0, "ignore_eos": True,
        "stream": True}
print(json.dumps(body))
PY
  curl -sfN --max-time 600 -H 'Content-Type: application/json' \
    --data @"$WORK/warmup.json" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null || {
      echo "FAILED: the warm-up request was rejected — see $SRVLOG" >&2
      tail -5 "$SRVLOG" >&2 || true
      exit 74
    }
}

# ---- one measured solo run: the rung-1 client + the serve-side length gate -----
# Emits one runs.jsonl line per metric: {cell, metric, value}.
rung1_run() {  # rung1_run <cell> <ptoks>
  local CELL=$1 PTOKS=$2
  local RESULT="$WORK/$CELL.json" RUNLOG="$WORK/$CELL.log"
  T0=$SECONDS
  warmup_curl "$PTOKS"
  local WARM_S=$((SECONDS - T0))
  T0=$SECONDS
  ( cd /work/spyre-inference && TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 \
      taskset -c "$CORES" uv run --no-sync python "$HERE/bench_spyre_rung1.py" \
      "http://127.0.0.1:$PORT" "$SERVED" "$PTOKS" 1 64 "$HERE/spyre_rung1_prompts" ) \
    > "$RESULT" 2> "$RUNLOG" || {
      echo "FAILED: $CELL: the rung-1 client failed; last 20 lines of its log:" >&2
      tail -20 "$RUNLOG" >&2 || true
      exit 74
    }
  local RUN_S=$((SECONDS - T0))
  echo "== $CELL: warm ${WARM_S}s, run ${RUN_S}s" >&2
  # The authoritative length gate: the measured request's serve-recorded
  # prompt_tokens, within 2% of the bucket. The warm-up's heuristic line
  # sits before it and is excluded by the tail.
  length_gate "$CELL" 1 "$PTOKS"
  python3 - "$CELL" "$RESULT" >> "$RUNS_JSONL" <<'PY'
import json, math, sys
cell, path = sys.argv[1:3]
try:
    with open(path) as f:
        got = json.load(f)
except (OSError, ValueError) as e:
    raise SystemExit(f"card_timing: FAILED: {cell}: client result is not JSON ({e})")
for key in ("ttft_ms", "itl_ms"):
    v = got.get(key)
    if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise SystemExit(
            f"card_timing: FAILED: {cell}: metric {key} is {v!r} -- refusing to "
            "print a number over nothing")
    print(json.dumps({"cell": cell, "metric": key, "value": float(v)}))
PY
  LAST_WARM_S=$WARM_S
  LAST_RUN_S=$RUN_S
}

# ---- one measured batched run: vllm bench serve, one wave ----------------------
# The legs' client line with the tool's own counts: the first <np> of the
# shared 16 rows (one clean wave of <conc>), 1 warm-up, 64 output tokens.
# The cell name picks the metric set: c8-1k reports TPOT p50 + tok/s; the
# ~10k long cells report TPOT p50 + TTFT p99 (the tail where #701's fix
# shows).
bench_run() {  # bench_run <cell> <dataset> <num-prompts> <max-conc> <gate-want> <gate-target>
  local CELL=$1 DS=$2 NP=$3 CONC=$4 GW=$5 GT=$6
  local RAW="$WORK/$CELL.json" RUNLOG="$WORK/$CELL.log"
  T0=$SECONDS
  ( cd /work/spyre-inference && TORCH_DEVICE_BACKEND_AUTOLOAD=0 taskset -c "$CORES" \
      uv run --no-sync vllm bench serve \
      --backend openai --base-url "http://127.0.0.1:$PORT" \
      --model "$SERVED" --tokenizer "$M" \
      --dataset-name custom --dataset-path "$DS" \
      --num-prompts "$NP" --num-warmups 1 --max-concurrency "$CONC" \
      --custom-output-len 64 \
      --temperature 0 --ignore-eos --seed 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --disable-shuffle --disable-tqdm \
      --save-result --result-dir "$(dirname "$RAW")" --result-filename "$(basename "$RAW")" ) \
    > "$RUNLOG" 2>&1 || {
      echo "FAILED: $CELL: vllm bench serve failed; last 20 lines of its log:" >&2
      tail -20 "$RUNLOG" >&2 || true
      exit 74
    }
  local BENCH_S=$((SECONDS - T0))
  echo "== $CELL: bench ${BENCH_S}s" >&2
  # Everything this run sent (1 warmup + <np> prompts) must have arrived
  # within 2% of the dataset's recorded scale (1083 for the ~1k set, 10300
  # for the ~10k set — the files' rows, not the site's buckets).
  length_gate "$CELL" "$GW" "$GT"
  python3 - "$CELL" "$RAW" >> "$RUNS_JSONL" <<'PY'
import json, math, sys
cell, path = sys.argv[1:3]
try:
    with open(path) as f:
        result = json.load(f)
except (OSError, ValueError) as e:
    raise SystemExit(f"card_timing: FAILED: {cell}: bench result is not JSON ({e})")
# vllm's key spellings move between versions: try the known ones, fail loudly
# listing the file's own keys — never guess a key. TPOT p50 is the ratio row;
# the c8-1k cell reports tok/s alongside (request_throughput is req/s, ~256x
# off — never emit it as tok/s); the ~10k long cells report TTFT p99 instead.
def first(keys, what):
    for k in keys:
        v = result.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            return float(v)
    raise SystemExit(
        f"card_timing: FAILED: {cell}: no {what} in keys {sorted(result)}; "
        "add the spelling this vllm version uses and re-run")
specs = [
    ("itl_ms", ["median_tpot_ms", "p50_tpot_ms", "mean_tpot_ms",
                "median_itl_ms", "p50_itl_ms", "mean_itl_ms", "tpot_ms"],
     "TPOT/ITL (ms)"),
]
if cell == "c8-1k":
    specs.append(("tok_s", ["output_throughput", "mean_output_throughput",
                            "total_output_throughput"],
                  "output throughput (tok/s)"))
else:  # the ~10k long cells: TTFT p99, where #701's wave admission shows
    specs.append(("ttft_p99_ms", ["p99_ttft_ms", "ttft_p99_ms",
                                  "99_percentile_ttft_ms"], "TTFT p99 (ms)"))
for metric, keys, what in specs:
    print(json.dumps({"cell": cell, "metric": metric, "value": first(keys, what)}))
PY
  LAST_BENCH_S=$BENCH_S
}

# ---- measure: the model's pinned boot plan -------------------------------------
LAST_WARM_S=0 LAST_RUN_S=0 LAST_BENCH_S=0
W1K_S=0 M1K_S=0 W10K_S=0 M10K_S=0 BENCH_S=0 C4_S=0 C8_S=0
case "$MODEL_KIND" in
  2b)
    # One boot hosts every base cell — validated on the card (solo cells
    # matched the sweep within ~3%).
    serve_up 8 12288
    rung1_run r1-1k 1061
    W1K_S=$LAST_WARM_S; M1K_S=$LAST_RUN_S
    rung1_run r1-10k 10091
    W10K_S=$LAST_WARM_S; M10K_S=$LAST_RUN_S
    bench_run c8-1k "$DS_1K" 8 8 9 1083
    BENCH_S=$LAST_BENCH_S
    serve_down
    SHAPES="one boot (mns 8, mml 12288)"
    if [ "$CELLS" = long ]; then
      # The long group, each cell on its own boot at the sweep's own shapes
      # (cells.tsv's s4/s8-2b-16-10k). The waves are prefill-heavy (conc x
      # ~7 s before any decode) — the wall is stated in the header, not
      # trimmed.
      serve_up 4 12288
      bench_run c4-10k "$DS_10K" 4 4 5 10300
      C4_S=$LAST_BENCH_S
      serve_down
      serve_up 8 12288
      bench_run c8-10k "$DS_10K" 8 8 9 10300
      C8_S=$LAST_BENCH_S
      serve_down
      SHAPES="$SHAPES + long boots (c4-10k: mns 4, mml 12288; c8-10k: mns 8, mml 12288)"
    fi
    ;;
  8b)
    # Two boots: mml 12288 at mns 8 needs 384 KV pages against the 273-page
    # pool, and the mml 8704 that mns 8 fits cannot host the ~10k prompt —
    # so the solo cells take mns 2 mml 12288, and the conc-8 cell takes the
    # sweep's own 8b batched shape, mns 8 mml 8704.
    serve_up 2 12288
    rung1_run r1-1k 1061
    W1K_S=$LAST_WARM_S; M1K_S=$LAST_RUN_S
    rung1_run r1-10k 10091
    W10K_S=$LAST_WARM_S; M10K_S=$LAST_RUN_S
    serve_down
    serve_up 8 8704
    bench_run c8-1k "$DS_1K" 8 8 9 1083
    BENCH_S=$LAST_BENCH_S
    serve_down
    SHAPES="two boots (solo: mns 2, mml 12288; conc-8: mns 8, mml 8704)"
    ;;
esac

# ---- the report (stdout) --------------------------------------------------------
TOTAL_S=$((SECONDS - T_START))
MD5_CAND=$(md5_of "$CAND")
MD5_REF=$(md5_of "$REF")
emit_report "$RUNS_JSONL" "$SWEEP" "$SHORT" "$SHAPES" "$CAND" "$MD5_CAND" "$REF" "$MD5_REF" \
  "$(hostname)" "$(date -u +%Y-%m-%d)" \
  "$BOOT_S_TOTAL" "$BOOTS" "$W1K_S" "$M1K_S" "$W10K_S" "$M10K_S" "$BENCH_S" \
  "$C4_S" "$C8_S" "$TOTAL_S"
echo "OK: card timing complete ($(date -u +%H:%M:%SZ))" >&2
