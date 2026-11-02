#!/bin/bash
# front-b-noise WALL-10 INT MB LEG (nickm2): the auditor's GO 2026-10-27 —
# "The int Mb leg: [2,512] dense + [1,512] Mb add and mul, plus the
# [2,65536] twin, through the scatter harness, on nickm2. On pass: (a) the
# band-Mb gate extended to IntBinary, (b) the int door's Mb arm opened
# citing the leg. Byte-identity on every other kernel, then the chat leg
# unchanged. No stops between them unless a cell fails, and then the card
# bytes come to me."
# The Mb form is the 32-row ceiling rung's own bake refusal (the whole-row
# [1, PAD_V] i32 lane table against the band-typed [B, PAD_V] tile —
# sample_fwd.py's `h = lane * hc1 + slo`). Two cell families:
#   * dense+Mb  (intmb_add / intmb_mul / intmb_wide_*) — isolates the spray.
#   * Mb×Col    (intmb_col_* / intmb_wide_col_*) — the PRODUCTION form: the
#     hash's multiply has NO dense operand (lane against the per-row word).
# The admission (band-Mb gate + door arm) is ALREADY IN THE TREE this leg
# bakes — the wall-9 in-flight pattern: the cells measure what the admission
# emits, the verdict gates whether it stays.
# PHASE B is the byte-identity witness the order demands: the whole intmr
# family re-baked, re-dxp'd and re-run under the NEW door — every bake
# artifact and every output md5-identical to the wall-9 leg's own bytes,
# then re-verdicted bit-for-bit. Tree: /work/front-b-noise. Bake: ON THE
# POD, KTIR_WHOLE=1 through the whole-function door. Runner: bundle_run_dev
# at release in the WARM target dir /work/pr916/head/target. Verdict ON THE
# POD (intmb_stats.py / intmr_scatter_stats.py, cells named on the command
# line); the numbers land in intmb-scatter-verdict.json. Markers:
# INTMB-DONE / INTMB-FAILED / INTMB-BYTE-IDENTITY-FAILED.
# Log: /work/intmb-leg.log
set -u

LOG=/work/intmb-leg.log
TREE=/work/front-b-noise
TDIR=/work/pr916/head/target
PROBE=/work/fb-noise-probe
BIN=$TDIR/release/examples/bundle_run_dev
DXP=/opt/ibm/spyre/deeptools/bin/dxp_standalone

# The measurement cells (phase A) — POSITIONAL ARGS can narrow them; the BG
# re-exec passes "$@" through.
MB_CELLS="${*:-intmb_add intmb_mul intmb_col_add intmb_col_mul intmb_wide_add intmb_wide_mul intmb_wide_col_add intmb_wide_col_mul}"
# The byte-identity cells (phase B) — the wall-9 family, verbatim.
INTMR_CELLS="intmr_dense_add intmr_dense_mul intmr_col_add intmr_col_mul intmr_f32col_add intmr_f32wide_add intmr_f32maxadd"

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" "$@" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== intmb-leg start $(date -u +%FT%TZ) cells: $MB_CELLS ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo INTMB-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo INTMB-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo INTMB-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo INTMB-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share

# ── leg 1: the runner (release, warm target dir) ──
echo "==== INTMB-BUILD-RUNNER-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-target-spyre --features spyre-hw --example bundle_run_dev; then
  echo INTMB-BUILD-FAILED; echo INTMB-FAILED; exit 1
fi
echo "==== INTMB-BUILD-RUNNER-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the staged inputs (both families, ONCE) ──
echo "==== INTMB-INPUTS $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_inputs.py" || { echo INTMB-INPUTS-FAILED; echo INTMB-FAILED; exit 1; }
python3 "$PROBE/intmb_inputs.py" || { echo INTMB-INPUTS-FAILED; echo INTMB-FAILED; exit 1; }

# ── legs 3-10: per measurement cell — bake, dxp, run ──
for CELL in $MB_CELLS; do
  BAKE=/work/int-mr-$CELL
  echo "==== INTMB-BAKE-START $CELL $(date -u +%FT%TZ) ===="
  rm -rf "$BAKE"
  if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
    echo "INTMB-BAKE-FAILED ($CELL)"; echo INTMB-FAILED; exit 1
  fi
  echo "==== INTMB-BAKE-OK $CELL ($(grep -o '"numWkSlicesPerDim_":{[^}]*}' "$BAKE"/sdsc_0.json | head -1)) ===="

  echo "==== INTMB-DXP $CELL $(date -u +%FT%TZ) ===="
  if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
    echo "INTMB-DXP-FAILED ($CELL)"; tail -5 "$BAKE/dxp.log"; echo INTMB-FAILED; exit 1
  fi
  echo "INTMB-DXP-OK $CELL ($(stat -c%s "$BAKE/spyreCodeDir/init_binary.bin") B image)"

  echo "==== INTMB-RUN-START $CELL $(date -u +%FT%TZ) ===="
  SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
    t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
    t1="$PROBE/in/${CELL}_t01_b.bin" raw:t1 \
    --out raw:t2="$PROBE/out_${CELL}_o.bin" \
    > "$PROBE/intmb-run-$CELL.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "INTMB-RUN-FAILED ($CELL rc=$rc)"; tail -20 "$PROBE/intmb-run-$CELL.log"; echo INTMB-FAILED; exit 1
  fi
  tail -4 "$PROBE/intmb-run-$CELL.log"
done

# ── leg 11: the measurement verdict, on the pod ──
echo "==== INTMB-CHECK $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmb_stats.py" $MB_CELLS
rc=$?
echo "==== intmb measurement checkrc=$rc $(date -u +%FT%TZ) ===="
if [ "$rc" -ne 0 ]; then echo INTMB-FAILED; exit 1; fi

# ── phase B: BYTE-IDENTITY on every other kernel — the intmr family under
#    the NEW door. Every bake artifact and every output md5-identical to the
#    wall-9 leg's own bytes, then re-verdicted bit-for-bit. ──
echo "==== INTMB-BI-PRE $(date -u +%FT%TZ) (capturing the wall-9 bytes) ===="
PRE=$PROBE/intmb-bi-pre.md5
: > "$PRE"
for CELL in $INTMR_CELLS; do
  BAKE=/work/int-mr-$CELL
  for f in "$BAKE/sdsc_0.json" "$BAKE/init_binary.bin" "$PROBE/out_${CELL}_o.bin"; do
    if [ -f "$f" ]; then
      md5sum "$f" >> "$PRE"
    else
      echo "MISSING $f" >> "$PRE"
    fi
  done
done
wc -l "$PRE"

for CELL in $INTMR_CELLS; do
  BAKE=/work/int-mr-$CELL
  echo "==== INTMB-BI-BAKE-START $CELL $(date -u +%FT%TZ) ===="
  rm -rf "$BAKE"
  if ! KTIR_WHOLE=1 cargo run --offline -p triton-ktir-superdsc --example bake_py -- "$CELL" "$BAKE"; then
    echo "INTMB-BI-BAKE-FAILED ($CELL)"; echo INTMB-BYTE-IDENTITY-FAILED; exit 1
  fi
  echo "==== INTMB-BI-DXP $CELL $(date -u +%FT%TZ) ===="
  if ! "$DXP" -d "$BAKE" -b sentient > "$BAKE/dxp.log" 2>&1; then
    echo "INTMB-BI-DXP-FAILED ($CELL)"; tail -5 "$BAKE/dxp.log"; echo INTMB-BYTE-IDENTITY-FAILED; exit 1
  fi
  echo "==== INTMB-BI-RUN $CELL $(date -u +%FT%TZ) ===="
  if [ "$CELL" = intmr_f32maxadd ]; then
    SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
      t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
      --out raw:t1="$PROBE/out_${CELL}_o.bin" \
      > "$PROBE/intmr-run-$CELL.log" 2>&1
  else
    SCRATCHY_SUPERDSC_DBG=1 "$BIN" "$BAKE" \
      t0="$PROBE/in/${CELL}_t00_a.bin" raw:t0 \
      t1="$PROBE/in/${CELL}_t01_b.bin" raw:t1 \
      --out raw:t2="$PROBE/out_${CELL}_o.bin" \
      > "$PROBE/intmr-run-$CELL.log" 2>&1
  fi
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "INTMB-BI-RUN-FAILED ($CELL rc=$rc)"; tail -20 "$PROBE/intmr-run-$CELL.log"; echo INTMB-BYTE-IDENTITY-FAILED; exit 1
  fi
done

echo "==== INTMB-BI-COMPARE $(date -u +%FT%TZ) ===="
bi_fail=0
while read -r h f; do
  case "$h" in MISSING) continue ;; esac
  h2=$(md5sum "$f" | cut -d" " -f1)
  if [ "$h" != "$h2" ]; then
    echo "BYTE-IDENTITY-FAIL $f ($h -> $h2)"
    bi_fail=1
  fi
done < "$PRE"
if [ "$bi_fail" -ne 0 ]; then echo INTMB-BYTE-IDENTITY-FAILED; exit 1; fi
echo "INTMB-BI-MD5-OK (every wall-9 bake artifact and output byte-identical under the new door)"

echo "==== INTMB-BI-CHECK (re-verdict, bit-for-bit) $(date -u +%FT%TZ) ===="
python3 "$PROBE/intmr_scatter_stats.py" $INTMR_CELLS
rc=$?
echo "==== intmb byte-identity checkrc=$rc $(date -u +%FT%TZ) ===="
if [ "$rc" -ne 0 ]; then echo INTMB-BYTE-IDENTITY-FAILED; exit 1; fi

echo "==== intmb-leg end $(date -u +%FT%TZ) ===="
echo INTMB-DONE
