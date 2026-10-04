#!/bin/bash
# REGENERATE THE ttir -> KTIR GOLDEN CHAIN, for every fixture, from the C++ toolchain.
#
#   bash third_party/spyre/test/goldens/ktir/regen.sh
#
# These goldens are the ORACLE for the Rust port of bridge two (ttir -> KTIR). Each
# config gets, per stage of make_ktir's pipeline plus the audit chain:
#
#   0_ttir.mlir     make_ttir output      -- the INPUT the Rust port consumes
#   1_ktir.mlir     make_ktir output      -- DotToLinalg -> ConvertTTIRToKTDP ->
#                                            DistributeWork -> LegalizeTypes ->
#                                            canonicalize -> DecomposeDenseConstants
#                                            -> PlanCorelets
#   2_layout.mlir   --spyre-lane-major-layout
#   3_sched.mlir    --spyre-to-scheduler-ktir=grid=...
#   4_groups.mlir   --spyre-carried-values-to-memory
#
# A fixture the C++ REFUSES records the refusal text instead, in `refusal.txt`, with
# the failing stage in `stage.txt`. Reproducing those refusals BY NAME is part of the
# port's contract, so the message is a golden too.
set -eu
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
CMAKE_DIR="$ROOT/build/cmake.macosx-26.0-arm64-cpython-3.14"
VENV=/private/tmp/triton-spyre-wt-venv
OUT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$CMAKE_DIR/bin:$PATH"
export PYTHONPATH="$ROOT/python:$CMAKE_DIR"
cd "$ROOT"

"$VENV/bin/python" third_party/spyre/test/goldens/ktir/regen.py "$OUT"

# The audit chain, per config that produced a KTIR.
for d in "$OUT"/*/; do
  cfg="$(basename "$d")"
  if [ ! -f "$d/1_ktir.mlir" ]; then
    echo "### $cfg  REFUSED at $(cat "$d/stage.txt" 2>/dev/null || echo '?')"
    continue
  fi
  grid="$(cat "$d/grid.txt")"
  echo "### $cfg  grid=$grid"
  triton-opt --spyre-lane-major-layout "$d/1_ktir.mlir" -o "$d/2_layout.mlir" \
      2>"$d/2_layout.err" || { echo "  2_layout FAILED"; continue; }
  triton-opt "--spyre-to-scheduler-ktir=grid=$grid" "$d/2_layout.mlir" \
      -o "$d/3_sched.mlir" 2>"$d/3_sched.err" || { echo "  3_sched FAILED"; continue; }
  set +e
  triton-opt --spyre-carried-values-to-memory "$d/3_sched.mlir" \
      -o "$d/4_groups.mlir" 2>"$d/4_groups.err"
  set -e
  printf '  lines: ttir=%s ktir=%s layout=%s sched=%s groups=%s   warn=%s\n' \
    "$(grep -c . "$d/0_ttir.mlir")" \
    "$(grep -c . "$d/1_ktir.mlir")" "$(grep -c . "$d/2_layout.mlir")" \
    "$(grep -c . "$d/3_sched.mlir")" \
    "$(grep -c . "$d/4_groups.mlir" 2>/dev/null || echo -)" \
    "$(grep -c warning "$d/4_groups.err" || true)"
done
