#!/bin/bash
# Front A: the THREE-WAY LegalizeTypes byte-identity proof diff.
#   /work/ktir-dump-before  = base legalize (md5 0c78afcf...)  — no island at all
#   /work/ktir-dump-island  = island-only probe (plain value walk, no sweep, no
#                             region-interior) — the run-3-era behavior
#   /work/ktir-dump-after   = worktree legalize (island + comparison sweep +
#                             region-interior) — md5 6f8975fe...
# Writes /work/legalize-proof.txt and prints it.
set -u
OUT=/work/legalize-proof.txt
{
  echo "==== LegalizeTypes byte-identity proof, three-way $(date -u +%FT%TZ) ===="
  echo "before  = base legalize (no island)"
  echo "island  = island-only probe (value walk only — no comparison sweep, no region-interior)"
  echo "after   = worktree legalize (island + comparison sweep + region-interior)"
  echo "Harness: splice production_ktir_dump, 24 entries over every production kernel except sample_fwd.py."
  echo
  echo "==== [1] island vs before (does the island port alone move bytes?) ===="
  diff -r /work/ktir-dump-before /work/ktir-dump-island
  echo "exit: $?"
  echo
  echo "==== [2] after vs island (does the sweep + region-interior move bytes?) ===="
  diff -r /work/ktir-dump-island /work/ktir-dump-after
  echo "exit: $?"
  echo
  echo "==== [3] after vs before (the whole tree's legalize diff) ===="
  diff -r /work/ktir-dump-before /work/ktir-dump-after
  echo "exit: $?"
  echo
  echo "==== per-kernel differing-line counts ===="
  printf "%-32s %8s %8s %8s\n" kernel before-island island-after before-after
  for f in /work/ktir-dump-before/*.ktir; do
    n=$(basename "$f")
    a=$(diff -U0 "$f" "/work/ktir-dump-island/$n" | grep -c "^[+-]" || true)
    b=$(diff -U0 "/work/ktir-dump-island/$n" "/work/ktir-dump-after/$n" | grep -c "^[+-]" || true)
    c=$(diff -U0 "$f" "/work/ktir-dump-after/$n" | grep -c "^[+-]" || true)
    printf "%-32s %8s %8s %8s\n" "$n" "$a" "$b" "$c"
  done
} > "$OUT" 2>&1
cat "$OUT"
