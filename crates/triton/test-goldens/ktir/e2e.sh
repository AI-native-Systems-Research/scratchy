#!/bin/bash
# END TO END: the RUST port's own KTIR -> SuperDSC -> dxp_standalone, per fixture.
#
#   bash third_party/spyre/test/goldens/ktir/e2e.sh
#
# WHAT THIS PROVES THAT THE GOLDEN DIFF DOES NOT. The diff says the port's KTIR is
# structurally identical to the C++'s. This says the port's KTIR is DOWNSTREAM-VALID:
# `triton-superdsc-lower` reads it and `dxp_standalone` compiles it to a real
# artifact. Structural equality plus a downstream refusal would still be a broken
# port.
#
# THE ARTIFACT IS THE CHECK, and it has to be, because `dxp_standalone` PRINTS
# NOTHING on success and an abort still returns 0 through a pipe. So the pass
# condition is `spyreCodeDir/init_binary.bin` existing and being non-empty -- never
# the exit status.
#
# THE CONTROL is the second half: the C++ golden's OWN KTIR is driven through the
# same chain. If the baseline did not produce an artifact either, the port's success
# would say nothing about the port.
set -eu
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
KTIR_CRATE="$ROOT/third_party/spyre/rust/triton-ktir"
SDSC_CRATE="$ROOT/third_party/spyre/rust/triton-superdsc"
DT=${DT:-$HOME/tmp/dt_src}
DXP="$DT/build/dxp/dxp_standalone"
WORK=${WORK:-/tmp/ktir_e2e}

if [ ! -x "$DXP" ]; then
  echo "MISSING $DXP -- this check cannot pass without it, and skipping would print a"
  echo "green that means nothing. Build deeptools, or set DT=<path>."
  exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK"
fail=0

# `$1` = a label, `$2` = the KTIR text file. Lower it, compile it, require the
# artifact.
drive() {
  label="$1"; ktir="$2"
  out="$WORK/$label"
  mkdir -p "$out"
  if ! (cd "$SDSC_CRATE" && cargo run --offline -q --example emit_bundles -- \
        "$ktir" 0 "$out" >"$out/.lower.log" 2>&1); then
    echo "  $label: LOWERING REFUSED"
    sed 's/^/      /' "$out/.lower.log" | head -5
    fail=1
    return
  fi
  sdscs=$(ls "$out"/sdsc_*.json 2>/dev/null | wc -l | tr -d ' ')
  (cd "$out" && DEEPTOOLS_PATH="$DT" "$DXP" -d "$out" -b sentient \
      >"$out/.dxp.log" 2>&1) || true
  bin="$out/spyreCodeDir/init_binary.bin"
  if [ -s "$bin" ]; then
    echo "  $label: $sdscs SuperDSCs -> ARTIFACT $(wc -c <"$bin" | tr -d ' ') bytes"
  else
    echo "  $label: $sdscs SuperDSCs -> NO ARTIFACT (dxp_standalone prints nothing on"
    echo "      success and returns 0 through a pipe, so the missing artifact IS the failure)"
    sed 's/^/      /' "$out/.dxp.log" | head -8
    fail=1
  fi
}

echo "=== THE PORT'S OWN KTIR ==="
for cfg in attention_flash_noncausal attention_flash_noncausal_unitscale \
           attention_flash_causal swiglu_mlp; do
  ktir="$WORK/$cfg.ours.ktir.mlir"
  (cd "$KTIR_CRATE" && cargo run --offline -q --example emit_ktir -- "$cfg" "$ktir") \
      2>"$WORK/$cfg.census.txt"
  drive "$cfg.ours" "$ktir"
done

echo "=== THE CONTROL: the C++ golden's own KTIR, same chain ==="
for cfg in attention_flash_noncausal attention_flash_causal swiglu_mlp; do
  drive "$cfg.cpp" "$ROOT/third_party/spyre/test/goldens/ktir/$cfg/1_ktir.mlir"
done

echo
if [ "$fail" = 0 ]; then
  echo "RESULT: every configuration reached an artifact, from BOTH the Rust port's KTIR"
  echo "and the C++ golden's -- so the port's output is downstream-valid, not merely"
  echo "structurally equal."
else
  echo "RESULT: at least one configuration did not reach an artifact -- see above."
  exit 1
fi
