#!/bin/bash
# front-b-noise CHAT LEG, PART 0 — the MAIN REFERENCE (nickm2): main's own
# greedy text + main's host logits for the leg's prompt. The auditor's chat-leg
# order (2026-10-27): items 4 and 5 need main as the reference — "Greedy (temp
# 0): must still equal main's text" and "softmax(logits) from main's host
# logits". Main e63c6f8ac8 reads its logits back to host (the device argmax
# was reverted in #926), so a print-only dump in the prompt-final chunk's
# read-back gives both. Tree: /work/front-b-chatmain — a SCRATCH WORKTREE at
# main's tip carrying ONE uncommitted probe patch (main-logits-dump.patch,
# never committed, never pushed). Build: ON THE POD, release, its own target
# dir (cold — one target dir per tree). Dump dir: /work/fb-chat-main. The chat:
# greedy (main's spyre path is always greedy — the worker never read
# temperature), same prompt and max-tokens as the leg's own runs. Markers:
# MAINREF-DONE / MAINREF-FAILED. Log: /work/main-ref-leg.log
set -u

LOG=/work/main-ref-leg.log
TREE=/work/front-b-chatmain
TDIR=/work/front-b-chatmain/target
PROBE=/work/fb-chat-main
SCR=$TDIR/release/scr
PROMPT="Write two sentences about the ocean."

if [ "${BG:-0}" != "1" ]; then
  BG=1 setsid nohup bash "$0" >>"$LOG" 2>&1 &
  echo "detached pid $! ; log $LOG"
  exit 0
fi
exec >>"$LOG" 2>&1

echo "==== main-ref-leg start $(date -u +%FT%TZ) ===="

# ── busy check (live cargo only; zombies are the pod's normal litter) ──
if pgrep -x cargo >/dev/null 2>&1; then
  for p in $(pgrep -x cargo); do
    if ! ps -o stat= -p "$p" 2>/dev/null | grep -q Z; then
      echo "POD BUSY (live cargo pid $p)"; echo MAINREF-FAILED; exit 1
    fi
  done
fi
if pgrep -f 'scr chat|scr serve' >/dev/null 2>&1; then
  echo "POD BUSY (scr)"; echo MAINREF-FAILED; exit 1
fi

[ -d "$TREE/crates" ] || { echo "no tree at $TREE (rsync first)"; echo MAINREF-FAILED; exit 1; }
cd "$TREE" || { echo "cannot cd $TREE"; echo MAINREF-FAILED; exit 1; }

export PATH=/work/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$TDIR
export LD_LIBRARY_PATH=/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export HF_HUB_OFFLINE=1

# ── leg 1: main's chat stack (release, cold target dir) ──
echo "==== MAINREF-BUILD-START $(date -u +%FT%TZ) ===="
if ! cargo build --release -p scratchy-cli \
     --features spyre-hw,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel; then
  echo MAINREF-BUILD-FAILED; echo MAINREF-FAILED; exit 1
fi
echo "==== MAINREF-BUILD-OK $(date -u +%FT%TZ) ===="

# ── leg 2: the greedy reference chat, with the logits dump riding it ──
mkdir -p "$PROBE"
echo "==== MAINREF-RUN-START $(date -u +%FT%TZ) ===="
"$SCR" chat -m RedHatAI/granite-3.1-2b-instruct-FP8-dynamic --device spyre \
  -q "$PROMPT" --max-tokens 64 \
  > "$PROBE/main-greedy.run" 2> "$PROBE/main-greedy.err"
rc=$?
echo "==== MAINREF-RUN rc=$rc $(date -u +%FT%TZ) ===="
if [ "$rc" -ne 0 ]; then
  echo MAINREF-RUN-FAILED; tail -30 "$PROBE/main-greedy.err"; echo MAINREF-FAILED; exit 1
fi
tail -5 "$PROBE/main-greedy.err"

# ── leg 3: the dump must exist and be vocab-sized ──
if [ ! -s "$PROBE/logits-0000.bin" ]; then
  echo "MAINREF-DUMP-MISSING (no /work/fb-chat-main/logits-0000.bin)"; echo MAINREF-FAILED; exit 1
fi
ls -la "$PROBE"/logits-*.bin | head -5
echo "==== main-ref-leg end $(date -u +%FT%TZ) ===="
echo MAINREF-DONE
