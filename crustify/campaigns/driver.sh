#!/bin/bash
# Bridge-2 campaign driver. Runs the remaining sub-campaigns end to end:
#   port wave(s) -> promote session branch -> review pass -> promote -> pipeline gate
# Every stage's own log is beside this file; this script's own trace is driver.trace.
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/bridge2
LOGDIR=$ROOT/crustify/campaigns/logs
TRACE=$LOGDIR/driver.trace
CAMP=$ROOT/crustify/campaigns
say() { echo "[$(date +%H:%M:%S)] $*" >> $TRACE; }

count() {
  filled=$(grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' $ROOT/crates/compiler/deeptools/src | sort -u | wc -l | tr -d ' ')
  open=$(grep -rhoE 'crustify:todo: e[0-9]{3}_[A-Za-z0-9_]+' $ROOT/crates/compiler/deeptools/src | sort -u | wc -l | tr -d ' ')
  say "ANCHORS: filled=$filled openTodo=$open"
}

promote() {
  br=$(grep -ohE 'crustify/session/[a-z]+-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{4}' "$1" | tail -1)
  if [ -z "$br" ]; then say "PROMOTE: no session branch found in $1"; return 0; fi
  if git -C $ROOT merge --ff-only "$br" >> $TRACE 2>&1; then
    say "PROMOTE ok: $br -> $(git -C $ROOT rev-parse --short HEAD)"
  else
    say "PROMOTE FAILED (not a fast-forward): $br — STOPPING"; exit 3
  fi
}

disk() {
  free=$(df -g $ROOT | awk 'NR==2{print $4}')
  say "DISK: ${free}G free"
  if [ "$free" -lt 12 ]; then say "DISK BELOW 12G — STOPPING"; exit 4; fi
}

stage() { # $1 wave json  $2 objective  $3 tag
  disk
  say "STAGE $3 start ($2, $1)"
  crustify --parallel-max 8 $ROOT . translate "$CAMP/$1" --objective "$2" > $LOGDIR/driver-$3.log 2>&1
  say "STAGE $3 exit=$?"
  grep -E '^\[crustify\] [0-9]+ failure' $LOGDIR/driver-$3.log >> $TRACE 2>/dev/null
  promote $LOGDIR/driver-$3.log
  count
}

gate() { # $1 tag
  say "GATE $1 start (scratchy-models superdsc pipeline)"
  ( cd $ROOT && cargo build -p scratchy-models \
      --features superdsc,granite-3.1-2b-instruct,scratchy-quantizations/fp8-dynamic-per-channel \
      > $LOGDIR/gate-$1.log 2>&1 )
  rc=$?
  say "GATE $1 exit=$rc"
  grep -E 'not yet implemented|panicked at|^error' $LOGDIR/gate-$1.log | tail -3 >> $TRACE
  ( cd $ROOT && cargo test -p deeptools > $LOGDIR/test-$1.log 2>&1 )
  say "TEST $1 exit=$? :: $(grep -E '^test result' $LOGDIR/test-$1.log | tr '\n' ' ')"
}

# 0. wait for the already-running sub-campaign-1 port wave to finish
if [ -n "$WAIT_PID" ]; then
  say "waiting on pid $WAIT_PID (sc1 port wave)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "sc1 port wave process exited"
  promote $LOGDIR/driver-sc1-port.log
  count
fi

stage level0-accessors-and-leaves/review.json          review sc1-review
gate sc1
stage levels1-2-transfer-and-compute/port.json         port   sc2-port
stage levels1-2-transfer-and-compute/review.json       review sc2-review
gate sc2
stage levels3-7-statements-and-passes/port.json        port   sc3-port
stage levels3-7-statements-and-passes/review.json      review sc3-review
gate sc3
stage levels8-10-pass-drivers/port.json                port   sc4-port
stage levels8-10-pass-drivers/review.json              review sc4-review
gate sc4
say "CAMPAIGN DRIVER DONE"
