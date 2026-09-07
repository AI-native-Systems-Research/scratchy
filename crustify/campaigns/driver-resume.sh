#!/bin/bash
# Bridge-2 campaign driver. Runs the remaining sub-campaigns end to end:
#   port wave(s) -> promote session branch -> review pass -> promote -> pipeline gate
# Every stage's own log is beside this file; this script's own trace is driver.trace.
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/bridge2
LOGDIR=$ROOT/crustify/campaigns/logs
TRACE=$LOGDIR/driver.trace
CAMP=$ROOT/crustify/campaigns
say() { echo "[$(date +%H:%M:%S)] $*" >> $TRACE; }

# ⛔⛔ SINGLE-INSTANCE LOCK. THREE SEPARATE INCIDENTS CAME FROM TWO DRIVERS RUNNING AT ONCE.
# They share every log path under $LOGDIR, so each overwrites the other's stage log — and `promote`
# reads the session branch out of that log. On 2026-09-07 the older driver promoted a branch the
# younger one owned, which moved HEAD sideways, and the younger driver's own promote then failed
# "not a fast-forward" and stopped the campaign with nine batches of finished work stranded.
LOCK=$ROOT/crustify/campaigns/.driver.lock
if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "driver already running as pid $(cat $LOCK) — refusing to start a second" >&2
  say "REFUSED: driver already running as pid $(cat $LOCK)"
  exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM
say "LOCK acquired by pid $$"


count() {
  filled=$(grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' $ROOT/crates/compiler/deeptools/src | sort -u | wc -l | tr -d ' ')
  open=$(grep -rhoE 'crustify:todo: e[0-9]{3}_[A-Za-z0-9_]+' $ROOT/crates/compiler/deeptools/src | sort -u | wc -l | tr -d ' ')
  say "ANCHORS: filled=$filled openTodo=$open"
}

promote() {
  br=$(grep -ohE 'crustify/session/[a-z]+-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{4}' "$1" | tail -1)
  if [ -z "$br" ]; then say "PROMOTE: no session branch found in $1"; return 0; fi
  # ⭐ REBASE THEN FAST-FORWARD, rather than fast-forward or die. A sideways HEAD — another promote,
  # a hand commit — must not strand a stage's finished work on a session branch. The rebase replays
  # the batches onto HEAD; only a genuine content conflict stops us now, and it stops us with the
  # branch intact and named.
  if git -C $ROOT merge --ff-only "$br" >> $TRACE 2>&1; then
    say "PROMOTE ok (ff): $br -> $(git -C $ROOT rev-parse --short HEAD)"
    return 0
  fi
  say "PROMOTE: not a fast-forward, rebasing $br onto $(git -C $ROOT rev-parse --short HEAD)"
  if git -C $ROOT rebase HEAD "$br" >> $TRACE 2>&1; then
    git -C $ROOT checkout - >> $TRACE 2>&1
    if git -C $ROOT merge --ff-only "$br" >> $TRACE 2>&1; then
      say "PROMOTE ok (rebased): $br -> $(git -C $ROOT rev-parse --short HEAD)"
      return 0
    fi
  fi
  git -C $ROOT rebase --abort >> $TRACE 2>&1
  git -C $ROOT checkout bridge2-campaign >> $TRACE 2>&1
  say "PROMOTE FAILED (rebase conflicted): $br — work is SAFE on that branch — STOPPING"
  exit 3
}

disk() {
  free=$(df -g $ROOT | awk 'NR==2{print $4}')
  say "DISK: ${free}G free"
  if [ "$free" -lt 12 ]; then say "DISK BELOW 12G — STOPPING"; exit 4; fi
}

stage() { # $1 wave json  $2 objective  $3 tag
  disk
  # ⭐ RETRY, BECAUSE THE FAILURES ARE THE NETWORK AND NOT THE WORK. Every batch lost so far died
  # with `API Error: Can't reach the API server (ENOTFOUND)` after 120-180 turns — 9 of 18 in the
  # first level-0 wave, all three in the run before it. crustify commits per batch, so a retry
  # re-runs only what is still unfilled if the schedule is a remainder; otherwise it re-runs the
  # stage and the already-landed anchors make the finished units cheap to redo.
  for attempt in 1 2 3; do
    say "STAGE $3 start ($2, $1) attempt $attempt"
    crustify --parallel-max 8 $ROOT . translate "$CAMP/$1" --objective "$2" > $LOGDIR/driver-$3.log 2>&1
    rc=$?
    say "STAGE $3 attempt $attempt exit=$rc"
    [ $rc -eq 0 ] && break
    if ! grep -q "ENOTFOUND\|Can't reach the API server\|exited 1 for TranslateAgent" $LOGDIR/driver-$3.log; then
      say "STAGE $3 failed for a reason that is NOT the API — not retrying"; break
    fi
    say "STAGE $3 retrying in 120s (API failure)"; sleep 120
  done
  say "STAGE $3 exit=$rc"
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

# RESUME. The original driver assumed sub-campaign 1's port wave was already running and only
# waited on it (WAIT_PID) before going to review. That wave died with 0 units landed — all three
# translator agents hit `API Error: ENOTFOUND` after ~35 min / 123 turns and `claude` exited 1 — so
# this run must PORT level 0 rather than review nothing.
say "RESUME: sc1 port wave died with filled=0 (agent API ENOTFOUND); restarting from sc1-port"
count

stage level0-accessors-and-leaves/port-remainder.json     port   sc1-port
stage level0-accessors-and-leaves/review.json            review sc1-review
gate sc1
stage levels1-2-transfer-and-compute/port-remainder.json port   sc2-port
stage levels1-2-transfer-and-compute/review.json         review sc2-review
gate sc2
stage levels3-7-statements-and-passes/port-remainder.json port  sc3-port
stage levels3-7-statements-and-passes/review.json        review sc3-review
gate sc3
stage levels8-10-pass-drivers/port-remainder.json        port   sc4-port
stage levels8-10-pass-drivers/review.json                review sc4-review
gate sc4
say "CAMPAIGN DRIVER DONE"
