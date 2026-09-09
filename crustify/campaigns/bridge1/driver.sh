#!/bin/bash
# Bridge-1 campaign driver: SuperDSC -> DataflowIR, 110 units in 3 sub-campaigns.
#   port wave(s) -> promote session branch -> review pass -> promote -> pipeline gate
# Every stage's own log is under logs/; this script's own trace is logs/driver.trace.
#
# Every guard below is a scar from the bridge-2 campaign. Do not simplify one away.
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/bridge1
BRANCH=bridge1-campaign
CAMP=$ROOT/crustify/campaigns/bridge1
LOGDIR=$CAMP/logs
TRACE=$LOGDIR/driver.trace
mkdir -p $LOGDIR
say() { echo "[$(date +%H:%M:%S)] $*" >> $TRACE; }

# ⛔⛔ (1) SINGLE-INSTANCE LOCK. THREE SEPARATE INCIDENTS ON BRIDGE 2 CAME FROM TWO DRIVERS AT ONCE.
# Duplicate drivers share every log path under $LOGDIR, so each overwrites the other's stage log —
# and `promote` reads the session branch name OUT of that log. On 2026-09-07 the older driver
# promoted a branch the younger one owned, HEAD moved sideways, the younger driver's own promote
# then failed "not a fast-forward", and the campaign stopped with nine batches of finished work
# stranded.
# ⛔⛔ NEVER `rm -f` THIS FILE TO CLEAR A "STALE" LOCK. Doing exactly that on bridge 2 while pid
# 17581 was alive let a second driver start. The `kill -0` below already handles a genuinely stale
# lock: the owner is gone, the check fails, this driver starts. If a LIVE driver holds it, KILL THE
# DRIVER and let its EXIT trap release the lock — and kill its `crustify` CHILD too, because killing
# the bash script alone reparents the child to init, which keeps spawning agents:
#     pkill -f 'crustify .*worktrees/bridge1'
# ⚠ This path is deliberately NOT the bridge-2 campaign's .driver.lock. That file is tracked in git
# and this worktree's checkout of it holds a pid that is ALIVE in the bridge-2 worktree; sharing it
# would make this driver refuse to start forever.
LOCK=$CAMP/.driver.lock
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
  say "ANCHORS: filled=$filled openTodo=$open  (of 110)"
}

# ⭐ (5) REMAINDER SCHEDULES. Rebuild every port-remainder.json from the anchors actually filled on
# the branch. Without this a restart re-ports what already landed: the schedules are static and
# crustify has no idea which units are done.
remainders() {
  grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' $ROOT/crates/compiler/deeptools/src \
    | sed 's|/// Replaces: ||' | sort -u > /tmp/filled-bridge1.txt
  python3 $CAMP/regen-remainders.py >> $TRACE 2>&1
}

# ⭐ (2) PROMOTE BY FF-THEN-REBASE, never ff-or-die. A sideways HEAD — another promote, a hand
# commit — must not strand a stage's finished work on a session branch. On bridge 2 that stranded
# 48 finished functions until they were rebased by hand.
promote() {
  br=$(grep -ohE 'crustify/session/[a-z]+-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{4}' "$1" | tail -1)
  if [ -z "$br" ]; then say "PROMOTE: no session branch found in $1"; return 0; fi
  if git -C $ROOT merge --ff-only "$br" >> $TRACE 2>&1; then
    say "PROMOTE ok (ff): $br -> $(git -C $ROOT rev-parse --short HEAD)"
    remainders; return 0
  fi
  say "PROMOTE: not a fast-forward, rebasing $br onto $(git -C $ROOT rev-parse --short HEAD)"
  if git -C $ROOT rebase HEAD "$br" >> $TRACE 2>&1; then
    git -C $ROOT checkout - >> $TRACE 2>&1
    if git -C $ROOT merge --ff-only "$br" >> $TRACE 2>&1; then
      say "PROMOTE ok (rebased): $br -> $(git -C $ROOT rev-parse --short HEAD)"
      remainders; return 0
    fi
  fi
  git -C $ROOT rebase --abort >> $TRACE 2>&1
  git -C $ROOT checkout $BRANCH >> $TRACE 2>&1
  say "PROMOTE FAILED (rebase conflicted): $br — work is SAFE on that branch — STOPPING"
  exit 3
}

# ⭐ (8) DISK. Each agent worktree grows its own target/. Bridge 2 runs 8 agents and has priority,
# so this campaign takes 4. Abort below 12G rather than wedge the host.
disk() {
  free=$(df -g $ROOT | awk 'NR==2{print $4}')
  say "DISK: ${free}G free"
  if [ "$free" -lt 12 ]; then say "DISK BELOW 12G — STOPPING"; exit 4; fi
}

# ⭐ (8) Prune promoted agent worktrees between stages, or the accumulated target/ dirs fill the disk.
prune() {
  git -C $ROOT worktree list --porcelain | grep -oE "$ROOT/crustify/.worktrees/[^ ]+" | while read -r w; do
    git -C $ROOT worktree remove --force "$w" >> $TRACE 2>&1
  done
  git -C $ROOT worktree prune >> $TRACE 2>&1
  say "PRUNED agent worktrees; $(df -g $ROOT | awk 'NR==2{print $4}')G free"
}

stage() { # $1 schedule json (relative to $CAMP)  $2 objective  $3 tag
  # ⭐ (6) SKIP AN EMPTY OR FINISHED SCHEDULE. A finished level leaves 0 units in its remainder.
  # Without this the driver hands crustify an empty wave list — or worse re-runs a REVIEW over a
  # level already reviewed: on bridge 2 a restart started 9 agents re-reviewing 142 done functions.
  n=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['unit_count'])" "$CAMP/$1" 2>/dev/null || echo 1)
  if [ "$n" = "0" ]; then say "STAGE $3 SKIPPED (0 units left in $1)"; return 0; fi
  disk
  # ⭐ (3)+(4) SMALL BATCHES, AND RETRY ONLY API FAILURES. max_syms is 8 for port and 24 for review
  # in the schedules themselves; crustify promotes nothing until an agent completes, and at 50-unit
  # batches an API drop at hour 4½ discarded three agents' work. Every batch lost on bridge 2 died
  # with `API Error: Can't reach the API server (ENOTFOUND)` after 120-180 turns — so retry 3× at
  # 120s, but ONLY when the log names an API failure. Anything else fails fast.
  for attempt in 1 2 3; do
    say "STAGE $3 start ($2, $1, $n units) attempt $attempt"
    crustify --parallel-max 4 $ROOT . translate "$CAMP/$1" --objective "$2" > $LOGDIR/driver-$3.log 2>&1
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
  prune
}

gate() { # $1 tag
  ( cd $ROOT && cargo test -p deeptools > $LOGDIR/test-$1.log 2>&1 )
  say "TEST $1 exit=$? :: $(grep -E '^test result' $LOGDIR/test-$1.log | tr '\n' ' ')"
  say "GATE $1 start (scratchy-models superdsc pipeline)"
  ( cd $ROOT && cargo build -p scratchy-models \
      --features superdsc,granite-3.1-2b-instruct,scratchy-quantizations/fp8-dynamic-per-channel \
      > $LOGDIR/gate-$1.log 2>&1 )
  say "GATE $1 exit=$? "
  grep -E 'not yet implemented|panicked at|^error' $LOGDIR/gate-$1.log | tail -3 >> $TRACE
}

say "=== BRIDGE 1 CAMPAIGN START on $BRANCH at $(git -C $ROOT rev-parse --short HEAD) ==="
count
remainders

stage levels0-1-leaves-and-helpers/port-remainder.json      port   sc1-port
stage levels0-1-leaves-and-helpers/review.json              review sc1-review
gate sc1
stage levels2-4-transfer-and-compute/port-remainder.json    port   sc2-port
stage levels2-4-transfer-and-compute/review.json            review sc2-review
gate sc2
stage levels5-10-statements-and-drivers/port-remainder.json port   sc3-port
stage levels5-10-statements-and-drivers/review.json         review sc3-review
gate sc3
say "CAMPAIGN DRIVER DONE"
