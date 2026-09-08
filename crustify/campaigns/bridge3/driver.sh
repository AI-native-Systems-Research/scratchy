#!/bin/bash
# Bridge-3 campaign driver (SentientIR -> ProgIR, dcc pass D76). Runs the whole campaign:
#   port wave(s) -> promote session branch -> review pass -> promote -> gate,  three times.
# Every stage's own log is under logs/ beside this file; this script's trace is logs/driver.trace.
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/bridge3
CAMP=$ROOT/crustify/campaigns/bridge3
LOGDIR=$CAMP/logs
TRACE=$LOGDIR/driver.trace
BRANCH=bridge3-campaign
mkdir -p "$LOGDIR"
say() { echo "[$(date +%H:%M:%S)] $*" >> "$TRACE"; }

# ⛔⛔ 1. SINGLE-INSTANCE LOCK. THREE SEPARATE INCIDENTS ON BRIDGE 2 CAME FROM TWO DRIVERS AT ONCE.
# They share every log path under $LOGDIR, so each overwrites the other's stage log — and `promote`
# reads the session branch out of that log. On 2026-09-07 the older driver promoted a branch the
# younger one owned, which moved HEAD sideways, and the younger driver's own promote then failed
# "not a fast-forward" and stopped the campaign with nine batches of finished work stranded.
#
# ⛔⛔ NEVER `rm -f` THIS FILE TO CLEAR A "STALE" LOCK. On bridge 2 that was done while pid 17581 was
# still alive, which let a second driver start. The `kill -0` below already handles a genuinely stale
# lock — the pid is gone, the check fails, and this driver starts. If a LIVE driver holds it, KILL THE
# DRIVER and let its EXIT trap release the lock. And KILL ITS `crustify` CHILD TOO: killing the bash
# script alone reparents the child to init, which keeps spawning agents.
#
# ⭐ This lock is deliberately NOT bridge 1's or bridge 2's. Theirs are git-tracked and their
# checked-out copies in this worktree hold pids that are alive in THEIR worktrees, so sharing one
# would make this driver refuse to start forever.
LOCK=$CAMP/.driver.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "driver already running as pid $(cat "$LOCK") — refusing to start a second" >&2
  say "REFUSED: driver already running as pid $(cat "$LOCK")"
  exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM
say "LOCK acquired by pid $$"

count() {
  filled=$(grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' "$ROOT/crates/compiler/deeptools/src/bridges/sentient_to_progir" \
           | sort -u | wc -l | tr -d ' ')
  open=$(grep -rhoE 'crustify:todo: e[0-9]{3}_[A-Za-z0-9_]+' "$ROOT/crates/compiler/deeptools/src/bridges/sentient_to_progir" \
         | sort -u | wc -l | tr -d ' ')
  say "ANCHORS: filled=$filled openTodo=$open"
}

# ⭐ 5. REMAINDER SCHEDULES, REBUILT AFTER EVERY PROMOTE from the anchors actually on the branch.
# Without this a restart re-ports what already landed: the schedules are static and crustify has no
# idea which units are done.
regen() {
  grep -rhoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' "$ROOT/crates/compiler/deeptools/src/bridges/sentient_to_progir" \
    | sed 's|/// Replaces: ||' | sort -u > /tmp/b3-filled.txt
  python3 "$CAMP/regen-remainders.py" >> "$TRACE" 2>&1
}

# ⭐ 2. PROMOTE BY FF-THEN-REBASE, never ff-or-die. A sideways HEAD — another promote, a hand commit —
# stranded 48 finished bridge-2 functions for hours. The rebase replays the batches onto HEAD; only a
# genuine content conflict stops us now, and it stops us with the branch INTACT AND NAMED.
promote() {
  br=$(grep -ohE 'crustify/session/[a-z]+-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{4}' "$1" | tail -1)
  if [ -z "$br" ]; then say "PROMOTE: no session branch found in $1"; return 0; fi
  if git -C "$ROOT" merge --ff-only "$br" >> "$TRACE" 2>&1; then
    say "PROMOTE ok (ff): $br -> $(git -C "$ROOT" rev-parse --short HEAD)"
    regen; return 0
  fi
  say "PROMOTE: not a fast-forward, rebasing $br onto $(git -C "$ROOT" rev-parse --short HEAD)"
  if git -C "$ROOT" rebase HEAD "$br" >> "$TRACE" 2>&1; then
    git -C "$ROOT" checkout "$BRANCH" >> "$TRACE" 2>&1
    if git -C "$ROOT" merge --ff-only "$br" >> "$TRACE" 2>&1; then
      say "PROMOTE ok (rebased): $br -> $(git -C "$ROOT" rev-parse --short HEAD)"
      regen; return 0
    fi
  fi
  git -C "$ROOT" rebase --abort >> "$TRACE" 2>&1
  git -C "$ROOT" checkout "$BRANCH" >> "$TRACE" 2>&1
  say "PROMOTE FAILED (rebase conflicted): $br — work is SAFE on that branch — STOPPING"
  exit 3
}

# ⭐ 8. DISK FLOOR AND PRUNING. Bridge 2's agent worktrees alone consumed 34 GB, and two sibling
# campaigns are running. Abort below 12 G; prune promoted agent worktrees between stages.
disk() {
  free=$(df -g "$ROOT" | awk 'NR==2{print $4}')
  say "DISK: ${free}G free"
  if [ "$free" -lt 12 ]; then say "DISK BELOW 12G — STOPPING"; exit 4; fi
}
prune() {
  before=$(df -g "$ROOT" | awk 'NR==2{print $4}')
  for wt in "$ROOT"/crustify/.worktrees/*; do
    [ -d "$wt" ] || continue
    # only a worktree whose HEAD is already an ancestor of the campaign branch — i.e. promoted
    h=$(git -C "$wt" rev-parse HEAD 2>/dev/null) || continue
    if git -C "$ROOT" merge-base --is-ancestor "$h" "$BRANCH" 2>/dev/null; then
      git -C "$ROOT" worktree remove --force "$wt" >> "$TRACE" 2>&1
    fi
  done
  git -C "$ROOT" worktree prune >> "$TRACE" 2>&1
  say "PRUNE: ${before}G -> $(df -g "$ROOT" | awk 'NR==2{print $4}')G free"
}

stage() { # $1 wave json (relative to $CAMP)  $2 objective  $3 tag
  # ⭐ 6. SKIP AN EMPTY OR ALREADY-COMPLETED SCHEDULE. regen() rewrites port-remainder.json after
  # every promote with only the units still lacking a filled anchor, so a finished level leaves 0.
  # Without this check the driver hands crustify an empty wave list — or worse re-runs a REVIEW over
  # a level already reviewed: a bridge-2 restart started 9 agents re-reviewing 142 done functions.
  local f="$CAMP/$1"
  if [ ! -f "$f" ]; then say "STAGE $3 SKIPPED ($1 does not exist)"; return 0; fi
  n=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['unit_count'])" "$f" 2>/dev/null || echo 1)
  if [ "$n" = "0" ]; then say "STAGE $3 SKIPPED (0 units left in $1)"; return 0; fi
  # ⭐ 6. A REVIEW STAGE IS NOT A REMAINDER, so nothing in the schedule shrinks when it finishes and
  # a restart would re-review every unit — a bridge-2 restart started 9 agents re-reviewing 142 done
  # functions. A done-marker beside the schedule is what makes the skip possible at all.
  if [ "$2" = "review" ]; then
    if [ -f "$(dirname "$f")/.reviewed" ]; then
      say "STAGE $3 SKIPPED (already reviewed: $(cat "$(dirname "$f")/.reviewed"))"; return 0
    fi
    # reviewing a sub-campaign that still has open ports would review half-finished work
    if [ -f "$(dirname "$f")/port-remainder.json" ]; then
      left=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['unit_count'])" \
             "$(dirname "$f")/port-remainder.json" 2>/dev/null || echo 0)
      if [ "$left" != "0" ]; then say "STAGE $3 SKIPPED ($left units still unported in this sub-campaign)"; return 0; fi
    fi
  fi
  disk
  local rc=1
  # ⭐ 4. RETRY ONLY API FAILURES. Every bridge-2 batch lost so far died with
  # `API Error: Can't reach the API server (ENOTFOUND)` after 120-180 turns. crustify commits per
  # batch, so a retry over a REMAINDER schedule re-runs only what is still unfilled. Anything that is
  # NOT the API fails fast — a compile error retried three times is three times the cost and the same
  # error.
  for attempt in 1 2 3; do
    say "STAGE $3 start ($2, $1) attempt $attempt"
    # ⭐ 3. max_syms 8 for port / 24 for review is in the SCHEDULES; ⭐ 8. --parallel-max 2 because
    # two sibling campaigns are already running and API drops are the largest source of lost work.
    crustify --parallel-max 2 "$ROOT" . translate "$f" --objective "$2" > "$LOGDIR/driver-$3.log" 2>&1
    rc=$?
    say "STAGE $3 attempt $attempt exit=$rc"
    [ $rc -eq 0 ] && break
    if ! grep -q "ENOTFOUND\|Can't reach the API server\|exited 1 for TranslateAgent" "$LOGDIR/driver-$3.log"; then
      say "STAGE $3 failed for a reason that is NOT the API — not retrying"; break
    fi
    say "STAGE $3 retrying in 120s (API failure)"; sleep 120
  done
  say "STAGE $3 exit=$rc"
  grep -E '^\[crustify\] [0-9]+ failure' "$LOGDIR/driver-$3.log" >> "$TRACE" 2>/dev/null
  promote "$LOGDIR/driver-$3.log"
  if [ "$2" = "review" ] && [ $rc -eq 0 ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$(dirname "$f")/.reviewed"
    say "STAGE $3 marked reviewed"
  fi
  count
  prune
  # ⛔⛔ 9. A PORT STAGE THAT LANDED NOTHING MUST STOP THE CAMPAIGN, NOT FALL THROUGH TO THE NEXT ONE.
  # On 2026-09-07 a port stage was killed mid-flight, landed 0 units and — correctly not retried, the
  # failure was not the API — promoted nothing, skipped its own review, ran the gate and then ADVANCED
  # a sub-campaign, so the driver began porting levels 2-3 while levels 0-1 were still entirely
  # unported. The wave barrier only holds WITHIN a stage; between stages nothing stopped it. A stage
  # that was ALREADY at zero returned early at the top of this function, so this cannot fire
  # spuriously; `promote` on success re-runs regen(), which rewrites the count read here.
  if [ "$2" = "port" ]; then
    local left_after
    left_after=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['unit_count'])" \
                 "$(dirname "$f")/port-remainder.json" 2>/dev/null || echo 0)
    if [ "$left_after" != "0" ]; then
      say "STAGE $3 LANDED TOO LITTLE: $left_after units still unported in $(dirname "$1") — refusing to advance to the next sub-campaign (that is what started levels 2-3 over unported levels 0-1); work already landed is SAFE on $BRANCH — STOPPING"
      exit 5
    fi
  fi
}

gate() { # $1 tag
  say "GATE $1 start (cargo check + test -p deeptools)"
  ( cd "$ROOT" && cargo check -p deeptools > "$LOGDIR/gate-$1.log" 2>&1 )
  say "GATE $1 check exit=$? :: $(grep -cE '^error' "$LOGDIR/gate-$1.log") error lines"
  grep -E '^error' "$LOGDIR/gate-$1.log" | head -3 >> "$TRACE"
  ( cd "$ROOT" && cargo test -p deeptools > "$LOGDIR/test-$1.log" 2>&1 )
  say "TEST $1 exit=$? :: $(grep -E '^test result' "$LOGDIR/test-$1.log" | tr '\n' ' ')"
}

say "CAMPAIGN START: bridge 3, SentientIR -> ProgIR (dcc D76), 130 units, 8 levels, 3 sub-campaigns"
count
regen

S1=levels0-1-leaves-and-instruction-builders
S2=levels2-3-computes-and-transfers
S3=levels4-7-uniform-regions-and-the-driver

stage $S1/port-remainder.json port   sc1-port
stage $S1/review.json          review sc1-review
gate sc1
stage $S2/port-remainder.json port   sc2-port
stage $S2/review.json          review sc2-review
gate sc2
stage $S3/port-remainder.json port   sc3-port
stage $S3/review.json          review sc3-review
gate sc3
say "CAMPAIGN DRIVER DONE"
