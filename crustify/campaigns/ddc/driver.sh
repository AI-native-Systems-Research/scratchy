#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════════
# ddc / L3-SCHEDULER CAMPAIGN DRIVER
#   the scheduling and address-placement stage of deeptools: 382 units, 4 stages, levels 0..9
#
# port wave -> promote -> review wave -> promote -> gate, once per sub-campaign.
# Every stage's own log is under logs/; this script's trace is logs/driver.trace.
#
# Every guard below is a lesson that cost a launch. Do not simplify one away.
# ══════════════════════════════════════════════════════════════════════════════════════════════
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/ddc
CAMPDIR=$ROOT/crustify/campaigns/ddc
LOGDIR=$CAMPDIR/logs
TRACE=$LOGDIR/driver.trace
UNITS=$ROOT/crustify-ddc/UNITS.tsv
TOOLS=$ROOT/crustify-ddc/tools
# ⛔ PROGRESS IS COUNTED UNDER THE CAMPAIGN'S OWN OUTPUT DIRECTORY. Session refs and eNNN_ unit
# names are both repo-wide — bridge1-campaign already carries an e001_checkConstraints and an
# e002_createDataConnectMetadata in src/bridges/ — so a branch-based or crate-wide count returns
# another campaign's number.
OUT=$ROOT/crates/compiler/deeptools/src/schedule
BRANCH=ddc-campaign
# ⭐ 4, not 8: campaign 5 (senpass) is LIVE on this host with 6 agents of its own.
PARALLEL=4

# ⛔⛔ ONE SHARED target/ FOR EVERY AGENT WORKTREE. crustify spawns each porting agent in its own
# worktree under $ROOT/crustify/.worktrees/, and each one running `cargo check -p deeptools` builds
# its OWN target/ by default. That is where bridge 2's 10G went (dozens of leftover agent
# worktrees), and it nearly tripped campaign 5's disk floor. Exported here so the crustify child and
# every agent it spawns inherit it.
#
# Cargo takes a file lock per target dir, so concurrent `cargo check` calls SERIALISE rather than
# corrupt. That is the trade: a few seconds of waiting per batch against ~15G of duplicated build
# artifacts. Model time dominates either way.
export CARGO_TARGET_DIR=$ROOT/target

mkdir -p "$LOGDIR"
say() { echo "[$(date +%H:%M:%S)] $*" >> "$TRACE"; }

# ── SINGLE-INSTANCE LOCK ──────────────────────────────────────────────────────────────────────
# ⛔⛔ THREE SEPARATE INCIDENTS CAME FROM TWO DRIVERS RUNNING AT ONCE. They share every log path
# under $LOGDIR, so each overwrites the other's stage log — and `promote` reads the session branch
# out of that log. One driver then promoted a branch the other owned, HEAD moved sideways, and the
# younger driver's promote failed "not a fast-forward" with nine batches of finished work stranded.
#
# ⛔⛔ NEVER `rm -f` THIS LOCK TO CLEAR A "STALE" ONE. That was done once while pid 17581 was still
# alive and it let a second driver start. The `kill -0` below already handles a genuinely stale
# lock: the pid is gone, the check fails, the driver starts. If a LIVE driver holds it, kill the
# driver AND its `crustify` child — killing the bash script alone reparents the child to init,
# which keeps spawning agents. This was defeated five times.
#     pkill -TERM -P "$(cat .driver.lock)" ; kill -TERM "$(cat .driver.lock)"
LOCK=$CAMPDIR/.driver.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "driver already running as pid $(cat "$LOCK") — refusing to start a second" >&2
  say "REFUSED: driver already running as pid $(cat "$LOCK")"
  exit 9
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"; [ -n "$AUTOPID" ] && kill "$AUTOPID" 2>/dev/null' EXIT INT TERM
say "LOCK acquired by pid $$"

# ── COUNTING ──────────────────────────────────────────────────────────────────────────────────
# ⛔⛔ OUTSTANDING WORK COMES FROM UNITS.tsv, NEVER FROM ANCHORS IN THE TREE. A deleted
# `crustify:todo:` anchor is indistinguishable from a finished unit: bridge 2 printed
# CAMPAIGN DRIVER DONE having ported 235 of 384, silently losing 149 that ran consecutively from
# e214 to e384 — the biggest functions in its span. `unported` is scheduled-minus-filled read from
# UNITS.tsv and is what completion is judged on; openTodo is a DIAGNOSTIC only, and a gap between
# it and `unported` means anchors have gone missing again.
FILLED=0
count() {
  filled=$(grep -rhoE '///[[:space:]]*Replaces:[[:space:]]*e[0-9]+_[A-Za-z0-9_]+' "$OUT" 2>/dev/null \
            | grep -oE 'e[0-9]+_[A-Za-z0-9_]+' | sort -u | wc -l | tr -d ' ')
  open=$(grep -rhoE 'crustify:todo:[[:space:]]*e[0-9]+_[A-Za-z0-9_]+' "$OUT" 2>/dev/null \
            | grep -oE 'e[0-9]+_[A-Za-z0-9_]+' | sort -u | wc -l | tr -d ' ')
  scheduled=$(tail -n +2 "$UNITS" | grep -c .)
  unported=$(( scheduled - filled ))
  FILLED=$filled
  say "ANCHORS: filled=$filled/$scheduled  unported=$unported  openTodo=$open"
  if [ "$unported" -gt 0 ] && [ "$open" -lt "$unported" ]; then
    say "⛔ ANCHOR LOSS: $unported unported but only $open TODO markers — $(( unported - open )) anchors were DELETED WITHOUT A PORT. Re-run gen_homes.py (it refuses to touch ported files) to re-anchor, then --remainder the schedules."
  fi
}

# ── PROMOTE: FAST-FORWARD, THEN REBASE ────────────────────────────────────────────────────────
# ⭐ NEVER ff-or-die. A sideways HEAD — another promote, a hand commit — must not strand a stage's
# finished work on a session branch. The rebase replays the batches onto HEAD; only a genuine
# content conflict stops us, and it stops with the branch intact and named.
promote() { # $1 = a stage log to read the session branch out of
  br=$(grep -ohE "crustify/session/[a-z]+-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{4}" "$1" 2>/dev/null | tail -1)
  if [ -z "$br" ]; then say "PROMOTE: no session branch found in $1"; return 0; fi
  if git -C "$ROOT" merge --ff-only "$br" >> "$TRACE" 2>&1; then
    say "PROMOTE ok (ff): $br -> $(git -C "$ROOT" rev-parse --short HEAD)"; return 0
  fi
  say "PROMOTE: not a fast-forward, rebasing $br onto $(git -C "$ROOT" rev-parse --short HEAD)"
  if git -C "$ROOT" rebase HEAD "$br" >> "$TRACE" 2>&1; then
    git -C "$ROOT" checkout "$BRANCH" >> "$TRACE" 2>&1
    if git -C "$ROOT" merge --ff-only "$br" >> "$TRACE" 2>&1; then
      say "PROMOTE ok (rebased): $br -> $(git -C "$ROOT" rev-parse --short HEAD)"; return 0
    fi
  fi
  git -C "$ROOT" rebase --abort >> "$TRACE" 2>&1
  git -C "$ROOT" checkout "$BRANCH" >> "$TRACE" 2>&1
  say "PROMOTE FAILED (rebase conflicted): $br — the work is SAFE on that branch — STOPPING"
  exit 3
}

disk() {
  free=$(df -g "$ROOT" | awk 'NR==2{print $4}')
  say "DISK: ${free}G free"
  if [ "$free" -lt 12 ]; then say "DISK BELOW 12G — STOPPING"; exit 4; fi
}

# ⭐ Prune promoted agent worktrees BETWEEN stages: each carries its own target/ and this host is
# tight. Only worktrees whose branch is already an ancestor of HEAD are removed, so nothing
# unpromoted is ever touched.
prune() {
  n=0
  for wt in "$ROOT"/crustify/.worktrees/*/; do
    [ -d "$wt" ] || continue
    b=$(git -C "$wt" branch --show-current 2>/dev/null) || continue
    if [ -n "$b" ] && git -C "$ROOT" merge-base --is-ancestor "$b" HEAD 2>/dev/null; then
      git -C "$ROOT" worktree remove --force "$wt" >> "$TRACE" 2>&1 && n=$((n+1))
    fi
  done
  say "PRUNE: removed $n promoted agent worktree(s)"
}

# ── ONE STAGE ─────────────────────────────────────────────────────────────────────────────────
stage() { # $1 schedule (relative to $CAMPDIR)  $2 objective  $3 tag
  n=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['unit_count'])" \
        "$CAMPDIR/$1" 2>/dev/null || echo 1)
  # ⭐ SKIP AN EMPTY SCHEDULE. The remainder is rewritten after every promote, so a finished level
  # leaves 0. Without this the driver hands crustify an empty wave list — or re-runs a REVIEW over
  # a level already reviewed, which once started 9 agents re-reviewing 142 done functions.
  if [ "$n" = "0" ]; then say "STAGE $3 SKIPPED (0 units left in $1)"; return 0; fi
  disk
  count; before=$FILLED
  rc=1
  for attempt in 1 2 3; do
    say "STAGE $3 start ($2, $1, $n units) attempt $attempt"
    crustify --parallel-max $PARALLEL "$ROOT" . translate "$CAMPDIR/$1" --objective "$2" \
      > "$LOGDIR/driver-$3.log" 2>&1
    rc=$?
    say "STAGE $3 attempt $attempt exit=$rc"
    [ $rc -eq 0 ] && break
    # ⭐ RETRY ONLY API FAILURES. Every batch lost so far died with `API Error: Can't reach the API
    # server (ENOTFOUND)` after 120-180 turns. Anything else fails fast: a config or gate failure
    # will fail identically three times and only burn an hour.
    if ! grep -q "ENOTFOUND\|Can't reach the API server\|exited 1 for TranslateAgent" \
           "$LOGDIR/driver-$3.log"; then
      say "STAGE $3 failed for a reason that is NOT the API — not retrying"; break
    fi
    say "STAGE $3 retrying in 120s (API failure)"; sleep 120
  done
  grep -E '^\[crustify\] [0-9]+ failure' "$LOGDIR/driver-$3.log" >> "$TRACE" 2>/dev/null
  promote "$LOGDIR/driver-$3.log"
  count
  # ⛔ A DEAD STAGE MUST NOT ADVANCE THE CAMPAIGN. The wave barrier only holds WITHIN a stage, so a
  # port stage that landed nothing would otherwise let the next level start on absent dependencies.
  if [ "$2" = "port" ] && [ "$FILLED" -le "$before" ]; then
    say "⛔ STAGE $3 LANDED 0 NEW UNITS (filled still $FILLED) — the next level would start on absent dependencies — STOPPING"
    exit 5
  fi
  python3 "$TOOLS/gen_campaign.py" --remainder "$ROOT" >> "$TRACE" 2>&1
  prune
}

gate() { # $1 tag
  say "GATE $1 start (cargo check + test -p deeptools)"
  ( cd "$ROOT" && cargo check -p deeptools > "$LOGDIR/gate-$1.log" 2>&1 )
  say "GATE $1 check exit=$?"
  grep -E '^error' "$LOGDIR/gate-$1.log" | head -5 >> "$TRACE"
  ( cd "$ROOT" && cargo test -p deeptools > "$LOGDIR/test-$1.log" 2>&1 )
  say "TEST $1 exit=$? :: $(grep -E '^test result' "$LOGDIR/test-$1.log" | tr '\n' ' ')"
}

# ── AUTO-PROMOTER ─────────────────────────────────────────────────────────────────────────────
# ⭐ A stage promotes only when its SLOWEST batch finishes, and that once hid 54 finished units for
# five hours. This fires every 5 min against the branches THIS campaign's own stage logs name.
"$CAMPDIR/autopromote.sh" >> "$LOGDIR/autopromote.log" 2>&1 &
AUTOPID=$!
say "AUTOPROMOTER started as pid $AUTOPID"

say "CAMPAIGN START: 382 units, 4 stages of runDdc, levels 0..9, --parallel-max $PARALLEL"
count

stage sc1-level0-leaves/port.json            port   sc1-port
stage sc1-level0-leaves/review.json          review sc1-review
gate sc1
stage sc2-level1-helpers/port.json           port   sc2-port
stage sc2-level1-helpers/review.json         review sc2-review
gate sc2
stage sc3-levels2-3-bodies/port.json         port   sc3-port
stage sc3-levels2-3-bodies/review.json       review sc3-review
gate sc3
stage sc4-levels4-6-drivers/port.json        port   sc4-port
stage sc4-levels4-6-drivers/review.json      review sc4-review
gate sc4
stage sc5-levels7-9-stage-roots/port.json    port   sc5-port
stage sc5-levels7-9-stage-roots/review.json  review sc5-review
gate sc5

count
say "CAMPAIGN DRIVER DONE — filled=$FILLED of $(tail -n +2 "$UNITS" | grep -c .). ⛔ THAT IS ONLY DONE IF THOSE TWO NUMBERS MATCH."
