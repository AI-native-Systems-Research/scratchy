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
# ⭐ 14, RAISED FROM 4. Concurrency was the binding constraint, not model time: sc3 held 67 units in
# 14 BATCHES and a cap of 4 ran them in four sequential waves, taking 6h58m. sc4 is 8 batches and sc5
# is 5, so 14 clears each remaining stage in one wave. The sibling campaign is nearly done, so the
# host is ours; ~375MB of agent worktree per batch against 60G free, and the shared CARGO_TARGET_DIR
# below stops the build artifacts multiplying.
PARALLEL=14

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
# ⛔⛔ STAGE MARKERS LIVE OUTSIDE $CAMPDIR. crustify link_shares the campaign directory into every
# agent worktree, and that turned each .done-<tag> into a symlink POINTING AT ITSELF: `[ -f ]`
# follows the link, finds nothing and reports absent, so the driver re-ran a completed 256-unit
# review; while crustify's own `open(path,'x')` sees the link and dies `FileExistsError`, which
# failed all 7 batches of sc2-port in seven seconds. One cause, both failures. $STATE is not under
# crustify/ and not under target/ (which gets cargo clean'd), so nothing links or sweeps it.
STATE=$ROOT/.campaign-done
mkdir -p "$STATE"
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
  # A COMPLETED STAGE MUST NOT RE-RUN ON RESTART. A port stage is idempotent through its remainder
  # (below), but a REVIEW stage is not: its schedule always names every unit of the level. On the
  # sibling campaign a restart began re-reviewing all 256 units of sc1, already reviewed and gated
  # green hours earlier, and this driver's own comment records the same thing happening to 142.
  # Each stage writes a marker once it has landed and promoted; delete one to force a re-run.
  # ⛔⛔ A MARKER SAYS THE STAGE RAN, NOT THAT THE LEVEL IS COMPLETE — AND THE REMAINDER OVERRULES IT.
  # senpass's sc2-port exited 0 on 2026-09-09 while 43 of its units sat unpromoted on AGENT branches
  # that had run ahead of their session branch. The stage was therefore marked done, its remainder
  # still listed 49 units, and every later run SKIPPED it — so the campaign would have reported DONE
  # with 49 units never ported. Completeness is the remainder's answer, never a marker's.
  if [ -f "$STATE/.done-$3" ]; then
    left=0
    if [ "$2" = "port" ]; then
      rj="$CAMPDIR/${1%/port.json}/port-remainder.json"
      [ -f "$rj" ] && left=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['unit_count'])" "$rj" 2>/dev/null || echo 0)
    fi
    if [ "${left:-0}" = "0" ]; then
      say "STAGE $3 SKIPPED (already completed: .done-$3 present, remainder empty)"
      return 0
    fi
    say "⛔ STAGE $3 IS MARKED DONE BUT ITS REMAINDER STILL HOLDS $left UNIT(S) — the marker recorded that the stage RAN, not that the level is complete. RE-RUNNING it."
    rm -f "$STATE/.done-$3"
  fi
  # A PORT STAGE MUST RUN THE REMAINDER, NOT THE ORIGINAL SCHEDULE. `gen_campaign.py --remainder`
  # writes `port-remainder.json`; the stage list names `port.json`, which still holds every unit the
  # level ever had, so the empty-schedule skip below can never fire and a restart re-ports finished
  # work. Preferring the remainder makes every restart self-correcting.
  if [ "$2" = "port" ]; then
    rem="${1%/port.json}/port-remainder.json"
    if [ -f "$CAMPDIR/$rem" ]; then
      say "STAGE $3 using the remainder ($rem) rather than $1"
      set -- "$rem" "$2" "$3"
    fi
  fi
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
    # ⛔⛔ AN AUTH FAILURE IS NOT A TRANSIENT — CHECK IT BEFORE THE RETRY PREDICATE. A revoked or
    # expired key returns `401 Authentication Error` and the agent still exits 1 for TranslateAgent,
    # so it matched the transient predicate below and burned all three attempts at ~200s per batch.
    # That is exactly how BOTH campaigns died on 2026-09-09: 22 of 22 batches on one and 14 on the
    # other, every one a 401, then the dead-stage guard stopped the run. Retrying cannot fix a key.
    if grep -q "401 Authentication Error\|Invalid proxy server token\|Failed to authenticate" \
         "$LOGDIR/driver-$3.log"; then
      say "⛔ STAGE $3 FAILED TO AUTHENTICATE (401) — the API key is dead, not the network. Fix the key and re-run; NOT retrying."
      break
    fi
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
  # The stage landed work and promoted it, so a restart must not redo it.
  touch "$STATE/.done-$3"
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
