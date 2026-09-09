#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════════════════════
# AUTO-PROMOTER — every 5 minutes, land whatever a finished batch has already committed.
#
# ⭐ WHY IT EXISTS: a crustify stage promotes only when its SLOWEST batch finishes. That hid 54
# finished units for five hours on the senpass campaign.
#
# ⛔ THE SELECTION ORDER IS THE WHOLE TRICK, AND GETTING IT WRONG MADE THIS FIRE ZERO TIMES IN
# THREE HOURS. A version that took the FURTHEST AHEAD first locked onto a stale branch (ahead=5,
# behind=16) that could never fast-forward, and never moved again. The order must be:
#     1. consider only session branches named in THIS campaign's own stage logs
#        (session refs are repo-wide — another campaign's branch is not ours to promote);
#     2. filter to those with behind == 0 (i.e. they contain HEAD and CAN fast-forward);
#     3. only THEN take the furthest ahead.
#
# ⛔ NO `mapfile`: this host's /bin/bash is 3.2.57 and has no such builtin — the first version of
# this loop died on its first iteration, which is one more way the promoter fires zero times.
#
# It only ever fast-forwards. A branch that needs a rebase is left to the driver's `promote`, which
# does that under the stage's own accounting.
# ══════════════════════════════════════════════════════════════════════════════════════════════
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/ddc
CAMPDIR=$ROOT/crustify/campaigns/ddc
LOGDIR=$CAMPDIR/logs
BRANCH=ddc-campaign
OUT=$ROOT/crates/compiler/deeptools/src/schedule
INTERVAL=300

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "autopromoter up (interval ${INTERVAL}s, watching $LOGDIR/driver-*.log)"
while true; do
  sleep $INTERVAL
  cur=$(git -C "$ROOT" branch --show-current 2>/dev/null)
  if [ "$cur" != "$BRANCH" ]; then
    say "skip: worktree is on '$cur', not '$BRANCH' (a promote or rebase is mid-flight)"
    continue
  fi
  # 1. branches this campaign's own stage logs name.
  brs=""
  while IFS= read -r line; do
    brs="$brs $line"
  done < <(grep -ohE "crustify/session/[a-z]+-[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9a-f]{4}" \
             "$LOGDIR"/driver-*.log 2>/dev/null | sort -u)
  case "$brs" in *[![:space:]]*) ;; *) continue ;; esac

  best=""; best_ahead=0
  for b in $brs; do
    git -C "$ROOT" rev-parse --verify "$b" >/dev/null 2>&1 || continue
    read -r behind ahead < <(git -C "$ROOT" rev-list --left-right --count "HEAD...$b" 2>/dev/null \
                              | awk '{print $1, $2}')
    [ -z "$ahead" ] && continue
    # 2. only branches that CONTAIN HEAD can fast-forward
    [ "$behind" != "0" ] && continue
    # 3. among those, the furthest ahead
    if [ "$ahead" -gt "$best_ahead" ]; then best_ahead=$ahead; best=$b; fi
  done

  if [ -z "$best" ]; then continue; fi
  if git -C "$ROOT" merge --ff-only "$best" >/dev/null 2>&1; then
    filled=$(grep -rhoE '///[[:space:]]*Replaces:[[:space:]]*e[0-9]+_[A-Za-z0-9_]+' "$OUT" 2>/dev/null \
              | grep -oE 'e[0-9]+_[A-Za-z0-9_]+' | sort -u | wc -l | tr -d ' ')
    say "PROMOTED $best (+$best_ahead commits) -> $(git -C "$ROOT" rev-parse --short HEAD); filled=$filled"
  else
    say "ff refused for $best despite behind=0 — leaving it to the driver's promote"
  fi
done
