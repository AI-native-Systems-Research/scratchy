#!/bin/bash
# ⭐ LESSON 7. Appends a line to driver.trace whenever a session branch gains a commit, so progress is
# visible BETWEEN promotes. The driver only counts anchors at stage boundaries, which leaves the
# campaign branch looking dead for ~3h per wave while batches are in fact landing on the session
# branch. Bridge 2 spent hours believing a live wave was hung.
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/bridge3
TRACE=$ROOT/crustify/campaigns/bridge3/logs/driver.trace
mkdir -p "$(dirname "$TRACE")"
last=-1
lastbr=""
while true; do
  br=$(git -C "$ROOT" for-each-ref --sort=-committerdate \
        --format='%(refname:short)' 'refs/heads/crustify/session/*' | head -1)
  if [ -z "$br" ]; then sleep 60; continue; fi
  n=$(git -C "$ROOT" rev-list --count bridge3-campaign.."$br" 2>/dev/null)
  if [ "$n" != "$last" ] || [ "$br" != "$lastbr" ]; then
    a=$(git -C "$ROOT" grep -hoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' "$br" \
          -- crates/compiler/deeptools/src/bridges/sentient_to_progir 2>/dev/null | sort -u | wc -l | tr -d ' ')
    t=$(git -C "$ROOT" grep -hoE 'crustify:todo: e[0-9]{3}_[A-Za-z0-9_]+' "$br" \
          -- crates/compiler/deeptools/src/bridges/sentient_to_progir 2>/dev/null | sort -u | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] BATCH: $n committed on $(basename "$br"), anchors filled=$a openTodo=$t" >> "$TRACE"
    last=$n; lastbr=$br
  fi
  sleep 90
done
