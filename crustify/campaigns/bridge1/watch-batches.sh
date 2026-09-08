#!/bin/bash
# (7) A BATCH WATCHER. Appends a line to driver.trace whenever a session branch gains a commit, so
# progress is visible BETWEEN promotes. The driver only counts anchors at stage boundaries, which
# leaves the campaign branch looking dead for hours per wave while batches are in fact landing on
# the session branch.
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/bridge1
BRANCH=bridge1-campaign
TRACE=$ROOT/crustify/campaigns/bridge1/logs/driver.trace
last=-1
while true; do
  br=$(git -C $ROOT for-each-ref --sort=-committerdate --format='%(refname:short)' 'refs/heads/crustify/session/*' | head -1)
  [ -z "$br" ] && { sleep 60; continue; }
  n=$(git -C $ROOT rev-list --count $BRANCH..$br 2>/dev/null)
  if [ "$n" != "$last" ]; then
    a=$(git -C $ROOT grep -hoE '/// Replaces: e[0-9]{3}_[A-Za-z0-9_]+' $br -- crates/compiler/deeptools/src 2>/dev/null | sort -u | wc -l | tr -d ' ')
    echo "[$(date +%H:%M:%S)] BATCH: $n committed on $(basename $br), anchors on that branch=$a" >> $TRACE
    last=$n
  fi
  sleep 90
done
