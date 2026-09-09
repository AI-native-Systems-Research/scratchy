#!/bin/bash
# One-shot commit of the ddc / L3-scheduler campaign scaffolding.
#
# WHY THIS SCRIPT EXISTS: the session that generated the scaffolding was isolation-pinned to a
# different worktree, so its git operations (and its subagent's) were refused by the harness guard.
# Everything below is already on disk and verified; only the commit remains.
#
# Run it from anywhere:  bash crustify/campaigns/ddc/commit-scaffold.sh
set -euo pipefail
ROOT=/Users/nickm/git/scratchy/.claude/worktrees/ddc
cd "$ROOT"

br=$(git branch --show-current)
if [ "$br" != "ddc-campaign" ]; then
  echo "refusing: expected branch ddc-campaign, on '$br'" >&2
  exit 1
fi

# The two deliberately-corrupted verifier fixtures must never be committed: they would read as
# authority. crustify-ddc/work/.gitignore already excludes them.
git add crustify-ddc crustify/crates.json crustify/build.json \
        crustify/wavefront/wavefront-config.json crustify/campaigns/ddc \
        crates/compiler/deeptools/src/schedule crates/compiler/deeptools/src/lib.rs

echo "--- staged ---"
git diff --cached --stat | tail -5

git commit -F - <<'MSG'
chore(deeptools): scaffold the ddc / L3-scheduler crustify campaign

The scheduling and address-placement stage of deeptools -- runDdc's four
stages -- enumerated, extracted and scheduled. 382 units over dependency
levels 0..9, from L3DlOpsScheduler (144), ddc/ (178), ddc/ddl/ (44) and
dcg_manager (16). No unit is ported yet: every home carries only
`// crustify:todo:` anchors.

444 definitions found by comment- and string-aware brace matching, 62
excluded with a stated reason each in EXCLUSIONS.tsv and none for being
hard. Levels computed over the SCC condensation; the span is acyclic.
The four consolidated C++ TUs were verified INDEPENDENTLY -- a second
character state machine, never re-slicing with the extractor's own
(file, line, length) -- at 382/382 bodies byte-identical to the
authority, 1,069,773 bytes compared, with both negative controls
detected.

Authority: /Users/nickm/git/deeptools-src @ a0d29abbed.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG

echo "--- committed ---"
git log --oneline -1
git status --short
