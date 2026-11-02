# Issue tracker: GitHub

Issues live on GitHub at `AI-native-Systems-Research/scratchy`. Specs and plans don't: they're checked-in documents under `docs/superpowers/specs/` and `docs/superpowers/plans/` (see `CONTRIBUTING.md`, Tier 2).

Every `gh issue` / `gh label` command below passes `-R AI-native-Systems-Research/scratchy`. Keep it: a bare `gh` resolves to whatever your clone's default remote is, which on a fork files issues where the team never sees them. Needs `gh` 2.94+.

## Conventions

- **Create an issue**: `gh issue create -R AI-native-Systems-Research/scratchy --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **List issues**: `gh issue list -R AI-native-Systems-Research/scratchy --state open --limit 200 --json number,title,labels`, with `--label` filters as needed. `--limit` defaults to 30, which silently drops most of the backlog. List only these fields, then view just the issue you're working.
- **Read an issue**: `gh issue view <number> -R AI-native-Systems-Research/scratchy --json number,title,body,labels,comments`.
- **Comment**: `gh issue comment <number> -R AI-native-Systems-Research/scratchy --body "..."`
- **Apply / remove labels**: `gh issue edit <number> -R AI-native-Systems-Research/scratchy --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> -R AI-native-Systems-Research/scratchy --comment "..."`
- **Sub-issue**: `gh issue create -R AI-native-Systems-Research/scratchy --parent <parent> ...`, or `gh issue edit <parent> -R AI-native-Systems-Research/scratchy --add-sub-issue <child>`.
- **Blocking**: `gh issue create -R AI-native-Systems-Research/scratchy --blocked-by <n>,<n> ...`, or `gh issue edit <child> -R AI-native-Systems-Research/scratchy --add-blocked-by <n>`. Issue numbers, not database ids.

External PRs are not a triage surface here; `/triage` covers issues only.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42 -R AI-native-Systems-Research/scratchy` and fall back to `gh issue view 42 -R AI-native-Systems-Research/scratchy`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue. If what it's publishing is a spec or plan, write it under `docs/superpowers/` instead.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as its sub-issues.

- **Setup, once per repo**: the `wayfinder:*` labels don't exist yet and `gh` won't create them on use. Create them first (`--force` makes this safe to repeat):
  ```bash
  for l in map research prototype grilling task; do
    gh label create "wayfinder:$l" -R AI-native-Systems-Research/scratchy --force
  done
  ```
- **Map**: an issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: a sub-issue of the map (see **Sub-issue**), labelled `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, it's assigned to the driving dev.
- **Blocking**: native issue dependencies (see **Blocking**). A ticket is unblocked when every blocker is closed.
- **Frontier query**: the map's open children with no open blocker and no assignee; first in map order wins.
  ```bash
  gh issue view <map> -R AI-native-Systems-Research/scratchy --json subIssues \
    --jq '.subIssues.nodes[] | select(.state == "OPEN") | .number' |
  while read n; do
    gh issue view "$n" -R AI-native-Systems-Research/scratchy --json number,assignees,blockedBy \
      --jq 'select((.assignees | length) == 0 and ([.blockedBy.nodes[] | select(.state == "OPEN")] | length) == 0) | .number'
  done | head -1
  ```
- **Claim**: `gh issue edit <n> -R AI-native-Systems-Research/scratchy --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> -R AI-native-Systems-Research/scratchy --body "<answer>"`, then `gh issue close <n> -R AI-native-Systems-Research/scratchy`, then append a context pointer (gist + link) to the map's Decisions-so-far.
