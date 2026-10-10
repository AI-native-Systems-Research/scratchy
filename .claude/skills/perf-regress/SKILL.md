---
name: perf-regress
description: Perf regression gate. Use to check a change against the locked per-arch perf baseline on this Mac (lockdown run or gate run), or to re-record the baseline and produce the table a perf: commit carries. Companion gate to metal-bench (which compares against other frameworks).
---

# Perf regression gate (this Mac vs its locked baseline)

Mechanics, cells and policy are in `docs/BENCHMARKING.md` ("The perf
regression gate"); flags are in the script header (`--help`). The script is
`scripts/bench_perf_regress.sh` and everything it needs is self-contained.

**What a run is:** one `scr bench latency` per (model, cell) over the
in-script manifest — one canonical model per supported arch — against a
binary built from the worktree it runs in. Three cells per model (`decode1`,
`fold4`, `prefill2k`), prefix caching off, metric = p50 per-iteration
latency. Compare exits 1 on any cell outside the band, either direction.

**Before you start:**
- The worktree is at the commit under test, tree clean (`--skip-build`
  reuses a binary you built there on purpose; say so in the PR).
- Mac on AC power; no other scr process alive (the script refuses both).
- The manifest's preset guesses are verified by the loader at run time —
  a wrong preset fails the cell loudly, it never silently runs dense.

## Gate run (does a change regress anything?)

```
scripts/bench_perf_regress.sh --markdown
```

- `--quick` for the small-model tier when the change is narrow; a change
  that touches a shared lowering/kernel path deserves the full pass over
  every cached model.
- Every cell must report `ok`. `OUT OF BAND` in either direction, a FAILED
  cell, or a SKIPPED row that the baseline covers are all findings to put
  in the PR — never reasons to rerun until green.

**Done when:** the table is printed, every run cell is `ok`, and the table
is in the PR body.

## Lockdown run (record / re-record this chip's baseline)

```
scripts/bench_perf_regress.sh --update --markdown
```

- Baselines live in `scripts/perf-regress/baselines/<chip>.json` and are
  committed with the change they legitimize — a `perf:` commit carries the
  re-recorded baseline and the `--markdown` table in its message.
- `--update` only rewrites the models it ran; rows for models not cached on
  this Mac keep their old numbers. That is by design — a chip's locked
  coverage is exactly what it ran.
- Never hand-edit a baseline JSON; rerun with `--update`.

**Done when:** the baseline JSON is written, the table is printed, and both
are staged with the change they measure.
