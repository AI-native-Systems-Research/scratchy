---
name: spyre-bench
description: Spyre A/B sweeps, triton-scratchy against torch-spyre on a Spyre card. Use to put a sweep's numbers on the site's Spyre page (spyre.html) as site/data/spyre/<date>-<sha8>.json, covering what to measure, every field of the file (including the page's per-sweep text: what moved, open levers, notes, a target), the checks the build runs, the preview and the PR.
---

# Spyre A/B sweep (one card → one JSON → the site)

`spyre.html` charts the newest file in `site/data/spyre/` and lists every
file in its "About these numbers" table. Everything the page says about one
sweep comes from that sweep's file: the cells, and also the text (what
moved since the previous sweep, the open levers, the notes under each
section, the target tile). So a new sweep is **one new file and no code**.
The page code knows nothing about any particular sweep; keep it that way.

Never edit or delete an older sweep's file. It is the record of what that
sweep measured, and the About table links to it.

Each step ends with a **Done when** check. Meet it before you start the
next step.

## 1. The commit

The page says "triton-scratchy at `<commit>`", so the commit has to be one
anybody can check out: on upstream `main`, and from a clean tree.

```bash
up=$(git remote -v | awk '/AI-native-Systems-Research\/scratchy(\.git)? \(fetch\)/ {print $1; exit}')
git fetch -q "$up" main
git merge-base --is-ancestor HEAD "$up/main" && echo "on upstream main" || echo "NOT on upstream main"
git status --short
```

**Done when:** it prints `on upstream main` and `git status --short` prints
nothing, or the user has said to measure this commit anyway.

## 2. Measure

Measure on the Spyre card only. Numbers from the emulator or the cost model
never go in a cell; a cost-model number can go in `target` (step 3).

**The harness is not checked in yet.** The two sides have been measured by
different harnesses. The 2026-10-17 sweep recorded these:

- torch-spyre: `vllm bench serve`, 256 output tokens, prompts of 123, 1083
  and 10300 tokens.
- triton-scratchy: `scr serve`, greedy, 64–128 output tokens, prompts of
  1051 and 10086 tokens.

They are not matched (different prompt lengths and output lengths), which is
why the page sorts cells into context buckets (`~123`, `~1k`, `~10k`) and
compares ratios, not absolutes. Until there is one shared harness:

- keep each side's harness, prompt lengths and buckets as in the newest
  sweep, so this sweep's ratios compare with the last one's;
- if anything changes, say exactly what in that side's `harness`.

What a full sweep measures, per model and dtype:

- **decode:** inter-token latency in ms (the median), one request at a time
  (batch 1), at every bucket;
- **prefill:** time to first token in ms, one request at a time (batch 1),
  at every bucket;
- **batch scaling:** the same cells at the other batch rungs the page rows
  its grids by — 4 and 8 requests decoded at once (16 and up corrupt on
  this stack, so 8 is the cap). Past batch 1, record both where the
  harnesses give both: decode as `itl_ms` (each request's median
  inter-token latency under that batch, TPOT — the like-for-like metric,
  which the page prefers for a decode square) and as `tok_s` (decoded
  tokens per second at that batch, for the concurrency blocks); a decode
  square with no `itl_ms` cell lights from its `tok_s` cell. `conc` in the
  file is the batch: the offered concurrency, both stacks serving that many
  requests at once.

A cell needs both sides, measured in this sweep. When one side can't run a
cell (torch-spyre's fp8 path past short context, say), leave the cell out:
the page draws it as a striped square. Say why in a cell `note` on a cell
that is there, or in `notes`. Never put a 0, a guess or an older run's
number in a cell. An older number can go in a note, with its date.

**Done when:** you have both sides' numbers for every cell you'll write, and
each side's harness line says how they were taken.

## 3. Write the file

Start from a copy of the newest file in `site/data/spyre/`, then replace
**everything** in it, the text included. Name it
`<date>-<first 8 characters of the commit>.json`, e.g.
`2026-10-17-b7c4c889.json`.

| field | what it holds |
|---|---|
| `schema` | `1` |
| `date` | the day of the sweep, `YYYY-MM-DD`, the same as in the file name |
| `contexts` | each bucket and the prompt tokens it means on scratchy's side: `{ "~123": 123, "~1k": 1051, "~10k": 10086 }` |
| `torch_spyre` | `measured` (date), `host` (optional), `harness` (step 2) |
| `triton_scratchy` | `measured` (date), `commit` (the full commit as run), `harness` |
| `cells` | one per measured point: `model`, `dtype` (`fp16` or `fp8`), `metric` (`itl_ms`, `ttft_ms` or `tok_s`), `conc` (the batch: requests decoded at once; 1 if left out), `context` (a `contexts` key), `torch`, `scratchy`, and an optional `note` the page prints under the cell's section |
| `since` | the PRs merged on scratchy's side since the previous sweep: `"#563–#583"` (leave out for a first sweep) |
| `moved` | one row per change: `{ "pr": "#573", "what": "dxp chunk explorer restored: 2b 10k TTFT 18.1 → 8.9 s" }` |
| `levers` | where torch-spyre still leads, and the work: `{ "lever": "8b 10k decode", "work": "torch-spyre 1.17× ahead — …" }` |
| `notes` | paragraphs under a section: `{ "decode": [...], "prefill": [...], "conc": [...], "fp8": [...] }` |
| `target` | optional: a cell's target beyond parity, a headline tile of its own: `label`, `model`, `dtype`, `metric`, `conc`, `context`, `value` (in the metric's unit), `why` |

Write the raw values in each side's own unit. The page works out every ratio
(× faster) itself, and which way is better (lower for ms, higher for tok/s).

The text fields are where a copied file goes wrong. Every number in `moved`,
`levers`, `notes` and `target` has to be true of **this** sweep: a lever that
says "torch-spyre 1.17× ahead" must match this sweep's cell, and a lever that
this sweep closed comes out (and probably goes into `moved`). If you don't
know a field's new text, leave the field out (the page leaves its section
out) and ask the user. Don't carry the old text over.

**Done when:** the file is named for its `date` and commit, and every number
in its text fields matches its cells or names where else it comes from.

## 4. Build

```bash
cd site && cargo run
```

The build checks every file in `site/data/spyre/` and lists every problem at
once: a file not named `<date>-<sha8>.json`, a schema other than 1, a missing
`triton_scratchy.commit`, a field it doesn't know (a misspelt key fails here
rather than vanishing from the page), a cell at a bucket `contexts` lacks, a
value missing or not above 0, a cell written twice, a `target` with no cell
under it. CI's site build fails on the same problems, so fix them here.

**Done when:** it prints `spyre page: N sweeps`, one more than before.

## 5. Preview

The dev server keeps serving, and re-renders on every change, until it's
stopped, so run it in the background:

```bash
site/open.sh --port 8001 --no-open    # then open http://localhost:8001/spyre.html
```

Check the page against the file:

- the intro names the new date, commit and `since`;
- the headline tiles and the "At a glance" heat maps show every model you
  measured, each grid running the batch rungs 1, 4, 8 down (a striped
  square is a cell you left out, and only those);
- "What moved" and "Open levers" read as this sweep's, not the last one's;
- the About table lists the new sweep at the top and the older ones below it.

**Done when:** all four hold.

## 6. PR

Commit only the new file, on a new branch, and open a PR titled:

```
data(site): torch-spyre A/B sweep at <sha8>
```

The commit message says when and where each side ran, and with what
harness.

**Done when:** the user has the PR link.
