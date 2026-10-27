---
name: metal-bench
description: Metal benchmark matrix runs on a Mac. Use to run scripts/bench_metal_matrix.sh against mlx-lm, ollama, llama.cpp, vllm-metal, oMLX and mistral.rs on this Mac (a full run, or a quick smoke run to check the setup), or to add a finished run's JSON to the site's Metal page (site/data/metal).
---

# Metal benchmark run (one Mac → one JSON → the site)

Rung definitions and fairness rules are in `docs/BENCHMARKING.md`. The
script's flags are documented in its header (`--help`).

The run records its mistakes silently, as `repo.dirty`, `machine.power`, a
skipped engine or a commit that isn't upstream `main`, and every one of them
has landed on the site at least once. Each step below ends with a **Done
when** check. Meet it before you start the next step.

**Who runs what:** a `sudo` that asks for a password, or anything that opens
an editor, needs the user at a terminal. Give them the exact lines to run in
a separate terminal window, then wait until they say it's done. The agent
runs everything else.

**Three kinds of run:**
- A **quick run** is the default, and what the user means by "run the
  benchmark" unless they say otherwise. **It asks no questions:** see
  **Quick run** below. The script with no flags but `--unpinned`: this Mac's
  four models (the small set below 64 GB, the large set at 64 GB), scratchy
  alone, cold and warm, one rep, the users × prompt grid at 1, 2 and 4
  users and prompts to 8k, the same on every Mac (its memory picks the models),
  and 64-token answers, two rounds of each users count a cell.
  scratchy runs at its defaults on every Mac, and every cell runs: one
  that runs out of memory is recorded as failed. The page keeps every
  other engine's numbers, and scratchy's frozen launches, from the last full
  run, each tagged with its date. Its length is not measured yet in this
  form: the one M3 Pro quick run took 43 minutes, before every cell ran and
  at four rounds a cell. The aim is about 25 minutes. Report the run's length.
- A **full run** renews every number: `--full` adds the comparison engines
  (oMLX and oMLX TurboQuant), frozen (needs `sudo purge`), three reps, four
  rounds of each users count a cell, and the 512- and 2048-token answers at
  1 user (the decode-vs-answer-length curve). It takes hours; run it when the engines or the grid change, or
  for a published study. It's pinned (`--pin`) and goes through every
  pre-flight question in step 2.
- A **smoke run** checks the setup: `--models gpt-oss-20b --no-scaling
  --parity --out-dir /tmp/metal-smoke`. Its JSON never goes on the site, so after
  step 4 you're done.

**The rounds:** one model per class, a small one for 36 GB and a large one
for 64 GB. Sliding window: gpt-oss-20b / gpt-oss-120b-3bit. Mamba-2:
nemotron-3-nano-30b-a3b / nemotron-3-super-120b-a12b. Gated DeltaNet:
qwen3.6-35b-a3b / qwen3.5-122b-a10b. Dense: granite-4.2-3b / granite-4.2-30b. The large set is 2- and 3-bit (OptiQ or 3-bit), since no 4-bit 120B fits 64 GB: its numbers aren't comparable with the 4-bit small set on quality.
`--models small`, `large` or `full` picks a set; glm-4.5-air and the gemma
and qwen3.6-27b models run by name. The run file records the class table
(`config.classes`), and the page's overview reads its columns from it.

**Not on a 36 GB M3 Pro yet.** At scratchy's default budget it has panicked
the kernel twice mid long prefill (#780), restarting the Mac. Don't run it
there until #540 lands; tell the user why and stop.

## Quick run

Run it straight through. Don't ask about the commit, a dirty tree, the
models or the flags: report them in the closing summary instead. Stop and
tell the user only for:
- **Battery:** `pmset -g batt | head -1` doesn't say `'AC Power'`. Battery
  runs are throttled. Tell them to plug in, and wait.
- **Something else running:** the `pgrep`/`lsof` lines in step 2 show a
  process this skill didn't start. Ask before stopping it.

Then, without `--fail-fast`, so one model's build failure does not stop the
models after it. A quick run is scratchy alone, so it skips the mlx-lm parity
gate and needs no mlx venv: each model records `parity_skipped` instead.
```bash
caffeinate -dims ./scripts/bench_metal_matrix.sh --unpinned
```
and go on to step 4. In the closing summary say: the commit benchmarked and
where it is against upstream `main`, whether the tree was dirty (the JSON
records it as `repo.dirty`), the models run and skipped, and the run's
length. A clean quick run goes to the round issue (step 7, quick run), not
straight into `site/data/metal`: skip steps 5 and 6.

## 1. One-time setup (per Mac)

Install every comparison engine at the versions agreed for this round (the
pins in step 3). The script runs each one it finds and prints it as skipped
otherwise:

```bash
# mlx-lm in its own venv; the script finds it via $VIRTUAL_ENV
~/.venvs/mlx/bin/python -c 'import mlx_lm' 2>/dev/null \
  || { python3 -m venv ~/.venvs/mlx && ~/.venvs/mlx/bin/pip install -U mlx-lm; }
command -v ollama || brew install ollama           # the CLI, not the desktop app
command -v llama-server || brew install llama.cpp
command -v vllm || { brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal \
                     && brew install vllm-project/vllm-metal/vllm-metal; }
command -v mistralrs || curl --proto '=https' --tlsv1.2 -sSf \
  https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | MISTRALRS_INSTALL_TAG=<tag> sh
```

oMLX goes in from source, with its native kernels: a plain install (Homebrew
included) skips them, and the Qwen3.5/3.6 family then falls back to much
slower paths. Building them needs full Xcode, not only the Command Line Tools:

```bash
git clone --branch <tag> https://github.com/jundot/omlx ~/omlx && cd ~/omlx
python3.13 -m venv .venv && OMLX_WITH_CUSTOM_KERNEL=1 .venv/bin/pip install -e .
.venv/bin/python -c "from omlx.custom_kernels import native_kernel_status; print(native_kernel_status())"
ln -s ~/omlx/.venv/bin/omlx /opt/homebrew/bin/omlx   # the script looks for omlx on PATH
```

**Passwordless `purge`.** The frozen rung runs `sudo -n purge` throughout a
run that lasts hours. A cached `sudo -v`, even with a keepalive loop, expires
partway through. A one-time sudoers rule is the reliable fix. Have the user
run this in a separate terminal. It checks the rule with `visudo -c` before
installing it, because a broken file in `/etc/sudoers.d` breaks `sudo`
entirely:

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/sbin/purge" > /tmp/bench-purge
sudo visudo -cf /tmp/bench-purge && sudo install -m 0440 -o root -g wheel /tmp/bench-purge /etc/sudoers.d/bench-purge
```

Then check the rule the way the harness does, by asking sudo whether it may
run `purge` without a password (it lists the rule rather than running it).
`sudo -k` drops any cached unlock first, so a pass proves the rule itself
works. An older rule that also lists `/usr/bin/true` still works:

```bash
sudo -k && sudo -n -l /usr/sbin/purge >/dev/null && echo "rule works"
```

**Without the rule** (the user would rather not add it, or can't), pass
`--scenarios cold,warm` in step 3. Nothing then needs sudo, but the run has
no frozen rung, so treat it like any partial run in step 5.

**Done when:** `~/.venvs/mlx/bin/python -c 'import mlx_lm'` exits 0,
`command -v` prints a path for `ollama`, `llama-server`, `vllm`, `omlx` and
`mistralrs` (or the user has chosen to leave one out), oMLX's kernel status
shows them available, and the check prints `rule works` without asking for a
password, or the user has chosen to run without the frozen rung.

## 2. Pre-flight

A quick run runs these commands and acts on them as **Quick run** above
says, without the questions below. A full run asks them.

Find the upstream remote. In a fork clone, `origin` is the fork, and its
`main` can be well behind upstream.

```bash
up=$(git remote -v | awk '/AI-native-Systems-Research\/scratchy(\.git)? \(fetch\)/ {print $1; exit}')
git fetch -q "$up" main
echo "HEAD $(git rev-parse --short HEAD) on $(git branch --show-current);" \
     "vs $up/main: $(git rev-list --left-right --count "$up/main...HEAD" | awk '{print $1 " behind, " $2 " ahead"}')"
git status --short
pmset -g batt | head -1
sysctl -n hw.memsize | awk '{print $1/2^30 " GB"}'
osascript -e 'quit app "Ollama"' 2>/dev/null   # the desktop app competes for memory
sleep 2; pkill -f '/Applications/Ollama.app/'  # quitting the app leaves its `ollama serve` on :11434
pgrep -fl 'bench_metal_matrix|scr bench|scr-|mlx_lm.(server|generate)|llama-server|vllm serve|omlx|mistralrs serve|caffeinate -dims'   # leftovers from an earlier run
lsof -nP -iTCP:8751 -sTCP:LISTEN               # the script's --port
```

- **Commit:** tell the user where `HEAD` is relative to upstream `main`
  (e.g. "on `skills`, 3 behind and 1 ahead"), and let them choose: benchmark
  this commit, or check out upstream `main` first. Stay on their branch
  unless they choose to switch. Only upstream `main` goes on the site
  without asking.
- **Dirty tree:** ask the user how to handle their changes. Leave the
  changes for them to commit or stash themselves.
- **On battery:** ask the user to plug in, then check again. Battery runs
  are throttled, and the JSON records the power source.
- **Leftovers:** if `pgrep` or `lsof` prints anything, an earlier run is
  still going or didn't clean up. Ask before stopping it (see **Stopping a
  run** under step 3).
- **Memory:** the script skips a model that can't fit and records why
  (`skipped: needs 64 GB, this Mac has 36 GB`), so the default round runs as
  is. glm-4.5-air (46.8 GB) runs only on a 64 GB Mac, at no more than 4 users
  at once. No model in the current rounds fits a 16 or 24 GB Mac: don't run
  there.
- **64 GB Macs:** GLM's longest cells need about 53 GB, above the GPU's
  default limit (about 48 GB). Ask the user to raise it to the agreed value
  with `sudo sysctl iogpu.wired_limit_mb=<MB>` (it resets on reboot); the
  run records it as `machine.gpu_wired_limit_mb`. That raise covers scratchy
  only — oMLX's own guard doesn't care about it (next bullet).
- **oMLX's memory guard:** oMLX loads a model only when the Mac can free the
  model's size plus ~8 GB of reserve and overhead, out of what is reclaimable
  right now (its dynamic ceiling — nothing the runner sets moves it). So a
  model within ~20 GB of the Mac's RAM — glm-4.5-air on 64 GB this round, a
  nemotron-super or gpt-oss-120b on whatever Mac a later round sizes for it —
  needs the Mac near idle as well: ask the user to quit the other apps and
  sessions, or oMLX declines the model at load and the run records it as
  declined instead of timing it. A model within ~8 GB of the Mac's RAM never
  passes the guard at all: expect it declined on every run, however idle.
- **First run on this Mac:** it downloads the HF weights and ollama's models,
  so it needs internet.

**Done when:** the user has chosen the commit to benchmark,
`git status --short` prints nothing, `pmset` says `'AC Power'`, the
`pkill`/`pgrep`/`lsof` lines leave nothing running, and you've decided on
the `--models` flag (none for this Mac's set, or `--models full`).

## 3. Run

Full run. Pin every engine the run uses and scratchy's commit to the versions
agreed for this round, so all the Macs run the same thing; the script refuses
to start while any pin is missing or doesn't match, and lists what's wrong.
Names match in any case, scratchy's pin is a commit prefix, and llama.cpp's
is its build number:

```bash
source ~/.venvs/mlx/bin/activate
caffeinate -dims ./scripts/bench_metal_matrix.sh --fail-fast [--models full] \
  --pin scratchy=<sha> --pin oMLX=<v> 2>&1 | tee ~/metal-matrix-run.log
echo "exit ${PIPESTATUS[0]}"
```

With `--engines all`, pin the others too: `--pin mlx-lm=<v> --pin ollama=<v>
--pin llama.cpp=b<build> --pin vllm-metal=<v> --pin mistral.rs=<v>`. oMLX
TurboQuant shares oMLX's pin.

`--unpinned` runs without pins and records that in the JSON (`pinned:
false`); use it only for a smoke run or a test the user asked for.

Smoke run (a few minutes when the builds are cached):

```bash
source ~/.venvs/mlx/bin/activate
caffeinate -dims ./scripts/bench_metal_matrix.sh --fail-fast --unpinned --models gpt-oss-20b \
  --reps 1 --prime 1 --no-scaling --parity --out-dir /tmp/metal-smoke 2>&1 | tee /tmp/metal-smoke.log
echo "exit ${PIPESTATUS[0]}"
```

The script exits 1 when a model got no scratchy numbers. `tee` would hide
that, which is why `PIPESTATUS[0]` is printed. `--fail-fast` stops at the
first scratchy build or parity-gate failure, instead of going on to the next
model.

Before timing each model, the script runs a blocking **parity gate**: `scr
chat` and `mlx_lm.generate`, both greedy, must give the same answer to a few
short prompts that have one right answer. Where the text differs,
`scripts/parity_ties.py` runs mlx-lm over scratchy's own answer and judges
each token scratchy picked: one within a logit of mlx-lm's top is a near-tie
(reasoning models' thoughts open on them), and the prompt passes as `TIE`.
A token further behind, or more than a few near-ties, fails. A model that fails the
gate isn't timed on any engine, because there would be no scratchy numbers
to compare against. The gate needs mlx-lm, and runs wherever a comparison
engine is timed — every full run — or wherever `--parity` forces it (the
smoke run passes it). A scratchy-only quick run skips it — the mlx-lm side
reloads the whole checkpoint once per gate prompt, which on the large models
can outlast the run — and records `parity_mlx_lm: null` with `parity_skipped`
saying why. Rules and rationale are in `docs/BENCHMARKING.md` (fairness
rule 3).

Run the command in the background, so a full run doesn't hit a shell timeout,
and wait for it to finish. While it runs, watch the log for these lines and
act on each one as it appears:

- the engine lines in the header, one per engine (`llama.cpp : <path>
  (<version>)`): a `skipped` line under one means it isn't installed or
  isn't selected (by default only oMLX and oMLX TurboQuant run). If it's an
  engine the user meant to run, stop the run and go back to step 1.
- `skipped: needs N GB`: expected for glm-4.5-air on a Mac under 64 GB.
- `--- mistral.rs: nothing to serve`: expected for every MoE model; it runs
  only the dense ones, since it runs out of memory on MoE GGUFs on Metal.
- `N FAILED` on a cell, then `requests failed`, or `timed out after 600s`:
  that engine ran out of memory (or errored, or was too slow) at that size.
  The server is restarted and the next cell runs (`(server restarted)`); the
  page marks the cell with ‡. Note which engine and cell for the user; it's
  a finding, not a broken run. `server would not restart` ends that sweep.
- `<engine> declined <model> — its own memory guard refused it: ...`: oMLX's
  admission guard refused to load the model outright (its dynamic ceiling
  from system-reclaimable memory — not the Metal cap, so the wired-limit
  raise doesn't fix it). Expected for any model within ~20 GB of the Mac's
  RAM unless it is near idle, and for any model within ~8 GB of the Mac's
  RAM on every run — not a glm-4.5-air quirk; the next nemotron-super or
  gpt-oss-120b round hits the same band. Also a finding, not a broken run:
  the engine's cells are skipped, the JSON records `engine_declined` with
  the engine's message, and the page shows "Declined before serving" on the
  model's section.
- `BUILD FAILED`: the run stops by itself with `--fail-fast`. Go to **On a
  build failure** below.
- `--- parity gate vs mlx-lm (blocking)`, followed by either `PARITY OK` or
  `PARITY FAILED — not timing <stem>` (full and `--parity` runs; a quick run
  prints a `skipped` line instead). A `TIE` line, with its `token N:`
  lines under it, is a pass: note the gaps for the user, nothing to do. On a failure, tell the user right
  away: that model has no numbers, and with `--fail-fast` the run stops
  there (without it, it moves on and exits 1 at the end). The lines under it
  show both answers; `the ... child crashed` means one side crashed rather
  than disagreed. Then go to **On a parity failure** below.
- `json -> <path>`: the run has finished writing its JSON.

**On a build failure:** check whether it's a regression on `main` or a
problem on this Mac. Rebuild the same features at the commit of the newest
run on the site. The features are on the log's `--- build -F <features>`
line.

```bash
sha=$(ls site/data/metal/*.json | xargs -n1 python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["generated_utc"], d["repo"]["sha"])' | sort | tail -1 | cut -d' ' -f2)
git worktree add /tmp/scratchy-known-good "$sha"
(cd /tmp/scratchy-known-good && cargo build --release -p scratchy-cli --features <features>)
git worktree remove /tmp/scratchy-known-good
```

The worktree starts from a cold cache, so this takes minutes. If it builds,
the failure is a regression on `main` since `$sha`. Tell the user, and point
them to the build log, the failing commit and the known-good one so they can
file an issue. If it fails too, the problem is with this Mac's toolchain.

**On a parity failure:** don't retry, loosen the gate or edit the script's
commands to get it through. A mismatch on these prompts means scratchy and
mlx-lm computed different things, and timing either of them would compare
different computations. The gate's full output is in
`<out-dir>/<machine>/parity-<stem>.log`; the path is on the `PARITY FAILED`
line. For each prompt it shows `OK`, `TIE` or `FAIL`, then `a:` (scratchy)
and `b:` (mlx-lm) for every `FAIL`, with the reason in brackets: a token some
logits behind mlx-lm's top, too many near-ties, or no score at all. Report
the `FAIL` lines to the user, and say which of these the outputs look like:

- **The two sides were given different prompts.** One side starts with a
  thought or reasoning block, or with template text the other doesn't have,
  and runs out of tokens before it answers. `scr chat` renders the
  checkpoint's chat template with its declared defaults, while
  `mlx_lm.generate` forces `enable_thinking` on for any vocab with think
  tokens. That was #314 (gemma-4). The fix is a new entry in
  `mlx_parity_config` in `scripts/bench_metal_matrix.sh`, not a change to
  scratchy.
- **A real disagreement.** Both sides answer, and the answers differ, or
  scratchy's output is garbage. That points at the load path: wrong quant
  preset, wrong group size, or a broken dequant. Check whether it's a
  regression on `main` the same way as **On a build failure**: build the
  same features at the known-good commit. Then rerun the gate on its own
  with that binary. The command is the `bench startup ... --parity-cmd`
  call in `scripts/bench_metal_matrix.sh`, run with `--child-cmd` pointing
  at the known-good build's `scr`.

Either way, the user decides what to file. Point them to the parity log and
the commit.

**Stopping a run** (only when the user asks, or after a failed check):
stopping just the script isn't enough. It handles signals only between
steps, and the `scr bench` and server it started keep running. Stop all of
them:

```bash
pkill -TERM -f 'bench_metal_matrix.sh|scr bench (startup|serve)|mlx_lm.(server|generate)|llama-server|vllm serve|omlx|mistralrs serve|caffeinate -dims'
pkill -TERM -f ' chat -m .* --device metal'    # the parity gate's scratchy side
sleep 10
pgrep -fl 'bench_metal_matrix|scr bench|scr-|mlx_lm.(server|generate)|llama-server|vllm serve|omlx|mistralrs serve|caffeinate -dims'
lsof -nP -iTCP:8751 -sTCP:LISTEN -t | xargs kill 2>/dev/null   # whatever server the run started
```

**Done when:** the header shows a real path for every engine the user meant
to run, every model printed `PARITY OK`, the log ends with a
`json -> <path>` line, and `exit 0` is printed. A parity failure exits 1,
with `(parity gate failed)` on the final `FAILED:` line.

## 4. Check the JSON

Take the path from the run's own `json -> ...` line, not whichever file is
newest. For a smoke run, read `/tmp/metal-smoke.log` instead:

```bash
f=$(grep -o 'json -> .*' ~/metal-matrix-run.log | tail -1 | cut -d' ' -f3)
up=$(git remote -v | awk '/AI-native-Systems-Research\/scratchy(\.git)? \(fetch\)/ {print $1; exit}')
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"]["sha"])' "$f")
if [ "$sha" = "$(git rev-parse "$up/main")" ]; then echo "commit  upstream main (tip)"
elif git merge-base --is-ancestor "$sha" "$up/main"; then echo "commit  upstream main, $(git rev-list --count "$sha..$up/main") commits behind its tip"
else echo "commit  NOT in upstream main"; fi
python3 - "$f" <<'EOF'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1])); c = d["config"]
print("chip   ", d["machine"]["chip"], "|", d["machine"]["power"])
print("repo   ", d["repo"]["sha"][:8], d["repo"]["branch"], "dirty=" + str(d["repo"]["dirty"]))
for e in c.get("engines") or []:
    print("engine ", f"{e['label']:16}", "installed=" + str(e["installed"]),
          e.get("version") or "-", "pin=" + (e.get("pin") or "-"))
print("pinned ", c.get("pinned"), c.get("pin_problems") or "")
for m in d["models"]:
    reps = Counter(r["scenario"] for r in m.get("cache_ladder") or [])
    print("model  ", m["stem"], "built=" + str(m.get("built")),
          "parity=" + str(m.get("parity_mlx_lm")), dict(reps))
for x in d.get("skipped_models") or []:
    print("skipped", x["stem"], x["reason"])
ladders = [Counter(r["scenario"] for r in m.get("cache_ladder") or []) for m in d["models"]]
plain = c["scaling"] is not None and not c["scratchy_serve_args"]
full = (plain and c["scenarios"] == ["frozen", "cold", "warm"] and c["cold_priming_launches"] == 3
        and all(l.get("frozen") == 3 for l in ladders))
quick = (plain and c["scenarios"] == ["cold", "warm"] and c["cold_priming_launches"] == 1
         and all(l.get("cold") == 1 for l in ladders))
print("scope  ", "full" if full else "quick" if quick else
      "PARTIAL (rungs, reps, priming, scaling or serve args not at a quick or full run's)")
EOF
```

**Done when:** you've reported every line to the user. These all have to
hold for a run to go on the site: `AC Power`, `dirty=False`, `installed=True`
for every engine the user meant to run, `pinned True` (each version equal to
its pin), and every model `built=True` and `parity=True`. `parity=False`
means the model failed the gate and has no numbers. `parity=None` on a built
model means the gate didn't run, so its numbers were never checked —
expected on a quick run, which skips the gate for scratchy alone
(`parity_skipped` says so). A
`skipped` model is fine when its reason is memory. If any of these fails,
report it and let the user decide whether the run is usable. Report the
`commit` and `scope` lines too, but they don't fail the run on their own;
step 5 asks about them.
A smoke run ends here.

## 5. Copy into the site

Before copying, ask the user to confirm if any of these is true:
- `scope` is `PARTIAL` (`quick` and `full` are both fine)
- the run has fewer models than the round it ran (this Mac's four, or all
  eight with `--models full`), not counting a model skipped for memory
- `commit` isn't upstream `main`
- any model's `parity` is `False` (it failed the gate), or `None` with no
  `parity_skipped` reason (the gate should have run but didn't, as on a full
  run without mlx-lm). `None` with `parity_skipped` is a quick run, as
  expected.

Never copy a smoke run, or anything under `/tmp/metal-smoke`.

The site shows each chip's two newest runs, as current and previous, and
computes the ▲/▼ change badges from them. Add the new run, keep the run
before it as the previous one, and remove anything older. Copy first, and
prune only once the copy has worked: `cp -n` refuses to overwrite a run that
has the same name, and in that case nothing should be removed.

```bash
m=$(basename "$f" | sed -E 's/-[0-9]{4}-[0-9]{2}-[0-9]{2}(T[0-9]{6}Z)?-[0-9a-f]{8}\.json$//')   # e.g. apple-m3-pro
if cp -n "$f" site/data/metal/; then
  ls site/data/metal/"$m"-*.json | sort | sed '$d' | sed '$d' | xargs -r git rm -q   # keep the newest two
else
  echo "$(basename "$f") is already in site/data/metal; nothing copied or removed"
fi
```

If it says the file is already there, show the user both files and let them
decide which one to keep.

**Done when:** `ls site/data/metal/$m-*` lists at most two files, and the
newest one is `$(basename "$f")`.

## 6. Preview

The site's dev server keeps serving (and re-rendering on every change) until
it's stopped, so run it in the background:

```bash
site/open.sh --port 8001    # then open http://localhost:8001/metal.html
```

If the build fails, it lists every problem in the data files. CI would fail
on the same ones, so fix them here.

**Done when:** the build succeeds, and `metal.html` shows this chip with the
new run's date and commit.

## 7. Share or commit

**A full run** goes in its own PR. Commit it on a new branch and open a PR
titled:

```
data(site): <machine> Metal run at <short-sha>
```

**A quick run** goes to the metal quick round: one open issue, labelled
`metal-round`, whose comments collect a quick run from each of the four Macs
(base M5, M3 Pro, M4 Max, M5 Max). Once all four are in, a workflow
(`.github/workflows/metal-round.yml`) opens a PR adding the newest run per Mac
to `site/data/metal` and closes the issue; an admin merges the PR. Post only a
run that exited 0; a run that went wrong is not posted. A second run of the
same Mac before the round closes is posted too: the newest run, by when it
ran, is the one committed. Only runs from the repo's owners, members and
collaborators count.

First check it can go in a round: a quick run, scratchy alone at its
defaults, one of the four Macs, a clean tree, and a commit on upstream `main`:

```bash
git fetch -q "$up" main
python3 scripts/metal_round.py check "$f" --main-ref "$up/main"
```

If it says `not for a round`, tell the user why and stop: nothing is
posted. Otherwise ask the user whether to post it, and post only on a yes.
Then find the round's open issue, or open one:

```bash
repo=AI-native-Systems-Research/scratchy
n=$(gh issue list -R "$repo" --label metal-round --state open --json number --jq '.[0].number')
if [ -z "$n" ]; then
  n=$(gh issue create -R "$repo" --label metal-round \
    --title "Metal quick round $(date -u +%F)" \
    --body "Collects one scratchy quick run from each Mac (base M5, M3 Pro, M4 Max, M5 Max) as comments, posted by the metal-bench skill. Once all four are in, the Metal round workflow opens a PR adding the newest run per Mac to site/data/metal, and closes this issue." \
    | grep -o '[0-9]*$')
fi
```

Two people opening a round at once makes two issues: post to the lower
number, and close the other.

Post the run as a comment: one line, then the JSON, compact (no indents: a
64 GB Mac's run with every cell is about 64 KB indented, at GitHub's comment
limit), in a collapsed block:

```bash
{
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["machine"]["chip"], "quick run,", d["generated_utc"] + ", at", d["repo"]["sha"][:8])' "$f"
  printf '\n<details><summary>run JSON</summary>\n\n```json\n'
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' "$f"
  printf '```\n</details>\n'
} > /tmp/metal-round-comment.md
wc -c < /tmp/metal-round-comment.md
[ "$(wc -c < /tmp/metal-round-comment.md)" -le 65536 ] && gh issue comment "$n" -R "$repo" --body-file /tmp/metal-round-comment.md
```

If it's over 65,536 bytes, GitHub won't take it: tell the user, and don't
split it. Within a minute the round's status comment on the issue lists this Mac
as in, or a reply says why the run wasn't counted.

**Done when:** the user has the PR link (a full run) or the comment's link
and the round's status (a quick run).
