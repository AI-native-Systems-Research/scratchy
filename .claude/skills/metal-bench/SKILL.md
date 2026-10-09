---
name: metal-bench
description: Metal benchmark matrix runs on a Mac. Use to run scripts/bench_metal_matrix.sh against mlx-lm and ollama on this Mac (a full run, or a quick smoke run to check the setup), or to add a finished run's JSON to the site's Metal page (site/data/metal).
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

**Two kinds of run:**
- A **full run** is what goes on the site: the default models (or the
  small-RAM set from step 2), every rung, and default reps, priming and
  scaling. It takes hours.
- A **smoke run** checks the setup in a few minutes. It runs granite only,
  with one rep, no scaling and output in `/tmp`. Its JSON never goes on the
  site, so after step 4 you're done.

## 1. One-time setup (per Mac)

```bash
# mlx-lm in its own venv; the script finds it via $VIRTUAL_ENV
~/.venvs/mlx/bin/python -c 'import mlx_lm' 2>/dev/null \
  || { python3 -m venv ~/.venvs/mlx && ~/.venvs/mlx/bin/pip install -U mlx-lm; }
command -v ollama || brew install ollama   # the CLI, not the desktop app
```

**Passwordless `purge`.** The frozen rung runs `sudo -n purge` throughout a
run that lasts hours. A cached `sudo -v`, even with a keepalive loop, expires
partway through. A one-time sudoers rule is the reliable fix. Have the user
run this in a separate terminal. It checks the rule with `visudo -c` before
installing it, because a broken file in `/etc/sudoers.d` breaks `sudo`
entirely:

```bash
echo "$(whoami) ALL=(root) NOPASSWD: /usr/sbin/purge, /usr/bin/true" > /tmp/bench-purge
sudo visudo -cf /tmp/bench-purge && sudo install -m 0440 -o root -g wheel /tmp/bench-purge /etc/sudoers.d/bench-purge
```

Then check the rule. `sudo -k` drops any cached unlock first, so a pass
proves the rule itself works:

```bash
sudo -k && sudo -n true && echo "rule works"
```

**Without the rule** (the user would rather not add it, or can't), pass
`--scenarios cold,warm` in step 3. Nothing then needs sudo, but the run has
no frozen rung, so treat it like any partial run in step 5.

**Done when:** `~/.venvs/mlx/bin/python -c 'import mlx_lm'` exits 0,
`command -v ollama` prints a path, and the check prints `rule works` without
asking for a password, or the user has chosen to run without the frozen
rung.

## 2. Pre-flight (every run)

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
pgrep -fl 'bench_metal_matrix|scr bench|mlx_lm.server|caffeinate -dims'   # leftovers from an earlier run
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
- **16 or 24 GB of RAM:** pass `--models granite-3.3-2b-instruct,qwen2.5-7b`.
  gemma-4-26b-a4b-it has 15.4 GB of weights and doesn't fit next to the
  other engines.
- **36 GB of RAM:** gemma runs, but macOS pages its weights out, so its RSS
  reads low. Point that out when you report the results.
- **First run on this Mac:** it downloads the HF weights and ollama's models,
  so it needs internet.

**Done when:** the user has chosen the commit to benchmark,
`git status --short` prints nothing, `pmset` says `'AC Power'`, the
`pkill`/`pgrep`/`lsof` lines leave nothing running, and you've decided on
the `--models` flag.

## 3. Run

Full run:

```bash
source ~/.venvs/mlx/bin/activate
caffeinate -dims ./scripts/bench_metal_matrix.sh --fail-fast [--models ...] 2>&1 | tee ~/metal-matrix-run.log
echo "exit ${PIPESTATUS[0]}"
```

Smoke run (a few minutes when the builds are cached):

```bash
source ~/.venvs/mlx/bin/activate
caffeinate -dims ./scripts/bench_metal_matrix.sh --fail-fast --models granite-3.3-2b-instruct \
  --reps 1 --prime 1 --no-scaling --out-dir /tmp/metal-smoke 2>&1 | tee /tmp/metal-smoke.log
echo "exit ${PIPESTATUS[0]}"
```

The script exits 1 when a model got no scratchy numbers. `tee` would hide
that, which is why `PIPESTATUS[0]` is printed. `--fail-fast` stops at the
first scratchy build failure, instead of spending hours measuring only the
other engines.

Run the command in the background, so a full run doesn't hit a shell timeout,
and wait for it to finish. While it runs, watch the log for these lines and
act on each one as it appears:

- the `mlx-lm  :` and `ollama  :` header lines: if either says `skipped`,
  stop the run and go back to step 1.
- `BUILD FAILED`: the run stops by itself with `--fail-fast`. Go to **On a
  build failure** below.
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

**Stopping a run** (only when the user asks, or after a failed check):
stopping just the script isn't enough. It handles signals only between
steps, and the `scr bench` and server it started keep running. Stop all of
them:

```bash
pkill -TERM -f 'bench_metal_matrix.sh|scr bench (startup|serve)|mlx_lm.server|caffeinate -dims'
sleep 10
pgrep -fl 'bench_metal_matrix|scr bench|mlx_lm.server|caffeinate -dims'
lsof -nP -iTCP:8751 -sTCP:LISTEN -t | xargs kill 2>/dev/null   # the ollama serve the run started
```

**Done when:** the header shows real paths for both `mlx-lm  :` and
`ollama  :`, the log ends with a `json -> <path>` line, and `exit 0` is
printed.

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
print("engines", c["comparison"])
for m in d["models"]:
    reps = Counter(r["scenario"] for r in m.get("cache_ladder") or [])
    print("model  ", m["stem"], "built=" + str(m.get("built")), dict(reps))
full = (c["scenarios"] == ["frozen", "cold", "warm"] and c["cold_priming_launches"] == 3
        and c["scaling"] is not None and not c["scratchy_serve_args"]
        and all(Counter(r["scenario"] for r in m.get("cache_ladder") or []).get("frozen") == 3
                for m in d["models"]))
print("scope  ", "full" if full else "PARTIAL (rungs, reps, priming, scaling or serve args not at defaults)")
EOF
```

**Done when:** you've reported every line to the user. These all have to
hold for a run to go on the site: `AC Power`, `dirty=False`, both engines
`True`, and every model `built=True`. If any of them fails, report it and let
the user decide whether the run is usable. Report the `commit` and `scope`
lines too, but they don't fail the run on their own; step 5 asks about them.
A smoke run ends here.

## 5. Copy into the site

Before copying, ask the user to confirm if any of these is true:
- `scope` is `PARTIAL`
- the run has fewer models than the default three (or fewer than the
  small-RAM pair on a 16/24 GB Mac)
- `commit` isn't upstream `main`

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

`site/open.sh` keeps serving until it's stopped, so run it in the background:

```bash
PORT=8001 site/open.sh    # then open http://localhost:8001/metal.html
```

If the build fails, it lists every problem in the data files. CI would fail
on the same ones, so fix them here.

**Done when:** the build succeeds, and `metal.html` shows this chip with the
new run's date and commit.

## 7. Share or commit

Either send the JSON to whoever collects the runs from all the Macs, or
commit it on a new branch and open a PR titled:

```
data(site): <machine> Metal run at <short-sha>
```

**Done when:** the user has the JSON or the PR link.
