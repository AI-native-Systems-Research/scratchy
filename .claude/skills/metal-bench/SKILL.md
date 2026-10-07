---
name: metal-bench
description: Metal benchmark matrix runs on a Mac. Use to run scripts/bench_metal_matrix.sh against mlx-lm and ollama on this Mac, or to add a finished run's JSON to the site's Metal page (site/data/metal).
---

# Metal benchmark run (one Mac → one JSON → the site)

Rung definitions and fairness rules are in `docs/BENCHMARKING.md`. The
script's flags are documented in its header (`--help`).

The run records its mistakes silently, as `repo.dirty`, `machine.power` or a
skipped engine, and every one of them has landed on the site at least once.
Each step below ends with a **Done when** check. Meet it before you start the
next step.

**Who runs what:** anything with `sudo` needs a password, so ask the user to
run those lines with the `!` prefix (e.g. `! sudo -v`). Claude runs
everything else.

## 1. One-time setup (per Mac)

```bash
# mlx-lm in its own venv; the script finds it via $VIRTUAL_ENV
~/.venvs/mlx/bin/python -c 'import mlx_lm' 2>/dev/null \
  || { python3 -m venv ~/.venvs/mlx && ~/.venvs/mlx/bin/pip install -U mlx-lm; }
command -v ollama || brew install ollama   # the CLI, not the desktop app
```

**Passwordless `purge`.** The frozen rung runs `sudo -n purge` throughout a
run that lasts hours. A cached `sudo -v`, even with a keepalive loop, expires
partway through. A one-time sudoers rule is the reliable fix, so have the
user add it:

```bash
! sudo visudo -f /etc/sudoers.d/bench-purge
#   add, with their username from `whoami`:
#   <user> ALL=(root) NOPASSWD: /usr/sbin/purge, /usr/bin/true
```

Then check it. `sudo -k` first drops any cached unlock, so a pass proves the
rule itself works:

```bash
sudo -k && sudo -n true && echo "rule works"
```

**Done when:** `~/.venvs/mlx/bin/python -c 'import mlx_lm'` exits 0,
`command -v ollama` prints a path, and the check prints `rule works` without
asking for a password.

## 2. Pre-flight (every run)

```bash
git checkout main && git pull --ff-only
git status --short
pmset -g batt | head -1
sysctl -n hw.memsize | awk '{print $1/2^30 " GB"}'
osascript -e 'quit app "Ollama"' 2>/dev/null   # the desktop app competes for memory
sleep 2; pkill -f '/Applications/Ollama.app/' # quitting the app leaves its `ollama serve` on :11434
```

- **Dirty tree:** ask the user how to handle their changes. Leave the
  changes for them to commit or stash themselves.
- **On battery:** ask the user to plug in, then check again. Battery runs
  are throttled, and the JSON records the power source.
- **16 or 24 GB of RAM:** pass `--models granite-3.3-2b-instruct,qwen2.5-7b`.
  gemma-4-26b-a4b-it has 15.4 GB of weights and doesn't fit next to the
  other engines.
- **36 GB of RAM:** gemma runs, but macOS pages its weights out, so its RSS
  reads low. Point that out when you report the results.
- **First run on this Mac:** it downloads the HF weights and ollama's models,
  so it needs internet.

**Done when:** `git status --short` prints nothing, `pmset` says
`'AC Power'`, `pgrep -f /Applications/Ollama.app/` prints nothing, and
you've decided on the `--models` flag.

## 3. Run

The run takes hours. Start it with Bash `run_in_background` and wait for the
completion notification:

```bash
source ~/.venvs/mlx/bin/activate
caffeinate -dims ./scripts/bench_metal_matrix.sh [--models ...] 2>&1 | tee ~/metal-matrix-run.log
```

Once the header has printed, read the top of `~/metal-matrix-run.log`. If
`mlx-lm  :` or `ollama  :` says `skipped`, stop the run (TaskStop, or Ctrl-C
in the user's terminal) and go back to step 1.

The script exits 0 even when a scratchy build fails. It only logs
`BUILD FAILED` and carries on with the other engines for hours. Watch for it
with a Monitor on the log:

```bash
tail -n +1 -f ~/metal-matrix-run.log | grep --line-buffered -E 'BUILD FAILED|^FAILED:|json -> '
```

On `BUILD FAILED`, stop the run and show the user the build log path printed
on that line. That run's numbers aren't usable for the site.

**Done when:** the header shows real paths for both `mlx-lm  :` and
`ollama  :`, the log has no `BUILD FAILED` or `FAILED:` line, and the run has
finished with a `json -> <path>` line at the end of the log.

## 4. Sanity-check the JSON

Take the file path from the run's own `json -> ...` line, not whichever file
is newest:

```bash
f=$(grep -o 'json -> .*' ~/metal-matrix-run.log | tail -1 | cut -d' ' -f3)
python3 - "$f" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("chip   ", d["machine"]["chip"], "|", d["machine"]["power"])
print("repo   ", d["repo"]["sha"][:8], d["repo"]["branch"], "dirty=" + str(d["repo"]["dirty"]))
print("engines", d["config"]["comparison"])
for m in d["models"]:
    print("model  ", m["stem"], "built=" + str(m.get("built")))
EOF
```

**Done when:** the output shows `AC Power`, `main`, `dirty=False`, both
engines `True`, and every model `built=True`. If any of these fails, report
it to the user and let them decide whether the run is usable.

## 5. Copy into the site

The site shows each chip's two newest runs, as current and previous, and
computes the ▲/▼ change badges from them. Keep this machine's newest
existing run as the previous one, remove anything older, and add the new
run:

```bash
m=$(basename "$f" | sed -E 's/-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{8}\.json$//')   # e.g. apple-m3-pro
ls site/data/metal/"$m"-*.json 2>/dev/null | sort | sed '$d' | xargs -r git rm -q
cp "$f" site/data/metal/
```

**Done when:** `ls site/data/metal/$m-*` lists at most two files, and the
newest one is `$(basename "$f")`.

## 6. Preview

`site/open.sh` keeps serving until it's stopped, so start it with
`run_in_background`:

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
