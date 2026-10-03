#!/usr/bin/env python3
"""Generate site/_site/metal.html from site/data/metal/*.json.

Each data file is one run of scripts/bench_metal_matrix.sh, copied in
unedited and named <machine>-<YYYY-MM-DD>-<sha8>.json. The page reads them at
site-build time, so it cannot drift from what the runner measured, and a file
missing what the page needs fails the build rather than rendering a blank.

Per machine, each model shows its newest run; older runs of the same model
stay visible in that model's history table, so a re-run after a Metal change
reads as a before/after.

Standalone: site/build_metal.py [out.html] [data_dir]
"""
import html
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data" / "metal"
REPO = "https://github.com/AI-native-Systems-Research/scratchy"

# (key, label, ladder field, scaling field, CSS series class). Fixed order, so
# an engine keeps its colour whichever engines a run happens to include.
ENGINES = [
    ("scratchy", "scratchy", "cache_ladder", "scaling", "s1"),
    ("mlx-lm", "mlx-lm", "cache_ladder_mlx_lm", "scaling_mlx_lm", "s2"),
    ("ollama", "ollama", "cache_ladder_ollama", "scaling_ollama", "s3"),
]

REQUIRED = {
    "": ("schema", "generated_utc", "generator", "machine", "repo", "config", "models"),
    "machine": ("chip", "cores_total", "memory_gb", "macos"),
    "repo": ("sha",),
    "config": ("scenarios",),
}
MODEL_REQUIRED = ("stem", "model_id", "built")


def esc(s):
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------- load

def load(data_dir):
    """Every run file, validated. Exits naming each problem, so one build
    reports all of them rather than the first."""
    runs, errors = [], []
    for f in sorted(Path(data_dir).glob("*.json")):
        try:
            run = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            errors.append(f"{f.name}: not valid JSON ({e})")
            continue
        for block, keys in REQUIRED.items():
            obj = run if not block else run.get(block) or {}
            for k in keys:
                if obj.get(k) in (None, ""):
                    errors.append(f"{f.name}: missing {block + '.' if block else ''}{k}")
        if run.get("schema") not in (None, 2):
            errors.append(f"{f.name}: schema {run.get('schema')!r}, this page reads schema 2")
        for i, m in enumerate(run.get("models") or []):
            for k in MODEL_REQUIRED:
                if k not in m:
                    errors.append(f"{f.name}: models[{i}] missing {k}")
        want = f"{slug(run.get('machine', {}).get('chip', ''))}-{run.get('generated_utc', '')[:10]}-{str(run.get('repo', {}).get('sha', ''))[:8]}.json"
        if f.name != want:
            errors.append(f"{f.name}: name it {want} (<machine>-<date>-<sha8>.json)")
        run["_file"] = f
        runs.append(run)
    if errors:
        for e in errors:
            print(f"metal page: {e}", file=sys.stderr)
        sys.exit(1)
    return runs


def slug(chip):
    return re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-")


# ------------------------------------------------------------------ numbers

def med(reps, scenario, field):
    vals = [r[field] for r in reps or [] if r.get("scenario") == scenario and r.get(field) is not None]
    return statistics.median(vals) if vals else None


def cell(cells, axis, rung):
    for c in cells or []:
        if c.get("axis") == axis and c.get("rung") == rung:
            return c
    return None


def lfmt(ladder, scenario, field, nd=0, scale=1):
    """A startup median, or why there is none: launches that reached ready but
    never streamed a token say so instead of showing a dash."""
    v = med(ladder, scenario, field)
    if v is None and field != "t_ready_s" and med(ladder, scenario, "t_ready_s") is not None:
        return "no first token"
    return fmt(v, nd, scale)


def served(m):
    """Whether scratchy produced any number for this model."""
    return m.get("built") and (m.get("cache_ladder") or m.get("warm_serving") or m.get("scaling"))


def fmt(v, nd=0, scale=1):
    return "—" if v is None else f"{v * scale:,.{nd}f}"


TIMINGS = ("median_ttft_ms", "median_tpot_ms")
NO_STREAM = "no stream"


def streamed(c):
    """Whether a serving cell's answer actually streamed. A zero median TPOT
    means the tokens arrived in one piece (or there was only one), so neither
    timing was measured: its TTFT is the whole answer's time, not the first
    token's. ollama's gemma4 does this."""
    return bool(c) and (c.get("median_tpot_ms") or 0) > 0


def timing(c, metric):
    """A cell's value, or None where it was not really measured."""
    if not c or (metric in TIMINGS and not streamed(c)):
        return None
    return c.get(metric)


def cfmt(c, metric, nd=0):
    """A cell's value as text: says "no stream" rather than show a timing the
    engine never produced."""
    if c and metric in TIMINGS and not streamed(c):
        return NO_STREAM
    return fmt(timing(c, metric), nd)


def nice_max(v):
    """Smallest 1/2/5 x 10^k at or above v, so ticks land on round numbers."""
    if v <= 0:
        return 1
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if v <= m * e:
            return m * e
    return 10 * e


# ------------------------------------------------------------------- render

def tip(rows):
    """A tooltip payload: [[value, label, series-class], ...]. Rendered by the
    page's script with textContent, never innerHTML."""
    return esc(json.dumps(rows, separators=(",", ":")))


def summary_table(m, run):
    base = (run["config"].get("scaling") or {}).get("base") or {}
    shape = f"{base.get('input', '?')} in / {base.get('output', '?')} out" if base else ""
    rows = []
    for key, label, lk, sk, cls in ENGINES:
        ladder, one = m.get(lk), cell(m.get(sk), "conc", 1)
        if key == "scratchy" and not served(m):
            why = "build failed" if not m.get("built") else "the server never served"
            rows.append(f'<tr><th scope="row"><span class="key {cls}"></span>{label}</th>'
                        f'<td colspan="8" class="gap">No numbers: {why}. See the run\'s raw logs.</td></tr>')
            continue
        if key == "mlx-lm" and not (ladder or m.get(sk)):
            continue
        if key == "ollama" and not (ladder or m.get(sk)):
            continue
        name = label
        if key == "ollama" and m.get("ollama"):
            o = m["ollama"]
            name += f' <span class="dim">{esc(o.get("tag", ""))}{", " + esc(o["quantization"]) if o.get("quantization") else ""}</span>'
        vals = [
            lfmt(ladder, "frozen", "t_ready_s", 2),
            lfmt(ladder, "cold", "t_ready_s", 2),
            lfmt(ladder, "cold", "ttft_from_send_s", 2),
            lfmt(ladder, "warm", "ttft_from_send_s", 0, 1000),
            cfmt(one, "median_ttft_ms"),
            cfmt(one, "median_tpot_ms", 1),
            fmt(one and one.get("output_throughput"), 1),
            fmt(med(ladder, "cold", "peak_rss_mib")),
        ]
        rows.append(f'<tr><th scope="row"><span class="key {cls}"></span>{name}</th>'
                    + "".join(f"<td>{v}</td>" for v in vals) + "</tr>")
    return f"""<div class="mtable"><table>
  <thead>
    <tr><th rowspan="2" scope="col" class="first">engine</th>
        <th colspan="4" scope="colgroup">startup</th>
        <th colspan="3" scope="colgroup">one user{', ' + shape if shape else ''}</th>
        <th rowspan="2" scope="col">peak RSS<br><span class="dim">MiB</span></th></tr>
    <tr><th scope="col">frozen<br><span class="dim">s</span></th><th scope="col">cold<br><span class="dim">s</span></th>
        <th scope="col">1st req<br><span class="dim">s</span></th><th scope="col">warm<br><span class="dim">ms</span></th>
        <th scope="col">TTFT<br><span class="dim">ms</span></th><th scope="col">TPOT<br><span class="dim">ms</span></th>
        <th scope="col">tok/s</th></tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table></div>"""


def conc_chart(m, mid):
    """tok/s against offered users, one line per engine."""
    series = []
    for key, label, _lk, sk, cls in ENGINES:
        pts = sorted((c["rung"], c.get("output_throughput")) for c in m.get(sk) or []
                     if c.get("axis") == "conc" and c.get("output_throughput") is not None)
        if pts:
            series.append((label, cls, pts))
    if not series:
        return ""
    xs = sorted({x for _, _, pts in series for x, _ in pts})
    W, H, L, R, T, B = 560, 240, 52, 96, 14, 40
    pw, ph = W - L - R, H - T - B
    ymax = nice_max(max(v for _, _, pts in series for _, v in pts))
    xpos = {x: L + (pw * i / (len(xs) - 1) if len(xs) > 1 else pw / 2) for i, x in enumerate(xs)}
    ypos = lambda v: T + ph - ph * v / ymax

    out = [f'<svg viewBox="0 0 {W} {H}" class="mchart" role="img" '
           f'aria-labelledby="{mid}-ct"><title id="{mid}-ct">Output tokens per second against '
           f'offered concurrent users, per engine</title>']
    for i in range(6):
        v = ymax * i / 5
        y = ypos(v)
        out.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    for x in xs:
        out.append(f'<text class="tick" x="{xpos[x]:.1f}" y="{T + ph + 18}" text-anchor="middle">{x}</text>')
    out.append(f'<text class="axis" x="{L + pw / 2:.1f}" y="{H - 4}" text-anchor="middle">offered concurrent users</text>')
    out.append(f'<text class="axis" x="12" y="{T + ph / 2:.1f}" text-anchor="middle" '
               f'transform="rotate(-90 12 {T + ph / 2:.1f})">tok/s</text>')
    out.append(f'<line class="xhair" x1="0" x2="0" y1="{T}" y2="{T + ph}" style="display:none"/>')
    ends = []
    for label, cls, pts in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{xpos[x]:.1f},{ypos(v):.1f}" for i, (x, v) in enumerate(pts))
        out.append(f'<path class="ln {cls}" d="{d}"/>')
        out += [f'<circle class="dot {cls}" cx="{xpos[x]:.1f}" cy="{ypos(v):.1f}" r="4"/>' for x, v in pts]
        x, v = pts[-1]
        ends.append([ypos(v), xpos[x], label, cls])
    # End labels in text ink beside their line; pushed apart where lines
    # converge, with a leader back to the line end so none detaches.
    ends.sort()
    prev = None
    for end in ends:
        prev = end[0] if prev is None else max(end[0], prev + 14)
        end.append(prev)
    for y, x, label, cls, ly in ends:
        if abs(ly - y) > 1:
            out.append(f'<line class="leader" x1="{x + 6:.1f}" y1="{y:.1f}" x2="{x + 12:.1f}" y2="{ly:.1f}"/>')
        out.append(f'<text class="endlbl" x="{x + 14:.1f}" y="{ly + 4:.1f}">{esc(label)}</text>')
    # One hit band per x, wider than any mark: the crosshair finds the x and
    # the tooltip lists every engine there.
    for i, x in enumerate(xs):
        left = (xpos[xs[i - 1]] + xpos[x]) / 2 if i else L
        right = (xpos[x] + xpos[xs[i + 1]]) / 2 if i + 1 < len(xs) else L + pw
        rows = [[fmt(dict(pts).get(x), 1), label, cls] for label, cls, pts in series]
        out.append(f'<rect class="hit" x="{left:.1f}" y="{T}" width="{right - left:.1f}" height="{ph}" '
                   f'tabindex="0" data-x="{xpos[x]:.1f}" data-head="{x} offered users · tok/s" '
                   f'data-tip="{tip(rows)}"><title>{x} users</title></rect>')
    out.append("</svg>")

    # One engine needs no legend: its end label and the caption already name it.
    legend = "".join(f'<span><span class="lkey {cls}"></span>{esc(label)}</span>'
                     for label, cls, _ in series) if len(series) > 1 else ""
    head = "".join(f"<th scope=\"col\">{x}</th>" for x in xs)
    trs = []
    for key, label, _lk, sk, cls in ENGINES:
        cs = {c["rung"]: c for c in m.get(sk) or [] if c.get("axis") == "conc"}
        if not cs:
            continue
        for metric, nd, name in (("output_throughput", 1, "tok/s"), ("median_ttft_ms", 0, "TTFT ms"),
                                 ("median_tpot_ms", 1, "TPOT ms")):
            trs.append(f'<tr><th scope="row">{esc(label)} · {name}</th>'
                       + "".join(f"<td>{cfmt(cs.get(x), metric, nd)}</td>" for x in xs) + "</tr>")
    return f"""<figure class="mfig">
  <figcaption>Throughput as users are added <span class="dim">(input {esc(_base(m, 'input'))}, output {esc(_base(m, 'output'))} tokens)</span></figcaption>
  <div class="legend">{legend}</div>
  <div class="chartbox">{''.join(out)}</div>
  <details><summary>Table view</summary><div class="mtable"><table>
    <thead><tr><th scope="col">offered users</th>{head}</tr></thead><tbody>{''.join(trs)}</tbody>
  </table></div></details>
</figure>"""


def _base(m, k):
    return m.get("_base", {}).get(k, "?")


def ratio_tint(r):
    """Diverging tint for a 'how many times faster is scratchy' ratio: blue
    above 1, red below, neutral at 1, saturating at 4x either way."""
    if r is None:
        return ""
    k = min(abs(math.log2(r)) / 2, 1) * 50
    pole = "var(--div-pos)" if r >= 1 else "var(--div-neg)"
    return f' style="background: color-mix(in oklab, {pole} {k:.0f}%, var(--bg-soft))"'


def grid_maps(m, run):
    sc = run["config"].get("scaling") or {}
    ins, outs = sc.get("grid_input") or [], sc.get("grid_output") or []
    have = {e[0]: {c["rung"]: c for c in m.get(e[3]) or [] if c.get("axis") == "grid"} for e in ENGINES}
    if not have["scratchy"] or not ins or not outs:
        return ""
    conc = (sc.get("base") or {}).get("conc", "?")
    # Colour against mlx-lm, which runs the same MLX checkpoint; failing that,
    # against ollama; with neither, show scratchy's own values uncoloured rather
    # than a grid of blanks. The caption and scale say which.
    present = [k for k in ("mlx-lm", "ollama") if have[k]]
    rival = present[0] if present else None
    other = present[1] if len(present) > 1 else None

    def one(metric, title, faster):
        rows = []
        for i in ins:
            tds = []
            for o in outs:
                rung = f"{i}x{o}"
                get = lambda k: have[k].get(rung) if k else None
                cs, cr, co = get("scratchy"), get(rival), get(other)
                s, vr, vo = timing(cs, metric), timing(cr, metric), timing(co, metric)
                r_r, r_o = faster(s, vr), faster(s, vo)
                nd = 1 if metric == "output_throughput" else 0
                tiprows = [[cfmt(have[k].get(rung), metric, nd), k, cls]
                           for k, _l, _lk, _sk, cls in ENGINES if have[k]]
                why = lambda c, name: f"{name}: {NO_STREAM}" if c else f"no {name}"
                unit = "tok/s" if metric == "output_throughput" else "ms"
                if rival is None:                 # nothing to compare against
                    big, small = cfmt(cs, metric, nd), unit if s is not None else ""
                else:
                    big = (f"×{r_r:.2f}" if r_r is not None
                           else why(cs, "scratchy") if s is None else why(cr, rival))
                    small = (f"vs {other} ×{r_o:.2f}" if r_o is not None
                             else "" if s is None or not other else why(co, other))
                tds.append(f'<td{ratio_tint(r_r)} tabindex="0" data-head="{i} in × {o} out · {esc(title)}" '
                           f'data-tip="{tip(tiprows)}"><b>{big}</b><span>{small}</span></td>')
            rows.append(f'<tr><th scope="row">{i}</th>{"".join(tds)}</tr>')
        head = "".join(f'<th scope="col">{o}</th>' for o in outs)
        return f"""<div class="heat">
    <div class="heatttl">{esc(title)}</div>
    <table><thead><tr><th scope="col"><span class="dim">in ↓ / out →</span></th>{head}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
  </div>"""

    tput = lambda s, o: s / o if s and o else None    # higher is better
    ttft = lambda s, o: o / s if s and o else None    # lower is better
    if rival is None:
        note = "scratchy's own values; this run has no mlx-lm or ollama to compare against"
        scale = ""
    else:
        missing = [k for k in ("mlx-lm", "ollama") if not have[k]]
        note = (f"how many times faster scratchy is than {rival}"
                + (f", since this run has no {missing[0]}" if missing else "")
                + "; hover for every engine's value")
        scale = (f'<div class="scale"><span><i class="sw neg"></i>{rival} faster</span>'
                 '<span><i class="sw mid"></i>about the same</span>'
                 '<span><i class="sw pos"></i>scratchy faster</span></div>')
    return f"""<figure class="mfig">
  <figcaption>Prompt size × answer size, {esc(conc)} users <span class="dim">({note})</span></figcaption>
  <div class="heatrow">
  {one("output_throughput", "throughput", tput)}
  {one("median_ttft_ms", "time to first token", ttft)}
  </div>
  {scale}
</figure>"""


def history(entries):
    if len(entries) < 2:
        return ""
    rows = []
    for run, m in reversed(entries):
        sha = str(run["repo"]["sha"])[:8]
        ladder, one = m.get("cache_ladder"), cell(m.get("scaling"), "conc", 1)
        if served(m):
            vals = [fmt(med(ladder, "cold", "t_ready_s"), 2), fmt(med(ladder, "cold", "ttft_from_send_s"), 2),
                    fmt(med(ladder, "warm", "ttft_from_send_s"), 0, 1000),
                    fmt(one and one.get("output_throughput"), 1)]
            tds = "".join(f"<td>{v}</td>" for v in vals)
        else:
            tds = '<td colspan="4" class="gap">no numbers</td>'
        rows.append(f'<tr><th scope="row">{run["generated_utc"][:10]}</th>'
                    f'<td><a href="{REPO}/commit/{esc(run["repo"]["sha"])}"><code>{sha}</code></a>'
                    f'{" <span class=dim>+ uncommitted</span>" if run["repo"].get("dirty") else ""}</td>'
                    f'<td>{esc(m.get("quant") or "")}</td>{tds}</tr>')
    return f"""<details class="hist"><summary>Earlier runs of this model ({len(entries) - 1})</summary><div class="mtable"><table>
  <thead><tr><th scope="col">run</th><th scope="col">scratchy</th><th scope="col">quant</th>
  <th scope="col">cold s</th><th scope="col">1st req s</th><th scope="col">warm ms</th><th scope="col">tok/s, 1 user</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div></details>"""


def runs_table(runs):
    rows = []
    for run in reversed(runs):
        r, c, mc = run["repo"], run["config"], run["machine"]
        prime = c.get("cold_priming_launches")
        rows.append(
            f'<tr><th scope="row">{esc(run["generated_utc"][:16].replace("T", " "))} UTC</th>'
            f'<td><a href="{REPO}/commit/{esc(r["sha"])}"><code>{esc(str(r["sha"])[:8])}</code></a>'
            f'{" <span class=dim>+ uncommitted changes</span>" if r.get("dirty") else ""}</td>'
            f'<td>{esc(", ".join(c.get("scenarios") or []))}</td>'
            f'<td>{prime if prime is not None else "not recorded"}</td>'
            f'<td>{esc(c.get("kv_cache_dtype") or "")}</td>'
            f'<td>{esc(mc.get("power") or "not recorded")}</td>'
            f'<td><a href="data/metal/{esc(run["_file"].name)}">json</a></td></tr>')
    return f"""<details class="hist"><summary>Runs on this machine ({len(runs)})</summary><div class="mtable"><table>
  <thead><tr><th scope="col">when</th><th scope="col">scratchy</th><th scope="col">steps</th>
  <th scope="col">cold priming launches</th><th scope="col">scratchy KV cache</th><th scope="col">power</th><th scope="col">data</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table></div></details>"""


def machine_section(chip, runs):
    mslug = slug(chip)
    latest, seen = {}, {}
    for run in runs:                       # oldest first, so newer runs win
        for m in run["models"]:
            latest[m["stem"]] = (run, m)
            seen.setdefault(m["stem"], []).append((run, m))
    mc = runs[-1]["machine"]
    cores = f'{mc["cores_total"]} cores'
    if mc.get("cores_performance"):
        cores += f' ({mc["cores_performance"]}P + {mc.get("cores_efficiency", 0)}E)'
    parts = [f"""<section class="msection" id="{mslug}">
  <h2>{esc(chip)}</h2>
  <p class="mmeta">{cores} · {esc(mc["memory_gb"])} GB unified memory · macOS {esc(mc["macos"])}</p>
  {runs_table(runs)}"""]
    for stem, (run, m) in latest.items():
        m = dict(m, _base=(run["config"].get("scaling") or {}).get("base") or {})
        mid = f"{mslug}-{slug(stem)}"
        f = m.get("footprint") or {}
        build = ""
        if f.get("build_seconds") is not None:
            build = f' · scratchy build {f["build_seconds"]} s, {round(f["binary_bytes"] / 1048576)} MiB binary'
        parts.append(f"""  <article class="mmodel" id="{mid}">
    <h3>{esc(stem)}</h3>
    <p class="mmeta"><a href="https://huggingface.co/{esc(m["model_id"])}">{esc(m["model_id"])}</a>
      · {esc(m.get("quant") or "default")}{build}
      · run {esc(run["generated_utc"][:10])}, <code>{esc(str(run["repo"]["sha"])[:8])}</code></p>
    {summary_table(m, run)}
    {conc_chart(m, mid)}
    {grid_maps(m, run)}
    {history(seen[stem])}
  </article>""")
    parts.append("</section>")
    return "\n".join(parts)


def page(header, data_dir):
    runs = load(data_dir)
    machines = {}
    for run in sorted(runs, key=lambda r: r["generated_utc"]):
        machines.setdefault(run["machine"]["chip"], []).append(run)

    if machines:
        body = "\n".join(machine_section(chip, rs) for chip, rs in sorted(machines.items()))
    else:
        body = ('<p class="empty">No runs published yet. Run <code>scripts/bench_metal_matrix.sh</code> '
                'and copy its JSON into <code>site/data/metal/</code>.</p>')
    nav = "\n".join(
        f'      <cds-side-nav-link href="#{slug(chip)}">{esc(chip)}</cds-side-nav-link>'
        for chip in sorted(machines))
    flags = []
    if any(r["repo"].get("dirty") for r in runs):
        flags.append("<li>Some runs were made from a tree with <b>uncommitted changes</b>; the runs table marks which.</li>")
    if any("battery" in (r["machine"].get("power") or "").lower() for r in runs):
        flags.append("<li>Some runs were made <b>on battery power</b>, which can slow a Mac down; the runs table marks which.</li>")

    out = PAGE
    for key, val in {
        "{header}": header,
        "{nav}": nav,
        "{flags}": "\n".join(flags),
        "{body}": body,
        "{count}": str(len(runs)),
        "{machines}": str(len(machines)),
        "{repo}": REPO,
    }.items():
        out = out.replace(key, val)
    return out, runs, len(machines)


def build(out_path, header, data_dir=DATA):
    out, runs, n = page(header, data_dir)
    out_path = Path(out_path)
    out_path.write_text(out)
    dest = out_path.parent / "data" / "metal"
    dest.mkdir(parents=True, exist_ok=True)
    for run in runs:
        shutil.copy(run["_file"], dest / run["_file"].name)
    print(f"metal page: {len(runs)} runs on {n} machines -> {out_path}")


def main():
    sys.path.insert(0, str(HERE))
    from build import header_html  # the one definition of the site chrome

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_site/metal.html"
    data = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA
    build(out, header_html("", active="metal.html"), data)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Metal performance — scratchy</title>
<meta name="description" content="scratchy against mlx-lm and ollama on Apple silicon: startup, single-user speed, and scaling, per machine.">
<link rel="icon" type="image/png" href="favicon.png">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@carbon/styles@1/css/styles.min.css">
<link rel="stylesheet" href="styles.css">
<script type="module" src="https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest/ui-shell.min.js"></script>
<script>
(function () {
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  function apply(dark) {
    document.documentElement.classList.remove('cds--g100', 'cds--white');
    document.documentElement.classList.add(dark ? 'cds--g100' : 'cds--white');
  }
  apply(mq.matches);
  mq.addEventListener('change', function (e) { apply(e.matches); });
})();
</script>
</head>
<body>

{header}

<cds-side-nav aria-label="Performance" class="docs-side-nav">
  <cds-side-nav-items>
    <cds-side-nav-menu title="Metal" expanded>
      <cds-side-nav-link href="#about">About these numbers</cds-side-nav-link>
{nav}
    </cds-side-nav-menu>
  </cds-side-nav-items>
</cds-side-nav>

<div class="docs-layout">
<main class="docs-content metalpage">
  <div class="hero-body">
    <h1>Metal performance</h1>
    <p class="lede">scratchy against mlx-lm and ollama on Apple silicon: how long each takes
    to start, how fast it answers one user, and how it holds up as prompts, answers and users
    grow. {count} runs on {machines} machines, every number measured by
    <a href="{repo}/blob/main/scripts/bench_metal_matrix.sh"><code>scripts/bench_metal_matrix.sh</code></a>.</p>
  </div>

  <section class="msection" id="about">
    <h2>About these numbers</h2>
    <ul class="caveats">
      <li><b>Made-up prompts, not real traffic.</b> Each request gets a unique random prompt at a
      fixed size, so no engine can answer from a cache. This measures the engine, not a workload.</li>
      <li><b>Startup:</b> <i>frozen</i> is a first-ever launch (memory cleared, scratchy's weights
      cache deleted); <i>cold</i> is a normal launch after earlier ones. Both are launch until the
      server answers "ready", with no request in them. <i>1st req</i> is the cold server's first
      request, send to first token. <i>warm</i> is the median send to first token over unique
      prompts to a server that has already answered once.</li>
      <li><b>ollama says "ready" before it loads the model.</b> Its server answers at once and
      loads the model on the first request, so its frozen and cold times are small and its
      load time shows up in <i>1st req</i> instead.</li>
      <li><b>scratchy compiles the model into its binary.</b> Build time and binary size are shown
      per model and are not part of any startup number. Build time is the runner's own
      <code>cargo build</code>, so a model built just before reads as a second or two.</li>
      <li><b>ollama runs different weights</b> (its own GGUF quantization, named per row), so its
      numbers mix engine and quantizer. mlx-lm runs the same MLX checkpoint as scratchy.</li>
      <li><b>mlx-lm and ollama can stop answers early</b> (they ignore "don't stop"), so compare
      TTFT and TPOT; tok/s can flatter an engine that generated less.</li>
      <li><b>"no stream"</b> means the engine sent its whole answer at once (or a single
      token), so there was no first token or per-token interval to time; its TTFT would be
      the whole answer's time. ollama's gemma4 does this.</li>
      <li><b>Greedy decoding everywhere.</b> Every request sends <code>temperature: 0</code>
      explicitly, so no engine falls back to its own default sampling; each prompt is still
      unique.</li>
      <li><b>Users are offered, not achieved:</b> "16 users" means up to 16 requests in flight;
      an engine may batch fewer.</li>
      <li><b>Peak RSS</b> is the server process. For ollama that is <code>ollama serve</code>
      only; its runner process is not counted.</li>
{flags}
    </ul>
  </section>

{body}
</main>
</div>

<div id="mtip" class="mtip" role="tooltip" hidden></div>
<script>
// Hover and keyboard focus show the same readout. Values come from data-tip
// and are written with textContent only.
(function () {
  var tipEl = document.getElementById('mtip');
  function show(el) {
    var rows = JSON.parse(el.getAttribute('data-tip') || '[]');
    tipEl.textContent = '';
    var h = document.createElement('div');
    h.className = 'mtiphead';
    h.textContent = el.getAttribute('data-head') || '';
    tipEl.appendChild(h);
    rows.forEach(function (r) {
      var row = document.createElement('div');
      var key = document.createElement('span');
      key.className = 'lkey ' + r[2];
      var v = document.createElement('b');
      v.textContent = r[0];
      var l = document.createElement('span');
      l.className = 'dim';
      l.textContent = ' ' + r[1];
      row.append(key, v, l);
      tipEl.appendChild(row);
    });
    tipEl.hidden = false;
    var b = el.getBoundingClientRect();
    tipEl.style.left = (window.scrollX + Math.min(window.innerWidth - tipEl.offsetWidth - 8, b.left + b.width / 2 + 12)) + 'px';
    tipEl.style.top = (b.top + window.scrollY + 8) + 'px';
    var x = el.getAttribute('data-x');
    var hair = x && el.ownerSVGElement && el.ownerSVGElement.querySelector('.xhair');
    if (hair) { hair.setAttribute('x1', x); hair.setAttribute('x2', x); hair.style.display = ''; }
  }
  function hide(el) {
    tipEl.hidden = true;
    var hair = el.ownerSVGElement && el.ownerSVGElement.querySelector('.xhair');
    if (hair) hair.style.display = 'none';
  }
  document.querySelectorAll('[data-tip]').forEach(function (el) {
    el.addEventListener('pointerenter', function () { show(el); });
    el.addEventListener('pointerleave', function () { hide(el); });
    el.addEventListener('focus', function () { show(el); });
    el.addEventListener('blur', function () { hide(el); });
  });
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
