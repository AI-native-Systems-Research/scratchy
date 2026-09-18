#!/usr/bin/env python3
"""Generate site/_site/architectures.html from crates/models/arch/dsl/*.rs.in.

The DSL files stay the single source of truth: this reads them at site-build
time, so the page cannot drift from the compiler input. Called by site/build.sh.

Usage: build-archs.py <repo-root> <output-html>
"""

import html
import json
import re
import sys
from pathlib import Path

# Same token classes as the snippet on the landing page (.c-* in styles.css).
TOKENS = re.compile(
    r"""(?P<cm>//.*$)
      | (?P<at>\#\[[A-Za-z_][A-Za-z0-9_]*\])
      | (?P<s>"(?:[^"\\]|\\.)*")
      | (?P<kw>\b(?:fn|for|in|let|if|else|match|while|return|mut|as|true|false)\b)
      | (?P<fn>\b[A-Za-z_][A-Za-z0-9_]*\b(?=\s*\())
      | (?P<n>\b\d[A-Za-z0-9_.]*\b)
    """,
    re.VERBOSE,
)

# Loop/branch keywords read as verbs to the `name(` rule; never call them ops.
NOT_OPS = {"fn", "for", "in", "if", "else", "match", "while", "let", "return"}


def highlight(line):
    out, at = [], 0
    for m in TOKENS.finditer(line):
        out.append(html.escape(line[at:m.start()]))
        cls = m.lastgroup
        out.append(f'<span class="c-{cls}">{html.escape(m.group())}</span>')
        at = m.end()
    out.append(html.escape(line[at:]))
    return "".join(out)


def strip_comment(line):
    return line.split("//", 1)[0]


def ops_of(text):
    """Verbs the architecture calls, minus the ones it defines itself."""
    body = "\n".join(strip_comment(l) for l in text.splitlines())
    called = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", body))
    defined = set(re.findall(r"\bfn\s+([a-z_][a-z0-9_]*)", body))
    return sorted(called - defined - NOT_OPS)


def norm(line):
    """Comparison key: whitespace-insensitive, comments ignored."""
    return " ".join(strip_comment(line).split())


def diff_count(a, b):
    """(removed, added) code lines under an LCS alignment. A rewritten line
    counts once on each side, which is what a side-by-side view shows."""
    A = [norm(l) for l in a if norm(l)]
    B = [norm(l) for l in b if norm(l)]
    n, m = len(A), len(B)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if A[i] == B[j] else max(dp[i + 1][j], dp[i][j + 1])
    common = dp[0][0]
    return n - common, m - common


def main():
    root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    dsl_dir = root / "crates/models/arch/dsl"
    files = sorted(dsl_dir.glob("*.rs.in"))
    if not files:
        sys.exit(f"no DSL files under {dsl_dir}")

    archs = {}
    for f in files:
        text = f.read_text()
        lines = text.splitlines()
        archs[f.name[: -len(".rs.in")]] = {
            "path": str(f.relative_to(root)),
            "raw": lines,
            "html": [highlight(l) for l in lines],
            "ops": ops_of(text),
            "lines": len(lines),
            "code": sum(1 for l in lines if norm(l)),
        }

    base = "llama" if "llama" in archs else next(iter(archs))
    for name, a in archs.items():
        a["del"], a["add"] = diff_count(archs[base]["raw"], a["raw"])
        a["distance"] = a["del"] + a["add"]

    # Sorted by distance from the baseline: the ordering is the argument.
    order = sorted(archs, key=lambda n: (archs[n]["distance"], n))
    total = sum(a["lines"] for a in archs.values())

    data = json.dumps({"base": base, "order": order, "archs": archs}, separators=(",", ":"))
    # Plain substitution, not str.format: the template is full of JS/CSS braces.
    page = PAGE
    for key, val in {
        "{data}": data,
        "{count}": str(len(archs)),
        "{total}": f"{total:,}",
        "{base_lines}": str(archs[base]["lines"]),
        "{base}": base,
        "{repo}": REPO,
    }.items():
        page = page.replace(key, val)
    out_path.write_text(page)
    print(f"architectures page: {len(archs)} models, {total} DSL lines -> {out_path}")


REPO = "https://github.com/AI-native-Systems-Research/scratchy"

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model architectures &middot; scratchy</title>
<meta name="description" content="Every model architecture scratchy supports, written in its math DSL, diffable side by side.">
<link rel="icon" href="favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="styles.css">
</head>
<body>
<header class="header-global">
  <a class="brand" href="index.html">scratchy</a>
  <nav class="navlinks">
    <a href="architectures.html" aria-current="page">Models</a>
    <a href="book/index.html">Docs</a>
    <a href="book/COMPILER.html">Compiler</a>
    <a href="{repo}">GitHub</a>
  </nav>
</header>

<main class="archpage">
  <h1>Model architectures</h1>
  <p class="lede">{count} architectures, {total} lines of DSL between them. Each one is the
  model's math, written once; the compiler turns it into the code for every target.
  Pick any two to see what actually differs.</p>

  <div class="archbar">
    <label>Baseline
      <select id="left"></select>
    </label>
    <button id="swap" class="btn" type="button" title="Swap the two sides">&#8646;</button>
    <label>Compare
      <select id="right"></select>
    </label>
    <label class="chk"><input type="checkbox" id="fold"> Changed lines only</label>
    <span id="tally" class="tally"></span>
  </div>

  <div class="archgrid">
    <div class="archpane">
      <div class="codehead"><span id="lpath"></span><span id="lmeta" class="cmeta"></span></div>
      <pre class="archcode"><code id="lcode"></code></pre>
    </div>
    <div class="archpane">
      <div class="codehead"><span id="rpath"></span><span id="rmeta" class="cmeta"></span></div>
      <pre class="archcode"><code id="rcode"></code></pre>
    </div>
  </div>

  <div id="opsbox" class="opsbox"></div>

  <h2>Distance from <code>{base}</code></h2>
  <p class="sub">{base} is {base_lines} lines. Most architectures are a handful of edits
  away from it &mdash; the ones at the bottom of the list are where the real work is.</p>
  <div class="archlist" id="list"></div>
</main>

<script type="application/json" id="archdata">{data}</script>
<script>
const DATA = JSON.parse(document.getElementById('archdata').textContent);
const A = DATA.archs, ORDER = DATA.order;
const $ = id => document.getElementById(id);
const norm = s => s.replace(/\/\/.*$/, '').split(/\s+/).join(' ').trim();

function fill(sel, chosen) {
  sel.innerHTML = ORDER.map(n =>
    `<option value="${n}"${n === chosen ? ' selected' : ''}>${n} &nbsp; ${A[n].lines} lines</option>`
  ).join('');
}

// Longest common subsequence over comment- and whitespace-insensitive lines,
// so the panes stay aligned and only real edits light up.
function align(a, b) {
  const n = a.length, m = b.length;
  const dp = Array.from({length: n + 1}, () => new Int32Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const ops = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { ops.push(['=', i++, j++]); }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push(['-', i++, -1]); }
    else { ops.push(['+', -1, j++]); }
  }
  while (i < n) ops.push(['-', i++, -1]);
  while (j < m) ops.push(['+', -1, j++]);
  return ops;
}

// One flex row per line so a whole line can carry a diff background, and so the
// two panes stay aligned when one side has no counterpart.
function row(arch, idx, cls) {
  if (idx < 0) return '<span class="r fill"><span class="ln"></span><span class="lx">&nbsp;</span></span>';
  return `<span class="r ${cls}"><span class="ln">${idx + 1}</span>` +
         `<span class="lx">${A[arch].html[idx] || '&nbsp;'}</span></span>`;
}

function render() {
  const l = $('left').value, r = $('right').value;
  const fold = $('fold').checked;
  const ln = A[l].raw.map(norm), rn = A[r].raw.map(norm);
  const ops = align(ln, rn);
  let adds = 0, dels = 0, lout = '', rout = '';
  for (const [t, i, j] of ops) {
    // A blank or comment-only line is not an edit, whichever side it is on.
    const real = t === '-' ? ln[i] !== '' : t === '+' ? rn[j] !== '' : false;
    if (real) { if (t === '+') adds++; else dels++; }
    if (fold && !real) continue;
    lout += row(l, i, real && t === '-' ? 'del' : '');
    rout += row(r, j, real && t === '+' ? 'add' : '');
  }
  $('lcode').innerHTML = lout;
  $('rcode').innerHTML = rout;
  $('lpath').textContent = A[l].path;
  $('rpath').textContent = A[r].path;
  $('lmeta').textContent = A[l].lines + ' lines';
  $('rmeta').textContent = A[r].lines + ' lines';
  $('tally').textContent = l === r ? 'same file'
    : (adds + dels) === 0 ? 'identical math'
    : `+${adds} \u2212${dels} lines`;

  const lops = new Set(A[l].ops);
  const extra = A[r].ops.filter(o => !lops.has(o));
  const gone = A[l].ops.filter(o => !new Set(A[r].ops).has(o));
  $('opsbox').innerHTML = (l === r) ? '' :
    `<div class="opsrow"><span class="opslabel">only in ${r}</span>${
      extra.length ? extra.map(o => `<code class="op add">${o}</code>`).join('') : '<span class="none">nothing &mdash; same operations</span>'}</div>` +
    `<div class="opsrow"><span class="opslabel">only in ${l}</span>${
      gone.length ? gone.map(o => `<code class="op del">${o}</code>`).join('') : '<span class="none">nothing</span>'}</div>`;

  location.replace('#' + l + '..' + r);
}

function list() {
  $('list').innerHTML = ORDER.map(n => {
    return `<button class="archrow" data-arch="${n}">
      <span class="an">${n}</span>
      <span class="ad">${n === DATA.base ? 'baseline' : '+' + A[n].add + ' \u2212' + A[n].del}</span>
      <span class="al">${A[n].lines} lines</span>
      <span class="ao">${A[n].ops.length} ops</span>
    </button>`;
  }).join('');
  for (const b of document.querySelectorAll('.archrow'))
    b.onclick = () => { $('right').value = b.dataset.arch; render();
      document.querySelector('.archgrid').scrollIntoView({behavior: 'smooth', block: 'center'}); };
}

const pair = decodeURIComponent(location.hash.slice(1)).split('..');
const start = (n, dflt) => (n && A[n]) ? n : dflt;
fill($('left'), start(pair[0], DATA.base));
const showcase = A['granite'] ? 'granite' : ORDER.find(n => n !== DATA.base) || DATA.base;
fill($('right'), start(pair[1], showcase));
$('left').onchange = $('right').onchange = $('fold').onchange = render;
$('swap').onclick = () => {
  const l = $('left').value; $('left').value = $('right').value; $('right').value = l; render();
};
list();
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
