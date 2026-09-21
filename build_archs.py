#!/usr/bin/env python3
"""Generate site/_site/architectures.html from crates/models/arch/dsl/*.rs.in.

The DSL files stay the single source of truth: this reads them at site-build
time, so the page cannot drift from the compiler input. Called from build.py,
which passes in the shared site header so the chrome matches every other page.

Standalone: site/build_archs.py [out.html]
"""

import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

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



# Carbon Web Components used below, one module per component family.
# https://web-components.carbondesignsystem.com
CDN = "https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest"
MODULES = ["ui-shell", "button", "tile", "select", "checkbox"]


def select(el_id, label, chosen, order, archs):
    items = "\n".join(
        f'    <cds-select-item value="{n}"{" selected" if n == chosen else ""}>'
        f'{n} \u2014 {archs[n]["lines"]} lines</cds-select-item>'
        for n in order
    )
    return (f'  <cds-select id="{el_id}" label-text="{label}" value="{chosen}">\n'
            f'{items}\n  </cds-select>')


def side_nav(order, chosen, compare):
    lines = []
    for n in order:
        active = " active" if n == chosen else ""
        lines.append(
            f'    <cds-side-nav-link href="#{n}..{compare}" data-arch="{n}"{active}>'
            f'{n}</cds-side-nav-link>'
        )
    return "\n".join(lines)


def page(header):
    """The whole standalone page. `header` is build.py's shared site chrome."""
    files = sorted((ROOT / "crates/models/arch/dsl").glob("*.rs.in"))
    if not files:
        sys.exit(f"no DSL files under {ROOT / 'crates/models/arch/dsl'}")

    archs = {}
    for f in files:
        text = f.read_text()
        lines = text.splitlines()
        archs[f.name[: -len(".rs.in")]] = {
            "path": str(f.relative_to(ROOT)),
            "raw": lines,
            "html": [highlight(l) for l in lines],
            "lines": len(lines),
        }

    base = "llama" if "llama" in archs else next(iter(archs))
    for a in archs.values():
        a["del"], a["add"] = diff_count(archs[base]["raw"], a["raw"])
        a["distance"] = a["del"] + a["add"]

    # Sorted by distance from the baseline: the ordering is the argument.
    order = sorted(archs, key=lambda n: (archs[n]["distance"], n))
    compare = "granite" if "granite" in archs else next(n for n in order if n != base)
    total = sum(a["lines"] for a in archs.values())

    out = PAGE
    for key, val in {
        "{modules}": "\n".join(
            f'<script type="module" src="{CDN}/{m}.min.js"></script>' for m in MODULES
        ),
        "{header}": header,
        "{right}": select("right", "Compare", compare, order, archs),
        "{nav}": side_nav(order, base, compare),
        "{data}": json.dumps({"base": base, "order": order, "archs": archs,
                              "left": base, "right": compare}, separators=(",", ":")),
        "{count}": str(len(archs)),
        "{total}": f"{total:,}",
        "{repo}": REPO,
    }.items():
        out = out.replace(key, val)
    return out, len(archs), total


def build(out_path, header):
    out, count, total = page(header)
    Path(out_path).write_text(out)
    print(f"architectures page: {count} models, {total} DSL lines -> {out_path}")


def main():
    sys.path.insert(0, str(HERE))
    from build import header_html  # the one definition of the site chrome

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_site/architectures.html"
    build(out, header_html("", active="architectures.html"))


REPO = "https://github.com/AI-native-Systems-Research/scratchy"

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model architectures — scratchy</title>
<meta name="description" content="Every model architecture scratchy supports, written in its math DSL, diffable side by side.">
<link rel="icon" type="image/png" href="favicon.png">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@carbon/styles@1/css/styles.min.css">
<link rel="stylesheet" href="styles.css">
{modules}
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

<cds-side-nav aria-label="Model architectures" class="docs-side-nav" id="model-nav">
  <cds-side-nav-items>
{nav}
  </cds-side-nav-items>
</cds-side-nav>

<div class="docs-layout">
<main class="docs-content archpage">
  <div class="hero-body">
  <h1>Scratchy model architectures</h1>
  <p class="lede">{count} architectures, {total} lines of DSL between them. Each one is
  the model's math, written once; the compiler turns it into the code for every
  target. Pick a model from the list on the left as the baseline, then choose
  what to diff it against below.</p>
</div>

  <div class="archbar">
{right}
    <span id="tally" class="tally"></span>
  </div>

  <div class="archgrid">
    <cds-tile class="archpane">
      <div class="codehead"><span id="lpath"></span><span id="lmeta" class="cmeta"></span></div>
      <pre class="archcode"><code id="lcode"></code></pre>
    </cds-tile>
    <cds-tile class="archpane">
      <div class="codehead"><span id="rpath"></span><span id="rmeta" class="cmeta"></span></div>
      <pre class="archcode"><code id="rcode"></code></pre>
    </cds-tile>
  </div>
</main>
</div>

<script type="application/json" id="archdata">{data}</script>
<script>
const DATA = JSON.parse(document.getElementById('archdata').textContent);
const A = DATA.archs, ORDER = DATA.order;
const $ = id => document.getElementById(id);
const norm = s => s.replace(/\/\/.*$/, '').split(/\s+/).join(' ').trim();

// The selects are rendered with their initial selection server-side, so the
// page is correct before the component modules finish loading; L and R are the
// authority afterwards.
let L = DATA.left, R = DATA.right, FOLD = false;

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
  const ln = A[L].raw.map(norm), rn = A[R].raw.map(norm);
  const ops = align(ln, rn);
  let adds = 0, dels = 0, lout = '', rout = '';
  for (const [t, i, j] of ops) {
    // A blank or comment-only line is not an edit, whichever side it is on.
    const real = t === '-' ? ln[i] !== '' : t === '+' ? rn[j] !== '' : false;
    if (real) { if (t === '+') adds++; else dels++; }
    if (FOLD && !real) continue;
    lout += row(L, i, real && t === '-' ? 'del' : '');
    rout += row(R, j, real && t === '+' ? 'add' : '');
  }
  $('lcode').innerHTML = lout;
  $('rcode').innerHTML = rout;
  $('lpath').textContent = A[L].path;
  $('rpath').textContent = A[R].path;
  $('lmeta').textContent = A[L].lines + ' lines';
  $('rmeta').textContent = A[R].lines + ' lines';
  $('tally').textContent = L === R ? 'same file'
    : (adds + dels) === 0 ? 'identical math'
    : `+${adds} −${dels} lines`;

  for (const link of document.querySelectorAll('#model-nav cds-side-nav-link[data-arch]')) {
    link.toggleAttribute('active', link.dataset.arch === L);
    link.href = '#' + link.dataset.arch + '..' + R;
  }

  history.replaceState(null, '', '#' + L + '..' + R);
}

function pick(side, name) {
  if (!A[name]) return;
  if (side === 'left') { L = name; }
  else { R = name; $('right').value = name; }
  render();
}

$('right').addEventListener('cds-select-selected', e => pick('right', e.detail.value));

for (const link of document.querySelectorAll('#model-nav cds-side-nav-link[data-arch]')) {
  link.addEventListener('click', e => {
    e.preventDefault();
    pick('left', link.dataset.arch);
  });
}

// A #left..right fragment makes one specific comparison linkable, and stays
// live afterward so the side-nav's own generated hrefs (and back/forward)
// keep working.
function applyHash() {
  const pair = decodeURIComponent(location.hash.slice(1)).split('..');
  if (A[pair[0]]) pick('left', pair[0]);
  if (A[pair[1]]) pick('right', pair[1]);
  render();
}
window.addEventListener('hashchange', applyHash);
applyHash();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
