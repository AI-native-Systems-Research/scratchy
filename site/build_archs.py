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


# Some architectures wrap in a `mod <name> { fn forward() { ... } }` and
# already call their entry point `forward`; the simpler ones (no `mod`
# wrapper) still name it after themselves, `fn llama() {`. That's a purely
# nominal difference, not a structural one — left alone it shows up as a
# diff line (and inflates the ranking's "distance from baseline") on every
# single comparison. Rewritten for display/diffing only; the compiler's
# actual DSL files are untouched.
def normalize_entry_fn(stem, lines):
    ident = stem.replace("-", "_")
    pattern = re.compile(rf"^(fn ){re.escape(ident)}(\(\))")
    return [pattern.sub(r"\1forward\2", l, count=1) for l in lines]


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
MODULES = ["ui-shell", "button", "tile", "select"]

# diff-view-element: a real, maintained diff web component (PrismJS-backed)
# \u2014 https://konnorrogers.github.io/diff-view-element/. Pinned to the major
# version, matching this file's @carbon/styles@1 pin.
DIFF_VIEW_CDN = (
    "https://cdn.jsdelivr.net/npm/diff-view-element@1"
    "/cdn/exports/components/diff-view-element/diff-view-element-register.js"
)
# Rust isn't one of diff-view-element's built-in languages; this loads the
# grammar into its own PrismJS instance on first use (see the JS below).
PRISM_RUST_CDN = "https://cdn.jsdelivr.net/npm/prism-esm/components/prism-rust.js"


def select(el_id, label, chosen, order, archs, placeholder=None):
    items = "\n".join(
        f'    <cds-select-item value="{n}"{" selected" if n == chosen else ""}>'
        f'{n} \u2014 {archs[n]["lines"]} lines</cds-select-item>'
        for n in order
    )
    lead = ""
    if placeholder is not None:
        selected = " selected" if chosen == "" else ""
        lead = f'    <cds-select-item value=""{selected}>{placeholder}</cds-select-item>\n'
    return (f'  <cds-select id="{el_id}" label-text="{label}" value="{chosen}">\n'
            f'{lead}{items}\n  </cds-select>')


def side_nav(order, chosen):
    lines = []
    for n in order:
        active = " active" if n == chosen else ""
        lines.append(
            f'    <cds-side-nav-link href="#{n}" data-arch="{n}"{active}>'
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
        stem = f.name[: -len(".rs.in")]
        lines = normalize_entry_fn(stem, f.read_text().splitlines())
        archs[stem] = {
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
    total = sum(a["lines"] for a in archs.values())

    out = PAGE
    for key, val in {
        "{modules}": "\n".join(
            f'<script type="module" src="{CDN}/{m}.min.js"></script>' for m in MODULES
        ),
        "{diff_view_cdn}": DIFF_VIEW_CDN,
        "{header}": header,
        "{right}": select("right", "Diff against", "", order, archs, placeholder="— none —"),
        "{nav}": side_nav(order, base),
        "{data}": json.dumps({"base": base, "order": order, "archs": archs,
                              "left": base, "right": "", "prismRustUrl": PRISM_RUST_CDN},
                             separators=(",", ":")),
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
<script type="module" src="{diff_view_cdn}"></script>
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
  target. Pick a model from the list on the left, then optionally diff it
  against another below.</p>
</div>

  <div class="archbar">
{right}
  </div>

  <cds-tile id="archpane" class="archpane">
    <div class="codehead"><span id="path"></span><span id="meta" class="cmeta"></span></div>
    <pre class="archcode"><code id="code"></code></pre>
  </cds-tile>

  <cds-tile id="diffpane" class="archpane" style="display: none">
    <div class="codehead"><span id="diffhead"></span></div>
    <diff-view-element id="diffview" language="rust" disable-line-numbers></diff-view-element>
  </cds-tile>
</main>
</div>

<script type="application/json" id="archdata">{data}</script>
<script>
const DATA = JSON.parse(document.getElementById('archdata').textContent);
const A = DATA.archs, ORDER = DATA.order;
const $ = id => document.getElementById(id);

// The select is rendered with its initial (empty) selection server-side, so
// the page is correct before the component modules finish loading; L and R
// are the authority afterwards. R === '' means "no diff — just show L".
let L = DATA.left, R = DATA.right;

// diff-view-element ships without Rust highlighting; load the grammar into
// its own PrismJS instance once, the first time a diff is actually shown —
// not on page load, since most visits never open one. Must wait for the
// element to actually be upgraded first: its register <script type=module>
// loads asynchronously, so on a fresh page load (or a #left..right deep
// link) this can run before `.highlighter`/`.requestUpdate` exist yet.
let rustLoaded = false;
async function ensureRustHighlighting() {
  if (rustLoaded) return;
  const [{ loader }] = await Promise.all([
    import(DATA.prismRustUrl),
    customElements.whenDefined('diff-view-element'),
  ]);
  if (rustLoaded) return;
  rustLoaded = true;
  loader($('diffview').highlighter);
  $('diffview').requestUpdate();
}

function render() {
  // cds-tile's own shadow CSS sets `:host(cds-tile) { display: block }`
  // unconditionally, which an author-origin rule always beats the UA-only
  // `[hidden] { display: none }` default — so toggling `.hidden` on a
  // cds-tile does nothing. Inline `style.display` outranks any stylesheet
  // rule (short of !important), including that one.
  const diffing = R !== '' && R !== L;
  $('archpane').style.display = diffing ? 'none' : '';
  $('diffpane').style.display = diffing ? '' : 'none';

  if (!diffing) {
    $('code').innerHTML = A[L].html.join('\n');
    $('path').textContent = A[L].path;
    $('meta').textContent = A[L].lines + ' lines';
  } else {
    const diffview = $('diffview');
    diffview.oldValue = A[L].raw.join('\n');
    diffview.newValue = A[R].raw.join('\n');
    $('diffhead').textContent = `${A[L].path} → ${A[R].path}`;
    ensureRustHighlighting();
  }

  for (const link of document.querySelectorAll('#model-nav cds-side-nav-link[data-arch]')) {
    link.toggleAttribute('active', link.dataset.arch === L);
  }

  history.replaceState(null, '', diffing ? `#${L}..${R}` : `#${L}`);
}

function pick(side, name) {
  if (side === 'left') {
    if (!A[name]) return;
    L = name;
  } else {
    if (name !== '' && !A[name]) return;
    R = name;
    $('right').value = name;
  }
  render();
}

$('right').addEventListener('cds-select-selected', e => pick('right', e.detail.value));

for (const link of document.querySelectorAll('#model-nav cds-side-nav-link[data-arch]')) {
  link.addEventListener('click', e => {
    e.preventDefault();
    pick('left', link.dataset.arch);
  });
}

// A #left..right fragment makes one specific comparison linkable (and #left
// alone links just that model), and stays live afterward so the side-nav's
// own generated hrefs (and back/forward) keep working.
function applyHash() {
  const [left, right] = decodeURIComponent(location.hash.slice(1)).split('..');
  if (A[left]) pick('left', left);
  pick('right', right && A[right] ? right : '');
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
