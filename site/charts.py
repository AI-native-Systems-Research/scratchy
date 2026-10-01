#!/usr/bin/env python3
"""SVG chart primitives for the leaderboard page.

Every coordinate on the page is computed here, in Python, at site-build time.
The browser adds only a crosshair/tooltip/keyboard layer (`HOVER_JS`) that reads
pre-computed pixel positions out of an embedded JSON block — it never does scale
math of its own.

That split is deliberate:

  * the page still shows every number with JavaScript off, and a CDN outage
    cannot blank it;
  * "renders honestly when a cell has no data" becomes a build-time property
    that site/tests/ can assert, instead of a JS branch nothing can gate;
  * no new Python dependency — site/ is stdlib-only and CI runs build.sh with
    the runner's system python3, no setup-python step.

Colour lives in styles.css (`--lb-s1`/`--lb-s2`/`--lb-s3` and friends), never
here: marks carry classes, so both Carbon themes work with no branching.
"""

import html
import math

# Plot box padding: left for y tick labels, right for direct end-labels (a
# 2-series chart is direct-labelled as well as legended), top for the tallest
# label's ascender, bottom for the x axis.
PAD = (52, 76, 14, 28)  # left, right, top, bottom


def scale(lo, hi, px0, px1):
    """A linear domain->pixel mapping as a plain 4-tuple.

    A flat series (lo == hi) would divide by zero; widen it instead so the
    single value lands mid-box rather than on an edge.
    """
    if hi < lo:
        lo, hi = hi, lo
    if lo == hi:
        pad = abs(lo) * 0.5 or 1.0
        lo, hi = lo - pad, hi + pad
    return (float(lo), float(hi), float(px0), float(px1))


def project(sc, v):
    """Domain value -> pixel along `sc`."""
    lo, hi, px0, px1 = sc
    return px0 + (float(v) - lo) / (hi - lo) * (px1 - px0)


def nice_ticks(lo, hi, want=5):
    """1/2/5-per-decade ticks spanning [lo, hi], as the axis domain.

    The returned bounds are the *nice* bounds, so callers build their scale from
    `ticks[0]`/`ticks[-1]` rather than the raw data range.

    Every tick is an exact multiple of the step, so a range that straddles zero
    always contains 0.0 — `diverging_bars` reads its zero rule off this list and
    depends on that.
    """
    if hi < lo:
        lo, hi = hi, lo
    if lo == hi:
        if lo == 0:
            return [0.0, 1.0]
        pad = abs(lo) * 0.5
        lo, hi = lo - pad, hi + pad

    raw = (hi - lo) / max(1, want)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10):
        step = mult * mag
        if raw <= step:
            break

    start = math.floor(lo / step) * step
    stop = math.ceil(hi / step) * step
    n = int(round((stop - start) / step))
    ticks = [round(start + i * step, 12) for i in range(n + 1)]
    # Kill -0.0 and float dust so an exact 0.0 is comparable by ==.
    return [0.0 if abs(t) < step * 1e-9 else t for t in ticks]


def fmt(v, unit="", digits=3):
    """A value as published: trimmed fixed digits, thin-space unit."""
    if v is None:
        return "—"
    s = f"{float(v):.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return f"{s} {unit}" if unit else s


def esc(s):
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------- svg scaffold

def svg(w, h, body, title, desc="", cls=""):
    """An <svg> with an accessible name and description.

    `role="img"` plus <title>/<desc> is what makes a chart announce itself; the
    watermark and hatch that mark example data are visual-only, so the same fact
    is repeated in <desc> for a screen reader.
    """
    c = f' class="{esc(cls)}"' if cls else ""
    d = f"<desc>{esc(desc)}</desc>" if desc else ""
    return (
        f'<svg{c} viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet" role="img">'
        f"<title>{esc(title)}</title>{d}{body}</svg>"
    )


def hatch_defs(uid):
    """A diagonal-hatch pattern marking example (non-measured) data.

    Filled marks reference it by id; stroked marks (lines, whiskers) carry a dash
    instead, since a hatch needs an area. Pattern ids are document-scoped, not
    per-<svg>, so they are namespaced by `uid`.
    """
    return (
        "<defs>"
        f'<pattern id="hx{esc(uid)}" patternUnits="userSpaceOnUse" '
        'width="6" height="6" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="6" class="lb-hatch"/>'
        "</pattern></defs>"
    )


def watermark(w, h, text="EXAMPLE DATA"):
    """A watermark inside the viewBox, so a cropped screenshot still carries it.

    A banner at the top of the page does not survive someone screenshotting one
    chart; this does.
    """
    cx, cy = w / 2, h / 2
    return (
        f'<text class="lb-watermark" x="{cx:.1f}" y="{cy:.1f}" '
        f'text-anchor="middle" dominant-baseline="middle" '
        f'transform="rotate(-18 {cx:.1f} {cy:.1f})">{esc(text)}</text>'
    )


# ------------------------------------------------------------------- primitives

def axes(w, h, xs, ys, x_ticks, y_ticks, x_fmt=str, y_fmt=str, pad=PAD):
    """Recessive hairline grid + tick labels.

    Solid, never dashed: a dashed gridline competes with a dashed series line for
    the same meaning.
    """
    left, right, top, bottom = pad
    x0, x1 = left, w - right
    y0, y1 = h - bottom, top
    out = []
    for t in y_ticks:
        y = project(ys, t)
        out.append(f'<line class="lb-grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(
            f'<text class="lb-tick lb-tick-y" x="{x0 - 8}" y="{y:.1f}" '
            f'text-anchor="end" dominant-baseline="middle">{esc(y_fmt(t))}</text>'
        )
    for t in x_ticks:
        x = project(xs, t)
        out.append(
            f'<text class="lb-tick lb-tick-x" x="{x:.1f}" y="{y0 + 18}" '
            f'text-anchor="middle">{esc(x_fmt(t))}</text>'
        )
    out.append(f'<line class="lb-axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    return "".join(out), (x0, x1, y0, y1)


def polyline(xs, ys, points, cls):
    pts = " ".join(f"{project(xs, x):.1f},{project(ys, y):.1f}" for x, y in points)
    return f'<polyline class="lb-line {esc(cls)}" points="{pts}"/>'


def dots(xs, ys, points, cls, r=4):
    """Markers with a surface ring, so an overlap still reads as two marks."""
    return "".join(
        f'<circle class="lb-dot {esc(cls)}" cx="{project(xs, x):.1f}" '
        f'cy="{project(ys, y):.1f}" r="{r}"/>'
        for x, y in points
    )


def end_label(xs, ys, point, text, cls):
    """A direct label at the series end.

    Mandatory here: two series get a legend *and* direct labels, so identity
    never rests on colour alone.
    """
    x, y = point
    return (
        f'<text class="lb-endlabel {esc(cls)}" x="{project(xs, x) + 8:.1f}" '
        f'y="{project(ys, y):.1f}" dominant-baseline="middle">{esc(text)}</text>'
    )


def row_label(x, y, text, sub=None):
    """A right-aligned row label, split over two lines when a sub-label is given.

    SVG text does not wrap, so a long "optimization + model" label would run off
    the left edge of the plot. Two lines halve the width it needs.
    """
    if not sub:
        return (
            f'<text class="lb-rowlabel" x="{x:.1f}" y="{y:.1f}" text-anchor="end" '
            f'dominant-baseline="middle">{esc(text)}</text>'
        )
    return (
        f'<text class="lb-rowlabel lb-rowlabel-main" x="{x:.1f}" y="{y - 6:.1f}" '
        f'text-anchor="end" dominant-baseline="middle">{esc(text)}</text>'
        f'<text class="lb-rowlabel lb-rowlabel-sub" x="{x:.1f}" y="{y + 7:.1f}" '
        f'text-anchor="end" dominant-baseline="middle">{esc(sub)}</text>'
    )


def legend(entries, kind="line"):
    """entries: [(label, css_class)]. Always present for >= 2 series."""
    items = []
    for label, cls in entries:
        key = f'<span class="lb-key lb-key-{esc(kind)} {esc(cls)}" aria-hidden="true"></span>'
        items.append(f'<li class="lb-legend-item">{key}{esc(label)}</li>')
    return f'<ul class="lb-legend">{"".join(items)}</ul>'


def table_view(headers, rows, caption, collapsed=True):
    """A data table. `collapsed` wraps it in <details>.

    Every figure carries one collapsed, so a value is never reachable only
    through a tooltip. Content that is itself the point — provenance — passes
    `collapsed=False`, because a disclosure nobody opens is not a disclosure.
    """
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<th scope="row">{esc(c)}</th>' if i == 0 else f"<td>{esc(c)}</td>"
            for i, c in enumerate(r)
        )
        + "</tr>"
        for r in rows
    )
    table = (
        f'<table class="lb-datatable"><caption>{esc(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )
    if not collapsed:
        return f'<div class="cds--data-table-container lbplain">{table}</div>'
    return (
        '<details class="lb-tableview"><summary>Table view</summary>'
        f"{table}</details>"
    )


# ------------------------------------------------------------------- composites

def line_figure(series, title, desc="", uid="f", w=440, h=240,
                x_label="turn", x_fmt=None, y_fmt=None, example=False):
    """A multi-line chart plus the hover payload the browser needs.

    `series` is [{"label", "cls", "points": [(x, y), ...]}]. Returns
    `(markup, payload)`; `payload["series"][i]["points"]` carries pixel positions
    so HOVER_JS does no scale math.
    """
    x_fmt = x_fmt or (lambda v: f"{v:g}")
    y_fmt = y_fmt or (lambda v: f"{v:g}")
    allx = [x for s in series for x, _ in s["points"]]
    ally = [y for s in series for _, y in s["points"]]
    if not allx:
        return "", {}

    xt = nice_ticks(min(allx), max(allx), want=6)
    # A magnitude axis starts at zero: a truncated y axis exaggerates a ratio,
    # which is the most common way a benchmark chart misleads.
    yt = nice_ticks(min(0.0, min(ally)), max(ally), want=5)
    left, right, top, bottom = PAD
    xs = scale(xt[0], xt[-1], left, w - right)
    ys = scale(yt[0], yt[-1], h - bottom, top)

    body = []
    grid, _box = axes(w, h, xs, ys, xt, yt, x_fmt, y_fmt)
    body.append(grid)

    marks = []
    for s in series:
        pts = sorted(s["points"])
        body.append(polyline(xs, ys, pts, s["cls"]))
        body.append(dots(xs, ys, pts, s["cls"]))
        body.append(end_label(xs, ys, pts[-1], s["label"], s["cls"]))
        marks.append({
            "label": s["label"],
            "cls": s["cls"],
            "points": [
                {"x": round(project(xs, x), 1), "y": round(project(ys, y), 1),
                 "vx": x, "vy": y}
                for x, y in pts
            ],
        })
    if example:
        body.append(watermark(w, h))

    full_desc = desc
    if example:
        full_desc = (desc + " " if desc else "") + EXAMPLE_DESC
    markup = svg(w, h, "".join(body), title, full_desc,
                 cls="lb-svg lb-example" if example else "lb-svg")
    payload = {
        "w": w, "h": h, "xLabel": x_label,
        "box": [left, w - right, h - bottom, top],
        "series": marks,
    }
    return markup, payload


def sparkline(values, w=96, h=24, cls="lb-s1", example=False):
    """A bare trend, no axes — context beside a stat tile, never a chart."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    ys = scale(min(0.0, min(vals)), max(vals), h - 2, 2)
    xs = scale(0, len(vals) - 1, 1, w - 1)
    pts = " ".join(f"{project(xs, i):.1f},{project(ys, v):.1f}" for i, v in enumerate(vals))
    mark = f'<polyline class="lb-line {esc(cls)}" points="{pts}"/>'
    return svg(w, h, mark, f"trend over {len(vals)} points",
               EXAMPLE_DESC if example else "",
               cls="lb-spark lb-example" if example else "lb-spark")


def diverging_bars(rows, title, desc="", uid="d", w=700, bar_h=20, gap=20,
                   unit="%", example=False):
    """Signed deltas against a baseline: two poles, a neutral zero rule.

    `rows` is [{"label", "value", "badge", "ci": (lo, hi) | None}]. A signed
    change is polarity, so this is diverging — never a one-hue bar chart, which
    would make "slower" read as "less faster".
    """
    if not rows:
        return ""
    label_w = 268
    left, right, top, bottom = label_w, 64, 10, 28
    h = top + bottom + len(rows) * (bar_h + gap)
    span = [v for r in rows
            for v in (r["value"], (r.get("ci") or (None, None))[0],
                      (r.get("ci") or (None, None))[1])
            if v is not None]
    ticks = nice_ticks(min(0.0, min(span)), max(0.0, max(span)), want=5)
    xs = scale(ticks[0], ticks[-1], left, w - right)
    zero = project(xs, 0.0)

    body = [hatch_defs(uid)] if example else []
    for t in ticks:
        x = project(xs, t)
        body.append(
            f'<line class="lb-grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}"/>'
        )
        body.append(
            f'<text class="lb-tick lb-tick-x" x="{x:.1f}" y="{h - bottom + 18}" '
            f'text-anchor="middle">{esc(f"{t:g}")}</text>'
        )

    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        v = r["value"]
        # The pole is chosen by sign: for a lower-is-better metric a negative
        # delta is the improvement.
        pole = "lb-pole-fast" if v < 0 else "lb-pole-slow"
        x = project(xs, v)
        bx, bw = min(zero, x), abs(x - zero)
        body.append(
            f'<rect class="lb-bar {pole}" x="{bx:.1f}" y="{y}" '
            f'width="{max(bw, 1):.1f}" height="{bar_h}" rx="4"/>'
        )
        if example:
            # Hatch over the fill, so an example bar is unmistakable even in a
            # screenshot that crops the watermark away.
            body.append(
                f'<rect class="lb-bar-hatch" x="{bx:.1f}" y="{y}" '
                f'width="{max(bw, 1):.1f}" height="{bar_h}" rx="4" '
                f'fill="url(#hx{esc(uid)})"/>'
            )
        ci = r.get("ci")
        if ci and ci[0] is not None and ci[1] is not None:
            a, b = project(xs, ci[0]), project(xs, ci[1])
            cy = y + bar_h / 2
            body.append(
                f'<line class="lb-ci" x1="{a:.1f}" y1="{cy:.1f}" x2="{b:.1f}" y2="{cy:.1f}"/>'
            )
        body.append(row_label(label_w - 12, y + bar_h / 2, r["label"],
                              r.get("sublabel")))
        # The value is always written out: the bar is the comparison, the number
        # is the fact, and a tooltip must never be the only way to read it.
        side = -6 if v < 0 else 6
        anchor = "end" if v < 0 else "start"
        body.append(
            f'<text class="lb-barvalue" x="{x + side:.1f}" y="{y + bar_h / 2:.1f}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">'
            f'{esc(f"{v:+g}{unit}")}</text>'
        )

    body.append(
        f'<line class="lb-zero" x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{h - bottom}"/>'
    )
    if example:
        body.append(watermark(w, h))
    full = desc
    if example:
        full = (desc + " " if desc else "") + EXAMPLE_DESC
    return svg(w, h, "".join(body), title, full,
               cls="lb-svg lb-example" if example else "lb-svg")


def dot_range(rows, title, desc="", uid="r", w=700, row_h=32, unit="",
              digits=2, example=False):
    """Median dot with a p10-p90 whisker — a median of N=5 with its spread.

    Never a bar: a bar implies a total, and its length would claim a precision
    five samples do not have.
    """
    if not rows:
        return ""
    label_w = 268
    left, right, top, bottom = label_w, 78, 10, 28
    h = top + bottom + len(rows) * row_h
    span = [v for r in rows for v in (r["median"], r.get("p10"), r.get("p90"))
            if v is not None]
    ticks = nice_ticks(min(0.0, min(span)), max(span), want=5)
    xs = scale(ticks[0], ticks[-1], left, w - right)

    body = []
    for t in ticks:
        x = project(xs, t)
        body.append(
            f'<line class="lb-grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom}"/>'
        )
        body.append(
            f'<text class="lb-tick lb-tick-x" x="{x:.1f}" y="{h - bottom + 18}" '
            f'text-anchor="middle">{esc(f"{t:g}")}</text>'
        )
    for i, r in enumerate(rows):
        y = top + i * row_h + row_h / 2
        cls = r.get("cls", "lb-s1")
        p10, p90 = r.get("p10"), r.get("p90")
        if p10 is not None and p90 is not None:
            a, b = project(xs, p10), project(xs, p90)
            body.append(
                f'<line class="lb-whisker {esc(cls)}" x1="{a:.1f}" y1="{y:.1f}" '
                f'x2="{b:.1f}" y2="{y:.1f}"/>'
            )
        cx = project(xs, r["median"])
        body.append(f'<circle class="lb-dot {esc(cls)}" cx="{cx:.1f}" cy="{y:.1f}" r="5"/>')
        body.append(row_label(label_w - 12, y, r["label"], r.get("sublabel")))
        body.append(
            f'<text class="lb-barvalue" x="{w - right + 8}" y="{y:.1f}" '
            f'dominant-baseline="middle">{esc(fmt(r["median"], unit, digits))}</text>'
        )
    if example:
        body.append(watermark(w, h))
    full = desc
    if example:
        full = (desc + " " if desc else "") + EXAMPLE_DESC
    return svg(w, h, "".join(body), title, full,
               cls="lb-svg lb-example" if example else "lb-svg")


# The one wording for "this is not a measurement", repeated into every example
# figure's <desc> so the signal is not visual-only. `validate()` greps for it.
EXAMPLE_DESC = (
    "Example data, not a measurement: the shape is illustrative and the values "
    "are not measured."
)


# ------------------------------------------------------------------ hover layer

# Crosshair + tooltip + keyboard readout. Reads only pre-computed pixel
# positions from the page's JSON block: no scales, no domains, no math that could
# disagree with what Python drew. Degrades to nothing without JS, which is why
# every value also appears in a direct label or the table view.
HOVER_JS = r"""
(function () {
  var PAY = JSON.parse(document.getElementById('lbfigures').textContent);
  var tip = document.createElement('div');
  tip.className = 'lbtip';
  tip.setAttribute('role', 'status');
  tip.setAttribute('aria-live', 'polite');
  tip.hidden = true;
  document.body.appendChild(tip);

  function hide() { tip.hidden = true; }

  Object.keys(PAY).forEach(function (id) {
    var fig = document.getElementById(id);
    if (!fig) return;
    var data = PAY[id];
    var svg = fig.querySelector('svg');
    if (!svg || !data.series.length) return;
    var idx = -1;

    // The x positions are identical across series (same turn indices), so one
    // flat list of candidate columns is enough to snap to.
    var cols = data.series[0].points.map(function (p) { return p.x; });

    function clearCross() {
      var old = svg.querySelectorAll('.lb-cross');
      for (var i = 0; i < old.length; i++) { old[i].remove(); }
    }

    function show(i, clientX, clientY) {
      if (i < 0 || i >= cols.length) return;
      idx = i;
      var rows = data.series.map(function (s) {
        var p = s.points[i];
        if (!p) { return ''; }
        return '<span class="lbtip-k ' + s.cls + '"></span>' + s.label +
               ' <b>' + p.vy + '</b>';
      }).filter(Boolean).join('<br>');
      tip.innerHTML = '<div class="lbtip-head">' + data.xLabel + ' ' +
                      data.series[0].points[i].vx + '</div>' + rows;
      tip.hidden = false;
      var box = svg.getBoundingClientRect();
      var px = clientX != null ? clientX : box.left + box.width * (cols[i] / data.w);
      var py = clientY != null ? clientY : box.top + box.height / 2;
      tip.style.left = Math.round(Math.min(px + 14,
          window.innerWidth - tip.offsetWidth - 8)) + 'px';
      tip.style.top = Math.round(py + window.scrollY - 8) + 'px';
      clearCross();
      var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('class', 'lb-cross');
      line.setAttribute('x1', cols[i]); line.setAttribute('x2', cols[i]);
      line.setAttribute('y1', data.box[3]); line.setAttribute('y2', data.box[2]);
      svg.appendChild(line);
    }

    function nearest(ev) {
      var box = svg.getBoundingClientRect();
      var vx = (ev.clientX - box.left) / box.width * data.w;
      var best = 0, bd = Infinity;
      for (var i = 0; i < cols.length; i++) {
        var d = Math.abs(cols[i] - vx);
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    }

    svg.addEventListener('pointermove', function (ev) {
      show(nearest(ev), ev.clientX, ev.clientY);
    });
    svg.addEventListener('pointerleave', function () { hide(); clearCross(); });

    // Keyboard parity: the same readout without a pointer.
    svg.setAttribute('tabindex', '0');
    svg.addEventListener('keydown', function (ev) {
      if (ev.key === 'ArrowRight') { show(Math.min(idx + 1, cols.length - 1)); }
      else if (ev.key === 'ArrowLeft') { show(Math.max(idx - 1, 0)); }
      else if (ev.key === 'Home') { show(0); }
      else if (ev.key === 'End') { show(cols.length - 1); }
      else if (ev.key === 'Escape') { hide(); clearCross(); svg.blur(); return; }
      else { return; }
      ev.preventDefault();
    });
    svg.addEventListener('blur', function () { hide(); clearCross(); });
  });
})();
"""
