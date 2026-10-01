#!/usr/bin/env python3
"""Generate site/_site/leaderboard.html from site/data/leaderboard/.

The Launch Claude Leaderboard: local inference engines compared under a real
coding-agent workload. Called from build.py, which passes in the shared site
header so the chrome matches every other page.

The data files are the contract, and `validate()` is what makes them one. Epic
#158 declares its metrics and fairness rules *before* measuring, so that the
metrics cannot be chosen after seeing which ones flatter us (#161); every rule in
that issue that can be checked mechanically is a check here, and a board that
breaks one does not build. `build()` therefore prints problems and exits non-zero
*before* writing the file, so an invalid board is unpublishable and the Pages
upload step never runs.

Two data shapes, because the declared matrix has to exist before any run does —
that is what lets an incomplete board render without inventing placeholders:

    board.json          the declared universe: lanes, metrics, engines, models,
                        machines, rungs, optimizations, applicability, the
                        cross-product the board is expected to cover, and the
                        disclosures that travel with every number.
    runs/<id>.json      one published run: its provenance block, its
                        measurements, its declared gaps, its ablations.

Standalone: site/build_leaderboard.py [out.html]
"""

import json
import sys
from pathlib import Path

import charts

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "leaderboard"
REPO = "https://github.com/AI-native-Systems-Research/scratchy"

SCHEMA = 1

# Every one of these must be present and non-empty on every numeric cell.
# Acceptance criterion 2 of #158 ("every published row names ...") mechanized:
# a cell whose provenance is incomplete is not publishable, not a footnote.
REQUIRED_PROVENANCE = (
    "machine",
    "rung",
    "scratchy_sha",
    "claude_version",
    "ollama_version",
    "context_length_tokens",
    "kv_cache_dtype",
    "quantization",
    "checkpoint_revision",
    "sampling",
)
# The three that must resolve per engine, so "we matched the KV dtype" is a
# statement about both sides rather than one (#161 fairness rule 4).
PER_ENGINE_PROVENANCE = ("kv_cache_dtype", "quantization", "checkpoint_revision")

# A missing cell is a declared record with a reason, never a blank. These are the
# reasons it may give.
GAP_STATES = {
    "inert": "inert",
    "off-by-design": "off by design",
    "deferred": "deferred",
    "not-run": "not run",
    "blocked": "blocked",
}
# An applicability state is the same vocabulary plus "live".
APPLICABILITY_STATES = {"live": "live", **GAP_STATES}
ATTRIBUTIONS = {"optimization", "architecture"}
RUN_STATUSES = {"published", "example"}
TURN_CLASSES = ("all", "tool-call", "file-write")

# Unknown keys are errors, not no-ops: a typo in a data file must not silently
# do nothing. Mirrors the schema-2 aggregate's filtered-key discipline.
BOARD_KEYS = {
    "schema", "title", "tagline", "epic", "method_issue", "measurement_issue",
    "repo", "unaffiliated", "entry_rule", "workload", "correctness_gate",
    "no_composite_score", "lanes", "metrics", "engines", "models", "machines",
    "rungs", "optimizations", "applicability", "expects", "kpis", "disclosures",
    "status",
}
RUN_KEYS = {"schema", "run_id", "lane", "status", "provenance", "measurements",
            "gaps", "ablations"}
MEAS_KEYS = {"lane", "metric", "engine", "model", "machine", "rung", "facet",
             "n", "stat", "series"}
GAP_KEYS = {"lane", "metric", "engine", "model", "machine", "rung", "state",
            "reason", "issue"}
ABL_KEYS = {"optimization", "model", "metric", "delta_pct", "ci_pct", "n",
            "attribution", "quality_check"}
PROV_KEYS = set(REQUIRED_PROVENANCE) | {
    "trace_corpus", "engine_order", "thermal_state_start", "powermetrics",
    "artifacts", "disclosed",
}

CDN = "https://1.www.s81c.com/common/carbon/web-components/tag/v2/latest"
MODULES = ["ui-shell", "select"]


# ------------------------------------------------------------------------ load

def load(data_dir=None):
    """Read board.json plus every runs/*.json into one dict.

    Indexing happens here so `validate()` and the renderers share one view of
    which cell a record belongs to.
    """
    data_dir = Path(data_dir or DATA)
    board = json.loads((data_dir / "board.json").read_text())
    board["runs"] = [
        json.loads(p.read_text()) for p in sorted((data_dir / "runs").glob("*.json"))
    ]
    index(board)
    return board


def facet_key(rec):
    """A stable key for a record's facet. '' is the headline (unfaceted) cell.

    Faceted records sit *beside* the headline cell rather than replacing it, so
    they do not participate in the coverage cross-product.
    """
    f = rec.get("facet") or {}
    return ",".join(f"{k}={f[k]}" for k in sorted(f))


def cell_key(rec, run):
    prov = rec.get("machine") or run["provenance"].get("machine")
    rung = rec.get("rung") or run["provenance"].get("rung")
    return (rec.get("metric"), rec.get("engine"), rec.get("model"), prov, rung)


def index(board):
    """Attach `_cells`, `_gaps` and `_runs_by_cell` lookups to `board`."""
    cells, gaps = {}, {}
    for run in board["runs"]:
        for m in run.get("measurements", []):
            cells.setdefault((cell_key(m, run), facet_key(m)), []).append((m, run))
        for g in run.get("gaps", []):
            gaps.setdefault(cell_key(g, run), []).append((g, run))
    board["_cells"] = cells
    board["_gaps"] = gaps
    return board


def effective_provenance(run, rec):
    """A record's provenance: the run's block, with any per-cell override."""
    prov = dict(run.get("provenance") or {})
    for k in ("machine", "rung"):
        if rec.get(k):
            prov[k] = rec[k]
    return prov


def get(board, metric, engine, model, machine, rung, facet=""):
    """The one measurement for a cell, or None. Duplicates are a build error, so
    by the time a renderer calls this there is at most one."""
    hit = board["_cells"].get(((metric, engine, model, machine, rung), facet))
    return hit[0] if hit else None


def gap_for(board, metric, engine, model, machine, rung):
    hit = board["_gaps"].get((metric, engine, model, machine, rung))
    return hit[0] if hit else None


def by_id(items, key="id"):
    return {i[key]: i for i in items}


def expect_cells(board):
    """The declared cross-product, in render order."""
    e = board["expects"]
    for model in e["models"]:
        for machine in e["machines"]:
            for rung in e["rungs"]:
                for metric in e["metrics"]:
                    for engine in e["engines"]:
                        yield metric, engine, model, machine, rung


# -------------------------------------------------------------------- validate

def validate(board):
    """Every mechanizable rule from #161 and docs/BENCHMARKING.md, as problems.

    Returns a list of strings, one per problem. Pure: no printing, no exit — the
    caller decides, the same way `build.py`'s `check_markdown_links()` does.
    """
    p = []
    metrics = by_id(board.get("metrics", []))
    engines = by_id(board.get("engines", []))
    models = by_id(board.get("models", []))
    machines = by_id(board.get("machines", []))
    rungs = by_id(board.get("rungs", []))
    opts = by_id(board.get("optimizations", []))
    discl = by_id(board.get("disclosures", []))
    lanes = board.get("lanes", {})

    # 1. shape and referential integrity
    if board.get("schema") != SCHEMA:
        p.append(f"board.json: schema is {board.get('schema')!r}, expected {SCHEMA}")
    for k in set(board) - BOARD_KEYS - {"runs", "_cells", "_gaps"}:
        p.append(f"board.json: unknown key {k!r}")
    for lane, spec in lanes.items():
        for mid in spec.get("owns", []):
            if mid not in metrics:
                p.append(f"board.json: lane {lane} owns undeclared metric {mid!r}")
    # Every declared metric belongs to exactly one lane. That partition is what
    # makes lane purity checkable for a metric the board does not render yet:
    # `metrics` is the full contract from #161, `expects` selects today's subset.
    for mid in metrics:
        owners = [lane for lane, s in lanes.items() if mid in s.get("owns", [])]
        if not owners:
            p.append(f"board.json: metric {mid!r} belongs to no lane")
        elif len(owners) > 1:
            p.append(f"board.json: metric {mid!r} is owned by lanes {owners}")
    for e in board.get("expects", {}).get("metrics", []):
        if e not in metrics:
            p.append(f"board.json: expects an undeclared metric {e!r}")

    # applicability: one declared state per optimization x model
    seen_app = set()
    for a in board.get("applicability", []):
        if a["optimization"] not in opts:
            p.append(f"applicability: undeclared optimization {a['optimization']!r}")
        if a["model"] not in models:
            p.append(f"applicability: undeclared model {a['model']!r}")
        if a.get("state") not in APPLICABILITY_STATES:
            p.append(
                f"applicability {a['optimization']}/{a['model']}: "
                f"unknown state {a.get('state')!r}"
            )
        if not a.get("reason"):
            p.append(f"applicability {a['optimization']}/{a['model']}: no reason given")
        # 6. deferred is not absent: it is named with its evidence.
        if a.get("state") == "deferred" and not a.get("evidence"):
            p.append(
                f"applicability {a['optimization']}/{a['model']}: deferred without "
                "evidence — a deferral is named with its evidence, not left as silence"
            )
        k = (a["optimization"], a["model"])
        if k in seen_app:
            p.append(f"applicability: duplicate entry for {k[0]}/{k[1]}")
        seen_app.add(k)

    for run in board.get("runs", []):
        rid = run.get("run_id", "<no run_id>")
        for k in set(run) - RUN_KEYS:
            p.append(f"{rid}: unknown key {k!r}")
        if run.get("schema") != SCHEMA:
            p.append(f"{rid}: schema is {run.get('schema')!r}, expected {SCHEMA}")
        if run.get("status") not in RUN_STATUSES:
            p.append(f"{rid}: unknown status {run.get('status')!r}")
        if run.get("lane") not in lanes:
            p.append(f"{rid}: unknown lane {run.get('lane')!r}")
            continue
        lane = run["lane"]
        prov = run.get("provenance") or {}
        for k in set(prov) - PROV_KEYS:
            p.append(f"{rid}: unknown provenance key {k!r}")

        for m in run.get("measurements", []):
            tag = f"{rid} {m.get('metric')}/{m.get('engine')}/{m.get('model')}"
            for k in set(m) - MEAS_KEYS:
                p.append(f"{tag}: unknown key {k!r}")
            # 2. lane purity, at load. Enforced again at the render boundary.
            if m.get("lane") != lane:
                p.append(f"{tag}: record lane {m.get('lane')!r} != run lane {lane!r}")
            elif m.get("metric") not in lanes[lane].get("owns", []):
                p.append(
                    f"{tag}: lane {lane} does not own {m.get('metric')!r} — a latency "
                    "claim comes only from Lane A and an outcome claim only from Lane B"
                )
            for field, table in (("engine", engines), ("model", models)):
                if m.get(field) not in table:
                    p.append(f"{tag}: undeclared {field} {m.get(field)!r}")

            # 3. provenance completeness
            ep = effective_provenance(run, m)
            for key in REQUIRED_PROVENANCE:
                if ep.get(key) in (None, "", [], {}):
                    p.append(f"{tag}: provenance is missing {key!r}")
            if ep.get("machine") and ep["machine"] not in machines:
                p.append(f"{tag}: undeclared machine {ep['machine']!r}")
            if ep.get("rung") and ep["rung"] not in rungs:
                p.append(f"{tag}: undeclared rung {ep['rung']!r}")
            for key in PER_ENGINE_PROVENANCE:
                val = ep.get(key)
                if isinstance(val, dict):
                    for eng in board["expects"]["engines"]:
                        if val.get(eng) in (None, ""):
                            p.append(f"{tag}: provenance {key!r} has no entry for {eng!r}")
                elif val is not None:
                    p.append(f"{tag}: provenance {key!r} must be a per-engine object")

            # 4. stat shape
            st = m.get("stat") or {}
            n = m.get("n")
            if st.get("median") is None:
                p.append(f"{tag}: stat has no median")
            if not isinstance(n, int) or n < 1:
                p.append(f"{tag}: n is {n!r}")
            else:
                if n < lanes[lane].get("min_n", 1):
                    p.append(
                        f"{tag}: n={n} is below lane {lane}'s declared minimum "
                        f"{lanes[lane]['min_n']}"
                    )
                if n > 1 and (st.get("p10") is None or st.get("p90") is None):
                    p.append(
                        f"{tag}: n={n} without p10/p90 — a cell is published as "
                        "median (p10–p90) ×reps, never a bare median"
                    )
            if None not in (st.get("p10"), st.get("median"), st.get("p90")):
                if not st["p10"] <= st["median"] <= st["p90"]:
                    p.append(f"{tag}: p10 <= median <= p90 does not hold: {st}")

            ser = m.get("series")
            if ser is not None:
                if not ser.get("points"):
                    p.append(f"{tag}: series present but has no points")
                elif any(len(pt) != 2 for pt in ser["points"]):
                    p.append(f"{tag}: series points must be [x, y] pairs")

        for g in run.get("gaps", []):
            tag = f"{rid} gap {g.get('metric')}/{g.get('engine')}/{g.get('model')}"
            for k in set(g) - GAP_KEYS:
                p.append(f"{tag}: unknown key {k!r}")
            if g.get("state") not in GAP_STATES:
                p.append(f"{tag}: unknown state {g.get('state')!r}")
            if not g.get("reason"):
                p.append(
                    f"{tag}: no reason — n/a is a value carrying a reason, never a blank"
                )
            if g.get("lane") != lane:
                p.append(f"{tag}: record lane {g.get('lane')!r} != run lane {lane!r}")

        for a in run.get("ablations", []):
            tag = f"{rid} ablation {a.get('optimization')}/{a.get('model')}"
            for k in set(a) - ABL_KEYS:
                p.append(f"{tag}: unknown key {k!r}")
            if a.get("optimization") not in opts:
                p.append(f"{tag}: undeclared optimization")
            if a.get("attribution") not in ATTRIBUTIONS:
                p.append(
                    f"{tag}: attribution is {a.get('attribution')!r}, must be one of "
                    f"{sorted(ATTRIBUTIONS)} — each result is attributed either to the "
                    "architecture or to the optimization"
                )
            if not a.get("quality_check"):
                p.append(f"{tag}: no quality_check — a delta ships with its quality check")
            # 6. applicability consistency
            state = next(
                (x.get("state") for x in board.get("applicability", [])
                 if x["optimization"] == a.get("optimization")
                 and x["model"] == a.get("model")),
                None,
            )
            if state is None:
                p.append(f"{tag}: no applicability entry declares this pair")
            elif state != "live":
                p.append(
                    f"{tag}: carries a delta but its declared state is {state!r} — only a "
                    "live optimization is ablated; the grid explains the rest"
                )

        for key in run.get("provenance", {}).get("disclosed", []):
            if key not in discl:
                p.append(f"{rid}: references undeclared disclosure {key!r}")

    # 5. coverage: no silent blanks, no silent duplicates
    for ck in expect_cells(board):
        metric, engine, model, machine, rung = ck
        hits = board["_cells"].get((ck, ""))
        gaps = board["_gaps"].get(ck)
        where = f"{metric}/{engine}/{model}/{machine}/{rung}"
        if hits and len(hits) > 1:
            p.append(
                f"{where}: {len(hits)} measurements for one cell — a board must not "
                "silently pick one"
            )
        if not hits and not gaps:
            p.append(
                f"{where}: uncovered — every declared cell needs a measurement or a "
                "gap with a reason"
            )
        if hits and gaps:
            p.append(f"{where}: has both a measurement and a gap")

    # 7. comparability, and 9. the disclosure that goes with it
    kv_differs = False
    for metric, model, machine, rung in {
        (m, mo, ma, r) for m, _e, mo, ma, r in expect_cells(board)
    }:
        provs = {}
        for engine in board["expects"]["engines"]:
            rec = get(board, metric, engine, model, machine, rung)
            if rec:
                provs[engine] = effective_provenance(rec[1], rec[0])
        if len(provs) < 2:
            continue
        vals = list(provs.values())
        where = f"{metric}/{model}/{machine}/{rung}"
        ctx = {v.get("context_length_tokens") for v in vals}
        if len(ctx) > 1:
            p.append(
                f"{where}: engines disagree on context_length_tokens {sorted(ctx)} — "
                "match it explicitly across engines and prove no truncation"
            )
        if len({v.get("rung") for v in vals}) > 1:
            p.append(f"{where}: engines compared across different rungs")
        # Both engines' records must agree on what the KV dtype *was* on each
        # side; if the two sides then differ from each other, that is an
        # asymmetry the page has to disclose rather than normalise away.
        maps = [v.get("kv_cache_dtype") or {} for v in vals]
        if len({json.dumps(d, sort_keys=True) for d in maps}) > 1:
            p.append(f"{where}: engines report different kv_cache_dtype maps")
        if len({v for v in maps[0].values() if v}) > 1:
            kv_differs = True
    if kv_differs and "kv-dtype-unmatched" not in discl:
        p.append(
            "kv_cache_dtype differs across engines but no 'kv-dtype-unmatched' "
            "disclosure is declared"
        )

    # 8. no cross-rung, cross-lane or cross-machine quoting in a headline ratio.
    # This scans every cell rather than the declared cross-product on purpose: an
    # operand taken from a rung or host the board does not declare is exactly the
    # case the check exists to catch, and iterating `expects` would never see it.
    def kpi_operand(k, engine):
        for (ck, facet), hits in board["_cells"].items():
            metric, eng, model, _machine, _rung = ck
            if (facet == "" and metric == k["metric"] and eng == engine
                    and model == k.get("model")):
                return hits[0]
        return None

    for k in board.get("kpis", []):
        a = kpi_operand(k, k["numerator"])
        b = kpi_operand(k, k["denominator"])
        if not a or not b:
            continue
        pa, pb = effective_provenance(a[1], a[0]), effective_provenance(b[1], b[0])
        for field in ("machine", "rung"):
            if pa.get(field) != pb.get(field):
                p.append(
                    f"kpi {k['id']}: operands differ in {field} "
                    f"({pa.get(field)!r} vs {pb.get(field)!r}) — one engine's number is "
                    "never quoted against the other's from a different rung or host"
                )
        if a[1]["lane"] != b[1]["lane"]:
            p.append(f"kpi {k['id']}: operands come from different lanes")
        if a[1].get("status") != b[1].get("status"):
            p.append(f"kpi {k['id']}: mixes example and published data")

    # 10. example isolation and empty-state honesty
    for ck, hits in board["_cells"].items():
        statuses = {run.get("status") for _rec, run in hits}
        if len(statuses) > 1:
            p.append(
                f"{ck[0]}: both example and published records exist for one cell — "
                "examples are deleted when real data lands, never blended with it"
            )
    measured = sum(
        len(r.get("measurements", [])) for r in board.get("runs", [])
        if r.get("status") == "published"
    )
    if measured == 0 and not (board.get("status", {}).get("pending", {}).get("text")):
        p.append(
            "no published measurements and no status.pending.text — an empty board must "
            "say why it is empty"
        )
    # Keyed on run status, not on how many measurements each run happens to carry:
    # an example run alongside a published one is a mixed board even while empty.
    statuses = {r.get("status") for r in board.get("runs", [])}
    if "example" in statuses and "published" in statuses:
        p.append(
            "example and published runs are both present — delete the example files "
            "when the measurements land"
        )
    return p


# --------------------------------------------------------------------- render

def cell(rec, metric, gap=None):
    """A numeric cell, or the declared reason it is missing.

    `median (p10–p90) ×reps`, the one published cell format
    (docs/BENCHMARKING.md §2). The terminal harness prints the same thing in
    ASCII at crates/benches/src/startup_exec/mod.rs:734-749; the en dash and
    × here are the published spelling, and site/tests pins both.
    """
    if rec is None:
        state = GAP_STATES.get((gap or {}).get("state"), "not measured")
        return (
            f'<span class="lb-na">—</span>'
            f'<span class="cds--tag cds--tag--outline lb-state">{charts.esc(state)}</span>'
        )
    m, n = rec["stat"], rec["n"]
    d = metric.get("digits", 3)
    unit = metric.get("unit", "")
    body = charts.fmt(m["median"], unit, d)
    if n > 1 and m.get("p10") is not None:
        body += (
            f' <span class="lb-spread">({charts.fmt(m["p10"], "", d)}–'
            f'{charts.fmt(m["p90"], "", d)})</span>'
            f' <span class="lb-reps">×{n}</span>'
        )
    else:
        body += f' <span class="lb-reps">×{n}</span>'
    return body


def winner(a, b, lower_is_better):
    """Which cell to bold, or None when the spreads overlap.

    A median gap that sits inside two overlapping p10–p90 ranges is not a
    result, and bolding it would manufacture one.
    """
    if not a or not b:
        return None
    sa, sb = a["stat"], b["stat"]
    if None not in (sa.get("p10"), sa.get("p90"), sb.get("p10"), sb.get("p90")):
        if sa["p10"] <= sb["p90"] and sb["p10"] <= sa["p90"]:
            return None
    if sa["median"] == sb["median"]:
        return None
    a_better = sa["median"] < sb["median"] if lower_is_better else sa["median"] > sb["median"]
    return "a" if a_better else "b"


def suspect_engines(board, model, machine, rung):
    """Engines whose tool-call validity failed on this row.

    Rule 7: a correctness failure outranks a latency win, so those cells are
    de-emphasised and badged. You cannot win a row with a fast wrong answer.
    """
    gate = board.get("correctness_gate")
    out = set()
    if not gate:
        return out
    for engine in board["expects"]["engines"]:
        rec = get(board, gate, engine, model, machine, rung)
        if rec and (rec[0]["stat"].get("median") or 0) > 0:
            out.add(engine)
    return out


def board_table(board, lane, model, machine, rung):
    """One lane's cells for one (model, machine, rung) group.

    Grouping is structural, not a filter: cross-host and cross-rung cells never
    share a ranking (docs/BENCHMARKING.md §6, #161 fairness rule 5).
    """
    metrics = by_id(board["metrics"])
    lane_metrics = [
        m for m in board["expects"]["metrics"]
        if m in board["lanes"][lane].get("owns", [])
    ]
    if not lane_metrics:
        return ""
    engines = [by_id(board["engines"])[e] for e in board["expects"]["engines"]]
    suspects = suspect_engines(board, model, machine, rung)
    gate = board.get("correctness_gate")

    head = "".join(
        f'<th scope="col">{charts.esc(metrics[m]["label"])}'
        + (f'<span class="lb-unit">{charts.esc(metrics[m]["unit"])}</span>'
           if metrics[m].get("unit") else "")
        + "</th>"
        for m in lane_metrics
    )
    rows = []
    for eng in engines:
        tds = []
        for mid in lane_metrics:
            metric = metrics[mid]
            hit = get(board, mid, eng["id"], model, machine, rung)
            rec = hit[0] if hit else None
            ghit = gap_for(board, mid, eng["id"], model, machine, rung)
            gap = ghit[0] if ghit else None
            other = next(e for e in engines if e["id"] != eng["id"])
            rhit = get(board, mid, other["id"], model, machine, rung)
            rival = rhit[0] if rhit else None
            w = winner(rec, rival, metric.get("lower_is_better", True))
            cls = ["lb-cell"]
            if rec is None:
                cls.append("empty")
            elif w == "a":
                cls.append("win")
            # A latency number from an engine that malformed a tool call is not a
            # win, whatever the clock said.
            if eng["id"] in suspects and mid != gate:
                cls.append("suspect")
            attrs = f' class="{" ".join(cls)}"'
            if rec is None and gap:
                attrs += f' data-state="{charts.esc(gap["state"])}"'
                attrs += f' title="{charts.esc(gap["reason"])}"'
            elif rec is not None:
                attrs += f' data-sort="{rec["stat"]["median"]}"'
            body = cell(rec, metric, gap)
            # The turn-class detail sits beside the headline value; the control
            # above the board switches which is shown, in CSS alone.
            variants = []
            for tc in TURN_CLASSES[1:]:
                fhit = get(board, mid, eng["id"], model, machine, rung,
                           facet=f"turn_class={tc}")
                if fhit:
                    variants.append(
                        f'<span class="lbfacet" data-facet="{tc}">'
                        f"{cell(fhit[0], metric)}</span>"
                    )
            if variants:
                body = (
                    f'<span class="lbfacet" data-facet="all">{body}</span>'
                    + "".join(variants)
                )
            tds.append(f"<td{attrs}>{body}</td>")
        badge = ""
        if eng["id"] in suspects:
            badge = (
                '<span class="cds--tag cds--tag--outline lb-suspect-tag" '
                'title="This engine produced malformed tool calls on this row, so its '
                'latency cells are not a win.">tool-call failures</span>'
            )
        rows.append(
            f'<tr><th scope="row"><span class="lb-key lb-key-line lb-s{eng["series"]}" '
            f'aria-hidden="true"></span>{charts.esc(eng["label"])}{badge}</th>'
            + "".join(tds)
            + "</tr>"
        )

    spec = board["lanes"][lane]
    return (
        '<div class="cds--data-table-container lbboard">'
        f'<table class="cds--data-table cds--data-table--sm lb-datatable">'
        f'<caption>Lane {lane} — {charts.esc(spec["name"])}. '
        f'{charts.esc(spec["summary"])}</caption>'
        f'<thead><tr><th scope="col">engine</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def footnotes(board):
    """Every declared gap, numbered, under the board. A dash in a cell always
    has somewhere to point."""
    seen, items = set(), []
    for run in board["runs"]:
        for g in run.get("gaps", []):
            key = (g["metric"], g["state"], g["reason"])
            if key in seen:
                continue
            seen.add(key)
            issue = ""
            if g.get("issue"):
                issue = (
                    f' <a href="{REPO}/issues/{g["issue"]}">#{g["issue"]}</a>'
                )
            items.append(
                f'<li><code>{charts.esc(g["metric"])}</code> '
                f'<span class="cds--tag cds--tag--outline lb-state">'
                f'{charts.esc(GAP_STATES[g["state"]])}</span> '
                f'{charts.esc(g["reason"])}{issue}</li>'
            )
    if not items:
        return ""
    return f'<ol class="lb-notes">{"".join(items)}</ol>'


def the_board(board):
    """Every (model, machine, rung) group, each with its two lane tables."""
    models, machines, rungs = (by_id(board[k]) for k in ("models", "machines", "rungs"))
    out = []
    for model in board["expects"]["models"]:
        for machine in board["expects"]["machines"]:
            for rung in board["expects"]["rungs"]:
                r = rungs[rung]
                out.append(
                    f'<section class="lbgroup"><h3>{charts.esc(models[model]["label"])}'
                    f'<span class="lb-groupmeta">{charts.esc(machines[machine]["label"])}'
                    f' · {charts.esc(rung)} “{charts.esc(r["label"])}”</span></h3>'
                    f'<p class="lb-groupnote">{charts.esc(models[model].get("note", ""))}</p>'
                )
                for lane in ("A", "B"):
                    out.append(board_table(board, lane, model, machine, rung))
                out.append("</section>")
    return "".join(out)


def empty_panel(title, reason, issue=None):
    """What a figure renders instead of itself when it has no data.

    No axes, no frame, no legend: an empty axis frame is exactly what "looks
    broken" means, and a placeholder number would be worse.
    """
    link = ""
    if issue:
        link = f' <a href="{REPO}/issues/{issue}">#{issue}</a>'
    return (
        f'<figure class="lbfig lbfig-empty"><figcaption>{charts.esc(title)}</figcaption>'
        f'<p class="lb-emptynote">{charts.esc(reason)}{link}</p></figure>'
    )


def is_example(board):
    return any(r.get("status") == "example" for r in board["runs"])


def kpi_row(board):
    """Headline ratios as stat tiles.

    Returns "" when nothing is measured: no hero, no tiles, and above all no
    dashes set in display type.
    """
    metrics = by_id(board["metrics"])
    engines = by_id(board["engines"])
    machine = board["expects"]["machines"][0]
    rung = board["expects"]["rungs"][0]
    tiles = []
    for k in board["kpis"]:
        metric = metrics[k["metric"]]
        num = get(board, k["metric"], k["numerator"], k["model"], machine, rung)
        den = get(board, k["metric"], k["denominator"], k["model"], machine, rung)
        if not num or not den:
            continue
        a, b = num[0]["stat"]["median"], den[0]["stat"]["median"]
        if not b:
            continue
        ratio = a / b
        faster = ratio > 1
        verdict = (
            f'{charts.esc(engines[k["denominator"]]["label"])} '
            f"{ratio:.2f}× {'faster' if faster else 'slower'}"
        )
        spark = ""
        if k.get("spark"):
            s = get(board, k["metric"], k["spark"], k["model"], machine, rung)
            if s and s[0].get("series"):
                spark = charts.sparkline(
                    [y for _x, y in s[0]["series"]["points"]],
                    cls=f'lb-s{engines[k["spark"]]["series"]}',
                    example=s[1].get("status") == "example",
                )
        hero = " lb-hero" if k.get("hero") else ""
        # A plain div, not cds-tile: styles.css hides an un-upgraded custom
        # element (the FOUC guard), so a Carbon tile would take the headline
        # numbers with it whenever the component module does not load. These are
        # bordered boxes; they do not need a component.
        tiles.append(
            f'<div class="lbkpi{hero}">'
            f'<p class="lbkpi-label">{charts.esc(k["label"])}</p>'
            f'<p class="lbkpi-value{hero}">{ratio:.2f}×</p>'
            f'<p class="lbkpi-sub">{verdict}</p>'
            f'<p class="lbkpi-detail">'
            f'{charts.esc(engines[k["denominator"]]["label"])} '
            f'{charts.fmt(b, metric.get("unit", ""), metric.get("digits", 3))}'
            f' vs {charts.esc(engines[k["numerator"]]["label"])} '
            f'{charts.fmt(a, metric.get("unit", ""), metric.get("digits", 3))}</p>'
            f"{spark}</div>"
        )
    if not tiles:
        return ""
    return f'<div class="lbkpis">{"".join(tiles)}</div>'


def ttft_facets(board):
    """Per-turn TTFT, one small multiple per model, two series each.

    The shape *is* the finding: a flat line means the prefix was reused, a rising
    one means every turn repays the whole prompt.
    """
    models, engines = by_id(board["models"]), by_id(board["engines"])
    machine = board["expects"]["machines"][0]
    rung = board["expects"]["rungs"][0]
    panels, payload = [], {}
    for i, model in enumerate(board["expects"]["models"]):
        series, rows, example = [], [], False
        for eid in board["expects"]["engines"]:
            rec = get(board, "ttft_ms", eid, model, machine, rung)
            if not rec or not rec[0].get("series"):
                continue
            example = example or rec[1].get("status") == "example"
            pts = [(x, y) for x, y in rec[0]["series"]["points"]]
            series.append({
                "label": engines[eid]["label"],
                "cls": f'lb-s{engines[eid]["series"]}',
                "points": pts,
            })
            rows.append([engines[eid]["label"]] + [charts.fmt(y, "", 1) for _x, y in pts])
        if not series:
            panels.append(empty_panel(
                models[model]["label"],
                "Per-turn TTFT appears here once the replay lane has run.",
                board.get("measurement_issue"),
            ))
            continue
        fid = f"fig-ttft-{i}"
        markup, pay = charts.line_figure(
            series,
            title=f'TTFT per turn — {models[model]["label"]}',
            desc="Time to first token against turn index, for each engine.",
            uid=f"t{i}", x_label="turn", example=example,
        )
        payload[fid] = pay
        turns = [x for x, _y in series[0]["points"]]
        table = charts.table_view(
            ["engine"] + [f"turn {t}" for t in turns], rows,
            ("Example data, not measurements. " if example else "")
            + f'TTFT in ms per turn, {models[model]["label"]}.',
        )
        panels.append(
            f'<figure class="lbfig" id="{fid}">'
            f'<figcaption>{charts.esc(models[model]["label"])}'
            f'<span class="lb-figmeta">{charts.esc(models[model]["family"])}'
            f' · TTFT in ms, lower is better</span>'
            f"</figcaption>{markup}"
            + charts.legend([(s["label"], s["cls"]) for s in series])
            + table
            + "</figure>"
        )
    return f'<div class="lbfacets">{"".join(panels)}</div>', payload


def ablation_chart(board):
    """Signed ablation deltas, scratchy only, with each row's attribution."""
    models, opts = by_id(board["models"]), by_id(board["optimizations"])
    rows, example, tv = [], False, []
    for run in board["runs"]:
        for a in run.get("ablations", []):
            example = example or run.get("status") == "example"
            opt = opts[a["optimization"]]["label"]
            model = models[a["model"]]["label"]
            ci = tuple(a["ci_pct"]) if a.get("ci_pct") else None
            rows.append({"label": opt, "sublabel": model,
                         "value": a["delta_pct"],
                         "badge": a["attribution"], "ci": ci})
            tv.append([
                f"{opt} · {model}", f'{a["delta_pct"]:+g}%',
                f'{ci[0]:+g}% to {ci[1]:+g}%' if ci else "—",
                a["attribution"], str(a["n"]), a["quality_check"],
            ])
    if not rows:
        return empty_panel(
            "Ablation deltas",
            "Each optimization's measured delta appears here, including any that "
            "measured about zero.",
            board.get("measurement_issue"),
        )
    svg = charts.diverging_bars(
        rows,
        title="Ablation deltas on TTFT, scratchy only",
        desc="Percent change in TTFT when each optimization is switched off, "
             "against the shipping default. Negative means the optimization helps.",
        uid="abl", example=example,
    )
    return (
        '<figure class="lbfig lbfig-wide" id="fig-ablations">'
        '<figcaption>Ablation deltas on TTFT'
        '<span class="lb-figmeta">scratchy only · negative means the optimization '
        'helps</span></figcaption>'
        f"{svg}"
        + charts.legend([("helps (faster)", "lb-pole-fast"),
                         ("hurts (slower)", "lb-pole-slow")], kind="rect")
        + charts.table_view(
            ["optimization · model", "delta", "95% CI", "attribution", "N",
             "quality check"], tv,
            ("Example data, not measurements. " if example else "")
            + "Ablation deltas with their attribution and quality check.")
        + "</figure>"
    )


def status_grid(board):
    """Applicability: which optimization is even live on which model.

    This is the panel that makes an about-zero legible. The same optimization is
    fully live on one model and structurally absent on another, so a near-zero
    delta is attributed here rather than read as a failure (#158 AC 4).
    """
    models, opts = by_id(board["models"]), by_id(board["optimizations"])
    app = {(a["optimization"], a["model"]): a for a in board["applicability"]}
    head = "".join(
        f'<th scope="col">{charts.esc(models[m]["label"])}</th>'
        for m in board["expects"]["models"]
    )
    rows = []
    for oid in [o["id"] for o in board["optimizations"]]:
        tds = []
        for m in board["expects"]["models"]:
            a = app.get((oid, m))
            if not a:
                tds.append('<td class="lb-state-cell empty">—</td>')
                continue
            state = a["state"]
            # Icon plus word plus colour: state is never carried by colour alone.
            icon = {"live": "●", "inert": "○", "off-by-design": "⊘",
                    "deferred": "◐", "not-run": "○",
                    "blocked": "✕"}.get(state, "○")
            issue = ""
            if a.get("issue"):
                issue = f' <a href="{REPO}/issues/{a["issue"]}">#{a["issue"]}</a>'
            tds.append(
                f'<td class="lb-state-cell" data-state="{charts.esc(state)}">'
                f'<span class="lb-state-icon" aria-hidden="true">{icon}</span>'
                f'<span class="lb-state-word">'
                f'{charts.esc(APPLICABILITY_STATES[state])}</span>'
                f'<span class="lb-state-why">{charts.esc(a["reason"])}'
                f'{issue}</span></td>'
            )
        rows.append(
            f'<tr><th scope="row">{charts.esc(opts[oid]["label"])}'
            f'<code class="lb-switch">{charts.esc(opts[oid]["switch"])}</code></th>'
            + "".join(tds) + "</tr>"
        )
    return (
        '<div class="cds--data-table-container lbstatus">'
        '<table class="cds--data-table cds--data-table--sm lb-datatable">'
        '<caption>Whether each optimization is even live on each model. An '
        'optimization that is inert or off by design cannot show a delta, and a '
        'near-zero there is a fact about the architecture, not about the '
        'optimization.</caption>'
        f'<thead><tr><th scope="col">optimization</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def lane_b_dots(board):
    """Lane B outcomes as a median dot with its p10-p90 whisker."""
    models, engines = by_id(board["models"]), by_id(board["engines"])
    machine = board["expects"]["machines"][0]
    rung = board["expects"]["rungs"][0]
    rows, tv, example = [], [], False
    for model in board["expects"]["models"]:
        for eid in board["expects"]["engines"]:
            rec = get(board, "task_pass", eid, model, machine, rung)
            if not rec:
                continue
            example = example or rec[1].get("status") == "example"
            st = rec[0]["stat"]
            rows.append({
                "label": models[model]["label"],
                "sublabel": engines[eid]["label"],
                "median": st["median"], "p10": st.get("p10"), "p90": st.get("p90"),
                "cls": f'lb-s{engines[eid]["series"]}',
            })
            tv.append([
                f'{models[model]["label"]} · {engines[eid]["label"]}',
                charts.fmt(st["median"], "", 2),
                charts.fmt(st.get("p10"), "", 2),
                charts.fmt(st.get("p90"), "", 2),
                str(rec[0]["n"]),
            ])
    if not rows:
        return empty_panel(
            "Task pass rate",
            "Graded task outcomes appear here once the live lane has run.",
            board.get("measurement_issue"),
        )
    svg = charts.dot_range(
        rows, title="Task pass rate by model and engine",
        desc="Median of N graded runs, with the p10 to p90 spread.",
        uid="lb", example=example,
    )
    return (
        '<figure class="lbfig lbfig-wide" id="fig-laneb">'
        '<figcaption>Task pass rate'
        '<span class="lb-figmeta">median of N with p10–p90 spread</span>'
        "</figcaption>"
        f"{svg}"
        + charts.legend([(engines[e]["label"], f'lb-s{engines[e]["series"]}')
                         for e in board["expects"]["engines"]], kind="dot")
        + charts.table_view(["model · engine", "median", "p10", "p90", "N"], tv,
                            ("Example data, not measurements. " if example else "")
                            + "Graded task pass rate.")
        + "</figure>"
    )


def disclosures_html(board):
    items = "".join(
        f'<li><strong>{charts.esc(d["id"])}</strong> — {charts.esc(d["text"])}'
        f' <span class="lb-src">{charts.esc(d.get("source", ""))}</span></li>'
        for d in board["disclosures"]
    )
    return f'<ul class="lb-disclosures">{items}</ul>'


def provenance_table(board):
    """What every number on this page was measured with."""
    rows = []
    for run in board["runs"]:
        p = run["provenance"]
        per = lambda key: ", ".join(  # noqa: E731 - a local formatter, not a policy
            f"{k}: {v}" for k, v in sorted((p.get(key) or {}).items())
        )
        rows.append([
            run["run_id"], run["lane"], run.get("status", ""),
            p.get("machine", ""), p.get("rung", ""),
            p.get("scratchy_sha", ""), p.get("claude_version", ""),
            p.get("ollama_version", ""), str(p.get("context_length_tokens", "")),
            per("kv_cache_dtype"), per("quantization"),
            p.get("thermal_state_start", ""),
        ])
    return charts.table_view(
        ["run", "lane", "status", "machine", "rung", "scratchy SHA", "claude",
         "ollama", "context", "KV dtype", "quantization", "thermal at start"],
        rows,
        "Provenance for every run on this page. A cell with no provenance does "
        "not build.",
        # Shown, not collapsed: a disclosure nobody opens is not a disclosure.
        collapsed=False,
    )


def turnclass_control(board):
    opts = "".join(
        f'<cds-select-item value="{tc}"{" selected" if tc == "all" else ""}>'
        f'{tc}</cds-select-item>'
        for tc in TURN_CLASSES
    )
    return (
        '<div class="lbcontrols"><cds-select id="turnclass" label-text="Turn class" '
        f'value="all">{opts}</cds-select>'
        '<p class="lb-controlnote">A ~30-token tool call is prefill-bound and a long '
        'file write is decode-bound. They have opposite profiles, so a blended '
        'number hides both.</p></div>'
    )


def page(header, board):
    problems = validate(board)
    if problems:
        return None, problems, {}

    example = is_example(board)
    facets, payload = ttft_facets(board)
    counts = {
        "rows": len(list(expect_cells(board))),
        "measured": sum(
            1 for ck in expect_cells(board) if board["_cells"].get((ck, ""))
        ),
        "gaps": sum(1 for ck in expect_cells(board) if board["_gaps"].get(ck)),
    }

    banner = ""
    if example:
        banner = (
            '<div class="lbbanner" role="note"><strong>Example data.</strong> '
            + charts.esc(board["status"]["pending"]["text"])
            + f' <a href="{REPO}/issues/{board["status"]["pending"]["issue"]}">'
            f'#{board["status"]["pending"]["issue"]}</a> Every figure below is '
            'hatched and watermarked for the same reason: no value on this page is '
            'a measurement yet.</div>'
        )

    out = PAGE
    for key, val in {
        "{modules}": "\n".join(
            f'<script type="module" src="{CDN}/{m}.min.js"></script>' for m in MODULES
        ),
        "{header}": header,
        "{title}": charts.esc(board["title"]),
        "{tagline}": charts.esc(board["tagline"]),
        "{banner}": banner,
        "{entry_rule}": charts.esc(board["entry_rule"]),
        "{unaffiliated}": charts.esc(board["unaffiliated"]),
        "{workload_why}": charts.esc(board["workload"]["why"]),
        "{no_composite}": charts.esc(board["no_composite_score"]),
        "{disclosures}": disclosures_html(board),
        "{kpis}": kpi_row(board),
        "{controls}": turnclass_control(board),
        "{board}": the_board(board),
        "{footnotes}": footnotes(board),
        "{facets}": facets,
        "{ablations}": ablation_chart(board),
        "{status_grid}": status_grid(board),
        "{laneb}": lane_b_dots(board),
        "{provenance}": provenance_table(board),
        "{figures}": json.dumps(payload, separators=(",", ":")),
        "{hover_js}": charts.HOVER_JS,
        "{counts}": (
            f'{counts["rows"]} declared cells · {counts["measured"]} with a '
            f'number · {counts["gaps"]} declared gaps'
        ),
        "{epic}": str(board["epic"]),
        "{method_issue}": str(board["method_issue"]),
        "{repo}": REPO,
    }.items():
        out = out.replace(key, val)
    return out, [], counts


def build(out_path, header):
    board = load()
    out, problems, counts = page(header, board)
    if problems:
        for p in problems:
            print(f"leaderboard: {p}", file=sys.stderr)
        print(
            f"leaderboard: {len(problems)} problem(s); refusing to write "
            f"{out_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    Path(out_path).write_text(out)
    print(
        f"leaderboard page: {counts['rows']} declared cells, {counts['measured']} "
        f"measured, {counts['gaps']} gaps -> {out_path}"
    )


def main():
    sys.path.insert(0, str(HERE))
    from build import header_html  # the one definition of the site chrome

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_site/leaderboard.html"
    build(out, header_html("", active="leaderboard.html"))


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — scratchy</title>
<meta name="description" content="Local inference engines compared under a real coding-agent workload: per-turn latency, prefix reuse, task outcomes and tool-call validity.">
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
<body data-turnclass="all">

{header}

<div class="docs-layout">
<main class="docs-content lbpage">

  <div class="hero-body">
    <h1>{title}</h1>
    <p class="lede">{tagline}</p>
  </div>

  {banner}

  <section class="lbsection">
    <h2>What is on the board</h2>
    <p><strong>Entry rule.</strong> {entry_rule}</p>
    <p><strong>Why this workload.</strong> {workload_why}</p>
    <p><strong>No overall rank.</strong> {no_composite}</p>
    <p class="lb-fineprint">{unaffiliated}</p>
  </section>

  <!-- The disclosures come before the headline, not after it: the fairness
       rules are the product here, and a reader meets them first. -->
  <section class="lbsection">
    <h2>Fairness disclosures</h2>
    <p>These travel with every number on this page. The method is fixed in
    <a href="{repo}/issues/{method_issue}">#{method_issue}</a> before any
    measurement, so the metrics cannot be chosen after seeing which ones
    flatter us.</p>
    {disclosures}
  </section>

  <section class="lbsection">
    <h2>Headline</h2>
    {kpis}
  </section>

  <section class="lbsection">
    <h2>The board</h2>
    <p>Ranked within a metric, inside one model, machine and rung. A cell is
    <code>median (p10–p90) ×reps</code>; bold marks a win only where the
    spreads do not overlap. A dash is a declared gap with a reason, never a
    zero.</p>
    {controls}
    {board}
    {footnotes}
  </section>

  <section class="lbsection">
    <h2>TTFT per turn</h2>
    <p>The shape is the finding. A flat line means the prefix was reused; a
    rising one means every turn repays the whole prompt.</p>
    {facets}
  </section>

  <section class="lbsection">
    <h2>What each optimization bought</h2>
    {ablations}
    {status_grid}
  </section>

  <section class="lbsection">
    <h2>Did it finish the job</h2>
    <p>Latency claims come from the replay lane; outcome claims come from the
    live lane. The two are never quoted as each other.</p>
    {laneb}
  </section>

  <section class="lbsection">
    <h2>Provenance</h2>
    {provenance}
    <p class="lb-count">{counts}</p>
    <p class="lb-fineprint">Method and rung definitions:
    <a href="{repo}/blob/main/docs/BENCHMARKING.md">docs/BENCHMARKING.md</a>.
    Epic: <a href="{repo}/issues/{epic}">#{epic}</a>.</p>
  </section>

</main>
</div>

<script type="application/json" id="lbfigures">{figures}</script>
<script>
// The turn-class control only flips an attribute on <body>; which cells that
// shows is decided in CSS, so the default view is correct before this runs and
// remains correct if it never does.
(function () {
  var sel = document.getElementById('turnclass');
  if (!sel) return;
  sel.addEventListener('cds-select-selected', function (e) {
    document.body.setAttribute('data-turnclass', e.detail.value);
  });
})();
{hover_js}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
