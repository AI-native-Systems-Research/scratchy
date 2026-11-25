# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""A metal matrix cell's energy: the power log's joules over the wall clock the cell's requests
ran (bench serve's started_unix_s to ended_unix_s), folded into the cell's JSON as `energy`.

    python3 scripts/cell_energy.py CELL_JSON POWER_CSV

The power log is scratchy-power-sampler's: a row a sample, unix_s, interval_s, cpu_j, gpu_j,
ane_j, the joules of the interval that ended at unix_s. A sample the window covers part of counts
by the share it covers. The counters are CPU, GPU and ANE only, not DRAM, and include the Mac's
idle floor. The matrix samples every 100 ms: fine enough to split a request's energy at its first
token where requests run one at a time ([`per_phase`]); a 10 ms sample often reads zero on an idle
GPU (a counter step is about 2 mJ), 100 ms never does.
"""

import csv
import json
import sys

# How much of a window the log may miss and still be read: the larger of a fixed slack, a
# sample's own interval running a little over a second, and a share of the window, so a cell of
# hours keeps its energy over a gap of seconds in the log.
SLACK_S = 1.5
SLACK_SHARE = 0.001


def read(power_csv):
    """The power log's rows, as floats, skipping any short or unreadable line."""
    out = []
    with open(power_csv) as f:
        for r in csv.reader(f):
            if len(r) != 5:
                continue
            try:
                out.append([float(x) for x in r])
            except ValueError:
                continue
    return out


def window(rows, start, end):
    """The CPU, GPU and ANE joules over start to end (Unix seconds), each sample by the share of
    its interval inside the window; None where the log misses more of it than the slack allows
    (SLACK_S, SLACK_SHARE). Seconds it misses within that count at the window's mean power, and
    the result says how many (uncovered_s)."""
    if end <= start:
        return None
    cpu = gpu = ane = covered = 0.0
    n = 0
    for t, dt, c, g, a in rows:
        if dt <= 0:
            continue
        lo, hi = max(t - dt, start), min(t, end)
        if hi <= lo:
            continue
        share = (hi - lo) / dt
        cpu, gpu, ane = cpu + c * share, gpu + g * share, ane + a * share
        covered += hi - lo
        n += 1
    length = end - start
    uncovered = max(length - covered, 0.0)
    if covered <= 0 or uncovered > max(SLACK_S, SLACK_SHARE * length):
        return None
    scale = length / covered if uncovered > SLACK_S else 1.0
    r = lambda x: round(x * scale, 2)  # noqa: E731
    return {"cpu_j": r(cpu), "gpu_j": r(gpu), "ane_j": r(ane), "soc_j": r(cpu + gpu + ane),
            "window_s": round(length, 3), "samples": n, "uncovered_s": round(uncovered, 3)}


def per(cell, e):
    """`e` with the cell's energy per request sent, per output token, its mean power, and for a
    scored cell per right answer (None where none was right)."""
    out = dict(e)
    sent = (cell.get("completed") or 0) + (cell.get("failed") or 0)
    tokens = cell.get("total_output_tokens") or 0
    soc = e["soc_j"]
    out["j_per_request"] = round(soc / sent, 3) if sent else None
    out["j_per_output_token"] = round(soc / tokens, 4) if tokens else None
    out["soc_avg_w"] = round(soc / e["window_s"], 2) if e["window_s"] else None
    if cell.get("accuracy") is not None:
        right = cell["accuracy"] * (cell.get("num_tasks") or sent)
        out["j_per_right_answer"] = round(soc / right, 2) if right > 0.5 else None
    return out


def shares(rows, start, end, spans):
    """Each span's joules: each power sample's joules inside start to end shared among the spans
    open over its interval, by the time each was open in it, so the spans' joules sum to the
    window's busy joules. A span is (from, to) in Unix seconds, or None (no joules); None for all
    where the log covers none of them. The one rule every per-request figure shares."""
    joules = [0.0 if span else None for span in spans]
    met = False
    for t, dt, c, g, a in rows:
        if dt <= 0:
            continue
        lo, hi = max(t - dt, start), min(t, end)
        if hi <= lo:
            continue
        flight = [(k, min(hi, b) - max(lo, a0)) for k, span in enumerate(spans) if span
                  for a0, b in [span]]
        flight = [(k, f) for k, f in flight if f > 0]
        busy = sum(f for _, f in flight)
        if busy <= 0:
            continue
        inside = (c + g + a) * (hi - lo) / dt
        for k, f in flight:
            joules[k] += inside * f / busy
            met = True
    return [round(j, 3) if j is not None else None for j in joules] if met else None


def request_spans(cell):
    """Each request's (sent, first token, last token) in Unix seconds, in the order sent: the
    cell's started_unix_s plus its start_times entry, then its ttfts_ms and e2els_ms after it. None
    for a request that failed; None for a cell that kept no request times."""
    start = cell.get("started_unix_s")
    starts, e2els = cell.get("start_times"), cell.get("e2els_ms")
    if start is None or not starts or not e2els or len(starts) != len(e2els):
        return None
    ttfts = cell.get("ttfts_ms") or [None] * len(starts)
    if len(ttfts) != len(starts):
        return None
    return [(start + s, start + s + f / 1000.0 if f is not None else None, start + s + d / 1000.0)
            if s is not None and d is not None else None
            for s, f, d in zip(starts, ttfts, e2els)]


def per_request(cell, rows):
    """Each request's joules, in the order sent (`task_joules`): each power sample's joules inside
    the cell's window shared among the requests in flight over its interval, by the time each was
    in flight in it ([`shares`]), so the requests' joules sum to the cell's. A request runs from
    its send to its last token. None for a request that failed; None for a cell that kept no
    request times, or none of whose requests the log covers."""
    start, end, spans = cell.get("started_unix_s"), cell.get("ended_unix_s"), request_spans(cell)
    if start is None or end is None or spans is None:
        return None
    return shares(rows, start, end, [(span[0], span[2]) if span else None for span in spans])


def per_phase(cell, rows):
    """Each request's prefill and decode joules, in the order sent (`task_prefill_joules`,
    `task_decode_joules`): its joules up to its first token, and after it ([`shares`] over the two
    phases as spans). A MEASUREMENT only where no two requests were in flight at once — then every
    joule a sample holds is one request's one phase. Where requests overlap, a step's energy is its
    batch's whole (one request's prompt tokens beside another's decode tokens), which no counter
    splits: None, never a share by tokens or time. None for a request that failed or has no first
    token; None for a cell that kept no request times, or that the log does not cover."""
    start, end, spans = cell.get("started_unix_s"), cell.get("ended_unix_s"), request_spans(cell)
    if start is None or end is None or spans is None:
        return None
    open_ = sorted((s, e) for span in spans if span for s, _, e in [span])
    if any(b[0] < a[1] for a, b in zip(open_, open_[1:])):
        return None
    phases = []
    for span in spans:
        s, f, e = span or (None, None, None)
        ok = span is not None and f is not None and s <= f <= e
        phases += [(s, f), (f, e)] if ok else [None, None]
    j = shares(rows, start, end, phases)
    if j is None:
        return None
    return j[0::2], j[1::2]


def fold(cell_json, power_csv):
    """Fold a cell's energy into its JSON, where it says when its requests ran and the log
    covers it; else leave it as it is."""
    with open(cell_json) as f:
        cell = json.load(f)
    start, end = cell.get("started_unix_s"), cell.get("ended_unix_s")
    if start is None or end is None:
        return
    log = read(power_csv)
    e = window(log, float(start), float(end))
    if e is None:
        return
    cell["energy"] = per(cell, e)
    joules = per_request(cell, log)
    if joules is not None:
        cell["task_joules"] = joules
    if (phases := per_phase(cell, log)) is not None:
        cell["task_prefill_joules"], cell["task_decode_joules"] = phases
    with open(cell_json, "w") as f:
        json.dump(cell, f, indent=2)


if __name__ == "__main__":
    fold(sys.argv[1], sys.argv[2])
