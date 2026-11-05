# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""A metal matrix cell's energy: the power log's joules over the wall clock the cell's requests
ran (bench serve's started_unix_s to ended_unix_s), folded into the cell's JSON as `energy`.

    python3 scripts/cell_energy.py CELL_JSON POWER_CSV

The power log is scratchy-power-sampler's: a row a sample, unix_s, interval_s, cpu_j, gpu_j,
ane_j, the joules of the interval that ended at unix_s. A sample the window covers part of counts
by the share it covers. The counters are CPU, GPU and ANE only, not DRAM, and include the Mac's
idle floor; at one sample a second, a cell of under about 20 s is coarse.
"""

import csv
import json
import sys

# How much of a window the log may miss, in seconds, and still be read: a sample's own interval
# runs a little over a second.
SLACK_S = 1.5


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
    its interval inside the window; None where the log does not cover the window."""
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
    if covered < (end - start) - SLACK_S:
        return None
    r = lambda x: round(x, 2)  # noqa: E731
    return {"cpu_j": r(cpu), "gpu_j": r(gpu), "ane_j": r(ane), "soc_j": r(cpu + gpu + ane),
            "window_s": round(end - start, 3), "samples": n}


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


def fold(cell_json, power_csv):
    """Fold a cell's energy into its JSON, where it says when its requests ran and the log
    covers it; else leave it as it is."""
    with open(cell_json) as f:
        cell = json.load(f)
    start, end = cell.get("started_unix_s"), cell.get("ended_unix_s")
    if start is None or end is None:
        return
    e = window(read(power_csv), float(start), float(end))
    if e is None:
        return
    cell["energy"] = per(cell, e)
    with open(cell_json, "w") as f:
        json.dump(cell, f, indent=2)


if __name__ == "__main__":
    fold(sys.argv[1], sys.argv[2])
