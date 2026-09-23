#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""Summarize a bench_startup_compare.sh results directory into markdown.

Kept separate from the orchestrator (rather than embedded as a heredoc like
bench_serve_compare.sh does) so a finished results directory can be
re-summarized without re-running an hour of purge cycles:

    python3 scripts/startup_summary.py bench_results/startup_compare/<dir>

Reporting rules:
  - MEDIAN + p10/p90, never a bare mean. This repo already paid for that
    lesson: crates/cli/scr/src/commands/chat.rs:90 records a mean hiding a
    bimodal ITL distribution for a week.
  - Every cell carries its rep count, so a "median" over 3 reps cannot be
    mistaken for a tight one.
  - Totals use E2E* = ttft_exec + (out-1)*tpot, because mlx-lm ignores
    ignore_eos and under-generates; raw wall-clock totals would reward it for
    doing less work.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

SCENARIO_ORDER = ["frozen", "cold", "warm"]
MODE_ORDER = ["cli", "server"]


def pct(vals: list[float], q: float) -> float:
    """numpy.percentile(method='linear'), matching crates/benches/src/serve.rs."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def med(vals: list[float]) -> float:
    return pct(vals, 0.5)


def fmt_s(v: float | None) -> str:
    return "—" if v is None or v != v else f"{v:.3f}"


def cell(vals: list[float]) -> str:
    """median (p10–p90) ×n — the spread and the rep count travel with it."""
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{med(vals):.3f} ×1"
    return f"**{med(vals):.3f}** ({pct(vals,0.1):.3f}–{pct(vals,0.9):.3f}) ×{len(vals)}"


def load(out_dir: str) -> list[dict]:
    recs = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name.startswith("run_meta"):
            continue
        path = os.path.join(out_dir, name)
        try:
            with open(path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "backend" not in r or r.get("scenario") == "parity":
            continue
        r["_file"] = name
        r["_long"] = ".long." in name
        recs.append(r)
    return recs


def main() -> int:
    if len(sys.argv) < 2:
        return print("usage: startup_summary.py <results-dir>") or 2
    out_dir = sys.argv[1]
    recs = load(out_dir)
    if not recs:
        return print(f"no result JSONs in {out_dir}") or 1

    meta_path = os.path.join(out_dir, "run_meta.txt")
    meta = open(meta_path).read().strip() if os.path.exists(meta_path) else ""

    # Column order follows the run's --backends order (scratchy first, so the
    # ratio column reads "how many times scratchy's number is mlx-lm's"),
    # falling back to alphabetical if the metadata is missing.
    found = {r["backend"] for r in recs}
    declared = []
    for line in meta.splitlines():
        if line.startswith("scenarios:") and "backends:" in line:
            declared = [b.strip() for b in line.split("backends:")[1].split(",")]
    backends = [b for b in declared if b in found] or sorted(found)
    backends += sorted(found - set(backends))

    print("# startup compare — scratchy vs mlx-lm")
    print()
    print(f"results: `{out_dir}`")
    print()
    print("```")
    print(meta)
    print("```")
    print()
    print("`ttft_exec` = seconds from **process exec to the first token byte**, "
          "measured by one external stopwatch (`scripts/startup_probe.py`) "
          "shared by both backends — never self-reported by either framework. "
          "Cells are `median (p10–p90) ×reps`.")
    print()

    # ---- Table 1: the ladder ------------------------------------------------
    print("## exec → first token")
    print()
    hdr = ["scenario", "mode", "prompt"] + backends
    if len(backends) == 2:
        hdr += [f"{backends[1]}/{backends[0]}"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))

    groups: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    ready: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    # Request-level TTFT (clock starts at request send, not at exec) for every
    # scenario. This is the TTFT people mean when they say TTFT, and it is the
    # only piece of ttft_exec that survives once a server is already up -- so
    # it is what makes frozen/cold/warm directly comparable to each other.
    ttft_req: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in recs:
        key = (r["scenario"], r["mode"], "long" if r["_long"] else "short")
        for q in r.get("requests", []):
            if q.get("ttft_from_send_s"):
                ttft_req[key][r["backend"]].append(
                    q["ttft_from_send_s"] * 1000)
        if r["scenario"] == "warm":
            continue
        if r.get("ttft_exec_s"):
            groups[key][r["backend"]].append(r["ttft_exec_s"])
        if r.get("t_ready_s"):
            ready[key][r["backend"]].append(r["t_ready_s"])

    def sort_key(k):
        sc, mo, pl = k
        return (SCENARIO_ORDER.index(sc) if sc in SCENARIO_ORDER else 9,
                MODE_ORDER.index(mo) if mo in MODE_ORDER else 9, pl != "short")

    for key in sorted(groups, key=sort_key):
        sc, mo, pl = key
        row = [sc.upper(), mo, pl]
        for b in backends:
            row.append(cell(groups[key][b]))
        if len(backends) == 2:
            a_, b_ = groups[key][backends[0]], groups[key][backends[1]]
            row.append(f"{med(b_)/med(a_):.2f}×"
                       if a_ and b_ and med(a_) > 0 else "—")
        print("| " + " | ".join(row) + " |")
    print()

    if ready:
        print("### split: exec → server ready (`/v1/models` answers 200)")
        print()
        print("The remainder of `ttft_exec` is the request itself: queue, "
              "prefill, first sample. Poll granularity 20 ms.")
        print()
        h2 = ["scenario", "prompt"] + backends
        print("| " + " | ".join(h2) + " |")
        print("|" + "---|" * len(h2))
        for key in sorted(ready, key=sort_key):
            sc, mo, pl = key
            print("| " + " | ".join([sc.upper(), pl]
                                    + [cell(ready[key][b]) for b in backends]) + " |")
        print()

    # ---- Table 1c: request-level TTFT, all scenarios -----------------------
    if ttft_req:
        print("### TTFT — time to first token, measured from request send (ms)")
        print()
        print("The startup-independent half of `ttft_exec`: the clock starts "
              "when the request is sent to an already-running server. FROZEN "
              "and COLD show it for the very first request a fresh process "
              "ever serves (so it still carries lazy pipeline compilation and "
              "first-touch faults); WARM shows it in steady state. Greedy "
              "(temperature 0) on both backends.")
        print()
        h = ["scenario", "mode", "prompt"] + backends
        if len(backends) == 2:
            h += [f"{backends[1]}/{backends[0]}"]
        print("| " + " | ".join(h) + " |")
        print("|" + "---|" * len(h))
        for key in sorted(ttft_req, key=sort_key):
            sc, mo, pl = key
            row = [sc.upper(), mo, pl]
            for b in backends:
                v = ttft_req[key][b]
                row.append(f"**{med(v):.1f}** ({pct(v,0.1):.1f}–{pct(v,0.9):.1f}) ×{len(v)}"
                           if v else "—")
            if len(backends) == 2:
                a_, b_ = ttft_req[key][backends[0]], ttft_req[key][backends[1]]
                row.append(f"{med(b_)/med(a_):.2f}×"
                           if a_ and b_ and med(a_) else "—")
            print("| " + " | ".join(row) + " |")
        print()

    # ---- Table 2: warm steady state ---------------------------------------
    warm = [r for r in recs if r["scenario"] == "warm"]
    if warm:
        print("## warm: steady-state request latency")
        print()
        print("Process already resident and past its first request. Each "
              "request gets a **unique prompt** so neither scratchy's prefix "
              "cache nor mlx-lm's prompt cache can serve a repeat.")
        print()
        # `scr bench serve` is the authoritative warm client: it is the
        # repo's own load generator, base-URL driven, so scratchy and mlx-lm
        # are driven by identical request logic and identical percentile math
        # (numpy-linear, matched to Python vLLM). The probe's own requests are
        # reported beside it as an independent cross-check -- two clients
        # against one server should agree, and where they do not, something in
        # the request shape differs and needs explaining.
        bs: dict[str, dict] = {}
        for r in warm:
            m = (r.get("bench_serve") or {}).get("metrics")
            if m:
                bs[r["backend"]] = m
        if bs:
            print("Authoritative client: `scr bench serve` "
                  f"(n={next(iter(bs.values())).get('completed','?')} requests, "
                  "concurrency 1, greedy).")
            print()
            hb = ["metric (bench serve)"] + backends
            if len(backends) == 2:
                hb += [f"{backends[1]}/{backends[0]}"]
            print("| " + " | ".join(hb) + " |")
            print("|" + "---|" * len(hb))
            rows = [("median_ttft_ms", "TTFT p50 (ms)", 1.0),
                    ("p99_ttft_ms", "TTFT p99 (ms)", 1.0),
                    ("median_tpot_ms", "TPOT p50 (ms/token)", 1.0),
                    ("median_itl_ms", "ITL p50 (ms)", 1.0),
                    ("output_throughput", "output throughput (tok/s)", 1.0)]
            for k, label, _ in rows:
                if not any(k in bs.get(b, {}) for b in backends):
                    continue
                row = [label]
                for b in backends:
                    v = bs.get(b, {}).get(k)
                    row.append(f"**{v:.1f}**" if isinstance(v, (int, float)) else "—")
                if len(backends) == 2:
                    x = bs.get(backends[0], {}).get(k)
                    y = bs.get(backends[1], {}).get(k)
                    row.append(f"{y/x:.2f}×" if x and y else "—")
                print("| " + " | ".join(row) + " |")
            row = ["generated tokens / request"]
            for b in backends:
                m = bs.get(b, {})
                row.append(f"{m['total_output_tokens']/max(m.get('completed',1),1):.1f}"
                           if m.get("total_output_tokens") else "—")
            if len(backends) == 2:
                row.append("—")
            print("| " + " | ".join(row) + " |")
            print()
            counts = {b: bs[b]["total_output_tokens"] / max(bs[b].get("completed", 1), 1)
                      for b in bs if bs[b].get("total_output_tokens")}
            if len(counts) > 1 and max(counts.values()) - min(counts.values()) > 0.5:
                print("> Generated-token counts differ "
                      f"({ {k: round(v,1) for k,v in counts.items()} }) — mlx-lm "
                      "ignores `ignore_eos`. Per-token metrics above stay "
                      "comparable; totals must use E2E*.")
                print()
            print("Cross-check — the probe's own requests against the same "
                  "server. Two clients, one server: they should agree, and "
                  "where they do not, the request shape differs and needs "
                  "explaining rather than averaging.")
            print()
        h3 = ["metric (probe)"] + backends
        if len(backends) == 2:
            h3 += [f"{backends[1]}/{backends[0]}"]
        print("| " + " | ".join(h3) + " |")
        print("|" + "---|" * len(h3))

        per: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        for r in warm:
            for q in r.get("requests", []):
                if q.get("kind") != "warm":
                    continue
                if q.get("ttft_from_send_s"):
                    per["ttft_ms"][r["backend"]].append(
                        q["ttft_from_send_s"] * 1000)
                if q.get("tpot_ms"):
                    per["tpot_ms"][r["backend"]].append(q["tpot_ms"])
                per["gen_toks"][r["backend"]].append(q.get("output_tokens", 0))
                if q.get("prompt_tokens"):
                    per["prompt_toks"][r["backend"]].append(q["prompt_tokens"])

        labels = [("ttft_ms", "TTFT (ms, from request send)"),
                  ("tpot_ms", "TPOT (ms/token)"),
                  ("gen_toks", "generated tokens / request"),
                  ("prompt_toks", "prompt tokens / request")]
        for k, label in labels:
            if not per[k]:
                continue
            row = [label]
            for b in backends:
                v = per[k][b]
                row.append(f"**{med(v):.1f}** ({pct(v,0.1):.1f}–{pct(v,0.9):.1f}) ×{len(v)}"
                           if v else "—")
            if len(backends) == 2:
                a_, b_ = per[k][backends[0]], per[k][backends[1]]
                row.append(f"{med(b_)/med(a_):.2f}×"
                           if a_ and b_ and med(a_) else "—")
            print("| " + " | ".join(row) + " |")

        # decode tok/s is just 1000/TPOT, but it is the number people quote.
        row = ["decode tok/s (= 1000/TPOT)"]
        for b in backends:
            v = per["tpot_ms"][b]
            row.append(f"**{1000/med(v):.1f}**" if v else "—")
        if len(backends) == 2:
            a_, b_ = per["tpot_ms"][backends[0]], per["tpot_ms"][backends[1]]
            row.append(f"{med(a_)/med(b_):.2f}×" if a_ and b_ else "—")
        print("| " + " | ".join(row) + " |")
        print()

        gen = {b: med(per["gen_toks"][b]) for b in backends
               if per["gen_toks"][b]}
        if len(set(gen.values())) > 1:
            print(f"> Generated-token counts differ ({gen}) — mlx-lm ignores "
                  "`ignore_eos` and stops early. TTFT and TPOT above are "
                  "per-token and stay comparable; any end-to-end total must "
                  "use E2E* below.")
            print()

    # ---- Table 3: E2E* ----------------------------------------------------
    e2e_rows = []
    for key in sorted(groups, key=sort_key):
        sc, mo, pl = key
        tp = {}
        for r in recs:
            if (r["scenario"], r["mode"]) != (sc, mo):
                continue
            if ("long" if r["_long"] else "short") != pl:
                continue
            for q in r.get("requests", []):
                if q.get("tpot_ms"):
                    tp.setdefault(r["backend"], []).append(q["tpot_ms"])
        if not tp:
            continue
        out_len = next((r["output_len"] for r in recs), 32)
        vals = {}
        for b in backends:
            if groups[key][b] and tp.get(b):
                vals[b] = med(groups[key][b]) + (out_len - 1) * med(tp[b]) / 1000
        if len(vals) == len(backends):
            e2e_rows.append((sc, mo, pl, vals))
    if e2e_rows:
        print(f"## E2E* — work-normalized total (ttft_exec + (out−1)·TPOT)")
        print()
        print("The honest total: it charges each backend for the same amount "
              "of generation even when mlx-lm stops early.")
        print()
        h4 = ["scenario", "mode", "prompt"] + [f"{b} (s)" for b in backends]
        if len(backends) == 2:
            h4 += [f"{backends[1]}/{backends[0]}"]
        print("| " + " | ".join(h4) + " |")
        print("|" + "---|" * len(h4))
        for sc, mo, pl, vals in e2e_rows:
            row = [sc.upper(), mo, pl] + [fmt_s(vals[b]) for b in backends]
            if len(backends) == 2:
                row.append(f"{vals[backends[1]]/vals[backends[0]]:.2f}×")
            print("| " + " | ".join(row) + " |")
        print()

    # ---- Table 4: memory + purge evidence ---------------------------------
    print("## memory and page-fault evidence")
    print()
    print("`peak_rss` is the child's `ru_maxrss` — the same call for both "
          "backends. On Apple Silicon unified memory this includes GPU "
          "buffers. `major_faults` (`ru_majflt`) is **the evidence the purge "
          "worked**: FROZEN must fault far more than COLD, or the cache "
          "control failed and the run is void.")
    print()
    h5 = ["scenario", "mode", "backend", "peak_rss (MiB)", "major_faults"]
    print("| " + " | ".join(h5) + " |")
    print("|" + "---|" * len(h5))
    mem: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        if not r["_long"]:
            mem[(r["scenario"], r["mode"], r["backend"])].append(r)
    faults: dict[tuple, float] = {}
    for key in sorted(mem, key=lambda k: (
            SCENARIO_ORDER.index(k[0]) if k[0] in SCENARIO_ORDER else 9,
            MODE_ORDER.index(k[1]) if k[1] in MODE_ORDER else 9, k[2])):
        rs = mem[key]
        rss = [x["peak_rss_bytes"] / 1048576 for x in rs
               if x.get("peak_rss_bytes")]
        flt = [x["major_faults"] for x in rs if x.get("major_faults") is not None]
        faults[key] = med(flt) if flt else float("nan")
        print("| " + " | ".join([key[0].upper(), key[1], key[2],
                                 f"{med(rss):.0f}" if rss else "—",
                                 f"{med(flt):.0f}" if flt else "—"]) + " |")
    print()

    # ---- validity checks ---------------------------------------------------
    print("## validity checks")
    print()
    problems = []
    printed_any_check = False
    for (sc, mo, b) in list(faults):
        if sc != "frozen":
            continue
        cold_key = ("cold", mo, b)
        if cold_key in faults:
            printed_any_check = True
            f_fro, f_col = faults[(sc, mo, b)], faults[cold_key]
            ok = f_fro > f_col
            print(f"- {'PASS' if ok else '**FAIL**'} — {b}/{mo}: FROZEN "
                  f"major faults {f_fro:.0f} vs COLD {f_col:.0f} "
                  f"({'purge evicted the weights' if ok else 'purge did NOT take effect'})")
            if not ok:
                problems.append(f"{b}/{mo} page-cache control")
    for mo in MODE_ORDER:
        for b in backends:
            k_f, k_c = ("frozen", mo, "short"), ("cold", mo, "short")
            if k_f in groups and k_c in groups and groups[k_f][b] and groups[k_c][b]:
                printed_any_check = True
                t_f, t_c = med(groups[k_f][b]), med(groups[k_c][b])
                ok = t_f > t_c
                print(f"- {'PASS' if ok else '**FAIL**'} — {b}/{mo}: FROZEN "
                      f"ttft_exec {t_f:.3f}s > COLD {t_c:.3f}s"
                      f"{'' if ok else '  (a frozen start should never be faster)'}")
                if not ok:
                    problems.append(f"{b}/{mo} frozen<cold")
    if not printed_any_check:
        print("- n/a — these checks compare FROZEN against COLD, and this run "
              "does not contain both. Run `--scenarios frozen,cold` to "
              "exercise them.")
    if problems:
        print()
        print(f"> **{len(problems)} check(s) failed**: {', '.join(problems)}. "
              "Treat the affected rows as invalid.")
    print()

    # ---- disclosures ------------------------------------------------------
    print("## what is not in these numbers")
    print()
    print("- **Weight download.** Out of scope: the HF snapshot is on disk in "
          "all three scenarios, and both run with `HF_HUB_OFFLINE=1`. Left "
          "online, mlx-lm makes a hub round trip on every launch and network "
          "jitter lands inside the measurement.")
    build = next((l.split(":", 1)[1].strip() for l in meta.splitlines()
                  if l.startswith("build_secs:")), None)
    print("- **scratchy's build-time model compilation.** scratchy expands the "
          "whole forward pass at build time (`#[forward]`), so work that MLX "
          "does at runtime is already paid for in the binary. That is a real "
          "engineering advantage and it is also why these startup numbers "
          "flatter scratchy relative to a from-scratch comparison. The "
          "matching build cost is a one-off, not per launch."
          + (f" Recorded for this binary: {build} s." if build else ""))
    print("- **scratchy's aligned-weights sidecar is removed for FROZEN** "
          "(`~/.cache/scratchy/metal-aligned-weights`), but measurement shows "
          "it is neither needed nor rebuilt for this checkpoint: scratchy "
          "writes it only from the realign-copy path, and Llama-3.2-3B-4bit "
          "loads 648/648 tensors zero-copy straight from the HF mmap. So for "
          "this model FROZEN→COLD is a page-cache delta for *both* backends "
          "and the sidecar rung is inert. It is not inert for checkpoints "
          "that need realignment (the code cites Qwen3.5-35B at 18.99 GiB), "
          "so re-check per model instead of assuming.")
    print("- **KV-cache dtype is not matched by default.** On metal scratchy "
          "enables TurboQuant 3-bit KV compression; mlx-lm uses an "
          "uncompressed cache. That affects memory and speed, and it is a "
          "quality difference, not just a perf one. Re-run with "
          "`--kv-cache-dtype fp16` for a like-for-like cache.")
    print("- **A background integrity hash overlaps early decode.** On an "
          "aligned-cache *hit* scratchy content-hashes the cached blob on a "
          "background thread (`metal_allocator.rs:1002`) — 0.39–2.44 s "
          "observed for this model's ~1.7 GiB sidecar, varying with page-cache "
          "state. It competes with the first requests, and it does not happen "
          "at all when no sidecar exists, so it is present in some COLD/WARM "
          "runs and absent from others. WARM samples therefore start only "
          "after a settle delay.")
    print("- **`scr bench startup` measures something different** — an "
          "in-process `LLM` rebuild with no cache wiping and no process exec. "
          "Do not compare its numbers with `ttft_exec`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
