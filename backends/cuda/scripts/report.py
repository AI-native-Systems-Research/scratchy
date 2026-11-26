"""The end-to-end report (/results/ab.md): SHA A against SHA B on one GPU.

usage: SHA_A=… SHA_B=… MODEL=… python3 -I report.py RESULTS BASELINE_COHERENCE OUT.md
  RESULTS       the pod's /results: ab-{a,b}-r{1,2}-{IN}-{OUT}.json (the bench cells, run in the order
                A1, B1, A2, B2), coherence-{a,b}.json (each SHA's 5 prompts), coherence-vllm.json
                (vLLM 0.31.0 in the same pod, with its top-2 logprobs) and gpu-bus.txt;
  BASELINE_COHERENCE  an earlier run's coherence-scr.json for this model, or a missing file.
Per SHA and per run: TTFT and TPOT; the mean of the two runs; the delta B − A; each SHA's drift (run
2 − run 1). Coherence, the D2 rule (cmp.py's `coherence`) against vLLM per SHA, and each SHA's
first differing token index against the baseline's tokens. Exit 1 when a cell or a coherence file
is missing or B's D2 verdict is not 5/5 PASS.
"""
import importlib.util
import json
import os
import sys

res, w1, out = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("cmp", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmp.py"))
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

SHAS = [("a", f"A ({os.environ.get('SHA_A', '?')[:8]})"), ("b", f"B ({os.environ.get('SHA_B', '?')[:8]})")]
CELLS = [(128, 128), (2048, 128)]
L = [f"# {os.environ.get('MODEL', 'the model')}: {SHAS[0][1]} against {SHAS[1][1]} on one GPU", ""]
bad = []
bus = os.path.join(res, "gpu-bus.txt")
if os.path.exists(bus):
    L += [f"GPU: `{open(bus).read().strip()}`; run order A1, B1, A2, B2, back to back.", ""]


def load(name):
    p = os.path.join(res, name)
    if not os.path.exists(p):
        bad.append(f"missing {name}")
        return None
    return json.load(open(p))


for i, o in CELLS:
    L += [f"## {i} in × {o} out, 10 requests after 2 warm-ups", "",
          "| SHA | run | completed | mean TTFT ms | median TTFT | std TTFT | mean TPOT ms | median TPOT | std TPOT |",
          "|---|---|---|---|---|---|---|---|---|"]
    mean, runs = {}, {}
    for side, name in SHAS:
        got = []
        for r in (1, 2):
            d = load(f"ab-{side}-r{r}-{i}-{o}.json")
            runs[(side, r)] = d
            if d is None:
                L.append(f"| {name} | {r} | MISSING | | | | | | |")
                continue
            got.append(d)
            L.append(f"| {name} | {r} | {d['completed']} | {d['mean_ttft_ms']:.2f} | {d['median_ttft_ms']:.2f} "
                     f"| {d['std_ttft_ms']:.2f} | {d['mean_tpot_ms']:.3f} | {d['median_tpot_ms']:.3f} "
                     f"| {d['std_tpot_ms']:.3f} |")
        if got:
            mean[side] = (sum(d["mean_ttft_ms"] for d in got) / len(got), sum(d["mean_tpot_ms"] for d in got) / len(got))
            L.append(f"| **{name}** | mean of {len(got)} | | **{mean[side][0]:.2f}** | | | **{mean[side][1]:.3f}** | | |")
    L.append("")
    if "a" in mean and "b" in mean:
        (ta, pa), (tb, pb) = mean["a"], mean["b"]
        L.append(f"- **delta B − A:** TTFT {tb - ta:+.2f} ms ({100 * (tb - ta) / ta:+.1f}%), "
                 f"TPOT {pb - pa:+.3f} ms ({100 * (pb - pa) / pa:+.1f}%)")
    for side, name in SHAS:
        r1, r2 = runs.get((side, 1)), runs.get((side, 2))
        if r1 and r2:
            L.append(f"- drift, {name} run 2 − run 1: TTFT {r2['mean_ttft_ms'] - r1['mean_ttft_ms']:+.2f} ms, "
                     f"TPOT {r2['mean_tpot_ms'] - r1['mean_tpot_ms']:+.3f} ms")
    L.append("")

vllm = load("coherence-vllm.json")
w1c = json.load(open(w1)) if os.path.exists(w1) else None
if w1c is None:
    L += [f"(no baseline coherence at {w1}: no comparison to its tokens)", ""]
for side, name in SHAS:
    scr = load(f"coherence-{side}.json")
    if scr is None:
        continue
    rows, verdicts = cmp.coherence(scr, vllm)
    passed = sum(v == "PASS" for v in verdicts)
    L += [f"## Coherence, {name} (D2: non-empty, and the first 8 tokens equal vLLM's or vLLM's top-2 gap < 1.0 nat at the first difference): {passed}/{len(verdicts)} PASS", "",
          "| prompt | first differing token vs vLLM | vLLM top-2 gap (nat) | verdict | why | repeat rule | first differing token vs the baseline |",
          "|---|---|---|---|---|---|---|"]
    w1_recs = {r["idx"]: r for r in (w1c or {}).get("records", [])}
    for idx, _, _, first, gap, verdict, why, repeat in rows:
        s_ids = next((r.get("token_ids") or [] for r in scr["records"] if r["idx"] == idx), [])
        w_ids = (w1_recs.get(idx) or {}).get("token_ids")
        if w_ids is None:
            vs_w1 = "-"
        else:
            d = next((k for k in range(min(len(s_ids), len(w_ids))) if s_ids[k] != w_ids[k]), None)
            if d is None and len(s_ids) != len(w_ids):
                d = min(len(s_ids), len(w_ids))
            vs_w1 = "identical" if d is None else str(d)
        L.append(f"| {idx} | {'-' if first is None else first} | {'-' if gap is None else f'{gap:.3f}'} "
                 f"| {verdict} | {why} | {repeat} | {vs_w1} |")
    L.append("")
    if side == "b" and (passed != 5 or len(verdicts) != 5):
        bad.append(f"{name} coherence {passed}/{len(verdicts)} PASS (D2 needs 5/5)")
if bad:
    L += ["## Problems", ""] + [f"- {b}" for b in bad]
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
sys.exit(1 if bad else 0)
