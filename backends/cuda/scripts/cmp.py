"""The report of one model's pod against vLLM in the same pod: /results/compare.md.

usage: python3 -I cmp.py TAG RESULTS S0DIR OUT
       python3 -I cmp.py --chat TAG RESULTS OUT   (coherence only: coherence-{scr,vllm}-chat.json)
  RESULTS: the pod's /results (scr-*.json, vllm-*.json, coherence-{scr,vllm}.json, the key lines,
           the GPU memory files, ldd.txt, readelf.txt, gpu-hygiene.txt)
  S0DIR:   an earlier vLLM run's results for this model (vllm-*.json), or a missing dir

The coherence verdict (the D2 rule), per prompt: scratchy's reply is non-empty, and
either its first 8 token ids equal vLLM's, or at the first differing index vLLM's own top-2 logprob
gap is below 1.0 nat. The repeat rule (no token id more than 8 times in a row) is reported, and
applied only where vLLM's reply passes it. §3.4's "obviously wrong" checks print as WARN lines, in
the report and on stdout. Exit 0 always: the report is the output, the WARN lines are for a person.
"""
import json
import os
import re
import sys

CELLS = [(128, 128), (512, 128), (2048, 128), (128, 512)]
POOL_TOKENS = 129 * 64
# §3.2-3.3: the expected scratchy TPOT ranges (ms), TTFT at 128 and 2048 (ms), and GPU after load (GB).
EXPECT = {
    "27b": {"tpot": {128: (20, 25), 512: (20, 25.5), 2048: (21.5, 27), "128x512": (20, 25)},
            "floor_ms": 9.0, "after_load_gb": 35, "kv_per_token": 32768},
    "35b": {"tpot": {128: (8.5, 9.3), 512: (8.6, 9.5), 2048: (9.3, 10.2), "128x512": (8.5, 9.4)},
            "floor_ms": 1.5, "after_load_gb": 41, "kv_per_token": 10240},
    "gemma": {"tpot": {128: (12.0, 13.3), 512: (12.4, 13.8), 2048: (14.0, 15.5), "128x512": (12.2, 13.5)},
              "floor_ms": 1.5, "after_load_gb": 33, "kv_per_token": 112640},
}
# §3.3: the device weight bytes the header sums give (GB, e4m3 / bf16 / total).
HEADER_GB = {"27b": (24.35, 5.10, 29.5), "35b": (32.61, 4.16, 36.8), "gemma": (24.48, 3.01, 27.5)}

warns = []


def warn(msg):
    warns.append(msg)
    print("WARN " + msg)


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def text(path):
    try:
        return open(path).read()
    except OSError:
        return ""


def mib(path):
    m = re.findall(r"(\d+)\s*MiB", text(path))
    return [int(x) for x in m]


def cell_key(i, o):
    return "128x512" if (i, o) == (128, 512) else i


def ratio(a, b):
    return f"{a / b:.2f}" if a and b else "-"


def coherence(scr, vllm):
    rows, verdicts = [], []
    s_recs = {r["idx"]: r for r in (scr or {}).get("records", [])}
    v_recs = {r["idx"]: r for r in (vllm or {}).get("records", [])}
    for i in sorted(set(s_recs) | set(v_recs)):
        s, v = s_recs.get(i, {}), v_recs.get(i, {})
        st, vt = s.get("token_ids") or [], v.get("token_ids") or []
        if "error" in s or not s.get("text"):
            verdict, why = "FAIL", "scratchy reply empty or failed: " + s.get("error", "empty")
            first, gap = None, None
        elif not vt:
            verdict, why, first, gap = "NO-REF", "no vLLM reply to compare", None, None
        else:
            first = next((k for k in range(min(len(st), len(vt))) if st[k] != vt[k]), None)
            if first is None and len(st) != len(vt):
                first = min(len(st), len(vt))
            gap = None
            if first is not None and first < len(v.get("top2", [])):
                lps = sorted(v["top2"][first].values(), reverse=True)
                gap = lps[0] - lps[1] if len(lps) >= 2 else None
            if first is None or first >= 8:
                verdict, why = "PASS", "first 8 tokens equal vLLM's"
            elif gap is not None and gap < 1.0:
                verdict, why = "PASS", f"diverges at {first}, vLLM's top-2 gap {gap:.3f} < 1.0 nat"
            else:
                verdict, why = "FAIL", f"diverges at {first}, vLLM's top-2 gap {gap if gap is None else round(gap, 3)}"
        rep_s, rep_v = max_run(st), max_run(vt)
        repeat = "-"
        if vt and rep_v <= 8:
            repeat = "PASS" if rep_s <= 8 else f"FAIL (a run of {rep_s})"
        elif vt:
            repeat = f"n/a (vLLM's own run {rep_v})"
        if verdict == "FAIL" and first == 0:
            warn(f"coherence prompt {i}: first divergence at index 0, not excused by vLLM's top-2 gap")
        if verdict == "FAIL" and not s.get("text"):
            warn(f"coherence prompt {i}: empty scratchy reply")
        verdicts.append(verdict)
        rows.append((i, s.get("text", ""), v.get("text", ""), first, gap, verdict, why, repeat))
    return rows, verdicts


def max_run(ids):
    best, run = 0, 0
    for k, t in enumerate(ids):
        run = run + 1 if k and ids[k - 1] == t else 1
        best = max(best, run)
    return best


def chat_report(tag, res, out):
    """The chat-templated coherence pod: the D2 table, each server's rendered prompt ids side by
    side (the same ids rule out the template before a divergence is blamed on the forward), and a
    plain verdict line for a token-0 divergence against a large vLLM gap."""
    scr, vllm = load(f"{res}/coherence-scr-chat.json"), load(f"{res}/coherence-vllm-chat.json")
    L = [f"# {tag}: chat-templated coherence, scratchy against vLLM 0.31.0 in the same pod\n"]
    L.append("Each of the 5 prompts as one user message to `/v1/chat/completions`, `max_tokens 32`, "
             "temperature 0, `chat_template_kwargs {\"enable_thinking\": false}`. vLLM with "
             "`logprobs: true, top_logprobs: 2`; scratchy without logprobs, its ids from `/tokenize` "
             "of the reply text. The rule is D2.\n")
    s_recs = {r["idx"]: r for r in (scr or {}).get("records", [])}
    v_recs = {r["idx"]: r for r in (vllm or {}).get("records", [])}
    L.append("## Rendered prompts\n")
    L.append("| prompt | scratchy prompt ids | vLLM prompt ids | same |")
    L.append("|---|---|---|---|")
    for i in sorted(set(s_recs) | set(v_recs)):
        sp, vp = s_recs.get(i, {}).get("prompt_ids"), v_recs.get(i, {}).get("prompt_ids")
        same = "yes" if sp is not None and sp == vp else "NO"
        if same == "NO":
            warn(f"chat prompt {i}: the two servers render different prompt ids "
                 f"({len(sp or [])} vs {len(vp or [])}): compare the templates before the forward")
        L.append(f"| {i} | {len(sp) if sp else '-'} ids | {len(vp) if vp else '-'} ids | {same} |")
    L.append("\n## Coherence (D2)\n")
    rows, verdicts = coherence(scr, vllm)
    L.append("| prompt | first differing token | vLLM top-2 gap (nat) | verdict | why | repeat rule |")
    L.append("|---|---|---|---|---|---|")
    forward_bug = []
    for i, st, vt, first, gap, verdict, why, repeat in rows:
        L.append(f"| {i} | {'-' if first is None else first} | {'-' if gap is None else f'{gap:.3f}'} | {verdict} | {why} | {repeat} |")
        same_prompt = s_recs.get(i, {}).get("prompt_ids") == v_recs.get(i, {}).get("prompt_ids")
        if verdict == "FAIL" and first == 0 and gap is not None and gap >= 1.0 and same_prompt:
            forward_bug.append(i)
    for i, st, vt, *_ in rows:
        L.append(f"\n**Prompt {i}**\n- scratchy: `{st!r}`\n- vLLM: `{vt!r}`")
    L.append(f"\nCoherence (chat): {verdicts.count('PASS')}/{len(verdicts)} PASS.")
    if forward_bug:
        msg = (f"prompts {forward_bug}: the same rendered prompt, and scratchy's first token differs "
               "from vLLM's where vLLM's top-2 gap is at least 1 nat: a Gemma forward bug, to "
               "diagnose before S8")
        warn(msg)
        L.append(f"\n**VERDICT: {msg}.**")
    L.append("\n## WARN\n")
    L.extend([f"- WARN {w}" for w in warns] or ["- none"])
    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out}: {len(warns)} WARN line(s)")


def main():
    if sys.argv[1] == "--chat":
        return chat_report(*sys.argv[2:5])
    tag, res, s0, out = sys.argv[1:5]
    exp = EXPECT.get(tag, {})
    L = [f"# {tag}: scratchy against vLLM 0.31.0 in the same pod\n"]
    L.append("One request at a time (`--max-concurrency 1`, batch size 1), `--ignore-eos`, temperature 0, "
             "`scr bench serve --dataset-name random`, 10 requests after 2 warm-ups. "
             "Both servers: `--max-num-seqs 1 --dtype bfloat16 --kv-cache-dtype fp8`; vLLM also `--max-model-len 8192`.\n")

    # ── cells ──
    L.append("## Cells\n")
    L.append("| input × output | scratchy TTFT mean / median (ms) | vLLM TTFT | vLLM S0 TTFT | scratchy TPOT (ms) | vLLM TPOT | vLLM S0 TPOT | scratchy tok/s | vLLM tok/s | vLLM S0 tok/s | tok/s scratchy / vLLM |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tpots = {}
    for i, o in CELLS:
        s = load(f"{res}/scr-{i}-{o}.json") or {}
        v = load(f"{res}/vllm-{i}-{o}.json") or {}
        z = load(f"{s0}/vllm-{i}-{o}.json") or {}
        g = lambda d, k: d.get(k)
        f = lambda x: "-" if x is None else f"{x:.1f}"
        L.append(f"| {i} × {o} | {f(g(s,'mean_ttft_ms'))} / {f(g(s,'median_ttft_ms'))} | {f(g(v,'mean_ttft_ms'))} | "
                 f"{f(g(z,'mean_ttft_ms'))} | {f(g(s,'mean_tpot_ms'))} | {f(g(v,'mean_tpot_ms'))} | {f(g(z,'mean_tpot_ms'))} | "
                 f"{f(g(s,'output_throughput'))} | {f(g(v,'output_throughput'))} | {f(g(z,'output_throughput'))} | "
                 f"{ratio(g(s,'output_throughput'), g(v,'output_throughput'))} |")
        tp = g(s, "mean_tpot_ms")
        tpots[(i, o)] = tp
        if not s:
            warn(f"cell {i}x{o}: no scratchy result")
            continue
        if s.get("completed") != 10 or s.get("failed"):
            warn(f"cell {i}x{o}: scratchy completed {s.get('completed')} failed {s.get('failed')}")
        if s.get("total_output_tokens") != 10 * o:
            warn(f"cell {i}x{o}: scratchy output tokens {s.get('total_output_tokens')} != {10 * o} (--ignore-eos?)")
        if tp is not None and tp < exp.get("floor_ms", 0):
            warn(f"cell {i}x{o}: TPOT {tp:.2f} ms below the weight-stream floor {exp['floor_ms']} ms: not every weight read?")
        if v.get("output_throughput") and s.get("output_throughput", 0) > 1.10 * v["output_throughput"]:
            warn(f"cell {i}x{o}: scratchy {s['output_throughput']:.1f} tok/s beats vLLM {v['output_throughput']:.1f} by >10%: check the served model, OUT and --ignore-eos")
        rng = exp.get("tpot", {}).get(cell_key(i, o))
        if tp is not None and rng and not (rng[0] * 0.75 <= tp <= rng[1] * 1.25):
            warn(f"cell {i}x{o}: TPOT {tp:.2f} ms far outside the expected {rng[0]}-{rng[1]} ms (±25%)")
    t128, t2048 = tpots.get((128, 128)), tpots.get((2048, 128))
    rng = exp.get("tpot", {})
    if t128 and t2048 and 128 in rng and 2048 in rng:
        want = (rng[2048][0] + rng[2048][1]) / 2 - (rng[128][0] + rng[128][1]) / 2
        if t2048 - t128 > 2.5 * max(want, 0.1):
            warn(f"TPOT grows {t2048 - t128:.2f} ms from 128 to 2048, over 2.5x the expected {want:.2f}: the fp8-cache decode slowdown is worse than modelled")

    # ── memory ──
    L.append("\n## Memory\n")
    keys = text(f"{res}/scr-key-lines.txt")
    after = mib(f"{res}/scr-gpu-after-load.txt")
    during = mib(f"{res}/scr-gpu-during-cells.txt")
    vafter = mib(f"{res}/vllm-gpu-after-load.txt")
    L.append(f"- scratchy GPU memory after load: {after[0] if after else '-'} MiB; peak during the cells: {max(during) if during else '-'} MiB.")
    L.append(f"- vLLM after load: {vafter[0] if vafter else '-'} MiB, **not comparable**: vLLM preallocates its KV cache to `--gpu-memory-utilization 0.9`.")
    m = re.search(r"weight bytes on the device (\d+) \(e4m3 (\d+), f32 scales (\d+), bf16 (\d+), GDN f32 (\d+)", keys)
    hg = HEADER_GB.get(tag)
    if m:
        tot, e4, sc, bf, gd = (int(x) for x in m.groups())
        L.append(f"- weights on the device: total {tot/1e9:.2f} GB = e4m3 {e4/1e9:.2f} + f32 scales {sc/1e9:.3f} + bf16 {bf/1e9:.2f} + GDN f32 {gd/1e9:.3f} GB, alignment {(tot-e4-sc-bf-gd)/1e6:.1f} MB.")
        if hg:
            L.append(f"  - the header sums (§3.3): e4m3 {hg[0]} GB, bf16 {hg[1]} GB, total ≈ {hg[2]} GB.")
            if abs(tot / 1e9 - hg[2]) > 3:
                warn(f"weights on the device {tot/1e9:.2f} GB, more than 3 GB off the header sums' {hg[2]} GB: wrong staging?")
    else:
        warn("no weight-bytes line in the serve log")
    kvm = re.search(r"CUDA worker: (\d+) KV blocks \((\d+) bytes\)", keys)
    if kvm:
        blocks, kvb = int(kvm.group(1)), int(kvm.group(2))
        per = kvb / (blocks * 64) if blocks else 0
        L.append(f"- KV pool: {blocks} blocks, {kvb} bytes ({kvb/2**20:.1f} MiB), {per:.0f} bytes a token (e4m3; expected {exp.get('kv_per_token', '-')}).")
        if exp.get("kv_per_token") and round(per) != exp["kv_per_token"]:
            warn(f"KV bytes a token {per:.0f} != the expected {exp['kv_per_token']}")
    else:
        warn("no 'CUDA worker: N KV blocks' line in the serve log")
    if after and exp.get("after_load_gb") and abs(after[0] * 2**20 / 1e9 - exp["after_load_gb"]) > 3:
        warn(f"GPU after load {after[0]} MiB, more than 3 GB off the expected ≈{exp['after_load_gb']} GB")
    L.append("\n```\n" + keys.strip() + "\n```")

    # ── coherence ──
    L.append("\n## Coherence (D2: non-empty, and the first 8 tokens equal vLLM's or vLLM's top-2 gap < 1.0 nat at the first difference)\n")
    rows, verdicts = coherence(load(f"{res}/coherence-scr.json"), load(f"{res}/coherence-vllm.json"))
    L.append("| prompt | first differing token | vLLM top-2 gap (nat) | verdict | why | repeat rule |")
    L.append("|---|---|---|---|---|---|")
    for i, st, vt, first, gap, verdict, why, repeat in rows:
        L.append(f"| {i} | {'-' if first is None else first} | {'-' if gap is None else f'{gap:.3f}'} | {verdict} | {why} | {repeat} |")
    for i, st, vt, *_ in rows:
        L.append(f"\n**Prompt {i}**\n- scratchy: `{st!r}`\n- vLLM: `{vt!r}`")
    L.append(f"\nCoherence: {verdicts.count('PASS')}/{len(verdicts)} PASS.")

    # ── linkage and hygiene ──
    L.append("\n## Linkage and GPU hygiene\n")
    L.append("```\n" + text(f"{res}/ldd.txt").strip() + "\n" +
             "\n".join(l for l in text(f"{res}/readelf.txt").splitlines() if "NEEDED" in l) + "\n```")
    if re.search(r"cublas|cudnn", text(f"{res}/ldd.txt") + text(f"{res}/readelf.txt"), re.I):
        warn("the scr binary links cuBLAS or cuDNN")
    L.append("```\n" + text(f"{res}/gpu-hygiene.txt").strip() + "\n```")
    L.append("\n## WARN\n")
    L.extend([f"- WARN {w}" for w in warns] or ["- none"])
    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out}: {len(warns)} WARN line(s)")


if __name__ == "__main__":
    main()
