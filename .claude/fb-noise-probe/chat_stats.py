#!/usr/bin/env python3
"""The seeded-chat leg's CARD VERDICT (front-b-noise, auditor GO 2026-10-27).

Five parts, read from the leg's own artifacts (a verdict is never computed
over runs that did not happen):
  1. same seed (1234) twice at temp 1.0        -> texts identical.
  2. a different seed (9999)                    -> text differs.
  3. 2-concurrent (1234 + 9999), run TWICE      -> each request's text is
     IDENTICAL across the two pair runs (auditor ruling (b), 2026-10-27:
     the batched-numerics class — the rows=2 rung is a different baked
     tape than the solo one, the same class main's own batched greedy
     shows, and main has no seeded-batched reference — so the pair is
     gated against ITSELF, never equated with solo; the pair-vs-solo
     comparison is reported as information only).
  4. greedy (temp 0)                            -> equals MAIN's greedy text
     (main-ref-leg: main e63c6f8ac8, same prompt, same max-tokens), and the
     in-process `scr chat` greedy equals the server's.
  5. 200 seeded single-token draws at temp 1.0  -> each top-10 token's
     frequency vs softmax(main's host logits), within 3 sigma (binomial).
  5c. the SAME 200 seeds fired 2-CONCURRENT (auditor order 2026-10-27:
     "plus a 2-concurrent P5") -> each concurrent draw equals its own SOLO
     draw's token (the rows=2 decode rung reproduces the solo noise), and
     the same top-10 distribution verdict over the concurrent draws.

Plus the ORDERED LOG EVIDENCE: the server log must SHOW the seat binding the
seeds (SEAT_BIND lines) and the batched rung launching (PREFILL_PATH=batched /
decode-batch lines), including a 2-row launch carrying both seeds.

The phase-B twins (server restarted with --no-prefix-caching, parts 1 and 3
re-run) are DIAGNOSTICS, never gates: they separate a cache-path salt
difference from a seat-law one. Reported beside the gated parts.

Exit 0 iff every gated part passes and the evidence lines exist.
"""
import glob
import json
import math
import pathlib
import re
import sys
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REF = pathlib.Path("/work/fb-chat-main")
SRVLOG = HERE / "chat-server.log"
SRVLOG_B = HERE / "chatb-server.log"
DRAWS = 200
Z_MAX = 3.0
# A `.run` file mixes log lines (a leading ISO timestamp), a `Using model:`
# preamble, and the reply. The auditor's P4 ruling: compare the REPLY, never
# the wrapper — a whole-file compare "diverged" at byte 0 on a timestamp the
# model never wrote.
_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_PREAMBLE = re.compile(r"^Using (cached )?model:")


def content(name: str) -> str:
    r = json.loads((HERE / name).read_text())
    return r["choices"][0]["message"]["content"]


def chat_stdout(name: str) -> str:
    """A `.run` file in the probe dir: log lines stripped, the reply kept."""
    return strip_run(HERE / name)


def chat_stdout_at(dir_: pathlib.Path, name: str) -> str:
    """A `.run` file elsewhere (main's banked reference) — the same strip."""
    return strip_run(dir_ / name)


def strip_run(path: pathlib.Path) -> str:
    lines = path.read_text().splitlines()
    keep = [
        l for l in lines
        if not _LOG_LINE.match(l) and not _PREAMBLE.match(l)
    ]
    while keep and not keep[0].strip():
        keep.pop(0)
    while keep and not keep[-1].strip():
        keep.pop()
    return "\n".join(keep)


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def main() -> int:
    tok_path = sorted(
        glob.glob(
            "/root/.cache/huggingface/hub/models--RedHatAI--granite-3.1-2b-instruct-FP8-dynamic/"
            "snapshots/*/tokenizer.json"
        )
        + glob.glob(
            str(pathlib.Path.home() / ".cache/huggingface/hub/"
                "models--RedHatAI--granite-3.1-2b-instruct-FP8-dynamic/snapshots/*/tokenizer.json")
        )
        + glob.glob(
            "/work/huggingface/hub/"
            "models--RedHatAI--granite-3.1-2b-instruct-FP8-dynamic/snapshots/*/tokenizer.json"
        )
    )
    if not tok_path:
        print("no tokenizer.json in the HF cache — cannot map draw texts to ids")
        return 1
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path[0])

    report: dict = {"probe": "seeded-chat", "prompt": "Write two sentences about the ocean."}
    ok = True

    # ── parts 1-3 ──
    a = content("chat-p1-seed1234-a.json")
    b = content("chat-p1-seed1234-b.json")
    c = content("chat-p2-seed9999.json")
    # The pair, run TWICE (ruling (b)): run a and run b, one file each.
    da = content("chat-p3-seed1234-a.json")
    db = content("chat-p3-seed1234-b.json")
    ea = content("chat-p3-seed9999-a.json")
    eb = content("chat-p3-seed9999-b.json")
    p1 = a == b
    p2 = a != c
    # THE GATE: the pair against ITSELF, deterministic across the two runs.
    p3a = da == db
    p3b = ea == eb
    # INFORMATION ONLY (never gated, ruling (b)): the pair vs the solo texts.
    info1234 = da == a
    info9999 = ea == c
    report["p1_same_seed_identical"] = p1
    report["p2_diff_seed_different"] = p2
    report["p3_concurrent_pair_deterministic"] = {
        "seed1234_run_a_equals_run_b": p3a,
        "seed9999_run_a_equals_run_b": p3b,
        "gate": "the concurrent pair against itself across two runs (ruling "
                "(b), 2026-10-27 — batched numerics, never equated with solo)",
        "informational_vs_solo": {
            "seed1234_pair_equals_solo": info1234,
            "seed9999_pair_equals_solo": info9999,
            "note": "information only — the rows=2 rung is a different baked "
                    "tape than the solo one (the batched-numerics class)",
        },
    }
    report["texts"] = {
        "solo_seed1234_a": a, "solo_seed1234_b": b, "solo_seed9999": c,
        "concurrent_seed1234_run_a": da, "concurrent_seed1234_run_b": db,
        "concurrent_seed9999_run_a": ea, "concurrent_seed9999_run_b": eb,
    }
    if not p1:
        i = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        report["p1_first_divergence"] = {"at_char": i, "a": a[max(0, i - 20):i + 20],
                                         "b": b[max(0, i - 20):i + 20]}
    if not (p3a and p3b):
        for name, x, y in (("seed1234", da, db), ("seed9999", ea, eb)):
            if x != y:
                i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]),
                         min(len(x), len(y)))
                report.setdefault("p3_first_divergence", {})[name] = {
                    "at_char": i, "run_a": x[max(0, i - 20):i + 20],
                    "run_b": y[max(0, i - 20):i + 20]}
    ok = ok and p1 and p2 and p3a and p3b

    # ── part 4: greedy vs main ──
    main_text = chat_stdout_at(REF, "main-greedy.run")
    srv_g = content("chat-p4-greedy.json")
    chat_g = chat_stdout("chat-p4-greedy-chat.run")
    p4_srv = srv_g == main_text
    p4_chat = chat_g == main_text
    p4_self = srv_g == chat_g
    report["p4_greedy"] = {
        "server_equals_main": p4_srv,
        "scr_chat_equals_main": p4_chat,
        "server_equals_scr_chat": p4_self,
    }
    report["texts"]["greedy_server"] = srv_g
    report["texts"]["greedy_scr_chat"] = chat_g
    report["texts"]["greedy_main"] = main_text
    if not (p4_srv and p4_chat):
        for name, t in (("server", srv_g), ("scr_chat", chat_g)):
            i = next((i for i in range(min(len(t), len(main_text)))
                      if t[i] != main_text[i]), min(len(t), len(main_text)))
            report["p4_greedy"][f"{name}_first_divergence"] = {
                "at_char": i, "got": t[max(0, i - 20):i + 20],
                "main": main_text[max(0, i - 20):i + 20]}
    ok = ok and p4_srv and p4_chat and p4_self

    # ── parts 5 + 5c: the distributions (solo, and 2-concurrent) ──
    logits = np.fromfile(REF / "logits-0000.bin", dtype="<f4")
    p = softmax(logits.astype(np.float64))
    top10 = np.argsort(p)[::-1][:10]
    # text -> id over the reference's own top 1000 (draws outside stay counted
    # against the complement, never silently dropped)
    top1000 = np.argsort(p)[::-1][:1000]
    text2id: dict = {}
    ambiguous = []
    for tid in top1000:
        s = tok.decode([int(tid)], skip_special_tokens=True)
        if s in text2id:
            ambiguous.append(s)
        text2id[s] = int(tid)

    def distribution(draws: list) -> dict:
        """The top-10 frequency verdict over one list of draws (solo or
        concurrent — the same reference, the same 3-sigma bar)."""
        counts = Counter(draws)
        rows = []
        for tid in top10:
            s = tok.decode([int(tid)], skip_special_tokens=True)
            obs = counts.get(s, 0) if s not in ambiguous else None
            exp = DRAWS * p[tid]
            sig = math.sqrt(DRAWS * p[tid] * (1 - p[tid]))
            z = (obs - exp) / max(sig, 0.5) if obs is not None else None
            rows.append({"id": int(tid), "text": s, "p": float(p[tid]),
                         "expected": exp, "observed": obs, "sigma": sig, "z": z,
                         "ambiguous_text": s in ambiguous})
        in_top1000 = sum(n for t, n in counts.items() if t in text2id)
        exp_top1000 = DRAWS * float(p[top1000].sum())
        outside = {t: n for t, n in counts.items() if t not in text2id}
        return {
            "top10": rows,
            "pass_within_3sigma": all(
                r["z"] is not None and abs(r["z"]) <= Z_MAX for r in rows
            ),
            "draws_in_ref_top1000": in_top1000,
            "expected_in_ref_top1000": exp_top1000,
            "draws_outside_top1000": {t: n for t, n in list(outside.items())[:10]},
            "n_outside": sum(outside.values()),
        }

    solo_draws = [json.loads(l) for l in (HERE / "chat-p5-draws.txt").read_text().splitlines()]
    assert len(solo_draws) == DRAWS, f"expected {DRAWS} draws, read {len(solo_draws)}"
    report["p5_distribution"] = {
        "draws": DRAWS, "seeds": "100000..100199", "temperature": 1.0,
        "ambiguous_texts": ambiguous,
        **distribution(solo_draws),
    }
    p5 = report["p5_distribution"]["pass_within_3sigma"]
    ok = ok and p5

    # ── part 5c: the same seeds, 2-concurrent — each draw equals its own
    #    solo draw (the rows=2 rung must reproduce the solo noise), and the
    #    same distribution verdict over the concurrent draws. HARD GATES. ──
    conc_draws = [json.loads(l) for l in (HERE / "chat-p5c-draws.txt").read_text().splitlines()]
    assert len(conc_draws) == DRAWS, f"expected {DRAWS} draws, read {len(conc_draws)}"
    mism = [i for i, (s, c) in enumerate(zip(solo_draws, conc_draws)) if s != c]
    p5c = {
        "draws": DRAWS, "temperature": 1.0,
        "pairing": "seeds 100000..100199, concurrent pairs (2k, 2k+1)",
        "each_equals_solo": not mism,
        **distribution(conc_draws),
    }
    if mism:
        i = mism[0]
        p5c["first_mismatch"] = {
            "seed": 100000 + i, "solo": solo_draws[i], "concurrent": conc_draws[i],
        }
    report["p5c_concurrent"] = p5c
    ok = ok and p5c["each_equals_solo"] and p5c["pass_within_3sigma"]

    # ── the ordered log evidence ──
    log = SRVLOG.read_text(errors="replace")
    seat_lines = [l for l in log.splitlines() if "SEAT_BIND" in l]
    both_seeds_one_launch = any("seed=1234" in l and "seed=9999" in l for l in seat_lines)
    ev = {
        "seat_bind_lines": len(seat_lines),
        "seat_bound_seed_1234": any("seed=1234" in l for l in seat_lines),
        "seat_bound_seed_9999": any("seed=9999" in l for l in seat_lines),
        "both_seeds_one_launch": both_seeds_one_launch,
        "prefill_batched_lines": log.count("PREFILL_PATH=batched"),
        "decode_batch_lines": sum(1 for l in log.splitlines()
                                  if "decode batch:" in l and "rung" in l),
        "two_row_rung_line": any("2 live on the 2-row rung" in l for l in log.splitlines()),
        "seat_readback_lines": log.count("SEAT_READBACK"),
        "sample_seat_lines": [l.strip()[:200] for l in seat_lines[:6]],
    }
    report["log_evidence"] = ev
    ok = ok and ev["seat_bind_lines"] > 0 and ev["both_seeds_one_launch"] \
        and ev["prefill_batched_lines"] > 0 and ev["decode_batch_lines"] > 0

    # ── phase-B diagnostics (never gates) ──
    try:
        bb = content("chatb-p1-seed1234-b.json")
        ba = content("chatb-p1-seed1234-a.json")
        bd = content("chatb-p3-seed1234.json")
        be = content("chatb-p3-seed9999.json")
        logb = SRVLOG_B.read_text(errors="replace")
        seat_b = [l for l in logb.splitlines() if "SEAT_BIND" in l]
        report["phaseB_no_prefix_caching"] = {
            "note": "server restarted with --no-prefix-caching; diagnostics, not gates",
            "p1_same_seed_identical": ba == bb,
            "p3_seed1234_pair_vs_solo_info": bd == ba,
            "p3_seed9999_pair_vs_solo_info": be == bb,
            "both_seeds_one_launch": any("seed=1234" in l and "seed=9999" in l
                                         for l in seat_b),
            "seat_bind_lines": len(seat_b),
        }
    except FileNotFoundError:
        report["phaseB_no_prefix_caching"] = "not run"

    (HERE / "chat-verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ok:
        print("CHAT-VERDICT-FAILED (see report above)")
        return 1
    print(f"CHAT-VERDICT-PASS (all five parts + the concurrent P5 + log evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
