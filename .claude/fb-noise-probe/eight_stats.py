#!/usr/bin/env python3
"""The 8b fp8 device-sampling leg's CARD VERDICT (front-b-noise, auditor
order 2026-10-27): every device-sampling claim so far is 2b only; 8b is
where l_sticks = 2 and the IBR wrap differ.

Parts, read from the leg's own artifacts (a verdict is never computed over
runs that did not happen):
  G1. greedy solo (temp 0, 64 tokens) -> equals MAIN's 8b fp8 greedy solo
      text (main8-ref-leg.sh: main e63c6f8ac8, same prompt, same tokens).
  P1. same seed (1234) twice at temp 1.0 -> texts identical.
  P2. a different seed (9999) -> text differs.
  B2. 2-concurrent greedy (temp 0, 64 tokens, same prompt twice) -> both
      texts equal MAIN's own 2-concurrent greedy texts — main's batched
      decode is the reference for the batched-numerics class, never solo.

Plus the ORDERED LOG EVIDENCE (no silent rung skips): the branch server log
must SHOW the seat binding the seeds (SEAT_BIND lines) and the batched rung
launching (PREFILL_PATH=batched / decode-batch lines), including a 2-row
launch — a rung that fails to bake or load is a loud build error upstream,
never a silent skip here.

Exit 0 iff every gated part passes and the evidence lines exist.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRVLOG = HERE / "eight-server.log"


def content(name: str) -> str:
    r = json.loads((HERE / name).read_text())
    return r["choices"][0]["message"]["content"]


def first_div(x: str, y: str) -> dict:
    i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]),
             min(len(x), len(y)))
    return {"at_char": i,
            "x": x[max(0, i - 20):i + 20],
            "y": y[max(0, i - 20):i + 20]}


def main() -> int:
    report: dict = {
        "probe": "8b-fp8-device-sampling",
        "model": "RedHatAI/granite-3.1-8b-instruct-FP8-dynamic",
        "prompt": "Write two sentences about the ocean.",
    }
    ok = True

    main_solo = content("main8-solo-greedy.json")
    main_b2_a = content("main8-b2-a.json")
    main_b2_b = content("main8-b2-b.json")

    # ── G1: greedy solo vs main's 8b greedy solo ──
    g1 = content("eight-g1-greedy.json")
    g1_pass = g1 == main_solo
    report["g1_greedy_solo_equals_main"] = g1_pass
    if not g1_pass:
        report["g1_first_divergence"] = first_div(g1, main_solo)
    ok = ok and g1_pass

    # ── P1 / P2: the seeded draws ──
    a = content("eight-p1-seed1234-a.json")
    b = content("eight-p1-seed1234-b.json")
    c = content("eight-p2-seed9999.json")
    p1 = a == b
    p2 = a != c
    report["p1_same_seed_identical"] = p1
    report["p2_diff_seed_different"] = p2
    if not p1:
        report["p1_first_divergence"] = first_div(a, b)
    if not p2:
        report["p2_note"] = "seed 9999 text equals seed 1234's — the noise died"
    ok = ok and p1 and p2

    # ── B2: 2-concurrent greedy vs MAIN's own 2-concurrent greedy ──
    #    The reference is main's batched texts (multiset — HTTP completion
    #    order may swap the two identical-prompt requests), never solo.
    ba = content("eight-b2-a.json")
    bb = content("eight-b2-b.json")
    b2_multiset = sorted([ba, bb]) == sorted([main_b2_a, main_b2_b])
    b2_pairwise = ba == main_b2_a and bb == main_b2_b
    report["b2_concurrent_greedy"] = {
        "gate_multiset_equals_main_batched": b2_multiset,
        "pairwise_by_file": b2_pairwise,
        "note": "main's own 2-concurrent greedy texts are the reference "
                "(batched numerics), never solo",
        # informational: how batching moved each tree off its own solo
        "info_branch_b2_equals_branch_solo": ba == g1 and bb == g1,
        "info_main_b2_equals_main_solo": main_b2_a == main_solo and main_b2_b == main_solo,
    }
    if not b2_multiset:
        report["b2_first_divergence"] = {
            "branch_a_vs_main_a": first_div(ba, main_b2_a),
            "branch_b_vs_main_b": first_div(bb, main_b2_b),
        }
    ok = ok and b2_multiset

    # ── the ordered log evidence (no silent rung skips) ──
    log = SRVLOG.read_text(errors="replace")
    lines = log.splitlines()
    seat_lines = [l for l in lines if "SEAT_BIND" in l]
    ev = {
        "seat_bind_lines": len(seat_lines),
        "seat_bound_seed_1234": any("seed=1234" in l for l in seat_lines),
        "seat_bound_seed_9999": any("seed=9999" in l for l in seat_lines),
        "prefill_batched_lines": log.count("PREFILL_PATH=batched"),
        "decode_batch_lines": sum(1 for l in lines
                                  if "decode batch:" in l and "rung" in l),
        "two_row_rung_line": any("2 live on the 2-row rung" in l for l in lines),
        "seat_readback_lines": log.count("SEAT_READBACK"),
        "refusals_or_panics": [l.strip()[:200] for l in lines
                               if "REFUSING" in l or "panicked" in l][:10],
        "sample_seat_lines": [l.strip()[:200] for l in seat_lines[:6]],
    }
    report["log_evidence"] = ev
    ok = ok and ev["seat_bind_lines"] > 0 and ev["seat_bound_seed_1234"] \
        and ev["seat_bound_seed_9999"] and ev["prefill_batched_lines"] > 0 \
        and ev["decode_batch_lines"] > 0 and not ev["refusals_or_panics"]

    report["texts"] = {
        "main_greedy_solo": main_solo,
        "main_b2_a": main_b2_a, "main_b2_b": main_b2_b,
        "branch_greedy_solo": g1,
        "branch_b2_a": ba, "branch_b2_b": bb,
        "branch_solo_seed1234_a": a, "branch_solo_seed1234_b": b,
        "branch_solo_seed9999": c,
    }

    (HERE / "eight-verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ok:
        print("EIGHT-VERDICT-FAILED (see report above)")
        return 1
    print("EIGHT-VERDICT-PASS (greedy solo vs main, P1, P2, b2 vs main's "
          "batched, log evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
