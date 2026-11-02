#!/usr/bin/env python3
"""The 8b fp8 B2-ONLY releg's CARD VERDICT (front-b-noise, auditor order
2026-10-27, on the tree rebased onto 8143cf0ebe): "If B2 matches main's
batched, it's done. If it's still garbage, it's the 8b-specific l_sticks = 2
class, and you localize it with argmax-coder's calibrated instrument."

One gate: the branch's 2-concurrent greedy texts (multiset — HTTP completion
order may swap the two identical-prompt requests) equal MAIN's own
2-concurrent greedy texts (main8-b2-a/b.json — main's batched decode is the
reference for the batched-numerics class, never solo).

Plus the ordered log evidence (no silent rung skips): the branch server log
must SHOW the batched rung launching (PREFILL_PATH=batched / decode-batch
lines), including the 2-row launch — the rung the pre-rebase verdict caught
degenerating. Exit 0 iff the gate passes and the evidence lines exist.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRVLOG = HERE / "eight-b2-server.log"


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
        "probe": "8b-fp8-b2only-releg",
        "model": "RedHatAI/granite-3.1-8b-instruct-FP8-dynamic",
        "prompt": "Write two sentences about the ocean.",
        "note": "rebased onto 8143cf0ebe (the scatter commit); b2 only per "
                "the auditor's order",
    }
    ok = True

    main_b2_a = content("main8-b2-a.json")
    main_b2_b = content("main8-b2-b.json")

    ba = content("eight-b2-a.json")
    bb = content("eight-b2-b.json")
    b2_multiset = sorted([ba, bb]) == sorted([main_b2_a, main_b2_b])
    b2_pairwise = ba == main_b2_a and bb == main_b2_b
    report["b2_concurrent_greedy"] = {
        "gate_multiset_equals_main_batched": b2_multiset,
        "pairwise_by_file": b2_pairwise,
        "note": "main's own 2-concurrent greedy texts are the reference "
                "(batched numerics), never solo",
        "info_branch_pair_texts_identical": ba == bb,
        "info_lengths": {"branch_a": len(ba), "branch_b": len(bb),
                         "main_a": len(main_b2_a), "main_b": len(main_b2_b)},
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
    ev = {
        "prefill_batched_lines": log.count("PREFILL_PATH=batched"),
        "decode_batch_lines": sum(1 for l in lines
                                  if "decode batch:" in l and "rung" in l),
        "two_row_rung_line": any("2 live on the 2-row rung" in l for l in lines),
        "rung_load_lines": sum(1 for l in lines if "rung" in l and
                               ("load" in l or "bake" in l)),
        "refusals_or_panics": [l.strip()[:200] for l in lines
                               if "REFUSING" in l or "panicked" in l][:10],
        "sample_rung_lines": [l.strip()[:200] for l in lines
                              if "rung" in l][:6],
    }
    report["log_evidence"] = ev
    ok = ok and ev["prefill_batched_lines"] > 0 and ev["decode_batch_lines"] > 0 \
        and not ev["refusals_or_panics"]

    report["texts"] = {
        "main_b2_a": main_b2_a, "main_b2_b": main_b2_b,
        "branch_b2_a": ba, "branch_b2_b": bb,
    }

    (HERE / "eight-b2-verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ok:
        print("EIGHT-B2-VERDICT-FAILED (see report above)")
        return 1
    print("EIGHT-B2-VERDICT-PASS (b2 vs main's batched, log evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
