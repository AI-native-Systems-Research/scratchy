#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""Blocking parity gate for the scratchy-vs-mlx-lm startup benchmark.

If the two backends are not actually running the same model correctly, every
timing number is meaningless -- a broken dequant path can be fast. So this
runs before any measurement and refuses to proceed on a real mismatch.

WHAT IT DOES AND DOES NOT ASSERT
--------------------------------
It asserts **exact agreement on short, high-confidence prompts** (a factual
answer, a fixed format, a short list). A wrong quant preset, a mis-sized
group, a bad dequant or a swapped tensor garbles these immediately.

It does NOT assert bit-exact logits, and it deliberately does not fail on
divergence deep inside a long, open-ended generation. Greedy decoding follows
argmax, so wherever the top two tokens are nearly tied, two different int4
kernel stacks legitimately pick differently ("The *sentence*..." vs "The
*phrase*...") and every token after that is conditioned on a different
history. Measured on this pair: short confident prompts agree exactly, long
open-ended ones diverge at the first near-tie. Treating that as a failure
would block on ordinary floating-point noise; ignoring the confident prompts
would let a genuinely broken load path through. Hence: enforce the former,
record the latter as a diagnostic.

For actual numerical fidelity work, this repo already has golden-reference
tooling (scripts/generate_mlx_goldens.py, tools/vision_parity/) -- that is a
different job from gating a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "startup_probe.py")

# Deterministic answers or rigid formats: the model is confident, so a
# disagreement here means a real defect, not a coin flip on tied logits.
ENFORCED = [
    ("colors", "Name three primary colors.", 40),
    ("capital", "What is the capital of France? Answer in one word.", 12),
    ("count", "Count from 1 to 10, separated by commas.", 40),
    ("arith", "What is 17 plus 25? Reply with just the number.", 12),
]
# Open-ended: recorded, never enforced.
DIAGNOSTIC = [
    ("openended", "Explain what a hash table is, in two sentences.", 60),
]


def clean(stdout: str, backend: str) -> str:
    if backend == "mlx-lm":
        parts = stdout.split("=" * 10)
        stdout = parts[1] if len(parts) > 1 else stdout
    else:
        stdout = "\n".join(l for l in stdout.splitlines()
                           if not l.startswith("Using model:"))
    return re.sub(r"\s+", " ", stdout).strip()


def run(a, backend: str, name: str, prompt: str, ntok: int) -> str:
    out = os.path.join(a.out_dir, f"{name}.{backend}.json")
    cmd = [sys.executable, PROBE, "--backend", backend, "--mode", "cli",
           "--model", a.model, "--out", out,
           "--log", os.path.join(a.out_dir, f"{name}.{backend}.log"),
           "--prompt", prompt, "--output-len", str(ntok),
           "--scenario", "parity", "--device", a.device]
    if backend == "scratchy":
        cmd += ["--scr-bin", a.scr_bin]
    else:
        cmd += ["--mlx-python", a.mlx_python]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
    if not os.path.exists(out):
        return ""
    with open(out) as f:
        return clean(json.load(f).get("stdout", ""), backend)


def agree_chars(x: str, y: str) -> int:
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--scr-bin", required=True)
    p.add_argument("--mlx-python", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="metal")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    report, failures = [], []
    for name, prompt, ntok in ENFORCED + DIAGNOSTIC:
        enforced = (name, prompt, ntok) in ENFORCED
        sc = run(a, "scratchy", name, prompt, ntok)
        mx = run(a, "mlx-lm", name, prompt, ntok)
        common = min(len(sc), len(mx))
        n = agree_chars(sc, mx)
        exact = bool(common) and n == common
        kind = "enforced" if enforced else "diagnostic"
        status = "OK" if exact else ("FAIL" if enforced else "drift")
        if enforced and not exact:
            failures.append(name)
        print(f"  [{kind}] {name:11s} {status:5s} agree {n}/{common} chars")
        if not exact:
            print(f"      scratchy: ...{sc[max(0,n-25):n]}|{sc[n:n+40]}")
            print(f"      mlx-lm  : ...{mx[max(0,n-25):n]}|{mx[n:n+40]}")
        report.append({"name": name, "kind": kind, "prompt": prompt,
                       "agree_chars": n, "common_chars": common,
                       "exact": exact, "scratchy": sc, "mlx_lm": mx})

    with open(os.path.join(a.out_dir, "parity.json"), "w") as f:
        json.dump(report, f, indent=1)

    if failures:
        print()
        print("PARITY GATE FAILED on high-confidence prompts: "
              f"{', '.join(failures)}.")
        print("These have deterministic answers, so a mismatch points at the "
              "load path (wrong quant preset, group size, or dequant), not at "
              "floating-point noise. Every timing below would be measuring a "
              "different computation. Refusing to benchmark.")
        return 1
    print("  PARITY OK — all high-confidence prompts match exactly.")
    drift = [r for r in report if r["kind"] == "diagnostic" and not r["exact"]]
    if drift:
        print("  (open-ended prompts diverge at the first near-tie, as "
              "expected between two int4 kernel stacks; recorded, not gating)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
