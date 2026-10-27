#!/usr/bin/env python3
"""The sample-noise probe's BIND VERIFIER (front-b-noise, nickm2 leg).

Reads a bake's own placements.json, maps the kernel's tensors to tids in
PARAMETER ORDER (the door's own numbering law — hash_noise's and the intmr
kit's measured t0..tN), and VERIFIES every tensor's placement size and the
outputs' Logits role against `sn_inputs.expected()` BEFORE a single byte is
staged: a door that numbers, sizes, or roles differently fails loudly with
both tables printed, never a silent misbind. On success prints one line of
space-separated `bundle_run_dev` arguments — the input binds and one
`--out raw:t<id>=<path>` PER OUTPUT (the runner's parser takes exactly one
argument per `--out` flag) — for the leg to word-split.

use: sn_binds.py <bake-dir> <case>
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sn_inputs  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("use: sn_binds.py <bake-dir> <case>", file=sys.stderr)
        return 2
    bake, case = sys.argv[1], sys.argv[2]
    with open(pathlib.Path(bake) / "placements.json") as f:
        places = json.load(f)["places"]
    by_tid = {p["tid"]: p for p in places}
    tids = sorted(by_tid)
    exp = sn_inputs.expected(case)

    def dump_and_fail(why: str) -> int:
        print(f"SN-BINDS-FAILED ({why})", file=sys.stderr)
        print("bake placements (tid, role, size):", file=sys.stderr)
        for tid in tids:
            p = by_tid[tid]
            print(f"  t{tid} {p['role']} {p['size']} B", file=sys.stderr)
        print("expected (param order, size, out):", file=sys.stderr)
        for e in exp:
            print(f"  {e['name']} {e['size']} B out={e['out']}", file=sys.stderr)
        return 1

    if len(tids) != len(exp):
        return dump_and_fail(
            f"the bake placed {len(tids)} tensors, the kernel names {len(exp)}"
        )
    args: list[str] = []
    outs: list[str] = []
    for tid, e in zip(tids, exp):
        p = by_tid[tid]
        if p["size"] != e["size"]:
            return dump_and_fail(
                f"t{tid} ({e['name']}): placement {p['size']} B, expected {e['size']} B"
            )
        if e["out"]:
            if p["role"] != "Logits":
                return dump_and_fail(
                    f"t{tid} ({e['name']}): role {p['role']}, expected Logits"
                )
            outs.append(f"raw:t{tid}={HERE / ('out_sn_' + case + '_' + e['name'] + '.bin')}")
        else:
            args.append(f"t{tid}={sn_inputs.INDIR / ('sn_' + case + '_' + e['name'] + '.bin')}")
            args.append(f"raw:t{tid}")
    # ONE `--out` PER OUTPUT — bundle_run_dev's parser consumes exactly one
    # argument after each `--out` flag, so a single `--out` followed by the
    # whole list leaves every later output in the INPUT list (where `raw:` is
    # a marker and takes no `=<path>`).
    print(f"{' '.join(args)} {' '.join('--out ' + o for o in outs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
