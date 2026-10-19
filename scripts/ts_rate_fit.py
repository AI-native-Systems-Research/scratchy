#!/usr/bin/env python3
"""Per-run TS-rate calibration for RB_TIME legs.

The device TS counter's cyc/us is NOT a constant to assert: the 1012 that
every earlier table used is ~1.4% fast (measured 998.0 on nickm4 and 998.6
pooled with nickm, 2026-10-18; the SOC clock is nominally 1000 MHz per
topo.json). A wrong rate manufactures a phantom host tail that SCALES WITH
DEVICE TIME — a model-scaled "tail" derived from ITL - device_time is a
clock-rate symptom until this fit says otherwise.

Method: regress the stamped per-step host walls (predict_enter[i+1] -
predict_enter[i]) on the raw device cycles per step (all-window busy span +
inter-step gap, wrap-straddling windows excluded). One shared slope across
all logs (one device counter), one intercept per log. slope = 1/R;
intercept ~= the per-step host overhead not covered by device cycles.

The pooled slope is THEIL-SEN (median of within-log pairwise slopes): a
single stalled step (a multi-ms inter-step gap) is a high-leverage point
that rotates a least-squares fit — measured on the nickm 2026-10-18 cell,
where one 9 ms gap moved LS from ~998 to 1228.7 while the bulk sat clean.
The LS fit is printed for reference only.

Input: one or more cell logs carrying BOTH [sdsc-host] predict_enter stamps
and [sdsc-rbgap] all[...] windows (the hoststeps stamps on the RB_TIME
gate). Every RB_TIME leg with stamps should run this and report R.

Usage: ts_rate_fit.py LOG [LOG ...]

Refuses loudly on empty extraction of either kind (verdict-over-nothing).
"""
import re
import sys
import statistics

RB = re.compile(r"\[sdsc-rbgap\] fwd (\d+) all\[(-?\d+),(-?\d+),\d+\]")
PE = re.compile(r"\[sdsc-host\] predict_enter (\d+)")
HALF = 1 << 31


def load_pairs(path):
    """(device_cycles, wall_us) per step, wrap-excluded; None if unusable."""
    fwds, pes = {}, []
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        print(f"{path}: cannot open ({e}) — refusing", file=sys.stderr)
        sys.exit(1)
    for ln in fh:
        m = RB.search(ln)
        if m:
            f = int(m.group(1))
            a0, a1 = int(m.group(2)), int(m.group(3))
            if (a1 - a0) & 0xFFFFFFFF <= HALF:
                fwds[f] = (a0, a1)
        m = PE.search(ln)
        if m:
            pes.append(int(m.group(1)))
    if not pes:
        print(f"{path}: NO [sdsc-host] predict_enter lines — refusing "
              f"(the rate fit needs the hoststeps stamps)", file=sys.stderr)
        sys.exit(1)
    if not fwds:
        print(f"{path}: NO [sdsc-rbgap] all[...] lines — refusing", file=sys.stderr)
        sys.exit(1)
    pts = []
    for i in range(4, len(pes) - 1):  # skip warmup steps
        f = i + 1  # rbgap fwd numbering is 1-based over the same forwards
        if f not in fwds or (f + 1) not in fwds:
            continue
        a0, a1 = fwds[f]
        b0 = fwds[f + 1][0]
        cyc = (a1 - a0) & 0xFFFFFFFF
        gap = (b0 - a1) & 0xFFFFFFFF
        if cyc > HALF or gap > HALF:
            continue
        pts.append((cyc + gap, pes[i + 1] - pes[i]))
    if len(pts) < 8:
        print(f"{path}: only {len(pts)} paired steps — refusing "
              f"(need >= 8 for a fit)", file=sys.stderr)
        sys.exit(1)
    return pts


def main():
    if len(sys.argv) < 2:
        print("usage: ts_rate_fit.py LOG [LOG ...]", file=sys.stderr)
        sys.exit(1)
    logs = []
    for path in sys.argv[1:]:
        pts = load_pairs(path)
        label = path.rsplit("/", 1)[-1]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
        if sx == 0 or sy == 0:
            print(f"{label}: n={len(pts)} steps  zero variance — refusing",
                  file=sys.stderr)
            sys.exit(1)
        r = statistics.correlation(xs, ys)
        slope = r * sy / sx  # us per cycle
        print(f"{label}: n={len(pts)} steps  solo-fit R={1/slope:.1f} cyc/us"
              f"  (reference only; the pooled fit below is the number)")
        logs.append((label, pts))

    # Pooled: wall = a_k + beta*cyc  (one slope, one intercept per log).
    # TWO estimators, and the robust one is THE NUMBER:
    #   - least squares: broken by a single high-leverage stalled step (a
    #     9 ms inter-step gap on the nickm 2026-10-18 cell rotated the fit
    #     to R=1228.7 with a +7.8 ms intercept while the bulk sat clean at
    #     998 — the stall extends the x-range, and the slope IS the x-spread);
    #   - Theil–Sen (median of within-log pairwise slopes, intercepts differenced
    #     out): immune to any minority of stalled steps.
    n = len(logs)
    allpts = []
    for k, (_, pts) in enumerate(logs):
        allpts.extend((c, w, k) for c, w in pts)
    K = n
    X = lambda p: [1.0 if j == p[2] else 0.0 for j in range(K)] + [float(p[0])]
    Y = [p[1] for p in allpts]
    dim = K + 1
    A = [[sum(X(p)[i] * X(p)[j] for p in allpts) for j in range(dim)] for i in range(dim)]
    b = [sum(X(p)[i] * y for p, y in zip(allpts, Y)) for i in range(dim)]
    for i in range(dim):
        pv = A[i][i]
        if pv == 0:
            print("degenerate fit (a log contributes zero variance) — refusing",
                  file=sys.stderr)
            sys.exit(1)
        for j in range(dim):
            A[i][j] /= pv
        b[i] /= pv
        for k in range(dim):
            if k != i:
                f = A[k][i]
                for j in range(dim):
                    A[k][j] -= f * A[i][j]
                b[k] -= f * b[i]
    ls_beta = b[-1]
    ls_intercepts = b[:K]

    # Theil–Sen: pairwise slopes within each log (the per-log intercept
    # differences out), median across all logs' pairs.
    slopes = []
    for _, pts in logs:
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dx = pts[i][0] - pts[j][0]
                if dx != 0:
                    slopes.append((pts[i][1] - pts[j][1]) / dx)
    if not slopes:
        print("no within-log pairs with x-spread — refusing", file=sys.stderr)
        sys.exit(1)
    slopes.sort()
    beta = slopes[len(slopes) // 2]
    intercepts = []
    for _, pts in logs:
        intercepts.append(statistics.median([w - beta * c for c, w in pts]))

    ybar = statistics.mean(Y)
    rss = sum((y - (intercepts[p[2]] + beta * p[0])) ** 2 for p, y in zip(allpts, Y))
    r2 = 1 - rss / sum((y - ybar) ** 2 for y in Y)
    ls_r2 = 1 - sum(
        (y - (ls_intercepts[p[2]] + ls_beta * p[0])) ** 2
        for p, y in zip(allpts, Y)
    ) / sum((y - ybar) ** 2 for y in Y)
    print(f"\nPOOLED (Theil–Sen, THE NUMBER): R = {1/beta:.1f} cyc/us   "
          f"R^2 = {r2:.5f}   n = {len(allpts)} steps over {K} log(s)")
    print(f"  (least squares for reference: R = {1/ls_beta:.1f} cyc/us   "
          f"R^2 = {ls_r2:.5f} — a single stalled step can rotate it; trust "
          f"the median)")
    for (label, _), a in zip(logs, intercepts):
        print(f"  {label}: intercept {a:+.1f} us (per-step host overhead ~= 0 if ~0)")
    print(f"\nDevice times from this run's tables scale by x{1012/(1/beta):.4f} "
          f"vs the old 1012 constant (raw cycles / {1/beta:.1f}).")


if __name__ == "__main__":
    main()
