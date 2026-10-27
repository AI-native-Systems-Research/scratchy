#!/usr/bin/env python3
"""One-off: the full bind-law + readback listing of the p3-loc pair slice,
one line per launch, so the pair's actual launch geometry is read off the
log (not guessed): which seed each launch served, at what step, rows, ctx."""
import re
from pathlib import Path

S = Path("/work/fb-noise-probe/p3-loc-server.log")
l1, l2 = 1215, 1439
lines = S.read_text().splitlines()[l1:l2]

BIND_RE = re.compile(r"SEAT_BIND rows=(\d+) seed_words=\[([^\]]*)\] invt=t(\d+) laws=(.*)")
READ_CTX_RE = re.compile(r"SEAT_READBACK (.*?) rows=(\d+) ")
ROW_RE = re.compile(r"r(\d+):words=\[([^\]]*)\]:invt=([^:]*):pick=(None|Some\(\d+\))")

binds = []
for ln in lines:
    if "SEAT_BIND" in ln:
        m = BIND_RE.search(ln)
        binds.append((m.group(1), m.group(4)) if m else ("?", ln.strip()[:120]))
reads = []
for ln in lines:
    if "SEAT_READBACK" in ln:
        m = READ_CTX_RE.search(ln)
        rows = [rm.group(0) for rm in ROW_RE.finditer(ln)]
        reads.append((m.group(1), m.group(2), rows) if m else ("?", "?", [ln.strip()[:120]]))

print(f"pair slice: {len(binds)} binds, {len(reads)} readbacks")
for i, (r, laws) in enumerate(binds):
    print(f"bind[{i:3}] rows={r} laws={laws}")
print()
for i, (ctx, r, rows) in enumerate(reads):
    print(f"read[{i:3}] ctx='{ctx}' rows={r} {' '.join(rows)}")
