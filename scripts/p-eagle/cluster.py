# The R-row steps' GPU span from verify-cost-trace.sh's legs: per R, the median of the decode
# command buffers slower than the plain leg's p95 (a step whose n-gram matched nothing runs one
# row and costs a plain step), against the plain leg's median. Usage: cluster.py <out dir> <process>
import os
import statistics
import sys

out, process = sys.argv[1:3]


def spans(r):
    import xml.etree.ElementTree as ET
    ids = {}

    def value(el):
        if "ref" in el.attrib:
            return ids[el.attrib["ref"]]
        v = (el.text or "", el.attrib.get("fmt", ""), el.tag)
        if "id" in el.attrib:
            ids[el.attrib["id"]] = v
        for child in el:
            value(child)
        return v

    gpu = {}
    for row in ET.parse(os.path.join(out, f"r{r}", "gpu.xml")).getroot().iter("row"):
        cols = {}
        for c in row:
            v = value(c)
            cols.setdefault(v[2], v)
        proc, cb = cols.get("process"), cols.get("metal-command-buffer-id")
        depth, chan = cols.get("metal-nesting-level"), cols.get("gpu-channel-name")
        if not (proc and process in proc[1] and cb and depth and depth[0] == "0" and chan and chan[0] == "Compute"):
            continue
        start, dur = int(cols["start-time"][0]), int(cols["duration"][0])
        s, e = gpu.get(cb[0], (start, start + dur))
        gpu[cb[0]] = (min(s, start), max(e, start + dur))
    return [(e - s) / 1e6 for s, e in sorted(gpu.values())][5:]


plain = spans(1)
p95 = sorted(plain)[int(0.95 * (len(plain) - 1))]
base = statistics.median(plain)
print(f"R=1: {len(plain)} steps, median {base:.2f} ms (p95 {p95:.2f})")
for r in sorted(int(d[1:]) for d in os.listdir(out) if d.startswith("r") and d[1:].isdigit()):
    if r == 1:
        continue
    xs = spans(r)
    up = [x for x in xs if x > p95]
    m = statistics.median(up)
    print(f"R={r}: {len(up)} of {len(xs)} steps past the plain p95, median {m:.2f} ms = {m / base:.2f}x plain")
