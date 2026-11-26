# One process's compute command buffers' GPU spans, from an Instruments Metal System Trace's
# `metal-gpu-intervals` export (depth 0, the Compute channel): a histogram in 0.5 ms bins and the
# modal span (the median of the spans in the most common bin and its neighbours). On a reply that
# copies its prompt under the n-gram proposer, most steps carry every draft, so the mode is the
# R-row verify step. Usage: spans.py <gpu.xml> <process> <R> [note]
import collections
import statistics
import sys
import xml.etree.ElementTree as ET

path, process, r = sys.argv[1:4]
note = sys.argv[4] if len(sys.argv) > 4 else ""
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
for row in ET.parse(path).getroot().iter("row"):
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
spans = [(e - s) / 1e6 for s, e in sorted(gpu.values())]
if len(spans) < 20:
    sys.exit(f"r{r}: {len(spans)} command buffers of {process}: nothing to time")
spans = spans[5:]  # the prompt's prefill and the first steps
bins = collections.Counter(int(x / 0.5) for x in spans)
top = bins.most_common(1)[0][0]
near = [x for x in spans if abs(int(x / 0.5) - top) <= 1]
mode = statistics.median(near)
print(f"r{r}: {len(spans)} command buffers, median {statistics.median(spans):.2f} ms, "
      f"upper mode {mode:.2f} ms ({len(near)} in its bins); drafts a drafting step {note}")
for b in sorted(bins):
    if bins[b] >= max(3, len(spans) // 100):
        print(f"  {b * 0.5:5.1f}-{b * 0.5 + 0.5:5.1f} ms {bins[b]:5d} {'#' * min(60, bins[b] * 60 // len(spans))}")
