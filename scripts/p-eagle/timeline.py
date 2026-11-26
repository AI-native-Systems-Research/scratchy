# One process's command buffers on the GPU, from an Instruments Metal System Trace's exports: each
# command buffer's GPU span (`metal-gpu-intervals`, depth 0, compute) and when the host submitted it
# (`metal-application-command-buffer-submissions`). Prints the spans, the GPU's idle time between
# them (a gap > 0.2 ms is host late when the command buffer was submitted after the GPU went idle,
# held on the device when before), and the host's turnaround: a command buffer's submission after
# the GPU finished the one two before it. Usage: timeline.py <submissions.xml> <gpu.xml> <process>
import statistics
import sys
import xml.etree.ElementTree as ET

subs_path, gpu_path, process = sys.argv[1:4]


def rows_of(path):
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

    return [[value(c) for c in row] for row in ET.parse(path).getroot().iter("row")]


def first(cols, tag):
    return next((c for c in cols if c[2] == tag), None)


subs, gpu = {}, {}
for cols in rows_of(subs_path):
    proc, cb = first(cols, "process"), first(cols, "metal-command-buffer-id")
    if proc and process in proc[1] and cb:
        subs.setdefault(cb[0], int(first(cols, "start-time")[0]))
for cols in rows_of(gpu_path):
    proc, cb = first(cols, "process"), first(cols, "metal-command-buffer-id")
    depth, chan = first(cols, "metal-nesting-level"), first(cols, "gpu-channel-name")
    if not (proc and process in proc[1] and cb and depth and depth[0] == "0" and chan and chan[0] == "Compute"):
        continue
    start, dur = int(first(cols, "start-time")[0]), int(first(cols, "duration")[0])
    s, e = gpu.get(cb[0], (start, start + dur))
    gpu[cb[0]] = (min(s, start), max(e, start + dur))
order = sorted(gpu.items(), key=lambda kv: kv[1][0])
if len(order) < 3:
    sys.exit(f"{len(order)} command buffers of {process}: nothing to time")
spans = [(e - s) / 1e6 for _, (s, e) in order]
idle, late, held, turn = [], 0, 0, []
end = order[0][1][1]
for k in range(1, len(order)):
    cb, (s, e) = order[k]
    gap = max(0.0, (s - end) / 1e6)
    idle.append(gap)
    if gap > 0.2 and cb in subs:
        late += subs[cb] > end
        held += subs[cb] <= end
    if k >= 2 and cb in subs:
        turn.append((subs[cb] - order[k - 2][1][1]) / 1e6)
    end = max(end, e)
total = (end - order[0][1][0]) / 1e6
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print(f"{process}: {len(order)} command buffers over {total:.1f} ms; GPU span a command buffer median "
      f"{statistics.median(spans):.3f} ms (p10 {q(spans, .1):.3f}, p90 {q(spans, .9):.3f})")
print(f"  idle {sum(idle):.1f} ms ({100 * sum(idle) / total:.1f}%); before a command buffer median "
      f"{statistics.median(idle):.3f} ms, p90 {q(idle, .9):.3f}; gaps > 0.2 ms: {late} host late, "
      f"{held} held on the device")
if turn:
    print(f"  host turnaround (submitted after the GPU finished the one two before): median "
          f"{statistics.median(turn):.3f} ms (p10 {q(turn, .1):.3f}, p90 {q(turn, .9):.3f})")
