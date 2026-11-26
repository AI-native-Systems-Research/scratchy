# Each request of a traced run, against the GPU's timeline (an Instruments Metal System Trace's
# exports): its wall, time to first token, and over its decode (first token to end) the GPU's busy
# and idle time, its command buffers' spans — the first 20 against the rest, a slow start reading
# apart from a slow step — and the decode wall past its command buffers. Usage: request-trace.py
# <toc.xml> <metal-gpu-intervals.xml> <process> <requests.json> (a list of {name, start, first,
# end}, epoch seconds)
import datetime
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET

toc_path, gpu_path, process, requests_path = sys.argv[1:5]
start_date = re.search(r"<start-date>([^<]*)</start-date>", open(toc_path).read()).group(1)
t0 = datetime.datetime.fromisoformat(start_date).timestamp()
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


spans = {}
for row in ET.parse(gpu_path).getroot().iter("row"):
    cols = [value(c) for c in row]
    get = lambda tag: next((c for c in cols if c[2] == tag), None)
    proc, cb, depth, chan = get("process"), get("metal-command-buffer-id"), get("metal-nesting-level"), get("gpu-channel-name")
    if not (proc and process in proc[1] and cb and depth and depth[0] == "0" and chan and chan[0] == "Compute"):
        continue
    s = t0 + int(get("start-time")[0]) / 1e9
    e = s + int(get("duration")[0]) / 1e9
    a, b = spans.get(cb[0], (s, e))
    spans[cb[0]] = (min(a, s), max(b, e))
cbs = sorted(spans.values())
print(f"{process}: {len(cbs)} command buffers in the trace")
for r in json.load(open(requests_path)):
    first, end = r["first"], r["end"]
    inside = [(max(s, first), min(e, end)) for s, e in cbs if e > first and s < end]
    busy = sum(e - s for s, e in inside)
    decode = end - first
    lens = [(e - s) * 1e3 for s, e in cbs if s >= first and e <= end]
    head = statistics.median(lens[:20]) if len(lens) >= 20 else float("nan")
    rest = statistics.median(lens[20:]) if len(lens) > 30 else float("nan")
    excess = decode * 1e3 - len(lens) * (rest if rest == rest else 0)
    print(f"  {r['name']:12} wall {r['end'] - r['start']:6.2f} s, ttft {1e3 * (first - r['start']):5.0f} ms, decode "
          f"{decode:6.2f} s: GPU busy {100 * busy / decode:5.1f}%, {len(lens)} command buffers, median span first 20 "
          f"{head:6.2f} ms, rest {rest:6.2f} ms; decode past {len(lens)} x rest {excess:+6.0f} ms")
