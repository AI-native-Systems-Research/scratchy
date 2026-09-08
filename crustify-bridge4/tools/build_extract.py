#!/usr/bin/env python3
"""Build the bridge-4 consolidated C++ translation unit + UNITS.tsv.

Unit set = the transitive call closure of the senprog EMISSION (seeds
Dpc::convertIr2Senprog and Dpc::checkProgFormatCompatibility), with bare-name
ambiguity resolved BY RECEIVER TYPE (see OVERRIDES). Bodies are copied VERBATIM
from the authority tree; nothing is reflowed and no brace is ever appended.
"""
import json
import os
import re
import subprocess
import sys

SRC = '/Users/nickm/git/deeptools-src'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'staged', 'crustify-bridge4')
os.makedirs(os.path.join(OUT, 'cpp'), exist_ok=True)

# Names the closure could not resolve from a bare identifier. The receiver in
# convertIr2Senprog is an OperandAttr (`operandPair.second`, `regPair.second`),
# so `.print(id)` is OperandAttr::print -- NOT ProgIrGraph::print, which is the
# ProgIR text dumper reached only from dpc_standalone's --ir option.
DROP = {
    ('sys-arch-spec/progir/progir.cpp', 'ProgIrGraph::print'),
    ('sys-arch-spec/progir/progir.cpp', 'ProgIrGraph::traverseGraphDFS'),
}
ADD = [
    # (file, qname) forced into the closure, resolved by receiver type
    ('sys-arch-spec/progir/progir.cpp', 'OperandAttr::print'),
]

closure = json.load(open(os.path.join(HERE, 'closure.json')))

# re-enumerate so we can slice bodies
FILES = sorted({v['file'] for v in closure.values()} |
               {f for f, _ in ADD})
alldefs = {}
for rel in FILES:
    p = os.path.join(SRC, rel)
    j = os.path.join(HERE, 'fns_' + rel.replace('/', '_') + '.json')
    if not os.path.exists(j):
        subprocess.run([sys.executable, os.path.join(HERE, 'enum_fns.py'), p, j],
                       check=True, stdout=subprocess.DEVNULL)
    for f in json.load(open(j)):
        f['file'] = rel
        alldefs.setdefault((rel, f['qname']), []).append(f)

units = []
for name, v in closure.items():
    key = (v['file'], v['qname'])
    if key in DROP:
        continue
    units.append(key)
for key in ADD:
    if key not in units:
        units.append(key)

# dependency order: callees before callers. Use closure edges; ADD'ed nodes are
# leaves for ordering purposes (OperandAttr::print calls only stdlib + accessors).
edge = {}
qn_of = {}
for name, v in closure.items():
    qn_of[name] = (v['file'], v['qname'])
for name, v in closure.items():
    key = (v['file'], v['qname'])
    edge[key] = [qn_of[c] for c in v['callees'] if c in qn_of]
for key in ADD:
    edge.setdefault(key, [])

ordered = []
seen = set()
temp = set()


def visit(k):
    if k in seen or k not in edge:
        return
    if k in temp:          # cycle: emit now, C++ needs a fwd decl anyway
        return
    temp.add(k)
    for d in edge.get(k, []):
        if d in edge and d not in DROP:
            visit(d)
    temp.discard(k)
    if k not in seen:
        seen.add(k)
        ordered.append(k)


for k in units:
    visit(k)
for k in units:
    if k not in seen:
        seen.add(k)
        ordered.append(k)

raws = {rel: open(os.path.join(SRC, rel), errors='replace').read() for rel in FILES}

banner_fmt = ('/* ==========================================================================\n'
              ' * ENTRY %(n)03d  %(qname)s\n'
              ' * AUTHORITY: %(file)s:%(start)d-%(end)d   (%(loc)d lines)\n'
              ' * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED\n'
              ' * ========================================================================== */\n')

tsv = ['\t'.join(['entry', 'unit', 'qname', 'authority_file', 'auth_start',
                  'auth_end', 'loc', 'extract_start', 'extract_end', 'callees'])]
parts = ['#include "prelude.inc"\n\n']
extract_line = parts[0].count('\n') + 1

for n, key in enumerate(ordered, 1):
    rel, qname = key
    cands = alldefs.get(key)
    if not cands:
        sys.stderr.write('NO DEFINITION for %s in %s\n' % (qname, rel))
        continue
    f = max(cands, key=lambda e: e['close_off'] - e['open_off'])
    # Snap the slice back to the START OF THE SIGNATURE'S LINE. enum_fns skips
    # leading whitespace to get start_line right, but slicing from there drops
    # the indentation of an in-class definition in a header -- which makes the
    # body not verbatim. verify_extract.py caught exactly this on 24 units.
    ls = raws[rel].rfind('\n', 0, f['start_off']) + 1
    body = raws[rel][ls:f['close_off'] + 1]
    loc = f['end_line'] - f['start_line'] + 1
    ban = banner_fmt % dict(n=n, qname=qname, file=rel,
                            start=f['start_line'], end=f['end_line'], loc=loc)
    parts.append(ban)
    extract_line += ban.count('\n')
    es = extract_line
    parts.append(body + '\n\n')
    extract_line += body.count('\n') + 2
    ee = es + body.count('\n')
    short = re.sub(r'.*::', '', qname)
    unit = 'e%03d_%s' % (n, short)
    callees = ','.join(sorted({re.sub(r'.*::', '', c[1]) for c in edge.get(key, [])}))
    tsv.append('\t'.join([str(n), unit, qname, rel, str(f['start_line']),
                          str(f['end_line']), str(loc), str(es), str(ee), callees]))

open(os.path.join(OUT, 'cpp', 'bridge4.cpp'), 'w').write(''.join(parts))
open(os.path.join(OUT, 'UNITS.tsv'), 'w').write('\n'.join(tsv) + '\n')
print('wrote %d units' % (len(tsv) - 1))
print('  cpp/bridge4.cpp : %d lines' % ''.join(parts).count('\n'))
print('  UNITS.tsv       : %d rows' % (len(tsv) - 1))
