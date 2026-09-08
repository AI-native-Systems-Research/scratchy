#!/usr/bin/env python3
"""Transitive call closure of the senprog EMISSION, over the authority tree.

Seeds are the two dpc.cpp entry points. We enumerate every function definition in
the candidate files by brace matching (enum_fns machinery), then walk call sites
(`ident(`) inside each reached body and follow them by bare name. Also reports the
declared-data tables the emission indexes, which are part of the port surface.
"""
import re
import sys
import json
import subprocess
import os

SRC = '/Users/nickm/git/deeptools-src'
FILES = [
    'sys-arch-spec/dpc/dpc.cpp',
    'sys-arch-spec/dpc/dpc.h',
    'sys-arch-spec/progir/progir.cpp',
    'sys-arch-spec/progir/progir.h',
    'sys-arch-spec/progir/regvisitor.cpp',
    'sys-arch-spec/progir/regvisitor.h',
    'sys-arch-spec/isa/isa.cpp',
    'sys-arch-spec/isa/isa.hpp',
]
HERE = os.path.dirname(os.path.abspath(__file__))

# --- reuse the enumerator ----------------------------------------------------
sys.path.insert(0, HERE)


def strip(raw):
    b = list(raw)
    i, n = 0, len(raw)
    state = None
    while i < n:
        c = raw[i]
        if state is None:
            if c == '/' and i + 1 < n and raw[i + 1] == '/':
                state = 'line'; b[i] = b[i + 1] = ' '; i += 2; continue
            if c == '/' and i + 1 < n and raw[i + 1] == '*':
                state = 'block'; b[i] = b[i + 1] = ' '; i += 2; continue
            if c == '"':
                state = 'str'; b[i] = ' '; i += 1; continue
            if c == "'":
                state = 'chr'; b[i] = ' '; i += 1; continue
            i += 1; continue
        if state == 'line':
            if c == '\n': state = None
            else: b[i] = ' '
            i += 1; continue
        if state == 'block':
            if c == '*' and i + 1 < n and raw[i + 1] == '/':
                b[i] = b[i + 1] = ' '; state = None; i += 2; continue
            if c != '\n': b[i] = ' '
            i += 1; continue
        q = '"' if state == 'str' else "'"
        if c == '\\':
            b[i] = ' '
            if i + 1 < n and raw[i + 1] != '\n': b[i + 1] = ' '
            i += 2; continue
        if c == q:
            b[i] = ' '; state = None; i += 1; continue
        if c != '\n': b[i] = ' '
        i += 1; continue
    return ''.join(b)


defs = {}          # bare name -> list of entries
entries = []
for rel in FILES:
    p = os.path.join(SRC, rel)
    out = os.path.join(HERE, 'fns_' + rel.replace('/', '_') + '.json')
    subprocess.run([sys.executable, os.path.join(HERE, 'enum_fns.py'), p, out],
                   check=True, stdout=subprocess.DEVNULL)
    raw = open(p, errors='replace').read()
    code = strip(raw)
    for f in json.load(open(out)):
        f['file'] = rel
        f['body'] = code[f['open_off']:f['close_off'] + 1]
        f['raw'] = raw[f['start_off']:f['close_off'] + 1]
        entries.append(f)
        defs.setdefault(f['name'], []).append(f)

print('enumerated %d function definitions across %d files' % (len(entries), len(FILES)))
for rel in FILES:
    k = [e for e in entries if e['file'] == rel]
    print('  %-40s %4d' % (rel, len(k)))

KEYWORDS = {'if', 'for', 'while', 'switch', 'catch', 'return', 'sizeof', 'new',
            'delete', 'throw', 'and', 'or', 'not', 'do', 'else', 'case',
            'static_cast', 'dynamic_cast', 'reinterpret_cast', 'const_cast',
            'printf', 'snprintf', 'sprintf', 'strcmp', 'atoi', 'stoi', 'stoul',
            'stoull', 'stof', 'stod', 'to_string', 'push_back', 'insert',
            'count', 'at', 'find', 'size', 'empty', 'begin', 'end', 'clear',
            'erase', 'substr', 'length', 'c_str', 'str', 'emplace_back',
            'make_pair', 'move', 'max', 'min', 'abs', 'pop_back', 'back',
            'front', 'first', 'second', 'assign', 'resize', 'reserve',
            'DT_ERROR', 'DT_ERROR_FMT', 'DT_ASSERT', 'DT_WARN', 'DT_INFO'}

seeds = ['convertIr2Senprog', 'checkProgFormatCompatibility']
AMBIG = {}
reached = {}
order = []
queue = list(seeds)
edges = {}
while queue:
    nm = queue.pop(0)
    if nm in reached:
        continue
    cands = defs.get(nm)
    if not cands:
        reached[nm] = None
        continue
    if len(cands) > 1:
        AMBIG.setdefault(nm, cands)
    # prefer the largest definition (the real one, not a 5-line overload shim)
    f = max(cands, key=lambda e: e['close_off'] - e['open_off'])
    reached[nm] = f
    order.append(nm)
    callees = set()
    for m in re.finditer(r'([A-Za-z_]\w*)\s*\(', f['body']):
        c = m.group(1)
        if c in KEYWORDS or c == nm:
            continue
        if c in defs:
            callees.add(c)
    edges[nm] = sorted(callees)
    for c in sorted(callees):
        if c not in reached:
            queue.append(c)

hit = [n for n in order]
print('\n=== CALL CLOSURE from the senprog emission: %d defined functions ===' % len(hit))
tot = 0
byfile = {}
for nm in hit:
    f = reached[nm]
    loc = f['end_line'] - f['start_line'] + 1
    tot += loc
    byfile.setdefault(f['file'], []).append((nm, f))
for rel in FILES:
    lst = byfile.get(rel, [])
    if not lst:
        continue
    print('\n-- %s : %d functions --' % (rel, len(lst)))
    for nm, f in sorted(lst, key=lambda x: x[1]['start_line']):
        print('   %5d-%5d %5dL  %-42s -> %s' % (
            f['start_line'], f['end_line'], f['end_line'] - f['start_line'] + 1,
            f['qname'], ','.join(edges.get(nm, [])[:6]) or '(leaf)'))
print('\nTOTAL in-closure lines: %d' % tot)

print('\n=== AMBIGUOUS names in the closure (overloads / same name, many classes) ===')
print('    the closure picked the LARGEST; a porter must resolve by receiver type')
for nm, cands in sorted(AMBIG.items()):
    print('   %s:' % nm)
    for c in sorted(cands, key=lambda e: (e['file'], e['start_line'])):
        print('      %-38s %5d-%5d %4dL  %s' % (c['file'], c['start_line'],
                                                c['end_line'],
                                                c['end_line'] - c['start_line'] + 1,
                                                c['qname']))

TABLES = ['typeToFieldEncoding', 'typeToFieldBitShift', 'senComponentsToString',
          'regTypeToString', 'progFormatFeaturesMap', 'typeToFieldName',
          'opCodeToString', 'operandToString']
print('\n=== declared-data tables the emission indexes ===')
seedbodies = ''.join(reached[n]['body'] for n in hit if reached[n])
for t in TABLES:
    print('   %-26s referenced in closure: %s' % (t, 'YES' if t in seedbodies else 'no'))

json.dump({n: dict(file=reached[n]['file'], start=reached[n]['start_line'],
                   end=reached[n]['end_line'], qname=reached[n]['qname'],
                   callees=edges.get(n, []))
           for n in hit}, open(os.path.join(HERE, 'closure.json'), 'w'), indent=1)
