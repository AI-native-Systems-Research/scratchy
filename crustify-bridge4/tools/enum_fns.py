#!/usr/bin/env python3
"""Enumerate C++ function definitions by BRACE MATCHING, not by signature regex.

Strip comments and string/char literals (length-preserving), then walk the file
tracking brace depth. A function definition is a '{' whose preceding tokens end in
')' (optionally followed by qualifiers or a ctor-init-list) and whose declarator
head is not a control keyword. The matching close brace is found by depth counting.
"""
import re
import sys
import json

path = sys.argv[1]
raw = open(path, 'r', errors='replace').read()

# --- length-preserving blank-out of comments and literals -------------------
b = list(raw)
i, n = 0, len(raw)
state = None
while i < n:
    c = raw[i]
    if state is None:
        if c == '/' and i + 1 < n and raw[i + 1] == '/':
            state = 'line'
            b[i] = b[i + 1] = ' '
            i += 2
            continue
        if c == '/' and i + 1 < n and raw[i + 1] == '*':
            state = 'block'
            b[i] = b[i + 1] = ' '
            i += 2
            continue
        if c == '"':
            state = 'str'
            b[i] = ' '
            i += 1
            continue
        if c == "'":
            state = 'chr'
            b[i] = ' '
            i += 1
            continue
        i += 1
        continue
    if state == 'line':
        if c == '\n':
            state = None
        else:
            b[i] = ' '
        i += 1
        continue
    if state == 'block':
        if c == '*' and i + 1 < n and raw[i + 1] == '/':
            b[i] = b[i + 1] = ' '
            state = None
            i += 2
            continue
        if c != '\n':
            b[i] = ' '
        i += 1
        continue
    if state in ('str', 'chr'):
        q = '"' if state == 'str' else "'"
        if c == '\\':
            b[i] = ' '
            if i + 1 < n and raw[i + 1] != '\n':
                b[i + 1] = ' '
            i += 2
            continue
        if c == q:
            b[i] = ' '
            state = None
            i += 1
            continue
        if c != '\n':
            b[i] = ' '
        i += 1
        continue
code = ''.join(b)
assert len(code) == len(raw), (len(code), len(raw))

starts = [0]
for m in re.finditer('\n', raw):
    starts.append(m.end())


def lineno(off):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= off:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


CTRL = {'if', 'for', 'while', 'switch', 'catch', 'do', 'else', 'return', 'case',
        'try', 'namespace', 'extern', 'enum', 'class', 'struct', 'union',
        'template', 'using', 'sizeof', 'decltype', 'static_assert'}
QUAL = re.compile(r'^(?:const|noexcept|override|final|volatile|mutable|'
                  r'throw\s*\([^)]*\)|->\s*[\w:<>,\s\*&\[\]]+)\s*$')

fns = []
depth = 0
i = 0
N = len(code)
while i < N:
    c = code[i]
    if c == '}':
        depth -= 1
        i += 1
        continue
    if c != '{':
        i += 1
        continue
    j = i - 1
    while j >= 0 and code[j] in ' \t\n\r':
        j -= 1
    k = j
    close = -1
    while k >= 0:
        if code[k] == ')':
            close = k
            break
        if code[k] in ';{}':
            break
        k -= 1
    if close < 0:
        depth += 1
        i += 1
        continue
    between = code[close + 1:i].strip()
    if not (between == '' or QUAL.match(between) or between.startswith(':')):
        depth += 1
        i += 1
        continue
    d = 0
    p = close
    while p >= 0:
        if code[p] == ')':
            d += 1
        elif code[p] == '(':
            d -= 1
            if d == 0:
                break
        p -= 1
    if p < 0:
        depth += 1
        i += 1
        continue
    h = p - 1
    while h >= 0 and code[h] not in ';{}':
        h -= 1
    head = code[h + 1:p].strip()
    m = re.search(r'([A-Za-z_~]\w*)\s*$', head)
    if not head or not m or m.group(1) in CTRL:
        depth += 1
        i += 1
        continue
    name = m.group(1)
    d2 = 0
    q = i
    while q < N:
        if code[q] == '{':
            d2 += 1
        elif code[q] == '}':
            d2 -= 1
            if d2 == 0:
                break
        q += 1
    if q >= N:
        sys.stderr.write('UNTERMINATED at line %d\n' % lineno(i))
        depth += 1
        i += 1
        continue
    qm = re.search(r'((?:[A-Za-z_]\w*\s*::\s*)*[A-Za-z_~]\w*)\s*$', head)
    qname = re.sub(r'\s+', '', qm.group(1)) if qm else name
    # h+1 is just past the PREVIOUS ';' '{' or '}': skip whitespace so the
    # recorded start_line is the signature's own first line, not the previous
    # function's closing brace.
    s = h + 1
    while s < p and code[s] in ' \t\n\r':
        s += 1
    fns.append(dict(name=name, qname=qname, depth=depth,
                    start_off=s, open_off=i, close_off=q,
                    start_line=lineno(s), end_line=lineno(q),
                    head=re.sub(r'\s+', ' ', head)[-160:]))
    depth += 1
    i += 1

print('%s: %d function definitions found by brace matching' % (path, len(fns)))
top = [f for f in fns if f['depth'] == 0]
nested = [f for f in fns if f['depth'] > 0]
print('  depth 0 (file scope): %d' % len(top))
print('  depth >0 (nested / lambda / in-class): %d' % len(nested))
json.dump(fns, open(sys.argv[2], 'w'), indent=1)
print('--- depth 0 ---')
for f in top:
    print('  %5d-%5d %5dL  %s' % (f['start_line'], f['end_line'],
                                  f['end_line'] - f['start_line'] + 1, f['qname']))
print('--- depth >0 ---')
for f in nested:
    print('  d%d %5d-%5d %5dL  %s | %s' % (f['depth'], f['start_line'], f['end_line'],
                                           f['end_line'] - f['start_line'] + 1,
                                           f['qname'], f['head'][-70:]))
