#!/usr/bin/env python3
"""INDEPENDENT verification of cpp/bridge4.cpp against the authority tree.

Bridge 2's extract silently truncated 366 of 384 bodies because its check
re-sliced the source with the SAME (file, line, length) the extractor had used --
a tautology. This verifier must therefore share NOTHING with the extractor:

  * it reads the (file, start_line) from the extract's own BANNER, never from the
    extractor's JSON;
  * it re-derives each function's END with its own line-oriented scanner, a
    different algorithm from enum_fns.py's offset-based one;
  * it compares LENGTH and CONTENT of the extract body against the authority
    slice it derived itself;
  * it asserts brace depth reaches zero exactly at the body's last character,
    so a truncated tail cannot pass;
  * it NEVER appends a brace to balance anything -- that is what hid the damage.

Two negative controls at the end prove the check actually bites.
"""
import os
import re
import sys

SRC = '/Users/nickm/git/deeptools-src'
HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACT = os.path.join(HERE, '..', 'cpp', 'bridge4.cpp')  # repointed after install into the worktree

BANNER = re.compile(
    r'/\* =+\n \* ENTRY (\d+)\s+(\S+)\n \* AUTHORITY: (\S+):(\d+)-(\d+)\s+\((\d+) lines\)\n'
    r' \* revision \S+\s+--\s+BODY VERBATIM, NOTHING APPENDED\n \* =+ \*/\n')


def scan_end(lines, start_idx):
    """Re-derive the end line of the definition beginning at lines[start_idx].

    Independent, line-oriented: walk characters, maintain our own comment/string
    state across lines, count brace depth, and stop at the '}' that returns depth
    to 0 after the first '{' has been seen. Returns (end_idx, depth_at_end).
    """
    depth = 0
    seen_open = False
    in_block = False
    for i in range(start_idx, len(lines)):
        line = lines[i]
        j = 0
        in_line = False
        in_str = False
        in_chr = False
        while j < len(line):
            c = line[j]
            nxt = line[j + 1] if j + 1 < len(line) else ''
            if in_block:
                if c == '*' and nxt == '/':
                    in_block = False
                    j += 2
                    continue
                j += 1
                continue
            if in_line:
                break
            if in_str:
                if c == '\\':
                    j += 2
                    continue
                if c == '"':
                    in_str = False
                j += 1
                continue
            if in_chr:
                if c == '\\':
                    j += 2
                    continue
                if c == "'":
                    in_chr = False
                j += 1
                continue
            if c == '/' and nxt == '*':
                in_block = True
                j += 2
                continue
            if c == '/' and nxt == '/':
                break
            if c == '"':
                in_str = True
                j += 1
                continue
            if c == "'":
                in_chr = True
                j += 1
                continue
            if c == '{':
                depth += 1
                seen_open = True
            elif c == '}':
                depth -= 1
                if seen_open and depth == 0:
                    return i, 0
            j += 1
    return None, depth


def verify(text, label='extract'):
    """Returns (ok, [problems])."""
    problems = []
    marks = [(m.start(), m.end(), m.groups()) for m in BANNER.finditer(text)]
    if not marks:
        return False, ['no banners found -- extract unreadable']
    checked = 0
    for k, (bs, be, g) in enumerate(marks):
        entry, qname, rel, a_start, a_end, a_loc = g
        a_start, a_end, a_loc = int(a_start), int(a_end), int(a_loc)
        body_end = marks[k + 1][0] if k + 1 < len(marks) else len(text)
        body = text[be:body_end].rstrip('\n')
        if not body.strip():
            problems.append('E%s %s: EMPTY body' % (entry, qname))
            continue

        auth = open(os.path.join(SRC, rel), errors='replace').read()
        alines = auth.split('\n')
        # independently re-derive the end from the banner's START line only
        end_idx, depth = scan_end(alines, a_start - 1)
        if end_idx is None:
            problems.append('E%s %s: authority def never closes from line %d '
                            '(depth left %d)' % (entry, qname, a_start, depth))
            continue
        my_end = end_idx + 1
        if my_end != a_end:
            problems.append('E%s %s: END MISMATCH banner says %d, I derive %d'
                            % (entry, qname, a_end, my_end))
        auth_slice = '\n'.join(alines[a_start - 1:my_end])

        # 1. LENGTH
        if len(body) != len(auth_slice):
            problems.append('E%s %s: LENGTH %d in %s vs %d in authority '
                            '(%s%d chars)' % (entry, qname, len(body), label,
                                              len(auth_slice),
                                              '+' if len(body) > len(auth_slice) else '',
                                              len(body) - len(auth_slice)))
        # 2. CONTENT
        if body != auth_slice:
            bl, al = body.split('\n'), auth_slice.split('\n')
            if len(bl) != len(al):
                problems.append('E%s %s: LINE COUNT %d vs %d'
                                % (entry, qname, len(bl), len(al)))
            for n, (x, y) in enumerate(zip(bl, al), a_start):
                if x != y:
                    problems.append('E%s %s: first differing line %d\n'
                                    '      %s: %r\n      auth   : %r'
                                    % (entry, qname, n, label, x, y))
                    break
        # 3. the body's own last line must CLOSE the function, and depth must
        #    reach zero exactly at the last character -- never by appending one.
        e2, d2 = scan_end(body.split('\n'), 0)
        if e2 is None:
            problems.append('E%s %s: body does not close (depth left %d) -- '
                            'TRUNCATED' % (entry, qname, d2))
        elif e2 != len(body.split('\n')) - 1:
            problems.append('E%s %s: body closes at its line %d but has %d '
                            'lines -- trailing junk' % (entry, qname, e2 + 1,
                                                        len(body.split('\n'))))
        last = [l for l in body.split('\n') if l.strip()][-1].strip()
        if not last.endswith('}') and not last.endswith('};'):
            problems.append('E%s %s: last line does not close: %r'
                            % (entry, qname, last))
        checked += 1
    return not problems, problems, checked


text = open(EXTRACT).read()
ok, problems, checked = verify(text)
print('=== VERIFY %s ===' % EXTRACT)
print('entries checked: %d' % checked)
if ok:
    print('PASS -- every body matches the authority in length and content, '
          'and closes its own function')
else:
    print('FAIL -- %d problem(s):' % len(problems))
    for p in problems:
        print('  ' + p)

# ---------------------------------------------------------------- controls ---
print('\n=== NEGATIVE CONTROLS (the check must FAIL on both) ===')
marks = [(m.start(), m.end()) for m in BANNER.finditer(text)]
# pick the largest body so the mutation is unambiguous
sizes = [(marks[i + 1][0] - marks[i][1] if i + 1 < len(marks)
          else len(text) - marks[i][1], i) for i in range(len(marks))]
_, big = max(sizes)
bs, be = marks[big]
bend = marks[big + 1][0] if big + 1 < len(marks) else len(text)
body = text[be:bend]

# control 1: drop the last 3 non-blank lines (bridge 2's exact failure)
bl = body.rstrip('\n').split('\n')
trunc = '\n'.join(bl[:-3]) + '\n\n'
t1 = text[:be] + trunc + text[bend:]
ok1, pr1, _ = verify(t1, 'truncated')
print('control 1 TRUNCATE 3 lines : %s' % ('FAILED AS REQUIRED' if not ok1
                                           else '!!! PASSED -- CHECK IS BLIND'))
if not ok1:
    print('   first problem: %s' % pr1[0].split('\n')[0])

# control 2: append a brace to "balance" -- the move that hid the damage
t2 = text[:be] + body.rstrip('\n') + '\n}\n\n' + text[bend:]
ok2, pr2, _ = verify(t2, 'brace-appended')
print('control 2 APPEND a brace   : %s' % ('FAILED AS REQUIRED' if not ok2
                                           else '!!! PASSED -- CHECK IS BLIND'))
if not ok2:
    print('   first problem: %s' % pr2[0].split('\n')[0])

sys.exit(0 if (ok and not ok1 and not ok2) else 1)
