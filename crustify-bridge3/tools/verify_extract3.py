#!/usr/bin/env python3
"""INDEPENDENT verification of cpp/bridge3.cpp against the authority tree.

⛔ THE FAILURE THIS EXISTS TO CATCH. Bridge 2's extract truncated 366 of 384 bodies and its check
reported "verbatim, 12 sampled, none differed" -- because the check re-sliced the source with the
SAME (file, line, length) the extractor had used. It compared what was copied against what was
copied. A check that re-slices with the length under test is a tautology.

So this file:
  * does NOT import cppscan -- it carries its own, differently-written scanner;
  * takes ONLY (file, head line) from each banner and re-derives the function's END itself;
  * finds each extracted body's end in the EXTRACT by its own brace matching, never from the
    banner's `(NNNL)`;
  * compares LENGTHS (lines and non-whitespace characters) and then full CONTENT;
  * separately asserts each extracted body's last non-blank line actually closes the function, and
    that the extract was not brace-balanced by appending `}`.

Run with --self-test to execute two negative controls that MUST both be caught.
"""
import pathlib
import re
import sys

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
EXTRACT = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3/cpp/bridge3.cpp")

BANNER = re.compile(r"^// ---- (\d+)/(\d+)\s+(\S+)\s+--\s+(\S+):(\d+)\s+\((\d+)L\)")


class Scan:
    """A character state machine. Deliberately not cppscan: different structure, same job."""

    CODE, LINE_C, BLOCK_C, STR, CHR = range(5)

    def __init__(self, text):
        self.text = text
        self._mask = None

    def code_mask(self):
        """True for every index that is real code (not comment, string, char or a `#` line)."""
        if self._mask is not None:
            return self._mask
        t = self.text
        n = len(t)
        mask = bytearray(n)
        st = self.CODE
        i = 0
        at_line_start = True
        in_pp = False
        while i < n:
            c = t[i]
            if c == "\n":
                in_pp = False
                at_line_start = True
                if st == self.LINE_C:
                    st = self.CODE
                i += 1
                continue
            if at_line_start and st == self.CODE:
                if c in " \t":
                    i += 1
                    continue
                at_line_start = False
                if c == "#":
                    in_pp = True
            if in_pp:
                i += 1
                continue
            if st == self.CODE:
                if c == "/" and i + 1 < n and t[i + 1] == "/":
                    st = self.LINE_C
                    i += 2
                    continue
                if c == "/" and i + 1 < n and t[i + 1] == "*":
                    st = self.BLOCK_C
                    i += 2
                    continue
                if c == '"':
                    if i > 0 and t[i - 1] == "R":
                        j = t.find("(", i)
                        if j != -1 and j - i < 32:
                            close = ")" + t[i + 1 : j] + '"'
                            k = t.find(close, j)
                            i = n if k == -1 else k + len(close)
                            continue
                    st = self.STR
                    i += 1
                    continue
                if c == "'":
                    st = self.CHR
                    i += 1
                    continue
                mask[i] = 1
                i += 1
                continue
            if st == self.BLOCK_C:
                if c == "*" and i + 1 < n and t[i + 1] == "/":
                    st = self.CODE
                    i += 2
                    continue
                i += 1
                continue
            if st in (self.STR, self.CHR):
                if c == "\\":
                    i += 2
                    continue
                if (st == self.STR and c == '"') or (st == self.CHR and c == "'"):
                    st = self.CODE
                i += 1
                continue
            i += 1
        self._mask = mask
        return mask

    def line_starts(self):
        offs = [0]
        for i, c in enumerate(self.text):
            if c == "\n":
                offs.append(i + 1)
        return offs

    def span_from(self, line_no):
        """(open_off, close_off) of the body of the definition starting at 1-based `line_no`."""
        t = self.text
        mask = self.code_mask()
        offs = self.line_starts()
        if line_no - 1 >= len(offs):
            return None
        start = offs[line_no - 1]
        o = -1
        for i in range(start, len(t)):
            if t[i] == "{" and mask[i]:
                o = i
                break
        if o == -1:
            return None
        d = 0
        for i in range(o, len(t)):
            if not mask[i]:
                continue
            if t[i] == "{":
                d += 1
            elif t[i] == "}":
                d -= 1
                if d == 0:
                    return (o, i)
        return None


def line_of(text, off):
    return text.count("\n", 0, off) + 1


def norm(s):
    return re.sub(r"\s+", "", s)


def verify(extract_text):
    ex_lines = extract_text.split("\n")
    banners = []
    for idx, l in enumerate(ex_lines):
        m = BANNER.match(l)
        if m:
            banners.append((idx + 1, m))

    cache = {}
    bad_len, bad_content, bad_close, padded, missing = [], [], [], [], []
    ok = 0

    for pos, m in banners:
        entry, name, rel, hl, loc = (int(m.group(1)), m.group(3), m.group(4),
                                     int(m.group(5)), int(m.group(6)))
        if rel not in cache:
            cache[rel] = Scan((SRC / rel).read_text(errors="ignore"))
        sc = cache[rel]
        span = sc.span_from(hl)
        if span is None:
            missing.append((entry, name, rel, hl))
            continue
        o, c = span
        auth_body = sc.text[o : c + 1]
        auth_lines = line_of(sc.text, c) - line_of(sc.text, o) + 1
        auth_decl = line_of(sc.text, c) - hl + 1

        # The extract's body: found WITHOUT using `loc` -- brace-match from the line after the
        # banner with this file's own scanner.
        seg_text = "\n".join(ex_lines[pos:])
        sc2 = Scan(seg_text)
        sp2 = sc2.span_from(1)
        if sp2 is None:
            bad_close.append((entry, name, "extract body never closes"))
            continue
        o2, c2 = sp2
        ex_body = seg_text[o2 : c2 + 1]
        ex_lines_n = seg_text.count("\n", o2, c2) + 1

        if ex_lines_n != auth_lines or len(norm(ex_body)) != len(norm(auth_body)):
            bad_len.append((entry, name, f"{rel}:{hl}", f"auth {auth_lines}L/{len(norm(auth_body))}c",
                            f"extract {ex_lines_n}L/{len(norm(ex_body))}c"))
            continue
        if norm(ex_body) != norm(auth_body):
            bad_content.append((entry, name, f"{rel}:{hl}"))
            continue
        tail = [l for l in ex_body.split("\n") if l.strip()]
        if not tail or "}" not in tail[-1]:
            bad_close.append((entry, name, "last body line does not close"))
            continue
        auth_tail = [l for l in auth_body.split("\n") if l.strip()][-1]
        if auth_tail.strip() != tail[-1].strip():
            padded.append((entry, name, auth_tail.strip(), tail[-1].strip()))
            continue
        if loc != auth_decl:
            bad_len.append((entry, name, f"{rel}:{hl}", f"banner LoC {loc}",
                            f"authority {auth_decl}"))
            continue
        ok += 1
    return banners, ok, bad_len, bad_content, bad_close, padded, missing


def report(extract_text, quiet=False):
    banners, ok, bad_len, bad_content, bad_close, padded, missing = verify(extract_text)
    if not quiet:
        print(f"banners found: {len(banners)}")
        print(f"VERIFIED verbatim + untruncated: {ok}")
        for label, rows in (("LENGTH MISMATCH", bad_len), ("CONTENT MISMATCH", bad_content),
                            ("DOES NOT CLOSE", bad_close), ("BRACE-PADDED", padded),
                            ("NOT FOUND IN AUTHORITY", missing)):
            if rows:
                print(f"\n{label}: {len(rows)}")
                for r in rows[:25]:
                    print("   ", r)
    bad = len(bad_len) + len(bad_content) + len(bad_close) + len(padded) + len(missing)
    if not quiet:
        print(f"\nTRUNCATION / MISMATCH TOTAL: {bad} of {len(banners)}")
    return bad, len(banners)


def self_test(extract_text):
    """⭐ TWO NEGATIVE CONTROLS. A verifier nobody has seen fail proves nothing."""
    lines = extract_text.split("\n")
    fails = 0

    # control 1: cut a function's tail and re-balance it with a bare `}` -- bridge 2's exact failure
    b = next(i for i, l in enumerate(lines) if BANNER.match(l))
    end = next(i for i in range(b + 8, len(lines)) if lines[i].startswith("}"))
    c1 = lines[: end - 2] + ["}"] + lines[end + 1 :]
    bad, _ = report("\n".join(c1), quiet=True)
    print(f"  control 1 (tail cut, re-balanced with a bare }}): {'CAUGHT' if bad else 'MISSED'}")
    fails += bad == 0

    # control 2: delete a brace-balanced run from the MIDDLE of a body
    body = lines[b + 1 : end]
    mid = len(body) // 2
    cut = [l for l in body[mid : mid + 4] if l.count("{") == l.count("}")]
    if len(cut) == 4:
        c2 = lines[: b + 1 + mid] + lines[b + 1 + mid + 4 :]
        bad, _ = report("\n".join(c2), quiet=True)
        print(f"  control 2 (brace-neutral 4-line cut from the middle): "
              f"{'CAUGHT' if bad else 'MISSED'}")
        fails += bad == 0
    else:
        # fall back to a single brace-neutral line so the control always runs
        for k in range(mid, end - b - 2):
            if body[k].count("{") == body[k].count("}"):
                c2 = lines[: b + 1 + k] + lines[b + 2 + k :]
                bad, _ = report("\n".join(c2), quiet=True)
                print(f"  control 2 (brace-neutral 1-line cut from the middle): "
                      f"{'CAUGHT' if bad else 'MISSED'}")
                fails += bad == 0
                break
    return fails


if __name__ == "__main__":
    text = EXTRACT.read_text()
    bad, n = report(text)
    rc = 1 if bad else 0
    if "--self-test" in sys.argv:
        print("\nNEGATIVE CONTROLS (both must be CAUGHT):")
        if self_test(text):
            print("  ⛔ A CONTROL WAS MISSED -- this verifier does not bite")
            rc = 2
    sys.exit(rc)
