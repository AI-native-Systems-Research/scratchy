#!/usr/bin/env python3
"""INDEPENDENT verification of cpp/bridge1.cpp against the authority tree.

⛔ THE FAILURE THIS EXISTS TO CATCH. Bridge 2's extract truncated 366 of 384 bodies and its check
reported "verbatim, 12 sampled, none differed" -- because the check re-sliced the source with the
SAME (file, line, length) the extractor had used. It compared what was copied against what was
copied. A check that re-slices with the length under test is a tautology.

So this file:
  * does NOT import cppscan -- it carries its own, differently-written scanner;
  * takes ONLY (file, head line) from each banner and re-derives the function's END itself;
  * compares LENGTHS (lines and non-whitespace characters) and then full CONTENT;
  * separately asserts each extracted body's last non-blank line actually closes the function,
    and that the extract was not brace-balanced by appending `}`.
"""
import pathlib
import re
import sys

SRC = pathlib.Path("/Users/nickm/git/deeptools-src")
EXTRACT = pathlib.Path("/tmp/bridge1-setup/cpp/bridge1.cpp")

BANNER = re.compile(
    r"^// ---- (\d+)/(\d+)\s+(\S+)\s+--\s+(\S+):(\d+)\s+\((\d+)L\)"
)


# ---------------------------------------------------------------- independent scanner
class Scan:
    """A character state machine. Deliberately not cppscan: different structure, same job."""

    CODE, LINE_C, BLOCK_C, STR, CHR = range(5)

    def __init__(self, text):
        self.text = text

    def code_mask(self):
        """True for every index that is real code (not comment, string, char or a `#` line)."""
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
                if st in (self.LINE_C,):
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
                    # raw string?
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
        return mask

    def span_from(self, line_no):
        """(open_off, close_off) of the body of the definition starting at 1-based `line_no`."""
        t = self.text
        mask = self.code_mask()
        offs = [0]
        for i, c in enumerate(t):
            if c == "\n":
                offs.append(i + 1)
        start = offs[line_no - 1]
        # first code `{` at or after `start`
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


def main():
    ex = EXTRACT.read_text()
    ex_lines = ex.split("\n")
    banners = []
    for idx, l in enumerate(ex_lines):
        m = BANNER.match(l)
        if m:
            banners.append((idx + 1, m))

    cache = {}
    bad_len = []
    bad_content = []
    bad_close = []
    padded = []
    missing = []
    ok = 0

    for pos, m in banners:
        entry, total, name, rel, hl, loc = (
            int(m.group(1)), int(m.group(2)), m.group(3), m.group(4),
            int(m.group(5)), int(m.group(6)),
        )
        path = SRC / rel
        if rel not in cache:
            cache[rel] = Scan(path.read_text(errors="ignore"))
        sc = cache[rel]
        span = sc.span_from(hl)
        if span is None:
            missing.append((entry, name, rel, hl))
            continue
        o, c = span
        auth_body = sc.text[o : c + 1]
        auth_lines = line_of(sc.text, c) - line_of(sc.text, o) + 1
        auth_decl = line_of(sc.text, c) - hl + 1

        # the extract's body: from the `{` on the signature line after the banner, to the last line
        # before the next banner / level rule / EOF -- found WITHOUT using `loc`.
        j = pos  # 0-based index of the line after the banner is pos (banner is 1-based at pos)
        # locate the body open brace in the extract
        k = j
        while k < len(ex_lines) and "{" not in ex_lines[k]:
            k += 1
        if k >= len(ex_lines):
            missing.append((entry, name, rel, hl))
            continue
        seg_start = k
        # brace-match forward in the extract with its own scanner
        seg_text = "\n".join(ex_lines[seg_start:])
        sc2 = Scan(seg_text)
        sp2 = sc2.span_from(1)
        if sp2 is None:
            bad_close.append((entry, name, "extract body never closes"))
            continue
        o2, c2 = sp2
        ex_body = seg_text[o2 : c2 + 1]
        ex_lines_n = seg_text.count("\n", o2, c2) + 1

        if ex_lines_n != auth_lines or len(norm(ex_body)) != len(norm(auth_body)):
            bad_len.append((entry, name, f"{rel}:{hl}", auth_lines, ex_lines_n,
                            len(norm(auth_body)), len(norm(ex_body))))
            continue
        if norm(ex_body) != norm(auth_body):
            bad_content.append((entry, name, f"{rel}:{hl}"))
            continue
        # last non-blank line of the extracted body must close the function
        tail = [l for l in ex_body.split("\n") if l.strip()]
        if not tail or "}" not in tail[-1]:
            bad_close.append((entry, name, "last body line does not close"))
            continue
        # brace-padding check: the authority's own last line must match the extract's
        auth_tail = [l for l in auth_body.split("\n") if l.strip()][-1]
        if auth_tail.strip() != tail[-1].strip():
            padded.append((entry, name, auth_tail.strip(), tail[-1].strip()))
            continue
        if loc != auth_decl:
            bad_len.append((entry, name, f"{rel}:{hl}", f"banner LoC {loc}", f"authority {auth_decl}",
                            "", ""))
            continue
        ok += 1

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
    print(f"\nTRUNCATION / MISMATCH TOTAL: {bad} of {len(banners)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
