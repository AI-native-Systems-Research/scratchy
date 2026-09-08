#!/usr/bin/env python3
"""Comment- and string-aware C++ scanning for the bridge-1 extraction.

Bridge 2's extract silently truncated 366 of 384 bodies because its brace matching counted
`{`/`}` characters in RAW text -- braces inside string literals and comments moved the depth --
and the check that was supposed to catch it re-sliced the source with the SAME (file, line,
length) the extractor had used. This module is the extractor's ONE notion of "where does this
function end". `verify_extract.py` deliberately does NOT import it: the verifier re-derives every
end with its own independent matcher and compares LENGTHS.

`sanitize()` returns a string of exactly the same length as its input, with the contents of
comments, string literals and character literals replaced by spaces (newlines preserved), so byte
offsets and line numbers are identical between raw and sanitized text.
"""

import re

_SPECIFIER = r"(?:const|noexcept|override|final|volatile|constexpr|mutable)"
_CONTROL = {
    "if", "for", "while", "switch", "catch", "do", "else", "return", "try", "case", "default",
    "sizeof", "static_assert", "decltype", "typedef", "using", "and", "or", "not", "alignof",
    "new", "delete", "throw", "namespace", "class", "struct", "union", "enum", "template",
    "operator", "explicit", "static", "inline", "virtual", "extern",
}
# what may sit between the parameter list's `)` and the body's `{`
_POST_OK = re.compile(
    r"\s*(?:" + _SPECIFIER + r"\s*)*"
    r"(?:noexcept\s*\([^()]*\)\s*)?"
    r"(?:throw\s*\([^()]*\)\s*)?"
    r"(?:->\s*[^;{]*)?"
    r"(?::\s*[^;{]*)?\s*$"
)
_QUAL_TAIL = re.compile(
    r"((?:[A-Za-z_][A-Za-z_0-9]*\s*(?:<[^<>]*>)?\s*::\s*)*)([A-Za-z_~][A-Za-z_0-9]*)\s*$"
)
_LAMBDA_NAME = re.compile(r"\b(?:auto|const\s+auto)\s*&?\s*([A-Za-z_][A-Za-z_0-9]*)\s*=\s*$")


def sanitize(text):
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "R" and i + 1 < n and text[i + 1] == '"':
            j = text.find("(", i + 2)
            if j != -1 and j - i < 32:
                close = ")" + text[i + 2 : j] + '"'
                k = text.find(close, j + 1)
                end = n if k == -1 else k + len(close)
                for p in range(i, min(end, n)):
                    if out[p] != "\n":
                        out[p] = " "
                i = end
                continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j == -1 else j
            for p in range(i, j):
                out[p] = " "
            i = j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            for p in range(i, j):
                if out[p] != "\n":
                    out[p] = " "
            i = j
            continue
        if c in ('"', "'"):
            quote, j = c, i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote or text[j] == "\n":
                    break
                j += 1
            end = min(j + 1, n)
            for p in range(i, end):
                if out[p] != "\n":
                    out[p] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def strip_preprocessor(san):
    """Blank whole preprocessor lines: an `#if 0` block's braces must not move the depth."""
    return "\n".join(
        " " * len(l) if l.lstrip().startswith("#") else l for l in san.split("\n")
    )


def prepared(text):
    return strip_preprocessor(sanitize(text))


def match_brace(san, open_idx):
    depth = 0
    for i in range(open_idx, len(san)):
        if san[i] == "{":
            depth += 1
        elif san[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _match_paren_back(san, close_idx):
    depth = 0
    for i in range(close_idx, -1, -1):
        if san[i] == ")":
            depth += 1
        elif san[i] == "(":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _head_start(san, idx):
    """Start of the declaration containing idx: past the previous `;`, `{`, `}` or access label."""
    best = 0
    for i in range(idx - 1, -1, -1):
        if san[i] in ";{}":
            best = i + 1
            break
    seg = san[best:idx]
    last = None
    for m in re.finditer(r"\b(?:public|private|protected)\s*:", seg):
        last = m
    if last:
        best += last.end()
    return best


class LineMap:
    def __init__(self, san):
        self.starts = [0] + [i + 1 for i, c in enumerate(san) if c == "\n"]

    def line_of(self, off):
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1


def find_defs(text, want_lambdas=False):
    """Every function definition in `text`, in source order, non-overlapping.

    Each dict carries 1-based `head_line` (first line of the signature), `open_line` (the line
    holding the body's `{`), `close_line` (the line holding the matching `}`), and
    `body_lines` = close_line - open_line + 1.

    A `{` that is not classified as a function body is scanned INTO, so member functions inside a
    class body and functions inside a namespace are found without any container bookkeeping.
    """
    san = prepared(text)
    lm = LineMap(san)
    defs = []
    i, n = 0, len(san)
    while i < n:
        if san[i] != "{":
            i += 1
            continue
        hs = _head_start(san, i)
        head = san[hs:i]

        hit = None
        # candidate parameter-list closers, LEFT to RIGHT: the leftmost `)` whose tail is only
        # specifiers / `-> type` / a constructor initializer list is the parameter list.
        for m in re.finditer(r"\)", head):
            c = m.start()
            if not _POST_OK.fullmatch(head[c + 1 :]):
                continue
            op = _match_paren_back(san, hs + c)
            if op == -1 or op < hs:
                continue
            before = head[: op - hs]
            qm = _QUAL_TAIL.search(before)
            if qm:
                qual = re.sub(r"\s+", "", qm.group(1))
                name = qm.group(2)
                lead = before[: qm.start(2)].strip()
                if name in _CONTROL and not qual:
                    continue
                if not lead and not qual:
                    continue
                if re.search(r"[=,?]\s*$", lead) or lead.endswith("&&") or lead.endswith("||"):
                    continue
                hit = (qual, name, op)
                break
            if want_lambdas:
                lm2 = re.search(r"\[[^\]]*\]\s*$", before)
                if lm2:
                    nm = _LAMBDA_NAME.search(before[: lm2.start()])
                    if nm:
                        hit = ("", nm.group(1), op)
                        break

        if hit is None:
            i += 1
            continue

        close = match_brace(san, i)
        if close == -1:
            i += 1
            continue
        qual, name, op = hit
        head_abs = hs + (len(head) - len(head.lstrip()))
        defs.append(
            {
                "qual": qual,
                "name": name,
                "head_off": head_abs,
                "head_line": lm.line_of(head_abs),
                "open_off": i,
                "open_line": lm.line_of(i),
                "close_off": close,
                "close_line": lm.line_of(close),
                "head": text[head_abs:i].strip(),
                "body": text[i : close + 1],
            }
        )
        i = close + 1
    for d in defs:
        d["body_lines"] = d["close_line"] - d["open_line"] + 1
        d["decl_lines"] = d["close_line"] - d["head_line"] + 1
    return defs


def calls_in(text):
    return set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*\(", prepared(text)))
