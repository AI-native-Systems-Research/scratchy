#!/usr/bin/env python3
"""INDEPENDENT verification that campaign/cpp/{l3,ddc,ddl}.cpp hold each body VERBATIM.

⛔ WHY THIS FILE EXISTS. Bridge 2's extract silently truncated 366 of 384 bodies and its check
PASSED, because the check RE-SLICED THE SOURCE WITH THE SAME (file, line, length) THE EXTRACTOR HAD
USED -- it compared what was copied against what was copied. A spot check against its own
parameters verifies nothing.

So this file:
  * does NOT import cppscan (the extractor's only notion of "where does this function end").
    Its scanner below is a character state machine, a deliberately different algorithm;
  * re-derives EVERY end twice -- once in the extract, once in the authority file -- from the
    banner's citation alone, and NEVER from the recorded extract_lines;
  * cross-checks the recorded extract_lines against its own derivation, so the extractor's
    bookkeeping is evidence rather than an assumption;
  * compares LENGTHS first, then bytes;
  * asserts each body's last line closes the function and the depth reaches 0 exactly once, at the
    final character;
  * NEVER appends `}` to balance braces -- a body that does not balance is a FAILURE, not a thing
    to repair;
  * proves it bites, with two negative controls that must both be detected.

Exit status is non-zero if any unit fails or if either negative control goes undetected.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

CAMP = _paths.CAMP
AUTH_ROOT = _paths.AUTH_ROOT
WORK = _paths.WORK

# --- an independent scanner: a character state machine, not a regex sanitizer -----------------
CODE, LINE_C, BLOCK_C, STR, CHR, RAW = range(6)


def scan_match(text, start):
    """Index of the `}` matching the first `{` at or after `start`, and that `{`'s index.

    Braces inside comments, string literals, character literals, raw strings and preprocessor
    lines do not move the depth. Returns (open_idx, close_idx) or (-1, -1).
    """
    st = CODE
    depth = 0
    open_idx = -1
    i = start
    n = len(text)
    raw_close = ""
    at_line_start = True
    in_pp = False
    while i < n:
        c = text[i]
        if st == CODE:
            if at_line_start and c == "#":
                in_pp = True
            if c == "\n":
                at_line_start = True
                if in_pp and not (i and text[i - 1] == "\\"):
                    in_pp = False
                i += 1
                continue
            if not c.isspace():
                at_line_start = False
            if in_pp:
                i += 1
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "/":
                st = LINE_C
                i += 2
                continue
            if c == "/" and i + 1 < n and text[i + 1] == "*":
                st = BLOCK_C
                i += 2
                continue
            if c == "R" and i + 1 < n and text[i + 1] == '"':
                j = text.find("(", i + 2)
                if j != -1 and j - i < 34:
                    raw_close = ")" + text[i + 2:j] + '"'
                    st = RAW
                    i = j + 1
                    continue
            if c == '"':
                st = STR
                i += 1
                continue
            if c == "'":
                st = CHR
                i += 1
                continue
            if c == "{":
                if depth == 0:
                    open_idx = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return open_idx, i
                if depth < 0:
                    return -1, -1
            i += 1
            continue
        if st == LINE_C:
            if c == "\n":
                st = CODE
                at_line_start = True
            i += 1
            continue
        if st == BLOCK_C:
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                st = CODE
                i += 2
                continue
            i += 1
            continue
        if st == RAW:
            if text.startswith(raw_close, i):
                st = CODE
                i += len(raw_close)
                continue
            i += 1
            continue
        # STR / CHR
        if c == "\\":
            i += 2
            continue
        if (st == STR and c == '"') or (st == CHR and c == "'") or c == "\n":
            st = CODE
        i += 1
    return -1, -1


def depth_profile_ok(body):
    """True when the body's depth hits 0 exactly once, at its final character."""
    o, c = scan_match(body, 0)
    return o == 0 and c == len(body) - 1


BANNER = re.compile(
    r"^// unit: (?P<unit>e\d+_[A-Za-z0-9_]+)\n"
    r"^// authority: (?P<path>[^\s:]+):(?P<line>\d+)\n",
    re.M)
CLOSING_RULE = "\n// " + "-" * 96 + "\n"


def parse_extract(path):
    """Every entry in the extract, from its banner alone. extract_lines is read but NOT used to
    slice -- it is compared against this function's own derivation."""
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    starts = [0]
    for l in lines[:-1]:
        starts.append(starts[-1] + len(l) + 1)

    def line_of(off):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    entries = []
    for m in BANNER.finditer(text):
        end_of_banner = text.find(CLOSING_RULE, m.end())
        if end_of_banner == -1:
            raise SystemExit("no closing banner rule for %s" % m.group("unit"))
        sig_start = end_of_banner + len(CLOSING_RULE)
        o, c = scan_match(text, sig_start)
        entries.append({
            "unit": m.group("unit"),
            "auth_path": m.group("path"),
            "auth_line": int(m.group("line")),
            # a body that does not balance is recorded as UNBALANCED and reported as a failure.
            # ⛔ It is never repaired, and no `}` is ever appended to make it balance.
            "body": None if o == -1 else text[o:c + 1],
            "sig_line": line_of(sig_start),
            "close_line": None if c == -1 else line_of(c),
        })
    return entries, text


def authority_body(rel, line):
    p = os.path.join(AUTH_ROOT, rel)
    text = open(p, encoding="utf-8", errors="replace").read()
    lines = text.split("\n")
    off = sum(len(l) + 1 for l in lines[:line - 1])
    o, c = scan_match(text, off)
    if o == -1:
        return None
    return text[o:c + 1]


def check(entries, recorded):
    fails = []
    for e in entries:
        u = e["unit"]
        if e["body"] is None:
            fails.append((u, "body in the extract does not balance - TRUNCATED or corrupted"))
            continue
        auth = authority_body(e["auth_path"], e["auth_line"])
        if auth is None:
            fails.append((u, "authority body did not balance"))
            continue
        if len(auth) != len(e["body"]):
            fails.append((u, "LENGTH %d in extract vs %d in authority (delta %d)"
                          % (len(e["body"]), len(auth), len(e["body"]) - len(auth))))
            continue
        if auth != e["body"]:
            d = next(i for i in range(len(auth)) if auth[i] != e["body"][i])
            fails.append((u, "CONTENT differs at byte %d: %r vs %r"
                          % (d, e["body"][d:d + 40], auth[d:d + 40])))
            continue
        if not e["body"].rstrip().endswith("}"):
            fails.append((u, "last line does not close the function"))
            continue
        if not depth_profile_ok(e["body"]):
            fails.append((u, "brace depth does not return to 0 exactly at the final character"))
            continue
        rec = recorded.get(u)
        if rec and rec != "%d-%d" % (e["sig_line"], e["close_line"]):
            fails.append((u, "extract_lines recorded %s but derived %d-%d"
                          % (rec, e["sig_line"], e["close_line"])))
    return fails


def main():
    # UNITS.tsv columns: 0 unit .. 6 extract_lines .. 9 extract_file
    recorded, ex_file = {}, {}
    with open(CAMP + "/UNITS.tsv") as fh:
        next(fh)
        for row in fh:
            f = row.rstrip("\n").split("\t")
            recorded[f[0]] = f[6]
            ex_file[f[0]] = f[9]

    all_entries, fails, total_bytes = [], [], 0
    for scope in _paths.SCOPES:
        p = CAMP + "/cpp/%s.cpp" % scope
        entries, _ = parse_extract(p)
        all_entries += entries
        total_bytes += sum(len(e["body"]) for e in entries if e["body"])
        print("cpp/%-8s entries: %4d" % (scope + ".cpp", len(entries)))
        fails += check(entries, recorded)
        # every unit the TSV assigns to this TU must actually be in it
        want = {u for u, f in ex_file.items() if f.endswith("/%s.cpp" % scope)}
        got = {e["unit"] for e in entries}
        for u in sorted(want - got):
            fails.append((u, "UNITS.tsv assigns it to %s.cpp but it is not in that file" % scope))
        for u in sorted(got - want):
            fails.append((u, "present in %s.cpp but UNITS.tsv does not assign it there" % scope))

    print("entries in extracts: %d" % len(all_entries))
    print("rows in UNITS.tsv  : %d" % len(recorded))
    missing = set(recorded) - {e["unit"] for e in all_entries}
    extra = {e["unit"] for e in all_entries} - set(recorded)
    if missing or extra:
        print("⛔ UNITS.tsv/extract mismatch: missing %s extra %s"
              % (sorted(missing)[:5], sorted(extra)[:5]))
    print("body bytes compared: %d across %d units" % (total_bytes, len(all_entries)))
    if fails:
        print("⛔ %d FAILURES" % len(fails))
        for u, why in fails[:25]:
            print("   %s: %s" % (u, why))
    else:
        print("✅ every body is byte-identical to the authority at its cited line")

    # ---- negative controls: the verifier must BITE ------------------------------------------
    print("\n-- negative controls (each MUST be detected) --")
    os.makedirs(WORK, exist_ok=True)
    victim_scope = "ddc"
    src = CAMP + "/cpp/%s.cpp" % victim_scope
    text = open(src, encoding="utf-8").read()
    entries, _ = parse_extract(src)
    entries = [e for e in entries if e["body"]]
    proofs = []

    # (a) drop the last line of one body (a real closing-brace line) - the truncation that
    #     defeated bridge 2's check
    victim = max(entries, key=lambda e: len(e["body"]))
    lines = text.split("\n")
    cut = victim["close_line"] - 1
    pa = os.path.join(WORK, "neg_a.cpp")
    open(pa, "w").write("\n".join(lines[:cut] + lines[cut + 1:]))
    fa = check(parse_extract(pa)[0], {})
    hit_a = any(u == victim["unit"] for u, _ in fa)
    proofs.append(("(a) last line of %s removed" % victim["unit"], hit_a,
                   [w for u, w in fa if u == victim["unit"]][:1]))

    # (b) blank a `}` in the middle of another body: the depth then closes early, so the body the
    #     verifier derives is SHORTER than the authority's
    victim2 = sorted(entries, key=lambda e: -len(e["body"]))[1]
    i = text.index(victim2["body"])
    inner = victim2["body"].rindex("}", 0, len(victim2["body"]) - 1)
    pb = os.path.join(WORK, "neg_b.cpp")
    open(pb, "w").write(text[:i + inner] + " " + text[i + inner + 1:])
    fb = check(parse_extract(pb)[0], {})
    hit_b = any(u == victim2["unit"] for u, _ in fb)
    proofs.append(("(b) an inner `}` of %s blanked" % victim2["unit"], hit_b,
                   [w for u, w in fb if u == victim2["unit"]][:1]))

    # record the numbers the agent-facing configs quote, so they can never go stale
    import json as _json
    _json.dump({"units": len(all_entries), "bytes": total_bytes,
                "failures": len(fails),
                "negative_controls_detected": sum(1 for _w, h, _y in proofs if h),
                "negative_controls": len(proofs)},
               open(os.path.join(WORK, "verify-stats.json"), "w"), indent=1)

    ok = True
    for what, hit, why in proofs:
        print("   %s  %s%s" % ("DETECTED" if hit else "⛔ NOT DETECTED", what,
                               (" -> %s" % why[0]) if why else ""))
        ok = ok and hit
    if not ok:
        print("⛔ A NEGATIVE CONTROL WENT UNDETECTED - this verifier does not bite. Do not trust it.")
    sys.exit(0 if (not fails and ok) else 1)


if __name__ == "__main__":
    main()
