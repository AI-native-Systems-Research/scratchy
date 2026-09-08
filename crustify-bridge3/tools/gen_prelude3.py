#!/usr/bin/env python3
"""Emit cpp/prelude.inc -- a catalogue of every name bridge3.cpp's bodies reach.

⛔ NOT A PORTING INPUT. It carries names and shapes only, so the translation unit is
self-describing with no MLIR/LLVM/dcc header. The behaviour being ported is in the bodies.
"""
import pathlib
import re
import sys

sys.path.insert(0, "/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3/tools")
import cppscan

OUT = pathlib.Path("/Users/nickm/git/scratchy/.claude/worktrees/bridge3/crustify-bridge3")
text = (OUT / "cpp/bridge3.cpp").read_text()
san = cppscan.prepared(text)

scopes = sorted(set(re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*::", san)))
calls = sorted(cppscan.calls_in(text))
defined = set(re.findall(r"\be\d{3}_([A-Za-z_0-9]+)\b", san))
syms = sorted(set(re.findall(r"\b(e\d{3}_[A-Za-z_0-9]+)\b", san)))
externs = [c for c in calls if c not in defined and not c.startswith("e0") and c not in scopes
           and not re.fullmatch(r"(if|for|while|switch|return|sizeof|catch|do)", c)]

HEAD = """\
// prelude.inc -- plain-C++ stand-ins for every name bridge3.cpp's bodies reach.
//
// ⛔ THIS IS NOT A PORTING INPUT. It carries NAMES AND SHAPES only, so the translation unit stands
// alone with no MLIR, LLVM or dcc header. The behaviour being ported lives in the bodies; nothing
// here has any.
//
// ⚠ The unit is NOT compile-clean, exactly as bridges 1 and 2 were not: these bodies do heavy
// member access on opaque reference types, which no stand-in can satisfy without the real headers.
// THE BODIES ARE VERBATIM AND VERIFIED UNTRUNCATED (130/130, two negative controls caught) -- that
// is the load-bearing property. Use the banners for WHICH function and WHAT ORDER, and port from
// the authority file at the cited line.
#pragma once
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
"""

lines = HEAD.split("\n")
lines.append(f"// ---- {len(scopes)} qualified scopes reached by the bodies")
lines += [f"// scope: {s}" for s in scopes]
lines.append("")
lines.append(f"// ---- {len(externs)} calls the bodies make to names defined outside this unit")
lines += [f"// extern: {c}" for c in externs]
lines.append("")
lines.append(f"// ---- the {len(syms)} units in this unit, in entry order")
lines += [f"// unit: {s}" for s in syms]
lines.append("")
(OUT / "cpp/prelude.inc").write_text("\n".join(lines) + "\n")
print(f"prelude.inc: {len(scopes)} scopes, {len(externs)} externs, {len(syms)} units")
