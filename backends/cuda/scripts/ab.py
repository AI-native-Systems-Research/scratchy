"""A/B of two `cargo test --nocapture` logs: the ratio= / mismatched_bits= lines per (binary, test).
Timing text and host-timed bandwidth lines are dropped (they vary run to run)."""
import re
import sys
from collections import Counter, defaultdict


def parse(path):
    binary, test = "?", "?"
    out = defaultdict(Counter)
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"\s*Running (?:unittests )?(\S+)", line)
        if m:
            binary = m.group(1)
        m = re.match(r"test (\S+) \.\.\.", line)
        if m:
            test = m.group(1)
        if "ratio=" not in line and "mismatched_bits=" not in line:
            continue
        if "TB/s" in line or "median=" in line or "host-timed" in line:
            continue
        text = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
        text = re.sub(r"\s+\d+(\.\d+)?\s*(ms|us|µs)\b", "", text)
        out[(binary, test)][text] += 1
    return out


base, pr = parse(sys.argv[1]), parse(sys.argv[2])
keys = sorted(set(base) | set(pr))
same = [k for k in keys if k in base and k in pr and base[k] == pr[k]]
only_base = [k for k in keys if k in base and k not in pr]
only_pr = [k for k in keys if k in pr and k not in base]
differ = [k for k in keys if k in base and k in pr and base[k] != pr[k]]
nb = sum(sum(c.values()) for c in base.values())
np_ = sum(sum(c.values()) for c in pr.values())
print(f"base: {len(base)} tests with {nb} lines; pr: {len(pr)} tests with {np_} lines")
print(f"identical tests: {len(same)}; differing tests: {len(differ)}; tests only at base: {len(only_base)}; tests only at the PR: {len(only_pr)}")
for k in only_base:
    print("ONLY AT BASE:", k)
for k in only_pr:
    print("ONLY AT PR:", k, f"({sum(pr[k].values())} lines)")
non_gemv = [k for k in differ if "gemv" not in k[1].lower()]
print(f"differing tests whose name has 'gemv': {len(differ) - len(non_gemv)}; without: {len(non_gemv)}")
for k in differ:
    print("DIFFERS:", k)
    for line in sorted((base[k] - pr[k]).elements()):
        print("   base:", line[:400])
    for line in sorted((pr[k] - base[k]).elements()):
        print("   pr:  ", line[:400])
