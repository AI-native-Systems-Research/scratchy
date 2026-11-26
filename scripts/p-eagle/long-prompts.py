# Two long copy-heavy prompts for the drafter-window test, from this repo's own files: a stretch of
# `mtp.rs` and a rewrite of a function in it; the README and a verbatim-quote request.
# Usage: python long-prompts.py <repo root> <out dir>
import os, sys

root, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
mtp = open(f"{root}/crates/serving/engine/src/spec_decode/mtp.rs").read().splitlines()
code = "\n".join(mtp[539:800]) + (
    "\n\nRewrite the function `chain` above exactly as it is written, adding a one-line comment "
    "above each statement. Output only the code.\n"
)
readme = open(f"{root}/README.md").read()[:18000]
prose = readme + (
    "\n\nQuote, verbatim and in order, every sentence above that says how to scope or narrow a "
    "build to one model. Output only the quotes.\n"
)
for name, text in [("code", code), ("prose", prose)]:
    open(f"{out}/{name}.txt", "w").write(text)
    print(name, len(text), "chars")
