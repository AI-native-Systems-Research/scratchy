#!/usr/bin/env bash
# THE 13-SECOND BATCHED-ATTENTION GATE. Self-contained: it builds its own probe.
#
# ⛔⛔⛔ WHY THIS EXISTS. The batch fault that this branch spent weeks hunting in the SuperDSC emitter
# reproduces on METAL — a different worker, different kernels, no Spyre code in the path — in ~13 seconds
# on a laptop. Root cause (2026-08-11): `RopeOnce{Nax,Steel,GqaShared}` stages K into a scratch with NO
# batch dimension using a hard-coded `seq_idx = 0` (attention.metal ~1263 and the two mlx_steel_attn paged
# headers), so ONE K image built from batch row 0 is served to every row of the batch. A decode sharing
# the step IS row 0 and looks fine; every seq_idx > 0 attends row 0's keys and returns fluent, wrong text.
#
# CURRENT: 1/12. TARGET: 12/12. The fix goes at the `roped_k_scratch` selection in
# interpreter/metal/lowering.rs — the sdpa-paged path (its `else`) already indexes each request's own
# block-table row and is correct for batches.
#
# ⭐⭐ THE PROBE IS DESIGNED AGAINST THREE CONFOUNDS THAT EACH COST A WRONG CONCLUSION TODAY:
#  1. PREFIX CACHING. The original probe was the same filler text at four lengths, so every request but
#     the first reused another's blocks and the cache — not the batch — explained the failures. Here each
#     prompt draws from a DISJOINT sentence bank; the longest shared prefix between any two is 0 chars.
#     The run asserts the cache reports 0 hits.
#  2. A SUBSTRING ORACLE IS NOT AN ORACLE. "2 prefills in one step scored 2/2" was a FALSE PASS: the
#     answer was "the capital of the United States is Tokyo" — wrong content containing the searched
#     word. This scores the FIRST WORD of the continuation only, which is where the capital must appear.
#  3. ONE REQUEST PER STEP HIDES IT. Twelve short prompts overflow the 2048-token budget by COUNT, not
#     length, so the scheduler must put prefills in a step that also carries a decode — which is the
#     failing configuration. Equal or ragged lengths make no difference; only that mixing does.
#
# The control is built in: `--max-num-seqs 1` must score 12/12 on the same file and binary. If the
# control ever fails, the probe or the model is broken, not the batching — do not read the batched number.
set -uo pipefail
MODEL="${MODEL:-mlx-community/granite-3.3-8b-instruct-4bit}"
BIN="${BIN:-./target/release/scr}"
OUT="${OUT:-$(mktemp -d)}"
[ -x "$BIN" ] || { echo "⛔ no binary at $BIN — cargo build --release --features metal,model/granite-3.3-8b-instruct --bin scr"; exit 2; }

python3 - "$OUT" <<'PY'
import json, sys
out = sys.argv[1]
# (country, capital, disjoint filler bank) — banks share no leading text, so no cross-request prefix hit.
rows = [
    ("France","Paris","Copper conducts electricity. Glass is made from sand. "),
    ("Japan","Tokyo","Whales migrate each winter. Bamboo grows quickly. "),
    ("Italy","Rome","Granite forms underground. Owls hunt after dusk. "),
    ("Spain","Madrid","Deserts cool at night. Ants follow scent trails. "),
    ("Portugal","Lisbon","Yeast makes bread rise. Coral builds slowly. "),
    ("Greece","Athens","Magnets attract iron. Steam rises from kettles. "),
    ("Egypt","Cairo","Frost forms on clear nights. Kelp anchors to rock. "),
    ("Kenya","Nairobi","Tin resists corrosion. Amber preserves insects. "),
    ("Norway","Oslo","Moss prefers shade. Wool insulates well. "),
    ("Poland","Warsaw","Tides follow the moon. Lightning precedes thunder. "),
    ("Cuba","Havana","Clay hardens when fired. Salt dissolves in water. "),
    ("Peru","Lima","Bees pollinate flowers. Iron rusts in damp air. "),
]
want, jsonl = {}, []
for i, (country, cap, bank) in enumerate(rows):
    cid = f"c{i:02d}_{country}"
    want[cid] = cap
    jsonl.append({"custom_id": cid, "method": "POST", "url": "/v1/completions",
                  "body": {"prompt": (bank * 30)[:750] + f"The capital of {country} is",
                           "max_tokens": 8, "temperature": 0.0}})
with open(f"{out}/probe.jsonl", "w") as f:
    for r in jsonl:
        f.write(json.dumps(r) + "\n")
json.dump(want, open(f"{out}/want.json", "w"))
# Fail loudly rather than measure a confounded probe. The bound is ONE BLOCK: the prefix cache is
# block-granular (16 tokens), so a couple of shared characters cannot produce a hit, but a shared
# leading BLOCK can — and that is exactly what confounded the original probe.
import os
ps = [r["body"]["prompt"] for r in jsonl]
worst = max((len(os.path.commonprefix([a, b])) for i, a in enumerate(ps) for b in ps[i+1:]), default=0)
BLOCK_CHARS = 16 * 3  # 16-token block, ~3 chars/token — deliberately conservative
assert worst < BLOCK_CHARS, f"prompts share a {worst}-char prefix (>= ~1 block) — prefix caching would confound this"
print(f"probe: 12 prompts, ~{len(ps[0])} chars each, longest shared prefix {worst} chars (< 1 block)")
PY

run() { # $1 = label, rest = extra scr args
  local label="$1"; shift
  "$BIN" batch -m "$MODEL" --device metal "$@" \
      -i "$OUT/probe.jsonl" -o "$OUT/out_$label.jsonl" >"$OUT/log_$label" 2>&1
  local hits
  hits=$(grep -c "blocks hit, [1-9]" "$OUT/log_$label" 2>/dev/null || true)
  [ "${hits:-0}" -eq 0 ] || echo "  ⚠️ $label: $hits request(s) got a PREFIX CACHE HIT — result is confounded"
  python3 - "$OUT" "$label" <<'PY'
import json, sys, re
out, label = sys.argv[1], sys.argv[2]
want = json.load(open(f"{out}/want.json"))
ok = bad = 0
for line in open(f"{out}/out_{label}.jsonl"):
    r = json.loads(line)
    cid = r["custom_id"]
    text = r["response"]["body"]["choices"][0]["text"]
    # FIRST WORD ONLY — the capital must be the continuation, not merely present somewhere in it.
    first = next(iter(re.findall(r"[A-Za-z]+", text)), "")
    if first == want[cid]:
        ok += 1
    else:
        bad += 1
        print(f"    FAIL {cid:14s} want={want[cid]:9s} got={first!r}  {text[:34]!r}")
print(f"  {label} = {ok}/{ok+bad}")
PY
}

echo "== batched (the gate; currently 1/12, target 12/12)"
run batched
echo "== control: --max-num-seqs 1 (MUST be 12/12 or the probe itself is broken)"
run serial --max-num-seqs 1
echo "artifacts: $OUT"
