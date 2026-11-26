# The prompts of the P-EAGLE model card's benchmarks: MT-Bench's first turns, HumanEval's and GSM-8K's
# first 100 (datasets-server rows, fetched beside them). Usage: python bench-prompts.py <data dir>
import json, sys
from huggingface_hub import hf_hub_download

out = sys.argv[1]
mt = hf_hub_download("HuggingFaceH4/mt_bench_prompts", "raw/question.jsonl", repo_type="dataset")
sets = {
    "mtbench": [json.loads(l)["prompt"][0] for l in open(mt)],
    "humaneval": [r["row"]["prompt"] for r in json.load(open(f"{out}/humaneval-rows.json"))["rows"]],
    "gsm8k": [r["row"]["question"] for r in json.load(open(f"{out}/gsm8k-rows.json"))["rows"]],
}
for name, xs in sets.items():
    json.dump(xs, open(f"{out}/{name}.json", "w"))
    print(name, len(xs), repr(xs[0][:70]))
