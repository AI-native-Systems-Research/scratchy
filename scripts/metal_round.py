#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""A metal quick round: one issue whose comments carry each Mac's quick run.

Each team member posts a clean quick run of scripts/bench_metal_matrix.sh as a
comment on the open round issue (the metal-bench skill, step 7). Once every Mac
in MACHINES has one, .github/workflows/metal-round.yml opens a PR that adds the
newest run per Mac to site/data/metal, in one commit, and closes the issue; an
admin merges the PR.

  metal_round.py check RUN.json
      Whether the run can go in a round; exits 1 with the reason if not.
  metal_round.py collect --comments C.json --out-dir D [--main-ref REF]
      Read the issue's comments (gh api .../comments, a JSON array), write the
      newest accepted run per Mac to D under the bench script's file name, and
      print a JSON summary: the status comment's body, the rejections to reply
      to, and whether the round is complete.
  metal_round.py commit-input --branch B --head SHA --repo OWNER/NAME --dir D FILE...
      The GraphQL request that adds the files to site/data/metal on branch B
      as one commit (createCommitOnBranch, which GitHub signs).

Standard library only.
"""

import argparse
import base64
import binascii
import gzip
import json
import re
import subprocess
import sys
from collections import Counter

# The round's Macs, by machine.chip as the bench script records it, in the
# page's order (memory).
MACHINES = ["Apple M5", "Apple M3 Pro", "Apple M4 Max", "Apple M5 Max"]

# The repo is public: anyone can comment. Runs count only from the team (the
# comment's author_association), and the bot's own comments are known by their
# author as well as their marker, so a marker anyone else writes is ignored.
TEAM = {"OWNER", "MEMBER", "COLLABORATOR"}
BOT = "github-actions[bot]"
STATUS_MARK = "<!-- metal-round status -->"
REPLY_MARK = "<!-- metal-round reply {id} -->"

# Every engine's per-model cell keys besides scratchy's own: a quick run has none.
OTHER_ENGINES = ["mlx_lm", "ollama", "omlx", "omlx_tq", "vllm_metal", "llama_cpp", "mistralrs"]

FENCE = re.compile(r"```json[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.S)
# A run gzipped and base64'd, so one that keeps every request (50 a cell) still fits a comment's
# 64 KB: the metal-bench skill posts it so. Line breaks inside the block are ignored.
PACKED = re.compile(r"```json\+gzip\+base64[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.S)
UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
SHA = re.compile(r"[0-9a-f]{40}")


def scope(run):
    """quick, full or partial, as the metal-bench skill's step 4 reads a run."""
    c = run["config"]
    ladders = [Counter(r["scenario"] for r in m.get("cache_ladder") or []) for m in run["models"]]
    # Any serving setting of its own makes a run partial: extra serve flags, or a memory setting.
    plain = (c.get("scaling") is not None and not c.get("scratchy_serve_args")
             and c.get("scratchy_memory_util") is None)
    if (plain and c["scenarios"] == ["frozen", "cold", "warm"] and c.get("cold_priming_launches") == 3
            and all(lad.get("frozen") == 3 for lad in ladders)):
        return "full"
    if (plain and c["scenarios"] == ["cold", "warm"] and c.get("cold_priming_launches") == 1
            and all(lad.get("cold") == 1 for lad in ladders)):
        return "quick"
    return "partial"


def on_main(sha, main_ref):
    """Whether the commit is in main's history (a full clone's checkout)."""
    r = subprocess.run(["git", "merge-base", "--is-ancestor", sha, main_ref],
                       capture_output=True)
    return r.returncode == 0


def problem(run, is_on_main, posted=None):
    """Why a run cannot go in a round, or None if it can. `posted` is when its comment was
    made: a run cannot have started after it."""
    if not isinstance(run, dict):
        return f"its JSON is a {type(run).__name__}, not a bench run"
    try:
        if run.get("schema") != 2:
            return f"schema {run.get('schema')}, a round takes schema 2"
        if not isinstance(run["models"], list) or not all(isinstance(m, dict) for m in run["models"]):
            return "its models are not a list of models"
        chip = run["machine"]["chip"]
        if chip not in MACHINES:
            return f"{chip} is not one of the round's Macs ({', '.join(MACHINES)})"
        kind = scope(run)
        if kind != "quick":
            return f"a {kind} run; a round takes quick runs only (a full run goes in its own PR)"
        for m in run["models"]:
            for k in OTHER_ENGINES:
                if m.get(f"cache_ladder_{k}") or m.get(f"scaling_{k}"):
                    return f"{m['stem']} has {k} numbers; a quick run is scratchy alone"
        if run["repo"]["dirty"]:
            return "made from a checkout with uncommitted changes"
        sha = run["repo"]["sha"]
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            return f"repo.sha {sha!r} is not a full commit hash"
        utc = run["generated_utc"]
        if not isinstance(utc, str) or not UTC.fullmatch(utc):
            return f"generated_utc {utc!r} is not YYYY-MM-DDTHH:MM:SSZ"
        if posted and utc > posted:
            return f"it says it ran at {utc}, after it was posted ({posted})"
        if not is_on_main(sha):
            return f"made at {sha[:8]}, which is not on main"
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return f"not a bench run ({type(e).__name__}: {e})"
    return None


def stamp(run):
    """The bench script's file name for a run: <chip-slug>-<stamp>-<sha8>.json."""
    slug = re.sub(r"[^a-z0-9-]", "", run["machine"]["chip"].lower().replace(" ", "-"))
    utc = run["generated_utc"]
    return f"{slug}-{utc[:10]}T{utc[11:19].replace(':', '')}Z-{run['repo']['sha'][:8]}.json"


def runs_in(body):
    """The JSON blocks in a comment, plain or packed (PACKED), parsed; a block that does not parse
    is an error string."""
    out = []
    for block in FENCE.findall(body or ""):
        try:
            out.append(json.loads(block))
        except json.JSONDecodeError as e:
            out.append(f"its JSON does not parse ({e})")
    for block in PACKED.findall(body or ""):
        try:
            out.append(json.loads(gzip.decompress(base64.b64decode("".join(block.split()), validate=True))))
        except (binascii.Error, OSError, EOFError, UnicodeDecodeError) as e:
            out.append(f"its packed run does not unpack ({e})")
        except json.JSONDecodeError as e:
            out.append(f"its JSON does not parse ({e})")
    return out


def by_bot(c):
    return (c.get("user") or {}).get("login") == BOT


def collect(comments, is_on_main):
    """The newest accepted run per Mac, the rejections, the bot's replies so far, and its
    status comment."""
    newest = {}   # chip -> (run, comment)
    rejected = []  # (comment id, reason)
    replied = set()
    status = None
    for c in comments:
        body = c.get("body") or ""
        if by_bot(c):
            if STATUS_MARK in body and status is None:
                status = c
            replied.update(int(i) for i in re.findall(r"<!-- metal-round reply (\d+) -->", body))
            continue
        runs = runs_in(body)
        if not runs:
            continue  # discussion
        if c.get("author_association") not in TEAM:
            rejected.append((c["id"], "runs count only from the repo's owners, members and "
                             "collaborators"))
            continue
        for run in runs:
            why = run if isinstance(run, str) else problem(run, is_on_main, c.get("created_at"))
            if why:
                rejected.append((c["id"], why))
                continue
            chip = run["machine"]["chip"]
            if chip not in newest or run["generated_utc"] > newest[chip][0]["generated_utc"]:
                newest[chip] = (run, c)
    return newest, rejected, replied, status


def status_body(newest):
    lines = [STATUS_MARK, "**Metal quick round**", "", "| Mac | run | commit | posted by |",
             "|---|---|---|---|"]
    for chip in MACHINES:
        if chip in newest:
            run, c = newest[chip]
            lines.append(f"| {chip} | {run['generated_utc'][:16].replace('T', ' ')} UTC "
                         f"| {run['repo']['sha'][:8]} | [@{c['user']['login']}]({c['html_url']}) |")
        else:
            lines.append(f"| {chip} | still missing | | |")
    missing = [m for m in MACHINES if m not in newest]
    lines.append("")
    if missing:
        lines.append(f"Waiting for {', '.join(missing)}. Once all four are in, the bot opens a PR "
                     "with the newest run per Mac and closes this issue.")
    else:
        lines.append("All four Macs are in: opening a PR with the newest run per Mac.")
    return "\n".join(lines)


def commit_input(a):
    adds = []
    for name in a.files:
        with open(f"{a.dir}/{name}", "rb") as f:
            adds.append({"path": f"site/data/metal/{name}",
                         "contents": base64.b64encode(f.read()).decode()})
    return {
        "query": "mutation($input: CreateCommitOnBranchInput!) "
                 "{ createCommitOnBranch(input: $input) { commit { oid url } } }",
        "variables": {"input": {
            "branch": {"repositoryNameWithOwner": a.repo, "branchName": a.branch},
            "expectedHeadOid": a.head,
            "message": {"headline": a.headline},
            "fileChanges": {"additions": adds},
        }},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("run")
    ck.add_argument("--main-ref", default="origin/main")
    co = sub.add_parser("collect")
    co.add_argument("--comments", required=True)
    co.add_argument("--out-dir", required=True)
    co.add_argument("--main-ref", default="origin/main")
    ci = sub.add_parser("commit-input")
    ci.add_argument("--branch", required=True)
    ci.add_argument("--head", required=True)
    ci.add_argument("--repo", required=True)
    ci.add_argument("--dir", required=True)
    ci.add_argument("--headline", required=True)
    ci.add_argument("files", nargs="+")
    a = ap.parse_args(argv)

    if a.cmd == "commit-input":
        json.dump(commit_input(a), sys.stdout)
        return 0

    def is_on_main(sha):
        return on_main(sha, a.main_ref)

    if a.cmd == "check":
        with open(a.run) as f:
            why = problem(json.load(f), is_on_main)
        if why:
            print(f"not for a round: {why}")
            return 1
        print("ok for a round")
        return 0

    with open(a.comments) as f:
        newest, rejected, replied, status = collect(json.load(f), is_on_main)
    files = []
    for chip in MACHINES:
        if chip in newest:
            run = newest[chip][0]
            name = stamp(run)
            with open(f"{a.out_dir}/{name}", "w") as f:
                json.dump(run, f, indent=2)
                f.write("\n")
            files.append(name)
    json.dump({
        "complete": len(newest) == len(MACHINES),
        "files": files,
        "status_comment_id": status["id"] if status else None,
        "status_body": status_body(newest),
        "replies": [{"id": i, "body": f"{REPLY_MARK.format(id=i)}\nNot counted in the round: {why}."}
                    for i, why in rejected if i not in replied],
    }, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
