# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Tests for scripts/metal_round.py. Standard library only:

    python3 -m unittest discover -s scripts/tests
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metal_round  # noqa: E402

SHA = "aa" + "0" * 38


def on_main(sha):
    return sha.startswith("aa")


def run(chip="Apple M3 Pro", utc="2026-10-21T10:23:54Z", sha=SHA, dirty=False,
        kind="quick", engines=()):
    cold = 3 if kind == "full" else 1
    ladder = ([{"scenario": "frozen"}] * 3 if kind == "full" else []) + [{"scenario": "cold"}] * cold
    model = {"stem": "gpt-oss-20b", "cache_ladder": ladder}
    for e in engines:
        model[f"scaling_{e}"] = [{}]
    return {
        "schema": 2, "generated_utc": utc,
        "machine": {"chip": chip}, "repo": {"sha": sha, "dirty": dirty},
        "config": {"scenarios": ["frozen", "cold", "warm"] if kind == "full" else ["cold", "warm"],
                   "cold_priming_launches": 3 if kind == "full" else 1,
                   "scaling": {}, "scratchy_serve_args": None},
        "models": [model],
    }


def comment(cid, *runs, login="someone", who="MEMBER", at="2026-10-22T12:00:00Z", text="a run"):
    body = text + "".join(f"\n<details>\n\n```json\n{json.dumps(r, indent=2)}\n```\n</details>"
                          for r in runs)
    return {"id": cid, "body": body, "user": {"login": login}, "author_association": who,
            "created_at": at, "html_url": f"https://example/{cid}"}


def bot(cid, body):
    return {"id": cid, "body": body, "user": {"login": metal_round.BOT},
            "author_association": "NONE", "created_at": "2026-10-22T12:00:00Z", "html_url": ""}


class Check(unittest.TestCase):
    def test_a_clean_quick_run_on_main_goes_in(self):
        self.assertIsNone(metal_round.problem(run(), on_main))

    def test_a_full_run_does_not(self):
        self.assertIn("full run", metal_round.problem(run(kind="full"), on_main))

    def test_a_quick_run_with_another_engine_is_not_quick(self):
        self.assertIn("omlx", metal_round.problem(run(engines=["omlx"]), on_main))

    def test_a_dirty_tree_or_a_commit_off_main_does_not(self):
        self.assertIn("uncommitted", metal_round.problem(run(dirty=True), on_main))
        self.assertIn("not on main", metal_round.problem(run(sha="bb" + "0" * 38), on_main))

    def test_a_mac_outside_the_round_does_not(self):
        self.assertIn("not one of", metal_round.problem(run(chip="Apple M1 Max"), on_main))

    def test_serving_settings_of_its_own_make_a_run_partial(self):
        r = run()
        r["config"]["scratchy_serve_args"] = "--max-num-seqs 4"
        self.assertIn("partial", metal_round.problem(r, on_main))
        r = run()
        r["config"]["scratchy_memory_util"] = 0.5
        self.assertIn("partial", metal_round.problem(r, on_main))

    def test_a_commit_must_be_a_full_hash(self):
        # git merge-base takes any revision: HEAD would pass, and name the file -HEAD.json.
        self.assertIn("full commit hash", metal_round.problem(run(sha="HEAD"), lambda _: True))

    def test_a_run_cannot_have_started_after_it_was_posted(self):
        why = metal_round.problem(run(utc="2099-01-01T00:00:00Z"), on_main, "2026-10-22T12:00:00Z")
        self.assertIn("after it was posted", why)

    def test_json_that_is_not_a_run_is_refused_not_a_crash(self):
        for v in ([], 3, "x", {"schema": 2, "models": [1]}, {"schema": 2, "models": "x"}):
            self.assertIsNotNone(metal_round.problem(v, on_main), repr(v))

    def test_its_file_name_is_the_bench_scripts(self):
        self.assertEqual(metal_round.stamp(run()),
                         "apple-m3-pro-2026-10-21T102354Z-aa000000.json")


class Collect(unittest.TestCase):
    def test_the_newest_run_per_mac_counts_by_when_it_ran(self):
        late_post_of_old_run = run(utc="2026-10-20T09:00:00Z")
        newest, rejected, _, _ = metal_round.collect(
            [comment(1, run()), comment(2, late_post_of_old_run)], on_main)
        self.assertEqual(newest["Apple M3 Pro"][0]["generated_utc"], "2026-10-21T10:23:54Z")
        self.assertEqual(rejected, [])

    def test_a_round_is_complete_with_all_four_macs(self):
        cs = [comment(i, run(chip=c)) for i, c in enumerate(metal_round.MACHINES)]
        newest, *_ = metal_round.collect(cs[:3], on_main)
        self.assertIn("still missing", metal_round.status_body(newest))
        newest, *_ = metal_round.collect(cs, on_main)
        self.assertEqual(len(newest), 4)
        self.assertIn("All four Macs are in", metal_round.status_body(newest))

    def test_the_status_comment_after_the_first_run_does_not_close_the_round(self):
        # The status comment comes right after the first run; every run after it still counts.
        cs = [comment(1, run(chip="Apple M3 Pro")), bot(2, metal_round.STATUS_MARK),
              comment(3, run(chip="Apple M5")), comment(4, run(chip="Apple M4 Max")),
              comment(5, run(chip="Apple M5 Max")), comment(6, text="LGTM")]
        newest, rejected, _, status = metal_round.collect(cs, on_main)
        self.assertEqual(len(newest), 4)
        self.assertEqual(rejected, [])
        self.assertEqual(status["id"], 2)

    def test_runs_count_only_from_the_team(self):
        newest, rejected, _, _ = metal_round.collect(
            [comment(1, run(), who="NONE"), comment(2, run(), who="CONTRIBUTOR")], on_main)
        self.assertEqual(newest, {})
        self.assertEqual([i for i, _ in rejected], [1, 2])

    def test_a_marker_from_anyone_but_the_bot_is_ignored(self):
        fake = comment(1, text=metal_round.STATUS_MARK + " " + metal_round.REPLY_MARK.format(id=2),
                       who="NONE")
        newest, rejected, replied, status = metal_round.collect(
            [fake, comment(2, run(dirty=True))], on_main)
        self.assertIsNone(status)
        self.assertEqual(replied, set())
        self.assertEqual([i for i, _ in rejected], [2])

    def test_discussion_is_not_a_run(self):
        newest, rejected, _, _ = metal_round.collect(
            [comment(1, text="looks good", who="NONE")], on_main)
        self.assertEqual((newest, rejected), ({}, []))

    def test_a_rejected_run_gets_one_reply(self):
        reply = bot(3, metal_round.REPLY_MARK.format(id=1) + "\nNot counted")
        _, rejected, replied, _ = metal_round.collect(
            [comment(1, run(dirty=True)), reply, comment(2, text="```json\n{oops\n```")], on_main)
        self.assertEqual([i for i, _ in rejected], [1, 2])
        self.assertEqual(replied, {1})

    def test_a_packed_run_counts_like_a_plain_one(self):
        import base64, gzip
        packed = base64.b64encode(gzip.compress(json.dumps(run()).encode())).decode()
        wrapped = "\n".join(packed[i:i + 76] for i in range(0, len(packed), 76))
        newest, rejected, _, _ = metal_round.collect(
            [comment(1, text=f"a run\n```json+gzip+base64\n{wrapped}\n```")], on_main)
        self.assertEqual(rejected, [])
        self.assertEqual(newest[run()["machine"]["chip"]][0], run())

    def test_a_packed_block_that_does_not_unpack_is_rejected_not_a_crash(self):
        _, rejected, _, _ = metal_round.collect(
            [comment(1, text="```json+gzip+base64\nnot base64!\n```")], on_main)
        self.assertEqual(len(rejected), 1)
        self.assertIn("does not unpack", rejected[0][1])

    def test_a_block_that_is_not_an_object_is_rejected_and_the_rest_still_counts(self):
        newest, rejected, _, _ = metal_round.collect(
            [comment(1, text="```json\n[]\n```"), comment(2, run())], on_main)
        self.assertEqual([i for i, _ in rejected], [1])
        self.assertIn("Apple M3 Pro", newest)


if __name__ == "__main__":
    unittest.main()
