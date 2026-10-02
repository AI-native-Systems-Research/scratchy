"""The spec of the leaderboard's honesty rules.

One case per check in `build_leaderboard.validate()`. Each starts from a valid
board, breaks exactly one rule, and asserts the substring of the problem it must
produce — so a failure here names the rule rather than showing a diff.
"""

import unittest

import fixture

import build_leaderboard as L


def problems(mutate=None):
    b = fixture.board()
    if mutate:
        mutate(b)
    return L.validate(fixture.indexed(b))


def only(mutate):
    """The problems a single mutation introduces, over a clean baseline."""
    base = set(problems())
    return [p for p in problems(mutate) if p not in base]


class ValidBoard(unittest.TestCase):
    def test_baseline_is_clean(self):
        self.assertEqual(problems(), [])


class Check1ShapeAndReferences(unittest.TestCase):
    def test_wrong_schema(self):
        got = only(lambda b: b.update(schema=99))
        self.assertTrue(any("schema is 99" in p for p in got), got)

    def test_unknown_board_key(self):
        got = only(lambda b: b.update(hedaline="typo"))
        self.assertTrue(any("unknown key 'hedaline'" in p for p in got), got)

    def test_lane_owns_undeclared_metric(self):
        got = only(lambda b: b["lanes"]["A"]["owns"].append("no_such_metric"))
        self.assertTrue(
            any("owns undeclared metric 'no_such_metric'" in p for p in got), got)

    def test_metric_owned_by_no_lane(self):
        def m(b):
            b["metrics"].append({"id": "orphan", "label": "o", "unit": "",
                                 "lower_is_better": True, "digits": 1})
        got = only(m)
        self.assertTrue(any("'orphan' belongs to no lane" in p for p in got), got)

    def test_metric_owned_by_two_lanes(self):
        got = only(lambda b: b["lanes"]["B"]["owns"].append("ttft_ms"))
        self.assertTrue(any("is owned by lanes" in p for p in got), got)

    def test_undeclared_engine_on_a_record(self):
        got = only(lambda b: b["runs"][0]["measurements"][0].update(engine="nope"))
        self.assertTrue(any("undeclared engine 'nope'" in p for p in got), got)

    def test_unknown_measurement_key(self):
        got = only(lambda b: b["runs"][0]["measurements"][0].update(tpot=1))
        self.assertTrue(any("unknown key 'tpot'" in p for p in got), got)


class Check2LanePurity(unittest.TestCase):
    def test_lane_b_metric_in_a_lane_a_run(self):
        def m(b):
            b["runs"][0]["measurements"][0]["metric"] = "task_pass"
        got = only(m)
        self.assertTrue(
            any("lane A does not own 'task_pass'" in p for p in got), got)

    def test_record_lane_disagrees_with_run_lane(self):
        got = only(lambda b: b["runs"][0]["measurements"][0].update(lane="B"))
        self.assertTrue(any("!= run lane 'A'" in p for p in got), got)


class Check3Provenance(unittest.TestCase):
    def test_missing_required_key(self):
        got = only(lambda b: b["runs"][0]["provenance"].pop("scratchy_sha"))
        self.assertTrue(any("missing 'scratchy_sha'" in p for p in got), got)

    def test_empty_required_key(self):
        got = only(lambda b: b["runs"][0]["provenance"].update(claude_version=""))
        self.assertTrue(any("missing 'claude_version'" in p for p in got), got)

    def test_per_engine_map_missing_one_engine(self):
        def m(b):
            b["runs"][0]["provenance"]["kv_cache_dtype"].pop("ollama")
        got = only(m)
        self.assertTrue(
            any("'kv_cache_dtype' has no entry for 'ollama'" in p for p in got), got)

    def test_per_engine_key_given_as_a_scalar(self):
        def m(b):
            b["runs"][0]["provenance"]["quantization"] = "mlx-affine-b4-g64"
        got = only(m)
        self.assertTrue(
            any("must be a per-engine object" in p for p in got), got)

    def test_undeclared_machine(self):
        got = only(lambda b: b["runs"][0]["provenance"].update(machine="ghost"))
        self.assertTrue(any("undeclared machine 'ghost'" in p for p in got), got)


class Check4StatShape(unittest.TestCase):
    def test_no_median(self):
        got = only(lambda b: b["runs"][0]["measurements"][0]["stat"].pop("median"))
        self.assertTrue(any("stat has no median" in p for p in got), got)

    def test_spread_missing_when_n_above_one(self):
        def m(b):
            b["runs"][0]["measurements"][0]["stat"] = {"median": 100}
        got = only(m)
        self.assertTrue(any("without p10/p90" in p for p in got), got)

    def test_median_outside_its_own_spread(self):
        def m(b):
            b["runs"][0]["measurements"][0]["stat"] = {"median": 500, "p10": 90,
                                                       "p90": 110}
        got = only(m)
        self.assertTrue(any("does not hold" in p for p in got), got)

    def test_n_below_the_lane_minimum(self):
        got = only(lambda b: b["runs"][1]["measurements"][0].update(n=2))
        self.assertTrue(
            any("below lane B's declared minimum 5" in p for p in got), got)

    def test_series_without_points(self):
        def m(b):
            b["runs"][0]["measurements"][0]["series"] = {"x": "turn_index",
                                                         "points": []}
        got = only(m)
        self.assertTrue(any("no points" in p for p in got), got)


class Check5Coverage(unittest.TestCase):
    def test_uncovered_cell(self):
        got = only(lambda b: b["runs"][1]["measurements"].pop(0))
        self.assertTrue(any("uncovered" in p for p in got), got)

    def test_duplicate_cell(self):
        def m(b):
            ms = b["runs"][0]["measurements"]
            ms.append(dict(ms[0]))
        got = only(m)
        self.assertTrue(
            any("must not silently pick one" in p for p in got), got)

    def test_gap_without_a_reason(self):
        def m(b):
            b["runs"][1]["measurements"].pop(0)
            b["runs"][1]["gaps"].append({
                "lane": "B", "metric": "task_pass", "engine": "scratchy",
                "model": "m1", "machine": "h1", "rung": "R1",
                "state": "not-run", "reason": "",
            })
        got = only(m)
        self.assertTrue(any("no reason" in p for p in got), got)

    def test_gap_with_an_unknown_state(self):
        def m(b):
            b["runs"][1]["measurements"].pop(0)
            b["runs"][1]["gaps"].append({
                "lane": "B", "metric": "task_pass", "engine": "scratchy",
                "model": "m1", "machine": "h1", "rung": "R1",
                "state": "shrug", "reason": "because",
            })
        got = only(m)
        self.assertTrue(any("unknown state 'shrug'" in p for p in got), got)

    def test_a_gap_covers_the_cell(self):
        """Removing a measurement is fine if a reasoned gap replaces it."""
        def m(b):
            b["runs"][1]["measurements"].pop(0)
            b["runs"][1]["gaps"].append({
                "lane": "B", "metric": "task_pass", "engine": "scratchy",
                "model": "m1", "machine": "h1", "rung": "R1",
                "state": "blocked", "reason": "the grader could not run",
                "issue": 164,
            })
        self.assertEqual(only(m), [])


class Check6Applicability(unittest.TestCase):
    def test_ablation_on_a_non_live_optimization(self):
        got = only(lambda b: b["applicability"][0].update(state="inert"))
        self.assertTrue(
            any("declared state is 'inert'" in p for p in got), got)

    def test_ablation_without_an_attribution(self):
        got = only(lambda b: b["runs"][0]["ablations"][0].pop("attribution"))
        self.assertTrue(any("attribution is None" in p for p in got), got)

    def test_ablation_without_a_quality_check(self):
        got = only(lambda b: b["runs"][0]["ablations"][0].update(quality_check=""))
        self.assertTrue(any("no quality_check" in p for p in got), got)

    def test_deferred_without_evidence(self):
        def m(b):
            b["runs"][0]["ablations"] = []
            b["applicability"][0].update(state="deferred", evidence="")
        got = only(m)
        self.assertTrue(
            any("deferred without evidence" in p for p in got), got)

    def test_applicability_without_a_reason(self):
        got = only(lambda b: b["applicability"][0].update(reason=""))
        self.assertTrue(any("no reason given" in p for p in got), got)


class Check7Comparability(unittest.TestCase):
    def test_context_length_differs_across_engines(self):
        def m(b):
            # Give one engine its own run, with a different context length.
            run = fixture.lane_a()
            run["run_id"] = "r-a2"
            run["measurements"] = [run["measurements"][1]]
            run["ablations"] = []
            run["provenance"]["context_length_tokens"] = 8192
            b["runs"][0]["measurements"] = [b["runs"][0]["measurements"][0]]
            b["runs"].append(run)
        got = only(m)
        self.assertTrue(
            any("disagree on context_length_tokens" in p for p in got), got)

    def test_kv_dtype_differs_without_the_disclosure(self):
        got = only(lambda b: b.update(disclosures=[]))
        self.assertTrue(
            any("no 'kv-dtype-unmatched' disclosure" in p for p in got), got)

    def test_undeclared_disclosure_reference(self):
        def m(b):
            b["runs"][0]["provenance"]["disclosed"].append("made-up")
        got = only(m)
        self.assertTrue(
            any("undeclared disclosure 'made-up'" in p for p in got), got)


class Check8NoCrossQuoting(unittest.TestCase):
    def test_kpi_operands_from_different_rungs(self):
        def m(b):
            b["rungs"].append({"id": "R2", "label": "matched", "detail": "d"})
            run = fixture.lane_a()
            run["run_id"] = "r-a2"
            run["measurements"] = [run["measurements"][1]]
            run["ablations"] = []
            run["provenance"]["rung"] = "R2"
            b["runs"][0]["measurements"] = [b["runs"][0]["measurements"][0]]
            b["runs"].append(run)
        got = only(m)
        self.assertTrue(
            any("kpi k1: operands differ in rung" in p for p in got), got)

    def test_kpi_operands_from_different_lanes(self):
        def m(b):
            b["kpis"][0].update(metric="ttft_ms")
            b["lanes"]["B"]["owns"].append("ttft_ms")
            run = fixture.lane_b()
            run["run_id"] = "r-b2"
            run["measurements"] = [{
                "lane": "B", "metric": "ttft_ms", "engine": "ollama",
                "model": "m1", "n": 5, "stat": {"median": 300, "p10": 280,
                                                "p90": 320},
            }]
            b["runs"][0]["measurements"] = [b["runs"][0]["measurements"][0]]
            b["runs"].append(run)
        got = only(m)
        self.assertTrue(
            any("different lanes" in p or "owned by lanes" in p for p in got), got)


class Check10ExampleIsolation(unittest.TestCase):
    def test_example_and_published_for_one_cell(self):
        def m(b):
            run = fixture.lane_a()
            run["run_id"] = "r-a-example"
            run["status"] = "example"
            run["ablations"] = []
            b["runs"].append(run)
        got = only(m)
        self.assertTrue(
            any("never blended with it" in p for p in got), got)

    def test_example_and_published_runs_together(self):
        def m(b):
            run = fixture.lane_b()
            run["run_id"] = "r-b-example"
            run["status"] = "example"
            run["measurements"] = []
            b["runs"].append(run)
        got = only(m)
        self.assertTrue(
            any("delete the example files" in p for p in got), got)

    def test_empty_board_must_say_why(self):
        def m(b):
            for r in b["runs"]:
                r["status"] = "example"
            b["status"]["pending"]["text"] = ""
        got = only(m)
        self.assertTrue(
            any("must say why it is empty" in p for p in got), got)

    def test_unknown_run_status(self):
        got = only(lambda b: b["runs"][0].update(status="draft"))
        self.assertTrue(any("unknown status 'draft'" in p for p in got), got)


class ShippedData(unittest.TestCase):
    """The committed data must itself pass, or the site does not build."""

    def test_committed_board_is_valid(self):
        self.assertEqual(L.validate(L.load()), [])


if __name__ == "__main__":
    unittest.main()
