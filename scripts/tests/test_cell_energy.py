# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Tests for scripts/cell_energy.py. Standard library only.

    python3 -m unittest discover -s scripts/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cell_energy  # noqa: E402


def rows(*joules, start=100.0):
    """One-second power rows from `start`: (unix_s, interval_s, cpu, gpu, ane), each `joules`
    a (cpu, gpu, ane) for the second that ended at unix_s."""
    return [[start + i + 1, 1.0, *j] for i, j in enumerate(joules)]


class Window(unittest.TestCase):
    def test_a_window_takes_the_seconds_inside_it_whole(self):
        log = rows((1, 2, 0), (3, 4, 0), (5, 6, 1), (7, 8, 0))
        e = cell_energy.window(log, 101.0, 103.0)
        self.assertEqual((e["cpu_j"], e["gpu_j"], e["ane_j"]), (8.0, 10.0, 1.0))
        self.assertEqual(e["soc_j"], 19.0)
        self.assertEqual(e["window_s"], 2.0)
        self.assertEqual(e["samples"], 2)

    def test_a_window_takes_a_part_second_by_its_share(self):
        # 100.5 to 102.25: half of the first second, all of the second, a quarter of the third.
        log = rows((2, 0, 0), (4, 0, 0), (8, 0, 0))
        e = cell_energy.window(log, 100.5, 102.25)
        self.assertAlmostEqual(e["cpu_j"], 1 + 4 + 2)
        self.assertAlmostEqual(e["window_s"], 1.75)

    def test_a_window_the_log_does_not_cover_is_none(self):
        log = rows((1, 1, 1), (1, 1, 1))
        self.assertIsNone(cell_energy.window(log, 50.0, 60.0))
        # Partly covered: the log missing a stretch of it is no reading.
        self.assertIsNone(cell_energy.window(log, 101.0, 110.0))


class Per(unittest.TestCase):
    def test_a_cells_energy_per_request_token_and_right_answer(self):
        cell = {"completed": 10, "failed": 0, "total_output_tokens": 500, "num_tasks": 10,
                "accuracy": 0.5}
        e = cell_energy.per(cell, {"soc_j": 200.0, "window_s": 20.0})
        self.assertEqual(e["j_per_request"], 20.0)
        self.assertEqual(e["j_per_output_token"], 0.4)
        self.assertEqual(e["soc_avg_w"], 10.0)
        # 200 J over 5 right answers.
        self.assertEqual(e["j_per_right_answer"], 40.0)

    def test_a_failed_request_still_costs_its_share(self):
        # Every request sent spent energy, answered or not.
        cell = {"completed": 3, "failed": 1, "total_output_tokens": 30}
        e = cell_energy.per(cell, {"soc_j": 40.0, "window_s": 4.0})
        self.assertEqual(e["j_per_request"], 10.0)
        self.assertNotIn("j_per_right_answer", e, "nothing scored")

    def test_no_right_answer_is_no_figure_not_infinity(self):
        cell = {"completed": 4, "total_output_tokens": 40, "num_tasks": 4, "accuracy": 0.0}
        e = cell_energy.per(cell, {"soc_j": 40.0, "window_s": 4.0})
        self.assertIsNone(e["j_per_right_answer"])


class Fold(unittest.TestCase):
    def test_a_cell_file_gets_its_energy(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "power.csv"
            log.write_text("".join(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n"
                                   for r in rows((1, 1, 0), (1, 1, 0), (1, 1, 0))))
            cell = Path(d) / "cell.json"
            cell.write_text(json.dumps({"started_unix_s": 100.0, "ended_unix_s": 102.0,
                                        "completed": 2, "total_output_tokens": 8}))
            cell_energy.fold(str(cell), str(log))
            e = json.loads(cell.read_text())["energy"]
            self.assertEqual(e["soc_j"], 4.0)
            self.assertEqual(e["j_per_request"], 2.0)
            # A cell with no window (an older bench serve's) is left as it is.
            cell.write_text(json.dumps({"completed": 2}))
            cell_energy.fold(str(cell), str(log))
            self.assertNotIn("energy", json.loads(cell.read_text()))


if __name__ == "__main__":
    unittest.main()
