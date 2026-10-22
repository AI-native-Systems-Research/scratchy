# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Tests for scripts/parity_ties.py. Standard library only: the gap measure takes any rows
that index and give items, so it runs without mlx.

    python3 -m unittest discover -s scripts/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parity_ties  # noqa: E402


class Num(float):
    """A float with mlx's `.item()`, and subtraction that keeps it."""

    def item(self):
        return float(self)

    def __sub__(self, other):
        return Num(float(self) - float(other))


class Row(list):
    def __getitem__(self, i):
        return Num(list.__getitem__(self, i))


class FakeMx:
    @staticmethod
    def argmax(row):
        return Num(max(range(len(row)), key=lambda i: list.__getitem__(row, i)))


class Gaps(unittest.TestCase):
    def setUp(self):
        sys.modules["mlx"] = type(sys)("mlx")
        sys.modules["mlx.core"] = FakeMx
        sys.modules["mlx"].core = FakeMx

    def tearDown(self):
        sys.modules.pop("mlx.core", None)
        sys.modules.pop("mlx", None)

    def test_a_token_the_reference_would_pick_has_no_gap(self):
        # Prompt of 2 positions; position 1 predicts answer token 0, position 2 token 1.
        logits = [Row([0, 0, 0]), Row([1.0, 5.0, 0.0]), Row([0.0, 0.0, 3.0])]
        self.assertEqual(parity_ties.gaps(logits, 2, [1, 2]), [])

    def test_a_token_behind_the_top_is_its_gap(self):
        logits = [Row([0, 0, 0]), Row([4.75, 5.0, 0.0]), Row([0.0, 0.0, 3.0])]
        self.assertEqual(parity_ties.gaps(logits, 2, [0, 2]), [(0, 1, 0.25)])

    def test_an_exact_tie_the_reference_breaks_the_other_way_still_counts(self):
        # gpt-oss-20b's arith answer had one: " need" for " have", 0.0 logits apart.
        logits = [Row([0, 0, 0]), Row([5.0, 5.0, 0.0])]
        self.assertEqual(parity_ties.gaps(logits, 2, [1]), [(0, 0, 0.0)])


if __name__ == "__main__":
    unittest.main()
