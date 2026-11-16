# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""How many tasks each of the metal matrix's MuSiQue cells sends, as bench_metal_matrix.sh sets it
from the Mac's memory and its flags: its own lines run under bash, nothing else of the script.
Standard library only.

    python3 -m unittest discover -s scripts/tests
"""

import subprocess
import unittest
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[1] / "bench_metal_matrix.sh"
START = "# A MuSiQue cell's tasks: as asked"
END = '[[ "${ACCURACY_TASKS}" =~ ^[0-9]+$ ]] ||'


def block():
    """The script's lines that size a MuSiQue cell, from START through the END line."""
    lines = MATRIX.read_text().splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith(START))
    last = next(i for i, l in enumerate(lines) if i > first and l.startswith(END))
    return "\n".join(lines[first : last + 1])


def size(mem_gb, full, asked=""):
    """(tasks, exit code) the block sets on a Mac of mem_gb, as asked."""
    preamble = (
        "LARGE_GB=64; ACCURACY_ALL=2417; ACCURACY_SMALL=200; ACCURACY_QUICK=100; "
        f"MEM_GB={mem_gb}; FULL={int(full)}; ACCURACY_TASKS='{asked}'\n"
    )
    out = subprocess.run(
        ["bash", "-c", preamble + block() + '\necho "${ACCURACY_TASKS}"'],
        capture_output=True, text=True,
    )
    if out.returncode:
        return None, out.returncode
    return int(out.stdout), 0


class AccuracySize(unittest.TestCase):
    def test_a_full_run_asks_each_cells_whole_share_on_a_large_mac(self):
        # The whole dev set's count: a cell's share is a third of it at 1, 2 and 4 users, and
        # bench serve sends all of a share asked for more than it holds.
        self.assertEqual(size(64, True), (2417, 0))
        self.assertEqual(size(128, True), (2417, 0))

    def test_a_full_run_asks_200_a_cell_on_a_smaller_mac(self):
        self.assertEqual(size(36, True), (200, 0), "the M3 Pro")
        self.assertEqual(size(32, True), (200, 0), "the base M5")

    def test_a_quick_run_asks_100_a_cell_on_any_mac(self):
        for mem in (32, 36, 64):
            self.assertEqual(size(mem, False), (100, 0), mem)

    def test_a_count_or_all_overrides_the_machine(self):
        self.assertEqual(size(36, True, "100"), (100, 0))
        self.assertEqual(size(36, False, "all"), (2417, 0))
        self.assertEqual(size(64, True, "0"), (0, 0), "a cell's rounds alone")

    def test_anything_else_is_refused(self):
        self.assertEqual(size(36, True, "lots")[1], 2)


if __name__ == "__main__":
    unittest.main()
