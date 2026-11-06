# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""The metal matrix's accuracy cell size, as bench_metal_matrix.sh sets it from the Mac's memory
and its flags: its own lines run under bash, nothing else of the script. Standard library only.

    python3 -m unittest discover -s scripts/tests
"""

import subprocess
import unittest
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[1] / "bench_metal_matrix.sh"
START = "# The accuracy cell's size: as asked"
END = "(( ACCURACY_TIMEOUT_S )) ||"


def block():
    """The script's lines that size the accuracy cell, from START through the END line."""
    lines = MATRIX.read_text().splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith(START))
    last = next(i for i, l in enumerate(lines) if i > first and l.startswith(END))
    return "\n".join(lines[first : last + 1])


def size(mem_gb, full, asked=""):
    """(tasks, limit seconds, exit code) the block sets on a Mac of mem_gb, as asked."""
    preamble = (
        "LARGE_GB=64; ACCURACY_ALL=2417; ACCURACY_SMALL=400; ACCURACY_TIMEOUT_S=0; "
        f"MEM_GB={mem_gb}; FULL={int(full)}; ACCURACY_TASKS='{asked}'\n"
    )
    out = subprocess.run(
        ["bash", "-c", preamble + block() + '\necho "${ACCURACY_TASKS} ${ACCURACY_TIMEOUT_S}"'],
        capture_output=True, text=True,
    )
    if out.returncode:
        return None, None, out.returncode
    tasks, limit = out.stdout.split()
    return int(tasks), int(limit), 0


class AccuracySize(unittest.TestCase):
    def test_a_full_run_asks_the_whole_dev_set_on_a_large_mac(self):
        self.assertEqual(size(64, True), (2417, 2417 * 18, 0))
        self.assertEqual(size(128, True)[0], 2417)

    def test_a_full_run_asks_400_on_a_smaller_mac(self):
        self.assertEqual(size(36, True), (400, 7200, 0), "the M3 Pro")
        self.assertEqual(size(32, True)[0], 400, "the base M5")

    def test_a_quick_run_has_no_accuracy_cell(self):
        self.assertEqual(size(64, False), (0, 2400, 0))

    def test_a_count_or_all_overrides_the_machine(self):
        self.assertEqual(size(36, True, "100"), (100, 2400, 0), "at least 2400 s")
        self.assertEqual(size(36, False, "all")[0], 2417)
        self.assertEqual(size(64, True, "0")[0], 0)

    def test_anything_else_is_refused(self):
        self.assertEqual(size(36, True, "lots")[2], 2)


if __name__ == "__main__":
    unittest.main()
