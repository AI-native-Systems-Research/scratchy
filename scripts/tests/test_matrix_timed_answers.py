# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Which of the metal matrix's dataset cells force their answers to length (`--timed-answers`),
as bench_metal_matrix.sh's task cell sets its flags: those lines run under bash, nothing else of
the script. MuSiQue's answers end on their own (#1208). Standard library only.

    python3 -m unittest discover -s scripts/tests
"""

import subprocess
import unittest
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[1] / "bench_metal_matrix.sh"
START = "    # A timed cell forces each answer to its task's length"
END = "    esac"


def block():
    """The task cell's lines from START through the `esac` that closes its workload case."""
    lines = MATRIX.read_text().splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith(START))
    last = next(i for i, l in enumerate(lines) if i > first and l == END)
    return "\n".join(lines[first : last + 1])


def flags(workload, rate=""):
    """The bench serve flags the block gives `workload`'s cell: its pace, then its dataset."""
    script = (
        "cell() {\n"
        f"local timed=1 w='{workload}' rate='{rate}' n=8 ACCURACY_TASKS=100\n"
        "local -a extra=() pace=()\n"
        '[[ -n "${rate}" ]] && pace=(--request-rate "${rate}")\n'
        + block()
        + '\necho "${pace[*]} | ${extra[*]}"\n}\ncell'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(out.stderr)
    return out.stdout.strip()


class TimedAnswers(unittest.TestCase):
    def test_musique_answers_end_on_their_own(self):
        for w in ("rag", "rag-spans", "rag-accuracy"):
            self.assertNotIn("--timed-answers", flags(w), w)

    def test_an_open_loop_musique_cell_keeps_its_rate(self):
        self.assertTrue(flags("rag", "0.5").startswith("--request-rate 0.5 |"))

    def test_the_other_datasets_are_still_forced_to_length(self):
        for w in ("sharegpt", "sharegpt-multi-turn", "longbench", "shared-prefix"):
            self.assertIn("--timed-answers", flags(w), w)


if __name__ == "__main__":
    unittest.main()
