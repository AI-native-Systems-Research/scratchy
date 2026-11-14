# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Which of the metal matrix's dataset cells force their answers to length (`--timed-answers`),
as bench_metal_matrix.sh's task cell sets its flags: those lines run under bash, nothing else of
the script. MuSiQue's 1-user cell answers on its own, its cells above 1 user are forced (#1208).
Standard library only.

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


def flags(workload, conc=1, rate=""):
    """The bench serve flags the block gives `workload`'s cell at `conc` users: its pace, then
    its dataset."""
    script = (
        "cell() {\n"
        f"local timed=1 w='{workload}' conc={conc} rate='{rate}' n=8 ACCURACY_TASKS=100\n"
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
    def test_musiques_one_user_cell_answers_on_its_own(self):
        self.assertNotIn("--timed-answers", flags("rag", conc=1))

    def test_musiques_cells_above_one_user_are_forced(self):
        for conc in (2, 4):
            self.assertIn("--timed-answers", flags("rag", conc=conc), conc)

    def test_the_span_and_accuracy_cells_answer_on_their_own(self):
        for w in ("rag-spans", "rag-accuracy"):
            for conc in (1, 4):
                self.assertNotIn("--timed-answers", flags(w, conc=conc), (w, conc))

    def test_an_open_loop_musique_cell_keeps_its_rate_and_is_forced(self):
        # An open-loop rate cell runs at the largest users count.
        self.assertEqual(flags("rag", conc=4, rate="0.5").split(" | ")[0], "--request-rate 0.5 --timed-answers")

    def test_the_other_datasets_are_still_forced_to_length(self):
        for w in ("sharegpt", "sharegpt-multi-turn", "longbench", "shared-prefix"):
            for conc in (1, 4):
                self.assertIn("--timed-answers", flags(w, conc=conc), (w, conc))


if __name__ == "__main__":
    unittest.main()
