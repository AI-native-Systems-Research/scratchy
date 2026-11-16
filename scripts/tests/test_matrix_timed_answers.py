# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Which of the metal matrix's dataset cells force their answers to length (`--timed-answers`),
as bench_metal_matrix.sh's task cell sets its flags: those lines run under bash, nothing else of
the script. MuSiQue's cells answer on their own at every users count (#1208, #994), each from
its own share of the dev set (`task_share`). Standard library only.

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


def flags(workload, conc=1, rate="", share="", tasks=100):
    """The bench serve flags the block gives `workload`'s cell at `conc` users: its pace, then
    its dataset, then how many tasks it sends."""
    script = (
        "cell() {\n"
        f"local timed=1 w='{workload}' conc={conc} rate='{rate}' share='{share}' n=48 ACCURACY_TASKS={tasks}\n"
        "local -a extra=() pace=()\n"
        '[[ -n "${rate}" ]] && pace=(--request-rate "${rate}")\n'
        + block()
        + '\necho "${pace[*]} | ${extra[*]} | ${n}"\n}\ncell'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(out.stderr)
    return out.stdout.rstrip("\n")


class TimedAnswers(unittest.TestCase):
    def test_musiques_cells_answer_on_their_own_at_every_users_count(self):
        for conc in (1, 2, 4):
            self.assertNotIn("--timed-answers", flags("rag", conc=conc), conc)

    def test_a_musique_cell_takes_its_share_and_its_task_count(self):
        _, extra, n = flags("rag", conc=2, share="2/3").split(" | ")
        self.assertEqual(extra, "--dataset-name rag --rag-source musique --task-share 2/3")
        self.assertEqual(n, "100", "ACCURACY_TASKS a cell")
        self.assertEqual(flags("rag", conc=4, share="3/3", tasks=0).split(" | ")[2], "48", "at least")
        self.assertEqual(flags("rag", conc=4, share="3/3", tasks=2417).split(" | ")[2], "2417", "all")

    def test_the_span_cell_answers_on_its_own_with_no_share(self):
        for conc in (1, 4):
            pace, extra, n = flags("rag-spans", conc=conc, share="1/3").split(" | ")
            self.assertNotIn("--timed-answers", pace, conc)
            self.assertNotIn("--task-share", extra, conc)
            self.assertEqual(n, "48", conc)

    def test_the_accuracy_cell_is_gone(self):
        # An unknown workload's cell is skipped before it sends anything.
        self.assertEqual(flags("rag-accuracy", conc=4), "")

    def test_an_open_loop_musique_cell_keeps_its_rate_and_answers_on_its_own(self):
        # An open-loop rate cell runs at the largest users count.
        self.assertEqual(flags("rag", conc=4, rate="0.5", share="3/3").split(" | ")[0], "--request-rate 0.5")

    def test_each_musique_cell_has_its_own_share_in_the_order_cells_run(self):
        self.assertEqual([share(c) for c in ("1", "2", "4")], ["1/3", "2/3", "3/3"])
        with self.assertRaises(AssertionError, msg="a count the run has not"):
            share("8")
        open_loop = {"OPEN_LOOP": "1", "TASK_RATE_STEPS": "0.5,1"}
        self.assertEqual([share(k, **open_loop) for k in ("base", "rate:1", "rate:2")], ["1/3", "2/3", "3/3"])

    def test_the_other_datasets_are_still_forced_to_length(self):
        for w in ("sharegpt", "sharegpt-multi-turn", "longbench", "shared-prefix"):
            for conc in (1, 4):
                self.assertIn("--timed-answers", flags(w, conc=conc), (w, conc))


def share(load, OPEN_LOOP="0", TASK_RATE_STEPS="", GRID_CONC_LIST="1,2,4"):
    """What the script's task_share gives a MuSiQue cell at `load`: its function, run alone."""
    lines = MATRIX.read_text().splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith("task_share() {"))
    last = next(i for i, l in enumerate(lines) if i > first and l == "}")
    script = (
        f"OPEN_LOOP={OPEN_LOOP} TASK_RATE_STEPS='{TASK_RATE_STEPS}' GRID_CONC_LIST='{GRID_CONC_LIST}'\n"
        + "\n".join(lines[first : last + 1])
        + f"\ntask_share '{load}'"
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(out.stderr)
    return out.stdout.strip()


if __name__ == "__main__":
    unittest.main()
