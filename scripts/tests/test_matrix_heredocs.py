# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""The metal matrix's embedded Python parses. Each `python3 - <<'PY'` block runs only when the
run reaches it, hours in for the per-model fold, so a block that does not parse is found by a run
failing there; `bash -n` reads none of them. Standard library only.

    python3 -m unittest discover -s scripts/tests
"""

import ast
import re
import unittest
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[1] / "bench_metal_matrix.sh"

# A quoted heredoc: its body runs as written, so it is the Python the run executes.
HEREDOC = re.compile(r"<<'(PY\w*)'\n(.*?)\n\1\n", re.S)


def blocks(text):
    """Each embedded Python block, with the line it starts on."""
    return [(text[: m.start()].count("\n") + 1, m.group(2)) for m in HEREDOC.finditer(text)]


class Heredocs(unittest.TestCase):
    def test_every_embedded_python_block_parses(self):
        found = blocks(MATRIX.read_text())
        self.assertGreaterEqual(len(found), 5, "the matrix's blocks were not found")
        for line, body in found:
            with self.subTest(line=line):
                ast.parse(body)

    def test_a_block_that_does_not_parse_is_found(self):
        broken = "x\npython3 - a b <<'PY'\nKEEP = [1,\n        2]\n        3]\nPY\ny\n"
        ((line, body),) = blocks(broken)
        self.assertEqual(line, 2)
        with self.assertRaises(SyntaxError):
            ast.parse(body)


if __name__ == "__main__":
    unittest.main()
