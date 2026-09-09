"""Path resolution shared by the campaign tools.

The tools are authored under /tmp/senpass and INSTALLED into
<worktree>/crustify-senpass/tools/. Both layouts must work, so nothing hardcodes /tmp: paths are
derived from this file's own location, with an env override for each.

    CAMP  = <...>/crustify-senpass      the campaign directory (this file's parent's parent)
    WORK  = $CAMP/work                  derived tables (inscope.json, ordered.json, units.json)
    TREE  = dirname(CAMP)               the repo root to read/write Rust and crustify/ under
    SENT  = the authority's Sentient directory (absolute; the C++ never moves)
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
CAMP = os.environ.get("SENPASS_CAMP") or os.path.dirname(_here)
WORK = os.environ.get("SENPASS_WORK") or os.path.join(CAMP, "work")
TREE = os.environ.get("SENPASS_TREE") or os.path.dirname(CAMP)
SENT = os.environ.get("SENPASS_SENT") or \
    "/Users/nickm/git/deeptools-src/dcc/src/Transform/Sentient"
AUTH_ROOT = os.environ.get("SENPASS_AUTH") or "/Users/nickm/git/deeptools-src"
REL = "dcc/src/Transform/Sentient"
REV = "a0d29abbedfa2dd44ec7255e59440b06a429118c"
EXTRACT = "crustify-senpass/cpp/sentient.cpp"
OUTDIR = "crates/compiler/deeptools/src/transform/sentient"
