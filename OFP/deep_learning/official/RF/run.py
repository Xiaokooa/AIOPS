from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from OFP.deep_learning.official.common.tree_runner import run_tree_model_cli


if __name__ == "__main__":
    run_tree_model_cli("rf")
