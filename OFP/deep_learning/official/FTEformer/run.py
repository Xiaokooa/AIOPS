from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from OFP.deep_learning.official.FTEformer.model import build_model
from OFP.deep_learning.official.common.trainer import run_single_model_cli


if __name__ == "__main__":
    run_single_model_cli("fteformer", build_model)
