from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_run_experiment_help_works_without_sibling_packages(tmp_path: Path) -> None:
    """The published v2 directory must be runnable as a standalone checkout."""

    source = Path(__file__).resolve().parents[1]
    standalone = tmp_path / "deep_rule_ofp_v2"
    shutil.copytree(
        source,
        standalone,
        ignore=shutil.ignore_patterns(
            "artifacts",
            "tests",
            "__pycache__",
            ".pytest_cache",
        ),
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-B", str(standalone / "run_experiment.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Native-row sequence-to-sequence" in completed.stdout
