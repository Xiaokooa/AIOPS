from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAINING_DIR = DATASET_DIR / "training"
OFFICIAL_TEST_DIR = DATASET_DIR / "test"
INDEX_FILE = DATASET_DIR / "train_test_set_index(in).csv"


@dataclass(frozen=True)
class OFPDRAMFeatureConfig:
    """Feature and sample settings for the OFP-DRAM experiment."""

    ahead_hours: int = 120
    history_minutes: tuple[int, ...] = (15, 60, 360, 1440)
    train_step_minutes: int = 60
    eval_step_minutes: int = 5
    min_history_points: int = 6
    min_observed_ratio: float = 0.50
    train_faulty_max_windows: int = 192
    train_healthy_max_windows: int = 32

    @property
    def ahead_seconds(self) -> int:
        return int(self.ahead_hours) * 3600

    @property
    def tag(self) -> str:
        hist = "-".join(str(v) for v in self.history_minutes)
        return (
            f"dram_ofp_ahead{self.ahead_hours}h_hist{hist}_"
            f"trstep{self.train_step_minutes}m_evalstep{self.eval_step_minutes}m"
        )


@dataclass
class OFPDRAMTrainConfig:
    random_state: int = 42
    n_jobs: int = 1
    two_stage: bool = True
    positive_lead_alpha: float = 2.0
    hard_negative_alpha: float = 2.0
    n_estimators: int = 240
    max_depth: int = 6
    learning_rate: float = 0.06
    subsample: float = 0.85
    colsample: float = 0.85
    threshold_grid_size: int = 99

