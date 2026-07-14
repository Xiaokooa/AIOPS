from __future__ import annotations

from run_suite import EXPERIMENTS
from run_threefold import DEFAULT_EXPERIMENT_NAME, DEFAULT_OUTPUT_DIR


def test_historical_n0_n1_suite_remains_a_pure_temporal_objective_ablation() -> None:
    for flags in EXPERIMENTS.values():
        assert "--architecture" in flags
        architecture_index = flags.index("--architecture")
        assert flags[architecture_index + 1] == "causal_depthwise_tcn"
        assert "--disable-safe-rule-fallback" in flags
    assert "--disable-fgl" in EXPERIMENTS["N0_native_s2s_ce"]
    assert "--disable-fgl" not in EXPERIMENTS["N1_native_s2s_ce_fgl"]


def test_default_threefold_identity_matches_the_n2_rule_guided_configuration() -> None:
    assert DEFAULT_EXPERIMENT_NAME == "N2_rule_guided_safe_3fold"
    assert DEFAULT_OUTPUT_DIR.name == DEFAULT_EXPERIMENT_NAME
