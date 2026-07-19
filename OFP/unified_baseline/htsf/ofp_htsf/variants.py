"""Canonical architecture variants; sampling and gains do not live here."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_temporal: bool
    use_engineered: bool
    fusion: str
    decision_inputs: tuple[str, ...]
    mechanism: str

    @property
    def needs_encoder(self) -> bool:
        return self.use_temporal or self.use_engineered


VARIANTS = {
    "sampled_b2_xgb": VariantSpec(
        name="sampled_b2_xgb",
        use_temporal=False,
        use_engineered=False,
        fusion="none",
        decision_inputs=("b2",),
        mechanism="same sampled endpoints, direct Raw+Statistical+Expert to XGBoost",
    ),
    "temporal_only": VariantSpec(
        name="temporal_only",
        use_temporal=True,
        use_engineered=False,
        fusion="single",
        decision_inputs=("temporal",),
        mechanism="Raw causal window representation only",
    ),
    "expert_stat_only": VariantSpec(
        name="expert_stat_only",
        use_temporal=False,
        use_engineered=True,
        fusion="single",
        decision_inputs=("engineered",),
        mechanism="Statistical+Expert endpoint representation only",
    ),
    "direct_concat": VariantSpec(
        name="direct_concat",
        use_temporal=True,
        use_engineered=True,
        fusion="concat_projection",
        decision_inputs=("fused",),
        mechanism="projected concatenation without cross-attention or gate",
    ),
    "cross_attention": VariantSpec(
        name="cross_attention",
        use_temporal=True,
        use_engineered=True,
        fusion="cross_attention",
        decision_inputs=("fused",),
        mechanism="cross-modal attention without a learned modality gate",
    ),
    "htsf_fusion": VariantSpec(
        name="htsf_fusion",
        use_temporal=True,
        use_engineered=True,
        fusion="gated_cross_attention",
        decision_inputs=("fused",),
        mechanism="cross-modal alignment and learned gate, followed by one XGBoost",
    ),
    "htsf_fusion_b2_skip": VariantSpec(
        name="htsf_fusion_b2_skip",
        use_temporal=True,
        use_engineered=True,
        fusion="gated_cross_attention",
        decision_inputs=("b2", "fused"),
        mechanism="nested B2 plus the learned HTSF representation",
    ),
}


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown HTSF variant={name!r}; expected {sorted(VARIANTS)}") from exc
