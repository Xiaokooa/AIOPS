from __future__ import annotations

import pytest
import torch

from drfp.losses import DRFPLoss, multi_horizon_weighted_bce
from drfp.model import (
    DRFPNet,
    cumulative_from_hazards,
    count_trainable_parameters,
    parameter_statistics,
)


VARIANTS = (
    "temporal_only",
    "deep_raw_stat",
    "rule_only",
    "fixed_fusion",
    "gated_fusion",
)


def _inputs(batch_size: int = 3):
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(batch_size, 168, 12, generator=generator)
    raw_mask = (torch.rand(batch_size, 168, 12, generator=generator) > 0.1).float()
    delta = torch.rand(batch_size, 168, 1, generator=generator) * 6.0
    stats = torch.randn(batch_size, 76, generator=generator)
    stats_mask = (torch.rand(batch_size, 76, generator=generator) > 0.05).float()
    rule = torch.randn(batch_size, 156, generator=generator)
    rule[:, -4:] = torch.rand(batch_size, 4, generator=generator)
    rule_mask = (torch.rand(batch_size, 156, generator=generator) > 0.08).float()
    return {
        "raw": raw,
        "raw_mask": raw_mask,
        "delta_hours": delta,
        "stats": stats,
        "stats_mask": stats_mask,
        "rule": rule,
        "rule_mask": rule_mask,
    }


def _small_model(variant: str) -> DRFPNet:
    return DRFPNet(
        variant=variant,
        patch_length=12,
        patch_stride=12,
        d_model=16,
        latent_dim=16,
        attention_heads=4,
        transformer_layers=1,
        ff_dim=32,
        dropout=0.0,
    )


def test_cumulative_from_hazards_is_monotone_and_differentiable():
    logits = torch.tensor(
        [[-3.0, 1.0, -2.0, 0.0], [2.0, -4.0, 0.5, -1.0]],
        requires_grad=True,
    )
    probabilities = cumulative_from_hazards(logits)
    assert probabilities.shape == (2, 4)
    assert torch.all(probabilities[:, 1:] >= probabilities[:, :-1])
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    probabilities.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_shapes_and_branch_activation(variant: str):
    model = _small_model(variant).eval()
    output = model(**_inputs())
    assert output["fused_hazard_logits"].shape == (3, 4)
    assert output["fused_probabilities"].shape == (3, 4)
    assert output["probabilities"].shape == (3, 4)
    assert output["gate"].shape == (3, 4)
    assert torch.all(output["fused_probabilities"][:, 1:] >= output["fused_probabilities"][:, :-1])

    if variant == "rule_only":
        assert output["deep_hazard_logits"] is None
        assert output["rule_hazard_logits"].shape == (3, 4)
    elif variant in {"temporal_only", "deep_raw_stat"}:
        assert output["deep_hazard_logits"].shape == (3, 4)
        assert output["rule_hazard_logits"] is None
    else:
        assert output["deep_hazard_logits"].shape == (3, 4)
        assert output["rule_hazard_logits"].shape == (3, 4)


@pytest.mark.parametrize(
    "variant, expected",
    [
        ("temporal_only", 0.0),
        ("deep_raw_stat", 0.0),
        ("rule_only", 1.0),
        ("fixed_fusion", 0.5),
    ],
)
def test_deterministic_gate_variants(variant: str, expected: float):
    output = _small_model(variant).eval()(**_inputs(batch_size=2))
    assert torch.allclose(output["gate"], torch.full((2, 4), expected))


def test_gated_fusion_gate_is_a_probability_and_uses_rule_mask():
    model = _small_model("gated_fusion").eval()
    inputs = _inputs(batch_size=2)
    output = model(**inputs)
    assert torch.all(output["gate"] >= 0.0)
    assert torch.all(output["gate"] <= 1.0)
    no_rule_mask = dict(inputs)
    no_rule_mask["rule_mask"] = torch.zeros_like(inputs["rule_mask"])
    masked_output = model(**no_rule_mask)
    assert torch.isfinite(masked_output["gate"]).all()


def test_fixed_fusion_occurs_in_hazard_logit_space():
    output = _small_model("fixed_fusion").eval()(**_inputs(batch_size=2))
    expected = 0.5 * output["deep_hazard_logits"] + 0.5 * output["rule_hazard_logits"]
    assert torch.allclose(output["fused_hazard_logits"], expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_loss_backpropagates_through_every_variant(variant: str):
    model = _small_model(variant).train()
    output = model(**_inputs())
    targets = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    criterion = DRFPLoss(
        pos_weight=(1.0, 2.0, 3.0, 4.0),
        horizon_weight=(1.0, 1.0, 1.0, 2.0),
    )
    loss = criterion(output, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)


def test_full_loss_has_fused_and_auxiliary_components_only():
    output = _small_model("gated_fusion").eval()(**_inputs())
    targets = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    parts = DRFPLoss()(output, targets, return_components=True)
    assert set(parts) == {"fused", "deep_aux", "rule_aux", "total"}
    assert torch.allclose(
        parts["total"],
        (parts["fused"] + 0.25 * parts["deep_aux"] + 0.25 * parts["rule_aux"])
        / 1.5,
    )


def test_weighted_bce_respects_valid_mask():
    probabilities = torch.tensor([[0.2, 0.4, 0.7, 0.8]])
    targets = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    actual = multi_horizon_weighted_bce(probabilities, targets, valid_mask=mask)
    expected = -(torch.log(torch.tensor(0.8)) + torch.log(torch.tensor(0.6))) / 2.0
    assert torch.allclose(actual, expected)


def test_parameter_statistics_are_consistent_and_variant_specific():
    temporal = _small_model("temporal_only")
    full = _small_model("gated_fusion")
    temporal_stats = parameter_statistics(temporal)
    full_stats = parameter_statistics(full)
    assert temporal_stats["trainable"] == count_trainable_parameters(temporal)
    assert temporal_stats["total"] == temporal_stats["trainable"] + temporal_stats["frozen"]
    assert full_stats["trainable"] > temporal_stats["trainable"]
