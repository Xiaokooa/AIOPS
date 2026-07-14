from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from fgofp.losses import (
    combined_future_guided_loss,
    fgl_kl_loss,
    module_normalized_cross_entropy,
)
from fgofp.model import (
    CausalSeq2SeqTCN,
    RuleGuidedResidualTCN,
    build_teacher_student,
)
from fgofp.rules import RULE_FEATURE_COUNT


def _inputs(batch: int = 2, time: int = 16):
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(batch, time, 12, generator=generator)
    raw_mask = torch.ones_like(raw, dtype=torch.bool)
    delta = torch.ones(batch, time)
    padding = torch.ones(batch, time, dtype=torch.bool)
    return raw, raw_mask, delta, padding


def _small_model(blocks: int = 3) -> CausalSeq2SeqTCN:
    return CausalSeq2SeqTCN(
        hidden_channels=16,
        dilation_blocks=blocks,
        kernel_size=3,
        dropout=0.0,
    )


def test_model_shape_padding_zero_and_independent_teacher_student():
    raw, raw_mask, delta, padding = _inputs()
    padding[1, 11:] = False
    raw[1, 11:] = 12345.0
    model = _small_model().eval()
    logits = model(raw, raw_mask, delta, padding)
    assert logits.shape == (2, 16, 2)
    assert torch.equal(logits[1, 11:], torch.zeros_like(logits[1, 11:]))

    config = SimpleNamespace(
        hidden_channels=16, dilation_blocks=2, kernel_size=3, dropout=0.0
    )
    student, teacher = build_teacher_student(config)
    assert student is not teacher
    student_parameter = next(student.parameters())
    teacher_parameter = next(teacher.parameters())
    assert student_parameter.data_ptr() != teacher_parameter.data_ptr()


def test_future_input_perturbation_cannot_change_past_logits():
    raw, raw_mask, delta, padding = _inputs(batch=1, time=24)
    model = _small_model(blocks=4).eval()
    split = 12
    original = model(raw, raw_mask, delta, padding)
    perturbed_raw = raw.clone()
    perturbed_mask = raw_mask.clone()
    perturbed_delta = delta.clone()
    perturbed_raw[:, split:] = torch.randn_like(perturbed_raw[:, split:]) * 1000.0
    perturbed_mask[:, split:, :5] = False
    perturbed_delta[:, split:] = 77.0
    changed = model(perturbed_raw, perturbed_mask, perturbed_delta, padding)
    assert torch.allclose(original[:, :split], changed[:, :split], atol=1e-6, rtol=0)


def test_receptive_field_matches_one_conv_per_exponential_dilation():
    model = CausalSeq2SeqTCN(
        hidden_channels=16,
        dilation_blocks=4,
        kernel_size=3,
        dropout=0.0,
    )
    assert model.dilations == (1, 2, 4, 8)
    assert model.receptive_field_steps == 1 + 2 * (1 + 2 + 4 + 8)


def _rule_inputs(batch: int, time: int):
    generator = torch.Generator().manual_seed(19)
    margin = torch.randn(batch, time, RULE_FEATURE_COUNT, generator=generator)
    mask = torch.ones_like(margin, dtype=torch.bool)
    hard = (margin > 0).any(dim=-1)
    return margin, mask, hard


def test_zero_initialized_rule_residual_is_exact_temporal_model() -> None:
    raw, raw_mask, delta, padding = _inputs(batch=2, time=12)
    margin, rule_mask, hard = _rule_inputs(2, 12)
    model = RuleGuidedResidualTCN(
        hidden_channels=16,
        dilation_blocks=3,
        kernel_size=3,
        dropout=0.0,
        rule_hidden_channels=8,
    ).eval()

    temporal = CausalSeq2SeqTCN.forward(model, raw, raw_mask, delta, padding)
    fused = model(
        raw,
        raw_mask,
        delta,
        padding,
        rule_margin=margin,
        rule_mask=rule_mask,
        rule_hard=hard,
    )

    assert torch.equal(fused, temporal)


def test_missing_rules_reduce_exactly_to_temporal_even_after_training_update() -> None:
    raw, raw_mask, delta, padding = _inputs(batch=1, time=10)
    margin, rule_mask, hard = _rule_inputs(1, 10)
    rule_mask.zero_()
    model = RuleGuidedResidualTCN(
        hidden_channels=16,
        dilation_blocks=2,
        dropout=0.0,
        rule_hidden_channels=8,
    ).eval()
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(0.25)
        model.residual_head[-1].bias.fill_(1.0)

    temporal = CausalSeq2SeqTCN.forward(model, raw, raw_mask, delta, padding)
    fused = model(
        raw,
        raw_mask,
        delta,
        padding,
        rule_margin=margin,
        rule_mask=rule_mask,
        rule_hard=hard,
    )

    assert torch.equal(fused, temporal)


def test_future_rule_perturbation_cannot_change_past_fused_logits() -> None:
    raw, raw_mask, delta, padding = _inputs(batch=1, time=20)
    margin, rule_mask, hard = _rule_inputs(1, 20)
    model = RuleGuidedResidualTCN(
        hidden_channels=16,
        dilation_blocks=3,
        dropout=0.0,
        rule_hidden_channels=8,
    ).eval()
    with torch.no_grad():
        model.residual_head[-1].weight.fill_(0.05)
    original = model(
        raw,
        raw_mask,
        delta,
        padding,
        rule_margin=margin,
        rule_mask=rule_mask,
        rule_hard=hard,
    )
    split = 9
    changed_margin = margin.clone()
    changed_mask = rule_mask.clone()
    changed_hard = hard.clone()
    changed_margin[:, split:] *= -100.0
    changed_mask[:, split:, :5] = False
    changed_hard[:, split:] = ~changed_hard[:, split:]
    changed = model(
        raw,
        raw_mask,
        delta,
        padding,
        rule_margin=changed_margin,
        rule_mask=changed_mask,
        rule_hard=changed_hard,
    )

    assert torch.allclose(original[:, :split], changed[:, :split], atol=0.0, rtol=0.0)


def test_module_normalized_ce_gives_short_and_long_modules_equal_weight():
    # Module 0 has one valid row with a deliberately poor prediction. Module 1
    # has four valid rows with easy predictions. The expected value is the mean
    # of the two per-module means, not the mean of all five rows.
    logits = torch.tensor(
        [
            [[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[2.0, -2.0], [2.0, -2.0], [2.0, -2.0], [2.0, -2.0]],
        ],
        requires_grad=True,
    )
    target = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.tensor([[True, False, False, False], [True, True, True, True]])
    loss, coverage = module_normalized_cross_entropy(
        logits, target, mask, return_coverage=True
    )
    module_zero = F.cross_entropy(logits[0, :1], target[0, :1])
    module_one = F.cross_entropy(logits[1], target[1])
    assert torch.allclose(loss, (module_zero + module_one) / 2)
    assert coverage == {"ce_modules": 2, "ce_rows": 5, "ce_positive_rows": 0}


def test_identical_aligned_distributions_have_zero_kl():
    student = torch.tensor([[[1.0, -1.0], [0.2, 0.4]]], requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    index = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = fgl_kl_loss(student, teacher, index, mask, temperature=3.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_fgl_detaches_teacher_and_backpropagates_to_student():
    student = torch.tensor([[[0.0, 0.0], [0.5, -0.5]]], requires_grad=True)
    teacher = torch.tensor([[[2.0, -2.0], [-2.0, 2.0]]], requires_grad=True)
    index = torch.tensor([[1, 0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = fgl_kl_loss(student, teacher, index, mask, temperature=2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) > 0
    assert teacher.grad is None


def test_sparse_fgl_batch_keeps_uncovered_module_as_zero_weighted_term():
    student = torch.tensor(
        [[[0.0, 0.0]], [[0.0, 0.0]]], requires_grad=True
    )
    teacher = torch.tensor([[[2.0, -2.0]], [[-2.0, 2.0]]])
    index = torch.tensor([[0], [-1]])
    mask = torch.tensor([[True], [False]])
    batch_loss = fgl_kl_loss(student, teacher, index, mask, temperature=2.0)
    single_loss = fgl_kl_loss(
        student[:1], teacher[:1], index[:1], mask[:1], temperature=2.0
    )
    assert torch.allclose(batch_loss, single_loss / 2.0)
    scaled = combined_future_guided_loss(
        student_logits=student,
        target=torch.zeros(2, 1, dtype=torch.long),
        target_mask=torch.ones(2, 1, dtype=torch.bool),
        teacher_logits=teacher,
        teacher_index=index,
        fgl_mask=mask,
        alpha=0.0,
        temperature=2.0,
        fgl_coverage_scale=2.0,
    )
    assert torch.allclose(scaled.fgl_kl, single_loss)


def test_no_fgl_pairs_returns_differentiable_zero_and_coverage():
    student = torch.randn(2, 3, 2, requires_grad=True)
    teacher = torch.randn(2, 3, 2, requires_grad=True)
    index = torch.full((2, 3), -1)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    loss, coverage = fgl_kl_loss(
        student, teacher, index, mask, return_coverage=True
    )
    assert loss.item() == pytest.approx(0.0)
    assert loss.requires_grad
    loss.backward()
    assert student.grad is not None
    assert torch.equal(student.grad, torch.zeros_like(student.grad))
    assert teacher.grad is None
    assert coverage == {"fgl_modules": 0, "fgl_pairs": 0}


def test_combined_loss_matches_alpha_formula_and_reports_coverage():
    student = torch.tensor(
        [[[1.0, -1.0], [-0.5, 0.5], [0.2, -0.2]]], requires_grad=True
    )
    teacher = torch.tensor(
        [[[-1.0, 1.0], [0.4, -0.4], [-0.2, 0.2]]], requires_grad=True
    )
    target = torch.tensor([[0, 1, 0]])
    target_mask = torch.ones(1, 3, dtype=torch.bool)
    teacher_index = torch.tensor([[1, 2, -1]])
    fgl_mask = torch.tensor([[True, True, False]])
    output = combined_future_guided_loss(
        student_logits=student,
        target=target,
        target_mask=target_mask,
        teacher_logits=teacher,
        teacher_index=teacher_index,
        fgl_mask=fgl_mask,
        alpha=0.7,
        temperature=2.0,
    )
    assert torch.allclose(output.total, 0.7 * output.cross_entropy + 0.3 * output.fgl_kl)
    assert output.coverage == {
        "ce_modules": 1,
        "ce_rows": 3,
        "ce_positive_rows": 1,
        "fgl_modules": 1,
        "fgl_pairs": 2,
    }
