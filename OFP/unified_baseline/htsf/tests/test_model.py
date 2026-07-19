import sys
import unittest
from pathlib import Path

import torch


HTSF_DIR = Path(__file__).resolve().parents[1]
UNIFIED_DIR = HTSF_DIR.parent
sys.path[:0] = [str(HTSF_DIR), str(UNIFIED_DIR)]

from helpers import tiny_payload
from ofp_htsf.config import HTSFConfig
from ofp_htsf.model import HTSFEncoder
from ofp_htsf.variants import get_variant


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = HTSFConfig.from_dict(tiny_payload())
        self.raw = torch.randn(3, 6, 12)
        self.raw_mask = torch.ones_like(self.raw)
        self.engineered = torch.randn(3, 118)
        self.engineered_mask = torch.ones_like(self.engineered)

    def test_all_learned_variants_return_same_representation_width(self):
        for name in (
            "temporal_only",
            "expert_stat_only",
            "direct_concat",
            "cross_attention",
            "htsf_fusion",
            "htsf_fusion_b2_skip",
        ):
            model = HTSFEncoder(self.config, get_variant(name))
            output = model(self.raw, self.raw_mask, self.engineered, self.engineered_mask)
            self.assertEqual(tuple(output.representations["representation"].shape), (3, 8))
            self.assertEqual(tuple(output.auxiliary_logit.shape), (3,))

    def test_temporal_backbone_receives_gradient_without_detach(self):
        model = HTSFEncoder(self.config, get_variant("temporal_only"))
        output = model(self.raw, self.raw_mask, self.engineered, self.engineered_mask)
        output.auxiliary_logit.sum().backward()
        gradient = model.temporal.patch_projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_full_htsf_has_gradient_in_both_branches_attention_and_gate(self):
        model = HTSFEncoder(self.config, get_variant("htsf_fusion"))
        output = model(self.raw, self.raw_mask, self.engineered, self.engineered_mask)
        output.auxiliary_logit.sum().backward()
        gradients = (
            model.temporal.patch_projection.weight.grad,
            model.engineered.network[0].weight.grad,
            model.fusion.attention.in_proj_weight.grad,
            model.fusion.gate[1].weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_temporal_representation_uses_endpoint_and_channel_identity(self):
        model = HTSFEncoder(self.config, get_variant("temporal_only"))
        model.eval()
        with torch.no_grad():
            original = model.temporal(self.raw, self.raw_mask)
            endpoint_changed = self.raw.clone()
            endpoint_changed[:, -1, :] += 1000.0
            changed = model.temporal(endpoint_changed, self.raw_mask)
            permuted = model.temporal(self.raw[:, :, [1, 0, *range(2, 12)]], self.raw_mask)
        self.assertGreater(float((original - changed).abs().max()), 1e-6)
        self.assertGreater(float((original - permuted).abs().max()), 1e-6)

    def test_attention_and_gate_are_separate_ablation_mechanisms(self):
        attention_model = HTSFEncoder(self.config, get_variant("cross_attention"))
        attention_output = attention_model(
            self.raw, self.raw_mask, self.engineered, self.engineered_mask
        )
        self.assertTrue(torch.all(attention_output.diagnostics["gate"] == 0.5))
        full_model = HTSFEncoder(self.config, get_variant("htsf_fusion"))
        full_output = full_model(
            self.raw, self.raw_mask, self.engineered, self.engineered_mask
        )
        self.assertEqual(tuple(full_output.diagnostics["attention"].shape), (3, 2, 2, 2))
        self.assertTrue(torch.all(full_output.diagnostics["gate"] > 0))
        self.assertTrue(torch.all(full_output.diagnostics["gate"] < 1))

    def test_full_with_half_gate_is_numerically_the_cross_attention_ablation(self):
        attention_model = HTSFEncoder(self.config, get_variant("cross_attention"))
        full_model = HTSFEncoder(self.config, get_variant("htsf_fusion"))
        full_model.load_state_dict(attention_model.state_dict(), strict=False)
        torch.nn.init.zeros_(full_model.fusion.gate[1].weight)
        torch.nn.init.zeros_(full_model.fusion.gate[1].bias)
        attention_model.eval()
        full_model.eval()
        with torch.no_grad():
            attention_output = attention_model(
                self.raw, self.raw_mask, self.engineered, self.engineered_mask
            ).representations["fused"]
            full_output = full_model(
                self.raw, self.raw_mask, self.engineered, self.engineered_mask
            ).representations["fused"]
        torch.testing.assert_close(attention_output, full_output, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
