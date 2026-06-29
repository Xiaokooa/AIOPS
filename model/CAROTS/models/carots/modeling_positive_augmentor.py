import random

import torch
import torch.nn as nn


class PositiveAugmentor(nn.Module):
	def __init__(self, cfg):
		super(PositiveAugmentor, self).__init__()
		self.cfg = cfg
		self.cfg_positive_augmentor = cfg.CAROTS.POSITIVE_AUGMENTOR

	def set_causal_discoverer(self, causal_discoverer):
		self.causal_discoverer = causal_discoverer
		self.cfg_causal_discoverer = self.cfg.CUTS_PLUS

	def forward(self, x, causality_mtx):
		x_pos = x.clone()
		causality_mtx = (causality_mtx > 0.5).int()
		start_node = random.choice(range(len(causality_mtx)))
		effects = [i for i, val in enumerate(causality_mtx[start_node]) if val == 1]
		self.start_node, self.effects = start_node, effects
		x_pos[:, self.cfg_causal_discoverer.INPUT_STEP, start_node] += torch.randn(x.size(0)).cuda() * self.cfg_positive_augmentor.NOISE_LEVEL
		with torch.no_grad():
			Graph = (self.causal_discoverer.causality_mtx > 0.5).float()
			Graph = Graph[None].expand(x.size(0), -1, -1)
			x_out = self.causal_discoverer(x_pos[:, :self.cfg_causal_discoverer.INPUT_STEP], Graph).transpose(1, 2)
		x_pos[:, -self.cfg_causal_discoverer.DATA_PRED.PRED_STEP, effects] = x_out[:, -self.cfg_causal_discoverer.DATA_PRED.PRED_STEP, effects]

		return x_pos