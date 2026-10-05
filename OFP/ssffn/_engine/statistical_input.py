# Numerical core retained for compatibility with the archived experiments.
"""Feature-content ablations of A4; preserve two token slots and shared SIT."""
from collections import OrderedDict
import hashlib
import torch
from torch import nn
from OFP.ssffn._engine.initialization import ExpertFollowupSIT

VARIANTS = {'full160': 160, 'no_rules84': 84, 'statistics80': 80}


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class StatisticalAblationSIT(ExpertFollowupSIT):
    def __init__(self, *args, feature_variant='full160', **kwargs):
        assert kwargs.get('followup', 'a4_typed') == 'a4_typed'
        super().__init__(*args, **kwargs)
        names = kwargs['stat_feature_names']
        assert len(names) == 160
        if feature_variant == 'full160':
            keep = list(range(160))
        elif feature_variant == 'no_rules84':
            keep = [i for i, n in enumerate(names) if not n.startswith(('Ma', 'Ru'))]
        elif feature_variant == 'statistics80':
            keep = [i for i, n in enumerate(names) if n.startswith('Fe')]
        else:
            raise ValueError(feature_variant)
        assert len(keep) == VARIANTS[feature_variant]
        reference_hash = parameter_hash(self)
        rng = torch.random.get_rng_state()
        if feature_variant != 'full160':
            source = self.expert_input
            groups = OrderedDict()
            layers = []
            for j, (name, indices) in enumerate(source.groups.items()):
                retained = [i for i in indices if i in keep]
                columns = [indices.index(i) for i in retained]
                # A zero-input projection is a learned constant token. Retaining
                # its bias/identity controls token count without moving features.
                layer = nn.Linear(len(retained), 64)
                with torch.no_grad():
                    layer.weight.copy_(source.projections[j].weight[:, columns])
                    layer.bias.copy_(source.projections[j].bias)
                groups[name] = retained
                setattr(source, f'indices_{j}', torch.tensor(retained, dtype=torch.long))
                layers.append(layer)
            source.groups = groups
            source.projections = nn.ModuleList(layers)
        torch.random.set_rng_state(rng)
        self.feature_variant = feature_variant
        self.temporal_encoder_cfg.update(
            feature_variant=feature_variant,
            active_expert_features=len(keep), active_statistic_features=len(keep),
            cached_statistic_features=160,
            active_feature_names=[names[i] for i in keep],
            removed_feature_names=[n for i, n in enumerate(names) if i not in keep],
            expert_groups={k: [names[i] for i in ids] for k, ids in self.expert_input.groups.items()},
            token_feature_dimensions=[len(ids) for ids in self.expert_input.groups.values()],
            constant_indicator_token=feature_variant == 'statistics80',
            a4_reference_initialization_sha256=reference_hash,
            initialization='Build identical A4; prune only removed feature projection columns; preserve all remaining weights, biases, token identities and RNG',
            initialization_sha256=parameter_hash(self),
            hss_scope='Unchanged original 64/8 HSS, including initial signal; only neural input features ablated')
