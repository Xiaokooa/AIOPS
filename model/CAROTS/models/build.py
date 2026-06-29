import torch

from models.carots.modeling_carots import CAROTS
from models.timesnet.modeling_timesnet import TimesNet


def build_model(cfg):
    model_name = cfg.MODEL.NAME

    model_mapping = {
        "CAROTS": CAROTS,
        "TIMESNET": TimesNet,
    }

    if model_name in model_mapping:
        model = model_mapping[model_name](cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if torch.cuda.is_available():
        model = model.cuda()

    return model
