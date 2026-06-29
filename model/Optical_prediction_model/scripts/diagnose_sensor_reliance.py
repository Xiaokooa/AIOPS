"""Diagnose whether MFP-Former truly relies only on temperature.

Tests:
1. Gradient-based sensor importance (input-level gradients)
2. Sensor ablation: zero out each sensor and measure F1 drop
3. Bypass vs concept path contribution analysis
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from model.Optical_prediction_model.scripts.run_dl_l4_typeaware_R4 import (
    r4_task_cfg, TypeAwareFailureWindowDataset, collate_type, TYPE_NAMES,
)
from model.Optical_prediction_model.deep_learning.data import (
    WindowSliceCache, compute_norm_stats,
)
from model.Optical_prediction_model.deep_learning.fault_type_labels import annotate_fault_type_frames
from model.Optical_prediction_model.deep_learning.train import load_task_frames

EXP_DIR = PROJECT / "output" / "Optical_prediction_model" / "experiments" / "dl_mfpformer_R4_full"
SENSORS = [
    "temperature", "current", "currentTXPower", "currentRXPower",
    "currentMultiRXPower1", "currentMultiRXPower2", "currentMultiRXPower3", "currentMultiRXPower4",
    "currentMultiTXPower1", "currentMultiTXPower2", "currentMultiTXPower3", "currentMultiTXPower4",
]
SHORT = ["Temp","Curr","TXPwr","RXPwr","mRX1","mRX2","mRX3","mRX4","mTX1","mTX2","mTX3","mTX4"]


def load_model_and_data():
    """Load trained model and test dataset."""
    from model.Optical_prediction_model.deep_learning.typeaware_transformer import (
        TypeAwareTransformerCfg, TypeAwareCrossAttnTransformer,
    )
    summary = json.load(open(EXP_DIR / "results" / "summary.json"))
    mcfg = summary["model_cfg"]
    cfg = TypeAwareTransformerCfg(**mcfg)
    model = TypeAwareCrossAttnTransformer(cfg)
    ckpt = torch.load(EXP_DIR / "models" / "model.pt", map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load test data
    task_cfg = r4_task_cfg()
    frames = load_task_frames(task_cfg=task_cfg, tps_lead_minutes=task_cfg.lead_minutes)
    frames, _ = annotate_fault_type_frames(frames)
    obs_steps = int(frames["train"].iloc[0]["obs_end_idx"] - frames["train"].iloc[0]["obs_start_idx"] + 1)
    slice_cache = WindowSliceCache()
    norm = compute_norm_stats(frames["train"], slice_cache, obs_steps, max_modules=400)
    test_ds = TypeAwareFailureWindowDataset(frames["test"], obs_steps, slice_cache, norm, TYPE_NAMES)
    threshold = summary["threshold"]
    return model, test_ds, device, threshold


def test_gradient_importance(model, test_ds, device):
    """Compute input-gradient-based sensor importance on positive samples."""
    print("\n=== Test 1: Gradient-based Sensor Importance ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_type)

    grad_accum = torch.zeros(12, device=device)
    count = 0

    for batch in loader:
        x = batch[0].to(device).requires_grad_(True)
        mask = batch[1].to(device)
        y = batch[2].to(device)
        pos_mask = y == 1
        if pos_mask.sum() == 0:
            continue

        out = model(x, mask)
        logit = out[0]
        # Only backprop on positive samples
        loss = logit[pos_mask].sum()
        loss.backward()

        # Input gradients: |grad| averaged over time, for positive samples
        g = x.grad[pos_mask].abs()  # (n_pos, T, M)
        g_per_sensor = g.mean(dim=1)  # (n_pos, M)
        grad_accum += g_per_sensor.sum(dim=0)
        count += pos_mask.sum().item()
        x.grad = None

    grad_importance = (grad_accum / count).cpu().numpy()
    grad_importance = grad_importance / grad_importance.sum()

    print("  Gradient-based sensor importance (positive samples):")
    for i in np.argsort(-grad_importance):
        print(f"    {SHORT[i]:6s}: {grad_importance[i]:.4f}")
    return grad_importance


def test_sensor_ablation(model, test_ds, device, threshold):
    """Zero out each sensor and measure prediction change."""
    print("\n=== Test 2: Sensor Ablation (zero-out each sensor) ===")
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_type)

    # Baseline predictions
    all_scores_base = []
    all_y = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            mask = batch[1].to(device)
            y = batch[2].numpy()
            out = model(x, mask)
            s = torch.sigmoid(out[0]).cpu().numpy()
            all_scores_base.append(s)
            all_y.append(y)
    all_scores_base = np.concatenate(all_scores_base)
    all_y = np.concatenate(all_y)

    base_preds = (all_scores_base >= threshold).astype(int)
    base_f1 = f1_score(all_y, base_preds)
    print(f"  Baseline F1={base_f1:.4f} (thr={threshold})")

    # Ablate each sensor
    for sensor_id in range(12):
        all_scores_abl = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device).clone()
                mask = batch[1].to(device).clone()
                # Zero out sensor
                x[:, :, sensor_id] = 0.0
                mask[:, :, sensor_id] = 0.0
                out = model(x, mask)
                s = torch.sigmoid(out[0]).cpu().numpy()
                all_scores_abl.append(s)
        all_scores_abl = np.concatenate(all_scores_abl)
        abl_preds = (all_scores_abl >= threshold).astype(int)
        abl_f1 = f1_score(all_y, abl_preds)
        f1_drop = base_f1 - abl_f1

        # Score change on positive samples
        pos_mask = all_y == 1
        score_change = (all_scores_abl[pos_mask] - all_scores_base[pos_mask]).mean()
        print(f"    Ablate {SHORT[sensor_id]:6s}: F1={abl_f1:.4f} (drop={f1_drop:+.4f})  "
              f"pos_score_change={score_change:+.4f}")


def test_path_decomposition(model, test_ds, device):
    """Decompose bypass vs concept contribution to final logit."""
    print("\n=== Test 3: Bypass vs Concept Path Decomposition ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_type)

    global_probs = []
    concept_probs = []
    mix_vals = []
    ys = []

    with torch.no_grad():
        mix = torch.sigmoid(model.concept_risk_mix_logit).item()
        print(f"  concept_risk_mix = {mix:.4f}")

        for batch in loader:
            x = batch[0].to(device)
            mask = batch[1].to(device)
            y = batch[2].numpy()

            # Run forward manually to get intermediate values
            if model.use_norm:
                from model.Optical_prediction_model.deep_learning.typeaware_transformer import _mask_aware_norm
                x_model = _mask_aware_norm(x, mask)
            else:
                x_model = x * mask

            if model.use_mask_channel:
                inp = torch.cat([x_model, mask], dim=-1)
            else:
                inp = x_model

            inp = model._pool_time_tokens(inp)
            length = inp.size(1)
            time_tokens = model.input_proj(inp)
            time_tokens = time_tokens + model.pos_embedding[:, :length, :]
            time_tokens = model.time_encoder(time_tokens)

            batch_size = x.size(0)
            sq = model.sensor_queries.unsqueeze(0).expand(batch_size, -1, -1)
            sensor_tokens, _ = model.sensor_time_attn(query=sq, key=time_tokens, value=time_tokens)

            if model.sensor_series_proj is not None:
                sensor_vals = x_model.transpose(1, 2)
                sensor_mask = mask.transpose(1, 2)
                denom = sensor_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                s_mean = (sensor_vals * sensor_mask).sum(dim=-1, keepdim=True) / denom
                s_var = ((sensor_vals - s_mean) ** 2 * sensor_mask).sum(dim=-1, keepdim=True) / denom
                s_std = torch.sqrt(s_var + 1e-5)
                s_min = sensor_vals.masked_fill(sensor_mask < 0.5, 1e9).min(dim=-1, keepdim=True).values
                s_max = sensor_vals.masked_fill(sensor_mask < 0.5, -1e9).max(dim=-1, keepdim=True).values
                s_last = sensor_vals[:, :, -1:]
                s_first = sensor_vals[:, :, :1]
                s_slope = (s_last - s_first) / sensor_vals.size(-1)
                stats = torch.cat([s_mean, s_std, s_min, s_max, s_slope, s_last], dim=-1)
                sensor_tokens = sensor_tokens + model.sensor_series_proj(stats)
            sensor_tokens = model.sensor_token_norm(sensor_tokens)

            concept_tokens = model.concept_queries.unsqueeze(0).expand(batch_size, -1, -1)
            bias = model.attn_bias
            for mfp in model.mfp_layers:
                sensor_tokens, concept_tokens, _, _ = mfp(sensor_tokens, concept_tokens, attn_bias=bias)

            type_logits = model.type_head(concept_tokens).squeeze(-1)
            concept_gate = torch.sigmoid(type_logits)
            pooled = sensor_tokens.mean(dim=1)
            import math
            time_alpha = torch.softmax(model.time_pool_score(time_tokens), dim=1)
            time_context = (time_tokens * time_alpha).sum(dim=1)
            bypass = model.bypass_proj(time_context) * model.cfg.bypass_ratio
            global_logit = model.global_head(torch.cat([pooled, bypass], dim=-1)).squeeze(-1)

            gp = torch.sigmoid(global_logit).cpu().numpy()
            any_cp = (1.0 - torch.prod((1.0 - concept_gate).clamp(min=1e-6, max=1.0), dim=-1)).cpu().numpy()

            global_probs.append(gp)
            concept_probs.append(any_cp)
            ys.append(y)

    global_probs = np.concatenate(global_probs)
    concept_probs = np.concatenate(concept_probs)
    ys = np.concatenate(ys)
    pos = ys == 1

    print(f"\n  On POSITIVE samples (n={pos.sum()}):")
    print(f"    global_prob:  mean={global_probs[pos].mean():.4f}  median={np.median(global_probs[pos]):.4f}")
    print(f"    concept_prob: mean={concept_probs[pos].mean():.4f}  median={np.median(concept_probs[pos]):.4f}")
    print(f"    fused = (1-{mix:.3f})*global + {mix:.3f}*concept")
    fused_pos = (1-mix)*global_probs[pos] + mix*concept_probs[pos]
    print(f"    fused_prob:   mean={fused_pos.mean():.4f}  median={np.median(fused_pos):.4f}")

    print(f"\n  On NEGATIVE samples (n={(~pos).sum()}):")
    print(f"    global_prob:  mean={global_probs[~pos].mean():.4f}")
    print(f"    concept_prob: mean={concept_probs[~pos].mean():.4f}")

    # Which path drives the positive predictions more?
    global_contrib = (1-mix) * global_probs[pos]
    concept_contrib = mix * concept_probs[pos]
    print(f"\n  Contribution to fused_prob (positive samples):")
    print(f"    Global path:  mean={global_contrib.mean():.4f}")
    print(f"    Concept path: mean={concept_contrib.mean():.4f}")
    print(f"    Ratio global/(global+concept): {global_contrib.mean()/(global_contrib.mean()+concept_contrib.mean()):.3f}")


if __name__ == "__main__":
    model, test_ds, device, threshold = load_model_and_data()
    test_gradient_importance(model, test_ds, device)
    test_sensor_ablation(model, test_ds, device, threshold)
    test_path_decomposition(model, test_ds, device)
