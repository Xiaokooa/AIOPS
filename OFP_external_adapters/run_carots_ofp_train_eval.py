from __future__ import annotations

import argparse
import gc
import json
import math
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_external_adapters.common import (
    SENSORS,
    AlarmCfg,
    aggregate_results,
    apply_alarm_strategy,
    evaluate_prediction_folder,
    file_label_map,
    files_for_index_fold,
    metrics_to_dict,
    read_index,
    select_threshold,
    split_train_val_files,
    write_prediction_frames,
)


def read_sensor_frame(path: Path) -> pd.DataFrame:
    cols = {"timestamp", "anomaly", *SENSORS}
    frame = pd.read_csv(path, usecols=lambda col: col in cols)
    for col in ["timestamp", "anomaly", *SENSORS]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp")


def finite_sensor_mask(frame: pd.DataFrame) -> np.ndarray:
    arr = frame[SENSORS].to_numpy(dtype=np.float64)
    return np.isfinite(arr).all(axis=1)


def fit_normalizer(data_dir: Path, train_files: list[str], label_by_file: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    def accumulate(normal_modules_only: bool) -> tuple[np.ndarray, np.ndarray, int]:
        sums = np.zeros(len(SENSORS), dtype=np.float64)
        sums_sq = np.zeros(len(SENSORS), dtype=np.float64)
        count = 0
        for name in train_files:
            if normal_modules_only and int(label_by_file.get(name, 0)) != 0:
                continue
            frame = read_sensor_frame(data_dir / name)
            arr = frame[SENSORS].to_numpy(dtype=np.float64)
            mask = finite_sensor_mask(frame)
            if "anomaly" in frame.columns:
                anomaly = pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=float)
                mask = mask & (anomaly <= 0)
            arr = arr[mask]
            if len(arr) == 0:
                continue
            sums += arr.sum(axis=0)
            sums_sq += (arr**2).sum(axis=0)
            count += int(arr.shape[0])
        return sums, sums_sq, count

    sums, sums_sq, count = accumulate(normal_modules_only=True)
    if count <= 0:
        sums, sums_sq, count = accumulate(normal_modules_only=False)
    if count <= 0:
        raise ValueError("No finite non-anomaly rows found for CAROTS normalizer")
    mean = sums / count
    var = np.maximum(sums_sq / count - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def normalized_arrays(data_dir: Path, file_name: str, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = read_sensor_frame(data_dir / file_name)
    timestamps = frame["timestamp"].to_numpy(dtype=np.int64)
    values = frame[SENSORS].to_numpy(dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = ((values - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-6)).astype(np.float32)
    anomaly = (
        pd.to_numeric(frame["anomaly"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if "anomaly" in frame.columns
        else np.zeros(len(frame), dtype=float)
    )
    return timestamps, values, anomaly


def valid_window_ends(values: np.ndarray, anomaly: np.ndarray, win_size: int, normal_only: bool) -> np.ndarray:
    n = int(values.shape[0])
    if n < int(win_size):
        return np.empty(0, dtype=np.int64)
    finite = np.isfinite(values).all(axis=1)
    ends = []
    for end in range(int(win_size) - 1, n):
        start = end - int(win_size) + 1
        if not finite[start : end + 1].all():
            continue
        if normal_only and np.any(anomaly[start : end + 1] > 0):
            continue
        ends.append(end)
    return np.asarray(ends, dtype=np.int64)


class OFPCAROTSWindowDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Path,
        file_names: list[str],
        label_by_file: dict[str, int],
        mean: np.ndarray,
        std: np.ndarray,
        win_size: int,
        samples_per_file: int,
        seed: int,
        train_on_normal_modules_only: bool,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.file_names = list(file_names)
        self.label_by_file = label_by_file
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.win_size = int(win_size)
        self.samples_per_file = int(samples_per_file)
        self.seed = int(seed)
        self.train_on_normal_modules_only = bool(train_on_normal_modules_only)
        self.active_files = [
            name
            for name in self.file_names
            if (not self.train_on_normal_modules_only) or int(self.label_by_file.get(name, 0)) == 0
        ]
        if not self.active_files:
            self.active_files = list(self.file_names)

    def __len__(self) -> int:
        return max(1, len(self.active_files) * max(1, self.samples_per_file))

    def __iter__(self):
        pairs = list(self.active_files)
        worker = get_worker_info()
        if worker is not None:
            pairs = pairs[worker.id :: worker.num_workers]
        for name in pairs:
            timestamps, values, anomaly = normalized_arrays(self.data_dir, name, self.mean, self.std)
            ends = valid_window_ends(values, anomaly, self.win_size, normal_only=True)
            if len(ends) == 0:
                ends = valid_window_ends(values, anomaly, self.win_size, normal_only=False)
            if len(ends) == 0:
                continue
            stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
            rng = np.random.default_rng(self.seed + int(stable))
            count = min(len(ends), max(1, self.samples_per_file))
            chosen = rng.choice(ends, size=count, replace=len(ends) < count)
            for end in np.sort(chosen):
                start = int(end) - self.win_size + 1
                yield torch.from_numpy(values[start : int(end) + 1].astype(np.float32))


class CAROTSOFPNet(nn.Module):
    """CAROTS-style encoder/projector for OFP module windows."""

    def __init__(self, n_features: int, hidden_dim: int, proj_dim: int, encoder: str = "lstm") -> None:
        super().__init__()
        self.encoder_name = str(encoder).lower()
        if self.encoder_name == "gru":
            self.encoder = nn.GRU(n_features, hidden_dim, num_layers=1, batch_first=True)
        elif self.encoder_name == "lstm":
            self.encoder = nn.LSTM(n_features, hidden_dim, num_layers=1, batch_first=True)
        else:
            raise ValueError("encoder must be lstm or gru")
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_name == "gru":
            _out, h = self.encoder(x)
        else:
            _out, (h, _c) = self.encoder(x)
        return h[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encode(x))


def positive_augment(x: torch.Tensor, noise_level: float) -> torch.Tensor:
    return x + torch.randn_like(x) * float(noise_level)


def negative_augment(x: torch.Tensor, bias_scale: float, percent: float) -> torch.Tensor:
    out = x.clone().detach()
    batch, steps, channels = out.shape
    n_steps = max(1, int(round(steps * float(percent))))
    n_channels = max(1, int(round(math.sqrt(channels))))
    step_idx = torch.randperm(steps, device=out.device)[:n_steps]
    chan_idx = torch.randperm(channels, device=out.device)[:n_channels]
    signs = torch.randint(0, 2, (batch, n_steps, n_channels), device=out.device).to(out.dtype) * 2 - 1
    for si, step in enumerate(step_idx):
        for ci, chan in enumerate(chan_idx):
            out[:, int(step), int(chan)] += signs[:, si, ci] * float(bias_scale)
    return out


def carots_contrastive_loss(z: torch.Tensor, sim_threshold: float, temperature: float) -> torch.Tensor:
    z = F.normalize(z.float(), p=2, dim=1)
    n_pos = z.size(0) // 2
    if n_pos <= 1:
        return z.new_tensor(0.0)
    sim = torch.matmul(z, z.T) / max(float(temperature), 1e-6)
    pos_block = torch.matmul(z[:n_pos], z[:n_pos].T)
    pos_mask = pos_block >= float(sim_threshold)
    eye = torch.eye(n_pos, device=z.device, dtype=torch.bool)
    pos_mask = pos_mask & ~eye
    if not pos_mask.any():
        half = n_pos // 2
        pos_mask = torch.zeros((n_pos, n_pos), device=z.device, dtype=torch.bool)
        if half > 0:
            idx = torch.arange(half, device=z.device)
            pos_mask[idx, idx + half] = True
            pos_mask[idx + half, idx] = True
    losses = []
    for i in range(n_pos):
        positives = torch.where(pos_mask[i])[0]
        if len(positives) == 0:
            continue
        neg_idx = torch.arange(n_pos, z.size(0), device=z.device)
        candidates = torch.cat([positives, neg_idx])
        logits = sim[i, candidates]
        target_mask = torch.zeros_like(logits, dtype=torch.bool)
        target_mask[: len(positives)] = True
        log_prob = logits - torch.logsumexp(logits, dim=0)
        losses.append(-torch.logsumexp(log_prob[target_mask], dim=0))
    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_one_epoch(
    model: CAROTSOFPNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    noise_level: float,
    bias_scale: float,
    bias_percent: float,
    sim_threshold: float,
    temperature: float,
) -> float:
    model.train()
    total = 0.0
    n_seen = 0
    for windows in loader:
        x = windows.to(device, non_blocking=True).float()
        x_pos = positive_augment(x, noise_level)
        x_seed = torch.cat([x, x_pos], dim=0)
        x_neg = negative_augment(x_seed, bias_scale, bias_percent)
        all_x = torch.cat([x_seed, x_neg], dim=0)
        z = model(all_x)
        loss = carots_contrastive_loss(z, sim_threshold, temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch = int(x.size(0))
        total += float(loss.detach().cpu()) * batch
        n_seen += batch
    return total / max(n_seen, 1)


@torch.no_grad()
def encode_windows(model: CAROTSOFPNet, windows: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, int(windows.size(0)), int(batch_size)):
        end = min(int(windows.size(0)), start + int(batch_size))
        outs.append(model(windows[start:end].to(device).float()).detach().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty((0, 1))


def make_windows(values: np.ndarray, win_size: int, stride: int) -> tuple[torch.Tensor, np.ndarray]:
    ends = np.arange(int(win_size) - 1, len(values), max(1, int(stride)), dtype=np.int64)
    if len(ends) == 0:
        return torch.empty((0, int(win_size), values.shape[1]), dtype=torch.float32), ends
    windows = np.stack([values[end - int(win_size) + 1 : end + 1] for end in ends]).astype(np.float32)
    return torch.from_numpy(windows), ends


@torch.no_grad()
def compute_centroid(
    model: CAROTSOFPNet,
    data_dir: Path,
    file_names: list[str],
    label_by_file: dict[str, int],
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> torch.Tensor:
    def collect(require_normal_module: bool) -> list[torch.Tensor]:
        embeddings = []
        used = 0
        for name in file_names:
            if require_normal_module and int(label_by_file.get(name, 0)) != 0:
                continue
            _ts, values, anomaly = normalized_arrays(data_dir, name, mean, std)
            ends = valid_window_ends(values, anomaly, int(args.win_size), normal_only=True)
            if len(ends) == 0:
                continue
            if int(args.centroid_windows_per_file) > 0 and len(ends) > int(args.centroid_windows_per_file):
                stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
                rng = np.random.default_rng(int(args.seed) + int(stable))
                ends = np.sort(rng.choice(ends, size=int(args.centroid_windows_per_file), replace=False))
            windows = np.stack([values[end - int(args.win_size) + 1 : end + 1] for end in ends]).astype(np.float32)
            emb = encode_windows(model, torch.from_numpy(windows), args.device, int(args.score_batch_size))
            embeddings.append(emb)
            used += int(emb.size(0))
            if int(args.max_centroid_windows) > 0 and used >= int(args.max_centroid_windows):
                break
        return embeddings

    embeddings = collect(require_normal_module=bool(args.train_on_normal_modules_only))
    if not embeddings and bool(args.train_on_normal_modules_only):
        embeddings = collect(require_normal_module=False)
    if not embeddings:
        raise ValueError("No windows available for CAROTS centroid")
    z = torch.cat(embeddings, dim=0)
    if int(args.max_centroid_windows) > 0 and int(z.size(0)) > int(args.max_centroid_windows):
        z = z[: int(args.max_centroid_windows)]
    return z.mean(dim=0, keepdim=True)


@torch.no_grad()
def score_file(
    model: CAROTSOFPNet,
    centroid: torch.Tensor,
    data_dir: Path,
    file_name: str,
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    timestamps, values, _anomaly = normalized_arrays(data_dir, file_name, mean, std)
    windows, ends = make_windows(values, int(args.win_size), int(args.score_stride))
    if int(windows.size(0)) == 0:
        return pd.DataFrame({"timestamp": timestamps.astype(np.int64), "score": np.zeros(len(timestamps), dtype=np.float32)})
    emb = encode_windows(model, windows, args.device, int(args.score_batch_size))
    score = torch.cdist(emb, centroid.cpu()).squeeze(-1).numpy().astype(np.float32)
    return pd.DataFrame({"timestamp": timestamps[ends].astype(np.int64), "score": score})


def score_files(
    model: CAROTSOFPNet,
    centroid: torch.Tensor,
    data_dir: Path,
    file_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for idx, name in enumerate(file_names, 1):
        frames[name] = score_file(model, centroid, data_dir, name, mean, std, args)
        if idx % 200 == 0:
            print(f"[carots-score] {idx}/{len(file_names)} files")
    return frames


def run_fold(args: argparse.Namespace, index_df: pd.DataFrame, fold: int, alarm_cfg: AlarmCfg) -> dict:
    started = time.time()
    torch.manual_seed(int(args.seed) + int(fold))
    np.random.seed(int(args.seed) + int(fold))
    label_by_file = file_label_map(index_df)
    train_files, test_files = files_for_index_fold(index_df, int(fold))
    if int(args.max_train_files) > 0:
        train_files = train_files[: int(args.max_train_files)]
    if int(args.max_test_files) > 0:
        test_files = test_files[: int(args.max_test_files)]
    fit_files, val_files = split_train_val_files(train_files, label_by_file, alarm_cfg, int(fold))
    run_dir = Path(args.out_root) / args.model_name / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[carots-train-init] fold={fold} train={len(fit_files)} val={len(val_files)} test={len(test_files)} "
        f"encoder={args.encoder} win={args.win_size} device={args.device}"
    )
    mean, std = fit_normalizer(args.data_dir, fit_files, label_by_file)
    dataset = OFPCAROTSWindowDataset(
        args.data_dir,
        fit_files,
        label_by_file,
        mean,
        std,
        int(args.win_size),
        int(args.samples_per_file),
        int(args.seed) + int(fold),
        bool(args.train_on_normal_modules_only),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    model = CAROTSOFPNet(len(SENSORS), int(args.hidden_dim), int(args.proj_dim), encoder=args.encoder).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        loss = train_one_epoch(
            model,
            loader,
            optimizer,
            args.device,
            float(args.noise_level),
            float(args.bias_scale),
            float(args.bias_percent),
            float(args.sim_threshold),
            float(args.temperature),
        )
        history.append({"epoch": epoch, "loss": float(loss)})
        print(f"[carots-train] fold={fold} epoch={epoch:02d} loss={loss:.6f}")
    centroid = compute_centroid(model, args.data_dir, fit_files, label_by_file, mean, std, args)
    val_scores = score_files(model, centroid, args.data_dir, val_files, mean, std, args)
    threshold, val_metrics, search_df = select_threshold(val_scores, args.data_dir, alarm_cfg)
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    search_df.to_csv(val_dir / "threshold_search.csv", index=False)
    pd.DataFrame({"Item": list(val_metrics.keys()), "Value": list(val_metrics.values())}).to_csv(
        val_dir / "best_evaluate_result.csv",
        index=False,
    )
    test_scores = score_files(model, centroid, args.data_dir, test_files, mean, std, args)
    score_dir = run_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in test_scores.items():
        frame.to_csv(score_dir / name, index=False)
    pred_frames = {name: apply_alarm_strategy(frame, threshold, alarm_cfg) for name, frame in test_scores.items()}
    pred_dir = run_dir / "predictions"
    eval_dir = run_dir / "evaluation"
    test_rows = write_prediction_frames(pred_frames, pred_dir)
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, args.data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    metrics = metrics_to_dict(summary_df)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "centroid": centroid,
            "mean": mean,
            "std": std,
            "args": vars(args),
        },
        run_dir / "model.pt",
    )
    result = {
        "model": args.model_name,
        "fold": int(fold),
        "protocol": "carots_ofp_contrastive_train_score_eval",
        "threshold": float(threshold),
        "val_metrics": val_metrics,
        "test_rows": int(test_rows),
        "seconds": time.time() - started,
        "metrics": metrics,
        "history": history,
        "alarm_cfg": asdict(alarm_cfg),
        "carots_cfg": {
            "encoder": args.encoder,
            "win_size": int(args.win_size),
            "hidden_dim": int(args.hidden_dim),
            "proj_dim": int(args.proj_dim),
            "noise_level": float(args.noise_level),
            "bias_scale": float(args.bias_scale),
            "bias_percent": float(args.bias_percent),
            "sim_threshold": float(args.sim_threshold),
            "temperature": float(args.temperature),
        },
    }
    (run_dir / "fold_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[carots-done] fold={fold} threshold={threshold:.4f} "
        f"f1={metrics.get('f1_score', 0.0):.5f} final={metrics.get('final_score', 0.0):.5f}"
    )
    del model, loader, dataset, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CAROTS-style contrastive OFP scorer and evaluate it.")
    parser.add_argument("--model_name", default="carots_ofp")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("OFP_external_adapter_results/carots_ofp"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--win_size", type=int, default=10)
    parser.add_argument("--samples_per_file", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--bias_scale", type=float, default=0.5)
    parser.add_argument("--bias_percent", type=float, default=0.5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--score_stride", type=int, default=1)
    parser.add_argument("--score_batch_size", type=int, default=512)
    parser.add_argument("--centroid_windows_per_file", type=int, default=8)
    parser.add_argument("--max_centroid_windows", type=int, default=50000)
    parser.add_argument("--train_on_normal_modules_only", action="store_true", default=True)
    parser.add_argument("--train_on_all_modules", dest="train_on_normal_modules_only", action="store_false")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold_grid", default=AlarmCfg.threshold_grid)
    parser.add_argument("--threshold_metric", default="f1_score")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--smoothing", choices=["none", "ema", "mean", "rolling"], default="ema")
    parser.add_argument("--smooth_window", type=int, default=1)
    parser.add_argument("--consecutive_k", type=int, default=1)
    parser.add_argument("--no_first_alarm_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_files", type=int, default=0)
    parser.add_argument("--max_test_files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_df = read_index(args.index_path)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = str(args.device)
    alarm_cfg = AlarmCfg(
        threshold_grid=args.threshold_grid,
        threshold_metric=args.threshold_metric,
        val_fraction=args.val_fraction,
        smoothing=args.smoothing,
        smooth_window=args.smooth_window,
        consecutive_k=args.consecutive_k,
        first_alarm_only=not args.no_first_alarm_only,
        seed=args.seed,
    )
    results = []
    for fold in args.folds:
        result = run_fold(args, index_df, int(fold), alarm_cfg)
        results.append(result)
        aggregate_results([result], args.out_root)
    aggregate_results(results, args.out_root)


if __name__ == "__main__":
    main()
