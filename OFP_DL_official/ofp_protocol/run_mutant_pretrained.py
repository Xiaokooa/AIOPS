from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_protocol.evaluator import evaluate_prediction_folder
from OFP_DL_official.ofp_protocol.run_ofp_baselines import read_index


OFP_MODEL2 = Path(__file__).resolve().parents[1] / "OFP" / "model2"
sys.path.insert(0, str(OFP_MODEL2))

from MUTANT.Model import MUTANT  # noqa: E402


RAW_TO_OFP = {
    "timestamp": "Ts",
    "temperature": "Temp",
    "current": "Curr",
    "currentTXPower": "TxP0",
    "currentRXPower": "RxP0",
    "currentMultiRXPower1": "RxP1",
    "currentMultiRXPower2": "RxP2",
    "currentMultiRXPower3": "RxP3",
    "currentMultiRXPower4": "RxP4",
    "currentMultiTXPower1": "TxP1",
    "currentMultiTXPower2": "TxP2",
    "currentMultiTXPower3": "TxP3",
    "currentMultiTXPower4": "TxP4",
    "anomaly": "Ano",
}

ORIGIN_FEATURE_LIST = [
    "Ts",
    "Temp",
    "Curr",
    "TxP0",
    "RxP0",
    "RxP1",
    "RxP2",
    "RxP3",
    "RxP4",
    "TxP1",
    "TxP2",
    "TxP3",
    "TxP4",
]

TEMP_OUTLIER = -255.0
NA_DEFAULT = -999.0


@dataclass
class MutantConfig:
    input_dim: int = 13
    batch_size: int = 120
    out_dim: int = 5
    window_length: int = 1
    hidden_size: int = 18
    latent_size: int = 18


@dataclass
class FoldSummary:
    model: str
    fold: int
    weight_path: str
    train_modules: int
    test_modules: int
    train_rows_for_scaler: int
    test_rows: int
    percentile_threshold: float
    seconds: float
    metrics: dict[str, float]


def read_origin_frame(path: Path, label: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path).rename(columns=RAW_TO_OFP)
    for col in [*ORIGIN_FEATURE_LIST, "Ano"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    normal_mask = (df["Temp"] != TEMP_OUTLIER) & (df["Temp"] != NA_DEFAULT) & (~df["Temp"].isna())
    normal = df.loc[normal_mask, [*ORIGIN_FEATURE_LIST, "Ano"]].copy()
    extra = df.loc[~normal_mask, [*ORIGIN_FEATURE_LIST, "Ano"]].copy()
    if label is None:
        label = int(normal["Ano"].sum() > 0) if len(normal) else int(df["Ano"].sum() > 0)
    normal["Label"] = label
    extra["Label"] = label
    values = normal[ORIGIN_FEATURE_LIST].to_numpy(dtype=np.float32, copy=True)
    normal.loc[:, ORIGIN_FEATURE_LIST] = values
    if len(extra):
        extra_values = extra[ORIGIN_FEATURE_LIST].to_numpy(dtype=np.float32, copy=True)
        extra.loc[:, ORIGIN_FEATURE_LIST] = extra_values
    return normal, extra


def normal_row_count(data_dir: Path, file_names: list[str]) -> int:
    total = 0
    for idx, name in enumerate(file_names, 1):
        df = pd.read_csv(data_dir / name, usecols=["temperature"])
        temp = pd.to_numeric(df["temperature"], errors="coerce")
        total += int(((temp != TEMP_OUTLIER) & (temp != NA_DEFAULT) & (~temp.isna())).sum())
        if idx % 1000 == 0:
            print(f"  [count] {idx}/{len(file_names)} files, normal_rows={total}")
    return total


def fit_scaler_and_threshold(
    data_dir: Path,
    train_files: list[str],
    label_map: dict[str, int],
) -> tuple[MinMaxScaler, float, int]:
    total_rows = normal_row_count(data_dir, train_files)
    cutoff = total_rows - int(total_rows * 0.2)
    scaler = MinMaxScaler()
    seen = 0
    fit_rows = 0
    prefix_rows = 0
    prefix_normal_label_rows = 0
    fitted = False
    for idx, name in enumerate(train_files, 1):
        if seen >= cutoff:
            break
        label = int(label_map[name])
        normal, _extra = read_origin_frame(data_dir / name, label=label)
        take = min(len(normal), cutoff - seen)
        if take <= 0:
            continue
        part = normal.iloc[:take]
        prefix_rows += len(part)
        prefix_normal_label_rows += int((part["Ano"].to_numpy(dtype=np.float32) == 0).sum())
        if label == 0:
            X = part[ORIGIN_FEATURE_LIST].to_numpy(dtype=np.float32, copy=False)
            if len(X):
                scaler.partial_fit(np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=100.0))
                fit_rows += len(X)
                fitted = True
        seen += take
        if idx % 500 == 0:
            print(f"  [scaler] {idx}/{len(train_files)} files, prefix_rows={prefix_rows}, fit_rows={fit_rows}")
    if not fitted:
        raise RuntimeError("No healthy rows were available to fit the MUTANT scaler.")
    base_percentile = 100.0 * prefix_normal_label_rows / max(prefix_rows, 1)
    percentile = 100.0 - 0.9 * (100.0 - base_percentile)
    return scaler, percentile, fit_rows


def fixed_win1_adjacency(input_dim: int) -> torch.Tensor:
    off_diag = 1.0 / 14.0
    diag = 1.0 / 7.0
    arr = np.full((input_dim, input_dim), off_diag, dtype=np.float32)
    np.fill_diagonal(arr, diag)
    return torch.tensor(arr, dtype=torch.float32)


def load_mutant(weight_path: Path, cfg: MutantConfig) -> MUTANT:
    model = MUTANT(
        cfg.input_dim,
        cfg.input_dim * cfg.out_dim,
        cfg.hidden_size,
        cfg.latent_size,
        cfg.batch_size,
        cfg.window_length,
        cfg.out_dim,
    )
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def fast_scores(model: MUTANT, X: np.ndarray, cfg: MutantConfig) -> np.ndarray:
    if len(X) <= cfg.window_length:
        return np.empty((0,), dtype=np.float32)
    values = X[:-cfg.window_length]
    original_len = len(values)
    pad = (-original_len) % cfg.batch_size
    if pad:
        values = np.concatenate([values, np.zeros((pad, cfg.input_dim), dtype=np.float32)], axis=0)
    adj = fixed_win1_adjacency(cfg.input_dim)
    weight = model.GCN.gc1.weight.detach().float()
    bias = model.GCN.gc1.bias.detach().float() if model.GCN.gc1.bias is not None else None
    out: list[np.ndarray] = []
    for start in range(0, len(values), cfg.batch_size):
        batch_np = values[start : start + cfg.batch_size]
        batch = torch.tensor(batch_np, dtype=torch.float32)
        support = batch[:, :, None] * weight[0][None, None, :]
        x_g = torch.einsum("ij,bjk->bik", adj, support)
        if bias is not None:
            x_g = x_g + bias
        xt = torch.relu(x_g).permute(0, 2, 1).contiguous()
        x_w = model.att_encoder(xt)
        pred = model.predict(x_w)
        mu = pred["recon_mu"]
        flat = xt.view(xt.shape[0], -1)
        score = torch.abs(torch.sum(flat - mu, dim=1))
        out.append(score.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)[:original_len]


def write_predictions_for_fold(
    model: MUTANT,
    scaler: MinMaxScaler,
    percentile: float,
    data_dir: Path,
    test_files: list[str],
    label_map: dict[str, int],
    out_dir: Path,
    cfg: MutantConfig,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for idx, name in enumerate(test_files, 1):
        normal, extra = read_origin_frame(data_dir / name, label=int(label_map[name]))
        frames: list[pd.DataFrame] = []
        if len(normal):
            X = normal[ORIGIN_FEATURE_LIST].to_numpy(dtype=np.float32, copy=False)
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=100.0)
            X = scaler.transform(X).astype(np.float32)
            scores = fast_scores(model, X, cfg)
            thresh = np.percentile(scores, float(percentile)) if len(scores) else np.inf
            aligned = normal.iloc[cfg.window_length : cfg.window_length + len(scores)]
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_numeric(aligned["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                        "predict": (scores > thresh).astype(int),
                        "score": scores,
                    }
                )
            )
        if len(extra):
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_numeric(extra["Ts"], errors="coerce").to_numpy(dtype=np.int64),
                        "predict": (pd.to_numeric(extra["Temp"], errors="coerce").to_numpy(dtype=float) == TEMP_OUTLIER).astype(int),
                        "score": np.nan,
                    }
                )
            )
        pred = pd.concat(frames, ignore_index=True).sort_values("timestamp") if frames else pd.DataFrame(columns=["timestamp", "predict", "score"])
        pred.to_csv(out_dir / name, index=False)
        total_rows += len(pred)
        if idx % 500 == 0:
            print(f"  [predict] {idx}/{len(test_files)} files, rows={total_rows}")
    return total_rows


def evaluate_dir(pred_dir: Path, label_dir: Path, eval_dir: Path) -> dict[str, float]:
    summary_df, detail_df = evaluate_prediction_folder(pred_dir, label_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(eval_dir / "evaluate_result.csv", index=False)
    detail_df.to_csv(eval_dir / "module_decisions.csv", index=False)
    return {str(row["Item"]): float(row["Value"]) for _, row in summary_df.iterrows()}


def default_weight_for_fold(fold: int) -> Path:
    if fold == 1:
        return OFP_MODEL2 / "mutant_model-folder1-win1.pt"
    if fold == 2:
        return OFP_MODEL2 / "mutant_model-folder2-win1.pt"
    return OFP_MODEL2 / "mutant_model.pt"


def run_fold(
    fold: int,
    data_dir: Path,
    index_df: pd.DataFrame,
    out_root: Path,
    max_train_files: int | None,
    max_test_files: int | None,
) -> FoldSummary:
    started = time.time()
    train_files = index_df.loc[index_df["folder_index"] != fold, "file_name"].tolist()
    test_files = index_df.loc[index_df["folder_index"] == fold, "file_name"].tolist()
    if max_train_files is not None:
        train_files = train_files[: int(max_train_files)]
    if max_test_files is not None:
        test_files = test_files[: int(max_test_files)]
    label_map = index_df.set_index("file_name")["Label"].astype(int).to_dict()
    print(f"[mutant] fold={fold} train_modules={len(train_files)} test_modules={len(test_files)}")
    scaler, percentile, fit_rows = fit_scaler_and_threshold(data_dir, train_files, label_map)
    print(f"[mutant] fold={fold} scaler_rows={fit_rows} percentile={percentile:.4f}")
    cfg = MutantConfig()
    weight_path = default_weight_for_fold(fold)
    model = load_mutant(weight_path, cfg)
    pred_dir = out_root / "mutant_pretrained" / f"fold_{fold}" / "predictions"
    test_rows = write_predictions_for_fold(model, scaler, percentile, data_dir, test_files, label_map, pred_dir, cfg)
    eval_dir = out_root / "mutant_pretrained" / f"fold_{fold}" / "evaluation"
    metrics = evaluate_dir(pred_dir, data_dir, eval_dir)
    summary = FoldSummary(
        model="mutant_pretrained",
        fold=fold,
        weight_path=str(weight_path),
        train_modules=len(train_files),
        test_modules=len(test_files),
        train_rows_for_scaler=fit_rows,
        test_rows=test_rows,
        percentile_threshold=percentile,
        seconds=time.time() - started,
        metrics=metrics,
    )
    fold_dir = out_root / "mutant_pretrained" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "fold_summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    del model, scaler
    gc.collect()
    return summary


def aggregate(summaries: list[FoldSummary], out_root: Path) -> None:
    rows = []
    for item in summaries:
        row = {
            "model": item.model,
            "fold": item.fold,
            "weight_path": item.weight_path,
            "train_modules": item.train_modules,
            "test_modules": item.test_modules,
            "train_rows_for_scaler": item.train_rows_for_scaler,
            "test_rows": item.test_rows,
            "percentile_threshold": item.percentile_threshold,
            "seconds": item.seconds,
        }
        row.update(item.metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "fold_metrics.csv"
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        df.drop_duplicates(subset=["model", "fold"], keep="last", inplace=True)
        df.sort_values(["model", "fold"], inplace=True)
    df.to_csv(path, index=False)
    numeric = [c for c in df.columns if c not in {"model", "fold", "weight_path"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("model")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.reset_index().to_csv(out_root / "model_metrics_mean_std.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/training"))
    parser.add_argument("--index_path", type=Path, default=Path("dataset/train_test_set_index(in).csv"))
    parser.add_argument("--out_root", type=Path, default=Path("output/ofp_protocol/mutant_pretrained"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    index_df = read_index(args.index_path)
    summaries = []
    for fold in args.folds:
        summaries.append(
            run_fold(
                fold=fold,
                data_dir=args.data_dir,
                index_df=index_df,
                out_root=args.out_root,
                max_train_files=args.max_train_files,
                max_test_files=args.max_test_files,
            )
        )
        aggregate(summaries, args.out_root)
        metrics = summaries[-1].metrics
        print(
            f"[mutant done] fold={fold} final={metrics.get('final_score', 0):.6f} "
            f"f1={metrics.get('f1_score', 0):.6f} hit={metrics.get('all_hit_cnt', 0):.0f} "
            f"pred={metrics.get('all_predict_pos_cnt', 0):.0f}"
        )


if __name__ == "__main__":
    main()

