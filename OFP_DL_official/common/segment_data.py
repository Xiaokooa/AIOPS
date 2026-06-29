from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from OFP_DL_official.ofp_formal_protocol.protocol import ahead_labels_multi_first_event, first_anomaly_timestamp

from OFP_DL_official.common.formal_data import (
    FormalModuleCache,
    HORIZON_HOURS,
    PRIMARY_HORIZON_INDEX,
    primary_labels,
)
from OFP_DL_official.common.trainer import cuda_autocast

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is optional.
    _tqdm = None


def allocate_stratified_negative_counts(neg_counts: np.ndarray, target_neg: int) -> np.ndarray:
    neg_counts = np.asarray(neg_counts, dtype=np.int64)
    target_neg = int(min(max(target_neg, 0), int(neg_counts.sum())))
    alloc = np.zeros_like(neg_counts, dtype=np.int64)
    if target_neg <= 0 or int(neg_counts.sum()) <= 0:
        return alloc

    positive_modules = np.flatnonzero(neg_counts > 0)
    if target_neg >= len(positive_modules):
        alloc[positive_modules] = 1
        remaining = target_neg - len(positive_modules)
    else:
        order = positive_modules[np.argsort(-neg_counts[positive_modules], kind="stable")]
        alloc[order[:target_neg]] = 1
        return alloc

    capacity = neg_counts - alloc
    if remaining <= 0 or int(capacity.sum()) <= 0:
        return alloc

    raw = capacity.astype(np.float64) / float(capacity.sum()) * float(remaining)
    extra = np.floor(raw).astype(np.int64)
    extra = np.minimum(extra, capacity)
    alloc += extra
    left = target_neg - int(alloc.sum())
    if left > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac, kind="stable")
        for idx in order:
            if left <= 0:
                break
            room = int(neg_counts[idx] - alloc[idx])
            if room <= 0:
                continue
            take = min(room, left)
            alloc[idx] += take
            left -= take
    return alloc


def ensure_multi_horizon_labels(
    timestamps: np.ndarray,
    anomaly: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 2 and arr.shape[-1] == len(HORIZON_HOURS):
        return arr.astype(np.float32, copy=False)
    return ahead_labels_multi_first_event(
        np.asarray(timestamps, dtype=float),
        np.asarray(anomaly, dtype=np.int8),
        HORIZON_HOURS,
    ).astype(np.float32, copy=False)


class SegmentForecastDataset(IterableDataset):
    """OFP segment forecasting dataset.

    Each item is a batch of continuous historical segments and a future label
    block. With input_len=288 and pred_len=12 this is the 24h -> 1h protocol.
    """

    def __init__(
        self,
        file_names: list[str],
        cache: FormalModuleCache,
        input_len: int,
        pred_len: int,
        train_stride: int,
        mean: np.ndarray,
        std: np.ndarray,
        batch_segments: int = 512,
        negative_ratio: float = 10.0,
        seed: int = 42,
        shuffle_segments: bool = True,
    ) -> None:
        self.file_names = list(file_names)
        self.cache = cache
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.train_stride = int(train_stride)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.batch_segments = int(batch_segments)
        self.negative_ratio = float(negative_ratio)
        self.seed = int(seed)
        self.shuffle_segments = bool(shuffle_segments)
        self._iteration = 0
        if self.input_len <= 0 or self.pred_len <= 0:
            raise ValueError("input_len and pred_len must be positive")
        if self.train_stride <= 0:
            raise ValueError("train_stride must be positive")
        if self.train_stride > self.pred_len:
            raise ValueError("train_stride > pred_len can leave target timestamps uncovered")
        if self.batch_segments <= 0:
            raise ValueError("batch_segments must be positive")

        self.source_segments = 0
        self.source_pos_segments = 0
        self.source_neg_segments = 0
        self.source_target_rows = 0
        self.source_pos_rows = 0
        self.source_neg_rows = 0
        self.source_pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.source_neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        pos_starts: list[np.ndarray] = []
        neg_starts: list[np.ndarray] = []
        neg_counts: list[int] = []

        for idx, name in enumerate(self.file_names, 1):
            timestamps, _values, anomaly, labels, valid_mask = self.cache.get(name)
            labels = ensure_multi_horizon_labels(timestamps, anomaly, labels)
            labels_primary = primary_labels(labels)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            raw_starts = np.arange(0, len(labels), self.train_stride, dtype=np.int64)
            kept_starts: list[int] = []
            kept_is_pos: list[bool] = []
            for start in raw_starts:
                end = min(int(start) + self.pred_len, len(labels))
                target_mask = valid_mask[int(start) : end]
                valid = int(target_mask.sum())
                if valid <= 0:
                    continue
                target = labels[int(start) : end][target_mask]
                target_primary = labels_primary[int(start) : end][target_mask]
                pos = int(target_primary.sum())
                pos_h = target.astype(np.int64).sum(axis=0)
                self.source_target_rows += valid
                self.source_pos_rows += pos
                self.source_neg_rows += valid - pos
                self.source_pos_rows_by_horizon += pos_h
                self.source_neg_rows_by_horizon += valid - pos_h
                kept_starts.append(int(start))
                kept_is_pos.append(pos > 0)
            starts = np.asarray(kept_starts, dtype=np.int64)
            is_pos = np.asarray(kept_is_pos, dtype=bool)
            pos = starts[is_pos]
            neg = starts[~is_pos]
            pos_starts.append(pos.astype(np.int64))
            neg_starts.append(neg.astype(np.int64))
            neg_counts.append(int(len(neg)))
            self.source_segments += int(len(starts))
            self.source_pos_segments += int(len(pos))
            self.source_neg_segments += int(len(neg))
        target_neg = int(round(float(self.source_pos_segments) * self.negative_ratio))
        alloc = allocate_stratified_negative_counts(np.asarray(neg_counts, dtype=np.int64), target_neg)
        self.selected_starts: list[np.ndarray] = []
        self.pos_segments = int(self.source_pos_segments)
        self.neg_segments = int(alloc.sum())
        self.total_segments = int(self.pos_segments + self.neg_segments)
        self.pos_rows = 0
        self.neg_rows = 0
        self.pos_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.neg_rows_by_horizon = np.zeros(len(HORIZON_HOURS), dtype=np.int64)
        self.total_target_rows = 0
        self.total_batches = 0

        for name, pos, neg, n_neg in zip(self.file_names, pos_starts, neg_starts, alloc):
            if int(n_neg) >= len(neg):
                chosen_neg = neg
            elif int(n_neg) > 0:
                stable = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
                rng = np.random.default_rng(self.seed + int(stable))
                chosen_neg = np.sort(rng.choice(neg, size=int(n_neg), replace=False)).astype(np.int64)
            else:
                chosen_neg = np.empty(0, dtype=np.int64)
            selected = np.sort(np.concatenate([pos, chosen_neg])).astype(np.int64)
            self.selected_starts.append(selected)

        for name, starts in zip(self.file_names, self.selected_starts):
            if len(starts) <= 0:
                continue
            timestamps, _values, anomaly, labels, valid_mask = self.cache.get(name)
            labels = ensure_multi_horizon_labels(timestamps, anomaly, labels)
            labels_primary = primary_labels(labels)
            valid_mask = np.asarray(valid_mask, dtype=bool)
            for start in starts:
                end = min(int(start) + self.pred_len, len(labels))
                target_mask = valid_mask[int(start) : end]
                valid = int(target_mask.sum())
                if valid <= 0:
                    continue
                target = labels[int(start) : end][target_mask]
                target_primary = labels_primary[int(start) : end][target_mask]
                pos = int(target_primary.sum())
                pos_h = target.astype(np.int64).sum(axis=0)
                self.total_target_rows += valid
                self.pos_rows += pos
                self.neg_rows += valid - pos
                self.pos_rows_by_horizon += pos_h
                self.neg_rows_by_horizon += valid - pos_h
        self.total_batches = int(math.ceil(self.total_segments / self.batch_segments)) if self.total_segments else 0

        print(
            f"[segment dataset] indexed done files={len(self.file_names)} "
            f"source_segments={self.source_segments} source_pos_segments={self.source_pos_segments} "
            f"source_neg_segments={self.source_neg_segments} sampled_segments={self.total_segments} "
            f"sampled_pos_segments={self.pos_segments} sampled_neg_segments={self.neg_segments} "
            f"target_rows={self.total_target_rows} pos_rows={self.pos_rows} neg_rows={self.neg_rows} "
            f"negative_ratio={self.negative_ratio} batches={self.total_batches} "
            f"batch_segments={self.batch_segments}",
            flush=True,
        )
        self.cache.cache.clear()

    def __len__(self) -> int:
        return int(self.total_batches)

    def _worker_items(self, epoch_seed: int) -> list[tuple[str, np.ndarray]]:
        worker = get_worker_info()
        pairs = [(name, starts.copy()) for name, starts in zip(self.file_names, self.selected_starts)]
        if self.shuffle_segments:
            rng = np.random.default_rng(int(epoch_seed))
            rng.shuffle(pairs)
            pairs = [
                (name, rng.permutation(starts).astype(np.int64) if len(starts) > 1 else starts)
                for name, starts in pairs
            ]
        if worker is None:
            return pairs
        return pairs[worker.id :: worker.num_workers]

    def __iter__(self):
        buf_xs: list[torch.Tensor] = []
        buf_masks: list[torch.Tensor] = []
        buf_ys: list[torch.Tensor] = []
        buf_ymasks: list[torch.Tensor] = []
        buf_n = 0

        def append_and_yield(xs: torch.Tensor, masks: torch.Tensor, ys: torch.Tensor, ymask: torch.Tensor):
            nonlocal buf_xs, buf_masks, buf_ys, buf_ymasks, buf_n
            offset = 0
            while offset < int(xs.size(0)):
                take = min(self.batch_segments - buf_n, int(xs.size(0)) - offset)
                buf_xs.append(xs[offset : offset + take])
                buf_masks.append(masks[offset : offset + take])
                buf_ys.append(ys[offset : offset + take])
                buf_ymasks.append(ymask[offset : offset + take])
                buf_n += int(take)
                offset += int(take)
                if buf_n >= self.batch_segments:
                    yield (
                        torch.cat(buf_xs, dim=0).contiguous(),
                        torch.cat(buf_masks, dim=0).contiguous(),
                        torch.cat(buf_ys, dim=0).contiguous(),
                        torch.cat(buf_ymasks, dim=0).contiguous(),
                    )
                    buf_xs, buf_masks, buf_ys, buf_ymasks = [], [], [], []
                    buf_n = 0

        epoch_seed = self.seed + int(self._iteration) * 1000003
        self._iteration += 1
        for name, starts in self._worker_items(epoch_seed):
            if len(starts) <= 0:
                continue
            timestamps, values, anomaly, labels, valid_mask = self.cache.get(name)
            labels = ensure_multi_horizon_labels(timestamps, anomaly, labels).astype(np.float32)
            valid_mask = np.asarray(valid_mask, dtype=np.float32)
            valid = np.isfinite(values).astype(np.float32)
            raw = np.where(valid > 0, values, 0.0).astype(np.float32)
            x = ((raw - self.mean) / self.std) * valid

            pad_x = np.zeros((self.input_len, values.shape[1]), dtype=np.float32)
            pad_m = np.zeros_like(pad_x)
            x_pad = torch.from_numpy(np.concatenate([pad_x, x.astype(np.float32)], axis=0))
            m_pad = torch.from_numpy(np.concatenate([pad_m, valid], axis=0))
            x_windows = x_pad.unfold(0, self.input_len, 1).permute(0, 2, 1)
            m_windows = m_pad.unfold(0, self.input_len, 1).permute(0, 2, 1)

            y_tail = np.zeros((self.pred_len - 1, labels.shape[1]), dtype=np.float32)
            y_pad = torch.from_numpy(np.concatenate([labels, y_tail], axis=0))
            ym_pad = torch.from_numpy(
                np.concatenate([valid_mask, np.zeros(self.pred_len - 1, dtype=np.float32)])
            )
            y_windows = y_pad.unfold(0, self.pred_len, 1).permute(0, 2, 1)
            ym_windows = ym_pad.unfold(0, self.pred_len, 1)
            ym_windows = ym_windows.unsqueeze(-1).expand(-1, -1, len(HORIZON_HOURS))

            for start in range(0, len(starts), self.batch_segments):
                batch_starts = starts[start : start + self.batch_segments]
                index = torch.from_numpy(batch_starts)
                xs = x_windows.index_select(0, index).contiguous()
                masks = m_windows.index_select(0, index).contiguous()
                ys = y_windows.index_select(0, index).contiguous()
                ymask = ym_windows.index_select(0, index).contiguous()
                yield from append_and_yield(xs, masks, ys, ymask)
        if buf_n > 0:
            yield (
                torch.cat(buf_xs, dim=0).contiguous(),
                torch.cat(buf_masks, dim=0).contiguous(),
                torch.cat(buf_ys, dim=0).contiguous(),
                torch.cat(buf_ymasks, dim=0).contiguous(),
            )


def make_segment_loader(dataset: SegmentForecastDataset, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=None, num_workers=int(num_workers), pin_memory=False)


@torch.no_grad()
def score_one_module_segments(
    model: torch.nn.Module,
    values: np.ndarray,
    input_len: int,
    pred_len: int,
    test_stride: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
    use_amp: bool = False,
    aggregation: str = "max",
) -> np.ndarray:
    model.eval()
    n = int(len(values))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    valid = np.isfinite(values).astype(np.float32)
    raw = np.where(valid > 0, values, 0.0).astype(np.float32)
    x = ((raw - mean) / np.maximum(std, 1e-6)) * valid
    pad_x = np.zeros((int(input_len), values.shape[1]), dtype=np.float32)
    pad_m = np.zeros_like(pad_x)
    x_pad = torch.from_numpy(np.concatenate([pad_x, x.astype(np.float32)], axis=0))
    m_pad = torch.from_numpy(np.concatenate([pad_m, valid], axis=0))
    x_windows = x_pad.unfold(0, int(input_len), 1).permute(0, 2, 1)
    m_windows = m_pad.unfold(0, int(input_len), 1).permute(0, 2, 1)

    starts = np.arange(0, n, int(test_stride), dtype=np.int64)
    if aggregation == "mean":
        score_sum = np.zeros(n, dtype=np.float64)
        score_count = np.zeros(n, dtype=np.float64)
    elif aggregation == "max":
        score_max = np.full(n, -np.inf, dtype=np.float64)
        score_count = np.zeros(n, dtype=np.float64)
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")

    horizons = np.arange(int(pred_len), dtype=np.int64)
    amp_enabled = bool(use_amp) and str(device).startswith("cuda")
    for start in range(0, len(starts), int(batch_size)):
        batch_starts = starts[start : start + int(batch_size)]
        index = torch.from_numpy(batch_starts)
        xb = x_windows.index_select(0, index).contiguous().to(device)
        mb = m_windows.index_select(0, index).contiguous().to(device)
        with cuda_autocast(amp_enabled):
            logits = model(xb, mb)
        if isinstance(logits, tuple):
            logits = logits[0]
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        if logits.ndim == 2 and logits.shape[-1] == int(pred_len) * len(HORIZON_HOURS):
            logits = logits.reshape(logits.shape[0], int(pred_len), len(HORIZON_HOURS))
        if logits.ndim == 3:
            logits = logits[:, :, PRIMARY_HORIZON_INDEX]
        scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        positions = batch_starts[:, None] + horizons[None, :]
        valid_pos = positions < n
        flat_pos = positions[valid_pos]
        flat_scores = scores[valid_pos]
        if aggregation == "mean":
            np.add.at(score_sum, flat_pos, flat_scores)
            np.add.at(score_count, flat_pos, 1.0)
        else:
            np.maximum.at(score_max, flat_pos, flat_scores)
            np.add.at(score_count, flat_pos, 1.0)

    if aggregation == "mean":
        out = np.divide(score_sum, np.maximum(score_count, 1.0))
    else:
        out = np.where(score_count > 0, score_max, 0.0)
    return out.astype(np.float32)


def collect_segment_module_scores(
    model: torch.nn.Module,
    cache: FormalModuleCache,
    file_names: list[str],
    input_len: int,
    pred_len: int,
    test_stride: int,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
    use_amp: bool = False,
    aggregation: str = "max",
    use_tqdm: bool = True,
    force_tqdm: bool = False,
    progress_desc: str = "segment score",
) -> list[dict]:
    rows: list[dict] = []
    iterator = enumerate(file_names, 1)
    progress_bar = None
    if bool(use_tqdm) and _tqdm is not None and (bool(force_tqdm) or sys.stdout.isatty()):
        progress_bar = _tqdm(
            iterator,
            total=len(file_names),
            desc=progress_desc,
            unit="module",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
            file=sys.stdout,
            ascii=True,
        )
        iterator = progress_bar
    for idx, name in iterator:
        timestamps, values, anomaly, _labels, valid_mask = cache.get(name)
        scores = score_one_module_segments(
            model,
            values,
            input_len,
            pred_len,
            test_stride,
            mean,
            std,
            batch_size,
            device,
            use_amp=use_amp,
            aggregation=aggregation,
        )
        true_ts = first_anomaly_timestamp(timestamps.astype(float), anomaly)
        rows.append(
            {
                "file_name": name,
                "timestamps": timestamps,
                "scores": scores,
                "true_label": int(true_ts is not None),
                "true_ts": true_ts,
                "valid_mask": valid_mask,
            }
        )
        if bool(use_tqdm) and progress_bar is None and idx % 200 == 0:
            print(f"[segment score] {idx}/{len(file_names)} modules", flush=True)
    if progress_bar is not None:
        progress_bar.close()
    return rows

