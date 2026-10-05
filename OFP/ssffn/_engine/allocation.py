from __future__ import annotations
import numpy as np

def allocate_stratified_negative_counts(neg_counts: np.ndarray, target_total: int) -> np.ndarray:
    neg_counts = np.asarray(neg_counts, dtype=np.int64)
    target_total = int(max(0, min(int(target_total), int(neg_counts.sum()))))
    alloc = np.zeros_like(neg_counts, dtype=np.int64)
    if target_total <= 0 or int(neg_counts.sum()) <= 0:
        return alloc
    positive_modules = neg_counts > 0
    if target_total >= int(positive_modules.sum()):
        alloc[positive_modules] = 1
    else:
        order = np.argsort(-neg_counts, kind='mergesort')
        chosen = order[:target_total]
        alloc[chosen] = 1
        return alloc
    remaining = target_total - int(alloc.sum())
    capacity = neg_counts - alloc
    if remaining <= 0 or int(capacity.sum()) <= 0:
        return alloc
    raw = capacity.astype(np.float64) * (float(remaining) / float(capacity.sum()))
    extra = np.floor(raw).astype(np.int64)
    extra = np.minimum(extra, capacity)
    alloc += extra
    remaining = target_total - int(alloc.sum())
    if remaining > 0:
        fractions = raw - np.floor(raw)
        order = np.lexsort((-capacity, -fractions))
        for idx in order:
            if remaining <= 0:
                break
            if alloc[idx] < neg_counts[idx]:
                alloc[idx] += 1
                remaining -= 1
    return alloc
