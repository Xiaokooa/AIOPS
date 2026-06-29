"""List the 28 evaluable failure modules in the R4 test split."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from model.Optical_prediction_model.deep_learning.train import load_r4_frames

frames = load_r4_frames()
test = frames["test"]
em = test.groupby("file_name")["event_label"].max()
event_modules = em[em == 1].index.tolist()

evaluable = []
for fn in event_modules:
    sub = test[test["file_name"] == fn]
    n_pos = int((sub["label"] == 1).sum())
    n_total = int(len(sub))
    if n_pos == 0:
        continue
    # First failure timestamp (if available).
    first_fail = sub["first_failure_ts"].iloc[0] if "first_failure_ts" in sub.columns else None
    evaluable.append({
        "file_name": fn,
        "n_positive_windows": n_pos,
        "n_total_windows": n_total,
        "first_failure_ts": int(first_fail) if first_fail and first_fail > 0 else None,
    })

print(f"Total event modules (event_label=1) in test: {len(event_modules)}")
print(f"Evaluable (>=1 positive window):             {len(evaluable)}")
print()
print(f"{'idx':>3}  {'file_name':<48} {'#pos':>5} {'#total':>7}")
print("-" * 70)
for i, r in enumerate(sorted(evaluable, key=lambda d: d['file_name']), 1):
    print(f"{i:>3}  {r['file_name']:<48} {r['n_positive_windows']:>5} {r['n_total_windows']:>7}")

out_path = PROJECT_ROOT / "output" / "Optical_prediction_model" / "experiments" / "test_event_modules.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "n_event_modules": len(event_modules),
        "n_evaluable": len(evaluable),
        "modules": evaluable,
    }, f, indent=2)
print(f"\nsaved to: {out_path}")
