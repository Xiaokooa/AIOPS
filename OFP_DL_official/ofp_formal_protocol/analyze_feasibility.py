from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from OFP_DL_official.ofp_formal_protocol.protocol import SEC_IN_HOUR


def summarize_positive_opportunity(metadata_df: pd.DataFrame) -> dict:
    pos = metadata_df[metadata_df["anomaly_label"] > 0].copy()
    neg = metadata_df[metadata_df["anomaly_label"] == 0].copy()
    pos["pre_event_hours"] = (
        pd.to_numeric(pos["first_anomaly_ts"], errors="coerce")
        - pd.to_numeric(pos["start_ts"], errors="coerce")
    ) / SEC_IN_HOUR
    pos["pre_event_hours"] = pos["pre_event_hours"].fillna(0.0)
    buckets = {
        "le_0h": int((pos["pre_event_hours"] <= 0).sum()),
        "gt_0h": int((pos["pre_event_hours"] > 0).sum()),
        "gt_1h": int((pos["pre_event_hours"] > 1).sum()),
        "gt_24h": int((pos["pre_event_hours"] > 24).sum()),
        "gt_25h": int((pos["pre_event_hours"] > 25).sum()),
        "gt_1h_plus_24h_lookback": int((pos["pre_event_hours"] > 25).sum()),
    }
    return {
        "total_modules": int(len(metadata_df)),
        "positive_modules": int(len(pos)),
        "negative_modules": int(len(neg)),
        "label_mismatches": int(pd.to_numeric(metadata_df["label_mismatch"], errors="coerce").fillna(0).sum()),
        "pre_event_opportunity_counts": buckets,
        "pre_event_hours_mean": float(pos["pre_event_hours"].mean()) if len(pos) else 0.0,
        "pre_event_hours_median": float(pos["pre_event_hours"].median()) if len(pos) else 0.0,
        "pre_event_hours_p10": float(pos["pre_event_hours"].quantile(0.10)) if len(pos) else 0.0,
        "pre_event_hours_p90": float(pos["pre_event_hours"].quantile(0.90)) if len(pos) else 0.0,
    }


def summarize_manifest_roles(metadata_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    pos_meta = metadata_df.copy()
    pos_meta["pre_event_hours"] = (
        pd.to_numeric(pos_meta["first_anomaly_ts"], errors="coerce")
        - pd.to_numeric(pos_meta["start_ts"], errors="coerce")
    ) / SEC_IN_HOUR
    pos_meta["pre_event_hours"] = pos_meta["pre_event_hours"].fillna(0.0)
    merged = manifest_df.merge(
        pos_meta[["file_name", "start_ts", "end_ts", "pre_event_hours"]],
        on="file_name",
        how="left",
    )
    rows = []
    for (fold, role), group in merged.groupby(["fold", "role"]):
        pos = group[group["anomaly_label"] > 0]
        rows.append(
            {
                "fold": int(fold),
                "role": role,
                "modules": int(len(group)),
                "positive_modules": int(len(pos)),
                "positive_with_gt_0h": int((pos["pre_event_hours"] > 0).sum()),
                "positive_with_gt_1h": int((pos["pre_event_hours"] > 1).sum()),
                "positive_with_gt_24h": int((pos["pre_event_hours"] > 24).sum()),
                "positive_with_gt_1h_plus_24h_lookback": int((pos["pre_event_hours"] > 25).sum()),
                "positive_pre_event_hours_median": float(pos["pre_event_hours"].median()) if len(pos) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "role"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether the first-event OFP task has pre-event observation opportunity.")
    parser.add_argument("--metadata_path", type=Path, default=Path("output/ofp_formal_protocol/splits/module_metadata.csv"))
    parser.add_argument("--manifest_path", type=Path, default=Path("output/ofp_formal_protocol/splits/formal_manifest.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("output/ofp_formal_protocol/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_df = pd.read_csv(args.metadata_path)
    manifest_df = pd.read_csv(args.manifest_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_positive_opportunity(metadata_df)
    role_df = summarize_manifest_roles(metadata_df, manifest_df)
    (args.out_dir / "protocol_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    role_df.to_csv(args.out_dir / "protocol_feasibility_by_role.csv", index=False)
    print(json.dumps(summary, indent=2))
    print(role_df.to_string(index=False))


if __name__ == "__main__":
    main()


