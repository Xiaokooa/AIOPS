from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    required = {"file_name", "folder_index", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{index_path} missing columns: {sorted(missing)}")
    df["folder_index"] = pd.to_numeric(df["folder_index"], errors="raise").astype(int)
    df["Label"] = pd.to_numeric(df["Label"], errors="raise").astype(int)
    return df


def files_for_index_fold(index_df: pd.DataFrame, fold: int) -> tuple[list[str], list[str]]:
    train_files = index_df.loc[index_df["folder_index"] != int(fold), "file_name"].tolist()
    test_files = index_df.loc[index_df["folder_index"] == int(fold), "file_name"].tolist()
    if not train_files or not test_files:
        raise ValueError(f"fold {fold} must have non-empty train/test roles")
    return train_files, test_files
