from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_index(index_path: Path) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    required = {'file_name', 'Label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{index_path} missing columns: {sorted(missing)}')
    df['Label'] = pd.to_numeric(df['Label'], errors='raise').astype(int)
    return df
