from __future__ import annotations
import numpy as np
import pandas as pd
OFP_NAME_MAP = {'timestamp': 'Ts', 'temperature': 'Temp', 'current': 'Curr', 'currentTXPower': 'TxP0', 'currentRXPower': 'RxP0', 'currentMultiRXPower1': 'RxP1', 'currentMultiRXPower2': 'RxP2', 'currentMultiRXPower3': 'RxP3', 'currentMultiRXPower4': 'RxP4', 'currentMultiTXPower1': 'TxP1', 'currentMultiTXPower2': 'TxP2', 'currentMultiTXPower3': 'TxP3', 'currentMultiTXPower4': 'TxP4', 'anomaly': 'Ano'}
OFP_BASE_COLUMNS = ['Ts', 'Temp', 'Curr', 'TxP0', 'RxP0', 'RxP1', 'RxP2', 'RxP3', 'RxP4', 'TxP1', 'TxP2', 'TxP3', 'TxP4']
TEMP_OUTLIER = -255.0
NA_DEFAULT = -999.0
HORIZON_HOURS = 1
HORIZON_SECONDS = HORIZON_HOURS * 3600

def first_anomaly_ts(df: pd.DataFrame) -> float | None:
    if 'anomaly' not in df.columns:
        return None
    anomaly = pd.to_numeric(df['anomaly'], errors='coerce').fillna(0.0).to_numpy()
    if not np.any(anomaly > 0):
        return None
    timestamps = pd.to_numeric(df['timestamp'], errors='coerce').to_numpy(dtype=float)
    idx = int(np.argmax(anomaly > 0))
    return float(timestamps[idx])

def first_anomaly_ahead_label(df: pd.DataFrame, horizon_seconds: float=HORIZON_SECONDS) -> np.ndarray:
    timestamps = pd.to_numeric(df['timestamp'], errors='coerce').to_numpy(dtype=float)
    first_ts = first_anomaly_ts(df)
    if first_ts is None:
        return np.zeros(len(df), dtype=np.int8)
    label = (timestamps >= first_ts - float(horizon_seconds)) & (timestamps < first_ts)
    return label.astype(np.int8)

def first_event_valid_mask(df: pd.DataFrame) -> np.ndarray:
    timestamps = pd.to_numeric(df['timestamp'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(timestamps)
    first_ts = first_anomaly_ts(df)
    if first_ts is None:
        return valid.astype(bool)
    return (valid & (timestamps < first_ts)).astype(bool)

def rolling_corr(a: pd.Series, b: pd.Series, win_size: int=5, fill: float=99.0) -> np.ndarray:
    out = a.rolling(win_size, min_periods=win_size).corr(b).to_numpy(dtype=np.float32)
    out[:win_size - 1] = fill
    return np.nan_to_num(out, nan=NA_DEFAULT, posinf=NA_DEFAULT, neginf=NA_DEFAULT).astype(np.float32)
