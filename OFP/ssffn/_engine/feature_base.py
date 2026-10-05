from __future__ import annotations
import numpy as np
import pandas as pd
OFP_NAME_MAP = {'timestamp': 'Ts', 'temperature': 'Temp', 'current': 'Curr', 'currentTXPower': 'TxP0', 'currentRXPower': 'RxP0', 'currentMultiRXPower1': 'RxP1', 'currentMultiRXPower2': 'RxP2', 'currentMultiRXPower3': 'RxP3', 'currentMultiRXPower4': 'RxP4', 'currentMultiTXPower1': 'TxP1', 'currentMultiTXPower2': 'TxP2', 'currentMultiTXPower3': 'TxP3', 'currentMultiTXPower4': 'TxP4', 'anomaly': 'Ano'}
OFP_BASE_COLUMNS = ['Ts', 'Temp', 'Curr', 'TxP0', 'RxP0', 'RxP1', 'RxP2', 'RxP3', 'RxP4', 'TxP1', 'TxP2', 'TxP3', 'TxP4']
MODEL2_FEATURES = [*OFP_BASE_COLUMNS, 'FeCoCurrTemp', 'FeCoCurrTxP0', 'FeCoCurrRxP0', 'FeCoTxP0RxP0', 'FeTxP0-Max', 'FeTxP0-Min', 'FeRxP0-Max', 'FeRxP0-Min', 'TsDelta']
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

def make_model2_features(raw_df: pd.DataFrame, with_label: bool=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_df.rename(columns=OFP_NAME_MAP).copy()
    df = df[~df['Temp'].isna()].copy()
    for col in OFP_BASE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if with_label and 'Ano' in df.columns:
        df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce').fillna(0.0)
    feature = df[OFP_BASE_COLUMNS].copy()
    feature['FeCoCurrTemp'] = rolling_corr(df['Curr'], df['Temp'], fill=99.0)
    feature['FeCoCurrTxP0'] = rolling_corr(df['Curr'], df['TxP0'], fill=-99.0)
    feature['FeCoCurrRxP0'] = rolling_corr(df['Curr'], df['RxP0'], fill=-99.0)
    feature['FeCoTxP0RxP0'] = rolling_corr(df['TxP0'], df['RxP0'], fill=99.0)
    feature['FeTxP0-Max'] = df['TxP0'] - df[['TxP1', 'TxP2', 'TxP3', 'TxP4']].max(axis=1)
    feature['FeTxP0-Min'] = df['TxP0'] - df[['TxP1', 'TxP2', 'TxP3', 'TxP4']].min(axis=1)
    feature['FeRxP0-Max'] = df['RxP0'] - df[['RxP1', 'RxP2', 'RxP3', 'RxP4']].max(axis=1)
    feature['FeRxP0-Min'] = df['RxP0'] - df[['RxP1', 'RxP2', 'RxP3', 'RxP4']].min(axis=1)
    if with_label and 'Ano' in df.columns:
        feature['Ano'] = df['Ano']
    cut_mask = (df['Temp'] == TEMP_OUTLIER) | (df['Temp'] == NA_DEFAULT) | df['Temp'].isna()
    normal = feature.loc[~cut_mask].copy()
    extra = feature.loc[cut_mask].copy()
    normal['TsDelta'] = normal['Ts'].diff().fillna(NA_DEFAULT)
    extra['TsDelta'] = extra['Ts'].diff().fillna(NA_DEFAULT)
    normal_ts = normal['Ts'].to_numpy(dtype=np.float64, copy=True)
    extra_ts = extra['Ts'].to_numpy(dtype=np.float64, copy=True)
    normal = normal.astype(np.float32)
    extra = extra.astype(np.float32)
    normal['Ts'] = normal_ts
    extra['Ts'] = extra_ts
    return (normal, extra)
