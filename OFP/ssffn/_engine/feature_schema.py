from __future__ import annotations
from collections import OrderedDict
import numpy as np
import pandas as pd

NA_DEFAULT = -999.0
RAW_DDM_FEATURES = ['Temp','Curr','TxP0','RxP0','RxP1','RxP2','RxP3','RxP4','TxP1','TxP2','TxP3','TxP4']
PREFIX_CHANNELS = RAW_DDM_FEATURES
PREFIX_OPERATIONS = ['Min','Diff','Max','Skew','Kurt','Std']
CORRELATION_FEATURES = ['FeCoCurrTemp','FeCoCurrTxP0','FeCoCurrRxP0','FeCoTxP0RxP0']
LANE_CONSISTENCY_FEATURES = ['FeTxP0-Max','FeTxP0-Min','FeRxP0-Max','FeRxP0-Min']
STATISTIC_NAMES = [*CORRELATION_FEATURES,*LANE_CONSISTENCY_FEATURES,
                   *[f'Fe{s}{op}' for s in RAW_DDM_FEATURES for op in PREFIX_OPERATIONS]]

def _prefix_feature(channel,operation): return f'Fe{channel}{operation}'
def deduplicate(values): return list(dict.fromkeys(values))
def _numeric(frame,name,default=NA_DEFAULT):
    return pd.to_numeric(frame[name],errors='coerce').fillna(default).astype(float)
def _rolling_corr(left,right,window,fill):
    corr=left.rolling(window=window,min_periods=window).corr(right)
    corr.iloc[:max(window-1,0)]=fill
    return corr.replace([np.inf,-np.inf],np.nan).fillna(NA_DEFAULT).astype(float)
def add_statistical_features(frame: pd.DataFrame, correlation_window: int=5, na_default: float=NA_DEFAULT) -> pd.DataFrame:
    out = frame.copy()
    for name in ['Ts', *RAW_DDM_FEATURES]:
        if name not in out.columns:
            out[name] = float(na_default)
        out[name] = pd.to_numeric(out[name], errors='coerce').fillna(float(na_default))
    if 'TsDelta' not in out.columns:
        out['TsDelta'] = out['Ts'].diff().fillna(0.0)
    engineered: dict[str, pd.Series] = {}
    if 'FeCoCurrTemp' not in out.columns:
        engineered['FeCoCurrTemp'] = _rolling_corr(out['Curr'], out['Temp'], correlation_window, 99.0)
    if 'FeCoCurrTxP0' not in out.columns:
        engineered['FeCoCurrTxP0'] = _rolling_corr(out['Curr'], out['TxP0'], correlation_window, -99.0)
    if 'FeCoCurrRxP0' not in out.columns:
        engineered['FeCoCurrRxP0'] = _rolling_corr(out['Curr'], out['RxP0'], correlation_window, -99.0)
    if 'FeCoTxP0RxP0' not in out.columns:
        engineered['FeCoTxP0RxP0'] = _rolling_corr(out['TxP0'], out['RxP0'], correlation_window, 99.0)
    tx_lanes = ['TxP1', 'TxP2', 'TxP3', 'TxP4']
    rx_lanes = ['RxP1', 'RxP2', 'RxP3', 'RxP4']
    if 'FeTxP0-Max' not in out.columns:
        engineered['FeTxP0-Max'] = out['TxP0'] - out[tx_lanes].max(axis=1)
    if 'FeTxP0-Min' not in out.columns:
        engineered['FeTxP0-Min'] = out['TxP0'] - out[tx_lanes].min(axis=1)
    if 'FeRxP0-Max' not in out.columns:
        engineered['FeRxP0-Max'] = out['RxP0'] - out[rx_lanes].max(axis=1)
    if 'FeRxP0-Min' not in out.columns:
        engineered['FeRxP0-Min'] = out['RxP0'] - out[rx_lanes].min(axis=1)
    for channel in PREFIX_CHANNELS:
        series = _numeric(out, channel, float(na_default))
        expanding = series.expanding(min_periods=1)
        minimum = expanding.min()
        maximum = expanding.max()
        std = expanding.std().fillna(0.0)
        skew = expanding.skew().replace([np.inf, -np.inf], np.nan).fillna(float(na_default))
        kurt = expanding.kurt().replace([np.inf, -np.inf], np.nan).fillna(float(na_default))
        engineered[_prefix_feature(channel, 'Min')] = minimum
        engineered[_prefix_feature(channel, 'Diff')] = maximum - minimum
        engineered[_prefix_feature(channel, 'Max')] = maximum
        engineered[_prefix_feature(channel, 'Skew')] = skew
        engineered[_prefix_feature(channel, 'Kurt')] = kurt
        engineered[_prefix_feature(channel, 'Std')] = std
    if engineered:
        out = pd.concat([out, pd.DataFrame(engineered, index=out.index)], axis=1)
    return out

def select_feature_names(feature_names,include_groups=None,exclude_groups=None):
    return [name for name in feature_names if name in STATISTIC_NAMES]

def feature_group_manifest(feature_names):
    groups=OrderedDict()
    for name in feature_names:
        group=('sensors' if name in RAW_DDM_FEATURES else
               'channel_relations' if name in CORRELATION_FEATURES+LANE_CONSISTENCY_FEATURES else
               'statistics' if name in STATISTIC_NAMES else 'preprocessing')
        groups.setdefault(group,[]).append(name)
    return dict(groups=groups,counts={k:len(v) for k,v in groups.items()},total=len(feature_names))
