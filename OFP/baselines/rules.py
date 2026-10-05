"""DDM rule baseline, independent of SSFFN input features and sampling."""
import pandas as pd

def predict_rules(statistics):
    """Apply the 38 diagnostic conditions to named continuous statistics."""
    out=statistics
    rule_columns: dict[str, pd.Series] = {'RuTempMin': out['FeTempMin'] < 0.0, 'RuCurrMin': out['FeCurrMin'] < 5000.0, 'RuTempDiff': out['FeTempDiff'] > 100.0, 'RuCurrDiff': out['FeCurrDiff'] > 6000.0, 'RuTempStd': out['FeTempStd'] > 10.0, 'RuCurrStd': out['FeCurrStd'] > 1500.0, 'RuTempKurt': out['FeTempKurt'] > 500.0, 'RuTxP0Min': out['FeTxP0Min'] < 0.0, 'RuRxP0Min': out['FeRxP0Min'] < 0.0, 'RuTxP0Diff': out['FeTxP0Diff'] > 1000.0, 'RuRxP0Diff': out['FeRxP0Diff'] > 1000.0, 'RuTxP0Std': out['FeTxP0Std'] > 500.0, 'RuRxP0Std': out['FeRxP0Std'] > 500.0, 'RuTxRxP0Kurt': (out['FeTxP0Kurt'] > 500.0) & (out['FeRxP0Kurt'] > 500.0)}
    for channel in ['RxP1', 'RxP2', 'RxP3', 'RxP4']:
        rule_columns[f'Ru{channel}Min'] = out[f'Fe{channel}Min'] < 0.0
        rule_columns[f'Ru{channel}Diff'] = out[f'Fe{channel}Diff'] > 1000.0
        rule_columns[f'Ru{channel}Std'] = out[f'Fe{channel}Std'] > 200.0
    for channel in ['TxP1', 'TxP2', 'TxP3', 'TxP4']:
        rule_columns[f'Ru{channel}Min'] = out[f'Fe{channel}Min'] < 0.0
        rule_columns[f'Ru{channel}Diff'] = out[f'Fe{channel}Diff'] > 1000.0
        rule_columns[f'Ru{channel}Std'] = out[f'Fe{channel}Std'] > 100.0
    return pd.DataFrame(rule_columns,index=out.index).fillna(False).any(axis=1).astype(int)
