# Frozen B0/B1/B2 feature schema

Schema version: `strict-causal-v1`.

Every row is engineered independently inside one SN file after timestamp order
has been validated.  No label, anomaly value, SN identifier, row index, future
observation or cross-SN state is an input feature.

## Raw: 12 dimensions

The original columns are kept unchanged and in this order:

```text
temperature, current, currentTXPower, currentRXPower,
currentMultiRXPower1..4, currentMultiTXPower1..4
```

Numeric coercion is the same as B0.  Raw sentinel values are not rewritten, so
B1/B2 retain the exact B0 inputs as their first 12 columns.

## Statistical: 76 dimensions

### Expanding univariate statistics: 72

For each of the 12 raw channels and row `i`, let `H_i` contain valid values of
that channel from the start of the same SN through row `i`, inclusive:

- `Min = min(H_i)`
- `Max = max(H_i)`
- `Diff = Max - Min` (Model2 range semantics, **not** first difference)
- `Std = sample standard deviation(H_i)`, `ddof=1`
- `Skew = pandas unbiased skewness(H_i)`
- `Kurt = pandas unbiased Fisher excess kurtosis(H_i)`

Names follow Model2, for example `FeTempMin`, `FeCurrStd` and `FeTxP4Kurt`.
Moment statistics are zero when at least one valid historical value exists but
the moment is undefined because history is too short or constant.  When there
is no valid history, the result is `NaN` and XGBoost handles it as missing.

### Trailing correlations: 4

Pearson correlation over the current row and previous four rows
(`window=5`, `min_periods=5`).  All five channel pairs must be valid; missing
rows are not skipped to reach farther back:

- `FeCoCurrTemp`: current versus temperature
- `FeCoCurrTxP0`: current versus central Tx power
- `FeCoCurrRxP0`: current versus central Rx power
- `FeCoTxP0RxP0`: central Tx versus central Rx power

An incomplete or zero-variance window produces `NaN`, not Model2's artificial
`+/-99` values.

## Expert: 42 dimensions

All expert-derived inputs are exposed as one `expert` group even though their
formulas have two sources.

### Continuous physical lane relations: 4

- `FeTxP0-Max = TxP0 - max(TxP1..TxP4)`
- `FeTxP0-Min = TxP0 - min(TxP1..TxP4)`
- `FeRxP0-Max = RxP0 - max(RxP1..RxP4)`
- `FeRxP0-Min = RxP0 - min(RxP1..RxP4)`

All five involved values must be valid at the current row; otherwise the
relation is missing.

### Model2 deterministic rule indicators: 38

Each indicator is `1` only when its strict threshold is satisfied; equality is
`0`, and missing inputs never trigger.

| Family | Rules |
|---|---|
| Temperature/current | `TempMin<0`, `CurrMin<5000`, `TempDiff>100`, `CurrDiff>6000`, `TempStd>10`, `CurrStd>1500`, `TempKurt>500` |
| Central Tx/Rx | `TxP0Min<0`, `RxP0Min<0`, `TxP0Diff>1000`, `RxP0Diff>1000`, `TxP0Std>500`, `RxP0Std>500`, both `TxP0Kurt` and `RxP0Kurt >500` |
| Each Rx lane 1..4 | `Min<0`, `Diff>1000`, `Std>200` |
| Each Tx lane 1..4 | `Min<0`, `Diff>1000`, `Std>100` |

The rule names remain the Model2 `Ru*` names.  Disabled skew rules are not
restored because the teacher implementation contains duplicated-lane errors.

An audit of all 936 faulty SNs with pre-first-fault history (464,254 rows)
found that all 38 rule indicators are zero before the first fault.  They are
kept for traceability and to test whether an explicit rule prior helps XGBoost,
but they are expected to be degenerate under the strict protocol.  The output
feature-importance tables make this directly auditable after B2 runs.

## Invalid-value policy for derived features

- Raw columns preserve their numeric values.
- In the derived-feature view only, `-999` is missing for every channel.
- In the derived-feature view only, temperature `-255` is also missing.
- Non-finite values are missing.
- Missing values do not activate rules and are never replaced by an extreme
  sentinel before Min/Max/Diff/Std/Skew/Kurt are calculated.

This prevents a single missing marker from permanently corrupting expanding
Min/Diff and avoids false expert-rule activation.
