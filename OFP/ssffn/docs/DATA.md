# Input data

Use one CSV file per optical transceiver, with observations in nondecreasing
timestamp order. Timestamps are Unix seconds. Do not concatenate independent
transceivers or split observations from one transceiver across datasets.

Required training/evaluation columns:

| CSV column | Meaning |
|---|---|
| `timestamp` | Observation time in Unix seconds |
| `temperature` | Temperature |
| `current` | Bias current |
| `currentTXPower` | Aggregate transmitted optical power |
| `currentRXPower` | Aggregate received optical power |
| `currentMultiRXPower1` … `currentMultiRXPower4` | Four receive-lane powers |
| `currentMultiTXPower1` … `currentMultiTXPower4` | Four transmit-lane powers |
| `anomaly` | Positive event/reference label, excluded from all predictor inputs |

Keep measurements in the original dataset units. `-999` denotes a missing
measurement; `-255` is additionally invalid for temperature. Non-finite readings
are invalid. We forward-fill each sensor causally, using zero until a valid
reading exists. Cumulative features use only the current and earlier readings.
Preserve the original observation order. CSVs used only for prediction may omit
`anomaly`.

The index has one row per transceiver:

```csv
file_name,folder_index,Label
module_001.csv,1,0
module_002.csv,2,1
module_003.csv,3,0
```

`Label` is the module-level failure label. `folder_index` contains the original
three-fold allocation for archived-result reproduction. Fixed-holdout runs
derive their own 80:20 split and three inner folds; the original fold assignment
is not used for these new partitions. Use the original index to reproduce the
archived table. An index made with another split is a different experiment.

Training takes the module label as the target for retained observations before
the first positive `anomaly` record; normal-module targets are zero. The first
positive record's raw timestamp is the failure reference used by evaluation.
Do not interpret it as independently measured physical failure time.

SFB uses 80 active features:

- Minimum, range, maximum, skewness, kurtosis and standard deviation for each
  sensor's cumulative history (72).
- Correlations over the latest five observations: current/temperature,
  current/aggregate Tx, current/aggregate Rx, and aggregate Tx/Rx (4).
- Current aggregate-minus-lane-maximum/minimum gaps for Tx and Rx (4).

The old feature cache still computes compatibility columns to preserve the
original feature ordering and initialization. The model selects only the 80
continuous features listed by `STATISTIC_NAMES`; all threshold/rule columns are
inactive. Tests perturb these columns and verify unchanged predictions.

Production CSVs and the production index are not bundled with this release.
The `smoke` command generates explicitly synthetic data for checking the code.
