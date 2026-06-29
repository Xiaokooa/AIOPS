# OFP External Model Adapters

This folder converts external model scores into the same OFP event-level output
used by the other experiments:

```text
timestamp,predict
```

Each adapter writes:

```text
fold_metrics.csv
model_metrics_mean_std.csv
<model>/fold_<k>/predictions/*.csv
<model>/fold_<k>/evaluation/evaluate_result.csv
<model>/fold_<k>/evaluation/module_decisions.csv
```

## Shared Evaluation Logic

All adapters use the same evaluator as `OFP_DL_official`:

```text
predict_ts < true_first_anomaly_ts
```

So an alarm one second before the first anomaly counts as a hit. An alarm at the
same timestamp does not.

## CAROTS Adapter

### Train CAROTS-OFP From Raw OFP CSVs

This is the direct path when you do not already have CAROTS score files:

```bash
GPU_ID=0 bash OFP_external_adapters/scripts/run_carots_ofp_train_eval.sh
```

It performs:

```text
dataset/training/*.csv
  -> module-safe OFP windows
  -> CAROTS-style contrastive encoder training
  -> centroid-distance anomaly scores
  -> validation threshold search
  -> timestamp,predict
  -> OFP evaluator
```

Outputs:

```text
OFP_external_adapter_results/carots_ofp_<run_id>/fold_metrics.csv
OFP_external_adapter_results/carots_ofp_<run_id>/model_metrics_mean_std.csv
OFP_external_adapter_results/carots_ofp_<run_id>/carots_ofp/fold_*/scores/*.csv
OFP_external_adapter_results/carots_ofp_<run_id>/carots_ofp/fold_*/predictions/*.csv
```

Small smoke:

```bash
FOLDS=1 \
EPOCHS=1 \
MAX_TRAIN_FILES=50 \
MAX_TEST_FILES=20 \
GPU_ID=0 \
bash OFP_external_adapters/scripts/run_carots_ofp_train_eval.sh
```

### Score-Only CAROTS Adapter

Real CAROTS integration should produce one score CSV per OFP module:

```text
<score_dir>/<file_name>.csv
```

Required columns:

```text
timestamp,score
```

Then run:

```bash
CAROTS_SCORE_DIR=/data2/mxk/AIOps/carots_scores \
GPU_ID=0 \
bash OFP_external_adapters/scripts/run_carots_adapter.sh
```

Adapter smoke only:

```bash
SCORE_MODE=normal_residual \
MODEL_NAME=carots_normal_residual_adapter \
MAX_TRAIN_FILES=50 \
MAX_TEST_FILES=20 \
FOLDS=1 \
bash OFP_external_adapters/scripts/run_carots_adapter.sh
```

The `normal_residual` mode is not a CAROTS model result. It only verifies that
the OFP score-to-alarm and evaluator path works.

## TimesFM Adapter

Install TimesFM and download weights first:

```bash
cd /data2/mxk/AIOps/model/timesfm
pip install -e ".[torch]"
python ofp_download_timesfm_weights.py \
  --model-id google/timesfm-2.5-200m-pytorch \
  --cache-dir /data2/mxk/AIOps/model/timesfm_hf_cache
```

Run the OFP forecast-risk adapter:

```bash
GPU_ID=0 \
CACHE_DIR=/data2/mxk/AIOps/model/timesfm_hf_cache \
bash OFP_external_adapters/scripts/run_timesfm_adapter.sh
```

Local or dependency-free smoke:

```bash
MOCK_FORECAST=1 \
MAX_TRAIN_FILES=50 \
MAX_TEST_FILES=20 \
FOLDS=1 \
bash OFP_external_adapters/scripts/run_timesfm_adapter.sh
```

`MOCK_FORECAST=1` verifies the adapter and evaluator plumbing. It is not a
TimesFM result.
