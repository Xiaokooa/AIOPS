# AIOps / OFP Research Repository

This repository contains the Optical Fault Prediction (OFP) data pipeline,
feature baselines, deep-learning models, evaluation protocols, and paper assets.

## Repository layout

```text
OFP/
  model1/                    Legacy Model1 implementation
  model2/                    Model2 rules and engineered features
  unified_baseline/          Reproducible B0/B1/B2 and HTSF experiments
  deep_learning/
    official/                Protocol-aligned training and evaluation pipelines
    compat/                  Model2-compatible adapters
    optical_prediction/      Shared neural model implementations
third_party/
  iTransformer/              Vendored upstream backbone
  ModernTCN/                 Vendored upstream backbone
  PatchTST/                  Vendored upstream backbone
  timesfm/                   Git submodule
dataset/                     Local data (ignored by Git)
```

Generated results, datasets, caches, and local analysis output are excluded
from version control. New maintained code should live under `OFP/`; third-party
source should stay isolated under `third_party/`.

## Main entry points

- Unified baselines: `OFP/unified_baseline/run_experiments.py`
- Clean HTSF: `OFP/unified_baseline/htsf/run_htsf.py`
- Official deep benchmark: `OFP/deep_learning/official/run_official_benchmark.py`
- Model2-compatible deep pipeline: `OFP/deep_learning/compat/run_model2_compat_deep.py`

Run commands from the repository root so package imports and relative data
paths resolve consistently.
