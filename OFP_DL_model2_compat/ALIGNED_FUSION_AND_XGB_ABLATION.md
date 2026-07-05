# PatchTST Statistic-Aligned Fusion and XGB DRAM Ablation

This note records two experimental entry points added for quick validation.

## 1. PatchTST + statistic MLP + feature alignment + cross-attention + XGBoost

The script avoids direct feature concatenation into XGBoost. It trains a supervised
fusion encoder first:

```text
raw telemetry window -> PatchTST -> projected embedding(256) -> latent(128)
engineered/statistical features -> MLP -> latent(128)
two latent tokens -> feature alignment -> cross-attention -> fused latent(128)
fused latent -> XGBoost
```

Quick one-fold command:

```bash
GPU_ID=1 FOLD=1 bash OFP_DL_model2_compat/scripts/run_patchtst_stat_aligned_xgb_quick.sh
```

Direct Python command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u -B OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py \
  --folds 1 \
  --device cuda \
  --out_root OFP_DL_model2_compat_results/patchtst_stat_aligned_xgb_quick_fold1 \
  --seq_len 32 \
  --epochs 3 \
  --batch_size 192 \
  --feature_mode model2_plus \
  --stat_feature_mode all_engineered \
  --sampling_mode module_balanced \
  --sample_selection hybrid \
  --temporal_positive_weight 2.0 \
  --adaptive_negative_weight 1.0 \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --max_train_files 2500 \
  --max_test_files 1200 \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs 1
```

Main outputs:

- `fold_metrics.csv`
- `model_metrics_mean_std.csv`
- `patchtst_stat_aligned_xgb/fold_*/feature_groups.json`
- `patchtst_stat_aligned_xgb/fold_*/aligned_encoder.pt`
- `patchtst_stat_aligned_xgb/fold_*/xgb_fused_latent.pkl`

## 2. XGBoost-only DRAM-inspired ablation

The ablation fixes the final classifier as XGBoost and opens the improvements
cumulatively:

| Variant | Feature mode | Sampling | Temporal positive weight | Adaptive negative |
| --- | --- | --- | --- | --- |
| `plain_model2_random` | `model2` | random | 0 | 0 |
| `plus_features` | `model2_plus` | random | 0 | 0 |
| `plus_hybrid_sampling` | `model2_plus` | hybrid | 0 | 0 |
| `plus_hybrid_temporal` | `model2_plus` | hybrid | 2 | 0 |
| `full_dram_xgb` | `model2_plus` | hybrid | 2 | 1 |

Quick one-fold command:

```bash
FOLD=1 bash OFP_DL_model2_compat/scripts/run_xgb_dram_ablation_quick.sh
```

Direct Python command:

```bash
python -u -B OFP_DL_model2_compat/run_xgb_dram_ablation.py \
  --folds 1 \
  --out_root OFP_DL_model2_compat_results/xgb_dram_ablation_quick_fold1 \
  --variants all \
  --seq_len 32 \
  --batch_size 192 \
  --sampling_mode module_balanced \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --max_train_files 2500 \
  --max_test_files 1200 \
  --selector extra_trees \
  --select_k 96 \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs 1
```

Main outputs:

- `fold_metrics.csv`
- `variant_metrics_mean_std.csv`
- `<variant>/fold_*/variant.json`
- `<variant>/fold_*/xgb_model.pkl`
