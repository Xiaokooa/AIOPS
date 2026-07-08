# TSF-XGBoost Temporal Encoder and Protocol Ablations

This note records two experimental entry points added for quick validation.

## 1. Temporal encoder + statistic MLP + feature alignment + cross-attention + XGBoost

The script avoids direct feature concatenation into XGBoost. It trains a supervised
fusion encoder first:

```text
raw telemetry window -> temporal encoder -> projected embedding(256) -> latent(128)
engineered/statistical features -> MLP -> latent(128)
two latent tokens -> feature alignment -> cross-attention/gated fusion -> fused latent(128)
fused latent -> XGBoost
```

The default temporal encoder is PatchTST. The same TSF-XGBoost framework can
also replace the temporal branch with iTransformer, ModernTCN, or FITS:

```bash
GPU_ID=1 FOLDS="1" bash OFP_DL_model2_compat/scripts/run_tsf_temporal_encoder_sweep.sh
```

To include PatchTST in the same sweep:

```bash
GPU_ID=1 FOLDS="1" TEMPORAL_ENCODERS="patchtst itransformer moderntcn fits" bash OFP_DL_model2_compat/scripts/run_tsf_temporal_encoder_sweep.sh
```

Quick one-fold command:

```bash
GPU_ID=1 FOLD=1 bash OFP_DL_model2_compat/scripts/run_patchtst_stat_aligned_xgb_quick.sh
```

Precision-control one-fold command. Use this first when the quick run shows high
recall but many false positives:

```bash
GPU_ID=1 FOLD=1 bash OFP_DL_model2_compat/scripts/run_patchtst_stat_aligned_xgb_precision_quick.sh
```

Direct Python command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u -B OFP_DL_model2_compat/run_patchtst_stat_aligned_xgb.py \
  --temporal_encoder patchtst \
  --folds 1 \
  --device cuda \
  --out_root OFP_DL_model2_compat_results/patchtst_stat_aligned_xgb_quick_fold1 \
  --seq_len 32 \
  --epochs 3 \
  --batch_size 192 \
  --feature_mode model2_plus \
  --stat_feature_mode all_engineered \
  --fusion_mode gated_attn \
  --sampling_mode module_balanced \
  --sample_selection hybrid \
  --temporal_positive_weight 2.0 \
  --adaptive_negative_weight 1.0 \
  --positive_windows_per_module 32 \
  --negative_windows_per_faulty_module 8 \
  --normal_windows_per_module 8 \
  --threshold_grid 0.001,0.003,0.005,0.01,0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.93,0.95,0.97,0.98,0.99 \
  --max_train_files 2500 \
  --max_test_files 1200 \
  --xgb_balance_mode auto \
  --ml_n_estimators 300 \
  --xgb_tree_method hist \
  --n_jobs 1
```

Useful diagnostic switches:

- `--fusion_mode gated_attn` uses modality gating after cross-attention.
- `--fusion_mode attn_mean` reproduces the earlier mean-pooling fusion.
- `--temporal_encoder patchtst|itransformer|moderntcn|fits` replaces the temporal branch while keeping the TSF-XGBoost framework fixed.
- `--rule_mode none` evaluates the latent XGBoost without rule OR.
- `--xgb_balance_mode none --no_xgb_sample_weight` removes duplicate class/row weighting.
- `--stat_feature_mode model2_expert` tests only OFP expert relational features.

Main outputs:

- `fold_metrics.csv`
- `model_metrics_mean_std.csv`
- `patchtst_stat_aligned_xgb/fold_*/feature_groups.json`
- `patchtst_stat_aligned_xgb/fold_*/aligned_encoder.pt`
- `patchtst_stat_aligned_xgb/fold_*/xgb_fused_latent.pkl`

DRAM-style lead-time sensitivity can be evaluated in the same training run. The
lead-time grid uses the first-warning evaluator with a stricter minimum hit lead
time, and re-selects the operating threshold on validation traces for each lead
requirement:

```bash
GPU_ID=1 FOLDS="1 2 3" bash OFP_DL_model2_compat/scripts/run_patchtst_stat_aligned_xgb_lead_sweep.sh
```

Use another temporal encoder under the same lead-time protocol:

```bash
GPU_ID=1 TEMPORAL_ENCODER=itransformer FOLDS="1 2 3" bash OFP_DL_model2_compat/scripts/run_patchtst_stat_aligned_xgb_lead_sweep.sh
```

Default grid: `1m,5m,15m,30m,1h,2h,5h,12h,24h`.

Main outputs:

- `lead_time_sweep.csv`
- `lead_time_sweep_mean_std.csv`
- `patchtst_stat_aligned_xgb/fold_*/lead_time_sweep/aligned_latent_xgb/lead_time_sweep.csv`

## 2. XGBoost-only protocol ablation

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

Lead-time sensitivity for the XGBoost-only path:

```bash
FOLDS="1 2 3" VARIANTS="full_dram_xgb" bash OFP_DL_model2_compat/scripts/run_xgb_dram_lead_sweep.sh
```

Use `VARIANTS="all"` to run the lead-time table for every cumulative XGBoost
ablation variant. Outputs are written to `lead_time_sweep.csv` and
`lead_time_sweep_mean_std.csv` under the selected `OUT_ROOT`.
