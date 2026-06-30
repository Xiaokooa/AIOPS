# OFP Model2-Compatible Experiment Commands

All commands assume the repository root is the current directory:

```bash
cd /path/to/AIOps
```

The default backbone list is:

```text
fits itransformer moderntcn patchtst
```

Training logs print epoch progress in this form:

```text
========================================================================================
OFP MODEL2-COMPAT DEEP TRAIN
----------------------------------------------------------------------------------------
                 model: fits
                  fold: 1
        train/val/test: 8023/892/4457 modules
...
[compat-train] model=fits fold=1 ep=03/08 [#######-----------]  37.5% loss=... rows=... rate=... epoch=... elapsed=... eta=...
[deep-train] model=fits ep=03/08 [#######-----------]  37.5% loss=... rows=... rate=... epoch=... elapsed=... eta=...
```

## Main Three Methods

### 1. Pure Deep Backbone + OFP Adapter

Strict deep-only diagnostic. No rule OR, no tabular ML.

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal \
  --methods pure_deep \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/pure_deep_formal
```

### 2. OFP-Compatible Deep Adapter

Direct deep row classifier with model2-compatible objective and rule OR.

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal \
  --methods ofp_compat_deep \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/ofp_compat_deep_formal
```

### 3. Deep Embedding + Model2 Features + RF/XGB/LGBM/CatBoost

Hybrid/fusion route. Missing optional LightGBM/CatBoost packages are skipped
unless `--require_all_ml` is set.

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal \
  --methods hybrid_fusion \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/hybrid_fusion_formal
```

Run all three:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal \
  --methods all \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/paper_suite_formal
```

## Backbone-Level Runs

Run a single backbone by adding `--models`.

### Pure Deep Per Backbone

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods pure_deep --models fits --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/pure_deep_fits
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods pure_deep --models itransformer --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/pure_deep_itransformer
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods pure_deep --models moderntcn --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/pure_deep_moderntcn
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods pure_deep --models patchtst --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/pure_deep_patchtst
```

### OFP-Compatible Deep Per Backbone

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods ofp_compat_deep --models fits --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/ofp_compat_fits
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods ofp_compat_deep --models itransformer --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/ofp_compat_itransformer
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods ofp_compat_deep --models moderntcn --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/ofp_compat_moderntcn
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods ofp_compat_deep --models patchtst --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/ofp_compat_patchtst
```

### Hybrid Fusion Per Backbone

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods hybrid_fusion --models fits --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/hybrid_fits
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods hybrid_fusion --models itransformer --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/hybrid_itransformer
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods hybrid_fusion --models moderntcn --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/hybrid_moderntcn
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods hybrid_fusion --models patchtst --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/hybrid_patchtst
```

## DRAM-Inspired Ablations

Default paper suite enables:

```text
sample_selection=hybrid
temporal_positive_weight=2
adaptive_negative_weight=1
```

No DRAM-inspired sampling/weighting:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --sample_selection random \
  --temporal_positive_weight 0 \
  --adaptive_negative_weight 0 \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_no_dram_weighting
```

Top-k signal sampling only:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --sample_selection signal_topk \
  --temporal_positive_weight 2 \
  --adaptive_negative_weight 1 \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_signal_topk
```

Temporal positive weighting only:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --sample_selection random \
  --temporal_positive_weight 2 \
  --adaptive_negative_weight 0 \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_temporal_pos_only
```

Adaptive hard-negative weighting only:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --sample_selection random \
  --temporal_positive_weight 0 \
  --adaptive_negative_weight 1 \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_adaptive_neg_only
```

## Target / Feature / Rule Ablations

Use current anomaly target:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --compat_target_mode anomaly \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_target_anomaly
```

Use strict ahead target:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --compat_target_mode ahead120 \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_target_ahead120
```

Use base model2 features only:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --compat_feature_mode model2 \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_feature_model2
```

Remove rule OR:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods ofp_compat_deep \
  --compat_rule_mode none \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/abl_no_rule
```

## Hybrid/Fusion Ablations

Fusion with all available ML models:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods hybrid_fusion \
  --ml_models rf,xgb,lgbm,catboost \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/hybrid_all_ml
```

Require LightGBM/CatBoost to be installed:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods hybrid_fusion \
  --ml_models rf,xgb,lgbm,catboost \
  --require_all_ml \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/hybrid_all_ml_require
```

Model2 features only, no deep embedding features:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods hybrid_fusion \
  --ml_feature_set model2 \
  --ml_models rf,xgb,lgbm,catboost \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/hybrid_model2_only
```

Deep embedding/score only:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods hybrid_fusion \
  --ml_feature_set embedding \
  --deep_feature_parts embedding,score \
  --ml_models rf,xgb,lgbm,catboost \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/hybrid_embedding_only
```

Fusion without rule OR:

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py \
  --profile formal --methods hybrid_fusion \
  --hybrid_rule_mode none \
  --ml_models rf,xgb,lgbm,catboost \
  --device cuda --gpu_id 0 \
  --out_root OFP_DL_model2_compat_results/hybrid_no_rule
```

## Lead-Time Sensitivity

These experiments change the hit definition. Use them as operational
constraint/sensitivity experiments, not as the replacement for the main OFP
comparison unless the paper explicitly defines this stricter metric.

```bash
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods all --min_hit_lead_hours 1 --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/lead_min_1h
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods all --min_hit_lead_hours 2 --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/lead_min_2h
python -B OFP_DL_model2_compat/run_paper_suite.py --profile formal --methods all --min_hit_lead_hours 5 --device cuda --gpu_id 0 --out_root OFP_DL_model2_compat_results/lead_min_5h
```

## Server Setup With Existing Dataset

### Case A: Existing `AIOps` Is Already a Git Repository

Keep the existing `dataset/` in place and pull only code:

```bash
cd /path/to/AIOps
test -d dataset/training
git status --short
git remote -v
printf "\ndataset/\nOFP_DL_model2_compat_results/\n" >> .git/info/exclude
git fetch origin
git pull --ff-only origin main
```

If the branch is `master`, replace `main` with `master`.

If the server has local code changes that block pulling:

```bash
cd /path/to/AIOps
git stash push -u -m "server-backup-before-ofp-update-$(date +%Y%m%d_%H%M%S)"
git pull --ff-only origin main
```

### Case B: Existing `AIOps` Was Manually Copied To The Server

This is the recommended path when the server's old `AIOps` folder was uploaded
manually and is not a Git repository. Do not overwrite it. Rename the old folder,
clone fresh code, then symlink the old dataset into the new code checkout.

```bash
cd /path/to
OLD=AIOps_old_$(date +%Y%m%d_%H%M%S)
mv AIOps "$OLD"
git clone <YOUR_GITHUB_REPO_URL> AIOps
ln -s "$(pwd)/$OLD/dataset" "$(pwd)/AIOps/dataset"
cd AIOps
test -d dataset/training
```

If your dataset is on another disk, symlink that exact dataset path instead:

```bash
cd /path/to/AIOps
ln -s /absolute/path/to/old/AIOps/dataset dataset
test -d dataset/training
```

If `dataset` already exists in the fresh Git checkout as a tracked or empty
folder, remove only that empty folder first:

```bash
cd /path/to/AIOps
rmdir dataset
ln -s /absolute/path/to/old/AIOps/dataset dataset
test -d dataset/training
```

If symlinks are not allowed on the server, copy with `rsync`:

```bash
cd /path/to/AIOps
mkdir -p dataset
rsync -a --info=progress2 /absolute/path/to/old/AIOps/dataset/ dataset/
test -d dataset/training
```

Install optional tabular dependencies:

```bash
pip install xgboost lightgbm catboost
```
