# CAROTS Local Reproduction Status

Source repository: https://github.com/kimanki/CAROTS
Local path: `D:\AIOps\model\CAROTS`
Download method: GitHub `master.zip` snapshot, extracted on 2026-06-14.

## Status

Corrected status: the repository does contain the full CAROTS code.

The earlier local copy under `model\CAROTS` only had `README.md` and
`LICENSE`, so it was incomplete. The current folder has been updated with the
full source snapshot.

## Included Components

- `main.py`, `trainer.py`, `threshold.py`, `config.py`
- `models/carots/`: CAROTS model, CUTS+ causal discoverer, augmentors, scorer,
  predictor, loss, and encoder wrappers
- `models/itransformer/` and `models/timesnet/`: encoder/model components
- `layers/`: Transformer, attention, embedding, and convolution blocks
- `datasets/`: dataset builders, loaders, collate utilities
- `scripts/`: runnable scripts for SWaT, WADI, PSM, SMD, SMAP/MSL, Lorenz96,
  and VAR examples
- `data/`: synthetic generation code and some preprocessed public benchmark
  artifacts included in the repository snapshot
- `results/`: pretrained/intermediate result artifacts included in the
  repository snapshot

## Likely Runtime Dependencies

The repository does not include a `requirements.txt`. Imports indicate these
packages are needed:

- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `einops`
- `matplotlib`
- `tqdm`
- `yacs`
- `reformer-pytorch`

CUDA is assumed by the upstream implementation in several places, for example
CAROTS initializes submodules with `.cuda()`.

## Upstream Run Pattern

From the repository root:

```bash
cd /data2/mxk/AIOps/model/CAROTS
bash scripts/SWaT.sh
```

The upstream scripts call `python main.py ...` with dataset-specific overrides.

## OFP Adaptation Notes

CAROTS is an unsupervised/contrastive multivariate time-series anomaly detection
framework. For OFP first-warning evaluation, it should not be used as a
drop-in 120h binary classifier. A safer adaptation path is:

1. Train CAROTS on normal or mostly-normal module traces.
2. Produce row/window-level anomaly scores with `Predictor`/`Scorer`.
3. Convert scores to module-level first-warning alarms using the existing OFP
   evaluator: validation threshold search, smoothing, consecutive-K, first alarm
   only.
4. Compare separately against:
   - true long-lead modules;
   - start-in-window modules;
   - left-censored/already-failed modules.

This keeps CAROTS aligned with anomaly-state detection while preserving the OFP
event-level metric boundary.
