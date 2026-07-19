# Model2-compatible deep-learning adapters

This package contains the retained Model2-compatible deep, tabular, and fusion
pipelines. It is part of the unified `OFP.deep_learning` package and reuses
models, data helpers, losses, trainers, and evaluators from the sibling
`OFP.deep_learning.official` package.

## Contents

```text
compat/
├── run_model2_compat_deep.py
├── run_model2_compat_tabular_only.py
├── run_model2_compat_tabular_fusion.py
├── prepare_ofp_feature_cache.py
└── scripts/
```

Paper-suite orchestration, result-only utilities, and dedicated ablation/sweep
launchers were removed during repository cleanup. Use
`OFP/unified_baseline/htsf` for the maintained HTSF experiment framework.

Run entry points from the repository root, for example:

```powershell
python -B OFP\deep_learning\compat\run_model2_compat_deep.py --help
```
