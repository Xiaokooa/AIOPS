# Manuscript alignment required for clean HTSF

The current paper text and historical HTSF numbers must not be relabeled as
results of this implementation.  Before using new results, align the manuscript
to the following executable definitions.

## Terms that must match code

- Write **channel-aware PatchTST-style encoder** or **channel-aware patch
  Transformer**, not the exact named PatchTST implementation.  This repository
  uses shared channel-independent temporal patch encoding plus learned channel
  identity embeddings and ordered channel-aware pooling.
- Describe the fusion attention as self-attention over two modality tokens
  `[z_t,z_s]`, not attention over time patches.
- State that the prediction task is first-fault prediction within 120 hours and
  that all first-fault/post-fault rows are excluded from training.
- State that the strict main threshold is fixed at 0.5.  Validation-selected
  threshold results, if added later, require a separate threshold-policy table
  applied to every baseline.
- Use only Raw 12, Statistical 76 and Expert 42.  Do not claim `TsDelta` is an
  input to the clean baseline or HTSF.
- State that invalid sentinels remain unchanged in the Raw/B2 view but are
  masked before derived statistics and neural normalization.
- State the new matched-budget HSS definition exactly.  Do not describe the old
  signal-Top-K hybrid unless it is reintroduced as its own isolated experiment.
- Report TPW-only, ANW-only and TPW+ANW separately.  ANW is applied once after
  warm-up to negative representation-loss weights; the final XGBoost is not
  sample-weighted.

## Result-table mapping

- Strict B0/B1/B2 stay in the unified baseline table.
- B2-strict -> H0 sampled-B2 is the sampling/scale bridge.
- H1--H5 form the equal-width representation mechanism ablation.
- H6 is an incremental B2-skip diagnostic and is not the paper's pure HTSF.
- Historical old-runner results remain legacy only because that runner used an
  effective 1-hour label, detached temporal embeddings, an internally
  statistics-wrapped temporal backbone and a different threshold policy.

No paper number should be updated until a complete three-fold run passes the
formal-result gate in `PROTOCOL.md`.
