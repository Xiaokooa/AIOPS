# Experiment protocols

The model architecture is the same for both protocols. Results must retain their
protocol label; the published archived table cannot be relabeled as fixed-test
results.

## Archived three-fold evaluation (`legacy_cv`)

1. Use the supplied index's original `folder_index` values 1, 2 and 3.
2. Hold out one entire fold for testing. Stratify the other modules by failure
   label into training and validation at approximately 80:20 using the original
   NumPy splitting routine (`42 + 1009 * fold`).
3. Fit feature normalization and sensor-distribution parameters on training
   modules only. HSS candidates also come exclusively from these modules.
4. Train for all 16 epochs; there is no validation early stopping. Select the
   alarm threshold by validation F1 from the archived 21-value threshold grid.
5. Evaluate each held-out test fold, then pool unique module decisions from the
   three folds. Compute the seven reported metrics from pooled decisions, not
   by averaging per-fold metrics.

The split3 choice followed previous experimentation on this benchmark. The
archived numbers should not be presented as performance on an untouched new
holdout. This release preserves their original reproduction path.

## Fixed holdout (`fixed_holdout`)

Sort module names, then use `train_test_split(test_size=0.2, stratify=Label,
random_state=42)`. All observations of a transceiver remain together. For the
original index this gives 10,697 training modules and 2,675 test modules:

| Partition | Normal | Faulty | Total |
|---|---:|---:|---:|
| Training | 7,416 | 3,281 | 10,697 |
| Test | 1,854 | 821 | 2,675 |

Create three inner folds with `StratifiedKFold(3, shuffle=True, random_state=42)`
inside the training partition. Train each fold model for 16 epochs on the other
two inner folds and choose its threshold on its validation fold. Choose the
fold model with highest validation F1 (lower fold number breaks a tie). Use its
saved normalization, checkpoint and threshold for one evaluation of the fixed
test partition. The selected model is not refitted on all training modules.
Each ablation applies the same selection rule independently.

The test data are not scored during inner CV. Split manifests are saved with
each run, and module overlap is rejected. Production training under this new
protocol has not been rerun for this release. Also, this holdout is drawn from
the previously used dataset; changing the split cannot erase prior benchmark
exposure or make it an independent external dataset.

## Architecture and training budget

| Item | Value |
|---|---|
| Input branches | SMB + SFB, one SIT backbone |
| Sensor tokens | 12 |
| Statistical tokens | 3, grouped as 36 / 36 / 8 features |
| Module token | One learnable token |
| Width / depth / heads / FFN width | 64 / 3 / 4 / 192 |
| Transformer dropout | 0.15 |
| Joint statistical-token dropout during training | 0.15 |
| Loss | Ordinary BCE averaged within each module, then across modules |
| Optimizer | AdamW, learning rate 0.0003, weight decay 0.01 |
| Gradient clipping | 1.0 |
| Modules per optimizer step | 32 |
| Epochs | 16 per fold, no early stopping |
| Training seed | 42 by default; Torch uses `training_seed + fold` |
| Data/sampling seed | 42, with the original fold/module offsets |
| AMP / TF32 | Enabled for CUDA in the archived configuration |

The compatibility loader retains a 32-observation window, but the final SMB
uses only the endpoint. There is no temporal branch encoder. SFB features are
cumulative statistics plus local channel correlations/current gaps.

HSS retains up to 32 positive windows per faulty training module and 8 normal
windows per normal module. Each normal candidate pool has at most 64 windows;
retained normal windows combine risk (half), coverage (quarter), and random
selection (remainder). The initial signal is sensor distribution deviation.
Risk is refreshed after epochs 2, 4, 6, 8, 10, 12 and 14. HSS never uses alarm
rules, validation modules or test modules.

## Module ablations

| Variant | Intervention |
|---|---|
| `full` | SSFFN split3 |
| `no_sensor` | Remove all SMB tokens |
| `no_statistics` | Remove all SFB tokens |
| `no_sit` | Remove SIT and module token; mean-pool each branch and equally weight the branches |
| `no_hss` | Uniformly sample normal windows under the same quota and refresh schedule |

Every variant is trained from scratch. Common parameter initialization and the
training RNG sequence are retained from the original implementation. For that
reason `_engine` contains compatibility constructors and older internal names;
those names do not imply an additional encoder, rule fallback or tree model in
the released configuration.

## Metrics and timestamp convention

Evaluation is at transceiver level using the first qualifying warning. Recall
uses all faulty modules in the evaluated partition as its denominator. AvgLead
is the average lead time of hits, in hours. Always interpret it with Recall.

```
AFWS = F1 + Accuracy + tanh(AvgLead_hours)
```

MinLead is not part of the reported AFWS. For exact archived threshold
reproduction, validation F1 ties retain the old evaluator's `final_score`
tie-break, followed by Precision, Accuracy, Recall and threshold. That internal
legacy score includes a minimum-lead term and is not the reported AFWS.
Use `results.csv`, `RESULTS.md`, or `paper_metrics.json` for the seven paper
metrics; do not report internal `final_score` as AFWS.

The explicit historical policy converts observation timestamps through
`float32` before prediction export, while reference failure times come from raw
CSV timestamps. It is preserved for both protocols. Quantization can move an
event-row warning before its raw reference time. Therefore short positive lead
times under this policy alone do not demonstrate strict causal advance warning.
The release does not silently change this behavior or replace the archived
evaluation with a different timestamp policy.

## Outputs and reproducibility

Each fold writes its split, training history, final checkpoint, normalization,
threshold search and a `completed.json` only after the declared budget and
evaluation finish. The final checkpoint includes normalization, feature order,
model variant and seed; prediction requires no original research directory.
`run.json` records the protocol, budget, index hash and completion status.

Interrupted runs are not marked complete. Use a fresh output directory to
restart; this release does not implement optimizer/sampler resume. Exact CUDA
training results can still depend on hardware, PyTorch version and kernels.
No production retraining or new multi-seed claim is made by the release checks.
