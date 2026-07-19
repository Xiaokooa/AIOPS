# DRFP-Net Evidence Plan

No formal experimental result is claimed in this document. All result cells must be filled from completed runs under the frozen protocol.

## Claim–evidence matrix

| Claim | Reviewer question | Evidence | Comparison | Main metrics | Status |
| --- | --- | --- | --- | --- | --- |
| Deep temporal modeling helps OFP | Does a deep sequence model add useful failure signal? | Fixed fold3 main result | D0 vs classical same-split baselines | OFP final/F1/P/R/accuracy/lead | planned |
| Statistical context complements raw dynamics | Is the statistical branch necessary? | Architecture ablation | D1 vs D0 | same | implemented, result TBD |
| Rule signals have independent predictive value | Is “expert knowledge” real or decorative? | Standalone rule branch | D2 vs D0/D1 and legacy hard RuleModel | same + rule participation | implemented, result TBD |
| Learned reliability is better than naive fusion | Why not average the branches? | Fusion mechanism ablation | D4 vs D3 | same + disagreement/gate | implemented, result TBD |
| Full deep-rule model improves the fixed deep model | Does expert knowledge help rather than merely add parameters? | Main ablation | D4 vs D1 and D2 | same + parameter count/time | implemented, result TBD |

## Main architecture table template

| Variant | Primary branch | Threshold policy | Params | Final ↑ | F1 ↑ | Precision ↑ | Recall ↑ | Accuracy ↑ | AvgLead ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 temporal only | Deep | fixed / selected | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D1 deep raw+stat | Deep | fixed / selected | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D2 rule only | Rule | fixed / selected | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D3 fixed fusion | Fusion | fixed / selected | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| D4 gated fusion | Fusion | fixed / selected | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Required follow-up evidence

1. Same-split baselines: Model1 Raw-XGB, Raw+Stat-XGB, legacy RuleModel, and any selected statistical ML baseline must use the exact 8022/892/4458 split and the same threshold policy.
2. Repeated seeds: at least 3 training seeds for neural variants; report mean/std while keeping the module split fixed.
3. Horizon ablation: single `p120` vs multi-horizon supervision, and optionally 120h vs 168h primary horizon as a separate task-axis experiment.
4. Rule robustness: rule-signal dropout, missing-channel stress, and analysis of cases where gate suppresses or promotes the Rule branch.
5. Efficiency: training time, test throughput, peak memory and parameters; separate model compute from feature preprocessing.
6. Failure analysis: false alarms on normal modules, misses, >120h hits, late hits, Deep-only wins, Rule-only wins, and harmful fusion cases.
7. Objective-alignment study: fixed-window BCE is a surrogate for the module-any-alarm OFP metric. A later, separately labeled experiment may compare it with an OFP-aligned sequence/MIL loss; it must not be silently merged into v1.

Stop condition: if D4 does not improve or stabilize D1 across seeds, keep the rule branch as an explanatory ablation rather than claiming it as the default model.
