# FORT 紧凑模块消融

该套件只回答论文最关键的模块问题，并且所有学习模型都使用相同的
`legacy_inclusive_v1`、完整三折 OOF 模块集合和 validation-only 决策阈值。
消融时统一关闭 safe Rule fallback，避免“兜底策略”掩盖被消融模块的真实效果。

## 实验设计

| ID | 模型 | 时序编码 | Rule 引导 | PKD | 正类权重 | 训练次数 |
| --- | --- | --- | --- | --- | --- | --- |
| S0_static_xgb | 同协议静态 XGB | 否 | 否 | 否 | 是 | 3 |
| R0_rule_only | 确定性 RuleModel | 否 | 是 | 否 | 否 | 0（从 M4 产物派生） |
| M2_temporal_only | causal TCN | 是 | 否 | 是 | 是 | 3 |
| M3_fort_no_class_weight | FORT core w/o class weight | 是 | 是 | 是 | 否 | 3 |
| M4_fort | FORT core | 是 | 是 | 是 | 是 | 3 |

主表只需报告四个效应：

1. `M2 - S0`：原生频率时序建模相对静态机器学习是否有增益；
2. `M2 - R0`：时序分支相对纯规则模型是否有增益；
3. `M4 - M2`：RuleModel 对时序模型的引导是否有效；
4. `M4 - M3`：正类权重对类别不平衡是否有效。

PKD（代码中的历史字段名为 FGL）在本组“模块消融”中固定开启，不再额外扩展一行；
它属于训练约束，不是部署模块。S0 必须使用与 M2/M4 完全一致的模块划分、
Legacy-Inclusive 评价和三折 OOF 汇总，历史上不同协议的 XGB 分数不能直接用来证明
时序增益。

因此 `M2 - S0` 的严谨表述是“完整深度时序分支（causal TCN + 固定 PKD 训练）相对
同输入静态 XGB 的增益”，不是把差值全部归因于 TCN 网络结构本身。若论文要进一步
声称纯架构增益，才需要额外增加 CE-only 的成对实验；紧凑主表不作这一更强主张。
S0 也只是用于机制控制的 25 维 current-row XGB，不能替代论文主对比表中的 OFP
XGB/RF（含其统计或专家特征设置）。

表中的 `M4_fort` 特指关闭 safe fallback 的 **FORT core**；主对比表中当前 N2 仍是
带 validation-safe Rule fallback 的完整部署策略。消融统一关闭 fallback，是为了不让
决策兜底掩盖时序、规则引导和类别权重本身的真实差异。

## 为什么不再加入 HSS

FORT 使用原生约 5 分钟序列和 sequence-to-sequence 监督。HSS 会重新改变时间点的
采样分布，丢弃大量轨迹，并使 PKD 的未来对齐监督不再保持原来的含义。因此本套件
不引入 HSS；类别不平衡仅通过 `M4 - M3` 检验正类权重，模块归一化保持不变。

## Lead time

Lead time 不需要重新训练一组模型。套件直接从同一批 OOF 模块首次报警中计算：

- 官方 `avg_lead_hour`：命中提前量总和 / 全部故障模块数；
- `mean_lead_hour_among_hits`：命中提前量总和 / 命中模块数；
- `strict_early_recall`：提前量严格大于 0 的故障模块比例；
- `early_hit_rate_at_6/12/24/72h`：至少提前相应小时命中的故障模块比例。

`lead_time_profile.csv` 给出 0、6、12、24、72 小时的完整曲线。其中 0 小时行就是
Legacy-Inclusive recall/TP，可同时看出模型的总体命中与真正提前预警能力。

## 一条命令运行

从仓库根目录执行：

```bash
python -B OFP/unified_baseline/deep_rule_ofp_v2/run_module_ablation.py \
  --device auto \
  --data-dir dataset/training \
  --index-path 'dataset/train_test_set_index(in).csv' \
  --output-dir OFP/unified_baseline/deep_rule_ofp_v2/artifacts/FORT_module_ablation
```

中断后续跑使用 `--resume`：已有 `comparison.csv` 的完整子实验会复用，未完成的当前
子实验会从头重跑；主动覆盖全部已有子实验使用 `--overwrite`。快速代码闭环可加
`--smoke`，但 smoke 分数不能写入论文。如果已经有同协议的
`static_xgb/comparison.csv`，可加 `--skip-static-xgb` 复用它；即使跳过运行，汇总器仍会
自动纳入该目录中的静态 XGB。

最终生成：

- `module_ablation.csv`：论文主消融表；
- `module_ablation_fold_metrics.csv`：每折审计结果；
- `module_effects.csv`：四个预定义模块效应及指标差值；
- `lead_time_profile.csv`：各模型的提前预警曲线数据。

正式 S0 会构建约 890 万行、25 维的训练矩阵，内存峰值可能达到数 GB。推荐保留
`--device auto`：若 XGBoost 没有 CUDA 支持，静态对照会安全回退到 CPU，深度模型
仍会使用 PyTorch 可见的 GPU。只有需要强制检查两者都能使用 CUDA 时才改为
`--device cuda`。
