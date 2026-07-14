# V2 Rule-Guided Residual + Validation-Safe Rule Fallback

本版本把 Rule Model 以两种互不混淆的方式接入 V2：

1. **训练内的软引导**：38 条规则转换为逐行、因果的 signed margin，经过零初始化、有界的 residual adapter 修正时序 logits；
2. **训练外的硬兜底**：cleaned-causal canonical hard-rule OR 作为冻结的 `rule_only` 候选。只在融合模型的 validation OFP `final_score` 严格更好时启用融合，否则整折退回 canonical Rule Model。

这不是逐行 OR，也不是在 test 上挑更好的结果。决策源在读取 outer-test CSV 前已经冻结。

## 架构

```text
12 raw + 12 mask + 1 delta
            │
            ▼
     causal native TCN ─────────────── temporal logits

causal expanding statistics
            │
            ▼
38 signed rule margins + validity + hard OR
            │
            ▼
zero-init bounded rule adapter ─────── rule delta

temporal logits + rule delta ───────── residual score
```

Hard RuleModel 不参与 CE/FGL 参数训练。它始终保留为确定性候选。因此：

- residual 初始化时严格等于 temporal-only；
- 所有规则不可用时严格退化为 temporal-only；
- rule residual 只读当前及历史行；
- Rule-only 与 residual 在 validation 上分别评价；平局选择 Rule-only；
- Legacy-Inclusive 和 README/OFP strict 两种口径分别冻结决策源。

## 规则口径

规则注册表为 `canonical-legacy-38-causal-v1`：对应 `model2/Config.py` 中实际启用的 38 条规则，不包含被注释掉的 4 条 skew 规则。Min/Max/Diff/Std/Kurt 均按模块前缀计算，严格使用 `<`/`>`，38 条命中结果再做 OR。

V2 保留全部 native rows，并把 `-999`、温度 `-255` 视为缺失，不让其污染 expanding statistics。这与原 Model2 历史代码中的缺失值和删行 quirks 并非 bug-for-bug 相同，因此产物明确标记为 canonical 口径。

## 重要保证边界

`validation_safe_rule` 能保证：

- validation 的已选候选分数不低于同一 validation 上的 Rule-only；
- 一旦选择 `rule_only`，test 逐行预测与 canonical RuleModel 完全相同；
- test 标签从不参与阈值或决策源选择。

它不能在未知 test 上数学保证 `Final/F1 >= Rule-only`。任何不读取 test 标签的模型选择都不存在这种保证。为便于审计，程序会在选择冻结之后同时计算 test 的 Rule-only 与 residual 反事实结果；它们只用于报告，不会反向改变已选结果。

## 单折正式运行

默认配置已经启用 `rule_guided_residual_tcn + validation_safe_rule`：

```bash
python -B OFP/unified_baseline/deep_rule_ofp_v2/run_experiment.py \
  --device cuda \
  --experiment-name N2_rule_guided_safe \
  --data-dir dataset/training \
  --index-path 'dataset/train_test_set_index(in).csv' \
  --output-dir OFP/unified_baseline/deep_rule_ofp_v2/artifacts/N2_rule_guided_safe
```

若希望 residual 必须比 Rule-only 多出一定 validation 增益才启用，例如 `0.05`：

```bash
python -B OFP/unified_baseline/deep_rule_ofp_v2/run_experiment.py \
  --device cuda \
  --fallback-min-gain 0.05 \
  --experiment-name N2_rule_guided_safe_margin005 \
  --data-dir dataset/training \
  --index-path 'dataset/train_test_set_index(in).csv' \
  --output-dir OFP/unified_baseline/deep_rule_ofp_v2/artifacts/N2_rule_guided_safe_margin005
```

## OFP 全量三折运行

```bash
python -B OFP/unified_baseline/deep_rule_ofp_v2/run_threefold.py \
  --device cuda \
  --experiment-name N2_rule_guided_safe_3fold \
  --data-dir dataset/training \
  --index-path 'dataset/train_test_set_index(in).csv' \
  --output-dir OFP/unified_baseline/deep_rule_ofp_v2/artifacts/N2_rule_guided_safe_3fold
```

每个 outer fold 可以独立选择 `rule_only` 或 `rule_guided_residual`；最终仍然拼接 13,372 个 OOF 模块决策后重新计算 pooled 指标。

## 纯时序对照

下面的命令同时关闭规则残差和安全回退，得到真正的 temporal-only 对照：

```bash
python -B OFP/unified_baseline/deep_rule_ofp_v2/run_experiment.py \
  --device cuda \
  --architecture causal_depthwise_tcn \
  --disable-safe-rule-fallback \
  --experiment-name N2_temporal_only \
  --data-dir dataset/training \
  --index-path 'dataset/train_test_set_index(in).csv' \
  --output-dir OFP/unified_baseline/deep_rule_ofp_v2/artifacts/N2_temporal_only
```

## 关键产物

- `result.csv`：Legacy-Inclusive 下 validation 冻结后真正采用的 test 结果；
- `result_ofp_original.csv`：README/OFP strict 口径下冻结后的 test 结果；
- `validation_decision_selection.csv`：Legacy 两候选及选择原因；
- `validation_decision_selection_ofp_original.csv`：strict 两候选及选择原因；
- `test_candidate_comparison.csv`：选择冻结后，Rule-only 与 residual 的 test 审计结果；
- `test_candidate_comparison_ofp_original.csv`：strict 口径的候选审计结果；
- `test_rule_only_module_decisions*.csv`：Rule-only 模块级结果；
- `test_residual_module_decisions*.csv`：residual 模块级结果；
- `checkpoints/student_best.pt`：包含规则 schema、冻结决策源和阈值的部署 checkpoint。
