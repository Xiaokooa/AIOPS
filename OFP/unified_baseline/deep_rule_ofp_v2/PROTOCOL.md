# Frozen Protocol：`legacy_inclusive_v1`

本文档冻结 `deep_rule_ofp_v2` 的任务、数据、训练和评价边界。任何会改变这些边界的实验都必须新建协议名或作为单轴消融显式报告，不能仍标记为 `legacy_inclusive_v1`。

## 1. 任务与适用范围

- 论文任务：Optical Module Failure Prediction。
- 数据单位：一个光模块的一份完整原生观测序列。
- 首故障锚点：文件中按行位置出现的第一个 `anomaly > 0`。
- 主输出：student 在每个观测行输出未来 120 小时故障概率 `p120(t)`。
- 唯一正式 evaluator：`legacy_inclusive_v1`。
- 测试推理：仅 student；teacher 只属于训练期。

该任务不是无监督异常检测。模型虽然逐行输出风险，但最终按照完整模块的首次阈值越界进行 Failure Prediction 评价。

## 2. 划分

单折开发入口 `run_experiment.py` 默认固定使用：

- folds 1+2：development pool，再按模块和标签、seed 42 划分 train 与 validation；
- fold 3：默认 outer test fold；也可以显式指定 fold 1 或 fold 2，以供全量入口调用。

全量 OFP 比较入口 `run_threefold.py` 分别以 fold 1、2、3 作为 outer test fold，产生三个互斥的 out-of-fold 决策集合，再拼接为 13,372 个模块的完整评估集。它不做模型集成，不平均多折预测，也不平均三个 fold 的 F1 或 Final Score；所有 pooled 指标都从拼接后的模块决策重新计算。

每次运行必须保存 `split_manifest.csv` 与 split fingerprint。smoke 模式可以限制每个 split 的模块数，但其结果只能用于实现验证。

## 3. 原生行与输入契约

对每个模块：

1. 保留全部真实观测行及原始顺序；
2. 不做重采样、时间聚合、插值、前向填充或补造 5 分钟网格；
3. 时间戳必须是非递减的 `int64` Unix 秒；相同时间戳的多行仍按原始行位置区分；
4. 一个原始行对应一个 Seq2Seq 时间步；批量 padding 必须由 `padding_mask` 排除；
5. 输入严格为 12 个 raw 值、12 个有效性 mask 和 1 个 `delta_steps`；
6. `delta_steps[0] = 0`，其余为相邻真实观测时间差除以 300 秒，并按配置裁剪；
7. 12 维标准化器只用 train 中的有效行拟合，故障模块最多包含到首故障行。

为减少无效计算，teacher/student 的训练与 validation-loss loader 可以在首故障行处截断故障模块；这不会删除任何有效监督行。用于 validation 阈值和 fold 3 决策的 student-only inference 必须恢复完整原始序列，并为每个原始行输出分数。

任何统计特征、专家特征、rule score、未来窗口统计或测试集统计都不是 v2 输入。

## 4. 因果模型契约

student 与 teacher 都是完整模块级 causal TCN：

- 时刻 `t` 的 logits 只能由该模块 `<= t` 的输入行计算；
- 卷积只做左侧 padding，不得使用双向卷积或未来注意力；
- LayerNorm 只作用于单个 `(module, time)` 的通道向量，不使用跨时间或跨 batch 的统计；
- padding 在每个残差块后归零；
- 默认 10 个 dilation blocks、kernel size 3，对应 2047 个原生行的感受野，覆盖约 120 小时的常规 5 分钟序列。

FGL 允许 teacher 在训练期使用相对 student 更晚的观测前缀，但不能让 student 的前向图直接读取未来输入。

## 5. Legacy-Inclusive 标签

设首故障行位置为 `i_f`，其 `int64` 时间戳为 `T_f`。训练有效行位置满足 `i <= i_f`；首故障行之后的行全部从 CE 和 FGL 中排除。相同时间戳不会改变这一位置边界。

对故障模块：

```text
y_H(i) = 1{0 <= T_f - T_i <= H},  i <= i_f
```

对正常模块，所有真实观测行的 `y_H = 0`。当前冻结值为：

```text
student_horizon = 120 h
future_offset   =   6 h
teacher_horizon = 114 h
```

因此首故障行满足 `y120 = y114 = 1`，是合法训练样本，不被丢弃。

## 6. FGL 对齐与损失

对每个 student 行 `t`：

1. 计算精确目标时间 `t + 6 h`；
2. 在同一模块寻找首个时间戳不早于目标时间的真实 teacher 行；
3. 该行与目标时间的差必须不超过配置容差（默认 300 秒）；
4. teacher 行必须仍在训练有效范围内；
5. 必须满足 `y114(t+6h) = y120(t)`；
6. 不满足任一条件时，该 student 行不参与 KL，但仍可参与 CE。

teacher 先使用其 Future-window CE 独立训练。选定最佳 validation checkpoint 后，teacher 切换到 eval、冻结参数，并在 KL 中再次 detach logits。student 的损失为：

```text
L_student = alpha * L_CE_120
          + (1 - alpha) * tau^2 * L_KL

L_KL = KL(q_teacher_114(t+6h) || q_student_120(t))
q     = softmax(logits / tau)
```

默认 `alpha = 0.7`、`tau = 4.0`。`N0_native_s2s_ce` 令 `alpha = 1.0`，即严格 CE-only；`N1_native_s2s_ce_fgl` 使用默认 CE + FGL-KL。两个损失都按模块归一化，类别权重只从 train 目标计算。没有合法 future pair 的模块先在 KL 项中贡献 0，再乘由完整 train 固定得到的 `all_modules / fgl_covered_modules`，从而恢复“全局有配对模块均值”；该缩放不随 batch 改变，长度分桶不会隐式放大或削弱稀疏 KL 模块。

### 6 小时 offset 的审计依据

正式 train 划分的覆盖统计为：

| Offset | 正配对数 | 至少含一个正配对的故障模块 |
| ---: | ---: | ---: |
| 6 h | 74,868 | 357 |
| 12 h | 57,111 | 314 |
| 24 h | 39,997 | 123 |
| 48 h | 19,973 | 58 |
| 96 h | 2,596 | 38 |

默认 6 小时是为了避免更长 offset 导致训练信号覆盖骤降。它是数据覆盖选择，不是已经通过 fold 3 证明的最优超参数；offset 的性能比较若开展，只能在 validation 上定案或作为预注册消融报告。

## 7. 不平衡处理边界

- 默认 CE 可使用 `module_normalized_auto` 正类权重；权重只由 train 标签估计并按配置截断。
- 每个模块的行损失先在模块内归一化，再跨模块平均，避免长模块天然获得更高权重。
- v2 不默认启用 HSS、TPW、ANW、过采样、SMOTE、focal loss 或 bag loss。
- 任何额外不平衡策略必须形成新的单轴消融，不能混入 N0/N1 对比。

## 8. Validation-only 决策与测试读取顺序

严格顺序为：

```text
index membership
  -> train CSV
  -> validation CSV
  -> train-only normalization / teacher / student
  -> validation student scores
  -> validation threshold frozen
  -> test CSV
  -> test student scores
  -> frozen Legacy-Inclusive evaluation
```

阈值候选只由 validation score 生成，并按以下顺序排序：

1. `final_score` 更高；
2. F1 更高；
3. precision 更高；
4. 阈值更高。

默认候选集合由 `[0,1]` 均匀网格、固定 0.5 和 validation score 分位点组成；这是对 N0/N1 完全相同的冻结候选搜索，不应在论文中表述为“穷举全部唯一分数得到的精确全局最优阈值”。

fold 3 CSV 在阈值冻结前不得打开。测试标签只能进入最后的冻结 evaluator 和诊断表，不能用于归一化、early stopping、teacher 选择、阈值选择或重跑决策。

## 9. 模块级 Legacy-Inclusive 评价

给定冻结阈值 `theta`：

```text
alarm_ts = 模块从首行到首故障行（含）第一个满足 p120(t) >= theta 的时间戳
failure_ts = 第一个 anomaly > 0 行的时间戳
```

逐模块判定：

- 故障模块：当且仅当 `alarm_ts <= failure_ts` 时为 TP，否则为 FN；
- 正常模块：存在任意报警为 FP，否则为 TN；
- 只有故障后的报警仍是 FN，不额外计作 FP；
- 故障模块首故障行之后的 score 不进入阈值候选或模块决策；即使后续行重复故障时间戳，也不能伪装成同刻命中；
- `alarm_ts = failure_ts` 是合法 TP，`lead_hour = 0`；
- 命中提前量 `lead_i = (failure_ts - alarm_ts) / 3600`。

指标定义：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
accuracy  = (TP + TN) / all_modules

lead_sum_hour = sum(lead_i for all TP modules)
AvgLead       = lead_sum_hour / all_faulty_modules
MinLead       = min(lead_i for TP modules), or 0 when no TP

final_score = F1 + accuracy + tanh(AvgLead) + tanh(MinLead)
```

`AvgLead` 的分母是**全部故障模块数**，因此漏报自动贡献 0。若存在故障时刻命中，则其 0 小时会参与 `MinLead`。所有时间计算使用 `int64` 秒，不允许利用浮点时间戳误差制造同刻之前的报警。

## 10. 消融与结果声明

冻结的最小消融为：

- `N0_native_s2s_ce`：native full-module Seq2Seq causal TCN + Future-window CE；
- `N1_native_s2s_ce_fgl`：在 N0 上仅增加 frozen-teacher FGL-KL。

二者的输入、student 参数量、划分、训练预算、阈值策略和 evaluator 必须一致。结果表至少报告 final score、F1、precision、recall、accuracy、AvgLead、MinLead、阈值、student 参数量、teacher 训练参数量、总耗时和 FGL 覆盖率。

在正式 fold 3 结果产生前，只能说明“实现了”或“验证了机制闭环”，不能宣称效果提高。即使 N1 在一次测试上优于 N0，也应结合随机种子或预注册复现实验后再形成稳定论文结论。

### 10.1 OFP 全量三折比较

OFP 全量比较不是“在全部 13,372 个模块上训练并测试同一个模型”，而是三个 outer folds 的 out-of-fold 测试决策池化：

```text
fold 1 test: 4,457 modules (1,367 faulty)
fold 2 test: 4,457 modules (1,367 faulty)
fold 3 test: 4,458 modules (1,368 faulty)
pooled:     13,372 modules (4,102 faulty + 9,270 normal)
```

每次训练只能访问另外两个 folds；其内部 validation 完成 early stopping 和主协议阈值选择后，才允许读取 outer-test CSV。三个测试决策表必须模块级互斥且并集等于索引全集。pooled confusion matrix、lead 和 final score 必须从 13,372 行模块决策重新计算，禁止平均三个 fold 的 F1 或 final score。

同一份 out-of-fold score 输出两个彼此独立的评价协议：

1. `legacy_inclusive_v1`：当前论文主协议，`alarm_ts <= failure_ts`，每折 validation-selected threshold；
2. `ofp_original_strict_v1`：Model1/OFP 根 evaluator 兼容协议，`alarm_ts < failure_ts`，固定 0.5，`AvgLead = lead_sum / all_faulty_modules`。

第二行只冻结可由源码确认的评价语义。原 README 背后的部分模型训练与阈值来源无法从仓库快照完整重建，因此必须标注 Historical Reference，而不能声称完全复现其训练过程。

## 11. 产物审计要求

每次正式运行必须能够从以下信息重建实验：

- effective config 与 config fingerprint；
- split manifest 与 split fingerprint；
- train-only normalizer；
- teacher/student 训练历史与最佳 epoch；
- student checkpoint、冻结阈值和 `teacher_used_at_inference=false`；
- validation 阈值搜索表；
- test 模块级决策、OFP 兼容逐模块预测；
- CE/KL 有效行数、FGL pair 总数、正 pair 数及覆盖模块数；
- Python、PyTorch、NumPy、Pandas、设备和运行耗时。

方法来源：[A predictive approach to enhance time-series forecasting, Nature Communications (2025)](https://www.nature.com/articles/s41467-025-63786-4)。该引用用于说明 FGL teacher-student CE+KL 的成熟来源，不替代本项目对 OFP 标签、因果性和评价协议的独立审计。
