# DRFP 特征注册表

所有输入只能属于三类：Raw、Statistical、Expert/Rule。任意特征都以单模块为边界，并且只读取当前 endpoint 及其过去。

## Raw：12 维

原始物理通道保持不增不减：

`temperature, current, currentTXPower, currentRXPower, currentMultiRXPower1..4, currentMultiTXPower1..4`

时序分支取 endpoint 前 168 小时，在 1 小时因果网格上使用“该网格时刻之前最近一次真实观测”。数值仍来自原始 12 个通道，同时显式输入：

- 每通道有效 mask；
- 最近观测距当前网格点的 `delta_hours`；
- 历史不足时的左侧缺失 mask。

这一步是时序对齐，不把额外统计量混入 Raw 类。

## Statistical：76 维

对每个 Raw 通道分别计算 6 个 expanding prefix 统计量，共 72 维：

- Min
- Max
- Diff = Max - Min
- Std
- Skew
- Kurt

再加入 Model2 的 4 个 trailing-5 Pearson correlation：

- current / temperature
- current / currentTXPower
- current / currentRXPower
- currentTXPower / currentRXPower

实现使用显式因果前缀矩和 trailing window，不让追加的未来极端值改变已经生成的历史 Skew/Kurt；测试同时锁定了与统一 baseline 在固定序列上的数值、NaN、mask 和列顺序一致性。

## Expert/Rule：156 维

复用 Model2 当前启用的 38 条 RuleModel 阈值。每条硬规则先转换为一个 signed normalized margin：

- `all_gt`：`min(input - threshold) / physical_scale`
- `all_lt`：`min(threshold - input) / physical_scale`
- margin > 0 表示规则已满足；margin < 0 表示尚未越界。

每条规则构造 4 个因果信号，共 `38 × 4 = 152` 维：

1. 当前 margin；
2. 相对 12 小时前的 approach delta；
3. 最近 6 小时 margin > 0 的 persistence；
4. 最近 6 小时有效数据 coverage。

最后加入 4 个跨规则 summary：最大 margin、正 margin 比例、最大 approach、平均 coverage，总计 156 维。

margin 使用固定物理尺度并截断到 `[-10, 10]`；它不根据 test 数据拟合。RuleEncoder 同时接收 signal mask。Expert/Rule 最终在论文和结果中统一为一类，不再拆出额外 Expert42 输入。
