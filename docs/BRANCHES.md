# 分支索引

记录日期：2026-10-17。结果是已完成评估的历史快照，非实时监控。

| 分支 | 用途 | 结果与限制 |
|---|---|---|
| [exp/v63-counterfactual-closure](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v63-counterfactual-closure) | 反事实状态转移学习 | 从同一实际观测采集不同候选动作的结果，监督离散状态转移。当前普通场景 80 任务为 39/80。 |
| [exp/v68-goal-probability](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v68-goal-probability) | 44/80 基线 | 通过离散转移推导到达生成子目标的概率，并加入相应监督。第 300 次更新检查点在当前普通场景 80 任务上为 44/80（55%）；短距离 23/40，中距离 21/40。 |
| [exp/v69-history8](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v69-history8) | 八步离散历史预测 | 用历史观测编号和动作预测下一观测状态，保持符号规划。独立状态预测留出集表现退化，未通过候选筛选，没有宣称新的 80 任务成功率。 |
| [exp/v70-diverse-collection](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v70-diverse-collection) | 多样化训练轨迹采集 | 从训练池固定抽取 1200 个普通场景短中距离任务，预先隔离 960 个拟合任务与 240 个留出任务。采集实际轨迹及候选动作结果，不纳入导航验证任务。此分支是数据采集流程。 |
| [exp/v71-expanded-dynamics](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v71-expanded-dynamics) | 扩大数据后的视觉与动力学训练 | 在新采集数据上训练视觉残差和观测转移解码器，保留上下文、定义及目标模块。包含采集完成后的训练衔接、大批量选择和 GPU 符号规划。本次源码归档不声明最终导航成绩。 |
| [exp/v72-evidence-continuation](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v72-evidence-continuation) | 视觉证据与规则继续训练 | 从 44/80 基线继续拟合视觉证据与转移规则，使用固定的 288 条新增数据子集。训练 1000 次更新，选中第 600 次检查点；同一 80 任务为 35/80，未优于基线。 |
| [exp/v73-history2](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v73-history2) | 紧凑两步历史 | 固定视觉、定义和基线规则，以最近两步离散观测及动作修正当前步的下一观测分布。后续规划分支仍使用基线转移表。第 300 次检查点为 43/80，未超过 44/80。 |
| [perf/v74-gpu-symbolic](https://github.com/NaLv2004/NeSy-VLN/tree/perf/v74-gpu-symbolic) | GPU 符号推导与批处理优化 | 将相同有限状态 Bellman 递推移到 GPU，并测量批量大小。8 个程序、32 步组件测试约 27.9 倍加速；524288 个动作选择一致。该数值不是端到端训练加速倍数。 |
| [exp/v75-stable-rules](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v75-stable-rules) | 固定感知，只训练转移规则 | 从 44/80 基线固定视觉、定义及目标，仅训练下一观测转移解码器。使用新增数据前 768 条的预定划分：609 拟合、159 留出。包含自适应批量和 GPU 规划；本次归档不声明最终导航成绩。 |
| [exp/v76-expanded-history](https://github.com/NaLv2004/NeSy-VLN/tree/exp/v76-expanded-history) | 扩大数据的两步历史预测 | 固定 44/80 基线，在同一 768 条数据上训练两步历史的下一观测残差。包含大批量训练与 GPU 规划；本次归档不声明最终导航成绩。 |
| [exp/blocks-world-baseline](https://github.com/NaLv2004/NeSy-VLN/tree/exp/blocks-world-baseline) | Blocks World 验证 | 在积木世界中测试匿名状态、生成转移规则与子目标规划。此任务不同于 ObjectNav，不将其成绩与导航 80 任务结果混用。 |

`main` 保存公共实现及依赖的早期源码；`archive/research-history` 保存全部可导出的历史源码。历史分支包含已废弃方案，不能把它们的假设视为当前执行约定。各实验分支的 `EXPERIMENT.md` 指向对应入口。
