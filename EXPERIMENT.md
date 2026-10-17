# 44/80 基线

分支：`exp/v68-goal-probability`。

通过离散转移推导到达生成子目标的概率，并加入相应监督。第 300 次更新检查点在当前普通场景 80 任务上为 44/80（55%）；短距离 23/40，中距离 21/40。

## 主要入口

- [`train_joint.py`](predictive_rules/goal_probability_v68/train_joint.py)
- [`joint_objective.py`](predictive_rules/goal_probability_v68/joint_objective.py)
- [`planner.py`](predictive_rules/goal_probability_v68/planner.py)

完整源码位于 `predictive_rules/goal_probability_v68/`。共 17 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
