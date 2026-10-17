# 固定感知，只训练转移规则

分支：`exp/v75-stable-rules`。

从 44/80 基线固定视觉、定义及目标，仅训练下一观测转移解码器。使用新增数据前 768 条的预定划分：609 拟合、159 留出。包含自适应批量和 GPU 规划；本次归档不声明最终导航成绩。

## 主要入口

- [`train_joint.py`](predictive_rules/stable_rules_v75/train_joint.py)
- [`base_objective.py`](predictive_rules/stable_rules_v75/base_objective.py)
- [`gpu_planner.py`](predictive_rules/stable_rules_v75/gpu_planner.py)

完整源码位于 `predictive_rules/stable_rules_v75/`。共 16 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
