# 紧凑两步历史

分支：`exp/v73-history2`。

固定视觉、定义和基线规则，以最近两步离散观测及动作修正当前步的下一观测分布。后续规划分支仍使用基线转移表。第 300 次检查点为 43/80，未超过 44/80。

## 主要入口

- [`train_history.py`](predictive_rules/compact_history_v73/train_history.py)
- [`evaluate.py`](predictive_rules/compact_history_v73/evaluate.py)

完整源码位于 `predictive_rules/compact_history_v73/`。共 20 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
