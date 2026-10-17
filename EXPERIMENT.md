# 扩大数据的两步历史预测

分支：`exp/v76-expanded-history`。

固定 44/80 基线，在同一 768 条数据上训练两步历史的下一观测残差。包含大批量训练与 GPU 规划；本次归档不声明最终导航成绩。

## 主要入口

- [`train_history.py`](predictive_rules/expanded_history_v76/train_history.py)
- [`evaluate.py`](predictive_rules/expanded_history_v76/evaluate.py)
- [`gpu_planner.py`](predictive_rules/expanded_history_v76/gpu_planner.py)

完整源码位于 `predictive_rules/expanded_history_v76/`。共 19 个顶层 Python 文件。相关依赖源码随分支保留，远端资产要求见 [运行说明](docs/RUNTIME.md)。入口脚本是历史研究流程，请先阅读其中的路径和 GPU 设置。

源码以本次本地已部署工作副本为基础；模型和数据没有上传。结果只记录已核实的完成评估，不将训练采样成功率或预测准确率代替导航成功率。
